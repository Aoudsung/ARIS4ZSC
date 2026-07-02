from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import replace as _dc_replace
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:  # pragma: no cover - script execution path
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.aris_bellman.specs import FactorSpec, GraphSpec, OptionSpec

from experiments.overcooked_v2.env_adapter import OCV2Adapter
from experiments.overcooked_v2.layout_parser import parse_layout
from experiments.overcooked_v2.options import OCV2OptionLibrary
from experiments.overcooked_v2.partner_pool import (
    PREP_KINDS as _V4_PREP_KINDS,
    SUPPORT_KINDS as _V4_SUPPORT_KINDS,
    TERMINAL_KINDS as _V4_TERMINAL_KINDS,
)
from experiments.overcooked_v2.provenance import (
    PROVENANCE_SCHEMA_VERSION,
    sha256_numpy,
)


DEFAULT_FACTOR_MODES = {
    "bottleneck": 3,
    "resource": 2,
    "handoff": 3,
    "serving": 2,
    "pot_allocation": 2,
    "recipe_indicator": 2,
    "generic_option_pair": 3,
}

GRAPH_VARIANTS = (
    "full_support",
    "overcomplete",
    "overcomplete_minus_low_ce",
    "minus_critical",
    "minus_high_ce",
    "minus_serve_soup",
    "random_same_size",
    "complete_option_graph",
    "shuffled_routes",
    "shuffled_relevance",
)

# Noop is useful as an action baseline, but it is not an interaction factor.
# Including noop in CE support makes the graph reflect option duration/cost rather
# than coordination externality.
EXCLUDED_FACTOR_OPTION_KINDS = {"noop"}

REQUIRED_TASK_STAGE_OPTION_KINDS = {
    "deliver_ingredient_to_pot",
    "plate_soup",
    "serve_soup",
}

ROLE_CONTRAST_KIND_PAIRS: tuple[tuple[str, str], ...] = (
    ("serve_soup", "wait_at_bottleneck"),
    ("serve_soup", "clear_interaction_cell"),
    ("serve_soup", "deliver_ingredient_to_pot"),
    ("plate_soup", "deliver_ingredient_to_pot"),
    ("pick_plate", "fetch_ingredient"),
)


def infer_factor_kind(option_i: OptionSpec, option_j: OptionSpec) -> str:
    kinds = {option_i.kind, option_j.kind}
    regions = set(option_i.region_ids) | set(option_j.region_ids)

    if any(region.startswith("bottleneck") for region in regions):
        return "bottleneck"
    if "fetch_ingredient" in kinds:
        return "resource"
    if "handoff_counter" in kinds:
        return "handoff"
    if "plate_soup" in kinds or "serve_soup" in kinds:
        return "serving"
    if "deliver_ingredient_to_pot" in kinds:
        return "pot_allocation"
    if "press_recipe_button" in kinds:
        return "recipe_indicator"
    return "generic_option_pair"


class GraphCoverageError(RuntimeError):
    """Raised when CE candidates cannot satisfy required task-stage coverage.

    Distinct from a late-stage validate_task_stage_coverage() failure: this is a
    *construction-time* precondition for coverage-constrained selection — a required
    task-critical option has no CE candidate factor at all, so the factor-local
    Bellman controller could never have belief access for it.
    """


def _coverage_constrained_pairs(
    ce_matrix: np.ndarray,
    options: list[OptionSpec],
    eta: float,
    max_factors: int,
    selection_cfg: dict[str, Any],
) -> tuple[list[tuple[float, int, int]], dict[tuple[int, int], dict[str, Any]]]:
    """Coverage-constrained, diversity-capped CE factor selection.

    Raw top-K CE is NOT a value-sufficient support selector: dense/shaped intermediate
    value can dominate and exclude a terminal option that is necessary for completion
    (observed: serve_soup CE 0.73 fell below a top-16 cutoff of ~0.80). This selector
    keeps CE as the scoring signal but FIRST reserves slots so every task-critical
    option (and, optionally, every valid option id of a kind) has at least one relevant
    factor, THEN fills remaining slots by CE under diversity caps. The task-stage gate
    becomes a postcondition, not a late surprise.
    """
    all_pairs_unfiltered = _all_pairs(ce_matrix, options)
    all_pairs = [pair for pair in all_pairs_unfiltered if pair[0] > float(eta)]
    raw_rank = {
        (i, j): rank for rank, (_score, i, j) in enumerate(all_pairs_unfiltered)
    }
    cfg = selection_cfg or {}

    def _min_factors(spec: Any) -> int:
        return int(spec.get("min_factors", 1)) if isinstance(spec, dict) else int(spec)

    kind_min = {
        str(k): _min_factors(v)
        for k, v in (cfg.get("required_option_kind_coverage") or {}).items()
    }
    per_option_kinds = {
        str(k): int((v or {}).get("min_per_valid_option", 1))
        for k, v in (cfg.get("required_option_id_coverage") or {}).items()
    }
    diversity = cfg.get("diversity") or {}
    cap_kindpair = diversity.get("max_factors_per_option_kind_pair")
    cap_optid = diversity.get("max_factors_per_option_id")

    budget = max(0, int(max_factors))
    selected: list[tuple[float, int, int]] = []
    keys: set[tuple[int, int]] = set()
    reasons: dict[tuple[int, int], str] = {}
    kp_count: dict[tuple[str, str], int] = {}
    opt_count: dict[int, int] = {}
    eta_used: dict[tuple[int, int], float] = {}

    def kind(idx: int) -> str:
        return str(options[idx].kind)

    def kpkey(i: int, j: int) -> tuple[str, str]:
        return tuple(sorted((kind(i), kind(j))))  # type: ignore[return-value]

    def diversity_ok(i: int, j: int) -> bool:
        if cap_kindpair is not None and kp_count.get(kpkey(i, j), 0) >= int(cap_kindpair):
            return False
        if cap_optid is not None and (
            opt_count.get(i, 0) >= int(cap_optid) or opt_count.get(j, 0) >= int(cap_optid)
        ):
            return False
        return True

    def add(pair: tuple[float, int, int], why: str, *, enforce_div: bool) -> bool:
        _score, i, j = pair
        if (i, j) in keys or len(selected) >= budget:
            return False
        if enforce_div and not diversity_ok(i, j):
            return False
        selected.append(pair)
        keys.add((i, j))
        reasons[(i, j)] = why
        kp_count[kpkey(i, j)] = kp_count.get(kpkey(i, j), 0) + 1
        opt_count[i] = opt_count.get(i, 0) + 1
        opt_count[j] = opt_count.get(j, 0) + 1
        return True

    def best_touching_option(oid: int, candidate_eta: float) -> tuple[float, int, int] | None:
        return next(
            (
                p
                for p in all_pairs_unfiltered
                if p[0] > float(candidate_eta)
                and (p[1] == oid or p[2] == oid)
                and (p[1], p[2]) not in keys
            ),
            None,
        )

    def low_eta_thresholds() -> tuple[float, ...]:
        return tuple(x for x in (0.05, 0.01, 0.001, 0.0) if x < float(eta))

    # 1) Per-option-id coverage: every valid option of a required kind gets a relevant
    #    factor (kind-level alone can leave one serve destination with Rel(ω)=empty).
    for opt in options:
        need = per_option_kinds.get(str(opt.kind))
        if not need:
            continue
        oid = int(opt.id)
        while opt_count.get(oid, 0) < int(need):
            best = next(
                (p for p in all_pairs if (p[1] == oid or p[2] == oid) and (p[1], p[2]) not in keys),
                None,
            )
            reason = "mandatory_per_option_coverage"
            fallback_eta = None
            if best is None:
                fallback = None
                for candidate_eta in low_eta_thresholds():
                    fallback = best_touching_option(oid, candidate_eta)
                    if fallback is not None:
                        fallback_eta = float(candidate_eta)
                        break
                if fallback is None or fallback_eta is None:
                    raise GraphCoverageError(
                        f"No CE candidate above eta touches required option {oid}:{opt.kind}; "
                        "the factor-local controller would have no belief access for it."
                    )
                best = fallback
                reason = "mandatory_per_option_id_low_eta"
            if not add(best, reason, enforce_div=False):
                raise GraphCoverageError(
                    f"Coverage budget exhausted before covering required option {oid}:{opt.kind}."
                )
            if fallback_eta is not None:
                eta_used[(best[1], best[2])] = fallback_eta

    # 2) Per-kind coverage (pick_plate / plate_soup / serve_soup ...).
    for k, min_n in kind_min.items():
        while sum(1 for _s, i, j in selected if k in (kind(i), kind(j))) < int(min_n):
            best = next(
                (p for p in all_pairs if (p[1], p[2]) not in keys and k in (kind(p[1]), kind(p[2]))),
                None,
            )
            if best is None:
                raise GraphCoverageError(f"No CE candidate for required option kind {k!r}.")
            if not add(best, "mandatory_kind_coverage", enforce_div=False):
                raise GraphCoverageError(
                    f"Coverage budget exhausted before covering required option kind {k!r}."
                )

    # Step 2.5: role-contrast reservation. For each configured kind-pair, reserve
    # at least one factor whose option pair matches the kinds (in either order).
    # Structural prior for terminal role switching; not a gate.
    role_pairs_cfg = tuple(
        tuple(p) for p in (
            (selection_cfg or {}).get("required_role_contrast_pairs")
            or ROLE_CONTRAST_KIND_PAIRS
        )
    )
    for a_kind, b_kind in role_pairs_cfg:
        if len(selected) >= max_factors:
            break
        already = any(
            {kind(i), kind(j)} == {str(a_kind), str(b_kind)}
            for _s, i, j in selected
        )
        if already:
            continue
        best = next(
            (
                p for p in all_pairs
                if (p[1], p[2]) not in keys
                and {kind(p[1]), kind(p[2])} == {str(a_kind), str(b_kind)}
            ),
            None,
        )
        if best is not None:
            add(best, "mandatory_role_contrast", enforce_div=False)

    # 3) Fill remaining slots by CE under diversity caps.
    for pair in all_pairs:
        if len(selected) >= budget:
            break
        if (pair[1], pair[2]) in keys:
            continue
        add(pair, "ce_fill", enforce_div=True)

    ordered = sorted(selected, key=lambda s: (-s[0], s[1], s[2]))
    provenance = {}
    for _s, i, j in ordered:
        entry: dict[str, Any] = {
            "selected_by": reasons[(i, j)],
            "raw_ce_rank": raw_rank.get((i, j)),
        }
        if (i, j) in eta_used:
            entry["eta_used"] = eta_used[(i, j)]
        provenance[(i, j)] = entry
    return ordered, provenance


def _load_ego_selectable_from_replay(replay_path: str | None) -> frozenset[int] | None:
    """Read ego_option per replay row; return None when replay is absent or empty."""
    if not replay_path:
        return None

    p = Path(replay_path)
    if not p.exists():
        return None

    ids: set[int] = set()
    try:
        with np.load(p, allow_pickle=False) as npz:
            if "rows" not in npz.files:
                return None
            for raw in npz["rows"]:
                try:
                    row = json.loads(str(raw))
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
                if "ego_option" in row:
                    ids.add(int(row["ego_option"]))
    except Exception:
        return None

    return frozenset(ids) if ids else None


def _ego_kind_relevant_options(
    factor: FactorSpec,
    options: list[OptionSpec],
    ego_selectable: frozenset[int],
) -> list[int]:
    """Project a CE row/col factor onto ego-selectable options by option kind."""
    if not (0 <= int(factor.option_i) < len(options)):
        return []
    if not (0 <= int(factor.option_j) < len(options)):
        return []

    kind_i = str(options[int(factor.option_i)].kind)
    kind_j = str(options[int(factor.option_j)].kind)
    target_kinds = {kind_i, kind_j}
    entity_ids = set(factor.entity_ids)
    region_ids = set(factor.region_ids)

    relevant: set[int] = set()
    for opt in options:
        oid = int(opt.id)
        if oid not in ego_selectable:
            continue
        if str(opt.kind) in target_kinds:
            relevant.add(oid)
        elif entity_ids.intersection(opt.entity_ids):
            relevant.add(oid)
        elif region_ids.intersection(opt.region_ids):
            relevant.add(oid)
    return sorted(relevant)


def _ego_complement_relevant_options(
    factor: FactorSpec,
    options: list[OptionSpec],
    ego_selectable: frozenset[int],
) -> list[int]:
    """Project CE factors to ego-selectable complement task-chain actions."""
    if not (0 <= int(factor.option_i) < len(options)):
        return []
    if not (0 <= int(factor.option_j) < len(options)):
        return []

    kind_i = str(options[int(factor.option_i)].kind)
    kind_j = str(options[int(factor.option_j)].kind)
    kinds_in_factor = {kind_i, kind_j}

    if kinds_in_factor & _V4_TERMINAL_KINDS:
        target_kinds = _V4_TERMINAL_KINDS | _V4_PREP_KINDS | _V4_SUPPORT_KINDS
    elif kinds_in_factor & _V4_PREP_KINDS:
        target_kinds = _V4_PREP_KINDS | _V4_TERMINAL_KINDS
    elif kinds_in_factor & _V4_SUPPORT_KINDS:
        target_kinds = _V4_SUPPORT_KINDS | _V4_PREP_KINDS
    else:
        return _ego_kind_relevant_options(factor, options, ego_selectable)

    entity_ids = set(factor.entity_ids)
    region_ids = set(factor.region_ids)

    relevant: set[int] = set()
    for opt in options:
        oid = int(opt.id)
        if oid not in ego_selectable:
            continue
        if str(opt.kind) == "noop":
            continue
        if str(opt.kind) in target_kinds:
            relevant.add(oid)
        elif entity_ids.intersection(opt.entity_ids):
            relevant.add(oid)
        elif region_ids.intersection(opt.region_ids):
            relevant.add(oid)
    return sorted(relevant)


def build_support_graph(
    layout_name: str,
    options: list[OptionSpec],
    ce_matrix: np.ndarray,
    eta: float,
    max_factors: int,
    mode_config: dict[str, int] | None = None,
    *,
    selection_cfg: dict[str, Any] | None = None,
    ego_selectable: frozenset[int] | None = None,
    relevance_semantics: str = "legacy_id_pair",
) -> GraphSpec:
    selection = str((selection_cfg or {}).get("selection", "top_k"))
    if selection == "coverage_constrained_ce":
        pairs, provenance = _coverage_constrained_pairs(
            ce_matrix, options, eta, max_factors, selection_cfg or {}
        )
    else:
        pairs = _top_pairs_above_eta(ce_matrix, options, eta, max_factors)
        provenance = {
            (int(i), int(j)): {"selected_by": "ce_top_k", "raw_ce_rank": rank}
            for rank, (_s, i, j) in enumerate(pairs)
        }
    factors = _factor_specs_from_pairs(layout_name, options, pairs, mode_config)
    # FactorSpec is a frozen dataclass; attach selection provenance via replace().
    factors = [
        _dc_replace(
            factor,
            metadata={
                **(factor.metadata or {}),
                **provenance[(int(factor.option_i), int(factor.option_j))],
            },
        )
        if (int(factor.option_i), int(factor.option_j)) in provenance
        else factor
        for factor in factors
    ]
    return make_graph_spec(
        layout_name,
        options,
        factors,
        metadata={
            "graph_variant": "full_support",
            "eta": float(eta),
            "max_factors": int(max_factors),
            "full_max_factors": int(max_factors),
            "graph_selection": selection,
            "provenance": {
                "schema_version": PROVENANCE_SCHEMA_VERSION,
                "ce_matrix_array_sha256": sha256_numpy(ce_matrix),
            },
        },
        ego_selectable=ego_selectable,
        relevance_semantics=relevance_semantics,
    )


def make_graph_spec(
    layout_name: str,
    options: list[OptionSpec],
    factors: list[FactorSpec],
    *,
    route_source_factors: list[FactorSpec] | None = None,
    relevance_source_factors: list[FactorSpec] | None = None,
    metadata: dict[str, Any] | None = None,
    ego_selectable: frozenset[int] | None = None,
    relevance_semantics: str = "legacy_id_pair",
) -> GraphSpec:
    num_options = len(options)
    num_factors = len(factors)
    relevance = np.zeros((num_factors, num_options), dtype=bool)
    route_map: dict[int, tuple[int, ...]] = {}

    def _rel_options_for_q(factor: FactorSpec) -> list[int]:
        if relevance_semantics == "ego_complement_projection":
            if ego_selectable is None:
                raise ValueError(
                    "make_graph_spec: relevance_semantics='ego_complement_projection' "
                    "requires ego_selectable set (must not be None)."
                )
            return _ego_complement_relevant_options(factor, options, ego_selectable)
        if relevance_semantics == "ego_kind_projection":
            if ego_selectable is None:
                raise ValueError(
                    "make_graph_spec: relevance_semantics='ego_kind_projection' "
                    "requires ego_selectable set (must not be None)."
                )
            return _ego_kind_relevant_options(factor, options, ego_selectable)
        if relevance_semantics == "legacy_id_pair":
            return _relevant_options_for_factor(factor, options)
        raise ValueError(
            f"Unknown relevance_semantics {relevance_semantics!r}; expected "
            "'legacy_id_pair', 'ego_kind_projection', or 'ego_complement_projection'."
        )

    for idx, factor in enumerate(factors):
        relevance_factor = (
            relevance_source_factors[idx] if relevance_source_factors is not None else factor
        )
        for option_id in _rel_options_for_q(relevance_factor):
            relevance[idx, option_id] = True

        route_factor = route_source_factors[idx] if route_source_factors is not None else factor
        route_map[idx] = tuple(_relevant_options_for_factor(route_factor, options))

    option_mask = np.ones((num_options,), dtype=bool)
    factor_mask = np.ones((num_factors,), dtype=bool)
    max_modes = max((factor.num_modes for factor in factors), default=0)
    mode_mask = np.zeros((num_factors, max_modes), dtype=bool)
    for idx, factor in enumerate(factors):
        mode_mask[idx, : factor.num_modes] = True

    graph_metadata = dict(metadata or {})
    graph_metadata["relevance_semantics"] = str(relevance_semantics)
    if ego_selectable is not None:
        graph_metadata["ego_selectable_count"] = int(len(ego_selectable))
        graph_metadata["ego_selectable_ids"] = sorted(int(x) for x in ego_selectable)

    return GraphSpec(
        layout_name=layout_name,
        options=options,
        factors=factors,
        relevance=relevance,
        option_mask=option_mask,
        factor_mask=factor_mask,
        mode_mask=mode_mask,
        route_map=route_map,
        metadata=graph_metadata,
    )


def validate_task_stage_coverage(
    graph: GraphSpec,
    required_option_kinds: set[str] | None = None,
) -> None:
    required = set(required_option_kinds or REQUIRED_TASK_STAGE_OPTION_KINDS)
    covered: set[str] = set()
    for factor in graph.factors:
        if 0 <= int(factor.option_i) < len(graph.options):
            covered.add(str(graph.options[int(factor.option_i)].kind))
        if 0 <= int(factor.option_j) < len(graph.options):
            covered.add(str(graph.options[int(factor.option_j)].kind))
    missing = sorted(required - covered)
    if missing:
        raise RuntimeError(f"CE graph lacks required task-stage options: {missing}")


def build_graph_variant(
    variant: str,
    layout_name: str,
    options: list[OptionSpec],
    ce_matrix: np.ndarray,
    *,
    eta: float,
    max_factors: int,
    full_max_factors: int | None = None,
    overcomplete_extra_factors: int = 0,
    mode_config: dict[str, int] | None = None,
    criticality_scores: Any | None = None,
    seed: int = 17,
    require_task_stage_coverage: bool = False,
    selection_cfg: dict[str, Any] | None = None,
    ego_selectable: frozenset[int] | None = None,
    relevance_semantics: str = "legacy_id_pair",
) -> GraphSpec:
    full_budget = int(full_max_factors if full_max_factors is not None else max_factors)
    if variant == "full_support":
        graph = full_support_graph(
            layout_name, options, ce_matrix, eta, full_budget, mode_config,
            selection_cfg=selection_cfg,
            ego_selectable=ego_selectable,
            relevance_semantics=relevance_semantics,
        )
    elif variant == "overcomplete":
        graph = overcomplete_graph(
            layout_name,
            options,
            ce_matrix,
            eta,
            full_budget,
            mode_config,
            overcomplete_extra_factors=overcomplete_extra_factors,
            ego_selectable=ego_selectable,
            relevance_semantics=relevance_semantics,
        )
    elif variant == "overcomplete_minus_low_ce":
        graph = overcomplete_minus_low_ce_graph(
            layout_name,
            options,
            ce_matrix,
            eta,
            full_budget,
            mode_config,
            overcomplete_extra_factors=overcomplete_extra_factors,
            ego_selectable=ego_selectable,
            relevance_semantics=relevance_semantics,
        )
    elif variant == "minus_critical":
        graph = minus_critical_graph(
            layout_name,
            options,
            ce_matrix,
            eta,
            full_budget,
            mode_config,
            criticality_scores,
            ego_selectable=ego_selectable,
            relevance_semantics=relevance_semantics,
        )
    elif variant == "minus_high_ce":
        graph = minus_high_ce_graph(
            layout_name,
            options,
            ce_matrix,
            eta,
            full_budget,
            mode_config,
            ego_selectable=ego_selectable,
            relevance_semantics=relevance_semantics,
        )
    elif variant == "minus_serve_soup":
        graph = minus_serve_soup_graph(
            layout_name,
            options,
            ce_matrix,
            eta,
            full_budget,
            mode_config,
            selection_cfg=selection_cfg,
            ego_selectable=ego_selectable,
            relevance_semantics=relevance_semantics,
        )
    elif variant == "random_same_size":
        graph = random_same_size_graph(
            layout_name,
            options,
            ce_matrix,
            eta,
            full_budget,
            mode_config,
            seed,
            ego_selectable=ego_selectable,
            relevance_semantics=relevance_semantics,
        )
    elif variant == "complete_option_graph":
        graph = complete_option_graph(
            layout_name,
            options,
            ce_matrix,
            mode_config,
            ego_selectable=ego_selectable,
            relevance_semantics=relevance_semantics,
        )
    elif variant == "shuffled_routes":
        graph = shuffled_routes_graph(
            layout_name,
            options,
            ce_matrix,
            eta,
            full_budget,
            mode_config,
            ego_selectable=ego_selectable,
            relevance_semantics=relevance_semantics,
        )
    elif variant == "shuffled_relevance":
        graph = shuffled_relevance_graph(
            layout_name,
            options,
            ce_matrix,
            eta,
            full_budget,
            mode_config,
            ego_selectable=ego_selectable,
            relevance_semantics=relevance_semantics,
        )
    else:
        raise ValueError(f"Unknown graph variant {variant!r}; expected one of {GRAPH_VARIANTS}.")

    if require_task_stage_coverage:
        validate_task_stage_coverage(graph)
    graph.metadata = _metadata_with_ce_hash(graph.metadata, ce_matrix)
    return graph


def _metadata_with_ce_hash(
    metadata: dict[str, Any] | None,
    ce_matrix: np.ndarray,
) -> dict[str, Any]:
    updated = dict(metadata or {})
    provenance = dict(updated.get("provenance", {}))
    provenance.setdefault("schema_version", PROVENANCE_SCHEMA_VERSION)
    provenance["ce_matrix_array_sha256"] = sha256_numpy(ce_matrix)
    updated["provenance"] = provenance
    return updated


def full_support_graph(
    layout_name: str,
    options: list[OptionSpec],
    ce_matrix: np.ndarray,
    eta: float,
    max_factors: int,
    mode_config: dict[str, int] | None = None,
    selection_cfg: dict[str, Any] | None = None,
    *,
    ego_selectable: frozenset[int] | None = None,
    relevance_semantics: str = "legacy_id_pair",
) -> GraphSpec:
    return build_support_graph(
        layout_name, options, ce_matrix, eta, max_factors, mode_config,
        selection_cfg=selection_cfg,
        ego_selectable=ego_selectable,
        relevance_semantics=relevance_semantics,
    )


def overcomplete_graph(
    layout_name: str,
    options: list[OptionSpec],
    ce_matrix: np.ndarray,
    eta: float,
    full_max_factors: int,
    mode_config: dict[str, int] | None = None,
    *,
    overcomplete_extra_factors: int = 0,
    ego_selectable: frozenset[int] | None = None,
    relevance_semantics: str = "legacy_id_pair",
) -> GraphSpec:
    full_pairs = _top_pairs_above_eta(ce_matrix, options, eta, full_max_factors)
    full_keys = {(i, j) for _, i, j in full_pairs}
    extras = [
        pair
        for pair in _all_positive_pairs(ce_matrix, options)
        if (pair[1], pair[2]) not in full_keys
    ]
    extra_pairs = extras[: max(0, int(overcomplete_extra_factors))]
    selected = full_pairs + extra_pairs
    factors = _factor_specs_from_pairs(layout_name, options, selected, mode_config)
    return make_graph_spec(
        layout_name,
        options,
        factors,
        metadata={
            "graph_variant": "overcomplete",
            "eta": float(eta),
            "full_max_factors": int(full_max_factors),
            "overcomplete_extra_factors": int(overcomplete_extra_factors),
            "full_count": len(full_pairs),
            "extra_count": len(extra_pairs),
        },
        ego_selectable=ego_selectable,
        relevance_semantics=relevance_semantics,
    )


def overcomplete_minus_low_ce_graph(
    layout_name: str,
    options: list[OptionSpec],
    ce_matrix: np.ndarray,
    eta: float,
    full_max_factors: int,
    mode_config: dict[str, int] | None = None,
    *,
    overcomplete_extra_factors: int = 0,
    ego_selectable: frozenset[int] | None = None,
    relevance_semantics: str = "legacy_id_pair",
) -> GraphSpec:
    full_pairs = _top_pairs_above_eta(ce_matrix, options, eta, full_max_factors)
    overcomplete = overcomplete_graph(
        layout_name,
        options,
        ce_matrix,
        eta,
        full_max_factors,
        mode_config,
        overcomplete_extra_factors=overcomplete_extra_factors,
        ego_selectable=ego_selectable,
        relevance_semantics=relevance_semantics,
    )
    full_keys = {(i, j) for _, i, j in full_pairs}
    extras = [
        factor
        for factor in overcomplete.factors
        if (factor.option_i, factor.option_j) not in full_keys
    ]
    if extras:
        extras.sort(key=lambda factor: (factor.ce_score, factor.option_i, factor.option_j))
        remove_count = max(1, len(extras) // 2)
        removed = {(factor.option_i, factor.option_j) for factor in extras[:remove_count]}
        factors = [
            factor
            for factor in overcomplete.factors
            if (factor.option_i, factor.option_j) not in removed
        ]
    else:
        removed = set()
        factors = list(overcomplete.factors)

    factors = _renumber_factors(factors)
    return make_graph_spec(
        layout_name,
        options,
        factors,
        metadata={
            "graph_variant": "overcomplete_minus_low_ce",
            "eta": float(eta),
            "full_max_factors": int(full_max_factors),
            "overcomplete_extra_factors": int(overcomplete_extra_factors),
            "removed_pairs": sorted([list(pair) for pair in removed]),
        },
        ego_selectable=ego_selectable,
        relevance_semantics=relevance_semantics,
    )


def minus_critical_graph(
    layout_name: str,
    options: list[OptionSpec],
    ce_matrix: np.ndarray,
    eta: float,
    max_factors: int,
    mode_config: dict[str, int] | None = None,
    criticality_scores: Any | None = None,
    *,
    ego_selectable: frozenset[int] | None = None,
    relevance_semantics: str = "legacy_id_pair",
) -> GraphSpec:
    if criticality_scores is None:
        raise ValueError("minus_critical requires validation criticality scores.")
    full = build_support_graph(
        layout_name,
        options,
        ce_matrix,
        eta,
        max_factors,
        mode_config,
        ego_selectable=ego_selectable,
        relevance_semantics=relevance_semantics,
    )
    critical_pair, source = _critical_pair(full.factors, criticality_scores)
    if critical_pair is None:
        raise ValueError("minus_critical could not resolve a critical factor from scores.")
    factors = [
        factor
        for factor in full.factors
        if (factor.option_i, factor.option_j) != critical_pair
    ]
    return make_graph_spec(
        layout_name,
        options,
        _renumber_factors(factors),
        metadata={
            "graph_variant": "minus_critical",
            "eta": float(eta),
            "max_factors": int(max_factors),
            "full_max_factors": int(max_factors),
            "critical_pair": list(critical_pair) if critical_pair is not None else None,
            "criticality_source": source,
        },
        ego_selectable=ego_selectable,
        relevance_semantics=relevance_semantics,
    )


def minus_high_ce_graph(
    layout_name: str,
    options: list[OptionSpec],
    ce_matrix: np.ndarray,
    eta: float,
    max_factors: int,
    mode_config: dict[str, int] | None = None,
    *,
    ego_selectable: frozenset[int] | None = None,
    relevance_semantics: str = "legacy_id_pair",
) -> GraphSpec:
    full = build_support_graph(
        layout_name,
        options,
        ce_matrix,
        eta,
        max_factors,
        mode_config,
        ego_selectable=ego_selectable,
        relevance_semantics=relevance_semantics,
    )
    high_pair = _high_ce_pair(full.factors)
    factors = [
        factor
        for factor in full.factors
        if (factor.option_i, factor.option_j) != high_pair
    ]
    return make_graph_spec(
        layout_name,
        options,
        _renumber_factors(factors),
        metadata={
            "graph_variant": "minus_high_ce",
            "eta": float(eta),
            "max_factors": int(max_factors),
            "full_max_factors": int(max_factors),
            "removed_pair": list(high_pair) if high_pair is not None else None,
            "criticality_source": "ce_highest_debug",
        },
        ego_selectable=ego_selectable,
        relevance_semantics=relevance_semantics,
    )


def minus_serve_soup_graph(
    layout_name: str,
    options: list[OptionSpec],
    ce_matrix: np.ndarray,
    eta: float,
    max_factors: int,
    mode_config: dict[str, int] | None = None,
    selection_cfg: dict[str, Any] | None = None,
    *,
    ego_selectable: frozenset[int] | None = None,
    relevance_semantics: str = "legacy_id_pair",
) -> GraphSpec:
    """A7 critical-factor deletion: the coverage-constrained support graph with every
    factor touching a serve_soup option removed. Built from the SAME selection_cfg as
    full_support (so it is coverage-constrained-minus-serve, not raw-top-K-minus-serve)
    to isolate the causal role of the serve factor. Pair with
    `graph.require_task_stage_coverage: false` in the config so the deliberately
    serve-less graph is allowed to train rather than rejected at the coverage gate."""
    full = full_support_graph(
        layout_name, options, ce_matrix, eta, max_factors, mode_config,
        selection_cfg=selection_cfg,
        ego_selectable=ego_selectable,
        relevance_semantics=relevance_semantics,
    )
    serve_ids = {int(opt.id) for opt in options if str(opt.kind) == "serve_soup"}
    factors = [
        factor
        for factor in full.factors
        if int(factor.option_i) not in serve_ids
        and int(factor.option_j) not in serve_ids
    ]
    return make_graph_spec(
        layout_name,
        options,
        _renumber_factors(factors),
        metadata={
            "graph_variant": "minus_serve_soup",
            "eta": float(eta),
            "max_factors": int(max_factors),
            "full_max_factors": int(max_factors),
            "graph_selection": (selection_cfg or {}).get("selection"),
            "removed_option_kind": "serve_soup",
            "removed_count": int(len(full.factors) - len(factors)),
            "removed_serve_option_ids": sorted(serve_ids),
        },
        ego_selectable=ego_selectable,
        relevance_semantics=relevance_semantics,
    )


def random_same_size_graph(
    layout_name: str,
    options: list[OptionSpec],
    ce_matrix: np.ndarray,
    eta: float,
    max_factors: int,
    mode_config: dict[str, int] | None = None,
    seed: int = 17,
    *,
    ego_selectable: frozenset[int] | None = None,
    relevance_semantics: str = "legacy_id_pair",
) -> GraphSpec:
    full_pairs = _top_pairs_above_eta(ce_matrix, options, eta, max_factors)
    full_keys = {(i, j) for _, i, j in full_pairs}
    candidates = [
        pair
        for pair in _all_pairs(ce_matrix, options)
        if (pair[1], pair[2]) not in full_keys
    ]
    if len(candidates) < len(full_pairs):
        candidates = _all_pairs(ce_matrix, options)
    rng = random.Random(seed)
    selected = rng.sample(candidates, k=min(len(full_pairs), len(candidates)))
    factors = _factor_specs_from_pairs(layout_name, options, selected, mode_config)
    return make_graph_spec(
        layout_name,
        options,
        factors,
        metadata={
            "graph_variant": "random_same_size",
            "eta": float(eta),
            "full_max_factors": int(max_factors),
            "seed": int(seed),
            "full_support_count": len(full_pairs),
        },
        ego_selectable=ego_selectable,
        relevance_semantics=relevance_semantics,
    )


def complete_option_graph(
    layout_name: str,
    options: list[OptionSpec],
    ce_matrix: np.ndarray,
    mode_config: dict[str, int] | None = None,
    *,
    ego_selectable: frozenset[int] | None = None,
    relevance_semantics: str = "legacy_id_pair",
) -> GraphSpec:
    factors = _factor_specs_from_pairs(layout_name, options, _all_pairs(ce_matrix, options), mode_config)
    return make_graph_spec(
        layout_name,
        options,
        factors,
        metadata={
            "graph_variant": "complete_option_graph",
            "num_complete_pairs": len(factors),
            "max_factors_ignored": True,
        },
        ego_selectable=ego_selectable,
        relevance_semantics=relevance_semantics,
    )


def shuffled_routes_graph(
    layout_name: str,
    options: list[OptionSpec],
    ce_matrix: np.ndarray,
    eta: float,
    max_factors: int,
    mode_config: dict[str, int] | None = None,
    *,
    ego_selectable: frozenset[int] | None = None,
    relevance_semantics: str = "legacy_id_pair",
) -> GraphSpec:
    full = build_support_graph(
        layout_name,
        options,
        ce_matrix,
        eta,
        max_factors,
        mode_config,
        ego_selectable=ego_selectable,
        relevance_semantics=relevance_semantics,
    )
    permutation = _rotated_permutation(len(full.factors))
    route_source = [full.factors[idx] for idx in permutation]
    return make_graph_spec(
        layout_name,
        options,
        list(full.factors),
        route_source_factors=route_source,
        metadata={
            "graph_variant": "shuffled_routes",
            "eta": float(eta),
            "full_max_factors": int(max_factors),
            "route_permutation": permutation,
        },
        ego_selectable=ego_selectable,
        relevance_semantics=relevance_semantics,
    )


def shuffled_relevance_graph(
    layout_name: str,
    options: list[OptionSpec],
    ce_matrix: np.ndarray,
    eta: float,
    max_factors: int,
    mode_config: dict[str, int] | None = None,
    *,
    ego_selectable: frozenset[int] | None = None,
    relevance_semantics: str = "legacy_id_pair",
) -> GraphSpec:
    full = build_support_graph(
        layout_name,
        options,
        ce_matrix,
        eta,
        max_factors,
        mode_config,
        ego_selectable=ego_selectable,
        relevance_semantics=relevance_semantics,
    )
    permutation = _rotated_permutation(len(full.factors))
    relevance_source = [full.factors[idx] for idx in permutation]
    return make_graph_spec(
        layout_name,
        options,
        list(full.factors),
        relevance_source_factors=relevance_source,
        metadata={
            "graph_variant": "shuffled_relevance",
            "eta": float(eta),
            "full_max_factors": int(max_factors),
            "relevance_permutation": permutation,
        },
        ego_selectable=ego_selectable,
        relevance_semantics=relevance_semantics,
    )


def _factor_specs_from_pairs(
    layout_name: str,
    options: list[OptionSpec],
    pairs: list[tuple[float, int, int]],
    mode_config: dict[str, int] | None,
) -> list[FactorSpec]:
    del layout_name
    modes = {**DEFAULT_FACTOR_MODES, **(mode_config or {})}
    factors: list[FactorSpec] = []
    for factor_id, (score, option_i, option_j) in enumerate(pairs):
        opt_i = options[option_i]
        opt_j = options[option_j]
        factor_kind = infer_factor_kind(opt_i, opt_j)
        factors.append(
            FactorSpec(
                id=factor_id,
                option_i=option_i,
                option_j=option_j,
                ce_score=float(score),
                num_modes=int(modes.get(factor_kind, modes["generic_option_pair"])),
                entity_ids=_union_sorted(opt_i.entity_ids, opt_j.entity_ids),
                region_ids=_union_sorted(opt_i.region_ids, opt_j.region_ids),
                factor_kind=factor_kind,
            )
        )
    return factors


def _top_pairs_above_eta(
    ce_matrix: np.ndarray,
    options: list[OptionSpec],
    eta: float,
    max_factors: int,
) -> list[tuple[float, int, int]]:
    pairs = [
        pair
        for pair in _all_pairs(ce_matrix, options)
        if pair[0] > float(eta)
    ]
    return pairs[: max(0, int(max_factors))]


def _all_positive_pairs(
    ce_matrix: np.ndarray,
    options: list[OptionSpec],
) -> list[tuple[float, int, int]]:
    return [pair for pair in _all_pairs(ce_matrix, options) if pair[0] > 0.0]


def _all_pairs(
    ce_matrix: np.ndarray,
    options: list[OptionSpec],
) -> list[tuple[float, int, int]]:
    ce = np.asarray(ce_matrix, dtype=float)
    pairs = [
        (float(ce[i, j]), int(i), int(j))
        for i in range(ce.shape[0])
        for j in range(ce.shape[1])
        if _factorable_option_pair(options, int(i), int(j))
    ]
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    return pairs


def _factorable_option_pair(
    options: list[OptionSpec],
    option_i: int,
    option_j: int,
) -> bool:
    if option_i == option_j:
        return False
    if option_i < 0 or option_j < 0:
        return False
    if option_i >= len(options) or option_j >= len(options):
        return False
    return (
        options[option_i].kind not in EXCLUDED_FACTOR_OPTION_KINDS
        and options[option_j].kind not in EXCLUDED_FACTOR_OPTION_KINDS
    )

def _relevant_options_for_factor(
    factor: FactorSpec,
    options: list[OptionSpec],
) -> list[int]:
    relevant = {int(factor.option_i), int(factor.option_j)}
    entity_ids = set(factor.entity_ids)
    region_ids = set(factor.region_ids)
    for opt in options:
        if entity_ids.intersection(opt.entity_ids):
            relevant.add(int(opt.id))
        if region_ids.intersection(opt.region_ids):
            relevant.add(int(opt.id))
    return sorted(option_id for option_id in relevant if 0 <= option_id < len(options))


def _renumber_factors(factors: list[FactorSpec]) -> list[FactorSpec]:
    return [
        FactorSpec(
            id=idx,
            option_i=factor.option_i,
            option_j=factor.option_j,
            ce_score=factor.ce_score,
            num_modes=factor.num_modes,
            entity_ids=factor.entity_ids,
            region_ids=factor.region_ids,
            factor_kind=factor.factor_kind,
            metadata=factor.metadata,
        )
        for idx, factor in enumerate(factors)
    ]


def _critical_pair(
    factors: list[FactorSpec],
    criticality_scores: Any | None,
) -> tuple[tuple[int, int] | None, str]:
    if not factors:
        return None, "empty_graph"
    scored_pair = _critical_pair_from_scores(factors, criticality_scores)
    if scored_pair is not None:
        return scored_pair, "validation_return_drop"
    return None, "missing_validation_scores"


def _high_ce_pair(factors: list[FactorSpec]) -> tuple[int, int] | None:
    if not factors:
        return None
    factor = max(factors, key=lambda item: item.ce_score)
    return int(factor.option_i), int(factor.option_j)


def _critical_pair_from_scores(
    factors: list[FactorSpec],
    criticality_scores: Any | None,
) -> tuple[int, int] | None:
    if criticality_scores is None:
        return None
    if isinstance(criticality_scores, dict):
        best_pair = None
        best_score = float("-inf")
        for factor in factors:
            keys = (
                str(factor.id),
                f"{factor.option_i},{factor.option_j}",
                (factor.option_i, factor.option_j),
            )
            for key in keys:
                if key in criticality_scores and float(criticality_scores[key]) > best_score:
                    best_score = float(criticality_scores[key])
                    best_pair = (int(factor.option_i), int(factor.option_j))
        return best_pair
    scores = np.asarray(criticality_scores, dtype=float)
    if scores.ndim != 1 or scores.size == 0:
        return None
    best_idx = int(np.argmax(scores[: len(factors)]))
    factor = factors[best_idx]
    return int(factor.option_i), int(factor.option_j)


def _rotated_permutation(n_items: int) -> list[int]:
    if n_items <= 1:
        return list(range(n_items))
    return [(idx + 1) % n_items for idx in range(n_items)]


def _union_sorted(a: tuple[str, ...], b: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(set(a) | set(b)))


def _load_mode_config(path_or_json: str | None) -> dict[str, int] | None:
    if not path_or_json:
        return None
    path = Path(path_or_json)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = json.loads(path_or_json)
    return {str(key): int(value) for key, value in data.items()}


def _load_criticality(path: str | None) -> Any | None:
    if not path:
        return None
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _cmd_build(args: argparse.Namespace) -> None:
    env = OCV2Adapter(
        args.layout,
        max_steps=args.max_steps,
        observation_type="default",
        force_path_planning=False,
    )
    layout_graph = parse_layout(env, args.layout)
    option_lib = OCV2OptionLibrary(layout_graph, max_option_steps=args.max_option_steps)
    ce_matrix = np.load(args.ce)
    graph = build_graph_variant(
        args.variant,
        args.layout,
        option_lib.options,
        ce_matrix,
        eta=args.eta,
        max_factors=args.max_factors,
        full_max_factors=args.full_max_factors,
        overcomplete_extra_factors=args.overcomplete_extra_factors,
        mode_config=_load_mode_config(args.mode_config),
        criticality_scores=_load_criticality(args.criticality),
        seed=args.seed,
        require_task_stage_coverage=bool(args.require_task_stage_coverage),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(graph.to_json_dict(), indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build OvercookedV2 GraphSpec variants")
    parser.add_argument("--layout", required=True)
    parser.add_argument("--ce", required=True)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--max_factors", type=int, default=16)
    parser.add_argument("--full_max_factors", type=int, default=None)
    parser.add_argument("--overcomplete_extra_factors", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variant", choices=GRAPH_VARIANTS, default="full_support")
    parser.add_argument("--mode_config", default=None)
    parser.add_argument("--criticality", default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--max_option_steps", type=int, default=12)
    parser.add_argument("--require_task_stage_coverage", action="store_true")
    parser.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
