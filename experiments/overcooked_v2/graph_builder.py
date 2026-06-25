from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:  # pragma: no cover - script execution path
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.aris_bellman.specs import FactorSpec, GraphSpec, OptionSpec

from experiments.overcooked_v2.env_adapter import OCV2Adapter
from experiments.overcooked_v2.layout_parser import parse_layout
from experiments.overcooked_v2.options import OCV2OptionLibrary


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


def build_support_graph(
    layout_name: str,
    options: list[OptionSpec],
    ce_matrix: np.ndarray,
    eta: float,
    max_factors: int,
    mode_config: dict[str, int] | None = None,
) -> GraphSpec:
    factors = _factor_specs_from_pairs(
        layout_name,
        options,
        _top_pairs_above_eta(ce_matrix, options, eta, max_factors),
        mode_config,
    )
    return make_graph_spec(
        layout_name,
        options,
        factors,
        metadata={
            "graph_variant": "full_support",
            "eta": float(eta),
            "max_factors": int(max_factors),
            "full_max_factors": int(max_factors),
        },
    )


def make_graph_spec(
    layout_name: str,
    options: list[OptionSpec],
    factors: list[FactorSpec],
    *,
    route_source_factors: list[FactorSpec] | None = None,
    relevance_source_factors: list[FactorSpec] | None = None,
    metadata: dict[str, Any] | None = None,
) -> GraphSpec:
    num_options = len(options)
    num_factors = len(factors)
    relevance = np.zeros((num_factors, num_options), dtype=bool)
    route_map: dict[int, tuple[int, ...]] = {}

    for idx, factor in enumerate(factors):
        relevance_factor = (
            relevance_source_factors[idx] if relevance_source_factors is not None else factor
        )
        relevant_options = _relevant_options_for_factor(relevance_factor, options)
        for option_id in relevant_options:
            relevance[idx, option_id] = True

        route_factor = route_source_factors[idx] if route_source_factors is not None else factor
        route_map[idx] = tuple(_relevant_options_for_factor(route_factor, options))

    option_mask = np.ones((num_options,), dtype=bool)
    factor_mask = np.ones((num_factors,), dtype=bool)
    max_modes = max((factor.num_modes for factor in factors), default=0)
    mode_mask = np.zeros((num_factors, max_modes), dtype=bool)
    for idx, factor in enumerate(factors):
        mode_mask[idx, : factor.num_modes] = True

    return GraphSpec(
        layout_name=layout_name,
        options=options,
        factors=factors,
        relevance=relevance,
        option_mask=option_mask,
        factor_mask=factor_mask,
        mode_mask=mode_mask,
        route_map=route_map,
        metadata=metadata or {},
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
) -> GraphSpec:
    full_budget = int(full_max_factors if full_max_factors is not None else max_factors)
    if variant == "full_support":
        graph = full_support_graph(layout_name, options, ce_matrix, eta, full_budget, mode_config)
    elif variant == "overcomplete":
        graph = overcomplete_graph(
            layout_name,
            options,
            ce_matrix,
            eta,
            full_budget,
            mode_config,
            overcomplete_extra_factors=overcomplete_extra_factors,
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
        )
    elif variant == "minus_high_ce":
        graph = minus_high_ce_graph(
            layout_name,
            options,
            ce_matrix,
            eta,
            full_budget,
            mode_config,
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
        )
    elif variant == "complete_option_graph":
        graph = complete_option_graph(layout_name, options, ce_matrix, mode_config)
    elif variant == "shuffled_routes":
        graph = shuffled_routes_graph(layout_name, options, ce_matrix, eta, full_budget, mode_config)
    elif variant == "shuffled_relevance":
        graph = shuffled_relevance_graph(
            layout_name,
            options,
            ce_matrix,
            eta,
            full_budget,
            mode_config,
        )
    else:
        raise ValueError(f"Unknown graph variant {variant!r}; expected one of {GRAPH_VARIANTS}.")

    if require_task_stage_coverage:
        validate_task_stage_coverage(graph)
    return graph


def full_support_graph(
    layout_name: str,
    options: list[OptionSpec],
    ce_matrix: np.ndarray,
    eta: float,
    max_factors: int,
    mode_config: dict[str, int] | None = None,
) -> GraphSpec:
    return build_support_graph(layout_name, options, ce_matrix, eta, max_factors, mode_config)


def overcomplete_graph(
    layout_name: str,
    options: list[OptionSpec],
    ce_matrix: np.ndarray,
    eta: float,
    full_max_factors: int,
    mode_config: dict[str, int] | None = None,
    *,
    overcomplete_extra_factors: int = 0,
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
    )


def minus_critical_graph(
    layout_name: str,
    options: list[OptionSpec],
    ce_matrix: np.ndarray,
    eta: float,
    max_factors: int,
    mode_config: dict[str, int] | None = None,
    criticality_scores: Any | None = None,
) -> GraphSpec:
    if criticality_scores is None:
        raise ValueError("minus_critical requires validation criticality scores.")
    full = build_support_graph(layout_name, options, ce_matrix, eta, max_factors, mode_config)
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
    )


def minus_high_ce_graph(
    layout_name: str,
    options: list[OptionSpec],
    ce_matrix: np.ndarray,
    eta: float,
    max_factors: int,
    mode_config: dict[str, int] | None = None,
) -> GraphSpec:
    full = build_support_graph(layout_name, options, ce_matrix, eta, max_factors, mode_config)
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
    )


def random_same_size_graph(
    layout_name: str,
    options: list[OptionSpec],
    ce_matrix: np.ndarray,
    eta: float,
    max_factors: int,
    mode_config: dict[str, int] | None = None,
    seed: int = 17,
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
    )


def complete_option_graph(
    layout_name: str,
    options: list[OptionSpec],
    ce_matrix: np.ndarray,
    mode_config: dict[str, int] | None = None,
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
    )


def shuffled_routes_graph(
    layout_name: str,
    options: list[OptionSpec],
    ce_matrix: np.ndarray,
    eta: float,
    max_factors: int,
    mode_config: dict[str, int] | None = None,
) -> GraphSpec:
    full = build_support_graph(layout_name, options, ce_matrix, eta, max_factors, mode_config)
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
    )


def shuffled_relevance_graph(
    layout_name: str,
    options: list[OptionSpec],
    ce_matrix: np.ndarray,
    eta: float,
    max_factors: int,
    mode_config: dict[str, int] | None = None,
) -> GraphSpec:
    full = build_support_graph(layout_name, options, ce_matrix, eta, max_factors, mode_config)
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
