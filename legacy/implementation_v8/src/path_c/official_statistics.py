"""Registered statistics for the two formal DEPI scoreboards."""

from __future__ import annotations

from collections import defaultdict
import hashlib
from itertools import permutations
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


OFFICIAL_RUNS = 10
OFFICIAL_EPISODES = 500
OFFICIAL_BOOTSTRAP_REPLICATES = 9_999


def registered_superiority_gate(
    comparison: Mapping[str, Any],
    *,
    lcb_threshold: float,
    minimum_effect: float,
    minimum_effect_rule: str,
) -> Mapping[str, bool]:
    """Evaluate the common registered LCB and material-effect conjunction."""

    lcb = float(comparison["one_sided_lcb"])
    point = float(comparison["point_reference"])
    if not np.isfinite(lcb) or not np.isfinite(point):
        raise ValueError("Superiority comparison contains a non-finite statistic.")
    lcb_passed = lcb > float(lcb_threshold)
    if minimum_effect_rule == "point_estimate":
        effect_passed = point >= float(minimum_effect)
    elif minimum_effect_rule == "lower_confidence_bound":
        effect_passed = lcb >= float(minimum_effect)
    else:
        raise ValueError("Unknown formal minimum-effect rule.")
    return {
        "passed": bool(lcb_passed and effect_passed),
        "superiority_lcb_passed": bool(lcb_passed),
        "minimum_effect_passed": bool(effect_passed),
    }


def registered_bootstrap_seed(*labels: str) -> int:
    """Derive a non-tunable bootstrap seed from the frozen protocol identity."""

    payload = "overcooked_v2_iclr2025_5ce1707_v1\0" + "\0".join(labels)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "big")


def official_pairings(run_count: int = OFFICIAL_RUNS) -> tuple[tuple[int, int], ...]:
    """Return Official order: all directed XP cells, then all SP cells."""

    if int(run_count) <= 1:
        raise ValueError("Official cross-play requires at least two runs.")
    indexes = tuple(range(int(run_count)))
    return (*tuple(permutations(indexes, 2)), *((index, index) for index in indexes))


def validate_official_return_cube(
    returns: Any,
    *,
    run_count: int = OFFICIAL_RUNS,
    episodes: int = OFFICIAL_EPISODES,
) -> np.ndarray:
    array = np.asarray(returns, dtype=np.float64)
    expected = (int(run_count), int(run_count), int(episodes))
    if array.shape != expected:
        raise ValueError(f"Official return cube must have shape {expected}, got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Official return cube contains non-finite values.")
    return array


def official_scoreboard_summary(returns: Any) -> Mapping[str, Any]:
    cube = validate_official_return_cube(returns)
    cell_means = np.mean(cube, axis=-1)
    diagonal = np.diag(cell_means)
    off_diagonal = cell_means[~np.eye(cell_means.shape[0], dtype=bool)]
    sp = float(np.mean(diagonal))
    xp = float(np.mean(off_diagonal))
    row_xp = np.asarray(
        [np.mean(np.delete(cell_means[index], index)) for index in range(10)]
    )
    column_xp = np.asarray(
        [np.mean(np.delete(cell_means[:, index], index)) for index in range(10)]
    )
    return {
        "sp_mean": sp,
        "sp_population_sd": float(np.std(diagonal, ddof=0)),
        "xp_mean": xp,
        "xp_population_sd": float(np.std(off_diagonal, ddof=0)),
        "gap_point": sp - xp,
        "cell_means": cell_means.tolist(),
        "row_xp_means": row_xp.tolist(),
        "column_xp_means": column_xp.tolist(),
        "worst_run": float(min(np.min(row_xp), np.min(column_xp))),
        "worst_ordered_pairing": float(np.min(off_diagonal)),
        "median_ordered_pairing": float(np.median(off_diagonal)),
        "ordered_pairing_cvar_10": float(
            np.mean(np.sort(off_diagonal)[: max(1, int(np.ceil(0.1 * len(off_diagonal))))])
        ),
    }


def _positive_node_weights(generator: np.random.Generator, count: int) -> np.ndarray:
    while True:
        draw = generator.integers(0, count, size=count)
        weights = np.bincount(draw, minlength=count).astype(np.float64)
        denominator = np.sum(np.outer(weights, weights)) - np.sum(weights * weights)
        if denominator > 0.0:
            return weights


def _weighted_xp(cell_means: np.ndarray, weights: np.ndarray) -> float:
    pair_weights = np.outer(weights, weights)
    np.fill_diagonal(pair_weights, 0.0)
    return float(np.sum(pair_weights * cell_means) / np.sum(pair_weights))


def _weighted_sp(cell_means: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(weights * np.diag(cell_means)) / np.sum(weights))


def official_node_bootstrap(
    method_returns: Mapping[str, Any],
    *,
    target_method: str,
    baseline_methods: Sequence[str],
    fcp_method: str,
    replicates: int = OFFICIAL_BOOTSTRAP_REPLICATES,
    seed: int,
    alpha: float = 0.05,
    inference_mode: str = "independent_run",
) -> Mapping[str, Any]:
    """Node bootstrap with shared episode-index resampling.

    ``independent_run`` samples training nodes independently between methods.
    ``paired_run`` uses one shared node-weight vector for all methods and is
    valid only when run indexes were preregistered as paired units.  Both modes
    share the episode-index vector, preserving the common environment keys.
    """

    names = (target_method, *tuple(baseline_methods))
    if len(set(names)) != len(names):
        raise ValueError("DEPI and baseline method names must be distinct.")
    if fcp_method not in baseline_methods:
        raise ValueError("The competence comparator must be one of the baselines.")
    missing = set(names) - set(method_returns)
    if missing:
        raise ValueError(f"Missing method return cubes: {sorted(missing)}")
    cubes = {name: validate_official_return_cube(method_returns[name]) for name in names}
    if int(replicates) <= 0 or not 0.0 < float(alpha) < 1.0:
        raise ValueError("Bootstrap replicates and alpha are invalid.")
    mode = str(inference_mode)
    if mode not in {"independent_run", "paired_run"}:
        raise ValueError("inference_mode must be independent_run or paired_run.")

    generator = np.random.default_rng(int(seed))
    deltas = np.empty(int(replicates), dtype=np.float64)
    competence = np.empty(int(replicates), dtype=np.float64)
    xp_samples = {name: np.empty(int(replicates), dtype=np.float64) for name in names}
    sp_samples = {name: np.empty(int(replicates), dtype=np.float64) for name in names}
    for replicate in range(int(replicates)):
        episode_indexes = generator.integers(0, OFFICIAL_EPISODES, size=OFFICIAL_EPISODES)
        shared_weights = (
            _positive_node_weights(generator, OFFICIAL_RUNS)
            if mode == "paired_run"
            else None
        )
        for name in names:
            weights = (
                shared_weights
                if shared_weights is not None
                else _positive_node_weights(generator, OFFICIAL_RUNS)
            )
            means = np.mean(cubes[name][:, :, episode_indexes], axis=-1)
            xp_samples[name][replicate] = _weighted_xp(means, weights)
            sp_samples[name][replicate] = _weighted_sp(means, weights)
        deltas[replicate] = xp_samples[target_method][replicate] - max(
            xp_samples[name][replicate] for name in baseline_methods
        )
        competence[replicate] = (
            sp_samples[target_method][replicate] - sp_samples[fcp_method][replicate]
        )
    return {
        "replicates": int(replicates),
        "seed": int(seed),
        "alpha": float(alpha),
        "inference_mode": mode,
        "depi_vs_best_baseline": {
            "point_reference": float(
                official_scoreboard_summary(cubes[target_method])["xp_mean"]
                - max(
                    official_scoreboard_summary(cubes[name])["xp_mean"]
                    for name in baseline_methods
                )
            ),
            "one_sided_lcb": float(np.quantile(deltas, alpha, method="lower")),
        },
        "depi_sp_minus_fcp_sp": {
            "point_reference": float(
                official_scoreboard_summary(cubes[target_method])["sp_mean"]
                - official_scoreboard_summary(cubes[fcp_method])["sp_mean"]
            ),
            "one_sided_lcb": float(np.quantile(competence, alpha, method="lower")),
        },
    }


def official_two_method_bootstrap(
    left_returns: Any,
    right_returns: Any,
    *,
    left_name: str,
    right_name: str,
    replicates: int = OFFICIAL_BOOTSTRAP_REPLICATES,
    seed: int,
    alpha: float = 0.05,
) -> Mapping[str, Any]:
    """Compare two independently trained method populations on Official XP.

    Each method receives an independent node resample because run indexes do
    not denote paired training units.  The episode-index resample is shared,
    preserving the registered common environment-key condition.
    """

    if left_name == right_name:
        raise ValueError("Two-method bootstrap labels must differ.")
    if int(replicates) <= 0 or not 0.0 < float(alpha) < 1.0:
        raise ValueError("Bootstrap replicates and alpha are invalid.")
    left = validate_official_return_cube(left_returns)
    right = validate_official_return_cube(right_returns)
    generator = np.random.default_rng(int(seed))
    differences = np.empty(int(replicates), dtype=np.float64)
    for replicate in range(int(replicates)):
        episode_indexes = generator.integers(
            0, OFFICIAL_EPISODES, size=OFFICIAL_EPISODES
        )
        left_means = np.mean(left[:, :, episode_indexes], axis=-1)
        right_means = np.mean(right[:, :, episode_indexes], axis=-1)
        left_score = _weighted_xp(
            left_means, _positive_node_weights(generator, OFFICIAL_RUNS)
        )
        right_score = _weighted_xp(
            right_means, _positive_node_weights(generator, OFFICIAL_RUNS)
        )
        differences[replicate] = left_score - right_score
    return {
        "replicates": int(replicates),
        "seed": int(seed),
        "alpha": float(alpha),
        "left_method": str(left_name),
        "right_method": str(right_name),
        "point_reference": float(
            official_scoreboard_summary(left)["xp_mean"]
            - official_scoreboard_summary(right)["xp_mean"]
        ),
        "one_sided_lcb": float(
            np.quantile(differences, alpha, method="lower")
        ),
    }


def rows_to_official_cube(
    rows: Iterable[Mapping[str, Any]],
    *,
    run_count: int = OFFICIAL_RUNS,
    episodes: int = OFFICIAL_EPISODES,
) -> np.ndarray:
    cube = np.full((run_count, run_count, episodes), np.nan, dtype=np.float64)
    seen: set[tuple[int, int, int]] = set()
    for row in rows:
        index = (
            int(row["left_run_index"]),
            int(row["right_run_index"]),
            int(row["episode_index"]),
        )
        if index in seen:
            raise ValueError(f"Duplicate Official episode row: {index}")
        seen.add(index)
        cube[index] = float(row["raw_return"])
    return validate_official_return_cube(cube, run_count=run_count, episodes=episodes)


def common_partner_summary(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_ego_runs: int = 10,
    expected_mechanisms: int = 4,
    expected_partners_per_mechanism: int = 4,
    expected_partner_counts: Mapping[str, int] | None = None,
    expected_roles: int = 2,
    expected_episodes: int = 500,
) -> Mapping[str, Any]:
    """Summarize a method on a shared, mechanism-stratified partner panel."""

    grouped: dict[str, dict[str, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    ego_ids: set[str] = set()
    row_count = 0
    negative = []
    for row in rows:
        mechanism = str(row["partner_mechanism"])
        partner = str(row["partner_run_id"])
        ego = str(row["ego_run_id"])
        role = int(row["ego_role"])
        grouped[mechanism][partner][role].append(float(row["raw_return"]))
        ego_ids.add(ego)
        row_count += 1
        if "base_raw_return" in row:
            negative.append(float(row["raw_return"]) < float(row["base_raw_return"]))

    if len(ego_ids) != expected_ego_runs:
        raise ValueError(f"Expected {expected_ego_runs} ego runs, got {len(ego_ids)}.")
    expected_mechanism_count = (
        len(expected_partner_counts)
        if expected_partner_counts is not None
        else int(expected_mechanisms)
    )
    if len(grouped) != expected_mechanism_count:
        raise ValueError(
            f"Expected {expected_mechanism_count} partner mechanisms, got {len(grouped)}."
        )
    if expected_partner_counts is not None and set(grouped) != set(
        expected_partner_counts
    ):
        raise ValueError("Common-Partner mechanisms differ from the frozen panel.")
    expected_per_cell = expected_ego_runs * expected_episodes
    mechanism_means: dict[str, float] = {}
    partner_means = []
    for mechanism, partners in grouped.items():
        expected_partner_count = (
            int(expected_partner_counts[mechanism])
            if expected_partner_counts is not None
            else int(expected_partners_per_mechanism)
        )
        if len(partners) != expected_partner_count:
            raise ValueError(
                f"Mechanism {mechanism} has {len(partners)} partners, expected "
                f"{expected_partner_count}."
            )
        current = []
        for partner, roles in partners.items():
            if set(roles) != set(range(expected_roles)):
                raise ValueError(f"Partner {partner} does not cover both ego roles.")
            for role, values in roles.items():
                if len(values) != expected_per_cell:
                    raise ValueError(
                        f"Partner {partner}, role {role} has {len(values)} rows; "
                        f"expected {expected_per_cell}."
                    )
            partner_mean = float(np.mean([value for values in roles.values() for value in values]))
            current.append(partner_mean)
            partner_means.append(partner_mean)
        mechanism_means[mechanism] = float(np.mean(current))
    tail = max(1, int(np.ceil(0.1 * len(partner_means))))
    return {
        "row_count": row_count,
        "mean_common_partner_return": float(np.mean(list(mechanism_means.values()))),
        "mechanism_means": mechanism_means,
        "worst_mechanism_return": float(min(mechanism_means.values())),
        "partner_cvar_10": float(np.mean(np.sort(partner_means)[:tail])),
        "negative_transfer_rate": (
            None if not negative else float(np.mean(np.asarray(negative, dtype=np.float64)))
        ),
    }


def _common_array(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    materialized = list(rows)
    ego_ids = tuple(sorted({str(row["ego_run_id"]) for row in materialized}))
    partner_ids = tuple(sorted({str(row["partner_run_id"]) for row in materialized}))
    mechanisms_by_partner = {
        str(row["partner_run_id"]): str(row["partner_mechanism"])
        for row in materialized
    }
    if len(ego_ids) != 10 or len(partner_ids) not in {16, 18}:
        raise ValueError("Formal common panel requires 10 egos and 16 or 18 partners.")
    mechanisms = tuple(sorted(set(mechanisms_by_partner.values())))
    observed_counts = {
        mechanism: sum(
            mechanisms_by_partner[partner] == mechanism for partner in partner_ids
        )
        for mechanism in mechanisms
    }
    expected_counts = {
        **{mechanism: 4 for mechanism in mechanisms if mechanism != "heuristic"},
        **({"heuristic": 2} if "heuristic" in mechanisms else {}),
    }
    if observed_counts != expected_counts:
        raise ValueError(
            "Common-panel family counts differ from four trained runs per "
            "mechanism plus two heuristic policies."
        )
    ego_index = {value: index for index, value in enumerate(ego_ids)}
    partner_index = {value: index for index, value in enumerate(partner_ids)}
    array = np.full((10, len(partner_ids), 2, 500), np.nan, dtype=np.float64)
    for row in materialized:
        index = (
            ego_index[str(row["ego_run_id"])],
            partner_index[str(row["partner_run_id"])],
            int(row["ego_role"]),
            int(row["episode_index"]),
        )
        if np.isfinite(array[index]):
            raise ValueError(f"Duplicate common-partner episode row: {index}")
        array[index] = float(row["raw_return"])
    if not np.all(np.isfinite(array)):
        raise ValueError("Common-partner return array is incomplete.")
    partner_mechanisms = tuple(mechanisms_by_partner[p] for p in partner_ids)
    return array, ego_ids, partner_ids, partner_mechanisms


def common_partner_bootstrap(
    method_rows: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    target_method: str,
    baseline_methods: Sequence[str],
    replicates: int = OFFICIAL_BOOTSTRAP_REPLICATES,
    seed: int,
    alpha: float = 0.05,
) -> Mapping[str, Any]:
    """Stratified common-panel bootstrap with matched episode resampling."""

    names = (target_method, *tuple(baseline_methods))
    rows_by_method = {name: list(method_rows[name]) for name in names}
    parsed = {name: _common_array(rows_by_method[name]) for name in names}
    partner_ids = parsed[target_method][2]
    partner_mechanisms = parsed[target_method][3]
    for name in baseline_methods:
        if parsed[name][2:] != parsed[target_method][2:]:
            raise ValueError("All methods must face the identical common partner panel.")
    by_mechanism = {
        mechanism: np.asarray(
            [index for index, value in enumerate(partner_mechanisms) if value == mechanism],
            dtype=np.int64,
        )
        for mechanism in sorted(set(partner_mechanisms))
    }
    generator = np.random.default_rng(int(seed))
    deltas = np.empty(int(replicates), dtype=np.float64)
    for replicate in range(int(replicates)):
        episode_indexes = generator.integers(0, 500, size=500)
        sampled_partners = {
            mechanism: generator.choice(indexes, size=len(indexes), replace=True)
            for mechanism, indexes in by_mechanism.items()
        }
        scores = {}
        for name in names:
            ego_indexes = generator.integers(0, 10, size=10)
            array = parsed[name][0]
            mechanism_means = []
            for mechanism, indexes in sampled_partners.items():
                values = array[ego_indexes][:, indexes, :, :]
                values = values[..., episode_indexes]
                mechanism_means.append(float(np.mean(values)))
            scores[name] = float(np.mean(mechanism_means))
        deltas[replicate] = scores[target_method] - max(
            scores[name] for name in baseline_methods
        )
    expected_partner_counts = {
        mechanism: len(indexes) for mechanism, indexes in by_mechanism.items()
    }
    point = {
        name: common_partner_summary(
            rows_by_method[name],
            expected_partner_counts=expected_partner_counts,
        )[
            "mean_common_partner_return"
        ]
        for name in names
    }
    return {
        "replicates": int(replicates),
        "seed": int(seed),
        "alpha": float(alpha),
        "point_reference": float(point[target_method] - max(point[name] for name in baseline_methods)),
        "one_sided_lcb": float(np.quantile(deltas, alpha, method="lower")),
    }


__all__ = [
    "OFFICIAL_BOOTSTRAP_REPLICATES",
    "OFFICIAL_EPISODES",
    "OFFICIAL_RUNS",
    "common_partner_summary",
    "common_partner_bootstrap",
    "official_node_bootstrap",
    "official_two_method_bootstrap",
    "official_pairings",
    "official_scoreboard_summary",
    "registered_superiority_gate",
    "rows_to_official_cube",
    "registered_bootstrap_seed",
    "validate_official_return_cube",
]
