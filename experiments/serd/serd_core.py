"""Core metrics for Same-Trajectory Semantic Error Recovery Decomposition.

This module is environment-agnostic. Adapters for Overcooked-AI or JaxMARL
should only need to emit BranchRecord objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf, sqrt
from statistics import mean, stdev
from typing import Dict, Iterable, List, Mapping, Sequence


NumberMap = Mapping[str, float]


@dataclass(frozen=True)
class BranchRecord:
    probe_id: str
    policy: str
    domain: str
    disruption: str
    family: str
    no_shock_return: float
    branch_return: float
    shock_magnitude: float
    phi_pre_h: Dict[str, float]

    @property
    def loss(self) -> float:
        return self.no_shock_return - self.branch_return


@dataclass(frozen=True)
class FamilySerd:
    policy: str
    domain: str
    disruption: str
    family: str
    mean_serd: float
    ci95_low: float
    ci95_high: float
    n: int
    classification: str


@dataclass(frozen=True)
class WorstSerd:
    policy: str
    domain: str
    disruption: str
    mean_serd_worst: float
    ci95_low: float
    ci95_high: float
    n: int
    classification: str
    limiting_family: str


@dataclass(frozen=True)
class BalanceRow:
    policy: str
    domain: str
    disruption: str
    family: str
    covariate: str
    semantic_mean: float
    control_mean: float
    smd: float
    n: int


def normalized_pair_distance(left: NumberMap, right: NumberMap) -> float:
    """Max normalized distance over shared covariates for one matched pair.

    This is a deterministic sanity-stage proxy for the paper's balance table.
    Full experiments should also report standardized mean differences across
    matched probe sets.
    """

    if set(left) != set(right):
        missing_left = sorted(set(right) - set(left))
        missing_right = sorted(set(left) - set(right))
        raise ValueError(
            f"phi_pre_h keys differ; missing_left={missing_left}, "
            f"missing_right={missing_right}"
        )
    distances = []
    for key in left:
        denom = max(abs(float(left[key])), abs(float(right[key])), 1.0)
        distances.append(abs(float(left[key]) - float(right[key])) / denom)
    return max(distances, default=0.0)


def is_valid_match(
    semantic: BranchRecord,
    control: BranchRecord,
    epsilon_shock: float,
    epsilon_phi: float,
) -> bool:
    if semantic.probe_id != control.probe_id:
        return False
    if semantic.policy != control.policy or semantic.domain != control.domain:
        return False
    if semantic.disruption != control.disruption:
        return False
    shock_gap = abs(control.shock_magnitude - semantic.shock_magnitude)
    if shock_gap > epsilon_shock:
        return False
    return normalized_pair_distance(semantic.phi_pre_h, control.phi_pre_h) <= epsilon_phi


def serd_value(semantic: BranchRecord, control: BranchRecord) -> float:
    """SERD_f sample for one matched semantic/control pair.

    Positive values mean the policy loses less under semantic disruption than
    under a matched non-semantic shock.
    """

    return control.loss - semantic.loss


def confidence_interval_95(values: Sequence[float]) -> tuple[float, float, float]:
    if not values:
        return (float("nan"), float("nan"), float("nan"))
    center = mean(values)
    if len(values) == 1:
        return (center, center, center)
    half_width = 1.96 * stdev(values) / sqrt(len(values))
    return (center, center - half_width, center + half_width)


def classify_interval(low: float, high: float, delta: float) -> str:
    if low > delta:
        return "survival"
    if high < -delta:
        return "adverse_semantic_gap"
    if low >= -delta and high <= delta:
        return "collapse"
    return "inconclusive"


def summarize_family_serd(
    semantic_records: Iterable[BranchRecord],
    control_records: Iterable[BranchRecord],
    epsilon_shock: float,
    epsilon_phi: float,
    delta_serd: float,
) -> list[FamilySerd]:
    semantic_by_probe = {record.probe_id: record for record in semantic_records}
    grouped: dict[tuple[str, str, str, str], list[float]] = {}

    for control in control_records:
        semantic = semantic_by_probe.get(control.probe_id)
        if semantic is None:
            continue
        if not is_valid_match(semantic, control, epsilon_shock, epsilon_phi):
            continue
        key = (control.policy, control.domain, control.disruption, control.family)
        grouped.setdefault(key, []).append(serd_value(semantic, control))

    summaries = []
    for (policy, domain, disruption, family), values in sorted(grouped.items()):
        center, low, high = confidence_interval_95(values)
        summaries.append(
            FamilySerd(
                policy=policy,
                domain=domain,
                disruption=disruption,
                family=family,
                mean_serd=center,
                ci95_low=low,
                ci95_high=high,
                n=len(values),
                classification=classify_interval(low, high, delta_serd),
            )
        )
    return summaries


def summarize_worst_serd(
    family_summaries: Iterable[FamilySerd],
    delta_serd: float,
) -> list[WorstSerd]:
    grouped: dict[tuple[str, str, str], list[FamilySerd]] = {}
    for summary in family_summaries:
        key = (summary.policy, summary.domain, summary.disruption)
        grouped.setdefault(key, []).append(summary)

    worst = []
    for (policy, domain, disruption), summaries in sorted(grouped.items()):
        limiting = min(summaries, key=lambda item: item.mean_serd)
        center = limiting.mean_serd
        low = limiting.ci95_low
        high = limiting.ci95_high
        worst.append(
            WorstSerd(
                policy=policy,
                domain=domain,
                disruption=disruption,
                mean_serd_worst=center,
                ci95_low=low,
                ci95_high=high,
                n=limiting.n,
                classification=classify_interval(low, high, delta_serd),
                limiting_family=limiting.family,
            )
        )
    return worst


def standardized_mean_difference(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return float("nan")
    left_mean = mean(left)
    right_mean = mean(right)
    if len(left) < 2 and len(right) < 2:
        return 0.0 if left_mean == right_mean else float("inf")
    left_var = stdev(left) ** 2 if len(left) > 1 else 0.0
    right_var = stdev(right) ** 2 if len(right) > 1 else 0.0
    pooled = sqrt((left_var + right_var) / 2.0)
    if pooled == 0:
        return 0.0 if left_mean == right_mean else float("inf")
    return abs(left_mean - right_mean) / pooled


def summarize_pre_h_balance(
    semantic_records: Iterable[BranchRecord],
    control_records: Iterable[BranchRecord],
    epsilon_shock: float,
    epsilon_phi: float,
) -> list[BalanceRow]:
    semantic_by_probe = {record.probe_id: record for record in semantic_records}
    grouped: dict[tuple[str, str, str, str], list[tuple[BranchRecord, BranchRecord]]] = {}

    for control in control_records:
        semantic = semantic_by_probe.get(control.probe_id)
        if semantic is None:
            continue
        if not is_valid_match(semantic, control, epsilon_shock, epsilon_phi):
            continue
        key = (control.policy, control.domain, control.disruption, control.family)
        grouped.setdefault(key, []).append((semantic, control))

    rows: list[BalanceRow] = []
    for (policy, domain, disruption, family), pairs in sorted(grouped.items()):
        if not pairs:
            continue
        covariates = sorted(pairs[0][0].phi_pre_h.keys())
        for covariate in covariates:
            semantic_values = [pair[0].phi_pre_h[covariate] for pair in pairs]
            control_values = [pair[1].phi_pre_h[covariate] for pair in pairs]
            rows.append(
                BalanceRow(
                    policy=policy,
                    domain=domain,
                    disruption=disruption,
                    family=family,
                    covariate=covariate,
                    semantic_mean=mean(semantic_values),
                    control_mean=mean(control_values),
                    smd=standardized_mean_difference(semantic_values, control_values),
                    n=len(pairs),
                )
            )
    return rows


def require_family_coverage(
    family_summaries: Iterable[FamilySerd],
    required_families: Sequence[str],
) -> list[str]:
    seen = {summary.family for summary in family_summaries}
    return [family for family in required_families if family not in seen]


def finite_or_none(value: float) -> float | None:
    if value == inf or value == -inf or value != value:
        return None
    return value
