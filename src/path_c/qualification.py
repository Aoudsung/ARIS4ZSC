"""Immutable C0--C5 qualification records and paired confidence bounds."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Iterable, Mapping

import numpy as np


CONTRACT_NAMES = ("C0", "C1", "C2", "C3", "C4", "C5")


@dataclass(frozen=True, slots=True)
class PairedMeanCVARStatistics:
    mean_point: float
    mean_lcb: float
    mean_ucb: float
    cvar_point: float
    cvar_lcb: float
    cvar_ucb: float
    run_block_count: int


@dataclass(frozen=True, slots=True)
class GateDecision:
    passed: bool
    statistic: float
    lower_bound: float
    upper_bound: float
    sample_count: int
    random_domain: str
    artifact_fingerprint: str
    reason: str = ""

    def __post_init__(self) -> None:
        if self.sample_count < 0:
            raise ValueError("Qualification sample_count cannot be negative.")
        if self.lower_bound > self.upper_bound:
            raise ValueError("Qualification confidence interval is reversed.")
        if not self.random_domain:
            raise ValueError("Qualification random domain must be explicit.")
        if not self.artifact_fingerprint:
            raise ValueError("Qualification artifact fingerprint is required.")


@dataclass(frozen=True, slots=True)
class SignalQualification:
    decisions: Mapping[str, GateDecision] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = set(self.decisions).difference(CONTRACT_NAMES)
        if unknown:
            raise ValueError(f"Unknown signal contracts: {sorted(unknown)}")
        seen_failure = False
        for name in CONTRACT_NAMES:
            decision = self.decisions.get(name)
            if decision is None:
                continue
            if seen_failure and decision.passed:
                raise ValueError(f"{name} cannot pass after an earlier failed contract.")
            seen_failure = seen_failure or not decision.passed

    def with_decision(self, name: str, decision: GateDecision) -> "SignalQualification":
        if name not in CONTRACT_NAMES:
            raise ValueError(f"Unknown signal contract: {name}")
        index = CONTRACT_NAMES.index(name)
        for prerequisite in CONTRACT_NAMES[:index]:
            previous = self.decisions.get(prerequisite)
            if previous is None or not previous.passed:
                raise ValueError(f"{name} requires passed {prerequisite}.")
        updated = dict(self.decisions)
        updated[name] = decision
        for later in CONTRACT_NAMES[index + 1 :]:
            updated.pop(later, None)
        return SignalQualification(updated)

    def passed(self, name: str) -> bool:
        decision = self.decisions.get(name)
        return bool(decision is not None and decision.passed)

    @property
    def highest_passed(self) -> str | None:
        highest = None
        for name in CONTRACT_NAMES:
            if not self.passed(name):
                break
            highest = name
        return highest

    @property
    def fingerprint(self) -> str:
        payload = {
            name: asdict(self.decisions[name]) for name in sorted(self.decisions)
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def to_mapping(self) -> Mapping[str, Any]:
        return {
            "decisions": {name: asdict(value) for name, value in self.decisions.items()},
            "highest_passed": self.highest_passed,
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SignalQualification":
        raw = value.get("decisions", value)
        if not isinstance(raw, Mapping):
            raise ValueError("Qualification mapping is malformed.")
        decisions = {
            str(name): (
                decision
                if isinstance(decision, GateDecision)
                else GateDecision(**dict(decision))
            )
            for name, decision in raw.items()
            if name in CONTRACT_NAMES
        }
        return cls(decisions)


def bootstrap_interval(
    values: Iterable[float],
    *,
    seed: int,
    replicates: int = 9_999,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Deterministic one-sample bootstrap interval for independent run blocks."""

    sample = np.asarray(tuple(values), dtype=np.float64)
    if sample.ndim != 1 or sample.size < 2 or not np.all(np.isfinite(sample)):
        raise ValueError("At least two finite independent block values are required.")
    if replicates <= 0 or not 0.0 < alpha < 0.5:
        raise ValueError("Invalid bootstrap registration.")
    rng = np.random.default_rng(int(seed))
    indexes = rng.integers(0, sample.size, size=(int(replicates), sample.size))
    estimates = np.mean(sample[indexes], axis=1)
    return (
        float(np.mean(sample)),
        float(np.quantile(estimates, alpha)),
        float(np.quantile(estimates, 1.0 - alpha)),
    )


def paired_noninferiority_decision(
    paired_differences: Iterable[float],
    *,
    margin: float,
    seed: int,
    random_domain: str,
    artifact_fingerprint: str,
) -> GateDecision:
    point, lower, upper = bootstrap_interval(paired_differences, seed=seed)
    values = tuple(paired_differences)
    return GateDecision(
        passed=lower >= -float(margin),
        statistic=point,
        lower_bound=lower,
        upper_bound=upper,
        sample_count=len(values),
        random_domain=random_domain,
        artifact_fingerprint=artifact_fingerprint,
        reason=f"one-sided LCB must be >= {-float(margin):.6g}",
    )


def paired_mean_cvar_noninferiority_decision(
    paired_episode_differences: Any,
    *,
    partner_run_ids: Iterable[str | int],
    margin: float,
    cvar_level: float,
    seed: int,
    random_domain: str,
    artifact_fingerprint: str,
) -> GateDecision:
    """C0/generator paired mean and lower-tail CVaR run-block decision."""

    statistics = paired_mean_cvar_statistics(
        paired_episode_differences,
        partner_run_ids=partner_run_ids,
        cvar_level=cvar_level,
        seed=seed,
    )
    return GateDecision(
        passed=(
            statistics.mean_lcb >= -float(margin)
            and statistics.cvar_lcb >= -float(margin)
        ),
        statistic=min(statistics.mean_point, statistics.cvar_point),
        lower_bound=min(statistics.mean_lcb, statistics.cvar_lcb),
        upper_bound=min(statistics.mean_ucb, statistics.cvar_ucb),
        sample_count=statistics.run_block_count,
        random_domain=random_domain,
        artifact_fingerprint=artifact_fingerprint,
        reason=(
            f"paired mean LCB={statistics.mean_lcb:.6g} and "
            f"CVaR{100*cvar_level:.0f} LCB={statistics.cvar_lcb:.6g} "
            f"must both be >= {-float(margin):.6g}"
        ),
    )


def paired_mean_cvar_statistics(
    paired_episode_differences: Any,
    *,
    partner_run_ids: Iterable[str | int],
    cvar_level: float,
    seed: int,
) -> PairedMeanCVARStatistics:
    """Return both registered run-block bounds without collapsing their meaning."""

    values = np.asarray(paired_episode_differences, dtype=np.float64)
    ids = np.asarray(tuple(partner_run_ids))
    if values.ndim != 1 or ids.shape != values.shape or not np.all(np.isfinite(values)):
        raise ValueError("Paired episode differences and partner blocks do not align.")
    blocks = np.unique(ids)
    if blocks.size < 2 or not 0.0 < cvar_level <= 1.0:
        raise ValueError("Paired noninferiority requires independent run blocks.")
    mean_blocks = []
    cvar_blocks = []
    for block in blocks:
        sample = np.sort(values[ids == block])
        count = max(int(np.ceil(float(cvar_level) * sample.size)), 1)
        mean_blocks.append(float(np.mean(sample)))
        cvar_blocks.append(float(np.mean(sample[:count])))
    mean_point, mean_lcb, mean_ucb = bootstrap_interval(mean_blocks, seed=seed)
    cvar_point, cvar_lcb, cvar_ucb = bootstrap_interval(cvar_blocks, seed=seed + 1)
    return PairedMeanCVARStatistics(
        mean_point=mean_point,
        mean_lcb=mean_lcb,
        mean_ucb=mean_ucb,
        cvar_point=cvar_point,
        cvar_lcb=cvar_lcb,
        cvar_ucb=cvar_ucb,
        run_block_count=int(blocks.size),
    )


def compound_gate_decision(
    *,
    passed: bool,
    statistic: float,
    lower_bound: float,
    upper_bound: float,
    sample_count: int,
    random_domain: str,
    artifact_fingerprint: str,
    reason: str,
) -> GateDecision:
    """Create an auditable C1--C5 decision from its preregistered subgates."""

    values = (statistic, lower_bound, upper_bound)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("Signal-contract statistics must be finite.")
    return GateDecision(
        passed=bool(passed),
        statistic=float(statistic),
        lower_bound=float(lower_bound),
        upper_bound=float(upper_bound),
        sample_count=int(sample_count),
        random_domain=str(random_domain),
        artifact_fingerprint=str(artifact_fingerprint),
        reason=str(reason),
    )


__all__ = [
    "CONTRACT_NAMES",
    "GateDecision",
    "PairedMeanCVARStatistics",
    "SignalQualification",
    "bootstrap_interval",
    "compound_gate_decision",
    "paired_noninferiority_decision",
    "paired_mean_cvar_noninferiority_decision",
    "paired_mean_cvar_statistics",
]
