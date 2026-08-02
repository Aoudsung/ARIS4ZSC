"""Formal-seed-preserving deployment selection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from .qualification import SignalQualification


class DeploymentTier(str, Enum):
    OWNER_SP_FALLBACK = "owner_sp_source_fallback"
    QUALIFIED_ROBUST_BASE = "qualified_robust_base"
    CALIBRATED_PASSIVE_CONDITIONAL = "calibrated_passive_conditional_delta"
    CALIBRATED_FULL_ACTIVE = "calibrated_full_active_delta"


def deployment_tier(qualification: SignalQualification) -> DeploymentTier:
    if not qualification.passed("C0"):
        return DeploymentTier.OWNER_SP_FALLBACK
    if not (qualification.passed("C3") and qualification.passed("C4")):
        return DeploymentTier.QUALIFIED_ROBUST_BASE
    if not qualification.passed("C5"):
        return DeploymentTier.CALIBRATED_PASSIVE_CONDITIONAL
    return DeploymentTier.CALIBRATED_FULL_ACTIVE


@dataclass(frozen=True, slots=True)
class FallbackArtifactIdentity:
    seed_index: int
    tier: DeploymentTier
    qualification_fingerprint: str
    source_fingerprint: str
    formal_seed_preserved: bool = True

    def to_mapping(self) -> Mapping[str, object]:
        if not 0 <= int(self.seed_index) <= 9:
            raise ValueError("Formal seed indexes must remain 0..9.")
        if not self.formal_seed_preserved:
            raise ValueError("r3 never drops a formal seed after qualification failure.")
        return {
            "seed_index": int(self.seed_index),
            "deployment_tier": self.tier.value,
            "qualification_fingerprint": self.qualification_fingerprint,
            "source_fingerprint": self.source_fingerprint,
            "formal_seed_preserved": True,
        }


def assert_all_formal_seeds(artifacts: Mapping[int, object]) -> None:
    expected = set(range(10))
    actual = {int(index) for index in artifacts}
    if actual != expected:
        raise ValueError(
            f"Formal deployment set must contain exactly seeds 0..9; missing={sorted(expected-actual)}, extra={sorted(actual-expected)}."
        )


__all__ = [
    "DeploymentTier",
    "FallbackArtifactIdentity",
    "assert_all_formal_seeds",
    "deployment_tier",
]
