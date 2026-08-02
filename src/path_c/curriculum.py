"""Fail-closed r3 signal-contract curriculum.

The curriculum is deliberately a finite state machine.  No return curve or
training loss can advance it; only recorded C0--C5 qualification decisions may
do so.  A failed contract preserves the formal seed and selects the strongest
already-qualified controller.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Mapping


class CurriculumPhase(IntEnum):
    BOOTSTRAP_BASE = 0
    ESTABLISH_DECISION_SUPPORT = 1
    LEARN_RAW_Q = 2
    PASSIVE_ADAPTATION = 3
    GENERATOR_EXPANSION = 4
    ACTIVE_INFORMATION = 5
    FULL_DELTA = 6
    SAFE_FALLBACK = 7


CONTRACT_FOR_PHASE: Mapping[CurriculumPhase, str] = {
    CurriculumPhase.BOOTSTRAP_BASE: "C0",
    CurriculumPhase.ESTABLISH_DECISION_SUPPORT: "C1",
    CurriculumPhase.LEARN_RAW_Q: "C2",
    CurriculumPhase.PASSIVE_ADAPTATION: "C3_C4",
    CurriculumPhase.GENERATOR_EXPANSION: "GENERATOR_ADMISSION",
    CurriculumPhase.ACTIVE_INFORMATION: "C5",
}


def next_phase(phase: CurriculumPhase, *, contract_passed: bool) -> CurriculumPhase:
    """Advance exactly one phase; a failed interim audit remains fail-closed.

    Qualification may be re-audited at a later preregistered milestone.  A
    failure therefore keeps the current phase with all downstream mechanisms
    disabled.  Only finalization chooses the deployment fallback tier.
    """

    current = CurriculumPhase(phase)
    if current in (CurriculumPhase.FULL_DELTA, CurriculumPhase.SAFE_FALLBACK):
        return current
    if not contract_passed:
        return current
    return CurriculumPhase(int(current) + 1)


def decision_regret_enabled(phase: CurriculumPhase) -> bool:
    return CurriculumPhase(phase) >= CurriculumPhase.FULL_DELTA and phase != CurriculumPhase.SAFE_FALLBACK


def conditional_control_enabled(phase: CurriculumPhase) -> bool:
    return CurriculumPhase.PASSIVE_ADAPTATION < CurriculumPhase(phase) < CurriculumPhase.SAFE_FALLBACK


def generator_diversity_enabled(phase: CurriculumPhase) -> bool:
    return CurriculumPhase(phase) >= CurriculumPhase.GENERATOR_EXPANSION and phase != CurriculumPhase.SAFE_FALLBACK


def phase_from_qualification(qualification: Any, *, generator_admitted: bool) -> CurriculumPhase:
    """Derive the only legal runtime phase from immutable contract records."""

    if not qualification.passed("C0"):
        return CurriculumPhase.BOOTSTRAP_BASE
    if not qualification.passed("C1"):
        return CurriculumPhase.ESTABLISH_DECISION_SUPPORT
    if not qualification.passed("C2"):
        return CurriculumPhase.LEARN_RAW_Q
    if not (qualification.passed("C3") and qualification.passed("C4")):
        return CurriculumPhase.PASSIVE_ADAPTATION
    if not generator_admitted:
        return CurriculumPhase.GENERATOR_EXPANSION
    if not qualification.passed("C5"):
        return CurriculumPhase.ACTIVE_INFORMATION
    return CurriculumPhase.FULL_DELTA


__all__ = [
    "CONTRACT_FOR_PHASE",
    "CurriculumPhase",
    "conditional_control_enabled",
    "decision_regret_enabled",
    "generator_diversity_enabled",
    "next_phase",
    "phase_from_qualification",
]
