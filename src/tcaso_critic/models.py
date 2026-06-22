from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class CertificateStatus(str, Enum):
    DRAFT = "DRAFT"
    CERTIFIED = "CERTIFIED"
    REJECTED = "REJECTED"
    OBSTRUCTED = "OBSTRUCTED"


class MatchingResult(str, Enum):
    EXACT_MATCH = "EXACT_MATCH"
    NO_EXACT_MATCH = "NO_EXACT_MATCH"
    NOT_EVALUATED = "NOT_EVALUATED"


class ControlType(str, Enum):
    TASK_DETERMINED_NEGATIVE = "TASK_DETERMINED_NEGATIVE"
    NO_CLASS_DISRUPTION_NEGATIVE = "NO_CLASS_DISRUPTION_NEGATIVE"
    NONE = "NONE"


class ProbeClass(str, Enum):
    STRUCTURAL_POSITIVE_CANDIDATE = "STRUCTURAL_POSITIVE_CANDIDATE"
    TASK_DETERMINED_NEGATIVE_CONTROL = "TASK_DETERMINED_NEGATIVE_CONTROL"
    NO_CLASS_DISRUPTION_NEGATIVE_CONTROL = "NO_CLASS_DISRUPTION_NEGATIVE_CONTROL"
    VALID_SEMANTIC_RECOVERY_PROBE = "VALID_SEMANTIC_RECOVERY_PROBE"
    UNMATCHED_STRUCTURAL_POSITIVE = "UNMATCHED_STRUCTURAL_POSITIVE"
    REJECTED_LEAKAGE = "REJECTED_LEAKAGE"


class SameOverrideClaimLevel(str, Enum):
    SAME_OVERRIDE_TASK_DETERMINED_COUNTEREXAMPLE = "SAME_OVERRIDE_TASK_DETERMINED_COUNTEREXAMPLE"
    SAME_OVERRIDE_NO_CLASS_DISRUPTION_DIAGNOSTIC = "SAME_OVERRIDE_NO_CLASS_DISRUPTION_DIAGNOSTIC"
    UNMATCHED_STRUCTURAL_POSITIVE = "UNMATCHED_STRUCTURAL_POSITIVE"
    REJECTED_LEAKAGE = "REJECTED_LEAKAGE"


class ConstructSplitType(str, Enum):
    UNDERDETERMINED_DISRUPTION_VS_TASK_DETERMINED = "UNDERDETERMINED_DISRUPTION_VS_TASK_DETERMINED"
    UNDERDETERMINED_NO_CLASS_DISRUPTION = "UNDERDETERMINED_NO_CLASS_DISRUPTION"
    UNMATCHED = "UNMATCHED"
    LEAKAGE_REJECTED = "LEAKAGE_REJECTED"


@dataclass(frozen=True)
class Override:
    agent_i: int
    action: int
    channel_c: str = "primitive_action"

    def key(self) -> str:
        return f"agent_{self.agent_i}:{self.channel_c}:{self.action}"


@dataclass(frozen=True)
class TaskSpec:
    tau_id: str
    task_type: str
    params: Mapping[str, Any]


@dataclass(frozen=True)
class PublicState:
    layout_id: str
    L: Any
    R: Any
    I: Any
    C: Any
    W_dyn: Any
    C_cfg: Any
    T: Any
    R_recipe: Any
    ResetDomainTag: Any


@dataclass(frozen=True)
class EdgeRecord:
    edge_id: str
    src_hash: str
    dst_hash: str
    joint_action: tuple[int, ...]
    branch_labels: tuple[str, ...]
    source_certified: bool
    projection_checked: bool
    rejected_reason: str | None = None


@dataclass(frozen=True)
class KappaSignature:
    signature_id: str
    field_values: Mapping[str, Any]
    kappa_hash: str


@dataclass(frozen=True)
class ProbeRecord:
    probe_id: str
    layout_id: str
    tau_id: str
    state_hash: str
    override: Override
    U_tau: float
    D_tau: int
    distance_to_tau: int | None
    pi_signature: str
    kappa_hash: str
    probe_class: ProbeClass
    certificate_status: CertificateStatus
    matched_control_certificate_id: str | None = None
    leakage_audit_id: str | None = None
    claim_guard_passed: bool = False
    failure_label: str | None = None


@dataclass(frozen=True)
class MatchedControlCertificate:
    certificate_id: str
    positive_probe_id: str
    control_probe_id: str | None
    control_type: ControlType
    matching_result: MatchingResult
    certificate_status: CertificateStatus
    kappa_hash_positive: str
    kappa_hash_control: str | None
    leakage_audit_result: str | None
    leakage_audit_id: str | None
    failure_label: str | None = None


@dataclass(frozen=True)
class SameOverrideCounterexampleCertificate:
    certificate_id: str
    claim_level: SameOverrideClaimLevel
    construct_split_type: ConstructSplitType
    positive_probe_id: str
    control_probe_id: str | None
    matched_control_certificate_id: str | None
    leakage_audit_id: str | None
    certificate_status: CertificateStatus
    U_tau_positive: float
    D_tau_positive: int
    U_tau_control: float | None
    D_tau_control: int | None
    failure_label: str | None = None


@dataclass
class GraphBuildResult:
    nodes: dict[str, PublicState] = field(default_factory=dict)
    raw_representatives: dict[str, Any] = field(default_factory=dict, repr=False)
    edges: list[EdgeRecord] = field(default_factory=list)
    depths: dict[str, int] = field(default_factory=dict)
    rejected_edges: list[EdgeRecord] = field(default_factory=list)


@dataclass(frozen=True)
class QuotientStateInfo:
    state_hash: str
    distance_to_tau: int | None
    optimal_action_classes: tuple[int, ...]
    U_tau: float
    pi_signature: str


@dataclass(frozen=True)
class Gate3RunSummary:
    run_id: str
    backend: str
    layout_id: str
    tau_id: str
    max_depth: int
    num_nodes: int
    num_edges: int
    num_rejected_edges: int
    num_reachable_tau_states: int
    num_pi_classes: int
    num_probes: int
    num_positive_candidates: int
    num_exact_matched_controls: int
    num_same_override_counterexamples: int
    failure_label_histogram: Mapping[str, int]
    status: str
