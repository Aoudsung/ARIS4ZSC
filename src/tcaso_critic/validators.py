from __future__ import annotations

import re
from dataclasses import asdict, is_dataclass
from typing import Any, Iterable, Mapping

from .models import (
    CertificateStatus,
    ConstructSplitType,
    ControlType,
    KappaSignature,
    MatchedControlCertificate,
    MatchingResult,
    ProbeClass,
    ProbeRecord,
    SameOverrideClaimLevel,
    SameOverrideCounterexampleCertificate,
)


class InvariantViolation(ValueError):
    """Raised when a record cannot be written as a certificate artifact."""


HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

CANONICAL_FAILURE_LABELS = {
    "GATE2_INPUT_NOT_ACCEPTED",
    "KAPPA_FIELD_DOMAIN_VIOLATION",
    "KAPPA_FIELD_MISSING",
    "KAPPA_FIELD_EXTRA",
    "MATCHING_DESTROYS_PROBE_POOL",
    "MATCHING_PROTOCOL_UNCERTIFIABLE",
    "KAPPA_BUCKET_POSTHOC_TUNING",
    "OVERRIDE_ALIAS_LEAKAGE",
    "POLICY_VISITATION_FILTER_LEAKAGE",
    "DIAGNOSTIC_CERTIFICATE_CONFLATION",
    "LEAKAGE_AUDIT_FAILED",
    "NO_EXACT_MATCH",
    "NO_CERTIFIED_SAME_OVERRIDE_COUNTEREXAMPLE_FOUND",
    "UNCERTIFIED_GRAPH_TOO_LARGE",
    "GRAPH_BUILD_INFEASIBLE",
    "SOURCE_STEP_WRAPPER_FAILED",
    "PUBLIC_PROJECTION_MISMATCH",
    "CANDIDATE_EDGE_NOT_ADMITTED",
    "RECORD_INVARIANT_VIOLATION",
}

# Field domains are intentionally finite or scalar. Object/array values are
# forbidden inside field_values; structural values must be represented by
# canonical hashes or registered enum/bin ids.
KAPPA_FIELD_DOMAINS: dict[str, str] = {
    "layout_id": "string",
    "tau_id": "string",
    "agent_i": "int",
    "channel_c": "string",
    "override_action": "int",
    "distance_to_tau_bucket": "string",
    "terminal_bucket": "string",
    "active_recipe_hash": "hash_or_none",
    "public_geometry_hash": "hash",
    "inventory_profile_hash": "hash",
    "interface_profile_hash": "hash",
    "partner_action_feasibility_hash": "hash",
    "override_precondition": "string",
    "time_bucket": "string",
    "reset_domain_tag": "string",
}


def _record_to_dict(record: Any) -> dict[str, Any]:
    if is_dataclass(record):
        return asdict(record)
    if isinstance(record, dict):
        return record
    raise TypeError(f"Unsupported record type: {type(record).__name__}")


def _enum_value(value: Any) -> str:
    return getattr(value, "value", value)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise InvariantViolation(message)


def validate_failure_label(label: str | None, *, required: bool) -> None:
    if required:
        require(label is not None, "failure_label is required for non-certified records")
        require(label in CANONICAL_FAILURE_LABELS, f"failure_label {label!r} is not canonical")
    else:
        require(label is None, "certified records must have failure_label=null")


def validate_kappa_signature(sig: KappaSignature | Mapping[str, Any]) -> None:
    d = _record_to_dict(sig)
    fv = d.get("field_values")
    require(isinstance(fv, dict), "KappaSignature.field_values must be an object")
    expected = set(KAPPA_FIELD_DOMAINS)
    got = set(fv)
    missing = expected - got
    extra = got - expected
    require(not missing, f"KappaSignature missing fields: {sorted(missing)}")
    require(not extra, f"KappaSignature has extra fields: {sorted(extra)}")
    for name, domain in KAPPA_FIELD_DOMAINS.items():
        value = fv[name]
        require(not isinstance(value, (dict, list, tuple)), f"kappa field {name} must be scalar/hash, not object/array")
        if domain == "string":
            require(isinstance(value, str) and value != "", f"kappa field {name} must be a non-empty string")
        elif domain == "int":
            require(isinstance(value, int) and not isinstance(value, bool), f"kappa field {name} must be integer")
        elif domain == "hash":
            require(isinstance(value, str) and HEX64_RE.match(value) is not None, f"kappa field {name} must be 64-hex hash")
        elif domain == "hash_or_none":
            require(value is None or (isinstance(value, str) and HEX64_RE.match(value) is not None), f"kappa field {name} must be null or 64-hex hash")
        else:
            raise AssertionError(f"Unhandled kappa domain {domain}")
    require(isinstance(d.get("kappa_hash"), str) and HEX64_RE.match(d["kappa_hash"]) is not None, "kappa_hash must be 64-hex")


def validate_probe_record(record: ProbeRecord | Mapping[str, Any]) -> None:
    d = _record_to_dict(record)
    status = _enum_value(d.get("certificate_status"))
    probe_class = _enum_value(d.get("probe_class"))
    failure_label = d.get("failure_label")
    if status == CertificateStatus.CERTIFIED.value:
        validate_failure_label(failure_label, required=False)
    else:
        validate_failure_label(failure_label, required=True)
    if probe_class == ProbeClass.VALID_SEMANTIC_RECOVERY_PROBE.value:
        require(status == CertificateStatus.CERTIFIED.value, "valid semantic probe must be CERTIFIED")
        require(bool(d.get("matched_control_certificate_id")), "valid semantic probe needs matched_control_certificate_id")
        require(bool(d.get("leakage_audit_id")), "valid semantic probe needs leakage_audit_id")
        require(d.get("claim_guard_passed") is True, "valid semantic probe requires claim_guard_passed=true")
        require(d.get("D_tau") == 1, "valid semantic probe requires D_tau=1")
        require(float(d.get("U_tau", 0.0)) > 0.0, "valid semantic probe requires U_tau>0")


def validate_matched_control_certificate(cert: MatchedControlCertificate | Mapping[str, Any]) -> None:
    d = _record_to_dict(cert)
    status = _enum_value(d.get("certificate_status"))
    matching_result = _enum_value(d.get("matching_result"))
    control_type = _enum_value(d.get("control_type"))
    failure_label = d.get("failure_label")
    if status == CertificateStatus.CERTIFIED.value:
        validate_failure_label(failure_label, required=False)
        require(matching_result == MatchingResult.EXACT_MATCH.value, "CERTIFIED matched-control requires EXACT_MATCH")
        require(control_type in {ControlType.TASK_DETERMINED_NEGATIVE.value, ControlType.NO_CLASS_DISRUPTION_NEGATIVE.value}, "CERTIFIED matched-control requires a real negative control type")
        require(d.get("leakage_audit_result") == "PASS", "CERTIFIED matched-control requires leakage PASS")
        require(bool(d.get("control_probe_id")), "CERTIFIED matched-control requires control_probe_id")
        require(bool(d.get("leakage_audit_id")), "CERTIFIED matched-control requires leakage_audit_id")
        require(d.get("kappa_hash_positive") == d.get("kappa_hash_control"), "CERTIFIED matched-control requires equal kappa hashes")
    if matching_result == MatchingResult.EXACT_MATCH.value:
        require(status == CertificateStatus.CERTIFIED.value, "EXACT_MATCH implies CERTIFIED")
        require(control_type != ControlType.NONE.value, "EXACT_MATCH cannot have control_type=NONE")
        require(d.get("kappa_hash_positive") == d.get("kappa_hash_control"), "EXACT_MATCH requires equal kappa hashes")
    if status != CertificateStatus.CERTIFIED.value:
        validate_failure_label(failure_label, required=True)


def validate_same_override_certificate(cert: SameOverrideCounterexampleCertificate | Mapping[str, Any]) -> None:
    d = _record_to_dict(cert)
    status = _enum_value(d.get("certificate_status"))
    claim = _enum_value(d.get("claim_level"))
    split = _enum_value(d.get("construct_split_type"))
    failure_label = d.get("failure_label")
    if status == CertificateStatus.CERTIFIED.value:
        validate_failure_label(failure_label, required=False)
        require(bool(d.get("positive_probe_id")), "CERTIFIED same-override record requires positive_probe_id")
        require(bool(d.get("control_probe_id")), "CERTIFIED same-override record requires control_probe_id")
        require(bool(d.get("matched_control_certificate_id")), "CERTIFIED same-override record requires matched_control_certificate_id")
        require(bool(d.get("leakage_audit_id")), "CERTIFIED same-override record requires leakage_audit_id")
        require(float(d.get("U_tau_positive", 0.0)) > 0.0, "positive side requires U_tau>0")
        require(d.get("D_tau_positive") == 1, "positive side requires D_tau=1")
    else:
        validate_failure_label(failure_label, required=True)
    if claim == SameOverrideClaimLevel.SAME_OVERRIDE_TASK_DETERMINED_COUNTEREXAMPLE.value:
        require(status == CertificateStatus.CERTIFIED.value, "strongest same-override claim requires CERTIFIED")
        require(split == ConstructSplitType.UNDERDETERMINED_DISRUPTION_VS_TASK_DETERMINED.value, "strongest claim requires task-determined split")
        require(d.get("U_tau_control") == 0.0, "task-determined control requires U_tau_control=0")
        require(d.get("D_tau_control") == 0, "task-determined control requires D_tau_control=0")
    if claim == SameOverrideClaimLevel.SAME_OVERRIDE_NO_CLASS_DISRUPTION_DIAGNOSTIC.value:
        require(status == CertificateStatus.CERTIFIED.value, "weaker diagnostic still requires CERTIFIED")
        require(split == ConstructSplitType.UNDERDETERMINED_NO_CLASS_DISRUPTION.value, "weaker diagnostic split mismatch")
        require(float(d.get("U_tau_control", -1.0)) > 0.0, "no-class-disruption control requires U_tau_control>0")
        require(d.get("D_tau_control") == 0, "no-class-disruption control requires D_tau_control=0")
    if claim == SameOverrideClaimLevel.UNMATCHED_STRUCTURAL_POSITIVE.value:
        require(status != CertificateStatus.CERTIFIED.value, "unmatched positive cannot be CERTIFIED")
        require(split == ConstructSplitType.UNMATCHED.value, "unmatched positive split mismatch")


def validate_before_write(records: Iterable[Any]) -> None:
    """Apply all record invariants before any JSONL certificate write.

    This function is intentionally fail-fast. Callers must not catch this error
    to continue artifact writing; the CLI catches it only at top level to write a
    run report with status INVARIANT_FAILED.
    """

    for record in records:
        if isinstance(record, KappaSignature) or (isinstance(record, dict) and "field_values" in record and "kappa_hash" in record):
            validate_kappa_signature(record)
        elif isinstance(record, ProbeRecord) or (isinstance(record, dict) and "probe_class" in record):
            validate_probe_record(record)
        elif isinstance(record, MatchedControlCertificate) or (isinstance(record, dict) and "matching_result" in record):
            validate_matched_control_certificate(record)
        elif isinstance(record, SameOverrideCounterexampleCertificate) or (isinstance(record, dict) and "claim_level" in record):
            validate_same_override_certificate(record)
        else:
            raise InvariantViolation(f"No invariant validator registered for record type {type(record).__name__}")
