import pytest

from tcaso_critic.models import (
    CertificateStatus,
    ConstructSplitType,
    ControlType,
    KappaSignature,
    MatchedControlCertificate,
    MatchingResult,
    Override,
    ProbeClass,
    ProbeRecord,
    SameOverrideClaimLevel,
    SameOverrideCounterexampleCertificate,
)
from tcaso_critic.validators import InvariantViolation, validate_kappa_signature, validate_matched_control_certificate, validate_probe_record, validate_same_override_certificate

HEX = "a" * 64
FIELDS = {
    "layout_id": "cramped_room",
    "tau_id": "tau_agent0_left",
    "agent_i": 0,
    "channel_c": "primitive_action",
    "override_action": 2,
    "distance_to_tau_bucket": "d1",
    "terminal_bucket": "nonterminal",
    "active_recipe_hash": HEX,
    "public_geometry_hash": HEX,
    "inventory_profile_hash": HEX,
    "interface_profile_hash": HEX,
    "partner_action_feasibility_hash": HEX,
    "override_precondition": "move_into_empty",
    "time_bucket": "t0",
    "reset_domain_tag": "RESET_DEFAULT",
}


def test_kappa_rejects_object_field_value():
    bad = dict(FIELDS)
    bad["public_geometry_hash"] = {"raw": "object"}
    with pytest.raises(InvariantViolation):
        validate_kappa_signature(KappaSignature("ks", bad, HEX))


def test_probe_certified_requires_null_failure_label():
    rec = ProbeRecord(
        probe_id="p",
        layout_id="cramped_room",
        tau_id="tau",
        state_hash=HEX,
        override=Override(0, 2),
        U_tau=0.0,
        D_tau=0,
        distance_to_tau=1,
        pi_signature="pi:x",
        kappa_hash=HEX,
        probe_class=ProbeClass.TASK_DETERMINED_NEGATIVE_CONTROL,
        certificate_status=CertificateStatus.CERTIFIED,
        failure_label="NO_EXACT_MATCH",
    )
    with pytest.raises(InvariantViolation):
        validate_probe_record(rec)


def test_exact_match_obstructed_is_rejected():
    cert = MatchedControlCertificate(
        certificate_id="c",
        positive_probe_id="p",
        control_probe_id=None,
        control_type=ControlType.NONE,
        matching_result=MatchingResult.EXACT_MATCH,
        certificate_status=CertificateStatus.OBSTRUCTED,
        kappa_hash_positive=HEX,
        kappa_hash_control=HEX,
        leakage_audit_result=None,
        leakage_audit_id=None,
        failure_label="NO_EXACT_MATCH",
    )
    with pytest.raises(InvariantViolation):
        validate_matched_control_certificate(cert)


def test_certified_hash_mismatch_is_rejected():
    cert = MatchedControlCertificate(
        certificate_id="c",
        positive_probe_id="p",
        control_probe_id="q",
        control_type=ControlType.TASK_DETERMINED_NEGATIVE,
        matching_result=MatchingResult.EXACT_MATCH,
        certificate_status=CertificateStatus.CERTIFIED,
        kappa_hash_positive=HEX,
        kappa_hash_control="b" * 64,
        leakage_audit_result="PASS",
        leakage_audit_id="l",
        failure_label=None,
    )
    with pytest.raises(InvariantViolation):
        validate_matched_control_certificate(cert)


def test_weaker_same_override_requires_evidence_ids():
    cert = SameOverrideCounterexampleCertificate(
        certificate_id="soc",
        claim_level=SameOverrideClaimLevel.SAME_OVERRIDE_NO_CLASS_DISRUPTION_DIAGNOSTIC,
        construct_split_type=ConstructSplitType.UNDERDETERMINED_NO_CLASS_DISRUPTION,
        positive_probe_id="p",
        control_probe_id=None,
        matched_control_certificate_id=None,
        leakage_audit_id=None,
        certificate_status=CertificateStatus.CERTIFIED,
        U_tau_positive=1.0,
        D_tau_positive=1,
        U_tau_control=1.0,
        D_tau_control=0,
        failure_label=None,
    )
    with pytest.raises(InvariantViolation):
        validate_same_override_certificate(cert)
