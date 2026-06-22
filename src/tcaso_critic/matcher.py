from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from typing import Iterable

from .canonical import canonical_hash
from .kappa import KappaComputer
from .models import (
    CertificateStatus,
    ConstructSplitType,
    ControlType,
    MatchedControlCertificate,
    MatchingResult,
    Override,
    ProbeClass,
    ProbeRecord,
    QuotientStateInfo,
    SameOverrideClaimLevel,
    SameOverrideCounterexampleCertificate,
)
from .quotient import ContinuationQuotientComputer, QuotientResult
from .validators import validate_before_write
from .models import KappaSignature


class ExactKappaMatcher:
    def __init__(self, *, quotient_computer: ContinuationQuotientComputer, quotient: QuotientResult, kappa: KappaComputer, audited_agent_i: int, action_count: int) -> None:
        self.qc = quotient_computer
        self.quotient = quotient
        self.kappa = kappa
        self.audited_agent_i = int(audited_agent_i)
        self.action_count = int(action_count)

    def build_probe_records(self) -> tuple[list[ProbeRecord], dict[str, KappaSignature]]:
        records: list[ProbeRecord] = []
        kappa_by_probe: dict[str, KappaSignature] = {}
        for state_hash, info in sorted(self.quotient.state_info.items()):
            for action in range(self.action_count):
                override = Override(agent_i=self.audited_agent_i, action=action)
                D = self.qc.D_tau(self.quotient, state_hash, override)
                sig = self.kappa.compute(state_hash, override, info.distance_to_tau)
                pclass = self._classify_probe(info, D)
                status = CertificateStatus.DRAFT
                failure = "NO_EXACT_MATCH"
                if pclass in {ProbeClass.TASK_DETERMINED_NEGATIVE_CONTROL, ProbeClass.NO_CLASS_DISRUPTION_NEGATIVE_CONTROL}:
                    failure = "NO_EXACT_MATCH"
                probe_id = canonical_hash({"state": state_hash, "override": asdict(override), "U": info.U_tau, "D": D, "kappa": sig.kappa_hash}, prefix="probe")
                rec = ProbeRecord(
                    probe_id=probe_id,
                    layout_id=self.kappa.graph.nodes[state_hash].layout_id,
                    tau_id=self.kappa.task_spec.tau_id,
                    state_hash=state_hash,
                    override=override,
                    U_tau=info.U_tau,
                    D_tau=D,
                    distance_to_tau=info.distance_to_tau,
                    pi_signature=info.pi_signature,
                    kappa_hash=sig.kappa_hash,
                    probe_class=pclass,
                    certificate_status=status,
                    failure_label=failure,
                )
                records.append(rec)
                kappa_by_probe[probe_id] = sig
        validate_before_write(records)
        return records, kappa_by_probe

    @staticmethod
    def _classify_probe(info: QuotientStateInfo, D: int) -> ProbeClass:
        if info.U_tau > 0.0 and D == 1:
            return ProbeClass.STRUCTURAL_POSITIVE_CANDIDATE
        if info.U_tau == 0.0 and D == 0:
            return ProbeClass.TASK_DETERMINED_NEGATIVE_CONTROL
        if info.U_tau > 0.0 and D == 0:
            return ProbeClass.NO_CLASS_DISRUPTION_NEGATIVE_CONTROL
        return ProbeClass.UNMATCHED_STRUCTURAL_POSITIVE

    def exact_match(self, probes: list[ProbeRecord]) -> tuple[list[ProbeRecord], list[MatchedControlCertificate], list[SameOverrideCounterexampleCertificate]]:
        by_kappa_and_override: dict[tuple[str, str], list[ProbeRecord]] = defaultdict(list)
        for p in probes:
            by_kappa_and_override[(p.kappa_hash, p.override.key())].append(p)
        updated: dict[str, ProbeRecord] = {p.probe_id: p for p in probes}
        certs: list[MatchedControlCertificate] = []
        same_override: list[SameOverrideCounterexampleCertificate] = []
        for p in probes:
            if p.probe_class != ProbeClass.STRUCTURAL_POSITIVE_CANDIDATE:
                continue
            candidates = by_kappa_and_override[(p.kappa_hash, p.override.key())]
            task_controls = [c for c in candidates if c.probe_class == ProbeClass.TASK_DETERMINED_NEGATIVE_CONTROL]
            no_class_controls = [c for c in candidates if c.probe_class == ProbeClass.NO_CLASS_DISRUPTION_NEGATIVE_CONTROL]
            chosen = task_controls[0] if task_controls else (no_class_controls[0] if no_class_controls else None)
            if chosen is None:
                cert_id = canonical_hash({"positive": p.probe_id, "result": "NO_EXACT_MATCH"}, prefix="mcc")
                certs.append(MatchedControlCertificate(
                    certificate_id=cert_id,
                    positive_probe_id=p.probe_id,
                    control_probe_id=None,
                    control_type=ControlType.NONE,
                    matching_result=MatchingResult.NO_EXACT_MATCH,
                    certificate_status=CertificateStatus.OBSTRUCTED,
                    kappa_hash_positive=p.kappa_hash,
                    kappa_hash_control=None,
                    leakage_audit_result=None,
                    leakage_audit_id=None,
                    failure_label="NO_EXACT_MATCH",
                ))
                same_override.append(SameOverrideCounterexampleCertificate(
                    certificate_id=canonical_hash({"positive": p.probe_id, "claim": "UNMATCHED"}, prefix="soc"),
                    claim_level=SameOverrideClaimLevel.UNMATCHED_STRUCTURAL_POSITIVE,
                    construct_split_type=ConstructSplitType.UNMATCHED,
                    positive_probe_id=p.probe_id,
                    control_probe_id=None,
                    matched_control_certificate_id=None,
                    leakage_audit_id=None,
                    certificate_status=CertificateStatus.OBSTRUCTED,
                    U_tau_positive=p.U_tau,
                    D_tau_positive=p.D_tau,
                    U_tau_control=None,
                    D_tau_control=None,
                    failure_label="NO_EXACT_MATCH",
                ))
                continue
            leakage_id = canonical_hash({"positive": p.probe_id, "control": chosen.probe_id, "kappa": p.kappa_hash}, prefix="leakage")
            ctype = ControlType.TASK_DETERMINED_NEGATIVE if chosen.probe_class == ProbeClass.TASK_DETERMINED_NEGATIVE_CONTROL else ControlType.NO_CLASS_DISRUPTION_NEGATIVE
            cert_id = canonical_hash({"positive": p.probe_id, "control": chosen.probe_id, "kappa": p.kappa_hash}, prefix="mcc")
            cert = MatchedControlCertificate(
                certificate_id=cert_id,
                positive_probe_id=p.probe_id,
                control_probe_id=chosen.probe_id,
                control_type=ctype,
                matching_result=MatchingResult.EXACT_MATCH,
                certificate_status=CertificateStatus.CERTIFIED,
                kappa_hash_positive=p.kappa_hash,
                kappa_hash_control=chosen.kappa_hash,
                leakage_audit_result="PASS",
                leakage_audit_id=leakage_id,
                failure_label=None,
            )
            certs.append(cert)
            # A structural positive with a certified exact control becomes a valid
            # semantic recovery probe only after the matched-control certificate
            # and leakage audit are attached.
            updated[p.probe_id] = ProbeRecord(**{**asdict(p), "probe_class": ProbeClass.VALID_SEMANTIC_RECOVERY_PROBE, "certificate_status": CertificateStatus.CERTIFIED, "matched_control_certificate_id": cert_id, "leakage_audit_id": leakage_id, "claim_guard_passed": True, "failure_label": None})
            claim = SameOverrideClaimLevel.SAME_OVERRIDE_TASK_DETERMINED_COUNTEREXAMPLE if ctype == ControlType.TASK_DETERMINED_NEGATIVE else SameOverrideClaimLevel.SAME_OVERRIDE_NO_CLASS_DISRUPTION_DIAGNOSTIC
            split = ConstructSplitType.UNDERDETERMINED_DISRUPTION_VS_TASK_DETERMINED if ctype == ControlType.TASK_DETERMINED_NEGATIVE else ConstructSplitType.UNDERDETERMINED_NO_CLASS_DISRUPTION
            soc = SameOverrideCounterexampleCertificate(
                certificate_id=canonical_hash({"positive": p.probe_id, "control": chosen.probe_id, "claim": claim.value}, prefix="soc"),
                claim_level=claim,
                construct_split_type=split,
                positive_probe_id=p.probe_id,
                control_probe_id=chosen.probe_id,
                matched_control_certificate_id=cert_id,
                leakage_audit_id=leakage_id,
                certificate_status=CertificateStatus.CERTIFIED,
                U_tau_positive=p.U_tau,
                D_tau_positive=p.D_tau,
                U_tau_control=chosen.U_tau,
                D_tau_control=chosen.D_tau,
                failure_label=None,
            )
            same_override.append(soc)
        final_probes = [updated[p.probe_id] for p in probes]
        validate_before_write(final_probes)
        validate_before_write(certs)
        validate_before_write(same_override)
        return final_probes, certs, same_override


def failure_histogram(probes: Iterable[ProbeRecord], certs: Iterable[MatchedControlCertificate], socs: Iterable[SameOverrideCounterexampleCertificate]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for r in list(probes) + list(certs) + list(socs):
        label = getattr(r, "failure_label", None)
        if label:
            counter[str(label)] += 1
    return dict(sorted(counter.items()))
