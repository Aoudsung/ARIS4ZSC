from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from typing import Any, Mapping

from .models import MatchedControlCertificate, ProbeClass, ProbeRecord, KappaSignature


STRONG_KAPPA_FIELDS = {
    "public_geometry_hash",
    "interface_profile_hash",
    "partner_action_feasibility_hash",
    "inventory_profile_hash",
    "active_recipe_hash",
    "distance_to_tau_bucket",
}


def diagnose_exact_matching(
    *,
    probes: list[ProbeRecord],
    matched_controls: list[MatchedControlCertificate],
    kappa_signatures: Mapping[str, KappaSignature],
    num_tau_states: int,
    max_depth: int,
) -> dict[str, Any]:
    positives = [p for p in probes if p.probe_class == ProbeClass.STRUCTURAL_POSITIVE_CANDIDATE]
    controls = [p for p in probes if p.probe_class in {ProbeClass.TASK_DETERMINED_NEGATIVE_CONTROL, ProbeClass.NO_CLASS_DISRUPTION_NEGATIVE_CONTROL}]
    exact = [c for c in matched_controls if getattr(c.matching_result, "value", c.matching_result) == "EXACT_MATCH"]
    if exact:
        return {
            "diagnostic_primary_label": "EXACT_MATCH_NONEMPTY",
            "exact_match_count": len(exact),
            "positive_count": len(positives),
            "control_count": len(controls),
            "dominant_mismatch_fields": [],
            "notes": "At least one exact kappa matched control was certified; proceed to same-override certificate inspection.",
        }
    if num_tau_states == 0:
        return {
            "diagnostic_primary_label": "TAU_UNREACHABLE_OR_BOUND_TOO_SHALLOW",
            "exact_match_count": 0,
            "positive_count": len(positives),
            "control_count": len(controls),
            "dominant_mismatch_fields": [],
            "notes": f"No target state is reachable within max_depth={max_depth}; increase depth or choose a reachable tau.",
        }
    if not positives:
        return {
            "diagnostic_primary_label": "TAU_OR_OVERRIDE_NOT_PRODUCING_STRUCTURAL_POSITIVES",
            "exact_match_count": 0,
            "positive_count": 0,
            "control_count": len(controls),
            "dominant_mismatch_fields": [],
            "notes": "The quotient did not produce U_tau>0 and D_tau=1 probes; tau/layout may not create public-task underdetermination at this depth.",
        }
    if not controls:
        return {
            "diagnostic_primary_label": "NO_CONTROL_CANDIDATES_FOR_POSITIVES",
            "exact_match_count": 0,
            "positive_count": len(positives),
            "control_count": 0,
            "dominant_mismatch_fields": [],
            "notes": "Structural positives exist but no task-determined/no-class-disruption controls exist. This points to tau/layout/bound, not merely kappa strictness.",
        }

    mismatch_counter: Counter[str] = Counter()
    same_override_pair_count = 0
    for p in positives:
        psig = kappa_signatures.get(p.probe_id)
        if psig is None:
            continue
        for c in controls:
            if p.override.key() != c.override.key():
                continue
            same_override_pair_count += 1
            csig = kappa_signatures.get(c.probe_id)
            if csig is None:
                continue
            for field, pval in psig.field_values.items():
                if csig.field_values.get(field) != pval:
                    mismatch_counter[field] += 1
    dominant = [{"field": k, "count": v} for k, v in mismatch_counter.most_common(10)]
    if same_override_pair_count == 0:
        primary = "LAYOUT_TAU_DOES_NOT_PRODUCE_SAME_OVERRIDE_CONTROLS"
        notes = "Controls exist, but none share the same override key as structural positives. Change tau family/layout/depth before weakening kappa."
    elif mismatch_counter and all(item["field"] in STRONG_KAPPA_FIELDS for item in dominant[:3]):
        primary = "KAPPA_FIELD_STRICTNESS_DOMINATES_NO_MATCH"
        notes = "Same-override positive/control pairs exist, but exact matching is destroyed by hard kappa fields. Inspect dominant fields before considering any pre-registered kappa redesign. Do not merge bins post hoc."
    else:
        primary = "LAYOUT_OR_TAU_MISMATCH_DOMINATES_NO_MATCH"
        notes = "Same-override candidates exist but mismatches are distributed; current layout/tau/bound likely fails to create clean matched-control structure."
    return {
        "diagnostic_primary_label": primary,
        "exact_match_count": 0,
        "positive_count": len(positives),
        "control_count": len(controls),
        "same_override_positive_control_pairs": same_override_pair_count,
        "dominant_mismatch_fields": dominant,
        "notes": notes,
    }
