from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any, Mapping

from .canonical import dump_json, write_jsonl
from .matcher import failure_histogram
from .models import Gate3RunSummary, GraphBuildResult, MatchedControlCertificate, ProbeRecord, SameOverrideCounterexampleCertificate, KappaSignature
from .quotient import QuotientResult


def write_run_artifacts(
    out_dir: str,
    *,
    summary: Gate3RunSummary,
    graph: GraphBuildResult,
    quotient: QuotientResult,
    probes: list[ProbeRecord],
    matched_controls: list[MatchedControlCertificate],
    same_override: list[SameOverrideCounterexampleCertificate],
    kappa_signatures: Mapping[str, KappaSignature] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    dump_json(os.path.join(out_dir, "run_summary.json"), asdict(summary))
    dump_json(os.path.join(out_dir, "nodes.json"), {h: asdict(v) for h, v in graph.nodes.items()})
    write_jsonl(os.path.join(out_dir, "edges.jsonl"), [asdict(e) for e in graph.edges])
    write_jsonl(os.path.join(out_dir, "rejected_edges.jsonl"), [asdict(e) for e in graph.rejected_edges])
    dump_json(os.path.join(out_dir, "quotient_state_info.json"), {h: asdict(i) for h, i in quotient.state_info.items()})
    dump_json(os.path.join(out_dir, "pi_classes.json"), quotient.pi_classes)
    write_jsonl(os.path.join(out_dir, "probe_records.jsonl"), [asdict(p) for p in probes])
    write_jsonl(os.path.join(out_dir, "matched_control_certificates.jsonl"), [asdict(c) for c in matched_controls])
    write_jsonl(os.path.join(out_dir, "same_override_counterexamples.jsonl"), [asdict(c) for c in same_override])
    if kappa_signatures is not None:
        dump_json(os.path.join(out_dir, "kappa_signatures.json"), {k: asdict(v) for k, v in kappa_signatures.items()})
    if diagnostics is not None:
        dump_json(os.path.join(out_dir, "matching_diagnostics.json"), dict(diagnostics))
    with open(os.path.join(out_dir, "GATE3_RUN_REPORT.md"), "w", encoding="utf-8") as f:
        f.write(render_markdown_report(summary, graph, quotient, probes, matched_controls, same_override, diagnostics=diagnostics))


def render_markdown_report(
    summary: Gate3RunSummary,
    graph: GraphBuildResult,
    quotient: QuotientResult,
    probes: list[ProbeRecord],
    matched_controls: list[MatchedControlCertificate],
    same_override: list[SameOverrideCounterexampleCertificate],
    *,
    diagnostics: Mapping[str, Any] | None = None,
) -> str:
    certified_same_override = [c for c in same_override if getattr(c.certificate_status, "value", c.certificate_status) == "CERTIFIED"]
    positive = [p for p in probes if getattr(p.probe_class, "value", p.probe_class) in {"STRUCTURAL_POSITIVE_CANDIDATE", "VALID_SEMANTIC_RECOVERY_PROBE"}]
    valid = [p for p in probes if getattr(p.probe_class, "value", p.probe_class) == "VALID_SEMANTIC_RECOVERY_PROBE"]
    lines = [
        "# TCASO-CRITIC Gate 3 Depth-2/3 Run Report",
        "",
        "## Status",
        "",
        "```text",
        summary.status,
        "```",
        "",
        "This is a source-backed certifier run. It is not a policy experiment, pilot, benchmark, or ICLR-ready evidence package.",
        "",
        "## Counts",
        "",
        f"- backend: `{summary.backend}`",
        f"- layout: `{summary.layout_id}`",
        f"- tau: `{summary.tau_id}`",
        f"- max_depth: `{summary.max_depth}`",
        f"- public nodes: `{summary.num_nodes}`",
        f"- certified edges: `{summary.num_edges}`",
        f"- rejected edges: `{summary.num_rejected_edges}`",
        f"- tau states reachable in bounded graph: `{summary.num_reachable_tau_states}`",
        f"- Pi_tau classes: `{summary.num_pi_classes}`",
        f"- probes: `{summary.num_probes}`",
        f"- structural positives: `{len(positive)}`",
        f"- valid semantic recovery probes: `{len(valid)}`",
        f"- exact matched controls: `{summary.num_exact_matched_controls}`",
        f"- certified same-override records: `{len(certified_same_override)}`",
        "",
        "## Failure labels",
        "",
    ]
    if summary.failure_label_histogram:
        for k, v in summary.failure_label_histogram.items():
            lines.append(f"- `{k}`: {v}")
    else:
        lines.append("- none")
    if diagnostics:
        lines += [
            "",
            "## Matching diagnosis",
            "",
            f"- primary label: `{diagnostics.get('diagnostic_primary_label')}`",
            f"- notes: {diagnostics.get('notes')}",
        ]
        dominant = diagnostics.get("dominant_mismatch_fields") or []
        if dominant:
            lines.append("- dominant kappa mismatches:")
            for row in dominant[:10]:
                lines.append(f"  - `{row.get('field')}`: {row.get('count')}")
    lines += [
        "",
        "## Claim boundary",
        "",
        "```text",
        "MATCHED_CONTROL_POOL_CERTIFIED is claimed only if exact matched controls > 0 and all record invariants pass.",
        "VALID_SEMANTIC_RECOVERY_PROBE is claimed only for probe_records with CERTIFIED status and matched-control/leakage ids.",
        "No policy score, pilot score, benchmark result, or diagnostic reversal is claimed here.",
        "```",
        "",
    ]
    return "\n".join(lines)


def make_summary(
    *,
    run_id: str,
    backend: str,
    layout_id: str,
    tau_id: str,
    max_depth: int,
    graph: GraphBuildResult,
    quotient: QuotientResult,
    probes: list[ProbeRecord],
    matched_controls: list[MatchedControlCertificate],
    same_override: list[SameOverrideCounterexampleCertificate],
) -> Gate3RunSummary:
    exact = [c for c in matched_controls if getattr(c.matching_result, "value", c.matching_result) == "EXACT_MATCH"]
    certified_soc = [c for c in same_override if getattr(c.certificate_status, "value", c.certificate_status) == "CERTIFIED"]
    positives = [p for p in probes if getattr(p.probe_class, "value", p.probe_class) in {"STRUCTURAL_POSITIVE_CANDIDATE", "VALID_SEMANTIC_RECOVERY_PROBE"}]
    hist = failure_histogram(probes, matched_controls, same_override)
    status = "GATE3_DEPTH23_SLICE_COMPLETED"
    if not exact:
        status = "GATE3_DEPTH23_SLICE_COMPLETED_MATCHING_DESTROYS_PROBE_POOL_OR_NO_EXACT_MATCH"
    if not quotient.target_states:
        status = "GATE3_DEPTH23_SLICE_COMPLETED_NO_TAU_REACHABLE_WITHIN_BOUND"
    return Gate3RunSummary(
        run_id=run_id,
        backend=backend,
        layout_id=layout_id,
        tau_id=tau_id,
        max_depth=max_depth,
        num_nodes=len(graph.nodes),
        num_edges=len(graph.edges),
        num_rejected_edges=len(graph.rejected_edges),
        num_reachable_tau_states=len(quotient.target_states),
        num_pi_classes=len(quotient.pi_classes),
        num_probes=len(probes),
        num_positive_candidates=len(positives),
        num_exact_matched_controls=len(exact),
        num_same_override_counterexamples=len(certified_soc),
        failure_label_histogram=hist,
        status=status,
    )
