"""Score non-noop sweep bundles as a two-arm discrimination diagnostic.

This scorer does not create new evidence. It aggregates already returned
non-noop sweep bundles into recovery-arm and failure-arm summaries so reviewers
can judge whether return-independent positives discriminate from negative arms.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", action="append", type=Path, required=True)
    parser.add_argument("--recovery-disruption", action="append", required=True)
    parser.add_argument("--failure-disruption", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _arm_label(
    disruption: str,
    recovery_disruptions: set[str],
    failure_disruptions: set[str],
) -> str | None:
    if disruption in recovery_disruptions:
        return "recovery_arm"
    if disruption in failure_disruptions:
        return "failure_arm"
    return None


def _bundle_rows(
    bundle: Path,
    recovery_disruptions: set[str],
    failure_disruptions: set[str],
) -> list[dict[str, Any]]:
    summary = _read_json(bundle / "summary.json")
    run_config = _read_json(bundle / "run_config.json")
    return_independent = _read_csv(bundle / "return_independent_worst_serd.csv")
    standard = {
        (row["policy"], row["domain"], row["disruption"]): row
        for row in _read_csv(bundle / "worst_serd.csv")
    }

    rows: list[dict[str, Any]] = []
    for row in return_independent:
        arm = _arm_label(row["disruption"], recovery_disruptions, failure_disruptions)
        if arm is None:
            continue
        key = (row["policy"], row["domain"], row["disruption"])
        standard_row = standard.get(key, {})
        rows.append(
            {
                "bundle": str(bundle),
                "layout": run_config.get("layout", ""),
                "pairing": run_config.get("pairing", ""),
                "policy": row["policy"],
                "domain": row["domain"],
                "disruption": row["disruption"],
                "arm": arm,
                "return_independent_mean_serd_worst": row["mean_serd_worst"],
                "return_independent_ci95_low": row["ci95_low"],
                "return_independent_ci95_high": row["ci95_high"],
                "return_independent_n": row["n"],
                "return_independent_classification": row["classification"],
                "return_independent_limiting_family": row["limiting_family"],
                "standard_mean_serd_worst": standard_row.get("mean_serd_worst", ""),
                "standard_classification": standard_row.get("classification", ""),
                "standard_limiting_family": standard_row.get("limiting_family", ""),
                "bundle_claim_bearing": summary.get("claim_bearing", ""),
                "bundle_status": summary.get("m4_status", ""),
            }
        )
    return rows


def _count(rows: list[dict[str, Any]], key: str, value: str) -> int:
    return sum(1 for row in rows if row.get(key) == value)


def _summarize_arm(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    arm_rows = [row for row in rows if row["arm"] == arm]
    values = [
        float(row["return_independent_mean_serd_worst"])
        for row in arm_rows
        if row["return_independent_mean_serd_worst"] != ""
    ]
    survival_rows = [
        row
        for row in arm_rows
        if row["return_independent_classification"] == "survival"
    ]
    return {
        "arm": arm,
        "n_rows": len(arm_rows),
        "n_survival": _count(arm_rows, "return_independent_classification", "survival"),
        "n_collapse": _count(arm_rows, "return_independent_classification", "collapse"),
        "n_adverse": _count(arm_rows, "return_independent_classification", "adverse_semantic_gap"),
        "n_inconclusive": _count(arm_rows, "return_independent_classification", "inconclusive"),
        "mean_return_independent_worst": mean(values) if values else None,
        "survival_limiting_families": sorted(
            {row["return_independent_limiting_family"] for row in survival_rows}
        ),
        "n_standard_survival": _count(arm_rows, "standard_classification", "survival"),
        "n_standard_collapse": _count(arm_rows, "standard_classification", "collapse"),
        "n_standard_adverse": _count(arm_rows, "standard_classification", "adverse_semantic_gap"),
        "n_standard_inconclusive": _count(arm_rows, "standard_classification", "inconclusive"),
    }


def _decision(rows: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> dict[str, Any]:
    recovery = next(item for item in summaries if item["arm"] == "recovery_arm")
    failure = next(item for item in summaries if item["arm"] == "failure_arm")
    layouts = sorted({row["layout"] for row in rows if row["layout"]})
    pairings = sorted({row["pairing"] for row in rows if row["pairing"]})
    survival_families = set(recovery["survival_limiting_families"])
    failure_nonpositive = failure["n_collapse"] + failure["n_adverse"]
    blockers: list[str] = []
    if recovery["n_survival"] == 0:
        blockers.append("recovery_arm_has_no_return_independent_survival")
    if failure_nonpositive == 0:
        blockers.append("failure_arm_has_no_collapse_or_adverse_rows")
    if failure["n_survival"] > 0:
        blockers.append("failure_arm_has_survival_rows")
    if survival_families and survival_families <= {"naive_replanning"}:
        blockers.append("all_recovery_survival_limited_by_naive_replanning")
    if "cross_next" not in pairings:
        blockers.append("cross_play_pairing_missing")
    if len(layouts) < 2:
        blockers.append("second_layout_missing")

    if blockers:
        label = "TWO_ARM_DISCRIMINATION_INCOMPLETE"
    else:
        label = "TWO_ARM_DISCRIMINATION_SIGNAL_PRESENT_FOR_REVIEW"
    return {
        "decision": label,
        "blockers": blockers,
        "layouts": layouts,
        "pairings": pairings,
        "recovery_arm": recovery,
        "failure_arm": failure,
        "claim_boundary": (
            "diagnostic aggregation of returned non-noop sweeps; not a "
            "validated recovery detector or oral evidence without review"
        ),
    }


def _operator_note(output_dir: Path, bundles: list[Path], decision: dict[str, Any]) -> None:
    lines = [
        "# Operator Note",
        "",
        "No passwords, tokens, shell history, or private credentials are recorded",
        "in this artifact.",
        "",
        "Command class:",
        "",
        "```text",
        "python -m experiments.serd.score_nonnoop_two_arm_discrimination",
        "```",
        "",
        f"Decision: `{decision['decision']}`",
        "",
        "Input bundles:",
        "",
    ]
    lines.extend(f"- `{bundle}`" for bundle in bundles)
    lines.extend(
        [
            "",
            "This scorer aggregates already returned non-noop sweep outputs. It",
            "does not create branch records, alter SERD scoring, or relax",
            "`SERD_worst`.",
            "",
        ]
    )
    (output_dir / "operator_note.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    recovery_disruptions = set(args.recovery_disruption)
    failure_disruptions = set(args.failure_disruption)
    rows: list[dict[str, Any]] = []
    for bundle in args.bundle:
        rows.extend(_bundle_rows(bundle, recovery_disruptions, failure_disruptions))

    if not rows:
        raise RuntimeError("no selected recovery/failure rows found in input bundles")

    output_dir = args.output_dir
    fieldnames = list(rows[0])
    _write_csv(output_dir / "two_arm_selected_worst_rows.csv", rows, fieldnames)
    summaries = [
        _summarize_arm(rows, "recovery_arm"),
        _summarize_arm(rows, "failure_arm"),
    ]
    _write_csv(
        output_dir / "two_arm_discrimination_summary.csv",
        summaries,
        list(summaries[0]),
    )
    decision = _decision(rows, summaries)
    _write_json(output_dir / "two_arm_decision.json", decision)
    _operator_note(output_dir, args.bundle, decision)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
