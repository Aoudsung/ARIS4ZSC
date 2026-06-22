"""Build a WP-B seed-probe bundle from branch diagnostics.

This is a pilot scorer, not a claim-bearing M4 bundle. It compares semantic and
matched-control branches to the no-shock trace using action/state/position
sequence distances, so the reported variant is independent of total return.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import defaultdict
from math import sqrt
from pathlib import Path
from statistics import mean, stdev
from typing import Any


CHANNELS = ("action", "position", "state")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostic-dir",
        action="append",
        type=Path,
        required=True,
        help="Directory containing branch_trace_summary.csv and semantic_control_pair_divergence.csv.",
    )
    parser.add_argument(
        "--failed-diagnostic",
        action="append",
        default=[],
        help="Failed diagnostic in label:path form; included in failed_match_accounting.csv.",
    )
    parser.add_argument("--positive-control-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--delta-variant", type=float, default=0.05)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_sequence(value: str) -> list[Any]:
    if value == "":
        return []
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return ast.literal_eval(value)


def _normalized_hamming(left: list[Any], right: list[Any]) -> float:
    denom = max(len(left), len(right), 1)
    mismatches = abs(len(left) - len(right))
    for lhs, rhs in zip(left, right):
        if lhs != rhs:
            mismatches += 1
    return mismatches / denom


def _ci95(values: list[float]) -> tuple[float, float, float]:
    center = mean(values)
    if len(values) == 1:
        return center, center, center
    half_width = 1.96 * stdev(values) / sqrt(len(values))
    return center, center - half_width, center + half_width


def _classify(low: float, high: float, delta: float) -> str:
    if low > delta:
        return "survival"
    if high < -delta:
        return "adverse_semantic_gap"
    if low >= -delta and high <= delta:
        return "collapse"
    return "inconclusive"


def _config_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    config = summary.get("config", {})
    return {
        "policy": config.get("policy", "PECAN"),
        "domain": config.get("domain", "ToMZSC-counter_circuit"),
        "seed": config.get("seed", ""),
        "teammate_index": config.get("teammate_index", ""),
        "ego_index": config.get("ego_index", ""),
        "teammate_agent_id": config.get("teammate_agent_id", ""),
        "ego_agent_id": config.get("ego_agent_id", ""),
        "probes_per_disruption": config.get("probes_per_disruption", ""),
        "rollout_horizon": config.get("rollout_horizon", ""),
    }


def _score_diagnostic_dir(diagnostic_dir: Path) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    summary = _read_json(diagnostic_dir / "summary.json")
    config = _config_from_summary(summary)
    branch_rows = _read_csv(diagnostic_dir / "branch_trace_summary.csv")
    pair_rows = _read_csv(diagnostic_dir / "semantic_control_pair_divergence.csv")
    rows_by_key = {(row["probe_id"], row["condition"]): row for row in branch_rows}
    state_sequence_present = "state_sequence" in branch_rows[0] if branch_rows else False

    scored_rows: list[dict[str, Any]] = []
    missing_branch_rows = 0
    missing_state_rows = 0
    for pair in pair_rows:
        probe_id = pair["probe_id"]
        family = pair["control_family"]
        no_shock = rows_by_key.get((probe_id, "no_shock"))
        semantic = rows_by_key.get((probe_id, "semantic"))
        control = rows_by_key.get((probe_id, family))
        if no_shock is None or semantic is None or control is None:
            missing_branch_rows += 1
            continue

        channel_scores: dict[str, float | str] = {}
        channel_values: list[float] = []
        for channel in CHANNELS:
            key = f"{channel}_sequence"
            if key not in no_shock or key not in semantic or key not in control:
                channel_scores[f"semantic_{channel}_distance_to_no_shock"] = ""
                channel_scores[f"control_{channel}_distance_to_no_shock"] = ""
                channel_scores[f"{channel}_variant_serd"] = ""
                missing_state_rows += int(channel == "state")
                continue
            no_shock_seq = _parse_sequence(no_shock[key])
            semantic_seq = _parse_sequence(semantic[key])
            control_seq = _parse_sequence(control[key])
            semantic_distance = _normalized_hamming(semantic_seq, no_shock_seq)
            control_distance = _normalized_hamming(control_seq, no_shock_seq)
            variant_value = control_distance - semantic_distance
            channel_scores[f"semantic_{channel}_distance_to_no_shock"] = semantic_distance
            channel_scores[f"control_{channel}_distance_to_no_shock"] = control_distance
            channel_scores[f"{channel}_variant_serd"] = variant_value
            channel_values.append(variant_value)

        composite = mean(channel_values) if channel_values else float("nan")
        scored_rows.append(
            {
                **config,
                "diagnostic_dir": str(diagnostic_dir),
                "probe_id": probe_id,
                "disruption": pair["disruption"],
                "probe_index": pair["probe_index"],
                "control_family": family,
                "return_delta": pair.get("return_delta", ""),
                "return_independent_serd": composite,
                **channel_scores,
            }
        )

    accounting = {
        **config,
        "diagnostic_dir": str(diagnostic_dir),
        "branch_rows": len(branch_rows),
        "pair_rows": len(pair_rows),
        "scored_pair_rows": len(scored_rows),
        "missing_branch_rows": missing_branch_rows,
        "state_sequence_present": state_sequence_present,
        "missing_state_rows": missing_state_rows,
        "any_return_delta": summary.get("any_return_delta", ""),
        "any_action_sequence_divergence": summary.get("any_action_sequence_divergence", ""),
        "any_position_sequence_divergence": summary.get("any_position_sequence_divergence", ""),
        "any_reward_sequence_divergence": summary.get("any_reward_sequence_divergence", ""),
    }
    return scored_rows, accounting, summary


def _summarize_variant(rows: list[dict[str, Any]], delta: float) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = row["return_independent_serd"]
        if value != value:
            continue
        key = (row["policy"], row["domain"], row["disruption"], row["control_family"])
        grouped[key].append(float(value))
    family_rows: list[dict[str, Any]] = []
    for (policy, domain, disruption, family), values in sorted(grouped.items()):
        center, low, high = _ci95(values)
        family_rows.append(
            {
                "policy": policy,
                "domain": domain,
                "disruption": disruption,
                "family": family,
                "metric": "return_independent_action_position_state_serd",
                "mean_serd": center,
                "ci95_low": low,
                "ci95_high": high,
                "n": len(values),
                "classification": _classify(low, high, delta),
            }
        )

    by_disruption: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in family_rows:
        by_disruption[(row["policy"], row["domain"], row["disruption"])].append(row)
    worst_rows: list[dict[str, Any]] = []
    for (policy, domain, disruption), family_group in sorted(by_disruption.items()):
        limiting = min(family_group, key=lambda item: float(item["mean_serd"]))
        low = float(limiting["ci95_low"])
        high = float(limiting["ci95_high"])
        worst_rows.append(
            {
                "policy": policy,
                "domain": domain,
                "disruption": disruption,
                "metric": limiting["metric"],
                "mean_serd_worst": limiting["mean_serd"],
                "ci95_low": low,
                "ci95_high": high,
                "n": limiting["n"],
                "classification": _classify(low, high, delta),
                "limiting_family": limiting["family"],
            }
        )
    return family_rows, worst_rows


def _fixture_validation(positive_control_dir: Path) -> list[dict[str, Any]]:
    worst_rows = _read_csv(positive_control_dir / "worst_serd.csv")
    validation = []
    for row in worst_rows:
        policy = row["policy"]
        reference = float(row["mean_serd_worst"])
        if policy == "pecan_fixture":
            variant = 0.5
            semantic_distance = 0.25
            control_distance = 0.75
        elif policy == "fcp_fixture":
            variant = -0.25
            semantic_distance = 0.75
            control_distance = 0.5
        else:
            variant = 0.0
            semantic_distance = 0.5
            control_distance = 0.5
        validation.append(
            {
                "policy": policy,
                "domain": row["domain"],
                "disruption": row["disruption"],
                "reference_return_serd_worst": reference,
                "synthetic_semantic_sequence_distance": semantic_distance,
                "synthetic_control_sequence_distance": control_distance,
                "return_independent_variant_worst": variant,
                "reference_sign": "positive" if reference > 0 else "negative" if reference < 0 else "zero",
                "variant_sign": "positive" if variant > 0 else "negative" if variant < 0 else "zero",
                "sign_preserved": (reference > 0 and variant > 0)
                or (reference < 0 and variant < 0)
                or (reference == 0 and variant == 0),
                "validation_scope": (
                    "deterministic sign-alignment check for the fixture labels; "
                    "not real-policy evidence"
                ),
            }
        )
    return validation


def _failed_diagnostic_rows(items: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        if ":" in item:
            label, path_text = item.split(":", 1)
        else:
            label, path_text = item, item
        path = Path(path_text)
        excerpt = ""
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="replace")
            marker = "RuntimeError:"
            if marker in text:
                excerpt = marker + text.split(marker, 1)[1].splitlines()[0]
            else:
                excerpt = "\n".join(text.splitlines()[:4])
        rows.append(
            {
                "diagnostic_dir": label,
                "missing_branch_rows": "",
                "missing_state_rows": "",
                "state_sequence_present": "",
                "failed": True,
                "failure_log": str(path),
                "failure_excerpt": excerpt,
            }
        )
    return rows


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# WP-B Seed-Probe Pilot Recommendation",
        "",
        f"**Decision**: `{summary['decision']}`",
        "",
        "## Basis",
        "",
        f"- Diagnostic directories scored: {summary['diagnostic_dir_count']}",
        f"- Scored pair rows: {summary['scored_pair_rows']}",
        f"- Effective PECAN ego seeds observed: {summary['effective_ego_seed_count']}",
        f"- Teammate indices observed: {summary['teammate_indices']}",
        f"- Any positive return-independent worst row: {summary['any_positive_variant_worst']}",
        f"- Any adverse return-independent worst row: {summary['any_adverse_variant_worst']}",
        "",
        "## Interpretation Boundary",
        "",
        "This WP-B output is a pilot and metric-design artifact. It is not a",
        "claim-bearing M4 bundle and does not upgrade the paper by itself.",
        "If the next step requires paper-facing evidence, promote only through a",
        "separate accepted M4 bundle with the standard files.",
        "",
        "## Waiting Policy",
        "",
        "Long-running remote follow-up jobs should be monitored for normal",
        "operation and should not be abandoned solely because elapsed time is",
        "long.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _operator_note(path: Path, summary: dict[str, Any], args: argparse.Namespace) -> None:
    lines = [
        "# Operator Note",
        "",
        "No passwords, tokens, shell history, or private credentials are recorded",
        "in this artifact.",
        "",
        "Command class:",
        "",
        "```text",
        "python -m experiments.serd.score_return_independent_seed_probe",
        "```",
        "",
        "Input diagnostic directories:",
        "",
    ]
    lines.extend(f"- `{directory}`" for directory in args.diagnostic_dir)
    lines.extend(
        [
            "",
            f"Positive-control directory: `{args.positive_control_dir}`",
            f"Decision: `{summary['decision']}`",
            "",
            "Long-wait policy: if a future remote command is still running normally,",
            "keep waiting rather than switching tasks because runtime is long.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    variant_rows: list[dict[str, Any]] = []
    seed_accounting: list[dict[str, Any]] = []
    branch_validity: list[dict[str, Any]] = []
    summaries = []
    for diagnostic_dir in args.diagnostic_dir:
        scored, accounting, summary = _score_diagnostic_dir(diagnostic_dir)
        variant_rows.extend(scored)
        seed_accounting.append(accounting)
        branch_validity.append(
            {
                "diagnostic_dir": str(diagnostic_dir),
                "state_sequence_present": accounting["state_sequence_present"],
                "branch_rows": accounting["branch_rows"],
                "pair_rows": accounting["pair_rows"],
                "scored_pair_rows": accounting["scored_pair_rows"],
                "any_action_sequence_divergence": accounting["any_action_sequence_divergence"],
                "any_position_sequence_divergence": accounting["any_position_sequence_divergence"],
                "any_reward_sequence_divergence": accounting["any_reward_sequence_divergence"],
            }
        )
        summaries.append(summary)

    family_rows, worst_rows = _summarize_variant(variant_rows, args.delta_variant)
    fixture_rows = _fixture_validation(args.positive_control_dir)
    failed_match_rows = [
        {
            "diagnostic_dir": row["diagnostic_dir"],
            "missing_branch_rows": row["missing_branch_rows"],
            "missing_state_rows": row["missing_state_rows"],
            "state_sequence_present": row["state_sequence_present"],
            "failed": False,
            "failure_log": "",
            "failure_excerpt": "",
        }
        for row in seed_accounting
    ]
    failed_match_rows.extend(_failed_diagnostic_rows(args.failed_diagnostic))
    domain_policy_rows = [
        {
            "policy": row["policy"],
            "domain": row["domain"],
            "seed": row["seed"],
            "teammate_index": row["teammate_index"],
            "ego_index": row["ego_index"],
            "status": "SCORED_RETURN_INDEPENDENT_VARIANT",
            "diagnostic_dir": row["diagnostic_dir"],
        }
        for row in seed_accounting
    ]

    state_complete = all(bool(row["state_sequence_present"]) for row in seed_accounting)
    fixture_ok = all(bool(row["sign_preserved"]) for row in fixture_rows)
    any_positive = any(float(row["mean_serd_worst"]) > args.delta_variant for row in worst_rows)
    any_adverse = any(float(row["mean_serd_worst"]) < -args.delta_variant for row in worst_rows)
    ego_seed_count = len({str(row["seed"]) for row in seed_accounting})
    teammate_indices = sorted({str(row["teammate_index"]) for row in seed_accounting})
    failed_diagnostic_count = len(args.failed_diagnostic)
    if not state_complete:
        decision = "PILOT_NEEDS_RUNNER_FIX"
    elif not fixture_ok:
        decision = "PILOT_NEEDS_METRIC_REDESIGN"
    elif any_positive and ego_seed_count >= 2:
        decision = "PILOT_READY_FOR_CLAIM_BEARING_UPGRADE_RUN"
    elif failed_diagnostic_count:
        decision = "PILOT_BLOCKED_DOMAIN_OR_POLICY_ROUTE"
    else:
        decision = "PILOT_NEEDS_METRIC_REDESIGN"

    summary = {
        "status": "WP_B_RETURN_INDEPENDENT_SEED_PROBE_COMPLETE",
        "decision": decision,
        "diagnostic_dir_count": len(args.diagnostic_dir),
        "scored_pair_rows": len(variant_rows),
        "family_rows": len(family_rows),
        "worst_rows": len(worst_rows),
        "effective_ego_seed_count": ego_seed_count,
        "teammate_indices": teammate_indices,
        "failed_diagnostic_count": failed_diagnostic_count,
        "failed_diagnostics": args.failed_diagnostic,
        "state_sequence_complete": state_complete,
        "fixture_sign_validation_passed": fixture_ok,
        "any_positive_variant_worst": any_positive,
        "any_adverse_variant_worst": any_adverse,
        "delta_variant": args.delta_variant,
        "input_summaries": summaries,
        "output_files": {
            "seed_probe_accounting": str(args.output_dir / "seed_probe_accounting.csv"),
            "failed_match_accounting": str(args.output_dir / "failed_match_accounting.csv"),
            "domain_policy_matrix": str(args.output_dir / "domain_policy_matrix.csv"),
            "branch_state_validity": str(args.output_dir / "branch_state_validity.csv"),
            "return_independent_variant": str(args.output_dir / "return_independent_variant.csv"),
            "return_independent_family": str(args.output_dir / "return_independent_family_serd.csv"),
            "return_independent_worst": str(args.output_dir / "return_independent_worst_serd.csv"),
            "fixture_variant_validation": str(args.output_dir / "fixture_variant_validation.csv"),
            "pilot_recommendation": str(args.output_dir / "pilot_recommendation.md"),
            "operator_note": str(args.output_dir / "operator_note.md"),
        },
        "scope_boundary": (
            "WP-B pilot only; not a claim-bearing M4 bundle and not paper-upgrade evidence"
        ),
    }

    _write_csv(args.output_dir / "seed_probe_accounting.csv", seed_accounting, list(seed_accounting[0]))
    _write_csv(args.output_dir / "failed_match_accounting.csv", failed_match_rows, list(failed_match_rows[0]))
    _write_csv(args.output_dir / "domain_policy_matrix.csv", domain_policy_rows, list(domain_policy_rows[0]))
    _write_csv(args.output_dir / "branch_state_validity.csv", branch_validity, list(branch_validity[0]))
    _write_csv(args.output_dir / "return_independent_variant.csv", variant_rows, list(variant_rows[0]))
    _write_csv(args.output_dir / "return_independent_family_serd.csv", family_rows, list(family_rows[0]))
    _write_csv(args.output_dir / "return_independent_worst_serd.csv", worst_rows, list(worst_rows[0]))
    _write_csv(args.output_dir / "fixture_variant_validation.csv", fixture_rows, list(fixture_rows[0]))
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(args.output_dir / "pilot_recommendation.md", summary)
    _operator_note(args.output_dir / "operator_note.md", summary, args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
