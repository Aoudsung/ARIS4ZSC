"""Score latency-aware SERD from saved PECAN branch diagnostics."""

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

from .fixture_env import CONTROL_FAMILIES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostic-dir",
        type=Path,
        required=True,
        help="Directory containing branch_trace_summary.csv and pair divergence CSV.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--delta-latency", type=float, default=0.5)
    parser.add_argument("--delta-discounted", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.95)
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


def _reward_sequence(row: dict[str, str]) -> list[float]:
    return [float(item) for item in ast.literal_eval(row["reward_sequence"])]


def _discounted_return(rewards: list[float], gamma: float) -> float:
    return sum((gamma**idx) * reward for idx, reward in enumerate(rewards))


def _first_reward_step(row: dict[str, str], horizon: int) -> float:
    value = row["first_reward_step"]
    if value == "" or value.lower() == "none":
        return float(horizon)
    return float(value)


def _summarize_family(
    pair_rows: list[dict[str, Any]],
    metric_key: str,
    delta: float,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in pair_rows:
        grouped[(row["disruption"], row["control_family"])].append(float(row[metric_key]))
    summaries = []
    for (disruption, family), values in sorted(grouped.items()):
        center, low, high = _ci95(values)
        summaries.append(
            {
                "policy": "PECAN",
                "domain": "ToMZSC-counter_circuit",
                "disruption": disruption,
                "family": family,
                "metric": metric_key,
                "mean_serd": center,
                "ci95_low": low,
                "ci95_high": high,
                "n": len(values),
                "classification": _classify(low, high, delta),
            }
        )
    return summaries


def _summarize_worst(
    family_rows: list[dict[str, Any]],
    delta: float,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in family_rows:
        grouped[row["disruption"]].append(row)
    worst = []
    for disruption, rows in sorted(grouped.items()):
        limiting = min(rows, key=lambda row: float(row["mean_serd"]))
        low = float(limiting["ci95_low"])
        high = float(limiting["ci95_high"])
        worst.append(
            {
                "policy": "PECAN",
                "domain": "ToMZSC-counter_circuit",
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
    return worst


def main() -> int:
    args = parse_args()
    branch_rows = _read_csv(args.diagnostic_dir / "branch_trace_summary.csv")
    rows_by_key = {
        (row["probe_id"], row["condition"]): row
        for row in branch_rows
    }
    pair_rows: list[dict[str, Any]] = []
    horizon = max(len(_reward_sequence(row)) for row in branch_rows)
    for row in _read_csv(args.diagnostic_dir / "semantic_control_pair_divergence.csv"):
        probe_id = row["probe_id"]
        family = row["control_family"]
        no_shock = rows_by_key[(probe_id, "no_shock")]
        semantic = rows_by_key[(probe_id, "semantic")]
        control = rows_by_key[(probe_id, family)]

        no_shock_first = _first_reward_step(no_shock, horizon)
        semantic_first = _first_reward_step(semantic, horizon)
        control_first = _first_reward_step(control, horizon)
        semantic_latency_loss = semantic_first - no_shock_first
        control_latency_loss = control_first - no_shock_first

        no_shock_discounted = _discounted_return(_reward_sequence(no_shock), args.gamma)
        semantic_discounted = _discounted_return(_reward_sequence(semantic), args.gamma)
        control_discounted = _discounted_return(_reward_sequence(control), args.gamma)
        semantic_discounted_loss = no_shock_discounted - semantic_discounted
        control_discounted_loss = no_shock_discounted - control_discounted

        pair_rows.append(
            {
                "probe_id": probe_id,
                "disruption": row["disruption"],
                "probe_index": row["probe_index"],
                "control_family": family,
                "no_shock_first_reward_step": no_shock_first,
                "semantic_first_reward_step": semantic_first,
                "control_first_reward_step": control_first,
                "semantic_latency_loss": semantic_latency_loss,
                "control_latency_loss": control_latency_loss,
                "latency_serd": control_latency_loss - semantic_latency_loss,
                "gamma": args.gamma,
                "no_shock_discounted_return": no_shock_discounted,
                "semantic_discounted_return": semantic_discounted,
                "control_discounted_return": control_discounted,
                "semantic_discounted_loss": semantic_discounted_loss,
                "control_discounted_loss": control_discounted_loss,
                "discounted_serd": control_discounted_loss - semantic_discounted_loss,
            }
        )

    latency_family = _summarize_family(pair_rows, "latency_serd", args.delta_latency)
    latency_worst = _summarize_worst(latency_family, args.delta_latency)
    discounted_family = _summarize_family(pair_rows, "discounted_serd", args.delta_discounted)
    discounted_worst = _summarize_worst(discounted_family, args.delta_discounted)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        args.output_dir / "latency_pair_serd.csv",
        pair_rows,
        list(pair_rows[0]),
    )
    family_fields = [
        "policy",
        "domain",
        "disruption",
        "family",
        "metric",
        "mean_serd",
        "ci95_low",
        "ci95_high",
        "n",
        "classification",
    ]
    worst_fields = [
        "policy",
        "domain",
        "disruption",
        "metric",
        "mean_serd_worst",
        "ci95_low",
        "ci95_high",
        "n",
        "classification",
        "limiting_family",
    ]
    _write_csv(args.output_dir / "latency_family_serd.csv", latency_family, family_fields)
    _write_csv(args.output_dir / "latency_worst_serd.csv", latency_worst, worst_fields)
    _write_csv(
        args.output_dir / "discounted_family_serd.csv",
        discounted_family,
        family_fields,
    )
    _write_csv(args.output_dir / "discounted_worst_serd.csv", discounted_worst, worst_fields)
    summary = {
        "status": "LATENCY_SERD_RESCORING_COMPLETE",
        "diagnostic_dir": str(args.diagnostic_dir),
        "gamma": args.gamma,
        "delta_latency": args.delta_latency,
        "delta_discounted": args.delta_discounted,
        "pair_rows": len(pair_rows),
        "latency_worst_rows": latency_worst,
        "discounted_worst_rows": discounted_worst,
        "any_positive_latency_worst": any(
            float(row["mean_serd_worst"]) > 0.0 for row in latency_worst
        ),
        "any_positive_discounted_worst": any(
            float(row["mean_serd_worst"]) > 0.0 for row in discounted_worst
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
