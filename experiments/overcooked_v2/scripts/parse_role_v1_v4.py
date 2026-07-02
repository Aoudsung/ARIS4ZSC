from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any


METHODS = ("aris_bellman", "base_only", "global_gru", "flat_factor")
CLAIM_PARTNER = "heldout-yield-terminal-claim"
KIND_COLUMNS = (
    ("fetch", "fetch_ingredient"),
    ("deliver", "deliver_ingredient_to_pot"),
    ("pick", "pick_plate"),
    ("plate", "plate_soup"),
    ("serve", "serve_soup"),
    ("clear", "clear_interaction_cell"),
    ("drop", "drop_item_to_counter"),
    ("wait", "wait_at_bottleneck"),
    ("cross", "cross_bottleneck"),
    ("noop", "noop"),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print role_conditioned_v1 v4 claim-partner option-kind breakdown."
    )
    parser.add_argument("--results-dir", default="results_role_v1_v4")
    parser.add_argument("--partner", default=CLAIM_PARTNER)
    parser.add_argument("--methods", default=",".join(METHODS))
    args = parser.parse_args()

    methods = [item.strip() for item in str(args.methods).split(",") if item.strip()]
    rows = [
        _summarize_method(Path(args.results_dir), method, str(args.partner))
        for method in methods
    ]
    rows = [row for row in rows if row is not None]

    header = ["method", "total_opts"] + [name for name, _kind in KIND_COLUMNS] + ["timeout%"]
    widths = [max(len(header[idx]), 10 if idx == 0 else 8) for idx in range(len(header))]
    print(" ".join(value.ljust(widths[idx]) for idx, value in enumerate(header)))
    for row in rows:
        values = [
            str(row["method"]),
            f"{row['total_opts']:.1f}",
            *[f"{row[name]:.1f}" for name, _kind in KIND_COLUMNS],
            f"{row['timeout_pct']:.1f}",
        ]
        print(" ".join(values[idx].ljust(widths[idx]) for idx in range(len(values))))


def _summarize_method(results_dir: Path, method: str, partner: str) -> dict[str, Any] | None:
    files = sorted(glob.glob(str(results_dir / method / "seed*" / "eval_heldout.json")))
    if not files:
        return None
    per_seed = []
    for file_name in files:
        data = json.loads(Path(file_name).read_text(encoding="utf-8"))
        for result in data.get("results", []):
            if result.get("partner") == partner:
                per_seed.append(_row_from_aggregate(result.get("aggregate", {}) or {}))
    if not per_seed:
        return None

    out: dict[str, Any] = {"method": method}
    for key in ("total_opts", "timeout_pct", *(name for name, _kind in KIND_COLUMNS)):
        out[key] = sum(float(row.get(key, 0.0)) for row in per_seed) / float(len(per_seed))
    return out


def _row_from_aggregate(aggregate: dict[str, Any]) -> dict[str, float]:
    stats = aggregate.get("option_kind_stats", {}) or {}
    row: dict[str, float] = {}
    total_opts = 0.0
    timeout = 0.0
    for item in stats.values():
        total_opts += float(item.get("attempt_count", 0.0))
        timeout += float(item.get("timeout_count", 0.0))
    row["total_opts"] = total_opts
    row["timeout_pct"] = 100.0 * timeout / max(1.0, total_opts)
    for name, kind in KIND_COLUMNS:
        row[name] = float((stats.get(kind) or {}).get("attempt_count", 0.0))
    return row


if __name__ == "__main__":
    main()
