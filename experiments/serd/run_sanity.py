"""Run a CPU-only SERD sanity check on deterministic fixture branches."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from .fixture_env import CONTROL_FAMILIES, make_fixture_records
from .serd_core import (
    require_family_coverage,
    summarize_family_serd,
    summarize_worst_serd,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon-shock", type=float, default=0.05)
    parser.add_argument("--epsilon-phi", type=float, default=0.05)
    parser.add_argument("--delta-serd", type=float, default=0.05)
    parser.add_argument("--output-json", type=Path, default=Path("results/serd_sanity/sanity.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("results/serd_sanity/family_serd.csv"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    semantic, controls = make_fixture_records(probes=args.probes, seed=args.seed)
    family = summarize_family_serd(
        semantic,
        controls,
        epsilon_shock=args.epsilon_shock,
        epsilon_phi=args.epsilon_phi,
        delta_serd=args.delta_serd,
    )
    missing = require_family_coverage(family, CONTROL_FAMILIES)
    worst = summarize_worst_serd(family, delta_serd=args.delta_serd)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "config": {
            "probes": args.probes,
            "seed": args.seed,
            "epsilon_shock": args.epsilon_shock,
            "epsilon_phi": args.epsilon_phi,
            "delta_serd": args.delta_serd,
            "required_families": list(CONTROL_FAMILIES),
        },
        "missing_families": missing,
        "family_serd": [asdict(item) for item in family],
        "worst_serd": [asdict(item) for item in worst],
    }
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "policy",
                "domain",
                "disruption",
                "family",
                "mean_serd",
                "ci95_low",
                "ci95_high",
                "n",
                "classification",
            ],
        )
        writer.writeheader()
        for item in family:
            writer.writerow(asdict(item))

    print(json.dumps({"output_json": str(args.output_json), "output_csv": str(args.output_csv)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
