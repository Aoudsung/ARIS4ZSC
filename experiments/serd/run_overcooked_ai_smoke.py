"""Run a scripted Overcooked-AI SERD branch adapter smoke test."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from .fixture_env import CONTROL_FAMILIES
from .overcooked_ai_adapter import (
    OvercookedDependencyError,
    OvercookedSmokeConfig,
    all_records,
    make_overcooked_smoke_records,
)
from .serd_core import (
    require_family_coverage,
    summarize_family_serd,
    summarize_worst_serd,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overcooked-src", type=str, default=None)
    parser.add_argument("--layout", type=str, default="counter_circuit")
    parser.add_argument("--policy", type=str, default="scripted_smoke")
    parser.add_argument("--disruption", type=str, default="missed_handoff")
    parser.add_argument("--probes", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--shock-horizon", type=int, default=1)
    parser.add_argument("--shaped-reward-weight", type=float, default=1.0)
    parser.add_argument("--epsilon-shock", type=float, default=0.001)
    parser.add_argument("--epsilon-phi", type=float, default=0.0)
    parser.add_argument("--delta-serd", type=float, default=0.05)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/serd_overcooked_ai_smoke/smoke.json"),
    )
    parser.add_argument(
        "--branch-csv",
        type=Path,
        default=Path("results/serd_overcooked_ai_smoke/branch_records.csv"),
    )
    parser.add_argument(
        "--family-csv",
        type=Path,
        default=Path("results/serd_overcooked_ai_smoke/family_serd.csv"),
    )
    return parser.parse_args()


def _write_branch_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "probe_id",
        "policy",
        "domain",
        "disruption",
        "family",
        "no_shock_return",
        "branch_return",
        "shock_magnitude",
        "phi_pre_h",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            serialized["phi_pre_h"] = json.dumps(
                serialized["phi_pre_h"], sort_keys=True
            )
            writer.writerow(serialized)


def _write_family_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "policy",
        "domain",
        "disruption",
        "family",
        "mean_serd",
        "ci95_low",
        "ci95_high",
        "n",
        "classification",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    config = OvercookedSmokeConfig(
        layout=args.layout,
        policy=args.policy,
        disruption=args.disruption,
        probes=args.probes,
        horizon=args.horizon,
        shock_horizon=args.shock_horizon,
        shaped_reward_weight=args.shaped_reward_weight,
        overcooked_src=args.overcooked_src,
    )
    try:
        semantic, controls = make_overcooked_smoke_records(config)
    except OvercookedDependencyError as exc:
        raise SystemExit(str(exc)) from exc

    family = summarize_family_serd(
        semantic,
        controls,
        epsilon_shock=args.epsilon_shock,
        epsilon_phi=args.epsilon_phi,
        delta_serd=args.delta_serd,
    )
    missing = require_family_coverage(family, CONTROL_FAMILIES)
    worst = summarize_worst_serd(family, delta_serd=args.delta_serd)

    branch_rows = [asdict(item) for item in all_records(semantic, controls)]
    family_rows = [asdict(item) for item in family]
    worst_rows = [asdict(item) for item in worst]

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "smoke_only_not_scientific_evidence",
        "config": asdict(config),
        "matching": {
            "epsilon_shock": args.epsilon_shock,
            "epsilon_phi": args.epsilon_phi,
            "delta_serd": args.delta_serd,
            "required_families": list(CONTROL_FAMILIES),
        },
        "counts": {
            "semantic_records": len(semantic),
            "control_records": len(controls),
            "branch_records": len(branch_rows),
        },
        "missing_families": missing,
        "family_serd": family_rows,
        "worst_serd": worst_rows,
    }
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_branch_csv(args.branch_csv, branch_rows)
    _write_family_csv(args.family_csv, family_rows)

    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "branch_csv": str(args.branch_csv),
                "family_csv": str(args.family_csv),
                "branch_records": len(branch_rows),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
