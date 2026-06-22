"""Run policy-driven Overcooked-AI M1 SERD branch probes."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from .fixture_env import CONTROL_FAMILIES
from .overcooked_ai_adapter import (
    OvercookedDependencyError,
    OvercookedM1Config,
    all_records,
    make_overcooked_m1_policy_records,
)
from .serd_core import (
    require_family_coverage,
    summarize_family_serd,
    summarize_pre_h_balance,
    summarize_worst_serd,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overcooked-src", type=str, default=None)
    parser.add_argument("--layout", type=str, default="counter_circuit")
    parser.add_argument("--policy", type=str, default="handcoded_counter_circuit_policy")
    parser.add_argument(
        "--disruptions",
        type=str,
        default="missed_handoff,route_block,hesitation",
    )
    parser.add_argument("--probes-per-disruption", type=int, default=50)
    parser.add_argument("--rollout-horizon", type=int, default=20)
    parser.add_argument("--warmup-horizon", type=int, default=20)
    parser.add_argument("--shock-horizon", type=int, default=1)
    parser.add_argument("--shaped-reward-weight", type=float, default=1.0)
    parser.add_argument("--epsilon-shock", type=float, default=0.001)
    parser.add_argument("--epsilon-phi", type=float, default=0.0)
    parser.add_argument("--delta-serd", type=float, default=0.05)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/serd_overcooked_ai_m1/m1.json"),
    )
    parser.add_argument(
        "--branch-csv",
        type=Path,
        default=Path("results/serd_overcooked_ai_m1/branch_records.csv"),
    )
    parser.add_argument(
        "--family-csv",
        type=Path,
        default=Path("results/serd_overcooked_ai_m1/family_serd.csv"),
    )
    parser.add_argument(
        "--worst-csv",
        type=Path,
        default=Path("results/serd_overcooked_ai_m1/worst_serd.csv"),
    )
    parser.add_argument(
        "--balance-csv",
        type=Path,
        default=Path("results/serd_overcooked_ai_m1/pre_h_balance.csv"),
    )
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    disruptions = tuple(
        item.strip() for item in args.disruptions.split(",") if item.strip()
    )
    config = OvercookedM1Config(
        layout=args.layout,
        policy=args.policy,
        disruptions=disruptions,
        probes_per_disruption=args.probes_per_disruption,
        rollout_horizon=args.rollout_horizon,
        warmup_horizon=args.warmup_horizon,
        shock_horizon=args.shock_horizon,
        shaped_reward_weight=args.shaped_reward_weight,
        overcooked_src=args.overcooked_src,
    )
    try:
        semantic, controls = make_overcooked_m1_policy_records(config)
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
    balance = summarize_pre_h_balance(
        semantic,
        controls,
        epsilon_shock=args.epsilon_shock,
        epsilon_phi=args.epsilon_phi,
    )

    branch_rows = [asdict(item) for item in all_records(semantic, controls)]
    for row in branch_rows:
        row["phi_pre_h"] = json.dumps(row["phi_pre_h"], sort_keys=True)
    family_rows = [asdict(item) for item in family]
    worst_rows = [asdict(item) for item in worst]
    balance_rows = [asdict(item) for item in balance]

    valid_pairs_by_disruption = {}
    for disruption in disruptions:
        valid_pairs_by_disruption[disruption] = {
            family_name: 0 for family_name in CONTROL_FAMILIES
        }
    for item in family:
        valid_pairs_by_disruption[item.disruption][item.family] = item.n

    m1_gate_passed = (
        not missing
        and all(
            count >= args.probes_per_disruption
            for by_family in valid_pairs_by_disruption.values()
            for count in by_family.values()
        )
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "m1_policy_probe_adapter_passed_not_fcp_pecan_evidence"
        if m1_gate_passed
        else "m1_policy_probe_adapter_incomplete",
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
        "valid_pairs_by_disruption": valid_pairs_by_disruption,
        "missing_families": missing,
        "m1_gate_passed": m1_gate_passed,
        "family_serd": family_rows,
        "worst_serd": worst_rows,
        "pre_h_balance": balance_rows,
    }
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    _write_csv(
        args.branch_csv,
        branch_rows,
        [
            "probe_id",
            "policy",
            "domain",
            "disruption",
            "family",
            "no_shock_return",
            "branch_return",
            "shock_magnitude",
            "phi_pre_h",
        ],
    )
    _write_csv(
        args.family_csv,
        family_rows,
        [
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
    _write_csv(
        args.worst_csv,
        worst_rows,
        [
            "policy",
            "domain",
            "disruption",
            "mean_serd_worst",
            "ci95_low",
            "ci95_high",
            "n",
            "classification",
            "limiting_family",
        ],
    )
    _write_csv(
        args.balance_csv,
        balance_rows,
        [
            "policy",
            "domain",
            "disruption",
            "family",
            "covariate",
            "semantic_mean",
            "control_mean",
            "smd",
            "n",
        ],
    )

    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "branch_csv": str(args.branch_csv),
                "family_csv": str(args.family_csv),
                "worst_csv": str(args.worst_csv),
                "balance_csv": str(args.balance_csv),
                "branch_records": len(branch_rows),
                "m1_gate_passed": m1_gate_passed,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
