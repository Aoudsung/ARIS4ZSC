"""Write a fixture M4 bundle that must contain a positive SERD control.

This runner is a metric/bundle positive control, not FCP/PECAN evidence. It
uses deterministic fixture BranchRecords where `pecan_fixture` loses less under
semantic disruption than under every matched control family, so `SERD_worst`
must be positive if the metric and bundle writer are not flooring all rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .fixture_env import CONTROL_FAMILIES, make_fixture_records
from .m4_bundle import write_m4_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon-shock", type=float, default=0.05)
    parser.add_argument("--epsilon-phi", type=float, default=0.05)
    parser.add_argument("--delta-serd", type=float, default=0.05)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/serd_m4_positive_control/fixture_counter_circuit"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    semantic, controls = make_fixture_records(probes=args.probes, seed=args.seed)
    run_config = {
        "adapter": "experiments.serd.fixture_env",
        "adapter_config": {
            "domain": "fixture_counter_circuit",
            "probes": args.probes,
            "seed": args.seed,
            "expected_positive_policy": "pecan_fixture",
            "expected_non_positive_policy": "fcp_fixture",
        },
        "probes_per_disruption": args.probes,
        "rollout_horizon": None,
        "warmup_horizon": None,
        "epsilon_shock": args.epsilon_shock,
        "epsilon_phi": args.epsilon_phi,
        "delta_serd": args.delta_serd,
        "control_families": list(CONTROL_FAMILIES),
        "disruptions": ["missed_handoff"],
        "m3_transition": "M4_POSITIVE_CONTROL_NOT_POLICY_EVIDENCE",
    }
    provenance = {
        "policy_family": "fixture_positive_control",
        "source_repository_path": "experiments/serd/fixture_env.py",
        "source_commit_hash": None,
        "adapter_route": "deterministic fixture BranchRecord generator",
        "simulator_or_environment": "deterministic fixture",
        "target_domain_or_layout": "fixture_counter_circuit",
        "random_seeds": [args.seed],
        "run_command_or_queue_manifest": "python -m experiments.serd.run_m4_positive_control",
        "m3_acceptance_artifact": None,
        "claim_boundary": "metric positive control only; not FCP, PECAN, human, or policy evidence",
    }
    summary = write_m4_bundle(
        output_dir=args.output_dir,
        semantic_records=semantic,
        control_records=controls,
        provenance=provenance,
        run_config=run_config,
        acceptance_notes=[
            "Deterministic fixture M4 positive-control bundle.",
            "Expected positive control: pecan_fixture should have positive SERD_worst.",
            "Expected non-positive control: fcp_fixture should not support a survival claim.",
            "This bundle resolves metric-floor sanity only; it is not policy or paper evidence.",
        ],
    )
    print(json.dumps({"output_dir": str(args.output_dir), **summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
