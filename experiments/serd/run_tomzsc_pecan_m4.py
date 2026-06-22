"""Run reproduced ToMZSC PECAN SERD probes and write an M4 bundle."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .fixture_env import CONTROL_FAMILIES
from .m4_bundle import write_m4_bundle
from .tomzsc_pecan_serd_adapter import (
    TomzscPecanSerdConfig,
    make_tomzsc_pecan_branch_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tomzsc-root", type=str, default=TomzscPecanSerdConfig.tomzsc_root)
    parser.add_argument("--teammate-dir", type=str, default=TomzscPecanSerdConfig.teammate_dir)
    parser.add_argument("--ego-dir", type=str, default=TomzscPecanSerdConfig.ego_dir)
    parser.add_argument("--cluster-labels", type=str, default=TomzscPecanSerdConfig.cluster_labels)
    parser.add_argument(
        "--reproduction-manifest",
        type=str,
        default=TomzscPecanSerdConfig.reproduction_manifest,
    )
    parser.add_argument("--layout", type=str, default=TomzscPecanSerdConfig.layout)
    parser.add_argument("--policy", type=str, default=TomzscPecanSerdConfig.policy)
    parser.add_argument("--domain", type=str, default=TomzscPecanSerdConfig.domain)
    parser.add_argument("--disruptions", type=str, default="missed_handoff,route_block,hesitation")
    parser.add_argument("--probes-per-disruption", type=int, default=4)
    parser.add_argument("--warmup-horizon", type=int, default=8)
    parser.add_argument("--rollout-horizon", type=int, default=20)
    parser.add_argument(
        "--probe-mode",
        type=str,
        default=TomzscPecanSerdConfig.probe_mode,
        choices=("policy_warmup", "reward_event"),
    )
    parser.add_argument(
        "--max-probe-episodes",
        type=int,
        default=TomzscPecanSerdConfig.max_probe_episodes,
    )
    parser.add_argument("--seed", type=int, default=154)
    parser.add_argument("--teammate-index", type=int, default=0)
    parser.add_argument("--ego-index", type=int, default=0)
    parser.add_argument("--teammate-agent-id", type=int, default=0)
    parser.add_argument("--ego-agent-id", type=int, default=1)
    parser.add_argument("--epsilon-shock", type=float, default=0.001)
    parser.add_argument("--epsilon-phi", type=float, default=0.0)
    parser.add_argument("--delta-serd", type=float, default=0.05)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/serd_m4/pecan_tomzsc_counter_circuit"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    disruptions = tuple(item.strip() for item in args.disruptions.split(",") if item.strip())
    adapter_config = TomzscPecanSerdConfig(
        tomzsc_root=args.tomzsc_root,
        teammate_dir=args.teammate_dir,
        ego_dir=args.ego_dir,
        cluster_labels=args.cluster_labels,
        reproduction_manifest=args.reproduction_manifest,
        policy=args.policy,
        domain=args.domain,
        layout=args.layout,
        disruptions=disruptions,
        probes_per_disruption=args.probes_per_disruption,
        warmup_horizon=args.warmup_horizon,
        rollout_horizon=args.rollout_horizon,
        probe_mode=args.probe_mode,
        max_probe_episodes=args.max_probe_episodes,
        seed=args.seed,
        teammate_index=args.teammate_index,
        ego_index=args.ego_index,
        teammate_agent_id=args.teammate_agent_id,
        ego_agent_id=args.ego_agent_id,
    )
    semantic, controls, provenance = make_tomzsc_pecan_branch_records(adapter_config)
    run_config = {
        "adapter": "experiments.serd.tomzsc_pecan_serd_adapter",
        "adapter_config": asdict(adapter_config),
        "probes_per_disruption": args.probes_per_disruption,
        "rollout_horizon": args.rollout_horizon,
        "warmup_horizon": args.warmup_horizon,
        "probe_mode": args.probe_mode,
        "max_probe_episodes": args.max_probe_episodes,
        "epsilon_shock": args.epsilon_shock,
        "epsilon_phi": args.epsilon_phi,
        "delta_serd": args.delta_serd,
        "control_families": list(CONTROL_FAMILIES),
        "disruptions": list(disruptions),
        "m3_transition": "M3_TO_M4_FULL_SCOPE_READY",
    }
    provenance = {
        **provenance,
        "run_command_or_queue_manifest": "python -m experiments.serd.run_tomzsc_pecan_m4",
    }
    summary = write_m4_bundle(
        output_dir=args.output_dir,
        semantic_records=semantic,
        control_records=controls,
        provenance=provenance,
        run_config=run_config,
        acceptance_notes=[
            "Reproduced target-domain PECAN checkpoints emitted BranchRecord-compatible rows.",
            "Adapter reuses the patched ToMZSC/JaxMARL stack from the accepted M3 PECAN reproduction.",
            "Project-level M4 acceptance must combine this bundle with the accepted FCP bundle.",
        ],
    )
    print(json.dumps({"output_dir": str(args.output_dir), **summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
