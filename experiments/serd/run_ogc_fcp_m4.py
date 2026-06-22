"""Run native OGC FCP SERD BranchRecord probes and write an M4 bundle."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .fixture_env import CONTROL_FAMILIES
from .m4_bundle import write_m4_bundle
from .ogc_fcp_serd_adapter import OgcFcpSerdConfig, make_ogc_fcp_branch_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ogc-src", type=str, default=OgcFcpSerdConfig.ogc_src)
    parser.add_argument("--population-json", type=str, default=OgcFcpSerdConfig.population_json)
    parser.add_argument("--log-dir", type=str, default=OgcFcpSerdConfig.log_dir)
    parser.add_argument("--ego-xpid", type=str, required=True)
    parser.add_argument("--checkpoint-name", type=str, default=OgcFcpSerdConfig.checkpoint_name)
    parser.add_argument("--agent-id", type=int, default=OgcFcpSerdConfig.agent_id)
    parser.add_argument("--band", type=str, default=OgcFcpSerdConfig.band)
    parser.add_argument("--agent-idx", type=int, default=OgcFcpSerdConfig.agent_idx)
    parser.add_argument("--env-name", type=str, default=OgcFcpSerdConfig.env_name)
    parser.add_argument("--policy", type=str, default=OgcFcpSerdConfig.policy)
    parser.add_argument("--domain", type=str, default=OgcFcpSerdConfig.domain)
    parser.add_argument("--disruptions", type=str, default="missed_handoff,route_block,hesitation")
    parser.add_argument("--probes-per-disruption", type=int, default=4)
    parser.add_argument("--warmup-horizon", type=int, default=8)
    parser.add_argument("--rollout-horizon", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epsilon-shock", type=float, default=0.001)
    parser.add_argument("--epsilon-phi", type=float, default=0.0)
    parser.add_argument("--delta-serd", type=float, default=0.05)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/serd_m4/fcp_ogc_countercircuit6_9"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    disruptions = tuple(item.strip() for item in args.disruptions.split(",") if item.strip())
    adapter_config = OgcFcpSerdConfig(
        ogc_src=args.ogc_src,
        population_json=args.population_json,
        log_dir=args.log_dir,
        ego_xpid=args.ego_xpid,
        checkpoint_name=args.checkpoint_name,
        agent_id=args.agent_id,
        band=args.band,
        agent_idx=args.agent_idx,
        env_name=args.env_name,
        policy=args.policy,
        domain=args.domain,
        disruptions=disruptions,
        probes_per_disruption=args.probes_per_disruption,
        warmup_horizon=args.warmup_horizon,
        rollout_horizon=args.rollout_horizon,
        seed=args.seed,
    )
    semantic, controls, provenance = make_ogc_fcp_branch_records(adapter_config)
    run_config = {
        "adapter": "experiments.serd.ogc_fcp_serd_adapter",
        "adapter_config": asdict(adapter_config),
        "probes_per_disruption": args.probes_per_disruption,
        "rollout_horizon": args.rollout_horizon,
        "warmup_horizon": args.warmup_horizon,
        "epsilon_shock": args.epsilon_shock,
        "epsilon_phi": args.epsilon_phi,
        "delta_serd": args.delta_serd,
        "control_families": list(CONTROL_FAMILIES),
        "disruptions": list(disruptions),
        "m3_transition": "M3_TO_M4_FULL_SCOPE_READY",
    }
    provenance = {
        **provenance,
        "run_command_or_queue_manifest": "python -m experiments.serd.run_ogc_fcp_m4",
    }
    summary = write_m4_bundle(
        output_dir=args.output_dir,
        semantic_records=semantic,
        control_records=controls,
        provenance=provenance,
        run_config=run_config,
        acceptance_notes=[
            "Native OGC FCP adapter emitted BranchRecord-compatible rows.",
            "This bundle covers FCP only; full Workflow-2 readiness still requires PECAN M4 rows under the current scope.",
            "Project-level M4 acceptance must classify policy/domain completeness before review.",
        ],
    )
    print(json.dumps({"output_dir": str(args.output_dir), **summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
