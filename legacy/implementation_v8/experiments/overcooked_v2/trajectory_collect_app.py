"""E2 trajectory collection (exploration track).

Registered protocol: docs/research/TRAJECTORY_AND_ESTIMATION_SPEC.md section
one.  A fixed ego policy q plays full Official episodes against each upstream
partner run; every partner action index is recorded for the kappa/TV
estimators.  No training, no observation tensors.  Sharding follows the panel
tool pattern; episode keys are identical across partners (paired design) and
disjoint from the panel key stream by a fixed root offset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.overcooked_v2.official_adapter import (
    official_pairing_rollouts,
    official_policy,
    restore_official_checkpoint,
    validate_official_runtime,
)
from experiments.overcooked_v2.official_evaluation_app import _env_kwargs
from experiments.overcooked_v2.panel_opportunity_app import load_panel_manifest
from src.path_c.experiment import (
    OFFICIAL_EPISODE_STEPS,
    OFFICIAL_EVALUATION_ROOT_SEED,
    RunConfig,
    load_config,
)
from src.path_c.storage import write_json

TRAJECTORY_ROOT_OFFSET = 1_000
DEFAULT_EGO_RUN = "rnn-sp-seed-0"
ACTION_RANGE = 6


def make_environment(config: RunConfig) -> Any:
    from jaxmarl.environments.overcooked_v2.layouts import overcooked_v2_layouts
    from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2

    return OvercookedV2(
        layout=overcooked_v2_layouts[config.environment.layout],
        **dict(_env_kwargs(config)),
    )


def load_policy(checkpoint: str) -> Any:
    official_config, params = restore_official_checkpoint(Path(checkpoint))
    return official_policy(params, official_config)


def partner_entries(manifest: dict) -> list[dict]:
    entries = []
    for entry in manifest["entries"]:
        for run in entry["runs"]:
            entries.append(
                {
                    "partner_type": entry["type"],
                    "run_id": str(run["run_id"]),
                    "checkpoint": str(run["checkpoint"]),
                }
            )
    return entries


def collect(args: argparse.Namespace) -> None:
    import jax

    validate_official_runtime()
    config = load_config(args.config, run_kind="formal")
    manifest = load_panel_manifest(args.manifest)
    if manifest["layout"] != config.environment.layout:
        raise RuntimeError("Panel manifest layout differs from the config layout.")

    partners = partner_entries(manifest)
    ego = next((p for p in partners if p["run_id"] == args.ego_run), None)
    if ego is None:
        raise RuntimeError(f"ego run not found in manifest: {args.ego_run}")
    if args.shard_count > 1:
        if not 0 <= args.shard_index < args.shard_count:
            raise RuntimeError("shard-index out of range")
        block = (len(partners) + args.shard_count - 1) // args.shard_count
        partners = partners[args.shard_index * block : (args.shard_index + 1) * block]
        if not partners:
            raise RuntimeError("empty shard slice")

    environment = make_environment(config)
    ego_policy = load_policy(ego["checkpoint"])
    root_key = jax.random.PRNGKey(
        OFFICIAL_EVALUATION_ROOT_SEED + TRAJECTORY_ROOT_OFFSET
    )

    records = []
    for partner in partners:
        partner_policy = load_policy(partner["checkpoint"])
        rollouts, _keys = official_pairing_rollouts(
            left_policy=ego_policy,
            right_policy=partner_policy,
            environment=environment,
            root_key=root_key,
            episodes=int(args.episodes),
        )
        actions = np.asarray(rollouts.actions_seq["agent_1"], dtype=np.int64)
        rewards = np.asarray(rollouts.total_reward, dtype=np.float64)
        if actions.shape != (int(args.episodes), OFFICIAL_EPISODE_STEPS):
            raise RuntimeError(
                f"partner action tensor has invalid shape {actions.shape}"
            )
        if actions.min() < 0 or actions.max() >= ACTION_RANGE:
            raise RuntimeError("partner action index out of registered range")
        if not np.all(np.isfinite(rewards)):
            raise RuntimeError("rollout rewards contain non-finite values")
        records.append(
            {
                "partner_type": partner["partner_type"],
                "partner_run_id": partner["run_id"],
                "checkpoint": partner["checkpoint"],
                "episodes": int(args.episodes),
                "episode_index": list(range(int(args.episodes))),
                "total_reward": rewards.tolist(),
                "partner_actions": actions.tolist(),
            }
        )
        print(f"collected {partner['run_id']}")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "trajectories.json",
        {
            "ego_run": args.ego_run,
            "layout": manifest["layout"],
            "episodes_per_partner": int(args.episodes),
            "episode_steps": OFFICIAL_EPISODE_STEPS,
            "root_seed_offset": TRAJECTORY_ROOT_OFFSET,
            "records": records,
        },
    )
    print(json.dumps({"partners": len(records), "episodes": int(args.episodes)}))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E2 partner trajectory collection (exploration track)."
    )
    parser.add_argument("--config", required=True, help="Formal run config yaml.")
    parser.add_argument(
        "--manifest", required=True, help="Panel manifest listing all runs."
    )
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument(
        "--episodes", type=int, default=100, help="Episodes per partner."
    )
    parser.add_argument("--ego-run", default=DEFAULT_EGO_RUN, help="Ego run id.")
    parser.add_argument("--shard-index", type=int, default=0, help="Shard index.")
    parser.add_argument("--shard-count", type=int, default=1, help="Total shards.")
    args = parser.parse_args()
    if args.episodes <= 0:
        raise SystemExit("episodes must be positive")
    collect(args)


if __name__ == "__main__":
    main()
