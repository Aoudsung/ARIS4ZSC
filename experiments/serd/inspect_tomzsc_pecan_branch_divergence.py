"""Inspect branch-level divergence for reproduced ToMZSC PECAN SERD probes."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from .fixture_env import CONTROL_FAMILIES
from .tomzsc_pecan_serd_adapter import (
    ACTION_NAMES,
    TomzscPecanSerdConfig,
    _control_override,
    _make_transition_fn,
    _phi_pre_h,
    _reward_event_snapshots,
    _reward_scalar,
    _semantic_override,
    _transition,
    load_tomzsc_pecan_bundle,
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
    parser.add_argument("--rollout-horizon", type=int, default=20)
    parser.add_argument("--max-probe-episodes", type=int, default=60)
    parser.add_argument("--seed", type=int, default=154)
    parser.add_argument("--teammate-index", type=int, default=0)
    parser.add_argument("--ego-index", type=int, default=0)
    parser.add_argument("--teammate-agent-id", type=int, default=1)
    parser.add_argument("--ego-agent-id", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/serd_branch_diagnostics/pecan_tomzsc_counter_circuit"),
    )
    return parser.parse_args()


def _scalar(value: Any) -> float:
    return float(jax.device_get(jnp.asarray(value)))


def _state_summary(state: Any) -> dict[str, Any]:
    raw = jax.device_get(state.env_state)
    summary: dict[str, Any] = {}
    for name in ("time", "agent_pos", "agent_dir_idx", "agent_inv"):
        value = getattr(raw, name, None)
        if value is not None:
            array = jax.device_get(jnp.asarray(value))
            summary[name] = array.tolist()
    return summary


def _compact_state_key(state: Any) -> str:
    summary = _state_summary(state)
    return json.dumps(summary, sort_keys=True, separators=(",", ":"))


def _trace_rollout(
    bundle: Any,
    transition_fn: Any,
    snapshot: tuple[Any, ...],
    horizon: int,
    first_override: tuple[int, str] | None,
) -> dict[str, Any]:
    rng, state, obs, teammate_hs, ego_hs, done = snapshot
    rewards: list[float] = []
    actions: list[dict[str, int | str]] = []
    states: list[str] = [_compact_state_key(state)]
    positions: list[Any] = []
    total = 0.0
    first_reward_step: int | None = None
    for step_idx in range(horizon):
        override = first_override if step_idx == 0 else None
        (
            rng,
            state,
            obs,
            teammate_hs,
            ego_hs,
            done,
            reward,
            _info,
            step_actions,
        ) = _transition(
            bundle,
            transition_fn,
            rng,
            state,
            obs,
            teammate_hs,
            ego_hs,
            done,
            override,
        )
        reward_value = _reward_scalar(bundle, reward)
        if reward_value > 0.0 and first_reward_step is None:
            first_reward_step = step_idx
        total += reward_value
        rewards.append(reward_value)
        action_row: dict[str, int | str] = {}
        for agent_name in bundle.env.agents:
            action_id = int(jax.device_get(jnp.asarray(step_actions[agent_name])))
            action_row[agent_name] = action_id
            action_row[f"{agent_name}_name"] = ACTION_NAMES[action_id]
        actions.append(action_row)
        state_summary = _state_summary(state)
        positions.append(state_summary.get("agent_pos"))
        states.append(json.dumps(state_summary, sort_keys=True, separators=(",", ":")))
        if bool(jax.device_get(jnp.asarray(done))):
            break
    return {
        "total_return": total,
        "first_reward_step": first_reward_step,
        "rewards": rewards,
        "actions": actions,
        "positions": positions,
        "states": states,
    }


def _override_label(override: tuple[int, str] | None) -> str:
    if override is None:
        return "none"
    actor, action = override
    return f"agent_{actor}:{action}"


def main() -> int:
    args = parse_args()
    disruptions = tuple(item.strip() for item in args.disruptions.split(",") if item.strip())
    config = TomzscPecanSerdConfig(
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
        rollout_horizon=args.rollout_horizon,
        probe_mode="reward_event",
        max_probe_episodes=args.max_probe_episodes,
        seed=args.seed,
        teammate_index=args.teammate_index,
        ego_index=args.ego_index,
        teammate_agent_id=args.teammate_agent_id,
        ego_agent_id=args.ego_agent_id,
    )
    bundle = load_tomzsc_pecan_bundle(config)
    transition_fn = _make_transition_fn(bundle)
    rng = jax.random.PRNGKey(config.seed)
    needed = len(disruptions) * args.probes_per_disruption
    _rng, snapshots, reward_event_meta = _reward_event_snapshots(
        bundle,
        transition_fn,
        rng,
        needed,
    )

    branch_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    snapshot_index = 0
    for disruption in disruptions:
        for probe_index in range(args.probes_per_disruption):
            snapshot = snapshots[snapshot_index]
            snapshot_index += 1
            probe_id = f"{config.policy}:{config.domain}:{disruption}:{probe_index:04d}"
            phi = _phi_pre_h(snapshot, probe_index)
            conditions: list[tuple[str, tuple[int, str] | None]] = [
                ("no_shock", None),
                ("semantic", _semantic_override(disruption, probe_index)),
            ]
            conditions.extend(
                (family, _control_override(disruption, family, probe_index))
                for family in CONTROL_FAMILIES
            )
            traces = {
                condition: _trace_rollout(
                    bundle,
                    transition_fn,
                    snapshot,
                    args.rollout_horizon,
                    override,
                )
                for condition, override in conditions
            }
            for condition, override in conditions:
                trace = traces[condition]
                first_action = trace["actions"][0] if trace["actions"] else {}
                branch_rows.append(
                    {
                        "probe_id": probe_id,
                        "disruption": disruption,
                        "probe_index": probe_index,
                        "condition": condition,
                        "override": _override_label(override),
                        "total_return": trace["total_return"],
                        "first_reward_step": trace["first_reward_step"],
                        "first_agent_0_action": first_action.get("agent_0_name", ""),
                        "first_agent_1_action": first_action.get("agent_1_name", ""),
                        "reward_sequence": json.dumps(trace["rewards"]),
                        "action_sequence": json.dumps(trace["actions"], sort_keys=True),
                        "position_sequence": json.dumps(trace["positions"], sort_keys=True),
                        "state_sequence": json.dumps(trace["states"], sort_keys=True),
                        "phi_pre_h": json.dumps(phi, sort_keys=True),
                    }
                )
            semantic = traces["semantic"]
            for family in CONTROL_FAMILIES:
                control = traces[family]
                pair_rows.append(
                    {
                        "probe_id": probe_id,
                        "disruption": disruption,
                        "probe_index": probe_index,
                        "control_family": family,
                        "semantic_return": semantic["total_return"],
                        "control_return": control["total_return"],
                        "return_delta": control["total_return"] - semantic["total_return"],
                        "reward_sequence_equal": semantic["rewards"] == control["rewards"],
                        "action_sequence_equal": semantic["actions"] == control["actions"],
                        "position_sequence_equal": semantic["positions"] == control["positions"],
                        "state_sequence_equal": semantic["states"] == control["states"],
                        "semantic_first_reward_step": semantic["first_reward_step"],
                        "control_first_reward_step": control["first_reward_step"],
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    branch_path = args.output_dir / "branch_trace_summary.csv"
    pair_path = args.output_dir / "semantic_control_pair_divergence.csv"
    with branch_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(branch_rows[0]))
        writer.writeheader()
        writer.writerows(branch_rows)
    with pair_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)
    summary = {
        "status": "PECAN_BRANCH_DIVERGENCE_DIAGNOSTIC_COMPLETE",
        "branch_rows": len(branch_rows),
        "pair_rows": len(pair_rows),
        "config": asdict(config),
        "reward_event_meta": reward_event_meta,
        "any_return_delta": any(float(row["return_delta"]) != 0.0 for row in pair_rows),
        "any_reward_sequence_divergence": any(
            not bool(row["reward_sequence_equal"]) for row in pair_rows
        ),
        "any_action_sequence_divergence": any(
            not bool(row["action_sequence_equal"]) for row in pair_rows
        ),
        "any_position_sequence_divergence": any(
            not bool(row["position_sequence_equal"]) for row in pair_rows
        ),
        "output_files": {
            "branch_trace_summary": str(branch_path),
            "semantic_control_pair_divergence": str(pair_path),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
