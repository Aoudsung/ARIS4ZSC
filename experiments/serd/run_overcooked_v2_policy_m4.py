"""Run SERD M4 probes with real OvercookedV2 PPO checkpoints.

This runner connects the official OvercookedV2 RNN policy checkpoints to the
existing SERD BranchRecord schema. It is the direct policy/domain pivot after
the counter_circuit PECAN route exhausted the oral-upgrade path: source-only or
handcoded OvercookedV2 records are not used here.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from math import sqrt
from typing import Any, Iterable

from .fixture_env import CONTROL_FAMILIES
from .jaxmarl_overcooked_v2_adapter import (
    ensure_jaxmarl_overcooked_v2,
    state_phi_pre_h,
)
from .m4_bundle import write_m4_bundle
from .serd_core import BranchRecord


ACTION_NAMES_BY_ID = {
    0: "right",
    1: "down",
    2: "left",
    3: "up",
    4: "stay",
    5: "interact",
}

ACTION_ID_BY_NAME = {name: idx for idx, name in ACTION_NAMES_BY_ID.items()}
CHANNELS = ("action", "position", "state")


@dataclass(frozen=True)
class PolicySpec:
    label: str
    run_dir: Path
    method: str


@dataclass(frozen=True)
class LoadedPolicyPair:
    label: str
    ego_run_num: int
    partner_run_num: int
    ego_policy: Any
    partner_policy: Any
    config: dict[str, Any]


@dataclass(frozen=True)
class RolloutResult:
    total_return: float
    action_sequence: list[tuple[int, int]]
    position_sequence: list[tuple[tuple[int, int], tuple[int, int]]]
    state_sequence: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tg-ssa-root",
        type=Path,
        default=Path("/apps/users/cxw/Document/CodeSpace/Selfs/TG-SSA"),
    )
    parser.add_argument(
        "--jaxmarl-src",
        type=Path,
        default=Path("/apps/users/cxw/Document/CodeSpace/Selfs/TG-SSA/external/JaxMARL"),
    )
    parser.add_argument("--layout", type=str, default="grounded_coord_simple")
    parser.add_argument("--domain", type=str, default="jaxmarl_overcooked_v2:grounded_coord_simple")
    parser.add_argument(
        "--policy-run",
        action="append",
        required=True,
        help="Policy spec in label:method:run_dir form, e.g. rnn_sp:rnn-sp:/path/to/run_dir.",
    )
    parser.add_argument(
        "--run-nums",
        type=str,
        default="0,1,2,3,4,5,6,7,8,9",
        help="Comma-separated checkpoint run numbers to evaluate.",
    )
    parser.add_argument(
        "--pairing",
        choices=("self", "cross_next"),
        default="cross_next",
        help="self uses one checkpoint for both agents; cross_next pairs run i with run i+1.",
    )
    parser.add_argument("--disruptions", type=str, default="missed_handoff,route_block,hesitation")
    parser.add_argument("--probes-per-disruption", type=int, default=4)
    parser.add_argument("--warmup-horizon", type=int, default=8)
    parser.add_argument("--rollout-horizon", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shaped-reward-weight", type=float, default=1.0)
    parser.add_argument("--epsilon-shock", type=float, default=0.001)
    parser.add_argument("--epsilon-phi", type=float, default=0.0)
    parser.add_argument("--delta-serd", type=float, default=0.05)
    parser.add_argument("--delta-variant", type=float, default=0.05)
    parser.add_argument(
        "--positive-control-mode",
        choices=("none", "noop_semantic_forced_control"),
        default="none",
        help=(
            "When enabled, build a live real-policy positive control: semantic "
            "branches use the no-shock rollout while matched controls force "
            "bad real-environment actions for several steps. This is an "
            "instrument-sensitivity control, not policy evidence."
        ),
    )
    parser.add_argument(
        "--positive-control-control-steps",
        type=int,
        default=4,
        help="Number of initial control-branch steps to force in positive-control mode.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _install_remote_paths(tg_ssa_root: Path, jaxmarl_src: Path) -> None:
    candidates = [
        tg_ssa_root,
        tg_ssa_root / "external" / "overcookedv2_experiments" / "experiments",
        tg_ssa_root / "external" / "overcookedv2_experiments" / "JaxMARL",
        jaxmarl_src,
    ]
    for candidate in reversed(candidates):
        text = str(candidate)
        if text not in sys.path:
            sys.path.insert(0, text)


def _parse_policy_spec(value: str) -> PolicySpec:
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise ValueError(
            "--policy-run must use label:method:run_dir, "
            f"got {value!r}"
        )
    label, method, run_dir = parts
    return PolicySpec(label=label, method=method, run_dir=Path(run_dir))


def _parse_ints(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("at least one run number is required")
    return parsed


def _scalar(value: Any) -> float:
    try:
        import jax

        value = jax.device_get(value)
    except Exception:
        pass
    try:
        return float(value.item())
    except AttributeError:
        return float(value)


def _to_int(value: Any) -> int:
    return int(round(_scalar(value)))


def _jsonable(value: Any) -> Any:
    try:
        import jax

        value = jax.device_get(value)
    except Exception:
        pass
    try:
        item = value.item()
    except AttributeError:
        item = value
    if isinstance(item, (int, float, str, bool)) or item is None:
        return item
    return str(item)


def _state_signature(state: Any) -> str:
    phi = state_phi_pre_h(state)
    compact = {key: round(float(value), 4) for key, value in sorted(phi.items())}
    return json.dumps(compact, sort_keys=True, separators=(",", ":"))


def _positions(state: Any) -> tuple[tuple[int, int], tuple[int, int]]:
    try:
        import jax

        pos_x = jax.device_get(state.agents.pos.x).reshape(-1)
        pos_y = jax.device_get(state.agents.pos.y).reshape(-1)
    except Exception:
        pos_x = state.agents.pos.x.reshape(-1)
        pos_y = state.agents.pos.y.reshape(-1)
    return (
        (_to_int(pos_x[0]), _to_int(pos_y[0])),
        (_to_int(pos_x[1]), _to_int(pos_y[1])),
    )


def _load_policy_pair(
    spec: PolicySpec,
    run_num: int,
    run_nums: list[int],
    pairing: str,
) -> LoadedPolicyPair:
    from overcooked_v2_experiments.ppo.policy import PPOPolicy
    from overcooked_v2_experiments.ppo.utils.store import load_checkpoint

    if pairing == "self":
        partner_run_num = run_num
    else:
        current_index = run_nums.index(run_num)
        partner_run_num = run_nums[(current_index + 1) % len(run_nums)]

    ego_config, ego_params = load_checkpoint(spec.run_dir, run_num, "final")
    partner_config, partner_params = load_checkpoint(spec.run_dir, partner_run_num, "final")
    if ego_config.get("env", {}) != partner_config.get("env", {}):
        raise RuntimeError(
            f"config mismatch between run_{run_num} and run_{partner_run_num}"
        )

    ego_policy = PPOPolicy(ego_params, ego_config, stochastic=False)
    partner_policy = PPOPolicy(partner_params, partner_config, stochastic=False)
    pair_label = f"{spec.label}_{pairing}_ego{run_num}_partner{partner_run_num}"
    return LoadedPolicyPair(
        label=pair_label,
        ego_run_num=run_num,
        partner_run_num=partner_run_num,
        ego_policy=ego_policy,
        partner_policy=partner_policy,
        config=dict(ego_config),
    )


def _make_env(layout: str, env_kwargs: dict[str, Any], jaxmarl_src: Path):
    OvercookedV2, _Actions, _jax = ensure_jaxmarl_overcooked_v2(str(jaxmarl_src))
    return OvercookedV2(layout=layout, **env_kwargs)


def _initial_done(env) -> dict[str, Any]:
    done = {agent: False for agent in env.agents}
    done["__all__"] = False
    return done


def _init_hstates(pair: LoadedPolicyPair) -> dict[str, Any]:
    return {
        "agent_0": pair.ego_policy.init_hstate(1),
        "agent_1": pair.partner_policy.init_hstate(1),
    }


def _compute_actions(
    env,
    pair: LoadedPolicyPair,
    obs: dict[str, Any],
    done: dict[str, Any],
    hstates: dict[str, Any],
    key: Any,
) -> tuple[dict[str, Any], tuple[int, int], dict[str, Any]]:
    import jax

    policies = {
        "agent_0": pair.ego_policy,
        "agent_1": pair.partner_policy,
    }
    keys = jax.random.split(key, len(env.agents))
    actions: dict[str, Any] = {}
    action_tuple: list[int] = []
    next_hstates: dict[str, Any] = {}
    for index, agent in enumerate(env.agents):
        action, next_hstate = policies[agent].compute_action(
            obs[agent],
            done[agent],
            hstates[agent],
            keys[index],
        )
        action_int = _to_int(action)
        actions[agent] = action
        action_tuple.append(action_int)
        next_hstates[agent] = next_hstate
    return actions, (action_tuple[0], action_tuple[1]), next_hstates


def _override_action_dict(env, joint_action: tuple[int, int]) -> dict[str, int]:
    return {
        agent: int(joint_action[index])
        for index, agent in enumerate(env.agents)
    }


def _sum_reward(rewards: dict[str, Any], infos: dict[str, Any], shaped_reward_weight: float) -> float:
    sparse = sum(float(_scalar(reward)) for reward in rewards.values())
    shaped = 0.0
    for value in infos.get("shaped_reward", {}).values():
        shaped += float(_scalar(value))
    return sparse + shaped_reward_weight * shaped


def _advance_policy_prefix(
    *,
    env,
    jax,
    pair: LoadedPolicyPair,
    seed: int,
    warmup_horizon: int,
) -> tuple[dict[str, Any], Any, dict[str, Any], dict[str, Any]]:
    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key)
    done = _initial_done(env)
    hstates = _init_hstates(pair)
    for _step_index in range(warmup_horizon):
        key, sample_key, step_key = jax.random.split(key, 3)
        actions, _action_tuple, hstates = _compute_actions(
            env,
            pair,
            obs,
            done,
            hstates,
            sample_key,
        )
        obs, state, rewards, done, infos = env.step(step_key, state, actions)
    return obs, state, done, hstates


def _rollout_return(
    *,
    env,
    jax,
    pair: LoadedPolicyPair,
    obs,
    state,
    done,
    hstates,
    seed: int,
    horizon: int,
    shaped_reward_weight: float,
    first_joint_override: tuple[int, int] | None = None,
    joint_override_sequence: list[tuple[int, int]] | None = None,
) -> RolloutResult:
    key = jax.random.PRNGKey(seed)
    total = 0.0
    action_sequence: list[tuple[int, int]] = []
    position_sequence: list[tuple[tuple[int, int], tuple[int, int]]] = []
    state_sequence: list[str] = []

    for step_index in range(horizon):
        key, sample_key, step_key = jax.random.split(key, 3)
        computed_actions, computed_tuple, hstates = _compute_actions(
            env,
            pair,
            obs,
            done,
            hstates,
            sample_key,
        )
        if joint_override_sequence is not None and step_index < len(joint_override_sequence):
            action_tuple = joint_override_sequence[step_index]
            actions = _override_action_dict(env, action_tuple)
        elif step_index == 0 and first_joint_override is not None:
            actions = _override_action_dict(env, first_joint_override)
            action_tuple = first_joint_override
        else:
            actions = computed_actions
            action_tuple = computed_tuple
        obs, state, rewards, done, infos = env.step(step_key, state, actions)
        total += _sum_reward(rewards, infos, shaped_reward_weight)
        action_sequence.append((int(action_tuple[0]), int(action_tuple[1])))
        position_sequence.append(_positions(state))
        state_sequence.append(_state_signature(state))

    return RolloutResult(
        total_return=total,
        action_sequence=action_sequence,
        position_sequence=position_sequence,
        state_sequence=state_sequence,
    )


def _replace_one_action(
    base_joint: tuple[int, int],
    actor_index: int,
    preferred: int,
) -> tuple[int, int]:
    alternatives = (
        ACTION_ID_BY_NAME["stay"],
        ACTION_ID_BY_NAME["interact"],
        ACTION_ID_BY_NAME["up"],
        ACTION_ID_BY_NAME["down"],
        ACTION_ID_BY_NAME["left"],
        ACTION_ID_BY_NAME["right"],
    )
    chosen = preferred if preferred != base_joint[actor_index] else None
    if chosen is None:
        for candidate in alternatives:
            if candidate != base_joint[actor_index]:
                chosen = candidate
                break
    if chosen is None:
        raise RuntimeError("could not construct alternative action")
    branch = list(base_joint)
    branch[actor_index] = chosen
    return (branch[0], branch[1])


def _semantic_joint(
    base_joint: tuple[int, int],
    disruption: str,
    probe_index: int,
) -> tuple[int, int]:
    stay = ACTION_ID_BY_NAME["stay"]
    if disruption == "missed_handoff":
        return _replace_one_action(base_joint, 0, stay)
    if disruption == "route_block":
        return _replace_one_action(base_joint, 1, stay)
    if disruption == "hesitation":
        return _replace_one_action(base_joint, probe_index % 2, stay)
    raise ValueError(f"unknown disruption: {disruption}")


def _control_joint(
    base_joint: tuple[int, int],
    disruption: str,
    family: str,
    probe_index: int,
) -> tuple[int, int]:
    actor_index = 0 if disruption == "missed_handoff" else 1
    if disruption == "hesitation":
        actor_index = probe_index % 2
    preferred_by_family = {
        "random_lag": ACTION_ID_BY_NAME["interact"],
        "state_block": ACTION_ID_BY_NAME["up"],
        "reward_shaping": ACTION_ID_BY_NAME["down"],
        "naive_replanning": ACTION_ID_BY_NAME["left"],
    }
    try:
        preferred = preferred_by_family[family]
    except KeyError as exc:
        raise ValueError(f"unknown control family: {family}") from exc
    return _replace_one_action(base_joint, actor_index, preferred)


def _positive_control_joint(
    base_joint: tuple[int, int],
    family: str,
    probe_index: int,
) -> tuple[int, int]:
    forced_by_family = {
        "random_lag": (ACTION_ID_BY_NAME["stay"], ACTION_ID_BY_NAME["stay"]),
        "state_block": (ACTION_ID_BY_NAME["up"], ACTION_ID_BY_NAME["down"]),
        "reward_shaping": (ACTION_ID_BY_NAME["left"], ACTION_ID_BY_NAME["right"]),
        "naive_replanning": (ACTION_ID_BY_NAME["interact"], ACTION_ID_BY_NAME["stay"]),
    }
    forced = forced_by_family[family]
    if forced != base_joint:
        return forced
    alternates = [
        (ACTION_ID_BY_NAME["interact"], ACTION_ID_BY_NAME["interact"]),
        (ACTION_ID_BY_NAME["stay"], ACTION_ID_BY_NAME["interact"]),
        (ACTION_ID_BY_NAME["up"], ACTION_ID_BY_NAME["up"]),
        (ACTION_ID_BY_NAME["down"], ACTION_ID_BY_NAME["down"]),
    ]
    return alternates[probe_index % len(alternates)]


def _action_divergence(left: tuple[int, int], right: tuple[int, int]) -> float:
    return sum(1 for lhs, rhs in zip(left, right) if lhs != rhs) / 2.0


def _normalized_hamming(left: list[Any], right: list[Any]) -> float:
    denom = max(len(left), len(right), 1)
    mismatches = abs(len(left) - len(right))
    for lhs, rhs in zip(left, right):
        if lhs != rhs:
            mismatches += 1
    return mismatches / denom


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


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summarize_variant(
    rows: list[dict[str, Any]],
    delta: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str, str], list[float]] = {}
    for row in rows:
        key = (
            str(row["policy"]),
            str(row["domain"]),
            str(row["disruption"]),
            str(row["control_family"]),
        )
        grouped.setdefault(key, []).append(float(row["return_independent_serd"]))

    family_rows: list[dict[str, Any]] = []
    for (policy, domain, disruption, family), values in sorted(grouped.items()):
        center, low, high = _ci95(values)
        family_rows.append(
            {
                "policy": policy,
                "domain": domain,
                "disruption": disruption,
                "family": family,
                "metric": "return_independent_action_position_state_serd",
                "mean_serd": center,
                "ci95_low": low,
                "ci95_high": high,
                "n": len(values),
                "classification": _classify(low, high, delta),
            }
        )

    by_disruption: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in family_rows:
        key = (row["policy"], row["domain"], row["disruption"])
        by_disruption.setdefault(key, []).append(row)

    worst_rows: list[dict[str, Any]] = []
    for (policy, domain, disruption), group in sorted(by_disruption.items()):
        limiting = min(group, key=lambda item: float(item["mean_serd"]))
        low = float(limiting["ci95_low"])
        high = float(limiting["ci95_high"])
        worst_rows.append(
            {
                "policy": policy,
                "domain": domain,
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
    return family_rows, worst_rows


def _sequence_distances(
    no_shock: RolloutResult,
    branch: RolloutResult,
) -> dict[str, float]:
    return {
        "action": _normalized_hamming(no_shock.action_sequence, branch.action_sequence),
        "position": _normalized_hamming(no_shock.position_sequence, branch.position_sequence),
        "state": _normalized_hamming(no_shock.state_sequence, branch.state_sequence),
    }


def _build_records_for_pair(
    *,
    env,
    jax,
    pair: LoadedPolicyPair,
    domain: str,
    disruptions: tuple[str, ...],
    probes_per_disruption: int,
    warmup_horizon: int,
    rollout_horizon: int,
    shaped_reward_weight: float,
    seed: int,
    positive_control_mode: str,
    positive_control_control_steps: int,
) -> tuple[list[BranchRecord], list[BranchRecord], list[dict[str, Any]], list[dict[str, Any]]]:
    semantic_records: list[BranchRecord] = []
    control_records: list[BranchRecord] = []
    variant_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []

    for disruption in disruptions:
        for probe_index in range(probes_per_disruption):
            probe_seed = seed + pair.ego_run_num * 10_000 + probe_index
            recovery_seed = seed + 100_000 + pair.partner_run_num * 10_000 + probe_index
            obs, state, done, hstates = _advance_policy_prefix(
                env=env,
                jax=jax,
                pair=pair,
                seed=probe_seed,
                warmup_horizon=warmup_horizon + probe_index,
            )
            phi = state_phi_pre_h(state)
            key = jax.random.PRNGKey(recovery_seed)
            _actions, base_joint, _next_hstates = _compute_actions(
                env,
                pair,
                obs,
                done,
                hstates,
                key,
            )
            no_shock = _rollout_return(
                env=env,
                jax=jax,
                pair=pair,
                obs=obs,
                state=state,
                done=done,
                hstates=hstates,
                seed=recovery_seed,
                horizon=rollout_horizon,
                shaped_reward_weight=shaped_reward_weight,
            )
            if positive_control_mode == "noop_semantic_forced_control":
                semantic_joint = base_joint
                semantic_override_sequence = None
            else:
                semantic_joint = _semantic_joint(base_joint, disruption, probe_index)
                semantic_override_sequence = [semantic_joint]
            semantic = _rollout_return(
                env=env,
                jax=jax,
                pair=pair,
                obs=obs,
                state=state,
                done=done,
                hstates=hstates,
                seed=recovery_seed,
                horizon=rollout_horizon,
                shaped_reward_weight=shaped_reward_weight,
                joint_override_sequence=semantic_override_sequence,
            )
            probe_id = f"{pair.label}_{disruption}_{probe_index:03d}"
            semantic_records.append(
                BranchRecord(
                    probe_id=probe_id,
                    policy=pair.label,
                    domain=domain,
                    disruption=disruption,
                    family="semantic",
                    no_shock_return=no_shock.total_return,
                    branch_return=semantic.total_return,
                    shock_magnitude=_action_divergence(base_joint, semantic_joint),
                    phi_pre_h=dict(phi),
                )
            )
            trace_rows.extend(
                [
                    _trace_row(pair, domain, disruption, probe_id, "no_shock", no_shock),
                    _trace_row(pair, domain, disruption, probe_id, "semantic", semantic),
                ]
            )

            semantic_distances = _sequence_distances(no_shock, semantic)
            for family in CONTROL_FAMILIES:
                if positive_control_mode == "noop_semantic_forced_control":
                    control_joint = _positive_control_joint(base_joint, family, probe_index)
                    control_sequence = [
                        control_joint
                        for _ in range(max(1, positive_control_control_steps))
                    ]
                else:
                    control_joint = _control_joint(base_joint, disruption, family, probe_index)
                    control_sequence = [control_joint]
                control = _rollout_return(
                    env=env,
                    jax=jax,
                    pair=pair,
                    obs=obs,
                    state=state,
                    done=done,
                    hstates=hstates,
                    seed=recovery_seed,
                    horizon=rollout_horizon,
                    shaped_reward_weight=shaped_reward_weight,
                    joint_override_sequence=control_sequence,
                )
                control_records.append(
                    BranchRecord(
                        probe_id=probe_id,
                        policy=pair.label,
                        domain=domain,
                        disruption=disruption,
                        family=family,
                        no_shock_return=no_shock.total_return,
                        branch_return=control.total_return,
                        shock_magnitude=_action_divergence(base_joint, control_joint),
                        phi_pre_h=dict(phi),
                    )
                )
                trace_rows.append(
                    _trace_row(pair, domain, disruption, probe_id, family, control)
                )
                control_distances = _sequence_distances(no_shock, control)
                channel_values = [
                    control_distances[channel] - semantic_distances[channel]
                    for channel in CHANNELS
                ]
                variant_rows.append(
                    {
                        "policy": pair.label,
                        "domain": domain,
                        "ego_run_num": pair.ego_run_num,
                        "partner_run_num": pair.partner_run_num,
                        "probe_id": probe_id,
                        "disruption": disruption,
                        "probe_index": probe_index,
                        "control_family": family,
                        "return_delta": control.total_return - semantic.total_return,
                        "return_independent_serd": mean(channel_values),
                        "semantic_action_distance_to_no_shock": semantic_distances["action"],
                        "control_action_distance_to_no_shock": control_distances["action"],
                        "action_variant_serd": channel_values[0],
                        "semantic_position_distance_to_no_shock": semantic_distances["position"],
                        "control_position_distance_to_no_shock": control_distances["position"],
                        "position_variant_serd": channel_values[1],
                        "semantic_state_distance_to_no_shock": semantic_distances["state"],
                        "control_state_distance_to_no_shock": control_distances["state"],
                        "state_variant_serd": channel_values[2],
                    }
                )
    return semantic_records, control_records, variant_rows, trace_rows


def _trace_row(
    pair: LoadedPolicyPair,
    domain: str,
    disruption: str,
    probe_id: str,
    condition: str,
    rollout: RolloutResult,
) -> dict[str, Any]:
    return {
        "policy": pair.label,
        "domain": domain,
        "ego_run_num": pair.ego_run_num,
        "partner_run_num": pair.partner_run_num,
        "probe_id": probe_id,
        "disruption": disruption,
        "condition": condition,
        "total_return": rollout.total_return,
        "action_sequence": json.dumps(rollout.action_sequence),
        "position_sequence": json.dumps(rollout.position_sequence),
        "state_sequence": json.dumps(rollout.state_sequence),
    }


def _operator_note(output_dir: Path, args: argparse.Namespace, summary: dict[str, Any]) -> None:
    lines = [
        "# Operator Note",
        "",
        "No passwords, tokens, shell history, or private credentials are recorded",
        "in this artifact.",
        "",
        "Command class:",
        "",
        "```text",
        "python -m experiments.serd.run_overcooked_v2_policy_m4",
        "```",
        "",
        f"Decision/status: `{summary['m4_status']}`",
        f"Output directory: `{output_dir}`",
        f"Policy specs: `{args.policy_run}`",
        f"Run nums: `{args.run_nums}`",
        f"Pairing: `{args.pairing}`",
        f"Positive-control mode: `{args.positive_control_mode}`",
        f"Positive-control forced control steps: `{args.positive_control_control_steps}`",
        "",
        "This run uses real restored PPO checkpoints from official OvercookedV2",
        "training artifacts. It does not use the M2 handcoded adapter as policy",
        "evidence.",
        "",
        "If positive-control mode is enabled, this artifact is a live pipeline",
        "instrument-sensitivity control. It must not be interpreted as policy",
        "recovery, superiority, human, or oral evidence.",
        "",
    ]
    (output_dir / "operator_note.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    _install_remote_paths(args.tg_ssa_root, args.jaxmarl_src)

    import jax

    policy_specs = [_parse_policy_spec(item) for item in args.policy_run]
    run_nums = _parse_ints(args.run_nums)
    disruptions = tuple(item.strip() for item in args.disruptions.split(",") if item.strip())

    env_kwargs = {
        "agent_view_size": 2,
        "negative_rewards": True,
        "random_agent_positions": True,
        "sample_recipe_on_delivery": True,
    }
    env = _make_env(args.layout, env_kwargs, args.jaxmarl_src)

    all_semantic: list[BranchRecord] = []
    all_controls: list[BranchRecord] = []
    all_variant_rows: list[dict[str, Any]] = []
    all_trace_rows: list[dict[str, Any]] = []
    policy_matrix_rows: list[dict[str, Any]] = []

    for spec in policy_specs:
        for run_num in run_nums:
            pair = _load_policy_pair(spec, run_num, run_nums, args.pairing)
            semantic, controls, variant_rows, trace_rows = _build_records_for_pair(
                env=env,
                jax=jax,
                pair=pair,
                domain=args.domain,
                disruptions=disruptions,
                probes_per_disruption=args.probes_per_disruption,
                warmup_horizon=args.warmup_horizon,
                rollout_horizon=args.rollout_horizon,
                shaped_reward_weight=args.shaped_reward_weight,
                seed=args.seed,
                positive_control_mode=args.positive_control_mode,
                positive_control_control_steps=args.positive_control_control_steps,
            )
            all_semantic.extend(semantic)
            all_controls.extend(controls)
            all_variant_rows.extend(variant_rows)
            all_trace_rows.extend(trace_rows)
            policy_matrix_rows.append(
                {
                    "policy": pair.label,
                    "method": spec.method,
                    "domain": args.domain,
                    "layout": args.layout,
                    "ego_run_num": pair.ego_run_num,
                    "partner_run_num": pair.partner_run_num,
                    "run_dir": str(spec.run_dir),
                    "pairing": args.pairing,
                    "status": "RESTORED_AND_SCORED",
                }
            )

    run_config = {
        "adapter": "experiments.serd.run_overcooked_v2_policy_m4",
        "tg_ssa_root": str(args.tg_ssa_root),
        "jaxmarl_src": str(args.jaxmarl_src),
        "layout": args.layout,
        "domain": args.domain,
        "policy_run": args.policy_run,
        "run_nums": run_nums,
        "pairing": args.pairing,
        "disruptions": list(disruptions),
        "probes_per_disruption": args.probes_per_disruption,
        "warmup_horizon": args.warmup_horizon,
        "rollout_horizon": args.rollout_horizon,
        "shaped_reward_weight": args.shaped_reward_weight,
        "epsilon_shock": args.epsilon_shock,
        "epsilon_phi": args.epsilon_phi,
        "delta_serd": args.delta_serd,
        "delta_variant": args.delta_variant,
        "positive_control_mode": args.positive_control_mode,
        "positive_control_control_steps": args.positive_control_control_steps,
        "control_families": list(CONTROL_FAMILIES),
        "m3_transition": (
            "REAL_ROLLOUT_POSITIVE_CONTROL_REQUIRED"
            if args.positive_control_mode != "none"
            else "STOP_OR_PIVOT_POLICY_DOMAIN_ROUTE"
        ),
    }
    provenance = {
        "route": (
            "overcookedv2_real_rollout_positive_control"
            if args.positive_control_mode != "none"
            else "overcookedv2_real_ppo_checkpoint_policy_route"
        ),
        "policy_source": "official OvercookedV2 RNN checkpoint directories",
        "not_policy_source": "handcoded M2 adapter",
        "run_command_or_queue_manifest": "python -m experiments.serd.run_overcooked_v2_policy_m4",
        "policy_matrix_rows": policy_matrix_rows,
        "positive_control_mode": args.positive_control_mode,
        "claim_boundary": (
            "live pipeline instrument-sensitivity control only; not policy "
            "recovery, superiority, human, or oral evidence"
            if args.positive_control_mode != "none"
            else "real policy route evidence; interpret under project review state"
        ),
    }
    summary = write_m4_bundle(
        output_dir=args.output_dir,
        semantic_records=all_semantic,
        control_records=all_controls,
        provenance=provenance,
        run_config=run_config,
        acceptance_notes=[
            "Real OvercookedV2 PPO checkpoints were restored and used for policy calls.",
            "Pairing mode is recorded in domain_policy_matrix.csv; cross_next means ego run i is paired with run i+1.",
            "Return-independent action/position/state rows are written as a metric-design companion, not a substitute for standard SERD_worst.",
            (
                "Positive-control mode exercises the real rollout path only as "
                "a live pipeline sensitivity control; it is not policy "
                "recovery evidence."
                if args.positive_control_mode != "none"
                else "Positive-control mode was not enabled."
            ),
        ],
    )

    family_variant, worst_variant = _summarize_variant(all_variant_rows, args.delta_variant)
    _write_csv(args.output_dir / "domain_policy_matrix.csv", policy_matrix_rows, list(policy_matrix_rows[0]))
    _write_csv(args.output_dir / "return_independent_variant.csv", all_variant_rows, list(all_variant_rows[0]))
    _write_csv(args.output_dir / "return_independent_family_serd.csv", family_variant, list(family_variant[0]))
    _write_csv(args.output_dir / "return_independent_worst_serd.csv", worst_variant, list(worst_variant[0]))
    _write_csv(args.output_dir / "branch_trace_summary.csv", all_trace_rows, list(all_trace_rows[0]))

    summary_path = args.output_dir / "summary.json"
    enriched_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    enriched_summary.update(
        {
            "route_decision": (
                "OVERCOOKEDV2_REAL_ROLLOUT_POSITIVE_CONTROL_SCORED"
                if args.positive_control_mode != "none"
                else "OVERCOOKEDV2_REAL_PPO_POLICY_ROUTE_SCORED"
            ),
            "positive_control_mode": args.positive_control_mode,
            "positive_control_expected_survival": args.positive_control_mode != "none",
            "n_policy_pairs": len(policy_matrix_rows),
            "n_return_independent_rows": len(all_variant_rows),
            "n_return_independent_worst_rows": len(worst_variant),
            "return_independent_any_positive_worst": any(
                float(row["mean_serd_worst"]) > args.delta_variant
                for row in worst_variant
            ),
            "return_independent_any_adverse_worst": any(
                float(row["mean_serd_worst"]) < -args.delta_variant
                for row in worst_variant
            ),
            "output_files": {
                "domain_policy_matrix": "domain_policy_matrix.csv",
                "return_independent_variant": "return_independent_variant.csv",
                "return_independent_family": "return_independent_family_serd.csv",
                "return_independent_worst": "return_independent_worst_serd.csv",
                "branch_trace_summary": "branch_trace_summary.csv",
                "operator_note": "operator_note.md",
            },
        }
    )
    summary_path.write_text(
        json.dumps(enriched_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _operator_note(args.output_dir, args, enriched_summary)

    print(json.dumps({"output_dir": str(args.output_dir), **enriched_summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
