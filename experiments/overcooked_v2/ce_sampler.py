from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

if __package__ in {None, ""}:  # pragma: no cover - script execution path
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.overcooked_v2.env_adapter import OCV2Adapter
from experiments.overcooked_v2.event_extractor import OCV2Event, extract_event
from experiments.overcooked_v2.layout_parser import parse_layout
from experiments.overcooked_v2.option_termination import OptionRuntime, option_success
from experiments.overcooked_v2.options import OCV2OptionLibrary
from experiments.overcooked_v2.partner_pool import make_training_partners
from experiments.overcooked_v2.state_utils import (
    get_agent_pos,
    get_dynamic_objects_grid,
    get_inventory,
)


@dataclass
class OptionReplayRow:
    layout: str
    episode_id: int
    t_option: int
    ego_option: int
    partner_option: int | None
    partner_option_dist: np.ndarray | None
    partner_option_confidence: float
    state_key: str
    duration: int
    reward_sum: float
    shaped_reward_sum: float
    realized_cost: float
    local_return_h: float
    reward_to_go: float
    event_summary: dict[str, Any]
    partner_name: str
    partner_id: int


def collect_option_replay(
    env: Any,
    partner_pool: Iterable[Any],
    option_lib: OCV2OptionLibrary,
    *,
    layout_name: str | None = None,
    episodes: int = 100,
    max_options_per_episode: int | None = None,
    seed: int = 0,
    gamma: float = 0.99,
    horizon_options: int = 5,
    cost_per_step: float = 1.0,
    cost_coef: float = 1.0,
    shaped_reward_coef: float = 0.0,
) -> list[OptionReplayRow]:
    import time as _time
    rng = np.random.default_rng(seed)
    layout = layout_name or getattr(env, "layout_name", "unknown_layout")
    rows: list[OptionReplayRow] = []
    partners = list(partner_pool)
    _t0 = _time.monotonic()

    for partner_idx, partner in enumerate(partners):
        print(f"  partner {partner_idx+1}/{len(partners)}: {getattr(partner, 'name', '?')}", flush=True)
        for episode_idx in range(episodes):
            if episode_idx > 0 and episode_idx % 20 == 0:
                _elapsed = _time.monotonic() - _t0
                print(f"    ep {episode_idx}/{episodes}, {len(rows)} rows, {_elapsed:.1f}s", flush=True)
            episode_id = partner_idx * episodes + episode_idx
            reset_seed = int(rng.integers(0, 2**31 - 1))
            obs, _ = env.reset(reset_seed)
            if hasattr(partner, "reset"):
                partner.reset(reset_seed)

            t_option = 0
            done = False
            option_limit = max_options_per_episode or getattr(env, "max_steps", 400)

            while not done and t_option < option_limit:
                option_id = _sample_valid_option(option_lib, env.state, 0, rng)
                row, done = _rollout_option(
                    env,
                    obs,
                    partner,
                    option_lib,
                    layout,
                    episode_id,
                    t_option,
                    option_id,
                    rng,
                    cost_per_step,
                )
                rows.append(row)
                obs = env.obs or obs
                t_option += 1

    compute_local_returns(
        rows,
        gamma=gamma,
        horizon=horizon_options,
        cost_coef=cost_coef,
        shaped_reward_coef=shaped_reward_coef,
    )
    compute_reward_to_go(rows, gamma=gamma)
    return rows


def compute_local_returns(
    option_rows: list[OptionReplayRow],
    gamma: float,
    horizon: int,
    cost_coef: float = 1.0,
    shaped_reward_coef: float = 0.0,
) -> list[OptionReplayRow]:
    for idx, row in enumerate(option_rows):
        ret = 0.0
        discount = 1.0
        for jdx in range(idx, min(idx + horizon, len(option_rows))):
            next_row = option_rows[jdx]
            if next_row.episode_id != row.episode_id:
                break
            ret += discount * (
                next_row.reward_sum
                + float(shaped_reward_coef) * next_row.shaped_reward_sum
                - float(cost_coef) * next_row.realized_cost
            )
            discount *= gamma ** max(1, next_row.duration)
        row.local_return_h = float(ret)
    return option_rows


def compute_reward_to_go(
    option_rows: list[OptionReplayRow],
    gamma: float,
) -> list[OptionReplayRow]:
    running_by_episode: dict[int, float] = {}
    for row in reversed(option_rows):
        running = running_by_episode.get(row.episode_id, 0.0)
        row.reward_to_go = float(row.reward_sum + (gamma ** max(1, row.duration)) * running)
        running_by_episode[row.episode_id] = row.reward_to_go
    return option_rows


@dataclass
class _EnvSlot:
    episode_id: int
    t_option: int
    option_id: int | None
    option_runtime: Any
    state_key: str | None
    reward_sum: float
    shaped_reward_sum: float
    duration: int
    summary: dict[str, Any]
    partner_dists: list[np.ndarray]
    partner_options: list[int]
    partner_confidences: list[float]
    partner: Any
    done: bool
    needs_new_option: bool
    episodes_remaining: int
    rng_seed_base: int


def collect_option_replay_batched(
    env: Any,
    partner_pool: Iterable[Any],
    option_lib: "OCV2OptionLibrary",
    *,
    layout_name: str | None = None,
    episodes: int = 100,
    max_options_per_episode: int | None = None,
    seed: int = 0,
    gamma: float = 0.99,
    horizon_options: int = 5,
    cost_per_step: float = 1.0,
    cost_coef: float = 1.0,
    shaped_reward_coef: float = 0.0,
    batch_size: int = 64,
) -> list[OptionReplayRow]:
    from experiments.overcooked_v2.batched_rollout import BatchedEnvPool

    rng = np.random.default_rng(seed)
    layout = layout_name or getattr(env, "layout_name", "unknown_layout")
    rows: list[OptionReplayRow] = []
    partners = list(partner_pool)
    raw_env = getattr(env, "env", env)
    pool = BatchedEnvPool(raw_env, batch_size)
    option_limit = max_options_per_episode or getattr(env, "max_steps", 400)

    import time as _time
    _t0 = _time.monotonic()

    for partner_idx, partner_template in enumerate(partners):
        print(f"  partner {partner_idx+1}/{len(partners)}: {getattr(partner_template, 'name', '?')}", flush=True)
        episode_base = partner_idx * episodes
        slots = [
            _EnvSlot(
                episode_id=episode_base + i,
                t_option=0,
                option_id=None,
                option_runtime=None,
                state_key=None,
                reward_sum=0.0,
                shaped_reward_sum=0.0,
                duration=0,
                summary=_empty_event_summary(),
                partner_dists=[],
                partner_options=[],
                partner_confidences=[],
                partner=copy.deepcopy(partner_template),
                done=i >= episodes,
                needs_new_option=True,
                episodes_remaining=max(0, episodes - i),
                rng_seed_base=int(rng.integers(0, 2**31 - 1)),
            )
            for i in range(batch_size)
        ]

        init_seeds = np.array([
            int(rng.integers(0, 2**31 - 1)) for _ in range(batch_size)
        ], dtype=np.int64)
        pool.reset(init_seeds)
        for slot in slots:
            if not slot.done and hasattr(slot.partner, "reset"):
                slot.partner.reset(int(init_seeds[slots.index(slot)]))

        active_count = sum(1 for s in slots if not s.done)
        ego_actions = np.zeros(batch_size, dtype=np.int32)
        partner_actions = np.zeros(batch_size, dtype=np.int32)
        prev_states: list[Any] = [None] * batch_size

        while active_count > 0:
            pre_snap = pool.snapshot()
            obs_np = pool.snapshot_obs()

            for i, slot in enumerate(slots):
                if slot.done:
                    ego_actions[i] = 5  # Actions.stay
                    partner_actions[i] = 5
                    continue

                state_i = pre_snap[i]

                if slot.needs_new_option:
                    slot.option_id = _sample_valid_option(option_lib, state_i, 0, rng)
                    slot.option_runtime = OptionRuntime(
                        option_id=int(slot.option_id),
                        start_pos=get_agent_pos(state_i, 0),
                    )
                    slot.state_key = _state_key(state_i)
                    slot.reward_sum = 0.0
                    slot.shaped_reward_sum = 0.0
                    slot.duration = 0
                    slot.summary = _empty_event_summary()
                    slot.partner_dists = []
                    slot.partner_options = []
                    slot.partner_confidences = []
                    slot.needs_new_option = False

                ego_actions[i] = option_lib.primitive_action(state_i, 0, slot.option_id)
                partner_obs = obs_np.get("agent_1")
                partner_obs_i = partner_obs[i] if partner_obs is not None else None
                pa = slot.partner.act(partner_obs_i, state_i, rng)
                partner_actions[i] = int(pa.primitive_action)
                if pa.option_dist is not None:
                    slot.partner_dists.append(np.asarray(pa.option_dist, dtype=np.float32))
                if pa.option_id is not None:
                    slot.partner_options.append(int(pa.option_id))
                slot.partner_confidences.append(float(pa.option_confidence))
                prev_states[i] = state_i

            _, _, rewards, dones, info = pool.step(ego_actions, partner_actions)

            post_snap = pool.snapshot()
            rewards_np = {k: np.asarray(v) for k, v in rewards.items()}
            dones_np = {k: np.asarray(v) for k, v in dones.items()}

            reset_indices = []
            reset_seeds = []

            for i, slot in enumerate(slots):
                if slot.done:
                    continue

                state_i = post_snap[i]
                opt = option_lib.options[slot.option_id]
                info_i = pool.get_info_i(i, info)
                reward_i = float(rewards_np["agent_0"][i])
                done_i = bool(dones_np["__all__"][i])
                shaped_i = _shaped_reward_for_agent(info_i, "agent_0")

                event = extract_event(
                    prev_states[i],
                    int(ego_actions[i]),
                    int(partner_actions[i]),
                    state_i,
                    info_i,
                    partner_option=slot.partner_options[-1] if slot.partner_options else None,
                    partner_option_dist=slot.partner_dists[-1] if slot.partner_dists else None,
                )
                _accumulate_event_summary(slot.summary, event)
                slot.reward_sum += reward_i
                slot.shaped_reward_sum += shaped_i
                slot.duration += 1

                terminated, reason = option_lib.option_terminated(
                    opt, prev_states[i], state_i, event,
                    agent_id=0, elapsed=slot.duration, runtime=slot.option_runtime,
                )
                if done_i and not terminated:
                    reason = "env_max_steps"
                    terminated = True

                if terminated:
                    slot.summary["termination_reason"] = reason
                    slot.summary["done"] = done_i
                    slot.summary["option_kind"] = opt.kind
                    slot.summary["option_success"] = option_success(opt.kind, reason)
                    p_dist = _average_partner_dist(slot.partner_dists)
                    p_opt = _partner_option_from_trace(p_dist, slot.partner_options)
                    p_conf = _partner_confidence(p_dist, slot.partner_confidences)
                    rows.append(OptionReplayRow(
                        layout=layout,
                        episode_id=slot.episode_id,
                        t_option=slot.t_option,
                        ego_option=slot.option_id,
                        partner_option=p_opt,
                        partner_option_dist=p_dist,
                        partner_option_confidence=p_conf,
                        state_key=slot.state_key or "",
                        duration=slot.duration,
                        reward_sum=slot.reward_sum,
                        shaped_reward_sum=slot.shaped_reward_sum,
                        realized_cost=float(slot.duration * cost_per_step),
                        local_return_h=0.0,
                        reward_to_go=0.0,
                        event_summary=slot.summary,
                        partner_name=str(getattr(slot.partner, "name", "partner")),
                        partner_id=int(getattr(slot.partner, "partner_id", -1)),
                    ))
                    slot.t_option += 1
                    slot.needs_new_option = True

                if done_i or slot.t_option >= option_limit:
                    slot.episodes_remaining -= 1
                    if slot.episodes_remaining > 0:
                        slot.episode_id = episode_base + (episodes - slot.episodes_remaining)
                        slot.t_option = 0
                        slot.needs_new_option = True
                        new_seed = int(rng.integers(0, 2**31 - 1))
                        reset_indices.append(i)
                        reset_seeds.append(new_seed)
                        if hasattr(slot.partner, "reset"):
                            slot.partner.reset(new_seed)
                    else:
                        slot.done = True
                        active_count -= 1

            if reset_indices:
                pool.reset_indices(
                    np.array(reset_indices, dtype=np.int32),
                    np.array(reset_seeds, dtype=np.int64),
                )

        _elapsed = _time.monotonic() - _t0
        _ep_done = sum(episodes - s.episodes_remaining for s in slots)
        print(f"    → {_ep_done}/{episodes} episodes done, {len(rows)} rows, {_elapsed:.1f}s elapsed", flush=True)

    compute_local_returns(
        rows, gamma=gamma, horizon=horizon_options,
        cost_coef=cost_coef, shaped_reward_coef=shaped_reward_coef,
    )
    compute_reward_to_go(rows, gamma=gamma)
    return rows


def option_kind_stats(
    rows: list[OptionReplayRow],
    options: list[Any],
) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for row in rows:
        kind = _option_kind_for_row(row, options)
        reason = str((row.event_summary or {}).get("termination_reason", "unknown"))
        item = stats.setdefault(
            kind,
            {
                "attempt_count": 0,
                "success_count": 0,
                "timeout_count": 0,
                "success_rate": 0.0,
                "termination_reason_histogram": {},
            },
        )
        item["attempt_count"] = int(item["attempt_count"]) + 1
        item["success_count"] = int(item["success_count"]) + int(option_success(kind, reason))
        item["timeout_count"] = int(item["timeout_count"]) + int(
            reason in {"max_steps", "env_max_steps"}
        )
        histogram = item["termination_reason_histogram"]
        histogram[reason] = int(histogram.get(reason, 0)) + 1

    for item in stats.values():
        attempts = max(1, int(item["attempt_count"]))
        item["success_rate"] = float(int(item["success_count"]) / attempts)
    return stats


def replay_coverage(
    rows: list[OptionReplayRow],
    options: list[Any],
) -> dict[str, int]:
    coverage = {
        "picked_ingredient": 0,
        "ingredient_delivered_to_pot": 0,
        "pot_became_ready": 0,
        "plate_picked": 0,
        "plated_soup": 0,
        "served_soup": 0,
        "drop_item_to_counter": 0,
        "cleared_interaction_cell": 0,
    }
    for row in rows:
        summary = row.event_summary or {}
        reason = str(summary.get("termination_reason", "unknown"))
        coverage["picked_ingredient"] += int(reason == "picked_ingredient")
        coverage["ingredient_delivered_to_pot"] += int(
            reason == "ingredient_delivered_to_pot"
        )
        coverage["pot_became_ready"] += int(summary.get("pot_became_ready", 0))
        coverage["plate_picked"] += int(summary.get("plate_picked", 0))
        coverage["plated_soup"] += max(
            int(reason == "plated_soup"),
            int(summary.get("soup_picked", 0)),
        )
        coverage["served_soup"] += max(
            int(reason == "served_soup"),
            int(summary.get("delivery_event", 0)),
        )
        coverage["drop_item_to_counter"] += int(reason == "dropped_item_to_counter")
        coverage["cleared_interaction_cell"] += int(reason == "cleared_interaction_cell")
    return {key: int(value) for key, value in coverage.items()}


def replay_coverage_gate(
    coverage: dict[str, int],
    *,
    require_full_task_coverage: bool,
) -> dict[str, Any]:
    required = ("plated_soup", "served_soup")
    missing = [key for key in required if int(coverage.get(key, 0)) <= 0]
    result = {
        "require_full_task_coverage": bool(require_full_task_coverage),
        "missing_required_coverage": missing,
        "status": "passed" if not missing else "missing_required_coverage",
    }
    if require_full_task_coverage and missing:
        raise RuntimeError(f"CE replay lacks required task coverage: {missing}")
    return result


def _option_kind_for_row(row: OptionReplayRow, options: list[Any]) -> str:
    if 0 <= int(row.ego_option) < len(options):
        return str(options[int(row.ego_option)].kind)
    return str((row.event_summary or {}).get("option_kind", "unknown"))


def estimate_empirical_ce(
    replay: list[OptionReplayRow],
    num_options: int,
    min_weight: float = 20.0,
) -> np.ndarray:
    if not replay:
        raise ValueError("estimate_empirical_ce() requires at least one replay row.")

    ce = np.zeros((num_options, num_options), dtype=np.float32)
    returns = np.asarray([row.local_return_h for row in replay], dtype=np.float32)
    global_mean = float(returns.mean())

    ego_w = np.zeros((len(replay), num_options), dtype=np.float32)
    partner_w = np.zeros((len(replay), num_options), dtype=np.float32)

    for idx, row in enumerate(replay):
        ego_w[idx, row.ego_option] = 1.0
        if row.partner_option_dist is not None:
            dist = np.asarray(row.partner_option_dist, dtype=np.float32)
            partner_w[idx, : min(num_options, dist.size)] = dist[:num_options]
        elif row.partner_option is not None:
            partner_w[idx, row.partner_option] = float(row.partner_option_confidence)

    ego_mean = weighted_means(returns, ego_w, default=global_mean)
    partner_mean = weighted_means(returns, partner_w, default=global_mean)

    for ego_option in range(num_options):
        for partner_option in range(num_options):
            weight = ego_w[:, ego_option] * partner_w[:, partner_option]
            weight_sum = float(weight.sum())
            if weight_sum < min_weight:
                continue
            joint = float((weight * returns).sum() / weight_sum)
            ce[ego_option, partner_option] = abs(
                joint - ego_mean[ego_option] - partner_mean[partner_option] + global_mean
            )

    return ce


def weighted_means(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    default: float,
) -> np.ndarray:
    means = np.full((weights.shape[1],), float(default), dtype=np.float32)
    for idx in range(weights.shape[1]):
        weight = weights[:, idx]
        total = float(weight.sum())
        if total > 0.0:
            means[idx] = float((weight * values).sum() / total)
    return means


def refine_empirical_ce(
    ce_matrix: np.ndarray,
    replay: list[OptionReplayRow],
    num_options: int,
    *,
    top_k: int = 32,
    min_weight: float = 20.0,
    bootstrap_iterations: int = 8,
    seed: int = 17,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not replay:
        raise ValueError("refine_empirical_ce() requires replay rows.")

    rng = np.random.default_rng(seed)
    refined = np.asarray(ce_matrix, dtype=np.float32).copy()
    top_pairs = _top_pairs(refined, top_k)
    if not top_pairs:
        return refined, {"refine_mode": "empirical_refine", "top_k": 0}

    estimates = {(i, j): [] for _, i, j in top_pairs}
    relaxed_min_weight = max(1.0, float(min_weight) * 0.5)
    for _ in range(max(1, bootstrap_iterations)):
        sample_idx = rng.integers(0, len(replay), size=len(replay))
        sample = [replay[int(idx)] for idx in sample_idx]
        sample_ce = estimate_empirical_ce(sample, num_options, relaxed_min_weight)
        for _, i, j in top_pairs:
            estimates[(i, j)].append(float(sample_ce[i, j]))

    for _, i, j in top_pairs:
        values = [value for value in estimates[(i, j)] if value > 0.0]
        if values:
            refined[i, j] = float(np.mean(values))

    metadata = {
        "refine_mode": "empirical_refine",
        "top_k": len(top_pairs),
        "bootstrap_iterations": int(bootstrap_iterations),
        "min_weight": float(min_weight),
        "relaxed_min_weight": relaxed_min_weight,
        "forced_intervention": False,
    }
    return refined, metadata


def refine_interventional_ce(
    ce_matrix: np.ndarray,
    replay: list[OptionReplayRow],
    num_options: int,
    *,
    top_k: int = 16,
    samples_per_pair: int = 4,
    cost_coef: float = 1.0,
    shaped_reward_coef: float = 0.0,
    seed: int = 17,
    intervention_runner: Any | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if intervention_runner is None:
        raise ValueError(
            "refine_interventional_ce requires an intervention_runner; scalar "
            "empirical replay bootstrap is available via refine_empirical_ce."
        )
    if samples_per_pair <= 0:
        raise ValueError("samples_per_pair must be positive.")

    rng = np.random.default_rng(seed)
    refined = np.asarray(ce_matrix, dtype=np.float32).copy()
    top_pairs = _top_pairs(refined, top_k)
    if not top_pairs:
        return refined, {
            "refine_mode": "interventional_topk",
            "top_k": 0,
            "samples_per_pair": int(samples_per_pair),
            "cost_coef": float(cost_coef),
            "forced_intervention": True,
        }

    replay_returns = np.asarray(
        [
            row.reward_sum
            + float(shaped_reward_coef) * row.shaped_reward_sum
            - float(cost_coef) * row.realized_cost
            for row in replay
        ],
        dtype=np.float32,
    )
    baseline = float(replay_returns.mean()) if replay_returns.size else 0.0
    changed_pairs: list[list[int]] = []
    sample_counts: dict[str, int] = {}
    for _, ego_option, partner_option in top_pairs:
        values = []
        for sample_idx in range(int(samples_per_pair)):
            values.append(
                float(
                    intervention_runner(
                        int(ego_option),
                        int(partner_option),
                        int(rng.integers(0, 2**31 - 1)),
                        sample_idx,
                    )
                )
            )
        refined[int(ego_option), int(partner_option)] = abs(float(np.mean(values)) - baseline)
        changed_pairs.append([int(ego_option), int(partner_option)])
        sample_counts[f"{ego_option},{partner_option}"] = len(values)

    metadata = {
        "refine_mode": "interventional_topk",
        "top_k": len(top_pairs),
        "intervention_top_k": len(top_pairs),
        "samples_per_pair": int(samples_per_pair),
        "sample_counts": sample_counts,
        "changed_pairs": changed_pairs,
        "cost_coef": float(cost_coef),
        "forced_intervention": True,
    }
    return refined, metadata


def save_replay_npz(
    path: str | Path,
    rows: list[OptionReplayRow],
    metadata: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = np.asarray([json.dumps(_row_to_json_dict(row)) for row in rows])
    np.savez_compressed(
        path,
        rows=payload,
        metadata=np.asarray(json.dumps(metadata or {})),
    )


def load_replay_npz(path: str | Path) -> tuple[list[OptionReplayRow], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        rows = [_row_from_json_dict(json.loads(str(item))) for item in data["rows"]]
        metadata = json.loads(str(data["metadata"])) if "metadata" in data else {}
    return rows, metadata


def _rollout_option(
    env: Any,
    obs: dict[str, np.ndarray],
    partner: Any,
    option_lib: OCV2OptionLibrary,
    layout: str,
    episode_id: int,
    t_option: int,
    option_id: int,
    rng: np.random.Generator,
    cost_per_step: float,
) -> tuple[OptionReplayRow, bool]:
    opt = option_lib.options[option_id]
    start_state = env.state
    runtime = OptionRuntime(option_id=int(option_id), start_pos=get_agent_pos(start_state, 0))
    state_key = _state_key(start_state)
    reward_sum = 0.0
    shaped_reward_sum = 0.0
    duration = 0
    done = False
    termination_reason = "running"
    partner_dists: list[np.ndarray] = []
    partner_options: list[int] = []
    partner_confidences: list[float] = []
    summary = _empty_event_summary()

    while duration < opt.max_steps:
        ego_action = option_lib.primitive_action(env.state, 0, option_id)
        partner_obs = obs.get("agent_1") if isinstance(obs, dict) else None
        partner_action = partner.act(partner_obs, env.state, rng)
        if partner_action.option_dist is not None:
            partner_dists.append(np.asarray(partner_action.option_dist, dtype=np.float32))
        if partner_action.option_id is not None:
            partner_options.append(int(partner_action.option_id))
        partner_confidences.append(float(partner_action.option_confidence))

        prev_state = env.state
        step = env.step(ego_action, partner_action.primitive_action)
        event = extract_event(
            prev_state,
            ego_action,
            partner_action.primitive_action,
            step.state,
            step.info,
            partner_option=partner_action.option_id,
            partner_option_dist=partner_action.option_dist,
        )
        _accumulate_event_summary(summary, event)

        reward_sum += float(step.rewards.get("agent_0", 0.0))
        shaped_reward_sum += _shaped_reward_for_agent(step.info, "agent_0")
        duration += 1
        done = bool(step.dones.get("__all__", False))
        terminated, termination_reason = option_lib.option_terminated(
            opt,
            prev_state,
            step.state,
            event,
            agent_id=0,
            elapsed=duration,
            runtime=runtime,
        )
        if done and not terminated:
            termination_reason = "env_max_steps"
        obs = step.obs
        if done or terminated:
            break

    partner_dist = _average_partner_dist(partner_dists)
    partner_option = _partner_option_from_trace(partner_dist, partner_options)
    partner_confidence = _partner_confidence(partner_dist, partner_confidences)
    summary["termination_reason"] = termination_reason
    summary["done"] = done
    summary["option_kind"] = opt.kind
    summary["option_success"] = option_success(opt.kind, termination_reason)

    row = OptionReplayRow(
        layout=layout,
        episode_id=episode_id,
        t_option=t_option,
        ego_option=option_id,
        partner_option=partner_option,
        partner_option_dist=partner_dist,
        partner_option_confidence=partner_confidence,
        state_key=state_key,
        duration=duration,
        reward_sum=float(reward_sum),
        shaped_reward_sum=float(shaped_reward_sum),
        realized_cost=float(duration * cost_per_step),
        local_return_h=0.0,
        reward_to_go=0.0,
        event_summary=summary,
        partner_name=str(getattr(partner, "name", "partner")),
        partner_id=int(getattr(partner, "partner_id", -1)),
    )
    return row, done


def _sample_valid_option(
    option_lib: OCV2OptionLibrary,
    state: Any,
    agent_id: int,
    rng: np.random.Generator,
) -> int:
    valid = option_lib.valid_options(state, agent_id)
    valid_ids = np.flatnonzero(valid)
    if valid_ids.size:
        return int(rng.choice(valid_ids))
    return _noop_option_id(option_lib)


def _noop_option_id(option_lib: OCV2OptionLibrary) -> int:
    for opt in option_lib.options:
        if opt.kind == "noop":
            return int(opt.id)
    return 0


def _state_key(state: Any) -> str:
    dynamic = np.asarray(get_dynamic_objects_grid(state), dtype=np.int64)
    payload = {
        "agent_pos": [get_agent_pos(state, 0), get_agent_pos(state, 1)],
        "inventory": [get_inventory(state, 0), get_inventory(state, 1)],
        "dynamic_sha1": hashlib.sha1(dynamic.tobytes()).hexdigest(),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _empty_event_summary() -> dict[str, Any]:
    return {
        "delivery_event": 0,
        "wrong_delivery_event": 0,
        "pot_changed": 0,
        "object_pickup_or_drop": 0,
        "recipe_indicator_event": 0,
        "button_pressed": 0,
        "pot_became_full": 0,
        "pot_became_cooked": 0,
        "pot_became_ready": 0,
        "plate_picked": 0,
        "soup_picked": 0,
        "correct_delivery": 0,
        "collision_or_block": 0,
        "ego_waited": 0,
        "partner_waited": 0,
        "changed_cells": 0,
        "pot_changed_cells": 0,
    }


def _accumulate_event_summary(summary: dict[str, Any], event: OCV2Event) -> None:
    for key in (
        "delivery_event",
        "wrong_delivery_event",
        "pot_changed",
        "object_pickup_or_drop",
        "recipe_indicator_event",
        "button_pressed",
        "pot_became_full",
        "pot_became_cooked",
        "pot_became_ready",
        "plate_picked",
        "soup_picked",
        "correct_delivery",
        "collision_or_block",
        "ego_waited",
        "partner_waited",
    ):
        summary[key] += int(bool(getattr(event, key)))
    summary["changed_cells"] += len(event.changed_cells)
    summary["pot_changed_cells"] += len(getattr(event, "pot_changed_cells", ()))


def _average_partner_dist(dists: list[np.ndarray]) -> np.ndarray | None:
    if not dists:
        return None
    max_size = max(dist.size for dist in dists)
    padded = np.zeros((len(dists), max_size), dtype=np.float32)
    for idx, dist in enumerate(dists):
        padded[idx, : dist.size] = dist
    mean = padded.mean(axis=0)
    total = float(mean.sum())
    if total > 0.0:
        mean /= total
    return mean.astype(np.float32)


def _partner_option_from_trace(
    partner_dist: np.ndarray | None,
    partner_options: list[int],
) -> int | None:
    if partner_dist is not None and partner_dist.size and float(partner_dist.sum()) > 0.0:
        return int(np.argmax(partner_dist))
    if not partner_options:
        return None
    values, counts = np.unique(np.asarray(partner_options, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _partner_confidence(
    partner_dist: np.ndarray | None,
    confidences: list[float],
) -> float:
    if partner_dist is not None and partner_dist.size:
        return float(np.max(partner_dist))
    if confidences:
        return float(np.mean(confidences))
    return 0.0


def _shaped_reward_for_agent(info: dict[str, Any], agent_key: str) -> float:
    shaped = info.get("shaped_reward", 0.0)
    if isinstance(shaped, dict):
        if agent_key in shaped:
            return _as_float(shaped[agent_key])
        return float(sum(_as_float(value) for value in shaped.values()))
    return _as_float(shaped)


def _as_float(value: Any) -> float:
    return float(np.asarray(value).item())


def _row_to_json_dict(row: OptionReplayRow) -> dict[str, Any]:
    data = asdict(row)
    if row.partner_option_dist is not None:
        data["partner_option_dist"] = np.asarray(row.partner_option_dist).tolist()
    return data


def _row_from_json_dict(data: dict[str, Any]) -> OptionReplayRow:
    dist = data.get("partner_option_dist")
    data["partner_option_dist"] = None if dist is None else np.asarray(dist, dtype=np.float32)
    data.setdefault("partner_id", -1)
    return OptionReplayRow(**data)


def _top_pairs(ce_matrix: np.ndarray, top_k: int) -> list[tuple[float, int, int]]:
    ce = np.asarray(ce_matrix, dtype=float)
    pairs = [
        (float(ce[i, j]), int(i), int(j))
        for i in range(ce.shape[0])
        for j in range(ce.shape[1])
        if float(ce[i, j]) > 0.0
    ]
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    return pairs[: max(0, int(top_k))]


def _build_default_stack(args: argparse.Namespace) -> tuple[OCV2Adapter, OCV2OptionLibrary]:
    env = OCV2Adapter(
        args.layout,
        max_steps=args.max_steps,
        observation_type=args.observation_type,
        force_path_planning=False,
    )
    layout_graph = parse_layout(env, args.layout)
    option_lib = OCV2OptionLibrary(layout_graph, max_option_steps=args.max_option_steps)
    return env, option_lib


def _cmd_collect(args: argparse.Namespace) -> None:
    env, option_lib = _build_default_stack(args)
    if args.partners not in {"scripted_debug", "train", "all"}:
        raise ValueError(
            "Phase 4 CE collection supports scripted_debug/train/all partner selectors."
        )
    partners = make_training_partners(option_lib)
    collect_fn = collect_option_replay_batched if args.batch_size > 1 else collect_option_replay
    collect_kwargs = dict(
        layout_name=args.layout,
        episodes=args.episodes,
        max_options_per_episode=args.max_options_per_episode,
        seed=args.seed,
        gamma=args.gamma,
        horizon_options=args.horizon_options,
        cost_per_step=args.cost_per_step,
        cost_coef=args.cost_coef,
        shaped_reward_coef=args.shaped_reward_coef,
    )
    if args.batch_size > 1:
        collect_kwargs["batch_size"] = args.batch_size
    rows = collect_fn(env, partners, option_lib, **collect_kwargs)
    coverage = replay_coverage(rows, option_lib.options)
    coverage_gate = replay_coverage_gate(
        coverage,
        require_full_task_coverage=bool(args.require_full_task_coverage),
    )
    save_replay_npz(
        args.output,
        rows,
        metadata={
            "layout": args.layout,
            "episodes_per_partner": args.episodes,
            "num_rows": len(rows),
            "partners": [partner.name for partner in partners],
            "cost_coef": float(args.cost_coef),
            "cost_per_step": float(args.cost_per_step),
            "shaped_reward_coef": float(args.shaped_reward_coef),
            "coverage": coverage,
            "coverage_gate": coverage_gate,
            "option_kind_stats": option_kind_stats(rows, option_lib.options),
        },
    )


def _cmd_estimate(args: argparse.Namespace) -> None:
    rows, metadata = load_replay_npz(args.replay)
    ce = estimate_empirical_ce(rows, args.num_options, min_weight=args.min_weight)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, ce)
    _write_metadata_sidecar(
        args.output,
        {**metadata, "min_weight": args.min_weight, "num_options": args.num_options},
    )


def _cmd_refine(args: argparse.Namespace) -> None:
    rows, metadata = load_replay_npz(args.replay)
    ce = np.load(args.ce)
    refined, refine_metadata = refine_empirical_ce(
        ce,
        rows,
        args.num_options,
        top_k=args.top_k,
        min_weight=args.min_weight,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, refined)
    _write_metadata_sidecar(args.output, {**metadata, **refine_metadata})


def _write_metadata_sidecar(path: str | Path, metadata: dict[str, Any]) -> None:
    sidecar = Path(f"{path}.metadata.json")
    sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OvercookedV2 option-level CE sampler")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect")
    collect.add_argument("--layout", required=True)
    collect.add_argument("--episodes", type=int, default=500)
    collect.add_argument("--horizon_options", type=int, default=5)
    collect.add_argument("--output", required=True)
    collect.add_argument("--partners", default="scripted_debug")
    collect.add_argument("--seed", type=int, default=0)
    collect.add_argument("--gamma", type=float, default=0.99)
    collect.add_argument("--cost_per_step", type=float, default=1.0)
    collect.add_argument("--cost_coef", type=float, default=1.0)
    collect.add_argument("--shaped_reward_coef", type=float, default=0.0)
    collect.add_argument("--max_steps", type=int, default=200)
    collect.add_argument("--max_option_steps", type=int, default=12)
    collect.add_argument("--max_options_per_episode", type=int, default=None)
    collect.add_argument("--observation_type", default="default")
    collect.add_argument("--require_full_task_coverage", action="store_true")
    collect.add_argument("--batch_size", type=int, default=1)
    collect.set_defaults(func=_cmd_collect)

    estimate = subparsers.add_parser("estimate")
    estimate.add_argument("--replay", required=True)
    estimate.add_argument("--num_options", type=int, required=True)
    estimate.add_argument("--min_weight", type=float, default=20.0)
    estimate.add_argument("--output", required=True)
    estimate.set_defaults(func=_cmd_estimate)

    refine = subparsers.add_parser("refine")
    refine.add_argument("--ce", required=True)
    refine.add_argument("--replay", required=True)
    refine.add_argument("--num_options", type=int, required=True)
    refine.add_argument("--top_k", type=int, default=32)
    refine.add_argument("--min_weight", type=float, default=20.0)
    refine.add_argument("--bootstrap_iterations", type=int, default=8)
    refine.add_argument("--seed", type=int, default=17)
    refine.add_argument("--output", required=True)
    refine.set_defaults(func=_cmd_refine)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
