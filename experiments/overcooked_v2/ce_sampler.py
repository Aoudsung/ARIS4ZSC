from __future__ import annotations

import argparse
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
from experiments.overcooked_v2.option_termination import OptionRuntime
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
    rng = np.random.default_rng(seed)
    layout = layout_name or getattr(env, "layout_name", "unknown_layout")
    rows: list[OptionReplayRow] = []
    partners = list(partner_pool)

    for partner_idx, partner in enumerate(partners):
        for episode_idx in range(episodes):
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
        obs = step.obs
        if done or terminated:
            break

    partner_dist = _average_partner_dist(partner_dists)
    partner_option = _partner_option_from_trace(partner_dist, partner_options)
    partner_confidence = _partner_confidence(partner_dist, partner_confidences)
    summary["termination_reason"] = termination_reason
    summary["done"] = done

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
    rows = collect_option_replay(
        env,
        partners,
        option_lib,
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
