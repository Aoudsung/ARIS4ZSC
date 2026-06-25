from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:  # pragma: no cover - script execution path
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.overcooked_v2.ce_sampler import (
    OptionReplayRow,
    collect_option_replay,
    estimate_empirical_ce,
    load_replay_npz,
)
from experiments.overcooked_v2.env_adapter import OCV2Adapter
from experiments.overcooked_v2.layout_parser import LayoutGraph, parse_layout
from experiments.overcooked_v2.options import OCV2OptionLibrary
from experiments.overcooked_v2.partner_pool import make_training_partners


def preflight_layout(
    env: Any,
    layout_name: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = config or {}
    layout_graph = parse_layout(env, layout_name)
    option_lib = OCV2OptionLibrary(
        layout_graph,
        max_option_steps=int(_cfg(config, "options.max_option_steps", 12)),
    )
    eta = float(_cfg(config, "graph.ce_eta", _cfg(config, "graph.eta", 0.2)))
    max_factors = int(_cfg(config, "graph.max_factors", 16))
    min_weight = float(_cfg(config, "graph.ce_min_weight", _cfg(config, "graph.min_weight", 20.0)))
    gamma = float(_cfg(config, "training.gamma", 0.99))
    horizon = int(_cfg(config, "graph.local_return_horizon_options", 5))
    cost_coef = float(_cfg(config, "training.cost_coef", 1.0))
    shaped_reward_coef = float(_cfg(config, "training.shaped_reward_coef", 0.0))

    ce_matrix, replay_rows, ce_source = _ce_matrix_and_replay(
        env,
        option_lib,
        layout_name,
        config,
        gamma=gamma,
        horizon_options=horizon,
        min_weight=min_weight,
        cost_coef=cost_coef,
        shaped_reward_coef=shaped_reward_coef,
    )
    ce_stats = _ce_stats(ce_matrix, eta, max_factors, option_lib.options)
    proxy_stats = _partner_return_proxy_stats(
        env,
        option_lib,
        layout_name,
        config,
        replay_rows,
        gamma=gamma,
        horizon_options=horizon,
        cost_coef=cost_coef,
        shaped_reward_coef=shaped_reward_coef,
    )

    reference_gap = proxy_stats["partner_return_variance_proxy"]
    default_gap_threshold = 0.05 * abs(proxy_stats["mean_return_across_all"])
    gap_threshold = _cfg(config, "layouts.filter.min_reference_gap", None)
    if gap_threshold is None:
        gap_threshold = _cfg(config, "diagnostics.gap_threshold", default_gap_threshold)

    stats: dict[str, Any] = {
        "layout_name": layout_name,
        "accepted": False,
        "ce_source": ce_source,
        "num_valid_options": int(option_lib.num_options),
        "num_bottlenecks": int(len(layout_graph.bottlenecks)),
        "num_articulation_bottlenecks": int(len(layout_graph.bottlenecks)),
        "num_entities": int(len(layout_graph.entities)),
        "num_passable_cells": int(np.asarray(layout_graph.passable).sum()),
        "num_factors_ce_above_eta": ce_stats["num_factors_ce_above_eta"],
        "num_ce_factors": ce_stats["num_factors_ce_above_eta"],
        "ce_matrix_sparsity": ce_stats["ce_matrix_sparsity"],
        "top_ce_mean": ce_stats["top_ce_mean"],
        "top_ce_pairs": ce_stats["top_ce_pairs"],
        "partner_induced_return_variance": proxy_stats["partner_return_mean_variance"],
        "partner_return_variance_proxy": reference_gap,
        "reference_base_gap_proxy": reference_gap,
        "reference_base_gap": reference_gap,
        "mean_return_across_all": proxy_stats["mean_return_across_all"],
        "mean_return_by_partner": proxy_stats["mean_return_by_partner"],
        "resource_contention_score": _resource_contention_score(layout_graph),
        "route_overlap_score": _route_overlap_score(layout_graph, option_lib),
        "recipe_uncertainty_score": _recipe_uncertainty_score(layout_graph),
        "min_options": int(_cfg(config, "layouts.filter.min_options", 8)),
        "min_ce_factors": int(_cfg(config, "layouts.filter.min_ce_factors", 4)),
        "ce_threshold": float(_cfg(config, "layouts.filter.ce_threshold", eta)),
        "gap_threshold": float(gap_threshold),
    }
    stats["accepted"] = accept_layout(stats)
    return stats


def accept_layout(stats: dict[str, Any]) -> bool:
    return (
        int(stats["num_valid_options"]) >= int(stats["min_options"])
        and int(stats["num_factors_ce_above_eta"]) >= int(stats["min_ce_factors"])
        and float(stats["top_ce_mean"]) >= float(stats["ce_threshold"])
        and float(stats["partner_return_variance_proxy"]) >= float(stats["gap_threshold"])
    )


def estimate_reference_base_gap_proxy(
    env: Any,
    option_lib: OCV2OptionLibrary,
    partners: list[Any],
    *,
    layout_name: str,
    episodes: int = 100,
    max_options_per_episode: int | None = None,
    seed: int = 0,
    gamma: float = 0.99,
    horizon_options: int = 5,
    cost_coef: float = 1.0,
    shaped_reward_coef: float = 0.0,
) -> dict[str, Any]:
    rows = collect_option_replay(
        env,
        partners,
        option_lib,
        layout_name=layout_name,
        episodes=episodes,
        max_options_per_episode=max_options_per_episode,
        seed=seed,
        gamma=gamma,
        horizon_options=horizon_options,
        cost_coef=cost_coef,
        shaped_reward_coef=shaped_reward_coef,
    )
    return _partner_return_stats(rows)


def _ce_matrix_and_replay(
    env: Any,
    option_lib: OCV2OptionLibrary,
    layout_name: str,
    config: dict[str, Any],
    *,
    gamma: float,
    horizon_options: int,
    min_weight: float,
    cost_coef: float = 1.0,
    shaped_reward_coef: float = 0.0,
) -> tuple[np.ndarray, list[OptionReplayRow] | None, str]:
    ce_path = _cfg(config, "graph.ce_path", None)
    replay_path = _cfg(config, "graph.replay_path", None)
    ce_matrix = _explicit_ce_matrix(config)
    replay_rows: list[OptionReplayRow] | None = None

    if ce_matrix is None and ce_path is not None:
        ce_matrix = np.load(ce_path)
        ce_source = str(ce_path)
    elif ce_matrix is not None:
        ce_source = "config.ce_matrix"
    else:
        ce_source = "collected_replay"

    if replay_path is not None:
        replay_rows, _ = load_replay_npz(replay_path)

    if ce_matrix is None:
        partners = make_training_partners(option_lib)
        episodes = int(_cfg(config, "graph.ce_episodes", 100))
        max_options_per_episode = int(_cfg(config, "graph.ce_max_options_per_episode", 20))
        replay_rows = collect_option_replay(
            env,
            partners,
            option_lib,
            layout_name=layout_name,
            episodes=episodes,
            max_options_per_episode=max_options_per_episode,
            seed=int(_cfg(config, "diagnostics.seed", 0)),
            gamma=gamma,
            horizon_options=horizon_options,
            cost_per_step=float(_cfg(config, "training.cost_per_step", 1.0)),
            cost_coef=cost_coef,
            shaped_reward_coef=shaped_reward_coef,
        )
        ce_matrix = estimate_empirical_ce(
            replay_rows,
            option_lib.num_options,
            min_weight=min_weight,
        )

    return np.asarray(ce_matrix, dtype=np.float32), replay_rows, ce_source


def _partner_return_proxy_stats(
    env: Any,
    option_lib: OCV2OptionLibrary,
    layout_name: str,
    config: dict[str, Any],
    replay_rows: list[OptionReplayRow] | None,
    *,
    gamma: float,
    horizon_options: int,
    cost_coef: float = 1.0,
    shaped_reward_coef: float = 0.0,
) -> dict[str, Any]:
    if replay_rows:
        return _partner_return_stats(
            replay_rows,
            cost_coef=cost_coef,
            shaped_reward_coef=shaped_reward_coef,
        )

    partners = make_training_partners(option_lib)
    episodes = int(_cfg(config, "diagnostics.proxy_episodes", 100))
    max_options_per_episode = int(_cfg(config, "graph.ce_max_options_per_episode", 20))
    return estimate_reference_base_gap_proxy(
        env,
        option_lib,
        partners,
        layout_name=layout_name,
        episodes=episodes,
        max_options_per_episode=max_options_per_episode,
        seed=int(_cfg(config, "diagnostics.seed", 0)),
        gamma=gamma,
        horizon_options=horizon_options,
        cost_coef=cost_coef,
        shaped_reward_coef=shaped_reward_coef,
    )


def _partner_return_stats(
    rows: list[OptionReplayRow],
    cost_coef: float = 1.0,
    shaped_reward_coef: float = 0.0,
) -> dict[str, Any]:
    returns: dict[str, dict[int, float]] = {}
    for row in rows:
        returns.setdefault(row.partner_name, {})
        training_return = (
            row.reward_sum
            + float(shaped_reward_coef) * row.shaped_reward_sum
            - float(cost_coef) * row.realized_cost
        )
        returns[row.partner_name][row.episode_id] = (
            returns[row.partner_name].get(row.episode_id, 0.0) + training_return
        )

    mean_by_partner = {
        name: float(np.mean(list(episode_returns.values())))
        for name, episode_returns in returns.items()
        if episode_returns
    }
    values = list(mean_by_partner.values())
    if values:
        proxy = float(max(values) - min(values))
        mean_all = float(np.mean(values))
        variance = float(np.var(values))
    else:
        proxy = 0.0
        mean_all = 0.0
        variance = 0.0

    return {
        "partner_return_variance_proxy": proxy,
        "partner_return_mean_variance": variance,
        "mean_return_across_all": mean_all,
        "mean_return_by_partner": mean_by_partner,
    }


def _ce_stats(
    ce_matrix: np.ndarray,
    eta: float,
    max_factors: int,
    options: list[Any] | None = None,
) -> dict[str, Any]:
    ce = np.asarray(ce_matrix, dtype=float)
    positive = ce > float(eta)
    if options is not None:
        for i in range(ce.shape[0]):
            for j in range(ce.shape[1]):
                if not _factorable_for_stats(options, i, j):
                    positive[i, j] = False
    top_pairs = [
        (float(ce[i, j]), int(i), int(j))
        for i in range(ce.shape[0])
        for j in range(ce.shape[1])
        if ce[i, j] > 0.0 and _factorable_for_stats(options, i, j)
    ]
    top_pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    top_scores = [score for score, _, _ in top_pairs[:max_factors]]
    return {
        "num_factors_ce_above_eta": int(min(np.count_nonzero(positive), max_factors)),
        "ce_matrix_sparsity": float(np.mean(positive)) if ce.size else 0.0,
        "top_ce_mean": float(np.mean(top_scores)) if top_scores else 0.0,
        "top_ce_pairs": [
            {"ce_score": score, "ego_option": i, "partner_option": j}
            for score, i, j in top_pairs[:max_factors]
        ],
    }


def _factorable_for_stats(options: list[Any] | None, i: int, j: int) -> bool:
    if options is None:
        return i != j
    if i == j or i >= len(options) or j >= len(options):
        return False
    return getattr(options[i], "kind", None) != "noop" and getattr(options[j], "kind", None) != "noop"


def _resource_contention_score(layout_graph: LayoutGraph) -> float:
    ingredient_piles = [
        entity
        for entity in layout_graph.entities.values()
        if entity.kind.startswith("ingredient_pile")
    ]
    if not ingredient_piles:
        return 0.0
    return float(min(1.0, 2.0 / max(1, len(ingredient_piles))))


def _route_overlap_score(
    layout_graph: LayoutGraph,
    option_lib: OCV2OptionLibrary,
) -> float:
    bottlenecks = tuple(layout_graph.bottlenecks)
    if not bottlenecks:
        return 0.0

    targets = [
        tuple((opt.metadata or {}).get("interaction_cells", ()))
        for opt in option_lib.options
        if opt.kind not in {"noop", "wait_at_bottleneck"}
    ]
    checked = 0
    through_bottleneck = 0
    for left_idx, left_targets in enumerate(targets):
        for right_targets in targets[left_idx + 1 :]:
            if not left_targets or not right_targets:
                continue
            checked += 1
            if _target_pair_uses_bottleneck(layout_graph, left_targets, right_targets):
                through_bottleneck += 1
    if checked == 0:
        return 0.0
    return float(through_bottleneck / checked)


def _recipe_uncertainty_score(layout_graph: LayoutGraph) -> float:
    has_indicator = bool(layout_graph.entities_by_kind.get("recipe_indicator"))
    has_button = bool(layout_graph.entities_by_kind.get("button_recipe_indicator"))
    return float(has_indicator) * 0.5 + float(has_button) * 0.5


def _target_pair_uses_bottleneck(
    layout_graph: LayoutGraph,
    left_targets: tuple[tuple[int, int], ...],
    right_targets: tuple[tuple[int, int], ...],
) -> bool:
    for left in left_targets:
        for right in right_targets:
            direct = layout_graph.shortest_path_dist.get((left, right))
            if direct is None:
                continue
            for bottleneck in layout_graph.bottlenecks:
                left_dist = layout_graph.shortest_path_dist.get((left, bottleneck))
                right_dist = layout_graph.shortest_path_dist.get((bottleneck, right))
                if left_dist is not None and right_dist is not None:
                    if left_dist + right_dist == direct:
                        return True
    return False

def _explicit_ce_matrix(config: dict[str, Any]) -> np.ndarray | None:
    value = _cfg(config, "graph.ce_matrix", None)
    if value is None:
        value = config.get("ce_matrix")
    if value is None:
        return None
    return np.asarray(value, dtype=np.float32)


def _cfg(config: dict[str, Any], dotted: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _load_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - optional CLI dependency
            raise ImportError("YAML config loading requires PyYAML.") from exc
        return dict(yaml.safe_load(text) or {})
    return dict(json.loads(text))


def _candidate_layouts(config: dict[str, Any], explicit_layout: str | None) -> list[str]:
    if explicit_layout:
        return [explicit_layout]
    configured_layout = config.get("layout")
    if configured_layout:
        return [str(configured_layout)]
    candidates = _cfg(config, "layouts.candidate", None)
    if not candidates:
        raise ValueError("No layout supplied. Use --layout or layouts.candidate in config.")
    return [str(layout) for layout in candidates]


def _make_env(layout: str, config: dict[str, Any]) -> OCV2Adapter:
    env_cfg = config.setdefault("env", {})
    env_cfg["observation_type"] = "default"
    env_cfg["force_path_planning"] = False
    return OCV2Adapter(
        layout,
        max_steps=int(env_cfg.get("max_steps", 200)),
        observation_type="default",
        negative_rewards=bool(env_cfg.get("negative_rewards", True)),
        sample_recipe_on_delivery=bool(env_cfg.get("sample_recipe_on_delivery", True)),
        random_reset=bool(env_cfg.get("random_reset", False)),
        random_agent_positions=bool(env_cfg.get("random_agent_positions", False)),
        force_path_planning=False,
    )


def _cmd_preflight(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    results = []
    for layout in _candidate_layouts(config, args.layout):
        env = _make_env(layout, config)
        results.append(preflight_layout(env, layout, config))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OvercookedV2 layout preflight checks")
    parser.add_argument("--config", default=None)
    parser.add_argument("--layout", default=None)
    parser.add_argument("--output", required=True)
    parser.set_defaults(func=_cmd_preflight)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
