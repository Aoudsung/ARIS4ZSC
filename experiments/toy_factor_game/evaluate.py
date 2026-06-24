"""Evaluation for ARIS-Bellman Toy Factor Game experiments."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.metrics import bootstrap_ci, expected_calibration_error
from common.utils import get_device, save_results, set_seed
from toy_factor_game.env import NUM_FACTORS, ToyFactorGameEnv
from toy_factor_game.graph_config import GRAPH_VARIANTS, GraphConfig, get_graph_config, stable_convention_seed
from toy_factor_game.gtvoi import bellman_delta_info, compute_mi
from toy_factor_game.options import NUM_OPTIONS, OptionID, get_option_action, get_option_cost, get_valid_options
from toy_factor_game.policy import METHODS, TRUE_BELIEF_METHODS, ActiveFactorAgent, uniform_marginals
from toy_factor_game.train import all_conventions, collect_episode, model_output_dir, oracle_marginals


EVAL_METHODS = METHODS
EXPERIMENT_SCHEMA = "aris_bellman_v4.1"
EVAL_SCHEMA = "aris_bellman_eval_v4.1"
PROPOSAL_VERSION = "v4"
CODE_FIX_LEVEL = "ce-all-conventions-criticality-diagnostics-oracle-planner"
EXP3_ROUTING_CONDITIONS = (
    ("aris_bellman", "full_support"),
    ("aris_bellman", "shuffled_routes"),
    ("aris_bellman", "shuffled_relevance"),
    ("aris_bellman", "random_same_size"),
)


def model_path_for(output_dir: Path, seed: int, method: str, graph_variant: str) -> Path:
    return model_output_dir(output_dir, seed, method, graph_variant) / "model.pt"


def results_path_for(output_dir: Path, seed: int, method: str, graph_variant: str) -> Path:
    return model_output_dir(output_dir, seed, method, graph_variant) / "results.json"


def load_checkpoint(model_path: Path, device):
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != EXPERIMENT_SCHEMA:
        raise RuntimeError(f"{model_path} is not an ARIS-Bellman v4.1 checkpoint")
    return checkpoint["state_dict"]


def load_agent(model_path: Path, device, graph_config: GraphConfig, hidden_dim: int, method: str):
    env = ToyFactorGameEnv()
    agent = ActiveFactorAgent(
        obs_dim=env.obs_dim,
        n_actions=env.n_actions,
        n_options=NUM_OPTIONS,
        graph_config=graph_config,
        hidden_dim=hidden_dim,
        method=method,
    ).to(device)
    agent.load_state_dict(load_checkpoint(model_path, device))
    agent.eval()
    return agent


def random_agent(device, graph_config: GraphConfig, hidden_dim: int):
    env = ToyFactorGameEnv()
    return ActiveFactorAgent(
        obs_dim=env.obs_dim,
        n_actions=env.n_actions,
        n_options=NUM_OPTIONS,
        graph_config=graph_config,
        hidden_dim=hidden_dim,
        method="random_policy",
    ).to(device).eval()


def _empty_diagnostics(oracle_gap_status: str = "not_applicable") -> dict:
    return {
        "actual_mi_mean": None,
        "delta_info_mean": None,
        "positive_delta_info_rate": None,
        "topk_delta_info_mean": None,
        "diagnostic_action_count": 0,
        "diagnostic_opportunity_cost": 0.0,
        "reward_after_first_diagnostic": None,
        "reward_after_first_high_delta_info": None,
        "delta_info_future_return_corr": None,
        "mi_future_return_corr": None,
        "belief_swap_delta_q_mean": None,
        "belief_swap_selected_delta_mean": None,
        "belief_swap_maxq_delta_mean": None,
        "belief_swap_abs_maxq_delta_mean": None,
        "belief_swap_action_flip_rate": None,
        "per_factor_abs_maxq_delta": None,
        "per_factor_action_flip_rate": None,
        "q_base_action": None,
        "uniform_belief_action": None,
        "full_q_action": None,
        "diagnostic_count_base_reference": 0,
        "diagnostic_count_uniform_reference": 0,
        "oracle_gap_closed_after_diagnostic": None,
        "oracle_gap_status": oracle_gap_status,
    }


def _env_terminal(env: ToyFactorGameEnv) -> bool:
    return env.step_count >= env.max_steps or (env.deliveries_left > 0 and env.deliveries_right > 0)


def _planner_state_key(env: ToyFactorGameEnv, horizon: int) -> tuple:
    return (
        horizon,
        tuple(env.ego_pos),
        tuple(env.partner_pos),
        bool(env.ego_carrying),
        bool(env.partner_carrying),
        bool(env.resource_a_available),
        bool(env.resource_b_available),
        int(env.deliveries_left),
        int(env.deliveries_right),
        int(env.step_count),
        tuple(sorted(env.partner_convention.modes.items())),
    )


def oracle_planner_option_value(
    env: ToyFactorGameEnv,
    option: OptionID,
    horizon: int,
    gamma: float,
    memo: dict[tuple, float] | None = None,
) -> float:
    if horizon <= 0 or _env_terminal(env):
        return 0.0
    sim_env = copy.deepcopy(env)
    action = get_option_action(option, sim_env.ego_pos, sim_env.ego_carrying)
    _obs, reward, done, _info = sim_env.step(action)
    value = float(reward - get_option_cost(option))
    if not done and horizon > 1:
        value += gamma * oracle_planner_value(sim_env, horizon - 1, gamma, memo)
    return value


def oracle_planner_value(
    env: ToyFactorGameEnv,
    horizon: int,
    gamma: float,
    memo: dict[tuple, float] | None = None,
) -> float:
    if horizon <= 0 or _env_terminal(env):
        return 0.0
    if memo is None:
        memo = {}
    key = _planner_state_key(env, horizon)
    if key in memo:
        return memo[key]
    valid_options = get_valid_options(env)
    if not valid_options:
        memo[key] = 0.0
        return 0.0
    best = max(
        oracle_planner_option_value(env, option, horizon, gamma, memo)
        for option in valid_options
    )
    memo[key] = float(best)
    return memo[key]


def oracle_planner_select(env: ToyFactorGameEnv, horizon: int, gamma: float) -> OptionID:
    valid_options = get_valid_options(env)
    if not valid_options:
        return OptionID.NOOP
    memo: dict[tuple, float] = {}
    scored = [
        (oracle_planner_option_value(env, option, horizon, gamma, memo), int(option), option)
        for option in valid_options
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][2]


def factor_metrics(agent, device, episode_data: dict, graph_config: GraphConfig) -> tuple[list[float], list[int]]:
    if agent.method in ("base_only", "global_gru", "random_policy"):
        return [], []
    if not episode_data["evidence"]:
        return [], []
    evidence_seq = torch.stack(episode_data["evidence"]).unsqueeze(0).to(device)
    labels = graph_config.labels_from_convention(episode_data["infos"][0]["convention"])
    with torch.no_grad():
        if agent.method in TRUE_BELIEF_METHODS:
            marginals = oracle_marginals(graph_config, episode_data["infos"][0]["convention"], 1, device)
        else:
            marginals = agent.belief_model(evidence_seq)
    probs, correct = [], []
    for factor_idx, label in enumerate(labels):
        if not graph_config.ground_truth_mask[factor_idx]:
            continue
        pred = int(marginals[factor_idx][0].argmax().item())
        probs.append(float(marginals[factor_idx][0].max().item()))
        correct.append(int(pred == int(label)))
    return probs, correct


def _safe_corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2:
        return None
    x = np.array(xs, dtype=np.float64)
    y = np.array(ys, dtype=np.float64)
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _uniform_like(marginal: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(marginal) / float(marginal.shape[-1])


def _masked_values(q_values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    if valid_mask.dim() == 1:
        valid_mask = valid_mask.unsqueeze(0)
    return q_values.masked_fill(~valid_mask.bool(), -1e9)


def per_factor_swap_diagnostics(
    agent,
    obs: torch.Tensor,
    marginals: list[torch.Tensor],
    valid_mask: torch.Tensor,
) -> list[dict]:
    q_actual = _masked_values(agent.q_values(obs, marginals), valid_mask)
    actual_action = int(q_actual.argmax(dim=-1).item())
    rows = []
    for factor_idx in range(len(marginals)):
        swapped = [m.clone() for m in marginals]
        swapped[factor_idx] = _uniform_like(swapped[factor_idx])
        q_swap = _masked_values(agent.q_values(obs, swapped), valid_mask)
        swapped_action = int(q_swap.argmax(dim=-1).item())
        maxq_delta = float(q_actual.max().item() - q_swap.max().item())
        rows.append(
            {
                "factor_idx": factor_idx,
                "selected_delta": float(q_actual[0, actual_action].item() - q_swap[0, actual_action].item()),
                "maxq_delta": maxq_delta,
                "abs_maxq_delta": abs(maxq_delta),
                "action_flip": int(actual_action != swapped_action),
                "actual_action": actual_action,
                "swapped_action": swapped_action,
            }
        )
    return rows


def diagnostic_metrics(agent, device, episode_data: dict, graph_config: GraphConfig, gamma: float):
    if agent.method in ("base_only", "flat_latent", "global_gru", "true_belief_factorq", "true_belief_flatq", "random_policy"):
        return _empty_diagnostics()
    factor_hidden = agent.belief_model.initial_hidden(1, device)
    mi_values = []
    delta_values = []
    future_returns = []
    belief_swap_delta_q = []
    belief_swap_maxq_delta = []
    belief_swap_abs_maxq_delta = []
    belief_swap_action_flip = []
    per_factor_abs_maxq: dict[int, list[float]] = {}
    per_factor_action_flip: dict[int, list[int]] = {}
    q_base_actions = []
    q_uniform_actions = []
    q_full_actions = []
    opportunity_costs = []
    diagnostic_indices = []
    diagnostic_uniform_indices = []
    high_delta_indices = []
    for t, evidence in enumerate(episode_data["evidence"]):
        obs_t = episode_data["obs"][t].unsqueeze(0).to(device)
        next_obs_t = episode_data["next_obs"][t].unsqueeze(0).to(device)
        before = agent.belief_model._marginals_from_h(factor_hidden)
        valid_mask_t = episode_data["valid_masks"][t].unsqueeze(0).to(device)
        q_task = _masked_values(agent.q_base_values(obs_t), valid_mask_t)
        selected = int(episode_data["options"][t])
        best_task = int(q_task.argmax(dim=-1).item())
        opportunity_cost = max(0.0, float(q_task[0, best_task].item() - q_task[0, selected].item()))
        uniform_before = uniform_marginals(graph_config.factor_modes, 1, device)
        q_actual = _masked_values(agent.q_values(obs_t, before), valid_mask_t)
        q_uniform = _masked_values(agent.q_values(obs_t, uniform_before), valid_mask_t)
        best_uniform = int(q_uniform.argmax(dim=-1).item())
        best_full = int(q_actual.argmax(dim=-1).item())
        q_base_actions.append(best_task)
        q_uniform_actions.append(best_uniform)
        q_full_actions.append(best_full)
        belief_swap_delta_q.append(float(q_actual[0, selected].item() - q_uniform[0, selected].item()))
        maxq_delta = float(q_actual.max().item() - q_uniform.max().item())
        belief_swap_maxq_delta.append(maxq_delta)
        belief_swap_abs_maxq_delta.append(abs(maxq_delta))
        belief_swap_action_flip.append(int(best_full != best_uniform))
        for swap_row in per_factor_swap_diagnostics(agent, obs_t, before, valid_mask_t):
            per_factor_abs_maxq.setdefault(swap_row["factor_idx"], []).append(swap_row["abs_maxq_delta"])
            per_factor_action_flip.setdefault(swap_row["factor_idx"], []).append(swap_row["action_flip"])
        with torch.no_grad():
            factor_hidden = agent.belief_model.step_history(evidence.to(device), factor_hidden)
            after = agent.belief_model._marginals_from_h(factor_hidden)
            mi = float(compute_mi(before, after).item())
            delta_info = float(
                bellman_delta_info(
                    agent,
                    next_obs_t,
                    before,
                    after,
                    valid_mask=episode_data["next_valid_masks"][t].unsqueeze(0).to(device),
                    gamma=gamma,
                ).item()
            )
        mi_values.append(mi)
        delta_values.append(delta_info)
        future_returns.append(float(sum(episode_data["env_rewards"][t + 1:])))
        if delta_info > 1e-3:
            high_delta_indices.append(t)
        if delta_info > 1e-3 and selected != best_task:
            diagnostic_indices.append(t)
            opportunity_costs.append(opportunity_cost)
        if delta_info > 1e-3 and selected != best_uniform:
            diagnostic_uniform_indices.append(t)

    first_reward = None
    if diagnostic_indices:
        first = diagnostic_indices[0]
        first_reward = float(sum(episode_data["env_rewards"][first:]))
    first_high_delta_reward = None
    if high_delta_indices:
        first = high_delta_indices[0]
        first_high_delta_reward = float(sum(episode_data["env_rewards"][first:]))
    topk_delta = None
    if delta_values:
        k = min(3, len(delta_values))
        topk_delta = float(np.mean(sorted(delta_values, reverse=True)[:k]))
    return {
        "actual_mi_mean": float(np.mean(mi_values)) if mi_values else None,
        "delta_info_mean": float(np.mean(delta_values)) if delta_values else None,
        "positive_delta_info_rate": float(np.mean([value > 1e-3 for value in delta_values])) if delta_values else None,
        "topk_delta_info_mean": topk_delta,
        "diagnostic_action_count": int(len(diagnostic_indices)),
        "diagnostic_opportunity_cost": float(np.sum(opportunity_costs)) if opportunity_costs else 0.0,
        "reward_after_first_diagnostic": first_reward,
        "reward_after_first_high_delta_info": first_high_delta_reward,
        "delta_info_future_return_corr": _safe_corr(delta_values, future_returns),
        "mi_future_return_corr": _safe_corr(mi_values, future_returns),
        "belief_swap_delta_q_mean": float(np.mean(belief_swap_delta_q)) if belief_swap_delta_q else None,
        "belief_swap_selected_delta_mean": float(np.mean(belief_swap_delta_q)) if belief_swap_delta_q else None,
        "belief_swap_maxq_delta_mean": float(np.mean(belief_swap_maxq_delta)) if belief_swap_maxq_delta else None,
        "belief_swap_abs_maxq_delta_mean": float(np.mean(belief_swap_abs_maxq_delta)) if belief_swap_abs_maxq_delta else None,
        "belief_swap_action_flip_rate": float(np.mean(belief_swap_action_flip)) if belief_swap_action_flip else None,
        "per_factor_abs_maxq_delta": {
            str(factor_idx): float(np.mean(values)) for factor_idx, values in sorted(per_factor_abs_maxq.items())
        },
        "per_factor_action_flip_rate": {
            str(factor_idx): float(np.mean(values)) for factor_idx, values in sorted(per_factor_action_flip.items())
        },
        "q_base_action": int(q_base_actions[0]) if q_base_actions else None,
        "uniform_belief_action": int(q_uniform_actions[0]) if q_uniform_actions else None,
        "full_q_action": int(q_full_actions[0]) if q_full_actions else None,
        "diagnostic_count_base_reference": int(len(diagnostic_indices)),
        "diagnostic_count_uniform_reference": int(len(diagnostic_uniform_indices)),
        "oracle_gap_closed_after_diagnostic": None,
        "oracle_gap_status": "missing_oracle",
    }


def training_efficiency(output_dir: Path, seed: int, method: str, graph_variant: str) -> dict:
    path = results_path_for(output_dir, seed, method, graph_variant)
    if not path.exists():
        return {"eval_reward_auc": None, "first_eval_episode_reward_ge_0": None}
    data = json.loads(path.read_text())
    eval_log = data.get("eval", [])
    if not eval_log:
        return {"eval_reward_auc": None, "first_eval_episode_reward_ge_0": None}
    episodes = np.array([row["episode"] for row in eval_log], dtype=np.float64)
    rewards = np.array([row.get("episode_reward", 0.0) for row in eval_log], dtype=np.float64)
    first_ge_0 = None
    for row in eval_log:
        if row.get("episode_reward", -np.inf) >= 0.0:
            first_ge_0 = int(row["episode"])
            break
    return {
        "eval_reward_auc": float(np.trapz(rewards, episodes)) if len(rewards) > 1 else float(rewards[0]),
        "first_eval_episode_reward_ge_0": first_ge_0,
    }


def summarize_rows(rows: list[dict]) -> dict:
    if not rows:
        return {"status": "empty", "n": 0}
    rewards = np.array([row["episode_reward"] for row in rows], dtype=np.float64)
    td_rewards = np.array([row["td_reward"] for row in rows], dtype=np.float64)
    early_rewards = np.array([row["early_reward"] for row in rows], dtype=np.float64)
    collisions = np.array([row["collisions"] for row in rows], dtype=np.float64)
    completions = np.array([row["completion"] for row in rows], dtype=np.float64)
    times = np.array([row["time_to_completion"] for row in rows], dtype=np.float64)
    factor_probs = np.array([p for row in rows for p in row["factor_probs"]], dtype=np.float64)
    factor_correct = np.array([c for row in rows for c in row["factor_correct"]], dtype=np.float64)
    diagnostic_counts = np.array([row["diagnostic_action_count"] for row in rows], dtype=np.float64)
    diagnostic_costs = np.array([row["diagnostic_opportunity_cost"] for row in rows], dtype=np.float64)
    delta_infos = np.array(
        [row["delta_info_mean"] for row in rows if row["delta_info_mean"] is not None],
        dtype=np.float64,
    )
    actual_mis = np.array(
        [row["actual_mi_mean"] for row in rows if row["actual_mi_mean"] is not None],
        dtype=np.float64,
    )
    positive_delta_rates = np.array(
        [row["positive_delta_info_rate"] for row in rows if row["positive_delta_info_rate"] is not None],
        dtype=np.float64,
    )
    topk_delta_infos = np.array(
        [row["topk_delta_info_mean"] for row in rows if row["topk_delta_info_mean"] is not None],
        dtype=np.float64,
    )
    reward_after_diag = np.array(
        [row["reward_after_first_diagnostic"] for row in rows if row["reward_after_first_diagnostic"] is not None],
        dtype=np.float64,
    )
    reward_after_high_delta = np.array(
        [row["reward_after_first_high_delta_info"] for row in rows if row["reward_after_first_high_delta_info"] is not None],
        dtype=np.float64,
    )
    delta_corrs = np.array(
        [row["delta_info_future_return_corr"] for row in rows if row["delta_info_future_return_corr"] is not None],
        dtype=np.float64,
    )
    mi_corrs = np.array(
        [row["mi_future_return_corr"] for row in rows if row["mi_future_return_corr"] is not None],
        dtype=np.float64,
    )
    belief_swap_delta_q = np.array(
        [row["belief_swap_delta_q_mean"] for row in rows if row["belief_swap_delta_q_mean"] is not None],
        dtype=np.float64,
    )
    belief_swap_maxq_delta = np.array(
        [row["belief_swap_maxq_delta_mean"] for row in rows if row["belief_swap_maxq_delta_mean"] is not None],
        dtype=np.float64,
    )
    belief_swap_abs_maxq_delta = np.array(
        [row["belief_swap_abs_maxq_delta_mean"] for row in rows if row["belief_swap_abs_maxq_delta_mean"] is not None],
        dtype=np.float64,
    )
    belief_swap_action_flip = np.array(
        [row["belief_swap_action_flip_rate"] for row in rows if row["belief_swap_action_flip_rate"] is not None],
        dtype=np.float64,
    )
    diagnostic_base_counts = np.array([row["diagnostic_count_base_reference"] for row in rows], dtype=np.float64)
    diagnostic_uniform_counts = np.array([row["diagnostic_count_uniform_reference"] for row in rows], dtype=np.float64)
    per_factor_abs: dict[str, list[float]] = {}
    per_factor_flip: dict[str, list[float]] = {}
    for row in rows:
        for factor_idx, value in (row.get("per_factor_abs_maxq_delta") or {}).items():
            per_factor_abs.setdefault(factor_idx, []).append(float(value))
        for factor_idx, value in (row.get("per_factor_action_flip_rate") or {}).items():
            per_factor_flip.setdefault(factor_idx, []).append(float(value))
    oracle_gap_statuses = sorted({row.get("oracle_gap_status") for row in rows if row.get("oracle_gap_status")})

    return {
        "status": "done",
        "n": len(rows),
        "episode_reward_mean": float(rewards.mean()),
        "episode_reward_ci": list(bootstrap_ci(rewards)),
        "td_reward_mean": float(td_rewards.mean()),
        "early_reward_mean": float(early_rewards.mean()),
        "completion_rate": float(completions.mean()),
        "time_to_completion_mean": float(times.mean()),
        "collisions_mean": float(collisions.mean()),
        "factor_accuracy_mean": float(factor_correct.mean()) if len(factor_correct) else None,
        "ece": expected_calibration_error(factor_probs, factor_correct) if len(factor_correct) else None,
        "factor_metrics_note": "sanity_only_ground_truth_factors_excluding_synthetic",
        "actual_mi_mean": float(actual_mis.mean()) if len(actual_mis) else None,
        "delta_info_mean": float(delta_infos.mean()) if len(delta_infos) else None,
        "positive_delta_info_rate": float(positive_delta_rates.mean()) if len(positive_delta_rates) else None,
        "topk_delta_info_mean": float(topk_delta_infos.mean()) if len(topk_delta_infos) else None,
        "diagnostic_action_count_mean": float(diagnostic_counts.mean()),
        "diagnostic_opportunity_cost_mean": float(diagnostic_costs.mean()),
        "reward_after_first_diagnostic_mean": float(reward_after_diag.mean()) if len(reward_after_diag) else None,
        "reward_after_first_high_delta_info_mean": float(reward_after_high_delta.mean()) if len(reward_after_high_delta) else None,
        "delta_info_future_return_corr_mean": float(delta_corrs.mean()) if len(delta_corrs) else None,
        "mi_future_return_corr_mean": float(mi_corrs.mean()) if len(mi_corrs) else None,
        "belief_swap_delta_q_mean": float(belief_swap_delta_q.mean()) if len(belief_swap_delta_q) else None,
        "belief_swap_selected_delta_mean": float(belief_swap_delta_q.mean()) if len(belief_swap_delta_q) else None,
        "belief_swap_maxq_delta_mean": float(belief_swap_maxq_delta.mean()) if len(belief_swap_maxq_delta) else None,
        "belief_swap_abs_maxq_delta_mean": float(belief_swap_abs_maxq_delta.mean()) if len(belief_swap_abs_maxq_delta) else None,
        "belief_swap_action_flip_rate": float(belief_swap_action_flip.mean()) if len(belief_swap_action_flip) else None,
        "per_factor_abs_maxq_delta": {
            factor_idx: float(np.mean(values)) for factor_idx, values in sorted(per_factor_abs.items())
        },
        "per_factor_action_flip_rate": {
            factor_idx: float(np.mean(values)) for factor_idx, values in sorted(per_factor_flip.items())
        },
        "diagnostic_count_base_reference_mean": float(diagnostic_base_counts.mean()),
        "diagnostic_count_uniform_reference_mean": float(diagnostic_uniform_counts.mean()),
        "oracle_gap_closed_after_diagnostic": None,
        "oracle_gap_status": ",".join(oracle_gap_statuses) if oracle_gap_statuses else "missing_oracle",
    }


def evaluate_agent(agent, device, graph_config: GraphConfig, seed: int, n_per_conv: int, max_steps: int, gamma: float):
    rows = []
    for conv in all_conventions():
        convention_key = "-".join(str(conv.modes[factor_id]) for factor_id in range(NUM_FACTORS))
        for trial in range(n_per_conv):
            env_seed = stable_convention_seed(seed, conv.modes, trial)
            env = ToyFactorGameEnv(partner_convention=conv, max_steps=max_steps, seed=env_seed)
            with torch.no_grad():
                episode_data = collect_episode(
                    env,
                    agent,
                    device,
                    graph_config=graph_config,
                    explore_eps=0.0,
                    deterministic=True,
                )
            rewards = episode_data["env_rewards"]
            early_k = max(1, len(rewards) // 5)
            factor_probs, factor_correct = factor_metrics(agent, device, episode_data, graph_config)
            diagnostics = diagnostic_metrics(agent, device, episode_data, graph_config, gamma)
            completed = any(info.get("completed", False) for info in episode_data["infos"])
            rows.append(
                {
                    "convention": convention_key,
                    "trial": trial,
                    "episode_reward": float(sum(rewards)),
                    "td_reward": float(sum(episode_data["rewards"])),
                    "early_reward": float(sum(rewards[:early_k])),
                    "completion": float(completed),
                    "time_to_completion": len(rewards) if completed else max_steps + 1,
                    "collisions": float(sum(1 for info in episode_data["infos"] if info["collision"])),
                    "factor_probs": factor_probs,
                    "factor_correct": factor_correct,
                    **diagnostics,
                }
            )
    return rows


def collect_oracle_planner_episode(
    env: ToyFactorGameEnv,
    gamma: float,
    horizon: int,
) -> dict:
    obs = env.reset()
    transitions = {
        "obs": [],
        "next_obs": [],
        "options": [],
        "rewards": [],
        "env_rewards": [],
        "done": [],
        "infos": [],
    }
    for _t in range(env.max_steps):
        option = oracle_planner_select(env, horizon, gamma)
        action = get_option_action(option, env.ego_pos, env.ego_carrying)
        next_obs, env_reward, done, info = env.step(action)
        transitions["obs"].append(torch.tensor(obs, dtype=torch.float32))
        transitions["next_obs"].append(torch.tensor(next_obs, dtype=torch.float32))
        transitions["options"].append(int(option))
        transitions["rewards"].append(float(env_reward - get_option_cost(option)))
        transitions["env_rewards"].append(float(env_reward))
        transitions["done"].append(bool(done))
        transitions["infos"].append(info)
        obs = next_obs
        if done:
            break
    return transitions


def evaluate_oracle_planner(
    seed: int,
    n_per_conv: int,
    max_steps: int,
    gamma: float,
    horizon: int,
) -> list[dict]:
    rows = []
    for conv in all_conventions():
        convention_key = "-".join(str(conv.modes[factor_id]) for factor_id in range(NUM_FACTORS))
        for trial in range(n_per_conv):
            env_seed = stable_convention_seed(seed, conv.modes, trial)
            env = ToyFactorGameEnv(partner_convention=conv, max_steps=max_steps, seed=env_seed)
            episode_data = collect_oracle_planner_episode(env, gamma=gamma, horizon=horizon)
            rewards = episode_data["env_rewards"]
            early_k = max(1, len(rewards) // 5) if rewards else 1
            completed = any(info.get("completed", False) for info in episode_data["infos"])
            rows.append(
                {
                    "convention": convention_key,
                    "trial": trial,
                    "episode_reward": float(sum(rewards)),
                    "td_reward": float(sum(episode_data["rewards"])),
                    "early_reward": float(sum(rewards[:early_k])),
                    "completion": float(completed),
                    "time_to_completion": len(rewards) if completed else max_steps + 1,
                    "collisions": float(sum(1 for info in episode_data["infos"] if info["collision"])),
                    "factor_probs": [],
                    "factor_correct": [],
                    **_empty_diagnostics("planning_oracle"),
                }
            )
    return rows


def evaluate_method_for_graph(
    output_dir: Path,
    device,
    seed: int,
    graph_variant: str,
    method: str,
    hidden_dim: int,
    n_per_conv: int,
    max_steps: int,
    gamma: float,
    oracle_horizon: int,
    require_checkpoint: bool = False,
):
    graph_config = get_graph_config(graph_variant)
    if method == "oracle_planner":
        rows = evaluate_oracle_planner(seed, n_per_conv, max_steps, gamma, oracle_horizon)
        summary = summarize_rows(rows)
        summary["baseline_type"] = "planning_oracle"
        summary["oracle_horizon"] = int(oracle_horizon)
        return summary
    if method == "random_policy":
        agent = random_agent(device, graph_config, hidden_dim)
        rows = evaluate_agent(agent, device, graph_config, seed, n_per_conv, max_steps, gamma)
        return summarize_rows(rows)

    path = model_path_for(output_dir, seed, method, graph_variant)
    if not path.exists():
        if require_checkpoint:
            raise FileNotFoundError(
                f"Missing checkpoint for {method}/{graph_variant}: {path}. "
                "Exp 3 routing controls require separately trained checkpoints."
            )
        return {"status": "missing", "model_path": str(path)}
    agent = load_agent(path, device, graph_config, hidden_dim, method)
    rows = evaluate_agent(agent, device, graph_config, seed, n_per_conv, max_steps, gamma)
    summary = summarize_rows(rows)
    summary.update(training_efficiency(output_dir, seed, method, graph_variant))
    return summary


def exp1_policy_baselines(output_dir: Path, device, seed: int, hidden_dim: int, methods: list[str], graph_variants: list[str], n_per_conv: int, max_steps: int, gamma: float, oracle_horizon: int):
    return {
        graph_variant: {
            method: evaluate_method_for_graph(
                output_dir, device, seed, graph_variant, method, hidden_dim, n_per_conv, max_steps, gamma, oracle_horizon
            )
            for method in methods
        }
        for graph_variant in graph_variants
    }


def exp3_value_sufficiency(output_dir: Path, device, seed: int, hidden_dim: int, methods: list[str], n_per_conv: int, max_steps: int, gamma: float, oracle_horizon: int):
    method_comparison = {
        method: evaluate_method_for_graph(
            output_dir, device, seed, "full_support", method, hidden_dim, n_per_conv, max_steps, gamma, oracle_horizon
        )
        for method in methods
    }
    routing_controls = {}
    for method, graph_variant in EXP3_ROUTING_CONDITIONS:
        routing_controls[f"{method}/{graph_variant}"] = evaluate_method_for_graph(
            output_dir,
            device,
            seed,
            graph_variant,
            method,
            hidden_dim,
            n_per_conv,
            max_steps,
            gamma,
            oracle_horizon,
            require_checkpoint=True,
        )
    return {
        "method_comparison": method_comparison,
        "routing_relevance_controls": routing_controls,
        "routing_relevance_conditions": [
            {"method": method, "graph_variant": graph_variant}
            for method, graph_variant in EXP3_ROUTING_CONDITIONS
        ],
    }


def exp4_graph_robustness(output_dir: Path, device, seed: int, hidden_dim: int, method: str, graph_variants: list[str], n_per_conv: int, max_steps: int, gamma: float, oracle_horizon: int):
    results = {
        graph_variant: evaluate_method_for_graph(
            output_dir, device, seed, graph_variant, method, hidden_dim, n_per_conv, max_steps, gamma, oracle_horizon
        )
        for graph_variant in graph_variants
    }
    full_reward = None
    if "full_support" in results and results["full_support"].get("status") == "done":
        full_reward = results["full_support"].get("episode_reward_mean")
    for graph_variant, summary in results.items():
        if summary.get("status") != "done" or full_reward is None:
            summary["delta_return_vs_full_support"] = None
            summary["counterfactual_delta_status"] = "missing_full_support_or_variant"
            continue
        summary["delta_return_vs_full_support"] = float(summary["episode_reward_mean"] - full_reward)
        if graph_variant == "minus_critical":
            summary["counterfactual_delta_type"] = "factor_deletion_delta_return"
        elif graph_variant == "shuffled_routes":
            summary["counterfactual_delta_type"] = "route_deletion_delta_return"
        elif graph_variant == "shuffled_relevance":
            summary["counterfactual_delta_type"] = "relevance_deletion_delta_return"
        else:
            summary["counterfactual_delta_type"] = "graph_variant_delta_return"
        summary["counterfactual_delta_status"] = "computed_against_full_support"
    return results


def parse_csv(raw: str, allowed: tuple[str, ...]) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown values {unknown}; expected subset of {allowed}")
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/toy")
    parser.add_argument("--experiments", type=str, default="1,3,4")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument(
        "--methods",
        type=str,
        default="base_only,aris_bellman,flat_latent,global_gru,true_belief_factorq,true_belief_flatq,oracle_planner,random_policy",
    )
    parser.add_argument("--graph_variants", type=str, default=",".join(GRAPH_VARIANTS))
    parser.add_argument("--exp1_graph_variants", type=str, default="full_support,overcomplete")
    parser.add_argument("--n_per_conv", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--oracle_horizon", type=int, default=6)
    args = parser.parse_args()

    if args.n_per_conv <= 0:
        raise ValueError("--n_per_conv must be positive")
    if args.max_steps <= 0:
        raise ValueError("--max_steps must be positive")
    if args.oracle_horizon <= 0:
        raise ValueError("--oracle_horizon must be positive")

    set_seed(args.seed)
    device = get_device()
    output_dir = Path(args.output_dir)
    experiments = [int(part) for part in args.experiments.split(",") if part.strip()]
    methods = parse_csv(args.methods, EVAL_METHODS)
    graph_variants = parse_csv(args.graph_variants, GRAPH_VARIANTS)
    exp1_graph_variants = parse_csv(args.exp1_graph_variants, GRAPH_VARIANTS)
    exp4_method = "aris_bellman" if "aris_bellman" in methods else next(
        (method for method in methods if method not in ("random_policy", "oracle_planner")),
        "aris_bellman",
    )

    all_results = {}
    if 1 in experiments:
        all_results["exp1"] = exp1_policy_baselines(
            output_dir, device, args.seed, args.hidden_dim, methods, exp1_graph_variants,
            args.n_per_conv, args.max_steps, args.gamma, args.oracle_horizon,
        )
    if 3 in experiments:
        all_results["exp3"] = exp3_value_sufficiency(
            output_dir, device, args.seed, args.hidden_dim, methods,
            args.n_per_conv, args.max_steps, args.gamma, args.oracle_horizon,
        )
    if 4 in experiments:
        all_results["exp4"] = exp4_graph_robustness(
            output_dir, device, args.seed, args.hidden_dim, exp4_method, graph_variants,
            args.n_per_conv, args.max_steps, args.gamma, args.oracle_horizon,
        )

    results_path = output_dir / f"eval_results_seed{args.seed}.json"
    save_results(
        {
            "schema_version": EVAL_SCHEMA,
            "proposal_version": PROPOSAL_VERSION,
            "code_fix_level": CODE_FIX_LEVEL,
            "config": vars(args),
            "ce_estimation": get_graph_config("full_support").ce_metadata(),
            "results": all_results,
            "diagnostic_note": (
                "G-TVOI and MI are post-hoc trajectory diagnostics computed from real belief "
                "updates. They are not selectors, losses, or training signals."
            ),
            "ground_truth_note": (
                "Factor accuracy and ECE are computed only for ToyFactorGameEnv ground-truth "
                "factors; synthetic graph factors are excluded."
            ),
        },
        str(results_path),
    )
    print(f"All results saved to {results_path}")


if __name__ == "__main__":
    main()
