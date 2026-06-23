"""Evaluation for ARIS-Bellman Toy Factor Game experiments."""

from __future__ import annotations

import argparse
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
from toy_factor_game.options import NUM_OPTIONS
from toy_factor_game.policy import METHODS, ActiveFactorAgent, labels_to_marginals
from toy_factor_game.train import all_conventions, collect_episode, model_output_dir


EVAL_METHODS = METHODS


def model_path_for(output_dir: Path, seed: int, method: str, graph_variant: str) -> Path:
    return model_output_dir(output_dir, seed, method, graph_variant) / "model.pt"


def results_path_for(output_dir: Path, seed: int, method: str, graph_variant: str) -> Path:
    return model_output_dir(output_dir, seed, method, graph_variant) / "results.json"


def load_checkpoint(model_path: Path, device):
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != "aris_bellman_v1":
        raise RuntimeError(f"{model_path} is not an ARIS-Bellman checkpoint")
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


def factor_metrics(agent, device, episode_data: dict, graph_config: GraphConfig) -> tuple[list[float], list[int]]:
    if agent.method in ("global_gru", "random_policy"):
        return [], []
    if not episode_data["evidence"]:
        return [], []
    evidence_seq = torch.stack(episode_data["evidence"]).unsqueeze(0).to(device)
    labels = graph_config.labels_from_convention(episode_data["infos"][0]["convention"])
    with torch.no_grad():
        if agent.method == "oracle_belief":
            label_tensor = torch.tensor([labels], dtype=torch.long, device=device)
            marginals = labels_to_marginals(label_tensor, graph_config.factor_modes)
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


def diagnostic_metrics(agent, device, episode_data: dict, graph_config: GraphConfig, gamma: float):
    if agent.method in ("global_gru", "random_policy", "oracle_belief"):
        return {
            "actual_mi_mean": None,
            "delta_info_mean": None,
            "diagnostic_action_count": 0,
            "diagnostic_opportunity_cost": 0.0,
            "reward_after_first_diagnostic": None,
            "oracle_gap_closed_after_diagnostic": None,
        }
    factor_hidden = agent.belief_model.initial_hidden(1, device)
    mi_values = []
    delta_values = []
    opportunity_costs = []
    diagnostic_indices = []
    for t, evidence in enumerate(episode_data["evidence"]):
        obs_t = episode_data["obs"][t].unsqueeze(0).to(device)
        next_obs_t = episode_data["next_obs"][t].unsqueeze(0).to(device)
        before = agent.belief_model._marginals_from_h(factor_hidden)
        q_before = agent.q_values(obs_t, before)
        q_before = q_before.masked_fill(~episode_data["valid_masks"][t].unsqueeze(0).bool(), -1e9)
        selected = int(episode_data["options"][t])
        best = int(q_before.argmax(dim=-1).item())
        opportunity_cost = max(0.0, float(q_before[0, best].item() - q_before[0, selected].item()))
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
        if delta_info > 1e-3 and selected != best:
            diagnostic_indices.append(t)
            opportunity_costs.append(opportunity_cost)

    first_reward = None
    if diagnostic_indices:
        first = diagnostic_indices[0]
        first_reward = float(sum(episode_data["env_rewards"][first:]))
    return {
        "actual_mi_mean": float(np.mean(mi_values)) if mi_values else None,
        "delta_info_mean": float(np.mean(delta_values)) if delta_values else None,
        "diagnostic_action_count": int(len(diagnostic_indices)),
        "diagnostic_opportunity_cost": float(np.sum(opportunity_costs)) if opportunity_costs else 0.0,
        "reward_after_first_diagnostic": first_reward,
        "oracle_gap_closed_after_diagnostic": None,
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
    reward_after_diag = np.array(
        [row["reward_after_first_diagnostic"] for row in rows if row["reward_after_first_diagnostic"] is not None],
        dtype=np.float64,
    )

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
        "actual_mi_mean": float(actual_mis.mean()) if len(actual_mis) else None,
        "delta_info_mean": float(delta_infos.mean()) if len(delta_infos) else None,
        "diagnostic_action_count_mean": float(diagnostic_counts.mean()),
        "diagnostic_opportunity_cost_mean": float(diagnostic_costs.mean()),
        "reward_after_first_diagnostic_mean": float(reward_after_diag.mean()) if len(reward_after_diag) else None,
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
):
    graph_config = get_graph_config(graph_variant)
    if method == "random_policy":
        agent = random_agent(device, graph_config, hidden_dim)
        rows = evaluate_agent(agent, device, graph_config, seed, n_per_conv, max_steps, gamma)
        return summarize_rows(rows)

    path = model_path_for(output_dir, seed, method, graph_variant)
    if not path.exists():
        return {"status": "missing", "model_path": str(path)}
    agent = load_agent(path, device, graph_config, hidden_dim, method)
    rows = evaluate_agent(agent, device, graph_config, seed, n_per_conv, max_steps, gamma)
    summary = summarize_rows(rows)
    summary.update(training_efficiency(output_dir, seed, method, graph_variant))
    return summary


def exp1_policy_baselines(output_dir: Path, device, seed: int, hidden_dim: int, methods: list[str], graph_variants: list[str], n_per_conv: int, max_steps: int, gamma: float):
    return {
        graph_variant: {
            method: evaluate_method_for_graph(
                output_dir, device, seed, graph_variant, method, hidden_dim, n_per_conv, max_steps, gamma
            )
            for method in methods
        }
        for graph_variant in graph_variants
    }


def exp3_value_sufficiency(output_dir: Path, device, seed: int, hidden_dim: int, methods: list[str], n_per_conv: int, max_steps: int, gamma: float):
    return {
        method: evaluate_method_for_graph(
            output_dir, device, seed, "full_graph", method, hidden_dim, n_per_conv, max_steps, gamma
        )
        for method in methods
    }


def exp4_graph_robustness(output_dir: Path, device, seed: int, hidden_dim: int, method: str, graph_variants: list[str], n_per_conv: int, max_steps: int, gamma: float):
    return {
        graph_variant: evaluate_method_for_graph(
            output_dir, device, seed, graph_variant, method, hidden_dim, n_per_conv, max_steps, gamma
        )
        for graph_variant in graph_variants
    }


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
    parser.add_argument("--methods", type=str, default="aris_bellman,flat_latent,global_gru,oracle_belief,random_policy")
    parser.add_argument("--graph_variants", type=str, default=",".join(GRAPH_VARIANTS))
    parser.add_argument("--exp1_graph_variants", type=str, default="full_graph,plus_irrelevant")
    parser.add_argument("--n_per_conv", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.99)
    args = parser.parse_args()

    if args.n_per_conv <= 0:
        raise ValueError("--n_per_conv must be positive")
    if args.max_steps <= 0:
        raise ValueError("--max_steps must be positive")

    set_seed(args.seed)
    device = get_device()
    output_dir = Path(args.output_dir)
    experiments = [int(part) for part in args.experiments.split(",") if part.strip()]
    methods = parse_csv(args.methods, EVAL_METHODS)
    graph_variants = parse_csv(args.graph_variants, GRAPH_VARIANTS)
    exp1_graph_variants = parse_csv(args.exp1_graph_variants, GRAPH_VARIANTS)
    exp4_method = next((method for method in methods if method != "random_policy"), "aris_bellman")

    all_results = {}
    if 1 in experiments:
        all_results["exp1"] = exp1_policy_baselines(
            output_dir, device, args.seed, args.hidden_dim, methods, exp1_graph_variants,
            args.n_per_conv, args.max_steps, args.gamma,
        )
    if 3 in experiments:
        all_results["exp3"] = exp3_value_sufficiency(
            output_dir, device, args.seed, args.hidden_dim, methods,
            args.n_per_conv, args.max_steps, args.gamma,
        )
    if 4 in experiments:
        all_results["exp4"] = exp4_graph_robustness(
            output_dir, device, args.seed, args.hidden_dim, exp4_method, graph_variants,
            args.n_per_conv, args.max_steps, args.gamma,
        )

    results_path = output_dir / f"eval_results_seed{args.seed}.json"
    save_results(
        {
            "schema_version": "aris_bellman_eval_v1",
            "config": vars(args),
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
