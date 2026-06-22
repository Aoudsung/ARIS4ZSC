"""
Evaluation for the toy factor game.

Exp 1 compares deployment-time option selection modes on one trained model.
Exp 4 compares independently trained graph-variant models.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.metrics import bootstrap_ci, expected_calibration_error, pareto_frontier
from common.utils import get_device, save_results, set_seed
from toy_factor_game.env import NUM_FACTORS, ConventionAssignment, ToyFactorGameEnv
from toy_factor_game.graph_config import GRAPH_VARIANTS, GraphConfig, get_graph_config, stable_convention_seed
from toy_factor_game.options import NUM_OPTIONS
from toy_factor_game.policy import ActiveFactorAgent
from toy_factor_game.gtvoi import belief_to_features
from toy_factor_game.train import MODES, all_conventions, collect_episode


def model_path_for(output_dir: Path, seed: int, graph_variant: str, loss_variant: str) -> Path:
    return output_dir / f"seed{seed}" / graph_variant / loss_variant / "model.pt"


def load_state_dict_file(model_path: Path, device):
    try:
        return torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(model_path, map_location=device)


def load_agent(model_path: Path, device, graph_config: GraphConfig, hidden_dim: int):
    env = ToyFactorGameEnv()
    agent = ActiveFactorAgent(
        obs_dim=env.obs_dim,
        n_actions=env.n_actions,
        n_options=NUM_OPTIONS,
        n_factors=graph_config.n_factors,
        factor_modes=graph_config.factor_modes,
        hidden_dim=hidden_dim,
        pairwise_pairs=list(graph_config.pairwise_pairs),
    ).to(device)
    agent.load_state_dict(load_state_dict_file(model_path, device))
    agent.eval()
    return agent


def episode_diagnostics(agent, device, episode_data: dict, conv: ConventionAssignment, graph_config: GraphConfig):
    if len(episode_data["obs_history"]) < 3:
        return {
            "factor_probs": [],
            "factor_correct": [],
            "partner_pred_correct": None,
            "posterior_entropy": None,
            "per_factor_entropy": [],
            "value_gap_abs": None,
        }
    split = max(1, len(episode_data["obs_history"]) // 2)
    obs_seq = torch.stack(episode_data["obs_history"][:split]).unsqueeze(0).to(device)
    ego_seq = torch.stack(episode_data["ego_act_history"][:split]).unsqueeze(0).to(device)
    partner_seq = torch.stack(episode_data["partner_act_history"][:split]).unsqueeze(0).to(device)
    with torch.no_grad():
        h = agent.belief_model.encode_history(obs_seq, ego_seq, partner_seq)
        marginals = agent.belief_model._marginals_from_h(h)
        response_logits = agent.response_predictor(h)
        belief_features = belief_to_features(marginals)
        value_pred = agent.value_net(
            episode_data["obs_history"][split].unsqueeze(0).to(device),
            belief_features,
        )

    probs = []
    labels = []
    factor_labels = graph_config.labels_from_convention(conv.modes)
    for factor_idx, label in enumerate(factor_labels):
        if not graph_config.ground_truth_mask[factor_idx]:
            continue
        pred = marginals[factor_idx][0].argmax().item()
        probs.append(float(marginals[factor_idx][0].max().item()))
        labels.append(int(pred == label))

    entropies = agent.belief_model.get_entropy(marginals)[0]
    gt_entropies = [
        float(entropies[i].item())
        for i, is_gt in enumerate(graph_config.ground_truth_mask)
        if is_gt
    ]
    partner_pred = response_logits.argmax(dim=-1)
    actual_partner = episode_data["partner_act_history"][split].argmax().to(device)
    future_return = float(sum(episode_data["rewards"][split:]))
    return {
        "factor_probs": probs,
        "factor_correct": labels,
        "partner_pred_correct": int(partner_pred.item() == actual_partner.item()),
        "posterior_entropy": float(np.mean(gt_entropies)) if gt_entropies else None,
        "per_factor_entropy": gt_entropies,
        "value_gap_abs": abs(float(value_pred.item()) - future_return),
    }


def training_efficiency(output_dir: Path, seed: int, graph_variant: str, loss_variant: str) -> dict:
    path = output_dir / f"seed{seed}" / graph_variant / loss_variant / "results.json"
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


def summarize_rows(rows: list[dict], oracle_rows: dict[tuple[str, int], dict] | None = None) -> dict:
    rewards = np.array([row["episode_reward"] for row in rows], dtype=np.float64)
    early_rewards = np.array([row["early_reward"] for row in rows], dtype=np.float64)
    probe_costs = np.array([row["probe_cost"] for row in rows], dtype=np.float64)
    alignments = np.array([row["time_to_alignment"] for row in rows], dtype=np.float64)
    collisions = np.array([row["collisions"] for row in rows], dtype=np.float64)
    factor_probs = np.array([p for row in rows for p in row["factor_probs"]], dtype=np.float64)
    factor_correct = np.array([c for row in rows for c in row["factor_correct"]], dtype=np.float64)
    partner_pred = np.array(
        [row["partner_pred_correct"] for row in rows if row["partner_pred_correct"] is not None],
        dtype=np.float64,
    )
    posterior_entropy = np.array(
        [row["posterior_entropy"] for row in rows if row["posterior_entropy"] is not None],
        dtype=np.float64,
    )
    value_gaps = np.array(
        [row["value_gap_abs"] for row in rows if row["value_gap_abs"] is not None],
        dtype=np.float64,
    )

    summary = {
        "status": "done",
        "n": len(rows),
        "episode_reward_mean": float(rewards.mean()),
        "episode_reward_ci": list(bootstrap_ci(rewards)),
        "early_reward_mean": float(early_rewards.mean()),
        "probe_cost_mean": float(probe_costs.mean()),
        "time_to_alignment_mean": float(alignments.mean()),
        "collisions_mean": float(collisions.mean()),
        "factor_accuracy_mean": float(factor_correct.mean()) if len(factor_correct) else 0.0,
        "ece": expected_calibration_error(factor_probs, factor_correct),
        "partner_pred_accuracy": float(partner_pred.mean()) if len(partner_pred) else None,
        "posterior_entropy_mean": float(posterior_entropy.mean()) if len(posterior_entropy) else None,
        "value_gap_entropy_corr": None,
    }
    if len(value_gaps) > 1 and len(posterior_entropy) == len(value_gaps):
        if np.std(value_gaps) > 0 and np.std(posterior_entropy) > 0:
            summary["value_gap_entropy_corr"] = float(np.corrcoef(value_gaps, posterior_entropy)[0, 1])

    if oracle_rows is not None:
        regrets = []
        early_regrets = []
        for row in rows:
            oracle = oracle_rows[(row["convention"], row["trial"])]
            regrets.append(oracle["episode_reward"] - row["episode_reward"])
            early_regrets.append(oracle["early_reward"] - row["early_reward"])
        summary["regret_to_oracle_mean"] = float(np.mean(regrets))
        summary["early_regret_to_oracle_mean"] = float(np.mean(early_regrets))
    return summary


def real_factor_belief_aligned(
    marginals: list[torch.Tensor],
    conv: ConventionAssignment,
    graph_config: GraphConfig,
) -> bool:
    checked = False
    for factor_idx, factor in enumerate(graph_config.factors):
        if factor.env_factor_id is None:
            continue
        checked = True
        label = int(conv.modes[factor.env_factor_id])
        pred = int(marginals[factor_idx][0].argmax().item())
        if pred != label:
            return False
    return checked


def first_alignment_time(
    agent,
    device,
    episode_data: dict,
    conv: ConventionAssignment,
    graph_config: GraphConfig,
    max_steps: int,
) -> int:
    hidden = None
    for idx in range(len(episode_data["obs_history"])):
        with torch.no_grad():
            hidden = agent.belief_model.step_history(
                episode_data["obs_history"][idx].to(device),
                episode_data["ego_act_history"][idx].to(device),
                episode_data["partner_act_history"][idx].to(device),
                hidden,
            )
            marginals = agent.belief_model._marginals_from_h(hidden)
        if real_factor_belief_aligned(marginals, conv, graph_config):
            return idx + 1
    return max_steps + 1


def evaluate_mode(agent, device, graph_config: GraphConfig, mode: str, seed: int, n_per_conv: int, max_steps: int):
    rows = []
    for conv in all_conventions():
        convention_key = "-".join(str(conv.modes[f]) for f in range(NUM_FACTORS))
        for trial in range(n_per_conv):
            env_seed = stable_convention_seed(seed, conv.modes, trial)
            env = ToyFactorGameEnv(partner_convention=conv, max_steps=max_steps, seed=env_seed)
            with torch.no_grad():
                episode_data = collect_episode(
                    env,
                    agent,
                    device,
                    mode=mode,
                    graph_config=graph_config,
                    explore_eps=0.0,
                    deterministic=True,
                )
            rewards = episode_data["rewards"]
            diagnostics = episode_diagnostics(agent, device, episode_data, conv, graph_config)
            early_k = max(1, len(rewards) // 5)
            rows.append(
                {
                    "convention": convention_key,
                    "trial": trial,
                    "episode_reward": float(sum(rewards)),
                    "early_reward": float(sum(rewards[:early_k])),
                    "probe_cost": float(sum(episode_data["probe_costs"])),
                    "time_to_alignment": first_alignment_time(
                        agent, device, episode_data, conv, graph_config, max_steps
                    ),
                    "collisions": float(sum(1 for info in episode_data["infos"] if info["collision"])),
                    **diagnostics,
                }
            )
    return rows


def evaluate_modes_for_graph(output_dir: Path, device, seed: int, graph_variant: str, loss_variant: str, hidden_dim: int, modes: list[str], n_per_conv: int, max_steps: int):
    graph_config = get_graph_config(graph_variant)
    path = model_path_for(output_dir, seed, graph_variant, loss_variant)
    if not path.exists():
        return {"status": "missing", "model_path": str(path)}
    agent = load_agent(path, device, graph_config, hidden_dim)
    oracle_rows_list = evaluate_mode(agent, device, graph_config, "oracle", seed, n_per_conv, max_steps)
    oracle_rows = {(row["convention"], row["trial"]): row for row in oracle_rows_list}
    efficiency = training_efficiency(output_dir, seed, graph_variant, loss_variant)
    results = {}
    for mode in modes:
        rows = oracle_rows_list if mode == "oracle" else evaluate_mode(
            agent, device, graph_config, mode, seed, n_per_conv, max_steps
        )
        results[mode] = summarize_rows(rows, oracle_rows)
        results[mode].update(efficiency)

    costs = np.array([results[mode]["probe_cost_mean"] for mode in modes], dtype=np.float64)
    rewards = np.array([results[mode]["episode_reward_mean"] for mode in modes], dtype=np.float64)
    frontier_idx = pareto_frontier(costs, rewards)
    results["reward_probe_cost_pareto_modes"] = [modes[int(idx)] for idx in frontier_idx]
    return results


def exp1_gtvoi_vs_mi(output_dir: Path, device, seed: int, loss_variant: str, hidden_dim: int, modes: list[str], n_per_conv: int, max_steps: int, graph_variants: list[str]):
    return {
        graph_variant: evaluate_modes_for_graph(
            output_dir, device, seed, graph_variant, loss_variant, hidden_dim, modes, n_per_conv, max_steps
        )
        for graph_variant in graph_variants
    }


def exp3_value_sufficiency(output_dir: Path, device, seed: int, hidden_dim: int, n_per_conv: int, max_steps: int):
    graph_config = get_graph_config("full_graph")
    results = {}
    for loss_variant in ["response_only", "response_value", "full"]:
        path = model_path_for(output_dir, seed, "full_graph", loss_variant)
        if not path.exists():
            results[loss_variant] = {"status": "missing", "model_path": str(path)}
            continue
        agent = load_agent(path, device, graph_config, hidden_dim)
        rows = evaluate_mode(agent, device, graph_config, "gtvoi", seed, n_per_conv, max_steps)
        results[loss_variant] = summarize_rows(rows)
        results[loss_variant].update(training_efficiency(output_dir, seed, "full_graph", loss_variant))
    return results


def exp4_graph_robustness(output_dir: Path, device, seed: int, loss_variant: str, hidden_dim: int, graph_variants: list[str], n_per_conv: int, max_steps: int):
    results = {}
    for graph_variant in graph_variants:
        graph_config = get_graph_config(graph_variant)
        path = model_path_for(output_dir, seed, graph_variant, loss_variant)
        if not path.exists():
            results[graph_variant] = {"status": "missing", "model_path": str(path)}
            continue
        agent = load_agent(path, device, graph_config, hidden_dim)
        rows = evaluate_mode(agent, device, graph_config, "gtvoi", seed, n_per_conv, max_steps)
        results[graph_variant] = summarize_rows(rows)
        results[graph_variant].update(training_efficiency(output_dir, seed, graph_variant, loss_variant))
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
    parser.add_argument("--experiments", type=str, default="1,3,4",
                        help="Comma-separated experiment numbers to run")
    parser.add_argument("--loss_variant", type=str, default="full",
                        choices=["full", "response_only", "response_value"])
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--modes", type=str, default="gtvoi,mi,passive,random,oracle")
    parser.add_argument("--exp1_graph_variants", type=str, default="full_graph,plus_irrelevant")
    parser.add_argument("--graph_variants", type=str, default=",".join(GRAPH_VARIANTS))
    parser.add_argument("--n_per_conv", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=50)
    args = parser.parse_args()

    if args.n_per_conv <= 0:
        raise ValueError("--n_per_conv must be positive")
    if args.max_steps <= 0:
        raise ValueError("--max_steps must be positive")

    set_seed(args.seed)
    device = get_device()
    output_dir = Path(args.output_dir)
    exps = [int(x) for x in args.experiments.split(",") if x.strip()]
    modes = parse_csv(args.modes, MODES)
    exp1_graph_variants = parse_csv(args.exp1_graph_variants, GRAPH_VARIANTS)
    graph_variants = parse_csv(args.graph_variants, GRAPH_VARIANTS)

    all_results = {}
    if 1 in exps:
        all_results["exp1"] = exp1_gtvoi_vs_mi(
            output_dir, device, args.seed, args.loss_variant, args.hidden_dim,
            modes, args.n_per_conv, args.max_steps, exp1_graph_variants
        )
    if 3 in exps:
        all_results["exp3"] = exp3_value_sufficiency(
            output_dir, device, args.seed, args.hidden_dim, args.n_per_conv, args.max_steps
        )
    if 4 in exps:
        all_results["exp4"] = exp4_graph_robustness(
            output_dir, device, args.seed, args.loss_variant, args.hidden_dim, graph_variants, args.n_per_conv, args.max_steps
        )

    results_path = output_dir / f"eval_results_seed{args.seed}.json"
    save_results(
        {
            "config": vars(args),
            "results": all_results,
            "ground_truth_note": (
                "Factor accuracy and ECE are computed against ToyFactorGameEnv ConventionAssignment labels, "
                "not another model's outputs. Synthetic graph factors are excluded from these metrics."
            ),
            "oracle_note": "oracle mode uses perfect belief with the same Q/cost scoring, i.e. perfect_belief_q_only.",
        },
        str(results_path),
    )
    print(f"All results saved to {results_path}")


if __name__ == "__main__":
    main()
