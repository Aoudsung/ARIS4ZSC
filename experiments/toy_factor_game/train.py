"""
Training script for the toy interaction factor game.

The controller is trained at the macro-option level. Options are selected by
G-TVOI, MI, passive, random, or oracle scoring, then translated to primitive
environment actions through the option library.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.utils import get_device, save_results, set_seed
from toy_factor_game.env import ConventionAssignment, FACTOR_MODES, NUM_FACTORS, ToyFactorGameEnv
from toy_factor_game.graph_config import GRAPH_VARIANTS, GraphConfig, get_graph_config
from toy_factor_game.gtvoi import OptionSelector, belief_to_features
from toy_factor_game.options import (
    NUM_OPTIONS,
    OptionID,
    get_option_action,
    get_option_cost,
    get_valid_options,
)
from toy_factor_game.policy import ActiveFactorAgent


MODES = ("gtvoi", "mi", "passive", "random", "oracle")


def all_conventions() -> list[ConventionAssignment]:
    conventions = []
    for f0 in range(FACTOR_MODES[0]):
        for f1 in range(FACTOR_MODES[1]):
            for f2 in range(FACTOR_MODES[2]):
                conventions.append(ConventionAssignment(modes={0: f0, 1: f1, 2: f2}))
    return conventions


def one_hot_action(action: int, n_actions: int, device: torch.device) -> torch.Tensor:
    out = torch.zeros(n_actions, device=device)
    out[action] = 1.0
    return out


def initial_marginals(graph_config: GraphConfig, device: torch.device) -> list[torch.Tensor]:
    return [
        torch.ones(1, n_modes, device=device) / n_modes
        for n_modes in graph_config.factor_modes
    ]


def oracle_marginals(graph_config: GraphConfig, convention: dict[int, int], device: torch.device) -> list[torch.Tensor]:
    labels = graph_config.labels_from_convention(convention)
    out = []
    for label, n_modes in zip(labels, graph_config.factor_modes):
        probs = torch.zeros(1, n_modes, device=device)
        probs[0, label] = 1.0
        out.append(probs)
    return out


def valid_option_mask(env: ToyFactorGameEnv, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(1, NUM_OPTIONS, dtype=torch.bool, device=device)
    for option in get_valid_options(env):
        mask[0, int(option)] = True
    return mask


def option_costs(device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [[get_option_cost(OptionID(option_id)) for option_id in range(NUM_OPTIONS)]],
        dtype=torch.float32,
        device=device,
    )


def select_option(
    env: ToyFactorGameEnv,
    agent: ActiveFactorAgent,
    obs_t: torch.Tensor,
    marginals: list[torch.Tensor],
    graph_config: GraphConfig,
    device: torch.device,
    mode: str,
    deterministic: bool,
    temperature: float,
) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}; expected one of {MODES}")

    scoring_marginals = marginals
    if mode == "oracle":
        scoring_marginals = oracle_marginals(graph_config, env.partner_convention.modes, device)

    belief_features = belief_to_features(scoring_marginals)
    simulated_marginals, simulated_beliefs = agent.belief_transition.forward_all_options(
        belief_features
    )

    selector = OptionSelector(agent.q_net, agent.value_net, mode="gtvoi" if mode == "oracle" else mode)
    selected, log_prob, entropy = selector.select(
        obs_t,
        belief_features,
        simulated_beliefs=simulated_beliefs,
        option_costs=option_costs(device),
        marginals_before=scoring_marginals,
        simulated_marginals=simulated_marginals,
        valid_mask=valid_option_mask(env, device),
        deterministic=deterministic,
        temperature=temperature,
    )
    value = agent.value_net(obs_t, belief_features)
    return int(selected.item()), log_prob.squeeze(0), entropy.squeeze(0), value.squeeze(0), belief_features


def collect_episode(
    env: ToyFactorGameEnv,
    agent: ActiveFactorAgent,
    device: torch.device,
    mode: str = "gtvoi",
    graph_config: GraphConfig | None = None,
    explore_eps: float = 0.1,
    deterministic: bool = False,
    temperature: float = 1.0,
) -> dict:
    graph_config = graph_config or get_graph_config("full_graph")
    obs = env.reset()
    obs_history = []
    ego_act_history = []
    partner_act_history = []
    option_history = []
    primitive_action_history = []
    rewards = []
    infos = []
    option_log_probs = []
    value_predictions = []
    option_entropies = []
    policy_mask = []
    probe_costs = []

    for _t in range(env.max_steps):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        if obs_history:
            obs_seq = torch.stack(obs_history).unsqueeze(0)
            ego_seq = torch.stack(ego_act_history).unsqueeze(0)
            partner_seq = torch.stack(partner_act_history).unsqueeze(0)
            with torch.no_grad():
                marginals = agent.get_belief(obs_seq, ego_seq, partner_seq)
        else:
            marginals = initial_marginals(graph_config, device)

        explore = bool(np.random.random() < explore_eps)
        if explore:
            valid_options = get_valid_options(env)
            option = valid_options[int(np.random.randint(0, len(valid_options)))]
            option_id = int(option)
            belief_features = belief_to_features(marginals)
            log_prob = torch.zeros((), device=device)
            entropy = torch.zeros((), device=device)
            value = agent.value_net(obs_t, belief_features).squeeze(0)
            use_policy_loss = False
        else:
            option_id, log_prob, entropy, value, _belief_features = select_option(
                env,
                agent,
                obs_t,
                marginals,
                graph_config,
                device,
                mode,
                deterministic=deterministic,
                temperature=temperature,
            )
            use_policy_loss = mode != "random"

        action = get_option_action(OptionID(option_id), env.ego_pos, env.ego_carrying)
        next_obs, reward, done, info = env.step(action)

        obs_history.append(obs_t.squeeze(0))
        ego_act_history.append(one_hot_action(action, env.n_actions, device))
        partner_act_history.append(one_hot_action(info["partner_action"], env.n_actions, device))
        option_history.append(option_id)
        primitive_action_history.append(action)
        rewards.append(float(reward))
        infos.append(info)
        option_log_probs.append(log_prob)
        value_predictions.append(value)
        option_entropies.append(entropy)
        policy_mask.append(use_policy_loss)
        probe_costs.append(get_option_cost(OptionID(option_id)))

        obs = next_obs
        if done:
            break

    return {
        "obs_history": obs_history,
        "ego_act_history": ego_act_history,
        "partner_act_history": partner_act_history,
        "option_history": option_history,
        "primitive_action_history": primitive_action_history,
        "option_log_probs": option_log_probs,
        "value_predictions": value_predictions,
        "option_entropies": option_entropies,
        "policy_mask": policy_mask,
        "probe_costs": probe_costs,
        "rewards": rewards,
        "infos": infos,
    }


def discounted_returns(rewards: list[float], gamma: float, device: torch.device) -> torch.Tensor:
    returns = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        returns.append(running)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32, device=device)


def normalize_advantages(advantages: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if advantages.numel() <= 1:
        return advantages
    return (advantages - advantages.mean()) / (advantages.std(unbiased=False) + eps)


def batch_episodes(
    episodes: list[dict] | dict,
    device: torch.device,
    graph_config: GraphConfig,
    gamma: float,
) -> dict:
    if isinstance(episodes, dict):
        episodes = [episodes]
    usable = [episode for episode in episodes if len(episode["obs_history"]) >= 3]
    if not usable:
        return {}

    obs_prefixes, ego_prefixes, partner_prefixes = [], [], []
    obs_next_prefixes, ego_next_prefixes, partner_next_prefixes = [], [], []
    lengths, next_lengths, full_lengths = [], [], []
    obs_at_eval, value_targets, partner_next, factor_labels, option_at_split = [], [], [], [], []
    returns_seq, values_seq, log_prob_seq, entropy_seq, policy_mask_seq = [], [], [], [], []
    rewards_sum = []

    for episode in usable:
        obs_history = episode["obs_history"]
        ego_history = episode["ego_act_history"]
        partner_history = episode["partner_act_history"]
        rewards = episode["rewards"]
        infos = episode["infos"]
        t_steps = len(obs_history)
        split = max(1, t_steps // 2)

        obs_prefixes.append(torch.stack(obs_history[:split]))
        ego_prefixes.append(torch.stack(ego_history[:split]))
        partner_prefixes.append(torch.stack(partner_history[:split]))
        obs_next_prefixes.append(torch.stack(obs_history[:split + 1]))
        ego_next_prefixes.append(torch.stack(ego_history[:split + 1]))
        partner_next_prefixes.append(torch.stack(partner_history[:split + 1]))
        lengths.append(split)
        next_lengths.append(split + 1)
        full_lengths.append(t_steps)

        obs_at_eval.append(obs_history[split])
        returns_i = discounted_returns(rewards, gamma, device)
        value_targets.append(returns_i[split])
        partner_next.append(partner_history[split].argmax())
        factor_labels.append(graph_config.labels_from_convention(infos[0]["convention"]))
        option_at_split.append(episode["option_history"][split])

        returns_seq.append(returns_i)
        values_seq.append(torch.stack(episode["value_predictions"]))
        log_prob_seq.append(torch.stack(episode["option_log_probs"]))
        entropy_seq.append(torch.stack(episode["option_entropies"]))
        policy_mask_seq.append(torch.tensor(episode["policy_mask"], dtype=torch.bool, device=device))
        rewards_sum.append(float(sum(rewards)))

    valid_mask = torch.zeros(len(usable), max(full_lengths), dtype=torch.bool, device=device)
    for row, length in enumerate(full_lengths):
        valid_mask[row, :length] = True

    return {
        "obs_seq": pad_sequence(obs_prefixes, batch_first=True).to(device),
        "ego_seq": pad_sequence(ego_prefixes, batch_first=True).to(device),
        "partner_seq": pad_sequence(partner_prefixes, batch_first=True).to(device),
        "obs_seq_next": pad_sequence(obs_next_prefixes, batch_first=True).to(device),
        "ego_seq_next": pad_sequence(ego_next_prefixes, batch_first=True).to(device),
        "partner_seq_next": pad_sequence(partner_next_prefixes, batch_first=True).to(device),
        "lengths": torch.tensor(lengths, dtype=torch.long, device=device),
        "next_lengths": torch.tensor(next_lengths, dtype=torch.long, device=device),
        "obs_at_eval": torch.stack(obs_at_eval).to(device),
        "value_target": torch.stack(value_targets).to(device),
        "partner_next": torch.stack(partner_next).to(device),
        "factor_labels": torch.tensor(factor_labels, dtype=torch.long, device=device),
        "option_at_split": torch.tensor(option_at_split, dtype=torch.long, device=device),
        "returns": pad_sequence(returns_seq, batch_first=True).to(device),
        "values": pad_sequence(values_seq, batch_first=True).to(device),
        "log_probs": pad_sequence(log_prob_seq, batch_first=True).to(device),
        "entropies": pad_sequence(entropy_seq, batch_first=True).to(device),
        "policy_mask": pad_sequence(policy_mask_seq, batch_first=True).to(device),
        "valid_mask": valid_mask,
        "episode_reward": float(np.mean(rewards_sum)),
        "n_episodes": len(usable),
    }


def train_step(
    agent: ActiveFactorAgent,
    optimizer: torch.optim.Optimizer,
    episode_data: dict | list[dict],
    device: torch.device,
    graph_config: GraphConfig,
    loss_variant: str = "full",
    gamma: float = 0.99,
    entropy_coef: float = 0.01,
) -> dict[str, float]:
    batch = batch_episodes(episode_data, device, graph_config, gamma)
    if not batch:
        return {}
    sparsity_weights = torch.tensor(graph_config.sparsity_weights, dtype=torch.float32, device=device)

    if loss_variant == "response_only":
        beta_kl, gamma_sparsity, mu_cal = 0.01, 0.0, 0.0
        value_coef, critic_coef, policy_coef = 0.0, 0.0, 0.0
    elif loss_variant == "response_value":
        beta_kl, gamma_sparsity, mu_cal = 0.01, 0.0, 0.0
        value_coef, critic_coef, policy_coef = 1.0, 0.0, 0.0
    elif loss_variant == "full":
        beta_kl, gamma_sparsity, mu_cal = 0.01, 0.1, 0.5
        value_coef, critic_coef, policy_coef = 1.0, 1.0, 1.0
    else:
        raise ValueError("loss_variant must be one of response_only, response_value, full")

    factor_losses = agent.compute_factor_losses(
        batch["obs_seq"],
        batch["ego_seq"],
        batch["partner_seq"],
        batch["partner_next"],
        batch["factor_labels"],
        lengths=batch["lengths"],
        obs_at_eval=batch["obs_at_eval"],
        value_target=batch["value_target"],
        value_coef=value_coef,
        beta_kl=beta_kl,
        gamma_sparsity=gamma_sparsity,
        mu_cal=mu_cal,
        sparsity_weights=sparsity_weights,
        gt_mask=graph_config.ground_truth_mask,
        return_intermediates=True,
    )

    returns = batch["returns"]
    values = batch["values"]
    critic_value_loss = ((values - returns) ** 2)[batch["valid_mask"]].mean()

    with torch.no_grad():
        target_marginals = agent.belief_model(
            batch["obs_seq_next"],
            batch["ego_seq_next"],
            batch["partner_seq_next"],
            lengths=batch["next_lengths"],
        )
    current_belief_features = factor_losses["belief_features"]
    target_belief_features = belief_to_features(target_marginals)
    option_onehot = F.one_hot(
        batch["option_at_split"],
        num_classes=NUM_OPTIONS,
    ).float()
    predicted_belief_features = agent.belief_transition(current_belief_features.detach(), option_onehot)
    transition_loss = F.mse_loss(predicted_belief_features, target_belief_features)

    policy_mask = batch["policy_mask"] & batch["valid_mask"]
    if policy_mask.any():
        log_probs = batch["log_probs"][policy_mask]
        entropies = batch["entropies"][policy_mask]
        advantages = (returns - values.detach())[policy_mask]
        advantages = normalize_advantages(advantages)
        policy_loss = -(log_probs * advantages).mean()
        entropy_bonus = entropies.mean()
    else:
        policy_loss = torch.zeros((), device=device)
        entropy_bonus = torch.zeros((), device=device)

    total = (
        factor_losses["total"]
        + critic_coef * critic_value_loss
        + transition_loss
        + policy_coef * policy_loss
        - policy_coef * entropy_coef * entropy_bonus
    )

    optimizer.zero_grad()
    total.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
    optimizer.step()

    out = {
        k: float(v.detach().item())
        for k, v in factor_losses.items()
        if isinstance(v, torch.Tensor) and v.ndim == 0
    }
    out.update(
        {
            "total": float(total.detach().item()),
            "value": float(factor_losses["belief_value"].detach().item()),
            "critic_value": float(critic_value_loss.detach().item()),
            "transition": float(transition_loss.detach().item()),
            "control": float(policy_loss.detach().item()),
            "entropy": float(entropy_bonus.detach().item()),
            "episode_reward": batch["episode_reward"],
            "batch_size": batch["n_episodes"],
        }
    )
    return out


def evaluate(agent, device, graph_config: GraphConfig, mode: str, n_episodes=50, seed=0, max_steps=50):
    results = defaultdict(list)
    conventions = all_conventions()

    for ep in range(n_episodes):
        conv = conventions[ep % len(conventions)]
        env = ToyFactorGameEnv(partner_convention=conv, max_steps=max_steps, seed=seed + ep)
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
        infos = episode_data["infos"]
        results["episode_reward"].append(sum(rewards))
        results["collisions"].append(sum(1 for info in infos if info["collision"]))
        results["probe_cost"].append(sum(episode_data["probe_costs"]))

        if len(episode_data["obs_history"]) > 3:
            split = len(episode_data["obs_history"]) // 2
            obs_seq = torch.stack(episode_data["obs_history"][:split]).unsqueeze(0).to(device)
            ego_seq = torch.stack(episode_data["ego_act_history"][:split]).unsqueeze(0).to(device)
            partner_seq = torch.stack(episode_data["partner_act_history"][:split]).unsqueeze(0).to(device)
            with torch.no_grad():
                marginals = agent.get_belief(obs_seq, ego_seq, partner_seq)

            labels = graph_config.labels_from_convention(conv.modes)
            for factor_idx, label in enumerate(labels):
                if not graph_config.ground_truth_mask[factor_idx]:
                    continue
                pred = marginals[factor_idx][0].argmax().item()
                results[f"factor_{factor_idx}_accuracy"].append(int(pred == label))
                results[f"factor_{factor_idx}_confidence"].append(marginals[factor_idx][0].max().item())

    return {k: float(np.mean(v)) for k, v in results.items() if v}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_episodes", type=int, default=8000)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--loss_variant", type=str, default="full",
                        choices=["full", "response_only", "response_value"])
    parser.add_argument("--graph_variant", type=str, default="full_graph", choices=GRAPH_VARIANTS)
    parser.add_argument("--mode", type=str, default="gtvoi", choices=MODES)
    parser.add_argument("--explore_eps", type=float, default=0.15)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--control_temperature", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="results/toy")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")

    set_seed(args.seed)
    device = get_device()
    graph_config = get_graph_config(args.graph_variant)
    print(
        f"Device: {device}, Seed: {args.seed}, Loss: {args.loss_variant}, "
        f"Graph: {args.graph_variant}, Mode: {args.mode}"
    )

    env = ToyFactorGameEnv(max_steps=args.max_steps, seed=args.seed)
    agent = ActiveFactorAgent(
        obs_dim=env.obs_dim,
        n_actions=env.n_actions,
        n_options=NUM_OPTIONS,
        n_factors=graph_config.n_factors,
        factor_modes=graph_config.factor_modes,
        hidden_dim=args.hidden_dim,
        pairwise_pairs=list(graph_config.pairwise_pairs),
    ).to(device)

    optimizer = Adam(agent.parameters(), lr=args.lr)
    train_log = []
    eval_log = []
    episode_buffer = []

    def flush_episode_buffer(episode_idx: int):
        if not episode_buffer:
            return
        losses = train_step(
            agent,
            optimizer,
            list(episode_buffer),
            device,
            graph_config,
            args.loss_variant,
            gamma=args.gamma,
            entropy_coef=args.entropy_coef,
        )
        episode_buffer.clear()
        if losses:
            train_log.append({"episode": episode_idx, **losses})

    for ep in range(args.n_episodes):
        conv = ConventionAssignment(
            modes={f: np.random.randint(0, FACTOR_MODES[f]) for f in range(NUM_FACTORS)}
        )
        env = ToyFactorGameEnv(partner_convention=conv, max_steps=args.max_steps, seed=args.seed + ep)
        episode_data = collect_episode(
            env,
            agent,
            device,
            mode=args.mode,
            graph_config=graph_config,
            explore_eps=args.explore_eps,
            deterministic=False,
            temperature=args.control_temperature,
        )
        episode_buffer.append(episode_data)
        if len(episode_buffer) >= args.batch_size:
            flush_episode_buffer(ep + 1)

        if (ep + 1) % args.eval_every == 0:
            flush_episode_buffer(ep + 1)
            eval_results = evaluate(
                agent,
                device,
                graph_config,
                args.mode,
                n_episodes=50,
                seed=args.seed + 10000,
                max_steps=args.max_steps,
            )
            eval_results["episode"] = ep + 1
            eval_log.append(eval_results)
            print(
                f"  Ep {ep+1}: reward={eval_results.get('episode_reward', 0):.2f}, "
                f"collisions={eval_results.get('collisions', 0):.2f}, "
                f"probe_cost={eval_results.get('probe_cost', 0):.2f}"
            )

    flush_episode_buffer(args.n_episodes)

    output_dir = Path(args.output_dir) / f"seed{args.seed}" / args.graph_variant / args.loss_variant
    output_dir.mkdir(parents=True, exist_ok=True)
    save_results(
        {
            "config": vars(args),
            "graph": {
                "name": graph_config.name,
                "n_factors": graph_config.n_factors,
                "factor_modes": graph_config.factor_modes,
            },
            "train": train_log,
            "eval": eval_log,
        },
        str(output_dir / "results.json"),
    )
    torch.save(agent.state_dict(), str(output_dir / "model.pt"))
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
