"""ARIS-Bellman training for the toy interaction factor game."""

from __future__ import annotations

import argparse
import copy
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.utils import get_device, save_results, set_seed
from toy_factor_game.env import ConventionAssignment, FACTOR_MODES, NUM_FACTORS, ToyFactorGameEnv
from toy_factor_game.evidence import route_event_to_factors
from toy_factor_game.graph_config import GRAPH_VARIANTS, GraphConfig, get_graph_config
from toy_factor_game.options import (
    NUM_OPTIONS,
    OptionID,
    get_option_action,
    get_option_cost,
    get_valid_options,
)
from toy_factor_game.policy import METHODS, ORACLE_BELIEF_METHODS, ActiveFactorAgent


TRAIN_METHODS = tuple(method for method in METHODS if method != "random_policy")
EXPERIMENT_SCHEMA = "aris_bellman_v4.1"
PROPOSAL_VERSION = "v4"
CODE_FIX_LEVEL = "ce-all-conventions-criticality-diagnostics"


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


def valid_option_mask_from_options(options: list[OptionID], device: torch.device) -> torch.Tensor:
    mask = torch.zeros(NUM_OPTIONS, dtype=torch.bool, device=device)
    for option in options:
        mask[int(option)] = True
    return mask


def valid_option_mask(env: ToyFactorGameEnv, device: torch.device) -> torch.Tensor:
    return valid_option_mask_from_options(get_valid_options(env), device)


def oracle_marginals(
    graph_config: GraphConfig,
    convention: dict[int, int],
    batch_size: int,
    device: torch.device,
) -> list[torch.Tensor]:
    marginals = []
    for factor, n_modes in zip(graph_config.factors, graph_config.factor_modes):
        if factor.env_factor_id is None:
            marginals.append(torch.ones(batch_size, n_modes, device=device) / float(n_modes))
            continue
        label = int(convention[factor.env_factor_id])
        labels = torch.full((batch_size,), label, dtype=torch.long, device=device)
        marginals.append(F.one_hot(labels, num_classes=n_modes).float())
    return marginals


def labels_to_oracle_marginals(
    factor_labels: torch.Tensor,
    graph_config: GraphConfig,
) -> list[torch.Tensor]:
    marginals = []
    batch_size = factor_labels.shape[0]
    device = factor_labels.device
    for factor_idx, (factor, n_modes) in enumerate(zip(graph_config.factors, graph_config.factor_modes)):
        if factor.env_factor_id is None:
            marginals.append(torch.ones(batch_size, n_modes, device=device) / float(n_modes))
            continue
        labels = factor_labels[:, factor_idx].clamp(min=0, max=n_modes - 1)
        marginals.append(F.one_hot(labels, num_classes=n_modes).float())
    return marginals


def _masked_q(q_values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    return q_values.masked_fill(~mask.bool(), -1e9)


def select_option(
    agent: ActiveFactorAgent,
    obs_t: torch.Tensor,
    marginals: list[torch.Tensor] | None,
    global_hidden: torch.Tensor | None,
    valid_mask: torch.Tensor,
    explore_eps: float,
    deterministic: bool,
) -> int:
    valid_ids = torch.nonzero(valid_mask, as_tuple=False).flatten()
    if agent.method == "random_policy" or (not deterministic and np.random.random() < explore_eps):
        idx = int(np.random.randint(0, len(valid_ids)))
        return int(valid_ids[idx].item())

    q_values = agent.q_values(obs_t, marginals=marginals, global_hidden=global_hidden)
    scores = _masked_q(q_values, valid_mask)
    if deterministic:
        return int(scores.argmax(dim=-1).item())
    dist = torch.distributions.Categorical(logits=scores)
    return int(dist.sample().item())


def collect_episode(
    env: ToyFactorGameEnv,
    agent: ActiveFactorAgent,
    device: torch.device,
    graph_config: GraphConfig,
    explore_eps: float = 0.1,
    deterministic: bool = False,
) -> dict:
    obs = env.reset()
    factor_hidden = None
    global_hidden = None
    transitions = defaultdict(list)

    for _t in range(env.max_steps):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        mask_t = valid_option_mask(env, device)

        if agent.method in ORACLE_BELIEF_METHODS:
            marginals = oracle_marginals(graph_config, env.partner_convention.modes, 1, device)
        elif agent.method == "global_gru":
            marginals = None
        else:
            if factor_hidden is None:
                factor_hidden = agent.belief_model.initial_hidden(1, device)
            marginals = agent.belief_model._marginals_from_h(factor_hidden)

        option_id = select_option(
            agent,
            obs_t,
            marginals,
            global_hidden,
            mask_t,
            explore_eps=explore_eps,
            deterministic=deterministic,
        )
        action = get_option_action(OptionID(option_id), env.ego_pos, env.ego_carrying)
        next_obs, env_reward, done, info = env.step(action)
        event = dict(info["event"])
        event["ego_option"] = option_id
        evidence = torch.tensor(
            route_event_to_factors(event, graph_config),
            dtype=torch.float32,
            device=device,
        )
        ego_onehot = one_hot_action(action, env.n_actions, device)
        partner_onehot = one_hot_action(info["partner_action"], env.n_actions, device)
        next_mask = valid_option_mask(env, device)

        transitions["obs"].append(obs_t.squeeze(0))
        transitions["next_obs"].append(torch.tensor(next_obs, dtype=torch.float32, device=device))
        transitions["evidence"].append(evidence)
        transitions["ego_actions"].append(ego_onehot)
        transitions["partner_actions"].append(partner_onehot)
        transitions["options"].append(option_id)
        transitions["rewards"].append(float(env_reward - get_option_cost(OptionID(option_id))))
        transitions["env_rewards"].append(float(env_reward))
        transitions["done"].append(bool(done))
        transitions["valid_masks"].append(mask_t)
        transitions["next_valid_masks"].append(next_mask)
        transitions["infos"].append(info)

        with torch.no_grad():
            if factor_hidden is None:
                factor_hidden = agent.belief_model.initial_hidden(1, device)
            factor_hidden = agent.belief_model.step_history(evidence, factor_hidden)
            if agent.method == "global_gru":
                global_hidden = agent.global_q.step_history(obs_t, ego_onehot, partner_onehot, global_hidden)

        obs = next_obs
        if done:
            break

    transitions["factor_labels"] = graph_config.labels_from_convention(env.partner_convention.modes)
    return dict(transitions)


def batch_episodes(
    episodes: list[dict] | dict,
    device: torch.device,
    graph_config: GraphConfig,
) -> dict:
    if isinstance(episodes, dict):
        episodes = [episodes]
    usable = [episode for episode in episodes if episode.get("obs")]
    if not usable:
        return {}

    keys_to_pad = [
        "obs",
        "next_obs",
        "evidence",
        "ego_actions",
        "partner_actions",
        "valid_masks",
        "next_valid_masks",
    ]
    batch = {
        key: pad_sequence([torch.stack(ep[key]) for ep in usable], batch_first=True).to(device)
        for key in keys_to_pad
    }
    batch["options"] = pad_sequence(
        [torch.tensor(ep["options"], dtype=torch.long, device=device) for ep in usable],
        batch_first=True,
    ).to(device)
    batch["rewards"] = pad_sequence(
        [torch.tensor(ep["rewards"], dtype=torch.float32, device=device) for ep in usable],
        batch_first=True,
    ).to(device)
    batch["env_rewards"] = pad_sequence(
        [torch.tensor(ep["env_rewards"], dtype=torch.float32, device=device) for ep in usable],
        batch_first=True,
    ).to(device)
    batch["done"] = pad_sequence(
        [torch.tensor(ep["done"], dtype=torch.bool, device=device) for ep in usable],
        batch_first=True,
    ).to(device)
    lengths = torch.tensor([len(ep["obs"]) for ep in usable], dtype=torch.long, device=device)
    valid = torch.zeros(len(usable), int(lengths.max().item()), dtype=torch.bool, device=device)
    for row, length in enumerate(lengths.tolist()):
        valid[row, :length] = True
    batch["lengths"] = lengths
    batch["step_mask"] = valid
    batch["factor_labels"] = torch.tensor(
        [ep["factor_labels"] for ep in usable],
        dtype=torch.long,
        device=device,
    )
    batch["episode_reward"] = float(np.mean([sum(ep["env_rewards"]) for ep in usable]))
    batch["n_episodes"] = len(usable)
    return batch


def _blend_hidden(next_hidden: torch.Tensor, hidden: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    if hidden.dim() == 3 and hidden.shape[0] == 1:
        mask_shape = [1, active.shape[0], 1]
    else:
        mask_shape = [active.shape[0]] + [1] * (hidden.dim() - 1)
    mask = active.to(dtype=hidden.dtype, device=hidden.device).view(*mask_shape)
    return next_hidden * mask + hidden * (1.0 - mask)


def _current_marginals(
    agent: ActiveFactorAgent,
    graph_config: GraphConfig,
    factor_hidden: torch.Tensor,
    factor_labels: torch.Tensor,
) -> list[torch.Tensor] | None:
    if agent.method in ORACLE_BELIEF_METHODS:
        return labels_to_oracle_marginals(factor_labels, graph_config)
    if agent.method == "global_gru":
        return None
    return agent.belief_model._marginals_from_h(factor_hidden)


def train_step(
    agent: ActiveFactorAgent,
    target_agent: ActiveFactorAgent,
    optimizer: torch.optim.Optimizer,
    episode_data: dict | list[dict],
    device: torch.device,
    graph_config: GraphConfig,
    gamma: float = 0.99,
) -> dict[str, float]:
    batch = batch_episodes(episode_data, device, graph_config)
    if not batch:
        return {}

    batch_size, max_time = batch["obs"].shape[:2]
    factor_hidden = agent.belief_model.initial_hidden(batch_size, device)
    target_factor_hidden = target_agent.belief_model.initial_hidden(batch_size, device)
    global_hidden = agent.global_q.initial_hidden(batch_size, device)
    target_global_hidden = target_agent.global_q.initial_hidden(batch_size, device)
    td_losses = []
    q_pred_values = []
    q_target_values = []

    for t in range(max_time):
        active = batch["step_mask"][:, t]
        if not active.any():
            continue

        marginals = _current_marginals(agent, graph_config, factor_hidden, batch["factor_labels"])
        q_values = agent.q_values(batch["obs"][:, t], marginals=marginals, global_hidden=global_hidden)
        q_pred = q_values.gather(1, batch["options"][:, t].unsqueeze(1)).squeeze(1)

        next_factor_hidden = agent.belief_model.step_history(
            batch["evidence"][:, t], factor_hidden, active_mask=active
        )
        if agent.method == "global_gru":
            next_global_hidden_raw = agent.global_q.step_history(
                batch["obs"][:, t],
                batch["ego_actions"][:, t],
                batch["partner_actions"][:, t],
                global_hidden,
            )
            next_global_hidden = _blend_hidden(next_global_hidden_raw, global_hidden, active)
        else:
            next_global_hidden = global_hidden

        with torch.no_grad():
            target_next_factor_hidden = target_agent.belief_model.step_history(
                batch["evidence"][:, t], target_factor_hidden, active_mask=active
            )
            if target_agent.method == "global_gru":
                target_next_global_raw = target_agent.global_q.step_history(
                    batch["obs"][:, t],
                    batch["ego_actions"][:, t],
                    batch["partner_actions"][:, t],
                    target_global_hidden,
                )
                target_next_global = _blend_hidden(target_next_global_raw, target_global_hidden, active)
            else:
                target_next_global = target_global_hidden
            target_marginals = _current_marginals(
                target_agent, graph_config, target_next_factor_hidden, batch["factor_labels"]
            )
            target_q = target_agent.q_values(
                batch["next_obs"][:, t],
                marginals=target_marginals,
                global_hidden=target_next_global,
            )
            target_q = target_q.masked_fill(~batch["next_valid_masks"][:, t].bool(), -1e9)
            bootstrapped = target_q.max(dim=-1).values
            td_target = batch["rewards"][:, t] + gamma * (~batch["done"][:, t]).float() * bootstrapped

        td_losses.append(F.mse_loss(q_pred[active], td_target[active]))
        q_pred_values.append(q_pred[active].detach())
        q_target_values.append(td_target[active].detach())

        factor_hidden = next_factor_hidden
        target_factor_hidden = target_next_factor_hidden.detach()
        global_hidden = next_global_hidden
        target_global_hidden = target_next_global.detach()

    if not td_losses:
        return {}

    td_loss = torch.stack(td_losses).mean()
    optimizer.zero_grad()
    td_loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
    optimizer.step()

    q_pred_all = torch.cat(q_pred_values)
    q_target_all = torch.cat(q_target_values)
    return {
        "total": float(td_loss.detach().item()),
        "td": float(td_loss.detach().item()),
        "q_pred_mean": float(q_pred_all.mean().item()),
        "q_target_mean": float(q_target_all.mean().item()),
        "episode_reward": batch["episode_reward"],
        "batch_size": batch["n_episodes"],
    }


def evaluate(
    agent: ActiveFactorAgent,
    device,
    graph_config: GraphConfig,
    n_episodes: int = 50,
    seed: int = 0,
    max_steps: int = 50,
) -> dict[str, float]:
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
                graph_config=graph_config,
                explore_eps=0.0,
                deterministic=True,
            )
        infos = episode_data["infos"]
        results["episode_reward"].append(sum(episode_data["env_rewards"]))
        results["td_reward"].append(sum(episode_data["rewards"]))
        results["collisions"].append(sum(1 for info in infos if info["collision"]))
        results["completion_rate"].append(float(any(info.get("completed", False) for info in infos)))
        results["time_to_completion"].append(len(episode_data["obs"]) if results["completion_rate"][-1] else max_steps + 1)
    return {key: float(np.mean(values)) for key, values in results.items() if values}


def model_output_dir(output_dir: Path, seed: int, method: str, graph_variant: str) -> Path:
    return output_dir / f"seed{seed}" / method / graph_variant


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_episodes", type=int, default=8000)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--method", type=str, default="aris_bellman", choices=TRAIN_METHODS)
    parser.add_argument("--graph_variant", type=str, default="full_support", choices=GRAPH_VARIANTS)
    parser.add_argument("--explore_eps", type=float, default=0.15)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--target_update_every", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="results/toy")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.target_update_every <= 0:
        raise ValueError("--target_update_every must be positive")

    set_seed(args.seed)
    device = get_device()
    graph_config = get_graph_config(args.graph_variant)
    print(
        f"Device: {device}, Seed: {args.seed}, Method: {args.method}, "
        f"Graph: {args.graph_variant}"
    )

    env = ToyFactorGameEnv(max_steps=args.max_steps, seed=args.seed)
    agent = ActiveFactorAgent(
        obs_dim=env.obs_dim,
        n_actions=env.n_actions,
        n_options=NUM_OPTIONS,
        graph_config=graph_config,
        hidden_dim=args.hidden_dim,
        method=args.method,
    ).to(device)
    target_agent = copy.deepcopy(agent).to(device)
    target_agent.eval()

    optimizer = Adam(agent.parameters(), lr=args.lr)
    train_log = []
    eval_log = []
    episode_buffer = []
    optimizer_steps = 0

    def flush_episode_buffer(episode_idx: int):
        nonlocal optimizer_steps, target_agent
        if not episode_buffer:
            return
        agent.train()
        losses = train_step(
            agent,
            target_agent,
            optimizer,
            list(episode_buffer),
            device,
            graph_config,
            gamma=args.gamma,
        )
        episode_buffer.clear()
        if losses:
            optimizer_steps += 1
            train_log.append({"episode": episode_idx, "optimizer_step": optimizer_steps, **losses})
            if optimizer_steps % args.target_update_every == 0:
                target_agent.load_state_dict(agent.state_dict())

    for ep in range(args.n_episodes):
        conv = ConventionAssignment(
            modes={factor_id: np.random.randint(0, FACTOR_MODES[factor_id]) for factor_id in range(NUM_FACTORS)}
        )
        env = ToyFactorGameEnv(partner_convention=conv, max_steps=args.max_steps, seed=args.seed + ep)
        agent.eval()
        episode_data = collect_episode(
            env,
            agent,
            device,
            graph_config=graph_config,
            explore_eps=args.explore_eps,
            deterministic=False,
        )
        episode_buffer.append(episode_data)
        if len(episode_buffer) >= args.batch_size:
            flush_episode_buffer(ep + 1)

        if (ep + 1) % args.eval_every == 0:
            flush_episode_buffer(ep + 1)
            target_agent.load_state_dict(agent.state_dict())
            agent.eval()
            eval_results = evaluate(
                agent,
                device,
                graph_config,
                n_episodes=50,
                seed=args.seed + 10000,
                max_steps=args.max_steps,
            )
            eval_results["episode"] = ep + 1
            eval_log.append(eval_results)
            print(
                f"  Ep {ep+1}: reward={eval_results.get('episode_reward', 0):.2f}, "
                f"completion={eval_results.get('completion_rate', 0):.2f}, "
                f"collisions={eval_results.get('collisions', 0):.2f}"
            )

    flush_episode_buffer(args.n_episodes)
    target_agent.load_state_dict(agent.state_dict())

    output_dir = model_output_dir(Path(args.output_dir), args.seed, args.method, args.graph_variant)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": EXPERIMENT_SCHEMA,
        "proposal_version": PROPOSAL_VERSION,
        "code_fix_level": CODE_FIX_LEVEL,
        "method": args.method,
        "graph_variant": args.graph_variant,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ce_estimation": graph_config.ce_metadata(),
        "criticality_source": "low_ce_heuristic" if args.graph_variant == "overcomplete_minus_low_ce" else None,
    }
    save_results(
        {
            **metadata,
            "config": vars(args),
            "graph": {
                "name": graph_config.name,
                "n_factors": graph_config.n_factors,
                "factor_modes": graph_config.factor_modes,
                "ground_truth_mask": graph_config.ground_truth_mask,
            },
            "train": train_log,
            "eval": eval_log,
        },
        str(output_dir / "results.json"),
    )
    torch.save({**metadata, "state_dict": agent.state_dict()}, str(output_dir / "model.pt"))
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
