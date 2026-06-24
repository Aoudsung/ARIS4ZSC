from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml

if __package__ in {None, ""}:  # pragma: no cover - script execution path
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.aris_bellman.factor_belief import FactorLocalBeliefModel
from src.aris_bellman.factor_q import FactorLocalQNetwork
from src.aris_bellman.replay import EvidenceBuffer, OptionReplayBuffer
from src.aris_bellman.specs import GraphSpec, OptionTransition
from src.aris_bellman.td import aris_td_loss

from experiments.overcooked_v2.ce_sampler import (
    collect_option_replay,
    estimate_empirical_ce,
)
from experiments.overcooked_v2.env_adapter import OCV2Adapter
from experiments.overcooked_v2.event_extractor import extract_event
from experiments.overcooked_v2.evidence_router import D_EVID, OCV2EvidenceRouter
from experiments.overcooked_v2.graph_builder import build_graph_variant
from experiments.overcooked_v2.layout_parser import LayoutGraph, parse_layout
from experiments.overcooked_v2.obs_encoder import OCV2ObsEncoder, infer_obs_dim
from experiments.overcooked_v2.options import OCV2OptionLibrary
from experiments.overcooked_v2.partner_pool import make_training_partners

METHODS = (
    "base_only",
    "aris_bellman",
    "flat_factor",
    "global_gru",
    "random_policy",
)


class ArisBellmanQNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, graph: GraphSpec):
        super().__init__()
        self.encoder = OCV2ObsEncoder(obs_dim, hidden_dim)
        self.q_net = FactorLocalQNetwork(
            obs_dim=hidden_dim,
            max_options=graph.num_options,
            max_factors=max(1, graph.num_factors),
            max_modes=max(1, graph.max_modes),
            hidden_dim=hidden_dim,
            relevance_mask=graph.relevance,
        )

    def forward(self, obs_feat: torch.Tensor, belief: torch.Tensor, **graph_kwargs):
        encoded = self.encoder(obs_feat)
        return self.q_net(encoded, belief, **graph_kwargs)


class BaseOnlyQNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, num_options: int):
        super().__init__()
        self.encoder = OCV2ObsEncoder(obs_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_options),
        )

    def forward(
        self,
        obs_feat: torch.Tensor,
        belief: torch.Tensor,
        option_mask: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        del belief
        q_values = self.head(self.encoder(obs_feat))
        return _mask_q_values(q_values, option_mask)


class FlatFactorQNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        num_options: int,
        num_factors: int,
        max_modes: int,
    ):
        super().__init__()
        self.encoder = OCV2ObsEncoder(obs_dim, hidden_dim)
        self.flat_dim = max(0, int(num_factors) * int(max_modes))
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + self.flat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_options),
        )

    def forward(
        self,
        obs_feat: torch.Tensor,
        belief: torch.Tensor,
        option_mask: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        encoded = self.encoder(obs_feat)
        flat = belief.reshape(belief.shape[0], -1)
        q_values = self.head(torch.cat([encoded, flat], dim=-1))
        return _mask_q_values(q_values, option_mask)


class GlobalGRUQNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        num_options: int,
        num_factors: int,
        evidence_dim: int,
    ):
        super().__init__()
        self.encoder = OCV2ObsEncoder(obs_dim, hidden_dim)
        self.history_input_dim = max(1, int(num_factors) * int(evidence_dim))
        self.history = nn.GRU(
            input_size=self.history_input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_options),
        )

    def forward(
        self,
        obs_feat: torch.Tensor,
        evidence_seq: torch.Tensor,
        option_mask: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        encoded = self.encoder(obs_feat)
        if evidence_seq.shape[1] == 0:
            history_in = evidence_seq.new_zeros(
                evidence_seq.shape[0],
                evidence_seq.shape[2],
                self.history_input_dim,
            )
        else:
            history_in = evidence_seq.permute(0, 2, 1, 3).reshape(
                evidence_seq.shape[0],
                evidence_seq.shape[2],
                -1,
            )
        _, hidden = self.history(history_in)
        q_values = self.head(torch.cat([encoded, hidden[-1]], dim=-1))
        return _mask_q_values(q_values, option_mask)


def train(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_config(args.config)
    _apply_cli_overrides(config, args)
    _set_seeds(args.seed)

    layout_name = str(config["layout"])
    env = _build_env(layout_name, config)
    obs, _ = env.reset(args.seed)
    layout_graph = parse_layout(env, layout_name)
    option_lib = OCV2OptionLibrary(
        layout_graph,
        max_option_steps=int(config["options"]["max_option_steps"]),
    )
    graph = _build_graph(env, layout_graph, option_lib, config, args)
    router = OCV2EvidenceRouter(graph, layout_graph.cell_to_entity, layout_graph.region_cells)
    obs_dim = infer_obs_dim(env, obs)

    output_dir = _result_dir(config, args, layout_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "resolved_config.json", config)
    _write_json(output_dir / "graph.json", graph.to_json_dict())
    _capture_git_metadata(output_dir)

    if args.method == "random_policy":
        metrics = _run_random_policy(env, obs, option_lib, router, config, args, output_dir)
        _save_checkpoint(output_dir, args.method, config, graph, metrics, None, None, None)
        return metrics

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_net = _build_q_network(args.method, obs_dim, graph, config).to(device)
    target_q_net = copy.deepcopy(q_net).to(device)
    target_q_net.eval()
    belief_model = _build_belief_model(graph, config).to(device)
    optimizer_params = list(q_net.parameters())
    if args.method in {"aris_bellman", "flat_factor"}:
        optimizer_params += list(belief_model.parameters())
    optimizer = torch.optim.Adam(
        optimizer_params,
        lr=float(config["training"]["learning_rate"]),
    )

    replay = OptionReplayBuffer(
        capacity=int(config["training"]["replay_size"]),
        seed=args.seed,
    )
    evidence_buffer = EvidenceBuffer(
        num_factors=graph.num_factors,
        window=int(config["training"]["evidence_window"]),
        evidence_dim=D_EVID,
    )
    partners = make_training_partners(option_lib)
    rng = np.random.default_rng(args.seed)
    metrics = _empty_metrics(args.method, graph, output_dir)
    obs, _ = _reset_episode(env, evidence_buffer, partners, rng, args.seed)
    episode_return = 0.0
    episode_options = 0
    updates_done = 0
    wall_start = time.time()

    while updates_done < int(config["training"]["total_updates"]):
        if episode_options >= int(config["training"]["max_episode_options"]):
            metrics["episode_returns"].append(float(episode_return))
            obs, _ = _reset_episode(env, evidence_buffer, partners, rng, args.seed)
            episode_return = 0.0
            episode_options = 0

        option_id = _select_option(
            args.method,
            q_net,
            belief_model,
            obs,
            env.state,
            evidence_buffer,
            option_lib,
            graph,
            config,
            updates_done,
            rng,
            device,
        )
        partner = partners[int(rng.integers(0, len(partners)))]
        transition, done, obs = _execute_option(
            env,
            obs,
            partner,
            option_lib,
            router,
            evidence_buffer,
            option_id,
            graph,
            rng,
            config,
        )
        replay.add(transition)
        episode_return += _transition_training_return(transition, config)
        episode_options += 1
        metrics["option_durations"].append(int(transition.duration))
        _increment_count(metrics["termination_counts"], transition.termination_reason)

        if done:
            metrics["episode_returns"].append(float(episode_return))
            obs, _ = _reset_episode(env, evidence_buffer, partners, rng, args.seed)
            episode_return = 0.0
            episode_options = 0

        if len(replay) < int(config["training"]["warmup_transitions"]):
            continue

        for _ in range(int(config["training"].get("updates_per_transition", 1))):
            if updates_done >= int(config["training"]["total_updates"]):
                break
            batch = replay.sample(int(config["training"]["batch_size"]))
            loss = _td_update(
                args.method,
                q_net,
                target_q_net,
                belief_model,
                optimizer,
                batch,
                graph,
                config,
                device,
            )
            if not math.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite TD loss at update {updates_done}: {loss}"
                )

            metrics["td_losses"].append(float(loss))
            updates_done += 1
            if updates_done % int(config["training"]["target_update_interval"]) == 0:
                target_q_net.load_state_dict(q_net.state_dict())
            if updates_done % int(config["training"]["log_interval"]) == 0:
                _write_metrics(output_dir, metrics, updates_done, wall_start)

    if episode_options:
        metrics["episode_returns"].append(float(episode_return))
    _write_metrics(output_dir, metrics, updates_done, wall_start, final=True)
    _save_checkpoint(
        output_dir,
        args.method,
        config,
        graph,
        metrics,
        q_net,
        belief_model,
        optimizer,
    )
    metrics["checkpoint_load_ok"] = _checkpoint_loads(output_dir / "checkpoint.pt")
    _write_metrics(output_dir, metrics, updates_done, wall_start, final=True)
    return metrics


def _build_env(layout_name: str, config: dict[str, Any]) -> OCV2Adapter:
    env_cfg = dict(config.get("env", {}))
    return OCV2Adapter(
        layout=layout_name,
        max_steps=int(env_cfg.get("max_steps", 200)),
        observation_type=str(env_cfg.get("observation_type", "featurized")),
        agent_view_size=env_cfg.get("agent_view_size"),
        negative_rewards=bool(env_cfg.get("negative_rewards", True)),
        sample_recipe_on_delivery=bool(env_cfg.get("sample_recipe_on_delivery", True)),
        random_reset=bool(env_cfg.get("random_reset", False)),
        random_agent_positions=bool(env_cfg.get("random_agent_positions", False)),
        force_path_planning=bool(env_cfg.get("force_path_planning", True)),
    )


def _build_graph(
    env: OCV2Adapter,
    layout_graph: LayoutGraph,
    option_lib: OCV2OptionLibrary,
    config: dict[str, Any],
    args: argparse.Namespace,
) -> GraphSpec:
    graph_cfg = config.get("graph", {})
    ce_path = graph_cfg.get("ce_path")
    if ce_path:
        ce_matrix = np.load(Path(ce_path))
    else:
        partners = make_training_partners(option_lib)
        replay = collect_option_replay(
            env,
            partners,
            option_lib,
            layout_name=layout_graph.layout_name,
            episodes=int(graph_cfg.get("ce_episodes", 2)),
            max_options_per_episode=graph_cfg.get("ce_max_options_per_episode"),
            seed=args.seed,
            gamma=float(config["training"].get("gamma", 0.99)),
            horizon_options=int(graph_cfg.get("local_return_horizon_options", 3)),
            cost_per_step=float(config["training"].get("cost_per_step", 1.0)),
        )
        ce_matrix = estimate_empirical_ce(
            replay,
            option_lib.num_options,
            min_weight=float(graph_cfg.get("ce_min_weight", 1.0)),
        )

    graph = build_graph_variant(
        args.graph_variant,
        layout_graph.layout_name,
        option_lib.options,
        ce_matrix,
        eta=float(graph_cfg.get("ce_eta", 0.0)),
        max_factors=int(graph_cfg.get("max_factors", 16)),
        mode_config=graph_cfg.get("modes"),
        seed=args.seed,
    )
    if graph.num_factors == 0 and bool(graph_cfg.get("debug_fallback_topk_when_empty", True)):
        graph = build_graph_variant(
            args.graph_variant,
            layout_graph.layout_name,
            option_lib.options,
            ce_matrix,
            eta=-1.0,
            max_factors=int(graph_cfg.get("max_factors", 16)),
            mode_config=graph_cfg.get("modes"),
            seed=args.seed,
        )
        graph.metadata = {
            **(graph.metadata or {}),
            "debug_empty_graph_fallback": True,
            "original_eta": float(graph_cfg.get("ce_eta", 0.0)),
        }
    if graph.num_factors == 0:
        raise RuntimeError("Graph construction produced zero factors; cannot train ARIS.")
    return graph


def _build_q_network(
    method: str,
    obs_dim: int,
    graph: GraphSpec,
    config: dict[str, Any],
) -> nn.Module:
    hidden_dim = int(config["training"]["hidden_dim"])
    if method == "aris_bellman":
        return ArisBellmanQNetwork(obs_dim, hidden_dim, graph)
    if method == "base_only":
        return BaseOnlyQNetwork(obs_dim, hidden_dim, graph.num_options)
    if method == "flat_factor":
        return FlatFactorQNetwork(
            obs_dim,
            hidden_dim,
            graph.num_options,
            graph.num_factors,
            graph.max_modes,
        )
    if method == "global_gru":
        return GlobalGRUQNetwork(
            obs_dim,
            hidden_dim,
            graph.num_options,
            graph.num_factors,
            D_EVID,
        )
    raise ValueError(f"Unsupported trainable method {method!r}.")


def _build_belief_model(graph: GraphSpec, config: dict[str, Any]) -> FactorLocalBeliefModel:
    return FactorLocalBeliefModel(
        evidence_dim=D_EVID,
        hidden_dim=int(config["training"]["hidden_dim"]),
        max_factors=max(1, graph.num_factors),
        max_modes=max(1, graph.max_modes),
    )


def _run_random_policy(
    env: OCV2Adapter,
    obs: dict[str, np.ndarray],
    option_lib: OCV2OptionLibrary,
    router: OCV2EvidenceRouter,
    config: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    del obs
    rng = np.random.default_rng(args.seed)
    partners = make_training_partners(option_lib)
    evidence_buffer = EvidenceBuffer(
        num_factors=router.graph.num_factors,
        window=int(config["training"]["evidence_window"]),
        evidence_dim=D_EVID,
    )
    obs, _ = _reset_episode(env, evidence_buffer, partners, rng, args.seed)
    metrics = _empty_metrics(args.method, router.graph, output_dir)
    episode_return = 0.0
    episode_options = 0
    wall_start = time.time()

    for update_idx in range(int(config["training"]["total_updates"])):
        if episode_options >= int(config["training"]["max_episode_options"]):
            metrics["episode_returns"].append(float(episode_return))
            obs, _ = _reset_episode(env, evidence_buffer, partners, rng, args.seed)
            episode_return = 0.0
            episode_options = 0

        option_id = _sample_valid_option(option_lib, env.state, 0, rng)
        partner = partners[int(rng.integers(0, len(partners)))]
        transition, done, obs = _execute_option(
            env,
            obs,
            partner,
            option_lib,
            router,
            evidence_buffer,
            option_id,
            router.graph,
            rng,
            config,
        )
        episode_return += _transition_training_return(transition, config)
        episode_options += 1
        metrics["option_durations"].append(int(transition.duration))
        _increment_count(metrics["termination_counts"], transition.termination_reason)
        if done:
            metrics["episode_returns"].append(float(episode_return))
            obs, _ = _reset_episode(env, evidence_buffer, partners, rng, args.seed)
            episode_return = 0.0
            episode_options = 0
        if update_idx % int(config["training"]["log_interval"]) == 0:
            _write_metrics(output_dir, metrics, update_idx, wall_start)

    if episode_options:
        metrics["episode_returns"].append(float(episode_return))
    _write_metrics(output_dir, metrics, int(config["training"]["total_updates"]), wall_start)
    return metrics


def _execute_option(
    env: OCV2Adapter,
    obs: dict[str, np.ndarray],
    partner: Any,
    option_lib: OCV2OptionLibrary,
    router: OCV2EvidenceRouter,
    evidence_buffer: EvidenceBuffer,
    option_id: int,
    graph: GraphSpec,
    rng: np.random.Generator,
    config: dict[str, Any],
) -> tuple[OptionTransition, bool, dict[str, np.ndarray]]:
    opt = option_lib.options[int(option_id)]
    evidence_t = evidence_buffer.snapshot()
    obs_feat_t = _obs_vector(obs, "agent_0")
    expected_cost = option_lib.expected_cost(env.state, 0, int(option_id))
    reward_sum = 0.0
    realized_cost = 0.0
    duration = 0
    done = False
    termination_reason = "running"

    while duration < opt.max_steps:
        ego_action = option_lib.primitive_action(env.state, 0, int(option_id))
        partner_obs = obs.get("agent_1") if isinstance(obs, dict) else None
        partner_action = partner.act(partner_obs, env.state, rng)
        prev_state = env.state
        step = env.step(ego_action, partner_action.primitive_action)
        event = extract_event(
            prev_state,
            ego_action,
            partner_action.primitive_action,
            step.state,
            step.info,
            partner_action.option_id,
            partner_action.option_dist,
        )
        evidence_buffer.append(router.route(event, ego_option_id=int(option_id)))
        reward_sum += _training_reward(step, config, "agent_0")
        realized_cost += float(config["training"].get("cost_per_step", 1.0))
        duration += 1
        done = bool(step.dones.get("__all__", False))
        terminated, termination_reason = option_lib.option_terminated(
            opt,
            prev_state,
            step.state,
            event,
            agent_id=0,
            elapsed=duration,
        )
        obs = step.obs
        if done or terminated:
            break

    return (
        OptionTransition(
            obs_feat_t=obs_feat_t,
            evidence_t=evidence_t,
            option_id=int(option_id),
            reward_sum=float(reward_sum),
            expected_cost=float(expected_cost),
            realized_cost=float(realized_cost),
            duration=max(1, int(duration)),
            obs_feat_next=_obs_vector(obs, "agent_0"),
            evidence_next=evidence_buffer.snapshot(),
            done=bool(done),
            termination_reason=termination_reason,
            graph_id=f"{graph.layout_name}:{graph.metadata.get('graph_variant', 'graph')}",
        ),
        done,
        obs,
    )


def _td_update(
    method: str,
    q_net: nn.Module,
    target_q_net: nn.Module,
    belief_model: FactorLocalBeliefModel,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, Any],
    graph: GraphSpec,
    config: dict[str, Any],
    device: torch.device,
) -> float:
    q_net.train()
    if method in {"aris_bellman", "flat_factor"}:
        belief_model.train()

    obs_t = _tensor(batch["obs_feat_t"], device)
    obs_next = _tensor(batch["obs_feat_next"], device)
    evidence_t = _tensor(batch["evidence_t"], device)
    evidence_next = _tensor(batch["evidence_next"], device)
    graph_batch = _graph_tensors(graph, obs_t.shape[0], device)
    state_t = _state_repr(method, belief_model, evidence_t, graph_batch)
    state_next = _state_repr(method, belief_model, evidence_next, graph_batch)

    optimizer.zero_grad(set_to_none=True)
    loss = aris_td_loss(
        q_net,
        target_q_net,
        obs_t,
        state_t,
        torch.as_tensor(batch["option_id"], dtype=torch.long, device=device),
        _tensor(batch["reward_sum"], device),
        _tensor(batch["realized_cost"], device),
        _tensor(batch["duration"], device),
        obs_next,
        state_next,
        _tensor(batch["done"], device),
        graph_batch,
        gamma=float(config["training"]["gamma"]),
        cost_coef=float(config["training"]["cost_coef"]),
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        _trainable_params(q_net, belief_model, method),
        float(config["training"]["grad_clip_norm"]),
    )
    optimizer.step()
    return float(loss.detach().cpu().item())


def _state_repr(
    method: str,
    belief_model: FactorLocalBeliefModel,
    evidence: torch.Tensor,
    graph_batch: dict[str, Any],
) -> torch.Tensor:
    if method == "global_gru":
        return evidence
    if method in {"aris_bellman", "flat_factor"}:
        return belief_model(
            evidence,
            graph_batch.get("factor_features"),
            graph_batch["factor_mask"],
            graph_batch["mode_mask"],
        )
    return evidence.new_zeros(
        evidence.shape[0],
        graph_batch["factor_mask"].shape[1],
        graph_batch["mode_mask"].shape[2],
    )


def _select_option(
    method: str,
    q_net: nn.Module,
    belief_model: FactorLocalBeliefModel,
    obs: dict[str, np.ndarray],
    state: Any,
    evidence_buffer: EvidenceBuffer,
    option_lib: OCV2OptionLibrary,
    graph: GraphSpec,
    config: dict[str, Any],
    update_idx: int,
    rng: np.random.Generator,
    device: torch.device,
) -> int:
    valid = option_lib.valid_options(state, 0)
    valid_ids = np.flatnonzero(valid)
    if valid_ids.size == 0:
        return _noop_option_id(option_lib)
    if method == "random_policy" or rng.random() < _epsilon(config, update_idx):
        return int(rng.choice(valid_ids))

    with torch.no_grad():
        obs_tensor = _tensor(_obs_vector(obs, "agent_0")[None, :], device)
        evidence = _tensor(evidence_buffer.snapshot()[None, ...], device)
        graph_batch = _graph_tensors(graph, 1, device)
        state_repr = _state_repr(method, belief_model, evidence, graph_batch)
        q_values = q_net(obs_tensor, state_repr, **_q_forward_kwargs(graph_batch)).squeeze(0)
        valid_tensor = torch.as_tensor(valid, dtype=torch.bool, device=device)
        q_values = q_values.masked_fill(~valid_tensor, -1e9)
        return int(torch.argmax(q_values).item())


def _q_forward_kwargs(graph_batch: dict[str, Any]) -> dict[str, Any]:
    return {
        "option_mask": graph_batch["option_mask"],
        "factor_mask": graph_batch["factor_mask"],
        "mode_mask": graph_batch["mode_mask"],
        "relevance_mask": graph_batch["relevance_mask"],
        "option_features": graph_batch.get("option_features"),
        "factor_features": graph_batch.get("factor_features"),
    }


def _graph_tensors(graph: GraphSpec, batch_size: int, device: torch.device) -> dict[str, Any]:
    option_mask = torch.as_tensor(graph.option_mask, dtype=torch.bool, device=device)
    factor_mask = torch.as_tensor(graph.factor_mask, dtype=torch.bool, device=device)
    mode_mask = torch.as_tensor(graph.mode_mask, dtype=torch.bool, device=device)
    relevance = torch.as_tensor(graph.relevance, dtype=torch.bool, device=device)
    return {
        "option_mask": option_mask.unsqueeze(0).expand(batch_size, -1),
        "option_mask_next": option_mask.unsqueeze(0).expand(batch_size, -1),
        "factor_mask": factor_mask.unsqueeze(0).expand(batch_size, -1),
        "mode_mask": mode_mask.unsqueeze(0).expand(batch_size, -1, -1),
        "relevance_mask": relevance.unsqueeze(0).expand(batch_size, -1, -1),
        "option_features": _optional_feature_tensor(graph.option_features, batch_size, device),
        "factor_features": _optional_feature_tensor(graph.factor_features, batch_size, device),
    }


def _optional_feature_tensor(
    values: np.ndarray | None,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor | None:
    if values is None:
        return None
    tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
    return tensor.unsqueeze(0).expand(batch_size, -1, -1)


def _reset_episode(
    env: OCV2Adapter,
    evidence_buffer: EvidenceBuffer,
    partners: list[Any],
    rng: np.random.Generator,
    base_seed: int,
) -> tuple[dict[str, np.ndarray], Any]:
    evidence_buffer.reset()
    seed = int(rng.integers(0, 2**31 - 1)) ^ int(base_seed)
    obs, state = env.reset(seed)
    for partner in partners:
        partner.reset(seed)
    return obs, state


def _training_reward(step: Any, config: dict[str, Any], agent_key: str) -> float:
    sparse = float(step.rewards.get(agent_key, 0.0))
    shaped_coef = float(config["training"].get("shaped_reward_coef", 0.0))
    return sparse + shaped_coef * _shaped_reward_for_agent(step.info, agent_key)


def _transition_training_return(
    transition: OptionTransition,
    config: dict[str, Any],
) -> float:
    cost_coef = float(config["training"]["cost_coef"])
    return float(transition.reward_sum - cost_coef * transition.realized_cost)


def _shaped_reward_for_agent(info: dict[str, Any], agent_key: str) -> float:
    shaped = info.get("shaped_reward", 0.0)
    if isinstance(shaped, dict):
        if agent_key in shaped:
            return _as_float(shaped[agent_key])
        return float(sum(_as_float(value) for value in shaped.values()))
    return _as_float(shaped)


def _as_float(value: Any) -> float:
    return float(np.asarray(value).item())


def _empty_metrics(method: str, graph: GraphSpec, output_dir: Path) -> dict[str, Any]:
    return {
        "method": method,
        "layout": graph.layout_name,
        "num_options": graph.num_options,
        "num_factors": graph.num_factors,
        "output_dir": str(output_dir),
        "td_losses": [],
        "episode_returns": [],
        "episode_return_kind": "reward_sum_minus_cost_coef_realized_cost",
        "option_durations": [],
        "termination_counts": {},
        "checkpoint_load_ok": False,
    }


def _write_metrics(
    output_dir: Path,
    metrics: dict[str, Any],
    updates_done: int,
    wall_start: float,
    final: bool = False,
) -> None:
    metrics["updates_done"] = int(updates_done)
    metrics["wall_time_sec"] = float(time.time() - wall_start)
    metrics["final"] = bool(final)
    metrics.update(_metrics_summary(metrics))
    _write_json(output_dir / "metrics.json", metrics)


def _metrics_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    losses = np.asarray(metrics.get("td_losses", []), dtype=np.float64)
    rewards = np.asarray(metrics.get("episode_returns", []), dtype=np.float64)
    first_mean = _window_mean(losses, first=True)
    last_mean = _window_mean(losses, first=False)
    return {
        "finite_td_loss": bool(losses.size == 0 or np.all(np.isfinite(losses))),
        "td_loss_first_window": first_mean,
        "td_loss_last_window": last_mean,
        "td_loss_decreased": _loss_decreased(first_mean, last_mean),
        "reward_mean": float(np.mean(rewards)) if rewards.size else None,
        "reward_variance": float(np.var(rewards)) if rewards.size else None,
        "finite_rewards": bool(rewards.size == 0 or np.all(np.isfinite(rewards))),
    }


def _loss_decreased(first_mean: float | None, last_mean: float | None) -> bool:
    if first_mean is None or last_mean is None:
        return False
    return bool(first_mean > last_mean)


def _window_mean(values: np.ndarray, *, first: bool) -> float | None:
    if values.size == 0:
        return None
    window = min(20, values.size)
    chunk = values[:window] if first else values[-window:]
    return float(np.mean(chunk))


def _save_checkpoint(
    output_dir: Path,
    method: str,
    config: dict[str, Any],
    graph: GraphSpec,
    metrics: dict[str, Any],
    q_net: nn.Module | None,
    belief_model: nn.Module | None,
    optimizer: torch.optim.Optimizer | None,
) -> None:
    payload: dict[str, Any] = {
        "method": method,
        "config": config,
        "graph": graph.to_json_dict(),
        "metrics_summary": _metrics_summary(metrics),
    }
    if q_net is not None:
        payload["q_net"] = q_net.state_dict()
    if belief_model is not None:
        payload["belief_model"] = belief_model.state_dict()
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, output_dir / "checkpoint.pt")


def _checkpoint_loads(path: Path) -> bool:
    try:
        torch.load(path, map_location="cpu")
    except Exception:
        return False
    return True


def _result_dir(config: dict[str, Any], args: argparse.Namespace, layout: str) -> Path:
    root = Path(args.output_dir or config.get("output_dir", "results/ocv2"))
    return root / layout / args.method / args.graph_variant / f"seed{args.seed}"


def _capture_git_metadata(output_dir: Path) -> None:
    for name, command in {
        "git_diff_stat.txt": ["git", "diff", "--stat"],
        "git_diff.patch": ["git", "diff", "--", "experiments/overcooked_v2", "src/aris_bellman"],
    }.items():
        result = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        (output_dir / name).write_text(result.stdout, encoding="utf-8")


def _load_config(path: str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping.")
    return data


def _apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    config.setdefault("training", {})
    if args.updates is not None:
        config["training"]["total_updates"] = int(args.updates)
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)


def _set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _epsilon(config: dict[str, Any], update_idx: int) -> float:
    train_cfg = config["training"]
    start = float(train_cfg.get("epsilon_start", 0.2))
    end = float(train_cfg.get("epsilon_end", 0.05))
    total = max(1, int(train_cfg.get("total_updates", 1)))
    frac = min(1.0, max(0.0, update_idx / total))
    return float(start + frac * (end - start))


def _sample_valid_option(
    option_lib: OCV2OptionLibrary,
    state: Any,
    agent_id: int,
    rng: np.random.Generator,
) -> int:
    valid_ids = np.flatnonzero(option_lib.valid_options(state, agent_id))
    if valid_ids.size:
        return int(rng.choice(valid_ids))
    return _noop_option_id(option_lib)


def _noop_option_id(option_lib: OCV2OptionLibrary) -> int:
    for opt in option_lib.options:
        if opt.kind == "noop":
            return int(opt.id)
    return 0


def _obs_vector(obs: dict[str, np.ndarray], agent_key: str) -> np.ndarray:
    return np.asarray(obs[agent_key], dtype=np.float32).reshape(-1)


def _tensor(value: Any, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(value).copy(), dtype=torch.float32, device=device)


def _trainable_params(
    q_net: nn.Module,
    belief_model: FactorLocalBeliefModel,
    method: str,
) -> list[nn.Parameter]:
    params = list(q_net.parameters())
    if method in {"aris_bellman", "flat_factor"}:
        params += list(belief_model.parameters())
    return params


def _mask_q_values(
    q_values: torch.Tensor,
    option_mask: torch.Tensor | None,
) -> torch.Tensor:
    if option_mask is None:
        return q_values
    return q_values.masked_fill(~option_mask.bool(), -1e9)


def _increment_count(counts: dict[str, int], key: str) -> None:
    counts[key] = int(counts.get(key, 0)) + 1


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ARIS on OvercookedV2 options.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--graph_variant", required=True)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    metrics = train(args)
    print(json.dumps(_metrics_summary(metrics), indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
