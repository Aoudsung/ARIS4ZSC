from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import pickle
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import yaml

if __package__ in {None, ""}:  # pragma: no cover - script execution path
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.aris_bellman.factor_belief import FactorLocalBeliefModel
from src.aris_bellman.factor_q import FactorLocalQNetwork
from src.aris_bellman.replay import (
    EpisodeSequenceReplayBuffer,
    EvidenceBuffer,
    OptionReplayBuffer,
)
from src.aris_bellman.specs import GraphSpec, OptionTransition
from src.aris_bellman.td import aris_td_loss

from experiments.overcooked_v2.env_adapter import OCV2Adapter
from experiments.overcooked_v2.event_extractor import (
    EVENT_SEMANTICS_VERSION,
    PARTNER_OPTION_EVIDENCE_POLICY,
    actor_sparse_reward,
    extract_event,
    sparse_credit_params,
)
from experiments.overcooked_v2.evidence_router import D_EVID, EVIDENCE_INDEX, OCV2EvidenceRouter
from experiments.overcooked_v2.graph_builder import (
    _load_ego_selectable_from_replay,
    build_graph_variant,
    validate_task_stage_coverage,
)
from experiments.overcooked_v2.layout_diagnostics import preflight_layout
from experiments.overcooked_v2.layout_parser import LayoutGraph, parse_layout
from experiments.overcooked_v2.obs_featurizer import NumpyFeaturizer
from experiments.overcooked_v2.obs_encoder import OCV2ObsEncoder, infer_obs_dim
from experiments.overcooked_v2.option_executor import option_primitive_step
from experiments.overcooked_v2.option_inferencer import (
    PartnerOptionInferencer,
    make_behavior_option_inferencer,
)
from experiments.overcooked_v2.option_termination import OptionRuntime, option_success
from experiments.overcooked_v2.options import OCV2OptionLibrary
from experiments.overcooked_v2.path_c_config import (
    normalize_path_c_config,
    path_c_metadata,
)
from experiments.overcooked_v2.path_c_evaluation import PATH_C_PROBE_PROVENANCE_IDS
from experiments.overcooked_v2.path_c_sequence import (
    DecisionEvidenceBuffer,
    EgoEvidenceSpecV1,
    EpisodeBootstrapRecord,
    EpisodeEvidenceBuffer,
    RecurrentEnsembleQ,
    SequenceTDBatch,
    collate_episode_records,
    ensemble_diversity_telemetry,
    sequence_td_loss,
)
from experiments.overcooked_v2.partner_pool import PARTNER_REGISTRIES, make_training_partners
from experiments.overcooked_v2.residual_signature import (
    normalized_advantage_disagreement,
    select_probe_candidate,
)
from experiments.overcooked_v2.reward_design import (
    ContributionLedger,
    terminal_progress_bonus,
    terminal_progress_params,
)
from experiments.overcooked_v2.provenance import (
    GRAPH_HASH_FIELD,
    PROVENANCE_SCHEMA_VERSION,
    graph_content_hash,
    graph_hash_from_spec,
    runtime_provenance,
    stamp_graph_hash,
)
from experiments.overcooked_v2.state_utils import get_agent_pos
from jaxmarl.environments.overcooked_v2.common import Actions as _OCActions

METHODS = (
    "base_only",
    "aris_bellman",
    "flat_factor",
    "global_gru",
    "partner_id_q",
    "random_policy",
)

logger = logging.getLogger(__name__)


class ArisBellmanQNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: Any,
        hidden_dim: int,
        graph: GraphSpec,
        encoder_type: str = "auto",
        advantage_norm: str = "none",
        value_bound: bool = False,
        vmax: float = 20.0,
        base_bound: float | None = None,
        adv_bound: float | None = None,
        adv_unit: float = 1.0,
    ):
        super().__init__()
        self.encoder = OCV2ObsEncoder(obs_dim, hidden_dim, encoder_type=encoder_type)
        self.q_net = FactorLocalQNetwork(
            obs_dim=hidden_dim,
            max_options=graph.num_options,
            max_factors=max(1, graph.num_factors),
            max_modes=max(1, graph.max_modes),
            hidden_dim=hidden_dim,
            relevance_mask=graph.relevance,
            advantage_norm=advantage_norm,
            value_bound=value_bound,
            vmax=vmax,
            base_bound=base_bound,
            adv_bound=adv_bound,
            adv_unit=adv_unit,
        )

    def forward(self, obs_feat: torch.Tensor, belief: torch.Tensor, **graph_kwargs):
        graph_kwargs.pop("partner_id", None)
        encoded = self.encoder(obs_feat)
        return self.q_net(encoded, belief, **graph_kwargs)

    def forward_mean(self, obs_feat: torch.Tensor, belief: torch.Tensor, **graph_kwargs):
        return self.forward(obs_feat, belief, **graph_kwargs)

    def forward_heads(self, obs_feat: torch.Tensor, belief: torch.Tensor, **graph_kwargs):
        return self.forward(obs_feat, belief, **graph_kwargs).unsqueeze(1)

    def disagreement_values(self, obs_feat: torch.Tensor, belief: torch.Tensor, **graph_kwargs):
        q_values = self.forward(obs_feat, belief, **graph_kwargs)
        return {
            "per_option": torch.zeros_like(q_values),
            "scalar": torch.zeros(q_values.shape[0], dtype=q_values.dtype, device=q_values.device),
        }

    def forward_with_belief_override(
        self,
        obs: torch.Tensor,
        belief_override: torch.Tensor,
        *,
        graph_kwargs: dict[str, Any] | None = None,
        bypass_evidence_recompute: bool = True,
    ) -> torch.Tensor:
        graph_kwargs = dict(graph_kwargs or {})
        graph_kwargs.pop("partner_id", None)
        encoded = self.encoder(obs)
        if hasattr(self.q_net, "forward_with_belief_override"):
            return self.q_net.forward_with_belief_override(
                encoded,
                belief_override,
                graph_kwargs=graph_kwargs,
                bypass_evidence_recompute=bypass_evidence_recompute,
            )
        return self.q_net(encoded, belief_override, **graph_kwargs)


class EnsembleArisBellmanQNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: Any,
        hidden_dim: int,
        graph: GraphSpec,
        *,
        n_heads: int,
        encoder_type: str = "auto",
        advantage_norm: str = "none",
        value_bound: bool = False,
        vmax: float = 20.0,
        base_bound: float | None = None,
        adv_bound: float | None = None,
        adv_unit: float = 1.0,
        disagreement_stat: str = "variance",
        prior_scale: float = 0.0,
    ):
        super().__init__()
        if int(n_heads) <= 1:
            raise ValueError("EnsembleArisBellmanQNetwork requires n_heads > 1.")
        self.n_heads = int(n_heads)
        self.disagreement_stat = str(disagreement_stat)
        self.prior_scale = float(prior_scale)
        self.encoder = OCV2ObsEncoder(obs_dim, hidden_dim, encoder_type=encoder_type)
        self.heads = nn.ModuleList(
            [
                FactorLocalQNetwork(
                    obs_dim=hidden_dim,
                    max_options=graph.num_options,
                    max_factors=max(1, graph.num_factors),
                    max_modes=max(1, graph.max_modes),
                    hidden_dim=hidden_dim,
                    relevance_mask=graph.relevance,
                    advantage_norm=advantage_norm,
                    value_bound=value_bound,
                    vmax=vmax,
                    base_bound=base_bound,
                    adv_bound=adv_bound,
                    adv_unit=adv_unit,
                )
                for _ in range(self.n_heads)
            ]
        )
        self.prior_encoder: nn.Module | None = None
        self.prior_heads = nn.ModuleList()
        if self.prior_scale != 0.0:
            self.prior_encoder = OCV2ObsEncoder(
                obs_dim,
                hidden_dim,
                encoder_type=encoder_type,
            )
            self.prior_heads = nn.ModuleList(
                [
                    FactorLocalQNetwork(
                        obs_dim=hidden_dim,
                        max_options=graph.num_options,
                        max_factors=max(1, graph.num_factors),
                        max_modes=max(1, graph.max_modes),
                        hidden_dim=hidden_dim,
                        relevance_mask=graph.relevance,
                        advantage_norm=advantage_norm,
                        value_bound=value_bound,
                        vmax=vmax,
                        base_bound=base_bound,
                        adv_bound=adv_bound,
                        adv_unit=adv_unit,
                    )
                    for _ in range(self.n_heads)
                ]
            )
            for param in self.prior_heads.parameters():
                param.requires_grad_(False)
            for param in self.prior_encoder.parameters():
                param.requires_grad_(False)
            self.prior_encoder.eval()
            self.prior_heads.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.prior_encoder is not None:
            self.prior_encoder.eval()
            self.prior_heads.eval()
        return self

    def forward_heads(self, obs_feat: torch.Tensor, belief: torch.Tensor, **graph_kwargs):
        graph_kwargs.pop("partner_id", None)
        encoded = self.encoder(obs_feat)
        learned = torch.stack(
            [head(encoded, belief, **graph_kwargs) for head in self.heads],
            dim=1,
        )
        if self.prior_scale == 0.0:
            return learned
        if self.prior_encoder is None:
            raise RuntimeError("Randomized prior encoder is missing.")
        with torch.no_grad():
            prior_encoded = self.prior_encoder(obs_feat)
            prior = torch.stack(
                [head(prior_encoded, belief, **graph_kwargs) for head in self.prior_heads],
                dim=1,
            )
        return learned + self.prior_scale * prior

    def forward_mean(self, obs_feat: torch.Tensor, belief: torch.Tensor, **graph_kwargs):
        return self.forward_heads(obs_feat, belief, **graph_kwargs).mean(dim=1)

    def forward(self, obs_feat: torch.Tensor, belief: torch.Tensor, **graph_kwargs):
        return self.forward_mean(obs_feat, belief, **graph_kwargs)

    def forward_with_belief_override(
        self,
        obs: torch.Tensor,
        belief_override: torch.Tensor,
        *,
        graph_kwargs: dict[str, Any] | None = None,
        bypass_evidence_recompute: bool = True,
    ) -> torch.Tensor:
        del bypass_evidence_recompute
        return self.forward_mean(obs, belief_override, **(graph_kwargs or {}))

    def disagreement_values(self, obs_feat: torch.Tensor, belief: torch.Tensor, **graph_kwargs):
        heads = self.forward_heads(obs_feat, belief, **graph_kwargs)
        if self.disagreement_stat == "range":
            per_option = heads.max(dim=1).values - heads.min(dim=1).values
        else:
            per_option = heads.var(dim=1, unbiased=False)
        option_mask = graph_kwargs.get("option_mask")
        if option_mask is not None:
            per_option = per_option.masked_fill(~option_mask.bool(), 0.0)
        return {
            "per_option": per_option,
            "scalar": per_option.max(dim=1).values,
        }


class BaseOnlyQNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: Any,
        hidden_dim: int,
        num_options: int,
        encoder_type: str = "auto",
        q_bound_vmax: float | None = None,
    ):
        super().__init__()
        self.q_bound_vmax = _validated_q_bound_vmax(q_bound_vmax)
        self.encoder = OCV2ObsEncoder(obs_dim, hidden_dim, encoder_type=encoder_type)
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
        q_values = _apply_final_q_value_bound(q_values, self.q_bound_vmax)
        return _mask_q_values(q_values, option_mask)


class FlatFactorQNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: Any,
        hidden_dim: int,
        num_options: int,
        num_factors: int,
        max_modes: int,
        encoder_type: str = "auto",
        q_bound_vmax: float | None = None,
    ):
        super().__init__()
        self.q_bound_vmax = _validated_q_bound_vmax(q_bound_vmax)
        self.encoder = OCV2ObsEncoder(obs_dim, hidden_dim, encoder_type=encoder_type)
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
        q_values = _apply_final_q_value_bound(q_values, self.q_bound_vmax)
        return _mask_q_values(q_values, option_mask)


class PartnerIDQNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: Any,
        hidden_dim: int,
        num_options: int,
        num_partners: int,
        encoder_type: str = "auto",
        q_bound_vmax: float | None = None,
    ):
        super().__init__()
        self.q_bound_vmax = _validated_q_bound_vmax(q_bound_vmax)
        self.encoder = OCV2ObsEncoder(obs_dim, hidden_dim, encoder_type=encoder_type)
        self.partner_embedding = nn.Embedding(max(1, int(num_partners)), hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_options),
        )

    def forward(
        self,
        obs_feat: torch.Tensor,
        belief: torch.Tensor,
        option_mask: torch.Tensor | None = None,
        partner_id: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        del belief
        encoded = self.encoder(obs_feat)
        if partner_id is None:
            partner_id = torch.zeros(
                encoded.shape[0],
                dtype=torch.long,
                device=encoded.device,
            )
        partner_id = partner_id.long().clamp(min=0, max=self.partner_embedding.num_embeddings - 1)
        partner_context = self.partner_embedding(partner_id.reshape(-1))
        q_values = self.head(torch.cat([encoded, partner_context], dim=-1))
        q_values = _apply_final_q_value_bound(q_values, self.q_bound_vmax)
        return _mask_q_values(q_values, option_mask)



def train(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_config(args.config)
    _apply_cli_overrides(config, args)
    _normalize_training_stability_config(config)
    normalize_path_c_config(config, config_path=args.config)
    _set_seeds(args.seed)

    layout_name = str(config["layout"])
    env = _build_env(layout_name, config)
    layout_graph = parse_layout(env, layout_name)
    env.set_featurizer(NumpyFeaturizer(layout_graph))
    obs, _ = env.reset(args.seed)
    option_lib = _build_option_lib(layout_graph, config)
    output_dir = _result_dir(config, args, layout_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight_gate = _enforce_preflight_gate(layout_name, config, args)
    _write_json(output_dir / "preflight_gate.json", preflight_gate)

    partners = _select_train_partners(option_lib, config)
    graph = _build_graph(env, layout_graph, option_lib, config, args)
    _enforce_graph_objective_metadata(
        graph,
        config,
        layout_graph=layout_graph,
        option_lib=option_lib,
        partners=partners,
    )
    graph.metadata = {
        **(graph.metadata or {}),
        "preflight_gate": preflight_gate,
        "path_c": path_c_metadata(config),
    }
    _stamp_runtime_provenance(
        graph,
        config,
        layout_graph,
        option_lib,
        partners,
        fill_missing_hashes=True,
    )
    router = OCV2EvidenceRouter(graph, layout_graph.cell_to_entity, layout_graph.region_cells)
    obs_dim = infer_obs_dim(env, obs)

    _write_json(output_dir / "resolved_config.json", config)
    _write_graph_json(output_dir / "graph.json", graph)
    _capture_git_metadata(output_dir)

    if args.method == "random_policy":
        metrics = _run_random_policy(env, obs, option_lib, router, config, args, output_dir)
        metrics["path_c"] = path_c_metadata(config)
        _save_checkpoint(output_dir, args.method, config, graph, metrics, None, None, None)
        return metrics

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_net = _build_q_network(args.method, obs_dim, graph, config).to(device)
    target_q_net = copy.deepcopy(q_net).to(device)
    target_q_net.eval()
    sequence_training = _uses_sequence_q(q_net)
    sequence_replay_contract = (
        _validate_sequence_replay_contract(config)
        if sequence_training
        else None
    )
    path_c_probe_base_q_net = _load_path_c_probe_baseline_q_net(
        config,
        obs_dim,
        graph,
        device,
    )
    belief_model = _build_belief_model(graph, config).to(device)
    optimizer_params = [
        parameter for parameter in q_net.parameters() if parameter.requires_grad
    ]
    if args.method in {"aris_bellman", "flat_factor"} and not sequence_training:
        optimizer_params += [
            parameter for parameter in belief_model.parameters() if parameter.requires_grad
        ]
    optimizer = torch.optim.Adam(
        optimizer_params,
        lr=float(config["training"]["learning_rate"]),
    )
    replay: OptionReplayBuffer | EpisodeSequenceReplayBuffer
    if sequence_training:
        replay = EpisodeSequenceReplayBuffer(
            capacity=int(config["training"]["replay_size"]),
            seed=args.seed,
        )
    else:
        replay = OptionReplayBuffer(
            capacity=int(config["training"]["replay_size"]),
            seed=args.seed,
        )
    evidence_buffer = EvidenceBuffer(
        num_factors=graph.num_factors,
        window=int(config["training"]["evidence_window"]),
        evidence_dim=D_EVID,
    )
    rng = np.random.default_rng(args.seed)
    metrics = _empty_metrics(args.method, graph, output_dir)
    metrics["path_c"] = path_c_metadata(config)
    if sequence_training:
        metrics["sequence_model"] = {
            "evidence_schema_version": _sequence_spec(q_net).schema_version,
            "evidence_spec_sha256": _sequence_spec(q_net).sha256(),
            "replay_unit": "complete_episode",
            "bootstrap_scope": "episode",
            "q_trace_shape": "B,S,K,A",
            "replay_capacity_transitions": int(
                sequence_replay_contract["replay_capacity_transitions"]
            ),
            "warmup_transitions": int(
                sequence_replay_contract["warmup_transitions"]
            ),
            "maximum_episode_transitions": int(
                sequence_replay_contract["maximum_episode_transitions"]
            ),
            "ensemble_telemetry": (
                "valid normalized-advantage effective rank, head correlation, "
                "and scaled frozen-prior contribution at every greedy decision"
            ),
        }
    probe_cfg = ((config.get("path_c") or {}).get("probe") or {})
    metrics["path_c_probe_baseline"] = {
        "loaded": path_c_probe_base_q_net is not None,
        "checkpoint": (
            config.get("path_c_secondary_base_checkpoint")
            or probe_cfg.get("base_checkpoint")
            or probe_cfg.get("public_baseline_checkpoint")
        ),
        "registered_as_trainable_module": False,
        "gradient_source": "none_secondary_visualization_only",
    }
    if path_c_probe_base_q_net is not None:
        metrics["path_c"]["probe"]["public_baseline_loaded"] = True
    seed_summary = {}
    if sequence_training:
        terminal_seed_cfg = (config.get("training", {}).get("terminal_replay_seed") or {})
        role_seed_cfg = (config.get("training", {}).get("role_replay_seed") or {})
        if bool(terminal_seed_cfg.get("enabled", False)) or bool(
            role_seed_cfg.get("enabled", False)
        ):
            seed_summary = {
                "enabled": False,
                "reason": "transition_curriculum_incompatible_with_full_episode_sequence_replay",
            }
    else:
        seed_summary = _maybe_seed_terminal_replay(
            env,
            partners,
            option_lib,
            router,
            graph,
            config,
            args.seed,
            replay,
            args.method,
            q_net,
            target_q_net,
            belief_model,
            optimizer,
            device,
        )
    if seed_summary:
        metrics["terminal_replay_seed"] = seed_summary
    checkpoint_policy = _checkpoint_policy(config)
    metrics["checkpoint_selection"] = {
        "checkpoint_every": checkpoint_policy["checkpoint_every"],
        "select_best_by": checkpoint_policy["select_best_by"],
        "checkpoint_eval_episodes": checkpoint_policy["checkpoint_eval_episodes"],
        "best_greedy_return": None,
        "best_update": None,
        "selected_checkpoint": None,
    }
    validation_env = None
    validation_router = None
    validation_partners: list[Any] = []
    if checkpoint_policy["greedy_enabled"]:
        validation_env = _build_env(layout_name, config)
        validation_env.set_featurizer(NumpyFeaturizer(layout_graph))
        validation_router = OCV2EvidenceRouter(
            graph,
            layout_graph.cell_to_entity,
            layout_graph.region_cells,
        )
        validation_partners = _select_train_partners(option_lib, config)
    best_greedy_return = -float("inf")
    obs, state, current_partner = _reset_episode(
        env,
        evidence_buffer,
        partners,
        rng,
        args.seed,
        router,
    )
    sequence_episode_index = 0
    _start_sequence_episode(
        evidence_buffer,
        q_net,
        obs,
        state,
        option_lib,
        config,
        episode_id=f"train:{int(args.seed)}:{sequence_episode_index}",
        manifest_seed=int(args.seed),
    )
    partner_option_inferencer = _new_partner_option_inferencer(option_lib, state, config)
    _initialise_persistent_belief(evidence_buffer, args.method, belief_model, graph, device, _belief_persistence_enabled(config))
    contribution_ledger = ContributionLedger.from_config(config.get("training"))
    episode_return = 0.0
    episode_options = 0
    updates_done = 0
    wall_start = time.time()

    while updates_done < int(config["training"]["total_updates"]):
        if (
            not sequence_training
            and episode_options >= int(config["training"]["max_episode_options"])
        ):
            metrics["episode_returns"].append(float(episode_return))
            obs, state, current_partner = _reset_episode(
                env,
                evidence_buffer,
                partners,
                rng,
                args.seed,
                router,
            )
            partner_option_inferencer = _new_partner_option_inferencer(option_lib, state, config)
            _initialise_persistent_belief(evidence_buffer, args.method, belief_model, graph, device, _belief_persistence_enabled(config))
            contribution_ledger = ContributionLedger.from_config(config.get("training"))
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
            partner_id=int(getattr(current_partner, "partner_id", 0)),
            selection_stats=metrics,
            path_c_probe_base_q_net=path_c_probe_base_q_net,
        )
        probe_decision = metrics.get("path_c_probe_last_decision") or {}
        selected_probe_cost = (
            float(probe_decision.get("realized_probe_cost", 0.0))
            if probe_decision.get("selected") is True
            else 0.0
        )
        transition, done, obs = _execute_option(
            env,
            obs,
            current_partner,
            option_lib,
            router,
            evidence_buffer,
            option_id,
            graph,
            rng,
            config,
            contribution_ledger=contribution_ledger,
            partner_option_inferencer=partner_option_inferencer,
            method=args.method,
            belief_model=belief_model,
            device=device,
            truncate_at_boundary=(
                sequence_training
                and episode_options + 1
                >= int(config["training"]["max_episode_options"])
            ),
            probe_cost=selected_probe_cost,
        )
        sequence_episode = evidence_buffer.sequence_episode()
        sequence_boundary = bool(
            sequence_training
            and isinstance(sequence_episode, EpisodeEvidenceBuffer)
            and sequence_episode.closed
        )
        if sequence_training:
            if sequence_boundary:
                replay.add(sequence_episode.to_record())
        else:
            replay.add(transition)
        episode_return += _transition_training_return(transition, config)
        episode_options += 1
        metrics["option_durations"].append(int(transition.duration))
        _increment_count(metrics["termination_counts"], transition.termination_reason)
        _update_option_kind_metrics(
            metrics,
            graph.options[int(transition.option_id)].kind,
            transition.termination_reason,
        )
        _update_task_progress_metrics(metrics, transition)

        if done or sequence_boundary:
            metrics["episode_returns"].append(float(episode_return))
            obs, state, current_partner = _reset_episode(
                env,
                evidence_buffer,
                partners,
                rng,
                args.seed,
                router,
            )
            sequence_episode_index += 1
            _start_sequence_episode(
                evidence_buffer,
                q_net,
                obs,
                state,
                option_lib,
                config,
                episode_id=f"train:{int(args.seed)}:{sequence_episode_index}",
                manifest_seed=int(args.seed),
            )
            partner_option_inferencer = _new_partner_option_inferencer(option_lib, state, config)
            _initialise_persistent_belief(evidence_buffer, args.method, belief_model, graph, device, _belief_persistence_enabled(config))
            contribution_ledger = ContributionLedger.from_config(config.get("training"))
            episode_return = 0.0
            episode_options = 0

        if (
            sequence_training
            and isinstance(replay, EpisodeSequenceReplayBuffer)
            and replay.episode_count == 0
        ):
            continue
        if len(replay) < int(config["training"]["warmup_transitions"]):
            continue

        for _ in range(int(config["training"].get("updates_per_transition", 1))):
            if updates_done >= int(config["training"]["total_updates"]):
                break
            sampled = replay.sample(int(config["training"]["batch_size"]))
            batch = (
                collate_episode_records(sampled)
                if sequence_training
                else sampled
            )
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
            if (
                checkpoint_policy["greedy_enabled"]
                and updates_done % checkpoint_policy["checkpoint_every"] == 0
            ):
                if validation_env is None or validation_router is None:
                    raise RuntimeError(
                        "Greedy checkpoint validation was not initialized."
                    )
                validation = _run_greedy_validation(
                    args.method,
                    q_net,
                    belief_model,
                    validation_env,
                    option_lib,
                    validation_router,
                    validation_partners,
                    graph,
                    config,
                    seed=int(args.seed) + 1_000_003 + int(updates_done),
                    episodes=checkpoint_policy["checkpoint_eval_episodes"],
                    update_idx=updates_done,
                    device=device,
                )
                metrics["greedy_validation"].append(validation)
                if getattr(args, "save_all_checkpoints", False):
                    _save_checkpoint(
                        output_dir,
                        args.method,
                        config,
                        graph,
                        metrics,
                        q_net,
                        belief_model,
                        optimizer,
                        filename=f"checkpoint_u{int(updates_done)}.pt",
                    )
                _cs = metrics["checkpoint_selection"]
                _val_mean = float(validation["mean_return"])
                _val_ego_sole = int(validation.get("ego_sole_correct_delivery_count", 0))
                _val_partner = int(validation.get("partner_correct_delivery_count", 0))
                _cs["max_partner_correct_delivery_seen"] = max(
                    int(_cs.get("max_partner_correct_delivery_seen", 0)), _val_partner
                )
                requires_ego_selection = bool(
                    config["training"].get("require_ego_delivery_selection", False)
                )
                eligible_for_deploy = (not requires_ego_selection) or _val_ego_sole > 0
                if _val_mean > best_greedy_return:
                    _cs["best_greedy_return_seen"] = _val_mean
                    _cs["best_update_seen"] = int(updates_done)
                    _cs["best_seen_ego_sole_correct_delivery_count"] = _val_ego_sole
                    _cs["best_seen_partner_correct_delivery_count"] = _val_partner
                if eligible_for_deploy and _val_mean > best_greedy_return:
                    best_greedy_return = _val_mean
                    _cs.update(
                        {
                            "best_greedy_return": best_greedy_return,
                            "best_update": int(updates_done),
                            "selected_checkpoint": "checkpoint.pt",
                            "selected_by": "greedy",
                            "deployable_checkpoint": "checkpoint.pt",
                            "selected_ego_correct_delivery_count": int(
                                validation.get("ego_correct_delivery_count", 0)
                            ),
                            "selected_ego_sole_correct_delivery_count": _val_ego_sole,
                            "selected_partner_correct_delivery_count": _val_partner,
                            "selected_team_delivery_episode_rate": float(
                                validation.get("team_delivery_episode_rate", 0.0)
                            ),
                            "selected_ego_correct_completion_rate": float(
                                validation.get("ego_correct_completion_rate", 0.0)
                            ),
                        }
                    )
                    _save_checkpoint(
                        output_dir,
                        args.method,
                        config,
                        graph,
                        metrics,
                        q_net,
                        belief_model,
                        optimizer,
                        filename="checkpoint.pt",
                    )
                    _save_checkpoint(
                        output_dir,
                        args.method,
                        config,
                        graph,
                        metrics,
                        q_net,
                        belief_model,
                        optimizer,
                        filename="checkpoint_best.pt",
                    )
                elif not eligible_for_deploy:
                    _cs["last_ineligible_checkpoint_reason"] = "no_ego_sole_correct_delivery"

    if episode_options:
        metrics["episode_returns"].append(float(episode_return))
    metrics["checkpoint_selection"]["final_checkpoint"] = "checkpoint_final.pt"
    select_final = (
        checkpoint_policy["select_best_by"] == "final"
        or metrics["checkpoint_selection"].get("best_update") is None
    )
    final_publish_allowed = not bool(config["training"].get("require_ego_delivery_selection", False))
    if select_final:
        selected_by = checkpoint_policy["select_best_by"]
        if selected_by == "greedy":
            selected_by = "final_no_greedy_validation"
        if final_publish_allowed:
            metrics["checkpoint_selection"].update(
                {
                    "selected_checkpoint": "checkpoint.pt",
                    "selected_by": selected_by,
                    "deployable_checkpoint": "checkpoint.pt",
                }
            )
        else:
            metrics["checkpoint_selection"].update(
                {
                    "selected_checkpoint": None,
                    "selected_by": f"{selected_by}_not_publishable_without_ego_delivery_validation",
                    "deployable_checkpoint": None,
                    "final_checkpoint_publish_blocked": True,
                }
            )
    # RC free-rider guard verdict. Under ego-terminal-aware selection a checkpoint
    # is selectable only if it actually serves, so a "fail" means greedy validation
    # never produced a serving checkpoint. Recorded as a machine-checkable Type-A
    # field in metrics.json. If it fails, the final checkpoint remains available as
    # a diagnostic artifact but is not published as checkpoint.pt.
    _greedy_ran = bool(metrics.get("greedy_validation"))
    metrics["checkpoint_selection"]["free_rider_guard"] = _free_rider_guard_verdict(
        metrics["checkpoint_selection"], config, greedy_ran=_greedy_ran
    )
    if metrics["checkpoint_selection"]["free_rider_guard"] == "fail":
        metrics["checkpoint_selection"]["free_rider_diagnosis"] = _free_rider_diagnosis(
            metrics["checkpoint_selection"]
        )
    if metrics["checkpoint_selection"].get("deployable_checkpoint") != "checkpoint.pt":
        _remove_stale_deployable_checkpoints(
            output_dir,
            metrics,
            reason=str(
                metrics["checkpoint_selection"].get(
                    "selected_by",
                    "no_deployable_checkpoint_selected",
                )
            ),
        )
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
        filename="checkpoint_final.pt",
    )
    if select_final and final_publish_allowed:
        _save_checkpoint(
            output_dir,
            args.method,
            config,
            graph,
            metrics,
            q_net,
            belief_model,
            optimizer,
            filename="checkpoint.pt",
        )
    checkpoint_path = output_dir / "checkpoint.pt"
    metrics["checkpoint_load_ok"] = (
        _checkpoint_loads(checkpoint_path) if checkpoint_path.exists() else False
    )
    metrics["run_status"] = "ok"
    _write_metrics(output_dir, metrics, updates_done, wall_start, final=True)
    return metrics



def _enforce_preflight_gate(
    layout_name: str,
    config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Require an accepted preflight report before any training run.

    This is intentionally a hard gate: a run on
    a rejected layout does not test ARIS-Bellman's diagnostic setting and must be
    executed through a separate engineering-smoke harness, not through the formal
    trainer.
    """
    preflight_cfg = dict(config.get("preflight", {}))
    graph_cfg = dict(config.get("graph", {}))
    preflight_path = (
        getattr(args, "preflight_path", None)
        or preflight_cfg.get("path")
        or graph_cfg.get("preflight_path")
    )
    if not preflight_path:
        raise ValueError(
            "OvercookedV2 training requires an accepted preflight report. "
            "Run layout_diagnostics.py first and pass --preflight_path or "
            "preflight.path. Rejected-layout engineering smoke must use a separate "
            "smoke script, not train_aris.py."
        )

    data = json.loads(Path(preflight_path).read_text(encoding="utf-8"))
    entries = data if isinstance(data, list) else [data]
    selected = None
    for entry in entries:
        if str(entry.get("layout_name", entry.get("layout", ""))) == str(layout_name):
            selected = entry
            break
    if selected is None:
        raise ValueError(
            f"Preflight report {preflight_path} has no entry for layout {layout_name!r}."
        )

    accepted = bool(selected.get("accepted", False))
    result = {
        "layout": layout_name,
        "required": True,
        "accepted": accepted,
        "preflight_path": str(preflight_path),
        "stats": selected,
        "formal_experiment": accepted,
        "status": "accepted" if accepted else "rejected",
    }
    if not accepted:
        raise RuntimeError(
            "Layout failed preflight; refusing formal experiment. "
            f"Stats: {json.dumps(_jsonable(selected), sort_keys=True)}"
        )
    return result


def _build_env(layout_name: str, config: dict[str, Any]) -> OCV2Adapter:
    env_cfg = config.setdefault("env", {})
    env_cfg["observation_type"] = "default"
    # RC-2b P1: respect the configured force_path_planning (was hard-coded False, silently overriding
    # config and mislabeling experiment conditions in logs). OCV2Adapter's own default is True;
    # this wrapper keeps the explicit default force_path_planning=False unless config opts in.
    fpp = bool(env_cfg.get("force_path_planning", False))
    return OCV2Adapter(
        layout=layout_name,
        max_steps=int(env_cfg.get("max_steps", 200)),
        observation_type="default",
        agent_view_size=env_cfg.get("agent_view_size"),
        negative_rewards=bool(env_cfg.get("negative_rewards", True)),
        sample_recipe_on_delivery=bool(env_cfg.get("sample_recipe_on_delivery", True)),
        random_reset=bool(env_cfg.get("random_reset", False)),
        random_agent_positions=bool(env_cfg.get("random_agent_positions", False)),
        force_path_planning=fpp,
    )


def _build_option_lib(layout_graph: Any, config: dict[str, Any]) -> OCV2OptionLibrary:
    """Single factory so train/eval/CE construct the option library identically and the executor
    semantics flags actually reach it. RC-2b P1: train/eval previously passed only max_option_steps,
    so strict_preconditions / dynamic_budget never took effect on the main path."""
    opt_cfg = config.get("options", {}) or {}
    return OCV2OptionLibrary(
        layout_graph,
        max_option_steps=int(opt_cfg.get("max_option_steps", 20)),
        strict_preconditions=bool(opt_cfg.get("strict_preconditions", False)),
        dynamic_budget=bool(opt_cfg.get("dynamic_budget", False)),
    )


def _select_train_partners(option_lib: OCV2OptionLibrary, config: dict[str, Any]) -> list[Any]:
    """Training + greedy-validation partner pool.

    The order in training.train_partners is preserved and duplicates are allowed.
    This is intentional: the terminal-stage curriculum can be expressed as data
    distribution, not as a fallback controller or an auxiliary loss. Optional
    training.partner_sampling weights are attached to the selected partner objects
    and consumed by _reset_episode().
    """
    train_cfg = config.get("training", {}) or {}
    partners = make_training_partners(
        option_lib,
        partner_set=str(train_cfg.get("partner_set", "standard7")),
    )
    names = train_cfg.get("train_partners")
    weights_cfg = train_cfg.get("partner_sampling", {}) or {}
    groups_cfg = train_cfg.get("partner_groups", {}) or {}
    by_name = {p.name: p for p in partners}
    heldout_names = [str(name) for name in (train_cfg.get("heldout_partners") or [])]
    missing_heldout = sorted(set(heldout_names) - set(by_name))
    if missing_heldout:
        raise ValueError(
            f"heldout_partners not found: {missing_heldout}; available={sorted(by_name)}"
        )
    overlap = sorted(set(str(name) for name in (names or [])) & set(heldout_names))
    if overlap:
        raise ValueError(f"S23: train_partners overlap heldout_partners: {overlap}")
    if not names:
        if train_cfg.get("heldout_partners") and not bool(
            train_cfg.get("allow_all_partners_for_no_split", False)
        ):
            raise ValueError(
                "S23: formal held-out split claims require explicit "
                "training.train_partners. Set allow_all_partners_for_no_split=true "
                "only for labeled no-split smoke runs."
            )
        selected = list(partners)
    else:
        missing = sorted(set(names) - set(by_name))
        if missing:
            raise ValueError(
                f"train_partners not found: {missing}; available={sorted(by_name)}"
            )
        selected = [by_name[str(name)] for name in names]
    if not selected:
        raise ValueError("train_partners selected zero partners.")
    for partner in selected:
        setattr(partner, "sampling_weight", float(weights_cfg.get(partner.name, 1.0)))
        default_group = getattr(getattr(partner, "protocol", None), "curriculum_group", None)
        setattr(
            partner,
            "sampling_group",
            str(groups_cfg.get(partner.name, default_group or partner.name)),
        )
    return selected

def _build_graph(
    env: OCV2Adapter,
    layout_graph: LayoutGraph,
    option_lib: OCV2OptionLibrary,
    config: dict[str, Any],
    args: argparse.Namespace,
) -> GraphSpec:
    graph_cfg = config.get("graph", {})
    graph_path = graph_cfg.get("graph_path")
    if graph_path:
        graph = GraphSpec.from_json_dict(
            json.loads(Path(graph_path).read_text(encoding="utf-8"))
        )
        if graph.num_factors == 0:
            raise RuntimeError("Loaded GraphSpec has zero factors; cannot train ARIS.")
        validate_task_stage_coverage(graph)
        graph.metadata = {
            **(graph.metadata or {}),
            "formal_experiment": True,
            "graph_source": "precomputed_graph",
        }
        return graph

    ce_path = graph_cfg.get("ce_path")
    graph_source = "precomputed_ce"
    if not ce_path:
        raise ValueError(
            "Formal training requires graph.graph_path or graph.ce_path. "
            "Run ce_sampler.py and graph_builder.py before train_aris.py; "
            "online CE construction inside training is disabled."
        )
    ce_path_obj = Path(ce_path)
    ce_matrix = np.load(ce_path_obj)
    ce_meta_path = _metadata_sidecar_path(ce_path_obj)
    sidecar_meta: dict[str, Any] = {}
    ce_support_mask_applied = False
    if ce_meta_path is not None:
        sidecar_meta = json.loads(ce_meta_path.read_text(encoding="utf-8"))
        ce_matrix, ce_support_mask_applied = _mask_ce_matrix_with_support_audit(
            ce_matrix,
            sidecar_meta,
        )

    max_factors = int(graph_cfg.get("max_factors", 16))
    relevance_semantics = str(graph_cfg.get("relevance_semantics", "legacy_id_pair"))
    ego_selectable_v3: frozenset[int] | None = None
    if relevance_semantics in {"ego_kind_projection", "ego_complement_projection"}:
        ego_selectable_v3 = _load_ego_selectable_from_replay(graph_cfg.get("replay_path"))
        if ego_selectable_v3 is None:
            raise RuntimeError(
                f"graph.relevance_semantics={relevance_semantics!r} requires "
                "replay rows with ego_option field. Loaded replay_path="
                f"{graph_cfg.get('replay_path')!r} does not provide any. Either "
                "regenerate the CE replay (which stores per-row ego_option) or "
                "switch relevance_semantics back to 'legacy_id_pair'."
            )
    graph = build_graph_variant(
        args.graph_variant,
        layout_graph.layout_name,
        option_lib.options,
        ce_matrix,
        eta=float(graph_cfg.get("ce_eta", 0.0)),
        max_factors=max_factors,
        full_max_factors=int(graph_cfg.get("full_max_factors", max_factors)),
        overcomplete_extra_factors=int(graph_cfg.get("overcomplete_extra_factors", 0)),
        mode_config=graph_cfg.get("modes"),
        seed=args.seed,
        require_task_stage_coverage=bool(graph_cfg.get("require_task_stage_coverage", True)),
        selection_cfg=graph_cfg,
        ego_selectable=ego_selectable_v3,
        relevance_semantics=relevance_semantics,
    )
    if graph_source == "online_debug_ce":
        graph.metadata = {
            **(graph.metadata or {}),
            "formal_experiment": False,
            "graph_source": graph_source,
        }
    else:
        graph.metadata = {
            **(graph.metadata or {}),
            "formal_experiment": True,
            "graph_source": graph_source,
        }
    if ce_meta_path is not None:
        graph.metadata = _merge_metadata(graph.metadata or {}, sidecar_meta)
        if ce_support_mask_applied:
            graph.metadata["ce_unsupported_cells_masked"] = True
    if graph.num_factors == 0:
        raise RuntimeError(
            "Graph construction produced zero factors. Run layout_preflight or lower "
            "eta explicitly; empty graphs are invalid for ARIS training."
        )
    return graph


def _mask_ce_matrix_with_support_audit(
    ce_matrix: np.ndarray,
    metadata: dict[str, Any],
) -> tuple[np.ndarray, bool]:
    ce_support = metadata.get("ce_support_audit")
    if not isinstance(ce_support, dict):
        return ce_matrix, False
    estimable_raw = ce_support.get("estimable_mask")
    if estimable_raw is None:
        return ce_matrix, False
    estimable = np.asarray(estimable_raw, dtype=bool)
    if estimable.shape != ce_matrix.shape:
        raise RuntimeError(
            "CE support audit estimable_mask shape does not match ce_path matrix: "
            f"mask={estimable.shape}, ce={ce_matrix.shape}."
        )
    masked = np.asarray(ce_matrix, dtype=np.float32).copy()
    masked[~estimable] = 0.0
    return masked, True


def _merge_metadata(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = {**base, **extra}
    merged["provenance"] = {
        **(base.get("provenance", {}) or {}),
        **(extra.get("provenance", {}) or {}),
    }
    return merged


def _stamp_runtime_provenance(
    graph: GraphSpec,
    config: dict[str, Any],
    layout_graph: LayoutGraph,
    option_lib: OCV2OptionLibrary,
    partners: list[Any],
    *,
    fill_missing_hashes: bool,
) -> None:
    existing = dict((graph.metadata or {}).get("provenance", {}) or {})
    generated = runtime_provenance(
        config=config,
        layout_graph=layout_graph,
        option_lib=option_lib,
        partners=partners,
        ce_path=config.get("graph", {}).get("ce_path"),
        replay_path=config.get("graph", {}).get("replay_path"),
        graph_path=config.get("graph", {}).get("graph_path"),
    )
    generated[GRAPH_HASH_FIELD] = graph_hash_from_spec(graph)
    if not fill_missing_hashes:
        generated = {
            key: value
            for key, value in generated.items()
            if key in existing or key in {"schema_version", "git_commit"}
        }
    provenance = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        **existing,
        **generated,
    }
    graph.metadata = {
        **(graph.metadata or {}),
        "provenance": provenance,
    }


def _metadata_sidecar_path(path: Path) -> Path | None:
    candidates = (
        Path(f"{path}.metadata.json"),
        path.parent / f"{path.stem}.meta.json",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _write_graph_json(path: Path, graph: GraphSpec) -> None:
    graph_json = stamp_graph_hash(graph.to_json_dict())
    graph.metadata = graph_json["metadata"]
    _write_json(path, graph_json)


def _expected_graph_objective_metadata(config: dict[str, Any]) -> dict[str, Any]:
    training = config["training"]
    params = sparse_credit_params(training)
    expected = {
        "layout": str(config["layout"]),
        "cost_coef": float(training["cost_coef"]),
        "cost_per_step": float(training["cost_per_step"]),
        "shaped_reward_coef": float(training["shaped_reward_coef"]),
        # RC root-cause fix: the actor-specific sparse-credit mode is part of the
        # CE objective. A graph built under a different credit mode is objective-
        # inconsistent. Absent graph metadata is treated as legacy "team" in
        # _graph_objective_metadata_status so pre-fix graphs still load for team runs.
        "sparse_credit": str(params["mode"]),
        "partner_set": str(training.get("partner_set", "standard7")),
        "event_semantics_version": int(EVENT_SEMANTICS_VERSION),
        # Research reward intervention: if terminal progression shaping is enabled,
        # the CE graph must be regenerated under the same objective. Record the
        # full actor-local shaping signature so stale pre-intervention graphs reject.
        "terminal_progress_shaping": terminal_progress_params(training),
        "relevance_semantics": str(
            (config.get("graph") or {}).get("relevance_semantics", "legacy_id_pair")
        ),
    }
    if config.get("graph"):
        expected["ce_support_objective"] = _expected_ce_support_objective(config)
    # ego_correct_delivery's reward magnitude comes from explicit constants; enforce
    # them so a graph built with different constants is rejected. team / ego_delivery
    # do not use constants, so they are not part of their objective signature.
    if params["mode"] == "ego_correct_delivery":
        expected["ego_delivery_reward"] = float(params["ego_delivery_reward"])
        expected["ego_wrong_delivery_penalty"] = float(params["ego_wrong_delivery_penalty"])
    if str(params.get("mode", "")) in {"contrib_team", "role_contrib_team"}:
        expected["contribution_credit"] = {
            "contrib_scale": float(
                (training.get("contrib_team") or {}).get("contrib_scale", 1.0)
            ),
        }
    if str(params.get("mode", "")) == "role_contrib_team":
        expected["role_contrib_team"] = {
            "ego_terminal_penalty_under_claim": float(
                params["ego_terminal_penalty_under_claim"]
            ),
        }
    return expected


def _expected_ce_support_objective(config: dict[str, Any]) -> dict[str, Any]:
    training = config["training"]
    graph_cfg = config.get("graph", {}) or {}
    sparse_ce_support = bool(graph_cfg.get("sparse_ce_support", False))
    support_objective_cfg = graph_cfg.get("support_objective", graph_cfg.get("ce_support_objective"))
    if isinstance(support_objective_cfg, dict):
        support_objective = str(
            support_objective_cfg.get(
                "support_objective",
                "sparse_excluding_terminal_progress" if sparse_ce_support else "training_reward_sum",
            )
        )
    else:
        support_objective = str(
            support_objective_cfg
            or ("sparse_excluding_terminal_progress" if sparse_ce_support else "training_reward_sum")
        )
    return {
        "gamma": float(graph_cfg.get("ce_gamma", training.get("gamma", 0.99))),
        "horizon_options": int(
            graph_cfg.get(
                "local_return_horizon_options",
                graph_cfg.get(
                    "horizon_options",
                    training.get("local_return_horizon_options", 5),
                ),
            )
        ),
        "min_weight": float(graph_cfg.get("ce_min_weight", 20.0)),
        "min_actor_delivery_support": int(graph_cfg.get("min_actor_delivery_support", 1)),
        "support_objective": support_objective,
        "sparse_ce_support": sparse_ce_support,
    }


def _graph_objective_metadata_status(
    graph: GraphSpec,
    config: dict[str, Any],
    *,
    layout_graph: LayoutGraph | None = None,
    option_lib: OCV2OptionLibrary | None = None,
    partners: list[Any] | None = None,
) -> dict[str, Any]:
    metadata = graph.metadata or {}
    try:
        expected = _expected_graph_objective_metadata(config)
    except KeyError as exc:
        return {
            "reward_scale_verified": False,
            "expected": {},
            "observed": {},
            "missing": [],
            "mismatches": {
                "config": {
                    "graph": None,
                    "config": f"missing required config key {exc}",
                }
            },
            "event_semantics_version": metadata.get("event_semantics_version"),
        }
    # sparse_credit is back-compatible: a legacy graph with no sparse_credit key
    # is treated as "team" (the historical objective), so it is NOT a hard-miss.
    # It only mismatches when the config asks for a non-team mode the graph lacks.
    optional_legacy_keys = {"sparse_credit", "partner_set", "relevance_semantics"}
    if not bool((expected.get("terminal_progress_shaping") or {}).get("enabled", False)):
        optional_legacy_keys.add("terminal_progress_shaping")
    missing = [
        key for key in expected
        if key not in metadata
        and key not in optional_legacy_keys
    ]
    mismatches: dict[str, dict[str, Any]] = {}

    for key, expected_value in expected.items():
        if key in missing:
            continue
        if key in optional_legacy_keys and key not in metadata:
            continue
        observed = metadata.get(key)
        if key in {
            "cost_coef",
            "cost_per_step",
            "shaped_reward_coef",
            "ego_delivery_reward",
            "ego_wrong_delivery_penalty",
        }:
            try:
                matches = math.isclose(
                    float(observed),
                    float(expected_value),
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                )
            except (TypeError, ValueError):
                matches = False
        elif key == "event_semantics_version":
            try:
                matches = int(observed) == int(expected_value)
            except (TypeError, ValueError):
                matches = False
        elif key == "sparse_credit":
            observed = metadata.get(key, "team")
            matches = str(observed) == str(expected_value)
        elif key == "partner_set":
            observed = metadata.get(key, "standard7")
            matches = str(observed) == str(expected_value)
        elif key == "terminal_progress_shaping":
            matches = json.dumps(observed, sort_keys=True) == json.dumps(
                expected_value, sort_keys=True
            )
        elif key == "contribution_credit":
            matches = json.dumps(observed, sort_keys=True) == json.dumps(
                expected_value, sort_keys=True
            )
        elif key == "role_contrib_team":
            matches = json.dumps(observed, sort_keys=True) == json.dumps(
                expected_value, sort_keys=True
            )
        elif key == "ce_support_objective":
            matches = json.dumps(observed, sort_keys=True) == json.dumps(
                expected_value, sort_keys=True
            )
        elif key == "relevance_semantics":
            observed = metadata.get(key, "legacy_id_pair")
            matches = str(observed) == str(expected_value)
        else:
            matches = str(observed) == str(expected_value)
        if not matches:
            mismatches[key] = {
                "graph": observed,
                "config": expected_value,
            }

    if str(graph.layout_name) != str(expected["layout"]):
        mismatches["layout_name"] = {
            "graph": graph.layout_name,
            "config": expected["layout"],
        }

    require_ce_support = bool(config.get("graph"))
    if require_ce_support:
        ce_support = metadata.get("ce_support_audit")
        if not isinstance(ce_support, dict) or ce_support.get("schema_version") != "ce_support_audit_v1":
            if "ce_support_audit" not in missing:
                missing.append("ce_support_audit")
        else:
            required_ce_keys = {"weight_sum", "estimable_mask", "skipped_mask", "measured_zero_mask", "min_weight"}
            absent_ce_keys = sorted(key for key in required_ce_keys if key not in ce_support)
            if absent_ce_keys:
                mismatches["ce_support_audit"] = {
                    "graph": sorted(ce_support),
                    "config": f"missing CE support keys {absent_ce_keys}",
                }
            expected_ce = expected.get("ce_support_objective")
            if isinstance(expected_ce, dict):
                nested_mismatches: dict[str, dict[str, Any]] = {}
                for field in ("min_weight", "gamma"):
                    try:
                        if not math.isclose(
                            float(ce_support.get(field)),
                            float(expected_ce[field]),
                            rel_tol=1e-9,
                            abs_tol=1e-12,
                        ):
                            nested_mismatches[field] = {
                                "graph": ce_support.get(field),
                                "config": expected_ce[field],
                            }
                    except (TypeError, ValueError):
                        nested_mismatches[field] = {
                            "graph": ce_support.get(field),
                            "config": expected_ce[field],
                        }
                try:
                    if int(ce_support.get("horizon_options")) != int(expected_ce["horizon_options"]):
                        nested_mismatches["horizon_options"] = {
                            "graph": ce_support.get("horizon_options"),
                            "config": expected_ce["horizon_options"],
                        }
                except (TypeError, ValueError):
                    nested_mismatches["horizon_options"] = {
                        "graph": ce_support.get("horizon_options"),
                        "config": expected_ce["horizon_options"],
                    }
                expected_reward_objective = str(expected.get("sparse_credit", "team"))
                if str(ce_support.get("reward_objective")) != expected_reward_objective:
                    nested_mismatches["reward_objective"] = {
                        "graph": ce_support.get("reward_objective"),
                        "config": expected_reward_objective,
                    }
                if str(ce_support.get("support_objective")) != str(expected_ce["support_objective"]):
                    nested_mismatches["support_objective"] = {
                        "graph": ce_support.get("support_objective"),
                        "config": expected_ce["support_objective"],
                    }
                if nested_mismatches:
                    mismatches["ce_support_audit_params"] = nested_mismatches
        if not bool(metadata.get("ce_unsupported_cells_masked", False)):
            mismatches["ce_unsupported_cells_masked"] = {
                "graph": metadata.get("ce_unsupported_cells_masked"),
                "config": True,
            }

    provenance_status = _graph_provenance_status(
        graph,
        config,
        layout_graph=layout_graph,
        option_lib=option_lib,
        partners=partners,
    )

    return {
        "reward_scale_verified": (
            not missing
            and not mismatches
            and not provenance_status["mismatches"]
        ),
        "expected": expected,
        "observed": {key: metadata.get(key) for key in expected},
        "missing": missing,
        "mismatches": mismatches,
        "event_semantics_version": metadata.get("event_semantics_version"),
        "provenance": provenance_status,
    }


def _enforce_graph_objective_metadata(
    graph: GraphSpec,
    config: dict[str, Any],
    *,
    layout_graph: LayoutGraph | None = None,
    option_lib: OCV2OptionLibrary | None = None,
    partners: list[Any] | None = None,
) -> None:
    status = _graph_objective_metadata_status(
        graph,
        config,
        layout_graph=layout_graph,
        option_lib=option_lib,
        partners=partners,
    )
    provenance = status.get("provenance", {})
    if provenance.get("missing"):
        warnings.warn(
            "Loaded CE graph metadata has no provenance hashes for "
            f"{provenance['missing']}; treating as legacy graph metadata.",
            RuntimeWarning,
        )
    if status["reward_scale_verified"]:
        return
    raise RuntimeError(
        "Loaded CE graph metadata is missing or inconsistent with the training "
        "objective. Regenerate CE with experiments/overcooked_v2/scripts/"
        "run_ce_pipeline.py using the current config, then set graph.graph_path "
        "to the regenerated graph.json before training. "
        f"Missing: {status['missing']}; mismatches: "
        f"{json.dumps(_jsonable(status['mismatches']), sort_keys=True)}; "
        "provenance mismatches: "
        f"{json.dumps(_jsonable(provenance.get('mismatches', {})), sort_keys=True)}"
    )


def _graph_provenance_status(
    graph: GraphSpec,
    config: dict[str, Any],
    *,
    layout_graph: LayoutGraph | None = None,
    option_lib: OCV2OptionLibrary | None = None,
    partners: list[Any] | None = None,
) -> dict[str, Any]:
    observed = dict((graph.metadata or {}).get("provenance", {}) or {})
    expected = _expected_provenance_hashes(
        graph,
        config,
        layout_graph=layout_graph,
        option_lib=option_lib,
        partners=partners,
    )
    # S28: graph identity is compared CONTENT-to-content, not record-to-record.
    # The embedded provenance record is stamped after runtime metadata additions
    # (formal_experiment/graph_source/preflight_gate), so the recorded hash can
    # NEVER equal the on-disk file's hash on the eval path (checkpoint-embedded
    # graph) — a guaranteed false positive first caught by the LDS-B2 hard gate.
    # Recomputing both sides with graph_content_hash keeps the real invariant
    # ("the graph in use is the file the config points to") fully enforced:
    # any factor/CE/semantic-metadata drift still mismatches.
    _graph_path = (config.get("graph", {}) or {}).get("graph_path")
    if _graph_path and Path(_graph_path).exists() and hasattr(graph, "to_json_dict"):
        expected[GRAPH_HASH_FIELD] = graph_content_hash(
            json.loads(Path(_graph_path).read_text(encoding="utf-8"))
        )
        observed[GRAPH_HASH_FIELD] = graph_content_hash(graph.to_json_dict())
    missing = sorted(key for key in expected if key not in observed)
    mismatches = {}
    for key, expected_value in expected.items():
        if key not in observed:
            continue
        if observed.get(key) != expected_value:
            mismatches[key] = {
                "graph": observed.get(key),
                "config": expected_value,
            }
    return {
        "expected": expected,
        "observed": {key: observed.get(key) for key in expected},
        "missing": missing,
        "mismatches": mismatches,
    }


def _expected_provenance_hashes(
    graph: GraphSpec,
    config: dict[str, Any],
    *,
    layout_graph: LayoutGraph | None = None,
    option_lib: OCV2OptionLibrary | None = None,
    partners: list[Any] | None = None,
) -> dict[str, Any]:
    graph_cfg = config.get("graph", {})
    provenance = runtime_provenance(
        config=config,
        layout_graph=layout_graph,
        option_lib=option_lib,
        partners=partners or [],
        ce_path=graph_cfg.get("ce_path"),
        replay_path=graph_cfg.get("replay_path"),
        graph_path=graph_cfg.get("graph_path"),
    )
    if layout_graph is None:
        provenance.pop("layout_parse_sha256", None)
    if option_lib is None:
        provenance.pop("option_library_sha256", None)
    if partners is None:
        provenance.pop("partner_pool_sha256", None)
    if not graph_cfg.get("graph_path") and hasattr(graph, "to_json_dict"):
        provenance[GRAPH_HASH_FIELD] = graph_hash_from_spec(graph)
    return {
        key: value
        for key, value in provenance.items()
        if key.endswith("_sha256") and value is not None
    }



def _load_path_c_probe_baseline_q_net(
    config: dict[str, Any],
    obs_dim: Any,
    graph: GraphSpec,
    device: torch.device,
) -> nn.Module | None:
    path_c = config.get("path_c") or {}
    probe = path_c.get("probe") or {}
    if not bool(probe.get("enable", False)):
        return None
    if int((path_c.get("ensemble") or {}).get("n_heads", 1)) <= 1:
        return None
    ckpt_path = (
        config.get("path_c_secondary_base_checkpoint")
        or probe.get("base_checkpoint")
        or probe.get("residual_baseline_checkpoint")
        or probe.get("public_baseline_checkpoint")
    )
    if ckpt_path in {None, ""}:
        # Public-baseline residuals are secondary diagnostics in Path C version
        # 3.  The primary normalized-advantage selector has no baseline
        # dependency, so absence of this optional checkpoint must not block the
        # active training path.
        return None
    path = Path(str(ckpt_path))
    if not path.exists():
        raise FileNotFoundError(f"Path C public baseline checkpoint not found: {path}")
    expected_sha256 = (
        config.get("path_c_secondary_base_checkpoint_sha256")
        or probe.get("base_checkpoint_sha256")
    )
    observed_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 not in {None, ""} and observed_sha256 != str(expected_sha256):
        raise ValueError(
            "Path C public baseline checkpoint SHA-256 does not match the frozen config."
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if str(checkpoint.get("method")) != "base_only":
        raise ValueError(
            "path_c.probe.base_checkpoint must be a base_only checkpoint; "
            f"got method={checkpoint.get('method')!r}."
        )
    base_graph = GraphSpec.from_json_dict(checkpoint["graph"])
    if graph_hash_from_spec(base_graph) != graph_hash_from_spec(graph):
        raise ValueError(
            "Path C probe baseline graph does not match the active graph."
        )
    base_option_order = [(option.id, option.name, option.kind) for option in base_graph.options]
    active_option_order = [(option.id, option.name, option.kind) for option in graph.options]
    if base_option_order != active_option_order:
        raise ValueError("Path C probe baseline option order does not match the active graph.")
    base_config = checkpoint.get("config") or config
    base_q_net = _build_q_network("base_only", obs_dim, graph, base_config).to(device)
    base_q_net.load_state_dict(checkpoint["q_net"])
    base_q_net.eval()
    for param in base_q_net.parameters():
        param.requires_grad_(False)
    return base_q_net


def _path_c_attach_probe_baseline_if_configured(
    q_net: nn.Module,
    obs_dim: Any,
    graph: GraphSpec,
    config: dict[str, Any],
    device: torch.device,
    *,
    config_path: Path | str | None = None,
) -> nn.Module | None:
    """Attach an optional public-state baseline for secondary visualization.

    The baseline is intentionally installed with ``object.__setattr__`` instead of
    as a registered child module: it is not part of the trainable ego, cannot
    receive optimizer updates, and never drives the primary probe selector.
    """
    base_q_net = _load_path_c_probe_baseline_q_net(config, obs_dim, graph, device)
    if base_q_net is None:
        return None
    object.__setattr__(q_net, "path_c_base_q_net", base_q_net)
    probe = (config.get("path_c") or {}).get("probe") or {}
    object.__setattr__(
        q_net,
        "path_c_base_checkpoint",
        probe.get("base_checkpoint")
        or probe.get("residual_baseline_checkpoint")
        or probe.get("public_baseline_checkpoint")
        or config_path,
    )
    return base_q_net


_PATH_C_PROGRESS_EVENT_NAMES = (
    "collision_or_block",
    "delivery_event",
    "wrong_delivery_event",
    "ego_delivery_event",
    "partner_delivery_event",
    "ego_sole_correct_delivery",
    "pot_became_full",
    "pot_became_cooked",
    "pot_became_ready",
    "plate_picked",
    "soup_picked",
    "button_pressed",
    "recipe_indicator_event",
    "object_pickup_or_drop",
)


def _uses_sequence_q(q_net: nn.Module) -> bool:
    return isinstance(q_net, RecurrentEnsembleQ)


def _validate_sequence_replay_contract(
    config: dict[str, Any],
) -> dict[str, int]:
    """Reject recurrent replay settings that can never reach a TD update."""

    training = config.get("training") or {}

    def require_integer(name: str, *, minimum: int) -> int:
        raw = training.get(name)
        if isinstance(raw, bool):
            raise ValueError(f"training.{name} must be an integer.")
        try:
            value = int(raw)
            numeric = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"training.{name} must be an integer.") from exc
        if not math.isfinite(numeric) or numeric != float(value) or value < minimum:
            raise ValueError(
                f"training.{name} must be an integer at least {minimum}."
            )
        return value

    replay_size = require_integer("replay_size", minimum=1)
    warmup = require_integer("warmup_transitions", minimum=0)
    maximum_episode = require_integer("max_episode_options", minimum=1)
    if warmup > replay_size:
        raise ValueError(
            "Sequence replay can never reach training.warmup_transitions because "
            "training.warmup_transitions exceeds training.replay_size."
        )
    if maximum_episode > replay_size:
        raise ValueError(
            "A complete sequence episode can exceed training.replay_size; refusing "
            "a configuration that would truncate or reject recurrent history."
        )
    return {
        "replay_capacity_transitions": replay_size,
        "warmup_transitions": warmup,
        "maximum_episode_transitions": maximum_episode,
    }


def _sequence_architecture_enabled(
    method: str,
    config: dict[str, Any],
) -> bool:
    if method == "global_gru":
        return True
    if method != "aris_bellman":
        return False
    path_c = config.get("path_c") or {}
    ensemble = path_c.get("ensemble") or {}
    evidence = path_c.get("evidence_spec") or {}
    return (
        str(ensemble.get("architecture", "legacy_default_off"))
        == "recurrent_sequence_v1"
        and bool(evidence.get("enable", False))
    )


def _flat_observation_dim(obs_dim: Any) -> int:
    if isinstance(obs_dim, (int, np.integer)):
        value = int(obs_dim)
    else:
        shape = tuple(int(dim) for dim in obs_dim)
        value = int(np.prod(np.asarray(shape, dtype=np.int64)))
    if value <= 0:
        raise ValueError(f"Invalid observation dimension {obs_dim!r}.")
    return value


def _build_ego_evidence_spec(
    obs_dim: Any,
    graph: GraphSpec,
    config: dict[str, Any],
) -> EgoEvidenceSpecV1:
    path_c = config.get("path_c") or {}
    evidence_cfg = path_c.get("evidence_spec") or {}
    schema_version = str(
        evidence_cfg.get("schema_version", "ego_evidence_spec_v1")
    )
    if schema_version != "ego_evidence_spec_v1":
        raise ValueError(
            "Path C evidence schema must be ego_evidence_spec_v1; "
            f"got {schema_version!r}."
        )
    max_primitive_steps = int(
        evidence_cfg.get("max_primitive_actions_per_decision", 64)
    )
    if max_primitive_steps <= 0:
        raise ValueError(
            "path_c.evidence_spec.max_primitive_actions_per_decision must be positive."
        )
    if bool(evidence_cfg.get("enable", False)):
        max_episode_decisions = int(evidence_cfg.get("max_episode_decisions", 0))
        if max_episode_decisions <= 0:
            raise ValueError(
                "Active Path C requires a positive evidence_spec.max_episode_decisions."
            )
        if int(config.get("training", {}).get("max_episode_options", 0)) > max_episode_decisions:
            raise ValueError(
                "training.max_episode_options exceeds the frozen Path C episode length."
            )
    graph_max_steps = max((int(option.max_steps) for option in graph.options), default=0)
    if graph_max_steps > max_primitive_steps:
        raise ValueError(
            "The frozen ego evidence window is shorter than an option's maximum "
            "primitive duration; refusing to truncate observable history."
        )
    option_ids = [int(option.id) for option in graph.options]
    if option_ids != list(range(graph.num_options)):
        raise ValueError(
            "EgoEvidenceSpecV1 requires option ids to match the frozen action order."
        )
    primitive_actions = (
        ("right", _OCActions.right),
        ("down", _OCActions.down),
        ("left", _OCActions.left),
        ("up", _OCActions.up),
        ("stay", _OCActions.stay),
        ("interact", _OCActions.interact),
    )
    primitive_by_id = sorted(
        ((int(action), name) for name, action in primitive_actions),
        key=lambda item: item[0],
    )
    if [item[0] for item in primitive_by_id] != list(range(len(primitive_by_id))):
        raise ValueError("Overcooked primitive action ids are not contiguous from zero.")
    spec = EgoEvidenceSpecV1(
        observation_dim=_flat_observation_dim(obs_dim),
        num_primitive_actions=len(primitive_by_id),
        num_options=graph.num_options,
        max_primitive_steps_per_decision=max_primitive_steps,
        progress_event_dim=len(_PATH_C_PROGRESS_EVENT_NAMES),
        observation_schema="ocv2_agent0_public_observation_flat_v1",
        primitive_action_names=tuple(name for _, name in primitive_by_id),
        option_names=tuple(str(option.name) for option in graph.options),
        progress_event_names=_PATH_C_PROGRESS_EVENT_NAMES,
    )
    frozen_payload = evidence_cfg.get("frozen_spec")
    if frozen_payload is not None:
        if not isinstance(frozen_payload, dict):
            raise ValueError("path_c.evidence_spec.frozen_spec must be a mapping.")
        frozen_spec = EgoEvidenceSpecV1.from_dict(frozen_payload)
        if frozen_spec.to_dict() != spec.to_dict():
            raise ValueError(
                "Environment-derived evidence dimensions or vocabulary differ "
                "from the frozen EgoEvidenceSpecV1."
            )
        spec = frozen_spec
    expected_sha256 = (path_c.get("preregistration") or {}).get(
        "evidence_spec_sha256"
    )
    if expected_sha256 not in {None, ""} and str(expected_sha256) != spec.sha256():
        raise ValueError(
            "Runtime EgoEvidenceSpecV1 does not match the preregistered evidence hash."
        )
    return spec


def _build_sequence_q_network(
    method: str,
    obs_dim: Any,
    hidden_dim: int,
    graph: GraphSpec,
    config: dict[str, Any],
) -> RecurrentEnsembleQ:
    spec = _build_ego_evidence_spec(obs_dim, graph, config)
    ensemble_cfg = ((config.get("path_c") or {}).get("ensemble") or {})
    if method == "global_gru":
        n_heads = 1
        prior_scale = 0.0
    else:
        n_heads = int(ensemble_cfg.get("n_heads", 1))
        prior_scale = float(ensemble_cfg.get("prior_scale", 0.0))
    model = RecurrentEnsembleQ(
        evidence_dim=spec.evidence_dim,
        n_actions=graph.num_options,
        n_heads=n_heads,
        encoder_dim=hidden_dim,
        recurrent_dim=hidden_dim,
        prior_scale=prior_scale,
        prior_seed=0,
        evidence_spec_sha256=spec.sha256(),
    )
    object.__setattr__(model, "ego_evidence_spec", spec)
    object.__setattr__(model, "sequence_method", str(method))
    return model


def _build_q_network(
    method: str,
    obs_dim: Any,
    graph: GraphSpec,
    config: dict[str, Any],
) -> nn.Module:
    hidden_dim = int(config["training"]["hidden_dim"])
    encoder_type = str(config["training"].get("obs_encoder", "auto"))
    vb = config["training"].get("value_bound", {}) or {}
    non_aris_q_bound_vmax = (
        float(vb.get("vmax", 20.0))
        if bool(vb.get("enabled", False)) and bool(vb.get("apply_to_all_methods", False))
        else None
    )
    if _sequence_architecture_enabled(method, config):
        return _build_sequence_q_network(method, obs_dim, hidden_dim, graph, config)
    if method == "aris_bellman":
        path_c = config.get("path_c") or {}
        ensemble_cfg = path_c.get("ensemble") or {}
        n_heads = int(ensemble_cfg.get("n_heads", 1))
        if n_heads > 1:
            return EnsembleArisBellmanQNetwork(
                obs_dim,
                hidden_dim,
                graph,
                n_heads=n_heads,
                encoder_type=encoder_type,
                advantage_norm=str(config["training"].get("advantage_norm", "none")),
                value_bound=bool(vb.get("enabled", False)),
                vmax=float(vb.get("vmax", 20.0)),
                base_bound=(float(vb["base_bound"]) if vb.get("base_bound") is not None else None),
                adv_bound=(float(vb["adv_bound"]) if vb.get("adv_bound") is not None else None),
                adv_unit=float(vb.get("adv_unit", 1.0)),
                disagreement_stat=str(ensemble_cfg.get("disagreement_stat", "variance")),
                prior_scale=float(ensemble_cfg.get("prior_scale", 0.0)),
            )
        return ArisBellmanQNetwork(
            obs_dim,
            hidden_dim,
            graph,
            encoder_type,
            advantage_norm=str(config["training"].get("advantage_norm", "none")),
            value_bound=bool(vb.get("enabled", False)),
            vmax=float(vb.get("vmax", 20.0)),
            base_bound=(float(vb["base_bound"]) if vb.get("base_bound") is not None else None),
            adv_bound=(float(vb["adv_bound"]) if vb.get("adv_bound") is not None else None),
            adv_unit=float(vb.get("adv_unit", 1.0)),
        )
    if method == "base_only":
        return BaseOnlyQNetwork(
            obs_dim,
            hidden_dim,
            graph.num_options,
            encoder_type,
            q_bound_vmax=non_aris_q_bound_vmax,
        )
    if method == "flat_factor":
        return FlatFactorQNetwork(
            obs_dim,
            hidden_dim,
            graph.num_options,
            graph.num_factors,
            graph.max_modes,
            encoder_type,
            q_bound_vmax=non_aris_q_bound_vmax,
        )
    if method == "global_gru":  # guarded by _sequence_architecture_enabled
        raise RuntimeError("global_gru must use the shared recurrent sequence core.")
    if method == "partner_id_q":
        return PartnerIDQNetwork(
            obs_dim,
            hidden_dim,
            graph.num_options,
            int(config["training"].get("num_partners", 6)),
            encoder_type,
            q_bound_vmax=non_aris_q_bound_vmax,
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
    partners = make_training_partners(
        option_lib,
        partner_set=str(config.get("training", {}).get("partner_set", "standard7")),
    )
    evidence_buffer = EvidenceBuffer(
        num_factors=router.graph.num_factors,
        window=int(config["training"]["evidence_window"]),
        evidence_dim=D_EVID,
    )
    obs, state, current_partner = _reset_episode(
        env,
        evidence_buffer,
        partners,
        rng,
        args.seed,
        router,
    )
    partner_option_inferencer = _new_partner_option_inferencer(option_lib, state, config)
    contribution_ledger = ContributionLedger.from_config(config.get("training"))
    metrics = _empty_metrics(args.method, router.graph, output_dir)
    episode_return = 0.0
    episode_options = 0
    wall_start = time.time()

    for update_idx in range(int(config["training"]["total_updates"])):
        if episode_options >= int(config["training"]["max_episode_options"]):
            metrics["episode_returns"].append(float(episode_return))
            obs, state, current_partner = _reset_episode(
                env,
                evidence_buffer,
                partners,
                rng,
                args.seed,
                router,
            )
            partner_option_inferencer = _new_partner_option_inferencer(option_lib, state, config)
            contribution_ledger = ContributionLedger.from_config(config.get("training"))
            episode_return = 0.0
            episode_options = 0

        option_id = _sample_valid_option(
            option_lib,
            env.state,
            0,
            rng,
            selection_stats=metrics,
        )
        transition, done, obs = _execute_option(
            env,
            obs,
            current_partner,
            option_lib,
            router,
            evidence_buffer,
            option_id,
            router.graph,
            rng,
            config,
            contribution_ledger=contribution_ledger,
            partner_option_inferencer=partner_option_inferencer,
            method=args.method,
            belief_model=None,
            device=None,
        )
        episode_return += _transition_training_return(transition, config)
        episode_options += 1
        metrics["option_durations"].append(int(transition.duration))
        _increment_count(metrics["termination_counts"], transition.termination_reason)
        _update_option_kind_metrics(
            metrics,
            router.graph.options[int(transition.option_id)].kind,
            transition.termination_reason,
        )
        _update_task_progress_metrics(metrics, transition)
        if done:
            metrics["episode_returns"].append(float(episode_return))
            obs, state, current_partner = _reset_episode(
                env,
                evidence_buffer,
                partners,
                rng,
                args.seed,
                router,
            )
            partner_option_inferencer = _new_partner_option_inferencer(option_lib, state, config)
            contribution_ledger = ContributionLedger.from_config(config.get("training"))
            episode_return = 0.0
            episode_options = 0
        if update_idx % int(config["training"]["log_interval"]) == 0:
            _write_metrics(output_dir, metrics, update_idx, wall_start)

    if episode_options:
        metrics["episode_returns"].append(float(episode_return))
    _write_metrics(output_dir, metrics, int(config["training"]["total_updates"]), wall_start)
    return metrics


def _sequence_spec(q_net: nn.Module) -> EgoEvidenceSpecV1:
    if not _uses_sequence_q(q_net):
        raise TypeError("A recurrent sequence Q-network is required.")
    spec = getattr(q_net, "ego_evidence_spec", None)
    if not isinstance(spec, EgoEvidenceSpecV1):
        raise RuntimeError("The sequence Q-network is missing EgoEvidenceSpecV1.")
    if spec.sha256() != q_net.evidence_spec_sha256:
        raise RuntimeError("The sequence model and attached evidence spec disagree.")
    return spec


def _start_sequence_episode(
    evidence_buffer: EvidenceBuffer,
    q_net: nn.Module,
    obs: dict[str, np.ndarray],
    state: Any,
    option_lib: OCV2OptionLibrary,
    config: dict[str, Any],
    *,
    episode_id: str,
    manifest_seed: int,
) -> EpisodeEvidenceBuffer | None:
    if not _uses_sequence_q(q_net):
        evidence_buffer.set_sequence_episode(None)
        return None
    spec = _sequence_spec(q_net)
    ensemble_cfg = ((config.get("path_c") or {}).get("ensemble") or {})
    bootstrap_p = (
        float(ensemble_cfg.get("bootstrap_p", 1.0))
        if int(q_net.n_heads) > 1
        else 1.0
    )
    bootstrap = EpisodeBootstrapRecord.sample(
        episode_id=str(episode_id),
        n_heads=int(q_net.n_heads),
        bootstrap_p=bootstrap_p,
        manifest_seed=int(manifest_seed) & ((1 << 63) - 1),
    )
    valid_actions = np.asarray(option_lib.valid_options(state, 0), dtype=bool)
    initial = spec.encode_decision(
        observation=_obs_vector(obs, "agent_0"),
        ego_primitive_actions=(),
        partner_primitive_actions=(),
        ego_option_id=None,
        duration=0,
        reward=0.0,
        progress_events=np.zeros(spec.progress_event_dim, dtype=np.float32),
        valid_actions=valid_actions,
        terminated=False,
        truncated=False,
    )
    episode = EpisodeEvidenceBuffer(spec, bootstrap)
    episode.start(initial, valid_actions)
    evidence_buffer.set_sequence_episode(episode)
    return episode


def _path_c_progress_event_vector(
    event: Any,
    spec: EgoEvidenceSpecV1,
) -> np.ndarray:
    if tuple(spec.progress_event_names) != _PATH_C_PROGRESS_EVENT_NAMES:
        raise ValueError("Unexpected Path C progress-event vocabulary.")
    return np.asarray(
        [float(bool(getattr(event, name, False))) for name in spec.progress_event_names],
        dtype=np.float32,
    )


def _sequence_step_return(step_reward: float, config: dict[str, Any]) -> float:
    value_bound = (config.get("training", {}).get("value_bound") or {})
    reward_scale = float(value_bound.get("reward_scale", 1.0))
    if not math.isfinite(reward_scale) or reward_scale <= 0.0:
        raise ValueError("training.value_bound.reward_scale must be positive and finite.")
    step_cost = float(config["training"].get("cost_per_step", 1.0))
    cost_coef = float(config["training"].get("cost_coef", 0.0))
    return float((float(step_reward) - cost_coef * step_cost) / reward_scale)


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
    *,
    contribution_ledger: ContributionLedger | None = None,
    partner_option_inferencer: PartnerOptionInferencer | None = None,
    method: str | None = None,
    belief_model: FactorLocalBeliefModel | None = None,
    device: torch.device | None = None,
    truncate_at_boundary: bool = False,
    probe_cost: float = 0.0,
) -> tuple[OptionTransition, bool, dict[str, np.ndarray]]:
    opt = option_lib.options[int(option_id)]
    runtime = OptionRuntime(
        option_id=int(option_id),
        start_pos=get_agent_pos(env.state, 0),
    )
    evidence_t = evidence_buffer.snapshot()
    evidence_mask_t = evidence_buffer.snapshot_mask()
    evidence_len_t = evidence_buffer.length()
    belief_hidden_t = evidence_buffer.belief_window_base_snapshot()
    obs_feat_t = _obs_vector(obs, "agent_0")
    expected_cost = option_lib.expected_cost(env.state, 0, int(option_id))
    if not math.isfinite(float(probe_cost)) or float(probe_cost) < 0.0:
        raise ValueError("Path C probe cost must be finite and non-negative.")
    reward_sum = -float(probe_cost)
    realized_cost = 0.0
    duration = 0
    done = False
    termination_reason = "running"
    event_summary = _empty_event_summary()
    event_summary["path_c_probe_cost"] = float(probe_cost)
    sequence_episode = evidence_buffer.sequence_episode()
    sequence_decision = (
        DecisionEvidenceBuffer(sequence_episode.spec)
        if isinstance(sequence_episode, EpisodeEvidenceBuffer)
        else None
    )

    _budget = option_lib.option_budget(env.state, 0, int(option_id))
    while duration < _budget:
        _ostep = option_primitive_step(
            env,
            option_lib,
            int(option_id),
            partner,
            obs,
            rng,
            partner_option_inferencer=partner_option_inferencer,
        )
        ego_action = _ostep.ego_action
        partner_action = _ostep.partner_action
        prev_state = _ostep.prev_state
        step = _ostep.step
        event = _ostep.event
        _accumulate_event_summary(event_summary, event)
        if contribution_ledger is not None:
            contribution_ledger.update(event, ego_option_kind=str(opt.kind))
        ego_contributed = False
        if contribution_ledger is not None:
            ego_contributed = contribution_ledger.query_and_reset_on_delivery(event)
        step_reward = _training_reward(
            step, config, "agent_0", event,
            ego_contributed=ego_contributed,
        )
        reward_sum += step_reward
        realized_cost += float(config["training"].get("cost_per_step", 1.0))
        duration += 1
        if sequence_decision is not None:
            evidence_step_return = _sequence_step_return(step_reward, config)
            if duration == 1 and float(probe_cost) > 0.0:
                reward_scale = float(
                    (config.get("training", {}).get("value_bound") or {}).get(
                        "reward_scale", 1.0
                    )
                )
                evidence_step_return -= float(probe_cost) / reward_scale
            sequence_decision.append_primitive(
                int(ego_action),
                int(partner_action.primitive_action),
                reward=evidence_step_return,
                progress_event=_path_c_progress_event_vector(
                    event,
                    sequence_decision.spec,
                ),
            )
        x_f = router.route(
            event,
            ego_option_id=int(option_id),
            ego_option_elapsed=duration,
            ego_option_max_steps=opt.max_steps,
        )
        evidence_buffer.append(x_f)
        _advance_persistent_belief(
            evidence_buffer,
            method,
            belief_model,
            graph,
            device,
            x_f,
            _belief_persistence_enabled(config),
        )
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

    # Diagnostic label fix: with dynamic option budgets the loop can exit because
    # `duration` reached `_budget` without a natural termination, leaving the last
    # option_terminated() verdict "running" as the recorded terminal reason. Relabel
    # it so option stats show a real terminal cause (does not change control flow).
    if not done and termination_reason == "running":
        termination_reason = "budget_exhausted"

    # Push the option-level FAILURE signal to evidence when the option did not succeed.
    # Value-sufficient belief update: repeated failure of an option should shift the
    # posterior on the factors touching that option, allowing the model to route away
    # from it without a probe/selector.
    _failed = termination_reason in {"budget_exhausted", "max_steps", "env_max_steps"}
    if _failed and duration > 0:
        x_fail = router.route_failure_boundary(
            ego_option_id=int(option_id),
            ego_option_elapsed=duration,
            ego_option_max_steps=opt.max_steps,
        )
        evidence_buffer.append(x_fail)
        _advance_persistent_belief(
            evidence_buffer,
            method,
            belief_model,
            graph,
            device,
            x_fail,
            _belief_persistence_enabled(config),
        )

    # Bellman target hygiene: action selection masks invalid options dynamically, so
    # the TD bootstrap must do the same for s_{t+1}. Store the next-state validity
    # inside the transition summary to avoid changing the public OptionTransition
    # schema; _td_update consumes it when the replay buffer exposes event_summary.
    valid_options_next = np.asarray(option_lib.valid_options(env.state, 0), dtype=bool)
    event_summary["valid_options_next"] = [bool(x) for x in valid_options_next.tolist()]

    sequence_truncated = bool(truncate_at_boundary and not done)
    if sequence_decision is not None:
        if not isinstance(sequence_episode, EpisodeEvidenceBuffer):
            raise RuntimeError("Sequence decision has no episode buffer.")
        next_evidence = sequence_decision.encode_boundary(
            observation=_obs_vector(obs, "agent_0"),
            ego_option_id=int(option_id),
            valid_actions=valid_options_next,
            terminated=bool(done),
            truncated=sequence_truncated,
        )
        value_bound = (config.get("training", {}).get("value_bound") or {})
        reward_scale = float(value_bound.get("reward_scale", 1.0))
        sequence_reward = (
            float(reward_sum)
            - float(config["training"].get("cost_coef", 0.0)) * float(realized_cost)
        ) / reward_scale
        sequence_episode.append_transition(
            action=int(option_id),
            reward=float(sequence_reward),
            discount=float(config["training"]["gamma"]) ** int(duration),
            done=bool(done),
            truncated=sequence_truncated,
            next_evidence=next_evidence,
            next_valid_actions=valid_options_next,
        )

    return (
        OptionTransition(
            obs_feat_t=obs_feat_t,
            evidence_t=evidence_t,
            evidence_mask_t=evidence_mask_t,
            evidence_len_t=int(evidence_len_t),
            belief_hidden_t=belief_hidden_t,
            option_id=int(option_id),
            reward_sum=float(reward_sum),
            expected_cost=float(expected_cost),
            realized_cost=float(realized_cost),
            duration=max(1, int(duration)),
            obs_feat_next=_obs_vector(obs, "agent_0"),
            evidence_next=evidence_buffer.snapshot(),
            evidence_mask_next=evidence_buffer.snapshot_mask(),
            evidence_len_next=int(evidence_buffer.length()),
            belief_hidden_next=evidence_buffer.belief_window_base_snapshot(),
            done=bool(done),
            termination_reason=termination_reason,
            graph_id=f"{graph.layout_name}:{graph.metadata.get('graph_variant', 'graph')}",
            partner_id=int(getattr(partner, "partner_id", 0)),
            event_summary=event_summary,
            bootstrap_mask=(
                None
                if sequence_decision is not None
                else _path_c_bootstrap_mask(config, rng)
            ),
        ),
        done,
        obs,
    )


def _path_c_bootstrap_mask(
    config: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray | None:
    ensemble = (config.get("path_c") or {}).get("ensemble") or {}
    if str(ensemble.get("architecture", "legacy_default_off")) == "recurrent_sequence_v1":
        return None
    n_heads = int(ensemble.get("n_heads", 1))
    if n_heads <= 1:
        return None
    p = float(ensemble.get("bootstrap_p", 1.0))
    mask = rng.random(n_heads) < p
    if not bool(mask.any()):
        mask[int(rng.integers(0, n_heads))] = True
    return mask.astype(np.bool_)


def _td_update(
    method: str,
    q_net: nn.Module,
    target_q_net: nn.Module,
    belief_model: FactorLocalBeliefModel,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, Any] | SequenceTDBatch,
    graph: GraphSpec,
    config: dict[str, Any],
    device: torch.device,
) -> float:
    q_net.train()
    if _uses_sequence_q(q_net):
        if not _uses_sequence_q(target_q_net):
            raise TypeError("Sequence training requires a recurrent target network.")
        if not isinstance(batch, SequenceTDBatch):
            raise TypeError("Sequence training requires a full-episode SequenceTDBatch.")
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        value_bound = config["training"].get("value_bound", {}) or {}
        loss = sequence_td_loss(
            q_net,
            target_q_net,
            batch,
            td_loss=str(config["training"].get("td_loss", "huber")),
            huber_delta=float(config["training"].get("huber_delta", 1.0)),
            double_q=_as_bool(config["training"].get("double_q", True), "double_q"),
            vmax=(
                float(value_bound["vmax"])
                if value_bound.get("enabled") and value_bound.get("vmax") is not None
                else None
            ),
            fail_on_zero_support_head=False,
            require_matching_prior=True,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in q_net.parameters() if parameter.requires_grad],
            float(config["training"]["grad_clip_norm"]),
        )
        optimizer.step()
        return float(loss.detach().cpu().item())

    if method in {"aris_bellman", "flat_factor"}:
        belief_model.train()

    obs_t = _tensor(batch["obs_feat_t"], device)
    obs_next = _tensor(batch["obs_feat_next"], device)
    evidence_t = _tensor(batch["evidence_t"], device)
    evidence_next = _tensor(batch["evidence_next"], device)
    graph_batch = _graph_tensors(graph, obs_t.shape[0], device)
    dynamic_next_mask = _option_mask_next_from_batch(batch, graph, device)
    if dynamic_next_mask is not None:
        graph_batch["option_mask_next"] = dynamic_next_mask
    evidence_len_t = _tensor(batch.get("evidence_len_t"), device) if batch.get("evidence_len_t") is not None else None
    evidence_len_next = _tensor(batch.get("evidence_len_next"), device) if batch.get("evidence_len_next") is not None else None
    evidence_mask_t = _batch_evidence_mask(batch.get("evidence_mask_t"), evidence_t, device)
    evidence_mask_next = _batch_evidence_mask(batch.get("evidence_mask_next"), evidence_next, device)
    belief_hidden_t = _optional_tensor(batch.get("belief_hidden_t"), device)
    belief_hidden_next = _optional_tensor(batch.get("belief_hidden_next"), device)
    state_t = _state_repr(
        method,
        belief_model,
        evidence_t,
        graph_batch,
        evidence_lengths=evidence_len_t,
        evidence_mask=evidence_mask_t,
        belief_hidden=belief_hidden_t,
    )
    state_next = _state_repr(
        method,
        belief_model,
        evidence_next,
        graph_batch,
        evidence_lengths=evidence_len_next,
        evidence_mask=evidence_mask_next,
        belief_hidden=belief_hidden_next,
    )

    optimizer.zero_grad(set_to_none=True)
    _vb = config["training"].get("value_bound", {}) or {}
    bootstrap_mask = _bootstrap_mask_from_batch(batch, device)
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
        q_extra_t=_q_extra_kwargs(method, batch, device),
        q_extra_next=_q_extra_kwargs(method, batch, device),
        td_loss=str(config["training"].get("td_loss", "huber")),
        huber_delta=float(config["training"].get("huber_delta", 1.0)),
        double_q=_as_bool(config["training"].get("double_q", True), "double_q"),
        reward_scale=float(_vb.get("reward_scale", 1.0)),
        vmax=(float(_vb["vmax"]) if _vb.get("enabled") and _vb.get("vmax") is not None else None),
        bootstrap_mask=bootstrap_mask,
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        _trainable_params(q_net, belief_model, method),
        float(config["training"]["grad_clip_norm"]),
    )
    optimizer.step()
    return float(loss.detach().cpu().item())


def _bootstrap_mask_from_batch(
    batch: dict[str, Any],
    device: torch.device,
) -> torch.Tensor | None:
    raw = batch.get("bootstrap_mask")
    if raw is None:
        return None
    if isinstance(raw, list) and all(item is None for item in raw):
        return None
    arr = np.asarray(raw, dtype=bool)
    if arr.ndim != 2:
        return None
    return torch.as_tensor(arr, dtype=torch.bool, device=device)


def _batch_evidence_mask(
    raw: Any,
    evidence: torch.Tensor,
    device: torch.device,
) -> torch.Tensor | None:
    """Return a boolean evidence mask for replay batches when available.

    P4/S2: replay evidence windows are left-padded with zeros early in an
    episode. Without this mask the GRU treats padding as real zero-valued
    behavioral evidence. Legacy replay without mask remains loadable but is
    explicitly weaker and should not support formal post-repair claims.
    """
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)) and all(item is None for item in raw):
        return None
    arr = np.asarray(raw, dtype=bool)
    expected_shape = tuple(evidence.shape[:-1])
    if arr.shape != expected_shape:
        warnings.warn(
            "Ignoring malformed evidence mask in replay batch: "
            f"shape={arr.shape}, expected={expected_shape}.",
            RuntimeWarning,
        )
        return None
    return torch.as_tensor(arr, dtype=torch.bool, device=device)


def _option_mask_next_from_batch(
    batch: dict[str, Any],
    graph: GraphSpec,
    device: torch.device,
) -> torch.Tensor | None:
    """Return the dynamic valid-option mask for bootstrap states when available.

    This keeps the Bellman target aligned with deployment action selection:
    _select_option() masks invalid options using option_lib.valid_options(state, 0),
    so the TD target must not bootstrap through options that are only present in
    the static graph support. The transition stores this as event_summary because
    the shared OptionTransition schema may be used by older experiments.
    """
    raw = batch.get("valid_options_next")
    if raw is None:
        summaries = batch.get("event_summary")
        if summaries is not None:
            values = []
            for summary in summaries:
                if isinstance(summary, dict):
                    values.append(summary.get("valid_options_next"))
                else:
                    values.append(None)
            if values and all(v is not None for v in values):
                raw = values
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=bool)
    if arr.ndim != 2 or arr.shape[1] != graph.num_options:
        warnings.warn(
            "Ignoring malformed valid_options_next mask in replay batch: "
            f"shape={arr.shape}, expected=(*,{graph.num_options}).",
            RuntimeWarning,
        )
        return None
    return torch.as_tensor(arr, dtype=torch.bool, device=device)


def _checkpoint_candidate_eligible(
    validation: dict[str, Any],
    config: dict[str, Any],
) -> bool:
    """Return whether a greedy checkpoint may be published as deployable.

    NEW-2/S20: when ego-delivery selection is required, a high-return
    free-riding checkpoint must not be saved as ``checkpoint.pt`` and then merely
    marked failed after the fact. Eligibility is enforced before publication.
    """
    if not bool(config["training"].get("require_ego_delivery_selection", False)):
        return True
    return int(validation.get("ego_sole_correct_delivery_count", 0)) > 0



def _free_rider_guard_verdict(
    checkpoint_selection: dict[str, Any],
    config: dict[str, Any],
    *,
    greedy_ran: bool,
) -> str:
    """RC free-rider guard verdict (Type-A).

    Under ego-terminal-aware selection a checkpoint can only be selected if it
    actually serves (ego_sole_correct_delivery_count > 0), so:
      * "not_required" — run did not opt in.
      * "not_evaluated" — no greedy validation ran (cannot judge).
      * "pass" — a serving checkpoint was selected.
      * "fail" — greedy ran but NO serving checkpoint was ever found (the run did
                 not learn to take the terminal stage). The accompanying
                 free_rider_diagnosis distinguishes free-riding from shaped-farming.
    """
    if not bool(config["training"].get("require_ego_delivery_selection", False)):
        return "not_required"
    if not greedy_ran:
        return "not_evaluated"
    sel_ego_sole = checkpoint_selection.get("selected_ego_sole_correct_delivery_count")
    if sel_ego_sole is not None and int(sel_ego_sole) > 0:
        return "pass"
    return "fail"


def _free_rider_diagnosis(checkpoint_selection: dict[str, Any]) -> str:
    """Why the free-rider guard failed: partner-serving (free-riding) vs no serve."""
    if int(checkpoint_selection.get("max_partner_correct_delivery_seen", 0)) > 0:
        return "free_riding_partner_serves_while_ego_idle"
    return "no_terminal_stage_shaped_farming_or_stall"


def _run_greedy_validation(
    method: str,
    q_net: nn.Module,
    belief_model: FactorLocalBeliefModel,
    env: OCV2Adapter,
    option_lib: OCV2OptionLibrary,
    router: OCV2EvidenceRouter,
    partners: list[Any],
    graph: GraphSpec,
    config: dict[str, Any],
    *,
    seed: int,
    episodes: int,
    update_idx: int,
    device: torch.device,
) -> dict[str, Any]:
    greedy_config = copy.deepcopy(config)
    greedy_config.setdefault("training", {})
    greedy_config["training"]["epsilon_start"] = 0.0
    greedy_config["training"]["epsilon_end"] = 0.0
    # Checkpoint-selection return is a formal greedy policy readout, not an active
    # probe-collection run. Probes are disabled here even if training used them.
    greedy_probe = greedy_config.setdefault("path_c", {}).setdefault("probe", {})
    greedy_probe["enable"] = False
    rng = np.random.default_rng(seed)
    returns: list[float] = []
    option_counts: list[int] = []
    # RC free-rider guard instrumentation: a high greedy return with zero EGO
    # deliveries is the free-riding signature (the partner finishes for the ego).
    # Aggregate the actor-specific terminal-stage counts so checkpoint selection
    # can refuse such a checkpoint. These are read off transition.event_summary,
    # which already carries ego/partner correct-delivery flags.
    ego_correct_deliveries = 0
    ego_sole_correct_deliveries = 0
    partner_correct_deliveries = 0
    team_delivery_episodes = 0
    ego_sole_completed_episodes = 0
    q_was_training = q_net.training
    belief_was_training = belief_model.training
    q_net.eval()
    belief_model.eval()
    try:
        with torch.no_grad():
            for episode_idx in range(int(episodes)):
                evidence_buffer = EvidenceBuffer(
                    num_factors=graph.num_factors,
                    window=int(config["training"]["evidence_window"]),
                    evidence_dim=D_EVID,
                )
                validation_partner = partners[int(episode_idx) % len(partners)]
                obs, state, partner = _reset_episode(
                    env,
                    evidence_buffer,
                    partners,
                    rng,
                    int(seed) + episode_idx,
                    router,
                    partner_override=validation_partner,
                )
                _start_sequence_episode(
                    evidence_buffer,
                    q_net,
                    obs,
                    state,
                    option_lib,
                    greedy_config,
                    episode_id=f"greedy_validation:{int(seed)}:{int(episode_idx)}",
                    manifest_seed=int(seed),
                )
                partner_option_inferencer = _new_partner_option_inferencer(option_lib, state, greedy_config)
                _initialise_persistent_belief(evidence_buffer, method, belief_model, graph, device, _belief_persistence_enabled(greedy_config))
                contribution_ledger = ContributionLedger.from_config(
                    greedy_config.get("training")
                )
                episode_return = 0.0
                option_count = 0
                episode_delivered = 0
                episode_ego_sole = 0
                done = False
                while (
                    not done
                    and option_count < int(config["training"]["max_episode_options"])
                ):
                    option_id = _select_option(
                        method,
                        q_net,
                        belief_model,
                        obs,
                        env.state,
                        evidence_buffer,
                        option_lib,
                        graph,
                        greedy_config,
                        update_idx,
                        rng,
                        device,
                        partner_id=int(getattr(partner, "partner_id", 0)),
                        selection_stats=None,
                        eval_mode=True,
                    )
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
                        greedy_config,
                        contribution_ledger=contribution_ledger,
                        partner_option_inferencer=partner_option_inferencer,
                        method=method,
                        belief_model=belief_model,
                        device=device,
                        truncate_at_boundary=(
                            option_count + 1
                            >= int(config["training"]["max_episode_options"])
                        ),
                    )
                    episode_return += _transition_training_return(
                        transition,
                        greedy_config,
                    )
                    _esummary = transition.event_summary or {}
                    ego_correct_deliveries += int(_esummary.get("ego_correct_delivery", 0))
                    ego_sole_correct_deliveries += int(
                        _esummary.get("ego_sole_correct_delivery", 0)
                    )
                    partner_correct_deliveries += int(
                        _esummary.get("partner_correct_delivery", 0)
                    )
                    episode_delivered += int(_esummary.get("delivery_event", 0))
                    episode_ego_sole += int(_esummary.get("ego_sole_correct_delivery", 0))
                    option_count += 1
                returns.append(float(episode_return))
                option_counts.append(int(option_count))
                team_delivery_episodes += int(episode_delivered > 0)
                ego_sole_completed_episodes += int(episode_ego_sole > 0)
    finally:
        q_net.train(q_was_training)
        belief_model.train(belief_was_training)

    mean_return = (
        float(np.mean(np.asarray(returns, dtype=np.float64)))
        if returns
        else float("nan")
    )
    return {
        "update": int(update_idx),
        "seed": int(seed),
        "episodes": int(episodes),
        "returns": returns,
        "mean_return": mean_return,
        "option_counts": option_counts,
        "ego_correct_delivery_count": int(ego_correct_deliveries),
        "ego_sole_correct_delivery_count": int(ego_sole_correct_deliveries),
        "partner_correct_delivery_count": int(partner_correct_deliveries),
        # F9 (FINDINGS_LEDGER): the former completion-rate key here carried TEAM
        # semantics (any delivery incl. partner's). Renamed so a team rate is
        # never quoted under a completion name; the ego-sole rate matches the
        # formal eval headline metric.
        "team_delivery_episode_rate": (
            float(team_delivery_episodes) / float(episodes) if int(episodes) > 0 else 0.0
        ),
        "ego_correct_completion_rate": (
            float(ego_sole_completed_episodes) / float(episodes)
            if int(episodes) > 0
            else 0.0
        ),
    }


def _state_repr(
    method: str,
    belief_model: FactorLocalBeliefModel,
    evidence: torch.Tensor,
    graph_batch: dict[str, Any],
    *,
    evidence_lengths: torch.Tensor | None = None,
    evidence_mask: torch.Tensor | None = None,
    belief_hidden: torch.Tensor | None = None,
) -> torch.Tensor:
    if method == "global_gru":
        # Compatibility value for legacy belief diagnostics only.  The active
        # global_gru model is RecurrentEnsembleQ and consumes EgoEvidenceSpecV1
        # directly in selection and sequence TD updates.
        return evidence
    if method in {"aris_bellman", "flat_factor"}:
        hidden = belief_model.encode_history(
            evidence,
            graph_batch.get("factor_features"),
            factor_mask=graph_batch["factor_mask"],
            evidence_lengths=evidence_lengths,
            evidence_mask=evidence_mask,
            initial_hidden=belief_hidden,
        )
        return belief_model.belief_from_hidden(
            hidden,
            factor_features=graph_batch.get("factor_features"),
            factor_mask=graph_batch["factor_mask"],
            mode_mask=graph_batch["mode_mask"],
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
    partner_id: int | None = None,
    selection_stats: dict[str, Any] | None = None,
    path_c_probe_base_q_net: nn.Module | None = None,
    eval_mode: bool = False,
) -> int:
    _record_selection_attempt(selection_stats)
    _path_c_reset_probe_decision(selection_stats)
    valid = option_lib.valid_options(state, 0)
    valid_ids = np.flatnonzero(valid)
    if valid_ids.size == 0:
        _record_forced_noop(selection_stats)
        return _noop_option_id(option_lib)
    if method == "random_policy" or rng.random() < _epsilon(config, update_idx):
        return _sample_exploration_option(
            option_lib,
            valid_ids,
            graph,
            config,
            update_idx,
            rng,
            selection_stats,
        )

    with torch.no_grad():
        obs_tensor = _tensor(_obs_vector(obs, "agent_0")[None, ...], device)
        if _uses_sequence_q(q_net):
            episode = evidence_buffer.sequence_episode()
            if not isinstance(episode, EpisodeEvidenceBuffer):
                raise RuntimeError(
                    "Sequence action selection requires a started full-episode evidence buffer."
                )
            sequence_batch = episode.evidence_batch(device)
            encoded_valid = sequence_batch.valid_actions[0, -1]
            valid_tensor = torch.as_tensor(valid, dtype=torch.bool, device=device)
            if not torch.equal(encoded_valid, valid_tensor):
                raise RuntimeError(
                    "Current valid actions disagree with EgoEvidenceSpecV1 history."
                )
            head_trace, _ = q_net.forward_sequence(sequence_batch)
            head_values = head_trace[:, -1]
            prior_head_values = None
            if float(q_net.prior_scale) > 0.0:
                prior_head_values = q_net.forward_prior_sequence(sequence_batch)[:, -1]
            _record_path_c_ensemble_telemetry(
                selection_stats,
                ensemble_diversity_telemetry(
                    head_values,
                    valid_tensor.unsqueeze(0),
                    prior_q_values=prior_head_values,
                    prior_scale=float(q_net.prior_scale),
                ),
            )
            q_values = head_values.mean(dim=1).squeeze(0)
            q_values = q_values.masked_fill(~valid_tensor, -1e9)
            probe_choice = _path_c_probe_choice(
                q_net,
                obs_tensor,
                None,
                _graph_tensors(graph, 1, device),
                q_values,
                valid_tensor,
                config,
                partner_id=partner_id,
                device=device,
                selection_stats=selection_stats,
                base_q_values=None,
                eval_mode=eval_mode,
                probe_rng=rng,
                head_values_override=head_values,
            )
            if probe_choice is not None:
                return probe_choice
            return int(torch.argmax(q_values).item())

        evidence = _tensor(evidence_buffer.snapshot()[None, ...], device)
        evidence_mask = torch.as_tensor(
            evidence_buffer.snapshot_mask()[None, ...], dtype=torch.bool, device=device
        )
        evidence_lengths = torch.as_tensor(
            [evidence_buffer.length()], dtype=torch.float32, device=device
        )
        graph_batch = _graph_tensors(graph, 1, device)
        belief_hidden_np = evidence_buffer.belief_window_base_snapshot()
        belief_hidden = (
            _tensor(belief_hidden_np[None, ...], device)
            if belief_hidden_np is not None
            else None
        )
        state_repr = _state_repr(
            method,
            belief_model,
            evidence,
            graph_batch,
            evidence_lengths=evidence_lengths,
            evidence_mask=evidence_mask,
            belief_hidden=belief_hidden,
        )
        q_values = q_net(
            obs_tensor,
            state_repr,
            **_q_forward_kwargs(graph_batch),
            partner_id=_partner_id_tensor(partner_id, 1, device),
        ).squeeze(0)
        valid_tensor = torch.as_tensor(valid, dtype=torch.bool, device=device)
        q_values = q_values.masked_fill(~valid_tensor, -1e9)
        base_q_values = None
        if path_c_probe_base_q_net is not None:
            base_q_values = path_c_probe_base_q_net(
                obs_tensor,
                state_repr,
                **_q_forward_kwargs(graph_batch),
            )
        probe_choice = _path_c_probe_choice(
            q_net,
            obs_tensor,
            state_repr,
            graph_batch,
            q_values,
            valid_tensor,
            config,
            partner_id=partner_id,
            device=device,
            selection_stats=selection_stats,
            base_q_values=base_q_values,
            eval_mode=eval_mode,
            probe_rng=rng,
        )
        if probe_choice is not None:
            return probe_choice
        return int(torch.argmax(q_values).item())


def _path_c_reset_probe_decision(selection_stats: dict[str, Any] | None) -> None:
    if selection_stats is None:
        return
    record = {
        "selected": False,
        "option_id": -1,
        "reason": "none",
        "scalar": None,
        "candidate_mean_q": None,
        "rule": "none",
        "selection_count": int(selection_stats.get("option_selection_count", 0)),
    }
    selection_stats["path_c_probe_last"] = dict(record)
    selection_stats["path_c_probe_last_decision"] = dict(record)


def _path_c_record_probe_decision(
    selection_stats: dict[str, Any] | None,
    selected: bool,
    reason: str,
    *,
    option_id: int = -1,
    scalar: float | None = None,
    candidate_mean_q: float | None = None,
    rule: str | None = None,
    candidate_scores: Sequence[float] | None = None,
    candidate_mask: Sequence[bool] | None = None,
    propensity: float | None = None,
    probe_cost_per_use: float | None = None,
) -> None:
    if selection_stats is None:
        return
    if reason not in PATH_C_PROBE_PROVENANCE_IDS:
        raise ValueError(f"Unknown Path C probe provenance reason: {reason!r}")
    unit_cost = (
        float(selection_stats.get("path_c_probe_cost_per_use", 0.0))
        if probe_cost_per_use is None
        else float(probe_cost_per_use)
    )
    if not math.isfinite(unit_cost) or unit_cost < 0.0:
        raise ValueError("Path C probe cost must be finite and non-negative.")
    record = {
        "selected": bool(selected),
        "option_id": int(option_id) if option_id is not None else -1,
        "reason": str(reason),
        "scalar": None if scalar is None else float(scalar),
        "candidate_mean_q": (
            None if candidate_mean_q is None else float(candidate_mean_q)
        ),
        "rule": "none" if rule is None else str(rule),
        "target": "normalized_advantage_disagreement",
        "candidate_scores": (
            None if candidate_scores is None
            else [float(value) for value in candidate_scores]
        ),
        "candidate_mask": (
            None if candidate_mask is None
            else [bool(value) for value in candidate_mask]
        ),
        "propensity": None if propensity is None else float(propensity),
        "probe_cost_per_use": unit_cost,
        "realized_probe_cost": unit_cost if selected else 0.0,
        "selection_count": int(selection_stats.get("option_selection_count", 0)),
    }
    selection_stats["path_c_probe_last"] = dict(record)
    selection_stats["path_c_probe_last_decision"] = dict(record)
    selection_stats.setdefault("path_c_probe_decisions", []).append(dict(record))


def _path_c_probe_choice(
    q_net: nn.Module,
    obs_tensor: torch.Tensor,
    state_repr: Any,
    graph_batch: dict[str, torch.Tensor],
    q_values: torch.Tensor,
    valid_tensor: torch.Tensor,
    config: dict[str, Any],
    *,
    partner_id: int | None,
    device: torch.device,
    selection_stats: dict[str, Any] | None,
    base_q_values: torch.Tensor | None = None,
    eval_mode: bool = False,
    probe_rng: np.random.Generator | None = None,
    head_values_override: torch.Tensor | None = None,
) -> int | None:
    path_c = config.get("path_c") or {}
    probe = path_c.get("probe") or {}
    if selection_stats is not None:
        selection_stats["path_c_probe_cost_per_use"] = float(
            (path_c.get("preregistration") or {}).get(
                "probe_cost_per_use", 0.0
            )
        )
    if not bool(probe.get("enable", False)):
        return None
    if eval_mode:
        _path_c_record_probe_decision(selection_stats, False, "disabled")
        return None
    if selection_stats is not None:
        _increment_count(selection_stats, "path_c_probe_opportunity_count")
    if int((path_c.get("ensemble") or {}).get("n_heads", 1)) <= 1:
        if selection_stats is not None:
            _increment_count(selection_stats, "path_c_probe_skipped_single_head_count")
        _path_c_record_probe_decision(selection_stats, False, "disabled")
        return None
    # A separately learned public-state baseline is intentionally not part of
    # the primary selector.  It remains available to callers for secondary
    # visualization only.
    del base_q_values
    if head_values_override is None and not hasattr(q_net, "forward_heads"):
        _path_c_record_probe_decision(selection_stats, False, "disabled")
        return None
    head_values = head_values_override
    if head_values is None:
        head_values = q_net.forward_heads(
            obs_tensor,
            state_repr,
            **_q_forward_kwargs(graph_batch),
            partner_id=_partner_id_tensor(partner_id, 1, device),
        )
    disagreement = normalized_advantage_disagreement(
        head_values,
        option_mask=valid_tensor.unsqueeze(0) if valid_tensor.ndim == 1 else valid_tensor,
        stat=str((path_c.get("ensemble") or {}).get("disagreement_stat", "variance")),
        tie_atol=float((path_c.get("ensemble") or {}).get("tie_atol", 1.0e-6)),
    )
    logged_candidate_scores = [
        float(value)
        for value in disagreement["per_option"].detach().cpu().reshape(-1).tolist()
    ]
    logged_candidate_mask = [
        bool(value)
        for value in valid_tensor.detach().cpu().reshape(-1).tolist()
    ]
    collection_selection_mode = str(
        probe.get("collection_selection_mode", "normalized_advantage")
    )
    if collection_selection_mode == "random":
        if probe_rng is None:
            raise ValueError("Random-probe collection requires an explicit RNG.")
        valid_ids = np.flatnonzero(valid_tensor.detach().cpu().numpy().astype(bool))
        choice = int(probe_rng.choice(valid_ids))
        score = float(
            disagreement["per_option"].detach().cpu().reshape(-1)[choice]
        )
        candidate_mean_q = float(q_values.reshape(-1)[choice].detach().cpu().item())
        return_floor = probe.get("return_floor")
        if return_floor is None or candidate_mean_q < float(return_floor):
            reason = "return_floor"
        else:
            reason = "selected"
        decision = {
            "selected": reason == "selected",
            "reason": reason,
            "option_id": choice,
            "score": score,
            "candidate_mean_q": candidate_mean_q,
            "propensity": 1.0 / float(len(valid_ids)),
        }
    elif collection_selection_mode == "normalized_advantage":
        decision = select_probe_candidate(
            disagreement["per_option"],
            q_values,
            valid_tensor,
            disagreement_threshold=probe.get("disagreement_threshold"),
            return_floor=probe.get("return_floor"),
        )
        decision["propensity"] = 1.0
    else:
        raise ValueError(
            "Path C collection_selection_mode must be 'normalized_advantage' "
            f"or 'random'; got {collection_selection_mode!r}."
        )
    choice = int(decision["option_id"])
    scalar = decision.get("score")
    candidate_mean_q = decision.get("candidate_mean_q")
    if decision["reason"] == "threshold":
        if selection_stats is not None:
            _increment_count(selection_stats, "path_c_probe_skipped_threshold_count")
        _path_c_record_probe_decision(
            selection_stats,
            False,
            "threshold",
            option_id=choice,
            scalar=scalar,
            candidate_mean_q=candidate_mean_q,
            rule=str(probe.get("rule", "max_normalized_advantage_disagreement")),
            candidate_scores=logged_candidate_scores,
            candidate_mask=logged_candidate_mask,
            propensity=decision.get("propensity"),
        )
        return None
    if decision["reason"] == "return_floor":
        if selection_stats is not None:
            _increment_count(selection_stats, "path_c_probe_skipped_return_floor_count")
        _path_c_record_probe_decision(
            selection_stats,
            False,
            "return_floor",
            option_id=choice,
            scalar=scalar,
            candidate_mean_q=candidate_mean_q,
            rule=(
                "random_probe_valid"
                if collection_selection_mode == "random"
                else str(probe.get("rule", "max_normalized_advantage_disagreement"))
            ),
            candidate_scores=logged_candidate_scores,
            candidate_mask=logged_candidate_mask,
            propensity=decision.get("propensity"),
        )
        return None
    if not decision["selected"]:
        _path_c_record_probe_decision(selection_stats, False, "disabled")
        return None
    if selection_stats is not None:
        _increment_count(selection_stats, "path_c_probe_selected_count")
    _path_c_record_probe_decision(
        selection_stats,
        True,
        "selected",
        option_id=choice,
        scalar=scalar,
        candidate_mean_q=candidate_mean_q,
        rule=(
            "random_probe_valid"
            if collection_selection_mode == "random"
            else str(probe.get("rule", "max_normalized_advantage_disagreement"))
        ),
        candidate_scores=logged_candidate_scores,
        candidate_mask=logged_candidate_mask,
        propensity=decision.get("propensity"),
    )
    return choice

def _sample_exploration_option(
    option_lib: OCV2OptionLibrary,
    valid_ids: np.ndarray,
    graph: GraphSpec,
    config: dict[str, Any],
    update_idx: int,
    rng: np.random.Generator,
    selection_stats: dict[str, Any] | None = None,
) -> int:
    """Training-only directed exploration over valid terminal-stage options.

    This changes only epsilon exploration, not greedy/deployment argmax. It is a
    data-distribution intervention for the terminal-stage stall: when serve/plate
    options are already valid, do not waste almost all exploratory samples on
    bottleneck/wait/fetch options.
    """
    training_cfg = config.get("training", {}) or {}
    role_cfg = training_cfg.get("role_exploration") or {}
    if bool(role_cfg.get("enabled", False)):
        # P5: main-method exploration may use only behavior-observable state. It
        # must not branch on partner terminal_policy. A role_exploration.default
        # block is accepted as an oracle-free curriculum; keyed yield/claim blocks
        # are ignored unless run as a separately labeled ablation outside the main
        # method path.
        cfg = role_cfg.get("default") or training_cfg.get("terminal_exploration") or {}
        enabled = bool(cfg)
    else:
        cfg = training_cfg.get("terminal_exploration") or {}
        enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return int(rng.choice(valid_ids))
    preferred_kinds = tuple(
        str(kind)
        for kind in cfg.get(
            "preferred_kinds",
            ("serve_soup", "plate_soup", "pick_plate"),
        )
    )
    preferred = [
        int(idx)
        for idx in valid_ids
        if graph.options[int(idx)].kind in preferred_kinds
    ]
    if preferred and rng.random() < _terminal_exploration_bias(cfg, update_idx):
        _record_terminal_exploration_pick(selection_stats)
        # Ordered preference matters at the terminal stage: if serve is valid,
        # sampling pick_plate instead is usually a regression. Among equally ranked
        # valid options, remain stochastic to preserve exploration.
        rank = {kind: pos for pos, kind in enumerate(preferred_kinds)}
        best_rank = min(rank.get(graph.options[int(idx)].kind, 10_000) for idx in preferred)
        best = [idx for idx in preferred if rank.get(graph.options[int(idx)].kind, 10_000) == best_rank]
        return int(rng.choice(np.asarray(best, dtype=np.int64)))
    return int(rng.choice(valid_ids))


def _terminal_exploration_bias(cfg: dict[str, Any], update_idx: int) -> float:
    start = float(cfg.get("bias_start", 0.9))
    end = float(cfg.get("bias_end", 0.35))
    horizon = max(1, int(cfg.get("anneal_updates", 2_500)))
    frac = min(1.0, max(0.0, float(update_idx) / float(horizon)))
    return float(start + frac * (end - start))


def _record_terminal_exploration_pick(selection_stats: dict[str, Any] | None) -> None:
    if selection_stats is None:
        return
    selection_stats["terminal_exploration_pick_count"] = int(
        selection_stats.get("terminal_exploration_pick_count", 0)
    ) + 1


def _q_forward_kwargs(graph_batch: dict[str, Any]) -> dict[str, Any]:
    return {
        "option_mask": graph_batch["option_mask"],
        "factor_mask": graph_batch["factor_mask"],
        "mode_mask": graph_batch["mode_mask"],
        "relevance_mask": graph_batch["relevance_mask"],
        "option_features": graph_batch.get("option_features"),
        "factor_features": graph_batch.get("factor_features"),
    }


def _q_extra_kwargs(
    method: str,
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    if method != "partner_id_q":
        return {}
    partner_ids = batch.get("partner_id")
    if partner_ids is None:
        return {"partner_id": _partner_id_tensor(None, len(batch["option_id"]), device)}
    return {
        "partner_id": torch.as_tensor(
            np.asarray(partner_ids, dtype=np.int64),
            dtype=torch.long,
            device=device,
        )
    }


def _partner_id_tensor(
    partner_id: int | None,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    value = 0 if partner_id is None else int(partner_id)
    return torch.full((int(batch_size),), value, dtype=torch.long, device=device)


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



def _maybe_seed_terminal_replay(
    env: OCV2Adapter,
    partners: list[Any],
    option_lib: OCV2OptionLibrary,
    router: OCV2EvidenceRouter,
    graph: GraphSpec,
    config: dict[str, Any],
    base_seed: int,
    replay: OptionReplayBuffer,
    method: str,
    q_net: nn.Module,
    target_q_net: nn.Module,
    belief_model: FactorLocalBeliefModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, Any]:
    """Seed Bellman replay with scripted ego-owned terminal-stage chains.

    This is a higher-risk curriculum/data intervention for the observed stall:
    online exploration never generates ego plate→serve transitions. It does not
    add imitation loss, supervised labels, or a deployment fallback. The seeded
    data are converted into normal OptionTransition rows and trained with the same
    TD loss as online experience.
    """
    train_cfg = config.get("training", {}) or {}
    role_cfg = train_cfg.get("role_replay_seed") or {}
    cfg = train_cfg.get("terminal_replay_seed") or {}
    if bool(role_cfg.get("enabled", False)):
        if bool(cfg.get("enabled", False)):
            logger.warning(
                "Both terminal_replay_seed.enabled=True and role_replay_seed.enabled=True; "
                "using role_replay_seed."
            )
        return _seed_role_replay(
            env,
            partners,
            option_lib,
            router,
            graph,
            config,
            train_cfg,
            role_cfg,
            base_seed,
            replay,
            method,
            q_net,
            target_q_net,
            belief_model,
            optimizer,
            device,
        )
    if not bool(cfg.get("enabled", False)):
        return {}
    if not partners:
        raise ValueError("terminal_replay_seed requires at least one training partner")

    rng = np.random.default_rng(int(base_seed) + int(cfg.get("seed_offset", 7_131_917)))
    target_serves = max(0, int(cfg.get("target_ego_serves", 64)))
    max_episodes = max(1, int(cfg.get("max_episodes", 120)))
    max_options = max(1, int(cfg.get("max_episode_options", train_cfg.get("max_episode_options", 20))))
    store_all_chain = bool(cfg.get("store_all_chain", True))
    store_kinds = set(
        str(k)
        for k in cfg.get(
            "store_option_kinds",
            ("pick_plate", "plate_soup", "serve_soup"),
        )
    )
    priority_kinds = tuple(
        str(k)
        for k in cfg.get(
            "priority_kinds",
            (
                "serve_soup",
                "plate_soup",
                "pick_plate",
                "deliver_ingredient_to_pot",
                "fetch_ingredient",
                "clear_interaction_cell",
                "wait_at_bottleneck",
            ),
        )
    )
    partner = _partner_by_name(partners, cfg.get("partner"))

    rows_added = 0
    ego_serves = 0
    episodes_used = 0
    option_counts_by_kind: dict[str, int] = {}
    terminal_rows_added = 0

    for episode_idx in range(max_episodes):
        if target_serves > 0 and ego_serves >= target_serves:
            break
        evidence_buffer = EvidenceBuffer(
            num_factors=graph.num_factors,
            window=int(train_cfg["evidence_window"]),
            evidence_dim=D_EVID,
        )
        obs, state, active_partner = _reset_episode(
            env,
            evidence_buffer,
            [partner],
            rng,
            int(base_seed) + 90_000_000 + episode_idx,
            router,
            partner_override=partner,
        )
        partner_option_inferencer = _new_partner_option_inferencer(option_lib, state, config)
        _initialise_persistent_belief(evidence_buffer, method, belief_model, graph, device, _belief_persistence_enabled(config))
        contribution_ledger = ContributionLedger.from_config(train_cfg)
        done = False
        episodes_used += 1
        for _ in range(max_options):
            if done:
                break
            option_id = _scripted_terminal_option(
                option_lib, graph, env.state, priority_kinds, rng
            )
            transition, done, obs = _execute_option(
                env,
                obs,
                active_partner,
                option_lib,
                router,
                evidence_buffer,
                option_id,
                graph,
                rng,
                config,
                contribution_ledger=contribution_ledger,
                partner_option_inferencer=partner_option_inferencer,
                method=method,
                belief_model=belief_model,
                device=device,
            )
            kind = graph.options[int(transition.option_id)].kind
            option_counts_by_kind[kind] = int(option_counts_by_kind.get(kind, 0)) + 1
            summary = transition.event_summary or {}
            terminal_event = bool(
                int(summary.get("ego_sole_correct_delivery", 0))
                or kind in {"pick_plate", "plate_soup", "serve_soup"}
            )
            if store_all_chain or kind in store_kinds or terminal_event:
                replay.add(transition)
                rows_added += 1
                terminal_rows_added += int(terminal_event)
            ego_serves += int(summary.get("ego_sole_correct_delivery", 0))
            if target_serves > 0 and ego_serves >= target_serves:
                break

    summary: dict[str, Any] = {
        "enabled": True,
        "partner": getattr(partner, "name", None),
        "target_ego_serves": int(target_serves),
        "episodes_used": int(episodes_used),
        "rows_added": int(rows_added),
        "terminal_rows_added": int(terminal_rows_added),
        "ego_sole_correct_delivery_count": int(ego_serves),
        "option_kind_counts": option_counts_by_kind,
    }

    seed_updates = max(0, int(cfg.get("seed_updates", 0)))
    if seed_updates > 0 and rows_added > 0:
        pretrain = _pretrain_from_seed_replay(
            replay,
            seed_updates,
            int(cfg.get("seed_batch_size", train_cfg.get("batch_size", 8))),
            method,
            q_net,
            target_q_net,
            belief_model,
            optimizer,
            graph,
            config,
            device,
        )
        summary.update(pretrain)
    return summary


def _seed_role_replay(
    env: OCV2Adapter,
    partners: list[Any],
    option_lib: OCV2OptionLibrary,
    router: OCV2EvidenceRouter,
    graph: GraphSpec,
    config: dict[str, Any],
    train_cfg: dict[str, Any],
    cfg: dict[str, Any],
    base_seed: int,
    replay: OptionReplayBuffer,
    method: str,
    q_net: nn.Module,
    target_q_net: nn.Module,
    belief_model: FactorLocalBeliefModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, Any]:
    if not partners:
        raise ValueError("role_replay_seed requires at least one training partner")
    chains = list(cfg.get("chains") or [])
    if not chains:
        raise ValueError("role_replay_seed.enabled=true requires at least one chain")

    rng = np.random.default_rng(int(base_seed) + int(cfg.get("seed_offset", 7_131_917)))
    max_options = max(1, int(cfg.get("max_episode_options", train_cfg.get("max_episode_options", 20))))
    rows_added = 0
    terminal_rows_added = 0
    chain_summaries: list[dict[str, Any]] = []

    for chain_idx, chain_raw in enumerate(chains):
        chain = chain_raw or {}
        name = str(chain.get("name", f"chain_{chain_idx}"))
        partner = _partner_by_name(partners, chain.get("partner"))
        target_actor = str(chain.get("target_actor", "ego"))
        if target_actor not in {"ego", "partner"}:
            raise ValueError(
                f"role_replay_seed chain {name!r} target_actor={target_actor!r}; "
                "expected 'ego' or 'partner'."
            )
        target_deliveries = max(0, int(chain.get("target_deliveries", 0)))
        max_episodes = max(1, int(chain.get("max_episodes", cfg.get("max_episodes", 120))))
        priority_kinds = tuple(
            str(k)
            for k in chain.get(
                "ego_priority_kinds",
                (
                    "serve_soup",
                    "plate_soup",
                    "pick_plate",
                    "deliver_ingredient_to_pot",
                    "fetch_ingredient",
                    "clear_interaction_cell",
                    "wait_at_bottleneck",
                ),
            )
        )
        chain_rows = 0
        chain_terminal_rows = 0
        chain_ego_deliveries = 0
        chain_partner_deliveries = 0
        chain_target_deliveries = 0
        episodes_used = 0
        option_counts_by_kind: dict[str, int] = {}

        for episode_idx in range(max_episodes):
            if target_deliveries > 0 and chain_target_deliveries >= target_deliveries:
                break
            evidence_buffer = EvidenceBuffer(
                num_factors=graph.num_factors,
                window=int(train_cfg["evidence_window"]),
                evidence_dim=D_EVID,
            )
            obs, state, active_partner = _reset_episode(
                env,
                evidence_buffer,
                [partner],
                rng,
                int(base_seed) + 91_000_000 + chain_idx * 1_000_000 + episode_idx,
                router,
                partner_override=partner,
            )
            partner_option_inferencer = _new_partner_option_inferencer(option_lib, state, config)
            _initialise_persistent_belief(evidence_buffer, method, belief_model, graph, device, _belief_persistence_enabled(config))
            contribution_ledger = ContributionLedger.from_config(train_cfg)
            done = False
            episodes_used += 1
            for _ in range(max_options):
                if done:
                    break
                option_id = _scripted_terminal_option(
                    option_lib, graph, env.state, priority_kinds, rng
                )
                transition, done, obs = _execute_option(
                    env,
                    obs,
                    active_partner,
                    option_lib,
                    router,
                    evidence_buffer,
                    option_id,
                    graph,
                    rng,
                    config,
                    contribution_ledger=contribution_ledger,
                    partner_option_inferencer=partner_option_inferencer,
                    method=method,
                    belief_model=belief_model,
                    device=device,
                )
                kind = graph.options[int(transition.option_id)].kind
                option_counts_by_kind[kind] = int(option_counts_by_kind.get(kind, 0)) + 1
                summary = transition.event_summary or {}
                replay.add(transition)
                rows_added += 1
                chain_rows += 1
                ego_count = int(summary.get("ego_sole_correct_delivery", 0))
                partner_count = int(summary.get("partner_correct_delivery", 0))
                chain_ego_deliveries += ego_count
                chain_partner_deliveries += partner_count
                if target_actor == "ego":
                    chain_target_deliveries += ego_count
                else:
                    chain_target_deliveries += partner_count
                terminal_event = bool(
                    int(summary.get("delivery_event", 0))
                    or kind in {"pick_plate", "plate_soup", "serve_soup"}
                )
                terminal_rows_added += int(terminal_event)
                chain_terminal_rows += int(terminal_event)
                if target_deliveries > 0 and chain_target_deliveries >= target_deliveries:
                    break

        chain_summaries.append(
            {
                "name": name,
                "partner": getattr(partner, "name", None),
                "target_actor": target_actor,
                "target_deliveries": int(target_deliveries),
                "episodes_used": int(episodes_used),
                "rows_added": int(chain_rows),
                "terminal_rows_added": int(chain_terminal_rows),
                "ego_sole_correct_delivery_count": int(chain_ego_deliveries),
                "partner_correct_delivery_count": int(chain_partner_deliveries),
                "target_delivery_count": int(chain_target_deliveries),
                "option_kind_counts": option_counts_by_kind,
            }
        )

    summary: dict[str, Any] = {
        "enabled": True,
        "mode": "role_replay_seed",
        "rows_added": int(rows_added),
        "terminal_rows_added": int(terminal_rows_added),
        "chains": chain_summaries,
    }

    seed_updates = max(0, int(cfg.get("seed_updates", 0)))
    if seed_updates > 0 and rows_added > 0:
        pretrain = _pretrain_from_seed_replay(
            replay,
            seed_updates,
            int(cfg.get("seed_batch_size", train_cfg.get("batch_size", 8))),
            method,
            q_net,
            target_q_net,
            belief_model,
            optimizer,
            graph,
            config,
            device,
        )
        summary.update(pretrain)
    return summary


def _pretrain_from_seed_replay(
    replay: OptionReplayBuffer,
    seed_updates: int,
    batch_size: int,
    method: str,
    q_net: nn.Module,
    target_q_net: nn.Module,
    belief_model: FactorLocalBeliefModel,
    optimizer: torch.optim.Optimizer,
    graph: GraphSpec,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    if len(replay) < max(1, int(batch_size)):
        return {
            "seed_updates_requested": int(seed_updates),
            "seed_updates_done": 0,
            "seed_update_skip_reason": "insufficient_seed_replay",
        }
    losses: list[float] = []
    target_interval = max(1, int(config["training"].get("target_update_interval", 50)))
    for update_idx in range(int(seed_updates)):
        batch = replay.sample(int(batch_size))
        loss = _td_update(
            method,
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
                f"Non-finite terminal seed TD loss at seed update {update_idx}: {loss}"
            )
        losses.append(float(loss))
        if (update_idx + 1) % target_interval == 0:
            target_q_net.load_state_dict(q_net.state_dict())
    if losses:
        target_q_net.load_state_dict(q_net.state_dict())
    return {
        "seed_updates_requested": int(seed_updates),
        "seed_updates_done": int(len(losses)),
        "seed_td_loss_start": float(losses[0]) if losses else None,
        "seed_td_loss_end": float(losses[-1]) if losses else None,
        "seed_td_loss_mean": float(np.mean(np.asarray(losses, dtype=np.float64))) if losses else None,
    }


def _partner_by_name(partners: list[Any], name: Any | None) -> Any:
    if name is None:
        return partners[0]
    for partner in partners:
        if getattr(partner, "name", None) == str(name):
            return partner
    raise ValueError(
        f"terminal_replay_seed.partner={name!r} not found; "
        f"available={[getattr(p, 'name', '?') for p in partners]}"
    )


def _scripted_terminal_option(
    option_lib: OCV2OptionLibrary,
    graph: GraphSpec,
    state: Any,
    priority_kinds: tuple[str, ...],
    rng: np.random.Generator,
) -> int:
    valid = option_lib.valid_options(state, 0)
    valid_ids = np.flatnonzero(valid)
    if valid_ids.size == 0:
        return _noop_option_id(option_lib)
    for kind in priority_kinds:
        candidates = [int(idx) for idx in valid_ids if graph.options[int(idx)].kind == kind]
        if candidates:
            return int(rng.choice(np.asarray(candidates, dtype=np.int64)))
    return int(rng.choice(valid_ids))


def _sample_partner(partners: list[Any], rng: np.random.Generator) -> Any:
    grouped: dict[str, list[Any]] = {}
    for partner in partners:
        group = str(getattr(partner, "sampling_group", getattr(partner, "name", "default")))
        grouped.setdefault(group, []).append(partner)

    if len(grouped) < len(partners):
        # Multiple groups -> sample group uniformly, then partner by weight within group.
        group_names = sorted(grouped)
        group = group_names[int(rng.integers(0, len(group_names)))]
        candidates = grouped[group]
    else:
        candidates = partners

    weights = np.asarray(
        [max(0.0, float(getattr(p, "sampling_weight", 1.0))) for p in candidates],
        dtype=np.float64,
    )
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("partner_sampling weights must be finite and have positive sum.")
    probs = weights / total
    idx = int(rng.choice(np.arange(len(candidates)), p=probs))
    return candidates[idx]


def _reset_episode(
    env: OCV2Adapter,
    evidence_buffer: EvidenceBuffer,
    partners: list[Any],
    rng: np.random.Generator,
    base_seed: int,
    router: OCV2EvidenceRouter | None = None,
    partner_override: Any | None = None,
) -> tuple[dict[str, np.ndarray], Any, Any]:
    evidence_buffer.reset()
    if router is not None:
        router.reset()
    seed = int(rng.integers(0, 2**31 - 1)) ^ int(base_seed)
    obs, state = env.reset(seed)
    if not partners:
        raise ValueError("_reset_episode() requires at least one partner.")
    partner = partner_override if partner_override is not None else _sample_partner(partners, rng)
    partner.reset(seed)
    return obs, state, partner


def _new_partner_option_inferencer(
    option_lib: OCV2OptionLibrary,
    state: Any,
    config: dict[str, Any] | None = None,
) -> PartnerOptionInferencer:
    """Behavior-only partner option inferencer for P1 de-oracled evidence.

    The inferencer consumes primitive partner actions plus state/event deltas via
    ``option_executor.option_primitive_step``. It never receives partner name,
    protocol, role, terminal_policy, or the scripted true option labels.
    """
    # E2 (review BLOCK fix): the zeroed ablation is eval-only. Fail loudly if a
    # training/CE config ever sets mode=zeroed instead of silently zeroing evidence.
    inferencer = make_behavior_option_inferencer(option_lib, config, require_inferred=True)
    inferencer.reset(state)
    return inferencer




def _belief_persistence_enabled(config: dict[str, Any] | None) -> bool:
    """E3 switch: read ``training.belief_persistence`` (default True = current P4 behavior)."""
    return _as_bool(
        ((config or {}).get("training", {}) or {}).get("belief_persistence", True),
        "belief_persistence",  # _as_bool already prefixes "training."
    )


def _initialise_persistent_belief(
    evidence_buffer: EvidenceBuffer,
    method: str | None,
    belief_model: FactorLocalBeliefModel | None,
    graph: GraphSpec,
    device: torch.device | None,
    persistent: bool = True,
) -> None:
    """Initialise cross-option factor belief state at episode start (P4).

    E3 ablation (EXPERIMENT_CHAIN_PLAN §10.4): ``persistent=False`` takes the same
    no-carry path as a non-belief method, so the belief re-encodes its window from a
    zero hidden at every option decision — i.e. the pre-P4 behavior. Default (True)
    is bit-identical to the current persistent behavior.
    """
    if method not in {"aris_bellman", "flat_factor"} or not persistent:
        evidence_buffer.set_belief_hidden(None)
        return
    if belief_model is None or device is None:
        raise ValueError("persistent belief requires belief_model and device for ARIS/flat methods")
    with torch.no_grad():
        hidden = belief_model.initial_hidden(1, graph.num_factors, device).squeeze(0)
    evidence_buffer.set_belief_hidden(hidden.detach().cpu().numpy())


def _advance_persistent_belief(
    evidence_buffer: EvidenceBuffer,
    method: str | None,
    belief_model: FactorLocalBeliefModel | None,
    graph: GraphSpec,
    device: torch.device | None,
    evidence_row: np.ndarray,
    persistent: bool = True,
) -> None:
    """Transfer factor-belief hidden state after each primitive evidence row (P4).

    E3 ablation: ``persistent=False`` skips the carry entirely — no window-base is
    recorded, so ``belief_window_base_snapshot()`` stays None and the belief re-encodes
    from zeros each decision (pre-P4 behavior). Default (True) is unchanged.
    """
    if method not in {"aris_bellman", "flat_factor"} or not persistent:
        return
    if belief_model is None or device is None:
        raise ValueError("persistent belief update requires belief_model and device")
    hidden_np = evidence_buffer.belief_hidden_snapshot()
    if hidden_np is None:
        _initialise_persistent_belief(evidence_buffer, method, belief_model, graph, device, persistent)
        hidden_np = evidence_buffer.belief_hidden_snapshot()
    if hidden_np is None:
        raise ValueError("persistent belief initialisation failed")
    with torch.no_grad():
        graph_batch = _graph_tensors(graph, 1, device)
        hidden = _tensor(hidden_np[None, ...], device)
        row_np = np.asarray(evidence_row, dtype=np.float32)
        evidence_t = _tensor(row_np[None, ...], device)
        if row_np.ndim != 2 or row_np.shape[0] != graph.num_factors:
            raise ValueError(
                f"evidence_row must have shape [num_factors, {D_EVID}], got {row_np.shape}."
            )
        active_np = row_np[:, EVIDENCE_INDEX["evidence_present"]] > 0.0
        active = torch.as_tensor(active_np[None, ...], dtype=torch.bool, device=device)
        evidence_buffer.record_belief_window_base(hidden_np)
        next_hidden = belief_model.step_history(
            evidence_t,
            hidden,
            factor_features=graph_batch.get("factor_features"),
            factor_mask=graph_batch["factor_mask"],
            active_mask=active,
        ).squeeze(0)
    evidence_buffer.set_belief_hidden(next_hidden.detach().cpu().numpy())


def _empty_event_summary() -> dict[str, Any]:
    return {
        "delivery_event": 0,
        "wrong_delivery_event": 0,
        "ego_delivery_event": 0,
        "partner_delivery_event": 0,
        "ego_correct_delivery": 0,
        "partner_correct_delivery": 0,
        "ego_sole_correct_delivery": 0,
        "ego_wrong_delivery_event": 0,
        "partner_wrong_delivery_event": 0,
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
    }


def _accumulate_event_summary(summary: dict[str, int], event: Any) -> None:
    for key in tuple(summary.keys()):
        summary[key] += int(bool(getattr(event, key, False)))


def _update_task_progress_metrics(metrics: dict[str, Any], transition: OptionTransition) -> None:
    counts = metrics.setdefault("task_progress_counts", _empty_progress_summary())
    summary = transition.event_summary or {}
    if transition.termination_reason == "picked_ingredient":
        counts["picked_ingredient"] = int(counts.get("picked_ingredient", 0)) + 1
    if transition.termination_reason == "ingredient_delivered_to_pot":
        counts["ingredient_delivered_to_pot"] = int(counts.get("ingredient_delivered_to_pot", 0)) + 1
    if transition.termination_reason == "plated_soup":
        counts["plated_soup"] = int(counts.get("plated_soup", 0)) + 1
    delivered = int(summary.get("delivery_event", 0))
    if delivered <= 0 and transition.termination_reason == "served_soup":
        delivered = 1
    counts["served_soup"] = int(counts.get("served_soup", 0)) + delivered
    if transition.termination_reason == "dropped_item_to_counter":
        counts["drop_item_to_counter"] = int(counts.get("drop_item_to_counter", 0)) + 1
    if transition.termination_reason == "cleared_interaction_cell":
        counts["cleared_interaction_cell"] = int(counts.get("cleared_interaction_cell", 0)) + 1
    for key in (
        "pot_became_ready",
        "plate_picked",
        "soup_picked",
        "correct_delivery",
        "wrong_delivery_event",
        "ego_delivery_event",
        "partner_delivery_event",
        "ego_correct_delivery",
        "partner_correct_delivery",
        "ego_sole_correct_delivery",
        "ego_wrong_delivery_event",
        "partner_wrong_delivery_event",
        "collision_or_block",
        "recipe_indicator_event",
        "button_pressed",
        "delivery_event",
        "pot_changed",
    ):
        if key in summary:
            counts[key] = int(counts.get(key, 0)) + int(summary.get(key, 0))


def _training_reward(
    step: Any,
    config: dict[str, Any],
    agent_key: str,
    event: Any,
    *,
    ego_contributed: bool = False,
    include_terminal_shaping: bool = True,
) -> float:
    # RC root-cause fix: the sparse term is the ego's actor-specific delivery
    # credit, not the shared team reward. With sparse_credit="team" (legacy
    # default) this reproduces the old `step.rewards[agent_key]` behaviour
    # exactly; with "ego_delivery"/"ego_correct_delivery" the partner's
    # deliveries no longer leak into the ego option's return. `event` is the
    # OCV2Event for this primitive step (always available at both call sites).
    team_sparse = float(step.rewards.get(agent_key, 0.0))
    sparse_params = sparse_credit_params(config.get("training"))
    sparse = actor_sparse_reward(
        team_sparse,
        event,
        ego_contributed=ego_contributed,
        **{k: v for k, v in sparse_params.items() if k != "partner_terminal_policy"},
    )
    shaped_coef = float(config["training"].get("shaped_reward_coef", 0.0))
    # LDS-B1 (latent-defect sweep): terminal-progress shaping is a TRAINING
    # scaffold. Eval-side return accounting passes include_terminal_shaping=False
    # so shaped (E1-rev) and unshaped (E1) checkpoints report returns on the same
    # scale. Training call sites keep the default True — behaviour unchanged.
    terminal_bonus = 0.0
    if include_terminal_shaping:
        terminal_bonus = terminal_progress_bonus(
            event, params=terminal_progress_params(config.get("training"))
        )
    return (
        sparse
        + terminal_bonus
        + shaped_coef
        * _shaped_reward_for_agent(
            step.info,
            agent_key,
            allow_shared=bool(config["training"].get("allow_shared_shaping", False)),
        )
    )


def _transition_training_return(
    transition: OptionTransition,
    config: dict[str, Any],
) -> float:
    cost_coef = float(config["training"]["cost_coef"])
    return float(transition.reward_sum - cost_coef * transition.realized_cost)


def _shaped_reward_for_agent(
    info: dict[str, Any],
    agent_key: str,
    *,
    allow_shared: bool = False,
) -> float:
    shaped = info.get("shaped_reward", 0.0)
    if isinstance(shaped, dict):
        if agent_key in shaped:
            return _as_float(shaped[agent_key])
        if not allow_shared:
            raise KeyError(
                f"shaped_reward is a dict but has no {agent_key!r} entry; pass "
                "--allow_shared_shaping only for legacy shared-shaping smoke runs."
            )
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
        "episode_return_kind": (
            "reward_sum_including_selected_path_c_probe_cost_"
            "minus_cost_coef_realized_primitive_cost"
        ),
        "path_c_probe_cost_accounting": (
            "A selected probe cost is subtracted exactly once from reward_sum; "
            "primitive-step cost is accounted for separately."
        ),
        "option_durations": [],
        "termination_counts": {},
        "option_kind_stats": {},
        "task_progress_counts": {},
        "checkpoint_load_ok": False,
        "greedy_validation": [],
        "checkpoint_selection": {},
        "option_selection_count": 0,
        "no_valid_option_count": 0,
        "forced_noop_count": 0,
        "partner_option_evidence_policy": PARTNER_OPTION_EVIDENCE_POLICY,
        "oracle_truth_main_path": False,
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
    progress = metrics.get("task_progress_counts", {}) or {}
    terminations = metrics.get("termination_counts", {}) or {}
    total_terminations = max(1, sum(int(v) for v in terminations.values()))
    noop_count = int(terminations.get("noop", 0))
    max_steps_count = int(terminations.get("max_steps", 0))
    selection_count = int(metrics.get("option_selection_count", 0))
    forced_noop_count = int(metrics.get("forced_noop_count", 0))
    no_valid_count = int(metrics.get("no_valid_option_count", 0))
    checkpoint_selection = metrics.get("checkpoint_selection", {}) or {}
    ensemble_accumulator = (
        metrics.get("path_c_ensemble_telemetry_accumulator", {}) or {}
    )
    ensemble_count = int(ensemble_accumulator.get("observation_count", 0))
    correlation_count = int(
        ensemble_accumulator.get("head_correlation_observation_count", 0)
    )
    return_floor_count = int(
        metrics.get("path_c_probe_skipped_return_floor_count", 0)
    )
    floor_adjudication_count = sum(
        int(metrics.get(key, 0))
        for key in (
            "path_c_probe_skipped_return_floor_count",
            "path_c_probe_selected_count",
        )
    )
    return {
        "finite_td_loss": bool(losses.size == 0 or np.all(np.isfinite(losses))),
        "td_loss_first_window": first_mean,
        "td_loss_last_window": last_mean,
        "td_loss_decreased": _loss_decreased(first_mean, last_mean),
        "reward_mean": float(np.mean(rewards)) if rewards.size else None,
        "reward_variance": float(np.var(rewards)) if rewards.size else None,
        "finite_rewards": bool(rewards.size == 0 or np.all(np.isfinite(rewards))),
        "noop_count": int(metrics.get("termination_counts", {}).get("noop", 0)),
        "max_steps_count": int(metrics.get("termination_counts", {}).get("max_steps", 0)),
        "env_max_steps_count": int(
            metrics.get("termination_counts", {}).get("env_max_steps", 0)
        ),
        "option_selection_count": selection_count,
        "forced_noop_count": forced_noop_count,
        "no_valid_option_count": no_valid_count,
        "forced_noop_fraction": float(
            forced_noop_count / max(1, selection_count)
        ),
        "no_valid_option_fraction": float(
            no_valid_count / max(1, selection_count)
        ),
        "path_c_probe_floor_binding": {
            "definition": (
                "return-floor rejections divided by candidates that passed the "
                "disagreement threshold and reached return-floor adjudication"
            ),
            "return_floor_rejection_count": return_floor_count,
            "candidate_decision_count": floor_adjudication_count,
            "rate": (
                float(return_floor_count / floor_adjudication_count)
                if floor_adjudication_count > 0
                else None
            ),
        },
        "path_c_ensemble_telemetry_summary": {
            "definition": ensemble_accumulator.get("definition"),
            "observation_count": ensemble_count,
            "effective_rank_mean": (
                float(ensemble_accumulator.get("effective_rank_sum", 0.0))
                / float(ensemble_count)
                if ensemble_count > 0
                else None
            ),
            "head_correlation_mean": (
                float(ensemble_accumulator.get("head_correlation_sum", 0.0))
                / float(correlation_count)
                if correlation_count > 0
                else None
            ),
            "head_correlation_absolute_mean": (
                float(ensemble_accumulator.get("head_correlation_absolute_sum", 0.0))
                / float(correlation_count)
                if correlation_count > 0
                else None
            ),
            "prior_contribution_ratio_mean": (
                float(
                    ensemble_accumulator.get("prior_contribution_ratio_sum", 0.0)
                )
                / float(ensemble_count)
                if ensemble_count > 0
                else None
            ),
            "latest": ensemble_accumulator.get("latest"),
        },
        "option_kind_stats": metrics.get("option_kind_stats", {}),
        "delivery_event_count": int(metrics.get("task_progress_counts", {}).get("delivery_event", 0)),
        "ego_delivery_event_count": int(
            metrics.get("task_progress_counts", {}).get("ego_delivery_event", 0)
        ),
        "partner_delivery_event_count": int(
            metrics.get("task_progress_counts", {}).get("partner_delivery_event", 0)
        ),
        "pot_changed_count": int(metrics.get("task_progress_counts", {}).get("pot_changed", 0)),
        "plate_picked_count": int(metrics.get("task_progress_counts", {}).get("plate_picked", 0)),
        "soup_picked_count": int(metrics.get("task_progress_counts", {}).get("soup_picked", 0)),
        "plated_soup_count": int(metrics.get("task_progress_counts", {}).get("plated_soup", 0)),
        "served_soup_count": int(metrics.get("task_progress_counts", {}).get("served_soup", 0)),
        "correct_delivery_count": int(
            metrics.get("task_progress_counts", {}).get("correct_delivery", 0)
        ),
        "wrong_delivery_event_count": int(
            metrics.get("task_progress_counts", {}).get("wrong_delivery_event", 0)
        ),
        "ego_correct_delivery_count": int(
            metrics.get("task_progress_counts", {}).get("ego_correct_delivery", 0)
        ),
        "partner_correct_delivery_count": int(
            metrics.get("task_progress_counts", {}).get("partner_correct_delivery", 0)
        ),
        "ego_wrong_delivery_event_count": int(
            metrics.get("task_progress_counts", {}).get("ego_wrong_delivery_event", 0)
        ),
        "partner_wrong_delivery_event_count": int(
            metrics.get("task_progress_counts", {}).get("partner_wrong_delivery_event", 0)
        ),
        "drop_item_to_counter_count": int(
            metrics.get("task_progress_counts", {}).get("drop_item_to_counter", 0)
        ),
        "cleared_interaction_cell_count": int(
            metrics.get("task_progress_counts", {}).get("cleared_interaction_cell", 0)
        ),
        "task_progress_events": int(sum(int(value) for value in progress.values())),
        "selected_checkpoint": checkpoint_selection.get("selected_checkpoint"),
        "selected_checkpoint_by": checkpoint_selection.get("selected_by"),
        "best_greedy_return": checkpoint_selection.get("best_greedy_return"),
        "best_greedy_update": checkpoint_selection.get("best_update"),
        "selected_ego_correct_delivery_count": checkpoint_selection.get(
            "selected_ego_correct_delivery_count"
        ),
        "selected_ego_sole_correct_delivery_count": checkpoint_selection.get(
            "selected_ego_sole_correct_delivery_count"
        ),
        "selected_partner_correct_delivery_count": checkpoint_selection.get(
            "selected_partner_correct_delivery_count"
        ),
        "selected_team_delivery_episode_rate": checkpoint_selection.get(
            "selected_team_delivery_episode_rate"
        ),
        "selected_ego_correct_completion_rate": checkpoint_selection.get(
            "selected_ego_correct_completion_rate"
        ),
        "max_partner_correct_delivery_seen": checkpoint_selection.get(
            "max_partner_correct_delivery_seen"
        ),
        "free_rider_guard": checkpoint_selection.get("free_rider_guard"),
        "free_rider_diagnosis": checkpoint_selection.get("free_rider_diagnosis"),
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
    *,
    filename: str = "checkpoint.pt",
) -> None:
    path_c = path_c_metadata(config)
    payload: dict[str, Any] = {
        "method": method,
        "config": config,
        "graph": graph.to_json_dict(),
        "metrics_summary": _metrics_summary(metrics),
        "path_c_artifact_binding": {
            "preregistration_sha256": (
                (path_c.get("preregistration") or {}).get("sha256")
            ),
            "resolved_path_c_sha256": path_c.get("resolved_path_c_sha256"),
            "ego_evidence_spec_sha256": (
                _sequence_spec(q_net).sha256()
                if q_net is not None and _uses_sequence_q(q_net)
                else None
            ),
        },
    }
    if q_net is not None:
        payload["q_net"] = q_net.state_dict()
        if _uses_sequence_q(q_net):
            payload["ego_evidence_spec"] = _sequence_spec(q_net).to_dict()
    if belief_model is not None:
        payload["belief_model"] = belief_model.state_dict()
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    checkpoint_path = output_dir / filename
    torch.save(payload, checkpoint_path)
    trainable_parameters: dict[str, list[int]] = {}
    for prefix, module in (("q_net", q_net), ("belief_model", belief_model)):
        if module is None:
            continue
        for name, parameter in module.named_parameters():
            if parameter.requires_grad:
                trainable_parameters[f"{prefix}.{name}"] = list(parameter.shape)
    parameter_manifest = {
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "preregistration_sha256": (
            (path_c.get("preregistration") or {}).get("sha256")
        ),
        "resolved_path_c_sha256": path_c.get("resolved_path_c_sha256"),
        "ego_evidence_spec_sha256": (
            _sequence_spec(q_net).sha256()
            if q_net is not None and _uses_sequence_q(q_net)
            else None
        ),
        "trainable_parameters": trainable_parameters,
        "parameter_count": int(sum(
            int(np.prod(np.asarray(shape, dtype=np.int64)))
            for shape in trainable_parameters.values()
        )),
    }
    _write_json(
        checkpoint_path.with_suffix(".parameters.json"),
        parameter_manifest,
    )


def _remove_stale_deployable_checkpoints(
    output_dir: Path,
    metrics: dict[str, Any],
    *,
    reason: str,
) -> None:
    removed: list[str] = []
    for filename in ("checkpoint.pt", "checkpoint_best.pt"):
        path = output_dir / filename
        if path.exists():
            path.unlink()
            removed.append(filename)
    if removed:
        metrics.setdefault("checkpoint_selection", {})[
            "removed_stale_deployable_checkpoints"
        ] = removed
        metrics.setdefault("checkpoint_selection", {})[
            "stale_deployable_removal_reason"
        ] = reason


def _checkpoint_loads(path: Path) -> bool:
    try:
        torch.load(path, map_location="cpu")
    except (OSError, RuntimeError, EOFError, ValueError, pickle.UnpicklingError) as exc:
        warnings.warn(
            f"Checkpoint load validation failed with {type(exc).__name__}: {exc}",
            RuntimeWarning,
        )
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
    if getattr(args, "preflight_path", None) is not None:
        config.setdefault("preflight", {})["path"] = str(args.preflight_path)
    if getattr(args, "path_c_preregistration", None) is not None:
        config.setdefault("path_c", {})["preregistration_path"] = str(
            args.path_c_preregistration
        )
    if getattr(args, "path_c_probe_base_checkpoint", None) is not None:
        config["path_c_secondary_base_checkpoint"] = str(
            args.path_c_probe_base_checkpoint
        )
    if bool(getattr(args, "allow_shared_shaping", False)):
        config["training"]["allow_shared_shaping"] = True
    for key in (
        "td_loss",
        "huber_delta",
        "double_q",
        "advantage_norm",
        "checkpoint_every",
        "select_best_by",
        "checkpoint_eval_episodes",
    ):
        value = getattr(args, key, None)
        if value is not None:
            config["training"][key] = value


def _normalize_training_stability_config(config: dict[str, Any]) -> None:
    train_cfg = config.setdefault("training", {})

    td_loss = str(train_cfg.get("td_loss", "huber")).lower()
    if td_loss not in {"huber", "mse"}:
        raise ValueError("training.td_loss must be one of {'huber', 'mse'}.")
    train_cfg["td_loss"] = td_loss

    huber_delta = float(train_cfg.get("huber_delta", 1.0))
    if huber_delta <= 0.0:
        raise ValueError("training.huber_delta must be positive.")
    train_cfg["huber_delta"] = huber_delta

    train_cfg["double_q"] = _as_bool(train_cfg.get("double_q", True), "double_q")

    advantage_norm = str(train_cfg.get("advantage_norm", "none")).lower()
    if advantage_norm not in {"none", "relevant_count", "sqrt_relevant"}:
        raise ValueError(
            "training.advantage_norm must be one of "
            "{'none', 'relevant_count', 'sqrt_relevant'}."
        )
    train_cfg["advantage_norm"] = advantage_norm

    checkpoint_every = int(train_cfg.get("checkpoint_every", 0))
    if checkpoint_every < 0:
        raise ValueError("training.checkpoint_every must be non-negative.")
    train_cfg["checkpoint_every"] = checkpoint_every

    select_best_by = str(train_cfg.get("select_best_by", "greedy")).lower()
    if select_best_by not in {"greedy", "final"}:
        raise ValueError("training.select_best_by must be one of {'greedy', 'final'}.")
    train_cfg["select_best_by"] = select_best_by

    checkpoint_eval_episodes = int(train_cfg.get("checkpoint_eval_episodes", 3))
    if checkpoint_eval_episodes <= 0:
        raise ValueError("training.checkpoint_eval_episodes must be positive.")
    train_cfg["checkpoint_eval_episodes"] = checkpoint_eval_episodes

    # RC free-rider guard wiring check: the guard is only enforceable when greedy
    # validation actually runs (it reads the selected checkpoint's ego deliveries).
    # Fail fast instead of letting an opted-in hard gate silently no-op. Reads the
    # EFFECTIVE (normalized) select_best_by / checkpoint_every, not logged intent.
    require_ego_delivery_selection = _as_bool(
        train_cfg.get("require_ego_delivery_selection", False),
        "require_ego_delivery_selection",
    )
    train_cfg["require_ego_delivery_selection"] = require_ego_delivery_selection
    if require_ego_delivery_selection and not (
        checkpoint_every > 0 and select_best_by == "greedy"
    ):
        raise ValueError(
            "training.require_ego_delivery_selection=true requires greedy validation "
            "(select_best_by='greedy' and checkpoint_every>0); otherwise the free-rider "
            "guard has no greedy-validation deliveries to check and would silently no-op."
        )

    oracle_role_ablation = _as_bool(
        train_cfg.get("oracle_role_conditioned_ablation", False),
        "oracle_role_conditioned_ablation",
    )
    train_cfg["oracle_role_conditioned_ablation"] = oracle_role_ablation
    if not oracle_role_ablation:
        sparse_credit = str(train_cfg.get("sparse_credit", "team"))
        if sparse_credit == "role_contrib_team":
            raise ValueError(
                "P5: sparse_credit='role_contrib_team' branches on true "
                "partner terminal_policy and is allowed only under "
                "training.oracle_role_conditioned_ablation=true, not in the "
                "main black-box method path."
            )
        role_exploration = train_cfg.get("role_exploration") or {}
        if bool(role_exploration.get("enabled", False)):
            raise ValueError(
                "P5: training.role_exploration.enabled=true is a true-role "
                "curriculum and is allowed only under "
                "training.oracle_role_conditioned_ablation=true."
            )
        role_replay_seed = train_cfg.get("role_replay_seed") or {}
        if bool(role_replay_seed.get("enabled", False)):
            raise ValueError(
                "P5: training.role_replay_seed.enabled=true is a true-role "
                "seeded-replay ablation and is allowed only under "
                "training.oracle_role_conditioned_ablation=true."
            )
        terminal_replay_seed = train_cfg.get("terminal_replay_seed") or {}
        if bool(terminal_replay_seed.get("enabled", False)):
            raise ValueError(
                "P5: training.terminal_replay_seed.enabled=true seeds replay by a "
                "named terminal-policy partner and is allowed only under "
                "training.oracle_role_conditioned_ablation=true, not in the main "
                "black-box method path."
            )


def _checkpoint_policy(config: dict[str, Any]) -> dict[str, Any]:
    train_cfg = config["training"]
    checkpoint_every = int(train_cfg.get("checkpoint_every", 0))
    select_best_by = str(train_cfg.get("select_best_by", "greedy")).lower()
    return {
        "checkpoint_every": checkpoint_every,
        "select_best_by": select_best_by,
        "checkpoint_eval_episodes": int(train_cfg.get("checkpoint_eval_episodes", 3)),
        "greedy_enabled": bool(
            checkpoint_every > 0 and select_best_by == "greedy"
        ),
    }


def _as_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    raise ValueError(f"training.{name} must be a boolean.")


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
    selection_stats: dict[str, Any] | None = None,
) -> int:
    _record_selection_attempt(selection_stats)
    valid_ids = np.flatnonzero(option_lib.valid_options(state, agent_id))
    if valid_ids.size:
        return int(rng.choice(valid_ids))
    _record_forced_noop(selection_stats)
    return _noop_option_id(option_lib)


def _record_selection_attempt(selection_stats: dict[str, Any] | None) -> None:
    if selection_stats is None:
        return
    selection_stats["option_selection_count"] = int(
        selection_stats.get("option_selection_count", 0)
    ) + 1


def _record_path_c_ensemble_telemetry(
    selection_stats: dict[str, Any] | None,
    observation: dict[str, Any],
) -> None:
    """Accumulate auditable ensemble diagnostics without affecting selection."""

    if selection_stats is None:
        return
    block = selection_stats.setdefault(
        "path_c_ensemble_telemetry_accumulator",
        {
            "definition": observation["definition"],
            "observation_count": 0,
            "effective_rank_sum": 0.0,
            "prior_contribution_ratio_sum": 0.0,
            "head_correlation_observation_count": 0,
            "head_correlation_sum": 0.0,
            "head_correlation_absolute_sum": 0.0,
            "latest": None,
        },
    )
    if block.get("definition") != observation.get("definition"):
        raise ValueError("Path C ensemble telemetry definition changed within one run.")
    for key in ("effective_rank", "prior_contribution_ratio"):
        value = float(observation[key])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"Path C ensemble telemetry {key} must be finite and non-negative.")
    block["observation_count"] = int(block["observation_count"]) + 1
    block["effective_rank_sum"] = float(block["effective_rank_sum"]) + float(
        observation["effective_rank"]
    )
    block["prior_contribution_ratio_sum"] = float(
        block["prior_contribution_ratio_sum"]
    ) + float(observation["prior_contribution_ratio"])
    correlation = observation.get("head_correlation_mean")
    absolute_correlation = observation.get("head_correlation_absolute_mean")
    if correlation is not None or absolute_correlation is not None:
        if correlation is None or absolute_correlation is None:
            raise ValueError("Head-correlation telemetry fields must be jointly available.")
        correlation = float(correlation)
        absolute_correlation = float(absolute_correlation)
        if not math.isfinite(correlation) or not math.isfinite(absolute_correlation):
            raise ValueError("Head-correlation telemetry must be finite.")
        block["head_correlation_observation_count"] = int(
            block["head_correlation_observation_count"]
        ) + 1
        block["head_correlation_sum"] = float(block["head_correlation_sum"]) + correlation
        block["head_correlation_absolute_sum"] = float(
            block["head_correlation_absolute_sum"]
        ) + absolute_correlation
    block["latest"] = dict(observation)


def _record_forced_noop(selection_stats: dict[str, Any] | None) -> None:
    if selection_stats is None:
        return
    selection_stats["no_valid_option_count"] = int(
        selection_stats.get("no_valid_option_count", 0)
    ) + 1
    selection_stats["forced_noop_count"] = int(
        selection_stats.get("forced_noop_count", 0)
    ) + 1


def _noop_option_id(option_lib: OCV2OptionLibrary) -> int:
    for opt in option_lib.options:
        if opt.kind == "noop":
            return int(opt.id)
    raise ValueError("The registered option library has no noop fallback option.")


def _obs_vector(obs: dict[str, np.ndarray], agent_key: str) -> np.ndarray:
    return np.asarray(obs[agent_key], dtype=np.float32)


def _optional_tensor(value: Any, device: torch.device) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, list) and all(item is None for item in value):
        return None
    return _tensor(value, device)


def _tensor(value: Any, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(value).copy(), dtype=torch.float32, device=device)


def _trainable_params(
    q_net: nn.Module,
    belief_model: FactorLocalBeliefModel,
    method: str,
) -> list[nn.Parameter]:
    params = list(q_net.parameters())
    if method in {"aris_bellman", "flat_factor"} and not _uses_sequence_q(q_net):
        params += list(belief_model.parameters())
    return [parameter for parameter in params if parameter.requires_grad]


def _mask_q_values(
    q_values: torch.Tensor,
    option_mask: torch.Tensor | None,
) -> torch.Tensor:
    if option_mask is None:
        return q_values
    return q_values.masked_fill(~option_mask.bool(), -1e9)


def _validated_q_bound_vmax(q_bound_vmax: float | None) -> float | None:
    if q_bound_vmax is None:
        return None
    vmax = float(q_bound_vmax)
    if vmax <= 0.0:
        raise ValueError("value_bound.vmax must be positive when apply_to_all_methods is enabled.")
    return vmax


def _apply_final_q_value_bound(
    q_values: torch.Tensor,
    q_bound_vmax: float | None,
) -> torch.Tensor:
    if q_bound_vmax is None:
        return q_values
    vmax = float(q_bound_vmax)
    return vmax * torch.tanh(q_values / vmax)


def _increment_count(counts: dict[str, int], key: str) -> None:
    counts[key] = int(counts.get(key, 0)) + 1


def _update_option_kind_metrics(
    metrics: dict[str, Any],
    option_kind: str,
    termination_reason: str,
) -> None:
    stats = metrics.setdefault("option_kind_stats", {})
    item = stats.setdefault(
        str(option_kind),
        {
            "attempt_count": 0,
            "success_count": 0,
            "timeout_count": 0,
            "success_rate": 0.0,
            "termination_reason_histogram": {},
        },
    )
    item["attempt_count"] = int(item["attempt_count"]) + 1
    item["success_count"] = int(item["success_count"]) + int(
        option_success(option_kind, termination_reason)
    )
    item["timeout_count"] = int(item["timeout_count"]) + int(
        str(termination_reason) in {"max_steps", "env_max_steps"}
    )
    histogram = item["termination_reason_histogram"]
    reason = str(termination_reason)
    histogram[reason] = int(histogram.get(reason, 0)) + 1
    item["success_rate"] = float(
        int(item["success_count"]) / max(1, int(item["attempt_count"]))
    )


def _empty_progress_summary() -> dict[str, int]:
    return {
        "picked_ingredient": 0,
        "ingredient_delivered_to_pot": 0,
        "pot_became_ready": 0,
        "plate_picked": 0,
        "plated_soup": 0,
        "served_soup": 0,
        "drop_item_to_counter": 0,
        "cleared_interaction_cell": 0,
        "soup_picked": 0,
        "correct_delivery": 0,
        "delivery_event": 0,
        "pot_changed": 0,
        "wrong_delivery_event": 0,
        "ego_delivery_event": 0,
        "partner_delivery_event": 0,
        "ego_correct_delivery": 0,
        "partner_correct_delivery": 0,
        "ego_sole_correct_delivery": 0,
        "ego_wrong_delivery_event": 0,
        "partner_wrong_delivery_event": 0,
        "collision_or_block": 0,
        "recipe_indicator_event": 0,
        "button_pressed": 0,
    }


def _accumulate_progress_summary(summary: dict[str, int], event: Any) -> None:
    if bool(getattr(event, "ego_inventory_before", 0) == 0 and getattr(event, "ego_inventory_after", 0) != 0):
        # Inventory changes are coarse; specific categories below add more semantics.
        pass
    summary["ingredient_delivered_to_pot"] += int(bool(getattr(event, "pot_became_full", False)))
    summary["pot_became_ready"] += int(bool(getattr(event, "pot_became_ready", False)))
    summary["plate_picked"] += int(bool(getattr(event, "plate_picked", False)))
    summary["soup_picked"] += int(bool(getattr(event, "soup_picked", False)))
    summary["delivery_event"] += int(bool(getattr(event, "delivery_event", False)))
    summary["correct_delivery"] += int(bool(getattr(event, "correct_delivery", False)))
    summary["wrong_delivery_event"] += int(bool(getattr(event, "wrong_delivery_event", False)))
    summary["ego_delivery_event"] += int(bool(getattr(event, "ego_delivery_event", False)))
    summary["partner_delivery_event"] += int(bool(getattr(event, "partner_delivery_event", False)))
    summary["ego_correct_delivery"] += int(bool(getattr(event, "ego_correct_delivery", False)))
    summary["partner_correct_delivery"] += int(bool(getattr(event, "partner_correct_delivery", False)))
    summary["ego_wrong_delivery_event"] += int(bool(getattr(event, "ego_wrong_delivery_event", False)))
    summary["partner_wrong_delivery_event"] += int(bool(getattr(event, "partner_wrong_delivery_event", False)))
    summary["collision_or_block"] += int(bool(getattr(event, "collision_or_block", False)))
    summary["recipe_indicator_event"] += int(bool(getattr(event, "recipe_indicator_event", False)))
    summary["button_pressed"] += int(bool(getattr(event, "button_pressed", False)))
    # Treat ingredient pickup as any inventory pickup that is not plate/soup.
    before = int(getattr(event, "ego_inventory_before", 0))
    after = int(getattr(event, "ego_inventory_after", 0))
    if before == 0 and after != 0 and not bool(getattr(event, "plate_picked", False)) and not bool(getattr(event, "soup_picked", False)):
        summary["picked_ingredient"] += 1


def _merge_progress_counts(metrics: dict[str, Any], summary: dict[str, Any] | None) -> None:
    if summary is None:
        return
    counts = metrics.setdefault("progress_counts", _empty_progress_summary())
    for key in _empty_progress_summary():
        counts[key] = int(counts.get(key, 0)) + int(summary.get(key, 0))


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
    parser.add_argument("--preflight_path", default=None)
    parser.add_argument(
        "--path_c_preregistration",
        default=None,
        help=(
            "Optional Path C preregistration YAML. Required when any path_c feature "
            "is enabled; read-only and recorded in resolved_config/metrics."
        ),
    )
    parser.add_argument(
        "--path_c_probe_base_checkpoint",
        default=None,
        help=(
            "Optional frozen base_only checkpoint for secondary residual "
            "visualization; it never drives Path C probe selection."
        ),
    )
    parser.add_argument("--td_loss", choices=("huber", "mse"), default=None)
    parser.add_argument("--huber_delta", type=float, default=None)
    double_q_group = parser.add_mutually_exclusive_group()
    double_q_group.add_argument(
        "--double_q",
        dest="double_q",
        action="store_true",
        default=None,
    )
    double_q_group.add_argument("--no_double_q", dest="double_q", action="store_false")
    parser.add_argument(
        "--advantage_norm",
        choices=("none", "relevant_count", "sqrt_relevant"),
        default=None,
    )
    parser.add_argument("--checkpoint_every", type=int, default=None)
    parser.add_argument(
        "--save_all_checkpoints",
        action="store_true",
        help="RC-1 diagnostic: also persist checkpoint_u<update>.pt at every greedy-validation "
        "interval (does not change training/selection; for per-checkpoint Q-decomposition audit).",
    )
    parser.add_argument("--select_best_by", choices=("greedy", "final"), default=None)
    parser.add_argument("--checkpoint_eval_episodes", type=int, default=None)
    parser.add_argument(
        "--allow_shared_shaping",
        action="store_true",
        help="Legacy smoke escape hatch: sum shared shaped_reward dicts when agent_0 is absent.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    metrics = train(args)
    print(json.dumps(_metrics_summary(metrics), indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
