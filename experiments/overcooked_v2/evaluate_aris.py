from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

if __package__ in {None, ""}:  # pragma: no cover - script execution path
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.aris_bellman.factor_belief import FactorLocalBeliefModel
from src.aris_bellman.replay import EvidenceBuffer
from src.aris_bellman.specs import GraphSpec

from experiments.overcooked_v2.diagnostics import (
    belief_swap_top_pairs,
    diagnostic_cost,
    factor_deletion_return_drop,
    graph_with_deleted_factor,
    mutual_information_proxy,
    realized_delta_info,
    reference_gap_closure,
)
from experiments.overcooked_v2.env_adapter import OCV2Adapter
from experiments.overcooked_v2.evidence_router import D_EVID, OCV2EvidenceRouter
from experiments.overcooked_v2.layout_parser import LayoutGraph, parse_layout
from experiments.overcooked_v2.obs_featurizer import NumpyFeaturizer
from experiments.overcooked_v2.obs_encoder import infer_obs_dim
from experiments.overcooked_v2.option_termination import OptionRuntime, option_success
from experiments.overcooked_v2.option_executor import option_primitive_step
from experiments.overcooked_v2.option_inferencer import (
    PartnerOptionInferencer,
    make_behavior_option_inferencer,
)
from experiments.overcooked_v2.options import OCV2OptionLibrary
from experiments.overcooked_v2.path_c_config import (
    normalize_path_c_config,
    path_c_metadata,
)
from experiments.overcooked_v2.path_c_evaluation import load_frozen_preregistration
from experiments.overcooked_v2.path_c_sequence import (
    DecisionEvidenceBuffer,
    EgoEvidenceSpecV1,
    EpisodeEvidenceBuffer,
)
from experiments.overcooked_v2.path_c_response_summary import ResponseSummarySpecV1
from experiments.overcooked_v2.path_c_protocol import (
    ActingEnvironmentStep,
    ActingEpisodeSpec,
    ActingTransition,
    DirectInformationActingPolicy,
    FORMAL_ACTING_POLICY_NAMES,
    FormalActingBenchmarkContract,
    FormalActingPolicyFactoryEntry,
    FormalActingPolicyFactoryRegistry,
    NoProbeActingPolicy,
    PathCActingEnvironment,
    PolicyAction,
    ProbeBudgetState,
    RandomProbeActingPolicy,
)
from experiments.overcooked_v2.baselines.belief_filter import (
    BayesianHMMBeliefFilter,
    BeliefFilterActingPolicy,
    HMMFilterConfig,
    ParticleBeliefFilter,
    ParticleFilterConfig,
)
from experiments.overcooked_v2.partner_pool import make_training_partners
from experiments.overcooked_v2.provenance import (
    option_library_hash,
    reward_config_payload,
    runtime_provenance,
    sha256_file,
    sha256_json,
)
from experiments.overcooked_v2.reward_design import (
    ContributionLedger,
    terminal_progress_params,
)
from experiments.overcooked_v2.sparse_credit import sparse_credit_params
from experiments.overcooked_v2.state_utils import (
    agent_facing_pos, get_agent_pos, get_inventory,
    get_pot_contents, is_pot_cooking, is_pot_ready_for_plate,
)
from jaxmarl.environments.overcooked_v2.common import Actions as _OCActions
from experiments.overcooked_v2.train_aris import (
    _advance_persistent_belief,
    _belief_persistence_enabled,
    _build_belief_model,
    _build_env,
    _build_option_lib,
    _build_q_network,
    _graph_objective_metadata_status,
    _graph_tensors,
    _initialise_persistent_belief,
    _load_path_c_probe_baseline_q_net,
    _obs_vector,
    _partner_id_tensor,
    _path_c_probe_choice,
    _path_c_reset_probe_decision,
    _q_forward_kwargs,
    _select_train_partners,
    _state_repr,
    _start_sequence_episode,
    _sequence_spec,
    _sequence_step_return,
    _tensor,
    _training_reward,
    _uses_sequence_q,
    _path_c_progress_event_vector,
)


@dataclass
class EvalContext:
    checkpoint_path: Path
    config: dict[str, Any]
    graph: GraphSpec
    method: str
    graph_variant: str
    seed_name: str
    q_net: torch.nn.Module
    belief_model: FactorLocalBeliefModel
    layout_graph: LayoutGraph
    option_lib: OCV2OptionLibrary
    obs_dim: int
    # Diagnostic-only hooks (default off → normal eval/selection unchanged). RC-1: when
    # `qaudit` is a list, _select_option appends a read-only (q_total, q_base) decomposition
    # per greedy decision. RC-2: when `scripted_priority` is a kind-priority list, selection
    # follows that priority over valid options instead of argmax-Q (bypasses the network).
    qaudit: list | None = None
    scripted_priority: list[str] | None = None
    # RC-2b per-step option-execution trace (diagnostic; default off). When trace_steps is a
    # list, _execute_eval_option appends one record per primitive step for options whose kind
    # equals trace_kind (or all kinds if trace_kind is None).
    trace_steps: list | None = None
    trace_kind: str | None = None
    # RC-2b state-aware scripted oracle: callable(ctx, state, valid_ids) -> option_id, used by the
    # reachability probe to drive the full pipeline with stage/inventory awareness (bypasses Q).
    scripted_fsm: object | None = None
    # sec18.14 (formal blind round): eval-only partner-registry override. When set,
    # ALL partner lookups for this context (main eval, random/reference baselines,
    # factor-deletion rollouts) resolve from this registry instead of the config's
    # training partner_set. The config itself is never mutated (the graph-objective
    # gate compares config.partner_set against graph metadata).
    partner_set_override: str | None = None
    # Path C exploratory collection and formal acting-budget evaluation are
    # separate. The latter remains benchmark-return eligible.
    active_probe_collection: bool = False
    path_c_probe_base_q_net: torch.nn.Module | None = None
    path_c_acting_probe_budget: int | None = None
    path_c_acting_probe_remaining: int | None = None
    path_c_probe_cost_per_use: float = 0.0


class OCV2FormalActingEnvironment:
    """Option-boundary OCV2 adapter used by every formal acting policy."""

    def __init__(
        self,
        *,
        context: EvalContext,
        spec: ActingEpisodeSpec,
        partner_name: str,
        max_episode_options: int,
        response_summary_spec: ResponseSummarySpecV1,
        probe_action_ids: Sequence[int],
    ) -> None:
        if int(max_episode_options) <= 0:
            raise ValueError("Formal acting max_episode_options must be positive.")
        if not isinstance(response_summary_spec, ResponseSummarySpecV1):
            raise TypeError("Formal acting requires a frozen ResponseSummarySpecV1.")
        self.context = context
        self.spec = spec
        self.partner_name = str(partner_name)
        self.max_episode_options = int(max_episode_options)
        self.response_summary_spec = response_summary_spec
        self.response_token_vocabulary = tuple(response_summary_spec.vocabulary)
        raw_probe_action_ids = tuple(probe_action_ids)
        if not raw_probe_action_ids or any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            for value in raw_probe_action_ids
        ):
            raise ValueError("Formal acting requires explicit integer probe action ids.")
        self.probe_action_ids = tuple(sorted(set(map(int, raw_probe_action_ids))))
        if self.probe_action_ids[0] < 0 or self.probe_action_ids[-1] >= int(
            context.graph.num_options
        ):
            raise ValueError("Formal acting probe action id is outside action support.")
        if set(self.probe_action_ids) == set(range(int(context.graph.num_options))):
            raise ValueError(
                "Formal acting probe actions must be a strict subset of action support."
            )
        self.env: OCV2Adapter | None = None
        self.partner: Any | None = None
        self.router: OCV2EvidenceRouter | None = None
        self.evidence_buffer: EvidenceBuffer | None = None
        self.partner_option_inferencer: PartnerOptionInferencer | None = None
        self.contribution_ledger: ContributionLedger | None = None
        self.rng = np.random.default_rng(0)
        self.obs: dict[str, np.ndarray] | None = None
        self.option_count = 0

    @property
    def partner_id(self) -> int:
        if self.partner is None:
            raise RuntimeError("Formal acting environment has not been reset.")
        return int(getattr(self.partner, "partner_id", 0))

    def reset(self, *, seed: int) -> Mapping[str, Any]:
        ctx = self.context
        self.rng = np.random.default_rng(int(seed))
        self.env = _build_env(ctx.graph.layout_name, ctx.config)
        self.env.set_featurizer(NumpyFeaturizer(ctx.layout_graph))
        self.router = OCV2EvidenceRouter(
            ctx.graph,
            ctx.layout_graph.cell_to_entity,
            ctx.layout_graph.region_cells,
            evidence_policy=_evidence_policy_for_config(ctx.config),
        )
        partner_set = (
            getattr(ctx, "partner_set_override", None)
            or str(ctx.config.get("training", {}).get("partner_set", "standard7"))
        )
        partners = {
            partner.name: partner
            for partner in make_training_partners(ctx.option_lib, partner_set=partner_set)
        }
        if self.partner_name not in partners:
            raise KeyError(
                f"Formal acting partner {self.partner_name!r} is unavailable; "
                f"choices={sorted(partners)}."
            )
        self.partner = partners[self.partner_name]
        self.evidence_buffer = EvidenceBuffer(
            num_factors=ctx.graph.num_factors,
            window=int(ctx.config["training"]["evidence_window"]),
            evidence_dim=D_EVID,
        )
        self.evidence_buffer.reset()
        self.router.reset()
        self.obs, state = self.env.reset(int(seed))
        self.partner.reset(int(seed))
        _start_sequence_episode(
            self.evidence_buffer,
            ctx.q_net,
            self.obs,
            state,
            ctx.option_lib,
            ctx.config,
            episode_id=self.spec.episode_uid,
            manifest_seed=int(seed),
        )
        self.partner_option_inferencer = make_behavior_option_inferencer(
            ctx.option_lib,
            ctx.config,
        )
        self.partner_option_inferencer.reset(state)
        _initialise_persistent_belief(
            self.evidence_buffer,
            ctx.method,
            ctx.belief_model,
            ctx.graph,
            torch.device("cpu"),
            _belief_persistence_enabled(ctx.config),
        )
        self.contribution_ledger = ContributionLedger.from_config(
            ctx.config.get("training", {})
        )
        self.option_count = 0
        initial_response_id = self.response_summary_spec.encode(
            "no_response",
            0,
        )
        return self._observation(
            self.response_summary_spec.decode(initial_response_id),
            response_token_id=initial_response_id,
        )

    def step(self, action_id: int) -> ActingEnvironmentStep:
        return self._step(int(action_id), realized_probe_cost=0.0)

    def step_policy_action(
        self,
        action: PolicyAction,
        *,
        realized_probe_cost: float,
    ) -> ActingEnvironmentStep:
        if action.is_probe and int(action.action_id) not in self.probe_action_ids:
            raise ValueError(
                "An action outside the frozen probe-action set cannot be charged as a probe."
            )
        expected = float(action.estimated_probe_cost) if action.is_probe else 0.0
        if not np.isclose(
            float(realized_probe_cost), expected, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError("Formal acting environment received inconsistent probe cost.")
        return self._step(
            int(action.action_id),
            realized_probe_cost=float(realized_probe_cost),
        )

    def _step(self, action_id: int, *, realized_probe_cost: float) -> ActingEnvironmentStep:
        if any(
            value is None
            for value in (
                self.env,
                self.partner,
                self.router,
                self.evidence_buffer,
                self.partner_option_inferencer,
                self.contribution_ledger,
                self.obs,
            )
        ):
            raise RuntimeError("Formal acting environment step called before reset.")
        next_count = self.option_count + 1
        option_return, done, next_obs, info = _execute_eval_option(
            self.context,
            self.env,
            self.obs,
            self.partner,
            self.router,
            self.context.graph,
            self.evidence_buffer,
            int(action_id),
            False,
            False,
            self.rng,
            contribution_ledger=self.contribution_ledger,
            partner_option_inferencer=self.partner_option_inferencer,
            truncate_at_boundary=(next_count >= self.max_episode_options),
            probe_cost=float(realized_probe_cost),
        )
        self.option_count = next_count
        self.obs = next_obs
        truncated = bool(not done and self.option_count >= self.max_episode_options)
        termination_reason = str(info["termination_reason"])
        response_id = self._canonical_response_token_id(
            info,
            terminated=bool(done),
            truncated=truncated,
        )
        token = self.response_summary_spec.decode(response_id)
        return ActingEnvironmentStep(
            observation=self._observation(token, response_token_id=response_id),
            reward=float(option_return),
            terminated=bool(done),
            truncated=truncated,
            environment_steps=int(info["primitive_steps"]),
            info={
                "termination_reason": termination_reason,
                "response_token_id": int(response_id),
                "response_token": token,
            },
        )

    def _canonical_response_token_id(
        self,
        info: Mapping[str, Any],
        *,
        terminated: bool,
        truncated: bool,
    ) -> int:
        """Encode one completed option into the frozen finite response alphabet."""

        if terminated:
            return self.response_summary_spec.encode(terminal=True)
        if truncated:
            return self.response_summary_spec.encode(censored=True)
        counts = info.get("delivery_counts") or {}
        if not isinstance(counts, Mapping):
            raise TypeError("Formal acting delivery counts must be a mapping.")
        reason = str(info.get("termination_reason", ""))
        if int(counts.get("wrong_delivery_event", 0)) > 0:
            response_class = "delivery_failure"
        elif int(counts.get("correct_delivery", 0)) > 0 or int(
            counts.get("delivery_event", 0)
        ) > 0:
            response_class = "delivery_success"
        elif "block" in reason or "collision" in reason:
            response_class = "block"
        elif "wait" in reason or reason == "noop":
            response_class = "wait"
        elif reason in {"budget_exhausted", "max_steps", "env_max_steps"}:
            response_class = "no_response"
        else:
            response_class = "progress"
        if response_class not in self.response_summary_spec.response_classes:
            raise ValueError(
                f"Frozen response vocabulary lacks class {response_class!r}."
            )
        latency = int(info.get("primitive_steps", 0))
        if latency < 0:
            raise ValueError("Formal acting response latency cannot be negative.")
        return self.response_summary_spec.encode(response_class, latency)

    def _observation(
        self,
        response_token: str,
        *,
        response_token_id: int | None = None,
    ) -> dict[str, Any]:
        if self.env is None or self.obs is None:
            raise RuntimeError("Formal acting observation requested before reset.")
        valid = tuple(
            int(value)
            for value in np.flatnonzero(
                self.context.option_lib.valid_options(self.env.state, 0)
            ).tolist()
        )
        token_id = (
            self.response_summary_spec.encode("no_response", 0)
            if response_token_id is None
            else int(response_token_id)
        )
        if self.response_summary_spec.decode(token_id) != str(response_token):
            raise ValueError("Formal acting response token and id disagree.")
        probe_candidates = tuple(
            action_id for action_id in valid if action_id in self.probe_action_ids
        )
        if valid and set(probe_candidates) == set(valid):
            raise ValueError(
                "A formal acting state must retain at least one valid non-probe action."
            )
        return {
            "agent_0": np.asarray(self.obs["agent_0"], dtype=np.float32).copy(),
            "belief_public_context": np.asarray(
                self.obs["agent_0"],
                dtype=np.float32,
            ).copy(),
            "valid_action_ids": valid,
            "probe_candidate_action_ids": probe_candidates,
            "probe_only_action_ids": (),
            "belief_observation_token": str(response_token),
            "response_token_id": token_id,
        }


class RecurrentContextActingPolicy:
    """Use a loaded recurrent EvalContext through the common acting protocol."""

    oracle_baseline = False

    def __init__(
        self,
        *,
        policy_name: str,
        context: EvalContext,
        environment: OCV2FormalActingEnvironment,
        spec: ActingEpisodeSpec,
        probing: bool,
        probe_budget_grid_sha256: str,
        benchmark_resources: Mapping[str, Any],
    ) -> None:
        if not _uses_sequence_q(context.q_net):
            raise ValueError(f"{policy_name} requires a recurrent sequence checkpoint.")
        if environment.evidence_buffer is not None:
            raise ValueError("Recurrent policy must be constructed before environment reset.")
        if _sequence_spec(context.q_net).sha256() != _sequence_spec(
            environment.context.q_net
        ).sha256():
            raise ValueError("Recurrent acting checkpoints use different evidence specs.")
        self.policy_name = str(policy_name)
        self.context = context
        self.environment = environment
        self.spec = spec
        self.probing = bool(probing)
        self.probe_budget_grid_sha256 = str(probe_budget_grid_sha256)
        self.benchmark_resources = dict(benchmark_resources)
        self._probe_count = 0
        self._environment_steps = 0
        self._realized_probe_cost = 0.0
        self._action_count = 0
        self._last_representation = np.zeros(1, dtype=np.float32)
        self._rng = np.random.default_rng(0)
        self._probe_cost_per_use = 0.0

    def reset(self, *, seed: int, probe_budget_state: ProbeBudgetState) -> None:
        self._rng = np.random.default_rng(int(seed))
        self._probe_cost_per_use = float(probe_budget_state.cost_per_probe)
        self._probe_count = 0
        self._environment_steps = 0
        self._realized_probe_cost = 0.0
        self._action_count = 0
        self._last_representation = np.zeros(1, dtype=np.float32)

    def act(
        self,
        observation: Mapping[str, Any],
        probe_budget_state: ProbeBudgetState,
    ) -> PolicyAction:
        evidence_buffer = self.environment.evidence_buffer
        if evidence_buffer is None:
            raise RuntimeError("Recurrent acting policy used before environment reset.")
        episode = evidence_buffer.sequence_episode()
        if not isinstance(episode, EpisodeEvidenceBuffer):
            raise RuntimeError("Recurrent acting environment lacks sequence evidence.")
        batch = episode.evidence_batch(torch.device("cpu"))
        valid_ids = tuple(int(value) for value in observation.get("valid_action_ids", ()))
        if not valid_ids:
            raise ValueError("Recurrent acting policy requires valid actions.")
        if (
            len(valid_ids) != len(set(valid_ids))
            or any(
                action_id < 0 or action_id >= self.context.graph.num_options
                for action_id in valid_ids
            )
        ):
            raise ValueError(
                "Recurrent acting valid-action ids must be unique and registered."
            )
        valid = torch.zeros(self.context.graph.num_options, dtype=torch.bool)
        valid[list(valid_ids)] = True
        if not torch.equal(batch.valid_actions[0, -1], valid):
            raise RuntimeError(
                "Formal acting observation and sequence history disagree on action support."
            )
        with torch.no_grad():
            head_trace, representation = self.context.q_net.forward_sequence(batch)
        head_values = head_trace[:, -1]
        q_values = head_values.mean(dim=1).squeeze(0).masked_fill(~valid, -1.0e9)
        self._last_representation = (
            representation[:, -1].detach().cpu().numpy().reshape(-1).astype(np.float32)
        )
        selected_probe: int | None = None
        decision: Mapping[str, Any] = {}
        if self.probing and probe_budget_state.remaining > 0:
            probe_candidate_ids = tuple(
                int(value)
                for value in observation.get("probe_candidate_action_ids", ())
            )
            if (
                len(probe_candidate_ids) != len(set(probe_candidate_ids))
                or any(action_id not in valid_ids for action_id in probe_candidate_ids)
            ):
                raise ValueError(
                    "Probe-candidate action ids must be unique members of valid_action_ids."
                )
            probe_candidates = torch.zeros(
                self.context.graph.num_options,
                dtype=torch.bool,
            )
            if probe_candidate_ids:
                probe_candidates[list(probe_candidate_ids)] = True
            configured_cost = (
                (self.context.config.get("path_c") or {})
                .get("preregistration", {})
                .get("probe_cost_per_use")
            )
            if configured_cost is None or not np.isclose(
                float(configured_cost),
                float(probe_budget_state.cost_per_probe),
                rtol=0.0,
                atol=0.0,
            ):
                raise ValueError("Recurrent probing checkpoint changed the frozen probe cost.")
            selection_stats: dict[str, Any] = {"option_selection_count": self._action_count + 1}
            _path_c_reset_probe_decision(selection_stats)
            selected_probe = _path_c_probe_choice(
                self.context.q_net,
                _tensor(
                    np.asarray(observation["agent_0"], dtype=np.float32)[None, ...],
                    torch.device("cpu"),
                ),
                None,
                _graph_tensors(self.context.graph, 1, torch.device("cpu")),
                q_values,
                probe_candidates,
                self.context.config,
                partner_id=self.environment.partner_id,
                device=torch.device("cpu"),
                selection_stats=selection_stats,
                eval_mode=False,
                probe_rng=self._rng,
                head_values_override=head_values,
            )
            decision = selection_stats.get("path_c_probe_last_decision") or {}
            if selected_probe is not None and int(selected_probe) not in probe_candidate_ids:
                raise RuntimeError(
                    "Probe selection returned an action outside the frozen probe candidates."
                )
        is_probe = selected_probe is not None
        selected = (
            int(selected_probe)
            if is_probe
            else int(torch.argmax(q_values).item())
        )
        if is_probe and decision.get("candidate_scores") is not None:
            all_scores = tuple(float(value) for value in decision["candidate_scores"])
            scores = tuple(all_scores[action] for action in valid_ids)
            propensity = float(decision.get("propensity") or 1.0)
            policy_kind = "probing_ego_normalized_advantage"
        else:
            scores = tuple(float(q_values[action].item()) for action in valid_ids)
            propensity = 1.0
            policy_kind = f"{self.policy_name}_recurrent_greedy"
        self._action_count += 1
        self._probe_count += int(is_probe)
        return PolicyAction(
            action_id=selected,
            is_probe=bool(is_probe),
            propensity=propensity,
            candidate_action_ids=valid_ids,
            candidate_scores=scores,
            estimated_probe_cost=(
                float(probe_budget_state.cost_per_probe) if is_probe else 0.0
            ),
            policy_kind=policy_kind,
        )

    def observe(self, transition: ActingTransition) -> None:
        self._environment_steps += int(transition.environment_steps)
        self._realized_probe_cost += float(transition.realized_probe_cost)

    def representation(self) -> np.ndarray:
        return self._last_representation.copy()

    def metrics(self) -> Mapping[str, Any]:
        parameter_count = int(
            sum(
                parameter.numel()
                for parameter in self.context.q_net.parameters()
                if parameter.requires_grad
            )
        )
        if int(self.benchmark_resources.get("trainable_parameters", -1)) != parameter_count:
            raise ValueError("Recurrent resource provenance changed trainable parameters.")
        return {
            **self.benchmark_resources,
            "acting_protocol_version": "path_c_acting_policy_v1",
            "probe_count": int(self._probe_count),
            "environment_steps": int(self._environment_steps),
            "realized_probe_cost": float(self._realized_probe_cost),
            "gradient_updates": int(self.spec.gradient_updates),
            "training_environment_steps": int(self.spec.training_environment_steps),
            "evaluation_environment_step_limit": int(
                self.spec.evaluation_environment_step_limit
            ),
            "evaluation_schedule_id": str(self.spec.evaluation_schedule_id),
            "probe_budget_grid_sha256": self.probe_budget_grid_sha256,
            "probe_cost_per_use": float(self._probe_cost_per_use),
            "action_support_sha256": sha256_json(
                list(range(self.context.graph.num_options))
            ),
            "trainable_parameters": parameter_count,
            "oracle_baseline": False,
        }


@dataclass(frozen=True)
class FormalOCV2ActingRuntime:
    factory_registry: FormalActingPolicyFactoryRegistry
    environment_factory: Any
    availability_manifest: Mapping[str, Any]


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    result = evaluate(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(_jsonable(result["summary"]), indent=2, sort_keys=True))


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    start = time.time()
    anchor = _resolve_checkpoint_path(Path(args.checkpoint))
    variants = _parse_csv(args.graph_variants)
    contexts = [
        _load_context(_sibling_checkpoint(anchor, variant), variant)
        for variant in variants
    ]
    # LDS-B3: explicit eval-only ablation switch. The E2 zeroed-channel run is
    # "same checkpoint, zeroed evidence" — before this flag existed it required
    # mutating the checkpoint's config by hand, and the gate admitted the zeroed
    # policy string with no run-level declaration. The flag overlays the mode on
    # the LOADED config (checkpoint file untouched) and the declaration is
    # enforced end-to-end: _validate_eval_integrity now requires the evidence
    # policy to exactly match the mode this run declared.
    zeroed_ablation = bool(getattr(args, "zeroed_partner_option_ablation", False))
    if zeroed_ablation:
        for ctx in contexts:
            _apply_zeroed_override(ctx.config)
    active_probe_collection = bool(
        getattr(args, "path_c_active_probe_collection", False)
        or getattr(args, "active_probe_collection", False)
    )
    acting_probe_budget = getattr(args, "path_c_acting_probe_budget", None)
    requested_probe_cost = getattr(args, "path_c_probe_cost_per_use", None)
    if active_probe_collection and acting_probe_budget is not None:
        raise ValueError(
            "Exploratory probe collection and formal acting-budget evaluation "
            "cannot be combined."
        )
    if requested_probe_cost is not None and acting_probe_budget is None:
        raise ValueError(
            "--path_c_probe_cost_per_use is only an assertion for a formal "
            "--path_c_acting_probe_budget run."
        )
    if acting_probe_budget is not None and bool(
        getattr(args, "random_policy_only", False)
    ):
        raise ValueError(
            "random_policy_only does not implement the registered Path C probe "
            "policy and cannot emit a formal probe-budget point."
        )
    if acting_probe_budget is not None and zeroed_ablation:
        raise ValueError(
            "A declared evidence ablation cannot be emitted as a formal Path C "
            "probe-budget point."
        )
    formal_probe_contracts: list[tuple[str, tuple[int, ...], float, int]] = []
    for ctx in contexts:
        probe = ctx.config.setdefault("path_c", {}).setdefault("probe", {})
        if getattr(args, "path_c_probe_base_checkpoint", None):
            ctx.config["path_c_secondary_base_checkpoint"] = str(
                args.path_c_probe_base_checkpoint
            )
        ctx.active_probe_collection = bool(active_probe_collection)
        ctx.path_c_acting_probe_budget = (
            None if acting_probe_budget is None else int(acting_probe_budget)
        )
        ctx.path_c_acting_probe_remaining = ctx.path_c_acting_probe_budget
        ctx.path_c_probe_cost_per_use = 0.0
        if active_probe_collection:
            frozen_collection_cost = (ctx.config.get("path_c") or {}).get(
                "preregistration", {}
            ).get("probe_cost_per_use")
            if frozen_collection_cost is None:
                raise ValueError(
                    "Path C probe collection requires a frozen per-probe cost."
                )
            ctx.path_c_probe_cost_per_use = float(frozen_collection_cost)
        if ctx.path_c_acting_probe_budget is not None:
            if ctx.path_c_acting_probe_budget < 0:
                raise ValueError("Path C acting probe budget must be non-negative.")
            ensemble = ((ctx.config.get("path_c") or {}).get("ensemble") or {})
            if (
                ctx.method != "aris_bellman"
                or not _uses_sequence_q(ctx.q_net)
                or int(ensemble.get("n_heads", 1)) <= 1
                or not bool(probe.get("enable", False))
                or not bool(probe.get("collection_enable", False))
            ):
                raise ValueError(
                    "Formal probe-budget evaluation requires the active multi-head "
                    "recurrent Path C probing policy."
                )
            if str(probe.get("rule")) != "max_normalized_advantage_disagreement" or str(
                probe.get("collection_selection_mode", "normalized_advantage")
            ) != "normalized_advantage":
                raise ValueError(
                    "Formal probing_ego evaluation requires the frozen "
                    "normalized-advantage selector."
                )
            preregistration_path = (ctx.config.get("path_c") or {}).get(
                "preregistration_path"
            )
            if preregistration_path in {None, ""}:
                raise ValueError(
                    "Formal Path C probe-budget evaluation requires a frozen preregistration."
                )
            frozen = load_frozen_preregistration(preregistration_path)
            budget_grid = tuple(
                int(value)
                for value in frozen.primary_endpoint["probe_budget_grid"]
            )
            if ctx.path_c_acting_probe_budget not in budget_grid:
                raise ValueError(
                    "The requested acting probe budget is outside the frozen grid."
                )
            frozen_cost = float(frozen.primary_endpoint["probe_cost_per_use"])
            if requested_probe_cost is not None and not np.isclose(
                float(requested_probe_cost), frozen_cost, rtol=0.0, atol=0.0
            ):
                raise ValueError("Probe cost differs from the frozen primary endpoint.")
            ctx.path_c_probe_cost_per_use = frozen_cost
            formal_probe_contracts.append((
                frozen.sha256,
                budget_grid,
                frozen_cost,
                int(frozen.payload["evidence_spec"]["max_episode_decisions"]),
            ))
        ctx.path_c_probe_base_q_net = None
        if active_probe_collection:
            ctx.path_c_probe_base_q_net = _load_path_c_probe_baseline_q_net(
                ctx.config,
                ctx.obs_dim,
                ctx.graph,
                torch.device("cpu"),
            )
    if formal_probe_contracts and len(set(formal_probe_contracts)) != 1:
        raise ValueError(
            "All graph variants in one formal probe-budget run must share the "
            "same preregistration, budget grid, probe cost, and episode limit."
        )
    # sec18.14 (formal blind round): eval-only partner-registry override. This is
    # deliberately NOT a config mutation — the graph-objective gate compares
    # config.training.partner_set against the graph metadata (train_aris.
    # _graph_objective_metadata_status), so the override is threaded only into
    # the partner LOOKUP sites below and declared in the output payload. Partner
    # names are globally unique across registries, so baseline caches cannot
    # collide across sets.
    partner_set_override = str(getattr(args, "partner_set", "") or "") or None
    if partner_set_override:
        for ctx in contexts:
            ctx.partner_set_override = partner_set_override
    else:
        # codex review blocker: close the OLD escape path too — a checkpoint
        # whose config was hand-mutated to a non-formal evidence mode must not
        # run as if it were a declared ablation (or as a formal run). Without
        # the flag, only the formal inferred mode is acceptable.
        for ctx in contexts:
            _policy = _evidence_policy_for_config(ctx.config)
            if _policy != "behavior_inferred_v1":
                raise RuntimeError(
                    "Checkpoint config carries a non-formal evidence mode "
                    f"({_policy!r}) but this run did not declare an ablation. "
                    "Pass --zeroed_partner_option_ablation to run the declared "
                    "E2 zeroed channel, or restore the formal config."
                )
    partner_names = _resolve_partner_names(
        contexts[0].option_lib,
        args.partners,
        partner_set=partner_set_override
        or str(contexts[0].config.get("training", {}).get("partner_set", "standard7")),
    )
    reward_scale_status = {
        ctx.graph_variant: _graph_objective_metadata_status(
            ctx.graph,
            ctx.config,
            layout_graph=ctx.layout_graph,
            option_lib=ctx.option_lib,
            partners=_select_train_partners(ctx.option_lib, ctx.config),
        )
        for ctx in contexts
    }
    # LDS-B2 (latent-defect sweep): reward-scale verification is a HARD gate for
    # formal eval — a graph/config objective mismatch must fail the run, not just
    # be recorded in the output JSON for someone to notice later.
    _enforce_reward_scale(
        reward_scale_status,
        allow_unverified=bool(getattr(args, "allow_unverified_reward_scale", False)),
    )
    seed = int(args.seed)
    if int(args.episodes) <= 0:
        raise ValueError("Evaluation episodes must be positive.")
    requested_max_episode_options = getattr(args, "max_episode_options", None)
    max_episode_options = int(
        contexts[0].config["training"].get("max_episode_options", 20)
        if requested_max_episode_options is None
        else requested_max_episode_options
    )
    if max_episode_options <= 0:
        raise ValueError("max_episode_options must be positive.")
    if (
        formal_probe_contracts
        and max_episode_options > int(formal_probe_contracts[0][3])
    ):
        raise ValueError(
            "max_episode_options exceeds the frozen EgoEvidenceSpecV1 episode limit."
        )

    fast = bool(getattr(args, "fast", False))
    allow_diag_skip = bool(getattr(args, "allow_diag_skip", False))
    random_policy_only = bool(getattr(args, "random_policy_only", False))
    factor_deletion_episodes = _factor_deletion_episode_count(args, fast)
    results = []
    for ctx in contexts:
        for partner_name in partner_names:
            aggregate, episodes = _evaluate_partner(
                ctx,
                partner_name,
                episodes=int(args.episodes),
                seed=seed,
                max_episode_options=max_episode_options,
                graph_override=ctx.graph,
                random_policy=random_policy_only,
                collect_diagnostics=not fast,
                allow_diag_skip=allow_diag_skip,
            )
            # P5: formal evaluation metadata must not inspect true partner
            # terminal_policy/role/protocol and derive role-match returns. Partner
            # names select the requested evaluation partner only; they are not routed
            # into decision evidence or return conditioning.
            aggregate["partner_protocol"] = "not_recorded_on_formal_main_path"
            q_proxy_factor_mask = [] if fast else _factor_deletion_q_proxy_diagnostics(ctx)
            result = {
                "method": "random_policy" if random_policy_only else ctx.method,
                "graph_variant": ctx.graph_variant,
                "partner": partner_name,
                "checkpoint": str(ctx.checkpoint_path),
                "primary_return_metric": "mean_net_return",
                "primary_return": float(aggregate["mean_net_return"]),
                "reward_scale_verified": bool(
                    reward_scale_status[ctx.graph_variant]["reward_scale_verified"]
                ),
                "event_semantics_version": reward_scale_status[ctx.graph_variant][
                    "event_semantics_version"
                ],
                "aggregate": aggregate,
                "episodes": episodes,
                "q_proxy_factor_mask": q_proxy_factor_mask,
                "rollout_factor_mask": [],
                "factor_deletion_q_proxy": q_proxy_factor_mask,
            }
            if factor_deletion_episodes > 0 and not random_policy_only:
                result["rollout_factor_mask"] = _factor_deletion_rollout_diagnostics(
                    ctx,
                    partner_name,
                    float(aggregate["mean_net_return"]),
                    episodes=factor_deletion_episodes,
                    seed=seed + 400_000,
                    max_episode_options=max_episode_options,
                    allow_diag_skip=allow_diag_skip,
                )
                result["factor_deletion_return_drop"] = result["rollout_factor_mask"]
            results.append(result)

    baseline_cache_dir = _resolve_baseline_cache_dir(args)
    baselines = _random_baselines(
        contexts[0],
        partner_names,
        episodes=int(args.episodes),
        seed=seed + 200_000,
        max_episode_options=max_episode_options,
        cache_dir=baseline_cache_dir,
    )
    external_references = None
    if args.reference_base_checkpoint or args.reference_ref_checkpoint:
        if not args.reference_base_checkpoint or not args.reference_ref_checkpoint:
            raise ValueError(
                "reference_gap_closure requires both --reference_base_checkpoint and "
                "--reference_ref_checkpoint."
            )
        external_references = _external_reference_baselines(
            args,
            partner_names,
            episodes=int(args.episodes),
            seed=seed + 600_000,
            max_episode_options=max_episode_options,
            cache_dir=baseline_cache_dir,
        )
        _attach_external_reference_gaps(results, external_references)
    else:
        _attach_within_run_relative_returns(results, baselines)

    evaluation_kind = (
        "data_collection_only" if active_probe_collection else "formal_benchmark"
    )
    summary = (
        _collection_only_summary(results, baselines, time.time() - start)
        if active_probe_collection
        else _summary(
            results,
            baselines,
            time.time() - start,
            evaluation_kind=evaluation_kind,
        )
    )
    return {
        "schema_version": "ocv2_eval_v1",
        "evaluation_kind": evaluation_kind,
        "collection_role": evaluation_kind,
        "benchmark_return_eligible": not active_probe_collection,
        "active_probe_collection": active_probe_collection,
        "path_c_probe_budget_evaluation": acting_probe_budget is not None,
        "anchor_checkpoint": str(anchor),
        "graph_variants": variants,
        "partners": partner_names,
        "episodes_per_partner": int(args.episodes),
        "max_episode_options": max_episode_options,
        "diagnostic_granularity": "option",
        "allow_diag_skip": allow_diag_skip,
        "factor_deletion_episodes": factor_deletion_episodes,
        "episode_return_kind": "reward_sum_minus_cost_coef_realized_cost",
        "path_c_primary_return_kind": "episode_return_minus_realized_probe_cost",
        "path_c_acting_probe_budget": (
            None if acting_probe_budget is None else int(acting_probe_budget)
        ),
        "path_c_probe_cost_per_use": (
            None if not formal_probe_contracts else float(formal_probe_contracts[0][2])
        ),
        "path_c_probe_budget_grid": (
            None if not formal_probe_contracts else list(formal_probe_contracts[0][1])
        ),
        "path_c_probe_budget_grid_sha256": (
            None
            if not formal_probe_contracts
            else sha256_json(list(formal_probe_contracts[0][1]))
        ),
        "path_c_preregistration_sha256": (
            None if not formal_probe_contracts else formal_probe_contracts[0][0]
        ),
        # LDS-B1: eval returns exclude training-only shaping so shaped (E1-rev)
        # and unshaped (E1) checkpoints report on the same scale.
        "eval_return_excludes": ["terminal_progress_shaping"],
        # LDS-B3: run-level ablation declaration (sec18.13.3). The integrity gate
        # requires the evidence policy to match this declaration exactly.
        "eval_ablation": {"zeroed_partner_option": zeroed_ablation},
        "eval_partner_set_override": partner_set_override,
        "reward_scale_verified": all(
            bool(item["reward_scale_verified"])
            for item in reward_scale_status.values()
        ),
        "graph_event_semantics_versions": {
            variant: status["event_semantics_version"]
            for variant, status in reward_scale_status.items()
        },
        "graph_reward_scale_status": reward_scale_status,
        "graph_provenance": {
            ctx.graph_variant: (ctx.graph.metadata or {}).get("provenance", {})
            for ctx in contexts
        },
        "path_c": {
            ctx.graph_variant: {
                **path_c_metadata(ctx.config),
                "active_probe_collection_eval": bool(ctx.active_probe_collection),
                "acting_probe_budget": ctx.path_c_acting_probe_budget,
                "probe_cost_per_use": (
                    float(ctx.path_c_probe_cost_per_use)
                    if ctx.path_c_acting_probe_budget is not None
                    else None
                ),
            }
            for ctx in contexts
        },
        "eval_provenance": {
            ctx.graph_variant: _eval_provenance(ctx)
            for ctx in contexts
        },
        "reference_gap_semantics": _reference_semantics(external_references is not None),
        "results": results,
        "reference_baselines": baselines,
        "external_reference_baselines": external_references,
        "summary": summary,
    }


def _load_context(checkpoint_path: Path, variant: str) -> EvalContext:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint for graph variant {variant}: {checkpoint_path}"
        )
    checkpoint = _torch_load(checkpoint_path)
    config = checkpoint["config"]
    normalize_path_c_config(config)
    graph = GraphSpec.from_json_dict(checkpoint["graph"])
    method = str(checkpoint["method"])

    env = _build_env(graph.layout_name, config)
    layout_graph = parse_layout(env, graph.layout_name)
    env.set_featurizer(NumpyFeaturizer(layout_graph))
    obs, _ = env.reset(0)
    option_lib = _build_option_lib(layout_graph, config)
    obs_dim = infer_obs_dim(env, obs)
    q_net = _build_q_network(method, obs_dim, graph, config).to(torch.device("cpu"))
    if _uses_sequence_q(q_net):
        raw_spec = checkpoint.get("ego_evidence_spec")
        if raw_spec is None:
            raise ValueError(
                "Recurrent sequence checkpoints must embed EgoEvidenceSpecV1; "
                "this checkpoint predates the frozen evidence contract."
            )
        checkpoint_spec = EgoEvidenceSpecV1.from_dict(raw_spec)
        runtime_spec = _sequence_spec(q_net)
        if checkpoint_spec.to_dict() != runtime_spec.to_dict():
            raise ValueError(
                "Checkpoint and runtime EgoEvidenceSpecV1 definitions differ."
            )
        bound_sha256 = (checkpoint.get("path_c_artifact_binding") or {}).get(
            "ego_evidence_spec_sha256"
        )
        if bound_sha256 != runtime_spec.sha256():
            raise ValueError("Checkpoint evidence-spec hash binding is missing or invalid.")
    q_net.load_state_dict(checkpoint["q_net"])
    q_net.eval()
    belief_model = _build_belief_model(graph, config).to(torch.device("cpu"))
    belief_model.load_state_dict(checkpoint["belief_model"])
    belief_model.eval()
    return EvalContext(
        checkpoint_path=checkpoint_path,
        config=config,
        graph=graph,
        method=method,
        graph_variant=variant,
        seed_name=checkpoint_path.parent.name,
        q_net=q_net,
        belief_model=belief_model,
        layout_graph=layout_graph,
        option_lib=option_lib,
        obs_dim=obs_dim,
    )


def build_formal_ocv2_acting_runtime(
    payload: Mapping[str, Any],
    *,
    base_directory: str | Path = ".",
) -> FormalOCV2ActingRuntime:
    """Build all nine explicit factories or retain a reason for every missing one."""

    expected = {
        "schema_version",
        "anchor_checkpoint",
        "anchor_checkpoint_sha256",
        "anchor_graph_variant",
        "partner_name_by_split_group_id",
        "layout_name_by_layout_group",
        "probe_action_ids",
        "max_episode_options",
        "response_token_vocabulary",
        "benchmark_contract",
        "policies",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected:
        observed = set(payload) if isinstance(payload, Mapping) else set()
        raise ValueError(
            "Formal acting runtime key mismatch; "
            f"missing={sorted(expected - observed)}, unknown={sorted(observed - expected)}."
        )
    if payload["schema_version"] != "path_c_ocv2_acting_factory_config_v1":
        raise ValueError("Formal OCV2 acting factory config version changed.")
    root = Path(base_directory)
    anchor_path = _resolve_factory_path(payload["anchor_checkpoint"], root)
    anchor_sha256 = _require_sha256_text(
        payload["anchor_checkpoint_sha256"],
        "anchor_checkpoint_sha256",
    )
    if sha256_file(anchor_path) != anchor_sha256:
        raise ValueError("Formal acting anchor checkpoint changed content.")
    if not str(payload["anchor_graph_variant"]).strip():
        raise ValueError("anchor_graph_variant must be non-empty.")
    anchor = _load_context(anchor_path, str(payload["anchor_graph_variant"]))
    if not _uses_sequence_q(anchor.q_net):
        raise ValueError("Formal acting anchor must be a recurrent sequence checkpoint.")
    contract = FormalActingBenchmarkContract.from_mapping(
        payload["benchmark_contract"]
    )
    grid = contract.probe_budget_grid
    probe_cost = float(contract.probe_cost_per_use)
    grid_sha256 = sha256_json(list(grid))
    expected_action_support = sha256_json(list(range(anchor.graph.num_options)))
    if contract.action_support_sha256 != expected_action_support:
        raise ValueError("Formal acting action support differs from the anchor graph.")
    frozen_response_payload = (
        (anchor.config.get("path_c") or {})
        .get("response_summary_spec", {})
        .get("frozen_spec")
    )
    if not isinstance(frozen_response_payload, Mapping):
        raise ValueError(
            "Formal acting anchor lacks a frozen response-summary specification."
        )
    response_summary_spec = ResponseSummarySpecV1.from_mapping(
        {
            key: frozen_response_payload[key]
            for key in (
                "schema_version",
                "response_classes",
                "latency_bin_upper_bounds",
                "structured_multilabel_role",
            )
        }
    )
    response_vocabulary = tuple(map(str, payload["response_token_vocabulary"]))
    if response_vocabulary != response_summary_spec.vocabulary:
        raise ValueError(
            "Formal acting response vocabulary differs from the frozen "
            "ResponseSummarySpecV1 vocabulary."
        )
    required_online_response_classes = {
        "no_response",
        "wait",
        "block",
        "progress",
        "delivery_success",
        "delivery_failure",
    }
    missing_online_classes = sorted(
        required_online_response_classes.difference(
            response_summary_spec.response_classes
        )
    )
    if missing_online_classes:
        raise ValueError(
            "Formal acting response summary lacks online class(es): "
            + ", ".join(missing_online_classes)
        )
    raw_probe_action_ids = payload["probe_action_ids"]
    if isinstance(raw_probe_action_ids, (str, bytes)) or not isinstance(
        raw_probe_action_ids, Sequence
    ):
        raise TypeError("Formal acting probe_action_ids must be a sequence.")
    probe_action_ids = tuple(raw_probe_action_ids)
    if not probe_action_ids or any(
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or int(value) < 0
        or int(value) >= int(anchor.graph.num_options)
        for value in probe_action_ids
    ):
        raise ValueError(
            "Formal acting probe_action_ids must be a non-empty subset of action support."
        )
    if len(set(map(int, probe_action_ids))) != len(probe_action_ids):
        raise ValueError("Formal acting probe_action_ids must be unique.")
    if set(map(int, probe_action_ids)) == set(range(int(anchor.graph.num_options))):
        raise ValueError(
            "Formal acting probe_action_ids must be a strict subset of action support."
        )
    anchor_response_hash = (
        (anchor.config.get("path_c") or {})
        .get("preregistration", {})
        .get("response_vocabulary_sha256")
    )
    if anchor_response_hash != response_summary_spec.sha256:
        raise ValueError(
            "Formal acting anchor changed the frozen response-vocabulary hash."
        )
    partner_map = payload["partner_name_by_split_group_id"]
    if (
        not isinstance(partner_map, Mapping)
        or not partner_map
        or any(
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(value, str)
            or not value.strip()
            for key, value in partner_map.items()
        )
    ):
        raise ValueError("Formal acting partner map must be a non-empty mapping.")
    layout_map = payload["layout_name_by_layout_group"]
    if (
        not isinstance(layout_map, Mapping)
        or not layout_map
        or any(
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(value, str)
            or not value.strip()
            for key, value in layout_map.items()
        )
    ):
        raise ValueError("Formal acting layout map must be a non-empty mapping.")
    max_episode_options = payload["max_episode_options"]
    if isinstance(max_episode_options, bool) or not isinstance(max_episode_options, int):
        raise TypeError("Formal acting max_episode_options must be an integer.")
    if int(max_episode_options) <= 0:
        raise ValueError("Formal acting max_episode_options must be positive.")
    configured_max_episode_options = int(
        (anchor.config.get("training") or {}).get("max_episode_options", 0)
    )
    if int(max_episode_options) != configured_max_episode_options:
        raise ValueError(
            "Formal acting max_episode_options differs from the anchor checkpoint."
        )
    policy_configs = payload["policies"]
    if not isinstance(policy_configs, Mapping):
        raise TypeError("Formal acting policies must be a mapping.")
    unknown_policies = sorted(set(policy_configs).difference(FORMAL_ACTING_POLICY_NAMES))
    if unknown_policies:
        raise ValueError(f"Formal acting config has unknown policies: {unknown_policies}.")
    anchor_contract = _formal_acting_context_contract(anchor)
    entries: dict[str, FormalActingPolicyFactoryEntry] = {}

    for policy_name in ("probing_ego", "full_history_rnn", "rnn_residualized"):
        config = policy_configs.get(policy_name)
        if not isinstance(config, Mapping):
            entries[policy_name] = _unavailable_formal_factory(
                policy_name,
                "recurrent factory configuration is missing",
            )
            continue
        try:
            recurrent_keys = {
                "factory_kind",
                "checkpoint",
                "checkpoint_sha256",
                "graph_variant",
                "architecture_manifest",
                "training_provenance",
            }
            if policy_name == "rnn_residualized":
                recurrent_keys.add("residualization_proof")
            _require_exact_factory_keys(
                config,
                recurrent_keys,
                f"policies.{policy_name}",
            )
            if config.get("factory_kind") != "recurrent_eval_context_v1":
                raise ValueError("factory_kind must be recurrent_eval_context_v1")
            checkpoint_path = _resolve_factory_path(config.get("checkpoint"), root)
            checkpoint_sha256 = _require_sha256_text(
                config.get("checkpoint_sha256"),
                f"policies.{policy_name}.checkpoint_sha256",
            )
            if sha256_file(checkpoint_path) != checkpoint_sha256:
                raise ValueError("checkpoint content does not match checkpoint_sha256")
            training_provenance, training_provenance_sha256 = (
                _load_bound_json_artifact(
                    config["training_provenance"],
                    root,
                    f"policies.{policy_name}.training_provenance",
                )
            )
            _validate_recurrent_training_provenance(
                training_provenance,
                checkpoint_sha256=checkpoint_sha256,
                benchmark_contract=contract,
            )
            architecture_manifest, architecture_manifest_sha256 = (
                _load_bound_json_artifact(
                    config["architecture_manifest"],
                    root,
                    f"policies.{policy_name}.architecture_manifest",
                )
            )
            context = _load_context(
                checkpoint_path,
                str(config.get("graph_variant")),
            )
            if not _uses_sequence_q(context.q_net):
                raise ValueError("checkpoint does not expose recurrent sequence Q")
            if _formal_acting_context_contract(context) != anchor_contract:
                raise ValueError("checkpoint changes the acting environment/evidence contract")
            recurrent_resources = _validated_training_resources(training_provenance)
            actual_parameter_count = int(
                sum(
                    parameter.numel()
                    for parameter in context.q_net.parameters()
                    if parameter.requires_grad
                )
            )
            if recurrent_resources["trainable_parameters"] != actual_parameter_count:
                raise ValueError(
                    "Recurrent training provenance changed trainable parameters."
                )
            _validate_recurrent_architecture_manifest(
                architecture_manifest,
                policy_name=policy_name,
                context=context,
                root=root,
                checkpoint_sha256=checkpoint_sha256,
                training_provenance_sha256=training_provenance_sha256,
            )
            residualization_proof_sha256 = None
            if policy_name == "rnn_residualized":
                residualization_proof, residualization_proof_sha256 = (
                    _load_bound_json_artifact(
                        config["residualization_proof"],
                        root,
                        "policies.rnn_residualized.residualization_proof",
                    )
                )
                _validate_residualization_proof(
                    residualization_proof,
                    checkpoint_sha256=checkpoint_sha256,
                    training_provenance_sha256=training_provenance_sha256,
                    split_manifest_sha256=contract.split_manifest_sha256,
                )
            if policy_name == "probing_ego":
                probe = (context.config.get("path_c") or {}).get("probe") or {}
                ensemble = (context.config.get("path_c") or {}).get("ensemble") or {}
                if not (
                    context.method == "aris_bellman"
                    and int(ensemble.get("n_heads", 1)) > 1
                    and bool(probe.get("enable", False))
                    and bool(probe.get("collection_enable", False))
                    and str(probe.get("rule"))
                    == "max_normalized_advantage_disagreement"
                ):
                    raise ValueError("checkpoint is not the registered probing_ego")
            elif context.method != "global_gru":
                raise ValueError(
                    f"{policy_name} must use the shared global_gru sequence core"
                )

            def recurrent_factory(
                spec: ActingEpisodeSpec,
                environment: PathCActingEnvironment,
                *,
                _context: EvalContext = context,
                _name: str = policy_name,
                _resources: Mapping[str, Any] = recurrent_resources,
            ) -> RecurrentContextActingPolicy:
                contract.validate_episode_spec(spec)
                if not isinstance(environment, OCV2FormalActingEnvironment):
                    raise TypeError("Recurrent OCV2 policy requires OCV2FormalActingEnvironment.")
                return RecurrentContextActingPolicy(
                    policy_name=_name,
                    context=_context,
                    environment=environment,
                    spec=spec,
                    probing=_name == "probing_ego",
                    probe_budget_grid_sha256=grid_sha256,
                    benchmark_resources=_resources,
                )

            entries[policy_name] = FormalActingPolicyFactoryEntry(
                policy_name=policy_name,
                factory=recurrent_factory,
                factory_id="recurrent_eval_context_v1:" + sha256_json(
                    {
                        "policy_name": policy_name,
                        "checkpoint_sha256": checkpoint_sha256,
                        "architecture_manifest_sha256": architecture_manifest_sha256,
                        "training_provenance_sha256": training_provenance_sha256,
                        "residualization_proof_sha256": residualization_proof_sha256,
                    }
                ),
                allowed_split_roles=("design", "locked_audit"),
            )
        except Exception as error:
            entries[policy_name] = _unavailable_formal_factory(
                policy_name,
                f"{type(error).__name__}: {error}",
            )

    for policy_name in (
        "exact_belief_filter",
        "learned_hmm_filter",
        "particle_belief_filter",
    ):
        config = policy_configs.get(policy_name)
        if not isinstance(config, Mapping):
            entries[policy_name] = _unavailable_formal_factory(
                policy_name,
                "belief-filter factory configuration is missing",
            )
            continue
        try:
            factory = _build_belief_acting_factory(
                policy_name,
                config,
                anchor,
                root=root,
                benchmark_contract=contract,
                grid_sha256=grid_sha256,
                probe_cost=probe_cost,
            )
            training_artifact_sha256 = _artifact_reference_sha256(
                config.get("training_artifact"),
                f"policies.{policy_name}.training_artifact",
            )
            entries[policy_name] = FormalActingPolicyFactoryEntry(
                policy_name=policy_name,
                factory=factory,
                factory_id=(
                    f"{config.get('factory_kind')}:"
                    + sha256_json(
                        {
                            "config": dict(config),
                            "training_artifact_sha256": training_artifact_sha256,
                        }
                    )
                ),
                allowed_split_roles=("design", "locked_audit"),
            )
        except Exception as error:
            entries[policy_name] = _unavailable_formal_factory(
                policy_name,
                f"{type(error).__name__}: {error}",
            )

    for policy_name, wrapper_type in (
        ("random_probe", RandomProbeActingPolicy),
        ("no_probe", NoProbeActingPolicy),
    ):
        config = policy_configs.get(policy_name)
        expected_kind = f"{policy_name}_wrapper_v1"
        base_name = str(config.get("base_policy", "")) if isinstance(config, Mapping) else ""
        base_entry = entries.get(base_name)
        if (
            not isinstance(config, Mapping)
            or config.get("factory_kind") != expected_kind
            or base_entry is None
            or not base_entry.available
        ):
            if not isinstance(config, Mapping):
                reason = "wrapper configuration is missing"
            elif config.get("factory_kind") != expected_kind:
                reason = f"factory_kind must be {expected_kind}"
            elif base_entry is None:
                reason = f"base policy {base_name!r} is not a concrete factory"
            else:
                reason = str(base_entry.unavailable_reason)
            entries[policy_name] = _unavailable_formal_factory(policy_name, reason)
            continue
        try:
            _require_exact_factory_keys(
                config,
                {"factory_kind", "base_policy"},
                f"policies.{policy_name}",
            )
        except (TypeError, ValueError) as error:
            entries[policy_name] = _unavailable_formal_factory(
                policy_name,
                f"{type(error).__name__}: {error}",
            )
            continue

        def wrapper_factory(
            spec: ActingEpisodeSpec,
            environment: PathCActingEnvironment,
            *,
            _base: FormalActingPolicyFactoryEntry = base_entry,
            _wrapper: Any = wrapper_type,
        ) -> Any:
            return _wrapper(exploitation_policy=_base.build(spec, environment))

        entries[policy_name] = FormalActingPolicyFactoryEntry(
            policy_name=policy_name,
            factory=wrapper_factory,
            factory_id=f"{expected_kind}:" + sha256_json(
                {
                    "base_policy": base_name,
                    "base_factory_id": base_entry.factory_id,
                }
            ),
            allowed_split_roles=("design", "locked_audit"),
        )

    direct_config = policy_configs.get("direct_information")
    direct_base_name = (
        str(direct_config.get("base_policy", ""))
        if isinstance(direct_config, Mapping)
        else ""
    )
    direct_base = entries.get(direct_base_name)
    try:
        if not isinstance(direct_config, Mapping):
            raise ValueError("direct-information config is missing")
        _require_exact_factory_keys(
            direct_config,
            {"factory_kind", "base_policy", "response_probabilities"},
            "policies.direct_information",
        )
        if direct_config.get("factory_kind") != "direct_information_design_v1":
            raise ValueError("direct-information factory_kind is invalid")
        if direct_base is None or not direct_base.available:
            raise ValueError(
                "direct-information exploitation base is unavailable: "
                f"{None if direct_base is None else direct_base.unavailable_reason}"
            )
        response_probabilities = np.asarray(
            direct_config.get("response_probabilities"),
            dtype=np.float64,
        )
        if response_probabilities.shape[1:] != (
            anchor.graph.num_options,
            len(response_vocabulary),
        ):
            raise ValueError(
                "direct-information response table must match action support and "
                "response vocabulary"
            )
        DirectInformationActingPolicy(
            response_probabilities=response_probabilities,
            exploitation_policy=_FactoryValidationPolicy(),
        )

        def direct_factory(
            spec: ActingEpisodeSpec,
            environment: PathCActingEnvironment,
            *,
            _base: FormalActingPolicyFactoryEntry = direct_base,
            _probabilities: np.ndarray = response_probabilities.copy(),
        ) -> DirectInformationActingPolicy:
            return DirectInformationActingPolicy(
                response_probabilities=_probabilities,
                exploitation_policy=_base.build(spec, environment),
            )

        entries["direct_information"] = FormalActingPolicyFactoryEntry(
            policy_name="direct_information",
            factory=direct_factory,
            factory_id="direct_information_design_v1:" + sha256_json(
                {
                    "config": dict(direct_config),
                    "base_factory_id": direct_base.factory_id,
                }
            ),
            allowed_split_roles=("design",),
        )
    except Exception as error:
        entries["direct_information"] = _unavailable_formal_factory(
            "direct_information",
            f"{type(error).__name__}: {error}",
        )

    registry = FormalActingPolicyFactoryRegistry(
        entries=tuple(entries[name] for name in FORMAL_ACTING_POLICY_NAMES),
        benchmark_contract=contract,
    )

    def environment_factory(spec: ActingEpisodeSpec) -> OCV2FormalActingEnvironment:
        contract.validate_episode_spec(spec)
        partner_name = partner_map.get(spec.split_group_id)
        if partner_name is None:
            raise KeyError(
                f"No formal acting partner is registered for group {spec.split_group_id!r}."
            )
        configured_limit = int((anchor.config.get("env") or {}).get("max_steps", 200))
        if configured_limit != int(contract.evaluation_environment_step_limit):
            raise ValueError(
                "Benchmark evaluation limit differs from the anchor OCV2 environment."
            )
        layout_name = layout_map.get(spec.layout_group)
        if layout_name is None:
            raise KeyError(
                f"No formal acting layout is registered for {spec.layout_group!r}."
            )
        if str(layout_name) != str(anchor.graph.layout_name):
            raise ValueError("Acting episode layout group maps to a different anchor layout.")
        return OCV2FormalActingEnvironment(
            context=anchor,
            spec=spec,
            partner_name=str(partner_name),
            max_episode_options=int(max_episode_options),
            response_summary_spec=response_summary_spec,
            probe_action_ids=probe_action_ids,
        )

    registry_manifest = registry.to_manifest()
    environment_manifest = {
        "schema_version": "path_c_ocv2_acting_environment_manifest_v1",
        "anchor_checkpoint_sha256": anchor_sha256,
        "anchor_context_contract_sha256": anchor_contract,
        "partner_name_by_split_group_id_sha256": sha256_json(dict(partner_map)),
        "layout_name_by_layout_group_sha256": sha256_json(dict(layout_map)),
        "max_episode_options": int(max_episode_options),
        "response_summary_spec_sha256": response_summary_spec.sha256,
        "response_token_vocabulary_sha256": sha256_json(list(response_vocabulary)),
        "probe_action_ids_sha256": sha256_json(list(map(int, probe_action_ids))),
    }
    availability_manifest = {
        "schema_version": "path_c_ocv2_formal_acting_runtime_manifest_v1",
        "factory_registry": registry_manifest,
        "environment": environment_manifest,
    }
    availability_manifest["sha256"] = sha256_json(availability_manifest)
    return FormalOCV2ActingRuntime(
        factory_registry=registry,
        environment_factory=environment_factory,
        availability_manifest=availability_manifest,
    )


def _resolve_factory_path(value: Any, root: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Formal acting checkpoint path must be non-empty.")
    path = Path(value)
    return path if path.is_absolute() else root / path


def _require_exact_factory_keys(
    payload: Mapping[str, Any],
    expected: set[str],
    name: str,
) -> None:
    observed = set(payload)
    missing = sorted(expected.difference(observed))
    unknown = sorted(observed.difference(expected))
    if missing or unknown:
        raise ValueError(
            f"{name} key mismatch; missing={missing}, unknown={unknown}."
        )


def _require_sha256_text(value: Any, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return text


def _artifact_reference_sha256(reference: Any, name: str) -> str:
    if not isinstance(reference, Mapping):
        raise TypeError(f"{name} must be a path/SHA-256 mapping.")
    _require_exact_factory_keys(reference, {"path", "sha256"}, name)
    return _require_sha256_text(reference["sha256"], f"{name}.sha256")


def _load_bound_json_artifact(
    reference: Any,
    root: Path,
    name: str,
) -> tuple[dict[str, Any], str]:
    expected_sha256 = _artifact_reference_sha256(reference, name)
    path = _resolve_factory_path(reference["path"], root)
    observed_sha256 = sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise ValueError(f"{name} content does not match its SHA-256 digest.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{name} JSON root must be a mapping.")
    return payload, observed_sha256


def _validate_bound_file_reference(reference: Any, root: Path, name: str) -> str:
    expected_sha256 = _artifact_reference_sha256(reference, name)
    path = _resolve_factory_path(reference["path"], root)
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{name} content does not match its SHA-256 digest.")
    return expected_sha256


def _validate_recurrent_training_provenance(
    payload: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    benchmark_contract: FormalActingBenchmarkContract,
) -> None:
    _require_exact_factory_keys(
        payload,
        {
            "schema_version",
            "collection_role",
            "split_manifest_sha256",
            "checkpoint_sha256",
            "effective_episodes",
            "effective_transitions",
            "training_environment_steps",
            "gradient_updates",
            "trainable_parameters",
            "training_flops",
            "wall_clock_seconds",
            "inference_latency_ms",
        },
        "recurrent training provenance",
    )
    if payload["schema_version"] != "path_c_recurrent_training_provenance_v1":
        raise ValueError("Recurrent training provenance version changed.")
    if payload["collection_role"] != "train":
        raise ValueError("Recurrent checkpoint provenance must use collection_role=train.")
    if payload["split_manifest_sha256"] != benchmark_contract.split_manifest_sha256:
        raise ValueError("Recurrent checkpoint provenance changed the split manifest.")
    if payload["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError("Recurrent training provenance names a different checkpoint.")
    _validate_training_budget_evidence(payload, benchmark_contract)
    _validated_training_resources(payload)


def _validate_training_budget_evidence(
    payload: Mapping[str, Any],
    benchmark_contract: FormalActingBenchmarkContract,
) -> None:
    for name in (
        "effective_episodes",
        "effective_transitions",
        "training_environment_steps",
        "gradient_updates",
    ):
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"Training provenance {name} must be an integer.")
    if int(payload["effective_episodes"]) < int(
        benchmark_contract.effective_episode_floor
    ):
        raise ValueError("Training provenance is below the effective-episode floor.")
    if int(payload["effective_transitions"]) < int(
        benchmark_contract.effective_transition_floor
    ):
        raise ValueError("Training provenance is below the effective-transition floor.")
    if int(payload["training_environment_steps"]) != int(
        benchmark_contract.training_environment_steps
    ):
        raise ValueError("Training provenance changed training environment steps.")
    if int(payload["gradient_updates"]) != int(benchmark_contract.gradient_updates):
        raise ValueError("Training provenance changed the gradient-update budget.")


def _validated_training_resources(payload: Mapping[str, Any]) -> dict[str, Any]:
    parameters = payload.get("trainable_parameters")
    if isinstance(parameters, bool) or not isinstance(parameters, int) or parameters < 0:
        raise ValueError("Training provenance trainable_parameters must be non-negative.")
    resources: dict[str, Any] = {"trainable_parameters": int(parameters)}
    for name in ("training_flops", "wall_clock_seconds", "inference_latency_ms"):
        value = float(payload.get(name, float("nan")))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"Training provenance {name} must be finite and non-negative.")
        resources[name] = value
    return resources


def _validate_recurrent_architecture_manifest(
    payload: Mapping[str, Any],
    *,
    policy_name: str,
    context: EvalContext,
    root: Path,
    checkpoint_sha256: str,
    training_provenance_sha256: str,
) -> None:
    _require_exact_factory_keys(
        payload,
        {
            "schema_version",
            "policy_name",
            "architecture_kind",
            "checkpoint_sha256",
            "q_class",
            "sequence_method",
            "ego_evidence_spec_sha256",
            "n_heads",
            "prior_scale",
            "training_provenance_sha256",
            "residualization",
        },
        "recurrent architecture manifest",
    )
    if payload["schema_version"] != "path_c_recurrent_architecture_manifest_v1":
        raise ValueError("Recurrent architecture manifest version changed.")
    if payload["policy_name"] != policy_name:
        raise ValueError("Recurrent architecture manifest names a different policy.")
    expected_kind = {
        "probing_ego": "probing_recurrent_ensemble_q",
        "full_history_rnn": "full_history_recurrent_q",
        "rnn_residualized": "public_state_residualized_recurrent_q",
    }[policy_name]
    expected_method = "aris_bellman" if policy_name == "probing_ego" else "global_gru"
    actual = {
        "architecture_kind": expected_kind,
        "checkpoint_sha256": checkpoint_sha256,
        "q_class": type(context.q_net).__name__,
        "sequence_method": str(getattr(context.q_net, "sequence_method", context.method)),
        "ego_evidence_spec_sha256": _sequence_spec(context.q_net).sha256(),
        "n_heads": int(getattr(context.q_net, "n_heads", 0)),
        "prior_scale": float(getattr(context.q_net, "prior_scale", float("nan"))),
        "training_provenance_sha256": training_provenance_sha256,
    }
    if actual["q_class"] != "RecurrentEnsembleQ":
        raise ValueError("Recurrent architecture does not use RecurrentEnsembleQ.")
    if actual["sequence_method"] != expected_method:
        raise ValueError("Recurrent architecture has the wrong sequence method.")
    if isinstance(payload["n_heads"], bool) or not isinstance(payload["n_heads"], int):
        raise TypeError("Recurrent architecture n_heads must be an integer.")
    if not np.isfinite(float(payload["prior_scale"])):
        raise ValueError("Recurrent architecture prior_scale must be finite.")
    if policy_name != "probing_ego" and (
        actual["n_heads"] != 1 or actual["prior_scale"] != 0.0
    ):
        raise ValueError(
            "Recurrent baseline must use one learned head and no frozen prior."
        )
    for name, observed in actual.items():
        expected = payload[name]
        if name == "prior_scale":
            matches = np.isclose(float(expected), float(observed), rtol=0.0, atol=0.0)
        else:
            matches = expected == observed
        if not matches:
            raise ValueError(f"Recurrent architecture manifest changed {name}.")
    residualization = payload["residualization"]
    if policy_name != "rnn_residualized":
        if residualization is not None:
            raise ValueError("Only rnn_residualized may declare residualization metadata.")
        return
    if not isinstance(residualization, Mapping):
        raise ValueError("rnn_residualized requires explicit residualization provenance.")
    _require_exact_factory_keys(
        residualization,
        {
            "schema_version",
            "implementation_id",
            "training_target",
            "public_baseline_artifact",
            "residual_training_artifact",
        },
        "recurrent architecture residualization",
    )
    if residualization["schema_version"] != "path_c_residualization_provenance_v1":
        raise ValueError("Residualization provenance version changed.")
    if residualization["training_target"] != "public_state_residualized_td":
        raise ValueError("rnn_residualized did not use the registered residual target.")
    if not str(residualization["implementation_id"]).strip():
        raise ValueError("Residualization implementation_id must be non-empty.")
    for name in ("public_baseline_artifact", "residual_training_artifact"):
        _validate_bound_file_reference(
            residualization[name],
            root,
            f"residualization.{name}",
        )


def _formal_acting_context_contract(context: EvalContext) -> str:
    return sha256_json(
        {
            "layout": context.graph.layout_name,
            "environment": dict(context.config.get("env") or {}),
            "reward": reward_config_payload(context.config),
            "option_library_sha256": option_library_hash(context.option_lib),
            "ego_evidence_spec_sha256": _sequence_spec(context.q_net).sha256(),
        }
    )


def _validate_residualization_proof(
    payload: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    training_provenance_sha256: str,
    split_manifest_sha256: str,
) -> None:
    _require_exact_factory_keys(
        payload,
        {
            "schema_version",
            "residual_checkpoint_sha256",
            "training_provenance_sha256",
            "split_manifest_sha256",
            "collection_role",
            "training_relation",
            "public_baseline_artifact_sha256",
            "residual_training_artifact_sha256",
            "verification_report_sha256",
        },
        "rnn residualization proof",
    )
    if payload["schema_version"] != "path_c_residualization_proof_v1":
        raise ValueError("Residualization proof version changed.")
    expected = {
        "residual_checkpoint_sha256": checkpoint_sha256,
        "training_provenance_sha256": training_provenance_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "collection_role": "train",
        "training_relation": "public_state_residualized_td",
    }
    for name, value in expected.items():
        if payload[name] != value:
            raise ValueError(f"Residualization proof changed {name}.")
    for name in (
        "public_baseline_artifact_sha256",
        "residual_training_artifact_sha256",
        "verification_report_sha256",
    ):
        _require_sha256_text(payload[name], f"residualization proof {name}")


def _unavailable_formal_factory(
    policy_name: str,
    reason: str,
) -> FormalActingPolicyFactoryEntry:
    roles = (
        ("design",)
        if policy_name == "direct_information"
        else ("design", "locked_audit")
    )
    return FormalActingPolicyFactoryEntry(
        policy_name=policy_name,
        factory=None,
        factory_id=f"unavailable:{policy_name}",
        allowed_split_roles=roles,
        unavailable_reason=str(reason),
    )


def _belief_benchmark_metadata(
    spec: ActingEpisodeSpec,
    context: EvalContext,
    *,
    benchmark_contract: FormalActingBenchmarkContract,
    benchmark_resources: Mapping[str, Any],
    grid_sha256: str,
    probe_cost: float,
) -> dict[str, Any]:
    benchmark_contract.validate_episode_spec(spec)
    return {
        **dict(benchmark_resources),
        "gradient_updates": int(spec.gradient_updates),
        "training_environment_steps": int(spec.training_environment_steps),
        "evaluation_environment_step_limit": int(
            spec.evaluation_environment_step_limit
        ),
        "evaluation_schedule_id": str(spec.evaluation_schedule_id),
        "probe_budget_grid_sha256": str(grid_sha256),
        "probe_cost_per_use": float(probe_cost),
        "action_support_sha256": sha256_json(
            list(range(context.graph.num_options))
        ),
    }


def _build_belief_acting_factory(
    policy_name: str,
    config: Mapping[str, Any],
    context: EvalContext,
    *,
    root: Path,
    benchmark_contract: FormalActingBenchmarkContract,
    grid_sha256: str,
    probe_cost: float,
) -> Any:
    common_keys = {
        "factory_kind",
        "mode_names",
        "transition_stay_prob",
        "observation_smoothing",
        "observation_key",
        "public_context_key",
        "training_artifact",
    }
    expected_keys = set(common_keys)
    expected_kind = {
        "exact_belief_filter": "exact_belief_filter_v1",
        "learned_hmm_filter": "learned_hmm_filter_v1",
        "particle_belief_filter": "particle_belief_filter_v1",
    }.get(policy_name)
    if expected_kind is None:
        raise ValueError(f"Unknown belief acting policy {policy_name!r}.")
    if policy_name == "exact_belief_filter":
        expected_keys.add("observation_likelihood")
    if policy_name == "particle_belief_filter":
        expected_keys.add("particle_count")
    _require_exact_factory_keys(config, expected_keys, f"policies.{policy_name}")
    if config["factory_kind"] != expected_kind:
        raise ValueError(f"{policy_name} factory_kind is invalid")
    mode_names = tuple(map(str, config.get("mode_names", ())))
    if not mode_names or len(set(mode_names)) != len(mode_names):
        raise ValueError("Belief-filter mode_names must be non-empty and unique.")
    training_artifact, _training_sha256 = _load_bound_json_artifact(
        config["training_artifact"],
        root,
        f"policies.{policy_name}.training_artifact",
    )
    training_data = _validated_belief_training_artifact(
        training_artifact,
        mode_names=mode_names,
        root=root,
        benchmark_contract=benchmark_contract,
        require_observations=policy_name != "exact_belief_filter",
    )
    public_context_weights = np.asarray(
        training_data["public_context_weights"],
        dtype=np.float64,
    )
    benchmark_resources = _validated_training_resources(training_data)
    if (
        public_context_weights.ndim != 2
        or public_context_weights.shape[1] != context.graph.num_options
        or public_context_weights.shape[0] <= len(mode_names) + 1
    ):
        raise ValueError("Belief-filter state-conditioned value head has the wrong shape.")
    transition_stay_prob = float(config.get("transition_stay_prob", 0.98))
    smoothing = float(config.get("observation_smoothing", 1.0e-3))
    observation_key = str(config.get("observation_key", "belief_observation_token"))
    public_context_key = str(config.get("public_context_key", "belief_public_context"))
    if not observation_key.strip() or not public_context_key.strip():
        raise ValueError("Belief-filter observation and public-context keys must be non-empty.")
    if training_data["public_context_key"] != public_context_key:
        raise ValueError("Belief-filter training artifact changed public_context_key.")
    observations = tuple(map(str, training_data["observations"]))
    mode_labels = tuple(map(str, training_data["mode_labels"]))
    hmm_config = HMMFilterConfig(
        mode_names=mode_names,
        transition_stay_prob=transition_stay_prob,
        observation_smoothing=smoothing,
    )
    particle_config = None
    if policy_name == "particle_belief_filter":
        particle_count = config["particle_count"]
        if isinstance(particle_count, bool) or not isinstance(particle_count, int):
            raise TypeError("particle_count must be an integer.")
        particle_config = ParticleFilterConfig(
            mode_names=mode_names,
            particle_count=particle_count,
            transition_stay_prob=transition_stay_prob,
            observation_smoothing=smoothing,
        )

    def build_filter() -> Any:
        if policy_name == "exact_belief_filter":
            if not isinstance(config["observation_likelihood"], Mapping) or not config[
                "observation_likelihood"
            ]:
                raise ValueError(
                    "Exact belief filter requires a non-empty scripted likelihood table."
                )
            return BayesianHMMBeliefFilter(
                hmm_config,
                config["observation_likelihood"],
                inference_kind="scripted_hmm_filter",
            )
        if policy_name == "learned_hmm_filter":
            return BayesianHMMBeliefFilter.fit_from_training_split(
                hmm_config,
                observations,
                mode_labels,
            )
        if particle_config is None:
            raise RuntimeError("Particle-filter configuration was not constructed.")
        return ParticleBeliefFilter.fit_from_training_split(
            particle_config,
            observations,
            mode_labels,
        )

    # Pre-fit once so malformed or non-train data block the registry before rollout.
    build_filter()

    def factory(
        spec: ActingEpisodeSpec,
        environment: PathCActingEnvironment,
    ) -> BeliefFilterActingPolicy:
        del environment
        benchmark_contract.validate_episode_spec(spec)
        belief_filter = build_filter()
        return BeliefFilterActingPolicy(
            policy_name=policy_name,
            belief_filter=belief_filter,
            public_context_weights=public_context_weights,
            public_context_key=public_context_key,
            observation_key=observation_key,
            benchmark_metadata=_belief_benchmark_metadata(
                spec,
                context,
                benchmark_contract=benchmark_contract,
                benchmark_resources=benchmark_resources,
                grid_sha256=grid_sha256,
                probe_cost=probe_cost,
            ),
        )
    return factory


def _validated_belief_training_artifact(
    payload: Mapping[str, Any],
    *,
    mode_names: tuple[str, ...],
    root: Path,
    benchmark_contract: FormalActingBenchmarkContract,
    require_observations: bool,
) -> dict[str, Any]:
    _require_exact_factory_keys(
        payload,
        {
            "schema_version",
            "collection_role",
            "split_manifest_sha256",
            "effective_episodes",
            "effective_transitions",
            "training_environment_steps",
            "gradient_updates",
            "trainable_parameters",
            "training_flops",
            "wall_clock_seconds",
            "inference_latency_ms",
            "mode_names",
            "observations",
            "mode_labels",
            "public_context_key",
            "public_context_dim",
            "public_context_weights",
            "action_value_provenance",
        },
        "belief-filter training artifact",
    )
    if payload["schema_version"] != "path_c_belief_training_artifact_v2":
        raise ValueError("Belief-filter training artifact version changed.")
    if payload["collection_role"] != "train":
        raise ValueError("Belief-filter training artifact must use collection_role=train.")
    if payload["split_manifest_sha256"] != benchmark_contract.split_manifest_sha256:
        raise ValueError("Belief-filter training artifact changed the split manifest.")
    _validate_training_budget_evidence(payload, benchmark_contract)
    _validated_training_resources(payload)
    if tuple(map(str, payload["mode_names"])) != mode_names:
        raise ValueError("Belief-filter training artifact changed mode_names.")
    observations = payload["observations"]
    labels = payload["mode_labels"]
    if isinstance(observations, (str, bytes)) or isinstance(labels, (str, bytes)):
        raise TypeError("Belief-filter observations and labels must be sequences.")
    observations = tuple(observations)
    labels = tuple(labels)
    if len(observations) != len(labels):
        raise ValueError("Belief-filter training observations and labels are unaligned.")
    if require_observations and not observations:
        raise ValueError("Learned belief filters require train-role observations.")
    unknown_labels = sorted(set(map(str, labels)).difference(mode_names))
    if unknown_labels:
        raise ValueError(f"Belief-filter training labels are unknown: {unknown_labels}.")
    public_context_key = str(payload["public_context_key"])
    public_context_dim = payload["public_context_dim"]
    if not public_context_key.strip():
        raise ValueError("Belief-filter public_context_key must be non-empty.")
    if isinstance(public_context_dim, bool) or not isinstance(public_context_dim, int):
        raise TypeError("Belief-filter public_context_dim must be an integer.")
    if int(public_context_dim) <= 0:
        raise ValueError("Belief-filter public_context_dim must be positive.")
    public_context_weights = np.asarray(
        payload["public_context_weights"],
        dtype=np.float64,
    )
    expected_rows = int(public_context_dim) + len(mode_names) + 1
    if (
        public_context_weights.ndim != 2
        or public_context_weights.shape[0] != expected_rows
        or not np.all(np.isfinite(public_context_weights))
    ):
        raise ValueError(
            "Belief-filter public-context value weights have the wrong shape."
        )
    provenance = payload["action_value_provenance"]
    if not isinstance(provenance, Mapping):
        raise TypeError("action_value_provenance must be a mapping.")
    _require_exact_factory_keys(
        provenance,
        {
            "schema_version",
            "collection_role",
            "split_manifest_sha256",
            "source_artifact",
            "estimator_id",
        },
        "action-value provenance",
    )
    if provenance["schema_version"] != "path_c_action_value_provenance_v1":
        raise ValueError("Action-value provenance version changed.")
    if provenance["collection_role"] != "train":
        raise ValueError("Action values must be fitted only from the train role.")
    if provenance["split_manifest_sha256"] != benchmark_contract.split_manifest_sha256:
        raise ValueError("Action-value provenance changed the split manifest.")
    _validate_bound_file_reference(
        provenance["source_artifact"],
        root,
        "action_value_provenance.source_artifact",
    )
    if not str(provenance["estimator_id"]).strip():
        raise ValueError("Action-value estimator_id must be non-empty.")
    return {
        **dict(payload),
        "observations": observations,
        "mode_labels": labels,
        "public_context_key": public_context_key,
        "public_context_dim": int(public_context_dim),
        "public_context_weights": public_context_weights,
    }


class _FactoryValidationPolicy:
    policy_name = "factory_validation"
    oracle_baseline = False

    def reset(self, *, seed: int, probe_budget_state: ProbeBudgetState) -> None:
        del seed, probe_budget_state

    def act(self, observation: Mapping[str, Any], probe_budget_state: ProbeBudgetState) -> PolicyAction:
        del observation, probe_budget_state
        raise RuntimeError("Factory validation policy cannot act.")

    def observe(self, transition: ActingTransition) -> None:
        del transition

    def representation(self) -> np.ndarray:
        return np.zeros(1, dtype=np.float32)

    def metrics(self) -> Mapping[str, Any]:
        return {"probe_count": 0, "environment_steps": 0, "realized_probe_cost": 0.0}


def _apply_zeroed_override(config: dict[str, Any]) -> None:
    """LDS-B3: overlay `evidence.partner_option_inference.mode = "zeroed"` on a
    loaded checkpoint config (in memory only). Single mutation point so the
    inferencer construction, the router policy string and the integrity gate all
    derive the SAME declared mode."""
    evidence = config.setdefault("evidence", {})
    inference = evidence.setdefault("partner_option_inference", {})
    inference["mode"] = "zeroed"


def _evidence_policy_for_config(config: dict[str, Any]) -> str:
    """E2: derive the evidence-policy string the router stamps and the gate checks.

    `evidence.partner_option_inference.mode == "zeroed"` ⇒ the zeroed-channel ablation
    policy (METHOD_LOCK sec18.8); anything else ⇒ the formal behavior-inferred policy.
    Kept in one place so the router-stamped string and the gate's admitted set cannot
    drift apart.
    """
    mode = str(
        ((config or {}).get("evidence", {}) or {})
        .get("partner_option_inference", {})
        .get("mode", "inferred")
    )
    return (
        "behavior_inferred_v1_zeroed_ablation"
        if mode == "zeroed"
        else "behavior_inferred_v1"
    )


def _evaluate_partner(
    ctx: EvalContext,
    partner_name: str,
    *,
    episodes: int,
    seed: int,
    max_episode_options: int,
    graph_override: GraphSpec,
    random_policy: bool,
    collect_diagnostics: bool,
    allow_diag_skip: bool = False,
    partner_set_override: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    env = _build_env(graph_override.layout_name, ctx.config)
    env.set_featurizer(NumpyFeaturizer(ctx.layout_graph))
    router = OCV2EvidenceRouter(
        graph_override,
        ctx.layout_graph.cell_to_entity,
        ctx.layout_graph.region_cells,
        evidence_policy=_evidence_policy_for_config(ctx.config),
    )
    partners = {partner.name: partner for partner in make_training_partners(
        ctx.option_lib,
        partner_set=partner_set_override
        or getattr(ctx, "partner_set_override", None)
        or str(ctx.config.get("training", {}).get("partner_set", "standard7")),
    )}
    if partner_name not in partners:
        raise KeyError(f"Unknown partner {partner_name!r}; choices={sorted(partners)}")
    partner = partners[partner_name]
    episode_rows: list[dict[str, Any]] = []

    for episode_idx in range(episodes):
        row = _run_episode(
            ctx,
            env,
            partner,
            router,
            graph_override,
            rng,
            seed + episode_idx,
            max_episode_options,
            random_policy=random_policy,
            collect_diagnostics=collect_diagnostics,
            allow_diag_skip=allow_diag_skip,
        )
        row["episode_id"] = int(episode_idx)
        episode_rows.append(row)

    aggregate = _aggregate_episodes(episode_rows)
    aggregate["partner_option_evidence"] = router.partner_option_evidence_summary()
    _validate_eval_integrity(
        aggregate,
        collect_diagnostics=collect_diagnostics,
        allow_diag_skip=allow_diag_skip,
        # LDS-B3: exact-match against the mode THIS run declared (derived from
        # the same config the router/inferencer were built from). An undeclared
        # zeroed run or a declared-zeroed run that silently ran inferred both fail.
        expected_policy=_evidence_policy_for_config(ctx.config),
    )
    return aggregate, episode_rows


def _run_episode(
    ctx: EvalContext,
    env: OCV2Adapter,
    partner: Any,
    router: OCV2EvidenceRouter,
    graph: GraphSpec,
    rng: np.random.Generator,
    seed: int,
    max_episode_options: int,
    *,
    random_policy: bool,
    collect_diagnostics: bool,
    allow_diag_skip: bool,
) -> dict[str, Any]:
    evidence_buffer = EvidenceBuffer(
        num_factors=graph.num_factors,
        window=int(ctx.config["training"]["evidence_window"]),
        evidence_dim=D_EVID,
    )
    evidence_buffer.reset()
    router.reset()
    obs, state0 = env.reset(seed)
    partner.reset(seed)
    ctx.path_c_acting_probe_remaining = ctx.path_c_acting_probe_budget
    _start_sequence_episode(
        evidence_buffer,
        ctx.q_net,
        obs,
        state0,
        ctx.option_lib,
        ctx.config,
        episode_id=f"evaluation:{ctx.graph_variant}:{int(seed)}",
        manifest_seed=int(seed),
    )
    partner_option_inferencer = make_behavior_option_inferencer(ctx.option_lib, ctx.config)
    partner_option_inferencer.reset(state0)
    _initialise_persistent_belief(
        evidence_buffer, ctx.method, ctx.belief_model, graph, torch.device("cpu"),
        _belief_persistence_enabled(ctx.config),
    )
    contribution_ledger = ContributionLedger.from_config(
        ctx.config.get("training", {})
    )

    episode_return = 0.0
    primitive_steps = 0
    option_count = 0
    blocking_events = 0
    delivery_seen = False
    ego_completion_seen = False
    partner_delivery_seen = False
    wrong_delivery_seen = False
    delta_values: list[float] = []
    mi_values: list[float] = []
    diagnostic_cost_values: list[float] = []
    swap_values: list[dict[str, Any]] = []
    belief_influence_values: list[dict[str, float]] = []
    termination_counts: dict[str, int] = {}
    option_kind_stats: dict[str, dict[str, Any]] = {}
    diagnostic_status_counts: dict[str, int] = {}
    selection_stats: dict[str, Any] = {
        "option_selection_count": 0,
        "forced_noop_count": 0,
        "no_valid_option_count": 0,
    }
    delivery_counts = _empty_delivery_counts()
    done = False

    while not done and option_count < max_episode_options:
        option_id = _select_option(
            ctx,
            obs,
            env.state,
            evidence_buffer,
            graph,
            rng,
            random_policy,
            partner_id=int(getattr(partner, "partner_id", 0)),
            selection_stats=selection_stats,
        )
        probe_decision = selection_stats.get("path_c_probe_last_decision") or {}
        selected_probe_cost = (
            float(probe_decision.get("realized_probe_cost", 0.0))
            if probe_decision.get("selected") is True
            else 0.0
        )
        option_return, done, obs, info = _execute_eval_option(
            ctx,
            env,
            obs,
            partner,
            router,
            graph,
            evidence_buffer,
            option_id,
            collect_diagnostics,
            allow_diag_skip,
            rng,
            contribution_ledger=contribution_ledger,
            partner_option_inferencer=partner_option_inferencer,
            truncate_at_boundary=(option_count + 1 >= max_episode_options),
            probe_cost=selected_probe_cost,
        )
        episode_return += option_return
        option_count += 1
        primitive_steps += int(info["primitive_steps"])
        blocking_events += int(info["blocking_events"])
        delivery_seen = bool(delivery_seen or info["delivery_seen"])
        _merge_counts(delivery_counts, info["delivery_counts"])
        ego_completion_seen = bool(
            ego_completion_seen or int(info["delivery_counts"].get("ego_sole_correct_delivery", 0)) > 0
        )
        partner_delivery_seen = bool(
            partner_delivery_seen or int(info["delivery_counts"].get("partner_delivery_event", 0)) > 0
        )
        wrong_delivery_seen = bool(
            wrong_delivery_seen or int(info["delivery_counts"].get("wrong_delivery_event", 0)) > 0
        )
        _merge_counts(diagnostic_status_counts, info["diagnostic_status_counts"])
        delta_values.extend(info["delta_info"])
        mi_values.extend(info["mi"])
        diagnostic_cost_values.extend(info["diagnostic_cost"])
        swap_values.extend(info["belief_swap"])
        belief_influence_values.extend(info["belief_influence"])
        _increment(termination_counts, str(info["termination_reason"]))
        _update_option_kind_stats(
            option_kind_stats,
            graph.options[int(option_id)].kind,
            str(info["termination_reason"]),
        )

    probe_count = int(selection_stats.get("path_c_probe_selected_count", 0))
    if (
        ctx.path_c_acting_probe_budget is not None
        and probe_count > ctx.path_c_acting_probe_budget
    ):
        raise RuntimeError("Acting policy exceeded its assigned probe budget.")
    if ctx.path_c_acting_probe_budget is not None:
        expected_remaining = int(ctx.path_c_acting_probe_budget) - probe_count
        if ctx.path_c_acting_probe_remaining != expected_remaining:
            raise RuntimeError(
                "Probe decision logs and the acting-budget controller disagree."
            )
    probe_cost = float(probe_count) * float(ctx.path_c_probe_cost_per_use)
    return {
        "return": float(episode_return),
        "raw_return": float(episode_return),
        "net_return": float(episode_return - probe_cost),
        # S20: headline completion is ego-owned success; team delivery is separate.
        "completed": bool(ego_completion_seen),
        "team_completed": bool(delivery_seen),
        "partner_delivery_episode": bool(partner_delivery_seen),
        "wrong_delivery_episode": bool(wrong_delivery_seen),
        "ego_correct_completed": bool(ego_completion_seen),
        "partner_correct_completed": bool(delivery_counts.get("partner_correct_delivery", 0) > 0),
        "wrong_delivery_completed": bool(wrong_delivery_seen),
        "delivery_counts": delivery_counts,
        "primitive_steps": int(primitive_steps),
        "option_count": int(option_count),
        "blocking_events": int(blocking_events),
        "blocking_rate": float(blocking_events / max(1, primitive_steps)),
        "delta_info_mean": _mean_or_nan(delta_values),
        "mi_mean": _mean_or_nan(mi_values),
        "diagnostic_cost_mean": _mean_or_nan(diagnostic_cost_values),
        "diagnostic_count": len(delta_values),
        "diagnostic_status_counts": diagnostic_status_counts,
        **selection_stats,
        # These fields are authoritative and therefore follow the expandable
        # diagnostic mapping, which is not allowed to override them.
        "path_c_probe_budget": ctx.path_c_acting_probe_budget,
        "path_c_probes_used": probe_count,
        "path_c_probe_budget_remaining": ctx.path_c_acting_probe_remaining,
        "path_c_probe_cost_per_use": float(ctx.path_c_probe_cost_per_use),
        "path_c_realized_probe_cost": probe_cost,
        "forced_noop_fraction": float(
            selection_stats["forced_noop_count"]
            / max(1, selection_stats["option_selection_count"])
        ),
        "no_valid_option_fraction": float(
            selection_stats["no_valid_option_count"]
            / max(1, selection_stats["option_selection_count"])
        ),
        "belief_swap": _aggregate_swap(swap_values),
        "belief_influence": _aggregate_belief_influence(belief_influence_values),
        "belief_influence_count": len(belief_influence_values),
        "termination_counts": termination_counts,
        "option_kind_stats": option_kind_stats,
    }


def _execute_eval_option(
    ctx: EvalContext,
    env: OCV2Adapter,
    obs: dict[str, np.ndarray],
    partner: Any,
    router: OCV2EvidenceRouter,
    graph: GraphSpec,
    evidence_buffer: EvidenceBuffer,
    option_id: int,
    collect_diagnostics: bool,
    allow_diag_skip: bool,
    rng: np.random.Generator,
    *,
    contribution_ledger: ContributionLedger | None = None,
    partner_option_inferencer: PartnerOptionInferencer | None = None,
    truncate_at_boundary: bool = False,
    probe_cost: float = 0.0,
) -> tuple[float, bool, dict[str, np.ndarray], dict[str, Any]]:
    opt = ctx.option_lib.options[int(option_id)]
    runtime = OptionRuntime(
        option_id=int(option_id),
        start_pos=get_agent_pos(env.state, 0),
    )
    reward_sum = 0.0
    realized_cost = 0.0
    duration = 0
    blocking_events = 0
    delivery_seen = False
    termination_reason = "running"
    delta_values: list[float] = []
    mi_values: list[float] = []
    diagnostic_cost_values: list[float] = []
    swap_values: list[dict[str, Any]] = []
    belief_influence_values: list[dict[str, float]] = []
    diagnostic_status_counts: dict[str, int] = {}
    delivery_counts = _empty_delivery_counts()
    done = False
    sequence_episode = evidence_buffer.sequence_episode()
    sequence_decision = (
        DecisionEvidenceBuffer(sequence_episode.spec)
        if isinstance(sequence_episode, EpisodeEvidenceBuffer)
        else None
    )
    if not np.isfinite(float(probe_cost)) or float(probe_cost) < 0.0:
        raise ValueError("Path C probe cost must be finite and non-negative.")
    reward_scale = float(
        (ctx.config.get("training", {}).get("value_bound") or {}).get(
            "reward_scale", 1.0
        )
    )
    sequence_reward_sum = -float(probe_cost) / reward_scale
    belief_before_option = _current_belief(ctx, evidence_buffer, graph)

    # F2 (RC-2b): executor patience. When the agent makes no progress toward the option's target
    # interaction cell for `block_patience` steps (livelock / partner camping the only stand cell),
    # terminate the option as blocked so the policy can re-decide, instead of thrashing/colliding
    # for the whole budget. Default 0 = off (current behavior).
    _patience = int((ctx.config.get("options") or {}).get("block_patience", 0))
    _spd = ctx.option_lib.layout_graph.shortest_path_dist
    _opt_targets = tuple(ctx.option_lib._target_cells(opt))

    def _dist_to_target(_st: Any) -> int | None:
        if not _opt_targets:
            return None
        _a = get_agent_pos(_st, 0)
        _ds = [_spd.get((_a, _t)) for _t in _opt_targets]
        _ds = [d for d in _ds if d is not None]
        return min(_ds) if _ds else None

    _best_dist = _dist_to_target(env.state)
    _stuck = 0
    _pot_positions = [e.pos for e in ctx.layout_graph.entities.values() if e.kind == "pot"]

    _budget = ctx.option_lib.option_budget(env.state, 0, int(option_id))
    while duration < _budget:
        _ostep = option_primitive_step(
            env,
            ctx.option_lib,
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
        if contribution_ledger is not None:
            contribution_ledger.update(event, ego_option_kind=str(opt.kind))
        ego_contributed = False
        if contribution_ledger is not None:
            ego_contributed = contribution_ledger.query_and_reset_on_delivery(event)
        # LDS-B1: eval return accounting must stay on the UNSHAPED scale —
        # terminal-progress shaping is a training scaffold (E1-rev), and
        # including it here would put shaped/unshaped checkpoints' returns on
        # different scales. Applies uniformly to checkpoint, random-baseline and
        # external-reference rollouts (they all execute options through here).
        eval_step_reward = _training_reward(
            step,
            ctx.config,
            "agent_0",
            event,
            ego_contributed=ego_contributed,
            include_terminal_shaping=False,
        )
        reward_sum += eval_step_reward
        if sequence_decision is not None:
            model_step_reward = _training_reward(
                step,
                ctx.config,
                "agent_0",
                event,
                ego_contributed=ego_contributed,
                include_terminal_shaping=True,
            )
            model_step_return = _sequence_step_return(
                model_step_reward,
                ctx.config,
            )
            sequence_reward_sum += model_step_return
            evidence_step_return = model_step_return
            if duration == 0 and float(probe_cost) > 0.0:
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
        realized_cost += float(ctx.config["training"].get("cost_per_step", 1.0))
        duration += 1
        x_f = router.route(
            event,
            ego_option_id=int(option_id),
            ego_option_elapsed=duration,
            ego_option_max_steps=opt.max_steps,
        )
        evidence_buffer.append(x_f)
        _advance_persistent_belief(
            evidence_buffer,
            ctx.method,
            ctx.belief_model,
            graph,
            torch.device("cpu"),
            x_f,
            _belief_persistence_enabled(ctx.config),
        )
        blocking_events += int(bool(event.collision_or_block))
        if getattr(ctx, "trace_steps", None) is not None and (
            ctx.trace_kind is None or opt.kind == ctx.trace_kind
        ):
            ctx.trace_steps.append({
                "kind": opt.kind,
                "step": int(duration),
                "agent": list(get_agent_pos(prev_state, 0)),
                "partner": list(get_agent_pos(prev_state, 1)),
                "target_pos": list(opt.target_pos) if opt.target_pos is not None else None,
                "facing": list(agent_facing_pos(prev_state, 0)),
                "action": int(ego_action),
                "is_interact": bool(int(ego_action) == int(_OCActions.interact)),
                "inv_before": int(get_inventory(prev_state, 0)),
                "inv_after": int(get_inventory(step.state, 0)),
                "blocked": bool(event.collision_or_block),
                "pots": [
                    [list(p), int(get_pot_contents(step.state, p)),
                     bool(is_pot_cooking(step.state, p)),
                     bool(is_pot_ready_for_plate(step.state, p, require_correct_recipe=False))]
                    for p in _pot_positions
                ],
            })
        _accumulate_delivery_counts(delivery_counts, event)
        delivery_seen = bool(delivery_seen or event.delivery_event)
        done = bool(step.dones.get("__all__", False))
        if _patience and _best_dist is not None and not done:
            _cur = _dist_to_target(step.state)
            if _cur is not None and _cur < _best_dist:
                _best_dist = _cur
                _stuck = 0
            elif _cur is not None and _cur > 0:
                _stuck += 1
            if _stuck >= _patience:
                obs = step.obs
                termination_reason = "blocked_no_progress"
                break
        terminated, termination_reason = ctx.option_lib.option_terminated(
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

    # Diagnostic label fix (mirrors train_aris._execute_option): relabel a budget
    # exhaustion so option stats don't record "running" as a terminal reason.
    if not done and termination_reason == "running":
        termination_reason = "budget_exhausted"

    # Push option-level FAILURE signal to evidence when the option did not succeed
    # (mirrors train_aris). Belief must see failure events at inference too.
    _failed = termination_reason in {"budget_exhausted", "max_steps", "env_max_steps", "blocked_no_progress"}
    if _failed and duration > 0:
        x_fail = router.route_failure_boundary(
            ego_option_id=int(option_id),
            ego_option_elapsed=duration,
            ego_option_max_steps=opt.max_steps,
        )
        evidence_buffer.append(x_fail)
        _advance_persistent_belief(
            evidence_buffer,
            ctx.method,
            ctx.belief_model,
            graph,
            torch.device("cpu"),
            x_fail,
            _belief_persistence_enabled(ctx.config),
        )

    if sequence_decision is not None:
        if not isinstance(sequence_episode, EpisodeEvidenceBuffer):
            raise RuntimeError("Sequence evaluation decision has no episode buffer.")
        valid_options_next = np.asarray(
            ctx.option_lib.valid_options(env.state, 0),
            dtype=bool,
        )
        sequence_truncated = bool(truncate_at_boundary and not done)
        next_evidence = sequence_decision.encode_boundary(
            observation=_obs_vector(obs, "agent_0"),
            ego_option_id=int(option_id),
            valid_actions=valid_options_next,
            terminated=bool(done),
            truncated=sequence_truncated,
        )
        sequence_episode.append_transition(
            action=int(option_id),
            reward=float(sequence_reward_sum),
            discount=float(ctx.config["training"]["gamma"]) ** int(duration),
            done=bool(done),
            truncated=sequence_truncated,
            next_evidence=next_evidence,
            next_valid_actions=valid_options_next,
        )

    if collect_diagnostics:
        belief_after_option = _current_belief(ctx, evidence_buffer, graph)
        diag = _option_diagnostics(
            ctx,
            obs,
            int(option_id),
            belief_before_option,
            belief_after_option,
            graph,
            allow_diag_skip=allow_diag_skip,
        )
        _increment(diagnostic_status_counts, str(diag.get("status", "unknown")))
        if diag.get("status") == "ok":
            delta_values.append(diag["delta_info"])
            mi_values.append(diag["mi"])
            diagnostic_cost_values.append(diag["diagnostic_cost"])
            swap_values.append(diag["belief_swap"])
            belief_influence_values.append(diag["belief_influence"])

    option_return = reward_sum - float(ctx.config["training"]["cost_coef"]) * realized_cost
    return (
        float(option_return),
        bool(done),
        obs,
        {
            "primitive_steps": int(duration),
            "blocking_events": int(blocking_events),
            "delivery_seen": bool(delivery_seen),
            "delivery_counts": delivery_counts,
            "termination_reason": termination_reason,
            "delta_info": delta_values,
            "mi": mi_values,
            "diagnostic_cost": diagnostic_cost_values,
            "belief_swap": swap_values,
            "belief_influence": belief_influence_values,
            "diagnostic_status_counts": diagnostic_status_counts,
        },
    )


_DIAG_SKIP = {
    "status": "shape_mismatch",
    "delta_info": float("nan"),
    "mi": float("nan"),
    "diagnostic_cost": float("nan"),
    "belief_swap": {"status": "shape_mismatch"},
    "belief_influence": {
        "belief_zero_delta": 0.0,
        "belief_uniform_delta": 0.0,
        "relevance_zero_delta": 0.0,
    },
}

_DIAG_UNSUPPORTED = {
    "status": "unsupported_method",
    "delta_info": float("nan"),
    "mi": float("nan"),
    "diagnostic_cost": float("nan"),
    "belief_swap": {"status": "unsupported_method"},
    "belief_influence": {
        "belief_zero_delta": 0.0,
        "belief_uniform_delta": 0.0,
        "relevance_zero_delta": 0.0,
    },
}


def _option_diagnostics(
    ctx: EvalContext,
    obs_next: dict[str, np.ndarray],
    option_id: int,
    belief_before: torch.Tensor,
    belief_after: torch.Tensor,
    graph: GraphSpec,
    *,
    allow_diag_skip: bool = False,
) -> dict[str, Any]:
    method = getattr(ctx, "method", "aris_bellman")
    q_net = getattr(ctx, "q_net", None)
    if method not in {"aris_bellman", "flat_factor"} or (
        q_net is not None and _uses_sequence_q(q_net)
    ):
        return dict(_DIAG_UNSUPPORTED)
    graph_batch = _graph_tensors(graph, 1, torch.device("cpu"))
    mode_mask = graph_batch["mode_mask"]
    if belief_before.shape != mode_mask.shape:
        if not allow_diag_skip:
            raise RuntimeError(
                "Diagnostic belief/mode-mask shape mismatch: "
                f"belief={tuple(belief_before.shape)} mode_mask={tuple(mode_mask.shape)}. "
                "Pass --allow_diag_skip only for non-formal smoke runs."
            )
        return dict(_DIAG_SKIP)
    obs_tensor = _tensor(_obs_vector(obs_next, "agent_0")[None, ...], torch.device("cpu"))
    with torch.no_grad():
        delta = realized_delta_info(
            ctx.q_net,
            obs_tensor,
            belief_before,
            belief_after,
            graph_batch,
            gamma=float(ctx.config["training"]["gamma"]),
        )
        mi = mutual_information_proxy(belief_before, belief_after, mode_mask)
        q_base = _base_q_values(ctx, obs_tensor, graph_batch, belief_after)
        _, cost = diagnostic_cost(q_base, int(option_id), delta, tau=0.0)
        swap = belief_swap_top_pairs(ctx.q_net, obs_tensor, belief_after, graph_batch, graph)
        q_actual = ctx.q_net(
            obs_tensor,
            belief_after,
            **_q_forward_kwargs(graph_batch),
        ).squeeze(0)
        belief_influence = _belief_influence_decomposition(
            ctx,
            obs_tensor,
            belief_after,
            _q_forward_kwargs(graph_batch),
            q_actual,
        )
        return {
            "status": "ok",
            "delta_info": float(delta.mean().item()),
            "mi": float(mi.mean().item()),
            "diagnostic_cost": float(cost.mean().item()),
            "belief_swap": swap,
            "belief_influence": belief_influence,
        }


def _belief_influence_decomposition(
    ctx: EvalContext,
    obs_tensor: torch.Tensor,
    belief_actual: torch.Tensor,
    graph_kwargs: dict[str, Any],
    q_actual: torch.Tensor,
) -> dict[str, float]:
    if not hasattr(ctx.q_net, "forward_with_belief_override"):
        return {
            "belief_zero_delta": 0.0,
            "belief_uniform_delta": 0.0,
            "relevance_zero_delta": 0.0,
        }
    with torch.no_grad():
        belief_zero = torch.zeros_like(belief_actual)
        q_zero = ctx.q_net.forward_with_belief_override(
            obs_tensor,
            belief_zero,
            graph_kwargs=graph_kwargs,
        ).squeeze(0)

        mode_mask = graph_kwargs.get("mode_mask")
        if mode_mask is not None:
            mm = mode_mask.to(dtype=belief_actual.dtype)
            n_valid = mm.sum(dim=-1, keepdim=True).clamp(min=1.0)
            belief_unif = mm / n_valid
        else:
            belief_unif = torch.ones_like(belief_actual) / max(1, belief_actual.shape[-1])
        q_unif = ctx.q_net.forward_with_belief_override(
            obs_tensor,
            belief_unif,
            graph_kwargs=graph_kwargs,
        ).squeeze(0)

        graph_kwargs_no_rel = dict(graph_kwargs)
        rm = graph_kwargs_no_rel.get("relevance_mask")
        if rm is not None:
            graph_kwargs_no_rel["relevance_mask"] = torch.zeros_like(rm)
        q_no_rel = ctx.q_net.forward_with_belief_override(
            obs_tensor,
            belief_actual,
            graph_kwargs=graph_kwargs_no_rel,
        ).squeeze(0)
    return {
        "belief_zero_delta": float((q_actual - q_zero).abs().max().item()),
        "belief_uniform_delta": float((q_actual - q_unif).abs().max().item()),
        "relevance_zero_delta": float((q_actual - q_no_rel).abs().max().item()),
    }


def _current_belief(
    ctx: EvalContext,
    evidence_buffer: EvidenceBuffer,
    graph: GraphSpec,
) -> torch.Tensor:
    graph_batch = _graph_tensors(graph, 1, torch.device("cpu"))
    evidence = _tensor(evidence_buffer.snapshot()[None, ...], torch.device("cpu"))
    evidence_mask = torch.as_tensor(
        evidence_buffer.snapshot_mask()[None, ...], dtype=torch.bool, device=torch.device("cpu")
    )
    evidence_lengths = torch.as_tensor([evidence_buffer.length()], dtype=torch.float32)
    hidden_np = evidence_buffer.belief_window_base_snapshot()
    belief_hidden = _tensor(hidden_np[None, ...], torch.device("cpu")) if hidden_np is not None else None
    with torch.no_grad():
        return _state_repr(
            ctx.method,
            ctx.belief_model,
            evidence,
            graph_batch,
            evidence_lengths=evidence_lengths,
            evidence_mask=evidence_mask,
            belief_hidden=belief_hidden,
        )


def _select_option(
    ctx: EvalContext,
    obs: dict[str, np.ndarray],
    state: Any,
    evidence_buffer: EvidenceBuffer,
    graph: GraphSpec,
    rng: np.random.Generator,
    random_policy: bool,
    partner_id: int | None = None,
    selection_stats: dict[str, Any] | None = None,
) -> int:
    _record_selection_attempt(selection_stats)
    _path_c_reset_probe_decision(selection_stats)
    valid = ctx.option_lib.valid_options(state, 0)
    valid_ids = np.flatnonzero(valid)
    if valid_ids.size == 0:
        _record_forced_noop(selection_stats)
        return _noop_option_id(ctx.option_lib)
    if random_policy:
        return int(rng.choice(valid_ids))

    if ctx.path_c_acting_probe_budget is not None and (
        getattr(ctx, "scripted_fsm", None) is not None
        or getattr(ctx, "scripted_priority", None) is not None
    ):
        raise RuntimeError(
            "Formal probe-budget evaluation cannot bypass the registered acting "
            "policy with a scripted selector."
        )

    if getattr(ctx, "scripted_fsm", None) is not None:
        chosen = ctx.scripted_fsm(ctx, state, valid_ids)
        if chosen is not None:
            return int(chosen)

    if getattr(ctx, "scripted_priority", None) is not None:
        scripted = _scripted_priority_select(ctx, graph, valid_ids)
        if scripted is not None:
            return scripted

    with torch.no_grad():
        graph_batch = _graph_tensors(graph, 1, torch.device("cpu"))
        obs_tensor = _tensor(_obs_vector(obs, "agent_0")[None, ...], torch.device("cpu"))
        if _uses_sequence_q(ctx.q_net):
            episode = evidence_buffer.sequence_episode()
            if not isinstance(episode, EpisodeEvidenceBuffer):
                raise RuntimeError(
                    "Sequence evaluation requires a started EgoEvidenceSpecV1 episode."
                )
            sequence_batch = episode.evidence_batch(torch.device("cpu"))
            valid_tensor = torch.as_tensor(valid, dtype=torch.bool)
            if not torch.equal(sequence_batch.valid_actions[0, -1], valid_tensor):
                raise RuntimeError(
                    "Evaluation valid actions disagree with EgoEvidenceSpecV1 history."
                )
            head_trace, _ = ctx.q_net.forward_sequence(sequence_batch)
            head_values = head_trace[:, -1]
            q_values = head_values.mean(dim=1).squeeze(0)
            q_values = q_values.masked_fill(~valid_tensor, -1e9)
            active_probe = bool(getattr(ctx, "active_probe_collection", False))
            acting_probe = bool(
                ctx.path_c_acting_probe_remaining is not None
                and ctx.path_c_acting_probe_remaining > 0
            )
            if active_probe or acting_probe:
                probe_choice = _path_c_probe_choice(
                    ctx.q_net,
                    obs_tensor,
                    None,
                    graph_batch,
                    q_values,
                    valid_tensor,
                    ctx.config,
                    partner_id=partner_id,
                    device=torch.device("cpu"),
                    selection_stats=selection_stats,
                    base_q_values=None,
                    eval_mode=not (active_probe or acting_probe),
                    probe_rng=rng,
                    head_values_override=head_values,
                )
                _annotate_path_c_probe_budget(ctx, selection_stats)
                if probe_choice is not None:
                    if acting_probe:
                        ctx.path_c_acting_probe_remaining -= 1
                    return probe_choice
            return int(torch.argmax(q_values).item())

        belief = _current_belief(ctx, evidence_buffer, graph)
        q_values = ctx.q_net(
            obs_tensor,
            belief,
            **_q_forward_kwargs(graph_batch),
            partner_id=_partner_id_tensor(partner_id, 1, torch.device("cpu")),
        ).squeeze(0)
        if getattr(ctx, "qaudit", None) is not None:
            _record_qaudit(ctx, obs_tensor, graph_batch, q_values, valid)
        valid_tensor = torch.as_tensor(valid, dtype=torch.bool)
        q_values = q_values.masked_fill(~valid_tensor, -1e9)
        active_probe = bool(getattr(ctx, "active_probe_collection", False))
        acting_probe = bool(
            ctx.path_c_acting_probe_remaining is not None
            and ctx.path_c_acting_probe_remaining > 0
        )
        if active_probe or acting_probe:
            base_q_values = None
            base_q_net = getattr(ctx, "path_c_probe_base_q_net", None)
            if active_probe and base_q_net is not None:
                base_q_values = base_q_net(
                    obs_tensor,
                    belief,
                    **_q_forward_kwargs(graph_batch),
                )
            probe_choice = _path_c_probe_choice(
                ctx.q_net,
                obs_tensor,
                belief,
                graph_batch,
                q_values,
                valid_tensor,
                ctx.config,
                partner_id=partner_id,
                device=torch.device("cpu"),
                selection_stats=selection_stats,
                base_q_values=base_q_values,
                eval_mode=not (active_probe or acting_probe),
                probe_rng=rng,
            )
            _annotate_path_c_probe_budget(ctx, selection_stats)
            if probe_choice is not None:
                if acting_probe:
                    ctx.path_c_acting_probe_remaining -= 1
                return probe_choice
        return int(torch.argmax(q_values).item())


def _annotate_path_c_probe_budget(
    ctx: EvalContext,
    selection_stats: dict[str, Any] | None,
) -> None:
    if selection_stats is None:
        return
    record = selection_stats.get("path_c_probe_last_decision")
    if not isinstance(record, dict):
        return
    remaining_before = ctx.path_c_acting_probe_remaining
    selected = record.get("selected") is True
    remaining_after = remaining_before
    probes_used_before: int | None = None
    probes_used_after: int | None = None
    if ctx.path_c_acting_probe_budget is not None:
        if remaining_before is None:
            raise RuntimeError("Formal probe budget has no remaining-count state.")
        probes_used_before = int(ctx.path_c_acting_probe_budget) - int(remaining_before)
        remaining_after = int(remaining_before) - (1 if selected else 0)
        if remaining_after < 0:
            raise RuntimeError("A probe was selected after exhausting the acting budget.")
        probes_used_after = int(ctx.path_c_acting_probe_budget) - int(remaining_after)
    record.update({
        "assigned_budget": ctx.path_c_acting_probe_budget,
        "budget_remaining_before": remaining_before,
        "budget_remaining_after": remaining_after,
        "probes_used_before": probes_used_before,
        "probes_used_after": probes_used_after,
        "probe_cost_per_use": float(ctx.path_c_probe_cost_per_use),
        "realized_probe_cost": (
            float(ctx.path_c_probe_cost_per_use)
            if selected
            else 0.0
        ),
        "cumulative_probe_cost_before": (
            None
            if probes_used_before is None
            else float(probes_used_before) * float(ctx.path_c_probe_cost_per_use)
        ),
        "cumulative_probe_cost_after": (
            None
            if probes_used_after is None
            else float(probes_used_after) * float(ctx.path_c_probe_cost_per_use)
        ),
    })
    selection_stats["path_c_probe_last_decision"] = dict(record)
    selection_stats["path_c_probe_last"] = dict(record)
    decisions = selection_stats.get("path_c_probe_decisions")
    if isinstance(decisions, list) and decisions:
        decisions[-1] = dict(record)


def _scripted_priority_select(
    ctx: EvalContext,
    graph: GraphSpec,
    valid_ids: np.ndarray,
) -> int | None:
    """RC-2 diagnostic: pick the valid option of the highest-priority kind, bypassing Q.

    Used to confirm whether the full fetch->cook->plate->serve pipeline is reachable and
    whether serve_soup success / ego delivery / completion ever fire under deliberate play.
    Returns None if no valid option matches any listed kind (caller falls back to Q).
    """
    kinds = {int(i): str(graph.options[int(i)].kind) for i in valid_ids}
    for target in ctx.scripted_priority:
        for vid in valid_ids:
            if kinds[int(vid)] == target:
                return int(vid)
    return None


def _record_qaudit(
    ctx: EvalContext,
    obs_tensor: torch.Tensor,
    graph_batch: dict[str, Any],
    q_values: torch.Tensor,
    valid: np.ndarray,
) -> None:
    """RC-1 diagnostic: log the (q_total, q_base) decomposition for the current decision.

    Read-only: it does NOT change the returned argmax. adv_sum is recovered downstream as
    q_total - q_base (the network defines q_values = q_base + sum_f A_f). Skips silently if
    the wrapped net does not expose a factor-local base head.
    """
    if not (hasattr(ctx.q_net, "q_net") and hasattr(ctx.q_net.q_net, "q_base_values")):
        return
    encoded = ctx.q_net.encoder(obs_tensor)
    q_base = ctx.q_net.q_net.q_base_values(encoded, graph_batch["option_mask"]).squeeze(0)
    ctx.qaudit.append(
        {
            "q_full": [float(x) for x in q_values.detach().tolist()],
            "q_base": [float(x) for x in q_base.detach().tolist()],
            "valid": [bool(x) for x in valid.tolist()],
        }
    )


def _factor_deletion_q_proxy_diagnostics(ctx: EvalContext) -> list[dict[str, Any]]:
    if ctx.graph.num_factors == 0 or _uses_sequence_q(ctx.q_net):
        return []

    q_drops = _factor_deletion_q_proxy(ctx)
    rows = []
    for factor in ctx.graph.factors:
        rows.append(
            {
                "factor_id": int(factor.id),
                "factor_kind": factor.factor_kind,
                "option_i": int(factor.option_i),
                "option_j": int(factor.option_j),
                "q_proxy_drop": float(q_drops[int(factor.id)]),
                "ablation_mode": "q_proxy_factor_mask",
                "ablation_scope": "single_initial_state_empty_evidence",
                "rollout_episodes": 0,
            }
        )
    rows.sort(key=lambda row: (-row["q_proxy_drop"], row["factor_id"]))
    return rows


def _factor_deletion_rollout_diagnostics(
    ctx: EvalContext,
    partner_name: str,
    base_net_return: float,
    *,
    episodes: int,
    seed: int,
    max_episode_options: int,
    allow_diag_skip: bool = False,
) -> list[dict[str, Any]]:
    deleted_returns: dict[int, float] = {}
    for factor in ctx.graph.factors:
        deleted_graph = graph_with_deleted_factor(ctx.graph, int(factor.id))
        aggregate, _ = _evaluate_partner(
            ctx,
            partner_name,
            episodes=episodes,
            seed=seed + int(factor.id),
            max_episode_options=max_episode_options,
            graph_override=deleted_graph,
            random_policy=False,
            collect_diagnostics=False,
            allow_diag_skip=allow_diag_skip,
        )
        deleted_returns[int(factor.id)] = float(aggregate["mean_net_return"])

    rows = factor_deletion_return_drop(
        ctx.graph,
        base_net_return,
        lambda factor_id: deleted_returns[int(factor_id)],
    )
    for row in rows:
        row["ablation_mode"] = "rollout_factor_mask"
        row["ablation_scope"] = "factor_mask_relevance_mode_disabled"
        row["rollout_episodes"] = int(episodes)
        row["return_kind"] = "mean_net_return"
    return rows


def _factor_deletion_q_proxy(ctx: EvalContext) -> dict[int, float]:
    if _uses_sequence_q(ctx.q_net):
        raise ValueError(
            "Factor-deletion Q proxies are undefined for EgoEvidenceSpecV1 sequence models."
        )
    env = _build_env(ctx.graph.layout_name, ctx.config)
    env.set_featurizer(NumpyFeaturizer(ctx.layout_graph))
    obs, _ = env.reset(0)
    evidence_buffer = EvidenceBuffer(
        num_factors=ctx.graph.num_factors,
        window=int(ctx.config["training"]["evidence_window"]),
        evidence_dim=D_EVID,
    )
    obs_tensor = _tensor(_obs_vector(obs, "agent_0")[None, ...], torch.device("cpu"))
    base_graph_batch = _graph_tensors(ctx.graph, 1, torch.device("cpu"))
    base_belief = _current_belief(ctx, evidence_buffer, ctx.graph)
    with torch.no_grad():
        base_value = ctx.q_net(
            obs_tensor,
            base_belief,
            **_q_forward_kwargs(base_graph_batch),
        ).max(dim=-1).values.item()

    drops = {}
    for factor in ctx.graph.factors:
        deleted_graph = graph_with_deleted_factor(ctx.graph, int(factor.id))
        deleted_batch = _graph_tensors(deleted_graph, 1, torch.device("cpu"))
        deleted_belief = _current_belief(ctx, evidence_buffer, deleted_graph)
        with torch.no_grad():
            deleted_value = ctx.q_net(
                obs_tensor,
                deleted_belief,
                **_q_forward_kwargs(deleted_batch),
            ).max(dim=-1).values.item()
        drops[int(factor.id)] = float(base_value - deleted_value)
    return drops


_ENV_CACHE_KEYS = (
    "max_steps",
    "agent_view_size",
    "negative_rewards",
    "sample_recipe_on_delivery",
    "random_reset",
    "random_agent_positions",
    "force_path_planning",
)


# Option-runtime config that changes valid options / budgets / termination and
# therefore rollout returns (review BLOCK fix — was missing from the key).
_OPTIONS_CACHE_KEYS = (
    "max_option_steps",
    "strict_preconditions",
    "dynamic_budget",
    "block_patience",
)
# Bump when the key scheme or entry shape changes so stale valid-JSON entries from
# an older code version are rejected rather than trusted (review MEDIUM fix).
# v3: key gained sparse-credit + terminal-progress reward signatures (LDS-C2).
# v4: baseline entries expose raw and net return fields explicitly.
_BASELINE_CACHE_SCHEMA = 4


def _baseline_env_payload(config: dict[str, Any]) -> dict[str, Any]:
    """The subset of config that changes a reference/random rollout's outcome."""
    env_cfg = (config or {}).get("env", {}) or {}
    opt_cfg = (config or {}).get("options", {}) or {}
    training_cfg = ((config or {}).get("training", {}) or {})
    return {
        "env": {k: env_cfg.get(k) for k in _ENV_CACHE_KEYS},
        "options": {k: opt_cfg.get(k) for k in _OPTIONS_CACHE_KEYS},
        "reward_config": reward_config_payload(config or {}),
        # LDS-C2: reward inputs beyond reward_config_payload that change rollout
        # return accounting — sparse-credit mode/constants and terminal-progress
        # shaping. Resolved via the SAME helpers the train reward path uses (no
        # hand-rolled signature that could drift). Terminal shaping is excluded
        # from eval returns since LDS-B1, but stays in the key: over-keying only
        # costs cache misses, never correctness.
        "sparse_credit": sparse_credit_params(training_cfg),
        "terminal_progress": terminal_progress_params(training_cfg),
        # codex diff-review blocker: shaped-reward SHARING mode also changes the
        # rollout return (train_aris._training_reward reads it), so it must key.
        "allow_shared_shaping": bool(training_cfg.get("allow_shared_shaping", False)),
        "partner_set": str(training_cfg.get("partner_set", "standard7")),
    }


def _baseline_cache_target(
    cache_dir: Path | None,
    *,
    kind: str,
    layout: str,
    config: dict[str, Any],
    partner: str,
    episodes: int,
    seed: int,
    max_episode_options: int,
    extra: Any = None,
) -> tuple[Path | None, str | None]:
    """(path, key) for a checkpoint-independent baseline rollout.

    Key covers everything that determines the result: kind, layout, the env/options/
    reward config subset, partner_set, partner, episodes, max_episode_options, seed.
    `extra` carries reference-checkpoint hashes for the external variant (which IS
    checkpoint-dependent). Returns (None, None) when caching is disabled.
    """
    if cache_dir is None:
        return None, None
    key = sha256_json(
        {
            "schema": _BASELINE_CACHE_SCHEMA,
            "kind": kind,
            "layout": layout,
            "config": _baseline_env_payload(config),
            "partner": partner,
            "episodes": int(episodes),
            "max_episode_options": int(max_episode_options),
            "seed": int(seed),
            "extra": extra,
        }
    )
    return cache_dir / f"{kind}_{key}.json", key


def _resolve_baseline_cache_dir(args: argparse.Namespace) -> Path | None:
    """Cache dir for baseline rollouts: explicit --baseline_cache_dir, 'none' to
    disable, or default <output_dir>/.baseline_cache."""
    raw = getattr(args, "baseline_cache_dir", None)
    if raw is not None and str(raw).lower() == "none":
        return None
    cache_dir = Path(raw) if raw else (Path(args.output).parent / ".baseline_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _read_baseline_cache(path: Path | None, expected_key: str | None) -> dict[str, Any] | None:
    """Return the cached value only if the file parses, matches the current schema,
    and its embedded key equals the expected key (defends against corrupt/partial
    files and stale entries from an older key scheme)."""
    if path is None or expected_key is None or not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None  # corrupted/partial → recompute
    if not isinstance(raw, dict):
        return None
    if raw.get("_cache_schema") != _BASELINE_CACHE_SCHEMA:
        return None
    if raw.get("_cache_key") != expected_key:
        return None
    value = raw.get("value")
    return value if isinstance(value, dict) else None


def _write_baseline_cache(
    path: Path | None, expected_key: str | None, value: dict[str, Any]
) -> None:
    if path is None or expected_key is None:
        return
    payload = {
        "_cache_schema": _BASELINE_CACHE_SCHEMA,
        "_cache_key": expected_key,
        "value": value,
    }
    # Atomic on POSIX: write a pid-tagged temp then rename. Parallel eval runs as
    # independent subprocesses; the key now covers every result-affecting input, so a
    # same-path collision means identical inputs and last-writer-wins is harmless.
    tmp = path.parent / f"{path.name}.tmp.{os.getpid()}"
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        tmp.unlink(missing_ok=True)


def _random_baselines(
    ctx: EvalContext,
    partner_names: list[str],
    *,
    episodes: int,
    seed: int,
    max_episode_options: int,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    baselines = {}
    for idx, partner_name in enumerate(partner_names):
        cache_path, cache_key = _baseline_cache_target(
            cache_dir,
            kind="random_policy",
            layout=ctx.graph.layout_name,
            config=ctx.config,
            partner=partner_name,
            episodes=episodes,
            seed=seed + idx,
            max_episode_options=max_episode_options,
        )
        cached = _read_baseline_cache(cache_path, cache_key)
        if cached is not None:
            baselines[partner_name] = cached
            continue
        aggregate, _ = _evaluate_partner(
            ctx,
            partner_name,
            episodes=episodes,
            seed=seed + idx,
            max_episode_options=max_episode_options,
            graph_override=ctx.graph,
            random_policy=True,
            collect_diagnostics=False,
        )
        entry = {
            "base_kind": "random_policy",
            "mean_return": aggregate["mean_return"],
            "mean_raw_return": aggregate["mean_raw_return"],
            "mean_net_return": aggregate["mean_net_return"],
            "mean_probes_used": aggregate["mean_probes_used"],
            "mean_realized_probe_cost": aggregate["mean_realized_probe_cost"],
            "completion_rate": aggregate["completion_rate"],
            "blocking_rate": aggregate["blocking_rate"],
        }
        _write_baseline_cache(cache_path, cache_key, entry)
        baselines[partner_name] = entry
    return baselines


def _external_reference_baselines(
    args: argparse.Namespace,
    partner_names: list[str],
    *,
    episodes: int,
    seed: int,
    max_episode_options: int,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    base_ctx = _load_context(
        _resolve_checkpoint_path(Path(args.reference_base_checkpoint)),
        "reference_base",
    )
    ref_ctx = _load_context(
        _resolve_checkpoint_path(Path(args.reference_ref_checkpoint)),
        "reference_ref",
    )
    # These baselines DO depend on the two reference checkpoints, so the cache key
    # includes their content hashes (guards against a checkpoint being swapped).
    ref_extra = {
        "base_ckpt_sha256": sha256_file(base_ctx.checkpoint_path),
        "ref_ckpt_sha256": sha256_file(ref_ctx.checkpoint_path),
    }
    baselines: dict[str, Any] = {}
    for idx, partner_name in enumerate(partner_names):
        cache_path, cache_key = _baseline_cache_target(
            cache_dir,
            kind="external_reference",
            layout=base_ctx.graph.layout_name,
            config=base_ctx.config,
            partner=partner_name,
            episodes=episodes,
            seed=seed + idx,
            max_episode_options=max_episode_options,
            extra=ref_extra,
        )
        cached = _read_baseline_cache(cache_path, cache_key)
        if cached is not None:
            baselines[partner_name] = cached
            continue
        base_aggregate, _ = _evaluate_partner(
            base_ctx,
            partner_name,
            episodes=episodes,
            seed=seed + idx,
            max_episode_options=max_episode_options,
            graph_override=base_ctx.graph,
            random_policy=False,
            collect_diagnostics=False,
        )
        ref_aggregate, _ = _evaluate_partner(
            ref_ctx,
            partner_name,
            episodes=episodes,
            seed=seed + 10_000 + idx,
            max_episode_options=max_episode_options,
            graph_override=ref_ctx.graph,
            random_policy=False,
            collect_diagnostics=False,
        )
        entry = {
            "base_checkpoint": str(base_ctx.checkpoint_path),
            "ref_checkpoint": str(ref_ctx.checkpoint_path),
            "base_method": base_ctx.method,
            "ref_method": ref_ctx.method,
            "base_mean_return": base_aggregate["mean_return"],
            "ref_mean_return": ref_aggregate["mean_return"],
            "base_mean_net_return": base_aggregate["mean_net_return"],
            "ref_mean_net_return": ref_aggregate["mean_net_return"],
            "reference_probe_budget": None,
            "reference_probe_cost_per_use": 0.0,
            "episodes": int(episodes),
        }
        _write_baseline_cache(cache_path, cache_key, entry)
        baselines[partner_name] = entry
    return baselines


def _attach_external_reference_gaps(
    results: list[dict[str, Any]],
    references: dict[str, Any],
) -> None:
    for row in results:
        reference = references[row["partner"]]
        row["aggregate"]["reference_gap_closure"] = reference_gap_closure(
            float(row["aggregate"]["mean_return"]),
            float(reference["base_mean_return"]),
            float(reference["ref_mean_return"]),
        )
        row["aggregate"]["reference_gap_closure_net_return"] = reference_gap_closure(
            float(row["aggregate"]["mean_net_return"]),
            float(reference["base_mean_net_return"]),
            float(reference["ref_mean_net_return"]),
        )


def _attach_within_run_relative_returns(
    results: list[dict[str, Any]],
    baselines: dict[str, Any],
) -> None:
    by_partner: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        by_partner.setdefault(result["partner"], []).append(result)
    for partner, rows in by_partner.items():
        r_base = float(baselines[partner]["mean_return"])
        r_ref = max(float(row["aggregate"]["mean_return"]) for row in rows)
        net_base = float(baselines[partner]["mean_net_return"])
        net_ref = max(float(row["aggregate"]["mean_net_return"]) for row in rows)
        for row in rows:
            metric = reference_gap_closure(
                float(row["aggregate"]["mean_return"]),
                r_base,
                r_ref,
            )
            if metric.get("status") == "ok" and float(row["aggregate"]["mean_return"]) == r_ref:
                metric.pop("raw_value", None)
                metric["status"] = "within_run_reference_variant"
            row["aggregate"]["within_run_relative_return"] = metric
            net_metric = reference_gap_closure(
                float(row["aggregate"]["mean_net_return"]),
                net_base,
                net_ref,
            )
            if (
                net_metric.get("status") == "ok"
                and float(row["aggregate"]["mean_net_return"]) == net_ref
            ):
                net_metric.pop("raw_value", None)
                net_metric["status"] = "within_run_reference_variant"
            row["aggregate"]["within_run_relative_net_return"] = net_metric


def _reference_semantics(has_external_references: bool) -> dict[str, str]:
    if has_external_references:
        return {
            "r_base": "external_reference_base_checkpoint",
            "r_ref": "external_reference_ref_checkpoint",
            "reference_type": "external_checkpoint_reference_gap_closure",
            "raw_return_field": "mean_return",
            "net_return_field": "mean_net_return",
            "reference_probe_policy": "no_probe",
        }
    return {
        "r_base": "random_policy_rollout",
        "r_ref": "best_mean_return_among_requested_graph_variants_per_partner",
        "reference_type": "within_run_relative_return_not_reference_gap_closure",
        "raw_return_field": "mean_return",
        "net_return_field": "mean_net_return",
        "reference_probe_policy": "random_action_no_probe",
    }


def _throughput_fields(
    delivery_counts: dict[str, Any], n_episodes: int
) -> dict[str, Any]:
    """LDS-C3: canonical throughput fields (METHOD_LOCK sec18.9.2 lens).

    Freezes the denominator (episodes) and actor attribution HERE so downstream
    aggregation scripts cannot derive divergent throughput definitions.
    Undefined values are ``None``, never 0.0 — absence of measurement is not a
    zero (LDS-C4 lesson).
    """
    serve_denom = int(delivery_counts["ego_correct_delivery"]) + int(
        delivery_counts["partner_correct_delivery"]
    )
    return {
        "team_correct_delivery_throughput_per_episode": (
            float(delivery_counts["correct_delivery"]) / n_episodes
            if n_episodes > 0
            else None
        ),
        "ego_correct_delivery_throughput_per_episode": (
            float(delivery_counts["ego_correct_delivery"]) / n_episodes
            if n_episodes > 0
            else None
        ),
        "ego_serve_share": (
            float(delivery_counts["ego_correct_delivery"]) / serve_denom
            if serve_denom > 0
            else None
        ),
    }


def _validate_episode_probe_accounting(row: dict[str, Any]) -> None:
    raw_return = float(row.get("raw_return", row["return"]))
    legacy_return = float(row["return"])
    net_return = float(row.get("net_return", legacy_return))
    realized_cost = float(row.get("path_c_realized_probe_cost", 0.0))
    cost_per_use = float(row.get("path_c_probe_cost_per_use", 0.0))
    raw_selected_count = row.get("path_c_probe_selected_count", 0)
    raw_opportunity_count = row.get("path_c_probe_opportunity_count", 0)
    raw_probes_used = row.get("path_c_probes_used", raw_selected_count)
    for name, value in (
        ("path_c_probe_selected_count", raw_selected_count),
        ("path_c_probe_opportunity_count", raw_opportunity_count),
        ("path_c_probes_used", raw_probes_used),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or int(value) < 0
        ):
            raise TypeError(f"{name} must be a non-negative integer.")
    selected_count = int(raw_selected_count)
    opportunity_count = int(raw_opportunity_count)
    probes_used = int(raw_probes_used)
    budget = row.get("path_c_probe_budget")
    if not all(np.isfinite(value) for value in (
        raw_return,
        legacy_return,
        net_return,
        realized_cost,
        cost_per_use,
    )):
        raise ValueError("Probe accounting and return fields must be finite.")
    if realized_cost < 0.0 or cost_per_use < 0.0:
        raise ValueError("Probe costs must be non-negative.")
    if probes_used < 0 or opportunity_count < 0 or probes_used != selected_count:
        raise ValueError("Probe counters disagree or contain a negative count.")
    if not np.isclose(raw_return, legacy_return, rtol=0.0, atol=1.0e-9):
        raise ValueError("raw_return and return disagree.")
    expected_cost = float(probes_used) * cost_per_use
    if not np.isclose(realized_cost, expected_cost, rtol=1.0e-9, atol=1.0e-9):
        raise ValueError("Realized probe cost does not equal probes_used times unit cost.")
    if not np.isclose(
        net_return,
        raw_return - realized_cost,
        rtol=1.0e-9,
        atol=1.0e-9,
    ):
        raise ValueError("net_return does not equal raw_return minus probe cost.")

    expected_remaining: int | None = None
    if budget is not None:
        if (
            isinstance(budget, bool)
            or not isinstance(budget, (int, np.integer))
            or int(budget) < 0
        ):
            raise ValueError("Probe budget must be a non-negative integer or None.")
        if probes_used > int(budget):
            raise ValueError("Observed probes exceed the assigned budget.")
        expected_remaining = int(budget) - probes_used
        if row.get("path_c_probe_budget_remaining") != expected_remaining:
            raise ValueError("Episode probe-budget remainder is inconsistent.")

    decisions = row.get("path_c_probe_decisions", [])
    if decisions is None:
        decisions = []
    if not isinstance(decisions, list) or any(
        not isinstance(record, dict) for record in decisions
    ):
        raise TypeError("path_c_probe_decisions must be a list of mappings.")
    if len(decisions) != opportunity_count:
        raise ValueError("Probe opportunity count disagrees with the decision logs.")
    logged_selected = 0
    logged_cost = 0.0
    logged_remaining = None if budget is None else int(budget)
    previous_selection_count = 0
    for record in decisions:
        if type(record.get("selected")) is not bool:
            raise TypeError("Probe decision selected must be boolean.")
        selection_count = record.get("selection_count")
        if (
            isinstance(selection_count, bool)
            or not isinstance(selection_count, (int, np.integer))
            or int(selection_count) <= previous_selection_count
            or int(selection_count) > int(row.get("option_selection_count", 0))
        ):
            raise ValueError("Probe decision logs have invalid selection ordering.")
        previous_selection_count = int(selection_count)
        selected = record.get("selected") is True
        logged_selected += int(selected)
        record_unit_cost = float(record.get("probe_cost_per_use", float("nan")))
        if not np.isclose(
            record_unit_cost,
            cost_per_use,
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError("A probe decision log changed the frozen per-use cost.")
        record_cost = float(record.get("realized_probe_cost", 0.0))
        logged_cost += record_cost
        if not np.isclose(
            record_cost,
            cost_per_use if selected else 0.0,
            rtol=1.0e-9,
            atol=1.0e-9,
        ):
            raise ValueError("A probe decision log has an inconsistent realized cost.")
        if record.get("assigned_budget") != budget:
            raise ValueError("A probe decision log has the wrong assigned budget.")
        if record.get("reason") in {"selected", "threshold", "return_floor"}:
            scores = record.get("candidate_scores")
            mask = record.get("candidate_mask")
            propensity = record.get("propensity")
            if (
                not isinstance(scores, list)
                or not isinstance(mask, list)
                or len(scores) != len(mask)
                or not scores
            ):
                raise ValueError("Probe candidate scores and mask must be aligned.")
            if not all(np.isfinite(float(value)) for value in scores) or any(
                type(value) is not bool for value in mask
            ):
                raise ValueError("Probe candidate scores and mask have invalid values.")
            option_id = record.get("option_id")
            if (
                isinstance(option_id, bool)
                or not isinstance(option_id, (int, np.integer))
                or not 0 <= int(option_id) < len(mask)
                or not mask[int(option_id)]
            ):
                raise ValueError("Probe decision option_id is outside the candidate support.")
            if (
                propensity is None
                or not np.isfinite(float(propensity))
                or not 0.0 < float(propensity) <= 1.0
            ):
                raise ValueError("Probe decision propensity must be in (0, 1].")
        if budget is not None:
            before = record.get("budget_remaining_before")
            after = record.get("budget_remaining_after")
            if not isinstance(before, int) or not isinstance(after, int):
                raise ValueError("Formal probe logs require integer budget remainders.")
            if before != logged_remaining:
                raise ValueError("Probe decision logs are not a contiguous budget trace.")
            if after != before - int(selected) or not 0 <= after <= int(budget):
                raise ValueError("Probe decision budget transition is inconsistent.")
            used_before = int(budget) - before
            used_after = int(budget) - after
            if (
                record.get("probes_used_before") != used_before
                or record.get("probes_used_after") != used_after
                or not np.isclose(
                    float(record.get("cumulative_probe_cost_before", float("nan"))),
                    float(used_before) * cost_per_use,
                    rtol=1.0e-9,
                    atol=1.0e-9,
                )
                or not np.isclose(
                    float(record.get("cumulative_probe_cost_after", float("nan"))),
                    float(used_after) * cost_per_use,
                    rtol=1.0e-9,
                    atol=1.0e-9,
                )
            ):
                raise ValueError("Probe decision cumulative accounting is inconsistent.")
            logged_remaining = after
    if logged_selected != probes_used or not np.isclose(
        logged_cost,
        realized_cost,
        rtol=1.0e-9,
        atol=1.0e-9,
    ):
        raise ValueError("Episode probe totals disagree with the decision logs.")
    if decisions and budget is not None:
        if decisions[-1].get("budget_remaining_after") != expected_remaining:
            raise ValueError("Final probe log does not match the episode budget remainder.")


def _aggregate_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    for row in episodes:
        _validate_episode_probe_accounting(row)
    budget_values = {row.get("path_c_probe_budget") for row in episodes}
    cost_values = {
        float(row.get("path_c_probe_cost_per_use", 0.0)) for row in episodes
    }
    if len(budget_values) > 1 or len(cost_values) > 1:
        raise ValueError("Evaluation episodes do not share one probe budget and cost.")
    returns = [float(row["return"]) for row in episodes]
    net_returns = [float(row.get("net_return", row["return"])) for row in episodes]
    probes_used = [
        int(row.get("path_c_probes_used", row.get("path_c_probe_selected_count", 0)))
        for row in episodes
    ]
    realized_probe_costs = [
        float(row.get("path_c_realized_probe_cost", 0.0)) for row in episodes
    ]
    primitive_steps = sum(int(row["primitive_steps"]) for row in episodes)
    blocking_events = sum(int(row["blocking_events"]) for row in episodes)
    term_counts: dict[str, int] = {}
    option_kind_stats: dict[str, dict[str, Any]] = {}
    diagnostic_status_counts: dict[str, int] = {}
    delivery_counts = _empty_delivery_counts()
    option_selection_count = 0
    forced_noop_count = 0
    no_valid_option_count = 0
    path_c_probe_selected_count = 0
    path_c_probe_opportunity_count = 0
    for row in episodes:
        _merge_counts(delivery_counts, row.get("delivery_counts", {}))
        _merge_counts(diagnostic_status_counts, row.get("diagnostic_status_counts", {}))
        option_selection_count += int(row.get("option_selection_count", 0))
        forced_noop_count += int(row.get("forced_noop_count", 0))
        no_valid_option_count += int(row.get("no_valid_option_count", 0))
        path_c_probe_selected_count += int(row.get("path_c_probe_selected_count", 0))
        path_c_probe_opportunity_count += int(row.get("path_c_probe_opportunity_count", 0))
        for key, value in row["termination_counts"].items():
            term_counts[key] = term_counts.get(key, 0) + int(value)
        for kind, item in row.get("option_kind_stats", {}).items():
            aggregate = option_kind_stats.setdefault(
                kind,
                {
                    "attempt_count": 0,
                    "success_count": 0,
                    "timeout_count": 0,
                    "success_rate": 0.0,
                    "termination_reason_histogram": {},
                },
            )
            aggregate["attempt_count"] = int(aggregate["attempt_count"]) + int(
                item.get("attempt_count", 0)
            )
            aggregate["success_count"] = int(aggregate["success_count"]) + int(
                item.get("success_count", 0)
            )
            aggregate["timeout_count"] = int(aggregate["timeout_count"]) + int(
                item.get("timeout_count", 0)
            )
            histogram = aggregate["termination_reason_histogram"]
            for reason, count in item.get("termination_reason_histogram", {}).items():
                histogram[reason] = int(histogram.get(reason, 0)) + int(count)
    for item in option_kind_stats.values():
        item["success_rate"] = float(
            int(item["success_count"]) / max(1, int(item["attempt_count"]))
        )
    return {
        **_throughput_fields(delivery_counts, len(episodes)),
        "mean_return": _mean_or_nan(returns),
        "mean_raw_return": _mean_or_nan(returns),
        "return_std": float(np.std(returns)) if returns else float("nan"),
        "raw_return_std": float(np.std(returns)) if returns else float("nan"),
        "mean_net_return": _mean_or_nan(net_returns),
        "net_return_std": (
            float(np.std(net_returns)) if net_returns else float("nan")
        ),
        "path_c_probe_budget": (
            episodes[0].get("path_c_probe_budget") if episodes else None
        ),
        "path_c_probe_cost_per_use": (
            float(episodes[0].get("path_c_probe_cost_per_use", 0.0))
            if episodes else 0.0
        ),
        "total_probes_used": int(sum(probes_used)),
        "mean_probes_used": _mean_or_nan([float(value) for value in probes_used]),
        "total_realized_probe_cost": float(sum(realized_probe_costs)),
        "mean_realized_probe_cost": _mean_or_nan(realized_probe_costs),
        "completion_rate": _mean_or_nan([float(row["completed"]) for row in episodes]),
        "headline_success_metric": "ego_correct_completion_rate",
        "ego_correct_completion_rate": _mean_or_nan([float(row.get("ego_correct_completed", False)) for row in episodes]),
        "team_completion_rate": _mean_or_nan([float(row.get("team_completed", False)) for row in episodes]),
        "partner_correct_completion_rate": _mean_or_nan([float(row.get("partner_correct_completed", False)) for row in episodes]),
        "wrong_delivery_rate": _mean_or_nan([float(row.get("wrong_delivery_completed", False)) for row in episodes]),
        "partner_delivery_episode_rate": _mean_or_nan([float(row.get("partner_delivery_episode", False)) for row in episodes]),
        "wrong_delivery_episode_rate": _mean_or_nan([float(row.get("wrong_delivery_episode", False)) for row in episodes]),
        "delivery_counts": delivery_counts,
        "ego_delivery_count": int(delivery_counts["ego_delivery_event"]),
        "partner_delivery_count": int(delivery_counts["partner_delivery_event"]),
        "correct_delivery_count": int(delivery_counts["correct_delivery"]),
        "wrong_delivery_count": int(delivery_counts["wrong_delivery_event"]),
        "ego_correct_delivery_count": int(delivery_counts["ego_correct_delivery"]),
        "ego_sole_correct_delivery_count": int(delivery_counts.get("ego_sole_correct_delivery", 0)),
        "partner_correct_delivery_count": int(delivery_counts["partner_correct_delivery"]),
        "ego_wrong_delivery_count": int(delivery_counts["ego_wrong_delivery_event"]),
        "partner_wrong_delivery_count": int(delivery_counts["partner_wrong_delivery_event"]),
        "blocking_rate": float(blocking_events / max(1, primitive_steps)),
        "mean_duration": _mean_or_nan([float(row["primitive_steps"]) for row in episodes]),
        "primitive_steps": int(primitive_steps),
        "option_selection_count": int(option_selection_count),
        "forced_noop_count": int(forced_noop_count),
        "no_valid_option_count": int(no_valid_option_count),
        "path_c_probe": {
            "selected": int(path_c_probe_selected_count),
            "opportunities": int(path_c_probe_opportunity_count),
            "assigned_budget_per_episode": (
                episodes[0].get("path_c_probe_budget") if episodes else None
            ),
            "cost_per_use": (
                float(episodes[0].get("path_c_probe_cost_per_use", 0.0))
                if episodes else 0.0
            ),
            "total_probes_used": int(sum(probes_used)),
            "total_realized_cost": float(sum(realized_probe_costs)),
        },
        "forced_noop_fraction": float(forced_noop_count / max(1, option_selection_count)),
        "no_valid_option_fraction": float(no_valid_option_count / max(1, option_selection_count)),
        "diagnostic_status_counts": diagnostic_status_counts,
        "termination_counts": term_counts,
        "option_kind_stats": option_kind_stats,
        "delta_info": _weighted_episode_summary(episodes, "delta_info_mean"),
        "mi": _weighted_episode_summary(episodes, "mi_mean"),
        "diagnostic_cost": _weighted_episode_summary(episodes, "diagnostic_cost_mean"),
        "belief_swap_delta": _aggregate_swap([row["belief_swap"] for row in episodes]),
        "belief_influence": _weighted_belief_influence_summary(episodes),
    }


def _base_q_values(
    ctx: EvalContext,
    obs_tensor: torch.Tensor,
    graph_batch: dict[str, Any],
    belief: torch.Tensor,
) -> torch.Tensor:
    if hasattr(ctx.q_net, "q_net") and hasattr(ctx.q_net.q_net, "q_base_values"):
        encoded = ctx.q_net.encoder(obs_tensor)
        return ctx.q_net.q_net.q_base_values(encoded, graph_batch["option_mask"])
    return ctx.q_net(obs_tensor, belief, **_q_forward_kwargs(graph_batch))


def _resolve_partner_names(
    option_lib: OCV2OptionLibrary,
    selector: str,
    *,
    partner_set: str = "standard7",
) -> list[str]:
    partners = [
        partner.name
        for partner in make_training_partners(option_lib, partner_set=partner_set)
    ]
    if selector == "all":
        return partners
    requested = _parse_csv(selector)
    missing = sorted(set(requested) - set(partners))
    if missing:
        raise KeyError(f"Unknown partners {missing}; choices={partners}")
    return requested


def _sibling_checkpoint(anchor: Path, variant: str) -> Path:
    seed_dir = anchor.parent
    method_dir = seed_dir.parent.parent
    return method_dir / variant / seed_dir.name / "checkpoint.pt"


def _resolve_checkpoint_path(path: Path) -> Path:
    if path.is_dir():
        path = path / "checkpoint.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _summary(
    results: list[dict[str, Any]],
    baselines: dict[str, Any],
    wall: float,
    *,
    evaluation_kind: str,
) -> dict[str, Any]:
    if evaluation_kind != "formal_benchmark":
        raise ValueError("Formal result aggregation rejects data-collection-only artifacts.")
    def _mean_or_zero(vals: list[float]) -> float:
        return float(np.mean(vals)) if vals else 0.0

    closures = _relative_values(results, "reference_gap_closure", "value")
    raw_closures = _relative_values(results, "reference_gap_closure", "raw_value")
    within_run = _relative_values(results, "within_run_relative_return", "value")
    raw_within_run = _relative_values(results, "within_run_relative_return", "raw_value")
    net_within_run = _relative_values(
        results,
        "within_run_relative_net_return",
        "value",
    )
    external_net_closures = _relative_values(
        results,
        "reference_gap_closure_net_return",
        "value",
    )
    result_budgets = {
        row["aggregate"].get("path_c_probe_budget") for row in results
    }
    result_costs = {
        float(row["aggregate"].get("path_c_probe_cost_per_use", 0.0))
        for row in results
    }
    if len(result_budgets) > 1 or len(result_costs) > 1:
        raise ValueError("Formal summary mixes probe budgets or unit costs.")
    summary = {
        "num_results": len(results),
        "partners": sorted(baselines),
        "mean_return": _mean_or_nan(
            [float(row["aggregate"]["mean_return"]) for row in results]
        ),
        "mean_raw_return": _mean_or_nan(
            [float(row["aggregate"]["mean_raw_return"]) for row in results]
        ),
        "mean_net_return": _mean_or_nan(
            [float(row["aggregate"]["mean_net_return"]) for row in results]
        ),
        "mean_probes_used_per_episode": _mean_or_nan(
            [float(row["aggregate"]["mean_probes_used"]) for row in results]
        ),
        "mean_realized_probe_cost_per_episode": _mean_or_nan(
            [float(row["aggregate"]["mean_realized_probe_cost"]) for row in results]
        ),
        "path_c_probe_budget": (
            next(iter(result_budgets)) if result_budgets else None
        ),
        "path_c_probe_cost_per_use": (
            next(iter(result_costs)) if result_costs else 0.0
        ),
        "primary_return_metric": "mean_net_return",
        "mean_completion_rate": _mean_or_nan(
            [float(row["aggregate"]["completion_rate"]) for row in results]
        ),
        "mean_blocking_rate": _mean_or_nan(
            [float(row["aggregate"]["blocking_rate"]) for row in results]
        ),
        "wall_time_sec": float(wall),
    }
    if closures:
        summary["mean_reference_gap_closure"] = _mean_or_nan(closures)
    if raw_closures:
        summary["mean_reference_gap_closure_raw"] = _mean_or_nan(raw_closures)
        summary["num_negative_raw_reference_gaps"] = int(
            sum(1 for value in raw_closures if float(value) < 0.0)
        )
    if within_run:
        summary["mean_within_run_relative_return"] = _mean_or_nan(within_run)
    if raw_within_run:
        summary["mean_within_run_relative_return_raw"] = _mean_or_nan(raw_within_run)
    if net_within_run:
        summary["mean_within_run_relative_net_return"] = _mean_or_nan(net_within_run)
    if external_net_closures:
        summary["mean_reference_gap_closure_net_return"] = _mean_or_nan(
            external_net_closures
        )
    summary["mean_role_match_rate"] = None
    summary["role_match_status"] = "removed_from_formal_main_path_p5"
    summary["mean_ego_correct_completion_rate"] = _mean_or_zero([
        float(r["aggregate"].get("ego_correct_completion_rate", 0.0)) for r in results
    ])
    summary["mean_partner_correct_completion_rate"] = _mean_or_zero([
        float(r["aggregate"].get("partner_correct_completion_rate", 0.0)) for r in results
    ])
    summary["mean_wrong_delivery_rate"] = _mean_or_zero([
        float(r["aggregate"].get("wrong_delivery_rate", 0.0)) for r in results
    ])
    summary["mean_ego_delivery_count"] = _mean_or_zero([
        float(r["aggregate"].get("ego_correct_delivery_count", 0.0)) for r in results
    ])
    summary["mean_partner_delivery_count"] = _mean_or_zero([
        float(r["aggregate"].get("partner_correct_delivery_count", 0.0)) for r in results
    ])
    # LDS-C3: throughput means. DIRECT indexing (fail closed) — these fields are
    # emitted by the same code version, so absence means schema corruption, not
    # zero. A None value (0-episode aggregate) must also crash rather than
    # silently average as 0.0 (LDS-C4 lesson).
    summary["mean_team_correct_delivery_throughput_per_episode"] = _mean_or_nan([
        float(r["aggregate"]["team_correct_delivery_throughput_per_episode"])
        for r in results
    ])
    summary["mean_ego_correct_delivery_throughput_per_episode"] = _mean_or_nan([
        float(r["aggregate"]["ego_correct_delivery_throughput_per_episode"])
        for r in results
    ])
    _serve_shares = [r["aggregate"]["ego_serve_share"] for r in results]
    summary["mean_ego_serve_share"] = _mean_or_nan(
        [float(s) for s in _serve_shares if s is not None]
    )
    return summary


def _collection_only_summary(
    results: list[dict[str, Any]],
    baselines: dict[str, Any],
    wall: float,
) -> dict[str, Any]:
    selected = 0
    opportunities = 0
    mean_raw_returns: list[float] = []
    mean_net_returns: list[float] = []
    realized_cost = 0.0
    for row in results:
        aggregate = row.get("aggregate", {})
        probe = aggregate.get("path_c_probe", {})
        if isinstance(probe, dict):
            selected += int(probe.get("selected", 0) or 0)
            opportunities += int(probe.get("opportunities", 0) or 0)
        mean_raw_returns.append(float(aggregate["mean_raw_return"]))
        mean_net_returns.append(float(aggregate["mean_net_return"]))
        realized_cost += float(aggregate["total_realized_probe_cost"])
    return {
        "evaluation_kind": "data_collection_only",
        "benchmark_return_eligible": False,
        "num_results": len(results),
        "partners": sorted(baselines),
        "probe_selected": selected,
        "probe_opportunities": opportunities,
        "diagnostic_mean_raw_return": _mean_or_nan(mean_raw_returns),
        "diagnostic_mean_net_return": _mean_or_nan(mean_net_returns),
        "total_realized_probe_cost": float(realized_cost),
        "wall_time_sec": float(wall),
    }


def _relative_values(
    results: list[dict[str, Any]],
    key: str,
    value_key: str,
) -> list[float]:
    values = []
    for row in results:
        metric = row["aggregate"].get(key)
        if not isinstance(metric, dict):
            continue
        value = metric.get(value_key)
        if value is not None:
            values.append(float(value))
    return values


def _finite_summary(values: list[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    return {
        "mean": _mean_or_nan(finite),
        "count": len(finite),
        "status": "ok" if finite else "no_values",
    }


def _aggregate_swap(values: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [value for value in values if value.get("status") == "ok"]
    if not ok:
        statuses = sorted({str(value.get("status", "unknown")) for value in values})
        return {"status": "no_valid_swaps", "value": None, "input_statuses": statuses}
    pair_rows = [
        pair
        for value in ok
        for pair in value.get("pairs", [])
        if pair.get("status") == "ok"
    ]
    return {
        "status": "ok",
        "num_option_diagnostics": len(ok),
        "num_pair_rows": len(pair_rows),
        "pairs": pair_rows,
        "mean_abs_maxq_delta": _mean_or_nan(
            [float(value["mean_abs_maxq_delta"]) for value in ok]
        ),
        "mean_abs_q_delta": _mean_or_nan(
            [float(value["mean_abs_q_delta"]) for value in ok]
        ),
        "action_flip_rate": _mean_or_nan(
            [float(value["action_flip_rate"]) for value in ok]
        ),
    }


def _aggregate_belief_influence(values: list[dict[str, float]]) -> dict[str, Any]:
    keys = ("belief_zero_delta", "belief_uniform_delta", "relevance_zero_delta")
    finite_by_key: dict[str, list[float]] = {key: [] for key in keys}
    for row in values:
        for key in keys:
            value = float(row.get(key, 0.0))
            if np.isfinite(value):
                finite_by_key[key].append(value)
    return {
        "mean_belief_zero_delta": _mean_or_nan(finite_by_key["belief_zero_delta"]),
        "mean_belief_uniform_delta": _mean_or_nan(finite_by_key["belief_uniform_delta"]),
        "mean_relevance_zero_delta": _mean_or_nan(finite_by_key["relevance_zero_delta"]),
        "count": max((len(v) for v in finite_by_key.values()), default=0),
        "status": "ok" if any(finite_by_key.values()) else "no_values",
    }


def _weighted_belief_influence_summary(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        ("mean_belief_zero_delta", "belief_zero_delta"),
        ("mean_belief_uniform_delta", "belief_uniform_delta"),
        ("mean_relevance_zero_delta", "relevance_zero_delta"),
    )
    totals = {out_key: 0.0 for out_key, _ in keys}
    count = 0
    for row in episodes:
        influence = row.get("belief_influence", {})
        if not isinstance(influence, dict):
            continue
        row_count = int(row.get("belief_influence_count", row.get("diagnostic_count", 0)))
        if row_count <= 0:
            continue
        row_values: dict[str, float] = {}
        valid = True
        for out_key, _raw_key in keys:
            value = float(influence.get(out_key, float("nan")))
            if not np.isfinite(value):
                valid = False
                break
            row_values[out_key] = value
        if not valid:
            continue
        for out_key in totals:
            totals[out_key] += row_values[out_key] * row_count
        count += row_count
    if count <= 0:
        return {
            "mean_belief_zero_delta": 0.0,
            "mean_belief_uniform_delta": 0.0,
            "mean_relevance_zero_delta": 0.0,
            "count": 0,
            "status": "no_values",
        }
    return {
        "mean_belief_zero_delta": float(totals["mean_belief_zero_delta"] / count),
        "mean_belief_uniform_delta": float(totals["mean_belief_uniform_delta"] / count),
        "mean_relevance_zero_delta": float(totals["mean_relevance_zero_delta"] / count),
        "count": int(count),
        "status": "ok",
    }


def _weighted_episode_summary(
    episodes: list[dict[str, Any]],
    key: str,
) -> dict[str, Any]:
    total = 0.0
    count = 0
    for row in episodes:
        row_count = int(row.get("diagnostic_count", 0))
        value = float(row.get(key, 0.0))
        if row_count <= 0 or not np.isfinite(value):
            continue
        total += value * row_count
        count += row_count
    return {
        "mean": float(total / count) if count else float("nan"),
        "count": count,
        "status": "ok" if count else "no_values",
    }


def _mean_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _empty_delivery_counts() -> dict[str, int]:
    return {
        "delivery_event": 0,
        "ego_delivery_event": 0,
        "partner_delivery_event": 0,
        "correct_delivery": 0,
        "wrong_delivery_event": 0,
        "ego_correct_delivery": 0,
        "ego_sole_correct_delivery": 0,
        "partner_correct_delivery": 0,
        "ego_wrong_delivery_event": 0,
        "partner_wrong_delivery_event": 0,
    }


def _accumulate_delivery_counts(counts: dict[str, int], event: Any) -> None:
    for key in counts:
        counts[key] = int(counts.get(key, 0)) + int(bool(getattr(event, key, False)))


def _merge_counts(target: dict[str, int], source: dict[str, Any]) -> None:
    for key, value in source.items():
        target[str(key)] = int(target.get(str(key), 0)) + int(value)


def _record_selection_attempt(selection_stats: dict[str, int] | None) -> None:
    if selection_stats is None:
        return
    selection_stats["option_selection_count"] = int(
        selection_stats.get("option_selection_count", 0)
    ) + 1


def _record_forced_noop(selection_stats: dict[str, int] | None) -> None:
    if selection_stats is None:
        return
    selection_stats["no_valid_option_count"] = int(
        selection_stats.get("no_valid_option_count", 0)
    ) + 1
    selection_stats["forced_noop_count"] = int(
        selection_stats.get("forced_noop_count", 0)
    ) + 1


def _enforce_reward_scale(
    reward_scale_status: dict[str, dict[str, Any]],
    *,
    allow_unverified: bool,
) -> None:
    """LDS-B2: hard-fail formal eval on unverified reward scale.

    Previously the per-variant ``reward_scale_verified`` booleans were only
    recorded in the output JSON, so a checkpoint evaluated against a stale or
    objective-mismatched CE graph still produced a formal-looking artifact.
    ``--allow_unverified_reward_scale`` is the smoke/debug escape hatch: it
    downgrades the failure to a stderr warning while the JSON keeps recording
    the false status for triage.
    """
    failures = {
        variant: status
        for variant, status in reward_scale_status.items()
        if not bool(status.get("reward_scale_verified"))
    }
    if not failures:
        return
    if allow_unverified:
        print(
            "[evaluate_aris] WARNING: reward-scale verification FAILED for "
            f"variants {sorted(failures)}; proceeding only because "
            "--allow_unverified_reward_scale was passed (smoke/debug use only).",
            file=sys.stderr,
        )
        return
    raise RuntimeError(
        "Formal eval refused: reward-scale verification failed for variants "
        f"{sorted(failures)}: {failures!r}. Regenerate the CE graph with the "
        "current config (run_ce_pipeline) or, for smoke/debug only, pass "
        "--allow_unverified_reward_scale."
    )


def _validate_eval_integrity(
    aggregate: dict[str, Any],
    *,
    collect_diagnostics: bool,
    allow_diag_skip: bool,
    expected_policy: str = "behavior_inferred_v1",
) -> None:
    if int(aggregate.get("forced_noop_count", 0)) > 0:
        raise RuntimeError(
            "Formal eval encountered no-valid-option forced noop events: "
            f"{aggregate['forced_noop_count']}. This hard integrity check is not "
            "bypassed by --allow_diag_skip."
        )
    evidence = aggregate.get("partner_option_evidence", {})
    # E2 (METHOD_LOCK sec18.8) + LDS-B3 hardening: the evidence policy must match
    # EXACTLY the mode this run declared (config-derived; the zeroed CLI overlay
    # sets it). A whitelist would admit an UNDECLARED zeroed run — now an
    # undeclared-zeroed and a declared-zeroed-but-ran-inferred both hard-fail.
    # The oracle_source/observed_dist/missing hard checks below still run
    # unconditionally, so the zeroed mode never weakens the real-path guarantees.
    if str(evidence.get("evidence_policy")) != str(expected_policy):
        raise RuntimeError(
            "Formal eval evidence policy mismatch: declared/expected "
            f"{expected_policy!r} but evidence carries "
            f"{evidence.get('evidence_policy')!r} ({evidence!r})"
        )
    if int(evidence.get("observed_dist_count", 0)) > 0:
        raise RuntimeError(
            "Formal eval observed non-behavior partner-option distributions: "
            f"{evidence['observed_dist_count']} events."
        )
    if int(evidence.get("missing_count", 0)) > 0:
        raise RuntimeError(
            "Formal eval observed missing partner-option evidence: "
            f"{evidence['missing_count']} events."
        )
    if int(evidence.get("oracle_source_count", 0)) > 0:
        raise RuntimeError(
            "Formal eval observed oracle partner-option evidence source: "
            f"{evidence['oracle_source_count']} events."
        )
    if collect_diagnostics:
        bad = {
            key: value
            for key, value in (
                ("delta_info", aggregate.get("delta_info", {})),
                ("mi", aggregate.get("mi", {})),
                ("diagnostic_cost", aggregate.get("diagnostic_cost", {})),
            )
            if isinstance(value, dict) and value.get("status") not in {"ok", "unsupported_method"}
        }
        if bad and not allow_diag_skip:
            raise RuntimeError(
                "Formal eval diagnostics produced no valid values: "
                f"{json.dumps(_jsonable(bad), sort_keys=True)}. "
                "Pass --allow_diag_skip only for smoke runs."
            )


def _factor_deletion_episode_count(args: argparse.Namespace, fast: bool) -> int:
    requested = getattr(args, "factor_deletion_episodes", None)
    if requested is not None:
        return int(requested)
    return 0 if fast else 3


def _eval_provenance(ctx: EvalContext) -> dict[str, Any]:
    return runtime_provenance(
        config=ctx.config,
        layout_graph=ctx.layout_graph,
        option_lib=ctx.option_lib,
        partners=_select_train_partners(ctx.option_lib, ctx.config),
        ce_path=ctx.config.get("graph", {}).get("ce_path"),
        replay_path=ctx.config.get("graph", {}).get("replay_path"),
        graph_path=ctx.config.get("graph", {}).get("graph_path"),
    )


def _noop_option_id(option_lib: OCV2OptionLibrary) -> int:
    for opt in option_lib.options:
        if opt.kind == "noop":
            return int(opt.id)
    raise ValueError("The registered option library has no noop fallback option.")


def _increment(counts: dict[str, int], key: str) -> None:
    counts[key] = int(counts.get(key, 0)) + 1


def _update_option_kind_stats(
    stats: dict[str, dict[str, Any]],
    option_kind: str,
    termination_reason: str,
) -> None:
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


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


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
    parser = argparse.ArgumentParser(description="Evaluate OvercookedV2 ARIS checkpoints.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--path-c-active-probe-collection",
        action="store_true",
        help=(
            "Enable Path C active probes during this eval run for data collection. "
            "Formal benchmark evaluation keeps probes disabled by default."
        ),
    )
    parser.add_argument("--graph_variants", required=True)
    parser.add_argument("--partners", default="all")
    parser.add_argument(
        "--partner_set",
        default=None,
        help="sec18.14: eval-only partner-registry override (e.g. blind_v1). "
        "Redirects partner lookup only; the checkpoint config is not mutated. "
        "Declared in the output as eval_partner_set_override.",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_episode_options", type=int, default=0)
    parser.add_argument("--factor_deletion_episodes", type=int, default=None)
    parser.add_argument(
        "--allow_diag_skip",
        action="store_true",
        help="Smoke/debug escape hatch: record diagnostic skips instead of failing formal eval.",
    )
    parser.add_argument(
        "--allow_unverified_reward_scale",
        action="store_true",
        help="Smoke/debug ONLY (LDS-B2): downgrade the hard reward-scale "
        "verification gate to a warning. Formal eval must never pass this.",
    )
    parser.add_argument(
        "--zeroed_partner_option_ablation",
        action="store_true",
        help="E2 zeroed-channel ablation (LDS-B3): overlay evidence mode 'zeroed' "
        "on the loaded checkpoint config (eval-only; checkpoint untouched) and "
        "declare it in the output. The integrity gate requires the evidence "
        "policy to match this declaration exactly.",
    )
    parser.add_argument(
        "--random_policy_only",
        action="store_true",
        help="Use the checkpoint only to load env/graph context and evaluate random valid options.",
    )
    parser.add_argument(
        "--active_probe_collection",
        action="store_true",
        help="Data-collection mode only: allow Path C active probes during this evaluation run. "
        "Formal benchmark return eval leaves active probes disabled even when the checkpoint config "
        "contains path_c.probe.enable=true.",
    )
    parser.add_argument(
        "--path_c_acting_probe_budget",
        type=int,
        default=None,
        help=(
            "Formal Path C acting budget. The value must belong to the frozen "
            "probe-budget grid; unlike exploratory collection, this run remains "
            "benchmark-return eligible."
        ),
    )
    parser.add_argument(
        "--path_c_probe_cost_per_use",
        type=float,
        default=None,
        help=(
            "Optional assertion of the frozen per-probe cost. A mismatch is rejected."
        ),
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Gate-only mode: skip per-option diagnostics and factor q-proxy "
        "(raw and net return metrics are unaffected). Used for fast verification matrices.",
    )
    parser.add_argument(
        "--path_c_probe_base_checkpoint",
        default=None,
        help=(
            "Optional base_only checkpoint for secondary residual visualization; "
            "normalized-advantage probe selection does not use it."
        ),
    )
    parser.add_argument("--reference_base_checkpoint", default=None)
    parser.add_argument("--reference_ref_checkpoint", default=None)
    parser.add_argument(
        "--baseline_cache_dir",
        default=None,
        help="Directory to cache checkpoint-independent reference/random baseline "
        "rollouts across eval invocations (E1 speedup). Defaults to "
        "<output_dir>/.baseline_cache. Pass 'none' to disable.",
    )
    return parser


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
