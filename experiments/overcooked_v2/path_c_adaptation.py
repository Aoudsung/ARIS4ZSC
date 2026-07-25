"""WP-C (§6): Path C v3 adaptation with critic-only trunk gradients."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from experiments.overcooked_v2.batched_rollout import BatchedEnvPool
from experiments.overcooked_v2.path_c_backbone_ppo import (
    RecurrentIPPOBackbone,
    sample_actor_actions,
)
from experiments.overcooked_v2.path_c_seed import (
    derive_ocv2_execution_seed,
    validate_unique_execution_seed_mapping,
)
from experiments.overcooked_v2.path_c_response_probe import (
    PartnerConditionedValueEnsemble,
    REGISTERED_LOCAL_RESPONSE_SPEC,
    ResponseModelEnsemble,
    calibrate_response_disagreement_threshold,
    episode_bootstrap_mask,
    initial_partner_belief,
    local_non_agent_change_from_default_observation,
    partner_visible_from_default_observation,
    response_tokens_from_observations,
    select_finite_prototype_two_action_surrogate,
    select_response_probe,
    update_partner_belief,
    validate_response_probe_config,
)
from experiments.overcooked_v2.path_c_sequence import sequence_td_core
from experiments.overcooked_v2.path_c_standard import (
    LocalObservationEncoder,
    PrimitiveObservationBatch,
    StandardEnvConfig,
    choose_primitive_actions,
    load_standard_checkpoint,
    save_standard_checkpoint,
    shared_team_reward,
    single_step_batch,
)
from experiments.overcooked_v2.path_c_standard_diagnostics import (
    decompose_raw_reward_events,
)
from experiments.overcooked_v2.path_c_standard_training import (
    derive_standard_seed,
    shaped_rewards_by_agent,
)


ADAPTATION_CONFIG_SCHEMA_VERSION = "path_c_adaptation_v1"


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _calibration_implementation_sha256() -> str:
    """Bind calibration to the scorer and the ledger-producing trainer."""

    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(__file__).with_name("path_c_response_probe.py"),
    ):
        content = path.read_bytes()
        digest.update(path.name.encode("utf-8"))
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _resolved_artifact_path(value: Any, *, relative_to: Path) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def _probe_calibration_contract_sha256(
    controller: str,
    probe_config: Mapping[str, Any],
    quantile: float,
) -> str:
    """Hash calibration semantics while excluding its placeholder threshold."""

    fields = (
        "response_model_mode",
        "response_ensemble_size",
        "response_hidden_dim",
        "partner_value_hidden_dim",
        "belief_probability_floor",
        "max_probe_task_cost",
        "max_probe_regret",
        "probe_budget_per_episode",
        "probe_window_environment_steps",
        "response_route",
    )
    payload = {
        "schema_version": "path_c_probe_calibration_contract_v1",
        "controller": str(controller),
        "quantile": float(quantile),
        "probe": {field: probe_config.get(field) for field in fields},
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_probe_calibration_artifacts(
    summary_path: str | Path,
    *,
    controller: str,
    probe_config: Mapping[str, Any],
    expected_partner_ids: Sequence[str],
    expected_response_checkpoint: str | Path,
    expected_backbone_checkpoint: str | Path,
    expected_pool_report: str | Path,
    expected_environment_config_sha256: str,
    expected_quantile: float,
    expected_summary_sha256: str | None,
) -> float:
    """Recompute the frozen threshold from the content-addressed row ledger."""

    summary_file = Path(summary_path).resolve()
    summary_bytes = summary_file.read_bytes()
    summary_sha256 = hashlib.sha256(summary_bytes).hexdigest()
    if expected_summary_sha256 is not None and summary_sha256 != expected_summary_sha256:
        raise ValueError("Probe calibration summary differs from its formal binding.")
    calibration = json.loads(summary_bytes.decode("utf-8"))
    if not isinstance(calibration, Mapping) or calibration.get(
        "schema_version"
    ) != "path_c_probe_calibration_summary_v1":
        raise ValueError("Unsupported response-probe calibration summary.")
    if calibration.get("scientific_readout_allowed") is not False:
        raise ValueError("Probe calibration must remain a non-scientific artifact.")
    if calibration.get("controller") != controller:
        raise ValueError("Probe calibration used a different controller.")
    quantile = float(expected_quantile)
    if not math.isfinite(quantile) or not 0.0 < quantile < 1.0:
        raise ValueError("The expected calibration quantile must lie in (0,1).")
    if not math.isclose(
        float(calibration.get("calibration_quantile", math.nan)),
        quantile,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("Probe calibration used a different frozen quantile.")

    rows_path = _resolved_artifact_path(
        calibration.get("calibration_rows_path", ""),
        relative_to=summary_file.parent,
    )
    rows_bytes = rows_path.read_bytes()
    rows_sha256 = hashlib.sha256(rows_bytes).hexdigest()
    if calibration.get("calibration_rows_sha256") != rows_sha256:
        raise ValueError("Probe calibration row ledger content hash changed.")
    raw_lines = rows_bytes.decode("utf-8").splitlines()
    if len(raw_lines) != 50_000 or int(
        calibration.get("calibration_rows_line_count", -1)
    ) != len(raw_lines):
        raise ValueError("Probe calibration must contain 50,000 decision rows.")

    expected_row_fields = {
        "schema_version",
        "episode_index",
        "ego_slot",
        "partner_hypothesis_id",
        "episode_step",
        "candidate_score",
        "candidate_regret",
        "candidate_equals_greedy",
        "candidate_equals_baseline",
        "disagreement_pass",
        "regret_pass",
        "budget_and_window_pass",
        "probe_executed",
    }
    episode_rows: dict[int, dict[str, Any]] = {}
    scores: list[float] = []
    regrets: list[float] = []
    equals_greedy: list[bool] = []
    equals_baseline: list[bool] = []
    regret_limit = float(
        probe_config["max_probe_task_cost"]
        if controller == "finite_prototype_two_action_surrogate"
        else probe_config["max_probe_regret"]
    )
    for line in raw_lines:
        row = json.loads(line)
        if not isinstance(row, Mapping) or set(row) != expected_row_fields:
            raise ValueError("Probe calibration row schema changed.")
        if row.get("schema_version") != "path_c_probe_calibration_row_v1":
            raise ValueError("Probe calibration row version changed.")
        episode_index = int(row["episode_index"])
        ego_slot = int(row["ego_slot"])
        step = int(row["episode_step"])
        partner_id = str(row["partner_hypothesis_id"])
        score = float(row["candidate_score"])
        regret = float(row["candidate_regret"])
        boolean_fields = (
            "candidate_equals_greedy",
            "candidate_equals_baseline",
            "disagreement_pass",
            "regret_pass",
            "budget_and_window_pass",
            "probe_executed",
        )
        if (
            episode_index < 0
            or ego_slot not in (0, 1)
            or not 0 <= step < 100
            or partner_id not in set(expected_partner_ids)
            or not math.isfinite(score)
            or not math.isfinite(regret)
            or any(not isinstance(row[name], bool) for name in boolean_fields)
            or row["probe_executed"] is not False
            or row["budget_and_window_pass"] is not True
            or row["regret_pass"] != (regret <= regret_limit)
        ):
            raise ValueError("Probe calibration row violates the frozen no-probe design.")
        episode = episode_rows.setdefault(
            episode_index,
            {"ego_slot": ego_slot, "partner_id": partner_id, "steps": set()},
        )
        if episode["ego_slot"] != ego_slot or episode["partner_id"] != partner_id:
            raise ValueError("Probe calibration changes role or partner within an episode.")
        if step in episode["steps"]:
            raise ValueError("Probe calibration repeats an episode decision point.")
        episode["steps"].add(step)
        scores.append(score)
        regrets.append(regret)
        equals_greedy.append(bool(row["candidate_equals_greedy"]))
        equals_baseline.append(bool(row["candidate_equals_baseline"]))
    if len(episode_rows) != 500 or any(
        episode["steps"] != set(range(100)) for episode in episode_rows.values()
    ):
        raise ValueError("Probe calibration must cover 500 complete 100-step windows.")
    role_counts = {
        f"slot_{slot}": sum(
            episode["ego_slot"] == slot for episode in episode_rows.values()
        )
        for slot in (0, 1)
    }
    partner_counts = {
        partner_id: sum(
            episode["partner_id"] == partner_id for episode in episode_rows.values()
        )
        for partner_id in expected_partner_ids
    }
    if role_counts != {"slot_0": 250, "slot_1": 250} or calibration.get(
        "role_counts"
    ) != role_counts:
        raise ValueError("Probe calibration did not balance ego positions.")
    if set(partner_counts.values()) != {500 // len(partner_counts)} or calibration.get(
        "partner_counts"
    ) != partner_counts:
        raise ValueError("Probe calibration did not balance the admitted partners.")

    threshold, safe_count = calibrate_response_disagreement_threshold(
        scores,
        regrets,
        equals_greedy,
        max_probe_regret=regret_limit,
        quantile=quantile,
        candidate_equals_baseline=equals_baseline,
    )
    summary_field = (
        "calibrated_surrogate_score_threshold"
        if controller == "finite_prototype_two_action_surrogate"
        else "calibrated_response_disagreement_threshold"
    )
    if (
        int(calibration.get("episode_count", -1)) != 500
        or int(calibration.get("decision_count", -1)) != len(raw_lines)
        or int(calibration.get("probe_actions_executed", -1)) != 0
        or int(calibration.get("safe_intervention_candidate_count", -1))
        != safe_count
        or not math.isclose(
            float(calibration.get(summary_field, math.nan)),
            threshold,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError("Probe calibration summary does not match its row ledger.")

    expected_bindings = {
        "calibration_implementation_sha256": _calibration_implementation_sha256(),
        "calibration_contract_sha256": _probe_calibration_contract_sha256(
            controller,
            probe_config,
            quantile,
        ),
        "response_initialization_checkpoint_sha256": _file_sha256(
            expected_response_checkpoint
        ),
        "backbone_checkpoint_sha256": _file_sha256(expected_backbone_checkpoint),
        "partner_pool_admission_report_sha256": _file_sha256(expected_pool_report),
        "environment_config_sha256": expected_environment_config_sha256,
    }
    if any(calibration.get(field) != value for field, value in expected_bindings.items()):
        raise ValueError("Probe calibration used different frozen artifacts or code.")
    if _resolved_artifact_path(
        calibration.get("response_initialization_checkpoint", ""),
        relative_to=summary_file.parent,
    ) != Path(expected_response_checkpoint).resolve():
        raise ValueError("Probe calibration used a different response initialization.")
    return threshold


@dataclass(frozen=True)
class AdaptationState:
    hidden: torch.Tensor


@dataclass(frozen=True)
class AdaptationActionDecision:
    actions: np.ndarray
    is_probe: np.ndarray
    old_log_prob: torch.Tensor
    candidate_score: np.ndarray
    candidate_regret: np.ndarray
    candidate_equals_greedy: np.ndarray
    candidate_equals_baseline: np.ndarray
    disagreement_pass: np.ndarray
    regret_pass: np.ndarray
    budget_pass: np.ndarray
    candidate_evaluated: np.ndarray
    actor_entropy: np.ndarray
    actual_response_probabilities: torch.Tensor | None
    actual_continuation_values: torch.Tensor | None


def _finite_quantiles(values: Sequence[float]) -> dict[str, float | None]:
    array = np.asarray(tuple(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            key: None for key in ("min", "p10", "p50", "p90", "p99", "max")
        }
    return {
        "min": float(array.min()),
        "p10": float(np.quantile(array, 0.10)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


class PathCAdaptationPolicy(nn.Module):
    """Actor, critic, response, and partner-value heads over an observable trunk."""

    def __init__(
        self,
        observation_shape: Sequence[int],
        *,
        n_actions: int = 6,
        recurrent_dim: int = 128,
        visual_embedding_dim: int = 128,
        conv_channels: Sequence[int] = (32, 32, 16),
        encoder_spatial_mode: str = "flatten",
        n_critic_heads: int = 5,
        partner_action_channel: bool = True,
        response_ensemble_size: int = 5,
        response_hidden_dim: int = 64,
        partner_value_hidden_dim: int = 64,
        response_vocabulary_size: int = 7,
        response_model_mode: str = "bootstrap_legacy",
    ) -> None:
        super().__init__()
        self.observation_shape = tuple(int(item) for item in observation_shape)
        self.n_actions = int(n_actions)
        self.recurrent_dim = int(recurrent_dim)
        self.visual_embedding_dim = int(visual_embedding_dim)
        self.conv_channels = tuple(int(item) for item in conv_channels)
        self.encoder_spatial_mode = str(encoder_spatial_mode)
        self.n_heads = int(n_critic_heads)
        self.partner_action_channel = bool(partner_action_channel)
        if (
            not self.partner_action_channel
            and int(response_vocabulary_size) != REGISTERED_LOCAL_RESPONSE_SPEC.q
        ):
            raise ValueError(
                "Local-only adaptation must use the registered response vocabulary."
            )
        self.response_vocabulary_sha256 = (
            None
            if self.partner_action_channel
            else REGISTERED_LOCAL_RESPONSE_SPEC.sha256
        )
        self.response_model_mode = str(response_model_mode)
        if self.response_model_mode not in {"bootstrap_legacy", "partner_conditioned"}:
            raise ValueError("Unknown response model mode.")
        self.input_contract = (
            "action_augmented_v3" if self.partner_action_channel else "local_only_v3"
        )
        self.observation_encoder = LocalObservationEncoder(
            self.observation_shape[-1],
            self.visual_embedding_dim,
            observation_height=self.observation_shape[0],
            observation_width=self.observation_shape[1],
            spatial_mode=self.encoder_spatial_mode,
            conv_channels=self.conv_channels,
        )
        evidence_dim = self.visual_embedding_dim + self.n_actions + 1 + 2
        if self.partner_action_channel:
            evidence_dim += self.n_actions + 1
        self.recurrent = nn.GRU(evidence_dim, self.recurrent_dim, batch_first=True)
        self.actor_head = nn.Linear(self.recurrent_dim, self.n_actions)
        self.value_heads = nn.ModuleList(
            nn.Linear(self.recurrent_dim, 1) for _ in range(self.n_heads)
        )
        self.advantage_heads = nn.ModuleList(
            nn.Linear(self.recurrent_dim, self.n_actions) for _ in range(self.n_heads)
        )
        self.response_ensemble = ResponseModelEnsemble(
            self.recurrent_dim,
            n_actions=self.n_actions,
            vocabulary_size=int(response_vocabulary_size),
            hidden_dim=int(response_hidden_dim),
            ensemble_size=int(response_ensemble_size),
        )
        self.partner_value_ensemble = PartnerConditionedValueEnsemble(
            self.recurrent_dim,
            n_actions=self.n_actions,
            hidden_dim=int(partner_value_hidden_dim),
            ensemble_size=int(response_ensemble_size),
        )

    @property
    def trunk_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.observation_encoder.parameters()) + tuple(self.recurrent.parameters())

    @property
    def critic_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.value_heads.parameters()) + tuple(self.advantage_heads.parameters())

    @property
    def partner_value_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.partner_value_ensemble.parameters())

    def initial_state(self, batch_size: int, *, device: torch.device | str) -> AdaptationState:
        return AdaptationState(
            hidden=torch.zeros(1, int(batch_size), self.recurrent_dim, device=device)
        )

    def _features(
        self,
        batch: PrimitiveObservationBatch,
        partner_previous_actions: torch.Tensor | None,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch.validate(observation_shape=self.observation_shape, n_actions=self.n_actions)
        visual = self.observation_encoder(batch.observations)
        previous_action = F.one_hot(
            batch.previous_actions, num_classes=self.n_actions + 1
        ).to(visual.dtype)
        evidence = [
            visual,
            previous_action,
            batch.previous_rewards.unsqueeze(-1).to(visual.dtype),
            batch.episode_starts.unsqueeze(-1).to(visual.dtype),
        ]
        if self.partner_action_channel:
            if partner_previous_actions is None or partner_previous_actions.shape != batch.previous_actions.shape:
                raise ValueError("The augmented input contract requires partner previous actions.")
            evidence.append(
                F.one_hot(
                    partner_previous_actions, num_classes=self.n_actions + 1
                ).to(visual.dtype)
            )
        elif partner_previous_actions is not None:
            raise ValueError("The local-only contract forbids a partner action channel.")
        features: list[torch.Tensor] = []
        current_hidden = hidden
        joined = torch.cat(evidence, dim=-1)
        for step in range(joined.shape[1]):
            reset = batch.episode_starts[:, step].reshape(1, -1, 1)
            current_hidden = torch.where(reset, torch.zeros_like(current_hidden), current_hidden)
            output, current_hidden = self.recurrent(joined[:, step : step + 1], current_hidden)
            features.append(output[:, 0])
        return torch.stack(features, dim=1), current_hidden

    def critic_q(self, features: torch.Tensor) -> torch.Tensor:
        values = torch.stack([head(features) for head in self.value_heads], dim=-2)
        advantages = torch.stack([head(features) for head in self.advantage_heads], dim=-2)
        return values + advantages - advantages.mean(dim=-1, keepdim=True)

    def forward_sequence(
        self,
        batch: PrimitiveObservationBatch,
        *,
        partner_previous_actions: torch.Tensor | None = None,
        initial_state: AdaptationState | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = initial_state or self.initial_state(
            batch.observations.shape[0], device=batch.observations.device
        )
        features, _ = self._features(batch, partner_previous_actions, state.hidden)
        return self.critic_q(features), self.actor_head(features.detach()), features

    def forward_step(
        self,
        batch: PrimitiveObservationBatch,
        state: AdaptationState,
        *,
        partner_previous_actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, AdaptationState]:
        if batch.observations.shape[1] != 1:
            raise ValueError("forward_step requires one time step.")
        features, hidden = self._features(batch, partner_previous_actions, state.hidden)
        return self.critic_q(features[:, 0]), features[:, 0], AdaptationState(hidden)

    def actor_logits(self, features: torch.Tensor) -> torch.Tensor:
        """The explicit detach is the method-contract gradient boundary."""

        return self.actor_head(features.detach())

    def architecture_manifest(self) -> dict[str, Any]:
        return {
            "model_class": type(self).__name__,
            "observation_shape": list(self.observation_shape),
            "n_actions": self.n_actions,
            "recurrent_dim": self.recurrent_dim,
            "visual_embedding_dim": self.visual_embedding_dim,
            "conv_channels": list(self.conv_channels),
            "encoder_spatial_mode": self.encoder_spatial_mode,
            "n_critic_heads": self.n_heads,
            "partner_action_channel": self.partner_action_channel,
            "response_ensemble_size": self.response_ensemble.ensemble_size,
            "response_hidden_dim": self.response_ensemble.models[0].hidden_dim,
            "partner_value_hidden_dim": self.partner_value_ensemble.hidden_dim,
            "response_vocabulary_size": self.response_ensemble.vocabulary_size,
            "response_vocabulary_sha256": self.response_vocabulary_sha256,
            "response_model_mode": self.response_model_mode,
            "input_contract": self.input_contract,
        }

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> "PathCAdaptationPolicy":
        if manifest.get("model_class") != cls.__name__:
            raise ValueError("Manifest does not describe a Path C adaptation policy.")
        return cls(
            manifest["observation_shape"],
            n_actions=int(manifest["n_actions"]),
            recurrent_dim=int(manifest["recurrent_dim"]),
            visual_embedding_dim=int(manifest["visual_embedding_dim"]),
            conv_channels=manifest["conv_channels"],
            encoder_spatial_mode=str(manifest["encoder_spatial_mode"]),
            n_critic_heads=int(manifest["n_critic_heads"]),
            partner_action_channel=bool(manifest["partner_action_channel"]),
            response_ensemble_size=int(manifest["response_ensemble_size"]),
            response_hidden_dim=int(manifest["response_hidden_dim"]),
            partner_value_hidden_dim=int(
                manifest.get("partner_value_hidden_dim", manifest["response_hidden_dim"])
            ),
            response_vocabulary_size=int(manifest["response_vocabulary_size"]),
            response_model_mode=str(
                manifest.get("response_model_mode", "bootstrap_legacy")
            ),
        )

    @classmethod
    def from_backbone(
        cls,
        backbone: RecurrentIPPOBackbone,
        **kwargs: Any,
    ) -> "PathCAdaptationPolicy":
        model = cls(
            backbone.observation_shape,
            recurrent_dim=backbone.recurrent_dim,
            visual_embedding_dim=backbone.visual_embedding_dim,
            conv_channels=backbone.conv_channels,
            encoder_spatial_mode=backbone.encoder_spatial_mode,
            **kwargs,
        )
        model.observation_encoder.load_state_dict(backbone.observation_encoder.state_dict())
        with torch.no_grad():
            model.recurrent.weight_hh_l0.copy_(backbone.recurrent.weight_hh_l0)
            model.recurrent.bias_hh_l0.copy_(backbone.recurrent.bias_hh_l0)
            model.recurrent.bias_ih_l0.copy_(backbone.recurrent.bias_ih_l0)
            model.recurrent.weight_ih_l0.zero_()
            old_width = backbone.recurrent.weight_ih_l0.shape[1]
            model.recurrent.weight_ih_l0[:, :old_width].copy_(
                backbone.recurrent.weight_ih_l0
            )
            model.actor_head.load_state_dict(backbone.actor_head.state_dict())
            for value_head in model.value_heads:
                value_head.load_state_dict(backbone.critic_head.state_dict())
            for advantage_head in model.advantage_heads:
                advantage_head.weight.zero_()
                advantage_head.bias.zero_()
        return model

    def prior_state_sha256(self) -> None:
        return None

    def compact_recurrent_parameters(self) -> None:
        self.recurrent.flatten_parameters()
        for model in self.response_ensemble.models:
            model.recurrent.flatten_parameters()


@dataclass(frozen=True)
class AdaptationTrainingSpec:
    run_kind: str
    scientific_readout_allowed: bool
    training_mode: str
    seed: int
    total_environment_steps: int
    batch_size_envs: int
    rollout_steps: int
    metrics_interval_environment_steps: int
    target_update_environment_steps: int
    actor_update_epochs: int
    shaping_horizon_environment_steps: int
    stop_if_no_delivery_by_environment_steps: int
    trunk_learning_rate: float
    actor_learning_rate: float
    response_learning_rate: float
    entropy_coefficient: float
    gamma: float
    gradient_clip_norm: float
    ppo_clip: float
    output_dir: Path

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "AdaptationTrainingSpec":
        training = payload.get("training")
        if not isinstance(training, Mapping):
            raise TypeError("Adaptation config requires a training mapping.")
        run_kind = str(payload.get("run_kind"))
        scientific = payload.get("scientific_readout_allowed")
        if run_kind not in {"formal", "diagnostic"} or not isinstance(scientific, bool):
            raise ValueError(
                "Adaptation configs must explicitly label run_kind and "
                "scientific_readout_allowed."
            )
        if scientific != (run_kind == "formal"):
            raise ValueError("Only explicitly formal adaptation runs may allow a scientific readout.")
        training_mode = str(payload.get("training_mode", "joint"))
        if training_mode not in {"joint", "response_only", "probe_calibration"}:
            raise ValueError(
                "training_mode must be joint, response_only, or probe_calibration."
            )
        if run_kind == "formal" and training_mode != "joint":
            raise ValueError("Formal adaptation must use joint training.")
        total = int(training["total_environment_steps"])
        horizon = int(training["shaping_horizon_environment_steps"])
        stop = int(training["stop_if_no_delivery_by_environment_steps"])
        if run_kind == "formal" and (horizon, stop) != (5_000_000, 5_000_000):
            raise ValueError("Formal adaptation shaping and stop boundaries must both be 5M.")
        if run_kind == "formal" and total != 10_000_000:
            raise ValueError("A formal adaptation run must use the pre-registered 10M budget.")
        if training_mode == "probe_calibration" and total != 200_000:
            raise ValueError("Probe calibration requires 500 complete 400-step episodes.")
        batch_size = int(training["batch_size_envs"])
        rollout_steps = int(training["rollout_steps"])
        if min(total, batch_size, rollout_steps) <= 0 or total % (batch_size * rollout_steps):
            raise ValueError("The adaptation budget must contain complete vector rollouts.")
        if rollout_steps != 400:
            raise ValueError("Adaptation uses complete 400-step episodes as recurrent rollouts.")
        metrics_interval = int(training["metrics_interval_environment_steps"])
        target_interval = int(training["target_update_environment_steps"])
        for name, interval in (
            ("metrics interval", metrics_interval),
            ("target-update interval", target_interval),
        ):
            if interval <= 0 or interval % (batch_size * rollout_steps):
                raise ValueError(f"The {name} must align with complete vector rollouts.")
        actor_update_epochs = int(training["actor_update_epochs"])
        if actor_update_epochs <= 0:
            raise ValueError("actor_update_epochs must be positive.")
        rates = {
            name: float(training[name])
            for name in ("trunk_learning_rate", "actor_learning_rate", "response_learning_rate")
        }
        if any(not math.isfinite(value) or value <= 0.0 for value in rates.values()):
            raise ValueError("All adaptation learning-rate slots must be positive.")
        return cls(
            run_kind=run_kind,
            scientific_readout_allowed=scientific,
            training_mode=training_mode,
            seed=int(payload["seed"]),
            total_environment_steps=total,
            batch_size_envs=batch_size,
            rollout_steps=rollout_steps,
            metrics_interval_environment_steps=metrics_interval,
            target_update_environment_steps=target_interval,
            actor_update_epochs=actor_update_epochs,
            shaping_horizon_environment_steps=horizon,
            stop_if_no_delivery_by_environment_steps=stop,
            trunk_learning_rate=rates["trunk_learning_rate"],
            actor_learning_rate=rates["actor_learning_rate"],
            response_learning_rate=rates["response_learning_rate"],
            entropy_coefficient=float(training["entropy_coefficient"]),
            gamma=float(training["gamma"]),
            gradient_clip_norm=float(training["gradient_clip_norm"]),
            ppo_clip=float(training["ppo_clip"]),
            output_dir=Path(payload["output_dir"]),
        )


def require_admitted_pool(report_path: str | Path) -> dict[str, Any]:
    """Read the admission artifact and reject missing, malformed, or failed pools."""

    path = Path(report_path)
    if not path.is_file():
        raise FileNotFoundError("The partner admission report is required.")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema_version") not in {
        "path_c_pool_admission_report_v1",
        "path_c_pool_admission_report_v2",
    }:
        raise ValueError("Unsupported partner admission report.")
    if report.get("scientific_readout_allowed") is not False:
        raise ValueError("Admission reports are non-scientific wiring artifacts.")
    if report.get("admitted") is not True:
        raise ValueError("The partner pool did not pass its ability admission gate.")
    if report.get("policy_action_selection") != "stochastic":
        raise ValueError("The admitted PPO partner pool must use stochastic actions.")
    members = report.get("members")
    if not isinstance(members, Mapping) or not members or not all(
        isinstance(member, Mapping) and member.get("admitted") is True
        for member in members.values()
    ):
        raise ValueError("Every partner pool member must be explicitly admitted.")
    training_seeds = {int(member["training_seed"]) for member in members.values()}
    if len(training_seeds) < 2:
        raise ValueError(
            "Partner-conditioned adaptation requires independently trained partners."
        )
    return report


class PathCAdaptationTrainer:
    """Own the v3 model, optimizers, and loss boundaries used by the rollout driver."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.spec = AdaptationTrainingSpec.from_mapping(config)
        self.env_config = StandardEnvConfig.from_mapping(config["environment"])
        self.device = torch.device(str(config.get("device", "cuda")))
        self.pool_report = require_admitted_pool(config["partner_pool_admission_report"])
        backbone, metadata = load_standard_checkpoint(
            config["backbone_checkpoint"], device=self.device
        )
        if not isinstance(backbone, RecurrentIPPOBackbone):
            raise ValueError("Adaptation must initialize from a backbone checkpoint.")
        if metadata.get("phase") != "backbone_final" or int(
            metadata.get("environment_steps", -1)
        ) != 30_000_000:
            raise ValueError("Adaptation requires the completed 30M-step backbone.")
        if metadata.get("layout") != self.env_config.layout:
            raise ValueError("Backbone and adaptation layouts differ.")
        if self.pool_report.get("layout") != self.env_config.layout:
            raise ValueError("The admitted partner pool uses a different layout.")
        if len(self.pool_report["members"]) < 2:
            raise ValueError("The adaptation partner pool must contain two capable policies.")
        evidence = config.get("evidence")
        if not isinstance(evidence, Mapping) or set(evidence) != {"partner_action_channel"}:
            raise ValueError("Evidence config must declare only partner_action_channel.")
        probe_payload = dict(config["probe"])
        raw_controller = str(probe_payload.get("controller"))
        if raw_controller == "decision_focused":
            raise ValueError(
                "decision_focused is reserved for the proposal's sequential paired-branch "
                "controller. The implemented diagnostic must be named "
                "finite_prototype_two_action_surrogate."
            )
        if raw_controller == "response_voi":
            if self.spec.run_kind == "formal":
                raise ValueError(
                    "The old response_voi name is a generic information baseline, "
                    "not the formal decision-focused method."
                )
            probe_payload["controller"] = "generic_response_information"
            raw_controller = "generic_response_information"
        self.condition_id = str(config.get("condition_id", "diagnostic"))
        if self.spec.run_kind == "formal":
            expected_controller = {
                "decision_focused": "registered_response_sequential_branch_v1",
                "random_safe_probe": "registered_random_safe_probe_v1",
                "no_probe": "off",
                "generic_response_information": "generic_response_information",
            }.get(self.condition_id)
            if expected_controller is None:
                raise ValueError(
                    "Formal adaptation requires a registered proposal condition_id."
                )
            if bool(evidence["partner_action_channel"]):
                raise ValueError(
                    "Formal proposal runs must use official local observations only."
                )
            if raw_controller != expected_controller:
                raise ValueError(
                    "Formal proposal condition_id and controller disagree."
                )
            if expected_controller in {
                "registered_response_sequential_branch_v1",
                "registered_random_safe_probe_v1",
            }:
                raise NotImplementedError(
                    "The requested formal controller is not implemented. The sequential "
                    "response-use/mask branch and matched random-safe calibration must "
                    "exist before either can produce a scientific readout. A finite-"
                    "prototype two-action surrogate cannot replace them."
                )
            if probe_payload.get("response_route", "use_registered_response") != (
                "use_registered_response"
            ):
                raise ValueError(
                    "Formal main-method runs must use the registered probe response."
                )
        calibration_path = config.get("probe_calibration_summary")
        self.probe_calibration_binding: dict[str, Any] | None = None
        if calibration_path is not None:
            threshold_field = (
                "surrogate_score_threshold"
                if raw_controller == "finite_prototype_two_action_surrogate"
                else "response_disagreement_threshold"
            )
            if probe_payload.get(threshold_field) is not None:
                raise ValueError(
                    "A calibrated run must not also declare a manual controller threshold."
                )
            response_initialization_path = config.get(
                "response_initialization_checkpoint"
            )
            if response_initialization_path is None:
                raise ValueError("Probe calibration requires a response initialization.")
            raw_summary_sha256 = config.get("probe_calibration_summary_sha256")
            expected_summary_sha256: str | None = None
            if self.spec.run_kind == "formal":
                expected_summary_sha256 = str(raw_summary_sha256)
                if len(expected_summary_sha256) != 64 or any(
                    character not in "0123456789abcdef"
                    for character in expected_summary_sha256
                ):
                    raise ValueError(
                        "Formal probing must bind the calibration summary SHA-256."
                    )
            elif raw_summary_sha256 not in {None, "pending"}:
                expected_summary_sha256 = str(raw_summary_sha256)
            calibrated_threshold = validate_probe_calibration_artifacts(
                calibration_path,
                controller=raw_controller,
                probe_config=probe_payload,
                expected_partner_ids=tuple(self.pool_report["members"]),
                expected_response_checkpoint=response_initialization_path,
                expected_backbone_checkpoint=config["backbone_checkpoint"],
                expected_pool_report=config["partner_pool_admission_report"],
                expected_environment_config_sha256=str(
                    config.get("_environment_config_sha256", "")
                ),
                expected_quantile=float(
                    config.get("probe_calibration_quantile", math.nan)
                ),
                expected_summary_sha256=expected_summary_sha256,
            )
            if not math.isfinite(calibrated_threshold) or calibrated_threshold <= 0.0:
                raise ValueError("The calibrated response threshold must be finite and positive.")
            probe_payload[threshold_field] = calibrated_threshold
            calibration_document = json.loads(
                Path(calibration_path).read_text(encoding="utf-8")
            )
            self.probe_calibration_binding = {
                "summary_path": str(Path(calibration_path).resolve()),
                "summary_sha256": _file_sha256(calibration_path),
                "rows_sha256": calibration_document["calibration_rows_sha256"],
                "calibration_contract_sha256": calibration_document[
                    "calibration_contract_sha256"
                ],
                "calibration_implementation_sha256": calibration_document[
                    "calibration_implementation_sha256"
                ],
            }
        probe = validate_response_probe_config(probe_payload)
        if probe["enabled"] != (probe["controller"] != "off"):
            raise ValueError("probe.enabled must agree with whether the controller is off.")
        if probe["response_model_mode"] != "partner_conditioned":
            raise ValueError(
                "New adaptation training requires partner-conditioned response models."
            )
        if int(probe["response_ensemble_size"]) != len(self.pool_report["members"]):
            raise ValueError(
                "Partner-conditioned response heads must match the admitted partner count."
            )
        if self.spec.training_mode == "response_only" and probe["enabled"]:
            raise ValueError("Response-only fitting must not execute probe actions.")
        if self.spec.training_mode == "probe_calibration" and (
            probe["controller"] not in {
                "finite_prototype_two_action_surrogate",
                "generic_response_information",
            }
            or not probe["enabled"]
        ):
            raise ValueError("Probe calibration requires the enabled response controller.")
        if self.spec.training_mode == "probe_calibration":
            quantile = float(config.get("probe_calibration_quantile", math.nan))
            if not 0.0 < quantile < 1.0:
                raise ValueError("Probe calibration quantile must lie in (0,1).")
        if (
            self.spec.run_kind == "formal"
            and raw_controller != "off"
            and calibration_path is None
        ):
            raise ValueError(
                "A formal probing condition requires a measured calibration summary."
            )
        self.probe_config = probe
        model_config = config["model"]
        initialization = str(config.get("initialization", "backbone"))
        self.initialization = initialization
        torch.manual_seed(
            derive_ocv2_execution_seed(
                derive_standard_seed(self.spec.seed, "adaptation", "torch")
            )
        )
        model_kwargs = {
            "n_critic_heads": int(model_config["n_critic_heads"]),
            "partner_action_channel": bool(evidence["partner_action_channel"]),
            "response_ensemble_size": int(probe["response_ensemble_size"]),
            "response_hidden_dim": int(probe["response_hidden_dim"]),
            "partner_value_hidden_dim": int(probe["partner_value_hidden_dim"]),
            "response_vocabulary_size": (
                7
                if evidence["partner_action_channel"]
                else REGISTERED_LOCAL_RESPONSE_SPEC.q
            ),
            "response_model_mode": str(probe["response_model_mode"]),
        }
        if initialization == "backbone":
            self.model = PathCAdaptationPolicy.from_backbone(
                backbone, **model_kwargs
            ).to(self.device)
        elif initialization == "random":
            self.model = PathCAdaptationPolicy(
                backbone.observation_shape,
                recurrent_dim=backbone.recurrent_dim,
                visual_embedding_dim=backbone.visual_embedding_dim,
                conv_channels=backbone.conv_channels,
                encoder_spatial_mode=backbone.encoder_spatial_mode,
                **model_kwargs,
            ).to(self.device)
        else:
            raise ValueError("initialization must be backbone or random.")
        response_initialization = config.get("response_initialization_checkpoint")
        if response_initialization is not None:
            response_model, response_metadata = load_standard_checkpoint(
                response_initialization, device=self.device
            )
            if not isinstance(response_model, PathCAdaptationPolicy):
                raise ValueError("Response initialization must be an adaptation checkpoint.")
            if response_metadata.get("phase") != "response_prefit":
                raise ValueError("Response initialization must come from response-only fitting.")
            if response_metadata.get("initialization") != self.initialization:
                raise ValueError(
                    "Response initialization used a different trunk initialization."
                )
            if response_model.response_model_mode != "partner_conditioned":
                raise ValueError("Response initialization is not partner-conditioned.")
            expected_hypotheses = list(self.pool_report["members"])
            if response_metadata.get("partner_hypothesis_ids") != expected_hypotheses:
                raise ValueError("Response initialization used a different partner pool.")
            if response_model.response_ensemble.ensemble_size != len(expected_hypotheses):
                raise ValueError("Response initialization has the wrong hypothesis count.")
            self.model.response_ensemble.load_state_dict(
                response_model.response_ensemble.state_dict()
            )
            self.model.partner_value_ensemble.load_state_dict(
                response_model.partner_value_ensemble.state_dict()
            )
        if (
            self.spec.training_mode in {"joint", "probe_calibration"}
            and response_initialization is None
        ):
            raise ValueError("Joint training and calibration require fitted response models.")
        self.target_model = copy.deepcopy(self.model).to(self.device)
        for parameter in self.target_model.parameters():
            parameter.requires_grad_(False)
        self.critic_optimizer = torch.optim.Adam(
            [
                {"params": self.model.trunk_parameters, "lr": self.spec.trunk_learning_rate},
                {"params": self.model.critic_parameters, "lr": self.spec.trunk_learning_rate},
            ]
        )
        self.actor_optimizer = torch.optim.Adam(
            self.model.actor_head.parameters(), lr=self.spec.actor_learning_rate
        )
        self.response_optimizer = torch.optim.Adam(
            self.model.response_ensemble.parameters(), lr=self.spec.response_learning_rate
        )
        self.partner_value_optimizer = torch.optim.Adam(
            self.model.partner_value_parameters,
            lr=self.spec.response_learning_rate,
        )
        self.pool = BatchedEnvPool(
            self.env_config.make_adapter().env,
            self.spec.batch_size_envs,
        )
        self.rng = np.random.default_rng(
            derive_ocv2_execution_seed(
                derive_standard_seed(self.spec.seed, "adaptation", "policy")
            )
        )
        self.partners: list[RecurrentIPPOBackbone] = []
        for member in self.pool_report["members"].values():
            partner, partner_metadata = load_standard_checkpoint(
                member["checkpoint_path"], device=self.device
            )
            if not isinstance(partner, RecurrentIPPOBackbone):
                raise ValueError("Every admitted partner must be a backbone checkpoint.")
            if partner_metadata.get("layout") != self.env_config.layout:
                raise ValueError("An admitted partner uses a different layout.")
            partner.eval()
            self.partners.append(partner)
        if not self.partners:
            raise ValueError("The admitted partner pool is empty.")
        self.spec.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.spec.output_dir / "training_metrics.jsonl"

    def optimize_losses(
        self,
        *,
        q_online_all: torch.Tensor,
        q_target_all: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        discounts: torch.Tensor,
        features: torch.Tensor,
        old_log_prob: torch.Tensor,
        advantages: torch.Tensor,
        response_actions: torch.Tensor,
        response_tokens: torch.Tensor,
        response_partner_assignments: torch.Tensor,
        value_candidate_actions: torch.Tensor,
        value_continuation_actions: torch.Tensor,
        raw_returns_to_go: torch.Tensor,
        critic_bootstrap_mask: torch.Tensor,
        actor_mask: torch.Tensor,
        ppo_clip: float,
    ) -> dict[str, float]:
        """Apply isolated objectives; only the critic call retains trunk features."""

        _, _, td = sequence_td_core(
            q_online_all,
            q_target_all,
            actions=actions,
            rewards=rewards,
            dones=dones,
            discounts=discounts,
            n_heads=self.model.n_heads,
        )
        if critic_bootstrap_mask.shape != (td.shape[0], td.shape[2]):
            raise ValueError("critic_bootstrap_mask must have shape [episodes, heads].")
        critic_weights = critic_bootstrap_mask[:, None, :].to(td.dtype).expand_as(td)
        critic_loss = (td * critic_weights).sum() / critic_weights.sum().clamp_min(1.0)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        trunk_critic_norm = float(torch.nn.utils.clip_grad_norm_(
            self.model.trunk_parameters + self.model.critic_parameters,
            self.spec.gradient_clip_norm,
        ))
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad(set_to_none=True)

        detached_features = features.detach()
        actor_losses: list[torch.Tensor] = []
        trunk_actor_norm = 0.0
        for _ in range(self.spec.actor_update_epochs):
            logits = self.model.actor_logits(detached_features)
            distribution = torch.distributions.Categorical(logits=logits[:, :-1])
            new_log_prob = distribution.log_prob(actions)
            ratio = torch.exp(new_log_prob - old_log_prob)
            actor_terms = -torch.minimum(
                ratio * advantages,
                torch.clamp(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip) * advantages,
            ) - self.spec.entropy_coefficient * distribution.entropy()
            actor_weights = actor_mask.to(actor_terms.dtype)
            actor_loss = (actor_terms * actor_weights).sum() / actor_weights.sum().clamp_min(1.0)
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            trunk_actor_norm = max(
                trunk_actor_norm,
                float(sum(
                    0.0 if parameter.grad is None else parameter.grad.detach().abs().sum().item()
                    for parameter in self.model.trunk_parameters
                )),
            )
            self.actor_optimizer.step()
            actor_losses.append(actor_loss.detach())
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss = torch.stack(actor_losses).mean()

        response_loss, response_head_losses = (
            self.model.response_ensemble.partner_conditioned_cross_entropy(
                features[:, :-2].detach(),
                response_actions,
                response_tokens,
                response_partner_assignments,
            )
        )
        self.response_optimizer.zero_grad(set_to_none=True)
        response_loss.backward()
        trunk_response_norm = float(sum(
            0.0 if parameter.grad is None else parameter.grad.detach().abs().sum().item()
            for parameter in self.model.trunk_parameters
        ))
        self.response_optimizer.step()
        partner_value_loss, partner_value_head_losses = (
            self.model.partner_value_ensemble.partner_conditioned_mse(
                features[:, :-2].detach(),
                value_candidate_actions,
                value_continuation_actions,
                raw_returns_to_go,
                response_partner_assignments,
            )
        )
        self.partner_value_optimizer.zero_grad(set_to_none=True)
        partner_value_loss.backward()
        trunk_partner_value_norm = float(sum(
            0.0 if parameter.grad is None else parameter.grad.detach().abs().sum().item()
            for parameter in self.model.trunk_parameters
        ))
        self.partner_value_optimizer.step()
        metrics = {
            "critic_td_loss": float(critic_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "response_cross_entropy": float(response_loss.detach().cpu()),
            "partner_value_mse": float(partner_value_loss.detach().cpu()),
            "trunk_grad_norm_from_critic": trunk_critic_norm,
            "trunk_grad_norm_from_actor": trunk_actor_norm,
            "trunk_grad_norm_from_response": trunk_response_norm,
            "trunk_grad_norm_from_partner_value": trunk_partner_value_norm,
        }
        metrics.update({
            f"response_head_{index}_cross_entropy": float(loss.detach().cpu())
            for index, loss in enumerate(response_head_losses)
        })
        metrics.update({
            f"partner_value_head_{index}_mse": float(loss.detach().cpu())
            for index, loss in enumerate(partner_value_head_losses)
        })
        return metrics

    def optimize_response_only(
        self,
        *,
        features: torch.Tensor,
        response_actions: torch.Tensor,
        response_tokens: torch.Tensor,
        response_partner_assignments: torch.Tensor,
        value_candidate_actions: torch.Tensor,
        value_continuation_actions: torch.Tensor,
        raw_returns_to_go: torch.Tensor,
    ) -> dict[str, float]:
        """Fit partner-specific response heads without changing actor, critic, or trunk."""

        response_loss, response_head_losses = (
            self.model.response_ensemble.partner_conditioned_cross_entropy(
                features[:, :-2].detach(),
                response_actions,
                response_tokens,
                response_partner_assignments,
            )
        )
        self.response_optimizer.zero_grad(set_to_none=True)
        response_loss.backward()
        trunk_response_norm = float(sum(
            0.0 if parameter.grad is None else parameter.grad.detach().abs().sum().item()
            for parameter in self.model.trunk_parameters
        ))
        self.response_optimizer.step()
        partner_value_loss, partner_value_head_losses = (
            self.model.partner_value_ensemble.partner_conditioned_mse(
                features[:, :-2].detach(),
                value_candidate_actions,
                value_continuation_actions,
                raw_returns_to_go,
                response_partner_assignments,
            )
        )
        self.partner_value_optimizer.zero_grad(set_to_none=True)
        partner_value_loss.backward()
        trunk_partner_value_norm = float(sum(
            0.0 if parameter.grad is None else parameter.grad.detach().abs().sum().item()
            for parameter in self.model.trunk_parameters
        ))
        self.partner_value_optimizer.step()
        metrics = {
            "critic_td_loss": 0.0,
            "actor_loss": 0.0,
            "response_cross_entropy": float(response_loss.detach().cpu()),
            "partner_value_mse": float(partner_value_loss.detach().cpu()),
            "trunk_grad_norm_from_critic": 0.0,
            "trunk_grad_norm_from_actor": 0.0,
            "trunk_grad_norm_from_response": trunk_response_norm,
            "trunk_grad_norm_from_partner_value": trunk_partner_value_norm,
        }
        metrics.update({
            f"response_head_{index}_cross_entropy": float(loss.detach().cpu())
            for index, loss in enumerate(response_head_losses)
        })
        metrics.update({
            f"partner_value_head_{index}_mse": float(loss.detach().cpu())
            for index, loss in enumerate(partner_value_head_losses)
        })
        return metrics

    def _sequence_batch(
        self,
        *,
        observations: torch.Tensor,
        previous_actions: torch.Tensor,
        previous_rewards: torch.Tensor,
        episode_starts: torch.Tensor,
    ) -> PrimitiveObservationBatch:
        batch_size, sequence_length = previous_actions.shape
        return PrimitiveObservationBatch(
            observations=observations,
            previous_actions=previous_actions,
            previous_rewards=previous_rewards,
            episode_starts=episode_starts,
            valid_actions=torch.ones(
                batch_size,
                sequence_length,
                self.model.n_actions,
                dtype=torch.bool,
                device=self.device,
            ),
            time_mask=torch.ones(
                batch_size, sequence_length, dtype=torch.bool, device=self.device
            ),
            lengths=torch.full(
                (batch_size,), sequence_length, dtype=torch.long, device=self.device
            ),
        )

    @torch.no_grad()
    def _partner_actions(
        self,
        *,
        observations: np.ndarray,
        previous_actions: np.ndarray,
        previous_rewards: np.ndarray,
        episode_starts: np.ndarray,
        assignments: np.ndarray,
        states: list[Any],
    ) -> np.ndarray:
        selected = np.zeros(self.spec.batch_size_envs, dtype=np.int64)
        uniforms = self.rng.random(self.spec.batch_size_envs)
        for partner_index in np.unique(assignments):
            rows = np.flatnonzero(assignments == partner_index)
            partner = self.partners[int(partner_index)]
            batch = single_step_batch(
                observations[rows],
                previous_actions[rows],
                previous_rewards[rows],
                episode_starts[rows],
                n_actions=partner.n_actions,
                device=self.device,
            )
            full_state = states[int(partner_index)]
            row_index = torch.as_tensor(rows, dtype=torch.long, device=self.device)
            sub_state = type(full_state)(hidden=full_state.hidden[:, row_index])
            logits, _, _, next_state = partner.forward_step(batch, sub_state)
            full_state.hidden[:, row_index] = next_state.hidden
            selected[rows] = sample_actor_actions(logits, uniforms[rows])
        return selected

    def _choose_ego_actions(
        self,
        *,
        actor_logits: torch.Tensor,
        critic_q: torch.Tensor,
        features: torch.Tensor,
        episode_step: int,
        probe_counts: np.ndarray,
        partner_belief: torch.Tensor | None,
        pending_continuation_values: torch.Tensor | None,
    ) -> AdaptationActionDecision:
        distribution = torch.distributions.Categorical(logits=actor_logits)
        sampled = distribution.sample()
        controller = self.probe_config["controller"]
        base_actions = sampled
        if (
            controller == "finite_prototype_two_action_surrogate"
            and pending_continuation_values is not None
        ):
            if partner_belief is None:
                raise ValueError("Partner-conditioned continuation needs a partner belief.")
            if pending_continuation_values.shape != (
                self.spec.batch_size_envs,
                partner_belief.shape[1],
                self.model.n_actions,
            ):
                raise ValueError("Pending continuation values have the wrong shape.")
            belief_control_values = torch.einsum(
                "bk,bka->ba", partner_belief, pending_continuation_values
            )
            base_actions = belief_control_values.argmax(dim=-1)
        selected = base_actions.detach().cpu().numpy().astype(np.int64)
        is_probe = np.zeros(self.spec.batch_size_envs, dtype=bool)
        disagreement = np.full(self.spec.batch_size_envs, np.nan, dtype=np.float32)
        candidate_regret = np.full(
            self.spec.batch_size_envs, np.nan, dtype=np.float32
        )
        candidate_equals_greedy = np.zeros(self.spec.batch_size_envs, dtype=bool)
        candidate_equals_baseline = np.zeros(self.spec.batch_size_envs, dtype=bool)
        disagreement_pass = np.zeros(self.spec.batch_size_envs, dtype=bool)
        regret_pass = np.zeros(self.spec.batch_size_envs, dtype=bool)
        budget_pass = np.zeros(self.spec.batch_size_envs, dtype=bool)
        candidate_evaluated = np.zeros(self.spec.batch_size_envs, dtype=bool)
        allowed = (
            self.probe_config["enabled"]
            and episode_step < self.probe_config["probe_window_environment_steps"]
        )
        allowed_rows = (
            probe_counts < self.probe_config["probe_budget_per_episode"]
            if allowed
            else np.zeros(self.spec.batch_size_envs, dtype=bool)
        )
        valid = torch.ones_like(actor_logits, dtype=torch.bool)
        candidate_actions = torch.arange(self.model.n_actions, device=self.device)
        candidate_probabilities = None
        candidate_continuation_values = None
        if self.model.response_model_mode == "partner_conditioned":
            candidate_probabilities = (
                self.model.response_ensemble.probabilities_for_candidates(
                    features.detach(), candidate_actions
                )
            )
            candidate_continuation_values = (
                self.model.partner_value_ensemble.values_for_candidates(
                    features.detach(), candidate_actions
                )
            )
        if (
            controller == "finite_prototype_two_action_surrogate"
            and bool(allowed_rows.any())
        ):
            if (
                partner_belief is None
                or candidate_probabilities is None
                or candidate_continuation_values is None
            ):
                raise ValueError(
                    "The finite-prototype surrogate requires belief, response, and value models."
                )
            decision = select_finite_prototype_two_action_surrogate(
                actor_logits=actor_logits,
                candidate_probabilities=candidate_probabilities,
                continuation_values=candidate_continuation_values,
                partner_belief=partner_belief,
                valid_actions=valid,
                surrogate_score_threshold=self.probe_config[
                    "surrogate_score_threshold"
                ],
                max_probe_task_cost=self.probe_config["max_probe_task_cost"],
                probe_allowed=allowed_rows,
            )
            if self.spec.training_mode != "probe_calibration":
                selected = decision.actions.copy()
                is_probe = decision.is_probe
            disagreement = decision.candidate_score
            candidate_regret = decision.candidate_regret
            candidate_equals_greedy = decision.candidate_equals_greedy
            candidate_equals_baseline = decision.candidate_equals_baseline
            disagreement_pass = decision.disagreement_pass
            regret_pass = decision.regret_pass
            budget_pass = decision.budget_pass
            candidate_evaluated[:] = True
        elif controller == "generic_response_information" and bool(allowed_rows.any()):
            if candidate_probabilities is None:
                raise ValueError("The response-information baseline needs response models.")
            decision = select_response_probe(
                actor_logits=actor_logits,
                critic_q_values=critic_q,
                candidate_probabilities=candidate_probabilities,
                valid_actions=valid,
                response_disagreement_threshold=self.probe_config[
                    "response_disagreement_threshold"
                ],
                max_probe_regret=self.probe_config["max_probe_regret"],
                probe_allowed=allowed_rows,
                partner_weights=partner_belief,
                baseline_actions=base_actions,
            )
            if self.spec.training_mode != "probe_calibration":
                selected[decision.is_probe] = decision.actions[decision.is_probe]
                is_probe = decision.is_probe
            disagreement = decision.candidate_score
            candidate_regret = decision.candidate_regret
            candidate_equals_greedy = decision.candidate_equals_greedy
            candidate_equals_baseline = decision.candidate_equals_baseline
            disagreement_pass = decision.disagreement_pass
            regret_pass = decision.regret_pass
            budget_pass = decision.budget_pass
            candidate_evaluated[:] = True
        elif controller == "advantage_disagreement" and bool(allowed_rows.any()):
            decision = choose_primitive_actions(
                critic_q,
                valid,
                rng=self.rng,
                probe_enabled=True,
                disagreement_threshold=self.probe_config[
                    "advantage_disagreement_threshold"
                ],
                max_probe_regret=self.probe_config["max_probe_regret"],
                probe_allowed=allowed_rows,
            )
            candidate_equals_baseline = decision.actions == selected
            is_probe = decision.is_probe & ~candidate_equals_baseline
            selected[is_probe] = decision.actions[is_probe]
            disagreement = decision.probe_candidate_disagreement
            candidate_regret = decision.probe_candidate_regret
            candidate_equals_greedy = decision.probe_candidate_equals_greedy
            disagreement_pass = decision.probe_disagreement_pass
            regret_pass = decision.probe_regret_pass
            budget_pass = decision.probe_budget_pass
            candidate_evaluated[:] = True
        elif controller == "random" and bool(allowed_rows.any()):
            greedy = actor_logits.argmax(dim=-1).detach().cpu().numpy()
            mean_q = critic_q.mean(dim=1).detach().cpu().numpy()
            random_candidates = self.rng.integers(
                0, self.model.n_actions, size=self.spec.batch_size_envs
            )
            rows = np.arange(self.spec.batch_size_envs)
            regret = mean_q[rows, greedy] - mean_q[rows, random_candidates]
            candidate_regret = regret.astype(np.float32)
            candidate_equals_greedy = random_candidates == greedy
            candidate_equals_baseline = random_candidates == selected
            regret_pass = regret <= self.probe_config["max_probe_regret"]
            budget_pass = allowed_rows.copy()
            candidate_evaluated[:] = True
            is_probe = (
                allowed_rows
                & ~candidate_equals_greedy
                & ~candidate_equals_baseline
                & regret_pass
            )
            selected[is_probe] = random_candidates[is_probe]
        elif controller != "off":
            if controller not in {
                "finite_prototype_two_action_surrogate",
                "generic_response_information",
                "advantage_disagreement",
                "random",
            }:
                raise ValueError(f"Unsupported adaptation probe controller {controller!r}.")
        executed = torch.as_tensor(selected, dtype=torch.long, device=self.device)
        entropy = distribution.entropy().detach().cpu().numpy().astype(np.float32)
        actual_response_probabilities = None
        actual_continuation_values = None
        if self.model.response_model_mode == "partner_conditioned":
            if candidate_probabilities is None or candidate_continuation_values is None:
                raise RuntimeError("Partner-conditioned candidate predictions are missing.")
            row_index = torch.arange(executed.shape[0], device=self.device)
            actual_response_probabilities = candidate_probabilities[
                row_index, executed
            ]
            actual_continuation_values = candidate_continuation_values[
                row_index, executed
            ]
        return AdaptationActionDecision(
            actions=selected,
            is_probe=is_probe,
            old_log_prob=distribution.log_prob(executed).detach(),
            candidate_score=disagreement,
            candidate_regret=candidate_regret,
            candidate_equals_greedy=candidate_equals_greedy,
            candidate_equals_baseline=candidate_equals_baseline,
            disagreement_pass=disagreement_pass,
            regret_pass=regret_pass,
            budget_pass=budget_pass,
            candidate_evaluated=candidate_evaluated,
            actor_entropy=entropy,
            actual_response_probabilities=actual_response_probabilities,
            actual_continuation_values=actual_continuation_values,
        )

    def run(self) -> dict[str, Any]:
        """Collect complete episodes, train isolated objectives, and checkpoint."""

        self.metrics_path.write_text("", encoding="utf-8")
        environment_steps = episodes = gradient_updates = total_probes = 0
        total_correct_deliveries = 0
        stopped = False
        calibration_scores: list[float] = []
        calibration_regrets: list[float] = []
        calibration_equals_greedy: list[bool] = []
        calibration_equals_baseline: list[bool] = []
        calibration_rows: list[dict[str, Any]] = []
        calibration_role_counts = {"slot_0": 0, "slot_1": 0}
        partner_hypothesis_ids = tuple(self.pool_report["members"])
        calibration_partner_counts = {
            partner_id: 0 for partner_id in partner_hypothesis_ids
        }
        batch_size = self.spec.batch_size_envs
        total_episodes = self.spec.total_environment_steps // self.spec.rollout_steps
        validate_unique_execution_seed_mapping(
            (
                derive_standard_seed(
                    self.spec.seed, "adaptation", "episode", episode_index
                )
                for episode_index in range(total_episodes)
            ),
            name="Path C adaptation episode seeds",
        )
        while environment_steps < self.spec.total_environment_steps:
            episode_numbers = np.arange(episodes, episodes + batch_size, dtype=np.int64)
            canonical_seeds = [
                derive_standard_seed(
                    self.spec.seed, "adaptation", "episode", int(number)
                )
                for number in episode_numbers
            ]
            self.pool.reset(np.asarray(
                [derive_ocv2_execution_seed(seed) for seed in canonical_seeds],
                dtype=np.uint32,
            ))
            observations = self.pool.snapshot_obs()
            if self.spec.training_mode == "probe_calibration":
                ego_slots = (episode_numbers % 2).astype(np.int64)
                partner_assignments = (
                    (episode_numbers // 2) % len(self.partners)
                ).astype(np.int64)
                calibration_role_counts["slot_0"] += int((ego_slots == 0).sum())
                calibration_role_counts["slot_1"] += int((ego_slots == 1).sum())
                for partner_index, partner_id in enumerate(partner_hypothesis_ids):
                    calibration_partner_counts[partner_id] += int(
                        (partner_assignments == partner_index).sum()
                    )
            else:
                ego_slots = self.rng.integers(0, 2, size=batch_size, dtype=np.int64)
                partner_assignments = self.rng.integers(
                    0, len(self.partners), size=batch_size, dtype=np.int64
                )
            row_index = np.arange(batch_size)
            previous_ego_actions = np.full(batch_size, 6, dtype=np.int64)
            previous_partner_actions = np.full(batch_size, 6, dtype=np.int64)
            previous_rewards = np.zeros(batch_size, dtype=np.float32)
            starts = np.ones(batch_size, dtype=bool)
            probe_counts = np.zeros(batch_size, dtype=np.int64)
            partner_belief = initial_partner_belief(
                batch_size,
                self.model.response_ensemble.ensemble_size,
                device=self.device,
            )
            pending_response_probabilities: torch.Tensor | None = None
            pending_continuation_values: torch.Tensor | None = None
            pending_response_is_probe: np.ndarray | None = None
            previous_ego_observation: np.ndarray | None = None
            ego_state = self.model.initial_state(batch_size, device=self.device)
            partner_states = [
                partner.initial_state(batch_size, device=self.device)
                for partner in self.partners
            ]
            stored: dict[str, list[torch.Tensor]] = {
                key: [] for key in (
                    "observations", "previous_actions", "previous_partner_actions",
                    "previous_rewards", "starts", "actions", "partner_actions",
                    "old_log_prob", "training_rewards", "raw_rewards", "dones", "actor_mask",
                    "partner_visible",
                )
            }
            window_raw_return = np.zeros(batch_size, dtype=np.float64)
            window_correct = window_wrong = window_indicator = window_ambiguous = 0
            window_shaped_sum = 0.0
            window_disagreement: list[float] = []
            window_candidate_regret: list[float] = []
            window_actor_entropy: list[float] = []
            window_belief_entropy: list[float] = []
            probe_gate_counts = {
                "candidate_evaluated_count": 0,
                "disagreement_threshold_pass_count": 0,
                "regret_pass_count": 0,
                "candidate_equals_greedy_count": 0,
                "candidate_equals_baseline_count": 0,
                "budget_and_window_pass_count": 0,
                "other_conditions_without_disagreement_pass_count": 0,
                "disagreement_only_block_count": 0,
                "regret_block_count": 0,
                "greedy_candidate_block_count": 0,
                "budget_block_count": 0,
                "joint_probe_count": 0,
            }
            probe_count_by_step_segment = {
                "steps_0_24": 0,
                "steps_25_49": 0,
                "steps_50_74": 0,
                "steps_75_99": 0,
            }
            for episode_step in range(self.spec.rollout_steps):
                ego_observation = np.stack(
                    (observations["agent_0"], observations["agent_1"]), axis=0
                )[ego_slots, row_index]
                partner_observation = np.stack(
                    (observations["agent_1"], observations["agent_0"]), axis=0
                )[ego_slots, row_index]
                partner_visible = partner_visible_from_default_observation(
                    ego_observation,
                    indicate_successful_delivery=self.env_config.indicate_successful_delivery,
                )
                if (
                    pending_response_probabilities is not None
                    and not self.model.partner_action_channel
                ):
                    if previous_ego_observation is None or pending_response_is_probe is None:
                        raise RuntimeError("Pending local response lacks its source history.")
                    local_change = local_non_agent_change_from_default_observation(
                        previous_ego_observation[:, None],
                        ego_observation[:, None],
                        indicate_successful_delivery=(
                            self.env_config.indicate_successful_delivery
                        ),
                    )[:, 0]
                    observed_response_tokens = response_tokens_from_observations(
                        partner_actions=np.zeros(batch_size, dtype=np.int64),
                        partner_visible=partner_visible,
                        local_non_agent_change=local_change,
                        partner_action_channel=False,
                    )
                    observed_mask = (
                        ~pending_response_is_probe
                        if self.probe_config["response_route"]
                        == "mask_current_probe_response"
                        else np.ones(batch_size, dtype=bool)
                    )
                    partner_belief = update_partner_belief(
                        partner_belief,
                        pending_response_probabilities,
                        torch.as_tensor(
                            observed_response_tokens,
                            dtype=torch.long,
                            device=self.device,
                        ),
                        probability_floor=float(
                            self.probe_config["belief_probability_floor"]
                        ),
                        observed=torch.as_tensor(observed_mask, device=self.device),
                    )
                ego_batch = single_step_batch(
                    ego_observation,
                    previous_ego_actions,
                    previous_rewards,
                    starts,
                    n_actions=6,
                    device=self.device,
                )
                partner_previous_tensor = torch.as_tensor(
                    previous_partner_actions,
                    dtype=torch.long,
                    device=self.device,
                ).reshape(batch_size, 1)
                with torch.no_grad():
                    critic_q, features, ego_state = self.model.forward_step(
                        ego_batch,
                        ego_state,
                        partner_previous_actions=(
                            partner_previous_tensor
                            if self.model.partner_action_channel else None
                        ),
                    )
                    actor_logits = self.model.actor_logits(features)
                    action_decision = self._choose_ego_actions(
                        actor_logits=actor_logits,
                        critic_q=critic_q,
                        features=features,
                        episode_step=episode_step,
                        probe_counts=probe_counts,
                        partner_belief=partner_belief,
                        pending_continuation_values=pending_continuation_values,
                    )
                ego_actions = action_decision.actions
                is_probe = action_decision.is_probe
                window_disagreement.extend(
                    float(value)
                    for value in action_decision.candidate_score[
                        action_decision.candidate_evaluated
                    ]
                    if np.isfinite(value)
                )
                window_candidate_regret.extend(
                    float(value)
                    for value in action_decision.candidate_regret[
                        action_decision.candidate_evaluated
                    ]
                    if np.isfinite(value)
                )
                if self.spec.training_mode == "probe_calibration":
                    calibration_scores.extend(
                        float(value)
                        for value in action_decision.candidate_score[
                            action_decision.candidate_evaluated
                        ]
                    )
                    calibration_regrets.extend(
                        float(value)
                        for value in action_decision.candidate_regret[
                            action_decision.candidate_evaluated
                        ]
                    )
                    calibration_equals_greedy.extend(
                        bool(value)
                        for value in action_decision.candidate_equals_greedy[
                            action_decision.candidate_evaluated
                        ]
                    )
                    calibration_equals_baseline.extend(
                        bool(value)
                        for value in action_decision.candidate_equals_baseline[
                            action_decision.candidate_evaluated
                        ]
                    )
                    for row in np.flatnonzero(
                        action_decision.candidate_evaluated
                    ):
                        calibration_rows.append({
                            "schema_version": "path_c_probe_calibration_row_v1",
                            "episode_index": int(episode_numbers[row]),
                            "ego_slot": int(ego_slots[row]),
                            "partner_hypothesis_id": partner_hypothesis_ids[
                                int(partner_assignments[row])
                            ],
                            "episode_step": int(episode_step),
                            "candidate_score": float(
                                action_decision.candidate_score[row]
                            ),
                            "candidate_regret": float(
                                action_decision.candidate_regret[row]
                            ),
                            "candidate_equals_greedy": bool(
                                action_decision.candidate_equals_greedy[row]
                            ),
                            "candidate_equals_baseline": bool(
                                action_decision.candidate_equals_baseline[row]
                            ),
                            "disagreement_pass": bool(
                                action_decision.disagreement_pass[row]
                            ),
                            "regret_pass": bool(
                                action_decision.regret_pass[row]
                            ),
                            "budget_and_window_pass": bool(
                                action_decision.budget_pass[row]
                            ),
                            "probe_executed": False,
                        })
                window_actor_entropy.extend(
                    float(value) for value in action_decision.actor_entropy
                )
                belief_entropy = -(
                    partner_belief
                    * partner_belief.clamp_min(torch.finfo(partner_belief.dtype).tiny).log()
                ).sum(dim=-1)
                window_belief_entropy.extend(
                    float(value) for value in belief_entropy.detach().cpu().numpy()
                )
                evaluated = action_decision.candidate_evaluated
                probe_gate_counts["candidate_evaluated_count"] += int(evaluated.sum())
                probe_gate_counts["disagreement_threshold_pass_count"] += int(
                    (evaluated & action_decision.disagreement_pass).sum()
                )
                probe_gate_counts["regret_pass_count"] += int(
                    (evaluated & action_decision.regret_pass).sum()
                )
                probe_gate_counts["candidate_equals_greedy_count"] += int(
                    (evaluated & action_decision.candidate_equals_greedy).sum()
                )
                probe_gate_counts["candidate_equals_baseline_count"] += int(
                    (evaluated & action_decision.candidate_equals_baseline).sum()
                )
                probe_gate_counts["budget_and_window_pass_count"] += int(
                    (evaluated & action_decision.budget_pass).sum()
                )
                probe_gate_counts[
                    "other_conditions_without_disagreement_pass_count"
                ] += int((
                    evaluated
                    & action_decision.regret_pass
                    & action_decision.budget_pass
                    & ~action_decision.candidate_equals_greedy
                    & ~action_decision.candidate_equals_baseline
                ).sum())
                probe_gate_counts["disagreement_only_block_count"] += int((
                    evaluated
                    & ~action_decision.disagreement_pass
                    & action_decision.regret_pass
                    & action_decision.budget_pass
                    & ~action_decision.candidate_equals_greedy
                    & ~action_decision.candidate_equals_baseline
                ).sum())
                probe_gate_counts["regret_block_count"] += int((
                    evaluated & ~action_decision.regret_pass
                ).sum())
                probe_gate_counts["greedy_candidate_block_count"] += int((
                    evaluated & action_decision.candidate_equals_greedy
                ).sum())
                probe_gate_counts["budget_block_count"] += int((
                    evaluated & ~action_decision.budget_pass
                ).sum())
                probe_gate_counts["joint_probe_count"] += int(is_probe.sum())
                if episode_step < 100:
                    segment_name = (
                        "steps_0_24" if episode_step < 25 else
                        "steps_25_49" if episode_step < 50 else
                        "steps_50_74" if episode_step < 75 else
                        "steps_75_99"
                    )
                    probe_count_by_step_segment[segment_name] += int(is_probe.sum())
                probe_counts += is_probe.astype(np.int64)
                total_probes += int(is_probe.sum())
                partner_actions = self._partner_actions(
                    observations=partner_observation,
                    previous_actions=previous_partner_actions,
                    previous_rewards=previous_rewards,
                    episode_starts=starts,
                    assignments=partner_assignments,
                    states=partner_states,
                )
                if (
                    pending_response_probabilities is not None
                    and self.model.partner_action_channel
                ):
                    if pending_response_is_probe is None:
                        raise RuntimeError("Pending action response lacks its source action.")
                    observed_response_tokens = response_tokens_from_observations(
                        partner_actions=partner_actions,
                        partner_visible=partner_visible,
                        local_non_agent_change=np.zeros(batch_size, dtype=bool),
                        partner_action_channel=self.model.partner_action_channel,
                    )
                    partner_belief = update_partner_belief(
                        partner_belief,
                        pending_response_probabilities,
                        torch.as_tensor(
                            observed_response_tokens,
                            dtype=torch.long,
                            device=self.device,
                        ),
                        probability_floor=float(
                            self.probe_config["belief_probability_floor"]
                        ),
                        observed=torch.as_tensor(
                            ~pending_response_is_probe
                            if self.probe_config["response_route"]
                            == "mask_current_probe_response"
                            else np.ones(batch_size, dtype=bool),
                            device=self.device,
                        ),
                    )
                pending_response_probabilities = (
                    action_decision.actual_response_probabilities
                )
                pending_continuation_values = (
                    action_decision.actual_continuation_values
                )
                pending_response_is_probe = is_probe.copy()
                actions_0 = np.where(ego_slots == 0, ego_actions, partner_actions)
                actions_1 = np.where(ego_slots == 1, ego_actions, partner_actions)
                next_observations, _, rewards_device, dones_device, info = (
                    self.pool.step_joint(actions_0, actions_1)
                )
                raw_reward = shared_team_reward(rewards_device).astype(np.float32)
                shaped_by_slot = shaped_rewards_by_agent(info, batch_size=batch_size)
                ego_shaped = np.where(
                    ego_slots == 0, shaped_by_slot[0], shaped_by_slot[1]
                ).astype(np.float32)
                anneal = max(
                    0.0,
                    1.0
                    - environment_steps
                    / float(self.spec.shaping_horizon_environment_steps),
                )
                training_reward = raw_reward + anneal * ego_shaped
                dones = np.asarray(dones_device["__all__"], dtype=bool)
                events = decompose_raw_reward_events(raw_reward)
                window_correct += events.correct_delivery_count
                window_wrong += events.wrong_delivery_count
                window_indicator += events.indicator_activation_count
                window_ambiguous += events.ambiguous_step_count
                total_correct_deliveries += events.correct_delivery_count
                window_raw_return += raw_reward
                window_shaped_sum += float((anneal * ego_shaped).sum())
                for key, value in (
                    ("observations", ego_batch.observations[:, 0]),
                    ("previous_actions", ego_batch.previous_actions[:, 0]),
                    ("previous_partner_actions", partner_previous_tensor[:, 0]),
                    ("previous_rewards", ego_batch.previous_rewards[:, 0]),
                    ("starts", ego_batch.episode_starts[:, 0]),
                    ("actions", torch.as_tensor(ego_actions, device=self.device)),
                    ("partner_actions", torch.as_tensor(partner_actions, device=self.device)),
                    ("old_log_prob", action_decision.old_log_prob),
                    ("training_rewards", torch.as_tensor(training_reward, device=self.device)),
                    ("raw_rewards", torch.as_tensor(raw_reward, device=self.device)),
                    ("dones", torch.as_tensor(dones, device=self.device)),
                    ("actor_mask", torch.as_tensor(~is_probe, device=self.device)),
                    ("partner_visible", torch.as_tensor(partner_visible, device=self.device)),
                ):
                    stored[key].append(value.detach())
                previous_ego_actions = ego_actions
                previous_partner_actions = partner_actions
                previous_rewards = raw_reward
                previous_ego_observation = ego_observation.copy()
                starts = dones
                observations = {
                    key: np.asarray(value) for key, value in next_observations.items()
                }
                environment_steps += batch_size
            if not bool(starts.all()):
                raise RuntimeError("Adaptation rollout did not end at 400 steps.")
            final_ego_observation = np.stack(
                (observations["agent_0"], observations["agent_1"]), axis=0
            )[ego_slots, row_index]
            sequence_observations = torch.cat(
                (
                    torch.stack(stored["observations"], dim=1),
                    torch.as_tensor(
                        final_ego_observation,
                        dtype=torch.float32,
                        device=self.device,
                    )[:, None],
                ),
                dim=1,
            )
            sequence_previous_actions = torch.cat((
                torch.stack(stored["previous_actions"], dim=1),
                torch.as_tensor(previous_ego_actions, device=self.device)[:, None],
            ), dim=1)
            sequence_partner_actions = torch.cat((
                torch.stack(stored["previous_partner_actions"], dim=1),
                torch.as_tensor(previous_partner_actions, device=self.device)[:, None],
            ), dim=1)
            sequence_previous_rewards = torch.cat((
                torch.stack(stored["previous_rewards"], dim=1),
                torch.zeros(batch_size, 1, device=self.device),
            ), dim=1)
            sequence_starts = torch.cat((
                torch.stack(stored["starts"], dim=1),
                torch.ones(batch_size, 1, dtype=torch.bool, device=self.device),
            ), dim=1)
            sequence_batch = self._sequence_batch(
                observations=sequence_observations,
                previous_actions=sequence_previous_actions,
                previous_rewards=sequence_previous_rewards,
                episode_starts=sequence_starts,
            )
            q_online, _, features = self.model.forward_sequence(
                sequence_batch,
                partner_previous_actions=(
                    sequence_partner_actions if self.model.partner_action_channel else None
                ),
            )
            with torch.no_grad():
                q_target, _, _ = self.target_model.forward_sequence(
                    sequence_batch,
                    partner_previous_actions=(
                        sequence_partner_actions
                        if self.target_model.partner_action_channel
                        else None
                    ),
                )
            actions_tensor = torch.stack(stored["actions"], dim=1).long()
            rewards_tensor = torch.stack(stored["training_rewards"], dim=1)
            raw_rewards_tensor = torch.stack(stored["raw_rewards"], dim=1)
            dones_tensor = torch.stack(stored["dones"], dim=1).bool()
            discounts = torch.full_like(rewards_tensor, self.spec.gamma)
            with torch.no_grad():
                predictions = q_online[:, :-1].mean(dim=2).gather(
                    -1, actions_tensor.unsqueeze(-1)
                ).squeeze(-1)
                returns_to_go = torch.zeros_like(rewards_tensor)
                raw_returns_to_go = torch.zeros_like(raw_rewards_tensor)
                running = torch.zeros(batch_size, device=self.device)
                raw_running = torch.zeros(batch_size, device=self.device)
                for step in reversed(range(self.spec.rollout_steps)):
                    running = rewards_tensor[:, step] + self.spec.gamma * running * (
                        ~dones_tensor[:, step]
                    ).to(running.dtype)
                    raw_running = raw_rewards_tensor[:, step] + self.spec.gamma * raw_running * (
                        ~dones_tensor[:, step]
                    ).to(raw_running.dtype)
                    returns_to_go[:, step] = running
                    raw_returns_to_go[:, step] = raw_running
                advantages = returns_to_go - predictions
                advantages = (advantages - advantages.mean()) / (
                    advantages.std(unbiased=False) + 1.0e-8
                )
            critic_mask = episode_bootstrap_mask(
                episode_numbers,
                ensemble_size=self.model.n_heads,
                bootstrap_seed=derive_ocv2_execution_seed(
                    derive_standard_seed(
                        self.spec.seed,
                        "adaptation",
                        "critic_bootstrap",
                        int(self.probe_config["bootstrap_seed"]),
                    )
                ),
                bootstrap_p=float(self.probe_config["critic_bootstrap_p"]),
            ).to(self.device)
            partner_action_array = (
                torch.stack(stored["partner_actions"], dim=1).cpu().numpy()
            )
            partner_visible_array = (
                torch.stack(stored["partner_visible"], dim=1).cpu().numpy()
            )
            observation_changes = local_non_agent_change_from_default_observation(
                sequence_observations[:, :-2].detach().cpu().numpy(),
                sequence_observations[:, 1:-1].detach().cpu().numpy(),
                indicate_successful_delivery=self.env_config.indicate_successful_delivery,
            )
            response_tokens = response_tokens_from_observations(
                partner_actions=partner_action_array[:, 1:],
                partner_visible=partner_visible_array[:, 1:],
                local_non_agent_change=observation_changes,
                partner_action_channel=self.model.partner_action_channel,
            )
            response_tokens_tensor = torch.as_tensor(
                response_tokens, dtype=torch.long, device=self.device
            )
            response_partner_assignments = torch.as_tensor(
                partner_assignments, dtype=torch.long, device=self.device
            )
            if self.spec.training_mode == "response_only":
                losses = self.optimize_response_only(
                    features=features,
                    response_actions=actions_tensor[:, :-1],
                    response_tokens=response_tokens_tensor,
                    response_partner_assignments=response_partner_assignments,
                    value_candidate_actions=actions_tensor[:, :-1],
                    value_continuation_actions=actions_tensor[:, 1:],
                    raw_returns_to_go=raw_returns_to_go[:, :-1],
                )
            elif self.spec.training_mode == "joint":
                losses = self.optimize_losses(
                    q_online_all=q_online,
                    q_target_all=q_target,
                    actions=actions_tensor,
                    rewards=rewards_tensor,
                    dones=dones_tensor,
                    discounts=discounts,
                    features=features,
                    old_log_prob=torch.stack(stored["old_log_prob"], dim=1),
                    advantages=advantages,
                    response_actions=actions_tensor[:, :-1],
                    response_tokens=response_tokens_tensor,
                    response_partner_assignments=response_partner_assignments,
                    value_candidate_actions=actions_tensor[:, :-1],
                    value_continuation_actions=actions_tensor[:, 1:],
                    raw_returns_to_go=raw_returns_to_go[:, :-1],
                    critic_bootstrap_mask=critic_mask,
                    actor_mask=torch.stack(stored["actor_mask"], dim=1),
                    ppo_clip=self.spec.ppo_clip,
                )
            else:
                losses = {
                    "critic_td_loss": 0.0,
                    "actor_loss": 0.0,
                    "response_cross_entropy": 0.0,
                    "partner_value_mse": 0.0,
                    "trunk_grad_norm_from_critic": 0.0,
                    "trunk_grad_norm_from_actor": 0.0,
                    "trunk_grad_norm_from_response": 0.0,
                    "trunk_grad_norm_from_partner_value": 0.0,
                }
            if self.spec.training_mode != "probe_calibration":
                gradient_updates += 1
            if (
                self.spec.training_mode == "joint"
                and environment_steps % self.spec.target_update_environment_steps == 0
            ):
                self.target_model.load_state_dict(self.model.state_dict())
            episodes += batch_size
            if probe_gate_counts["joint_probe_count"] != int(probe_counts.sum()):
                raise RuntimeError("Probe gate telemetry disagrees with executed probes.")
            metrics_row = {
                "schema_version": "path_c_adaptation_metrics_v2",
                "run_kind": self.spec.run_kind,
                "scientific_readout_allowed": self.spec.scientific_readout_allowed,
                "training_mode": self.spec.training_mode,
                "environment_steps": environment_steps,
                "episodes": episodes,
                "gradient_updates": gradient_updates,
                "raw_episode_return_mean": float(window_raw_return.mean()),
                "raw_episode_return_std": float(window_raw_return.std()),
                "correct_delivery_count": window_correct,
                "wrong_delivery_count": window_wrong,
                "indicator_activation_count": window_indicator,
                "ambiguous_reward_step_count": window_ambiguous,
                "annealed_shaped_reward_sum": window_shaped_sum,
                "probe_count": int(probe_counts.sum()),
                "probe_rate": float(probe_counts.sum() / (batch_size * self.spec.rollout_steps)),
                "normalized_probe_budget_use": float(
                    probe_counts.sum()
                    / (
                        batch_size
                        * int(self.probe_config["probe_budget_per_episode"])
                    )
                ),
                "response_information_quantiles": _finite_quantiles(
                    window_disagreement
                ),
                "probe_candidate_regret_quantiles": _finite_quantiles(
                    window_candidate_regret
                ),
                "partner_belief_entropy_quantiles": _finite_quantiles(
                    window_belief_entropy
                ),
                "probe_gate_counts": probe_gate_counts,
                "probe_count_by_step_segment": probe_count_by_step_segment,
                "response_token_counts": {
                    str(index): int(count)
                    for index, count in enumerate(
                        np.bincount(
                            response_tokens.reshape(-1),
                            minlength=self.model.response_ensemble.vocabulary_size,
                        )
                    )
                },
                "actor_entropy": float(np.mean(window_actor_entropy)),
                "input_contract": self.model.input_contract,
                "response_model_mode": self.model.response_model_mode,
                **losses,
            }
            if environment_steps % self.spec.metrics_interval_environment_steps == 0:
                with self.metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(metrics_row, sort_keys=True, allow_nan=False) + "\n"
                    )
            if (
                self.spec.training_mode == "joint"
                and environment_steps
                >= self.spec.stop_if_no_delivery_by_environment_steps
                and total_correct_deliveries == 0
            ):
                stopped = True
                break
        calibration_summary: dict[str, Any] | None = None
        if self.spec.training_mode == "probe_calibration":
            if stopped:
                raise RuntimeError("Probe calibration stopped before its complete schedule.")
            expected_decisions = (
                total_episodes
                * int(self.probe_config["probe_window_environment_steps"])
            )
            if len(calibration_scores) != expected_decisions:
                raise RuntimeError(
                    "Probe calibration did not record every scheduled decision."
                )
            if len(calibration_rows) != expected_decisions:
                raise RuntimeError("Probe calibration row ledger is incomplete.")
            if set(calibration_role_counts.values()) != {total_episodes // 2}:
                raise RuntimeError("Probe calibration did not balance the two ego positions.")
            expected_partner_episodes = total_episodes // len(self.partners)
            if set(calibration_partner_counts.values()) != {
                expected_partner_episodes
            }:
                raise RuntimeError("Probe calibration did not balance admitted partners.")
            threshold, safe_count = calibrate_response_disagreement_threshold(
                calibration_scores,
                calibration_regrets,
                calibration_equals_greedy,
                max_probe_regret=float(
                    self.probe_config["max_probe_task_cost"]
                    if self.probe_config["controller"]
                    == "finite_prototype_two_action_surrogate"
                    else self.probe_config["max_probe_regret"]
                ),
                quantile=float(self.config["probe_calibration_quantile"]),
                candidate_equals_baseline=calibration_equals_baseline,
            )
            calibration_rows_path = (
                self.spec.output_dir / "probe_calibration_rows.jsonl"
            )
            calibration_rows_text = "".join(
                json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                for row in calibration_rows
            )
            calibration_rows_bytes = calibration_rows_text.encode("utf-8")
            calibration_rows_path.write_text(
                calibration_rows_text,
                encoding="utf-8",
            )
            calibration_summary = {
                "schema_version": "path_c_probe_calibration_summary_v1",
                "scientific_readout_allowed": False,
                "episode_count": total_episodes,
                "decision_count": len(calibration_scores),
                "safe_intervention_candidate_count": safe_count,
                "calibration_quantile": float(
                    self.config["probe_calibration_quantile"]
                ),
                (
                    "calibrated_surrogate_score_threshold"
                    if self.probe_config["controller"]
                    == "finite_prototype_two_action_surrogate"
                    else "calibrated_response_disagreement_threshold"
                ): threshold,
                "controller": self.probe_config["controller"],
                "candidate_score_quantiles": _finite_quantiles(calibration_scores),
                "probe_actions_executed": total_probes,
                "role_counts": calibration_role_counts,
                "partner_counts": calibration_partner_counts,
                "response_initialization_checkpoint": str(
                    self.config["response_initialization_checkpoint"]
                ),
                "response_initialization_checkpoint_sha256": _file_sha256(
                    self.config["response_initialization_checkpoint"]
                ),
                "backbone_checkpoint_sha256": _file_sha256(
                    self.config["backbone_checkpoint"]
                ),
                "partner_pool_admission_report_sha256": _file_sha256(
                    self.config["partner_pool_admission_report"]
                ),
                "environment_config_sha256": str(
                    self.config["_environment_config_sha256"]
                ),
                "calibration_implementation_sha256": (
                    _calibration_implementation_sha256()
                ),
                "calibration_contract_sha256": (
                    _probe_calibration_contract_sha256(
                        self.probe_config["controller"],
                        self.probe_config,
                        float(self.config["probe_calibration_quantile"]),
                    )
                ),
                "calibration_rows_path": str(calibration_rows_path),
                "calibration_rows_sha256": hashlib.sha256(
                    calibration_rows_bytes
                ).hexdigest(),
                "calibration_rows_line_count": len(calibration_rows),
            }
            if total_probes != 0:
                raise RuntimeError("Probe calibration executed a probe action.")
            (self.spec.output_dir / "summary.json").write_text(
                json.dumps(calibration_summary, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            checkpoint = Path(self.config["response_initialization_checkpoint"])
        elif stopped:
            checkpoint = self.spec.output_dir / "adaptation_stopped_no_delivery.pt"
            save_standard_checkpoint(
                checkpoint,
                self.model,
                seed=self.spec.seed,
                environment_steps=environment_steps,
                episodes=episodes,
                phase="adaptation_stopped_no_delivery",
                extra_metadata={"layout": self.env_config.layout},
            )
        else:
            checkpoint = self.save_final(
                environment_steps=environment_steps, episodes=episodes
            )
        manifest = {
            "schema_version": "path_c_adaptation_artifact_v2",
            "run_status": "stopped_no_delivery" if stopped else "completed",
            "run_kind": self.spec.run_kind,
            "scientific_readout_allowed": self.spec.scientific_readout_allowed,
            "training_mode": self.spec.training_mode,
            "condition_id": self.condition_id,
            "initialization": self.initialization,
            "effective_environment_steps": environment_steps,
            "effective_episodes": episodes,
            "gradient_updates": gradient_updates,
            "correct_delivery_count": total_correct_deliveries,
            "probe_count": total_probes,
            "checkpoint": str(checkpoint),
            "input_contract": self.model.input_contract,
            "response_model_mode": self.model.response_model_mode,
            "response_vocabulary_sha256": self.model.response_vocabulary_sha256,
            "controller": self.probe_config["controller"],
            "response_route": self.probe_config["response_route"],
            "response_disagreement_threshold": float(
                self.probe_config["response_disagreement_threshold"]
            ),
            "partner_pool_member_count": len(self.partners),
            "partner_hypothesis_ids": list(self.pool_report["members"]),
            "partner_value_target": "discounted_raw_return_from_candidate_action",
            "controller_approximation_contract": (
                "finite_prototype_two_action_discounted_return_surrogate_v1"
                if self.probe_config["controller"]
                == "finite_prototype_two_action_surrogate"
                else None
            ),
            "probe_calibration_binding": self.probe_calibration_binding,
        }
        if self.probe_config["controller"] == "finite_prototype_two_action_surrogate":
            manifest["surrogate_score_threshold"] = float(
                self.probe_config["surrogate_score_threshold"]
            )
        if calibration_summary is not None:
            manifest["probe_calibration_summary"] = calibration_summary
        (self.spec.output_dir / "training_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        return manifest

    def save_final(self, *, environment_steps: int, episodes: int) -> Path:
        if environment_steps != self.spec.total_environment_steps:
            raise ValueError("A final adaptation checkpoint requires the configured budget.")
        response_only = self.spec.training_mode == "response_only"
        target = self.spec.output_dir / (
            "response_prefit.pt" if response_only else "path_c_final.pt"
        )
        save_standard_checkpoint(
            target,
            self.model,
            seed=self.spec.seed,
            environment_steps=environment_steps,
            episodes=episodes,
            phase="response_prefit" if response_only else "path_c_final",
            extra_metadata={
                "layout": self.env_config.layout,
                "environment": self.env_config.to_mapping(),
                "run_kind": self.spec.run_kind,
                "scientific_readout_allowed": self.spec.scientific_readout_allowed,
                "training_mode": self.spec.training_mode,
                "condition_id": self.condition_id,
                "initialization": self.initialization,
                "input_contract": self.model.input_contract,
                "response_model_mode": self.model.response_model_mode,
                "response_vocabulary_sha256": self.model.response_vocabulary_sha256,
                "controller": self.probe_config["controller"],
                "response_route": self.probe_config["response_route"],
                "response_disagreement_threshold": float(
                    self.probe_config["response_disagreement_threshold"]
                ),
                "surrogate_score_threshold": (
                    float(self.probe_config["surrogate_score_threshold"])
                    if self.probe_config["controller"]
                    == "finite_prototype_two_action_surrogate"
                    else None
                ),
                "controller_approximation_contract": (
                    "finite_prototype_two_action_discounted_return_surrogate_v1"
                    if self.probe_config["controller"]
                    == "finite_prototype_two_action_surrogate"
                    else None
                ),
                "probe_calibration_summary": self.config.get(
                    "probe_calibration_summary"
                ),
                "probe_calibration_binding": self.probe_calibration_binding,
                "partner_hypothesis_ids": list(self.pool_report["members"]),
                "partner_value_target": (
                    "discounted_raw_return_from_candidate_action"
                ),
                "backbone_formation_environment_steps": 30_000_000,
                "adaptation_environment_steps": environment_steps,
            },
        )
        return target


def load_adaptation_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    source_bytes = config_path.read_bytes()
    payload = yaml.safe_load(source_bytes.decode("utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema_version") != ADAPTATION_CONFIG_SCHEMA_VERSION:
        raise ValueError("Unsupported adaptation configuration.")
    config = dict(payload)
    environment = Path(config["environment_config"])
    if not environment.is_absolute():
        environment = (config_path.parent / environment).resolve()
    environment_bytes = environment.read_bytes()
    config["environment"] = yaml.safe_load(environment_bytes.decode("utf-8"))
    config["_source_config_sha256"] = hashlib.sha256(source_bytes).hexdigest()
    config["_environment_config_sha256"] = hashlib.sha256(
        environment_bytes
    ).hexdigest()
    for key in (
        "backbone_checkpoint",
        "partner_pool_admission_report",
        "output_dir",
        "response_initialization_checkpoint",
        "probe_calibration_summary",
    ):
        if key not in config:
            continue
        target = Path(config[key])
        if not target.is_absolute():
            config[key] = str((config_path.parent / target).resolve())
    return config


def run_adaptation_training(config_path: str | Path) -> dict[str, Any]:
    return PathCAdaptationTrainer(load_adaptation_config(config_path)).run()
