"""WP-A (§5): recurrent IPPO backbone training for the Path C v3 plan."""

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
from experiments.overcooked_v2.path_c_seed import derive_ocv2_execution_seed
from experiments.overcooked_v2.path_c_standard import (
    LocalObservationEncoder,
    PrimitiveObservationBatch,
    StandardEnvConfig,
    save_standard_checkpoint,
    shared_team_reward,
    single_step_batch,
)
from experiments.overcooked_v2.path_c_standard_diagnostics import (
    decompose_raw_reward_events,
)
from experiments.overcooked_v2.path_c_standard_training import (
    canonical_mapping_sha256,
    derive_standard_seed,
    partner_training_run_id,
    repository_source_dependency_closure,
    shaped_rewards_by_agent,
)


BACKBONE_CONFIG_SCHEMA_VERSION = "path_c_backbone_ppo_v1"

# This is the explicit repository-source closure that determines an IPPO
# partner-training run.  Admission imports this exact tuple and the shared
# closure-hashing function used by the config loader below.
IPPO_PARTNER_TRAINING_DEPENDENCIES = (
    "experiments/overcooked_v2/batched_rollout.py",
    "experiments/overcooked_v2/env_adapter.py",
    "experiments/overcooked_v2/path_c_backbone_ppo.py",
    "experiments/overcooked_v2/path_c_seed.py",
    "experiments/overcooked_v2/path_c_sequence.py",
    "experiments/overcooked_v2/path_c_standard.py",
    "experiments/overcooked_v2/path_c_standard_diagnostics.py",
    "experiments/overcooked_v2/path_c_standard_training.py",
    "experiments/overcooked_v2/residual_signature.py",
)

IPPO_PARTNER_FAMILY_DEFINITION: dict[str, str] = {
    "family_id": "recurrent_ippo_self_play_v1",
    "training_algorithm": "recurrent_independent_proximal_policy_optimization",
    "training_objective": "clipped_policy_and_value_objective_with_entropy",
    "convention_generation": "parameter_shared_two_agent_self_play",
    "model_class": "RecurrentIPPOBackbone",
    "checkpoint_phase": "backbone_snapshot",
    "action_rule": "categorical_actor_logits",
    "observation_contract": "official_local_5x5",
    "action_space": "six_primitive_actions",
    "reward_objective": (
        "raw_team_reward_plus_environment_native_per_agent_shaping_"
        "linear_to_zero_by_15000000_steps"
    ),
}
IPPO_PARTNER_FAMILY_SPEC_SHA256 = canonical_mapping_sha256(
    IPPO_PARTNER_FAMILY_DEFINITION
)


def ippo_partner_architecture_from_config(
    model_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize the registered IPPO model config to checkpoint form."""

    required = {
        "observation_shape",
        "visual_embedding_dim",
        "recurrent_dim",
        "conv_channels",
        "encoder_spatial_mode",
    }
    if set(model_config) != required:
        raise ValueError(
            "Registered IPPO model config must state exactly its frozen "
            "architecture fields."
        )
    return {
        "model_class": "RecurrentIPPOBackbone",
        "observation_shape": [int(item) for item in model_config["observation_shape"]],
        "n_actions": 6,
        "visual_embedding_dim": int(model_config["visual_embedding_dim"]),
        "recurrent_dim": int(model_config["recurrent_dim"]),
        "conv_channels": [int(item) for item in model_config["conv_channels"]],
        "encoder_spatial_mode": str(model_config["encoder_spatial_mode"]),
        "input_contract": (
            "own_default_local_observation+own_previous_action+"
            "own_previous_raw_reward+episode_start"
        ),
    }


def sample_actor_actions(
    logits: torch.Tensor,
    uniform_samples: np.ndarray,
) -> np.ndarray:
    """Sample categorical actor actions from caller-owned random numbers."""

    if logits.ndim != 2:
        raise ValueError("Actor logits must have shape [batch, actions].")
    uniforms = np.asarray(uniform_samples, dtype=np.float64)
    if uniforms.shape != (logits.shape[0],):
        raise ValueError("uniform_samples must have one value per actor row.")
    if not bool(np.isfinite(uniforms).all()) or bool(
        ((uniforms < 0.0) | (uniforms >= 1.0)).any()
    ):
        raise ValueError("uniform_samples must lie in [0,1).")
    probabilities = torch.softmax(logits.detach(), dim=-1).cpu().numpy()
    cumulative = np.cumsum(probabilities.astype(np.float64), axis=-1)
    actions = (uniforms[:, None] > cumulative).sum(axis=-1)
    return np.minimum(actions, logits.shape[-1] - 1).astype(np.int64)


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


@dataclass(frozen=True)
class BackbonePolicyState:
    hidden: torch.Tensor


class RecurrentIPPOBackbone(nn.Module):
    """Parameter-shared recurrent actor-critic used only for pool formation."""

    def __init__(
        self,
        observation_shape: Sequence[int],
        *,
        n_actions: int = 6,
        visual_embedding_dim: int = 128,
        recurrent_dim: int = 128,
        conv_channels: Sequence[int] = (32, 32, 16),
        encoder_spatial_mode: str = "flatten",
    ) -> None:
        super().__init__()
        shape = tuple(int(item) for item in observation_shape)
        if len(shape) != 3 or any(item <= 0 for item in shape):
            raise ValueError("observation_shape must be positive HWC dimensions.")
        self.observation_shape = shape
        self.n_actions = _positive_int(n_actions, name="n_actions")
        self.visual_embedding_dim = _positive_int(
            visual_embedding_dim,
            name="visual_embedding_dim",
        )
        self.recurrent_dim = _positive_int(recurrent_dim, name="recurrent_dim")
        self.conv_channels = tuple(int(item) for item in conv_channels)
        self.encoder_spatial_mode = str(encoder_spatial_mode)
        self.observation_encoder = LocalObservationEncoder(
            shape[-1],
            self.visual_embedding_dim,
            observation_height=shape[0],
            observation_width=shape[1],
            spatial_mode=self.encoder_spatial_mode,
            conv_channels=self.conv_channels,
        )
        evidence_dim = self.visual_embedding_dim + self.n_actions + 1 + 1 + 1
        self.recurrent = nn.GRU(evidence_dim, self.recurrent_dim, batch_first=True)
        self.actor_head = nn.Linear(self.recurrent_dim, self.n_actions)
        self.critic_head = nn.Linear(self.recurrent_dim, 1)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
    ) -> BackbonePolicyState:
        return BackbonePolicyState(
            hidden=torch.zeros(1, int(batch_size), self.recurrent_dim, device=device)
        )

    def _step_features(
        self,
        observations: torch.Tensor,
        previous_actions: torch.Tensor,
        previous_rewards: torch.Tensor,
        episode_starts: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = observations.shape[0]
        hidden = hidden * (~episode_starts).to(hidden.dtype).reshape(1, batch_size, 1)
        visual = self.observation_encoder(observations[:, None])[:, 0]
        previous_action = F.one_hot(
            previous_actions.clamp(min=0, max=self.n_actions),
            num_classes=self.n_actions + 1,
        ).to(dtype=visual.dtype)
        evidence = torch.cat(
            (
                visual,
                previous_action,
                previous_rewards[:, None].to(visual.dtype),
                episode_starts[:, None].to(visual.dtype),
            ),
            dim=-1,
        )
        output, next_hidden = self.recurrent(evidence[:, None], hidden)
        return output[:, 0], next_hidden

    def forward_step(
        self,
        batch: PrimitiveObservationBatch,
        state: BackbonePolicyState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, BackbonePolicyState]:
        batch.validate(
            observation_shape=self.observation_shape,
            n_actions=self.n_actions,
        )
        if batch.observations.shape[1] != 1:
            raise ValueError("Backbone forward_step requires one time step.")
        features, hidden = self._step_features(
            batch.observations[:, 0],
            batch.previous_actions[:, 0],
            batch.previous_rewards[:, 0],
            batch.episode_starts[:, 0],
            state.hidden,
        )
        return (
            self.actor_head(features),
            self.critic_head(features).squeeze(-1),
            features,
            BackbonePolicyState(hidden=hidden),
        )

    def forward_sequence(
        self,
        batch: PrimitiveObservationBatch,
        *,
        initial_state: BackbonePolicyState | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch.validate(
            observation_shape=self.observation_shape,
            n_actions=self.n_actions,
        )
        state = initial_state or self.initial_state(
            batch.observations.shape[0],
            device=batch.observations.device,
        )
        logits: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        features: list[torch.Tensor] = []
        hidden = state.hidden
        for step in range(batch.observations.shape[1]):
            feature, hidden = self._step_features(
                batch.observations[:, step],
                batch.previous_actions[:, step],
                batch.previous_rewards[:, step],
                batch.episode_starts[:, step],
                hidden,
            )
            logits.append(self.actor_head(feature))
            values.append(self.critic_head(feature).squeeze(-1))
            features.append(feature)
        return (
            torch.stack(logits, dim=1),
            torch.stack(values, dim=1),
            torch.stack(features, dim=1),
        )

    def architecture_manifest(self) -> dict[str, Any]:
        return {
            "model_class": type(self).__name__,
            "observation_shape": list(self.observation_shape),
            "n_actions": self.n_actions,
            "visual_embedding_dim": self.visual_embedding_dim,
            "recurrent_dim": self.recurrent_dim,
            "conv_channels": list(self.conv_channels),
            "encoder_spatial_mode": self.encoder_spatial_mode,
            "input_contract": (
                "own_default_local_observation+own_previous_action+"
                "own_previous_raw_reward+episode_start"
            ),
        }

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> "RecurrentIPPOBackbone":
        if manifest.get("model_class") != cls.__name__:
            raise ValueError("Manifest does not describe RecurrentIPPOBackbone.")
        return cls(
            manifest["observation_shape"],
            n_actions=int(manifest["n_actions"]),
            visual_embedding_dim=int(manifest["visual_embedding_dim"]),
            recurrent_dim=int(manifest["recurrent_dim"]),
            conv_channels=manifest["conv_channels"],
            encoder_spatial_mode=str(manifest["encoder_spatial_mode"]),
        )

    def prior_state_sha256(self) -> str:
        return "none"

    def compact_recurrent_parameters(self) -> None:
        self.recurrent.flatten_parameters()


@dataclass(frozen=True)
class BackboneTrainingSpec:
    seed: int
    total_environment_steps: int
    batch_size_envs: int
    rollout_steps: int
    update_epochs: int
    num_minibatches: int
    snapshot_steps: tuple[int, ...]
    metrics_interval_environment_steps: int
    stop_if_no_delivery_by_environment_steps: int
    learning_rate: float
    warmup_fraction: float
    gamma: float
    gae_lambda: float
    clip_coefficient: float
    entropy_coefficient: float
    value_coefficient: float
    gradient_clip_norm: float
    shaping_horizon_environment_steps: int
    partner_family_definition: Mapping[str, str]
    partner_family_spec_sha256: str
    training_config_sha256: str
    environment_config_sha256: str
    training_implementation_sha256: str
    training_implementation_dependencies: Mapping[str, Any]
    training_run_id: str
    output_dir: Path

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "BackboneTrainingSpec":
        training = payload.get("training")
        if not isinstance(training, Mapping):
            raise TypeError("Backbone config requires a training mapping.")
        family_definition = dict(payload.get("partner_family") or {})
        if family_definition != IPPO_PARTNER_FAMILY_DEFINITION:
            raise ValueError(
                "Backbone partner_family must match the registered IPPO family."
            )
        raw_model = payload.get("model")
        if not isinstance(raw_model, Mapping):
            raise TypeError("IPPO partner training requires a model mapping.")
        ippo_partner_architecture_from_config(raw_model)
        config_sha256 = str(payload.get("_source_config_sha256", ""))
        environment_sha256 = str(payload.get("_environment_config_sha256", ""))
        current_implementation_dependencies = repository_source_dependency_closure(
            IPPO_PARTNER_TRAINING_DEPENDENCIES
        )
        raw_implementation_dependencies = payload.get(
            "_training_implementation_dependencies"
        )
        if not isinstance(raw_implementation_dependencies, Mapping) or dict(
            raw_implementation_dependencies
        ) != current_implementation_dependencies:
            raise ValueError(
                "Backbone training dependency closure differs from the registered "
                "repository sources."
            )
        implementation_sha256 = str(
            payload.get("_training_implementation_sha256", "")
        )
        if implementation_sha256 != canonical_mapping_sha256(
            current_implementation_dependencies
        ):
            raise ValueError(
                "Backbone training implementation hash does not match its "
                "dependency closure."
            )
        for name, value in (
            ("training config", config_sha256),
            ("environment config", environment_sha256),
            ("training implementation", implementation_sha256),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"Backbone {name} requires a SHA-256 digest.")
        training_seed = int(payload["seed"])
        total = _positive_int(training["total_environment_steps"], name="total steps")
        batch = _positive_int(training["batch_size_envs"], name="batch_size_envs")
        if total != 30_000_000 or total % batch:
            raise ValueError("Backbone formal budget must be 30M and batch-divisible.")
        snapshots = tuple(int(item) for item in training["snapshot_steps"])
        if snapshots != (7_500_000, 15_000_000, 22_500_000, 30_000_000):
            raise ValueError("Backbone snapshots must be 7.5M, 15M, 22.5M and 30M.")
        metrics = _positive_int(
            training["metrics_interval_environment_steps"],
            name="metrics interval",
        )
        stop_step = _positive_int(
            training["stop_if_no_delivery_by_environment_steps"],
            name="stop step",
        )
        for boundary in (*snapshots, metrics, stop_step):
            if boundary % batch:
                raise ValueError("Every backbone boundary must be batch-divisible.")
        warmup = _finite_float(training["warmup_fraction"], name="warmup_fraction")
        if not 0.0 <= warmup < 1.0:
            raise ValueError("warmup_fraction must lie in [0,1).")
        shaping_horizon = _positive_int(
            training["shaping_horizon_environment_steps"],
            name="shaping horizon",
        )
        if shaping_horizon != 15_000_000:
            raise ValueError(
                "Registered IPPO partner training removes shaping at 15M steps."
            )
        return cls(
            seed=training_seed,
            total_environment_steps=total,
            batch_size_envs=batch,
            rollout_steps=_positive_int(training["rollout_steps"], name="rollout_steps"),
            update_epochs=_positive_int(training["update_epochs"], name="update_epochs"),
            num_minibatches=_positive_int(
                training["num_minibatches"], name="num_minibatches"
            ),
            snapshot_steps=snapshots,
            metrics_interval_environment_steps=metrics,
            stop_if_no_delivery_by_environment_steps=stop_step,
            learning_rate=_finite_float(training["learning_rate"], name="learning_rate"),
            warmup_fraction=warmup,
            gamma=_finite_float(training["gamma"], name="gamma"),
            gae_lambda=_finite_float(training["gae_lambda"], name="gae_lambda"),
            clip_coefficient=_finite_float(
                training["clip_coefficient"], name="clip_coefficient"
            ),
            entropy_coefficient=_finite_float(
                training["entropy_coefficient"], name="entropy_coefficient"
            ),
            value_coefficient=_finite_float(
                training["value_coefficient"], name="value_coefficient"
            ),
            gradient_clip_norm=_finite_float(
                training["gradient_clip_norm"], name="gradient_clip_norm"
            ),
            shaping_horizon_environment_steps=shaping_horizon,
            partner_family_definition=family_definition,
            partner_family_spec_sha256=IPPO_PARTNER_FAMILY_SPEC_SHA256,
            training_config_sha256=config_sha256,
            environment_config_sha256=environment_sha256,
            training_implementation_sha256=implementation_sha256,
            training_implementation_dependencies=(
                current_implementation_dependencies
            ),
            training_run_id=partner_training_run_id(
                family_spec_sha256=IPPO_PARTNER_FAMILY_SPEC_SHA256,
                training_seed=training_seed,
                training_config_sha256=config_sha256,
                environment_config_sha256=environment_sha256,
                training_implementation_sha256=implementation_sha256,
            ),
            output_dir=Path(payload["output_dir"]),
        )

    def shaping_factor(self, environment_steps: int) -> float:
        return max(
            0.0,
            1.0 - float(environment_steps) / self.shaping_horizon_environment_steps,
        )

    def scheduled_learning_rate(self, environment_steps: int) -> float:
        progress = min(1.0, max(0.0, environment_steps / self.total_environment_steps))
        if self.warmup_fraction > 0.0 and progress < self.warmup_fraction:
            return self.learning_rate * progress / self.warmup_fraction
        remaining = (1.0 - progress) / max(1.0 - self.warmup_fraction, 1e-12)
        return self.learning_rate * max(0.0, remaining)


@dataclass
class _BackboneMetricWindow:
    environment_steps: int = 0
    raw_reward_sum: float = 0.0
    correct_delivery_count: int = 0
    wrong_delivery_count: int = 0
    indicator_activation_count: int = 0
    ambiguous_reward_step_count: int = 0
    shaped_reward_sum: float = 0.0
    losses: list[float] = None  # type: ignore[assignment]
    actor_losses: list[float] = None  # type: ignore[assignment]
    critic_losses: list[float] = None  # type: ignore[assignment]
    entropies: list[float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.losses = []
        self.actor_losses = []
        self.critic_losses = []
        self.entropies = []

    def record_step(
        self,
        raw_reward: np.ndarray,
        shaped_reward_sum: float,
    ) -> None:
        events = decompose_raw_reward_events(raw_reward)
        self.environment_steps += int(raw_reward.size)
        self.raw_reward_sum += float(np.asarray(raw_reward).sum())
        self.correct_delivery_count += events.correct_delivery_count
        self.wrong_delivery_count += events.wrong_delivery_count
        self.indicator_activation_count += events.indicator_activation_count
        self.ambiguous_reward_step_count += events.ambiguous_step_count
        self.shaped_reward_sum += float(shaped_reward_sum)

    def build_row(self, *, total_steps: int, episodes: int, learning_rate: float) -> dict[str, Any]:
        def mean(values: list[float]) -> float | None:
            return None if not values else float(np.mean(np.asarray(values)))
        row = {
            "schema_version": "path_c_backbone_metrics_v1",
            "total_environment_steps": int(total_steps),
            "window_environment_steps": int(self.environment_steps),
            "cumulative_episodes": int(episodes),
            "learning_rate": float(learning_rate),
            "raw_reward_sum": float(self.raw_reward_sum),
            "correct_delivery_count": int(self.correct_delivery_count),
            "wrong_delivery_count": int(self.wrong_delivery_count),
            "indicator_activation_count": int(self.indicator_activation_count),
            "ambiguous_reward_step_count": int(self.ambiguous_reward_step_count),
            "environment_native_shaped_reward_sum": float(self.shaped_reward_sum),
            "ppo_loss_mean": mean(self.losses),
            "actor_loss_mean": mean(self.actor_losses),
            "critic_loss_mean": mean(self.critic_losses),
            "actor_entropy_mean": mean(self.entropies),
        }
        self.__init__()
        return row


class RecurrentIPPOBackboneTrainer:
    """Official-recipe recurrent IPPO self-play with shared parameters."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.spec = BackboneTrainingSpec.from_mapping(config)
        self.env_config = StandardEnvConfig.from_mapping(config["environment"])
        self.device = torch.device(str(config.get("device", "cuda")))
        self.model_config = dict(config["model"])
        torch.manual_seed(
            derive_ocv2_execution_seed(
                derive_standard_seed(self.spec.seed, "backbone", "torch")
            )
        )
        self.model = RecurrentIPPOBackbone(
            self.model_config["observation_shape"],
            n_actions=6,
            visual_embedding_dim=int(self.model_config["visual_embedding_dim"]),
            recurrent_dim=int(self.model_config["recurrent_dim"]),
            conv_channels=self.model_config["conv_channels"],
            encoder_spatial_mode=str(self.model_config["encoder_spatial_mode"]),
        ).to(self.device)
        if self.model.architecture_manifest() != ippo_partner_architecture_from_config(
            self.model_config
        ):
            raise RuntimeError(
                "IPPO partner model differs from its registered architecture config."
            )
        self.pool = BatchedEnvPool(
            self.env_config.make_adapter().env,
            self.spec.batch_size_envs,
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.spec.learning_rate)
        self.rng = np.random.default_rng(
            derive_ocv2_execution_seed(
                derive_standard_seed(self.spec.seed, "backbone", "policy")
            )
        )
        self.output_dir = self.spec.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "training_metrics.jsonl"

    def _partner_generation_metadata(self) -> dict[str, Any]:
        """Return immutable fields that prove how this partner run was generated."""

        return {
            "layout": self.env_config.layout,
            "partner_family": dict(self.spec.partner_family_definition),
            "partner_family_id": self.spec.partner_family_definition["family_id"],
            "partner_family_spec_sha256": self.spec.partner_family_spec_sha256,
            "training_config_sha256": self.spec.training_config_sha256,
            "environment_config_sha256": self.spec.environment_config_sha256,
            "training_implementation_sha256": (
                self.spec.training_implementation_sha256
            ),
            "training_implementation_dependencies": dict(
                self.spec.training_implementation_dependencies
            ),
            "training_run_id": self.spec.training_run_id,
        }

    def _reset(self) -> tuple[dict[str, np.ndarray], np.ndarray]:
        seeds = np.asarray(
            [
                derive_ocv2_execution_seed(
                    derive_standard_seed(self.spec.seed, "backbone", "env", index, 0)
                )
                for index in range(self.spec.batch_size_envs)
            ],
            dtype=np.uint32,
        )
        self.pool.reset(seeds)
        return self.pool.snapshot_obs(), np.zeros(self.spec.batch_size_envs, dtype=np.int64)

    def _batch_from_rollout(
        self,
        storage: Mapping[str, torch.Tensor],
        rows: torch.Tensor,
    ) -> PrimitiveObservationBatch:
        return PrimitiveObservationBatch(
            observations=storage["observations"][:, rows].transpose(0, 1),
            previous_actions=storage["previous_actions"][:, rows].transpose(0, 1),
            previous_rewards=storage["previous_rewards"][:, rows].transpose(0, 1),
            episode_starts=storage["episode_starts"][:, rows].transpose(0, 1),
            valid_actions=torch.ones(
                rows.numel(),
                storage["observations"].shape[0],
                6,
                dtype=torch.bool,
                device=self.device,
            ),
            time_mask=torch.ones(
                rows.numel(),
                storage["observations"].shape[0],
                dtype=torch.bool,
                device=self.device,
            ),
            lengths=torch.full(
                (rows.numel(),),
                storage["observations"].shape[0],
                dtype=torch.long,
                device=self.device,
            ),
        )

    def _ppo_update(
        self,
        storage: Mapping[str, torch.Tensor],
        initial_hidden: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        metric: _BackboneMetricWindow,
        environment_steps: int,
    ) -> int:
        sequence_count = storage["actions"].shape[1]
        if sequence_count % self.spec.num_minibatches:
            raise ValueError("Two-slot rollout count must divide num_minibatches.")
        updates = 0
        for group in self.optimizer.param_groups:
            group["lr"] = self.spec.scheduled_learning_rate(environment_steps)
        for _ in range(self.spec.update_epochs):
            order = self.rng.permutation(sequence_count)
            for indices in np.array_split(order, self.spec.num_minibatches):
                rows = torch.as_tensor(indices, dtype=torch.long, device=self.device)
                batch = self._batch_from_rollout(storage, rows)
                logits, values, _ = self.model.forward_sequence(
                    batch,
                    initial_state=BackbonePolicyState(
                        hidden=initial_hidden[:, rows],
                    ),
                )
                distribution = torch.distributions.Categorical(logits=logits)
                actions = storage["actions"][:, rows].transpose(0, 1)
                old_log_prob = storage["log_prob"][:, rows].transpose(0, 1)
                new_log_prob = distribution.log_prob(actions)
                ratio = torch.exp(new_log_prob - old_log_prob)
                minibatch_advantage = advantages[:, rows].transpose(0, 1)
                normalized_advantage = (
                    minibatch_advantage - minibatch_advantage.mean()
                ) / (minibatch_advantage.std(unbiased=False) + 1e-8)
                unclipped = ratio * normalized_advantage
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.spec.clip_coefficient,
                    1.0 + self.spec.clip_coefficient,
                ) * normalized_advantage
                actor_loss = -torch.minimum(unclipped, clipped).mean()
                old_values = storage["values"][:, rows].transpose(0, 1)
                return_targets = returns[:, rows].transpose(0, 1)
                clipped_values = old_values + torch.clamp(
                    values - old_values,
                    -self.spec.clip_coefficient,
                    self.spec.clip_coefficient,
                )
                value_loss_unclipped = (values - return_targets).square()
                value_loss_clipped = (clipped_values - return_targets).square()
                critic_loss = 0.5 * torch.maximum(
                    value_loss_unclipped,
                    value_loss_clipped,
                ).mean()
                entropy = distribution.entropy().mean()
                loss = (
                    actor_loss
                    + self.spec.value_coefficient * critic_loss
                    - self.spec.entropy_coefficient * entropy
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.spec.gradient_clip_norm,
                )
                self.optimizer.step()
                metric.losses.append(float(loss.detach().cpu()))
                metric.actor_losses.append(float(actor_loss.detach().cpu()))
                metric.critic_losses.append(float(critic_loss.detach().cpu()))
                metric.entropies.append(float(entropy.detach().cpu()))
                updates += 1
        return updates

    def run(self) -> dict[str, Any]:
        self.metrics_path.write_text("", encoding="utf-8")
        observations, episode_indices = self._reset()
        batch_size = self.spec.batch_size_envs
        previous_actions = {
            slot: np.full(batch_size, 6, dtype=np.int64) for slot in (0, 1)
        }
        previous_rewards = {
            slot: np.zeros(batch_size, dtype=np.float32) for slot in (0, 1)
        }
        episode_starts = np.ones(batch_size, dtype=bool)
        state = self.model.initial_state(2 * batch_size, device=self.device)
        environment_steps = episodes = gradient_updates = 0
        total_correct_deliveries = 0
        metric = _BackboneMetricWindow()
        snapshots: list[str] = []
        snapshot_bindings: list[dict[str, Any]] = []
        boundaries = set(self.spec.snapshot_steps) | {
            self.spec.stop_if_no_delivery_by_environment_steps,
            self.spec.total_environment_steps,
        }
        while environment_steps < self.spec.total_environment_steps:
            future_boundaries = [item for item in boundaries if item > environment_steps]
            next_boundary = min(future_boundaries) if future_boundaries else self.spec.total_environment_steps
            metric_boundary = (
                (environment_steps // self.spec.metrics_interval_environment_steps + 1)
                * self.spec.metrics_interval_environment_steps
            )
            next_boundary = min(next_boundary, metric_boundary)
            remaining_vector_steps = (next_boundary - environment_steps) // batch_size
            rollout_steps = min(self.spec.rollout_steps, remaining_vector_steps)
            if rollout_steps <= 0:
                raise RuntimeError("Backbone rollout cannot reach the next boundary.")
            initial_hidden = state.hidden.detach().clone()
            stored: dict[str, list[torch.Tensor]] = {
                name: []
                for name in (
                    "observations",
                    "previous_actions",
                    "previous_rewards",
                    "episode_starts",
                    "actions",
                    "log_prob",
                    "values",
                    "rewards",
                    "dones",
                )
            }
            for _ in range(rollout_steps):
                stacked_observation = np.concatenate(
                    (observations["agent_0"], observations["agent_1"]), axis=0
                )
                stacked_previous_actions = np.concatenate(
                    (previous_actions[0], previous_actions[1])
                )
                stacked_previous_rewards = np.concatenate(
                    (previous_rewards[0], previous_rewards[1])
                )
                stacked_starts = np.concatenate((episode_starts, episode_starts))
                batch = single_step_batch(
                    stacked_observation,
                    stacked_previous_actions,
                    stacked_previous_rewards,
                    stacked_starts,
                    n_actions=6,
                    device=self.device,
                )
                with torch.no_grad():
                    logits, values, _, state = self.model.forward_step(batch, state)
                    distribution = torch.distributions.Categorical(logits=logits)
                    actions = distribution.sample()
                    log_prob = distribution.log_prob(actions)
                action_array = actions.detach().cpu().numpy().astype(np.int64)
                next_observations, _, rewards_device, dones_device, info = self.pool.step_joint(
                    action_array[:batch_size], action_array[batch_size:]
                )
                raw_reward = shared_team_reward(rewards_device)
                shaped = shaped_rewards_by_agent(info, batch_size=batch_size)
                factor = self.spec.shaping_factor(environment_steps)
                training_reward = np.concatenate(
                    (raw_reward + factor * shaped[0], raw_reward + factor * shaped[1])
                ).astype(np.float32)
                dones = np.asarray(dones_device["__all__"], dtype=bool)
                metric.record_step(raw_reward, float(shaped[0].sum() + shaped[1].sum()))
                total_correct_deliveries += decompose_raw_reward_events(
                    raw_reward
                ).correct_delivery_count
                for name, value in (
                    ("observations", batch.observations[:, 0]),
                    ("previous_actions", batch.previous_actions[:, 0]),
                    ("previous_rewards", batch.previous_rewards[:, 0]),
                    ("episode_starts", batch.episode_starts[:, 0]),
                    ("actions", actions),
                    ("log_prob", log_prob),
                    ("values", values),
                    ("rewards", torch.as_tensor(training_reward, device=self.device)),
                    ("dones", torch.as_tensor(np.concatenate((dones, dones)), device=self.device)),
                ):
                    stored[name].append(value.detach())
                next_observations_np = {
                    key: np.asarray(value) for key, value in next_observations.items()
                }
                done_indices = np.flatnonzero(dones)
                if done_indices.size:
                    episodes += int(done_indices.size)
                    episode_indices[done_indices] += 1
                    reset_seeds = np.asarray(
                        [
                            derive_ocv2_execution_seed(
                                derive_standard_seed(
                                    self.spec.seed,
                                    "backbone",
                                    "env",
                                    int(index),
                                    int(episode_indices[index]),
                                )
                            )
                            for index in done_indices
                        ],
                        dtype=np.uint32,
                    )
                    self.pool.reset_indices(done_indices, reset_seeds)
                    next_observations_np = self.pool.snapshot_obs()
                previous_actions[0] = action_array[:batch_size].copy()
                previous_actions[1] = action_array[batch_size:].copy()
                previous_rewards[0] = raw_reward.copy()
                previous_rewards[1] = raw_reward.copy()
                for slot in (0, 1):
                    previous_actions[slot][dones] = 6
                    previous_rewards[slot][dones] = 0.0
                episode_starts = dones.copy()
                observations = next_observations_np
                environment_steps += batch_size
            next_batch = single_step_batch(
                np.concatenate((observations["agent_0"], observations["agent_1"])),
                np.concatenate((previous_actions[0], previous_actions[1])),
                np.concatenate((previous_rewards[0], previous_rewards[1])),
                np.concatenate((episode_starts, episode_starts)),
                n_actions=6,
                device=self.device,
            )
            with torch.no_grad():
                _, bootstrap_value, _, _ = self.model.forward_step(next_batch, state)
            storage = {name: torch.stack(values) for name, values in stored.items()}
            advantages = torch.zeros_like(storage["rewards"])
            gae = torch.zeros(2 * batch_size, device=self.device)
            next_value = bootstrap_value
            for step in reversed(range(rollout_steps)):
                nonterminal = (~storage["dones"][step]).to(torch.float32)
                delta = (
                    storage["rewards"][step]
                    + self.spec.gamma * next_value * nonterminal
                    - storage["values"][step]
                )
                gae = delta + self.spec.gamma * self.spec.gae_lambda * nonterminal * gae
                advantages[step] = gae
                next_value = storage["values"][step]
            returns = advantages + storage["values"]
            gradient_updates += self._ppo_update(
                storage,
                initial_hidden,
                advantages,
                returns,
                metric,
                environment_steps,
            )
            if environment_steps % self.spec.metrics_interval_environment_steps == 0:
                row = metric.build_row(
                    total_steps=environment_steps,
                    episodes=episodes,
                    learning_rate=self.spec.scheduled_learning_rate(environment_steps),
                )
                with self.metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            if environment_steps in self.spec.snapshot_steps:
                path = self.output_dir / "partner_pool" / f"step_{environment_steps}.pt"
                snapshot_binding = save_standard_checkpoint(
                    path,
                    self.model,
                    seed=self.spec.seed,
                    environment_steps=environment_steps,
                    episodes=episodes,
                    phase="backbone_snapshot",
                    extra_metadata=self._partner_generation_metadata(),
                )
                snapshots.append(str(path))
                snapshot_bindings.append(snapshot_binding)
            if (
                environment_steps == self.spec.stop_if_no_delivery_by_environment_steps
                and total_correct_deliveries == 0
            ):
                break
        stopped = environment_steps < self.spec.total_environment_steps
        final_path = self.output_dir / (
            "backbone_stopped_no_delivery.pt" if stopped else "backbone_final.pt"
        )
        final_checkpoint_binding = save_standard_checkpoint(
            final_path,
            self.model,
            seed=self.spec.seed,
            environment_steps=environment_steps,
            episodes=episodes,
            phase="backbone_stopped_no_delivery" if stopped else "backbone_final",
            extra_metadata=self._partner_generation_metadata(),
        )
        manifest = {
            "schema_version": "path_c_backbone_artifact_v2",
            "run_status": "stopped_no_delivery" if stopped else "completed",
            "effective_environment_steps": environment_steps,
            "effective_episodes": episodes,
            "gradient_updates": gradient_updates,
            "correct_delivery_count": total_correct_deliveries,
            "snapshots": snapshots,
            "snapshot_bindings": snapshot_bindings,
            "checkpoint": str(final_path),
            "checkpoint_binding": final_checkpoint_binding,
            "architecture": self.model.architecture_manifest(),
            "budget_accounting": "backbone formation cost reported separately from adaptation",
            "partner_family": dict(self.spec.partner_family_definition),
            "partner_family_spec_sha256": self.spec.partner_family_spec_sha256,
            "training_config_sha256": self.spec.training_config_sha256,
            "environment_config_sha256": self.spec.environment_config_sha256,
            "training_implementation_sha256": (
                self.spec.training_implementation_sha256
            ),
            "training_implementation_dependencies": dict(
                self.spec.training_implementation_dependencies
            ),
            "training_run_id": self.spec.training_run_id,
        }
        (self.output_dir / "training_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return manifest


def load_backbone_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    source_bytes = config_path.read_bytes()
    payload = yaml.safe_load(source_bytes.decode("utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema_version") != BACKBONE_CONFIG_SCHEMA_VERSION:
        raise ValueError("Unsupported backbone configuration.")
    config = dict(payload)
    environment_path = Path(config["environment_config"])
    if not environment_path.is_absolute():
        environment_path = (config_path.parent / environment_path).resolve()
    environment_bytes = environment_path.read_bytes()
    config["environment"] = yaml.safe_load(environment_bytes.decode("utf-8"))
    config["_source_config_sha256"] = hashlib.sha256(source_bytes).hexdigest()
    config["_environment_config_sha256"] = hashlib.sha256(
        environment_bytes
    ).hexdigest()
    implementation_dependencies = repository_source_dependency_closure(
        IPPO_PARTNER_TRAINING_DEPENDENCIES
    )
    config["_training_implementation_dependencies"] = implementation_dependencies
    config["_training_implementation_sha256"] = canonical_mapping_sha256(
        implementation_dependencies
    )
    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        config["output_dir"] = str((config_path.parent / output_dir).resolve())
    return config


def run_backbone_training(config_path: str | Path) -> dict[str, Any]:
    return RecurrentIPPOBackboneTrainer(load_backbone_config(config_path)).run()
