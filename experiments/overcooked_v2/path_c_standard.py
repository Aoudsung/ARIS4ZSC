"""Primitive-action Path C policy for the published OvercookedV2 protocol.

This module is intentionally independent of the legacy option, conditional-entropy,
global-state featurizer, and scripted-partner stack.  Every policy input is either the
agent's DEFAULT OvercookedV2 observation or that agent's own previous action/reward.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.overcooked_v2.env_adapter import OCV2Adapter
from experiments.overcooked_v2.path_c_sequence import (
    EvidenceBatch,
    RecurrentEnsembleQ,
    valid_action_dueling,
)
from experiments.overcooked_v2.residual_signature import (
    normalized_advantage_disagreement,
)


STANDARD_ENV_SCHEMA_VERSION = "path_c_standard_env_v1"
STANDARD_CHECKPOINT_SCHEMA_VERSION = "path_c_standard_checkpoint_v1"
PRIMITIVE_ACTION_NAMES = (
    "right",
    "down",
    "left",
    "up",
    "stay",
    "interact",
)


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
class StandardEnvConfig:
    """The exact environment semantics used by the published Test Time protocol."""

    layout: str
    max_steps: int = 400
    observation_type: str = "default"
    agent_view_size: int = 2
    negative_rewards: bool = True
    random_agent_positions: bool = True
    sample_recipe_on_delivery: bool = True
    indicate_successful_delivery: bool = True
    force_path_planning: bool = False
    random_reset: bool = False

    def __post_init__(self) -> None:
        if self.layout not in {"test_time_simple", "test_time_wide"}:
            raise ValueError("Standard Path C supports the two published Test Time layouts.")
        if int(self.max_steps) != 400:
            raise ValueError("The published Test Time protocol uses 400-step episodes.")
        if str(self.observation_type) != "default":
            raise ValueError("The standard path must consume DEFAULT local observations.")
        if int(self.agent_view_size) != 2:
            raise ValueError("The published Test Time protocol uses agent_view_size=2.")
        required_true = {
            "negative_rewards": self.negative_rewards,
            "random_agent_positions": self.random_agent_positions,
            "sample_recipe_on_delivery": self.sample_recipe_on_delivery,
            "indicate_successful_delivery": self.indicate_successful_delivery,
        }
        disabled = [name for name, enabled in required_true.items() if enabled is not True]
        if disabled:
            raise ValueError(
                "Standard Test Time environment requires true values for: "
                + ", ".join(disabled)
            )
        if self.force_path_planning is not False or self.random_reset is not False:
            raise ValueError(
                "Standard Test Time environment requires force_path_planning=False "
                "and random_reset=False."
            )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "StandardEnvConfig":
        if not isinstance(payload, Mapping):
            raise TypeError("Standard environment config must be a mapping.")
        schema_version = payload.get("schema_version", STANDARD_ENV_SCHEMA_VERSION)
        if schema_version != STANDARD_ENV_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported standard environment schema {schema_version!r}."
            )
        env = payload.get("env", payload)
        if not isinstance(env, Mapping):
            raise TypeError("Standard environment config env field must be a mapping.")
        fields = {
            "layout",
            "max_steps",
            "observation_type",
            "agent_view_size",
            "negative_rewards",
            "random_agent_positions",
            "sample_recipe_on_delivery",
            "indicate_successful_delivery",
            "force_path_planning",
            "random_reset",
        }
        unknown = set(env) - fields
        if unknown:
            raise ValueError(
                "Unknown standard environment field(s): "
                + ", ".join(sorted(str(item) for item in unknown))
            )
        return cls(**{key: env[key] for key in fields if key in env})

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": STANDARD_ENV_SCHEMA_VERSION,
            "env": {
                "layout": self.layout,
                "max_steps": int(self.max_steps),
                "observation_type": self.observation_type,
                "agent_view_size": int(self.agent_view_size),
                "negative_rewards": bool(self.negative_rewards),
                "random_agent_positions": bool(self.random_agent_positions),
                "sample_recipe_on_delivery": bool(self.sample_recipe_on_delivery),
                "indicate_successful_delivery": bool(
                    self.indicate_successful_delivery
                ),
                "force_path_planning": bool(self.force_path_planning),
                "random_reset": bool(self.random_reset),
            },
        }

    def make_adapter(self) -> OCV2Adapter:
        return OCV2Adapter(
            layout=self.layout,
            max_steps=self.max_steps,
            observation_type=self.observation_type,
            agent_view_size=self.agent_view_size,
            negative_rewards=self.negative_rewards,
            random_agent_positions=self.random_agent_positions,
            sample_recipe_on_delivery=self.sample_recipe_on_delivery,
            indicate_successful_delivery=self.indicate_successful_delivery,
            force_path_planning=self.force_path_planning,
            random_reset=self.random_reset,
            featurizer=None,
        )


@dataclass(frozen=True)
class PrimitiveObservationBatch:
    """Agent-local recurrent inputs with no partner action or simulator state."""

    observations: torch.Tensor
    previous_actions: torch.Tensor
    previous_rewards: torch.Tensor
    episode_starts: torch.Tensor
    valid_actions: torch.Tensor
    time_mask: torch.Tensor
    lengths: torch.Tensor

    def validate(
        self,
        *,
        observation_shape: Sequence[int],
        n_actions: int,
    ) -> "PrimitiveObservationBatch":
        expected_obs = tuple(int(item) for item in observation_shape)
        if self.observations.ndim != 5:
            raise ValueError("observations must have shape [B,S,H,W,C].")
        if tuple(self.observations.shape[2:]) != expected_obs:
            raise ValueError(
                f"Expected observation shape {expected_obs}; got "
                f"{tuple(self.observations.shape[2:])}."
            )
        batch_size, sequence_length = self.observations.shape[:2]
        state_shape = (batch_size, sequence_length)
        for name, value in (
            ("previous_actions", self.previous_actions),
            ("previous_rewards", self.previous_rewards),
            ("episode_starts", self.episode_starts),
            ("time_mask", self.time_mask),
        ):
            if tuple(value.shape) != state_shape:
                raise ValueError(f"{name} must have shape {state_shape}.")
        if tuple(self.valid_actions.shape) != (*state_shape, int(n_actions)):
            raise ValueError(
                f"valid_actions must have shape {(*state_shape, int(n_actions))}."
            )
        if tuple(self.lengths.shape) != (batch_size,):
            raise ValueError(f"lengths must have shape {(batch_size,)}.")
        if not torch.is_floating_point(self.observations):
            raise TypeError("observations must use a floating-point dtype.")
        if not bool(torch.isfinite(self.observations).all()):
            raise ValueError("observations must be finite.")
        if not bool(torch.isfinite(self.previous_rewards).all()):
            raise ValueError("previous_rewards must be finite.")
        if self.previous_actions.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise TypeError("previous_actions must use an integer dtype.")
        previous_actions = self.previous_actions.to(dtype=torch.long)
        if bool((previous_actions < 0).any()) or bool(
            (previous_actions > int(n_actions)).any()
        ):
            raise ValueError(
                "previous_actions must use primitive ids or the start padding id."
            )
        if self.episode_starts.dtype != torch.bool or self.time_mask.dtype != torch.bool:
            raise TypeError("episode_starts and time_mask must be boolean tensors.")
        if self.valid_actions.dtype != torch.bool:
            raise TypeError("valid_actions must be a boolean tensor.")
        if bool((self.lengths <= 0).any()) or bool(
            (self.lengths > sequence_length).any()
        ):
            raise ValueError("lengths must lie within the padded sequence length.")
        active_has_action = self.valid_actions.any(dim=-1) | ~self.time_mask
        if not bool(active_has_action.all()):
            raise ValueError("Every active state must expose a primitive action.")
        return self

    def to(self, device: torch.device | str) -> "PrimitiveObservationBatch":
        return PrimitiveObservationBatch(
            observations=self.observations.to(device),
            previous_actions=self.previous_actions.to(device),
            previous_rewards=self.previous_rewards.to(device),
            episode_starts=self.episode_starts.to(device),
            valid_actions=self.valid_actions.to(device),
            time_mask=self.time_mask.to(device),
            lengths=self.lengths.to(device),
        )


class LocalObservationEncoder(nn.Module):
    """Small convolutional encoder for one 5x5 local observation window."""

    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        *,
        conv_channels: Sequence[int] = (32, 32, 16),
    ) -> None:
        super().__init__()
        input_channels = _positive_int(input_channels, name="input_channels")
        output_dim = _positive_int(output_dim, name="output_dim")
        channels = tuple(
            _positive_int(item, name="conv_channels") for item in conv_channels
        )
        if len(channels) != 3:
            raise ValueError("conv_channels must contain exactly three widths.")
        self.input_channels = input_channels
        self.output_dim = output_dim
        self.conv_channels = channels
        self.convolutions = nn.Sequential(
            nn.Conv2d(input_channels, channels[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[2], output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 5:
            raise ValueError("LocalObservationEncoder expects [B,S,H,W,C].")
        if observations.shape[-1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} observation channels; got "
                f"{observations.shape[-1]}."
            )
        batch_size, sequence_length = observations.shape[:2]
        channels_first = observations.permute(0, 1, 4, 2, 3).reshape(
            batch_size * sequence_length,
            self.input_channels,
            observations.shape[2],
            observations.shape[3],
        )
        encoded = self.projection(self.convolutions(channels_first))
        return encoded.reshape(batch_size, sequence_length, self.output_dim)


@dataclass(frozen=True)
class PrimitiveCoreState:
    hidden: tuple[torch.Tensor, ...]


class PrimitiveVisualCore(nn.Module):
    """Trainable visual front-end followed by the shared Path C GRU ensemble."""

    def __init__(
        self,
        observation_shape: Sequence[int],
        n_actions: int,
        *,
        n_heads: int,
        visual_embedding_dim: int,
        encoder_dim: int,
        recurrent_dim: int,
        conv_channels: Sequence[int],
    ) -> None:
        super().__init__()
        shape = tuple(_positive_int(item, name="observation_shape") for item in observation_shape)
        if len(shape) != 3:
            raise ValueError("observation_shape must be [height,width,channels].")
        self.observation_shape = shape
        self.n_actions = _positive_int(n_actions, name="n_actions")
        self.n_heads = _positive_int(n_heads, name="n_heads")
        self.visual_embedding_dim = _positive_int(
            visual_embedding_dim,
            name="visual_embedding_dim",
        )
        self.encoder_dim = _positive_int(encoder_dim, name="encoder_dim")
        self.recurrent_dim = _positive_int(recurrent_dim, name="recurrent_dim")
        self.conv_channels = tuple(int(item) for item in conv_channels)
        self.observation_encoder = LocalObservationEncoder(
            shape[-1],
            self.visual_embedding_dim,
            conv_channels=self.conv_channels,
        )
        evidence_dim = self.visual_embedding_dim + self.n_actions + 1 + 2
        self.recurrent = RecurrentEnsembleQ(
            evidence_dim=evidence_dim,
            n_actions=self.n_actions,
            n_heads=self.n_heads,
            encoder_dim=self.encoder_dim,
            recurrent_dim=self.recurrent_dim,
            prior_scale=0.0,
        )

    def _evidence(self, batch: PrimitiveObservationBatch) -> EvidenceBatch:
        # The public entry points (forward_sequence / forward_step) validate the
        # batch once; validating again here would force redundant device syncs
        # on every acting step.
        visual = self.observation_encoder(batch.observations)
        previous_action = F.one_hot(
            batch.previous_actions.to(dtype=torch.long),
            num_classes=self.n_actions + 1,
        ).to(dtype=visual.dtype)
        evidence = torch.cat(
            (
                visual,
                previous_action,
                batch.previous_rewards.unsqueeze(-1).to(dtype=visual.dtype),
                batch.episode_starts.unsqueeze(-1).to(dtype=visual.dtype),
            ),
            dim=-1,
        )
        return EvidenceBatch(
            evidence=evidence,
            valid_actions=batch.valid_actions,
            lengths=batch.lengths,
            spec_sha256=None,
        )

    def forward_sequence(
        self,
        batch: PrimitiveObservationBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch.validate(
            observation_shape=self.observation_shape,
            n_actions=self.n_actions,
        )
        q_values, representation = self.recurrent.forward_sequence(self._evidence(batch))
        q_values = q_values.masked_fill(~batch.time_mask[:, :, None, None], 0.0)
        representation = representation.masked_fill(
            ~batch.time_mask[:, :, None],
            0.0,
        )
        return q_values, representation

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> PrimitiveCoreState:
        batch_size = _positive_int(batch_size, name="batch_size")
        hidden = tuple(
            torch.zeros(1, batch_size, self.recurrent_dim, device=device, dtype=dtype)
            for _ in range(self.n_heads)
        )
        return PrimitiveCoreState(hidden=hidden)

    def compact_recurrent_parameters(self) -> None:
        """Restore cuDNN's compact GRU weight layout after copies or loading."""

        for gru in self.recurrent.head_grus:
            gru.flatten_parameters()

    def forward_step(
        self,
        batch: PrimitiveObservationBatch,
        state: PrimitiveCoreState,
    ) -> tuple[torch.Tensor, torch.Tensor, PrimitiveCoreState]:
        batch.validate(
            observation_shape=self.observation_shape,
            n_actions=self.n_actions,
        )
        if batch.observations.shape[1] != 1:
            raise ValueError("forward_step requires exactly one recurrent step.")
        if len(state.hidden) != self.n_heads:
            raise ValueError("PrimitiveCoreState head count differs from the model.")
        evidence = self._evidence(batch)
        encoded = self.recurrent.shared_encoder(evidence.evidence)
        reset = batch.episode_starts[:, 0].reshape(1, -1, 1)
        learned_states: list[torch.Tensor] = []
        q_values: list[torch.Tensor] = []
        next_hidden: list[torch.Tensor] = []
        for head_idx, gru in enumerate(self.recurrent.head_grus):
            hidden = state.hidden[head_idx]
            if tuple(hidden.shape) != (
                1,
                batch.observations.shape[0],
                self.recurrent_dim,
            ):
                raise ValueError("PrimitiveCoreState hidden shape differs from the model.")
            hidden = torch.where(reset, torch.zeros_like(hidden), hidden)
            states, head_hidden = gru(encoded, hidden)
            learned_states.append(states[:, 0])
            next_hidden.append(head_hidden)
            q_values.append(
                valid_action_dueling(
                    self.recurrent.value_heads[head_idx](states[:, 0]),
                    self.recurrent.advantage_heads[head_idx](states[:, 0]),
                    batch.valid_actions[:, 0],
                    invalid_action_value=self.recurrent.invalid_action_value,
                )
            )
        q_stack = torch.stack(q_values, dim=1)
        representation = torch.stack(learned_states, dim=1).reshape(
            batch.observations.shape[0],
            -1,
        )
        return q_stack, representation, PrimitiveCoreState(tuple(next_hidden))


@dataclass(frozen=True)
class PrimitivePolicyState:
    learned: PrimitiveCoreState
    prior: PrimitiveCoreState | None


def _slice_core_state(core: PrimitiveCoreState, rows: torch.Tensor) -> PrimitiveCoreState:
    return PrimitiveCoreState(hidden=tuple(hidden[:, rows] for hidden in core.hidden))


def slice_policy_state(
    state: PrimitivePolicyState,
    rows: np.ndarray,
) -> tuple[PrimitivePolicyState, torch.Tensor]:
    """Row-slice a full-batch recurrent state for a sub-batch forward pass."""

    device = state.learned.hidden[0].device
    row_index = torch.as_tensor(np.asarray(rows, dtype=np.int64), device=device)
    sliced = PrimitivePolicyState(
        learned=_slice_core_state(state.learned, row_index),
        prior=None if state.prior is None else _slice_core_state(state.prior, row_index),
    )
    return sliced, row_index


def scatter_policy_state_(
    state: PrimitivePolicyState,
    row_index: torch.Tensor,
    sub_state: PrimitivePolicyState,
) -> None:
    """Write a sub-batch next state back into the full-batch state tensors.

    In-place is safe here: acting states are produced under ``torch.no_grad``
    and are owned exclusively by one rollout loop.
    """

    if (state.prior is None) != (sub_state.prior is None):
        raise ValueError("Full and sub-batch policy states disagree about the prior.")
    with torch.no_grad():
        core_pairs: list[tuple[PrimitiveCoreState, PrimitiveCoreState]] = [
            (state.learned, sub_state.learned)
        ]
        if state.prior is not None and sub_state.prior is not None:
            core_pairs.append((state.prior, sub_state.prior))
        for full_core, sub_core in core_pairs:
            for full_hidden, sub_hidden in zip(
                full_core.hidden,
                sub_core.hidden,
                strict=True,
            ):
                full_hidden[:, row_index] = sub_hidden


class PrimitiveRecurrentEnsembleQ(nn.Module):
    """Local-observation Path C network acting on six primitive actions."""

    def __init__(
        self,
        observation_shape: Sequence[int],
        *,
        n_actions: int = 6,
        n_heads: int = 5,
        visual_embedding_dim: int = 128,
        encoder_dim: int = 128,
        recurrent_dim: int = 128,
        conv_channels: Sequence[int] = (32, 32, 16),
        prior_scale: float = 0.0,
        prior_seed: int = 0,
    ) -> None:
        super().__init__()
        self.observation_shape = tuple(int(item) for item in observation_shape)
        self.n_actions = _positive_int(n_actions, name="n_actions")
        if self.n_actions != len(PRIMITIVE_ACTION_NAMES):
            raise ValueError("The published OvercookedV2 action space has six actions.")
        self.n_heads = _positive_int(n_heads, name="n_heads")
        self.visual_embedding_dim = _positive_int(
            visual_embedding_dim,
            name="visual_embedding_dim",
        )
        self.encoder_dim = _positive_int(encoder_dim, name="encoder_dim")
        self.recurrent_dim = _positive_int(recurrent_dim, name="recurrent_dim")
        self.conv_channels = tuple(int(item) for item in conv_channels)
        self.prior_scale = _finite_float(prior_scale, name="prior_scale")
        if self.prior_scale < 0.0:
            raise ValueError("prior_scale must be non-negative.")
        self.prior_seed = int(prior_seed)
        if self.prior_seed < 0:
            raise ValueError("prior_seed must be non-negative.")
        self.learned = PrimitiveVisualCore(
            self.observation_shape,
            self.n_actions,
            n_heads=self.n_heads,
            visual_embedding_dim=self.visual_embedding_dim,
            encoder_dim=self.encoder_dim,
            recurrent_dim=self.recurrent_dim,
            conv_channels=self.conv_channels,
        )
        self.fixed_prior: PrimitiveVisualCore | None = None
        if self.prior_scale > 0.0:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.prior_seed)
                self.fixed_prior = PrimitiveVisualCore(
                    self.observation_shape,
                    self.n_actions,
                    n_heads=self.n_heads,
                    visual_embedding_dim=self.visual_embedding_dim,
                    encoder_dim=self.encoder_dim,
                    recurrent_dim=self.recurrent_dim,
                    conv_channels=self.conv_channels,
                )
            for parameter in self.fixed_prior.parameters():
                parameter.requires_grad_(False)
            self.fixed_prior.eval()

    def train(self, mode: bool = True) -> "PrimitiveRecurrentEnsembleQ":
        super().train(mode)
        if self.fixed_prior is not None:
            self.fixed_prior.eval()
        return self

    def forward_sequence(
        self,
        batch: PrimitiveObservationBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        learned_q, representation = self.learned.forward_sequence(batch)
        if self.fixed_prior is None:
            return learned_q, representation
        with torch.no_grad():
            prior_q, _ = self.fixed_prior.forward_sequence(batch)
        valid = batch.valid_actions.unsqueeze(2)
        q_values = (learned_q + self.prior_scale * prior_q).masked_fill(
            ~valid,
            -torch.inf,
        )
        q_values = q_values.masked_fill(~batch.time_mask[:, :, None, None], 0.0)
        return q_values, representation

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
    ) -> PrimitivePolicyState:
        learned = self.learned.initial_state(batch_size, device=device)
        prior = (
            None
            if self.fixed_prior is None
            else self.fixed_prior.initial_state(batch_size, device=device)
        )
        return PrimitivePolicyState(learned=learned, prior=prior)

    def compact_recurrent_parameters(self) -> None:
        """Compact every active recurrent core after a copy or checkpoint load."""

        self.learned.compact_recurrent_parameters()
        if self.fixed_prior is not None:
            self.fixed_prior.compact_recurrent_parameters()

    def forward_step(
        self,
        batch: PrimitiveObservationBatch,
        state: PrimitivePolicyState,
    ) -> tuple[torch.Tensor, torch.Tensor, PrimitivePolicyState]:
        learned_q, representation, learned_state = self.learned.forward_step(
            batch,
            state.learned,
        )
        if self.fixed_prior is None:
            if state.prior is not None:
                raise ValueError("Policy state supplies a prior to a model without one.")
            return (
                learned_q,
                representation,
                PrimitivePolicyState(learned=learned_state, prior=None),
            )
        if state.prior is None:
            raise ValueError("Policy state is missing the fixed-prior recurrent state.")
        with torch.no_grad():
            prior_q, _, prior_state = self.fixed_prior.forward_step(batch, state.prior)
        q_values = (learned_q + self.prior_scale * prior_q).masked_fill(
            ~batch.valid_actions[:, 0, None, :],
            -torch.inf,
        )
        return (
            q_values,
            representation,
            PrimitivePolicyState(learned=learned_state, prior=prior_state),
        )

    def prior_state_sha256(self) -> str | None:
        if self.fixed_prior is None:
            return None
        digest = hashlib.sha256()
        for name, tensor in sorted(self.fixed_prior.state_dict().items()):
            value = tensor.detach().cpu().contiguous()
            digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode("utf-8"))
            digest.update(value.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def assert_same_fixed_prior(self, other: "PrimitiveRecurrentEnsembleQ") -> None:
        if not isinstance(other, PrimitiveRecurrentEnsembleQ):
            raise TypeError("Target model must be PrimitiveRecurrentEnsembleQ.")
        if self.prior_scale != other.prior_scale:
            raise ValueError("Online and target models use different prior scales.")
        if self.prior_state_sha256() != other.prior_state_sha256():
            raise ValueError("Online and target models use different fixed priors.")

    def architecture_manifest(self) -> dict[str, Any]:
        return {
            "model_class": "PrimitiveRecurrentEnsembleQ",
            "observation_shape": list(self.observation_shape),
            "primitive_action_names": list(PRIMITIVE_ACTION_NAMES),
            "n_actions": int(self.n_actions),
            "n_heads": int(self.n_heads),
            "visual_embedding_dim": int(self.visual_embedding_dim),
            "encoder_dim": int(self.encoder_dim),
            "recurrent_dim": int(self.recurrent_dim),
            "conv_channels": list(self.conv_channels),
            "prior_scale": float(self.prior_scale),
            "prior_seed": int(self.prior_seed),
            "input_contract": (
                "own_default_local_observation+own_previous_action+"
                "own_previous_reward+episode_start"
            ),
        }

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> "PrimitiveRecurrentEnsembleQ":
        if manifest.get("model_class") != "PrimitiveRecurrentEnsembleQ":
            raise ValueError("Checkpoint architecture is not a standard Path C model.")
        return cls(
            observation_shape=manifest["observation_shape"],
            n_actions=int(manifest["n_actions"]),
            n_heads=int(manifest["n_heads"]),
            visual_embedding_dim=int(manifest["visual_embedding_dim"]),
            encoder_dim=int(manifest["encoder_dim"]),
            recurrent_dim=int(manifest["recurrent_dim"]),
            conv_channels=manifest["conv_channels"],
            prior_scale=float(manifest["prior_scale"]),
            prior_seed=int(manifest["prior_seed"]),
        )


@dataclass(frozen=True)
class PrimitiveActionDecision:
    actions: np.ndarray
    is_probe: np.ndarray
    is_exploration: np.ndarray


@torch.no_grad()
def choose_primitive_actions(
    q_values: torch.Tensor,
    valid_actions: torch.Tensor,
    *,
    rng: np.random.Generator,
    epsilon: float = 0.0,
    probe_enabled: bool = False,
    disagreement_threshold: float | None = None,
    return_floor: float | None = None,
    disagreement_stat: str = "variance",
) -> PrimitiveActionDecision:
    """Choose primitive actions from ensemble Q values without external labels."""

    if q_values.ndim != 3:
        raise ValueError("q_values must have shape [B,K,A].")
    if valid_actions.shape != (q_values.shape[0], q_values.shape[2]):
        raise ValueError("valid_actions must have shape [B,A].")
    if q_values.shape[1] <= 0 or q_values.shape[2] <= 0:
        raise ValueError("q_values must contain ensemble heads and actions.")
    epsilon = _finite_float(epsilon, name="epsilon")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must lie in [0,1].")
    for name, value in (
        ("disagreement_threshold", disagreement_threshold),
        ("return_floor", return_floor),
    ):
        if value is not None:
            _finite_float(value, name=name)
    valid = valid_actions.bool()
    if not bool(valid.any(dim=-1).all()):
        raise ValueError("Every policy row requires at least one valid action.")
    mean_q = q_values.mean(dim=1).masked_fill(~valid, -torch.inf)
    greedy = mean_q.argmax(dim=-1)
    selected = greedy.clone()
    probe_mask = torch.zeros(q_values.shape[0], dtype=torch.bool, device=q_values.device)
    if probe_enabled:
        # The disagreement statistic is non-negative, so a missing or
        # non-positive threshold would turn every step into a probe and the
        # greedy branch would never act. Fail loudly instead of degenerating.
        if disagreement_threshold is None or float(disagreement_threshold) <= 0.0:
            raise ValueError(
                "Enabled probing requires a strictly positive disagreement_threshold."
            )
        disagreement = normalized_advantage_disagreement(
            q_values,
            option_mask=valid,
            stat=disagreement_stat,
        )["per_option"].masked_fill(~valid, -torch.inf)
        probe_actions = disagreement.argmax(dim=-1)
        candidate_score = disagreement.gather(1, probe_actions[:, None]).squeeze(1)
        candidate_q = mean_q.gather(1, probe_actions[:, None]).squeeze(1)
        probe_mask = candidate_score >= float(disagreement_threshold)
        if return_floor is not None:
            probe_mask &= candidate_q >= float(return_floor)
        selected = torch.where(probe_mask, probe_actions, selected)
    exploration = np.asarray(
        rng.random(q_values.shape[0]) < epsilon,
        dtype=bool,
    )
    selected_np = selected.detach().cpu().numpy().astype(np.int64, copy=True)
    valid_np = valid.detach().cpu().numpy()
    for row in np.flatnonzero(exploration):
        selected_np[row] = int(rng.choice(np.flatnonzero(valid_np[row])))
    probe_np = probe_mask.detach().cpu().numpy().astype(bool)
    probe_np[exploration] = False
    return PrimitiveActionDecision(
        actions=selected_np,
        is_probe=probe_np,
        is_exploration=exploration,
    )


def single_step_batch(
    observations: np.ndarray,
    previous_actions: np.ndarray,
    previous_rewards: np.ndarray,
    episode_starts: np.ndarray,
    *,
    n_actions: int = 6,
    device: torch.device | str,
) -> PrimitiveObservationBatch:
    # No defensive host copy: torch.as_tensor with a device target copies the
    # buffer anyway, and callers hand in freshly materialized arrays.
    observation_array = np.asarray(observations, dtype=np.float32)
    previous_action_array = np.asarray(previous_actions, dtype=np.int64)
    previous_reward_array = np.asarray(previous_rewards, dtype=np.float32)
    episode_start_array = np.asarray(episode_starts, dtype=bool)
    obs = torch.as_tensor(observation_array, dtype=torch.float32, device=device)
    if obs.ndim != 4:
        raise ValueError("single_step_batch observations must have shape [B,H,W,C].")
    batch_size = obs.shape[0]
    return PrimitiveObservationBatch(
        observations=obs.unsqueeze(1),
        previous_actions=torch.as_tensor(
            previous_action_array,
            dtype=torch.long,
            device=device,
        ).reshape(batch_size, 1),
        previous_rewards=torch.as_tensor(
            previous_reward_array,
            dtype=torch.float32,
            device=device,
        ).reshape(batch_size, 1),
        episode_starts=torch.as_tensor(
            episode_start_array,
            dtype=torch.bool,
            device=device,
        ).reshape(batch_size, 1),
        valid_actions=torch.ones(
            batch_size,
            1,
            n_actions,
            dtype=torch.bool,
            device=device,
        ),
        time_mask=torch.ones(
            batch_size,
            1,
            dtype=torch.bool,
            device=device,
        ),
        lengths=torch.ones(batch_size, dtype=torch.long, device=device),
    )


def validate_probe_config(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Fail-closed probe contract for the standard SP/XP path.

    The probe block must be explicit: probing on requires a strictly positive
    disagreement threshold (the statistic is non-negative, so zero or a missing
    threshold would probe on every step and the greedy rule would never act);
    probing off must be written out rather than implied by omission.
    """

    if payload is None:
        raise ValueError(
            "The standard path requires an explicit probe block: either "
            "enabled true with a positive disagreement_threshold, or enabled false."
        )
    if not isinstance(payload, Mapping):
        raise TypeError("probe must be a mapping.")
    config = dict(payload)
    allowed = {"enabled", "disagreement_threshold", "return_floor", "disagreement_stat"}
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(f"Unknown probe field(s): {', '.join(unknown)}.")
    if "enabled" not in config or not isinstance(config["enabled"], bool):
        raise ValueError("probe.enabled must be an explicit boolean.")
    enabled = bool(config["enabled"])
    threshold = config.get("disagreement_threshold")
    return_floor = config.get("return_floor")
    stat = str(config.get("disagreement_stat", "variance"))
    if stat not in {"variance", "range"}:
        raise ValueError("probe.disagreement_stat must be 'variance' or 'range'.")
    if enabled:
        if threshold is None:
            raise ValueError(
                "Enabled probing requires probe.disagreement_threshold."
            )
        threshold = _finite_float(threshold, name="probe.disagreement_threshold")
        if threshold <= 0.0:
            raise ValueError(
                "probe.disagreement_threshold must be strictly positive; "
                "zero probes on every step."
            )
    if return_floor is not None:
        return_floor = _finite_float(return_floor, name="probe.return_floor")
    return {
        "enabled": enabled,
        "disagreement_threshold": None if threshold is None else float(threshold),
        "return_floor": None if return_floor is None else float(return_floor),
        "disagreement_stat": stat,
    }


def shared_team_reward(rewards: Mapping[str, Any]) -> np.ndarray:
    """Return the one shared team reward, refusing silently per-agent rewards.

    OvercookedV2 broadcasts a single scalar to every agent; the standard SP/XP
    accounting depends on that invariant, so it is asserted instead of assumed.
    """

    reward_0 = np.asarray(rewards["agent_0"], dtype=np.float32)
    reward_1 = np.asarray(rewards["agent_1"], dtype=np.float32)
    if reward_0.shape != reward_1.shape or not np.array_equal(reward_0, reward_1):
        raise ValueError(
            "OvercookedV2 emitted different rewards per agent; the standard "
            "path records one shared team reward and cannot proceed."
        )
    return reward_0


def save_standard_checkpoint(
    path: str | Path,
    model: PrimitiveRecurrentEnsembleQ,
    *,
    seed: int,
    environment_steps: int,
    episodes: int,
    phase: str,
    extra_metadata: Mapping[str, Any] | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": STANDARD_CHECKPOINT_SCHEMA_VERSION,
        "seed": int(seed),
        "environment_steps": int(environment_steps),
        "episodes": int(episodes),
        "phase": str(phase),
        "architecture": model.architecture_manifest(),
        "prior_state_sha256": model.prior_state_sha256(),
    }
    if extra_metadata:
        overlap = set(metadata) & set(extra_metadata)
        if overlap:
            raise ValueError(
                "extra_metadata cannot replace checkpoint field(s): "
                + ", ".join(sorted(overlap))
            )
        metadata.update(dict(extra_metadata))
    torch.save(
        {
            "metadata": metadata,
            "model_state_dict": model.state_dict(),
        },
        target,
    )


def load_standard_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str,
) -> tuple[PrimitiveRecurrentEnsembleQ, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("Standard Path C checkpoint must be a mapping.")
    metadata = payload.get("metadata")
    state_dict = payload.get("model_state_dict")
    if not isinstance(metadata, Mapping) or not isinstance(state_dict, Mapping):
        raise ValueError("Standard Path C checkpoint is missing metadata or weights.")
    if metadata.get("schema_version") != STANDARD_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Unsupported standard Path C checkpoint schema.")
    model = PrimitiveRecurrentEnsembleQ.from_manifest(metadata["architecture"])
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.compact_recurrent_parameters()
    model.eval()
    if model.prior_state_sha256() != metadata.get("prior_state_sha256"):
        raise ValueError("Loaded fixed prior differs from checkpoint metadata.")
    return model, dict(metadata)
