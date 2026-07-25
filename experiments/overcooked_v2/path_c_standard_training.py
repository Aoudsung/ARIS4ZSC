"""Seed-contained training for the standard primitive-action path.

The main method uses two stages.  A separate registered mode stops after the
self-play stage to produce Q-learning partner candidates for the R015
opportunity audit.  Both modes
use vectorized JAX environments and PyTorch recurrent temporal-difference
updates.  Neither calls the legacy option graph, conditional-entropy matrix,
global-state featurizer, reward shaper, or scripted partner registry.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from experiments.overcooked_v2.batched_rollout import BatchedEnvPool
from experiments.overcooked_v2.path_c_seed import (
    canonical_uint64_seed,
    derive_ocv2_execution_seed,
)
from experiments.overcooked_v2.path_c_sequence import (
    EpisodeBootstrapRecord,
    per_head_masked_mean,
    sequence_td_core,
)
from experiments.overcooked_v2.path_c_standard_diagnostics import (
    TRAINING_METRICS_SCHEMA_VERSION,
    TrainingWindowAccumulator,
    decompose_raw_reward_events,
    snapshot_trainable_parameters,
    write_training_curves,
)
from experiments.overcooked_v2.path_c_standard import (
    PRIMITIVE_ACTION_NAMES,
    PrimitiveObservationBatch,
    PrimitivePolicyState,
    PrimitiveRecurrentEnsembleQ,
    StandardEnvConfig,
    choose_primitive_actions,
    save_standard_checkpoint,
    scatter_policy_state_,
    shared_team_reward,
    single_step_batch,
    slice_policy_state,
    validate_probe_config,
)


STANDARD_TRAINING_SCHEMA_VERSION = "path_c_standard_training_v1"
SELF_PLAY_ONLY_TRAINING_MODE = "self_play_only"
TWO_STAGE_TRAINING_MODE = "two_stage"
SOURCE_DEPENDENCY_CLOSURE_SCHEMA_VERSION = (
    "path_c_repository_source_dependency_closure_v1"
)

# These are the repository files whose executable semantics determine the
# registered Q-family training run.  The aggregate digest is computed from the
# path and content digest of every entry.  The admission evaluator imports this
# exact tuple and the same hashing function instead of maintaining a second
# approximation of the training dependency closure.
Q_PARTNER_TRAINING_DEPENDENCIES = (
    "experiments/overcooked_v2/batched_rollout.py",
    "experiments/overcooked_v2/env_adapter.py",
    "experiments/overcooked_v2/path_c_seed.py",
    "experiments/overcooked_v2/path_c_sequence.py",
    "experiments/overcooked_v2/path_c_standard.py",
    "experiments/overcooked_v2/path_c_standard_diagnostics.py",
    "experiments/overcooked_v2/path_c_standard_training.py",
    "experiments/overcooked_v2/residual_signature.py",
)

# This definition is deliberately data rather than prose in a planning file.  A
# checkpoint can therefore prove which training algorithm, objective, reward,
# observation, action and convention-generation mechanism produced it.
Q_PARTNER_FAMILY_DEFINITION: dict[str, str] = {
    "family_id": "recurrent_ensemble_q_self_play_v1",
    "training_algorithm": "recurrent_ensemble_double_q_learning",
    "training_objective": "bootstrap_head_huber_temporal_difference",
    "reward_objective": (
        "raw_team_reward_plus_environment_native_per_agent_shaping_"
        "linear_to_zero_by_5000000_steps"
    ),
    "convention_generation": "parameter_shared_two_agent_epsilon_greedy_self_play",
    "observation_contract": "official_local_5x5",
    "action_space": "six_primitive_actions",
    "model_class": "PrimitiveRecurrentEnsembleQ",
    "checkpoint_phase": "self_play_partner_snapshot",
    "action_rule": "greedy_mean_q",
}


def canonical_mapping_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a JSON-compatible mapping with stable key and separator rules."""

    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def repository_source_dependency_closure(
    relative_paths: Sequence[str],
) -> dict[str, Any]:
    """Bind an explicit repository-source dependency closure by content.

    Only repository-relative regular files are accepted.  This deliberately
    records a local source closure rather than pretending to hash third-party
    packages or the Python runtime.
    """

    repository_root = Path(__file__).resolve().parents[2]
    normalized = tuple(sorted(str(item) for item in relative_paths))
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("Source dependency closure must be non-empty and unique.")
    files: dict[str, str] = {}
    for relative in normalized:
        declared = Path(relative)
        if declared.is_absolute() or ".." in declared.parts:
            raise ValueError(
                "Source dependency closure accepts repository-relative paths only."
            )
        target = (repository_root / declared).resolve()
        try:
            target.relative_to(repository_root)
        except ValueError as error:
            raise ValueError("Source dependency escaped the repository root.") from error
        if not target.is_file():
            raise FileNotFoundError(f"Source dependency is missing: {target}")
        files[declared.as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()
    return {
        "schema_version": SOURCE_DEPENDENCY_CLOSURE_SCHEMA_VERSION,
        "files": files,
    }


def repository_source_dependency_closure_sha256(
    relative_paths: Sequence[str],
) -> str:
    """Return the canonical digest of an explicit source dependency closure."""

    return canonical_mapping_sha256(
        repository_source_dependency_closure(relative_paths)
    )


def q_partner_architecture_from_config(
    model_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize the registered Q-family model config to checkpoint form."""

    required = {
        "observation_shape",
        "n_heads",
        "visual_embedding_dim",
        "encoder_dim",
        "recurrent_dim",
        "conv_channels",
        "encoder_spatial_mode",
        "prior_scale",
        "prior_seed",
    }
    if set(model_config) != required:
        raise ValueError(
            "Registered Q-family model config must state exactly its frozen "
            "architecture fields."
        )
    return {
        "model_class": "PrimitiveRecurrentEnsembleQ",
        "observation_shape": [int(item) for item in model_config["observation_shape"]],
        "primitive_action_names": list(PRIMITIVE_ACTION_NAMES),
        "n_actions": 6,
        "n_heads": int(model_config["n_heads"]),
        "visual_embedding_dim": int(model_config["visual_embedding_dim"]),
        "encoder_dim": int(model_config["encoder_dim"]),
        "recurrent_dim": int(model_config["recurrent_dim"]),
        "conv_channels": [int(item) for item in model_config["conv_channels"]],
        "encoder_spatial_mode": str(model_config["encoder_spatial_mode"]),
        "prior_scale": float(model_config["prior_scale"]),
        "prior_seed": int(model_config["prior_seed"]),
        "input_contract": (
            "own_default_local_observation+own_previous_action+"
            "own_previous_raw_reward+episode_start"
        ),
    }


Q_PARTNER_FAMILY_SPEC_SHA256 = canonical_mapping_sha256(
    Q_PARTNER_FAMILY_DEFINITION
)


def partner_training_run_id(
    *,
    family_spec_sha256: str,
    training_seed: int,
    training_config_sha256: str,
    environment_config_sha256: str,
    training_implementation_sha256: str,
) -> str:
    """Bind one independent training run to all files that define it."""

    return canonical_mapping_sha256(
        {
            "family_spec_sha256": family_spec_sha256,
            "training_seed": int(training_seed),
            "training_config_sha256": training_config_sha256,
            "environment_config_sha256": environment_config_sha256,
            "training_implementation_sha256": training_implementation_sha256,
        }
    )


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative.")
    return result


def _probability(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0,1].")
    return result


def derive_standard_seed(base_seed: int, *parts: object) -> int:
    """Derive one canonical unsigned 64-bit identity from named coordinates."""

    base = canonical_uint64_seed(base_seed, name="standard training seed")
    payload = ":".join(["path_c_standard_seed_v1", str(base), *(str(p) for p in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def phase_shaping_anneal_factor(
    phase_environment_steps: int,
    phase_total_environment_steps: int,
) -> float:
    """Linearly remove native reward shaping over the first half of one phase."""

    phase_total = _positive_int(
        phase_total_environment_steps,
        name="phase_total_environment_steps",
    )
    steps = int(phase_environment_steps)
    if steps < 0:
        raise ValueError("phase_environment_steps must be non-negative.")
    horizon = float(phase_total) / 2.0
    return max(0.0, 1.0 - float(steps) / horizon)


def shaped_rewards_by_agent(
    info: Mapping[str, Any],
    *,
    batch_size: int,
) -> dict[int, np.ndarray]:
    """Read the environment-native per-agent shaped reward without changing it."""

    shaped = info.get("shaped_reward")
    if not isinstance(shaped, Mapping):
        raise ValueError("Environment info is missing per-agent shaped_reward.")
    result: dict[int, np.ndarray] = {}
    for slot in (0, 1):
        key = f"agent_{slot}"
        if key not in shaped:
            raise ValueError(f"Environment shaped_reward is missing {key!r}.")
        value = np.asarray(shaped[key], dtype=np.float32)
        if value.shape != (int(batch_size),):
            raise ValueError(
                f"Environment shaped_reward for {key!r} has shape {value.shape}, "
                f"expected {(int(batch_size),)}."
            )
        if not bool(np.isfinite(value).all()):
            raise ValueError("Environment shaped_reward must be finite.")
        result[slot] = value.copy()
    return result


@dataclass(frozen=True)
class PrimitiveEpisode:
    episode_id: str
    observations: np.ndarray
    previous_actions: np.ndarray
    previous_rewards: np.ndarray
    episode_starts: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    bootstrap_mask: np.ndarray

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])

    def validate(
        self,
        *,
        observation_shape: Sequence[int],
        n_actions: int,
        n_heads: int,
    ) -> "PrimitiveEpisode":
        length = self.length
        if length <= 0:
            raise ValueError("PrimitiveEpisode must contain at least one transition.")
        if self.observations.shape != (length + 1, *tuple(observation_shape)):
            raise ValueError("PrimitiveEpisode observation sequence is misaligned.")
        for name, value in (
            ("previous_actions", self.previous_actions),
            ("previous_rewards", self.previous_rewards),
            ("episode_starts", self.episode_starts),
        ):
            if value.shape != (length + 1,):
                raise ValueError(f"PrimitiveEpisode {name} is misaligned.")
        for name, value in (
            ("actions", self.actions),
            ("rewards", self.rewards),
            ("dones", self.dones),
        ):
            if value.shape != (length,):
                raise ValueError(f"PrimitiveEpisode {name} is misaligned.")
        if self.bootstrap_mask.shape != (n_heads,) or not bool(
            self.bootstrap_mask.any()
        ):
            raise ValueError("PrimitiveEpisode bootstrap mask is empty or misaligned.")
        if bool((self.actions < 0).any()) or bool((self.actions >= n_actions).any()):
            raise ValueError("PrimitiveEpisode contains an invalid primitive action.")
        if (
            not bool(np.isfinite(self.observations).all())
            or not bool(np.isfinite(self.previous_rewards).all())
            or not bool(np.isfinite(self.rewards).all())
        ):
            raise ValueError("PrimitiveEpisode contains non-finite data.")
        if not bool(self.episode_starts[0]) or bool(self.episode_starts[1:].any()):
            raise ValueError("PrimitiveEpisode has invalid episode-start markers.")
        if int(self.previous_actions[0]) != n_actions:
            raise ValueError("PrimitiveEpisode initial previous action must be padding.")
        if not bool(self.dones[-1]):
            raise ValueError("PrimitiveEpisode must end at an environment boundary.")
        return self


class PrimitiveEpisodeBuilder:
    def __init__(
        self,
        *,
        episode_id: str,
        initial_observation: np.ndarray,
        n_actions: int,
        n_heads: int,
        bootstrap_p: float,
        bootstrap_seed: int,
    ) -> None:
        self.episode_id = str(episode_id)
        self.n_actions = int(n_actions)
        bootstrap = EpisodeBootstrapRecord.sample(
            episode_id=self.episode_id,
            n_heads=n_heads,
            bootstrap_p=bootstrap_p,
            manifest_seed=int(bootstrap_seed),
        )
        self.bootstrap_mask = np.asarray(bootstrap.mask, dtype=bool)
        self.observations = [np.asarray(initial_observation, dtype=np.float32).copy()]
        self.previous_actions = [self.n_actions]
        self.previous_rewards = [0.0]
        self.episode_starts = [True]
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []

    def append(
        self,
        *,
        action: int,
        training_reward: float,
        previous_raw_reward: float,
        done: bool,
        next_observation: np.ndarray,
    ) -> None:
        action = int(action)
        if not 0 <= action < self.n_actions:
            raise ValueError("Primitive action lies outside the six-action space.")
        training_reward = float(training_reward)
        previous_raw_reward = float(previous_raw_reward)
        if not math.isfinite(training_reward) or not math.isfinite(previous_raw_reward):
            raise ValueError("Primitive training and raw rewards must be finite.")
        self.actions.append(action)
        self.rewards.append(training_reward)
        self.dones.append(bool(done))
        self.observations.append(
            np.asarray(next_observation, dtype=np.float32).copy()
        )
        self.previous_actions.append(action)
        self.previous_rewards.append(previous_raw_reward)
        self.episode_starts.append(False)

    def finish(self, *, observation_shape: Sequence[int]) -> PrimitiveEpisode:
        episode = PrimitiveEpisode(
            episode_id=self.episode_id,
            observations=np.stack(self.observations, axis=0),
            previous_actions=np.asarray(self.previous_actions, dtype=np.int64),
            previous_rewards=np.asarray(self.previous_rewards, dtype=np.float32),
            episode_starts=np.asarray(self.episode_starts, dtype=bool),
            actions=np.asarray(self.actions, dtype=np.int64),
            rewards=np.asarray(self.rewards, dtype=np.float32),
            dones=np.asarray(self.dones, dtype=bool),
            bootstrap_mask=self.bootstrap_mask.copy(),
        )
        return episode.validate(
            observation_shape=observation_shape,
            n_actions=self.n_actions,
            n_heads=len(self.bootstrap_mask),
        )


@dataclass(frozen=True)
class PrimitiveSequenceTDBatch:
    observations: PrimitiveObservationBatch
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    discounts: torch.Tensor
    transition_mask: torch.Tensor
    bootstrap_mask: torch.Tensor

    @property
    def n_heads(self) -> int:
        return int(self.bootstrap_mask.shape[1])


def collate_primitive_episodes(
    episodes: Sequence[PrimitiveEpisode],
    *,
    observation_shape: Sequence[int],
    n_actions: int,
    n_heads: int,
    gamma: float,
    device: torch.device | str,
) -> PrimitiveSequenceTDBatch:
    if not episodes:
        raise ValueError("At least one primitive episode is required.")
    gamma = float(gamma)
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must lie in [0,1].")
    records = [
        item.validate(
            observation_shape=observation_shape,
            n_actions=n_actions,
            n_heads=n_heads,
        )
        for item in episodes
    ]
    batch_size = len(records)
    max_transitions = max(item.length for item in records)
    max_states = max_transitions + 1
    shape = tuple(int(item) for item in observation_shape)
    observations = torch.zeros(
        batch_size,
        max_states,
        *shape,
        dtype=torch.float32,
        device=device,
    )
    previous_actions = torch.full(
        (batch_size, max_states),
        fill_value=n_actions,
        dtype=torch.long,
        device=device,
    )
    previous_rewards = torch.zeros(
        batch_size,
        max_states,
        dtype=torch.float32,
        device=device,
    )
    episode_starts = torch.zeros(
        batch_size,
        max_states,
        dtype=torch.bool,
        device=device,
    )
    time_mask = torch.zeros_like(episode_starts)
    valid_actions = torch.zeros(
        batch_size,
        max_states,
        n_actions,
        dtype=torch.bool,
        device=device,
    )
    actions = torch.full(
        (batch_size, max_transitions),
        fill_value=-1,
        dtype=torch.long,
        device=device,
    )
    rewards = torch.zeros(
        batch_size,
        max_transitions,
        dtype=torch.float32,
        device=device,
    )
    dones = torch.ones(
        batch_size,
        max_transitions,
        dtype=torch.bool,
        device=device,
    )
    transition_mask = torch.zeros_like(dones)
    lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
    bootstrap_mask = torch.zeros(
        batch_size,
        n_heads,
        dtype=torch.bool,
        device=device,
    )
    for row, episode in enumerate(records):
        transition_count = episode.length
        state_count = transition_count + 1
        observations[row, :state_count] = torch.as_tensor(
            episode.observations,
            dtype=torch.float32,
            device=device,
        )
        previous_actions[row, :state_count] = torch.as_tensor(
            episode.previous_actions,
            dtype=torch.long,
            device=device,
        )
        previous_rewards[row, :state_count] = torch.as_tensor(
            episode.previous_rewards,
            dtype=torch.float32,
            device=device,
        )
        episode_starts[row, :state_count] = torch.as_tensor(
            episode.episode_starts,
            dtype=torch.bool,
            device=device,
        )
        time_mask[row, :state_count] = True
        valid_actions[row, :state_count] = True
        actions[row, :transition_count] = torch.as_tensor(
            episode.actions,
            dtype=torch.long,
            device=device,
        )
        rewards[row, :transition_count] = torch.as_tensor(
            episode.rewards,
            dtype=torch.float32,
            device=device,
        )
        dones[row, :transition_count] = torch.as_tensor(
            episode.dones,
            dtype=torch.bool,
            device=device,
        )
        transition_mask[row, :transition_count] = True
        lengths[row] = state_count
        bootstrap_mask[row] = torch.as_tensor(
            episode.bootstrap_mask,
            dtype=torch.bool,
            device=device,
        )
    observation_batch = PrimitiveObservationBatch(
        observations=observations,
        previous_actions=previous_actions,
        previous_rewards=previous_rewards,
        episode_starts=episode_starts,
        valid_actions=valid_actions,
        time_mask=time_mask,
        lengths=lengths,
    ).validate(observation_shape=shape, n_actions=n_actions)
    return PrimitiveSequenceTDBatch(
        observations=observation_batch,
        actions=actions,
        rewards=rewards,
        dones=dones,
        discounts=torch.full_like(rewards, gamma),
        transition_mask=transition_mask,
        bootstrap_mask=bootstrap_mask,
    )


@dataclass(frozen=True)
class PrimitiveTDLossOutput:
    loss: torch.Tensor
    per_head_loss: torch.Tensor
    head_support: torch.Tensor


@dataclass(frozen=True)
class PrimitiveTDUpdateMetrics:
    loss: float
    per_head_loss: tuple[float, ...]
    per_head_support: tuple[int, ...]
    gradient_norm: float


def primitive_sequence_td_loss(
    online: PrimitiveRecurrentEnsembleQ,
    target: PrimitiveRecurrentEnsembleQ,
    batch: PrimitiveSequenceTDBatch,
    *,
    huber_delta: float = 1.0,
) -> PrimitiveTDLossOutput:
    online.assert_same_fixed_prior(target)
    if batch.n_heads != online.n_heads or target.n_heads != online.n_heads:
        raise ValueError("TD batch and model ensemble dimensions differ.")
    q_online, _ = online.forward_sequence(batch.observations)
    with torch.no_grad():
        q_target, _ = target.forward_sequence(batch.observations)
    _, _, elementwise = sequence_td_core(
        q_online,
        q_target,
        actions=batch.actions,
        rewards=batch.rewards,
        dones=batch.dones,
        discounts=batch.discounts,
        n_heads=online.n_heads,
        td_loss="huber",
        huber_delta=huber_delta,
        double_q=True,
        vmax=None,
    )
    per_head, support, supported = per_head_masked_mean(
        elementwise,
        batch.transition_mask,
        batch.bootstrap_mask,
    )
    if not bool(supported.any()):
        raise ValueError("No ensemble head has TD support in this batch.")
    return PrimitiveTDLossOutput(
        loss=per_head[supported].mean(),
        per_head_loss=per_head,
        head_support=support,
    )


def select_partner_actions(
    partner_pool: Sequence[PrimitiveRecurrentEnsembleQ],
    partner_states: list[PrimitivePolicyState],
    partner_indices: np.ndarray,
    observations: np.ndarray,
    previous_actions: np.ndarray,
    previous_rewards: np.ndarray,
    episode_starts: np.ndarray,
    *,
    n_actions: int,
    device: torch.device | str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[PrimitivePolicyState]]:
    """Act every environment row with its assigned partner snapshot.

    Each snapshot forwards only its assigned rows. Rows assigned elsewhere need
    no warm hidden state: partner assignment changes only on episode reset, and
    the episode-start flag zeroes the newly assigned partner's row state on its
    first forward, so sub-batching is exactly equivalent to a full-batch pass.
    """

    partner_indices = np.asarray(partner_indices, dtype=np.int64)
    batch_size = int(partner_indices.shape[0])
    selected = np.zeros(batch_size, dtype=np.int64)
    for pool_index in np.unique(partner_indices):
        partner = partner_pool[int(pool_index)]
        rows = np.flatnonzero(partner_indices == pool_index)
        partner.eval()
        sub_batch = single_step_batch(
            observations[rows],
            previous_actions[rows],
            previous_rewards[rows],
            episode_starts[rows],
            n_actions=n_actions,
            device=device,
        )
        sub_state, row_index = slice_policy_state(
            partner_states[int(pool_index)],
            rows,
        )
        with torch.no_grad():
            q_values, _, sub_next_state = partner.forward_step(sub_batch, sub_state)
        scatter_policy_state_(
            partner_states[int(pool_index)],
            row_index,
            sub_next_state,
        )
        decisions = choose_primitive_actions(
            q_values,
            sub_batch.valid_actions[:, 0],
            rng=rng,
            epsilon=0.0,
            probe_enabled=False,
        )
        selected[rows] = decisions.actions
    return selected, partner_states


class PrimitiveEpisodeReplay:
    def __init__(self, capacity_episodes: int) -> None:
        self.capacity_episodes = _positive_int(
            capacity_episodes,
            name="capacity_episodes",
        )
        self._episodes: deque[PrimitiveEpisode] = deque(maxlen=self.capacity_episodes)

    def __len__(self) -> int:
        return len(self._episodes)

    def add(self, episode: PrimitiveEpisode) -> None:
        self._episodes.append(episode)

    def sample(
        self,
        batch_size: int,
        *,
        rng: np.random.Generator,
    ) -> list[PrimitiveEpisode]:
        batch_size = _positive_int(batch_size, name="batch_size")
        if len(self._episodes) < batch_size:
            raise ValueError("Replay does not contain enough completed episodes.")
        indices = rng.choice(len(self._episodes), size=batch_size, replace=False)
        records = list(self._episodes)
        return [records[int(index)] for index in indices]


@dataclass(frozen=True)
class StandardTrainingSpec:
    run_kind: str
    training_mode: str
    scientific_readout_allowed: bool
    seed: int
    total_environment_steps: int
    self_play_environment_steps: int
    path_c_environment_steps: int
    batch_size_envs: int
    partner_pool_snapshot_steps: tuple[int, ...]
    ego_initialization: str
    replay_capacity_episodes: int
    minibatch_episodes: int
    learning_starts_episodes: int
    updates_per_vector_step: int
    target_update_environment_steps: int
    metrics_interval_environment_steps: int
    stop_if_no_path_c_delivery_by_environment_steps: int | None
    gamma: float
    learning_rate: float
    gradient_clip_norm: float
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_environment_steps: int
    bootstrap_p: float
    partner_family_definition: Mapping[str, str] | None
    partner_family_spec_sha256: str | None
    training_config_sha256: str
    environment_config_sha256: str
    training_implementation_sha256: str
    training_implementation_dependencies: Mapping[str, Any]
    training_run_id: str | None
    output_dir: Path

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "StandardTrainingSpec":
        training = payload.get("training")
        if not isinstance(training, Mapping):
            raise TypeError("Standard training config requires a training mapping.")
        run_kind = str(payload.get("run_kind", "smoke"))
        if run_kind not in {"smoke", "formal", "partner_family"}:
            raise ValueError(
                "run_kind must be 'smoke', 'formal', or 'partner_family'."
            )
        training_mode = str(
            payload.get(
                "training_mode",
                (
                    SELF_PLAY_ONLY_TRAINING_MODE
                    if run_kind == "partner_family"
                    else TWO_STAGE_TRAINING_MODE
                ),
            )
        )
        if training_mode not in {
            SELF_PLAY_ONLY_TRAINING_MODE,
            TWO_STAGE_TRAINING_MODE,
        }:
            raise ValueError("Unsupported standard training_mode.")
        if (run_kind == "partner_family") != (
            training_mode == SELF_PLAY_ONLY_TRAINING_MODE
        ):
            raise ValueError(
                "Only partner_family runs may use the self_play_only mode."
            )
        scientific_readout_allowed = bool(
            payload.get("scientific_readout_allowed", False)
        )
        if run_kind == "smoke" and scientific_readout_allowed:
            raise ValueError("Smoke configuration cannot permit scientific readout.")
        if run_kind == "partner_family" and scientific_readout_allowed:
            raise ValueError(
                "Partner-family formation is not a scientific result."
            )
        total = _positive_int(
            training["total_environment_steps"],
            name="training.total_environment_steps",
        )
        self_play = _positive_int(
            training["self_play_environment_steps"],
            name="training.self_play_environment_steps",
        )
        path_c = _nonnegative_int(
            training["path_c_environment_steps"],
            name="training.path_c_environment_steps",
        )
        if self_play + path_c != total:
            raise ValueError("The two training phases must sum to the total environment steps.")
        if run_kind == "formal" and total != 30_000_000:
            raise ValueError("Formal Test Time training uses 30,000,000 environment steps.")
        if training_mode == SELF_PLAY_ONLY_TRAINING_MODE:
            if total != 10_000_000 or self_play != total or path_c != 0:
                raise ValueError(
                    "The registered Q partner family uses exactly 10M self-play "
                    "steps and no Path C phase."
                )
        elif path_c <= 0:
            raise ValueError("Two-stage training requires positive Path C steps.")
        batch_size = _positive_int(
            training["batch_size_envs"],
            name="training.batch_size_envs",
        )
        if self_play % batch_size or (path_c and path_c % batch_size):
            raise ValueError("Each phase budget must be divisible by batch_size_envs.")
        snapshots = tuple(int(item) for item in training["partner_pool_snapshot_steps"])
        if not snapshots or tuple(sorted(set(snapshots))) != snapshots:
            raise ValueError("partner_pool_snapshot_steps must be sorted and unique.")
        if any(item <= 0 or item > self_play or item % batch_size for item in snapshots):
            raise ValueError("Partner snapshots must lie on self-play vector-step boundaries.")
        if snapshots[-1] != self_play:
            raise ValueError("The self-play final checkpoint must be in the partner pool.")
        if training_mode == SELF_PLAY_ONLY_TRAINING_MODE:
            if snapshots != (2_500_000, 5_000_000, 7_500_000, 10_000_000):
                raise ValueError(
                    "The registered Q family snapshots must be 2.5M, 5M, 7.5M "
                    "and 10M steps."
                )
            if training.get("ego_initialization") not in {None, "not_applicable"}:
                raise ValueError(
                    "Self-play-only partner formation has no ego initialization."
                )
            initialization = "not_applicable"
        else:
            initialization = str(training["ego_initialization"])
            if initialization not in {"fresh", "self_play_final"}:
                raise ValueError(
                    "ego_initialization must be 'fresh' or 'self_play_final'."
                )
        gamma = float(training.get("gamma", 0.99))
        if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
            raise ValueError("training.gamma must lie in [0,1].")
        learning_rate = float(training.get("learning_rate", 2.5e-4))
        gradient_clip = float(training.get("gradient_clip_norm", 10.0))
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("training.learning_rate must be positive.")
        if not math.isfinite(gradient_clip) or gradient_clip <= 0.0:
            raise ValueError("training.gradient_clip_norm must be positive.")
        epsilon_start = _probability(
            training.get("epsilon_start", 1.0),
            name="training.epsilon_start",
        )
        epsilon_end = _probability(
            training.get("epsilon_end", 0.05),
            name="training.epsilon_end",
        )
        bootstrap_p = _probability(
            training.get("bootstrap_p", 0.5),
            name="training.bootstrap_p",
        )
        if bootstrap_p <= 0.0:
            raise ValueError("training.bootstrap_p must lie in (0,1].")
        target_update_steps = _positive_int(
            training["target_update_environment_steps"],
            name="training.target_update_environment_steps",
        )
        metrics_interval_steps = _positive_int(
            training.get("metrics_interval_environment_steps", target_update_steps),
            name="training.metrics_interval_environment_steps",
        )
        if metrics_interval_steps % batch_size:
            raise ValueError(
                "training.metrics_interval_environment_steps must lie on a "
                "vector-step boundary."
            )
        if self_play % metrics_interval_steps or (
            path_c and path_c % metrics_interval_steps
        ):
            raise ValueError(
                "Each training phase must be divisible by the metrics interval."
            )
        raw_stop_step = training.get(
            "stop_if_no_path_c_delivery_by_environment_steps"
        )
        stop_step = None
        if raw_stop_step is not None:
            if training_mode == SELF_PLAY_ONLY_TRAINING_MODE:
                raise ValueError(
                    "Self-play-only partner formation has no Path C stop rule."
                )
            stop_step = _positive_int(
                raw_stop_step,
                name=(
                    "training."
                    "stop_if_no_path_c_delivery_by_environment_steps"
                ),
            )
            if stop_step > path_c:
                raise ValueError(
                    "The no-delivery stop step must lie within the Path C phase."
                )
            if stop_step % metrics_interval_steps:
                raise ValueError(
                    "The no-delivery stop step must coincide with a metrics window."
                )
        family_definition: Mapping[str, str] | None = None
        family_spec_sha256: str | None = None
        if training_mode == SELF_PLAY_ONLY_TRAINING_MODE:
            raw_family = payload.get("partner_family")
            if not isinstance(raw_family, Mapping):
                raise TypeError(
                    "Self-play-only training requires a partner_family mapping."
                )
            family_definition = {
                str(key): str(value) for key, value in raw_family.items()
            }
            if family_definition != Q_PARTNER_FAMILY_DEFINITION:
                raise ValueError(
                    "The Q partner-family definition differs from its registered semantics."
                )
            raw_model = payload.get("model")
            if not isinstance(raw_model, Mapping):
                raise TypeError(
                    "Q partner-family training requires a model architecture mapping."
                )
            q_partner_architecture_from_config(raw_model)
            family_spec_sha256 = canonical_mapping_sha256(family_definition)
        elif payload.get("partner_family") is not None:
            raise ValueError(
                "Two-stage training must not claim a partner-family definition."
            )
        config_sha256 = str(payload.get("_source_config_sha256", ""))
        if not config_sha256:
            hash_payload = {
                str(key): value
                for key, value in payload.items()
                if not str(key).startswith("_")
            }
            config_sha256 = canonical_mapping_sha256(hash_payload)
        if len(config_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in config_sha256
        ):
            raise ValueError("Training configuration SHA-256 is malformed.")
        environment_config_sha256 = str(
            payload.get(
                "_environment_config_sha256",
                canonical_mapping_sha256(
                    payload.get("environment")
                    if isinstance(payload.get("environment"), Mapping)
                    else {}
                ),
            )
        )
        current_implementation_dependencies = repository_source_dependency_closure(
            Q_PARTNER_TRAINING_DEPENDENCIES
        )
        raw_implementation_dependencies = payload.get(
            "_training_implementation_dependencies",
            current_implementation_dependencies,
        )
        if not isinstance(raw_implementation_dependencies, Mapping) or dict(
            raw_implementation_dependencies
        ) != current_implementation_dependencies:
            raise ValueError(
                "Training implementation dependency closure differs from the "
                "registered repository sources."
            )
        training_implementation_dependencies = dict(
            current_implementation_dependencies
        )
        expected_implementation_sha256 = canonical_mapping_sha256(
            training_implementation_dependencies
        )
        training_implementation_sha256 = str(
            payload.get(
                "_training_implementation_sha256",
                expected_implementation_sha256,
            )
        )
        if training_implementation_sha256 != expected_implementation_sha256:
            raise ValueError(
                "Training implementation hash does not match its dependency closure."
            )
        for name, value in (
            ("environment configuration", environment_config_sha256),
            ("training implementation", training_implementation_sha256),
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{name} SHA-256 is malformed.")
        training_run_identity: str | None = None
        if training_mode == SELF_PLAY_ONLY_TRAINING_MODE:
            if any(
                key not in payload
                for key in (
                    "_source_config_sha256",
                    "_environment_config_sha256",
                    "_training_implementation_sha256",
                    "_training_implementation_dependencies",
                )
            ):
                raise ValueError(
                    "Partner-family training must be loaded from its bound config file."
                )
            if family_spec_sha256 is None:
                raise RuntimeError("Partner-family hash was not constructed.")
            training_run_identity = partner_training_run_id(
                family_spec_sha256=family_spec_sha256,
                training_seed=int(payload["seed"]),
                training_config_sha256=config_sha256,
                environment_config_sha256=environment_config_sha256,
                training_implementation_sha256=training_implementation_sha256,
            )
        return cls(
            run_kind=run_kind,
            training_mode=training_mode,
            scientific_readout_allowed=scientific_readout_allowed,
            seed=canonical_uint64_seed(int(payload["seed"]), name="training seed"),
            total_environment_steps=total,
            self_play_environment_steps=self_play,
            path_c_environment_steps=path_c,
            batch_size_envs=batch_size,
            partner_pool_snapshot_steps=snapshots,
            ego_initialization=initialization,
            replay_capacity_episodes=_positive_int(
                training["replay_capacity_episodes"],
                name="training.replay_capacity_episodes",
            ),
            minibatch_episodes=_positive_int(
                training["minibatch_episodes"],
                name="training.minibatch_episodes",
            ),
            learning_starts_episodes=_positive_int(
                training["learning_starts_episodes"],
                name="training.learning_starts_episodes",
            ),
            updates_per_vector_step=_positive_int(
                training["updates_per_vector_step"],
                name="training.updates_per_vector_step",
            ),
            target_update_environment_steps=target_update_steps,
            metrics_interval_environment_steps=metrics_interval_steps,
            stop_if_no_path_c_delivery_by_environment_steps=stop_step,
            gamma=gamma,
            learning_rate=learning_rate,
            gradient_clip_norm=gradient_clip,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay_environment_steps=_positive_int(
                training["epsilon_decay_environment_steps"],
                name="training.epsilon_decay_environment_steps",
            ),
            bootstrap_p=bootstrap_p,
            partner_family_definition=family_definition,
            partner_family_spec_sha256=family_spec_sha256,
            training_config_sha256=config_sha256,
            environment_config_sha256=environment_config_sha256,
            training_implementation_sha256=training_implementation_sha256,
            training_implementation_dependencies=(
                training_implementation_dependencies
            ),
            training_run_id=training_run_identity,
            output_dir=Path(payload["output_dir"]),
        )

    def epsilon(self, phase_environment_steps: int) -> float:
        fraction = min(
            1.0,
            max(0.0, float(phase_environment_steps) / self.epsilon_decay_environment_steps),
        )
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)


def load_standard_training_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    source_bytes = config_path.read_bytes()
    payload = yaml.safe_load(source_bytes.decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("Standard training config must contain a mapping.")
    config = dict(payload)
    if config.get("schema_version") != STANDARD_TRAINING_SCHEMA_VERSION:
        raise ValueError("Unsupported standard training config schema.")
    env_path = config.get("environment_config")
    if not isinstance(env_path, str) or not env_path.strip():
        raise ValueError("Standard training config requires environment_config.")
    resolved_env_path = Path(env_path)
    if not resolved_env_path.is_absolute():
        resolved_env_path = (config_path.parent / resolved_env_path).resolve()
    environment_bytes = resolved_env_path.read_bytes()
    env_payload = yaml.safe_load(environment_bytes.decode("utf-8"))
    config["environment"] = env_payload
    config["environment_config_resolved"] = str(resolved_env_path)
    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        config["output_dir"] = str((config_path.parent / output_dir).resolve())
    config["_source_config_sha256"] = hashlib.sha256(source_bytes).hexdigest()
    config["_source_config_path"] = str(config_path.resolve())
    config["_environment_config_sha256"] = hashlib.sha256(
        environment_bytes
    ).hexdigest()
    implementation_dependencies = repository_source_dependency_closure(
        Q_PARTNER_TRAINING_DEPENDENCIES
    )
    config["_training_implementation_dependencies"] = implementation_dependencies
    config["_training_implementation_sha256"] = canonical_mapping_sha256(
        implementation_dependencies
    )
    return config


def build_standard_model(model_config: Mapping[str, Any]) -> PrimitiveRecurrentEnsembleQ:
    shape = model_config.get("observation_shape")
    if shape is None:
        raise ValueError(
            "Standard model config must state the layout-specific observation_shape."
        )
    return PrimitiveRecurrentEnsembleQ(
        observation_shape=shape,
        n_actions=6,
        n_heads=int(model_config.get("n_heads", 5)),
        visual_embedding_dim=int(model_config.get("visual_embedding_dim", 128)),
        encoder_dim=int(model_config.get("encoder_dim", 128)),
        recurrent_dim=int(model_config.get("recurrent_dim", 128)),
        conv_channels=model_config.get("conv_channels", [32, 32, 16]),
        encoder_spatial_mode=str(
            model_config.get("encoder_spatial_mode", "flatten")
        ),
        prior_scale=float(model_config.get("prior_scale", 0.0)),
        prior_seed=int(model_config.get("prior_seed", 0)),
    )


class StandardPathCTrainer:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.spec = StandardTrainingSpec.from_mapping(config)
        self.env_config = StandardEnvConfig.from_mapping(config["environment"])
        self.device = torch.device(str(config.get("device", "cuda")))
        self.rng = np.random.default_rng(self.spec.seed)
        torch.manual_seed(derive_ocv2_execution_seed(self.spec.seed))
        self.model_config = dict(config.get("model") or {})
        self.probe_config = validate_probe_config(config.get("probe"))
        if (
            self.spec.training_mode == SELF_PLAY_ONLY_TRAINING_MODE
            and self.probe_config["enabled"]
        ):
            raise ValueError(
                "Registered Q partner-family formation must keep probing disabled."
            )
        self.output_dir = self.spec.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "training_metrics.jsonl"
        self.metrics_rows = 0
        self.adapter = self.env_config.make_adapter()
        self.pool = BatchedEnvPool(self.adapter.env, self.spec.batch_size_envs)
        if "observation_shape" not in self.model_config:
            raise ValueError(
                "Standard training config must state model.observation_shape."
            )
        self.observation_shape = tuple(
            int(item) for item in self.model_config["observation_shape"]
        )
        self.n_actions = 6
        if tuple(self.adapter.env.obs_shape) != self.observation_shape:
            raise ValueError(
                "Configured model observation_shape differs from the standard "
                f"environment: {self.observation_shape} vs {self.adapter.env.obs_shape}."
            )
        if int(self.adapter.env.num_agents) != 2:
            raise ValueError("The standard SP/XP path requires exactly two agents.")
        if self.probe_config["enabled"] and int(
            self.model_config.get("n_heads", 5)
        ) <= 1:
            raise ValueError("Enabled Path C probing requires more than one ensemble head.")
        if self.spec.minibatch_episodes > self.spec.replay_capacity_episodes:
            raise ValueError("minibatch_episodes exceeds replay_capacity_episodes.")
        self.episode_counter = 0
        self.gradient_updates = 0
        self.probe_count = 0

    def run(self) -> dict[str, Any]:
        self.metrics_path.write_text("", encoding="utf-8")
        self.metrics_rows = 0
        self_play_model = build_standard_model(self.model_config).to(self.device)
        if self.spec.training_mode == SELF_PLAY_ONLY_TRAINING_MODE and (
            self_play_model.architecture_manifest()
            != q_partner_architecture_from_config(self.model_config)
        ):
            raise RuntimeError(
                "Q partner model differs from its registered architecture config."
            )
        partner_pool, self_play_metrics = self._run_self_play(self_play_model)
        if self.spec.training_mode == SELF_PLAY_ONLY_TRAINING_MODE:
            return self._finish_partner_family_run(
                self_play_model,
                partner_pool,
                self_play_metrics,
            )
        if self.spec.ego_initialization == "self_play_final":
            ego_model = copy.deepcopy(self_play_model)
            ego_model.compact_recurrent_parameters()
        else:
            ego_model = build_standard_model(self.model_config).to(self.device)
        path_c_metrics = self._run_path_c(ego_model, partner_pool)
        stopped_early = bool(path_c_metrics.get("stopped_early", False))
        realized_environment_steps = int(
            self_play_metrics["effective_environment_steps"]
        ) + int(path_c_metrics["effective_environment_steps"])
        expected_metric_rows = (
            realized_environment_steps
            // self.spec.metrics_interval_environment_steps
        )
        if self.metrics_rows != expected_metric_rows:
            raise RuntimeError(
                "Training metric row count differs from the configured schedule: "
                f"{self.metrics_rows} vs {expected_metric_rows}."
            )
        # Read the budget from realized phase counters. Only the preregistered
        # no-delivery rule may produce a shorter, explicitly labelled run.
        if not stopped_early and (
            realized_environment_steps != self.spec.total_environment_steps
        ):
            raise RuntimeError(
                "Realized environment steps differ from the configured budget: "
                f"{realized_environment_steps} vs {self.spec.total_environment_steps}."
            )
        checkpoint_path = self.output_dir / (
            "path_c_stopped_no_delivery.pt" if stopped_early else "path_c_final.pt"
        )
        save_standard_checkpoint(
            checkpoint_path,
            ego_model,
            seed=self.spec.seed,
            environment_steps=realized_environment_steps,
            episodes=self.episode_counter,
            phase=("path_c_stopped_no_delivery" if stopped_early else "path_c_final"),
            extra_metadata={
                "run_kind": self.spec.run_kind,
                "scientific_readout_allowed": self.spec.scientific_readout_allowed,
                "gradient_updates": self.gradient_updates,
                "probe_count": self.probe_count,
                "layout": self.env_config.layout,
                "environment": self.env_config.to_mapping(),
                "partner_pool_snapshot_steps": list(
                    self.spec.partner_pool_snapshot_steps
                ),
                "ego_initialization": self.spec.ego_initialization,
                "training_metrics_schema_version": TRAINING_METRICS_SCHEMA_VERSION,
                "metrics_interval_environment_steps": (
                    self.spec.metrics_interval_environment_steps
                ),
                "run_status": (
                    "stopped_no_delivery" if stopped_early else "completed"
                ),
                "training_reward_contract": (
                    "TD target uses raw team reward plus annealed environment-native "
                    "per-agent shaped reward; recurrent previous reward remains raw"
                ),
            },
        )
        curves_path = self.output_dir / "training_curves.png"
        write_training_curves(self.metrics_path, curves_path)
        manifest = {
            "schema_version": "path_c_standard_training_artifact_v1",
            "run_kind": self.spec.run_kind,
            "scientific_readout_allowed": self.spec.scientific_readout_allowed,
            "seed": self.spec.seed,
            "layout": self.env_config.layout,
            "run_status": (
                "stopped_no_delivery" if stopped_early else "completed"
            ),
            "effective_environment_steps": realized_environment_steps,
            "effective_episodes": self.episode_counter,
            "episode_counter_unit": (
                "joint environment episodes; the self-play phase stores two "
                "agent trajectories per counted episode"
            ),
            "gradient_updates": self.gradient_updates,
            "probe_count": self.probe_count,
            "self_play": self_play_metrics,
            "path_c": path_c_metrics,
            "checkpoint": str(checkpoint_path),
            "training_reward_contract": {
                "td_target": (
                    "raw team reward plus phase-annealed environment-native "
                    "per-agent shaped reward"
                ),
                "recurrent_previous_reward": "raw team reward",
                "evaluation_reward": "raw team reward",
                "anneal_horizon": "first half of each training phase",
            },
            "stop_if_no_path_c_delivery_by_environment_steps": (
                self.spec.stop_if_no_path_c_delivery_by_environment_steps
            ),
            "training_metrics": {
                "schema_version": TRAINING_METRICS_SCHEMA_VERSION,
                "path": str(self.metrics_path),
                "row_count": int(self.metrics_rows),
                "interval_environment_steps": int(
                    self.spec.metrics_interval_environment_steps
                ),
                "curve_path": str(curves_path),
            },
            "environment": self.env_config.to_mapping(),
            "architecture": ego_model.architecture_manifest(),
        }
        (self.output_dir / "training_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return manifest

    def _finish_partner_family_run(
        self,
        model: PrimitiveRecurrentEnsembleQ,
        partner_pool: Sequence[PrimitiveRecurrentEnsembleQ],
        self_play_metrics: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Close a candidate-producing run without claiming support admission."""

        realized_environment_steps = int(
            self_play_metrics["effective_environment_steps"]
        )
        if realized_environment_steps != self.spec.total_environment_steps:
            raise RuntimeError(
                "Q partner-family formation did not realize its frozen 10M budget."
            )
        if len(partner_pool) != len(self.spec.partner_pool_snapshot_steps):
            raise RuntimeError(
                "Q partner-family formation did not produce every frozen snapshot."
            )
        expected_metric_rows = (
            realized_environment_steps
            // self.spec.metrics_interval_environment_steps
        )
        if self.metrics_rows != expected_metric_rows:
            raise RuntimeError(
                "Partner-family metric row count differs from the frozen schedule."
            )
        if self.spec.partner_family_definition is None or (
            self.spec.partner_family_spec_sha256 is None
        ) or self.spec.training_run_id is None:
            raise RuntimeError("Partner-family semantics were not bound to the run.")
        final_checkpoint = (
            self.output_dir
            / "partner_pool"
            / f"step_{self.spec.self_play_environment_steps}.pt"
        )
        curves_path = self.output_dir / "training_curves.png"
        write_training_curves(self.metrics_path, curves_path)
        manifest = {
            "schema_version": "path_c_partner_family_training_artifact_v2",
            "run_kind": self.spec.run_kind,
            "training_mode": self.spec.training_mode,
            "scientific_readout_allowed": False,
            "support_status": "candidate_only",
            "seed": self.spec.seed,
            "layout": self.env_config.layout,
            "run_status": "completed",
            "effective_environment_steps": realized_environment_steps,
            "effective_episodes": self.episode_counter,
            "gradient_updates": self.gradient_updates,
            "probe_count": self.probe_count,
            "self_play": dict(self_play_metrics),
            "candidate_checkpoints": [
                str(self.output_dir / "partner_pool" / f"step_{step}.pt")
                for step in self.spec.partner_pool_snapshot_steps
            ],
            "candidate_checkpoint_bindings": list(
                self_play_metrics["candidate_checkpoint_bindings"]
            ),
            "final_candidate_checkpoint": str(final_checkpoint),
            "partner_family": dict(self.spec.partner_family_definition),
            "partner_family_spec_sha256": (
                self.spec.partner_family_spec_sha256
            ),
            "training_config_sha256": self.spec.training_config_sha256,
            "environment_config_sha256": self.spec.environment_config_sha256,
            "training_implementation_sha256": (
                self.spec.training_implementation_sha256
            ),
            "training_implementation_dependencies": dict(
                self.spec.training_implementation_dependencies
            ),
            "training_run_id": self.spec.training_run_id,
            "training_metrics": {
                "schema_version": TRAINING_METRICS_SCHEMA_VERSION,
                "path": str(self.metrics_path),
                "row_count": int(self.metrics_rows),
                "interval_environment_steps": int(
                    self.spec.metrics_interval_environment_steps
                ),
                "curve_path": str(curves_path),
            },
            "environment": self.env_config.to_mapping(),
            "architecture": model.architecture_manifest(),
        }
        (self.output_dir / "training_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return manifest

    def _append_metric_row(self, row: Mapping[str, Any]) -> None:
        if row.get("schema_version") != TRAINING_METRICS_SCHEMA_VERSION:
            raise ValueError("Training metric row uses an unexpected schema.")
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
        self.metrics_rows += 1
        health = row["model_parameter_health"]
        if int(health["learned"]["nonfinite_count"]) or int(
            health["fixed_prior"]["nonfinite_count"]
        ):
            raise RuntimeError("Training model contains non-finite parameters.")

    def _initial_pool_state(self, phase: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
        seeds = np.asarray(
            [
                derive_ocv2_execution_seed(
                    derive_standard_seed(self.spec.seed, phase, "env", index, 0)
                )
                for index in range(self.spec.batch_size_envs)
            ],
            dtype=np.uint32,
        )
        self.pool.reset(seeds)
        return self.pool.snapshot_obs(), np.zeros(self.spec.batch_size_envs, dtype=np.int64)

    def _new_builder(
        self,
        *,
        phase: str,
        env_index: int,
        agent_slot: int,
        episode_index: int,
        observation: np.ndarray,
        n_heads: int,
    ) -> PrimitiveEpisodeBuilder:
        return PrimitiveEpisodeBuilder(
            episode_id=(
                f"{phase}:seed={self.spec.seed}:env={env_index}:"
                f"slot={agent_slot}:episode={episode_index}"
            ),
            initial_observation=observation,
            n_actions=self.n_actions,
            n_heads=n_heads,
            bootstrap_p=self.spec.bootstrap_p,
            bootstrap_seed=derive_ocv2_execution_seed(
                derive_standard_seed(self.spec.seed, phase, "bootstrap")
            ),
        )

    def _update_model(
        self,
        model: PrimitiveRecurrentEnsembleQ,
        target: PrimitiveRecurrentEnsembleQ,
        optimizer: torch.optim.Optimizer,
        replay: PrimitiveEpisodeReplay,
    ) -> PrimitiveTDUpdateMetrics | None:
        if len(replay) < max(
            self.spec.learning_starts_episodes,
            self.spec.minibatch_episodes,
        ):
            return None
        records = replay.sample(self.spec.minibatch_episodes, rng=self.rng)
        batch = collate_primitive_episodes(
            records,
            observation_shape=self.observation_shape,
            n_actions=self.n_actions,
            n_heads=model.n_heads,
            gamma=self.spec.gamma,
            device=self.device,
        )
        model.train()
        target.eval()
        optimizer.zero_grad(set_to_none=True)
        output = primitive_sequence_td_loss(model, target, batch)
        if not bool(torch.isfinite(output.loss)):
            raise RuntimeError("Standard Path C TD loss became non-finite.")
        output.loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            self.spec.gradient_clip_norm,
            error_if_nonfinite=True,
        )
        optimizer.step()
        self.gradient_updates += 1
        return PrimitiveTDUpdateMetrics(
            loss=float(output.loss.detach().cpu().item()),
            per_head_loss=tuple(
                float(value) for value in output.per_head_loss.detach().cpu().tolist()
            ),
            per_head_support=tuple(
                int(value) for value in output.head_support.detach().cpu().tolist()
            ),
            gradient_norm=float(gradient_norm.detach().cpu().item()),
        )

    def _phase_training_setup(
        self,
        model: PrimitiveRecurrentEnsembleQ,
    ) -> tuple[
        PrimitiveRecurrentEnsembleQ,
        torch.optim.Optimizer,
        PrimitiveEpisodeReplay,
    ]:
        """Target network, optimizer and replay shared by both training phases."""

        target = copy.deepcopy(model).to(self.device).eval()
        target.compact_recurrent_parameters()
        model.assert_same_fixed_prior(target)
        optimizer = torch.optim.Adam(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=self.spec.learning_rate,
        )
        replay = PrimitiveEpisodeReplay(self.spec.replay_capacity_episodes)
        return target, optimizer, replay

    def _derive_reset_seeds(
        self,
        phase: str,
        done_indices: np.ndarray,
        episode_indices: np.ndarray,
    ) -> np.ndarray:
        """One deterministic execution seed per finished environment slot."""

        return np.asarray(
            [
                derive_ocv2_execution_seed(
                    derive_standard_seed(
                        self.spec.seed,
                        phase,
                        "env",
                        int(index),
                        int(episode_indices[index]),
                    )
                )
                for index in done_indices
            ],
            dtype=np.uint32,
        )

    def _drain_updates(
        self,
        model: PrimitiveRecurrentEnsembleQ,
        target: PrimitiveRecurrentEnsembleQ,
        optimizer: torch.optim.Optimizer,
        replay: PrimitiveEpisodeReplay,
        losses: list[float],
        diagnostics: TrainingWindowAccumulator,
    ) -> None:
        for _ in range(self.spec.updates_per_vector_step):
            update = self._update_model(model, target, optimizer, replay)
            if update is not None:
                losses.append(update.loss)
                diagnostics.record_update(
                    loss=update.loss,
                    per_head_loss=update.per_head_loss,
                    per_head_support=update.per_head_support,
                    gradient_norm=update.gradient_norm,
                )

    def _advance_target(
        self,
        model: PrimitiveRecurrentEnsembleQ,
        target: PrimitiveRecurrentEnsembleQ,
        phase_steps: int,
        next_target_update: int,
    ) -> int:
        if phase_steps >= next_target_update:
            target.load_state_dict(model.state_dict(), strict=True)
            while phase_steps >= next_target_update:
                next_target_update += self.spec.target_update_environment_steps
        return next_target_update

    def _run_self_play(
        self,
        model: PrimitiveRecurrentEnsembleQ,
    ) -> tuple[list[PrimitiveRecurrentEnsembleQ], dict[str, Any]]:
        phase_start_episodes = self.episode_counter
        target, optimizer, replay = self._phase_training_setup(model)
        obs, episode_indices = self._initial_pool_state("self_play")
        previous_actions = {
            slot: np.full(self.spec.batch_size_envs, self.n_actions, dtype=np.int64)
            for slot in (0, 1)
        }
        previous_rewards = {
            slot: np.zeros(self.spec.batch_size_envs, dtype=np.float32) for slot in (0, 1)
        }
        # Both parameter-shared slots reset on the same joint episode boundary.
        episode_starts = np.ones(self.spec.batch_size_envs, dtype=bool)
        states = {
            slot: model.initial_state(self.spec.batch_size_envs, device=self.device)
            for slot in (0, 1)
        }
        builders = {
            slot: [
                self._new_builder(
                    phase="self_play",
                    env_index=index,
                    agent_slot=slot,
                    episode_index=0,
                    observation=obs[f"agent_{slot}"][index],
                    n_heads=model.n_heads,
                )
                for index in range(self.spec.batch_size_envs)
            ]
            for slot in (0, 1)
        }
        partner_pool: list[PrimitiveRecurrentEnsembleQ] = []
        partner_checkpoint_bindings: list[dict[str, Any]] = []
        losses: list[float] = []
        diagnostics = TrainingWindowAccumulator(
            batch_size=self.spec.batch_size_envs,
            n_heads=model.n_heads,
            gradient_clip_norm=self.spec.gradient_clip_norm,
            reference=snapshot_trainable_parameters(model),
        )
        phase_steps = 0
        next_target_update = self.spec.target_update_environment_steps
        snapshot_set = set(self.spec.partner_pool_snapshot_steps)
        while phase_steps < self.spec.self_play_environment_steps:
            actions: dict[int, np.ndarray] = {}
            for slot in (0, 1):
                step_batch = single_step_batch(
                    obs[f"agent_{slot}"],
                    previous_actions[slot],
                    previous_rewards[slot],
                    episode_starts,
                    n_actions=self.n_actions,
                    device=self.device,
                )
                model.eval()
                with torch.no_grad():
                    q_values, _, states[slot] = model.forward_step(
                        step_batch,
                        states[slot],
                    )
                actions[slot] = choose_primitive_actions(
                    q_values,
                    step_batch.valid_actions[:, 0],
                    rng=self.rng,
                    epsilon=self.spec.epsilon(phase_steps),
                    probe_enabled=False,
                ).actions
            (
                next_obs_device,
                _,
                rewards_device,
                dones_device,
                info_device,
            ) = self.pool.step_joint(actions[0], actions[1])
            next_obs = {key: np.asarray(value) for key, value in next_obs_device.items()}
            rewards = shared_team_reward(rewards_device)
            shaped_by_slot = shaped_rewards_by_agent(
                info_device,
                batch_size=self.spec.batch_size_envs,
            )
            anneal_factor = phase_shaping_anneal_factor(
                phase_steps,
                self.spec.self_play_environment_steps,
            )
            annealed_shaped_by_slot = {
                slot: anneal_factor * shaped_by_slot[slot] for slot in (0, 1)
            }
            training_rewards_by_slot = {
                slot: rewards + annealed_shaped_by_slot[slot] for slot in (0, 1)
            }
            dones = np.asarray(dones_device["__all__"], dtype=bool)
            diagnostics.record_environment_batch(
                rewards,
                dones,
                probe_count=0,
                training_rewards=np.concatenate(
                    (training_rewards_by_slot[0], training_rewards_by_slot[1])
                ),
                training_shaped_rewards=np.concatenate(
                    (annealed_shaped_by_slot[0], annealed_shaped_by_slot[1])
                ),
                shaped_rewards_by_slot={
                    "agent_0": shaped_by_slot[0],
                    "agent_1": shaped_by_slot[1],
                },
                anneal_factor=anneal_factor,
            )
            for index in range(self.spec.batch_size_envs):
                for slot in (0, 1):
                    builders[slot][index].append(
                        action=int(actions[slot][index]),
                        training_reward=float(training_rewards_by_slot[slot][index]),
                        previous_raw_reward=float(rewards[index]),
                        done=bool(dones[index]),
                        next_observation=next_obs[f"agent_{slot}"][index],
                    )
                    if dones[index]:
                        replay.add(
                            builders[slot][index].finish(
                                observation_shape=self.observation_shape
                            )
                        )
                if dones[index]:
                    self.episode_counter += 1
            done_indices = np.flatnonzero(dones)
            if done_indices.size:
                episode_indices[done_indices] += 1
                reset_seeds = self._derive_reset_seeds(
                    "self_play",
                    done_indices,
                    episode_indices,
                )
                self.pool.reset_indices(done_indices, reset_seeds)
                next_obs = self.pool.snapshot_obs()
                for index in done_indices:
                    for slot in (0, 1):
                        builders[slot][int(index)] = self._new_builder(
                            phase="self_play",
                            env_index=int(index),
                            agent_slot=slot,
                            episode_index=int(episode_indices[index]),
                            observation=next_obs[f"agent_{slot}"][index],
                            n_heads=model.n_heads,
                        )
            for slot in (0, 1):
                previous_actions[slot] = actions[slot].copy()
                previous_rewards[slot] = rewards.copy()
                previous_actions[slot][dones] = self.n_actions
                previous_rewards[slot][dones] = 0.0
            episode_starts = dones.copy()
            obs = next_obs
            phase_steps += self.spec.batch_size_envs
            self._drain_updates(
                model,
                target,
                optimizer,
                replay,
                losses,
                diagnostics,
            )
            next_target_update = self._advance_target(
                model,
                target,
                phase_steps,
                next_target_update,
            )
            if phase_steps % self.spec.metrics_interval_environment_steps == 0:
                self._append_metric_row(
                    diagnostics.build_row(
                        seed=self.spec.seed,
                        layout=self.env_config.layout,
                        phase="self_play",
                        phase_environment_steps=phase_steps,
                        total_environment_steps=phase_steps,
                        cumulative_episodes=self.episode_counter,
                        gradient_updates=self.gradient_updates,
                        epsilon=self.spec.epsilon(phase_steps),
                        cumulative_probe_count=self.probe_count,
                        model=model,
                    )
                )
            if phase_steps in snapshot_set:
                snapshot = copy.deepcopy(model).to(self.device).eval()
                snapshot.compact_recurrent_parameters()
                for parameter in snapshot.parameters():
                    parameter.requires_grad_(False)
                partner_pool.append(snapshot)
                snapshot_metadata: dict[str, Any] = {
                    "layout": self.env_config.layout,
                }
                if self.spec.training_mode == SELF_PLAY_ONLY_TRAINING_MODE:
                    if self.spec.partner_family_definition is None or (
                        self.spec.partner_family_spec_sha256 is None
                    ) or self.spec.training_run_id is None:
                        raise RuntimeError(
                            "Partner-family checkpoint lacks frozen family semantics."
                        )
                    snapshot_metadata.update(
                        {
                            "partner_family": dict(
                                self.spec.partner_family_definition
                            ),
                            "partner_family_id": self.spec.partner_family_definition[
                                "family_id"
                            ],
                            "partner_family_spec_sha256": (
                                self.spec.partner_family_spec_sha256
                            ),
                            "training_config_sha256": (
                                self.spec.training_config_sha256
                            ),
                            "environment_config_sha256": (
                                self.spec.environment_config_sha256
                            ),
                            "training_implementation_sha256": (
                                self.spec.training_implementation_sha256
                            ),
                            "training_implementation_dependencies": dict(
                                self.spec.training_implementation_dependencies
                            ),
                            "training_run_id": self.spec.training_run_id,
                            "training_mode": self.spec.training_mode,
                            "scientific_readout_allowed": False,
                            "support_status": "candidate_only",
                        }
                    )
                checkpoint_binding = save_standard_checkpoint(
                    self.output_dir / "partner_pool" / f"step_{phase_steps}.pt",
                    snapshot,
                    seed=self.spec.seed,
                    environment_steps=phase_steps,
                    episodes=self.episode_counter,
                    phase="self_play_partner_snapshot",
                    extra_metadata=snapshot_metadata,
                )
                partner_checkpoint_bindings.append(checkpoint_binding)
        return partner_pool, {
            "effective_environment_steps": phase_steps,
            "completed_episodes": self.episode_counter - phase_start_episodes,
            "discarded_partial_agent_transitions": int(
                sum(
                    len(builder.actions)
                    for slot in (0, 1)
                    for builder in builders[slot]
                )
            ),
            "partner_pool_size": len(partner_pool),
            "candidate_checkpoint_bindings": partner_checkpoint_bindings,
            "mean_td_loss": None if not losses else float(np.mean(losses)),
        }

    def _partner_actions(
        self,
        partner_pool: Sequence[PrimitiveRecurrentEnsembleQ],
        partner_states: list[PrimitivePolicyState],
        partner_indices: np.ndarray,
        observations: np.ndarray,
        previous_actions: np.ndarray,
        previous_rewards: np.ndarray,
        episode_starts: np.ndarray,
    ) -> tuple[np.ndarray, list[PrimitivePolicyState]]:
        return select_partner_actions(
            partner_pool,
            partner_states,
            partner_indices,
            observations,
            previous_actions,
            previous_rewards,
            episode_starts,
            n_actions=self.n_actions,
            device=self.device,
            rng=self.rng,
        )

    def _run_path_c(
        self,
        model: PrimitiveRecurrentEnsembleQ,
        partner_pool: Sequence[PrimitiveRecurrentEnsembleQ],
    ) -> dict[str, Any]:
        phase_start_episodes = self.episode_counter
        phase_start_probes = self.probe_count
        if not partner_pool:
            raise ValueError("Path C phase requires a non-empty within-run partner pool.")
        target, optimizer, replay = self._phase_training_setup(model)
        obs, episode_indices = self._initial_pool_state("path_c")
        ego_slots = self.rng.integers(0, 2, size=self.spec.batch_size_envs, dtype=np.int64)
        partner_indices = self.rng.integers(
            0,
            len(partner_pool),
            size=self.spec.batch_size_envs,
            dtype=np.int64,
        )
        ego_state = model.initial_state(self.spec.batch_size_envs, device=self.device)
        partner_states = [
            partner.initial_state(self.spec.batch_size_envs, device=self.device)
            for partner in partner_pool
        ]
        ego_previous_actions = np.full(
            self.spec.batch_size_envs,
            self.n_actions,
            dtype=np.int64,
        )
        partner_previous_actions = ego_previous_actions.copy()
        ego_previous_rewards = np.zeros(self.spec.batch_size_envs, dtype=np.float32)
        partner_previous_rewards = np.zeros_like(ego_previous_rewards)
        episode_starts = np.ones(self.spec.batch_size_envs, dtype=bool)
        episode_step_indices = np.zeros(self.spec.batch_size_envs, dtype=np.int64)
        episode_probe_counts = np.zeros(self.spec.batch_size_envs, dtype=np.int64)

        row_range = np.arange(self.spec.batch_size_envs)

        def _slot_observation(
            slot_ids: np.ndarray,
            observations: Mapping[str, np.ndarray],
        ) -> np.ndarray:
            stacked = np.stack(
                (observations["agent_0"], observations["agent_1"]),
                axis=0,
            )
            return stacked[slot_ids, row_range]

        ego_obs = _slot_observation(ego_slots, obs)
        partner_obs = _slot_observation(1 - ego_slots, obs)
        builders = [
            self._new_builder(
                phase="path_c",
                env_index=index,
                agent_slot=int(ego_slots[index]),
                episode_index=0,
                observation=ego_obs[index],
                n_heads=model.n_heads,
            )
            for index in range(self.spec.batch_size_envs)
        ]
        losses: list[float] = []
        diagnostics = TrainingWindowAccumulator(
            batch_size=self.spec.batch_size_envs,
            n_heads=model.n_heads,
            gradient_clip_norm=self.spec.gradient_clip_norm,
            reference=snapshot_trainable_parameters(model),
        )
        phase_steps = 0
        correct_delivery_event_count = 0
        next_target_update = self.spec.target_update_environment_steps
        probe_enabled = self.probe_config["enabled"]
        while phase_steps < self.spec.path_c_environment_steps:
            ego_batch = single_step_batch(
                ego_obs,
                ego_previous_actions,
                ego_previous_rewards,
                episode_starts,
                n_actions=self.n_actions,
                device=self.device,
            )
            model.eval()
            with torch.no_grad():
                ego_q, _, ego_state = model.forward_step(ego_batch, ego_state)
            ego_decision = choose_primitive_actions(
                ego_q,
                ego_batch.valid_actions[:, 0],
                rng=self.rng,
                epsilon=self.spec.epsilon(phase_steps),
                probe_enabled=probe_enabled,
                disagreement_threshold=self.probe_config["disagreement_threshold"],
                max_probe_regret=self.probe_config["max_probe_regret"],
                probe_allowed=(
                    (episode_probe_counts < self.probe_config["probe_budget_per_episode"])
                    & (
                        episode_step_indices
                        < self.probe_config["probe_window_environment_steps"]
                    )
                ) if probe_enabled else None,
                disagreement_stat=self.probe_config["disagreement_stat"],
            )
            self.probe_count += int(ego_decision.is_probe.sum())
            episode_probe_counts += ego_decision.is_probe.astype(np.int64)
            partner_actions, partner_states = self._partner_actions(
                partner_pool,
                partner_states,
                partner_indices,
                partner_obs,
                partner_previous_actions,
                partner_previous_rewards,
                episode_starts,
            )
            actions_0 = np.where(ego_slots == 0, ego_decision.actions, partner_actions)
            actions_1 = np.where(ego_slots == 1, ego_decision.actions, partner_actions)
            (
                next_obs_device,
                _,
                rewards_device,
                dones_device,
                info_device,
            ) = self.pool.step_joint(actions_0, actions_1)
            next_obs = {key: np.asarray(value) for key, value in next_obs_device.items()}
            rewards = shared_team_reward(rewards_device)
            shaped_by_slot = shaped_rewards_by_agent(
                info_device,
                batch_size=self.spec.batch_size_envs,
            )
            ego_shaped_rewards = np.where(
                ego_slots == 0,
                shaped_by_slot[0],
                shaped_by_slot[1],
            )
            anneal_factor = phase_shaping_anneal_factor(
                phase_steps,
                self.spec.path_c_environment_steps,
            )
            annealed_ego_shaped_rewards = anneal_factor * ego_shaped_rewards
            training_rewards = rewards + annealed_ego_shaped_rewards
            dones = np.asarray(dones_device["__all__"], dtype=bool)
            correct_delivery_event_count += (
                decompose_raw_reward_events(rewards).correct_delivery_count
            )
            probe_diagnostic_arguments: dict[str, np.ndarray] = {}
            if probe_enabled:
                probe_diagnostic_arguments = {
                    "probe_candidate_disagreement": (
                        ego_decision.probe_candidate_disagreement
                    ),
                    "probe_candidate_q": ego_decision.probe_candidate_q,
                    "probe_candidate_regret": ego_decision.probe_candidate_regret,
                    "probe_candidate_equals_greedy": (
                        ego_decision.probe_candidate_equals_greedy
                    ),
                    "probe_disagreement_pass": (
                        ego_decision.probe_disagreement_pass
                    ),
                    "probe_regret_pass": ego_decision.probe_regret_pass,
                    "probe_budget_pass": (
                        ego_decision.probe_budget_pass
                    ),
                }
            diagnostics.record_environment_batch(
                rewards,
                dones,
                probe_count=int(ego_decision.is_probe.sum()),
                training_rewards=training_rewards,
                training_shaped_rewards=annealed_ego_shaped_rewards,
                shaped_rewards_by_slot={
                    "agent_0": shaped_by_slot[0],
                    "agent_1": shaped_by_slot[1],
                },
                anneal_factor=anneal_factor,
                **probe_diagnostic_arguments,
            )
            next_ego_obs = _slot_observation(ego_slots, next_obs)
            for index in range(self.spec.batch_size_envs):
                builders[index].append(
                    action=int(ego_decision.actions[index]),
                    training_reward=float(training_rewards[index]),
                    previous_raw_reward=float(rewards[index]),
                    done=bool(dones[index]),
                    next_observation=next_ego_obs[index],
                )
                if dones[index]:
                    replay.add(
                        builders[index].finish(observation_shape=self.observation_shape)
                    )
                    self.episode_counter += 1
            done_indices = np.flatnonzero(dones)
            if done_indices.size:
                episode_indices[done_indices] += 1
                reset_seeds = self._derive_reset_seeds(
                    "path_c",
                    done_indices,
                    episode_indices,
                )
                self.pool.reset_indices(done_indices, reset_seeds)
                next_obs = self.pool.snapshot_obs()
                ego_slots[done_indices] = self.rng.integers(
                    0,
                    2,
                    size=done_indices.size,
                    dtype=np.int64,
                )
                partner_indices[done_indices] = self.rng.integers(
                    0,
                    len(partner_pool),
                    size=done_indices.size,
                    dtype=np.int64,
                )
                next_ego_obs = _slot_observation(ego_slots, next_obs)
                for index in done_indices:
                    builders[int(index)] = self._new_builder(
                        phase="path_c",
                        env_index=int(index),
                        agent_slot=int(ego_slots[index]),
                        episode_index=int(episode_indices[index]),
                        observation=next_ego_obs[index],
                        n_heads=model.n_heads,
                    )
            next_partner_obs = _slot_observation(1 - ego_slots, next_obs)
            ego_previous_actions = ego_decision.actions.copy()
            partner_previous_actions = partner_actions.copy()
            ego_previous_rewards = rewards.copy()
            partner_previous_rewards = rewards.copy()
            episode_starts = dones.copy()
            episode_step_indices += 1
            episode_step_indices[dones] = 0
            episode_probe_counts[dones] = 0
            ego_previous_actions[dones] = self.n_actions
            partner_previous_actions[dones] = self.n_actions
            ego_previous_rewards[dones] = 0.0
            partner_previous_rewards[dones] = 0.0
            ego_obs = next_ego_obs
            partner_obs = next_partner_obs
            phase_steps += self.spec.batch_size_envs
            self._drain_updates(
                model,
                target,
                optimizer,
                replay,
                losses,
                diagnostics,
            )
            next_target_update = self._advance_target(
                model,
                target,
                phase_steps,
                next_target_update,
            )
            if phase_steps % self.spec.metrics_interval_environment_steps == 0:
                self._append_metric_row(
                    diagnostics.build_row(
                        seed=self.spec.seed,
                        layout=self.env_config.layout,
                        phase="path_c",
                        phase_environment_steps=phase_steps,
                        total_environment_steps=(
                            self.spec.self_play_environment_steps + phase_steps
                        ),
                        cumulative_episodes=self.episode_counter,
                        gradient_updates=self.gradient_updates,
                        epsilon=self.spec.epsilon(phase_steps),
                        cumulative_probe_count=self.probe_count,
                        model=model,
                    )
                )
            if (
                self.spec.stop_if_no_path_c_delivery_by_environment_steps
                is not None
                and phase_steps
                == self.spec.stop_if_no_path_c_delivery_by_environment_steps
                and correct_delivery_event_count == 0
            ):
                return {
                    "effective_environment_steps": phase_steps,
                    "completed_episodes": (
                        self.episode_counter - phase_start_episodes
                    ),
                    "discarded_partial_agent_transitions": int(
                        sum(len(builder.actions) for builder in builders)
                    ),
                    "probe_count": self.probe_count - phase_start_probes,
                    "mean_td_loss": (
                        None if not losses else float(np.mean(losses))
                    ),
                    "ego_slot_assignment": "uniform_per_episode",
                    "correct_delivery_event_count": 0,
                    "stopped_early": True,
                    "stop_reason": (
                        "no positive raw delivery reward by the preregistered "
                        f"{phase_steps}-step Path C checkpoint"
                    ),
                }
        return {
            "effective_environment_steps": phase_steps,
            "completed_episodes": self.episode_counter - phase_start_episodes,
            "discarded_partial_agent_transitions": int(
                sum(len(builder.actions) for builder in builders)
            ),
            "probe_count": self.probe_count - phase_start_probes,
            "mean_td_loss": None if not losses else float(np.mean(losses)),
            "ego_slot_assignment": "uniform_per_episode",
            "correct_delivery_event_count": int(correct_delivery_event_count),
            "stopped_early": False,
        }


def run_standard_training(config_path: str | Path) -> dict[str, Any]:
    config = load_standard_training_config(config_path)
    return StandardPathCTrainer(config).run()
