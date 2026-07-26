"""State, output, batch, and checkpoint records for Path C model version four."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from numbers import Integral, Real
from typing import Any, Mapping, NamedTuple


class VQBCPolicyState(NamedTuple):
    reference_carry: Any
    value_carry: Any
    slot_log_belief: Any
    previous_action: Any
    previous_team_reward: Any
    episode_start: Any
    log_temperature: Any
    generic_log_temperature: Any


class VQBCOutput(NamedTuple):
    features: Any
    q_values: Any
    learned_q_values: Any
    prior_q_values: Any
    centered_advantages: Any
    response_logits: Any
    response_probabilities: Any
    reward_mean: Any
    reward_log_standard_deviation: Any
    next_q_mean: Any
    next_q_log_standard_deviation: Any
    quotient_ids: Any
    j_use: Any
    j_mask: Any
    information_gain: Any
    execution_logits: Any


class VQBCDecisionRecord(NamedTuple):
    reference_logits: Any
    execution_logits: Any
    j_use: Any
    j_mask: Any
    kl_divergence: Any
    action: Any
    reference_greedy_action: Any
    quotient_count: Any
    belief_entropy: Any
    response_code: Any


class VQBCRolloutState(NamedTuple):
    environment_state: Any
    observations: Any
    policy_state: VQBCPolicyState
    partner_carry: Any
    partner_member_index: Any
    ego_seat: Any
    episode_step: Any
    episode_id: Any
    episode_return: Any
    completed_episodes: Any
    effective_environment_steps: Any
    random_key: Any


class VQBCRolloutBatch(NamedTuple):
    """Training tensors only; partner identity and provenance are excluded."""

    observations: Any
    response_next_observations: Any
    episode_start: Any
    previous_actions: Any
    previous_team_rewards: Any
    initial_value_carry: Any
    reference_logits: Any
    execution_logits: Any
    generic_execution_logits: Any
    slot_log_beliefs: Any
    actions: Any
    rewards: Any
    dones: Any
    response_codes: Any
    episode_ids: Any
    episode_steps: Any
    completed_episode_returns: Any
    quotient_counts: Any
    j_use: Any
    j_mask: Any


class VQBCCodebookState(NamedTuple):
    embeddings: Any
    exponential_counts: Any
    exponential_sums: Any
    unused_rollouts: Any
    initialized: Any
    replacement_count: Any
    last_replaced_codes: Any


class VQBCKLState(NamedTuple):
    log_temperature: Any
    generic_log_temperature: Any


class VQBCTrainState(NamedTuple):
    online_params: Any
    target_params: Any
    bellman_optimizer_state: Any
    outcome_optimizer_state: Any
    codebook: VQBCCodebookState
    kl_state: VQBCKLState
    rollout_state: VQBCRolloutState
    random_key: Any
    effective_environment_steps: Any
    completed_episodes: Any
    update_count: Any


@dataclass(frozen=True, slots=True)
class CheckpointMetadataV3:
    """Content bindings for a resumable fourth-version checkpoint."""

    schema_version: str
    run_kind: str
    scientific_readout_allowed: bool
    outer_unit_id: int | None
    reference_checkpoint_path: str
    reference_training_run_id: str
    reference_weights_sha256: str
    reference_launch_config_path: str
    reference_launch_config_sha256: str
    config_sha256: str
    model_weights_sha256: str
    state_sha256: str
    effective_environment_steps: int
    completed_episodes: int
    update_count: int
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != "path_c_model_checkpoint_metadata_v3":
            raise ValueError("Fourth-model checkpoint metadata schema changed.")
        if self.run_kind not in {"development", "formal"}:
            raise ValueError("Checkpoint run_kind must be development or formal.")
        if self.scientific_readout_allowed:
            raise ValueError("Fourth-model implementation checkpoints are non-claim.")
        if self.outer_unit_id is not None and (
            isinstance(self.outer_unit_id, bool)
            or not isinstance(self.outer_unit_id, Integral)
            or int(self.outer_unit_id) < 0
        ):
            raise ValueError("outer_unit_id must be null or non-negative.")
        for name in (
            "reference_checkpoint_path",
            "reference_training_run_id",
            "reference_launch_config_path",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be non-empty.")
        for name in (
            "reference_weights_sha256",
            "reference_launch_config_sha256",
            "config_sha256",
            "model_weights_sha256",
            "state_sha256",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or not set(value).issubset(frozenset("0123456789abcdef"))
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 string.")
        for name in (
            "effective_environment_steps",
            "completed_episodes",
            "update_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if not isinstance(self.extra, Mapping):
            raise TypeError("Checkpoint extra metadata must be a mapping.")

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CheckpointMetadataV3":
        if not isinstance(payload, Mapping):
            raise TypeError("Checkpoint metadata must be a mapping.")
        fields = {item.name for item in cls.__dataclass_fields__.values()}
        unknown = sorted(set(payload) - fields)
        missing = sorted((fields - {"extra"}) - set(payload))
        if unknown or missing:
            raise ValueError(
                f"Checkpoint metadata fields are invalid; unknown={unknown}, "
                f"missing={missing}."
            )
        return cls(**{name: payload[name] for name in fields if name in payload})


def finite_scalar(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


__all__ = [
    "CheckpointMetadataV3",
    "VQBCCodebookState",
    "VQBCDecisionRecord",
    "VQBCKLState",
    "VQBCOutput",
    "VQBCPolicyState",
    "VQBCRolloutBatch",
    "VQBCRolloutState",
    "VQBCTrainState",
]
