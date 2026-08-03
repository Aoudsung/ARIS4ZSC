"""Immutable PyTree records for DELTA-ZSC V6.

The deployable policy contains one legal-history task encoder, one diagonal
Gaussian partner belief and one belief-conditioned actor.  Training-only
objects are explicit so that deployment pruning and checkpoint identity can be
checked mechanically.
"""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple


class GaussianBelief(NamedTuple):
    recurrent_carry: Any
    mean: Any
    log_standard_deviation: Any
    normalized_uncertainty: Any


class PolicyState(NamedTuple):
    """Complete legal recurrent state exposed by the Official policy wrapper."""

    task_carry: Any
    belief: GaussianBelief
    previous_observation: Any
    previous_action: Any
    episode_start: Any


class ContextOutput(NamedTuple):
    task_features: Any
    belief_mean: Any
    belief_log_standard_deviation: Any
    normalized_uncertainty: Any


class ModelOutput(NamedTuple):
    task_features: Any
    belief_summary: Any
    belief_mean: Any
    belief_log_standard_deviation: Any
    normalized_uncertainty: Any
    policy_logits: Any
    state_value: Any
    raw_q1: Any
    raw_q2: Any
    action_values: Any


class ResponsePrediction(NamedTuple):
    visibility_logit: Any
    relative_position_logits: Any
    direction_logits: Any
    inventory_logits: Any
    interaction_change_logit: Any


class PartnerGeneratorState(NamedTuple):
    carry: Any
    code: Any
    episode_start: Any


class PartnerGeneratorOutput(NamedTuple):
    logits: Any
    value: Any
    action: Any
    log_probability: Any
    entropy: Any


class RolloutBatch(NamedTuple):
    """Recurrent PPO batch with T+1 legal-history states and T transitions."""

    observations: Any
    response_next_observations: Any
    previous_actions: Any
    episode_starts: Any
    action_keys: Any
    context_dropout_masks: Any
    actions: Any
    rewards: Any
    official_shaped_rewards: Any
    official_shaping_factors: Any
    decision_regret_shaping: Any
    shaped_rewards: Any
    dones: Any
    old_log_probabilities: Any
    old_values: Any
    behavior_probabilities: Any
    ppo_mask: Any
    partner_codes: Any
    partner_sources: Any
    partner_run_ids: Any
    initial_policy_state: PolicyState
    initial_target_policy_state: PolicyState


class CounterfactualAnchorBatch(NamedTuple):
    """Soft-policy-drift replay rows for real-return all-action supervision."""

    anchor_ids: Any
    rollout_flat_indexes: Any
    policy_states: PolicyState
    observations: Any
    partner_codes: Any
    partner_sources: Any
    partner_run_ids: Any
    fit_returns_by_action: Any
    return_sum_by_action: Any
    return_squared_sum_by_action: Any
    replica_count: Any
    collection_policy_logits: Any
    collection_update: Any
    collection_target_fingerprint: Any
    matched_pair_ids: Any
    action_mask: Any


class QuotientPairBatch(NamedTuple):
    anchor_index_a: Any
    anchor_index_b: Any
    decision_distance: Any
    weights: Any


class CalibrationArtifact(NamedTuple):
    """Optional E2E+Safety wrapper; never part of the primary method."""

    alpha: Any
    gain_residual_radius: Any
    support_threshold: Any
    support_mean: Any
    support_precision: Any
    support_distance_scale: Any
    calibration_run_count: Any
    model_fingerprint: Any


class TrainState(NamedTuple):
    """Complete V6 checkpoint state; deliberately incompatible with V5."""

    params: Any
    target_params: Any

    ppo_optimizer_state: Any
    raw_q_optimizer_state: Any
    response_optimizer_state: Any
    belief_optimizer_state: Any
    generator_optimizer_state: Any

    generator_params: Any
    generator_target_params: Any
    generator_competence_multiplier: Any
    generator_cvar_ema: Any
    external_reference_cvar_ema: Any

    anchor_replay: Any
    anchor_sampling_counter: Any

    belief_gradient_norm_ema: Any
    action_range_ema: Any
    anchor_advantage_scale_ema: Any

    ppo_optimizer_step: Any
    raw_q_optimizer_step: Any
    response_optimizer_step: Any
    belief_optimizer_step: Any
    generator_optimizer_step: Any

    runner_state: Any
    random_domains: Any
    update_count: Any
    effective_environment_steps: Any
    resource_ledger: Any


class TrainingCoreState(NamedTuple):
    params: Any
    target_params: Any
    ppo_optimizer_state: Any


class GeneratorCoreState(NamedTuple):
    params: Any
    target_params: Any
    optimizer_state: Any


class AuxiliaryCoreState(NamedTuple):
    params: Any
    raw_q_optimizer_state: Any
    response_optimizer_state: Any
    raw_q_optimizer_step: Any
    response_optimizer_step: Any


class LossBundle(NamedTuple):
    total: Any
    metrics: Mapping[str, Any]


class TrainingUpdate(NamedTuple):
    state: TrainState
    metrics: Mapping[str, Any]


class EvaluationRow(NamedTuple):
    ego_run_id: str
    partner_run_id: str
    partner_mechanism: str
    episode_index: int
    episode_seed: int
    raw_return: float
    correct_deliveries: int
    wrong_deliveries: int
    mean_belief_uncertainty: float
    mean_predicted_gain: float
    negative_transfer: bool


__all__ = [
    "AuxiliaryCoreState",
    "CalibrationArtifact",
    "ContextOutput",
    "CounterfactualAnchorBatch",
    "EvaluationRow",
    "GaussianBelief",
    "GeneratorCoreState",
    "LossBundle",
    "ModelOutput",
    "PartnerGeneratorOutput",
    "PartnerGeneratorState",
    "PolicyState",
    "QuotientPairBatch",
    "ResponsePrediction",
    "RolloutBatch",
    "TrainingCoreState",
    "TrainState",
    "TrainingUpdate",
]
