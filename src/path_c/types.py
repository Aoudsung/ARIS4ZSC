"""Immutable PyTree records for DEPI (DELTA-ZSC foundation batch).

METHOD_SPEC §1 defines the three scientific objects x_t / u / c_t with
structural input-layer isolation.  The deployable policy carries one task
encoder, one capability encoder and one protocol encoder; training-only
objects are explicit so that deployment pruning and checkpoint identity can be
checked mechanically.
"""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple


class ProtocolContext(NamedTuple):
    """Recurrent state of the capability and protocol pathways.

    METHOD_SPEC §1.1: u is the stable partner-capability embedding (16) and
    pi is the categorical protocol posterior over K=4 regimes.
    """

    capability_carry: Any
    protocol_carry: Any
    capability: Any
    protocol_logits: Any


class PolicyState(NamedTuple):
    """Complete legal recurrent state exposed by the Official policy wrapper.

    METHOD_SPEC §1.4 field list: (task_carry, capability_carry,
    protocol_carry, context_summary, previous_observation, previous_action,
    episode_start).  ``context_summary`` stores concat(u, c) from the last
    step (32 dimensions) for deployment continuity.
    """

    task_carry: Any
    capability_carry: Any
    protocol_carry: Any
    context_summary: Any
    previous_observation: Any
    previous_action: Any
    episode_start: Any


class ContextOutput(NamedTuple):
    """METHOD_SPEC §1.4: ContextOutput carries (task_features, u, pi, c)."""

    task_features: Any
    capability: Any
    protocol_probabilities: Any
    protocol_embedding: Any


class ModelOutput(NamedTuple):
    task_features: Any
    capability: Any
    protocol_probabilities: Any
    protocol_embedding: Any
    context_summary: Any
    posterior_entropy: Any
    policy_logits: Any
    state_value: Any
    raw_q1: Any
    raw_q2: Any
    action_values: Any


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
    """Matched-pair payload for the §5.3 frozen comparator data path.

    ``comparator_accuracy``/``ego_state_*``/``probe_observations`` stay
    ``None`` for legacy constructions; the §5 anchor pipeline fills them so
    ``separation_terms_from_matched_pairs`` can build the ``SeparationTerms``
    payload consumed inside the jit-compiled combined-loss scan.
    """

    anchor_index_a: Any
    anchor_index_b: Any
    decision_distance: Any
    weights: Any
    comparator_accuracy: Any = None
    ego_state_a: Any = None
    ego_state_b: Any = None
    probe_observations: Any = None


class SeparationTerms(NamedTuple):
    """§3.2 L_separation payload carried into the jit-compiled scan.

    The classification output of the §5.3 frozen comparator (equivalent /
    distinct masks, per-pair weights, margin) is precomputed at anchor
    trigger time and stays constant between triggers.  The two forward
    passes over ``ego_state_*`` / ``probe_observations`` run inside
    ``compute_loss`` against the *current* params, so L_separation joins
    the same ``value_and_grad`` as the other three losses and its gradient
    reaches the capability/protocol encoders (METHOD_SPEC §3.2/§3.4).
    Shapes are fixed per anchor trigger, keeping the compiled scan stable.
    """

    ego_state_a: Any
    ego_state_b: Any
    probe_observations: Any
    equivalent_mask: Any
    weights: Any
    margin: Any


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
    "GeneratorCoreState",
    "LossBundle",
    "ModelOutput",
    "PartnerGeneratorOutput",
    "PartnerGeneratorState",
    "PolicyState",
    "ProtocolContext",
    "QuotientPairBatch",
    "RolloutBatch",
    "TrainingCoreState",
    "TrainState",
    "TrainingUpdate",
]
