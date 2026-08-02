"""Core immutable PyTree-compatible records for DELTA-ZSC.

The active implementation deliberately contains no partner-specific actor/critic
branches or finite partner-type heads.  Every field
below is either a deployable legal-history quantity or an explicitly marked
training-only quantity.
"""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple


class GaussianMixtureBelief(NamedTuple):
    """Online posterior over the continuous decision context.

    Shapes use an arbitrary batch prefix ``...`` followed by
    ``[mixture_components]`` or ``[mixture_components, latent_dim]``.
    """

    recurrent_carry: Any
    mixture_logits: Any
    means: Any
    log_variances: Any
    support_score: Any


class PolicyState(NamedTuple):
    """Complete deployable recurrent state of one ego policy."""

    task_carry: Any
    belief: GaussianMixtureBelief
    previous_observation: Any
    previous_action: Any
    previous_reward: Any
    episode_start: Any


class ContextOutput(NamedTuple):
    """Task and continuous-belief state without policy/decoder evaluation."""

    task_features: Any
    mixture_logits: Any
    mixture_means: Any
    mixture_log_variances: Any
    support_score: Any


class ModelOutput(NamedTuple):
    """Compact policy/value output of the shared DELTA-ZSC model."""

    task_features: Any
    belief_embedding: Any
    mixture_logits: Any
    mixture_means: Any
    mixture_log_variances: Any
    support_score: Any
    base_logits: Any
    residual_logits: Any
    gate: Any
    execution_logits: Any
    state_value: Any
    raw_q1: Any
    raw_q2: Any
    action_values: Any


class ResponsePrediction(NamedTuple):
    """Structured visible-partner prediction for the executed action only."""

    visibility_logit: Any
    relative_position_logits: Any
    direction_logits: Any
    inventory_logits: Any
    interaction_change_logit: Any
    diagnostic_reward_mean: Any
    done_logit: Any


class PartnerGeneratorState(NamedTuple):
    """Recurrent state of the single continuous partner generator."""

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
    """A complete recurrent PPO batch.

    ``observations`` and recurrent inputs have T+1 states.  Transition fields
    have T rows.  ``response_next_observations`` stores terminal observations
    on done transitions, while ``observations`` continues with the reset state.
    """

    observations: Any
    response_next_observations: Any
    previous_actions: Any
    previous_rewards: Any
    episode_starts: Any
    action_keys: Any
    gate_overrides: Any
    actions: Any
    rewards: Any
    official_shaped_rewards: Any
    official_shaping_factors: Any
    decision_regret_shaping: Any
    shaped_rewards: Any
    dones: Any
    old_log_probabilities: Any
    old_values: Any
    ppo_mask: Any
    partner_codes: Any
    partner_sources: Any
    partner_run_ids: Any
    initial_policy_state: PolicyState
    initial_target_policy_state: PolicyState


class CounterfactualAnchorBatch(NamedTuple):
    """Simulator-return supervision for all ego actions at anchor states.

    The anchor retains the legal online policy state and current observation so
    task/belief features are recomputed under the candidate parameters.  It also
    stores the rollout index and partner source/code for lineage only.  No
    hidden code or privileged diagnostic context is a supervised online target.
    """

    anchor_ids: Any
    rollout_flat_indexes: Any
    policy_states: PolicyState
    observations: Any
    partner_codes: Any
    partner_sources: Any
    fit_returns_by_action: Any
    evaluation_returns_by_action: Any
    partner_run_ids: Any
    action_mask: Any


class QuotientPairBatch(NamedTuple):
    """Matched anchor pairs with empirical decision distances."""

    anchor_index_a: Any
    anchor_index_b: Any
    decision_distance: Any
    weights: Any


class CalibrationArtifact(NamedTuple):
    """Frozen partner-run-block conformal gate artifact.

    The support model is fit only from training-support posterior summaries and
    then thresholded on run-disjoint calibration partners.  Keeping the complete
    Mahalanobis model in the artifact makes deployment deterministic and avoids
    an untrained neural ``support_score`` head.
    """

    alpha: Any
    residual_radius: Any
    monte_carlo_radius: Any
    support_threshold: Any
    support_mean: Any
    support_precision: Any
    support_distance_scale: Any
    calibration_run_count: Any
    model_fingerprint: Any


class TrainState(NamedTuple):
    """Complete r3 signal-contract checkpoint state.

    There is intentionally no compatibility tail for r2.  Orbax restoration
    therefore fails before training when an archived r2 checkpoint is supplied.
    Every optimizer counter, qualification decision, policy epoch, random
    domain and replay fingerprint that can change the next update is explicit.
    """
    params: Any
    target_params: Any
    ppo_optimizer_state: Any
    raw_q_optimizer_state: Any
    response_optimizer_state: Any
    generator_optimizer_state: Any

    generator_params: Any
    generator_target_params: Any

    target_policy_epoch: Any
    qualified_base_params: Any
    owner_source_artifact: Any
    last_qualified_generator_params: Any
    last_qualified_generator_optimizer_state: Any
    last_qualified_generator_optimizer_step: Any
    generator_signature_readout: Any
    generator_snapshot_archive: Any
    generator_admission_passes: Any
    partner_source_probabilities: Any

    qualification: Any
    curriculum_phase: Any
    anchor_training_replay: Any
    anchor_audit_manifest: Any

    kl_multiplier: Any
    residual_multiplier: Any
    raw_q_calibration_error: Any
    raw_q_minimum_margin: Any

    ppo_optimizer_step: Any
    raw_q_optimizer_step: Any
    response_optimizer_step: Any
    generator_optimizer_step: Any

    runner_state: Any
    random_key: Any
    update_count: Any
    effective_environment_steps: Any
    random_domains: Any
    resource_ledger: Any
    calibration: CalibrationArtifact


class TrainingCoreState(NamedTuple):
    """The only state carried through the compiled PPO minibatch scan.

    Generator, runner, RNG, calibration, and accounting state deliberately stay
    outside the gradient executable.  This keeps the compiled signature stable
    without changing the sequential optimizer or Polyak-update semantics.
    """

    params: Any
    target_params: Any
    ppo_optimizer_state: Any


class GeneratorCoreState(NamedTuple):
    """Training-only generator state carried by its compiled update."""

    params: Any
    target_params: Any
    optimizer_state: Any


class AuxiliaryCoreState(NamedTuple):
    """Shared parameters with independent raw-Q and response optimizer states."""

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
    adaptation_enabled_steps: int
    base_policy_steps: int
    mean_support_score: float
    mean_predicted_gain: float
    negative_transfer: bool


__all__ = [
    "CalibrationArtifact",
    "CounterfactualAnchorBatch",
    "EvaluationRow",
    "GaussianMixtureBelief",
    "GeneratorCoreState",
    "LossBundle",
    "ModelOutput",
    "PartnerGeneratorOutput",
    "PartnerGeneratorState",
    "PolicyState",
    "QuotientPairBatch",
    "RolloutBatch",
    "ResponsePrediction",
    "TrainingCoreState",
    "TrainState",
    "TrainingUpdate",
]
