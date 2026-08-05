"""Immutable PyTree-compatible records for unified DELTA-ZSC.

The active state contains only legal deployment history.  Privileged
counterfactual observations appear exclusively in :class:`AnchorBatch` and are
never copied into :class:`PolicyState`.
"""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple


class BehaviorStatistics(NamedTuple):
    """Independent Beta posteriors for four observable behaviour rates."""

    alpha: Any
    beta: Any


class PolicyState(NamedTuple):
    """Complete deployable recurrent state."""

    task_carry: Any
    belief: Any
    behavior: BehaviorStatistics
    previous_observation: Any
    previous_action: Any
    episode_start: Any


class ResponsePrediction(NamedTuple):
    """Factorized response distribution with a latent-component axis."""

    visibility_logit: Any
    relative_position_logits: Any
    direction_logits: Any
    inventory_logits: Any
    inventory_change_logit: Any


class DecisionPrediction(NamedTuple):
    """Per-component action-return mean and positive model variance."""

    means: Any
    variances: Any


class VOIResult(NamedTuple):
    """Myopic response-value quadrature and report-only diagnostics."""

    value: Any
    raw_value: Any
    expected_posterior_value: Any
    prior_value: Any
    expected_posterior_entropy: Any
    predictive_entropy: Any
    expected_information_gain: Any
    quadrature_error_estimate: Any


class ModelOutput(NamedTuple):
    """One legal model step or a time-major sequence of steps."""

    task_features: Any
    instant_partner: Any
    base_policy_logits: Any
    policy_logits: Any
    value: Any
    predictive_belief: Any
    belief: Any
    behavior_features: Any
    response_prediction: ResponsePrediction
    response_negative_log_likelihood: Any
    component_decision_means: Any
    component_decision_variances: Any
    expected_decision_values: Any
    active_voi: Any
    active_voi_raw: Any
    active_information_gain: Any
    active_voi_quadrature_error: Any
    adaptation_kl: Any
    adaptation_temperature: Any


class RolloutBatch(NamedTuple):
    """A time-major on-policy rollout.

    State-like arrays contain ``T+1`` rows.  Transition arrays contain ``T``
    rows.  ``response_next_observations`` stores the terminal frame on a done
    transition even though ``observations[t+1]`` contains the reset frame.
    """

    observations: Any
    response_next_observations: Any
    previous_actions: Any
    episode_starts: Any
    actions: Any
    rewards: Any
    shaped_rewards: Any
    dones: Any
    old_log_probabilities: Any
    old_values: Any
    ppo_mask: Any
    initial_policy_state: PolicyState


class AnchorBatch(NamedTuple):
    """Sparse CRN all-action decision observations."""

    time_indexes: Any
    lane_indexes: Any
    fit_returns_by_action: Any
    evaluation_returns_by_action: Any
    measurement_covariances: Any
    action_mask: Any
    fit_replica_returns_by_action: Any
    evaluation_replica_returns_by_action: Any


class AnchorSnapshots(NamedTuple):
    """The sparse pre-action worlds selected from an anchor rollout."""

    time_indexes: Any
    lane_indexes: Any
    environment_state: Any
    observations: Any
    ego_state: Any
    partner_state: Any
    partner_episode_start: Any
    ego_roles: Any


class AdamState(NamedTuple):
    count: Any
    first_moment: Any
    second_moment: Any


class RunnerState(NamedTuple):
    environment_state: Any
    joint_observations: Any
    ego_policy_state: PolicyState
    partner_state: Any
    partner_episode_start: Any
    ego_roles: Any
    random_key: Any
    completed_episodes: Any


class TrainState(NamedTuple):
    base_params: Any
    latent_params: Any
    base_optimizer_state: AdamState
    latent_optimizer_state: AdamState
    runner_state: RunnerState
    update_count: Any
    effective_environment_steps: Any
    resource_ledger: Mapping[str, Any]


class LossResult(NamedTuple):
    total: Any
    metrics: Mapping[str, Any]


__all__ = [
    "AdamState",
    "AnchorBatch",
    "AnchorSnapshots",
    "BehaviorStatistics",
    "DecisionPrediction",
    "LossResult",
    "ModelOutput",
    "PolicyState",
    "ResponsePrediction",
    "RolloutBatch",
    "RunnerState",
    "TrainState",
    "VOIResult",
]
