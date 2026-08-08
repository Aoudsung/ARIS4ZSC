"""Immutable PyTree-compatible records for DELTA-ZSC v4.

The deployable state contains only legal interaction history.  Privileged
counterfactual observations appear exclusively in :class:`AnchorBatch`; they
are never copied into :class:`PolicyState` or the response-only posterior.
"""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple


class BehaviorStatistics(NamedTuple):
    """Independent Beta posteriors for six legal observable behaviour rates."""

    alpha: Any
    beta: Any


class PolicyState(NamedTuple):
    """Complete deployable recurrent state for one episode-static latent.

    ``probe_continuation_pending`` is true only for the single intervening
    action after an active probe.  That action is executed by the collection-
    time base policy, matching the delayed response and post-response decision
    estimands.  It is reset at every episode boundary.
    """

    task_carry: Any
    belief: Any
    behavior: BehaviorStatistics
    previous_observation: Any
    previous_action: Any
    episode_start: Any
    probe_continuation_pending: Any


class DirectResponseTarget(NamedTuple):
    """Immediate teammate geometry retained for passive semantic filtering."""

    visibility: Any
    relative_position: Any
    direction: Any
    inventory: Any
    inventory_change: Any
    visible_mask: Any
    event_mask: Any
    movement: Any
    carrying: Any


class ResponseTarget(NamedTuple):
    """Complete immediate direct/interface response observation."""

    direct: DirectResponseTarget
    interface_available: Any
    interface_changed: Any
    interface_event: Any
    recipe_mask: Any
    recipe_changed: Any


class ProbeResponseTarget(NamedTuple):
    """Two-step delayed response to an ego probe.

    The target is extracted from ``o[t+1] -> o[t+2]`` and removes the direct
    physical effect of the second ego action. ``valid_mask`` is zero whenever
    either intervening transition crosses an episode boundary.
    """

    visibility: Any
    interface_available: Any
    interface_changed: Any
    interface_event: Any
    valid_mask: Any


class DirectResponsePrediction(NamedTuple):
    """Immediate response distribution.

    Occurrence heads have no component axis.  Conditional geometry heads have
    a component axis and are the only direct-response factors allowed to alter
    the episode-static posterior.
    """

    visibility_logit: Any
    relative_position_logits: Any
    direction_logits: Any
    inventory_logits: Any
    inventory_change_logit: Any


class ResponsePrediction(NamedTuple):
    """Immediate response emission with shared occurrence and semantic factors."""

    direct: DirectResponsePrediction
    interface_availability_logit: Any
    interface_change_logit: Any
    interface_event_logits: Any
    recipe_change_logit: Any


class ProbeResponsePrediction(NamedTuple):
    """Compact delayed response used by exact active VOI."""

    visibility_logit: Any
    interface_availability_logit: Any
    interface_change_logit: Any
    interface_event_logits: Any


class DecisionPrediction(NamedTuple):
    """Shared action-value baseline plus centered component residual."""

    means: Any
    variances: Any
    shared_means: Any
    component_residuals: Any


class VOIResult(NamedTuple):
    """Exact value under the 66-outcome delayed probe-response marginal."""

    value: Any
    expected_posterior_value: Any
    prior_value: Any
    expected_posterior_entropy: Any
    predictive_entropy: Any
    expected_information_gain: Any


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
    probe_response_prediction: ProbeResponsePrediction
    response_negative_log_likelihood: Any
    component_decision_means: Any
    component_decision_variances: Any
    component_successor_decision_means: Any
    component_successor_decision_variances: Any
    expected_decision_values: Any
    active_voi: Any
    active_information_gain: Any
    active_probe_eligible: Any
    adaptation_kl: Any
    adaptation_temperature: Any


class RolloutBatch(NamedTuple):
    """A time-major on-policy rollout.

    State-like arrays contain ``T+1`` rows. Transition arrays contain ``T``
    rows. ``response_next_observations`` stores the terminal frame on a done
    transition even though ``observations[t+1]`` contains the reset frame.

    Consecutive rows encode the registered delayed probe window without an
    additional observation copy: for ``t < T-1``, the target transition is
    ``response_next_observations[t] -> response_next_observations[t+1]`` under
    ``actions[t+1]`` and is valid iff neither ``dones[t]`` nor ``dones[t+1]``.

    ``beliefs`` holds the posterior the actor was conditioned on at collection
    time.  PPO replays these rather than recomputing them, because the latent
    parameters commit before the PPO update and a recomputed posterior would
    make the replay off-policy with respect to the behaviour policy.
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
    beliefs: Any
    initial_policy_state: PolicyState


class AnchorBatch(NamedTuple):
    """Sparse CRN current and delayed post-response all-action observations.

    For each probe, the probe transition and one sampled base-continuation
    transition are executed before the all-action matrix is forced.  Their
    rewards are excluded because the delayed response is not usable until the
    resulting ``t+2`` observation.
    """

    time_indexes: Any
    lane_indexes: Any
    fit_returns_by_action: Any
    evaluation_returns_by_action: Any
    measurement_covariances: Any
    action_mask: Any
    fit_replica_returns_by_action: Any
    evaluation_replica_returns_by_action: Any
    probe_fit_returns_by_action: Any
    probe_evaluation_returns_by_action: Any
    probe_measurement_covariances: Any
    probe_action_mask: Any
    probe_fit_replica_returns_by_action: Any
    probe_evaluation_replica_returns_by_action: Any


class AnchorSnapshots(NamedTuple):
    """Sparse pre-action worlds selected from an anchor rollout."""

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
    "DirectResponsePrediction",
    "DirectResponseTarget",
    "LossResult",
    "ModelOutput",
    "PolicyState",
    "ProbeResponsePrediction",
    "ProbeResponseTarget",
    "ResponsePrediction",
    "ResponseTarget",
    "RolloutBatch",
    "RunnerState",
    "TrainState",
    "VOIResult",
]
