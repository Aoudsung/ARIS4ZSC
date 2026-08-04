"""Immutable state and batch records for unified DELTA-ZSC."""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple


class BehaviorPosterior(NamedTuple):
    """Four independent Beta posteriors over legal observable behaviour rates."""

    alpha: Any
    beta: Any


class AgentState(NamedTuple):
    """Complete deployment state.

    The base task recurrence, analytic behaviour statistics, categorical mode
    belief, and the previous legal ego observation/action are the only history
    carriers.  There is no learned capability GRU, comparator state, codebook,
    or training-only value ensemble in the deployment state.
    """

    task_carry: Any
    behavior: BehaviorPosterior
    belief: Any
    previous_observation: Any
    previous_action: Any
    episode_start: Any


class BasePolicyOutput(NamedTuple):
    task_features: Any
    instant_partner: Any
    base_logits: Any
    value: Any


class ResponseLogits(NamedTuple):
    """Base-plus-residual response distribution with one component axis K."""

    visibility: Any
    relative_position: Any
    direction: Any
    inventory: Any
    inventory_change: Any


class LatentOutput(NamedTuple):
    predictive_belief: Any
    belief: Any
    response_log_likelihood: Any
    response_log_evidence: Any
    decision_mean: Any
    decision_log_scale: Any
    expected_action_values: Any
    value_of_information: Any
    adapted_logits: Any
    adaptation_temperature: Any
    adaptation_kl: Any


class AgentOutput(NamedTuple):
    base: BasePolicyOutput
    behavior_features: Any
    latent: LatentOutput


class RolloutBatch(NamedTuple):
    observations: Any
    response_next_observations: Any
    previous_actions: Any
    episode_starts: Any
    actions: Any
    rewards: Any
    shaped_rewards: Any
    dones: Any
    behavior_probabilities: Any
    old_log_probabilities: Any
    old_values: Any
    ppo_mask: Any
    initial_state: AgentState


class DecisionAnchorBatch(NamedTuple):
    """Privileged decision observations aligned to one rollout.

    ``time_indexes`` and ``lane_indexes`` locate the legal states inside the
    rollout.  The latent forward algorithm recomputes task features, behaviour
    statistics, response beliefs, and decision emissions under the current
    parameters.  No stale stored posterior or learned comparator is needed.
    """

    time_indexes: Any
    lane_indexes: Any
    centered_returns: Any
    standard_errors: Any
    action_mask: Any
    fit_replica_returns: Any
    evaluation_returns: Any


class UnifiedTrainState(NamedTuple):
    """Complete resume-equivalent training state.

    Base PPO and latent maximum-likelihood estimation have independent parameter
    trees and optimizer moments.  The full runner state, including environment,
    partner carry, legal ego history, roles, and random key, is checkpointed so
    interrupted and uninterrupted runs are scientifically equivalent.
    """

    base_params: Any
    latent_params: Any
    base_optimizer_state: Any
    latent_optimizer_state: Any
    runner_state: Any
    random_key: Any
    environment_steps: Any
    update_count: Any
    resource_ledger: Mapping[str, Any]


class BaseLoss(NamedTuple):
    total: Any
    actor: Any
    value: Any
    entropy: Any
    exact_kl: Any


class LatentLoss(NamedTuple):
    total: Any
    response_nll: Any
    decision_nll: Any
    response_observation_count: Any
    decision_observation_count: Any


__all__ = [
    "AgentOutput",
    "AgentState",
    "BaseLoss",
    "BasePolicyOutput",
    "BehaviorPosterior",
    "DecisionAnchorBatch",
    "LatentLoss",
    "LatentOutput",
    "ResponseLogits",
    "RolloutBatch",
    "UnifiedTrainState",
]
