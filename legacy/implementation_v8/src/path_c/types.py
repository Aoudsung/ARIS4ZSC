"""Immutable PyTree records for DEPI.

METHOD_SPEC §1 defines the three scientific objects x_t / u / c_t with
structural input-layer isolation.  The deployable policy carries one task
encoder, one capability encoder, and the exact categorical protocol posterior;
training-only objects are explicit so deployment pruning and checkpoint
identity can be checked mechanically.
"""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple


class CapabilityCarry(NamedTuple):
    """Slow-timescale capability state.

    Evidence is accumulated every step in ``hidden`` but the actor-visible
    ``published`` embedding changes only at registered window boundaries.
    This prevents the stable capability path from becoming a second dynamic
    protocol encoder.
    """

    hidden: Any
    published: Any
    steps: Any


class PolicyState(NamedTuple):
    """Complete legal recurrent state exposed by the Official policy wrapper.

    METHOD_SPEC §1.4 field list: (task_carry, capability_carry,
    protocol_carry, context_summary, previous_observation, previous_action,
    episode_start).  ``context_summary`` stores concat(u, c) from the last
    step (32 dimensions) for deployment continuity. ``protocol_carry`` is
    the categorical posterior itself; no recognition-network hidden state is
    carried by the deployable policy.
    """

    task_carry: Any
    capability_carry: Any
    protocol_carry: Any
    context_summary: Any
    previous_observation: Any
    previous_action: Any
    episode_start: Any


class ContextOutput(NamedTuple):
    """Legal decision context ``(x_t, r_t, u_t, pi_t, c_t)``.

    ``instant_partner`` is computed from the current observation only.  It has
    no recurrent carry, so current collision/inventory geometry is available
    without opening a second partner-history path.
    """

    task_features: Any
    instant_partner: Any
    capability: Any
    protocol_probabilities: Any
    protocol_embedding: Any


class ModelOutput(NamedTuple):
    task_features: Any
    instant_partner: Any
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


class ComponentInterventionOutput(NamedTuple):
    """One-hot ``z=k`` interventions evaluated through the shared heads.

    The component axis is immediately before the action axis.  These values
    are diagnostics and anchor-supervision targets; component indexes remain
    exchangeable and are compared across runs only after permutation
    alignment.
    """

    protocol_probabilities: Any
    policy_logits: Any
    raw_q1: Any
    raw_q2: Any
    action_signatures: Any


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
    shaped_rewards: Any
    dones: Any
    old_log_probabilities: Any
    old_values: Any
    behavior_probabilities: Any
    ppo_mask: Any
    partner_sources: Any
    partner_members: Any
    partner_family_ids: Any
    partner_checkpoint_stages: Any
    partner_run_ids: Any
    initial_policy_state: PolicyState
    initial_target_policy_state: PolicyState


class CounterfactualAnchorBatch(NamedTuple):
    """Fixed real-return all-action supervision collected at an outer update."""

    anchor_ids: Any
    rollout_flat_indexes: Any
    policy_states: PolicyState
    observations: Any
    partner_sources: Any
    partner_members: Any
    partner_family_ids: Any
    partner_checkpoint_stages: Any
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
    evaluation_returns_by_action: Any = None
    evaluation_replica_count: Any = None
    fit_replica_returns_by_action: Any = None
    collection_policy_fingerprint: Any = None
    collection_context_fingerprint: Any = None


class QuotientPairBatch(NamedTuple):
    """Matched-pair payload for the §5.3 frozen comparator data path.

    ``comparator_distinct_probability``/``ego_state_*``/``probe_observations`` stay
    ``None`` for legacy constructions; the §5 anchor pipeline fills them so
    ``separation_terms_from_matched_pairs`` can build the ``SeparationTerms``
    payload consumed inside the jit-compiled combined-loss scan.
    """

    anchor_index_a: Any
    anchor_index_b: Any
    decision_distance: Any
    weights: Any
    comparator_distinct_probability: Any = None
    ego_state_a: Any = None
    ego_state_b: Any = None
    probe_observations: Any = None
    pair_valid: Any = None


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
    pair_valid: Any = None


class TrainState(NamedTuple):
    """Complete DEPI scientific state (checkpoint schema 8).

    Every value capable of changing the next outer update is explicit.  Dead
    per-head optimizers, reward-shaping EMAs, synthetic-partner state, and
    obsolete anchor-buffer state are not represented.
    """

    params: Any
    target_params: Any

    ppo_optimizer_state: Any

    supervision_anchor_batch: Any
    current_separation_terms: Any
    current_pair_comparator: Any
    supervision_readings: Any
    anchor_sampling_counter: Any
    anchor_microbatch_size: Any
    effective_update_epochs: Any

    bootstrap_encoder_params: Any
    bootstrap_optimizer_states: Any
    bootstrap_sampling_counters: Any
    m1_summary_state: Any
    m1_history: Any

    ppo_optimizer_step: Any

    runner_state: Any
    random_domains: Any
    update_count: Any
    effective_environment_steps: Any
    resource_ledger: Any


class TrainingCoreState(NamedTuple):
    params: Any
    target_params: Any
    ppo_optimizer_state: Any


class LossBundle(NamedTuple):
    total: Any
    metrics: Mapping[str, Any]


class TrainingUpdate(NamedTuple):
    state: TrainState
    metrics: Mapping[str, Any]


__all__ = [
    "CapabilityCarry",
    "ComponentInterventionOutput",
    "ContextOutput",
    "CounterfactualAnchorBatch",
    "LossBundle",
    "ModelOutput",
    "PolicyState",
    "QuotientPairBatch",
    "RolloutBatch",
    "TrainingCoreState",
    "TrainState",
    "TrainingUpdate",
]
