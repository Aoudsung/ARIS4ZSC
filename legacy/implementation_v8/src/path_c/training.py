"""DEPI PPO-scan plus one independently trust-regioned auxiliary transaction.

L = L_PPO + lambda_A * L_signature + lambda_R * L_response + lambda_S * L_separation

PPO minibatches are followed by one full-batch response/decision transaction.
Each transaction is differentiated once; the auxiliary transaction is rolled
back if its own exact categorical policy KL exceeds the registered bound.
After each update the rollout batch reports
``combined_policy_kl`` (exact behaviour-vs-updated categorical policy KL,
including every context-pathway effect); exceeding the registered threshold
terminates the remaining minibatches of the current PPO epoch (§3.4).
"""

from __future__ import annotations

from typing import Any, Mapping

from .types import LossBundle, RolloutBatch, TrainState, TrainingCoreState, TrainingUpdate


RESPONSE_VARIANTS = frozenset(
    {
        "b1",
        "b2",
        "deterministic_context",
        "q_only",
        "actor_only",
        "no_separation",
        "no_capability",
        "response_only_posterior",
    }
)
CAPABILITY_VARIANTS = RESPONSE_VARIANTS - {"no_capability"}
ANCHOR_VARIANTS = frozenset(
    {"b2", "decision_only", "q_only", "actor_only", "no_separation", "no_capability", "response_only_posterior"}
)
CRITIC_SIGNATURE_VARIANTS = frozenset(
    {"b2", "q_only", "no_separation", "no_capability", "response_only_posterior"}
)
ACTOR_DECISION_VARIANTS = frozenset(
    {"b2", "decision_only", "actor_only", "no_separation", "no_capability", "response_only_posterior"}
)
SEPARATION_VARIANTS = frozenset({"b2", "no_capability", "response_only_posterior"})
AUXILIARY_VARIANTS = RESPONSE_VARIANTS | ANCHOR_VARIANTS


def gather_actions(values: Any, actions: Any, *, action_axis: int = -1) -> Any:
    import jax
    import jax.numpy as jnp

    source = jnp.asarray(values)
    index = jnp.asarray(actions, dtype=jnp.int32)
    axis = action_axis if action_axis >= 0 else source.ndim + action_axis
    shape = list(index.shape)
    while len(shape) < source.ndim:
        shape.insert(axis, 1)
    return jnp.squeeze(jnp.take_along_axis(source, index.reshape(tuple(shape)), axis=axis), axis=axis)


def huber(error: Any, delta: float = 1.0) -> Any:
    import jax.numpy as jnp

    absolute = jnp.abs(jnp.asarray(error, dtype=jnp.float32))
    quadratic = jnp.minimum(absolute, float(delta))
    return 0.5 * jnp.square(quadratic) + float(delta) * (absolute - quadratic)


def categorical_log_probability(logits: Any, actions: Any) -> Any:
    import jax

    return gather_actions(jax.nn.log_softmax(logits, axis=-1), actions)


def categorical_entropy(logits: Any) -> Any:
    import jax
    import jax.numpy as jnp

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    probabilities = jnp.exp(log_probs)
    return -jnp.sum(probabilities * log_probs, axis=-1)


def generalized_advantage_estimation(*, rewards: Any, dones: Any, values: Any, gamma: float, gae_lambda: float):
    import jax
    import jax.numpy as jnp

    reward = jnp.asarray(rewards, dtype=jnp.float32)
    done = jnp.asarray(dones, dtype=jnp.bool_)
    value = jnp.asarray(values, dtype=jnp.float32)
    if value.shape[0] != reward.shape[0] + 1:
        raise ValueError("GAE values must contain T+1 states.")
    delta = reward + float(gamma) * (1.0 - done.astype(jnp.float32)) * value[1:] - value[:-1]

    def backward(carry: Any, items: tuple[Any, Any]):
        delta_t, done_t = items
        advantage = delta_t + float(gamma) * float(gae_lambda) * (1.0 - done_t.astype(jnp.float32)) * carry
        return advantage, advantage

    _, reverse = jax.lax.scan(backward, jnp.zeros_like(delta[-1]), (delta[::-1], done[::-1]))
    advantages = reverse[::-1]
    return jax.lax.stop_gradient(advantages), jax.lax.stop_gradient(advantages + value[:-1])


def ppo_actor_loss(*, logits: Any, actions: Any, old_log_probabilities: Any, advantages: Any, clip_epsilon: float, normalize_advantages: bool, mask: Any | None = None):
    import jax.numpy as jnp

    advantage = jnp.asarray(advantages, dtype=jnp.float32)
    weight = jnp.ones_like(advantage) if mask is None else jnp.asarray(mask, dtype=jnp.float32)
    denominator = jnp.maximum(jnp.sum(weight), 1.0e-8)
    if normalize_advantages:
        mean = jnp.sum(advantage * weight) / denominator
        variance = jnp.sum(jnp.square(advantage - mean) * weight) / denominator
        advantage = (advantage - mean) / (jnp.sqrt(variance) + 1.0e-8)
    log_ratio = categorical_log_probability(logits, actions) - jnp.asarray(old_log_probabilities)
    ratio = jnp.exp(log_ratio)
    clipped = jnp.clip(ratio, 1.0 - float(clip_epsilon), 1.0 + float(clip_epsilon))
    loss = -jnp.sum(jnp.minimum(ratio * advantage, clipped * advantage) * weight) / denominator
    approx_kl = jnp.sum(((ratio - 1.0) - log_ratio) * weight) / denominator
    return loss, {
        "actor_loss": loss,
        "approx_kl": approx_kl,
        "clip_fraction": jnp.sum((jnp.abs(ratio - 1.0) > float(clip_epsilon)).astype(jnp.float32) * weight) / denominator,
    }


def clipped_value_loss(*, predictions: Any, old_predictions: Any, targets: Any, clip_epsilon: float, mask: Any | None = None) -> Any:
    import jax.numpy as jnp

    predicted = jnp.asarray(predictions, dtype=jnp.float32)
    old = jnp.asarray(old_predictions, dtype=jnp.float32)
    target = jnp.asarray(targets, dtype=jnp.float32)
    clipped = old + jnp.clip(predicted - old, -float(clip_epsilon), float(clip_epsilon))
    items = 0.5 * jnp.maximum(
        jnp.square(predicted - target),
        jnp.square(clipped - target),
    )
    weight = jnp.ones_like(items) if mask is None else jnp.asarray(mask, dtype=jnp.float32)
    return jnp.sum(items * weight) / jnp.maximum(jnp.sum(weight), 1.0e-8)


def centered(values: Any) -> Any:
    import jax.numpy as jnp

    array = jnp.asarray(values, dtype=jnp.float32)
    return array - jnp.mean(array, axis=-1, keepdims=True)


def signature_loss(
    *,
    raw_q1: Any,
    raw_q2: Any,
    advantage_targets: Any,
    action_mask: Any,
    hinge_margin: float,
    hinge_advantage_gap: float,
) -> tuple[Any, Any, Any]:
    """L_signature: centered-Q Huber fit plus rank-preserving hinge (§3.2).

    Q_c = min(Q1, Q2) - mean_a min(Q1, Q2); targets are the centered CRN
    continuation returns A(a).  The hinge enforces the action ranking implied
    by advantages whose gap exceeds ``hinge_advantage_gap``.
    """

    import jax.numpy as jnp

    conservative = jnp.minimum(
        jnp.asarray(raw_q1, dtype=jnp.float32), jnp.asarray(raw_q2, dtype=jnp.float32)
    )
    q_centered = conservative - jnp.mean(conservative, axis=-1, keepdims=True)
    target = centered(advantage_targets)
    mask = jnp.asarray(action_mask, dtype=jnp.float32)
    denominator = jnp.maximum(jnp.sum(mask), 1.0)
    fit = jnp.sum(mask * huber(q_centered - target)) / denominator

    gap = jnp.abs(target[..., :, None] - target[..., None, :])
    order = jnp.sign(target[..., :, None] - target[..., None, :])
    q_gap = q_centered[..., :, None] - q_centered[..., None, :]
    pair_mask = (gap > float(hinge_advantage_gap)) & (order > 0.0)
    pair_mask = pair_mask & (mask[..., :, None] > 0.0) & (mask[..., None, :] > 0.0)
    hinge = jnp.maximum(
        0.0, float(hinge_margin) - q_gap * order
    )
    pair_denominator = jnp.maximum(jnp.sum(pair_mask.astype(jnp.float32)), 1.0)
    hinge_loss = jnp.sum(pair_mask.astype(jnp.float32) * hinge) / pair_denominator
    total = fit + hinge_loss
    return total, fit, hinge_loss


def decision_policy_loss(
    *,
    policy_logits: Any,
    advantage_targets: Any,
    action_mask: Any,
    temperature: float,
    confidence_weight: Any = 1.0,
    reward_scale_floor: float = 20.0,
) -> Any:
    """KL from empirical CRN continuation advantages to the actor policy.

    The target distribution and confidence are stopped gradients.  This makes
    legal all-action supervision alter executed action probabilities directly,
    closing the former critic-to-actor control break.
    """

    import jax
    import jax.numpy as jnp

    if float(temperature) <= 0.0:
        raise ValueError("Decision-policy temperature must be positive.")
    logits = jnp.asarray(policy_logits, dtype=jnp.float32)
    valid = jnp.asarray(action_mask, dtype=jnp.float32)
    target_advantage = jax.lax.stop_gradient(centered(advantage_targets))
    median = jnp.median(target_advantage, axis=-1, keepdims=True)
    mad_scale = 1.4826 * jnp.median(
        jnp.abs(target_advantage - median), axis=-1, keepdims=True
    )
    scale = jnp.maximum(mad_scale, float(reward_scale_floor))
    masked_target_logits = target_advantage / scale / float(temperature)
    masked_target_logits = jnp.where(valid > 0.0, masked_target_logits, -1.0e9)
    target_probability = jax.lax.stop_gradient(
        jax.nn.softmax(masked_target_logits, axis=-1)
    )
    target_log_probability = jnp.log(jnp.maximum(target_probability, 1.0e-12))
    policy_log_probability = jax.nn.log_softmax(logits, axis=-1)
    per_anchor = jnp.sum(
        target_probability * (target_log_probability - policy_log_probability),
        axis=-1,
    )
    confidence = jax.lax.stop_gradient(
        jnp.asarray(confidence_weight, dtype=jnp.float32)
    )
    confidence = jnp.broadcast_to(confidence, per_anchor.shape)
    return jnp.sum(confidence * per_anchor) / jnp.maximum(
        jnp.sum(confidence), 1.0
    )


def capability_consistency_loss(
    capability_sequence: Any,
    episode_starts: Any,
    *,
    update_period: int = 16,
) -> Any:
    """Penalize changes between non-overlapping same-episode ``u`` windows."""

    import jax
    import jax.numpy as jnp

    values = jnp.asarray(capability_sequence, dtype=jnp.float32)
    starts = jnp.asarray(episode_starts, dtype=jnp.bool_)
    period = int(update_period)
    if period <= 0:
        raise ValueError("Capability update period must be positive.")
    if values.shape[0] != starts.shape[0]:
        raise ValueError("Capability sequence and episode markers must align.")
    if values.shape[0] <= period:
        return jnp.asarray(0.0, dtype=jnp.float32)
    # Publication is aligned to episode-local time rather than the rollout's
    # global index.  Reconstruct the local counter exactly from start flags so
    # episodes that begin between global multiples of ``period`` are not
    # silently omitted.
    def count_step(previous: Any, start: Any) -> tuple[Any, Any]:
        current = jnp.where(start, 1, previous + 1)
        return current, current

    _, local_steps = jax.lax.scan(
        count_step, jnp.zeros_like(starts[0], dtype=jnp.int32), starts
    )
    indexes = jnp.arange(period, values.shape[0], dtype=jnp.int32)
    previous_indexes = indexes - period
    episode_ids = jnp.cumsum(starts.astype(jnp.int32), axis=0)
    same_episode = episode_ids[indexes] == episode_ids[previous_indexes]
    publication = (local_steps[indexes] % period == 0) & (
        local_steps[indexes] > period
    )
    items = jnp.mean(
        jnp.square(values[indexes] - values[previous_indexes]), axis=-1
    )
    weight = (same_episode & publication).astype(jnp.float32)
    return jnp.sum(weight * items) / jnp.maximum(jnp.sum(weight), 1.0)


def capability_auxiliary_objective(
    *,
    capability_sequence: Any,
    observations: Any,
    initial_previous_observation: Any,
    initial_steps: Any,
    episode_starts: Any,
    transition_mask: Any,
    variance_floor: float,
    update_period: int = 16,
) -> tuple[Any, Any, Any, Mapping[str, Any]]:
    """Direct semantic prediction plus variance-floor anti-collapse objective.

    At each complete 16-step publication window, coordinates 0:4 of ``u``
    predict legal local-history rates for visibility, visible movement,
    visible inventory holding, and visible inventory change.  No partner
    identity/action/lineage field is used.  A VICReg-style floor is applied to
    every published coordinate so the all-zero representation is not a
    stationary optimum of the auxiliary objective.
    """

    import jax
    import jax.numpy as jnp

    from .response_targets import (
        extract_partner_response_targets,
        official_partner_observation_planes,
    )

    values = jnp.asarray(capability_sequence, dtype=jnp.float32)
    frames = jnp.asarray(observations, dtype=jnp.float32)
    starts = jnp.asarray(episode_starts, dtype=jnp.bool_)
    valid_transitions = jnp.asarray(transition_mask, dtype=jnp.float32)
    period = int(update_period)
    if values.shape[0] != frames.shape[0] or values.shape[0] != starts.shape[0]:
        raise ValueError("Capability auxiliary sequence axes differ.")
    if values.shape[-1] < 4:
        raise ValueError("Capability semantic contract requires at least four coordinates.")
    previous = jnp.concatenate(
        (jnp.asarray(initial_previous_observation)[None], frames[:-1]), axis=0
    )
    planes = official_partner_observation_planes(frames.shape[-1])
    targets = extract_partner_response_targets(previous, frames, planes=planes)
    previous_targets = extract_partner_response_targets(
        previous, previous, planes=planes
    )
    both_visible = jnp.asarray(targets.event_mask, dtype=jnp.float32)
    moved = (
        (targets.relative_position != previous_targets.relative_position)
        & (both_visible > 0.0)
    ).astype(jnp.float32)
    holding = jnp.any(jnp.asarray(targets.inventory) > 0, axis=-1).astype(
        jnp.float32
    )
    numerators = jnp.stack(
        (
            jnp.asarray(targets.visibility, dtype=jnp.float32),
            moved,
            holding * jnp.asarray(targets.visible_mask, dtype=jnp.float32),
            jnp.asarray(targets.interaction_change, dtype=jnp.float32),
        ),
        axis=-1,
    )
    denominators = jnp.stack(
        (
            jnp.ones_like(both_visible),
            both_visible,
            jnp.asarray(targets.visible_mask, dtype=jnp.float32),
            both_visible,
        ),
        axis=-1,
    )
    # A transition crossing an episode boundary is not evidence.  Visibility
    # and holding in the first frame remain valid current-frame observations.
    denominators = denominators.at[..., 1].set(
        denominators[..., 1] * (~starts).astype(jnp.float32)
    )
    denominators = denominators.at[..., 3].set(
        denominators[..., 3] * (~starts).astype(jnp.float32)
    )
    numerators = numerators * denominators

    def accumulate(
        carry: tuple[Any, Any], item: tuple[Any, Any, Any]
    ) -> tuple[tuple[Any, Any], tuple[Any, Any]]:
        running_numerator, running_denominator = carry
        numerator, denominator, start = item
        running_numerator = jnp.where(
            start[..., None], jnp.zeros_like(running_numerator), running_numerator
        )
        running_denominator = jnp.where(
            start[..., None],
            jnp.zeros_like(running_denominator),
            running_denominator,
        )
        running_numerator = running_numerator + numerator
        running_denominator = running_denominator + denominator
        return (running_numerator, running_denominator), (
            running_numerator,
            running_denominator,
        )

    zero_stats = jnp.zeros_like(numerators[0])
    _, (cumulative_numerator, cumulative_denominator) = jax.lax.scan(
        accumulate, (zero_stats, zero_stats), (numerators, denominators, starts)
    )

    def count_step(previous_step: Any, start: Any) -> tuple[Any, Any]:
        current = jnp.where(start, 1, previous_step + 1)
        return current, current

    _, local_steps = jax.lax.scan(
        count_step, jnp.asarray(initial_steps, dtype=jnp.int32), starts
    )
    time_count = values.shape[0]
    indexes = jnp.arange(time_count, dtype=jnp.int32)
    previous_indexes = jnp.maximum(indexes - period, 0)
    episode_ids = jnp.cumsum(starts.astype(jnp.int32), axis=0)
    complete_window = indexes[:, None] >= (period - 1)
    previous_same_episode = episode_ids == episode_ids[previous_indexes]
    subtract_previous = complete_window & previous_same_episode & (
        local_steps > period
    )
    window_numerator = cumulative_numerator - jnp.where(
        subtract_previous[..., None], cumulative_numerator[previous_indexes], 0.0
    )
    window_denominator = cumulative_denominator - jnp.where(
        subtract_previous[..., None], cumulative_denominator[previous_indexes], 0.0
    )
    semantic_target = 2.0 * (
        window_numerator / jnp.maximum(window_denominator, 1.0)
    ) - 1.0
    publication = (
        complete_window
        & ((local_steps % period) == 0)
        & (local_steps >= period)
        & (valid_transitions > 0.0)
    )
    semantic_valid = publication[..., None] & (window_denominator > 0.0)
    semantic_weight = semantic_valid.astype(jnp.float32)
    semantic_error = jnp.square(values[..., :4] - semantic_target)
    prediction_loss = jnp.sum(semantic_weight * semantic_error) / jnp.maximum(
        jnp.sum(semantic_weight), 1.0
    )

    published_weight = publication.astype(jnp.float32)
    # Batch-level VICReg deliberately has no partner-run/lineage input.  The
    # semantic coordinates must vary across legal published samples without
    # becoming a learned run fingerprint.
    flat_weight = published_weight.reshape(-1)
    flat_values = values.reshape((-1, values.shape[-1]))
    sample_count = jnp.sum(flat_weight)
    mean = jnp.sum(flat_weight[:, None] * flat_values, axis=0) / jnp.maximum(
        sample_count, 1.0
    )
    centered_values = flat_values - mean
    variance = jnp.sum(
        flat_weight[:, None] * jnp.square(centered_values), axis=0
    ) / jnp.maximum(sample_count, 1.0)
    standard_deviation = jnp.sqrt(variance + 1.0e-8)
    variance_loss = jnp.mean(
        jnp.square(jnp.maximum(float(variance_floor) - standard_deviation, 0.0))
    )
    covariance = (
        (flat_weight[:, None] * centered_values).T @ centered_values
        / jnp.maximum(sample_count - 1.0, 1.0)
    )
    off_diagonal = covariance - jnp.diag(jnp.diag(covariance))
    covariance_loss = jnp.sum(jnp.square(off_diagonal)) / jnp.maximum(
        values.shape[-1] * (values.shape[-1] - 1), 1
    )
    return prediction_loss, variance_loss, covariance_loss, {
        "capability_semantic_prediction_loss": prediction_loss,
        "capability_variance_floor_loss": variance_loss,
        "capability_covariance_loss": covariance_loss,
        "capability_minimum_published_std": jnp.min(standard_deviation),
        "capability_publication_count": jnp.sum(published_weight),
        "capability_published_sample_count": sample_count,
    }


def signature_anchor_objective(
    *,
    model: Any,
    params: Any,
    anchors: Any,
    loss_v2: Any,
    critic_enabled: bool = True,
    actor_enabled: bool = True,
    posterior_decision_enabled: bool = True,
):
    """Unified legal-anchor decision loss for critic and actor."""

    import jax
    import jax.numpy as jnp

    from .gradient_routing import parameters_for_loss

    zero = jnp.asarray(0.0, dtype=jnp.float32)
    if anchors is None:
        return zero, {
            "signature_loss": zero,
            "signature_huber_loss": zero,
            "signature_hinge_loss": zero,
            "component_signature_loss": zero,
            "posterior_decision_loss": zero,
            "component_pairwise_action_signature_divergence": zero,
            "component_one_hot_actor_tv": zero,
            "component_one_hot_top_action_disagreement": zero,
            "decision_policy_loss": zero,
            "decision_supervision_confidence": zero,
            "decision_target_entropy": zero,
            "decision_top_action_stability": zero,
            "decision_confidence_q10": zero,
            "decision_confidence_q50": zero,
            "decision_confidence_q90": zero,
            "anchor_policy_drift_kl": zero,
            "anchor_drift_weight": zero,
            "anchor_context_fingerprint_match_fraction": zero,
            "decision_effective_anchor_count": zero,
        }
    _, signature_output = model.apply(
        {"params": parameters_for_loss(params, loss_name="signature")},
        anchors.policy_states,
        anchors.observations,
        method=model.step,
    )
    _, decision_output = model.apply(
        {"params": parameters_for_loss(params, loss_name="decision")},
        anchors.policy_states,
        anchors.observations,
        method=model.step,
    )
    _, component_output = model.apply(
        {"params": parameters_for_loss(params, loss_name="signature")},
        anchors.policy_states,
        anchors.observations,
        method=model.component_intervention_step,
    )
    _, posterior_output = model.apply(
        {"params": parameters_for_loss(params, loss_name="posterior_decision")},
        anchors.policy_states,
        anchors.observations,
        method=model.component_intervention_step,
    )
    q_total, fit, hinge_loss = signature_loss(
        raw_q1=signature_output.raw_q1,
        raw_q2=signature_output.raw_q2,
        advantage_targets=anchors.fit_returns_by_action,
        action_mask=anchors.action_mask,
        hinge_margin=float(loss_v2.rank_hinge_margin),
        hinge_advantage_gap=float(loss_v2.rank_hinge_advantage_gap),
    )
    replica_count = jnp.maximum(
        jnp.asarray(anchors.replica_count, dtype=jnp.float32), 1.0
    )
    mean = jnp.asarray(anchors.return_sum_by_action, dtype=jnp.float32) / replica_count
    second = (
        jnp.asarray(anchors.return_squared_sum_by_action, dtype=jnp.float32)
        / replica_count
    )
    standard_error = jnp.sqrt(
        jnp.maximum(second - jnp.square(mean), 0.0) / replica_count
    )
    anchor_valid = jnp.any(
        jnp.asarray(anchors.action_mask, dtype=jnp.bool_), axis=-1
    ).astype(jnp.float32)
    component_mask = jnp.asarray(anchors.action_mask, dtype=jnp.float32)
    component_responsibility = jax.lax.stop_gradient(
        jnp.asarray(component_output.protocol_probabilities, dtype=jnp.float32)
    )
    component_signatures = jnp.asarray(
        component_output.action_signatures, dtype=jnp.float32
    )
    # Direct component-conditioned signature constraint from the registered
    # definition: sum_k pi_k S_k(x,u,a) approximates the empirical centered
    # all-action continuation signature.  Individual indexes remain
    # exchangeable; the constraint does not assign a semantic name to k.
    posterior_mixed_signature = jnp.sum(
        component_responsibility[..., :, None] * component_signatures,
        axis=-2,
    )
    component_target = centered(anchors.fit_returns_by_action)
    component_fit = jnp.sum(
        component_mask
        * huber(posterior_mixed_signature - component_target)
    ) / jnp.maximum(jnp.sum(component_mask), 1.0)
    decision_component_signatures = jax.lax.stop_gradient(
        jnp.asarray(posterior_output.action_signatures, dtype=jnp.float32)
    )
    squared_error = jnp.sum(
        component_mask[..., None, :]
        * jnp.square(
            decision_component_signatures - component_target[..., None, :]
        ),
        axis=-1,
    ) / jnp.maximum(jnp.sum(component_mask, axis=-1, keepdims=True), 1.0)
    q_decision = jax.lax.stop_gradient(
        jax.nn.softmax(
            -squared_error
            / float(getattr(loss_v2, "posterior_decision_temperature", 1.0)),
            axis=-1,
        )
    )
    posterior_probability = jnp.asarray(
        posterior_output.protocol_probabilities, dtype=jnp.float32
    )
    posterior_decision_kl = jnp.sum(
        q_decision
        * (
            jnp.log(jnp.maximum(q_decision, 1.0e-12))
            - jnp.log(jnp.maximum(posterior_probability, 1.0e-12))
        ),
        axis=-1,
    )
    posterior_decision_loss = jnp.sum(anchor_valid * posterior_decision_kl) / jnp.maximum(
        jnp.sum(anchor_valid), 1.0
    )
    signature_delta = (
        component_signatures[..., :, None, :]
        - component_signatures[..., None, :, :]
    )
    signature_distance = jnp.sqrt(
        jnp.mean(jnp.square(signature_delta), axis=-1) + 1.0e-12
    )
    component_count = component_signatures.shape[-2]
    off_diagonal = 1.0 - jnp.eye(component_count, dtype=jnp.float32)
    pair_weight = anchor_valid[..., None, None] * off_diagonal
    pair_denominator = jnp.maximum(jnp.sum(pair_weight), 1.0)
    action_signature_divergence = (
        jnp.sum(pair_weight * signature_distance) / pair_denominator
    )
    intervention_probability = jax.nn.softmax(
        jnp.asarray(component_output.policy_logits, dtype=jnp.float32), axis=-1
    )
    actor_tv_matrix = 0.5 * jnp.sum(
        jnp.abs(
            intervention_probability[..., :, None, :]
            - intervention_probability[..., None, :, :]
        ),
        axis=-1,
    )
    actor_tv = jnp.sum(pair_weight * actor_tv_matrix) / pair_denominator
    top_actions = jnp.argmax(intervention_probability, axis=-1)
    top_disagreement = (
        top_actions[..., :, None] != top_actions[..., None, :]
    ).astype(jnp.float32)
    top_action_disagreement = (
        jnp.sum(pair_weight * top_disagreement) / pair_denominator
    )
    mean_returns = jnp.asarray(anchors.fit_returns_by_action, dtype=jnp.float32)
    masked_means = jnp.where(
        jnp.asarray(anchors.action_mask, dtype=jnp.bool_), mean_returns, -jnp.inf
    )
    top_indexes = jnp.argsort(masked_means, axis=-1)[..., -2:]
    top = jnp.take_along_axis(masked_means, top_indexes[..., 1:], axis=-1)[..., 0]
    second_top = jnp.take_along_axis(
        masked_means, top_indexes[..., :1], axis=-1
    )[..., 0]
    top_se = jnp.take_along_axis(
        standard_error, top_indexes[..., 1:], axis=-1
    )[..., 0]
    second_se = jnp.take_along_axis(
        standard_error, top_indexes[..., :1], axis=-1
    )[..., 0]
    gap_signal_to_noise = (top - second_top) / jnp.sqrt(
        jnp.square(top_se) + jnp.square(second_se) + 1.0e-6
    )
    signal_confidence = jax.nn.sigmoid(gap_signal_to_noise)
    if anchors.fit_replica_returns_by_action is not None:
        replica_returns = jnp.asarray(
            anchors.fit_replica_returns_by_action, dtype=jnp.float32
        )
        masked_replica_returns = jnp.where(
            jnp.asarray(anchors.action_mask, dtype=jnp.bool_)[..., None],
            replica_returns,
            -jnp.inf,
        )
        replica_top = jnp.argmax(masked_replica_returns, axis=-2)
        mean_top = jnp.argmax(masked_means, axis=-1)
        top_action_stability = jnp.mean(
            replica_top == mean_top[..., None], axis=-1
        )
        confidence = top_action_stability * anchor_valid
    else:
        top_action_stability = signal_confidence
        confidence = signal_confidence * anchor_valid

    collection_log_probability = jax.nn.log_softmax(
        jnp.asarray(anchors.collection_policy_logits, dtype=jnp.float32), axis=-1
    )
    current_log_probability = jax.nn.log_softmax(
        decision_output.policy_logits, axis=-1
    )
    current_probability = jnp.exp(current_log_probability)
    drift_kl = jnp.sum(
        current_probability * (current_log_probability - collection_log_probability),
        axis=-1,
    )
    drift_weight = jnp.exp(-drift_kl / 0.04) * anchor_valid
    context = jnp.asarray(decision_output.context_summary, dtype=jnp.float32).reshape(
        (anchor_valid.shape[0], -1)
    )
    quantized = jnp.rint(context * 10_000.0).astype(jnp.int32)
    positions = jnp.arange(1, quantized.shape[-1] + 1, dtype=jnp.int32)
    current_context_fingerprint = jnp.stack(
        (
            jnp.sum(quantized, axis=-1).astype(jnp.uint32),
            jnp.sum(quantized * positions[None], axis=-1).astype(jnp.uint32),
        ),
        axis=-1,
    )
    context_fingerprint_match = jnp.all(
        current_context_fingerprint
        == jnp.asarray(anchors.collection_context_fingerprint, dtype=jnp.uint32),
        axis=-1,
    ).astype(jnp.float32)
    combined_confidence = confidence * drift_weight
    policy_coupling = decision_policy_loss(
        policy_logits=decision_output.policy_logits,
        advantage_targets=anchors.fit_returns_by_action,
        action_mask=anchors.action_mask,
        temperature=float(getattr(loss_v2, "decision_policy_temperature", 1.0)),
        confidence_weight=combined_confidence,
        reward_scale_floor=20.0,
    )
    centered_target = centered(mean_returns)
    target_scale = jnp.maximum(
        1.4826
        * jnp.median(
            jnp.abs(
                centered_target
                - jnp.median(centered_target, axis=-1, keepdims=True)
            ),
            axis=-1,
            keepdims=True,
        ),
        20.0,
    )
    target_logits = jnp.where(
        jnp.asarray(anchors.action_mask, dtype=jnp.bool_),
        centered_target
        / target_scale
        / float(getattr(loss_v2, "decision_policy_temperature", 1.0)),
        -1.0e9,
    )
    target_probability = jax.nn.softmax(target_logits, axis=-1)
    target_entropy = -jnp.sum(
        target_probability * jnp.log(jnp.maximum(target_probability, 1.0e-12)),
        axis=-1,
    )
    confidence_sum = jnp.sum(combined_confidence)
    confidence_square_sum = jnp.sum(jnp.square(combined_confidence))
    policy_weight = float(getattr(loss_v2, "decision_policy_weight", 0.25))
    critic_weight = 1.0 if critic_enabled else 0.0
    actor_weight = 1.0 if actor_enabled else 0.0
    component_signature_weight = float(
        getattr(loss_v2, "component_signature_weight", 0.25)
    )
    total = (
        critic_weight * (q_total + component_signature_weight * component_fit)
        + actor_weight * policy_weight * policy_coupling
        + (1.0 if posterior_decision_enabled else 0.0)
        * float(getattr(loss_v2, "posterior_decision_weight", 0.25))
        * posterior_decision_loss
    )
    return total, {
        "signature_loss": total,
        "signature_huber_loss": critic_weight * fit,
        "signature_hinge_loss": critic_weight * hinge_loss,
        "component_signature_loss": critic_weight * component_fit,
        "posterior_decision_loss": (
            (1.0 if posterior_decision_enabled else 0.0)
            * posterior_decision_loss
        ),
        "component_pairwise_action_signature_divergence": action_signature_divergence,
        "component_one_hot_actor_tv": actor_tv,
        "component_one_hot_top_action_disagreement": top_action_disagreement,
        "decision_policy_loss": actor_weight * policy_coupling,
        "decision_supervision_confidence": jnp.mean(combined_confidence),
        "decision_target_entropy": jnp.mean(target_entropy * anchor_valid),
        "decision_top_action_stability": jnp.mean(
            top_action_stability * anchor_valid
        ),
        "decision_confidence_q10": jnp.quantile(combined_confidence, 0.10),
        "decision_confidence_q50": jnp.quantile(combined_confidence, 0.50),
        "decision_confidence_q90": jnp.quantile(combined_confidence, 0.90),
        "anchor_policy_drift_kl": jnp.sum(drift_kl * anchor_valid)
        / jnp.maximum(jnp.sum(anchor_valid), 1.0),
        "anchor_drift_weight": jnp.sum(drift_weight)
        / jnp.maximum(jnp.sum(anchor_valid), 1.0),
        "anchor_context_fingerprint_match_fraction": jnp.sum(
            anchor_valid * context_fingerprint_match
        )
        / jnp.maximum(jnp.sum(anchor_valid), 1.0),
        "decision_effective_anchor_count": jnp.square(confidence_sum)
        / jnp.maximum(confidence_square_sum, 1.0e-8),
    }


def separation_loss(
    *,
    context_a: Any,
    context_b: Any,
    equivalent_mask: Any,
    margin: Any,
    weights: Any,
) -> Any:
    """L_separation on matched pairs (§3.2).

    Observable-equivalent pairs (``equivalent_mask``) contribute squared
    cosine distance; decision-distinct pairs contribute
    ``max(0, m - d_cos(c_i,c_j))`` with ``m`` the registered margin.
    """

    import jax.numpy as jnp

    left = jnp.asarray(context_a, dtype=jnp.float32)
    right = jnp.asarray(context_b, dtype=jnp.float32)
    left = left / jnp.maximum(jnp.linalg.norm(left, axis=-1, keepdims=True), 1.0e-6)
    right = right / jnp.maximum(jnp.linalg.norm(right, axis=-1, keepdims=True), 1.0e-6)
    distance = 1.0 - jnp.sum(left * right, axis=-1)
    equivalent = jnp.asarray(equivalent_mask, dtype=jnp.bool_)
    margin_value = jnp.asarray(margin, dtype=jnp.float32)
    items = jnp.where(
        equivalent,
        jnp.square(distance),
        jnp.maximum(0.0, margin_value - distance),
    )
    pair_weights = jnp.asarray(weights, dtype=jnp.float32)
    return jnp.sum(pair_weights * items) / jnp.maximum(jnp.sum(pair_weights), 1.0)


def separation_anchor_objective(
    *, model: Any, params: Any, separation_terms: Any
) -> tuple[Any, Mapping[str, Any]]:
    """Evaluate matched-pair separation on the current parameters."""

    import jax.numpy as jnp

    from .gradient_routing import parameters_for_loss

    if separation_terms is None:
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        return zero, {
            "protocol_equivalent_pair_distance": zero,
            "protocol_distinct_pair_distance": zero,
            "protocol_separation_margin_saturation_fraction": zero,
            "protocol_separation_effective_pair_weight": zero,
        }
    pair_count = jnp.asarray(separation_terms.equivalent_mask).shape[0]
    dropped = jnp.zeros((pair_count,), dtype=jnp.bool_)
    owned_params = parameters_for_loss(params, loss_name="separation")
    _, output_a = model.apply(
        {"params": owned_params},
        separation_terms.ego_state_a,
        separation_terms.probe_observations[0],
        dropped,
        method=model.step,
    )
    _, output_b = model.apply(
        {"params": owned_params},
        separation_terms.ego_state_b,
        separation_terms.probe_observations[1],
        dropped,
        method=model.step,
    )
    left = output_a.protocol_embedding
    right = output_b.protocol_embedding
    left = left / jnp.maximum(jnp.linalg.norm(left, axis=-1, keepdims=True), 1.0e-6)
    right = right / jnp.maximum(jnp.linalg.norm(right, axis=-1, keepdims=True), 1.0e-6)
    distance = 1.0 - jnp.sum(left * right, axis=-1)
    equivalent = jnp.asarray(separation_terms.equivalent_mask, dtype=jnp.bool_)
    weights = jnp.asarray(separation_terms.weights, dtype=jnp.float32)

    def weighted_mean(mask: Any) -> Any:
        selected = weights * jnp.asarray(mask, dtype=jnp.float32)
        return jnp.sum(selected * distance) / jnp.maximum(jnp.sum(selected), 1.0)

    loss = separation_loss(
        context_a=output_a.protocol_embedding,
        context_b=output_b.protocol_embedding,
        equivalent_mask=separation_terms.equivalent_mask,
        margin=separation_terms.margin,
        weights=separation_terms.weights,
    )
    distinct = ~equivalent
    distinct_weight = weights * distinct.astype(jnp.float32)
    saturation = distance >= jnp.asarray(separation_terms.margin, dtype=jnp.float32)
    metrics = {
        "protocol_equivalent_pair_distance": weighted_mean(equivalent),
        "protocol_distinct_pair_distance": weighted_mean(distinct),
        "protocol_separation_margin_saturation_fraction": jnp.sum(
            distinct_weight * saturation.astype(jnp.float32)
        )
        / jnp.maximum(jnp.sum(distinct_weight), 1.0),
        "protocol_separation_effective_pair_weight": jnp.sum(weights),
    }
    return loss, metrics


def response_objective(*, model: Any, params: Any, batch: RolloutBatch):
    """Proper mixture NLL L_response over rollout transitions (§2.3)."""

    import jax.numpy as jnp

    from .response_targets import (
        extract_partner_response_targets,
        mixture_response_loss,
        official_partner_observation_planes,
        pairwise_component_response_divergence,
    )
    from .gradient_routing import parameters_for_loss

    _, prediction = model.apply(
        {"params": parameters_for_loss(params, loss_name="response")},
        batch.initial_policy_state,
        batch.observations,
        batch.previous_actions,
        batch.episode_starts,
        batch.actions,
        method=model.response_sequence,
    )
    _, no_component_prediction = model.apply(
        {"params": parameters_for_loss(params, loss_name="response")},
        batch.initial_policy_state,
        batch.observations,
        batch.previous_actions,
        batch.episode_starts,
        batch.actions,
        method=model.response_sequence_without_component,
    )
    planes = official_partner_observation_planes(batch.observations.shape[-1])
    targets = extract_partner_response_targets(
        batch.observations[:-1], batch.response_next_observations, planes=planes
    )
    losses = mixture_response_loss(prediction, targets, mask=batch.ppo_mask)
    no_component_losses = mixture_response_loss(
        no_component_prediction, targets, mask=batch.ppo_mask
    )
    event_weight = jnp.asarray(batch.ppo_mask, dtype=jnp.float32) * jnp.asarray(
        targets.event_mask, dtype=jnp.float32
    )
    return losses.total, {
        "response_total_loss": losses.total,
        "response_visibility_loss": losses.visibility,
        "response_position_loss": losses.relative_position,
        "response_direction_loss": losses.direction,
        "response_inventory_loss": losses.inventory,
        "response_event_loss": losses.interaction_change,
        "response_no_component_total_loss": no_component_losses.total,
        "component_predictive_nll_gain": no_component_losses.total - losses.total,
        "response_event_prevalence": jnp.sum(
            event_weight * jnp.asarray(targets.interaction_change, dtype=jnp.float32)
        ) / jnp.maximum(jnp.sum(event_weight), 1.0),
        "component_pairwise_response_divergence": (
            pairwise_component_response_divergence(
                prediction, mask=batch.ppo_mask
            )
        ),
    }


def compute_loss(
    *,
    model: Any,
    params: Mapping[str, Any],
    batch: RolloutBatch,
    config: Any,
    anchors: Any = None,
    separation_terms: Any = None,
    auxiliary_active: Any = True,
    ppo_active: Any = True,
    response_active: Any = True,
) -> LossBundle:
    """Combined four-loss objective at the current parameters (§3.1/§3.3).

    ``separation_terms`` is the §3.2 ``SeparationTerms`` payload: its
    classification fields are precomputed constants, but the two ego
    forward passes run here on ``params`` so L_separation contributes a
    live gradient (§3.4 ownership: capability/protocol encoders).
    """

    import jax
    import jax.numpy as jnp

    from .gradient_routing import parameters_for_loss

    variant = str(getattr(config, "method_variant", "r0")).lower()
    implemented = {"r0", "b0"} | AUXILIARY_VARIANTS
    if variant not in implemented:
        raise ValueError("The loss received an unimplemented method variant.")
    critic_signature_enabled = variant in CRITIC_SIGNATURE_VARIANTS
    actor_decision_enabled = variant in ACTOR_DECISION_VARIANTS

    _, output = model.apply(
        {"params": parameters_for_loss(params, loss_name="ppo")},
        batch.initial_policy_state,
        batch.observations,
        batch.previous_actions,
        batch.episode_starts,
        batch.context_dropout_masks,
        method=model.sequence,
    )
    advantages, returns = generalized_advantage_estimation(
        rewards=batch.shaped_rewards,
        dones=batch.dones,
        values=batch.old_values,
        gamma=config.ppo.gamma,
        gae_lambda=config.ppo.gae_lambda,
    )
    actor, actor_metrics = ppo_actor_loss(
        logits=output.policy_logits[:-1],
        actions=batch.actions,
        old_log_probabilities=batch.old_log_probabilities,
        advantages=advantages,
        clip_epsilon=config.ppo.clip_epsilon,
        normalize_advantages=config.ppo.normalize_advantages,
        mask=batch.ppo_mask,
    )
    value = clipped_value_loss(
        predictions=output.state_value[:-1],
        old_predictions=batch.old_values[:-1],
        targets=returns,
        clip_epsilon=config.ppo.value_clip_epsilon,
        mask=batch.ppo_mask,
    )
    entropy = jnp.sum(categorical_entropy(output.policy_logits[:-1]) * batch.ppo_mask) / jnp.maximum(jnp.sum(batch.ppo_mask), 1.0)
    ppo_total = (
        actor
        + float(config.ppo.value_weight) * value
        - float(config.ppo.entropy_weight) * entropy
    )

    response_total, response_metrics = response_objective(
        model=model, params=params, batch=batch
    )
    zero = jnp.asarray(0.0, dtype=jnp.float32)

    def active_anchor_terms(_: Any):
        signature, signature_values = signature_anchor_objective(
            model=model,
            params=params,
            anchors=anchors,
            loss_v2=config.loss_v2,
            critic_enabled=critic_signature_enabled,
            actor_enabled=actor_decision_enabled,
            posterior_decision_enabled=variant != "response_only_posterior",
        )
        separation, separation_values = separation_anchor_objective(
            model=model, params=params, separation_terms=separation_terms
        )
        return signature, separation, {**signature_values, **separation_values}

    def inactive_anchor_terms(_: Any):
        return zero, zero, {
            "signature_loss": zero,
            "signature_huber_loss": zero,
            "signature_hinge_loss": zero,
            "component_signature_loss": zero,
            "posterior_decision_loss": zero,
            "component_pairwise_action_signature_divergence": zero,
            "component_one_hot_actor_tv": zero,
            "component_one_hot_top_action_disagreement": zero,
            "decision_policy_loss": zero,
            "decision_supervision_confidence": zero,
            "decision_target_entropy": zero,
            "decision_top_action_stability": zero,
            "decision_confidence_q10": zero,
            "decision_confidence_q50": zero,
            "decision_confidence_q90": zero,
            "anchor_policy_drift_kl": zero,
            "anchor_drift_weight": zero,
            "anchor_context_fingerprint_match_fraction": zero,
            "decision_effective_anchor_count": zero,
            "protocol_equivalent_pair_distance": zero,
            "protocol_distinct_pair_distance": zero,
            "protocol_separation_margin_saturation_fraction": zero,
            "protocol_separation_effective_pair_weight": zero,
        }

    if anchors is None and separation_terms is None:
        signature_total, separation_total, signature_metrics = (
            inactive_anchor_terms(None)
        )
    else:
        signature_total, separation_total, signature_metrics = jax.lax.cond(
            jnp.asarray(auxiliary_active, dtype=jnp.bool_),
            active_anchor_terms,
            inactive_anchor_terms,
            operand=None,
        )

    anchor_weight = 1.0 if variant in ANCHOR_VARIANTS else 0.0
    architecture_weight = 1.0 if variant in RESPONSE_VARIANTS else 0.0
    capability_weight = 1.0 if variant in CAPABILITY_VARIANTS else 0.0
    separation_weight = 1.0 if variant in SEPARATION_VARIANTS else 0.0
    capability_stability = capability_consistency_loss(
        output.capability, batch.episode_starts
    )
    (
        capability_prediction,
        capability_variance,
        capability_covariance,
        capability_metrics,
    ) = capability_auxiliary_objective(
        capability_sequence=output.capability[:-1],
        observations=batch.observations[:-1],
        initial_previous_observation=(
            batch.initial_policy_state.previous_observation
        ),
        initial_steps=batch.initial_policy_state.capability_carry.steps,
        episode_starts=batch.episode_starts[:-1],
        transition_mask=batch.ppo_mask,
        variance_floor=float(
            getattr(config.loss_v2, "capability_variance_floor", 0.05)
        ),
    )
    posterior = jnp.asarray(output.protocol_probabilities[:-1], dtype=jnp.float32)
    posterior_entropy_values = -jnp.sum(
        posterior * jnp.log(jnp.maximum(posterior, 1.0e-12)), axis=-1
    )
    component_utilization = jnp.mean(
        posterior, axis=tuple(range(posterior.ndim - 1))
    )
    component_geometry = model.apply(
        {"params": params}, method=model.component_embedding_matrix
    )
    total = (
        jnp.asarray(ppo_active, dtype=jnp.float32) * ppo_total
        + anchor_weight * float(config.loss_v2.signature_weight) * signature_total
        + jnp.asarray(response_active, dtype=jnp.float32)
        * architecture_weight
        * float(config.loss_v2.response_weight)
        * response_total
        + separation_weight
        * float(config.loss_v2.separation_weight)
        * separation_total
        + capability_weight
        * float(getattr(config.loss_v2, "capability_consistency_weight", 0.01))
        * capability_stability
        * jnp.asarray(response_active, dtype=jnp.float32)
        + capability_weight
        * float(getattr(config.loss_v2, "capability_prediction_weight", 0.1))
        * capability_prediction
        * jnp.asarray(response_active, dtype=jnp.float32)
        + capability_weight
        * float(getattr(config.loss_v2, "capability_variance_weight", 0.01))
        * capability_variance
        * jnp.asarray(response_active, dtype=jnp.float32)
        + capability_weight
        * float(getattr(config.loss_v2, "capability_covariance_weight", 0.01))
        * capability_covariance
        * jnp.asarray(response_active, dtype=jnp.float32)
    )
    anchor_valid_count = (
        zero
        if anchors is None
        else jnp.sum(
            jnp.any(jnp.asarray(anchors.action_mask), axis=-1).astype(jnp.float32)
        )
    )
    auxiliary_indicator = (
        jnp.asarray(auxiliary_active, dtype=jnp.float32) * anchor_weight
    )
    return LossBundle(total, {
        **actor_metrics,
        **response_metrics,
        **signature_metrics,
        "separation_loss": separation_total,
        "capability_consistency_loss": capability_stability,
        **capability_metrics,
        "anchor_effective_sample_size": auxiliary_indicator * anchor_valid_count,
        "anchor_use_count_max": auxiliary_indicator,
        "anchor_auxiliary_actual_total_weight": auxiliary_indicator
        * (
            float(config.loss_v2.signature_weight)
            + separation_weight * float(config.loss_v2.separation_weight)
        ),
        "auxiliary_gradient_norm": zero,
        "ppo_total_loss": ppo_total,
        "total_loss": total,
        "method_variant_code": jnp.asarray(
            {
                "r0": -1,
                "b0": 0,
                "b1": 1,
                "b2": 2,
                "deterministic_context": 3,
                "decision_only": 4,
                "q_only": 5,
                "actor_only": 6,
                "no_separation": 7,
                "no_capability": 8,
                "response_only_posterior": 9,
            }[variant],
            dtype=jnp.float32,
        ),
        "value_loss": value,
        "entropy": entropy,
        "mean_raw_reward": jnp.mean(batch.rewards),
        "mean_shaped_reward": jnp.mean(batch.shaped_rewards),
        "mean_posterior_entropy": jnp.mean(output.posterior_entropy[:-1]),
        "protocol_effective_component_count": jnp.mean(
            jnp.exp(posterior_entropy_values)
        ),
        "protocol_minimum_component_utilization": jnp.min(component_utilization),
        "posterior_high_confidence_fraction": jnp.mean(
            (jnp.max(posterior, axis=-1) > 0.98).astype(jnp.float32)
        ),
        "protocol_dominant_component_fraction": jnp.max(component_utilization),
        "protocol_component_usage_entropy": -jnp.sum(
            component_utilization
            * jnp.log(jnp.maximum(component_utilization, 1.0e-12))
        ),
        "protocol_embedding_norm": jnp.mean(
            jnp.linalg.norm(output.protocol_embedding[:-1], axis=-1)
        ),
        "protocol_component_embedding_minimum_norm": jnp.min(
            jnp.linalg.norm(component_geometry, axis=-1)
        ),
        "protocol_component_embedding_maximum_norm": jnp.max(
            jnp.linalg.norm(component_geometry, axis=-1)
        ),
        "capability_dimension_variance": jnp.mean(
            jnp.var(output.capability[:-1], axis=(0, 1))
        ),
        "capability_mean_norm": jnp.mean(
            jnp.linalg.norm(output.capability[:-1], axis=-1)
        ),
        "context_dropout_fraction": jnp.mean(batch.context_dropout_masks[:-1].astype(jnp.float32)),
        # This objective is evaluated before the optimizer transaction.  The
        # registered post-update KL is computed in
        # ``apply_training_core_update`` after the candidate parameters exist.
        "combined_policy_kl": jnp.asarray(0.0, dtype=jnp.float32),
    })


def official_learning_rate_schedule(*, learning_rate: float, warmup_fraction: float, update_count: int, minibatches_per_epoch: int, update_epochs: int) -> Any:
    import optax

    warmup_updates = int(float(warmup_fraction) * int(update_count))
    steps_per_update = int(minibatches_per_epoch) * int(update_epochs)
    warmup = optax.linear_schedule(0.0, float(learning_rate), warmup_updates * steps_per_update)
    cosine = optax.cosine_decay_schedule(float(learning_rate), max(int(update_count) - warmup_updates, 1) * steps_per_update)
    return optax.join_schedules((warmup, cosine), (warmup_updates * steps_per_update,))


def official_reward_shaping_factor(environment_step: int | Any, *, horizon: int) -> Any:
    import jax.numpy as jnp

    return jnp.clip(1.0 - jnp.asarray(environment_step, dtype=jnp.float32) / float(horizon), 0.0, 1.0)


def make_optimizer(params: Any, *, learning_rate: float, gradient_clip_norm: float, adam_epsilon: float = 1.0e-5, anneal_learning_rate: bool = False, warmup_fraction: float = 0.0, update_count: int = 1, minibatches_per_epoch: int = 1, update_epochs: int = 1):
    import optax

    rate = official_learning_rate_schedule(
        learning_rate=learning_rate, warmup_fraction=warmup_fraction,
        update_count=update_count, minibatches_per_epoch=minibatches_per_epoch,
        update_epochs=update_epochs,
    ) if anneal_learning_rate else float(learning_rate)
    optimizer = optax.chain(optax.clip_by_global_norm(float(gradient_clip_norm)), optax.adam(rate, eps=float(adam_epsilon)))
    return optimizer, optimizer.init(params)


def polyak_update(target: Any, online: Any, coefficient: float) -> Any:
    import jax

    tau = float(coefficient)
    return jax.tree_util.tree_map(lambda old, new: (1.0 - tau) * old + tau * new, target, online)


def training_core_state(state: TrainState) -> TrainingCoreState:
    return TrainingCoreState(state.params, state.target_params, state.ppo_optimizer_state)


def merge_training_core(state: TrainState, core: TrainingCoreState) -> TrainState:
    return state._replace(params=core.params, target_params=core.target_params, ppo_optimizer_state=core.ppo_optimizer_state)


def post_update_combined_policy_kl(*, model: Any, params: Any, batch: RolloutBatch) -> Any:
    """Behaviour-vs-updated-policy KL over the complete recurrent pathway.

    The forward pass is deliberately performed after the optimizer update and
    replays ``(x, u, c)`` from the registered rollout batch.  This closes the
    gap where the pre-update PPO diagnostic was previously relabelled as a
    post-update combined-policy KL.
    """

    import jax
    import jax.numpy as jnp

    _, output = model.apply(
        {"params": params},
        batch.initial_policy_state,
        batch.observations,
        batch.previous_actions,
        batch.episode_starts,
        batch.context_dropout_masks,
        method=model.sequence,
    )
    new_log_probability = jax.nn.log_softmax(output.policy_logits[:-1], axis=-1)
    old_probability = jnp.asarray(batch.behavior_probabilities, dtype=jnp.float32)
    if old_probability.shape != new_log_probability.shape:
        raise ValueError(
            "Rollout behavior_probabilities must retain the complete action distribution."
        )
    old_log_probability = jnp.log(jnp.maximum(old_probability, 1.0e-12))
    weight = jnp.asarray(batch.ppo_mask, dtype=jnp.float32)
    return jnp.sum(
        weight
        * jnp.sum(
            old_probability * (old_log_probability - new_log_probability), axis=-1
        )
    ) / jnp.maximum(jnp.sum(weight), 1.0)


def policy_kl_between_params(
    *, model: Any, before_params: Any, after_params: Any, batch: RolloutBatch
) -> Any:
    """Exact before/after categorical KL on one complete recurrent batch."""

    import jax
    import jax.numpy as jnp

    def probabilities(params: Any) -> tuple[Any, Any]:
        _, output = model.apply(
            {"params": params},
            batch.initial_policy_state,
            batch.observations,
            batch.previous_actions,
            batch.episode_starts,
            batch.context_dropout_masks,
            method=model.sequence,
        )
        log_probability = jax.nn.log_softmax(output.policy_logits[:-1], axis=-1)
        return jnp.exp(log_probability), log_probability

    before_probability, before_log_probability = probabilities(before_params)
    _, after_log_probability = probabilities(after_params)
    weight = jnp.asarray(batch.ppo_mask, dtype=jnp.float32)
    return jnp.sum(
        weight
        * jnp.sum(
            before_probability * (before_log_probability - after_log_probability),
            axis=-1,
        )
    ) / jnp.maximum(jnp.sum(weight), 1.0)


def apply_training_core_update(
    *,
    model: Any,
    core: TrainingCoreState,
    optimizer: Any,
    batch: RolloutBatch,
    config: Any,
    anchors: Any = None,
    separation_terms: Any = None,
    auxiliary_active: Any = True,
    ppo_active: Any = True,
    response_active: Any = True,
    enforce_kl_stop: bool = True,
):
    """One combined gradient step over all four losses (§3.3)."""

    import jax
    import jax.numpy as jnp
    import optax

    def objective(candidate: Any):
        loss = compute_loss(
            model=model, params=candidate, batch=batch, config=config,
            anchors=anchors, separation_terms=separation_terms,
            auxiliary_active=auxiliary_active,
            ppo_active=ppo_active,
            response_active=response_active,
        )
        return loss.total, loss.metrics

    (_, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(core.params)
    variant = str(getattr(config, "method_variant", "r0")).lower()
    if variant in ANCHOR_VARIANTS and (
        anchors is not None or separation_terms is not None
    ):
        def anchor_objective(candidate_params: Any) -> Any:
            signature, _ = signature_anchor_objective(
                model=model,
                params=candidate_params,
                anchors=anchors,
                loss_v2=config.loss_v2,
                critic_enabled=variant in CRITIC_SIGNATURE_VARIANTS,
                actor_enabled=variant in ACTOR_DECISION_VARIANTS,
                posterior_decision_enabled=variant != "response_only_posterior",
            )
            separation, unused_separation_metrics = separation_anchor_objective(
                model=model,
                params=candidate_params,
                separation_terms=separation_terms,
            )
            del unused_separation_metrics
            return (
                float(config.loss_v2.signature_weight) * signature
                + (1.0 if variant in SEPARATION_VARIANTS else 0.0)
                * float(config.loss_v2.separation_weight)
                * separation
            )

        auxiliary_gradients = jax.lax.cond(
            jnp.asarray(auxiliary_active, dtype=jnp.bool_),
            lambda _: jax.grad(anchor_objective)(core.params),
            lambda _: jax.tree_util.tree_map(jnp.zeros_like, core.params),
            operand=None,
        )
        auxiliary_gradient_norm = optax.global_norm(auxiliary_gradients)
    else:
        auxiliary_gradient_norm = jnp.asarray(0.0, dtype=jnp.float32)
    updates, optimizer_state = optimizer.update(gradients, core.ppo_optimizer_state, core.params)
    params = optax.apply_updates(core.params, updates)
    candidate = TrainingCoreState(params, core.target_params, optimizer_state)
    leaves = jax.tree_util.tree_leaves((candidate.params, candidate.ppo_optimizer_state))
    finite = jnp.isfinite(metrics["total_loss"]) & jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in leaves]))
    committed = jax.lax.cond(finite, lambda _: candidate, lambda _: core, operand=None)
    combined_policy_kl = post_update_combined_policy_kl(
        model=model, params=committed.params, batch=batch
    )
    finite = finite & jnp.isfinite(combined_policy_kl)
    committed = jax.lax.cond(finite, lambda _: committed, lambda _: core, operand=None)
    kl_stop = finite & bool(enforce_kl_stop) & (combined_policy_kl > float(
        config.loss_v2.combined_policy_kl_threshold
    ))
    return committed, {
        **metrics,
        "auxiliary_gradient_norm": auxiliary_gradient_norm,
        "combined_policy_kl": combined_policy_kl,
        "optimizer_applied": finite.astype(jnp.float32),
        "nonfinite_update": (~finite).astype(jnp.float32),
        "nonfinite_failure": (~finite).astype(jnp.float32),
        "kl_early_stop": kl_stop.astype(jnp.float32),
    }


def scan_training_updates(*, model: Any, core: TrainingCoreState, optimizer: Any, batch: RolloutBatch, schedule: Any, config: Any, anchors: Any = None, separation_terms: Any = None):
    """Scan minibatches with the §3.4 early-stop on combined_policy_kl.

    ``anchors`` and ``separation_terms`` are anchor-batch-global payloads.
    They participate in exactly the first optimizer transaction of an outer
    update, so each anchor is used at most once and its scientific weight is
    invariant to PPO epoch/minibatch counts.  Later scan steps execute PPO and
    transition-response losses only.
    """

    import jax
    import jax.numpy as jnp

    indexes = jnp.asarray(schedule, dtype=jnp.int32)
    flat = indexes.reshape((-1, indexes.shape[-1]))

    def one(carry: tuple[Any, Any], values: tuple[Any, Any]):
        current, active = carry
        lane_indexes, scan_index = values

        def apply(value: Any):
            updated, metrics = apply_training_core_update(
                model=model, core=value, optimizer=optimizer,
                batch=slice_rollout_lanes(batch, lane_indexes), config=config,
                anchors=None, separation_terms=None,
                auxiliary_active=False,
                ppo_active=True,
                response_active=False,
            )
            stop = (metrics["nonfinite_update"] > 0.5) | (metrics["kl_early_stop"] > 0.5)
            return updated, {
                **metrics,
                # A valid KL stop is a schedule decision, not a numerical
                # failure.  Keep the legacy field as a strict alias of the
                # non-finite condition while callers migrate.
                "nonfinite_failure": metrics["nonfinite_update"],
                "training_aborted": metrics["nonfinite_update"],
                "minibatch_skipped": jnp.asarray(0.0, dtype=jnp.float32),
            }

        def skip(value: Any):
            zero = jnp.asarray(0.0, dtype=jnp.float32)
            return value, {
                "actor_loss": zero, "approx_kl": zero, "clip_fraction": zero,
                "response_total_loss": zero, "response_visibility_loss": zero,
                "response_position_loss": zero, "response_direction_loss": zero,
                "response_inventory_loss": zero, "response_event_loss": zero,
                "response_no_component_total_loss": zero,
                "component_predictive_nll_gain": zero,
                "response_event_prevalence": zero,
                "component_pairwise_response_divergence": zero,
                "signature_loss": zero, "signature_huber_loss": zero,
                "signature_hinge_loss": zero, "decision_policy_loss": zero,
                "component_signature_loss": zero,
                "posterior_decision_loss": zero,
                "component_pairwise_action_signature_divergence": zero,
                "component_one_hot_actor_tv": zero,
                "component_one_hot_top_action_disagreement": zero,
                "decision_supervision_confidence": zero, "separation_loss": zero,
                "decision_target_entropy": zero,
                "decision_top_action_stability": zero,
                "decision_confidence_q10": zero,
                "decision_confidence_q50": zero,
                "decision_confidence_q90": zero,
                "anchor_policy_drift_kl": zero,
                "anchor_drift_weight": zero,
                "anchor_context_fingerprint_match_fraction": zero,
                "decision_effective_anchor_count": zero,
                "protocol_equivalent_pair_distance": zero,
                "protocol_distinct_pair_distance": zero,
                "protocol_separation_margin_saturation_fraction": zero,
                "protocol_separation_effective_pair_weight": zero,
                "capability_consistency_loss": zero,
                "capability_semantic_prediction_loss": zero,
                "capability_variance_floor_loss": zero,
                "capability_covariance_loss": zero,
                "capability_minimum_published_std": zero,
                "capability_publication_count": zero,
                "capability_published_sample_count": zero,
                "anchor_effective_sample_size": zero,
                "anchor_use_count_max": zero,
                "anchor_auxiliary_actual_total_weight": zero,
                "auxiliary_gradient_norm": zero,
                "ppo_total_loss": zero, "total_loss": zero, "value_loss": zero,
                "entropy": zero, "mean_raw_reward": zero, "mean_shaped_reward": zero,
                "mean_posterior_entropy": zero, "context_dropout_fraction": zero,
                "protocol_effective_component_count": zero,
                "protocol_minimum_component_utilization": zero,
                "posterior_high_confidence_fraction": zero,
                "protocol_dominant_component_fraction": zero,
                "protocol_component_usage_entropy": zero,
                "protocol_embedding_norm": zero,
                "protocol_component_embedding_minimum_norm": zero,
                "protocol_component_embedding_maximum_norm": zero,
                "capability_dimension_variance": zero,
                "capability_mean_norm": zero,
                "method_variant_code": zero,
                "combined_policy_kl": zero, "optimizer_applied": zero,
                "nonfinite_update": zero, "nonfinite_failure": zero,
                "kl_early_stop": zero, "training_aborted": zero,
                "minibatch_skipped": jnp.asarray(1.0),
            }

        updated, metrics = jax.lax.cond(active, apply, skip, current)
        return (
            updated,
            active
            & (metrics["nonfinite_update"] < 0.5)
            & (metrics["kl_early_stop"] < 0.5),
        ), metrics

    scan_indexes = jnp.arange(flat.shape[0], dtype=jnp.int32)
    (final, _), metrics = jax.lax.scan(
        one, (core, jnp.asarray(True)), (flat, scan_indexes)
    )
    metrics = {**metrics, "ppo_policy_kl": metrics["combined_policy_kl"]}
    variant = str(getattr(config, "method_variant", "r0")).lower()
    if variant in AUXILIARY_VARIANTS:
        auxiliary_final, auxiliary_metrics = apply_training_core_update(
            model=model,
            core=final,
            optimizer=optimizer,
            batch=batch,
            config=config,
            anchors=anchors if variant in ANCHOR_VARIANTS else None,
            separation_terms=(
                separation_terms if variant in SEPARATION_VARIANTS else None
            ),
            auxiliary_active=True,
            ppo_active=False,
            response_active=True,
            enforce_kl_stop=False,
        )
        auxiliary_policy_kl = policy_kl_between_params(
            model=model,
            before_params=final.params,
            after_params=auxiliary_final.params,
            batch=batch,
        )
        auxiliary_accepted = (
            jnp.isfinite(auxiliary_policy_kl)
            & (
                auxiliary_policy_kl
                <= float(config.loss_v2.combined_policy_kl_threshold)
            )
            & (auxiliary_metrics["nonfinite_update"] < 0.5)
        )
        auxiliary_final = jax.lax.cond(
            auxiliary_accepted,
            lambda _: auxiliary_final,
            lambda _: final,
            operand=None,
        )
        total_outer_policy_kl = post_update_combined_policy_kl(
            model=model, params=auxiliary_final.params, batch=batch
        )
        # Preserve the fixed scan shape expected by the compiled training
        # lifecycle while exposing the one fixed auxiliary transaction in its
        # first telemetry row. PPO stop telemetry remains untouched.
        auxiliary_fields = {
            "response_total_loss",
            "response_visibility_loss",
            "response_position_loss",
            "response_direction_loss",
            "response_inventory_loss",
            "response_event_loss",
            "response_no_component_total_loss",
            "component_predictive_nll_gain",
            "response_event_prevalence",
            "component_pairwise_response_divergence",
            "signature_loss",
            "signature_huber_loss",
            "signature_hinge_loss",
            "component_signature_loss",
            "posterior_decision_loss",
            "component_pairwise_action_signature_divergence",
            "component_one_hot_actor_tv",
            "component_one_hot_top_action_disagreement",
            "decision_policy_loss",
            "decision_supervision_confidence",
            "decision_target_entropy",
            "decision_top_action_stability",
            "decision_confidence_q10",
            "decision_confidence_q50",
            "decision_confidence_q90",
            "anchor_policy_drift_kl",
            "anchor_drift_weight",
            "anchor_context_fingerprint_match_fraction",
            "decision_effective_anchor_count",
            "protocol_equivalent_pair_distance",
            "protocol_distinct_pair_distance",
            "protocol_separation_margin_saturation_fraction",
            "protocol_separation_effective_pair_weight",
            "separation_loss",
            "capability_consistency_loss",
            "capability_semantic_prediction_loss",
            "capability_variance_floor_loss",
            "capability_covariance_loss",
            "capability_minimum_published_std",
            "capability_publication_count",
            "capability_published_sample_count",
            "anchor_effective_sample_size",
            "anchor_use_count_max",
            "anchor_auxiliary_actual_total_weight",
            "auxiliary_gradient_norm",
        }
        metrics = {
            name: (
                values.at[0].set(auxiliary_metrics[name])
                if name in auxiliary_fields
                else values
            )
            for name, values in metrics.items()
        }
        template_values = next(iter(metrics.values()))
        metric_zeros = jnp.zeros_like(template_values)
        metrics["auxiliary_policy_kl"] = metric_zeros.at[0].set(
            auxiliary_policy_kl
        )
        metrics["total_outer_update_policy_kl"] = metric_zeros.at[0].set(
            total_outer_policy_kl
        )
        metrics["auxiliary_update_accepted"] = metric_zeros.at[0].set(
            auxiliary_accepted.astype(jnp.float32)
        )
        final = auxiliary_final
    else:
        template_values = next(iter(metrics.values()))
        metric_zeros = jnp.zeros_like(template_values)
        metrics["auxiliary_policy_kl"] = metric_zeros
        metrics["total_outer_update_policy_kl"] = metrics[
            "combined_policy_kl"
        ]
        metrics["auxiliary_update_accepted"] = metric_zeros
    return final, metrics


def slice_rollout_lanes(batch: RolloutBatch, indexes: Any) -> RolloutBatch:
    import jax

    values = {}
    for name, value in batch._asdict().items():
        values[name] = (
            jax.tree_util.tree_map(lambda leaf: leaf[indexes], value)
            if name in {"initial_policy_state", "initial_target_policy_state"}
            else value[:, indexes]
        )
    return RolloutBatch(**values)


def environment_minibatch_schedule(key: Any, *, environment_count: int, minibatches_per_epoch: int, update_epochs: int) -> Any:
    import jax
    import jax.numpy as jnp

    if environment_count % minibatches_per_epoch:
        raise ValueError("Environment lanes must divide exactly into minibatches.")
    lane_count = environment_count // minibatches_per_epoch
    keys = jax.random.split(key, update_epochs)
    return jnp.stack([jax.random.permutation(k, environment_count).reshape((minibatches_per_epoch, lane_count)) for k in keys])


def adapt_effective_update_epochs(
    kl_early_stop: Any,
    *,
    current_epochs: int,
    maximum_epochs: int,
    minimum_epochs: int = 2,
) -> tuple[int, float, bool]:
    """Apply the registered next-update epoch response to any KL stop event."""

    import numpy as np

    current = int(current_epochs)
    maximum = int(maximum_epochs)
    minimum = int(minimum_epochs)
    if not 1 <= minimum <= current <= maximum:
        raise ValueError("Effective PPO epoch bounds are inconsistent.")
    events = np.asarray(kl_early_stop, dtype=np.float32).reshape((-1,)) > 0.5
    if events.size == 0:
        raise ValueError("KL-stop telemetry cannot be empty.")
    fraction = float(np.mean(events))
    occurred = bool(np.any(events))
    if occurred:
        return max(minimum, current - 1), fraction, True
    return min(maximum, current + 1), fraction, False


def apply_training_update(*, model: Any, state: TrainState, optimizer: Any, batch: RolloutBatch, config: Any, anchors: Any = None, separation_terms: Any = None) -> TrainingUpdate:
    core, metrics = apply_training_core_update(
        model=model, core=training_core_state(state), optimizer=optimizer,
        batch=batch, config=config, anchors=anchors, separation_terms=separation_terms,
    )
    return TrainingUpdate(merge_training_core(state, core), metrics)


__all__ = [
    "adapt_effective_update_epochs", "apply_training_core_update", "apply_training_update",
    "capability_consistency_loss", "categorical_entropy", "categorical_log_probability", "centered",
    "clipped_value_loss", "compute_loss", "decision_policy_loss", "environment_minibatch_schedule",
    "gather_actions", "generalized_advantage_estimation", "huber",
    "make_optimizer", "merge_training_core", "official_learning_rate_schedule",
    "official_reward_shaping_factor", "polyak_update", "ppo_actor_loss",
    "post_update_combined_policy_kl",
    "response_objective", "scan_training_updates", "separation_anchor_objective", "separation_loss",
    "signature_anchor_objective", "signature_loss", "slice_rollout_lanes",
    "training_core_state",
]
