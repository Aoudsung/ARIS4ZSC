"""Dense twin raw-return Q learning and smooth V6 policy improvement."""

from __future__ import annotations

from typing import Any, NamedTuple


RETRACE_LAMBDA = 0.95


class RawQTargets(NamedTuple):
    targets: Any
    conservative_taken_q: Any
    target_expected_values: Any
    trace_coefficients: Any


def conservative_raw_q(raw_q1: Any, raw_q2: Any) -> Any:
    import jax.numpy as jnp

    first = jnp.asarray(raw_q1, dtype=jnp.float32)
    second = jnp.asarray(raw_q2, dtype=jnp.float32)
    if first.shape != second.shape:
        raise ValueError("Twin raw-Q heads must have identical shapes.")
    return jnp.minimum(first, second)


def gather_actions(action_values: Any, actions: Any) -> Any:
    import jax.numpy as jnp

    values = jnp.asarray(action_values, dtype=jnp.float32)
    selected = jnp.asarray(actions, dtype=jnp.int32)
    if values.shape[:-1] != selected.shape:
        raise ValueError("Raw-Q values and action indexes have incompatible axes.")
    return jnp.take_along_axis(values, selected[..., None], axis=-1)[..., 0]


def target_expected_q(action_values: Any, target_probabilities: Any) -> Any:
    import jax.numpy as jnp

    values = jnp.asarray(action_values, dtype=jnp.float32)
    probabilities = jnp.asarray(target_probabilities, dtype=jnp.float32)
    if values.shape != probabilities.shape:
        raise ValueError("Target probabilities must cover every raw-Q action.")
    probabilities = probabilities / jnp.maximum(
        jnp.sum(probabilities, axis=-1, keepdims=True), 1.0e-8
    )
    return jnp.sum(probabilities * values, axis=-1)


def recurrent_retrace_targets(
    *,
    rewards: Any,
    dones: Any,
    actions: Any,
    behavior_action_probabilities: Any,
    target_action_probabilities: Any,
    target_raw_q1: Any,
    target_raw_q2: Any,
    endpoint_target_probabilities: Any,
    endpoint_target_raw_q1: Any,
    endpoint_target_raw_q2: Any,
    gamma: float,
    trace_lambda: float = RETRACE_LAMBDA,
) -> RawQTargets:
    import jax
    import jax.numpy as jnp

    reward = jnp.asarray(rewards, dtype=jnp.float32)
    terminal = jnp.asarray(dones, dtype=jnp.bool_)
    action = jnp.asarray(actions, dtype=jnp.int32)
    behavior = jnp.asarray(behavior_action_probabilities, dtype=jnp.float32)
    target_probs = jnp.asarray(target_action_probabilities, dtype=jnp.float32)
    conservative = conservative_raw_q(target_raw_q1, target_raw_q2)
    taken_q = gather_actions(conservative, action)
    target_taken_probability = gather_actions(target_probs, action)
    coefficients = float(trace_lambda) * jnp.minimum(
        1.0, target_taken_probability / jnp.maximum(behavior, 1.0e-8)
    )
    state_values = target_expected_q(conservative, target_probs)
    endpoint = target_expected_q(
        conservative_raw_q(endpoint_target_raw_q1, endpoint_target_raw_q2),
        endpoint_target_probabilities,
    )
    next_taken = jnp.concatenate((taken_q[1:], endpoint[None]), axis=0)
    next_coeff = jnp.concatenate((coefficients[1:], jnp.zeros_like(coefficients[:1])), axis=0)
    next_values = jnp.concatenate((state_values[1:], endpoint[None]), axis=0)

    def backward(carry: Any, values: tuple[Any, ...]):
        reward_t, done_t, value_next, q_next, coefficient_next = values
        target = reward_t + float(gamma) * (1.0 - done_t.astype(jnp.float32)) * (
            value_next + coefficient_next * (carry - q_next)
        )
        return target, target

    _, reverse = jax.lax.scan(
        backward,
        endpoint,
        (reward[::-1], terminal[::-1], next_values[::-1], next_taken[::-1], next_coeff[::-1]),
    )
    return RawQTargets(
        jax.lax.stop_gradient(reverse[::-1]), taken_q, state_values, coefficients
    )


def twin_raw_q_loss(raw_q1: Any, raw_q2: Any, actions: Any, targets: Any, mask: Any) -> Any:
    import jax.numpy as jnp

    first = gather_actions(raw_q1, actions)
    second = gather_actions(raw_q2, actions)
    target = jnp.asarray(targets, dtype=jnp.float32)
    weight = jnp.asarray(mask, dtype=jnp.float32)
    denominator = jnp.maximum(jnp.sum(weight), 1.0)
    return jnp.sum(weight * (jnp.square(first - target) + jnp.square(second - target))) / (2.0 * denominator)


def rollout_retrace_loss(*, model: Any, params: Any, target_params: Any, batch: Any, gamma: float):
    import jax
    import jax.numpy as jnp

    zero_dropout = jnp.zeros_like(batch.context_dropout_masks)
    _, target = model.apply(
        {"params": target_params},
        batch.initial_target_policy_state,
        batch.observations,
        batch.previous_actions,
        batch.episode_starts,
        zero_dropout,
        method=model.sequence,
    )
    _, live = model.apply(
        {"params": params},
        batch.initial_policy_state,
        batch.observations,
        batch.previous_actions,
        batch.episode_starts,
        batch.context_dropout_masks,
        method=model.sequence,
    )
    target_probabilities = jax.nn.softmax(target.policy_logits, axis=-1)
    target_bundle = recurrent_retrace_targets(
        rewards=batch.rewards,
        dones=batch.dones,
        actions=batch.actions,
        behavior_action_probabilities=batch.behavior_probabilities,
        target_action_probabilities=target_probabilities[:-1],
        target_raw_q1=target.raw_q1[:-1],
        target_raw_q2=target.raw_q2[:-1],
        endpoint_target_probabilities=target_probabilities[-1],
        endpoint_target_raw_q1=target.raw_q1[-1],
        endpoint_target_raw_q2=target.raw_q2[-1],
        gamma=float(gamma),
    )
    loss = twin_raw_q_loss(
        live.raw_q1[:-1], live.raw_q2[:-1], batch.actions, target_bundle.targets, batch.ppo_mask
    )
    conservative = conservative_raw_q(live.raw_q1[:-1], live.raw_q2[:-1])
    action_range = jnp.max(conservative, axis=-1) - jnp.min(conservative, axis=-1)
    return loss, {
        "raw_q_retrace_loss": loss,
        "raw_q_action_range_mean": jnp.mean(action_range),
        "raw_q_action_range_p90": jnp.quantile(action_range, 0.9),
        "raw_q_head_disagreement": jnp.mean(jnp.abs(live.raw_q1[:-1] - live.raw_q2[:-1])),
        "retrace_coefficient_mean": jnp.mean(target_bundle.trace_coefficients),
        "raw_q_target_mean": jnp.mean(target_bundle.targets),
    }


def centered(values: Any) -> Any:
    import jax.numpy as jnp

    array = jnp.asarray(values, dtype=jnp.float32)
    return array - jnp.mean(array, axis=-1, keepdims=True)


def _huber(error: Any, delta: float = 1.0) -> Any:
    import jax.numpy as jnp

    absolute = jnp.abs(error)
    quadratic = jnp.minimum(absolute, float(delta))
    return 0.5 * jnp.square(quadratic) + float(delta) * (absolute - quadratic)


def anchor_all_action_loss(
    *, model: Any, params: Any, anchors: Any, replay_weights: Any | None = None
):
    import jax.numpy as jnp

    _, output = model.apply(
        {"params": params},
        anchors.policy_states,
        anchors.observations,
        jnp.zeros(anchors.anchor_ids.shape, dtype=jnp.bool_),
        method=model.step,
    )
    target = centered(anchors.fit_returns_by_action)
    first = centered(output.raw_q1)
    second = centered(output.raw_q2)
    row_weights = (
        jnp.ones(anchors.anchor_ids.shape, dtype=jnp.float32)
        if replay_weights is None
        else jnp.asarray(replay_weights, dtype=jnp.float32)
    )
    mask = jnp.asarray(anchors.action_mask, dtype=jnp.float32) * row_weights[:, None]
    denominator = jnp.maximum(jnp.sum(mask), 1.0)
    loss = 0.5 * jnp.sum(mask * (_huber(first - target) + _huber(second - target))) / denominator
    selected = jnp.argmax(conservative_raw_q(output.raw_q1, output.raw_q2), axis=-1)
    return loss, {
        "anchor_raw_q_loss": loss,
        "anchor_selected_fit_return": jnp.mean(gather_actions(anchors.fit_returns_by_action, selected)),
        "anchor_empirical_action_range": jnp.mean(jnp.max(target, axis=-1) - jnp.min(target, axis=-1)),
    }


def q_policy_coupling_loss(
    *,
    model: Any,
    params: Any,
    batch: Any,
    temperature: float,
    gap_midpoint: float,
    gap_temperature: float,
    disagreement_temperature: float,
):
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
    first, second = output.raw_q1[:-1], output.raw_q2[:-1]
    q = jax.lax.stop_gradient(conservative_raw_q(first, second))
    ordered = jnp.sort(q, axis=-1)
    gap = ordered[..., -1] - ordered[..., -2]
    disagreement = jnp.mean(jnp.abs(first - second), axis=-1)
    weight = jax.nn.sigmoid(
        (gap - float(gap_midpoint)) / float(gap_temperature)
    ) * jnp.exp(-disagreement / float(disagreement_temperature))
    target = jax.nn.softmax(q / float(temperature), axis=-1)
    log_target = jnp.log(jnp.maximum(target, 1.0e-8))
    log_policy = jax.nn.log_softmax(output.policy_logits[:-1], axis=-1)
    items = jnp.sum(target * (log_target - log_policy), axis=-1)
    valid = jnp.asarray(batch.ppo_mask, dtype=jnp.float32)
    denominator = jnp.maximum(jnp.sum(valid), 1.0)
    loss = jnp.sum(valid * weight * items) / denominator
    return loss, {
        "q_policy_coupling_loss": loss,
        "q_policy_weight_mean": jnp.sum(valid * weight) / denominator,
        "q_policy_gap_mean": jnp.sum(valid * gap) / denominator,
        "q_policy_head_disagreement_mean": jnp.sum(valid * disagreement) / denominator,
    }


def decision_equivalence_metric_loss(
    *,
    mean_a: Any,
    mean_b: Any,
    advantage_a: Any,
    advantage_b: Any,
    advantage_scale: Any,
    weights: Any,
) -> tuple[Any, Any, Any]:
    """Continuous post-evidence latent/action-signature metric objective."""

    import jax
    import jax.numpy as jnp

    latent_delta = (
        jnp.asarray(mean_a, dtype=jnp.float32)
        - jnp.asarray(mean_b, dtype=jnp.float32)
    )
    # The Euclidean norm has an undefined derivative at an exactly collapsed
    # pair.  Such pairs are expected at initialization and in masked replay
    # rows, so use the same 1e-12 squared-distance floor as the registered
    # decision-distance primitive.  This preserves the L2 geometry away from
    # zero while making its zero-point gradient finite (and equal to zero).
    latent_distance = jnp.sqrt(
        jnp.sum(jnp.square(latent_delta), axis=-1) + 1.0e-12
    )
    decision_distance = jax.lax.stop_gradient(
        jnp.sqrt(
            jnp.sum(
                jnp.square(
                    jnp.asarray(advantage_a, dtype=jnp.float32)
                    - jnp.asarray(advantage_b, dtype=jnp.float32)
                ),
                axis=-1,
            )
            + 1.0e-12
        )
        / (jnp.asarray(advantage_scale, dtype=jnp.float32) + 1.0e-8)
    )
    pair_weights = jnp.asarray(weights, dtype=jnp.float32)
    loss = jnp.sum(
        pair_weights * jnp.square(latent_distance - decision_distance)
    ) / jnp.maximum(jnp.sum(pair_weights), 1.0)
    return loss, latent_distance, decision_distance


def decision_equivalence_loss(
    *,
    model: Any,
    params: Any,
    anchors: Any,
    advantage_scale: Any,
    replay_weights: Any | None = None,
):
    import jax
    import jax.numpy as jnp

    pair_ids = jnp.asarray(anchors.matched_pair_ids, dtype=jnp.int32)
    # Rows of every matched pair are stored adjacently by replay construction;
    # ordinary adjacent rows carry -1 and are masked below.
    first_indexes = jnp.arange(0, pair_ids.shape[0] - 1, 2, dtype=jnp.int32)
    paired = jnp.stack((first_indexes, first_indexes + 1), axis=-1)
    _, output = model.apply(
        {"params": params},
        anchors.policy_states,
        anchors.observations,
        jnp.zeros(anchors.anchor_ids.shape, dtype=jnp.bool_),
        method=model.step,
    )
    mean_a, mean_b = output.belief_mean[paired[:, 0]], output.belief_mean[paired[:, 1]]
    advantage = centered(anchors.fit_returns_by_action)
    advantage_a, advantage_b = advantage[paired[:, 0]], advantage[paired[:, 1]]
    action_valid = jnp.any(anchors.action_mask, axis=-1)
    pair_valid = (
        (pair_ids[paired[:, 0]] == pair_ids[paired[:, 1]])
        & (pair_ids[paired[:, 0]] >= 0)
        & action_valid[paired[:, 0]]
        & action_valid[paired[:, 1]]
    )
    row_weights = (
        jnp.ones(pair_ids.shape, dtype=jnp.float32)
        if replay_weights is None
        else jnp.asarray(replay_weights, dtype=jnp.float32)
    )
    weights = pair_valid.astype(jnp.float32) * 0.5 * (
        row_weights[paired[:, 0]] + row_weights[paired[:, 1]]
    )
    loss, latent_distance, decision_distance = decision_equivalence_metric_loss(
        mean_a=mean_a,
        mean_b=mean_b,
        advantage_a=advantage_a,
        advantage_b=advantage_b,
        advantage_scale=advantage_scale,
        weights=weights,
    )
    return loss, {
        "decision_equivalence_loss": loss,
        "decision_equivalence_pair_count": jnp.sum(weights),
        "decision_equivalence_latent_distance": jnp.sum(weights * latent_distance) / jnp.maximum(jnp.sum(weights), 1.0),
        "decision_equivalence_target_distance": jnp.sum(weights * decision_distance) / jnp.maximum(jnp.sum(weights), 1.0),
    }


__all__ = [
    "RETRACE_LAMBDA",
    "RawQTargets",
    "anchor_all_action_loss",
    "centered",
    "conservative_raw_q",
    "decision_equivalence_loss",
    "decision_equivalence_metric_loss",
    "gather_actions",
    "q_policy_coupling_loss",
    "recurrent_retrace_targets",
    "rollout_retrace_loss",
    "target_expected_q",
    "twin_raw_q_loss",
]
