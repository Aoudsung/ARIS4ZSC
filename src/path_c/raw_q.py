"""Twin raw-return Q targets with target-policy-epoch semantics."""

from __future__ import annotations

from typing import Any, NamedTuple

from .gradient_routing import keep_owned_gradients


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
    normalizer = jnp.sum(probabilities, axis=-1, keepdims=True)
    probabilities = probabilities / jnp.maximum(normalizer, 1e-8)
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
    """Build stop-gradient recurrent Retrace targets from raw reward only.

    Time-major inputs have T rows.  Endpoint inputs describe state T.  The
    correction at t uses c_(t+1), matching the standard backward Retrace
    recursion; terminal transitions strictly remove both bootstrap terms.
    """

    import jax
    import jax.numpy as jnp

    reward = jnp.asarray(rewards, dtype=jnp.float32)
    terminal = jnp.asarray(dones, dtype=jnp.bool_)
    action = jnp.asarray(actions, dtype=jnp.int32)
    behavior = jnp.asarray(behavior_action_probabilities, dtype=jnp.float32)
    target_probs = jnp.asarray(target_action_probabilities, dtype=jnp.float32)
    if reward.shape != terminal.shape or reward.shape != action.shape:
        raise ValueError("Retrace transition time/batch axes differ.")
    if target_probs.shape[:-1] != reward.shape:
        raise ValueError("Retrace target policy axes differ from transitions.")
    if behavior.shape != reward.shape:
        raise ValueError("Behavior action probabilities must be recorded per transition.")
    if not 0.0 <= trace_lambda <= 1.0 or not 0.0 <= gamma <= 1.0:
        raise ValueError("Invalid Retrace discount or lambda.")

    conservative = conservative_raw_q(target_raw_q1, target_raw_q2)
    taken_q = gather_actions(conservative, action)
    target_taken_probability = gather_actions(target_probs, action)
    coefficients = float(trace_lambda) * jnp.minimum(
        1.0, target_taken_probability / jnp.maximum(behavior, 1e-8)
    )
    state_values = target_expected_q(conservative, target_probs)
    endpoint_values = target_expected_q(
        conservative_raw_q(endpoint_target_raw_q1, endpoint_target_raw_q2),
        endpoint_target_probabilities,
    )

    next_taken_q = jnp.concatenate((taken_q[1:], endpoint_values[None, ...]), axis=0)
    next_coefficients = jnp.concatenate(
        (coefficients[1:], jnp.zeros_like(coefficients[:1])), axis=0
    )
    next_state_values = jnp.concatenate(
        (state_values[1:], endpoint_values[None, ...]), axis=0
    )

    def backward(carry: Any, values: tuple[Any, ...]) -> tuple[Any, Any]:
        r_t, done_t, v_next, q_next, c_next = values
        not_done = 1.0 - done_t.astype(jnp.float32)
        target = r_t + float(gamma) * not_done * (
            v_next + c_next * (carry - q_next)
        )
        return target, target

    _, reversed_targets = jax.lax.scan(
        backward,
        endpoint_values,
        (
            reward[::-1],
            terminal[::-1],
            next_state_values[::-1],
            next_taken_q[::-1],
            next_coefficients[::-1],
        ),
    )
    targets = jax.lax.stop_gradient(reversed_targets[::-1])
    return RawQTargets(
        targets=targets,
        conservative_taken_q=taken_q,
        target_expected_values=state_values,
        trace_coefficients=coefficients,
    )


def twin_raw_q_loss(raw_q1: Any, raw_q2: Any, actions: Any, targets: Any, mask: Any) -> Any:
    import jax.numpy as jnp

    first = gather_actions(raw_q1, actions)
    second = gather_actions(raw_q2, actions)
    target = jnp.asarray(targets, dtype=jnp.float32)
    weight = jnp.asarray(mask, dtype=jnp.float32)
    denominator = jnp.maximum(jnp.sum(weight), 1.0)
    return jnp.sum(weight * (jnp.square(first - target) + jnp.square(second - target))) / (
        2.0 * denominator
    )


def rollout_retrace_loss(
    *,
    model: Any,
    params: Any,
    target_params: Any,
    batch: Any,
    gamma: float,
) -> tuple[Any, dict[str, Any]]:
    """Dense selected-action raw-Q supervision for every legal rollout step."""

    import jax
    import jax.numpy as jnp

    _, target = model.apply(
        {"params": target_params},
        batch.initial_target_policy_state,
        batch.observations,
        batch.previous_actions,
        batch.previous_rewards,
        batch.episode_starts,
        batch.gate_overrides,
        method=model.sequence,
    )
    _, live = model.apply(
        {"params": params},
        batch.initial_policy_state,
        batch.observations,
        batch.previous_actions,
        batch.previous_rewards,
        batch.episode_starts,
        batch.gate_overrides,
        method=model.sequence,
    )
    target_probabilities = jax.nn.softmax(target.execution_logits, axis=-1)
    target_bundle = recurrent_retrace_targets(
        rewards=batch.rewards,
        dones=batch.dones,
        actions=batch.actions,
        behavior_action_probabilities=jnp.exp(batch.old_log_probabilities),
        target_action_probabilities=target_probabilities[:-1],
        target_raw_q1=target.raw_q1[:-1],
        target_raw_q2=target.raw_q2[:-1],
        endpoint_target_probabilities=target_probabilities[-1],
        endpoint_target_raw_q1=target.raw_q1[-1],
        endpoint_target_raw_q2=target.raw_q2[-1],
        gamma=float(gamma),
        trace_lambda=RETRACE_LAMBDA,
    )
    loss = twin_raw_q_loss(
        live.raw_q1[:-1],
        live.raw_q2[:-1],
        batch.actions,
        target_bundle.targets,
        batch.ppo_mask,
    )
    conservative = conservative_raw_q(live.raw_q1[:-1], live.raw_q2[:-1])
    action_range = jnp.max(conservative, axis=-1) - jnp.min(conservative, axis=-1)
    head_disagreement = jnp.mean(jnp.abs(live.raw_q1[:-1] - live.raw_q2[:-1]))
    return loss, {
        "raw_q_retrace_loss": loss,
        "raw_q_action_range_mean": jnp.mean(action_range),
        "raw_q_action_range_p90": jnp.quantile(action_range, 0.9),
        "raw_q_head_disagreement": head_disagreement,
        "retrace_coefficient_mean": jnp.mean(target_bundle.trace_coefficients),
        "raw_q_target_mean": jnp.mean(target_bundle.targets),
    }


def anchor_all_action_loss(
    *,
    model: Any,
    params: Any,
    anchors: Any,
) -> tuple[Any, dict[str, Any]]:
    """All-action simulator-return supervision from current-epoch replay."""

    import jax.numpy as jnp

    _, output = model.apply(
        {"params": params},
        anchors.policy_states,
        anchors.observations,
        jnp.ones(anchors.anchor_ids.shape, dtype=jnp.float32),
        method=model.step,
    )
    target = jnp.asarray(anchors.fit_returns_by_action, dtype=jnp.float32)
    mask = jnp.asarray(anchors.action_mask, dtype=jnp.float32)
    denominator = jnp.maximum(jnp.sum(mask), 1.0)
    first = jnp.sum(mask * jnp.square(output.raw_q1 - target)) / denominator
    second = jnp.sum(mask * jnp.square(output.raw_q2 - target)) / denominator
    loss = 0.5 * (first + second)
    conservative = conservative_raw_q(output.raw_q1, output.raw_q2)
    selected = jnp.argmax(conservative, axis=-1)
    selected_return = gather_actions(target, selected)
    return loss, {
        "anchor_raw_q_loss": loss,
        "anchor_selected_fit_return": jnp.mean(selected_return),
        "anchor_empirical_action_range": jnp.mean(
            jnp.max(target, axis=-1) - jnp.min(target, axis=-1)
        ),
    }


def raw_q_gradients(
    objective: Any,
    params: Any,
) -> tuple[Any, Any, Any]:
    """Differentiate one raw-Q objective and enforce its parameter ownership."""

    import jax

    (loss, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(params)
    return loss, metrics, keep_owned_gradients(gradients, loss_name="raw_q")


def q_policy_coupling_loss(
    *,
    model: Any,
    params: Any,
    batch: Any,
    temperature: float,
    calibration_error: float,
    minimum_margin: float,
) -> tuple[Any, dict[str, Any]]:
    """Couple calibrated raw-Q to only the conditional residual actor."""

    import jax
    import jax.numpy as jnp

    _, output = model.apply(
        {"params": params},
        batch.initial_policy_state,
        batch.observations,
        batch.previous_actions,
        batch.previous_rewards,
        batch.episode_starts,
        batch.gate_overrides,
        method=model.sequence,
    )
    first = output.raw_q1[:-1]
    second = output.raw_q2[:-1]
    conservative = jax.lax.stop_gradient(conservative_raw_q(first, second))
    ordered = jnp.sort(conservative, axis=-1)
    margin = ordered[..., -1] - ordered[..., -2]
    agreement = jnp.max(jnp.abs(first - second), axis=-1)
    qualified = (
        (margin >= jnp.asarray(minimum_margin, dtype=jnp.float32))
        & (agreement <= jnp.asarray(calibration_error, dtype=jnp.float32))
        & (batch.gate_overrides[:-1] > 0.5)
    ).astype(jnp.float32)
    target = jax.nn.softmax(conservative / float(temperature), axis=-1)
    # Stop base actor gradients: C4 couples only the residual.
    conditional_logits = (
        jax.lax.stop_gradient(output.base_logits[:-1])
        + output.residual_logits[:-1]
    )
    log_conditional = jax.nn.log_softmax(conditional_logits, axis=-1)
    log_target = jnp.log(jnp.maximum(target, 1e-8))
    items = jnp.sum(target * (log_target - log_conditional), axis=-1)
    denominator = jnp.maximum(jnp.sum(qualified), 1.0)
    loss = jnp.sum(qualified * items) / denominator
    return loss, {
        "q_policy_coupling_loss": loss,
        "q_policy_qualified_fraction": jnp.mean(qualified),
        "q_policy_margin_mean": jnp.mean(margin),
        "q_policy_head_disagreement_mean": jnp.mean(agreement),
    }


__all__ = [
    "RETRACE_LAMBDA",
    "RawQTargets",
    "conservative_raw_q",
    "gather_actions",
    "recurrent_retrace_targets",
    "rollout_retrace_loss",
    "anchor_all_action_loss",
    "raw_q_gradients",
    "q_policy_coupling_loss",
    "target_expected_q",
    "twin_raw_q_loss",
]
