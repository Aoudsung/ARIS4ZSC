"""Belief-conditioned values and KL-constrained execution for VQBC V4.2.

Response-use and response-mask are evaluated under the same physical
slot/response law.  Their only counterfactual difference is the controller
belief supplied to the behavior-consistent continuation model.
"""

from __future__ import annotations

from typing import Any, NamedTuple


class ControlValues(NamedTuple):
    j_use_by_estimator: Any
    j_mask_by_estimator: Any
    j_use: Any
    j_mask: Any
    value_mask: Any
    per_action_response_value: Any
    information_net_value: Any
    response_marginal: Any
    updated_belief: Any
    physical_joint: Any


class PolicyValues(NamedTuple):
    logits: Any
    probabilities: Any
    kl_divergence: Any


class PolicyEffectValues(NamedTuple):
    use_policy: PolicyValues
    mask_policy: PolicyValues
    raw_response_effect: Any
    raw_policy_cost: Any
    raw_net_effect: Any
    regularized_net_effect: Any
    total_variation: Any


def uniform_slot_log_belief(batch_shape: tuple[int, ...], slot_count: int) -> Any:
    import jax.numpy as jnp

    if slot_count <= 1:
        raise ValueError("A slot belief requires at least two slots.")
    return jnp.full(
        (*batch_shape, slot_count),
        -jnp.log(jnp.asarray(slot_count, dtype=jnp.float32)),
        dtype=jnp.float32,
    )


def normalized_log_belief(log_belief: Any) -> Any:
    import jax
    import jax.numpy as jnp

    values = jnp.asarray(log_belief, dtype=jnp.float32)
    return values - jax.scipy.special.logsumexp(
        values, axis=-1, keepdims=True
    )


def deployment_belief_after_response(
    *, mode: str, current_log_belief: Any, updated_log_belief: Any
) -> Any:
    if mode not in {
        "posterior_use",
        "prior_only",
        "reference_only",
        "generic_response_information",
    }:
        raise ValueError("Unknown VQBC deployment mode.")
    if mode == "prior_only":
        current = normalized_log_belief(current_log_belief)
        return uniform_slot_log_belief(
            current.shape[:-1], int(current.shape[-1])
        )
    return normalized_log_belief(updated_log_belief)


def bellman_control_values(
    *,
    slot_belief: Any,
    response_probabilities: Any,
    reward_mean: Any,
    continuation_use_mean: Any,
    continuation_mask_mean: Any,
    gamma: float,
    probability_floor: float = 1.0e-8,
) -> ControlValues:
    """Compute raw-return use/mask values with common physical weighting.

    Shapes:
      belief ``[..., M]``;
      response ``[..., M, A, Y]``;
      reward ``[..., M, A]``;
      continuation ``[..., E, M, A, Y]``.

    Both branches integrate with ``b(m) p(y|m,a)``.  The continuation tensors
    differ only because their controller received the updated or masked belief.
    """

    import jax.numpy as jnp

    belief = jnp.asarray(slot_belief, dtype=jnp.float32)
    response = jnp.asarray(response_probabilities, dtype=jnp.float32)
    rewards = jnp.asarray(reward_mean, dtype=jnp.float32)
    use_continuation = jnp.asarray(continuation_use_mean, dtype=jnp.float32)
    mask_continuation = jnp.asarray(continuation_mask_mean, dtype=jnp.float32)
    if use_continuation.shape != mask_continuation.shape:
        raise ValueError("Use and mask continuation predictions must share shape.")
    if use_continuation.shape[-4] != 2:
        raise ValueError("Behavior-consistent control requires two estimators.")
    if response.shape[-3] != belief.shape[-1]:
        raise ValueError("Response slots and belief slots differ.")
    if rewards.shape != response.shape[:-1]:
        raise ValueError("Reward means require [..., slot, action] axes.")
    expected_shape = (
        *response.shape[:-3],
        2,
        response.shape[-3],
        response.shape[-2],
        response.shape[-1],
    )
    if use_continuation.shape != expected_shape:
        raise ValueError(
            "Continuation means require [..., estimator, slot, action, response]."
        )

    response = jnp.clip(response, float(probability_floor), 1.0)
    response = response / jnp.sum(response, axis=-1, keepdims=True)
    belief = belief / jnp.maximum(
        jnp.sum(belief, axis=-1, keepdims=True), float(probability_floor)
    )
    physical_joint = belief[..., :, None, None] * response
    response_marginal = jnp.sum(physical_joint, axis=-3)
    response_marginal = response_marginal / jnp.maximum(
        jnp.sum(response_marginal, axis=-1, keepdims=True),
        float(probability_floor),
    )
    updated = physical_joint / jnp.maximum(
        response_marginal[..., None, :, :], float(probability_floor)
    )

    immediate = jnp.einsum("...m,...ma->...a", belief, rewards)
    use_expected = jnp.einsum(
        "...may,...emay->...ea", physical_joint, use_continuation
    )
    mask_expected = jnp.einsum(
        "...may,...emay->...ea", physical_joint, mask_continuation
    )
    j_use_by_estimator = immediate[..., None, :] + float(gamma) * use_expected
    j_mask_by_estimator = immediate[..., None, :] + float(gamma) * mask_expected
    j_use = jnp.min(j_use_by_estimator, axis=-2)
    j_mask = jnp.min(j_mask_by_estimator, axis=-2)
    value_mask = jnp.max(j_mask, axis=-1)
    response_value = j_use - j_mask
    return ControlValues(
        j_use_by_estimator=j_use_by_estimator,
        j_mask_by_estimator=j_mask_by_estimator,
        j_use=j_use,
        j_mask=j_mask,
        value_mask=value_mask,
        per_action_response_value=response_value,
        information_net_value=j_use - value_mask[..., None],
        response_marginal=response_marginal,
        updated_belief=updated,
        physical_joint=physical_joint,
    )


def generic_response_information(
    *,
    slot_belief: Any,
    response_probabilities: Any,
    probability_floor: float = 1.0e-8,
) -> Any:
    import jax.numpy as jnp

    belief = jnp.asarray(slot_belief, dtype=jnp.float32)
    response = jnp.maximum(
        jnp.asarray(response_probabilities, dtype=jnp.float32),
        float(probability_floor),
    )
    response = response / jnp.sum(response, axis=-1, keepdims=True)
    mixture = jnp.einsum("...m,...may->...ay", belief, response)
    mixture_entropy = -jnp.sum(mixture * jnp.log(mixture), axis=-1)
    slot_entropy = -jnp.sum(response * jnp.log(response), axis=-1)
    expected_slot_entropy = jnp.einsum(
        "...m,...ma->...a", belief, slot_entropy
    )
    return jnp.maximum(mixture_entropy - expected_slot_entropy, 0.0)


def _temperature_axes(temperature: Any, action_values: Any) -> tuple[Any, Any]:
    import jax.numpy as jnp

    score = jnp.asarray(action_values)
    alpha = jnp.asarray(temperature, dtype=score.dtype)
    if alpha.ndim == score.ndim:
        if alpha.shape[-1] != 1:
            raise ValueError("Action-axis temperature must end in one element.")
        return alpha, alpha[..., 0]
    if alpha.ndim == score.ndim - 1:
        return alpha[..., None], alpha
    if alpha.ndim == 0:
        return alpha, alpha
    raise ValueError("Temperature cannot broadcast to action values.")


def regularized_policy(
    reference_logits: Any, score: Any, temperature: Any
) -> PolicyValues:
    import jax
    import jax.numpy as jnp

    reference_raw = jnp.asarray(reference_logits)
    score_values = jnp.asarray(score, dtype=reference_raw.dtype)
    if reference_raw.shape != score_values.shape:
        raise ValueError("Reference logits and policy scores must share shape.")
    reference = jax.nn.log_softmax(reference_raw, axis=-1)
    alpha, unused_scalar = _temperature_axes(temperature, score_values)
    del unused_scalar
    logits = reference_raw + score_values / alpha
    log_probabilities = jax.nn.log_softmax(logits, axis=-1)
    probabilities = jnp.exp(log_probabilities)
    kl = jnp.sum(probabilities * (log_probabilities - reference), axis=-1)
    return PolicyValues(
        logits=logits,
        probabilities=probabilities,
        kl_divergence=jnp.maximum(kl, 0.0),
    )


def regularized_objective_value(
    reference_logits: Any, score: Any, temperature: Any
) -> Any:
    """Return E[score] - alpha KL under the optimal regularized policy."""

    import jax
    import jax.numpy as jnp

    reference = jax.nn.log_softmax(jnp.asarray(reference_logits), axis=-1)
    score_values = jnp.asarray(score, dtype=reference.dtype)
    alpha, scalar_alpha = _temperature_axes(temperature, score_values)
    return scalar_alpha * jax.scipy.special.logsumexp(
        reference + score_values / alpha, axis=-1
    )


def policy_effect_decomposition(
    *, reference_logits: Any, j_use: Any, j_mask: Any, temperature: Any
) -> PolicyEffectValues:
    """Predict raw-return effects for the exact stochastic runtime policies."""

    import jax.numpy as jnp

    use_policy = regularized_policy(reference_logits, j_use, temperature)
    mask_policy = regularized_policy(reference_logits, j_mask, temperature)
    use_values = jnp.asarray(j_use)
    mask_values = jnp.asarray(j_mask)
    response_effect = jnp.sum(
        use_policy.probabilities * (use_values - mask_values), axis=-1
    )
    policy_cost = jnp.sum(
        (mask_policy.probabilities - use_policy.probabilities) * mask_values,
        axis=-1,
    )
    raw_net = (
        jnp.sum(use_policy.probabilities * use_values, axis=-1)
        - jnp.sum(mask_policy.probabilities * mask_values, axis=-1)
    )
    regularized_net = regularized_objective_value(
        reference_logits, use_values, temperature
    ) - regularized_objective_value(reference_logits, mask_values, temperature)
    total_variation = 0.5 * jnp.sum(
        jnp.abs(use_policy.probabilities - mask_policy.probabilities), axis=-1
    )
    return PolicyEffectValues(
        use_policy=use_policy,
        mask_policy=mask_policy,
        raw_response_effect=response_effect,
        raw_policy_cost=policy_cost,
        raw_net_effect=raw_net,
        regularized_net_effect=regularized_net,
        total_variation=total_variation,
    )


def deployment_policy(
    *,
    mode: str,
    reference_logits: Any,
    control_values: ControlValues,
    information_gain: Any,
    temperature: Any,
    generic_temperature: Any,
) -> PolicyValues:
    import jax

    if mode == "reference_only":
        reference = jax.numpy.asarray(reference_logits)
        return PolicyValues(
            logits=reference,
            probabilities=jax.nn.softmax(reference, axis=-1),
            kl_divergence=jax.numpy.zeros(reference.shape[:-1]),
        )
    if mode == "generic_response_information":
        return regularized_policy(
            reference_logits, information_gain, generic_temperature
        )
    if mode not in {"posterior_use", "prior_only"}:
        raise ValueError("Unknown VQBC deployment mode.")
    return regularized_policy(reference_logits, control_values.j_use, temperature)


def update_log_temperature(
    *,
    log_temperature: Any,
    mean_kl: Any,
    target_kl: float,
    learning_rate: float,
    minimum_temperature: float,
    maximum_temperature: float,
) -> Any:
    import jax.numpy as jnp

    lower = jnp.log(jnp.asarray(minimum_temperature, dtype=jnp.float32))
    upper = jnp.log(jnp.asarray(maximum_temperature, dtype=jnp.float32))
    return jnp.clip(
        jnp.asarray(log_temperature)
        + float(learning_rate) * (jnp.asarray(mean_kl) - float(target_kl)),
        lower,
        upper,
    )


__all__ = [
    "ControlValues",
    "PolicyEffectValues",
    "PolicyValues",
    "bellman_control_values",
    "deployment_belief_after_response",
    "deployment_policy",
    "generic_response_information",
    "normalized_log_belief",
    "policy_effect_decomposition",
    "regularized_objective_value",
    "regularized_policy",
    "uniform_slot_log_belief",
    "update_log_temperature",
]
