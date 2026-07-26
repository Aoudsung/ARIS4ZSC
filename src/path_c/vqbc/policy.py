"""Belief-conditioned Bellman values and KL-constrained execution policies."""

from __future__ import annotations

from typing import Any, NamedTuple


class ControlValues(NamedTuple):
    j_use_by_estimator: Any
    j_mask_by_estimator: Any
    j_use: Any
    j_mask: Any
    value_mask: Any
    information_net_value: Any
    response_marginal: Any
    updated_belief: Any


class PolicyValues(NamedTuple):
    logits: Any
    probabilities: Any
    kl_divergence: Any


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
    return values - jax.scipy.special.logsumexp(values, axis=-1, keepdims=True)


def deployment_belief_after_response(
    *,
    mode: str,
    current_log_belief: Any,
    updated_log_belief: Any,
) -> Any:
    """Apply the one deployment mode that refuses all response updates."""

    if mode not in {
        "posterior_use",
        "prior_only",
        "reference_only",
        "generic_response_information",
    }:
        raise ValueError("Unknown fourth-model deployment mode.")
    if mode == "prior_only":
        current = normalized_log_belief(current_log_belief)
        return uniform_slot_log_belief(
            current.shape[:-1], int(current.shape[-1])
        )
    return normalized_log_belief(updated_log_belief)


def bellman_control_values(
    *,
    class_belief: Any,
    response_probabilities: Any,
    reward_mean: Any,
    next_q_mean: Any,
    gamma: float,
    probability_floor: float = 1.0e-8,
) -> ControlValues:
    """Compute the registered response-using and response-masked values.

    ``next_q_mean`` has axes ``[..., estimator, class, action, response,
    next_action]``.  The response-using maximum applies the posterior updated by
    the candidate action and response.  The masked maximum keeps the
    pre-response class belief.
    """

    import jax.numpy as jnp

    belief = jnp.asarray(class_belief)
    response = jnp.asarray(response_probabilities)
    rewards = jnp.asarray(reward_mean)
    next_values = jnp.asarray(next_q_mean)
    if next_values.shape[-5] != 2:
        raise ValueError("Bellman control requires two continuation estimators.")
    response_marginal = jnp.einsum("...c,...cay->...ay", belief, response)
    response_marginal = jnp.maximum(response_marginal, probability_floor)
    response_marginal = response_marginal / jnp.sum(
        response_marginal, axis=-1, keepdims=True
    )
    numerator = belief[..., :, None, None] * response
    updated = numerator / jnp.maximum(
        response_marginal[..., None, :, :], probability_floor
    )
    immediate = jnp.einsum("...c,...ca->...a", belief, rewards)
    use_continuation = jnp.einsum(
        "...cay,...ecayu->...eayu", updated, next_values
    )
    use_best = jnp.max(use_continuation, axis=-1)
    use_expected = jnp.einsum("...ay,...eay->...ea", response_marginal, use_best)
    masked_continuation = jnp.einsum(
        "...c,...ecayu->...eayu", belief, next_values
    )
    masked_best = jnp.max(masked_continuation, axis=-1)
    masked_expected = jnp.einsum(
        "...ay,...eay->...ea", response_marginal, masked_best
    )
    j_use_by_estimator = immediate[..., None, :] + float(gamma) * use_expected
    j_mask_by_estimator = immediate[..., None, :] + float(gamma) * masked_expected
    j_use = jnp.min(j_use_by_estimator, axis=-2)
    j_mask = jnp.min(j_mask_by_estimator, axis=-2)
    value_mask = jnp.max(j_mask, axis=-1)
    return ControlValues(
        j_use_by_estimator=j_use_by_estimator,
        j_mask_by_estimator=j_mask_by_estimator,
        j_use=j_use,
        j_mask=j_mask,
        value_mask=value_mask,
        information_net_value=j_use - value_mask[..., None],
        response_marginal=response_marginal,
        updated_belief=updated,
    )


def generic_response_information(
    *,
    class_belief: Any,
    response_probabilities: Any,
    probability_floor: float = 1.0e-8,
) -> Any:
    import jax.numpy as jnp

    belief = jnp.asarray(class_belief)
    response = jnp.maximum(jnp.asarray(response_probabilities), probability_floor)
    response = response / jnp.sum(response, axis=-1, keepdims=True)
    mixture = jnp.einsum("...c,...cay->...ay", belief, response)
    mixture_entropy = -jnp.sum(mixture * jnp.log(mixture), axis=-1)
    class_entropy = -jnp.sum(response * jnp.log(response), axis=-1)
    expected_class_entropy = jnp.einsum("...c,...ca->...a", belief, class_entropy)
    return mixture_entropy - expected_class_entropy


def regularized_policy(
    reference_logits: Any, score: Any, temperature: Any
) -> PolicyValues:
    import jax
    import jax.numpy as jnp

    reference_raw = jnp.asarray(reference_logits)
    reference = jax.nn.log_softmax(reference_raw, axis=-1)
    alpha = jnp.asarray(temperature)
    if alpha.ndim == reference.ndim - 1:
        alpha = alpha[..., None]
    # Keep the public tensor as categorical logits.  This preserves exact
    # elementwise equality with the official logits when the learned score is
    # zero; normalization is applied only when probabilities or KL are needed.
    logits = reference_raw + jnp.asarray(score) / alpha
    log_probabilities = jax.nn.log_softmax(logits, axis=-1)
    probabilities = jnp.exp(log_probabilities)
    kl = jnp.sum(probabilities * (log_probabilities - reference), axis=-1)
    return PolicyValues(logits=logits, probabilities=probabilities, kl_divergence=kl)


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
        raise ValueError("Unknown fourth-model deployment mode.")
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
    "PolicyValues",
    "bellman_control_values",
    "deployment_belief_after_response",
    "deployment_policy",
    "generic_response_information",
    "normalized_log_belief",
    "regularized_policy",
    "uniform_slot_log_belief",
    "update_log_temperature",
]
