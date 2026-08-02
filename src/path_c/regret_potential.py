"""Decision-relevant uncertainty and policy-invariant potential shaping."""

from __future__ import annotations

from typing import Any, Callable

from .belief_set_encoder import stratified_mixture_samples


def decision_regret_from_action_values(
    action_values_by_context: Any,
    context_weights: Any | None = None,
) -> Any:
    """E_z[max_a Q(z,a)] - max_a E_z[Q(z,a)]."""

    import jax.numpy as jnp

    q = jnp.asarray(action_values_by_context, dtype=jnp.float32)
    if q.ndim < 2:
        raise ValueError("Decision regret requires context and action axes.")
    if context_weights is None:
        weights = jnp.full(
            q.shape[:-1], 1.0 / float(q.shape[-2]), dtype=jnp.float32
        )
    else:
        weights = jnp.asarray(context_weights, dtype=jnp.float32)
        if weights.shape != q.shape[:-1]:
            raise ValueError("Context weights do not align with action values.")
        weights = weights / jnp.maximum(
            jnp.sum(weights, axis=-1, keepdims=True), 1.0e-8
        )
    full_information = jnp.sum(weights * jnp.max(q, axis=-1), axis=-1)
    bayes_values = jnp.sum(weights[..., None] * q, axis=-2)
    bayes = jnp.max(bayes_values, axis=-1)
    return jnp.maximum(full_information - bayes, 0.0)


def posterior_decision_regret(
    key: Any,
    *,
    mixture_logits: Any,
    means: Any,
    log_variances: Any,
    sample_count: int,
    action_value_function: Callable[[Any], Any],
) -> Any:
    samples, weights = stratified_mixture_samples(
        key,
        mixture_logits=mixture_logits,
        means=means,
        log_variances=log_variances,
        sample_count=sample_count,
    )
    q = action_value_function(samples)
    return decision_regret_from_action_values(q, weights)


def potential_from_regret(regret: Any) -> Any:
    import jax.numpy as jnp

    return -jnp.asarray(regret, dtype=jnp.float32)


def potential_shaping(
    current_regret: Any,
    next_regret: Any,
    dones: Any,
    *,
    gamma: Any,
    weight: Any,
) -> Any:
    """Detached shaping F_t = lambda (R_t - gamma R_{t+1})."""

    import jax
    import jax.numpy as jnp

    current = jax.lax.stop_gradient(jnp.asarray(current_regret, dtype=jnp.float32))
    following = jax.lax.stop_gradient(jnp.asarray(next_regret, dtype=jnp.float32))
    terminal = jnp.asarray(dones, dtype=jnp.bool_)
    following = jnp.where(terminal, 0.0, following)
    scale = jnp.asarray(weight, dtype=jnp.float32)
    discount = jnp.asarray(gamma, dtype=jnp.float32)
    return scale * (current - discount * following)


def common_optimal_action_regret_zero(action_values_by_context: Any) -> Any:
    """Diagnostic: whether every context shares at least one optimal action."""

    import jax.numpy as jnp

    q = jnp.asarray(action_values_by_context, dtype=jnp.float32)
    maxima = jnp.max(q, axis=-1, keepdims=True)
    optimal = jnp.isclose(q, maxima, atol=1.0e-6, rtol=1.0e-6)
    common = jnp.any(jnp.all(optimal, axis=-2), axis=-1)
    regret = decision_regret_from_action_values(q)
    return common, regret


__all__ = [
    "common_optimal_action_regret_zero",
    "decision_regret_from_action_values",
    "posterior_decision_regret",
    "potential_from_regret",
    "potential_shaping",
]
