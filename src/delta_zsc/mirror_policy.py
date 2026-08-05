"""Analytic KL-constrained policy improvement.

The adapted policy is not another learned actor and has no loss weight or
softmax temperature.  It is the exact solution of

    max_pi E_pi[Q] - eta KL(pi || pi_base)

with ``eta`` chosen by one-dimensional dual search so the registered KL budget
is satisfied.
"""

from __future__ import annotations

from typing import Any, NamedTuple


class MirrorPolicy(NamedTuple):
    logits: Any
    probabilities: Any
    temperature: Any
    kl_to_base: Any


def categorical_kl(probability: Any, log_probability: Any, base_log_probability: Any) -> Any:
    import jax.numpy as jnp

    return jnp.sum(
        jnp.asarray(probability, dtype=jnp.float32)
        * (
            jnp.asarray(log_probability, dtype=jnp.float32)
            - jnp.asarray(base_log_probability, dtype=jnp.float32)
        ),
        axis=-1,
    )


def expected_action_values(belief: Any, component_action_values: Any) -> Any:
    import jax.numpy as jnp

    probabilities = jnp.asarray(belief, dtype=jnp.float32)
    values = jnp.asarray(component_action_values, dtype=jnp.float32)
    if values.shape[:-2] != probabilities.shape[:-1] or values.shape[-2] != probabilities.shape[-1]:
        raise ValueError("Belief and component action-value axes differ.")
    return jnp.sum(probabilities[..., :, None] * values, axis=-2)


def _candidate(base_log_probability: Any, centered_values: Any, temperature: Any) -> tuple[Any, Any, Any]:
    import jax.nn as jnn
    import jax.numpy as jnp

    eta = jnp.asarray(temperature, dtype=jnp.float32)
    candidate_log = jnn.log_softmax(
        base_log_probability + centered_values / eta[..., None], axis=-1
    )
    candidate = jnp.exp(candidate_log)
    kl = categorical_kl(candidate, candidate_log, base_log_probability)
    return candidate_log, candidate, kl


def kl_constrained_policy(
    base_logits: Any,
    action_values: Any,
    *,
    kl_budget: float,
    iterations: int = 48,
) -> MirrorPolicy:
    """Return the unique mirror-improvement policy within the KL budget."""

    import jax
    import jax.nn as jnn
    import jax.numpy as jnp

    if not 0.0 < float(kl_budget):
        raise ValueError("KL adaptation budget must be positive.")
    if int(iterations) <= 0:
        raise ValueError("Dual-search iteration count must be positive.")
    logits = jnp.asarray(base_logits, dtype=jnp.float32)
    values = jnp.asarray(action_values, dtype=jnp.float32)
    if logits.shape != values.shape:
        raise ValueError("Base logits and adaptation values must have equal shapes.")
    base_log = jnn.log_softmax(logits, axis=-1)
    centered_values = values - jnp.mean(values, axis=-1, keepdims=True)
    batch_shape = logits.shape[:-1]
    log_low = jnp.full(batch_shape, jnp.log(1.0e-4), dtype=jnp.float32)
    log_high = jnp.full(batch_shape, jnp.log(1.0e4), dtype=jnp.float32)
    _, _, low_kl = _candidate(base_log, centered_values, jnp.exp(log_low))
    constrained = low_kl > float(kl_budget)

    def refine(_: int, bounds: tuple[Any, Any]) -> tuple[Any, Any]:
        low, high = bounds
        middle = 0.5 * (low + high)
        _, _, current_kl = _candidate(
            base_log, centered_values, jnp.exp(middle)
        )
        needs_more_temperature = current_kl > float(kl_budget)
        return (
            jnp.where(needs_more_temperature, middle, low),
            jnp.where(needs_more_temperature, high, middle),
        )

    final_low, final_high = jax.lax.fori_loop(
        0, int(iterations), refine, (log_low, log_high)
    )
    temperature = jnp.where(
        constrained,
        jnp.exp(final_high),
        jnp.exp(log_low),
    )
    adapted_log, adapted_probability, adapted_kl = _candidate(
        base_log, centered_values, temperature
    )
    # The normalized log probabilities themselves are valid categorical logits.
    return MirrorPolicy(
        logits=adapted_log,
        probabilities=adapted_probability,
        temperature=temperature,
        kl_to_base=adapted_kl,
    )


__all__ = [
    "MirrorPolicy",
    "categorical_kl",
    "expected_action_values",
    "kl_constrained_policy",
]
