"""Analytic KL-bounded policy improvement."""

from __future__ import annotations

from typing import Any


def categorical_kl_from_logits(left_logits: Any, right_logits: Any) -> Any:
    import jax.nn
    import jax.numpy as jnp

    left_logp = jax.nn.log_softmax(left_logits, axis=-1)
    right_logp = jax.nn.log_softmax(right_logits, axis=-1)
    probability = jnp.exp(left_logp)
    return jnp.sum(probability * (left_logp - right_logp), axis=-1)


def mirror_policy_logits(
    base_logits: Any,
    action_values: Any,
    *,
    kl_budget: float,
    iterations: int = 48,
) -> tuple[Any, Any, Any]:
    """Solve max_pi E_pi[Q] subject to KL(pi||pi0)<=delta.

    For a fixed dual temperature eta the solution is
    ``pi_eta proportional pi0 * exp(Q/eta)``.  A deterministic bisection finds
    eta; no actor-supervision weight or temperature is exposed.
    """

    import jax
    import jax.nn
    import jax.numpy as jnp

    base = jnp.asarray(base_logits, dtype=jnp.float32)
    values = jnp.asarray(action_values, dtype=jnp.float32)
    values = values - jnp.mean(values, axis=-1, keepdims=True)
    delta = float(kl_budget)
    if delta < 0.0:
        raise ValueError("KL budget must be non-negative.")
    base_logp = jax.nn.log_softmax(base, axis=-1)

    def candidate(log_eta: Any) -> tuple[Any, Any]:
        eta = jnp.exp(log_eta)
        logits = base_logp + values / eta[..., None]
        logp = jax.nn.log_softmax(logits, axis=-1)
        probability = jnp.exp(logp)
        kl = jnp.sum(probability * (logp - base_logp), axis=-1)
        return logits, kl

    lead = values.shape[:-1]
    low = jnp.full(lead, -9.0, dtype=jnp.float32)
    high = jnp.full(lead, 12.0, dtype=jnp.float32)

    def one(_: int, bounds: tuple[Any, Any]) -> tuple[Any, Any]:
        lo, hi = bounds
        middle = 0.5 * (lo + hi)
        _, kl = candidate(middle)
        # KL decreases monotonically as eta grows.
        return jnp.where(kl > delta, middle, lo), jnp.where(kl > delta, hi, middle)

    low, high = jax.lax.fori_loop(0, int(iterations), one, (low, high))
    # The dual temperature is an analytic constraint solution, not a learned
    # pathway.  Stopping its gradient follows the envelope theorem and avoids
    # back-propagating through 48 bisection iterations.
    solved_log_eta = jax.lax.stop_gradient(high)
    logits, kl = candidate(solved_log_eta)
    no_signal = jnp.max(values, axis=-1) - jnp.min(values, axis=-1) < 1.0e-7
    inactive = no_signal | (delta <= 0.0)
    final_logits = jnp.where(inactive[..., None], base, logits)
    final_kl = jnp.where(inactive, 0.0, kl)
    eta = jnp.where(inactive, jnp.inf, jnp.exp(solved_log_eta))
    return final_logits, final_kl, eta


__all__ = ["categorical_kl_from_logits", "mirror_policy_logits"]
