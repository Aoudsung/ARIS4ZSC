"""Learned physical-time dynamics of the exchangeable coordination mode."""

from __future__ import annotations

from typing import Any


def init_transition_logits(key: Any, component_count: int) -> Any:
    import jax
    import jax.numpy as jnp

    count = int(component_count)
    if count < 2:
        raise ValueError("A coordination-mode model requires at least two components.")
    # Weak persistence bias with a tiny seed-dependent perturbation prevents
    # exact component symmetry while leaving maximum likelihood free to move.
    noise = 0.01 * jax.random.normal(key, (count, count), dtype=jnp.float32)
    return 2.0 * jnp.eye(count, dtype=jnp.float32) + noise


def transition_matrix(logits: Any) -> Any:
    import jax.nn
    import jax.numpy as jnp

    value = jnp.asarray(logits, dtype=jnp.float32)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("Transition logits must be a square matrix.")
    return jax.nn.softmax(value, axis=-1)


def predict_belief(belief: Any, transition_logits: Any) -> Any:
    import jax.numpy as jnp

    probability = jnp.asarray(belief, dtype=jnp.float32)
    result = probability @ transition_matrix(transition_logits)
    return result / jnp.maximum(jnp.sum(result, axis=-1, keepdims=True), 1.0e-12)


__all__ = ["init_transition_logits", "predict_belief", "transition_matrix"]
