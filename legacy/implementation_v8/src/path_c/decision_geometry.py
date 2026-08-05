"""Decision-equivalent continuation-signature geometry for DEPI."""

from __future__ import annotations

from typing import Any


def centered_action_values(values: Any) -> Any:
    """Remove the action-independent value offset from a six-action vector."""

    import jax.numpy as jnp

    array = jnp.asarray(values, dtype=jnp.float32)
    return array - jnp.mean(array, axis=-1, keepdims=True)


def decision_distance(signature_a: Any, signature_b: Any) -> Any:
    """Euclidean distance between centered empirical action signatures."""

    import jax.numpy as jnp

    left = centered_action_values(signature_a)
    right = centered_action_values(signature_b)
    if left.shape != right.shape:
        raise ValueError("Decision signatures must have identical shapes.")
    return jnp.sqrt(jnp.sum(jnp.square(left - right), axis=-1) + 1.0e-12)


__all__ = ["centered_action_values", "decision_distance"]
