"""Exact filtering under the learned physical-time latent model."""

from __future__ import annotations

from typing import Any

from .transition import predict_belief


def uniform_belief(batch_shape: tuple[int, ...], component_count: int) -> Any:
    import jax.numpy as jnp

    return jnp.full(
        tuple(int(v) for v in batch_shape) + (int(component_count),),
        1.0 / float(component_count),
        dtype=jnp.float32,
    )


def filter_update(
    previous_belief: Any,
    transition_logits: Any,
    component_log_likelihood: Any,
) -> Any:
    """Apply one Chapman-Kolmogorov prediction and Bayes correction.

    Missing response factors are represented inside the component likelihood by
    multiplicative ones (zero log contribution).  The physical-time transition
    is never silently replaced by identity.
    """

    import jax.nn
    import jax.numpy as jnp

    predictive = predict_belief(previous_belief, transition_logits)
    return jax.nn.softmax(
        jnp.log(jnp.maximum(predictive, 1.0e-30))
        + jnp.asarray(component_log_likelihood, dtype=jnp.float32),
        axis=-1,
    )


def belief_entropy(belief: Any) -> Any:
    import jax.numpy as jnp

    probability = jnp.asarray(belief, dtype=jnp.float32)
    return -jnp.sum(
        probability * jnp.log(jnp.maximum(probability, 1.0e-12)), axis=-1
    )


__all__ = ["belief_entropy", "filter_update", "uniform_belief"]
