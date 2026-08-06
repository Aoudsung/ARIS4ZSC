"""Exact response-only filtering for an episode-static coordination latent."""

from __future__ import annotations

from typing import Any


def uniform_belief(batch_shape: tuple[int, ...], component_count: int) -> Any:
    import jax.numpy as jnp

    return jnp.full(
        tuple(int(v) for v in batch_shape) + (int(component_count),),
        1.0 / float(component_count),
        dtype=jnp.float32,
    )


def episode_static_prior(
    previous_belief: Any,
    episode_start: Any,
    component_count: int | None = None,
) -> Any:
    """Return identity persistence inside an episode and uniform reset at its start."""

    import jax.numpy as jnp

    belief = jnp.asarray(previous_belief, dtype=jnp.float32)
    count = int(belief.shape[-1] if component_count is None else component_count)
    start = jnp.asarray(episode_start, dtype=jnp.bool_)
    uniform = uniform_belief(start.shape, count)
    normalized = belief / jnp.maximum(jnp.sum(belief, axis=-1, keepdims=True), 1.0e-30)
    return jnp.where(start[..., None], uniform, normalized)


def filter_update(previous_belief: Any, component_log_likelihood: Any) -> Any:
    """Apply one Bayes correction without a physical-time transition.

    Only component-semantic likelihoods may be passed here. Shared occurrence
    factors are deliberately absent, so a component cannot win merely by
    predicting partner-independent no-change/visibility frequencies.
    """

    import jax.nn as jnn
    import jax.numpy as jnp

    prior = jnp.asarray(previous_belief, dtype=jnp.float32)
    prior = prior / jnp.maximum(jnp.sum(prior, axis=-1, keepdims=True), 1.0e-30)
    return jnn.softmax(
        jnp.log(jnp.maximum(prior, 1.0e-30))
        + jnp.asarray(component_log_likelihood, dtype=jnp.float32),
        axis=-1,
    )


def belief_entropy(belief: Any) -> Any:
    import jax.numpy as jnp

    probability = jnp.asarray(belief, dtype=jnp.float32)
    return -jnp.sum(
        probability * jnp.log(jnp.maximum(probability, 1.0e-12)), axis=-1
    )


__all__ = [
    "belief_entropy",
    "episode_static_prior",
    "filter_update",
    "uniform_belief",
]
