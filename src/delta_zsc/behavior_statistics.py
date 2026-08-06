"""Analytic legal-history statistics with exact Beta uncertainty."""

from __future__ import annotations

from typing import Any

from .types import BehaviorStatistics


BEHAVIOR_STREAM_COUNT = 6
BEHAVIOR_FEATURE_DIM = 12


def initial_behavior_statistics(batch_shape: tuple[int, ...]) -> BehaviorStatistics:
    import jax.numpy as jnp

    shape = tuple(int(value) for value in batch_shape) + (BEHAVIOR_STREAM_COUNT,)
    return BehaviorStatistics(
        alpha=jnp.ones(shape, dtype=jnp.float32),
        beta=jnp.ones(shape, dtype=jnp.float32),
    )


def behavior_features(statistics: BehaviorStatistics) -> Any:
    import jax.numpy as jnp

    alpha = jnp.asarray(statistics.alpha, dtype=jnp.float32)
    beta = jnp.asarray(statistics.beta, dtype=jnp.float32)
    precision = alpha + beta
    mean = alpha / jnp.maximum(precision, 1.0e-12)
    mapped_mean = 2.0 * mean - 1.0
    # Bounded evidence magnitude: 0 at the Beta(1,1) prior and asymptotically 1.
    log_precision = jnp.log(jnp.maximum(precision / 2.0, 1.0))
    evidence = jnp.tanh(log_precision / 4.0)
    return jnp.concatenate((mapped_mean, evidence), axis=-1)


def update_behavior_statistics(
    statistics: BehaviorStatistics,
    response_target: Any,
    *,
    episode_start: Any,
) -> BehaviorStatistics:
    import jax.numpy as jnp

    start = jnp.asarray(episode_start, dtype=jnp.bool_)
    prior = initial_behavior_statistics(start.shape)
    alpha = jnp.where(start[..., None], prior.alpha, statistics.alpha)
    beta = jnp.where(start[..., None], prior.beta, statistics.beta)

    trials = jnp.stack(
        (
            jnp.ones_like(response_target.direct.visibility, dtype=jnp.float32),
            jnp.asarray(response_target.direct.event_mask, dtype=jnp.float32),
            jnp.asarray(response_target.direct.visible_mask, dtype=jnp.float32),
            jnp.asarray(response_target.direct.event_mask, dtype=jnp.float32),
            jnp.asarray(response_target.interface_available, dtype=jnp.float32),
            jnp.asarray(response_target.recipe_mask, dtype=jnp.float32),
        ),
        axis=-1,
    )
    successes = jnp.stack(
        (
            jnp.asarray(response_target.direct.visibility, dtype=jnp.float32),
            jnp.asarray(response_target.direct.movement, dtype=jnp.float32),
            jnp.asarray(response_target.direct.carrying, dtype=jnp.float32),
            jnp.asarray(response_target.direct.inventory_change, dtype=jnp.float32),
            jnp.asarray(response_target.interface_changed, dtype=jnp.float32),
            jnp.asarray(response_target.recipe_changed, dtype=jnp.float32),
        ),
        axis=-1,
    )
    # A reset observation is not a response to the previous episode's action.
    trials = jnp.where(start[..., None], 0.0, trials)
    successes = jnp.where(start[..., None], 0.0, successes)
    return BehaviorStatistics(
        alpha=alpha + successes,
        beta=beta + jnp.maximum(trials - successes, 0.0),
    )


__all__ = [
    "BEHAVIOR_FEATURE_DIM",
    "BEHAVIOR_STREAM_COUNT",
    "behavior_features",
    "initial_behavior_statistics",
    "update_behavior_statistics",
]
