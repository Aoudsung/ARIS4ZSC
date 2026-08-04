"""Analytic legal-history behaviour statistics.

The former learned capability GRU and its consistency/prediction/variance/
covariance losses are replaced by four Beta-Bernoulli sufficient statistics.
Each statistic has an explicit observable definition, an uncertainty estimate,
and no trainable parameter or anti-collapse objective.
"""

from __future__ import annotations

from typing import Any

from .types import BehaviorPosterior


BEHAVIOR_DIMENSION = 4


def initial_behavior_posterior(
    batch_shape: tuple[int, ...], *, prior_alpha: float = 1.0, prior_beta: float = 1.0
) -> BehaviorPosterior:
    import jax.numpy as jnp

    if prior_alpha <= 0.0 or prior_beta <= 0.0:
        raise ValueError("Beta prior parameters must be positive.")
    shape = tuple(int(value) for value in batch_shape) + (BEHAVIOR_DIMENSION,)
    return BehaviorPosterior(
        alpha=jnp.full(shape, float(prior_alpha), dtype=jnp.float32),
        beta=jnp.full(shape, float(prior_beta), dtype=jnp.float32),
    )


def observable_behavior_events(
    previous_observation: Any,
    current_observation: Any,
    episode_start: Any,
) -> tuple[Any, Any]:
    """Return successes and valid-observation counts for four legal events.

    Coordinates are, in order:

    1. partner visible in the current local frame;
    2. partner moved while visible at both adjacent frames;
    3. partner visibly holds any item;
    4. partner inventory changed while visible at both adjacent frames.

    No partner action, identity, global state, or learned label is used.
    """

    import jax.numpy as jnp

    from src.path_c.response_targets import (
        extract_partner_response_targets,
        official_partner_observation_planes,
    )

    previous = jnp.asarray(previous_observation, dtype=jnp.float32)
    current = jnp.asarray(current_observation, dtype=jnp.float32)
    if previous.shape != current.shape:
        raise ValueError("Adjacent observations must have identical shapes.")
    planes = official_partner_observation_planes(current.shape[-1])
    target = extract_partner_response_targets(previous, current, planes=planes)
    previous_target = extract_partner_response_targets(
        previous, previous, planes=planes
    )
    start = jnp.asarray(episode_start, dtype=jnp.bool_)

    both_visible = jnp.asarray(target.event_mask, dtype=jnp.float32)
    transition_valid = both_visible * (~start).astype(jnp.float32)
    moved = (
        jnp.asarray(target.relative_position)
        != jnp.asarray(previous_target.relative_position)
    ).astype(jnp.float32) * transition_valid
    holding = jnp.any(jnp.asarray(target.inventory) > 0, axis=-1).astype(
        jnp.float32
    ) * jnp.asarray(target.visible_mask, dtype=jnp.float32)
    inventory_change = jnp.asarray(
        target.interaction_change, dtype=jnp.float32
    ) * transition_valid

    successes = jnp.stack(
        (
            jnp.asarray(target.visibility, dtype=jnp.float32),
            moved,
            holding,
            inventory_change,
        ),
        axis=-1,
    )
    valid = jnp.stack(
        (
            jnp.ones_like(both_visible),
            transition_valid,
            jnp.asarray(target.visible_mask, dtype=jnp.float32),
            transition_valid,
        ),
        axis=-1,
    )
    return successes, valid


def update_behavior_posterior(
    state: BehaviorPosterior,
    *,
    previous_observation: Any,
    current_observation: Any,
    episode_start: Any,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> BehaviorPosterior:
    import jax.numpy as jnp

    start = jnp.asarray(episode_start, dtype=jnp.bool_)
    reset = initial_behavior_posterior(
        tuple(start.shape), prior_alpha=prior_alpha, prior_beta=prior_beta
    )
    alpha = jnp.where(start[..., None], reset.alpha, state.alpha)
    beta = jnp.where(start[..., None], reset.beta, state.beta)
    successes, valid = observable_behavior_events(
        previous_observation, current_observation, start
    )
    if successes.shape != alpha.shape or valid.shape != alpha.shape:
        raise ValueError("Behavior evidence and posterior shapes differ.")
    return BehaviorPosterior(
        alpha=alpha + successes,
        beta=beta + valid - successes,
    )


def behavior_features(state: BehaviorPosterior) -> Any:
    """Return posterior means and log precisions, shape ``[...,8]``."""

    import jax.numpy as jnp

    alpha = jnp.asarray(state.alpha, dtype=jnp.float32)
    beta = jnp.asarray(state.beta, dtype=jnp.float32)
    precision = alpha + beta
    mean = alpha / jnp.maximum(precision, 1.0e-8)
    return jnp.concatenate((mean, jnp.log1p(precision)), axis=-1)


def behavior_variance(state: BehaviorPosterior) -> Any:
    import jax.numpy as jnp

    alpha = jnp.asarray(state.alpha, dtype=jnp.float32)
    beta = jnp.asarray(state.beta, dtype=jnp.float32)
    total = alpha + beta
    return alpha * beta / jnp.maximum(
        jnp.square(total) * (total + 1.0), 1.0e-8
    )


__all__ = [
    "BEHAVIOR_DIMENSION",
    "behavior_features",
    "behavior_variance",
    "initial_behavior_posterior",
    "observable_behavior_events",
    "update_behavior_posterior",
]
