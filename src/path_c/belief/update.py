"""Numerically stable belief updates used by Path C."""

from __future__ import annotations

from typing import Any


def _log_normalize(log_weights: Any) -> Any:
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp

    values = jnp.asarray(log_weights)
    if values.ndim < 1:
        raise ValueError("Log belief must have a prototype axis.")
    return values - logsumexp(values, axis=-1, keepdims=True)


def uniform_log_belief(batch_size: int, num_prototypes: int) -> Any:
    """Return a normalized uniform log belief with shape ``[batch, prototype]``."""

    import jax.numpy as jnp

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if isinstance(num_prototypes, bool) or not isinstance(num_prototypes, int) or num_prototypes <= 1:
        raise ValueError("At least two prototypes are required.")
    return jnp.full(
        (batch_size, num_prototypes),
        -jnp.log(jnp.asarray(float(num_prototypes), dtype=jnp.float32)),
        dtype=jnp.float32,
    )


def log_bayes_update(
    log_belief: Any,
    per_prototype_likelihood: Any,
    *,
    probability_floor: float,
) -> Any:
    """Apply one categorical likelihood in log space and renormalize."""

    import jax.numpy as jnp

    prior = jnp.asarray(log_belief)
    likelihood = jnp.asarray(per_prototype_likelihood)
    if prior.shape != likelihood.shape or prior.ndim < 1:
        raise ValueError("Belief and likelihood must share a prototype axis.")
    floor = float(probability_floor)
    if not 0.0 < floor < 1.0:
        raise ValueError("probability_floor must lie strictly between zero and one.")
    log_likelihood = jnp.log(jnp.clip(likelihood, floor, 1.0))
    return _log_normalize(prior + log_likelihood)


def likelihood_for_token(per_prototype_response_probs: Any, observed_token: Any) -> Any:
    """Select the observed categorical likelihood for every prototype."""

    import jax.numpy as jnp

    probabilities = jnp.asarray(per_prototype_response_probs)
    token = jnp.asarray(observed_token, dtype=jnp.int32)
    if probabilities.ndim != 3:
        raise ValueError("Response probabilities must have shape [batch, prototype, token].")
    if token.shape != probabilities.shape[:1]:
        raise ValueError("observed_token must have shape [batch].")
    index = jnp.broadcast_to(token[:, None, None], (*probabilities.shape[:2], 1))
    return jnp.take_along_axis(probabilities, index, axis=-1)[..., 0]


def update_use(
    log_belief: Any,
    per_prototype_response_probs: Any,
    observed_token: Any,
    *,
    probability_floor: float,
) -> Any:
    """Update with the current candidate response coordinate."""

    return log_bayes_update(
        log_belief,
        likelihood_for_token(per_prototype_response_probs, observed_token),
        probability_floor=probability_floor,
    )


def update_mask(
    log_belief: Any,
    per_prototype_response_probs: Any,
    observed_token: Any,
    *,
    probability_floor: float,
) -> Any:
    """Mask only the current candidate response coordinate and keep prior evidence."""

    del per_prototype_response_probs, observed_token
    if not 0.0 < float(probability_floor) < 1.0:
        raise ValueError("probability_floor must lie strictly between zero and one.")
    return _log_normalize(log_belief)
