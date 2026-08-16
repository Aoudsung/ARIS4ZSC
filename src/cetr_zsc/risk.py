"""Parent-level lower-tail weights and the self-play dual update."""

from __future__ import annotations

from typing import Any

from .config import TAIL_DENSITY_RATIO_CAP


def renormalize_weights(nominal_weights: Any, observed_mask: Any) -> Any:
    """Renormalize nominal parent weights over the observed parent set."""

    import jax.numpy as jnp

    weights = jnp.asarray(nominal_weights, dtype=jnp.float32)
    observed = jnp.asarray(observed_mask, dtype=jnp.float32)
    restricted = weights * observed
    total = jnp.sum(restricted)
    return restricted / jnp.maximum(total, jnp.asarray(1.0e-8, dtype=weights.dtype))


def lower_half_cvar_weights(
    parent_returns: Any,
    nominal_weights: Any,
    observed_mask: Any = None,
) -> Any:
    """Return the closed-form lower-half tail weights.

    With uniform nominal weights, an even parent count, and every parent
    observed, the result gives each of the worst half parents
    ``TAIL_DENSITY_RATIO_CAP / M`` and gives zero to the rest.  Equal returns
    are ordered deterministically by original index because the ascending sort
    is stable.
    """

    import jax.numpy as jnp

    returns = jnp.asarray(parent_returns, dtype=jnp.float32)
    nominal = jnp.asarray(nominal_weights, dtype=jnp.float32)
    if observed_mask is None:
        observed = jnp.ones_like(returns, dtype=jnp.float32)
    else:
        observed = jnp.asarray(observed_mask, dtype=jnp.float32)
    nominal_observed = renormalize_weights(nominal, observed)
    order = jnp.argsort(returns, stable=True)
    sorted_nominal = nominal_observed[order]
    excluded_mass = jnp.cumsum(sorted_nominal) - sorted_nominal
    cap_ratio = jnp.asarray(TAIL_DENSITY_RATIO_CAP, dtype=returns.dtype)
    sorted_cap = cap_ratio * sorted_nominal
    sorted_weights = jnp.clip(
        1.0 - cap_ratio * excluded_mass,
        0.0,
        sorted_cap,
    )
    return jnp.zeros_like(nominal_observed).at[order].set(sorted_weights)


def tail_objective(
    parent_returns: Any,
    nominal_weights: Any,
    observed_mask: Any = None,
) -> Any:
    """Evaluate the parent-level lower-tail objective."""

    import jax.numpy as jnp

    weights = lower_half_cvar_weights(
        parent_returns,
        nominal_weights,
        observed_mask,
    )
    returns = jnp.asarray(parent_returns, dtype=jnp.float32)
    return jnp.sum(weights * returns)


def update_sp_dual(
    lambda_value: Any,
    reference_sp: Any,
    measured_sp: Any,
    learning_rate: Any,
) -> Any:
    """Apply one projected ascent step to the self-play constraint multiplier."""

    import jax.numpy as jnp

    value = jnp.asarray(lambda_value, dtype=jnp.float32)
    target = jnp.asarray(reference_sp, dtype=jnp.float32)
    measured = jnp.asarray(measured_sp, dtype=jnp.float32)
    rate = jnp.asarray(learning_rate, dtype=jnp.float32)
    return jnp.maximum(0.0, value + rate * (target - measured))


__all__ = [
    "lower_half_cvar_weights",
    "renormalize_weights",
    "tail_objective",
    "update_sp_dual",
]
