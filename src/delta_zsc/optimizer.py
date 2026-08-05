"""Minimal checkpoint-stable Adam implementation."""

from __future__ import annotations

from typing import Any

from .types import AdamState


def init_adam(params: Any) -> AdamState:
    import jax
    import jax.numpy as jnp

    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return AdamState(jnp.asarray(0, dtype=jnp.int32), zeros, zeros)


def _global_norm(tree: Any) -> Any:
    import jax
    import jax.numpy as jnp

    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    return jnp.sqrt(
        jnp.sum(jnp.stack([jnp.sum(jnp.square(leaf)) for leaf in leaves]))
    )


def adam_update(
    params: Any,
    gradients: Any,
    state: AdamState,
    *,
    learning_rate: Any,
    maximum_gradient_norm: float,
    epsilon: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
) -> tuple[Any, AdamState, Any]:
    import jax
    import jax.numpy as jnp

    norm = _global_norm(gradients)
    scale = jnp.minimum(1.0, float(maximum_gradient_norm) / jnp.maximum(norm, 1.0e-12))
    clipped = jax.tree_util.tree_map(lambda value: value * scale, gradients)
    count = state.count + 1
    first = jax.tree_util.tree_map(
        lambda old, value: float(beta1) * old + (1.0 - float(beta1)) * value,
        state.first_moment,
        clipped,
    )
    second = jax.tree_util.tree_map(
        lambda old, value: float(beta2) * old + (1.0 - float(beta2)) * jnp.square(value),
        state.second_moment,
        clipped,
    )
    correction1 = 1.0 - float(beta1) ** count.astype(jnp.float32)
    correction2 = 1.0 - float(beta2) ** count.astype(jnp.float32)
    updated = jax.tree_util.tree_map(
        lambda parameter, first_value, second_value: parameter
        - jnp.asarray(learning_rate, dtype=jnp.float32)
        * (first_value / correction1)
        / (jnp.sqrt(second_value / correction2) + float(epsilon)),
        params,
        first,
        second,
    )
    return updated, AdamState(count, first, second), norm


__all__ = ["adam_update", "init_adam"]
