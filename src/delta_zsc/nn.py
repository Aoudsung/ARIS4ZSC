"""Small pure-JAX neural-network primitives.

Using explicit parameter dictionaries keeps parameter ownership visible:
``base_params`` and ``latent_params`` are structurally disjoint and therefore
cannot share optimizer moments or accidental auxiliary gradients.
"""

from __future__ import annotations

from typing import Any, Sequence


def init_linear(
    key: Any,
    input_dim: int,
    output_dim: int,
    *,
    scale: float | None = None,
) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    fan_in = max(int(input_dim), 1)
    standard_deviation = (
        (2.0 / fan_in) ** 0.5 if scale is None else float(scale) / fan_in**0.5
    )
    return {
        "kernel": jax.random.normal(
            key, (int(input_dim), int(output_dim)), dtype=jnp.float32
        )
        * standard_deviation,
        "bias": jnp.zeros((int(output_dim),), dtype=jnp.float32),
    }


def linear(params: dict[str, Any], value: Any) -> Any:
    import jax.numpy as jnp

    return jnp.asarray(value, dtype=jnp.float32) @ params["kernel"] + params["bias"]


def init_mlp(key: Any, dimensions: Sequence[int]) -> tuple[dict[str, Any], ...]:
    import jax

    dims = tuple(int(value) for value in dimensions)
    if len(dims) < 2 or min(dims) <= 0:
        raise ValueError("MLP dimensions must contain at least two positive values.")
    keys = jax.random.split(key, len(dims) - 1)
    return tuple(
        init_linear(keys[index], dims[index], dims[index + 1])
        for index in range(len(dims) - 1)
    )


def mlp(params: Sequence[dict[str, Any]], value: Any, *, final_activation: bool) -> Any:
    import jax.nn

    hidden = value
    for index, layer in enumerate(params):
        hidden = linear(layer, hidden)
        if index < len(params) - 1 or final_activation:
            hidden = jax.nn.tanh(hidden)
    return hidden


def init_gru(key: Any, input_dim: int, hidden_dim: int) -> dict[str, Any]:
    import jax

    update_key, reset_key, candidate_key = jax.random.split(key, 3)
    combined = int(input_dim) + int(hidden_dim)
    return {
        "update": init_linear(update_key, combined, int(hidden_dim)),
        "reset": init_linear(reset_key, combined, int(hidden_dim)),
        "candidate": init_linear(candidate_key, combined, int(hidden_dim)),
    }


def gru_step(params: dict[str, Any], carry: Any, value: Any) -> Any:
    import jax.nn
    import jax.numpy as jnp

    previous = jnp.asarray(carry, dtype=jnp.float32)
    current = jnp.asarray(value, dtype=jnp.float32)
    combined = jnp.concatenate((current, previous), axis=-1)
    update = jax.nn.sigmoid(linear(params["update"], combined))
    reset = jax.nn.sigmoid(linear(params["reset"], combined))
    candidate_input = jnp.concatenate((current, reset * previous), axis=-1)
    candidate = jnp.tanh(linear(params["candidate"], candidate_input))
    return (1.0 - update) * previous + update * candidate


def layer_normalize(value: Any, epsilon: float = 1.0e-5) -> Any:
    import jax.numpy as jnp

    array = jnp.asarray(value, dtype=jnp.float32)
    mean = jnp.mean(array, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(array - mean), axis=-1, keepdims=True)
    return (array - mean) / jnp.sqrt(variance + float(epsilon))


def tree_stop_gradient(tree: Any) -> Any:
    import jax

    return jax.tree_util.tree_map(jax.lax.stop_gradient, tree)


__all__ = [
    "gru_step",
    "init_gru",
    "init_linear",
    "init_mlp",
    "layer_normalize",
    "linear",
    "mlp",
    "tree_stop_gradient",
]
