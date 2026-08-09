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



def init_orthogonal(
    key: Any, input_dim: int, output_dim: int, *, scale: float
) -> dict[str, Any]:
    """Orthogonal kernel with zero bias, matching the Official initialiser.

    The pinned OvercookedV2 baselines initialise every layer this way.  Using
    the same scheme is not cosmetic: it is what makes an untransplanted DELTA
    and an Official baseline start from the same distribution, so a difference
    between them is a difference of method rather than of initialisation.
    """

    import jax
    import jax.numpy as jnp

    initializer = jax.nn.initializers.orthogonal(scale=float(scale))
    return {
        "kernel": initializer(key, (int(input_dim), int(output_dim)), jnp.float32),
        "bias": jnp.zeros((int(output_dim),), dtype=jnp.float32),
    }


def init_conv(
    key: Any,
    kernel_size: tuple[int, int],
    input_channels: int,
    output_channels: int,
    *,
    scale: float,
) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    height, width = int(kernel_size[0]), int(kernel_size[1])
    initializer = jax.nn.initializers.orthogonal(scale=float(scale))
    return {
        "kernel": initializer(
            key,
            (height, width, int(input_channels), int(output_channels)),
            jnp.float32,
        ),
        "bias": jnp.zeros((int(output_channels),), dtype=jnp.float32),
    }


def conv(params: dict[str, Any], value: Any) -> Any:
    """NHWC convolution, unit stride, SAME padding.

    These are the pinned Official settings; a stride or padding difference
    would silently change the receptive field and make a transplanted kernel
    compute something other than what it was trained to compute.
    """

    import jax
    import jax.numpy as jnp

    array = jnp.asarray(value, dtype=jnp.float32)
    flat = array.reshape((-1,) + array.shape[-3:])
    output = jax.lax.conv_general_dilated(
        flat,
        jnp.asarray(params["kernel"], dtype=jnp.float32),
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    output = output + jnp.asarray(params["bias"], dtype=jnp.float32)
    return output.reshape(array.shape[:-3] + output.shape[-3:])


def init_layer_norm(dimension: int) -> dict[str, Any]:
    import jax.numpy as jnp

    return {
        "scale": jnp.ones((int(dimension),), dtype=jnp.float32),
        "bias": jnp.zeros((int(dimension),), dtype=jnp.float32),
    }


def layer_norm(params: dict[str, Any], value: Any, epsilon: float = 1.0e-6) -> Any:
    import jax.numpy as jnp

    normalized = layer_normalize(value, epsilon=epsilon)
    return normalized * jnp.asarray(
        params["scale"], dtype=jnp.float32
    ) + jnp.asarray(params["bias"], dtype=jnp.float32)


def init_gru(key: Any, input_dim: int, hidden_dim: int) -> dict[str, Any]:
    """A GRU in the exact form the pinned Official baselines use.

    Input and hidden projections are separate, and the candidate applies the
    reset gate to the *whole* hidden projection including its bias.  The
    previous formulation here concatenated ``[x, r*h]`` through one matrix,
    which cannot express the ``r * hn_bias`` term and therefore could not
    reproduce a trained Official recurrent cell even in principle.
    """

    import jax

    keys = jax.random.split(key, 6)
    inputs, hidden = int(input_dim), int(hidden_dim)
    return {
        "input_reset": init_orthogonal(keys[0], inputs, hidden, scale=1.0),
        "input_update": init_orthogonal(keys[1], inputs, hidden, scale=1.0),
        "input_candidate": init_orthogonal(keys[2], inputs, hidden, scale=1.0),
        "hidden_reset": init_orthogonal(keys[3], hidden, hidden, scale=1.0),
        "hidden_update": init_orthogonal(keys[4], hidden, hidden, scale=1.0),
        "hidden_candidate": init_orthogonal(keys[5], hidden, hidden, scale=1.0),
    }


def gru_step(params: dict[str, Any], carry: Any, value: Any) -> Any:
    import jax.nn
    import jax.numpy as jnp

    previous = jnp.asarray(carry, dtype=jnp.float32)
    current = jnp.asarray(value, dtype=jnp.float32)
    reset = jax.nn.sigmoid(
        linear(params["input_reset"], current)
        + previous @ params["hidden_reset"]["kernel"]
    )
    update = jax.nn.sigmoid(
        linear(params["input_update"], current)
        + previous @ params["hidden_update"]["kernel"]
    )
    candidate = jnp.tanh(
        linear(params["input_candidate"], current)
        + reset * linear(params["hidden_candidate"], previous)
    )
    return (1.0 - update) * candidate + update * previous


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
    "conv",
    "gru_step",
    "init_conv",
    "init_layer_norm",
    "init_orthogonal",
    "layer_norm",
    "init_gru",
    "init_linear",
    "init_mlp",
    "layer_normalize",
    "linear",
    "mlp",
    "tree_stop_gradient",
]
