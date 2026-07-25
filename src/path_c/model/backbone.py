"""Official-compatible convolutional encoder and recurrent backbone.

The layer order mirrors the accepted ``ActorCriticRNN`` implementation from
the external OvercookedV2 experiments package.  Imports are delayed so static
configuration and pipeline tools do not require the remote JAX environment.
"""

from __future__ import annotations

from typing import Any, Iterator, Mapping

import numpy as np


def backbone_class() -> Any:
    """Build the Flax backbone class only when the remote runtime needs it."""

    import functools

    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    from flax.linen.initializers import constant, orthogonal

    class OfficialCNN(nn.Module):
        output_size: int
        activation: str = "relu"

        @nn.compact
        def __call__(self, observations: Any) -> Any:
            activation = nn.relu if self.activation == "relu" else nn.tanh
            values = jnp.asarray(observations, dtype=jnp.float32)
            for index, (features, kernel_size) in enumerate(
                ((128, (1, 1)), (128, (1, 1)), (8, (1, 1)),
                 (16, (3, 3)), (32, (3, 3)), (32, (3, 3)))
            ):
                values = nn.Conv(
                    features=features,
                    kernel_size=kernel_size,
                    kernel_init=orthogonal(jnp.sqrt(2.0)),
                    bias_init=constant(0.0),
                    name=f"Conv_{index}",
                )(values)
                values = activation(values)
            values = values.reshape((values.shape[0], -1))
            values = nn.Dense(
                self.output_size,
                kernel_init=orthogonal(jnp.sqrt(2.0)),
                bias_init=constant(0.0),
                name="Dense_0",
            )(values)
            return activation(values)

    class ScannedRNN(nn.Module):
        """Resettable gated recurrent unit with the official scan semantics."""

        @functools.partial(
            nn.scan,
            variable_broadcast="params",
            in_axes=0,
            out_axes=0,
            split_rngs={"params": False},
        )
        @nn.compact
        def __call__(self, carry: Any, inputs: tuple[Any, Any]) -> tuple[Any, Any]:
            embedding, episode_start = inputs
            # Match the official module construction order exactly.  Its
            # zero-carry helper reserves GRUCell_0, so the parameter-bearing
            # recurrent cell is registered as GRUCell_1.
            reset_cell = nn.GRUCell(features=embedding.shape[1])
            reset_carry = reset_cell.initialize_carry(
                jax.random.PRNGKey(0),
                (embedding.shape[0], embedding.shape[1]),
            )
            carry = jnp.where(
                jnp.asarray(episode_start, dtype=jnp.bool_)[:, None],
                reset_carry,
                carry,
            )
            return nn.GRUCell(features=embedding.shape[1])(carry, embedding)

        @staticmethod
        def initialize_carry(batch_size: int, hidden_size: int) -> Any:
            if batch_size <= 0 or hidden_size <= 0:
                raise ValueError("Recurrent carry dimensions must be positive.")
            return jnp.zeros((batch_size, hidden_size), dtype=jnp.float32)

    class OfficialRecurrentBackbone(nn.Module):
        encoder_dim: int
        hidden_dim: int
        activation: str = "relu"

        def initial_carry(self, batch_size: int) -> Any:
            if batch_size <= 0:
                raise ValueError("The recurrent batch size must be positive.")
            return ScannedRNN.initialize_carry(batch_size, self.hidden_dim)

        @nn.compact
        def __call__(
            self, carry: Any, observations: Any, episode_start: Any
        ) -> tuple[Any, Any]:
            values = jnp.asarray(observations, dtype=jnp.float32)
            starts = jnp.asarray(episode_start, dtype=jnp.bool_)
            if values.ndim != 5 or starts.shape != values.shape[:2]:
                raise ValueError(
                    "Observations must have [time, batch, height, width, channel] "
                    "shape and episode_start must have [time, batch] shape."
                )
            encoder = OfficialCNN(
                output_size=self.hidden_dim,
                activation=self.activation,
                name="CNN_0",
            )
            embedded = jax.vmap(encoder)(values)
            embedded = nn.LayerNorm(name="LayerNorm_0")(embedded)
            return ScannedRNN(name="ScannedRNN_0")(carry, (embedded, starts))

    return OfficialRecurrentBackbone


def _plain(tree: Any) -> Any:
    if isinstance(tree, Mapping):
        return {str(key): _plain(value) for key, value in tree.items()}
    return tree


def iter_leaves(
    tree: Any, prefix: tuple[str, ...] = ()
) -> Iterator[tuple[tuple[str, ...], Any]]:
    if isinstance(tree, Mapping):
        for key in sorted(tree, key=str):
            yield from iter_leaves(tree[key], (*prefix, str(key)))
        return
    yield prefix, tree


def get_leaf(tree: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = tree
    for part in path:
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"Parameter tree is missing {'/'.join(path)}.")
        value = value[part]
    return value


def set_leaf(tree: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    if not path:
        raise ValueError("A parameter leaf path cannot be empty.")
    target = tree
    for part in path[:-1]:
        child = target.get(part)
        if not isinstance(child, dict):
            raise ValueError(f"Target parameter tree is missing {'/'.join(path)}.")
        target = child
    if path[-1] not in target:
        raise ValueError(f"Target parameter tree is missing {'/'.join(path)}.")
    target[path[-1]] = value


def copy_explicit_parameter_leaves(
    initial_params: Mapping[str, Any],
    official_params: Mapping[str, Any],
    leaf_mapping: Mapping[tuple[str, ...], tuple[str, ...]],
) -> dict[str, Any]:
    """Copy only named leaf pairs after exact shape comparison.

    No path inference, suffix matching, or shape-based fallback is permitted.
    """

    target = _plain(initial_params)
    source = _plain(official_params)
    if not isinstance(target, dict) or not isinstance(source, Mapping):
        raise TypeError("Both parameter trees must be mappings.")
    if not leaf_mapping:
        raise ValueError("Official initialization requires an explicit leaf mapping.")
    target_paths = [tuple(path) for path in leaf_mapping]
    source_paths = [tuple(path) for path in leaf_mapping.values()]
    if len(target_paths) != len(set(target_paths)) or len(source_paths) != len(set(source_paths)):
        raise ValueError("Official leaf mapping cannot repeat a target or source leaf.")
    for target_path, source_path in leaf_mapping.items():
        target_value = get_leaf(target, tuple(target_path))
        source_value = get_leaf(source, tuple(source_path))
        target_shape = tuple(int(value) for value in np.shape(target_value))
        source_shape = tuple(int(value) for value in np.shape(source_value))
        if target_shape != source_shape:
            raise ValueError(
                f"Official leaf {'/'.join(source_path)} has shape {source_shape}; "
                f"target {'/'.join(target_path)} requires {target_shape}."
            )
        set_leaf(target, tuple(target_path), source_value)
    return target
