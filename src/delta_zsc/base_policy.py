"""Immutable Official-SP reference plus a trainable residual/value branch."""

from __future__ import annotations

from typing import Any

from .nn import (
    gru_step,
    init_gru,
    init_layer_norm,
    init_linear,
    init_mlp,
    init_orthogonal,
    layer_normalize,
    linear,
    mlp,
)
from .observation import instantaneous_partner_observation


OFFICIAL_CONV_STACK = (
    ((1, 1), 128),
    ((1, 1), 128),
    ((1, 1), 8),
    ((3, 3), 16),
    ((3, 3), 32),
    ((3, 3), 32),
)
"""The pinned OvercookedV2 encoder, layer for layer."""


def init_base_params(
    key: Any,
    *,
    observation_shape: tuple[int, ...],
    task_hidden_dim: int,
    task_embedding_dim: int,
    instant_partner_dim: int,
    action_count: int,
    component_count: int,
) -> dict[str, Any]:
    """Initialise reference-shaped placeholders and trainable residual/value trees.

    The reference placeholders are replaced by the required Official SP
    checkpoint before a run starts.  They live inside ``base_params`` so a
    deployment is self-contained, but the optimiser receives only the
    ``trainable`` subtree.
    """

    import jax
    import jax.numpy as jnp

    from .nn import init_conv
    from .observation import partner_channel_indexes

    keys = jax.random.split(key, 10 + len(OFFICIAL_CONV_STACK))
    channels = int(observation_shape[-1])
    convolutions = []
    for index, (kernel_size, features) in enumerate(OFFICIAL_CONV_STACK):
        convolutions.append(
            init_conv(
                keys[index], kernel_size, channels, features, scale=2.0**0.5
            )
        )
        channels = features
    spatial = int(observation_shape[0]) * int(observation_shape[1]) * channels
    offset = len(OFFICIAL_CONV_STACK)
    reference = {
        "task_conv": tuple(convolutions),
        "task_dense": init_orthogonal(
            keys[offset], spatial, int(task_embedding_dim), scale=2.0**0.5
        ),
        "task_norm": init_layer_norm(int(task_embedding_dim)),
        "task_gru": init_gru(
            keys[offset + 1], int(task_embedding_dim), int(task_hidden_dim)
        ),
        "actor_trunk": init_orthogonal(
            keys[offset + 2], int(task_hidden_dim), int(task_hidden_dim), scale=2.0**0.5
        ),
        "actor": init_orthogonal(
            keys[offset + 3], int(task_hidden_dim), int(action_count), scale=0.01
        ),
    }

    partner_dim = int(
        observation_shape[0]
        * observation_shape[1]
        * len(partner_channel_indexes(observation_shape[-1]))
    )
    combined = int(task_hidden_dim) + int(instant_partner_dim) + int(component_count)
    residual = init_linear(keys[offset + 5], combined, int(action_count), scale=0.01)
    residual = {
        "kernel": jnp.zeros_like(residual["kernel"]),
        "bias": jnp.zeros_like(residual["bias"]),
    }
    trainable = {
        "instant_encoder": init_mlp(
            keys[offset + 4],
            (partner_dim, int(instant_partner_dim), int(instant_partner_dim)),
        ),
        "residual_actor": residual,
        "value_trunk": init_orthogonal(
            keys[offset + 6], combined, int(task_hidden_dim), scale=2.0**0.5
        ),
        "value": init_orthogonal(
            keys[offset + 7], int(task_hidden_dim), 1, scale=1.0
        ),
    }
    return {"reference": reference, "trainable": trainable}


def trainable_base_params(base_params: dict[str, Any]) -> dict[str, Any]:
    """Return the only subtree PPO may update or allocate Adam state for."""

    return base_params["trainable"]


def with_trainable_base_params(
    base_params: dict[str, Any], trainable: dict[str, Any]
) -> dict[str, Any]:
    return {"reference": base_params["reference"], "trainable": trainable}


def encode_task_frame(params: dict[str, Any], frame: Any) -> Any:
    """The Official convolutional encoder, ending in the pinned LayerNorm."""

    import jax.nn
    import jax.numpy as jnp

    from .nn import conv, layer_norm

    hidden = jnp.asarray(frame, dtype=jnp.float32)
    for layer in params["task_conv"]:
        hidden = jax.nn.relu(conv(layer, hidden))
    flat = hidden.reshape(hidden.shape[:-3] + (-1,))
    embedding = jax.nn.relu(linear(params["task_dense"], flat))
    return layer_norm(params["task_norm"], embedding)


def _heads(
    reference: dict[str, Any], trainable: dict[str, Any], task: Any, instant: Any, belief: Any
) -> tuple[Any, Any, Any, Any]:
    import jax.nn
    import jax.numpy as jnp

    reference_hidden = jax.nn.relu(linear(reference["actor_trunk"], task))
    reference_logits = linear(reference["actor"], reference_hidden)
    posterior = jnp.asarray(belief, dtype=jnp.float32)
    uniform = jnp.full_like(posterior, 1.0 / posterior.shape[-1])
    entropy = -jnp.sum(
        posterior * jnp.log(jnp.maximum(posterior, 1.0e-30)), axis=-1
    )
    confidence = 1.0 - entropy / jnp.log(
        jnp.asarray(posterior.shape[-1], dtype=jnp.float32)
    )
    innovation = confidence[..., None] * (posterior - uniform)
    residual_features = jnp.concatenate((task, instant, innovation), axis=-1)
    value_features = jnp.concatenate((task, instant, posterior), axis=-1)
    residual_logits = linear(trainable["residual_actor"], residual_features)
    critic = jax.nn.relu(linear(trainable["value_trunk"], value_features))
    value = linear(trainable["value"], critic)[..., 0]
    return reference_logits, residual_logits, reference_logits + residual_logits, value


def base_policy_step(
    params: dict[str, Any],
    task_carry: Any,
    observation: Any,
    episode_start: Any,
    belief: Any,
) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    """Advance the immutable full-frame SP reference and residual actor once."""

    import jax.numpy as jnp

    reference = params["reference"]
    trainable = params["trainable"]
    start = jnp.asarray(episode_start, dtype=jnp.bool_)
    carry = jnp.where(start[..., None], jnp.zeros_like(task_carry), task_carry)
    task_embedding = encode_task_frame(reference, observation)
    next_carry = gru_step(reference["task_gru"], carry, task_embedding)
    partner = instantaneous_partner_observation(observation)
    partner_flat = partner.reshape(partner.shape[:-3] + (-1,))
    instant = layer_normalize(
        mlp(trainable["instant_encoder"], partner_flat, final_activation=True)
    )
    reference_logits, residual_logits, logits, value = _heads(
        reference, trainable, next_carry, instant, belief
    )
    return (
        next_carry,
        next_carry,
        instant,
        reference_logits,
        residual_logits,
        logits,
        value,
    )


def base_policy_sequence(
    params: dict[str, Any],
    initial_task_carry: Any,
    observations: Any,
    episode_starts: Any,
    beliefs: Any,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    """Evaluate a time-major reference/residual policy with one GRU scan."""

    import jax
    import jax.numpy as jnp

    reference = params["reference"]
    trainable = params["trainable"]
    observation = jnp.asarray(observations, dtype=jnp.float32)
    task_embeddings = encode_task_frame(reference, observation)
    partner = instantaneous_partner_observation(observation)
    partner_flat = partner.reshape(partner.shape[:-3] + (-1,))
    instant = layer_normalize(
        mlp(trainable["instant_encoder"], partner_flat, final_activation=True)
    )

    def recurrent_step(carry: Any, values: tuple[Any, Any]):
        embedding, episode_start = values
        start = jnp.asarray(episode_start, dtype=jnp.bool_)
        reset = jnp.where(start[..., None], jnp.zeros_like(carry), carry)
        next_carry = gru_step(reference["task_gru"], reset, embedding)
        return next_carry, next_carry

    final_carry, task_features = jax.lax.scan(
        recurrent_step, initial_task_carry, (task_embeddings, episode_starts)
    )
    reference_logits, residual_logits, logits, value = _heads(
        reference, trainable, task_features, instant, beliefs
    )
    return (
        final_carry,
        task_features,
        task_features,
        instant,
        reference_logits,
        residual_logits,
        logits,
        value,
    )


__all__ = [
    "OFFICIAL_CONV_STACK",
    "base_policy_sequence",
    "base_policy_step",
    "encode_task_frame",
    "init_base_params",
    "trainable_base_params",
    "with_trainable_base_params",
]
