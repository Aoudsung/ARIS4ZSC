"""Single recurrent actor and training-only scalar value branch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .nn import (
    conv,
    gru_step,
    init_conv,
    init_gru,
    init_layer_norm,
    init_orthogonal,
    layer_norm,
    linear,
)


OFFICIAL_CONV_STACK = (
    ((1, 1), 128),
    ((1, 1), 128),
    ((1, 1), 8),
    ((3, 3), 16),
    ((3, 3), 32),
    ((3, 3), 32),
)
"""The six convolutional layers used by the Official actor."""


@dataclass(frozen=True, slots=True)
class CetrModel:
    """Shape-bound recurrent actor with a scalar value baseline.

    ``config.model`` supplies ``task_hidden_dim`` and ``task_embedding_dim``.
    The parameter tree is flat and contains one convolutional encoder, one GRU,
    one action head, and one scalar value branch.
    """

    config: Any
    observation_shape: tuple[int, ...]
    action_count: int

    def init_parameters(self, key: Any) -> dict[str, Any]:
        """Initialise the complete flat actor/value parameter tree."""

        import jax

        keys = jax.random.split(key, 10 + len(OFFICIAL_CONV_STACK))
        channels = int(self.observation_shape[-1])
        convolutions = []
        for index, (kernel_size, features) in enumerate(OFFICIAL_CONV_STACK):
            convolutions.append(
                init_conv(
                    keys[index],
                    kernel_size,
                    channels,
                    features,
                    scale=2.0**0.5,
                )
            )
            channels = int(features)

        task_hidden_dim = int(self.config.model.task_hidden_dim)
        task_embedding_dim = int(self.config.model.task_embedding_dim)
        spatial = (
            int(self.observation_shape[0])
            * int(self.observation_shape[1])
            * channels
        )
        offset = len(OFFICIAL_CONV_STACK)
        return {
            "task_conv": tuple(convolutions),
            "task_dense": init_orthogonal(
                keys[offset], spatial, task_embedding_dim, scale=2.0**0.5
            ),
            "task_norm": init_layer_norm(task_embedding_dim),
            "task_gru": init_gru(
                keys[offset + 1], task_embedding_dim, task_hidden_dim
            ),
            "actor_trunk": init_orthogonal(
                keys[offset + 2], task_hidden_dim, task_hidden_dim, scale=2.0**0.5
            ),
            "actor": init_orthogonal(
                keys[offset + 3], task_hidden_dim, int(self.action_count), scale=0.01
            ),
            "value_trunk": init_orthogonal(
                keys[offset + 6], task_hidden_dim, task_hidden_dim, scale=2.0**0.5
            ),
            "value": init_orthogonal(
                keys[offset + 7], task_hidden_dim, 1, scale=1.0
            ),
        }

    def initial_carry(self, batch_size: int) -> Any:
        """Return a zero carry for a batch of independent episodes."""

        import jax.numpy as jnp

        return jnp.zeros(
            (int(batch_size), int(self.config.model.task_hidden_dim)),
            dtype=jnp.float32,
        )

    def step(
        self,
        params: dict[str, Any],
        carry: Any,
        observation: Any,
        episode_start: Any,
    ) -> tuple[Any, Any, Any]:
        """Advance the recurrent actor and scalar value branch once."""

        return _encode_and_step(params, carry, observation, episode_start)

    def sequence(
        self,
        params: dict[str, Any],
        initial_carry: Any,
        observations: Any,
        episode_starts: Any,
    ) -> tuple[Any, Any]:
        """Evaluate a time-major batch with one recurrent scan."""

        return _model_sequence(
            params, initial_carry, observations, episode_starts
        )


def encode_task_frame(params: dict[str, Any], frame: Any) -> Any:
    """Encode an NHWC frame with the Official convolutional stack."""

    import jax
    import jax.numpy as jnp

    hidden = jnp.asarray(frame, dtype=jnp.float32)
    for layer in params["task_conv"]:
        hidden = jax.nn.relu(conv(layer, hidden))
    flat = hidden.reshape(hidden.shape[:-3] + (-1,))
    embedding = jax.nn.relu(linear(params["task_dense"], flat))
    return layer_norm(params["task_norm"], embedding)


def actor_parameters(params: dict[str, Any]) -> dict[str, Any]:
    """Select the parameter subtree used by deployed action inference."""

    names = (
        "task_conv",
        "task_dense",
        "task_norm",
        "task_gru",
        "actor_trunk",
        "actor",
    )
    return {name: params[name] for name in names}


def actor_step(
    actor_params: dict[str, Any],
    carry: Any,
    observation: Any,
    episode_start: Any,
) -> tuple[Any, Any]:
    """Advance the deployed actor without evaluating the value branch."""

    import jax

    embedding = encode_task_frame(actor_params, observation)
    next_carry = _step_recurrent(actor_params, carry, embedding, episode_start)
    actor_hidden = jax.nn.relu(linear(actor_params["actor_trunk"], next_carry))
    logits = linear(actor_params["actor"], actor_hidden)
    return next_carry, logits


def _actor_value(params: dict[str, Any], carry: Any) -> tuple[Any, Any]:
    import jax

    actor_hidden = jax.nn.relu(linear(params["actor_trunk"], carry))
    logits = linear(params["actor"], actor_hidden)
    value_hidden = jax.nn.relu(linear(params["value_trunk"], carry))
    value = linear(params["value"], value_hidden)[..., 0]
    return logits, value


def _reset_carry(carry: Any, episode_start: Any) -> Any:
    import jax.numpy as jnp

    return jnp.where(episode_start[..., None], jnp.zeros_like(carry), carry)


def _step_recurrent(
    params: dict[str, Any], carry: Any, embedding: Any, episode_start: Any
) -> Any:
    reset = _reset_carry(carry, episode_start)
    return gru_step(params["task_gru"], reset, embedding)


def _encode_and_step(
    params: dict[str, Any], carry: Any, observation: Any, episode_start: Any
) -> tuple[Any, Any, Any]:
    embedding = encode_task_frame(params, observation)
    next_carry = _step_recurrent(params, carry, embedding, episode_start)
    logits, value = _actor_value(params, next_carry)
    return next_carry, logits, value


def _model_sequence(
    params: dict[str, Any],
    initial_carry: Any,
    observations: Any,
    episode_starts: Any,
) -> tuple[Any, Any]:
    import jax
    import jax.numpy as jnp

    observation = jnp.asarray(observations, dtype=jnp.float32)
    task_embeddings = encode_task_frame(params, observation)

    def recurrent_step(carry: Any, values: tuple[Any, Any]):
        embedding, episode_start = values
        next_carry = _step_recurrent(params, carry, embedding, episode_start)
        return next_carry, next_carry

    _, task_features = jax.lax.scan(
        recurrent_step,
        initial_carry,
        (task_embeddings, episode_starts),
    )
    logits, values = _actor_value(params, task_features)
    return logits, values


__all__ = [
    "CetrModel",
    "OFFICIAL_CONV_STACK",
    "actor_parameters",
    "actor_step",
    "encode_task_frame",
]
