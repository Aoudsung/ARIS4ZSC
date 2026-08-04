"""Task-state encoders for the DEPI development matrix.

For B1/B2 the recurrent task pathway receives the current observation after
all other-agent semantic planes have been zeroed.  Consequently neither an
explicit history carrier nor a sequence of visible partner states can enter
its GRU.  B0 deliberately keeps the complete observation as the full-history
capacity control.
"""

from __future__ import annotations

from typing import Any

_TASK_CELL: Any | None = None
_TASK_SCAN: Any | None = None


def official_partner_channel_indexes(channel_count: int) -> tuple[int, ...]:
    """Return the pinned Official channels belonging to the other agent.

    At commit ``5ce1707`` each agent block is
    ``position[1], direction[4], inventory[num_ingredients+2]`` and the full
    observation has ``27 + 4*num_ingredients`` channels.  Failing this exact
    relation is safer than silently leaking partner history into the task GRU.
    """

    channels = int(channel_count)
    remainder = channels - 27
    if remainder < 0 or remainder % 4:
        raise ValueError("Observation channels do not match the pinned Official layout.")
    ingredient_count = remainder // 4
    if ingredient_count <= 0:
        raise ValueError("The pinned Official layout requires at least one ingredient.")
    agent_block = ingredient_count + 7
    return tuple(range(agent_block, 2 * agent_block))


def task_only_observation(observation: Any) -> Any:
    """Remove all current partner planes before the recurrent task pathway."""

    import jax.numpy as jnp

    obs = jnp.asarray(observation, dtype=jnp.float32)
    if obs.ndim < 3:
        raise ValueError("Official observations require spatial and channel axes.")
    partner_channels = jnp.asarray(
        official_partner_channel_indexes(obs.shape[-1]), dtype=jnp.int32
    )
    return obs.at[..., partner_channels].set(0.0)


def instantaneous_partner_observation(observation: Any) -> Any:
    """Keep only current other-agent planes for the non-recurrent branch."""

    import jax.numpy as jnp

    obs = jnp.asarray(observation, dtype=jnp.float32)
    partner_channels = official_partner_channel_indexes(obs.shape[-1])
    return obs[..., list(partner_channels)]


def task_encoder_classes() -> tuple[Any, Any]:
    global _TASK_CELL, _TASK_SCAN
    if _TASK_CELL is not None and _TASK_SCAN is not None:
        return _TASK_CELL, _TASK_SCAN

    import flax.linen as nn
    import jax.numpy as jnp
    from flax.linen.initializers import orthogonal, zeros

    class TaskEncoderCell(nn.Module):
        hidden_dim: int
        mask_partner_history: bool = True

        @nn.compact
        def __call__(
            self,
            carry: Any,
            inputs: tuple[Any, Any],
        ) -> tuple[Any, Any]:
            observation, episode_start = inputs
            obs = (
                task_only_observation(observation)
                if self.mask_partner_history
                else jnp.asarray(observation, dtype=jnp.float32)
            )
            start = jnp.asarray(episode_start, dtype=jnp.bool_)
            batch_axes = obs.shape[:-3]
            if start.shape != batch_axes:
                raise ValueError("Task encoder scalar inputs must share batch axes.")

            # This is the exact visual trunk used by the locked Official RNN:
            # 128x1x1, 128x1x1, 8x1x1, 16x3x3, 32x3x3, 32x3x3,
            # flatten, Dense-128, ReLU, LayerNorm, GRU-128.  DEPI attaches the
            # capability/protocol pathways after this shared public backbone;
            # the task pathway itself reads only the current observation.
            encoded = obs
            for index, (features, kernel) in enumerate(
                ((128, (1, 1)), (128, (1, 1)), (8, (1, 1)),
                 (16, (3, 3)), (32, (3, 3)), (32, (3, 3)))
            ):
                encoded = nn.relu(
                    nn.Conv(
                        features=features,
                        kernel_size=kernel,
                        kernel_init=orthogonal(jnp.sqrt(2.0)),
                        bias_init=zeros,
                        name=f"official_conv_{index}",
                    )(encoded)
                )
            encoded = encoded.reshape(batch_axes + (-1,))
            encoded = nn.relu(
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(jnp.sqrt(2.0)),
                    bias_init=zeros,
                    name="official_dense",
                )(encoded)
            )
            encoded = nn.LayerNorm(name="task_layer_norm")(encoded)
            carry = jnp.where(start[..., None], jnp.zeros_like(carry), carry)
            next_carry, feature = nn.GRUCell(
                features=self.hidden_dim,
                name="task_gru",
            )(carry, encoded)
            return next_carry, feature

    ScannedTaskEncoder = nn.scan(
        TaskEncoderCell,
        variable_broadcast="params",
        split_rngs={"params": False},
        in_axes=0,
        out_axes=0,
    )
    _TASK_CELL = TaskEncoderCell
    _TASK_SCAN = ScannedTaskEncoder
    return TaskEncoderCell, ScannedTaskEncoder


def initial_task_carry(batch_size: int, hidden_dim: int) -> Any:
    import jax.numpy as jnp

    return jnp.zeros((int(batch_size), int(hidden_dim)), dtype=jnp.float32)


__all__ = [
    "initial_task_carry",
    "instantaneous_partner_observation",
    "official_partner_channel_indexes",
    "task_encoder_classes",
    "task_only_observation",
]
