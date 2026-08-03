"""Deployable task-state encoder for DELTA-ZSC."""

from __future__ import annotations

from typing import Any

_TASK_CELL: Any | None = None
_TASK_SCAN: Any | None = None


def task_encoder_classes() -> tuple[Any, Any]:
    global _TASK_CELL, _TASK_SCAN
    if _TASK_CELL is not None and _TASK_SCAN is not None:
        return _TASK_CELL, _TASK_SCAN

    import flax.linen as nn
    import jax.numpy as jnp
    from flax.linen.initializers import orthogonal, zeros

    class TaskEncoderCell(nn.Module):
        hidden_dim: int
        action_count: int
        action_embedding_dim: int

        @nn.compact
        def __call__(
            self,
            carry: Any,
            inputs: tuple[Any, Any, Any],
        ) -> tuple[Any, Any]:
            observation, previous_action, episode_start = inputs
            obs = jnp.asarray(observation, dtype=jnp.float32)
            action = jnp.asarray(previous_action, dtype=jnp.int32)
            start = jnp.asarray(episode_start, dtype=jnp.bool_)
            if action.shape != start.shape:
                raise ValueError("Task encoder scalar inputs must share batch axes.")
            if obs.shape[:-3] != action.shape:
                raise ValueError("Observation batch axes do not match task inputs.")

            sentinel = jnp.where(start, self.action_count, action)
            action_embedding = nn.Embed(
                num_embeddings=self.action_count + 1,
                features=self.action_embedding_dim,
                embedding_init=nn.initializers.normal(0.02),
                name="previous_action_embedding",
            )(sentinel)
            # This is the exact visual trunk used by the locked Official RNN:
            # 128x1x1, 128x1x1, 8x1x1, 16x3x3, 32x3x3, 32x3x3,
            # flatten, Dense-128, ReLU, LayerNorm, GRU-128.  DELTA-specific
            # DELTA's belief and continuous low-rank actor modulation are
            # attached after this shared public-protocol backbone.
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
            encoded = encoded.reshape(action.shape + (-1,))
            encoded = nn.relu(
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(jnp.sqrt(2.0)),
                    bias_init=zeros,
                    name="official_dense",
                )(encoded)
            )
            encoded = encoded + nn.Dense(
                self.hidden_dim,
                kernel_init=zeros,
                bias_init=zeros,
                name="action_projection",
            )(action_embedding)
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


__all__ = ["initial_task_carry", "task_encoder_classes"]
