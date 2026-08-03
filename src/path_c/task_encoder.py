"""Deployable task-state encoder for DEPI.

METHOD_SPEC §1.4: the task pathway input shrinks to ``o_t`` only.  The
previous_action_embedding and action_projection wiring is deleted so that the
task GRU structurally never touches any partner-history carrier (METHOD_SPEC
§1.2 input-layer isolation).  The Official CNN trunk weights are untouched.
"""

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

        @nn.compact
        def __call__(
            self,
            carry: Any,
            inputs: tuple[Any, Any],
        ) -> tuple[Any, Any]:
            observation, episode_start = inputs
            obs = jnp.asarray(observation, dtype=jnp.float32)
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


__all__ = ["initial_task_carry", "task_encoder_classes"]
