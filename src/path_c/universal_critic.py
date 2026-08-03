"""Dueling critic for DEPI (METHOD_SPEC §1.3/§1.4).

``UniversalDuelingCritic(task_features, context)`` outputs (V, Q1, Q2) with
``context = concat(u, c_t)`` (32 dimensions).  The ``shaped_belief_embedding``
special case is removed: every head conditions on the same
``[x_t; u; c_t]`` context.  Head names are retained because the parameter
ownership table (METHOD_SPEC §3.4) routes ``shaped_*`` gradients to the PPO
loss and ``raw_q*`` gradients to the signature loss.
"""

from __future__ import annotations

from typing import Any

_CRITIC: Any | None = None


def universal_critic_class() -> Any:
    global _CRITIC
    if _CRITIC is not None:
        return _CRITIC

    import flax.linen as nn
    import jax.numpy as jnp
    from flax.linen.initializers import orthogonal, zeros

    class UniversalDuelingCritic(nn.Module):
        action_count: int
        hidden_dim: int

        @nn.compact
        def __call__(
            self,
            task_features: Any,
            context: Any,
        ) -> tuple[Any, Any, Any]:
            task = jnp.asarray(task_features, dtype=jnp.float32)
            ctx = jnp.asarray(context, dtype=jnp.float32)
            if task.shape[:-1] != ctx.shape[:-1]:
                raise ValueError("Critic task and context batch axes differ.")
            joined = jnp.concatenate((task, ctx), axis=-1)
            shaped_hidden = nn.tanh(
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(jnp.sqrt(2.0)),
                    bias_init=zeros,
                    name="shaped_hidden_0",
                )(joined)
            )
            shaped_hidden = nn.tanh(
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(jnp.sqrt(2.0)),
                    bias_init=zeros,
                    name="shaped_hidden_1",
                )(shaped_hidden)
            )
            state_value = nn.Dense(
                1,
                kernel_init=orthogonal(1.0),
                bias_init=zeros,
                name="shaped_state_value",
            )(shaped_hidden)[..., 0]

            raw_q1_hidden = nn.tanh(
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(jnp.sqrt(2.0)),
                    bias_init=zeros,
                    name="raw_q1_hidden_0",
                )(joined)
            )
            raw_q1_hidden = nn.tanh(
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(jnp.sqrt(2.0)),
                    bias_init=zeros,
                    name="raw_q1_hidden_1",
                )(raw_q1_hidden)
            )
            raw_q1 = nn.Dense(
                self.action_count,
                kernel_init=orthogonal(0.01),
                bias_init=zeros,
                name="raw_q1_values",
            )(raw_q1_hidden)

            raw_q2_hidden = nn.tanh(
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(jnp.sqrt(2.0)),
                    bias_init=zeros,
                    name="raw_q2_hidden_0",
                )(joined)
            )
            raw_q2_hidden = nn.tanh(
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(jnp.sqrt(2.0)),
                    bias_init=zeros,
                    name="raw_q2_hidden_1",
                )(raw_q2_hidden)
            )
            raw_q2 = nn.Dense(
                self.action_count,
                kernel_init=orthogonal(0.01),
                bias_init=zeros,
                name="raw_q2_values",
            )(raw_q2_hidden)
            return state_value, raw_q1, raw_q2

    _CRITIC = UniversalDuelingCritic
    return UniversalDuelingCritic


__all__ = ["universal_critic_class"]
