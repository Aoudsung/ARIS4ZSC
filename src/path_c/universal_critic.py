"""One shaped-return value and two independent raw-return Q heads."""

from __future__ import annotations

from typing import Any

_CRITIC: Any | None = None


def universal_critic_class() -> Any:
    global _CRITIC
    if _CRITIC is not None:
        return _CRITIC

    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    from flax.linen.initializers import orthogonal, zeros

    class UniversalDuelingCritic(nn.Module):
        action_count: int
        hidden_dim: int

        @nn.compact
        def __call__(
            self,
            task_features: Any,
            belief_embedding: Any,
            shaped_belief_embedding: Any | None = None,
        ) -> tuple[Any, Any, Any]:
            task = jnp.asarray(task_features, dtype=jnp.float32)
            belief = jnp.asarray(belief_embedding, dtype=jnp.float32)
            shaped_belief = (
                belief
                if shaped_belief_embedding is None
                else jnp.asarray(shaped_belief_embedding, dtype=jnp.float32)
            )
            if task.shape[:-1] != belief.shape[:-1]:
                raise ValueError("Critic task and belief batch axes differ.")
            if shaped_belief.shape[:-1] != task.shape[:-1]:
                raise ValueError("Shaped-value belief batch axes differ.")
            shaped_joined = jnp.concatenate((task, shaped_belief), axis=-1)
            raw_joined = jnp.concatenate((task, belief), axis=-1)
            shaped_hidden = nn.tanh(
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(jnp.sqrt(2.0)),
                    bias_init=zeros,
                    name="shaped_hidden_0",
                )(shaped_joined)
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
                )(raw_joined)
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
                )(raw_joined)
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
