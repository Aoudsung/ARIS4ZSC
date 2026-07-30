"""One shared belief-conditioned dueling critic."""

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
            belief_embedding: Any,
        ) -> tuple[Any, Any]:
            task = jnp.asarray(task_features, dtype=jnp.float32)
            belief = jnp.asarray(belief_embedding, dtype=jnp.float32)
            if task.shape[:-1] != belief.shape[:-1]:
                raise ValueError("Critic task and belief batch axes differ.")
            joined = jnp.concatenate((task, belief), axis=-1)
            shared = nn.tanh(
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(jnp.sqrt(2.0)),
                    bias_init=zeros,
                    name="critic_hidden_0",
                )(joined)
            )
            shared = nn.tanh(
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(jnp.sqrt(2.0)),
                    bias_init=zeros,
                    name="critic_hidden_1",
                )(shared)
            )
            state_value = nn.Dense(
                1,
                kernel_init=orthogonal(1.0),
                bias_init=zeros,
                name="state_value",
            )(shared)[..., 0]
            raw_advantage = nn.Dense(
                self.action_count,
                kernel_init=orthogonal(0.01),
                bias_init=zeros,
                name="action_advantage",
            )(shared)
            action_values = state_value[..., None] + raw_advantage - jnp.mean(
                raw_advantage, axis=-1, keepdims=True
            )
            return state_value, action_values

    _CRITIC = UniversalDuelingCritic
    return UniversalDuelingCritic


__all__ = ["universal_critic_class"]
