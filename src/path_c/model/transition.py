"""Partner-conditioned one-step transition and response predictor."""

from __future__ import annotations

from typing import Any


def transition_head_class() -> Any:
    import flax.linen as nn
    import jax
    import jax.numpy as jnp

    class PrototypeTransitionHeads(nn.Module):
        num_prototypes: int
        hidden_dim: int
        action_count: int
        response_count: int
        next_feature_dim: int
        activation: str = "relu"

        @nn.compact
        def __call__(self, features: Any) -> tuple[Any, Any, Any, Any]:
            activation = nn.relu if self.activation == "relu" else nn.tanh
            all_response_logits = []
            all_rewards = []
            all_next_features = []
            for prototype in range(self.num_prototypes):
                prototype_response = []
                prototype_reward = []
                prototype_next = []
                for action in range(self.action_count):
                    one_hot = jax.nn.one_hot(
                        jnp.full(features.shape[:-1], action, dtype=jnp.int32),
                        self.action_count,
                        dtype=features.dtype,
                    )
                    joined = jnp.concatenate((features, one_hot), axis=-1)
                    hidden = activation(
                        nn.Dense(self.hidden_dim, name=f"transition_{prototype}_{action}_hidden")(joined)
                    )
                    prototype_response.append(
                        nn.Dense(self.response_count, name=f"transition_{prototype}_{action}_response")(hidden)
                    )
                    prototype_reward.append(
                        nn.Dense(1, name=f"transition_{prototype}_{action}_reward")(hidden)[..., 0]
                    )
                    flattened = nn.Dense(
                        self.response_count * self.next_feature_dim,
                        name=f"transition_{prototype}_{action}_next",
                    )(hidden)
                    prototype_next.append(
                        flattened.reshape((*flattened.shape[:-1], self.response_count, self.next_feature_dim))
                    )
                all_response_logits.append(jnp.stack(prototype_response, axis=-2))
                all_rewards.append(jnp.stack(prototype_reward, axis=-1))
                all_next_features.append(jnp.stack(prototype_next, axis=-3))
            response_logits = jnp.stack(all_response_logits, axis=-3)
            rewards = jnp.stack(all_rewards, axis=-2)
            next_features = jnp.stack(all_next_features, axis=-4)
            return response_logits, jax.nn.softmax(response_logits, axis=-1), rewards, next_features

    return PrototypeTransitionHeads
