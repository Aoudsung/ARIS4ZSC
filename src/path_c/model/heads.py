"""Shared and partner-conditioned Flax heads with fixed gradient routing."""

from __future__ import annotations

from typing import Any


def head_classes() -> tuple[Any, Any, Any]:
    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    from flax.linen.initializers import constant, orthogonal

    class SharedActorCritic(nn.Module):
        hidden_dim: int
        action_count: int
        activation: str = "relu"

        @nn.compact
        def __call__(self, features: Any) -> tuple[Any, Any]:
            activation = nn.relu if self.activation == "relu" else nn.tanh
            detached = jax.lax.stop_gradient(features)
            actor_hidden = nn.Dense(
                self.hidden_dim,
                kernel_init=orthogonal(2.0),
                bias_init=constant(0.0),
                name="actor_hidden",
            )(detached)
            actor_hidden = activation(actor_hidden)
            raw_logits = nn.Dense(
                self.action_count,
                kernel_init=orthogonal(0.01),
                bias_init=constant(0.0),
                name="actor_logits",
            )(actor_hidden)
            # The official interface exposes Distrax Categorical.logits,
            # which are normalized log probabilities rather than the raw
            # final Dense output.
            logits = jax.nn.log_softmax(raw_logits, axis=-1)
            critic_hidden = nn.Dense(
                self.hidden_dim,
                kernel_init=orthogonal(2.0),
                bias_init=constant(0.0),
                name="critic_hidden",
            )(features)
            critic_hidden = activation(critic_hidden)
            value = nn.Dense(
                1,
                kernel_init=orthogonal(1.0),
                bias_init=constant(0.0),
                name="critic_value",
            )(critic_hidden)[..., 0]
            return logits, value

    class PrototypeValueHeads(nn.Module):
        num_prototypes: int
        hidden_dim: int
        activation: str = "relu"

        @nn.compact
        def __call__(self, features: Any) -> Any:
            activation = nn.relu if self.activation == "relu" else nn.tanh
            outputs = []
            for index in range(self.num_prototypes):
                hidden = activation(
                    nn.Dense(self.hidden_dim, name=f"value_{index}_hidden")(features)
                )
                outputs.append(
                    nn.Dense(1, name=f"value_{index}_out")(hidden)[..., 0]
                )
            return jnp.stack(outputs, axis=-1)

    class PrototypeResponseHeads(nn.Module):
        num_prototypes: int
        hidden_dim: int
        action_count: int
        response_count: int
        activation: str = "relu"

        @nn.compact
        def __call__(self, features: Any) -> tuple[Any, Any]:
            activation = nn.relu if self.activation == "relu" else nn.tanh
            prototype_logits = []
            for prototype in range(self.num_prototypes):
                action_logits = []
                for action in range(self.action_count):
                    one_hot = jax.nn.one_hot(
                        jnp.full(features.shape[:-1], action, dtype=jnp.int32),
                        self.action_count,
                        dtype=features.dtype,
                    )
                    joined = jnp.concatenate((features, one_hot), axis=-1)
                    hidden = activation(
                        nn.Dense(
                            self.hidden_dim,
                            name=f"response_{prototype}_{action}_hidden",
                        )(joined)
                    )
                    action_logits.append(
                        nn.Dense(
                            self.response_count,
                            name=f"response_{prototype}_{action}_out",
                        )(hidden)
                    )
                prototype_logits.append(jnp.stack(action_logits, axis=-2))
            logits = jnp.stack(prototype_logits, axis=-3)
            return logits, jax.nn.softmax(logits, axis=-1)

    return SharedActorCritic, PrototypeValueHeads, PrototypeResponseHeads
