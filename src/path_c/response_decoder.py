"""Structured decoder for deployable, visible-partner response evidence."""

from __future__ import annotations

from typing import Any

_DECODER: Any | None = None


def response_decoder_class() -> Any:
    global _DECODER
    if _DECODER is not None:
        return _DECODER

    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    from flax.linen.initializers import orthogonal, zeros
    from .response_targets import (
        PARTNER_DIRECTION_CLASSES,
        PARTNER_INVENTORY_FACTOR_CLASSES,
        PARTNER_POSITION_CLASSES,
    )

    class ResponseDecoder(nn.Module):
        action_count: int
        hidden_dim: int
        action_embedding_dim: int
        inventory_factor_count: int

        @nn.compact
        def __call__(
            self,
            task_features: Any,
            belief_embedding: Any,
            actions: Any,
        ) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
            task = jnp.asarray(task_features, dtype=jnp.float32)
            belief = jnp.asarray(belief_embedding, dtype=jnp.float32)
            action = jnp.asarray(actions, dtype=jnp.int32)
            if task.shape[:-1] != action.shape or belief.shape[:-1] != action.shape:
                raise ValueError("Response decoder batch axes differ.")
            action_embedding = nn.Embed(
                num_embeddings=self.action_count,
                features=self.action_embedding_dim,
                embedding_init=nn.initializers.normal(0.02),
                name="action_embedding",
            )(action)
            joined = jnp.concatenate((task, belief, action_embedding), axis=-1)
            hidden = nn.tanh(
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(jnp.sqrt(2.0)),
                    bias_init=zeros,
                    name="response_hidden_0",
                )(joined)
            )
            hidden = nn.tanh(
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(jnp.sqrt(2.0)),
                    bias_init=zeros,
                    name="response_hidden_1",
                )(hidden)
            )
            visibility_logit = nn.Dense(
                1,
                kernel_init=orthogonal(0.01),
                bias_init=zeros,
                name="partner_visibility_logit",
            )(hidden)[..., 0]
            relative_position_logits = nn.Dense(
                PARTNER_POSITION_CLASSES,
                kernel_init=orthogonal(0.01),
                bias_init=zeros,
                name="partner_relative_position_logits",
            )(hidden)
            direction_logits = nn.Dense(
                PARTNER_DIRECTION_CLASSES,
                kernel_init=orthogonal(0.01),
                bias_init=zeros,
                name="partner_direction_logits",
            )(hidden)
            inventory_logits = nn.Dense(
                self.inventory_factor_count * PARTNER_INVENTORY_FACTOR_CLASSES,
                kernel_init=orthogonal(0.01),
                bias_init=zeros,
                name="partner_inventory_logits",
            )(hidden).reshape(
                hidden.shape[:-1]
                + (self.inventory_factor_count, PARTNER_INVENTORY_FACTOR_CLASSES)
            )
            interaction_change_logit = nn.Dense(
                1,
                kernel_init=orthogonal(0.01),
                bias_init=zeros,
                name="partner_interaction_change_logit",
            )(hidden)[..., 0]
            diagnostic_reward_mean = nn.Dense(
                1,
                kernel_init=orthogonal(0.01),
                bias_init=zeros,
                name="diagnostic_reward_mean",
            )(hidden)[..., 0]
            done_logit = nn.Dense(
                1,
                kernel_init=orthogonal(0.01),
                bias_init=zeros,
                name="diagnostic_done_logit",
            )(hidden)[..., 0]
            return (
                visibility_logit,
                relative_position_logits,
                direction_logits,
                inventory_logits,
                interaction_change_logit,
                diagnostic_reward_mean,
                done_logit,
            )

    _DECODER = ResponseDecoder
    return ResponseDecoder


def bernoulli_logit_loss(target: Any, logit: Any) -> Any:
    import jax
    import jax.numpy as jnp

    label = jnp.asarray(target, dtype=jnp.float32)
    return jnp.maximum(logit, 0.0) - logit * label + jnp.log1p(
        jnp.exp(-jnp.abs(logit))
    )


__all__ = [
    "bernoulli_logit_loss",
    "response_decoder_class",
]
