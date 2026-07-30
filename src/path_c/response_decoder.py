"""Shared decoder for deployable partner-conditioned response evidence."""

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

    class ResponseDecoder(nn.Module):
        observation_size: int
        action_count: int
        hidden_dim: int
        action_embedding_dim: int
        log_std_minimum: float
        log_std_maximum: float

        @nn.compact
        def __call__(
            self,
            task_features: Any,
            belief_embedding: Any,
            actions: Any,
        ) -> tuple[Any, Any, Any, Any, Any]:
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
            observation_delta_mean = nn.Dense(
                self.observation_size,
                kernel_init=orthogonal(0.01),
                bias_init=zeros,
                name="observation_delta_mean",
            )(hidden)
            observation_delta_log_std = jnp.clip(
                nn.Dense(
                    self.observation_size,
                    kernel_init=zeros,
                    bias_init=zeros,
                    name="observation_delta_log_std",
                )(hidden),
                self.log_std_minimum,
                self.log_std_maximum,
            )
            reward_mean = nn.Dense(
                1,
                kernel_init=orthogonal(0.01),
                bias_init=zeros,
                name="reward_mean",
            )(hidden)[..., 0]
            reward_log_std = jnp.clip(
                nn.Dense(
                    1,
                    kernel_init=zeros,
                    bias_init=zeros,
                    name="reward_log_std",
                )(hidden)[..., 0],
                self.log_std_minimum,
                self.log_std_maximum,
            )
            done_logit = nn.Dense(
                1,
                kernel_init=orthogonal(0.01),
                bias_init=zeros,
                name="done_logit",
            )(hidden)[..., 0]
            return (
                observation_delta_mean,
                observation_delta_log_std,
                reward_mean,
                reward_log_std,
                done_logit,
            )

    _DECODER = ResponseDecoder
    return ResponseDecoder


def gaussian_negative_log_likelihood(
    target: Any,
    mean: Any,
    log_standard_deviation: Any,
) -> Any:
    import jax.numpy as jnp

    value = jnp.asarray(target, dtype=jnp.float32)
    mu = jnp.asarray(mean, dtype=jnp.float32)
    log_std = jnp.asarray(log_standard_deviation, dtype=jnp.float32)
    normalized = (value - mu) * jnp.exp(-log_std)
    return 0.5 * jnp.square(normalized) + log_std


def bernoulli_logit_loss(target: Any, logit: Any) -> Any:
    import jax
    import jax.numpy as jnp

    label = jnp.asarray(target, dtype=jnp.float32)
    return jnp.maximum(logit, 0.0) - logit * label + jnp.log1p(
        jnp.exp(-jnp.abs(logit))
    )


__all__ = [
    "bernoulli_logit_loss",
    "gaussian_negative_log_likelihood",
    "response_decoder_class",
]
