"""Permutation-invariant encoding and sampling for Gaussian-mixture beliefs."""

from __future__ import annotations

from typing import Any

_BELIEF_SET_ENCODER: Any | None = None


def normalized_mixture_weights(mixture_logits: Any) -> Any:
    import jax
    import jax.numpy as jnp

    logits = jnp.asarray(mixture_logits, dtype=jnp.float32)
    return jax.nn.softmax(logits, axis=-1)


def mixture_entropy(mixture_logits: Any, probability_floor: float = 1.0e-8) -> Any:
    import jax.numpy as jnp

    weights = normalized_mixture_weights(mixture_logits)
    return -jnp.sum(
        weights * jnp.log(jnp.maximum(weights, float(probability_floor))), axis=-1
    )


def mixture_moments(
    mixture_logits: Any,
    means: Any,
    log_variances: Any,
) -> tuple[Any, Any]:
    """Return exact first and diagonal second central moments."""

    import jax.numpy as jnp

    weights = normalized_mixture_weights(mixture_logits)
    mu = jnp.asarray(means, dtype=jnp.float32)
    log_var = jnp.asarray(log_variances, dtype=jnp.float32)
    if mu.shape != log_var.shape or mu.shape[-2] != weights.shape[-1]:
        raise ValueError("Gaussian-mixture parameter shapes are incompatible.")
    mean = jnp.sum(weights[..., :, None] * mu, axis=-2)
    second = jnp.sum(
        weights[..., :, None] * (jnp.exp(log_var) + jnp.square(mu)), axis=-2
    )
    variance = jnp.maximum(second - jnp.square(mean), 0.0)
    return mean, variance


def stratified_mixture_samples(
    key: Any,
    *,
    mixture_logits: Any,
    means: Any,
    log_variances: Any,
    sample_count: int,
) -> tuple[Any, Any]:
    """Reparameterized samples plus importance weights.

    Samples are drawn from the learned mixture.  The returned sample weights are
    uniform because component selection already follows the mixture weights.
    """

    import jax
    import jax.numpy as jnp

    if sample_count <= 0:
        raise ValueError("sample_count must be positive.")
    logits = jnp.asarray(mixture_logits, dtype=jnp.float32)
    mu = jnp.asarray(means, dtype=jnp.float32)
    log_var = jnp.asarray(log_variances, dtype=jnp.float32)
    component_key, noise_key = jax.random.split(key)
    components = jax.random.categorical(
        component_key,
        logits,
        axis=-1,
        shape=(int(sample_count),) + logits.shape[:-1],
    )
    expanded_mu = jnp.broadcast_to(mu, (int(sample_count),) + mu.shape)
    expanded_log_var = jnp.broadcast_to(
        log_var, (int(sample_count),) + log_var.shape
    )
    indexes = components[..., None, None]
    selected_mu = jnp.take_along_axis(expanded_mu, indexes, axis=-2)[..., 0, :]
    selected_log_var = jnp.take_along_axis(
        expanded_log_var, indexes, axis=-2
    )[..., 0, :]
    noise = jax.random.normal(noise_key, selected_mu.shape, dtype=jnp.float32)
    samples = selected_mu + jnp.exp(0.5 * selected_log_var) * noise
    samples = jnp.moveaxis(samples, 0, -2)
    weights = jnp.full(
        samples.shape[:-1], 1.0 / float(sample_count), dtype=jnp.float32
    )
    return samples, weights


def belief_set_encoder_class() -> Any:
    global _BELIEF_SET_ENCODER
    if _BELIEF_SET_ENCODER is not None:
        return _BELIEF_SET_ENCODER

    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    from flax.linen.initializers import orthogonal, zeros

    class BeliefSetEncoder(nn.Module):
        latent_dim: int
        hidden_dim: int
        output_dim: int

        @nn.compact
        def __call__(
            self,
            mixture_logits: Any,
            means: Any,
            log_variances: Any,
            support_score: Any,
        ) -> Any:
            weights = jax.nn.softmax(
                jnp.asarray(mixture_logits, dtype=jnp.float32), axis=-1
            )
            mu = jnp.asarray(means, dtype=jnp.float32)
            log_var = jnp.asarray(log_variances, dtype=jnp.float32)
            if mu.shape != log_var.shape:
                raise ValueError("Belief means and log variances must share shape.")
            if mu.shape[-1] != self.latent_dim:
                raise ValueError("Belief latent width differs from the encoder.")
            component_input = jnp.concatenate(
                (
                    mu,
                    log_var,
                    jnp.exp(log_var),
                    weights[..., :, None],
                ),
                axis=-1,
            )
            component_feature = nn.tanh(
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(1.0),
                    bias_init=zeros,
                    name="component_projection",
                )(component_input)
            )
            weighted = weights[..., :, None] * component_feature
            first = jnp.sum(weighted, axis=-2)
            second = jnp.sum(
                weights[..., :, None] * jnp.square(component_feature), axis=-2
            )
            entropy = -jnp.sum(
                weights * jnp.log(jnp.maximum(weights, 1.0e-8)),
                axis=-1,
                keepdims=True,
            )
            support = jnp.asarray(support_score, dtype=jnp.float32)[..., None]
            exact_mean, exact_variance = mixture_moments(
                mixture_logits, means, log_variances
            )
            summary = jnp.concatenate(
                (
                    first,
                    second,
                    exact_mean,
                    jnp.sqrt(jnp.maximum(exact_variance, 0.0) + 1.0e-8),
                    entropy,
                    support,
                ),
                axis=-1,
            )
            hidden = nn.tanh(
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(1.0),
                    bias_init=zeros,
                    name="summary_hidden",
                )(summary)
            )
            return nn.Dense(
                self.output_dim,
                kernel_init=orthogonal(1.0),
                bias_init=zeros,
                name="belief_embedding",
            )(hidden)

    _BELIEF_SET_ENCODER = BeliefSetEncoder
    return BeliefSetEncoder


__all__ = [
    "belief_set_encoder_class",
    "mixture_entropy",
    "mixture_moments",
    "normalized_mixture_weights",
    "stratified_mixture_samples",
]
