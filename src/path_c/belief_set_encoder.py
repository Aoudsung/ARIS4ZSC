"""Deterministic summaries and reparameterized particles for a single Gaussian."""

from __future__ import annotations

from typing import Any


def gaussian_summary(mean: Any, log_standard_deviation: Any, uncertainty: Any) -> Any:
    import jax.numpy as jnp

    mu = jnp.asarray(mean, dtype=jnp.float32)
    log_std = jnp.asarray(log_standard_deviation, dtype=jnp.float32)
    scalar = jnp.asarray(uncertainty, dtype=jnp.float32)
    if mu.shape != log_std.shape or scalar.shape != mu.shape[:-1]:
        raise ValueError("Gaussian belief summary shapes are incompatible.")
    return jnp.concatenate((mu, log_std, scalar[..., None]), axis=-1)


def prior_gaussian_summary(reference: Any, latent_dim: int) -> Any:
    import jax.numpy as jnp

    prefix = jnp.asarray(reference).shape[:-1]
    zeros = jnp.zeros(prefix + (2 * int(latent_dim),), dtype=jnp.float32)
    # With the frozen [-5, 2] log-standard-deviation range, the N(0, I)
    # prior has normalized uncertainty (0 - (-5)) / 7.
    uncertainty = jnp.full(prefix + (1,), 5.0 / 7.0, dtype=jnp.float32)
    return jnp.concatenate((zeros, uncertainty), axis=-1)


def gaussian_samples(
    key: Any,
    *,
    mean: Any,
    log_standard_deviation: Any,
    sample_count: int,
) -> tuple[Any, Any]:
    import jax
    import jax.numpy as jnp

    if int(sample_count) <= 0:
        raise ValueError("sample_count must be positive.")
    mu = jnp.asarray(mean, dtype=jnp.float32)
    log_std = jnp.asarray(log_standard_deviation, dtype=jnp.float32)
    if mu.shape != log_std.shape:
        raise ValueError("Gaussian mean and log standard deviation shapes differ.")
    noise = jax.random.normal(
        key, mu.shape[:-1] + (int(sample_count), mu.shape[-1]), dtype=jnp.float32
    )
    samples = mu[..., None, :] + jnp.exp(log_std[..., None, :]) * noise
    weights = jnp.full(samples.shape[:-1], 1.0 / float(sample_count), dtype=jnp.float32)
    return samples, weights


def gaussian_kl_standard_normal(
    mean: Any,
    log_standard_deviation: Any,
    *,
    free_bits_per_dimension: float,
) -> Any:
    import jax.numpy as jnp

    mu = jnp.asarray(mean, dtype=jnp.float32)
    log_std = jnp.asarray(log_standard_deviation, dtype=jnp.float32)
    per_dimension = 0.5 * (
        jnp.square(mu) + jnp.exp(2.0 * log_std) - 1.0 - 2.0 * log_std
    )
    allowance = float(free_bits_per_dimension)
    return jnp.mean(jnp.sum(jnp.maximum(per_dimension - allowance, 0.0), axis=-1))


# Compatibility aliases are deliberately mathematical only; active V6 code
# does not expose mixture components.
stratified_gaussian_samples = gaussian_samples


__all__ = [
    "gaussian_kl_standard_normal",
    "gaussian_samples",
    "gaussian_summary",
    "prior_gaussian_summary",
    "stratified_gaussian_samples",
]
