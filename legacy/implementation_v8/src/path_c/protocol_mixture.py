"""Deterministic summaries of the exchangeable categorical protocol state."""

from __future__ import annotations

from typing import Any

from .protocol_encoder import PROTOCOL_COMPONENTS


def mixture_summary(protocol_probabilities: Any, component_embeddings: Any) -> Any:
    """Return ``c = sum_k pi_k m_k`` while preserving arbitrary prefix axes."""

    import jax.numpy as jnp

    probabilities = jnp.asarray(protocol_probabilities, dtype=jnp.float32)
    embeddings = jnp.asarray(component_embeddings, dtype=jnp.float32)
    if probabilities.ndim < 1 or embeddings.ndim != 2:
        raise ValueError("Protocol probabilities need a class axis and embeddings a matrix.")
    if probabilities.shape[-1] != embeddings.shape[0]:
        raise ValueError("Protocol posterior and component count differ.")
    return jnp.einsum("...k,kd->...d", probabilities, embeddings)


def prior_mixture_summary(
    reference: Any,
    component_embeddings: Any,
    component_count: int = PROTOCOL_COMPONENTS,
) -> Any:
    """Return the mixture summary under the registered uniform prior."""

    import jax.numpy as jnp

    embeddings = jnp.asarray(component_embeddings, dtype=jnp.float32)
    prefix = jnp.asarray(reference).shape[:-1]
    uniform = jnp.full(
        prefix + (int(component_count),),
        1.0 / float(component_count),
        dtype=jnp.float32,
    )
    return mixture_summary(uniform, embeddings)


__all__ = ["mixture_summary", "prior_mixture_summary"]
