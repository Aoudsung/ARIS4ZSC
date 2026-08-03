"""Deterministic summaries for the K=4 categorical protocol posterior.

METHOD_SPEC §1.4: ``gaussian_summary``/``prior_gaussian_summary`` are replaced
by ``mixture_summary``/``prior_mixture_summary``.  The protocol state is
``c_t = sum_k pi_{t,k} m_k`` with component embeddings ``m_k ∈ R^16``
(METHOD_SPEC §1.1); the prior context uses the uniform posterior.
"""

from __future__ import annotations

from typing import Any

from .protocol_encoder import PROTOCOL_COMPONENTS


def mixture_summary(protocol_probabilities: Any, component_embeddings: Any) -> Any:
    """c = Σ_k π_k m_k.

    ``protocol_probabilities`` has shape (..., K) and ``component_embeddings``
    has shape (K, component_dim).
    """

    import jax.numpy as jnp

    probs = jnp.asarray(protocol_probabilities, dtype=jnp.float32)
    embeddings = jnp.asarray(component_embeddings, dtype=jnp.float32)
    if probs.shape[-1] != embeddings.shape[0]:
        raise ValueError("Protocol posterior and component count differ.")
    return jnp.einsum("...k,kd->...d", probs, embeddings)


def prior_mixture_summary(
    reference: Any,
    component_embeddings: Any,
    component_count: int = PROTOCOL_COMPONENTS,
) -> Any:
    """Prior context: uniform posterior over the K regimes (METHOD_SPEC §2.1)."""

    import jax.numpy as jnp

    embeddings = jnp.asarray(component_embeddings, dtype=jnp.float32)
    prefix = jnp.asarray(reference).shape[:-1]
    uniform = jnp.full(
        prefix + (int(component_count),),
        1.0 / float(component_count),
        dtype=jnp.float32,
    )
    return mixture_summary(uniform, embeddings)


def prior_capability(prefix_shape: tuple[int, ...]) -> Any:
    """Prior capability context is the zero vector (METHOD_SPEC §1.2)."""

    import jax.numpy as jnp

    return jnp.zeros(tuple(prefix_shape), dtype=jnp.float32)


__all__ = [
    "mixture_summary",
    "prior_capability",
    "prior_mixture_summary",
]
