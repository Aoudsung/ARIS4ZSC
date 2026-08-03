"""Decision-equivalence geometry and best-response diversity."""

from __future__ import annotations

from typing import Any


def centered_action_values(values: Any) -> Any:
    import jax.numpy as jnp

    q = jnp.asarray(values, dtype=jnp.float32)
    return q - jnp.mean(q, axis=-1, keepdims=True)


def decision_distance(signature_a: Any, signature_b: Any) -> Any:
    import jax.numpy as jnp

    left = jnp.asarray(signature_a, dtype=jnp.float32)
    right = jnp.asarray(signature_b, dtype=jnp.float32)
    if left.shape != right.shape:
        raise ValueError("Decision signatures must share shape.")
    return jnp.sqrt(jnp.sum(jnp.square(left - right), axis=-1) + 1.0e-12)


def quotient_geometry_loss(
    latent_a: Any,
    latent_b: Any,
    empirical_distance: Any,
    *,
    equivalence_epsilon: float,
    separation_epsilon: float,
    margin: float,
    intermediate_scale: float | None = None,
    weights: Any | None = None,
) -> Any:
    """Three-region loss from the DELTA-ZSC specification."""

    import jax.numpy as jnp

    if equivalence_epsilon < 0.0 or separation_epsilon <= equivalence_epsilon:
        raise ValueError("Quotient distance thresholds are invalid.")
    if margin <= 0.0:
        raise ValueError("Quotient margin must be positive.")
    left = jnp.asarray(latent_a, dtype=jnp.float32)
    right = jnp.asarray(latent_b, dtype=jnp.float32)
    target = jnp.asarray(empirical_distance, dtype=jnp.float32)
    latent_distance = jnp.sqrt(
        jnp.sum(jnp.square(left - right), axis=-1) + 1.0e-12
    )
    scale = (
        float(intermediate_scale)
        if intermediate_scale is not None
        else float(margin) / float(separation_epsilon)
    )
    collapse = jnp.square(latent_distance)
    separate = jnp.square(jnp.maximum(float(margin) - latent_distance, 0.0))
    interpolate = jnp.square(latent_distance - scale * target)
    items = jnp.where(
        target <= float(equivalence_epsilon),
        collapse,
        jnp.where(target >= float(separation_epsilon), separate, interpolate),
    )
    if weights is not None:
        weight = jnp.asarray(weights, dtype=jnp.float32)
        return jnp.sum(items * weight) / jnp.maximum(jnp.sum(weight), 1.0e-8)
    return jnp.mean(items)


def rbf_kernel(signatures: Any, bandwidth: float, jitter: float = 1.0e-5) -> Any:
    import jax.numpy as jnp

    values = jnp.asarray(signatures, dtype=jnp.float32)
    if values.ndim < 2:
        raise ValueError("Signatures require item and feature axes.")
    if bandwidth <= 0.0 or jitter <= 0.0:
        raise ValueError("RBF bandwidth and jitter must be positive.")
    flat = values.reshape((values.shape[0], -1))
    squared = jnp.sum(
        jnp.square(flat[:, None, :] - flat[None, :, :]), axis=-1
    )
    kernel = jnp.exp(-squared / (2.0 * float(bandwidth) ** 2))
    return kernel + float(jitter) * jnp.eye(values.shape[0], dtype=kernel.dtype)


def brdiv_logdet(signatures: Any, bandwidth: float, jitter: float = 1.0e-5) -> Any:
    import jax.numpy as jnp

    kernel = rbf_kernel(signatures, bandwidth, jitter)
    sign, log_abs = jnp.linalg.slogdet(kernel)
    return jnp.where(sign > 0.0, log_abs, -jnp.inf)


def mean_pairwise_signature_distance(signatures: Any) -> Any:
    """METHOD_SPEC §7.2 diversity: mean pairwise signature distance.

    Replaces BR-divergence.  Callers supply signatures computed from real
    all-action continuations over a shared environment-state bank (M states);
    this primitive only averages the off-diagonal pairwise distances.
    """

    import jax.numpy as jnp

    values = jnp.asarray(signatures, dtype=jnp.float32)
    if values.ndim < 2:
        raise ValueError("Signatures require item and feature axes.")
    flat = values.reshape((values.shape[0], -1))
    count = flat.shape[0]
    pairwise = jnp.sqrt(
        jnp.sum(jnp.square(flat[:, None, :] - flat[None, :, :]), axis=-1)
        + 1.0e-12
    )
    mask = 1.0 - jnp.eye(count, dtype=jnp.float32)
    return jnp.sum(pairwise * mask) / jnp.maximum(
        jnp.asarray(count * (count - 1), dtype=jnp.float32), 1.0
    )


def mean_pairwise_signature_contributions(signatures: Any) -> Any:
    """Per-item mean distance to every other signature on the shared bank."""

    import jax.numpy as jnp

    values = jnp.asarray(signatures, dtype=jnp.float32)
    if values.ndim < 2:
        raise ValueError("Signatures require item and feature axes.")
    flat = values.reshape((values.shape[0], -1))
    count = flat.shape[0]
    pairwise = jnp.sqrt(
        jnp.sum(jnp.square(flat[:, None, :] - flat[None, :, :]), axis=-1)
        + 1.0e-12
    )
    mask = 1.0 - jnp.eye(count, dtype=jnp.float32)
    return jnp.sum(pairwise * mask, axis=-1) / jnp.maximum(
        jnp.asarray(count - 1, dtype=jnp.float32), 1.0
    )


def smoothness_loss(codes: Any, signatures: Any) -> Any:
    """Penalize discontinuous generator responses in neighboring code regions."""

    import jax.numpy as jnp

    code = jnp.asarray(codes, dtype=jnp.float32)
    signature = jnp.asarray(signatures, dtype=jnp.float32).reshape(
        (code.shape[0], -1)
    )
    code_distance = jnp.sqrt(
        jnp.sum(jnp.square(code[:, None, :] - code[None, :, :]), axis=-1)
        + 1.0e-8
    )
    signature_distance = jnp.sqrt(
        jnp.sum(
            jnp.square(signature[:, None, :] - signature[None, :, :]), axis=-1
        )
        + 1.0e-8
    )
    mask = 1.0 - jnp.eye(code.shape[0], dtype=jnp.float32)
    local_weight = jnp.exp(-code_distance) * mask
    return jnp.sum(local_weight * jnp.square(signature_distance - code_distance)) / jnp.maximum(
        jnp.sum(local_weight), 1.0e-8
    )



def smoothness_marginal_penalties(codes: Any, signatures: Any) -> Any:
    """Per-code local smoothness penalties for score-function optimization."""

    import jax.numpy as jnp

    code = jnp.asarray(codes, dtype=jnp.float32)
    signature = jnp.asarray(signatures, dtype=jnp.float32).reshape(
        (code.shape[0], -1)
    )
    if code.shape[0] != signature.shape[0]:
        raise ValueError("Codes and signatures must share the item axis.")
    code_distance = jnp.sqrt(
        jnp.sum(jnp.square(code[:, None, :] - code[None, :, :]), axis=-1)
        + 1.0e-8
    )
    signature_distance = jnp.sqrt(
        jnp.sum(
            jnp.square(signature[:, None, :] - signature[None, :, :]), axis=-1
        )
        + 1.0e-8
    )
    mask = 1.0 - jnp.eye(code.shape[0], dtype=jnp.float32)
    local_weight = jnp.exp(-code_distance) * mask
    items = local_weight * jnp.square(signature_distance - code_distance)
    return jnp.sum(items, axis=-1) / jnp.maximum(
        jnp.sum(local_weight, axis=-1), 1.0e-8
    )

def lower_tail_cvar(values: Any, level: float) -> Any:
    import jax.numpy as jnp

    if not 0.0 < level <= 1.0:
        raise ValueError("CVaR level must lie in (0, 1].")
    items = jnp.sort(jnp.ravel(jnp.asarray(values, dtype=jnp.float32)))
    count = jnp.maximum(1, jnp.ceil(float(level) * items.shape[0]).astype(jnp.int32))
    mask = jnp.arange(items.shape[0]) < count
    return jnp.sum(jnp.where(mask, items, 0.0)) / count.astype(jnp.float32)


def empirical_effective_rank(signatures: Any, tolerance: float = 1.0e-6) -> Any:
    """Stable-rank diagnostic for H1; never used as a training label."""

    import jax.numpy as jnp

    values = jnp.asarray(signatures, dtype=jnp.float32)
    flat = values.reshape((values.shape[0], -1))
    centered = flat - jnp.mean(flat, axis=0, keepdims=True)
    singular = jnp.linalg.svd(centered, compute_uv=False)
    squared = jnp.square(singular)
    stable_rank = jnp.sum(squared) / jnp.maximum(jnp.max(squared), float(tolerance))
    return stable_rank


__all__ = [
    "brdiv_logdet",
    "brdiv_marginal_contributions",
    "centered_action_values",
    "decision_distance",
    "empirical_effective_rank",
    "lower_tail_cvar",
    "mean_pairwise_signature_contributions",
    "mean_pairwise_signature_distance",
    "quotient_geometry_loss",
    "rbf_kernel",
    "smoothness_loss",
    "smoothness_marginal_penalties",
]


def brdiv_marginal_contributions(
    signatures: Any,
    bandwidth: float,
    jitter: float = 1.0e-5,
) -> Any:
    """Leave-one-out log-determinant contributions used as REINFORCE rewards."""

    import jax
    import jax.numpy as jnp

    values = jnp.asarray(signatures, dtype=jnp.float32)
    full = brdiv_logdet(values, bandwidth, jitter)
    count = int(values.shape[0])
    if count <= 1:
        return jnp.zeros((count,), dtype=jnp.float32)

    def without(index: Any) -> Any:
        base = jnp.arange(count - 1, dtype=jnp.int32)
        indexes = base + (base >= index).astype(jnp.int32)
        subset = values[indexes]
        return full - brdiv_logdet(subset, bandwidth, jitter)

    return jax.vmap(without)(jnp.arange(count))
