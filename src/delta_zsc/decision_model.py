"""Decision emission in raw task-return units.

The six action returns are identified only up to a shared offset.  DELTA uses
an orthonormal Helmert contrast basis, so the likelihood lives in the
five-dimensional decision subspace and can use the full CRN measurement
covariance without introducing a singular centered Gaussian.
"""

from __future__ import annotations

from typing import Any

from .nn import init_linear, init_mlp, linear, mlp
from .types import DecisionPrediction


def action_contrast_matrix(action_count: int) -> Any:
    """Return an A x (A-1) orthonormal basis orthogonal to the all-ones vector."""

    import jax.numpy as jnp

    count = int(action_count)
    if count < 2:
        raise ValueError("Decision contrasts require at least two actions.")
    matrix = jnp.zeros((count, count - 1), dtype=jnp.float32)
    for column in range(count - 1):
        denominator = ((column + 1) * (column + 2)) ** 0.5
        matrix = matrix.at[: column + 1, column].set(1.0 / denominator)
        matrix = matrix.at[column + 1, column].set(-(column + 1) / denominator)
    return matrix


def init_decision_params(
    key: Any,
    *,
    task_dim: int,
    instant_dim: int,
    behavior_dim: int,
    component_embedding_dim: int,
    hidden_dim: int,
    action_count: int,
) -> dict[str, Any]:
    import jax

    keys = jax.random.split(key, 3)
    input_dim = task_dim + instant_dim + behavior_dim + component_embedding_dim
    return {
        "trunk": init_mlp(keys[0], (input_dim, hidden_dim, hidden_dim)),
        "mean": init_linear(keys[1], hidden_dim, action_count, scale=0.01),
        "log_variance": init_linear(keys[2], hidden_dim, action_count, scale=0.01),
    }


def decision_predict(
    params: dict[str, Any],
    component_embeddings: Any,
    task_features: Any,
    instant_partner: Any,
    behavior_features: Any,
) -> DecisionPrediction:
    import jax.numpy as jnp

    task = jnp.asarray(task_features, dtype=jnp.float32)
    instant = jnp.asarray(instant_partner, dtype=jnp.float32)
    behavior = jnp.asarray(behavior_features, dtype=jnp.float32)
    components = jnp.asarray(component_embeddings, dtype=jnp.float32)
    lead = task.shape[:-1]
    if instant.shape[:-1] != lead or behavior.shape[:-1] != lead:
        raise ValueError("Decision feature batch axes differ.")
    count = components.shape[0]

    def broadcast(value: Any) -> Any:
        array = jnp.asarray(value)
        return jnp.broadcast_to(
            array[..., None, :], lead + (count, array.shape[-1])
        )

    comp = jnp.broadcast_to(
        components.reshape((1,) * len(lead) + components.shape),
        lead + components.shape,
    )
    hidden = mlp(
        params["trunk"],
        jnp.concatenate(
            (broadcast(task), broadcast(instant), broadcast(behavior), comp), axis=-1
        ),
        final_activation=True,
    )
    raw_mean = linear(params["mean"], hidden)
    mean = raw_mean - jnp.mean(raw_mean, axis=-1, keepdims=True)
    log_variance = jnp.clip(linear(params["log_variance"], hidden), -6.0, 6.0)
    return DecisionPrediction(means=mean, variances=jnp.exp(log_variance))


def decision_component_log_probability(
    prediction: DecisionPrediction,
    targets: Any,
    measurement_covariances: Any,
    action_mask: Any | None = None,
) -> Any:
    """Full-covariance Gaussian score in the A-1 decision-contrast subspace.

    The model emits positive per-action variances.  They are projected through
    the same contrast basis as the empirical CRN covariance.  This preserves
    cross-action common-random-number correlations and avoids pretending that
    centered action returns are independent.
    """

    import jax.numpy as jnp
    import jax.scipy.linalg as jsl

    target = jnp.asarray(targets, dtype=jnp.float32)
    action_count = int(target.shape[-1])
    basis = action_contrast_matrix(action_count)
    target_contrast = target @ basis
    mean_contrast = jnp.einsum("...ka,ad->...kd", prediction.means, basis)
    model_covariance = jnp.einsum(
        "...ka,ad,ae->...kde", prediction.variances, basis, basis
    )
    measured = jnp.asarray(measurement_covariances, dtype=jnp.float32)
    expected_shape = target.shape[:-1] + (action_count - 1, action_count - 1)
    if measured.shape != expected_shape:
        raise ValueError("Decision measurement covariance shape differs.")
    covariance = model_covariance + measured[..., None, :, :]
    dimension = action_count - 1
    identity = jnp.eye(dimension, dtype=jnp.float32)
    covariance = covariance + 1.0e-5 * identity
    difference = target_contrast[..., None, :] - mean_contrast
    cholesky = jnp.linalg.cholesky(covariance)
    whitened = jsl.solve_triangular(
        cholesky, difference[..., None], lower=True
    )[..., 0]
    quadratic = jnp.sum(jnp.square(whitened), axis=-1)
    log_determinant = 2.0 * jnp.sum(
        jnp.log(jnp.maximum(jnp.diagonal(cholesky, axis1=-2, axis2=-1), 1.0e-12)),
        axis=-1,
    )
    log_probability = -0.5 * (
        quadratic + log_determinant + float(dimension) * jnp.log(2.0 * jnp.pi)
    )
    if action_mask is not None:
        valid = jnp.all(jnp.asarray(action_mask, dtype=jnp.bool_), axis=-1)
        log_probability = jnp.where(valid[..., None], log_probability, 0.0)
    return log_probability


__all__ = [
    "action_contrast_matrix",
    "decision_component_log_probability",
    "decision_predict",
    "init_decision_params",
]
