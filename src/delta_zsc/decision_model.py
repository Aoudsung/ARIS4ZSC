"""Current and probe-successor decision emissions in raw task-return units.

Both estimands use a shared action-value baseline plus a centered component
residual.  The model variance is shared across components, preventing a latent
component from winning the mixture by inflating or shrinking uncertainty rather
than predicting a different action ordering.
"""

from __future__ import annotations

from typing import Any, Mapping

from .nn import init_linear, init_mlp, linear, mlp
from .types import DecisionPrediction


def action_contrast_matrix(action_count: int) -> Any:
    """Return an A x (A-1) orthonormal basis orthogonal to all-ones."""

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
    action_embedding_dim: int,
    hidden_dim: int,
    action_count: int,
) -> dict[str, Any]:
    import jax

    keys = jax.random.split(key, 16)
    shared_input = int(task_dim) + int(instant_dim) + int(behavior_dim)
    component_input = int(hidden_dim) + int(component_embedding_dim)
    return {
        "shared_trunk": init_mlp(
            keys[0], (shared_input, int(hidden_dim), int(hidden_dim))
        ),
        "component_trunk": init_mlp(
            keys[1], (component_input, int(hidden_dim), int(hidden_dim))
        ),
        "shared_mean": init_linear(keys[2], hidden_dim, action_count, scale=0.01),
        "component_mean_context": init_linear(
            keys[3], hidden_dim, action_count, scale=1.0
        ),
        "component_mean_embedding": init_linear(
            keys[4], component_embedding_dim, action_count, scale=1.0
        ),
        "shared_log_variance": init_linear(
            keys[5], hidden_dim, action_count, scale=0.01
        ),
        "probe_action_embedding": jax.random.normal(
            keys[6], (int(action_count), int(action_embedding_dim))
        ) / max(int(action_embedding_dim), 1) ** 0.5,
        "successor_shared_trunk": init_mlp(
            keys[7],
            (
                int(hidden_dim) + int(action_embedding_dim),
                int(hidden_dim),
                int(hidden_dim),
            ),
        ),
        "successor_component_trunk": init_mlp(
            keys[8], (component_input, int(hidden_dim), int(hidden_dim))
        ),
        "successor_shared_mean": init_linear(
            keys[9], hidden_dim, action_count, scale=0.01
        ),
        "successor_component_mean_context": init_linear(
            keys[10], hidden_dim, action_count, scale=1.0
        ),
        "successor_component_mean_embedding": init_linear(
            keys[11], component_embedding_dim, action_count, scale=1.0
        ),
        "successor_shared_log_variance": init_linear(
            keys[12], hidden_dim, action_count, scale=0.01
        ),
    }


def _broadcast_components(components: Any, lead: tuple[int, ...]) -> Any:
    import jax.numpy as jnp

    value = jnp.asarray(components, dtype=jnp.float32)
    return jnp.broadcast_to(
        value.reshape((1,) * len(lead) + value.shape), lead + value.shape
    )


def _shared_context(
    params: Mapping[str, Any],
    task_features: Any,
    instant_partner: Any,
    behavior_features: Any,
) -> tuple[Any, tuple[int, ...]]:
    import jax.numpy as jnp

    task = jnp.asarray(task_features, dtype=jnp.float32)
    instant = jnp.asarray(instant_partner, dtype=jnp.float32)
    behavior = jnp.asarray(behavior_features, dtype=jnp.float32)
    lead = task.shape[:-1]
    if instant.shape[:-1] != lead or behavior.shape[:-1] != lead:
        raise ValueError("Decision feature batch axes differ.")
    shared = mlp(
        params["shared_trunk"],
        jnp.concatenate((task, instant, behavior), axis=-1),
        final_activation=True,
    )
    return shared, lead


def _prediction_from_hidden(
    *,
    shared_hidden: Any,
    component_hidden: Any,
    component_embeddings: Any,
    shared_mean_params: Any,
    context_residual_params: Any,
    embedding_residual_params: Any,
    variance_params: Any,
) -> DecisionPrediction:
    import jax.numpy as jnp

    shared_raw = linear(shared_mean_params, shared_hidden)
    shared_mean = shared_raw - jnp.mean(shared_raw, axis=-1, keepdims=True)
    raw_residual = linear(context_residual_params, component_hidden) + linear(
        embedding_residual_params, component_embeddings
    )
    residual = raw_residual - jnp.mean(raw_residual, axis=-2, keepdims=True)
    residual = residual - jnp.mean(residual, axis=-1, keepdims=True)
    means = shared_mean[..., None, :] + residual
    means = means - jnp.mean(means, axis=-1, keepdims=True)
    shared_log_variance = jnp.clip(
        linear(variance_params, shared_hidden), -6.0, 6.0
    )
    variances = jnp.broadcast_to(
        jnp.exp(shared_log_variance)[..., None, :], means.shape
    )
    return DecisionPrediction(
        means=means,
        variances=variances,
        shared_means=shared_mean,
        component_residuals=residual,
    )


def decision_predict(
    params: Mapping[str, Any],
    component_embeddings: Any,
    task_features: Any,
    instant_partner: Any,
    behavior_features: Any,
) -> DecisionPrediction:
    """Predict current-state all-action returns under each semantic component."""

    import jax.numpy as jnp

    shared, lead = _shared_context(
        params, task_features, instant_partner, behavior_features
    )
    components = _broadcast_components(component_embeddings, lead)
    count = int(jnp.asarray(component_embeddings).shape[0])
    shared_by_component = jnp.broadcast_to(
        shared[..., None, :], shared.shape[:-1] + (count, shared.shape[-1])
    )
    component_hidden = mlp(
        params["component_trunk"],
        jnp.concatenate((shared_by_component, components), axis=-1),
        final_activation=True,
    )
    return _prediction_from_hidden(
        shared_hidden=shared,
        component_hidden=component_hidden,
        component_embeddings=components,
        shared_mean_params=params["shared_mean"],
        context_residual_params=params["component_mean_context"],
        embedding_residual_params=params["component_mean_embedding"],
        variance_params=params["shared_log_variance"],
    )


def successor_decision_predict(
    params: Mapping[str, Any],
    component_embeddings: Any,
    task_features: Any,
    instant_partner: Any,
    behavior_features: Any,
    probe_actions: Any,
) -> DecisionPrediction:
    """Predict all post-response action returns after each probe.

    The prediction is conditioned on the pre-probe legal state and integrates
    over the probe transition plus one intervening collection-time base-policy
    action.  The predicted all-action matrix is therefore located at ``t+2``,
    when the partner's delayed response is first observable.  Its output shape
    is ``probe_actions.shape + (K, A)``.
    """

    import jax.numpy as jnp

    shared, base_lead = _shared_context(
        params, task_features, instant_partner, behavior_features
    )
    probes = jnp.asarray(probe_actions, dtype=jnp.int32)
    probe_lead = probes.shape
    if tuple(probe_lead[: len(base_lead)]) != tuple(base_lead):
        raise ValueError("Probe-action axes do not extend the decision batch axes.")
    extra = len(probe_lead) - len(base_lead)
    expanded_shared = jnp.broadcast_to(
        shared.reshape(base_lead + (1,) * extra + (shared.shape[-1],)),
        probe_lead + (shared.shape[-1],),
    )
    successor_shared = mlp(
        params["successor_shared_trunk"],
        jnp.concatenate(
            (expanded_shared, params["probe_action_embedding"][probes]), axis=-1
        ),
        final_activation=True,
    )
    components = _broadcast_components(component_embeddings, probe_lead)
    count = int(jnp.asarray(component_embeddings).shape[0])
    shared_by_component = jnp.broadcast_to(
        successor_shared[..., None, :],
        successor_shared.shape[:-1] + (count, successor_shared.shape[-1]),
    )
    component_hidden = mlp(
        params["successor_component_trunk"],
        jnp.concatenate((shared_by_component, components), axis=-1),
        final_activation=True,
    )
    return _prediction_from_hidden(
        shared_hidden=successor_shared,
        component_hidden=component_hidden,
        component_embeddings=components,
        shared_mean_params=params["successor_shared_mean"],
        context_residual_params=params["successor_component_mean_context"],
        embedding_residual_params=params["successor_component_mean_embedding"],
        variance_params=params["successor_shared_log_variance"],
    )


def decision_component_log_probability(
    prediction: DecisionPrediction,
    targets: Any,
    measurement_covariances: Any,
    action_mask: Any | None = None,
) -> Any:
    """Full-covariance Gaussian score in the A-1 decision-contrast subspace."""

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
    covariance = covariance + 1.0e-5 * jnp.eye(dimension, dtype=jnp.float32)
    difference = target_contrast[..., None, :] - mean_contrast
    cholesky = jnp.linalg.cholesky(covariance)
    whitened = jsl.solve_triangular(
        cholesky, difference[..., None], lower=True
    )[..., 0]
    quadratic = jnp.sum(jnp.square(whitened), axis=-1)
    log_determinant = 2.0 * jnp.sum(
        jnp.log(
            jnp.maximum(
                jnp.diagonal(cholesky, axis1=-2, axis2=-1), 1.0e-12
            )
        ),
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
    "successor_decision_predict",
]
