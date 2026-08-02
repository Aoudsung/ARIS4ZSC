"""Partner-run-block conformal calibration and deterministic OOD fallback.

The critic residual radius is calibrated by independent partner run, not by
individual state.  The latent support density is fitted from training-support
posterior summaries and only its threshold is selected on run-disjoint
calibration partners.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .types import CalibrationArtifact


@dataclass(frozen=True, slots=True)
class LatentSupportModel:
    mean: np.ndarray
    precision: np.ndarray
    distance_scale: float

    def score(self, latents: Any) -> np.ndarray:
        values = np.asarray(latents, dtype=np.float64)
        centered = values - self.mean
        squared = np.einsum("...i,ij,...j->...", centered, self.precision, centered)
        return np.exp(-0.5 * squared / max(self.distance_scale, 1.0e-12))


def fit_latent_support(latents: Any, *, ridge: float = 1.0e-4) -> LatentSupportModel:
    values = np.asarray(latents, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("Latent support fitting needs at least two rows.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Training support latents contain non-finite values.")
    mean = np.mean(values, axis=0)
    covariance = np.atleast_2d(np.cov(values, rowvar=False))
    covariance = covariance + float(ridge) * np.eye(values.shape[1], dtype=np.float64)
    precision = np.linalg.inv(covariance)
    centered = values - mean
    distances = np.einsum("ni,ij,nj->n", centered, precision, centered)
    scale = float(np.quantile(distances, 0.95, method="higher"))
    return LatentSupportModel(mean, precision, max(scale, 1.0))


def latent_support_score(latents: Any, calibration: CalibrationArtifact) -> Any:
    """JAX-compatible Mahalanobis support score stored in the artifact."""

    import jax.numpy as jnp

    values = jnp.asarray(latents, dtype=jnp.float32)
    mean = jnp.asarray(calibration.support_mean, dtype=jnp.float32)
    precision = jnp.asarray(calibration.support_precision, dtype=jnp.float32)
    centered = values - mean
    squared = jnp.einsum("...i,ij,...j->...", centered, precision, centered)
    scale = jnp.maximum(
        jnp.asarray(calibration.support_distance_scale, dtype=jnp.float32), 1.0e-8
    )
    return jnp.exp(-0.5 * squared / scale)


def finite_sample_conformal_quantile(scores: Sequence[float], alpha: float) -> float:
    """Split-conformal quantile ``ceil((n+1)(1-alpha))``.

    Returning infinity when the requested order statistic exceeds ``n`` is the
    finite-sample-safe abstention behavior, not a numerical failure.
    """

    values = np.sort(np.asarray(scores, dtype=np.float64))
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Conformal calibration needs non-empty scalar scores.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1).")
    rank = int(math.ceil((values.size + 1) * (1.0 - float(alpha))))
    if rank > values.size:
        return float("inf")
    return float(values[rank - 1])


def partner_run_block_scores(
    predicted_values: Any,
    empirical_values: Any,
    partner_run_ids: Sequence[str | int],
) -> Mapping[str, float]:
    """Maximum absolute all-action error within each independent partner run."""

    predicted = np.asarray(predicted_values, dtype=np.float64)
    empirical = np.asarray(empirical_values, dtype=np.float64)
    if predicted.shape != empirical.shape or predicted.shape[0] != len(partner_run_ids):
        raise ValueError("Calibration predictions, targets, and run IDs do not align.")
    result: dict[str, float] = {}
    for index, run_id in enumerate(partner_run_ids):
        error = float(np.max(np.abs(predicted[index] - empirical[index])))
        label = str(run_id)
        result[label] = max(result.get(label, 0.0), error)
    return result


def bounded_mean_radius(
    *,
    lower_bound: float,
    upper_bound: float,
    replicas: int,
    alpha: float,
    simultaneous_actions: int = 1,
) -> float:
    """Hoeffding radius for bounded empirical continuation means.

    A Bonferroni allocation across all evaluated actions is used because the
    gate compares policies selected after seeing the action vector.
    """

    if replicas <= 0 or simultaneous_actions <= 0:
        raise ValueError("replicas and simultaneous_actions must be positive.")
    if lower_bound >= upper_bound:
        raise ValueError("Return bounds are reversed.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1).")
    width = float(upper_bound) - float(lower_bound)
    allocated = float(alpha) / float(simultaneous_actions)
    return width * math.sqrt(math.log(2.0 / allocated) / (2.0 * replicas))


def empty_calibration(
    *, latent_dim: int, alpha: float, model_fingerprint: str | None = None
) -> CalibrationArtifact:
    """An abstaining artifact used before independent calibration."""

    if latent_dim <= 0:
        raise ValueError("latent_dim must be positive.")
    fingerprint = (
        np.zeros((2,), dtype=np.uint32)
        if model_fingerprint is None
        else np.frombuffer(
            hashlib.sha256(str(model_fingerprint).encode("utf-8")).digest()[:8],
            dtype=">u4",
        ).astype(np.uint32)
    )
    return CalibrationArtifact(
        alpha=np.asarray(alpha, dtype=np.float32),
        residual_radius=np.asarray(np.inf, dtype=np.float32),
        monte_carlo_radius=np.asarray(np.inf, dtype=np.float32),
        support_threshold=np.asarray(1.0, dtype=np.float32),
        support_mean=np.zeros((latent_dim,), dtype=np.float32),
        support_precision=np.eye(latent_dim, dtype=np.float32),
        support_distance_scale=np.asarray(1.0, dtype=np.float32),
        calibration_run_count=np.asarray(0, dtype=np.int32),
        model_fingerprint=fingerprint,
    )


def calibrate_adaptation_gate(
    *,
    predicted_values: Any,
    empirical_values: Any,
    partner_run_ids: Sequence[str | int],
    training_support_latents: Any,
    calibration_latents: Any,
    alpha: float,
    support_quantile: float,
    return_lower_bound: float,
    return_upper_bound: float,
    evaluation_replicas: int,
    action_count: int,
    model_fingerprint: str,
) -> CalibrationArtifact:
    """Fit the frozen simultaneous critic-error and latent-support gate."""

    if not 0.0 <= support_quantile < 1.0:
        raise ValueError("support_quantile must lie in [0, 1).")
    blocks = partner_run_block_scores(
        predicted_values, empirical_values, partner_run_ids
    )
    radius = finite_sample_conformal_quantile(tuple(blocks.values()), alpha)
    support_model = fit_latent_support(training_support_latents)
    calibration_values = np.asarray(calibration_latents, dtype=np.float64)
    if calibration_values.ndim != 2 or calibration_values.shape[0] != len(partner_run_ids):
        raise ValueError(
            "Calibration latent rows must align with calibration value rows."
        )
    support_scores = support_model.score(calibration_values)
    threshold = float(np.quantile(support_scores, support_quantile, method="lower"))
    # Centering by the action-vector mean can double the worst-case error of
    # one empirical action mean.  Calibrate that transformation explicitly.
    mc_radius = 2.0 * bounded_mean_radius(
        lower_bound=return_lower_bound,
        upper_bound=return_upper_bound,
        replicas=evaluation_replicas,
        alpha=alpha,
        simultaneous_actions=action_count,
    )
    fingerprint_bytes = hashlib.sha256(
        str(model_fingerprint).encode("utf-8")
    ).digest()[:8]
    fingerprint = np.frombuffer(fingerprint_bytes, dtype=">u4").astype(np.uint32)
    return CalibrationArtifact(
        alpha=np.asarray(alpha, dtype=np.float32),
        residual_radius=np.asarray(radius, dtype=np.float32),
        monte_carlo_radius=np.asarray(mc_radius, dtype=np.float32),
        support_threshold=np.asarray(threshold, dtype=np.float32),
        support_mean=np.asarray(support_model.mean, dtype=np.float32),
        support_precision=np.asarray(support_model.precision, dtype=np.float32),
        support_distance_scale=np.asarray(
            support_model.distance_scale, dtype=np.float32
        ),
        calibration_run_count=np.asarray(len(blocks), dtype=np.int32),
        model_fingerprint=fingerprint,
    )


def predicted_policy_gain(
    action_values: Any,
    base_logits: Any,
    conditional_logits: Any,
) -> Any:
    import jax
    import jax.numpy as jnp

    q = jnp.asarray(action_values, dtype=jnp.float32)
    base = jax.nn.softmax(jnp.asarray(base_logits, dtype=jnp.float32), axis=-1)
    conditional = jax.nn.softmax(
        jnp.asarray(conditional_logits, dtype=jnp.float32), axis=-1
    )
    return jnp.sum((conditional - base) * q, axis=-1)


def hard_adaptation_gate(
    predicted_gain: Any,
    support_score: Any,
    calibration: CalibrationArtifact,
) -> Any:
    import jax.numpy as jnp

    threshold = 2.0 * (
        jnp.asarray(calibration.residual_radius, dtype=jnp.float32)
        + jnp.asarray(calibration.monte_carlo_radius, dtype=jnp.float32)
    )
    return (
        (jnp.asarray(predicted_gain, dtype=jnp.float32) > threshold)
        & (
            jnp.asarray(support_score, dtype=jnp.float32)
            >= jnp.asarray(calibration.support_threshold, dtype=jnp.float32)
        )
    ).astype(jnp.float32)


def smooth_adaptation_gate(
    predicted_gain: Any,
    support_score: Any,
    calibration: CalibrationArtifact,
    *,
    temperature: float,
) -> Any:
    import jax
    import jax.numpy as jnp

    if temperature <= 0.0:
        raise ValueError("Gate temperature must be positive.")
    threshold = 2.0 * (
        jnp.asarray(calibration.residual_radius, dtype=jnp.float32)
        + jnp.asarray(calibration.monte_carlo_radius, dtype=jnp.float32)
    )
    gain_gate = jax.nn.sigmoid(
        (jnp.asarray(predicted_gain) - threshold) / float(temperature)
    )
    support_gate = jax.nn.sigmoid(
        (
            jnp.asarray(support_score)
            - jnp.asarray(calibration.support_threshold)
        )
        / float(temperature)
    )
    return gain_gate * support_gate


def calibration_to_mapping(artifact: CalibrationArtifact) -> Mapping[str, Any]:
    result: dict[str, Any] = {}
    for name, value in artifact._asdict().items():
        array = np.asarray(value)
        result[name] = array.item() if array.ndim == 0 else array.tolist()
    return result


def calibration_from_mapping(payload: Mapping[str, Any]) -> CalibrationArtifact:
    required = set(CalibrationArtifact._fields)
    if set(payload) != required:
        raise ValueError(
            "Calibration fields differ: "
            f"missing={sorted(required - set(payload))}, "
            f"unknown={sorted(set(payload) - required)}."
        )
    return CalibrationArtifact(
        alpha=np.asarray(payload["alpha"], dtype=np.float32),
        residual_radius=np.asarray(payload["residual_radius"], dtype=np.float32),
        monte_carlo_radius=np.asarray(payload["monte_carlo_radius"], dtype=np.float32),
        support_threshold=np.asarray(payload["support_threshold"], dtype=np.float32),
        support_mean=np.asarray(payload["support_mean"], dtype=np.float32),
        support_precision=np.asarray(payload["support_precision"], dtype=np.float32),
        support_distance_scale=np.asarray(
            payload["support_distance_scale"], dtype=np.float32
        ),
        calibration_run_count=np.asarray(
            payload["calibration_run_count"], dtype=np.int32
        ),
        model_fingerprint=np.asarray(payload["model_fingerprint"], dtype=np.uint32),
    )


__all__ = [
    "LatentSupportModel",
    "bounded_mean_radius",
    "calibrate_adaptation_gate",
    "calibration_from_mapping",
    "calibration_to_mapping",
    "empty_calibration",
    "finite_sample_conformal_quantile",
    "fit_latent_support",
    "hard_adaptation_gate",
    "latent_support_score",
    "partner_run_block_scores",
    "predicted_policy_gain",
    "smooth_adaptation_gate",
]
