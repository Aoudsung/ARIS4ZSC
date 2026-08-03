"""Optional V6 safety wrapper calibrated on direct policy-gain residuals.

This module is not imported by primary training or deployment.  It compares
the final full-context actor with the same actor under prior context and is
therefore an optional post-training safety ablation only.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .types import CalibrationArtifact


def predicted_policy_gain(
    action_values: Any, full_context_logits: Any, prior_context_logits: Any
) -> Any:
    try:
        import jax
        import jax.numpy as xp

        softmax = lambda value: jax.nn.softmax(value, axis=-1)
    except ModuleNotFoundError:  # Pure host-side calibration remains testable without JAX.
        import numpy as xp

        def softmax(value: Any) -> Any:
            shifted = value - xp.max(value, axis=-1, keepdims=True)
            exponential = xp.exp(shifted)
            return exponential / xp.sum(exponential, axis=-1, keepdims=True)

    values = xp.asarray(action_values, dtype=xp.float32)
    full = softmax(xp.asarray(full_context_logits, dtype=xp.float32))
    prior = softmax(xp.asarray(prior_context_logits, dtype=xp.float32))
    return xp.sum((full - prior) * values, axis=-1)


def empirical_policy_gain(
    returns_by_action: Any, full_context_logits: Any, prior_context_logits: Any
) -> Any:
    return predicted_policy_gain(
        returns_by_action, full_context_logits, prior_context_logits
    )


def _support_model(training_latents: Any) -> tuple[Any, Any, Any]:
    import numpy as np

    values = np.asarray(training_latents, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("Safety support needs at least two training latent rows.")
    mean = np.mean(values, axis=0)
    covariance = np.cov(values, rowvar=False)
    covariance = np.atleast_2d(covariance) + 1.0e-4 * np.eye(values.shape[1])
    precision = np.linalg.pinv(covariance)
    distances = np.einsum("ni,ij,nj->n", values - mean, precision, values - mean)
    scale = max(float(np.median(distances)), 1.0e-8)
    return mean, precision, scale


def latent_support_score(latents: Any, artifact: CalibrationArtifact) -> Any:
    try:
        import jax.numpy as xp
    except ModuleNotFoundError:
        import numpy as xp

    values = xp.asarray(latents, dtype=xp.float32)
    difference = values - xp.asarray(artifact.support_mean, dtype=xp.float32)
    distance = xp.einsum(
        "...i,ij,...j->...",
        difference,
        xp.asarray(artifact.support_precision, dtype=xp.float32),
        difference,
    )
    return xp.exp(
        -0.5
        * distance
        / xp.maximum(
            xp.asarray(artifact.support_distance_scale, dtype=xp.float32), 1.0e-8
        )
    )


def _finite_block_quantile(scores: Sequence[float], alpha: float) -> float:
    import math
    import numpy as np

    values = np.sort(np.asarray(scores, dtype=np.float64))
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Safety calibration needs non-empty partner-run scores.")
    rank = int(math.ceil((values.size + 1) * (1.0 - float(alpha))))
    if rank > values.size:
        raise ValueError("Too few run blocks for a finite conformal quantile.")
    return float(values[max(rank - 1, 0)])


def calibrate_safety_wrapper(
    *,
    predicted_gains: Any,
    empirical_gains: Any,
    partner_run_ids: Sequence[str],
    training_support_latents: Any,
    calibration_latents: Any,
    alpha: float,
    support_quantile: float,
    model_fingerprint: Any,
) -> CalibrationArtifact:
    import numpy as np

    predicted = np.asarray(predicted_gains, dtype=np.float64).reshape(-1)
    empirical = np.asarray(empirical_gains, dtype=np.float64).reshape(-1)
    labels = np.asarray(tuple(str(value) for value in partner_run_ids), dtype=object)
    latents = np.asarray(calibration_latents, dtype=np.float64)
    if predicted.shape != empirical.shape or predicted.shape != labels.shape:
        raise ValueError("Gain residual rows and partner-run blocks do not align.")
    if latents.ndim != 2 or latents.shape[0] != predicted.size:
        raise ValueError("Calibration latent rows do not align with gain rows.")
    blocks = []
    residual = np.abs(predicted - empirical)
    for label in sorted(set(labels.tolist())):
        blocks.append(float(np.max(residual[labels == label])))
    mean, precision, scale = _support_model(training_support_latents)
    temporary = CalibrationArtifact(
        alpha=np.asarray(alpha, dtype=np.float32),
        gain_residual_radius=np.asarray(0.0, dtype=np.float32),
        support_threshold=np.asarray(0.0, dtype=np.float32),
        support_mean=np.asarray(mean, dtype=np.float32),
        support_precision=np.asarray(precision, dtype=np.float32),
        support_distance_scale=np.asarray(scale, dtype=np.float32),
        calibration_run_count=np.asarray(len(blocks), dtype=np.int32),
        model_fingerprint=np.asarray(model_fingerprint, dtype=np.uint32),
    )
    support = np.asarray(latent_support_score(latents, temporary), dtype=np.float64)
    return temporary._replace(
        gain_residual_radius=np.asarray(
            _finite_block_quantile(blocks, alpha), dtype=np.float32
        ),
        support_threshold=np.asarray(
            np.quantile(support, float(support_quantile)), dtype=np.float32
        ),
    )


def safety_select_full_context(
    predicted_gain: Any, support_score: Any, artifact: CalibrationArtifact
) -> Any:
    try:
        import jax.numpy as xp
    except ModuleNotFoundError:
        import numpy as xp

    return (
        xp.asarray(predicted_gain, dtype=xp.float32)
        - xp.asarray(artifact.gain_residual_radius, dtype=xp.float32)
        > 0.0
    ) & (
        xp.asarray(support_score, dtype=xp.float32)
        >= xp.asarray(artifact.support_threshold, dtype=xp.float32)
    )


def calibration_to_mapping(artifact: CalibrationArtifact) -> Mapping[str, Any]:
    import numpy as np

    return {
        name: np.asarray(getattr(artifact, name)).tolist()
        for name in artifact._fields
    }


def calibration_from_mapping(payload: Mapping[str, Any]) -> CalibrationArtifact:
    import numpy as np

    if set(payload) != set(CalibrationArtifact._fields):
        raise ValueError("Safety calibration fields differ from V6.")
    return CalibrationArtifact(
        alpha=np.asarray(payload["alpha"], dtype=np.float32),
        gain_residual_radius=np.asarray(
            payload["gain_residual_radius"], dtype=np.float32
        ),
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
    "calibrate_safety_wrapper",
    "calibration_from_mapping",
    "calibration_to_mapping",
    "empirical_policy_gain",
    "latent_support_score",
    "predicted_policy_gain",
    "safety_select_full_context",
]
