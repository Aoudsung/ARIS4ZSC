from __future__ import annotations

import numpy as np
import pytest

from src.path_c.calibration import (
    calibrate_adaptation_gate,
    calibration_from_mapping,
    calibration_to_mapping,
    empty_calibration,
    finite_sample_conformal_quantile,
    hard_adaptation_gate,
    partner_run_block_scores,
)


def test_finite_sample_quantile_abstains_when_sample_is_too_small() -> None:
    assert np.isinf(finite_sample_conformal_quantile(np.arange(18), 0.05))
    assert finite_sample_conformal_quantile(np.arange(19), 0.05) == 18.0


def test_partner_run_scores_use_maximum_error_per_independent_run() -> None:
    predicted = np.zeros((3, 2), dtype=np.float32)
    empirical = np.asarray([[1.0, -2.0], [3.0, 0.0], [0.5, 0.25]])
    scores = partner_run_block_scores(predicted, empirical, ("a", "a", "b"))
    assert scores == {"a": 3.0, "b": 0.5}


def test_calibration_round_trip_and_hard_abstention() -> None:
    jnp = pytest.importorskip("jax.numpy")

    empty = empty_calibration(latent_dim=2, alpha=0.05)
    gate = hard_adaptation_gate(
        jnp.asarray([1.0e9]), jnp.asarray([1.0]), empty
    )
    assert np.asarray(gate).tolist() == [0.0]
    restored = calibration_from_mapping(calibration_to_mapping(empty))
    assert np.isinf(np.asarray(restored.residual_radius))


def test_partner_block_calibration_is_finite_with_registered_minimum() -> None:
    count = 19
    predicted = np.zeros((count, 3), dtype=np.float32)
    empirical = np.linspace(0.0, 0.18, count, dtype=np.float32)[:, None]
    empirical = np.repeat(empirical, 3, axis=1)
    training_latents = np.stack(
        (np.linspace(-1.0, 1.0, 40), np.linspace(1.0, -1.0, 40)), axis=-1
    )
    calibration_latents = training_latents[:count]
    artifact = calibrate_adaptation_gate(
        predicted_values=predicted,
        empirical_values=empirical,
        partner_run_ids=tuple(f"run-{index}" for index in range(count)),
        training_support_latents=training_latents,
        calibration_latents=calibration_latents,
        alpha=0.05,
        support_quantile=0.05,
        return_lower_bound=-1.0,
        return_upper_bound=1.0,
        evaluation_replicas=128,
        action_count=3,
        model_fingerprint="model-a",
    )
    assert int(np.asarray(artifact.calibration_run_count)) == count
    assert np.isfinite(np.asarray(artifact.residual_radius))
    assert np.asarray(artifact.model_fingerprint).shape == (2,)
