from __future__ import annotations

import numpy as np
import pytest

from src.path_c.calibration import (
    calibrate_safety_wrapper,
    calibration_from_mapping,
    calibration_to_mapping,
    empirical_policy_gain,
    predicted_policy_gain,
    safety_select_full_context,
)


def test_direct_policy_gain_uses_full_minus_prior_probabilities() -> None:
    values = np.asarray([[0.0, 20.0]], dtype=np.float32)
    full = np.asarray([[-10.0, 10.0]], dtype=np.float32)
    prior = np.asarray([[0.0, 0.0]], dtype=np.float32)
    predicted = np.asarray(predicted_policy_gain(values, full, prior))
    empirical = np.asarray(empirical_policy_gain(values, full, prior))
    assert predicted[0] == pytest.approx(10.0, abs=1.0e-3)
    np.testing.assert_allclose(predicted, empirical)


def test_optional_safety_calibrates_run_block_gain_residual() -> None:
    run_count = 19
    predicted = np.linspace(10.0, 28.0, run_count, dtype=np.float32)
    empirical = predicted - np.linspace(0.0, 1.8, run_count, dtype=np.float32)
    support = np.stack(
        (np.linspace(-1.0, 1.0, 40), np.linspace(1.0, -1.0, 40)), axis=-1
    )
    artifact = calibrate_safety_wrapper(
        predicted_gains=predicted,
        empirical_gains=empirical,
        partner_run_ids=tuple(f"run-{index}" for index in range(run_count)),
        training_support_latents=support,
        calibration_latents=support[:run_count],
        alpha=0.05,
        support_quantile=0.05,
        model_fingerprint=np.asarray([1, 2], dtype=np.uint32),
    )
    assert int(np.asarray(artifact.calibration_run_count)) == 19
    assert float(np.asarray(artifact.gain_residual_radius)) == pytest.approx(1.8)
    restored = calibration_from_mapping(calibration_to_mapping(artifact))
    assert calibration_to_mapping(restored) == calibration_to_mapping(artifact)


def test_safety_wrapper_is_not_an_always_on_gate() -> None:
    artifact = calibrate_safety_wrapper(
        predicted_gains=np.full(19, 30.0),
        empirical_gains=np.full(19, 20.0),
        partner_run_ids=tuple(f"run-{index}" for index in range(19)),
        training_support_latents=np.stack((np.arange(30), np.arange(30)), axis=-1),
        calibration_latents=np.stack((np.arange(19), np.arange(19)), axis=-1),
        alpha=0.05,
        support_quantile=0.0,
        model_fingerprint=np.asarray([3, 4], dtype=np.uint32),
    )
    selected = np.asarray(
        safety_select_full_context(
            np.asarray([11.0, 9.0]),
            np.asarray([1.0, 1.0]),
            artifact,
        )
    )
    assert selected.tolist() == [True, False]


def test_too_few_run_blocks_fail_instead_of_replacing_primary_method() -> None:
    with pytest.raises(ValueError, match="Too few"):
        calibrate_safety_wrapper(
            predicted_gains=np.zeros(18),
            empirical_gains=np.zeros(18),
            partner_run_ids=tuple(f"run-{index}" for index in range(18)),
            training_support_latents=np.zeros((20, 2)),
            calibration_latents=np.zeros((18, 2)),
            alpha=0.05,
            support_quantile=0.05,
            model_fingerprint=np.asarray([1, 2], dtype=np.uint32),
        )
