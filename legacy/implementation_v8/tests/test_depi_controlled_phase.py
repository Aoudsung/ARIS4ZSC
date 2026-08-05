"""Deterministic unit tests for the supporting controlled phase diagram.

All quantities are exact enumerations, so every assertion uses a hard
tolerance and no seeding.  The registered specification lives in
``docs/research/S2_PHASE_TASK_SPEC.md``.
"""

from __future__ import annotations

import math

from experiments.overcooked_v2.controlled_phase_app import (
    binomial_tv,
    directed_window_point,
    evaluate_point,
    run_scan,
    t3_lower_bound,
    t3_upper_bound,
    tv_sequence,
)


def test_tv_zero_and_monotone() -> None:
    assert binomial_tv(0, 0.4, 0.6) == 0.0
    sequence = tv_sequence(16, 0.2)
    assert sequence[0] == 0.0
    assert all(
        right >= left - 1e-12 for left, right in zip(sequence, sequence[1:])
    )


def test_tv_one_equals_kappa() -> None:
    # TV between Bernoulli((1-k)/2) and Bernoulli((1+k)/2) is exactly k.
    for kappa in (0.05, 0.2, 0.5):
        assert math.isclose(binomial_tv(1, (1 - kappa) / 2, (1 + kappa) / 2), kappa)


def test_t1_chain_and_t2_identity() -> None:
    point = evaluate_point(
        delta=4.0, kappa=0.2, n_e=8, c=0.0, n_lock=8, t_prefix=0, epsilon=0.5
    )
    checks = point["checks"]
    assert checks["t1_chain_ok"]
    assert checks["t2_ok"]
    assert checks["t2_residual_hz"] <= 1e-9
    assert checks["t2_residual_fix"] <= 1e-9


def test_chain_survives_waiting_cost() -> None:
    point = evaluate_point(
        delta=2.0, kappa=0.05, n_e=16, c=0.05, n_lock=16, t_prefix=8, epsilon=0.25
    )
    assert point["checks"]["t1_chain_ok"]


def test_t3_bounds_bracket_empirical_length() -> None:
    delta, kappa, epsilon = 8.0, 0.1, 1.0
    point = evaluate_point(
        delta=delta, kappa=kappa, n_e=32, c=0.0, n_lock=32, t_prefix=0, epsilon=epsilon
    )
    checks = point["checks"]
    assert checks["t3_ok"], checks["t3_detail"]
    assert t3_lower_bound(delta, epsilon, kappa) <= point["n_emp"]
    assert point["n_emp"] <= t3_upper_bound(delta, epsilon, kappa)


def test_directed_window_end_to_end_gain_nonpositive() -> None:
    window = directed_window_point()
    checks = window["checks"]
    assert checks["window_oracle_positive"]
    assert checks["window_end_to_end_nonpositive"]


def test_quick_scan_fully_passes() -> None:
    result = run_scan(quick=True)
    assert result["summary"]["failures"] == 0
    assert result["summary"]["t1_pass"] == result["summary"]["n_points"]
    assert result["summary"]["t2_pass"] == result["summary"]["n_points"]
    assert result["summary"]["t3_pass"] == result["summary"]["n_points"]
    assert result["summary"]["window_pass"]
