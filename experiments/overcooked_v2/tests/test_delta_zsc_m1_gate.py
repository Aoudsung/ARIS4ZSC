"""Extended M1 gate tests (METHOD_SPEC §6, REVIEW_AND_SUGGESTION_2026 §6).

Specification entries covered:
- The action-ranking Spearman check runs on both the posterior-mean path and
  every bootstrap member path (B=3), each against the shared CRN all-action
  signature of the same anchor state.
- The gate passes only if every path reaches Spearman >= 0.8 on >= 90% of
  anchor states; pooling across members is forbidden.
- The readout is report-only: nothing here shapes rewards or gradients.
"""

from __future__ import annotations

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from src.path_c.m1_gate import (  # noqa: E402
    M1_MINIMUM_ANCHOR_FRACTION,
    M1_SPEARMAN_THRESHOLD,
    evaluate_extended_m1_gate,
    m1_anchor_gate,
    m1_gate_report,
    spearman_rank_correlation,
)


def _signatures(anchor_count: int = 10, action_count: int = 6) -> jnp.ndarray:
    key = jax.random.PRNGKey(7)
    return jax.random.normal(key, (anchor_count, action_count))


def test_frozen_gate_thresholds_match_section_6() -> None:
    # §6 freezes Spearman >= 0.8 on >= 90% of anchor states.
    assert M1_SPEARMAN_THRESHOLD == 0.8
    assert M1_MINIMUM_ANCHOR_FRACTION == 0.9


def test_spearman_is_exactly_one_for_identical_rankings() -> None:
    signatures = _signatures()
    rho = spearman_rank_correlation(signatures, signatures)
    np.testing.assert_allclose(np.asarray(rho), np.ones(10), atol=1e-5)


def test_spearman_is_minus_one_for_reversed_rankings() -> None:
    signatures = _signatures()
    rho = spearman_rank_correlation(-signatures, signatures)
    np.testing.assert_allclose(np.asarray(rho), -np.ones(10), atol=1e-5)


def test_spearman_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="share shape"):
        spearman_rank_correlation(jnp.zeros((4, 6)), jnp.zeros((5, 6)))


def test_single_path_gate_passes_only_above_the_fraction_floor() -> None:
    signatures = _signatures()
    passed, fraction = m1_anchor_gate(signatures, crn_signatures=signatures)
    assert bool(np.asarray(passed))
    assert float(np.asarray(fraction)) == pytest.approx(1.0)

    reversed_values = -signatures
    passed, fraction = m1_anchor_gate(reversed_values, crn_signatures=signatures)
    assert not bool(np.asarray(passed))
    assert float(np.asarray(fraction)) == pytest.approx(0.0)


def test_extended_gate_passes_when_every_path_ranks_actions_correctly() -> None:
    signatures = _signatures()
    member_values = jnp.stack([signatures, signatures, signatures], axis=0)
    result = evaluate_extended_m1_gate(
        posterior_mean_action_values=signatures,
        member_action_values=member_values,
        crn_signatures=signatures,
    )
    assert bool(np.asarray(result.passed))
    fractions = np.asarray(result.path_passing_fractions)
    # 1 posterior-mean path + B=3 bootstrap member paths.
    assert fractions.shape == (4,)
    np.testing.assert_allclose(fractions, np.ones(4), atol=1e-6)


def test_extended_gate_fails_when_a_single_member_path_fails() -> None:
    # §6 forbids pooling across members: one failing bootstrap encoder must
    # keep the gate closed even when the posterior path and the other members
    # pass.
    signatures = _signatures()
    member_values = jnp.stack([signatures, -signatures, signatures], axis=0)
    result = evaluate_extended_m1_gate(
        posterior_mean_action_values=signatures,
        member_action_values=member_values,
        crn_signatures=signatures,
    )
    assert not bool(np.asarray(result.passed))
    fractions = np.asarray(result.path_passing_fractions)
    assert fractions[0] == pytest.approx(1.0)
    assert fractions[2] == pytest.approx(0.0)


def test_extended_gate_fails_when_the_posterior_path_fails() -> None:
    signatures = _signatures()
    member_values = jnp.stack([signatures, signatures, signatures], axis=0)
    result = evaluate_extended_m1_gate(
        posterior_mean_action_values=-signatures,
        member_action_values=member_values,
        crn_signatures=signatures,
    )
    assert not bool(np.asarray(result.passed))


def test_extended_gate_valid_mask_excludes_invalid_anchor_states() -> None:
    signatures = _signatures(anchor_count=20)
    # Corrupt the last two anchors on every path; masking them restores the
    # passing fraction on the remaining 18/20 = 90%.
    corrupted = signatures.at[-2:].set(-signatures[-2:])
    mask = jnp.ones((20,), dtype=jnp.float32).at[-2:].set(0.0)
    member_values = jnp.stack([corrupted, corrupted, corrupted], axis=0)
    result = evaluate_extended_m1_gate(
        posterior_mean_action_values=corrupted,
        member_action_values=member_values,
        crn_signatures=signatures,
        valid_mask=mask,
    )
    assert bool(np.asarray(result.passed))


def test_m1_gate_report_is_json_ready() -> None:
    signatures = _signatures()
    member_values = jnp.stack([signatures, signatures, signatures], axis=0)
    result = evaluate_extended_m1_gate(
        posterior_mean_action_values=signatures,
        member_action_values=member_values,
        crn_signatures=signatures,
    )
    report = m1_gate_report(result)
    assert report["m1_gate_passed"] is True
    assert len(report["m1_path_passing_fractions"]) == 4
    assert len(report["m1_path_mean_spearman"]) == 4
    assert report["m1_spearman_threshold"] == pytest.approx(0.8)
    assert report["m1_minimum_anchor_fraction"] == pytest.approx(0.9)
    for values in (
        report["m1_path_passing_fractions"],
        report["m1_path_mean_spearman"],
    ):
        assert all(isinstance(value, float) for value in values)
