"""Independent extended M1 diagnostic tests (METHOD_SPEC §7).

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
    evaluate_m1_gate_on_anchor_batch,
    initialize_bootstrap_value_ensemble,
    m1_anchor_gate,
    m1_gate_report,
    spearman_rank_correlation,
    train_bootstrap_value_ensemble,
)
from src.path_c.model import initial_policy_state  # noqa: E402
from src.path_c.storage import pytree_fingerprint  # noqa: E402


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


def test_spearman_uses_average_ranks_for_exact_ties() -> None:
    values = jnp.asarray([[1.0, 1.0, 3.0, 3.0]])
    assert float(spearman_rank_correlation(values, values)[0]) == pytest.approx(1.0)


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
    np.testing.assert_allclose(result.path_top_action_agreement, np.ones(4))
    np.testing.assert_allclose(result.path_mean_value_regret, np.zeros(4))


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


def test_bootstrap_members_have_independent_parameters_optimizers_and_counters() -> None:
    state = initial_policy_state(
        batch_size=2,
        observation_shape=(5, 5, 39),
        action_count=6,
        task_hidden_dim=8,
        capability_hidden_dim=8,
        capability_dim=4,
        component_embedding_dim=4,
        protocol_components=4,
    )
    models, params, optimizer_states, counters = initialize_bootstrap_value_ensemble(
        key=jax.random.PRNGKey(71),
        example_policy_state=state,
        example_observation=jnp.zeros((2, 5, 5, 39)),
    )
    assert len(models) == len(params) == len(optimizer_states) == 3
    assert len({pytree_fingerprint(value) for value in params}) == 3
    assert all(left is not right for left, right in zip(optimizer_states, optimizer_states[1:]))
    np.testing.assert_array_equal(np.asarray(counters), [0, 0, 0])


def test_bootstrap_members_train_independently_and_increment_sampling_counters() -> None:
    state = initial_policy_state(
        batch_size=4,
        observation_shape=(5, 5, 39),
        action_count=6,
        task_hidden_dim=8,
        capability_hidden_dim=8,
        capability_dim=4,
        component_embedding_dim=4,
        protocol_components=4,
    )
    observations = jax.random.normal(jax.random.PRNGKey(72), (4, 5, 5, 39))
    models, params, optimizer_states, counters = initialize_bootstrap_value_ensemble(
        key=jax.random.PRNGKey(73),
        example_policy_state=state,
        example_observation=observations,
    )
    anchors = type("Anchors", (), {})()
    anchors.policy_states = state
    anchors.observations = observations
    anchors.fit_returns_by_action = jnp.asarray(
        [[6, 5, 4, 3, 2, 1], [1, 2, 3, 4, 5, 6]] * 2,
        dtype=jnp.float32,
    )
    anchors.action_mask = jnp.ones((4, 6), dtype=jnp.bool_)
    before = tuple(pytree_fingerprint(value) for value in params)
    next_params, next_states, next_counters, metrics = train_bootstrap_value_ensemble(
        models=models,
        params=params,
        optimizer_states=optimizer_states,
        sampling_counters=counters,
        anchors=anchors,
        key=jax.random.PRNGKey(74),
        steps=3,
    )
    assert all(
        pytree_fingerprint(value) != fingerprint
        for value, fingerprint in zip(next_params, before, strict=True)
    )
    assert len({pytree_fingerprint(value) for value in next_params}) == 3
    assert all(left is not right for left, right in zip(next_states, next_states[1:]))
    np.testing.assert_array_equal(np.asarray(next_counters), [3, 3, 3])
    assert np.all(np.isfinite(np.asarray(metrics["bootstrap_training_loss"])))


def test_m1_anchor_gate_uses_evaluation_not_fit_replicas() -> None:
    predicted = jnp.arange(6, dtype=jnp.float32)[None].repeat(2, axis=0)

    class FakeModel:
        step = object()
        action_values_from_features_and_context = object()

        def apply(self, variables, *args, method):
            del variables
            if method is self.step:
                count = int(np.asarray(args[1]).shape[0])
                output = type("Output", (), {})()
                output.capability = jnp.zeros((count, 1))
                output.protocol_embedding = jnp.zeros((count, 1))
                output.task_features = jnp.zeros((count, 1))
                return args[0], output
            return predicted

    class FakeMember:
        def apply(self, variables, features):
            del variables, features
            return predicted

    anchors = type("Anchors", (), {})()
    anchors.anchor_ids = jnp.arange(2)
    anchors.policy_states = initial_policy_state(
        batch_size=2,
        observation_shape=(5, 5, 39),
        action_count=6,
        task_hidden_dim=1,
        capability_hidden_dim=1,
        capability_dim=1,
        component_embedding_dim=1,
        protocol_components=2,
    )
    anchors.observations = jnp.zeros((2, 5, 5, 39))
    anchors.fit_returns_by_action = -predicted
    anchors.evaluation_returns_by_action = predicted
    anchors.evaluation_replica_count = jnp.full((2, 6), 8)
    anchors.action_mask = jnp.ones((2, 6), dtype=jnp.bool_)
    result = evaluate_m1_gate_on_anchor_batch(
        model=FakeModel(),
        params={},
        bootstrap_models=(FakeMember(), FakeMember(), FakeMember()),
        bootstrap_params=({}, {}, {}),
        anchors=anchors,
    )
    assert bool(np.asarray(result.passed))
