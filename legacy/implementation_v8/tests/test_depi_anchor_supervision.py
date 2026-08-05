"""Decision-regime comparator and matched-pair supervision invariants."""

from __future__ import annotations

import numpy as np
import pytest

JAX = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from src.path_c.anchor_sampling import (  # noqa: E402
    PAIR_AMBIGUOUS,
    PAIR_DISTINCT,
    PAIR_EQUIVALENT,
    PAIR_FRACTION_ORDER,
    classify_matched_pairs,
    decision_distinction_pair_dataset,
    fit_pair_comparator,
    mask_supervision_partner_runs,
    manifest_partitioned_candidate_pairs,
    pair_feature_rows,
    predict_pair_comparator,
    run_disjoint_candidate_pairs,
    separation_terms_from_matched_pairs,
    time_source_stratified_indexes,
)
from src.path_c.types import CounterfactualAnchorBatch, QuotientPairBatch  # noqa: E402


def test_matched_candidate_stratification_preserves_partner_run_diversity() -> None:
    run_ids = jnp.asarray(
        [
            [10_000, 10_001, 10_002, 10_000],
            [10_000, 10_001, 10_002, 10_000],
            [10_000, 10_001, 10_002, 10_000],
            [10_000, 10_001, 10_002, 10_000],
        ]
    )
    indexes = time_source_stratified_indexes(
        JAX.random.PRNGKey(7),
        time_count=4,
        environment_count=4,
        requested=4,
        source_values=run_ids,
    )
    selected = np.asarray(run_ids).reshape((-1,))[np.asarray(indexes)]
    assert np.unique(selected).size >= 2


def test_anchor_rows_carry_scientific_identity_returns_and_provenance() -> None:
    fields = set(CounterfactualAnchorBatch._fields)
    assert {
        "policy_states",
        "observations",
        "partner_sources",
        "partner_members",
        "partner_family_ids",
        "partner_checkpoint_stages",
        "partner_run_ids",
        "fit_returns_by_action",
        "evaluation_returns_by_action",
        "return_sum_by_action",
        "return_squared_sum_by_action",
        "replica_count",
        "collection_policy_logits",
        "collection_update",
        "collection_target_fingerprint",
    } <= fields
    assert "partner_codes" not in fields


def test_frozen_comparator_target_is_direct_k_independent_signature_distance() -> None:
    history = np.asarray([[0.0], [0.1], [2.0], [2.1]])
    signatures = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.0, 0.0, -0.1],
            [2.0, -2.0, 0.0, 0.0, 0.0, 0.0],
            [2.1, -2.0, 0.0, 0.0, 0.0, -0.1],
        ]
    )
    rows, labels, blocks = decision_distinction_pair_dataset(
        history,
        signatures,
        signature_distance_threshold=1.0,
        block_ids=np.asarray([10, 11, 20, 21]),
    )
    assert rows.shape == (6, 2)
    assert labels.shape == blocks.shape == (6,)
    expected = []
    for left in range(4):
        for right in range(left + 1, 4):
            expected.append(float(np.linalg.norm(signatures[left] - signatures[right]) > 1.0))
    np.testing.assert_array_equal(labels, np.asarray(expected))


def _fitted_comparator():
    train_history = np.asarray([[0.0, 0.0], [0.2, 0.1], [3.0, 3.0], [3.2, 3.1]])
    train_signatures = np.asarray([[0.0, 0.0], [0.1, -0.1], [2.0, -2.0], [2.1, -2.1]])
    validation_history = np.asarray(
        [[0.1, -0.1], [0.3, 0.0], [2.9, 3.1], [3.1, 2.9]]
    )
    validation_signatures = np.asarray(
        [[0.0, 0.1], [0.1, 0.0], [1.9, -2.0], [2.0, -2.1]]
    )
    train_rows, train_labels, train_blocks = decision_distinction_pair_dataset(
        train_history,
        train_signatures,
        signature_distance_threshold=1.0,
        block_ids=np.asarray([1, 1, 2, 2]),
    )
    validation_rows, validation_labels, validation_blocks = (
        decision_distinction_pair_dataset(
            validation_history,
            validation_signatures,
            signature_distance_threshold=1.0,
            block_ids=np.asarray([3, 3, 4, 4]),
            require_both_classes=True,
        )
    )
    return fit_pair_comparator(
        train_rows,
        train_labels,
        validation_pair_features=validation_rows,
        validation_pair_labels=validation_labels,
        history_feature_dim=2,
        training_block_ids=train_blocks,
        validation_block_ids=validation_blocks,
        bootstrap_replicates=64,
        logistic_iterations=200,
    )


def test_frozen_comparator_has_explicit_dimensions_and_continuous_probability() -> None:
    comparator = _fitted_comparator()
    assert comparator.history_feature_dim == 2
    assert comparator.pair_feature_dim == 4
    probabilities = predict_pair_comparator(
        comparator,
        pair_feature_rows(np.asarray([[0.0, 0.0]]), np.asarray([[2.0, 2.0]])),
    )
    assert probabilities.shape == (1,)
    assert 0.0 < float(probabilities[0]) < 1.0


def test_comparator_rejects_partner_run_overlap_between_fit_and_validation() -> None:
    rows = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    labels = np.asarray([0.0, 1.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="overlap"):
        fit_pair_comparator(
            rows,
            labels,
            validation_pair_features=rows,
            validation_pair_labels=labels,
            history_feature_dim=1,
            training_block_ids=np.asarray(["1|2"] * 4),
            validation_block_ids=np.asarray(["2|3"] * 4),
            bootstrap_replicates=8,
        )


def test_comparator_development_runs_are_excluded_from_all_depi_supervision() -> None:
    mask = jnp.asarray(
        [
            [True, True, True],
            [True, False, True],
            [False, True, True],
            [True, True, True],
        ]
    )
    filtered = mask_supervision_partner_runs(
        mask,
        jnp.asarray([10, 20, 30, 40]),
        jnp.asarray([20, 40]),
    )
    np.testing.assert_array_equal(
        np.asarray(filtered),
        np.asarray(
            [
                [True, True, True],
                [False, False, False],
                [False, True, True],
                [False, False, False],
            ]
        ),
    )


def test_fixed_pair_budget_partitions_comparator_and_supervision_runs() -> None:
    run_ids = np.repeat(np.arange(1, 9), 3)
    features = np.stack(
        (run_ids.astype(np.float64), np.tile(np.arange(3), 8) / 10.0), axis=-1
    )
    pairs = run_disjoint_candidate_pairs(
        features,
        run_ids,
        pair_count=6,
        development_pair_count=3,
    )
    assert pairs.shape == (6, 2)
    development_runs = set(run_ids[pairs[:3].reshape((-1,))].tolist())
    supervision_runs = set(run_ids[pairs[3:].reshape((-1,))].tolist())
    assert len(development_runs) >= 4
    assert development_runs.isdisjoint(supervision_runs)


def test_frozen_comparator_runs_are_excluded_before_fixed_pairing() -> None:
    run_ids = np.repeat(np.arange(1, 7), 2)
    features = np.stack(
        (run_ids.astype(np.float64), np.tile(np.arange(2), 6)), axis=-1
    )
    pairs = run_disjoint_candidate_pairs(
        features,
        run_ids,
        pair_count=3,
        excluded_run_ids=(1, 2),
    )
    assert pairs.shape == (3, 2)
    assert set(run_ids[pairs.reshape((-1,))].tolist()).isdisjoint({1, 2})


def test_manifest_roles_form_fit_validation_and_support_pairs_without_overlap() -> None:
    run_ids = np.repeat(np.arange(12), 4)
    partitions = np.repeat(
        np.asarray([0, 0, 0, 0, 1, 1, 2, 2, 0, 0, 0, 0]), 4
    )
    features = np.stack(
        (run_ids.astype(np.float64), np.tile(np.arange(4), 12) / 10.0), axis=-1
    )
    pairs = manifest_partitioned_candidate_pairs(
        features,
        run_ids,
        partitions,
        pair_count=8,
        comparator_is_frozen=False,
    )
    assert pairs.shape == (8, 2)
    np.testing.assert_array_equal(partitions[pairs[:2]], 1)
    np.testing.assert_array_equal(partitions[pairs[2:4]], 2)
    np.testing.assert_array_equal(partitions[pairs[4:]], 0)
    fit_runs = set(run_ids[pairs[:2].reshape((-1,))])
    validation_runs = set(run_ids[pairs[2:4].reshape((-1,))])
    support_runs = set(run_ids[pairs[4:].reshape((-1,))])
    assert fit_runs.isdisjoint(validation_runs | support_runs)
    assert validation_runs.isdisjoint(support_runs)

    frozen = manifest_partitioned_candidate_pairs(
        features,
        run_ids,
        partitions,
        pair_count=4,
        comparator_is_frozen=True,
    )
    np.testing.assert_array_equal(partitions[frozen], 0)


def test_pair_class_contract_and_invalid_mask_are_frozen() -> None:
    assert PAIR_FRACTION_ORDER == ("equivalent", "distinct", "ambiguous")
    classes, equivalent, distinct, fractions = classify_matched_pairs(
        jnp.asarray([0.50, 0.80, 0.60, 0.90]),
        jnp.asarray([0.50, 2.00, 0.50, 2.00]),
        equivalent_probability_max=0.55,
        distinct_probability_min=0.70,
        signature_threshold=1.0,
        pair_valid=jnp.asarray([True, True, True, False]),
    )
    assert np.asarray(classes).tolist() == [
        PAIR_EQUIVALENT,
        PAIR_DISTINCT,
        PAIR_AMBIGUOUS,
        PAIR_AMBIGUOUS,
    ]
    assert np.asarray(equivalent).tolist() == [True, False, False, False]
    assert np.asarray(distinct).tolist() == [False, True, False, False]
    np.testing.assert_allclose(np.asarray(fractions), [1 / 3, 1 / 3, 1 / 3])


def test_ambiguous_and_invalid_pairs_receive_zero_separation_weight() -> None:
    quotient = QuotientPairBatch(
        anchor_index_a=jnp.asarray([0, 1, 2, 3]),
        anchor_index_b=jnp.asarray([4, 5, 6, 7]),
        decision_distance=jnp.asarray([0.5, 2.0, 0.8, 2.0]),
        weights=jnp.ones((4,)),
        comparator_distinct_probability=jnp.asarray([0.5, 0.8, 0.6, 0.9]),
        ego_state_a="a",
        ego_state_b="b",
        probe_observations=("a", "b"),
        pair_valid=jnp.asarray([True, True, True, False]),
    )
    terms, readings = separation_terms_from_matched_pairs(
        quotient=quotient,
        equivalent_probability_max=0.55,
        distinct_probability_min=0.70,
        signature_threshold=1.0,
        margin=0.5,
    )
    np.testing.assert_allclose(np.asarray(terms.weights), [1.0, 2.0, 0.0, 0.0])
    assert float(terms.margin) == pytest.approx(0.5)
    assert float(readings["irreducible_ambiguity_fraction"]) == pytest.approx(1 / 3)
