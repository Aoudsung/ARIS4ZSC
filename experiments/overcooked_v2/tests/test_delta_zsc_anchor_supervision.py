"""Anchor supervision tests (METHOD_SPEC §5, REVIEW_AND_SUGGESTION_2026 §5).

Specification entries covered:
- §5 snapshot checklist: the anchor batch carries the five records required
  by the legalized pipeline (policy states, observations, partner identity,
  CRN fit returns, and collection provenance).
- §5.3 frozen comparator: the three-way split uses accuracy thresholds
  0.55 / 0.70 and the signature distance threshold 1.0; irreducible-ambiguity
  pairs never enter a loss and are only recorded.
- §5.2 matched pairs come from different partner runs (the 2AFC dataset only
  admits cross-run pairs).
"""

from __future__ import annotations

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from src.path_c.anchor_sampling import (  # noqa: E402
    FrozenPairComparator,
    classify_matched_pairs,
    fit_pair_comparator,
    pair_comparator_accuracy,
)
from src.path_c.types import (  # noqa: E402
    CounterfactualAnchorBatch,
    QuotientPairBatch,
)


EQUIVALENT_ACCURACY_MAX = 0.55
DISTINCT_ACCURACY_MIN = 0.70
SIGNATURE_DISTANCE_THRESHOLD = 1.0


def test_anchor_snapshot_checklist_covers_the_five_required_records() -> None:
    # §5 legalization requires every anchor row to carry (1) the full DEPI
    # policy state, (2) the observation, (3) partner identity (code/source/
    # run), (4) CRN fit returns per action, and (5) collection provenance.
    fields = CounterfactualAnchorBatch._fields
    assert "policy_states" in fields
    assert "observations" in fields
    for identity_field in ("partner_codes", "partner_sources", "partner_run_ids"):
        assert identity_field in fields
    for return_field in (
        "fit_returns_by_action",
        "return_sum_by_action",
        "return_squared_sum_by_action",
        "replica_count",
    ):
        assert return_field in fields
    for provenance_field in (
        "collection_policy_logits",
        "collection_update",
        "collection_target_fingerprint",
    ):
        assert provenance_field in fields


def test_classify_matched_pairs_three_way_split_boundaries() -> None:
    # §5.3: equivalent needs accuracy <= 0.55 and distance <= 1.0; distinct
    # needs accuracy > 0.70 and distance > 1.0; everything else is
    # irreducible ambiguity (recorded only, never enters a loss).
    accuracy = jnp.asarray([0.50, 0.55, 0.60, 0.70, 0.80, 0.80], dtype=jnp.float32)
    distance = jnp.asarray([0.50, 1.00, 0.50, 2.00, 1.50, 0.50], dtype=jnp.float32)
    classes, equivalent, distinct, fractions = classify_matched_pairs(
        accuracy,
        distance,
        equivalent_accuracy_max=EQUIVALENT_ACCURACY_MAX,
        distinct_accuracy_min=DISTINCT_ACCURACY_MIN,
        signature_threshold=SIGNATURE_DISTANCE_THRESHOLD,
    )
    labels = np.asarray(classes).tolist()
    assert labels == [0, 0, 2, 2, 1, 2]
    assert np.asarray(equivalent).tolist() == [True, True, False, False, False, False]
    assert np.asarray(distinct).tolist() == [False, False, False, False, True, False]
    np.testing.assert_allclose(np.asarray(fractions), [3 / 6, 1 / 6, 2 / 6], atol=1e-6)
    assert float(np.sum(np.asarray(fractions))) == pytest.approx(1.0)


def test_ambiguous_pairs_receive_no_loss_weight_path() -> None:
    # Pairs that satisfy neither criterion must land in class 2; downstream
    # weighting maps class 2 to zero (verified here through the masks).
    classes, equivalent, distinct, _ = classify_matched_pairs(
        jnp.asarray([0.65, 0.90, 0.40], dtype=jnp.float32),
        jnp.asarray([0.90, 0.90, 1.20], dtype=jnp.float32),
        equivalent_accuracy_max=EQUIVALENT_ACCURACY_MAX,
        distinct_accuracy_min=DISTINCT_ACCURACY_MIN,
        signature_threshold=SIGNATURE_DISTANCE_THRESHOLD,
    )
    assert np.asarray(classes).tolist() == [2, 2, 2]
    assert not np.any(np.asarray(equivalent))
    assert not np.any(np.asarray(distinct))


def test_pair_comparator_accuracy_is_order_averaged() -> None:
    comparator = FrozenPairComparator(
        weights=np.asarray([1.0, -1.0], dtype=np.float32),
        bias=np.float32(0.0),
        train_accuracy=np.float32(1.0),
        feature_dim=2,
    )
    # Lane 0 dominates on the learned direction: both presentation orders are
    # answered correctly, so the 2AFC accuracy is 1.0.
    features = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    assert float(pair_comparator_accuracy(comparator, features)) == pytest.approx(1.0)
    swapped = features[::-1]
    assert float(pair_comparator_accuracy(comparator, swapped)) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="two lanes"):
        pair_comparator_accuracy(comparator, np.zeros((3, 2)))


def test_comparator_training_uses_cross_run_pairs_only() -> None:
    # §5.2: matched pairs come from different partner runs.  With a single
    # run the 2AFC dataset is empty and the fit must be rejected.
    features = np.random.default_rng(3).normal(size=(4, 8))
    with pytest.raises(ValueError, match="two partner runs"):
        fit_pair_comparator(
            features, probe_steps=4, partner_run_ids=np.zeros((4,), dtype=np.int64)
        )
    # Separable run signatures: run 0 features large on dimension 0.
    features = np.zeros((6, 8), dtype=np.float64)
    features[:3, 0] = 5.0
    features[3:, 0] = -5.0
    run_ids = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    comparator = fit_pair_comparator(
        features, probe_steps=4, partner_run_ids=run_ids
    )
    assert float(np.asarray(comparator.train_accuracy)) == pytest.approx(1.0)


def test_separation_payload_requires_the_comparator_payload() -> None:
    # Legacy quotient batches without the §5 payload must fail loudly instead
    # of silently producing an empty separation payload.
    from src.path_c.anchor_sampling import separation_terms_from_matched_pairs

    quotient = QuotientPairBatch(
        anchor_index_a=jnp.asarray([0]),
        anchor_index_b=jnp.asarray([1]),
        decision_distance=jnp.asarray([1.5]),
        weights=jnp.asarray([1.0]),
    )
    with pytest.raises(ValueError, match="matched-pair comparator payload"):
        separation_terms_from_matched_pairs(
            quotient=quotient,
            equivalent_accuracy_max=EQUIVALENT_ACCURACY_MAX,
            distinct_accuracy_min=DISTINCT_ACCURACY_MIN,
            signature_threshold=SIGNATURE_DISTANCE_THRESHOLD,
            margin_scale=0.25,
        )


def test_separation_payload_carries_classification_not_a_loss_scalar() -> None:
    # §3.2 gradient path: the payload carries the ego states/observations and
    # the precomputed classification so compute_loss can re-run the forward
    # passes on the current params; no eager loss scalar is produced here.
    from src.path_c.anchor_sampling import separation_terms_from_matched_pairs
    from src.path_c.types import SeparationTerms

    quotient = QuotientPairBatch(
        anchor_index_a=jnp.asarray([0, 1, 2]),
        anchor_index_b=jnp.asarray([3, 4, 5]),
        # pair 0: equivalent (acc<=0.55, dist<=1.0)
        # pair 1: distinct (acc>0.70, dist>1.0)
        # pair 2: irreducible ambiguity (zero weight)
        decision_distance=jnp.asarray([0.5, 2.0, 0.8]),
        weights=jnp.asarray([1.0, 1.0, 1.0]),
        comparator_accuracy=jnp.asarray([0.50, 0.85, 0.65]),
        ego_state_a="STATE_A",
        ego_state_b="STATE_B",
        probe_observations=("OBS_A", "OBS_B"),
    )
    terms, readings = separation_terms_from_matched_pairs(
        quotient=quotient,
        equivalent_accuracy_max=EQUIVALENT_ACCURACY_MAX,
        distinct_accuracy_min=DISTINCT_ACCURACY_MIN,
        signature_threshold=SIGNATURE_DISTANCE_THRESHOLD,
        margin_scale=0.25,
    )
    assert isinstance(terms, SeparationTerms)
    assert terms.ego_state_a == "STATE_A"
    assert terms.ego_state_b == "STATE_B"
    assert terms.probe_observations == ("OBS_A", "OBS_B")
    assert np.asarray(terms.equivalent_mask).tolist() == [True, False, False]
    # Weights: equivalent -> 1.0, distinct -> decision distance, ambiguous -> 0.
    np.testing.assert_allclose(
        np.asarray(terms.weights), [1.0, 2.0, 0.0], atol=1e-6
    )
    # margin = margin_scale x mean distinct signature distance.
    assert float(np.asarray(terms.margin)) == pytest.approx(0.25 * 2.0)
    assert float(np.asarray(readings["separation_margin"])) == pytest.approx(0.5)
    assert float(np.asarray(readings["irreducible_ambiguity_fraction"])) == (
        pytest.approx(1.0 / 3.0)
    )
