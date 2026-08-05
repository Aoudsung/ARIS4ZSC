"""Regression tests for the closed scientific contracts introduced in v8."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")

from experiments.overcooked_v2.decision_coverage_app import (  # noqa: E402
    _coverage_metrics,
)
from src.path_c.anchor_sampling import (  # noqa: E402
    task_matched_decision_distinction_pair_dataset,
)
from src.path_c.comparator_contract import ComparatorContract  # noqa: E402
from src.path_c.continuation import (  # noqa: E402
    ContinuationContract,
    RAW_REWARD_DEFINITION,
    discounted_reward_increment,
    validate_continuation_contract,
)
from src.path_c.experiment import (  # noqa: E402
    OFFICIAL_SOURCE_COMMIT,
    load_config,
    run_config_from_mapping,
)
from src.path_c.protocol_diagnostics import (  # noqa: E402
    analytic_posterior_memory_curves,
    empirical_protocol_memory_metrics,
)


ROOT = Path(__file__).resolve().parents[3]


def test_registered_continuation_is_discounted_and_masks_terminal_tail() -> None:
    rewards = jnp.asarray([1.0, 1.0, 1.0])
    active = jnp.asarray([True, True, False])
    increments = discounted_reward_increment(
        rewards, active=active, step=jnp.arange(3), gamma=0.5
    )
    np.testing.assert_allclose(np.asarray(increments), [1.0, 0.5, 0.0])


def test_continuation_contract_fails_closed_on_policy_or_estimand_drift() -> None:
    expected = ContinuationContract(
        gamma=0.99, horizon=128, continuation_policy_fingerprint="a" * 64
    )
    validate_continuation_contract(expected.to_mapping(), expected=expected)
    changed = replace(expected, gamma=1.0)
    with pytest.raises(ValueError, match="incompatible"):
        validate_continuation_contract(changed.to_mapping(), expected=expected)


def test_comparator_contract_fingerprint_covers_every_scientific_dimension() -> None:
    contract = ComparatorContract(
        layout="test_time_simple",
        observation_contract="official_l1_l5",
        view_radius=2,
        reward_definition=RAW_REWARD_DEFINITION,
        gamma=0.99,
        continuation_horizon=128,
        probe_steps=16,
        signature_distance_threshold=1.0,
        task_match_epsilon=2.0,
        official_source_commit=OFFICIAL_SOURCE_COMMIT,
        reference_policy_set_hash="1" * 64,
        fit_partner_parent_hash="2" * 64,
        validation_partner_parent_hash="3" * 64,
    )
    assert ComparatorContract.from_mapping(contract.to_mapping()) == contract
    assert replace(contract, gamma=0.98).fingerprint != contract.fingerprint
    assert replace(contract, layout="test_time_wide").fingerprint != contract.fingerprint


def test_task_matching_is_cross_run_reciprocal_and_reports_unmatched_rows() -> None:
    histories = np.arange(24, dtype=np.float64).reshape((6, 4))
    signatures = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.0],
            [3.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.0],
        ]
    )
    tasks = np.asarray([[0.0], [10.0], [0.1], [10.1], [50.0], [80.0]])
    runs = np.asarray(["a", "a", "b", "b", "c", "d"])
    rows, labels, blocks, report = task_matched_decision_distinction_pair_dataset(
        histories,
        signatures,
        tasks,
        runs,
        signature_distance_threshold=1.0,
        task_match_epsilon=0.2,
        episode_times=np.zeros(6, dtype=np.int64),
        recipe_order_states=np.asarray(["r"] * 6),
        ego_roles=np.zeros(6, dtype=np.int64),
    )
    assert rows.shape == (2, 8)
    assert sorted(labels.tolist()) == [0.0, 1.0]
    assert all(left != right for left, right in (block.split("|") for block in blocks))
    assert report["matched_pair_count"] == 2
    assert report["unmatched_fraction"] == pytest.approx(2.0 / 6.0)
    assert report["task_distance_max"] <= 0.2


def test_evidence_gate_preserves_posterior_through_occlusion_diagnostics() -> None:
    curves = analytic_posterior_memory_curves(
        component_count=4, maximum_occlusion_steps=8
    )
    assert curves["curves"]["0.97"]["evidence_gated_deviation_retention"] == [1.0] * 9
    metrics = empirical_protocol_memory_metrics(
        [
            {
                "posterior": [
                    [0.9, 0.1],
                    [0.9, 0.1],
                    [0.9, 0.1],
                    [0.85, 0.15],
                ],
                "evidence_valid": [True, False, False, True],
            }
        ]
    )
    assert metrics["occlusion_length_mean"] == 2.0
    assert metrics["false_switch_rate_during_occlusion"] == 0.0
    assert metrics["reidentification_time_mean"] == 1.0


def test_resolved_config_round_trips_for_self_contained_proxy_deployments() -> None:
    config = load_config(
        ROOT / "experiments/overcooked_v2/configs/depi_simple_development.yaml",
        run_kind="development",
    )
    restored = run_config_from_mapping(config.to_mapping())
    assert restored == config
    assert restored.fingerprint == config.fingerprint


def test_decision_coverage_reports_family_and_state_conditional_gaps() -> None:
    training = {
        "signatures": np.asarray([[1.0, -1.0], [-1.0, 1.0]]),
        "tasks": np.asarray([[0.0], [1.0]]),
        "runs": np.asarray(["train-a", "train-b"]),
        "families": np.asarray(["sp", "op"]),
        "times": np.asarray([4, 4]),
        "regimes": np.asarray(["r", "r"]),
        "roles": np.asarray([0, 0]),
    }
    coverage = {
        "signatures": np.asarray([[1.1, -1.1], [4.0, -4.0]]),
        "tasks": np.asarray([[0.0], [5.0]]),
        "runs": np.asarray(["coverage-a", "coverage-b"]),
        "families": np.asarray(["sa", "fcp"]),
        "times": np.asarray([4, 4]),
        "regimes": np.asarray(["r", "r"]),
        "roles": np.asarray([0, 0]),
    }
    report = _coverage_metrics(
        training, coverage, task_epsilon=0.1, signature_threshold=0.25
    )
    assert report["overall"]["signature_coverage"] == pytest.approx(0.5)
    assert report["overall"]["state_conditional_match_fraction"] == pytest.approx(0.5)
    assert report["maximum_family_coverage_gap"] == pytest.approx(1.0)
