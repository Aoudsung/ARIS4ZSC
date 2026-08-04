"""Executable matrix identity and mechanism-attribution schema tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from experiments.overcooked_v2.comparator_app import (  # noqa: E402
    MINIMUM_HISTORIES_PER_SPLIT,
    _rows as comparator_source_rows,
)
from experiments.overcooked_v2.development_matrix_app import (  # noqa: E402
    CORE_DEVELOPMENT_VARIANTS,
    DEVELOPMENT_VARIANTS,
    _validate_development_raw_rows,
    validate_development_entry_alignment,
)
from experiments.overcooked_v2.formal_claim_app import (  # noqa: E402
    _validate_common_summary,
    _validate_development_matrix,
    _validate_final_m1_from_official_summary,
    _validate_layout_artifact,
    _validate_posterior_calibration,
)
from experiments.overcooked_v2.official_evaluation_app import (  # noqa: E402
    _load_policy_manifest,
)
from src.path_c.experiment import (  # noqa: E402
    MECHANISM_ABLATION_VARIANTS,
    METHOD_VERSION,
    OFFICIAL_SOURCE_COMMIT,
)
from src.path_c.component_diagnostics import (  # noqa: E402
    best_permutation_alignment,
    component_intervention_summary,
    permutation_aligned_component_stability,
    validate_component_diagnostic_values,
)
from src.path_c.identifiability_controls import (  # noqa: E402
    continuation_swap_causal_consistency,
    history_shuffle_negative,
    match_different_partner_task_states,
    paired_bootstrap_drop,
    task_representation_drift_under_shuffle,
)
from src.path_c.model import (  # noqa: E402
    build_model,
    initial_policy_state,
    initialize_model_parameters,
)
from src.path_c.resources import parameter_count  # noqa: E402
from src.path_c.storage import sha256_path  # noqa: E402


def test_comparator_source_rows_require_independent_replica_signatures() -> None:
    rows = []
    for index in range(MINIMUM_HISTORIES_PER_SPLIT):
        rows.append(
            {
                "history_features": [float(index % 3), float(index % 5)],
                "fit_returns_by_action": [float(index % 6 == action) for action in range(6)],
                "evaluation_returns_by_action": [
                    float((index + 1) % 6 == action) for action in range(6)
                ],
                "partner_run_id": f"run-{index % 8}",
            }
        )
    features, fit_signatures, evaluation_signatures, runs = comparator_source_rows(
        rows, label="fixture"
    )
    assert features.shape == (MINIMUM_HISTORIES_PER_SPLIT, 2)
    assert fit_signatures.shape == evaluation_signatures.shape == (
        MINIMUM_HISTORIES_PER_SPLIT,
        6,
    )
    assert np.any(fit_signatures != evaluation_signatures)
    assert np.unique(runs).size == 8


def test_development_raw_rows_require_every_ordered_pair_and_episode_once() -> None:
    schedule = "a" * 64
    rows = [
        {
            "variant": "b2",
            "protocol_components": 4,
            "left_seed_index": left,
            "right_seed_index": right,
            "left_role": 0,
            "right_role": 1,
            "episode_index": episode,
            "raw_return": float(left - right),
            "evaluation_key_schedule_sha256": schedule,
        }
        for left in range(10)
        for right in range(10)
        for episode in range(500)
    ]
    scores = _validate_development_raw_rows(
        rows,
        variant="b2",
        protocol_components=4,
        seed_indexes=range(10),
        schedule_sha256=schedule,
    )
    assert set(scores) == set(range(10))
    broken = list(rows)
    broken[-1] = dict(broken[-2])
    with pytest.raises(ValueError, match="duplicated"):
        _validate_development_raw_rows(
            broken,
            variant="b2",
            protocol_components=4,
            seed_indexes=range(10),
            schedule_sha256=schedule,
        )


def _model_params(variant: str):
    model = build_model(
        observation_shape=(5, 5, 39),
        action_count=6,
        task_hidden_dim=8,
        capability_hidden_dim=8,
        capability_dim=4,
        protocol_components=4,
        component_embedding_dim=4,
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        response_hidden_dim=8,
        modulation_rank=2,
        action_embedding_dim=4,
        method_variant=variant,
    )
    state = initial_policy_state(
        batch_size=1,
        observation_shape=(5, 5, 39),
        action_count=6,
        task_hidden_dim=8,
        capability_hidden_dim=8,
        capability_dim=4,
        component_embedding_dim=4,
        protocol_components=4,
    )
    params = initialize_model_parameters(
        model,
        key=jax.random.PRNGKey(5),
        example_state=state,
        example_observation=jnp.zeros((1, 5, 5, 39)),
    )
    return params


def test_b0_b1_b2_are_first_class_and_capacity_matched_while_b3_fails() -> None:
    assert CORE_DEVELOPMENT_VARIANTS == ("r0", "b0", "b1", "b2")
    assert DEVELOPMENT_VARIANTS == (
        "r0",
        "b0",
        "b1",
        "b2",
        "r0_extra",
        "b0_extra",
        "b1_extra",
        *MECHANISM_ABLATION_VARIANTS,
    )
    counts = [
        parameter_count(_model_params(variant))
        for variant in (*CORE_DEVELOPMENT_VARIANTS, *MECHANISM_ABLATION_VARIANTS)
    ]
    assert len(set(counts)) == 1
    with pytest.raises(NotImplementedError, match="B3"):
        build_model(
            observation_shape=(5, 5, 39),
            action_count=6,
            task_hidden_dim=8,
            capability_hidden_dim=8,
            capability_dim=4,
            protocol_components=4,
            component_embedding_dim=4,
            actor_hidden_dim=8,
            critic_hidden_dim=8,
            response_hidden_dim=8,
            modulation_rank=2,
            action_embedding_dim=4,
            method_variant="b3",
        )


def test_identifiability_primitives_are_paired_and_directional() -> None:
    features = np.asarray([[0.0], [0.1], [1.0], [1.1]])
    run_ids = np.asarray([0, 1, 0, 1])
    source, donor, epsilon, distances = match_different_partner_task_states(
        features, run_ids, epsilon_quantile=1.0
    )
    assert source.shape == donor.shape == distances.shape
    assert np.all(run_ids[source] != run_ids[donor])
    assert epsilon >= float(np.max(distances))
    assert float(task_representation_drift_under_shuffle(features, features)) == 0.0
    drop, low, high = paired_bootstrap_drop(
        np.asarray([3.0, 4.0, 5.0]),
        np.asarray([1.0, 2.0, 3.0]),
        replications=100,
        seed=1,
    )
    assert (drop, low, high) == pytest.approx((2.0, 2.0, 2.0))
    assert not history_shuffle_negative(drop, low, high)
    consistency = continuation_swap_causal_consistency(
        swapped_returns_by_action=np.asarray([[0.0, 3.0]]),
        original_returns_by_action=np.asarray([[0.0, 1.0]]),
        source_signature=np.asarray([[0.0, 1.0]]),
        target_signature=np.asarray([[0.0, 4.0]]),
    )
    assert consistency == 1.0


def test_component_diagnostics_are_permutation_invariant_and_interventional() -> None:
    reference = np.asarray(
        [[1.0, 0.0, -1.0], [0.0, 2.0, -2.0], [-3.0, 1.0, 2.0]]
    )
    candidate = reference[[2, 0, 1]]
    order, error = best_permutation_alignment(reference, candidate)
    assert order == (1, 2, 0)
    assert error == 0.0
    stability = permutation_aligned_component_stability(
        {0: reference, 1: candidate}
    )
    assert stability["permutation_aligned_rmse"] == 0.0
    assert stability["permutation_aligned_stability"] == 1.0

    summary = component_intervention_summary(
        action_signatures=np.stack((reference, reference + 0.5), axis=0),
        policy_logits=np.asarray(
            [
                [[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 6.0]],
                [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
            ]
        ),
        protocol_probabilities=np.asarray(
            [[0.2, 0.3, 0.5], [0.4, 0.4, 0.2]]
        ),
    )
    assert len(summary["component_action_signatures"]) == 3
    assert sum(summary["component_utilization"]) == pytest.approx(1.0)
    assert summary["mean_pairwise_action_signature_divergence"] > 0.0
    assert summary["mean_pairwise_one_hot_actor_tv"] > 0.9
    assert summary["one_hot_top_action_disagreement_fraction"] == 1.0
    complete = {
        **summary,
        "mean_pairwise_response_divergence_all_actions": 0.25,
    }
    validate_component_diagnostic_values(complete, component_count=3, action_count=3)
    with pytest.raises(ValueError, match="probabilities"):
        validate_component_diagnostic_values(
            {**complete, "component_utilization": [1.0, 1.0, 1.0]},
            component_count=3,
            action_count=3,
        )


def test_development_matrix_rejects_budget_sampler_capacity_and_key_drift() -> None:
    entries = [
        {
            "variant": variant,
            "protocol_components": 4,
            "seed_index": 0,
            "partner_sampler_sha256": "same",
            "total_training_simulator_steps": 120 if (
                variant in {"r0_extra", "b0_extra", "b1_extra"}
                or variant in {"b2", "decision_only", "q_only", "actor_only", "no_separation", "no_capability"}
            ) else 100,
            "ppo_training_steps": (
                120 if variant in {"r0_extra", "b0_extra", "b1_extra"} else 100
            ),
            "auxiliary_training_steps": 20 if variant in {
                "b2",
                "decision_only",
                "q_only",
                "actor_only",
                "no_separation",
                "no_capability",
            } else 0,
            "deployable_parameters": 200,
            "episode_key_domains": {"evaluation": [1, 2]},
        }
        for variant in DEVELOPMENT_VARIANTS
    ]
    validate_development_entry_alignment(entries)
    for field, changed in (
        ("partner_sampler_sha256", "different"),
        ("deployable_parameters", 201),
        ("episode_key_domains", {"evaluation": [2, 1]}),
    ):
        broken = [dict(row) for row in entries]
        broken[-1][field] = changed
        with pytest.raises(RuntimeError, match="differs"):
            validate_development_entry_alignment(broken)
    broken = [dict(row) for row in entries]
    broken[-1]["total_training_simulator_steps"] = 121
    with pytest.raises(RuntimeError, match="total simulator budgets differ"):
        validate_development_entry_alignment(broken)
    broken = [dict(row) for row in entries]
    next(row for row in broken if row["variant"] == "b1_extra")[
        "ppo_training_steps"
    ] = 121
    with pytest.raises(RuntimeError, match="exactly replace"):
        validate_development_entry_alignment(broken)


def test_formal_claim_rejects_handwritten_scores_and_validates_mechanism_artifacts(
    tmp_path: Path,
) -> None:
    matrix_path = tmp_path / "matrix.json"
    scores_path = tmp_path / "scores.json"
    matrix_path.write_text(
        json.dumps(
            {
                "artifact_type": "depi_development_matrix",
                "method": METHOD_VERSION,
                "budget_capacity_and_key_matching_passed": True,
            }
        )
    )
    scores_path.write_text(json.dumps({"artifact_type": "depi_development_scores"}))
    development = {
        "version": 1,
        "artifact_type": "depi_development_matrix_summary",
        "method": METHOD_VERSION,
        "b3_status": "not_implemented",
        "primary_k4_paired_increments": {
            name: {"bootstrap_99_percent_ci": [1.0, 3.0]}
            for name in ("b1_minus_b0", "b2_minus_b1", "b2_minus_b0")
        },
        "sources": {
            "matrix": {"path": str(matrix_path), "sha256": sha256_path(matrix_path)},
            "scores": {"path": str(scores_path), "sha256": sha256_path(scores_path)},
        },
    }
    with pytest.raises(ValueError, match="Development-matrix summary identity"):
        _validate_development_matrix(development)

    common = {
        "method": METHOD_VERSION,
        "layout": "test_time_simple",
        "version": 3,
        "method_variant": "b2",
        "paired_crn": True,
        "resource_ledger": {"continuation_steps": 10, "total_simulator_steps": 20},
        "policy_manifest_sha256": "policy",
        "partner_manifest_sha256": "partner",
    }
    ident_raw_path = tmp_path / "ident_raw.json"
    recoverable_raw_path = tmp_path / "recoverable_raw.json"
    raw_common = {
        "method": METHOD_VERSION,
        "method_variant": "b2",
        "layout": "test_time_simple",
        "policy_manifest_sha256": "policy",
        "partner_manifest_sha256": "partner",
    }
    ident_raw_path.write_text(
        json.dumps({**raw_common, "artifact_type": "depi_identifiability_raw"})
    )
    recoverable_raw_path.write_text(
        json.dumps({**raw_common, "artifact_type": "depi_recoverable_value_raw"})
    )
    identifiability = {
        **common,
        "artifact_type": "depi_identifiability_evaluation",
        "task_leakage": {
            "representation_shuffle_drift": 0.0,
            "learned_representation_balanced_accuracy": 0.55,
            "task_state_baseline_balanced_accuracy": 0.52,
            "excess_balanced_accuracy": 0.03,
            "chance_accuracy": 0.5,
            "maximum_excess_over_task_state": 0.05,
        },
        "protocol_state_transplant": {
            "mean_return_drop": 2.0,
            "bootstrap_99_percent_ci": [1.0, 3.0],
        },
        "context_swap": {
            "measurement": "real_crn_all_action_continuation",
            "source_world_value_alignment_bootstrap_99_percent_ci": [0.5, 2.0],
        },
        "pass": {
            "task_excess_leakage": True,
            "protocol_state_transplant": True,
            "source_world_context_value": True,
            "overall": True,
        },
        "source": {
            "path": str(ident_raw_path),
            "sha256": sha256_path(ident_raw_path),
        },
    }
    recoverable = {
        **common,
        "artifact_type": "depi_recoverable_value_evaluation",
        "recoverable_fraction": {"estimable": True, "signal_threshold": 20.0},
        "comparisons": {
            "g1_minus_g2": {"bootstrap_99_percent_ci": [1.0, 2.0]},
            "g1_minus_g3": {"bootstrap_99_percent_ci": [1.0, 2.0]},
            "g4_minus_g1": {"point_difference": 0.0},
        },
        "cross_fitted_proxy": {
            "mean_top_action_selection_stability": 0.75,
            "not_a_mathematical_upper_bound": True,
        },
        "pass": {
            "legal_history_beats_shuffled": True,
            "legal_history_beats_state_only": True,
            "recoverable_fraction_estimable": True,
            "overall": True,
        },
        "source": {
            "path": str(recoverable_raw_path),
            "sha256": sha256_path(recoverable_raw_path),
        },
    }
    assert _validate_layout_artifact(
        identifiability,
        layout="test_time_simple",
        artifact_type="depi_identifiability_evaluation",
    )
    assert _validate_layout_artifact(
        recoverable,
        layout="test_time_simple",
        artifact_type="depi_recoverable_value_evaluation",
    )
    with pytest.raises(ValueError, match="simulator ledger"):
        _validate_layout_artifact(
            {**identifiability, "resource_ledger": {"continuation_steps": 0}},
            layout="test_time_simple",
            artifact_type="depi_identifiability_evaluation",
        )


def test_formal_claim_rejects_handwritten_common_and_calibration_booleans() -> None:
    with pytest.raises(ValueError, match="source tree"):
        _validate_common_summary(
            {
                "version": 2,
                "artifact_type": "depi_common_partner_evaluation",
                "method": METHOD_VERSION,
                "method_variant": "b2",
                "layout": "test_time_simple",
                "official_protocol_version": "overcooked_v2_iclr2025_5ce1707_v1",
                "official_source_commit": "5ce1707cf31c1c115e6f6ba96db7bc9cc80a850e",
                "paired_crn": True,
                "episode_key_schedule_sha256": "0" * 64,
                "sources": {},
                "common_partner_gate_passed": True,
                "br_prox_complete": True,
            },
            layout="test_time_simple",
        )
    with pytest.raises(ValueError, match="source tree"):
        _validate_posterior_calibration(
            {
                "version": 2,
                "artifact_type": "depi_posterior_calibration",
                "artifact_name": "DEPI-Posterior-Calibration",
                "method": METHOD_VERSION,
                "method_variant": "b2",
                "layout": "test_time_simple",
                "official_protocol_version": "overcooked_v2_iclr2025_5ce1707_v1",
                "official_source_commit": "5ce1707cf31c1c115e6f6ba96db7bc9cc80a850e",
                "sources": {},
                "pass": {"overall": True},
            },
            layout="test_time_simple",
        )


def _write_depi_manifest_with_final_m1(root: Path, *, layout: str) -> Path:
    runs = []
    m1_rows = []
    lineage = []
    for index in range(10):
        run_id = f"{layout}-run-{index}"
        fingerprint = f"params-{layout}-{index}"
        deployment = root / f"deployment-{index}"
        deployment.mkdir(parents=True)
        (deployment / "deployment_bundle.json").write_text(
            json.dumps({"params_fingerprint": fingerprint}), encoding="utf-8"
        )
        m1_path = root / f"m1-{index}.json"
        final = {
            "update": 10,
            "final_checkpoint_condition": True,
            "model_fingerprint": fingerprint,
            "deployment_params_fingerprint": fingerprint,
            "anchor_collection_policy_fingerprint": fingerprint,
            "continuation_policy_fingerprint": fingerprint,
            "partner_panel_fingerprint": "a" * 64,
            "fit_key_domain": {
                "name": "audit_anchor/final_fit",
                "root": [1, 2],
                "lane_key_derivation": "fold_in(root,100003),fold_in(replica_index)",
                "replica_index_range": [0, 4],
            },
            "evaluation_key_domain": {
                "name": "audit_anchor/final_evaluation",
                "root": [1, 2],
                "lane_key_derivation": "fold_in(root,100003),fold_in(replica_index)",
                "replica_index_range": [4, 12],
            },
            "fresh_final_anchor": True,
            "bootstrap_reinitialized_after_deployment_freeze": True,
            "bootstrap_training": {"loss": 0.1},
            "m1_gate_passed": True,
            "m1_path_passing_fractions": [0.9, 0.9, 0.9, 0.9],
            "m1_path_mean_spearman": [0.8, 0.8, 0.8, 0.8],
            "m1_path_top_action_agreement": [1.0, 1.0, 1.0, 1.0],
            "m1_path_mean_value_regret": [0.0, 0.0, 0.0, 0.0],
            "m1_spearman_threshold": 0.8,
            "m1_minimum_anchor_fraction": 0.9,
        }
        m1_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "artifact_type": "depi_final_m1_evaluation",
                    "method": METHOD_VERSION,
                    "method_variant": "b2",
                    "layout": layout,
                    "seed_index": index,
                    "specification": "METHOD_SPEC §7 final-checkpoint M1 gate",
                    "history": [
                        {
                            "update": 10,
                            "passed": True,
                            "path_passing_fractions": [0.9, 0.9, 0.9, 0.9],
                            "path_mean_spearman": [0.8, 0.8, 0.8, 0.8],
                            "path_top_action_agreement": [1.0, 1.0, 1.0, 1.0],
                            "path_mean_value_regret": [0.0, 0.0, 0.0, 0.0],
                        }
                    ],
                    "latest": final,
                    "final": final,
                }
            ),
            encoding="utf-8",
        )
        runs.append(
            {
                "run_index": index,
                "run_id": run_id,
                "parent_training_run_id": run_id,
                "co_training_group_id": None,
                "checkpoint": str(deployment),
                "checkpoint_sha256": sha256_path(deployment),
            }
        )
        m1_rows.append(
            {
                "run_index": index,
                "run_id": run_id,
                "path": str(m1_path),
                "sha256": sha256_path(m1_path),
                "model_fingerprint": fingerprint,
                "passed": True,
            }
        )
        lineage.append(
            {
                "checkpoint_sha256": sha256_path(deployment),
                "parent_training_run_id": run_id,
                "co_training_group_id": None,
                "role": "formal_ego",
            }
        )
    manifest_path = root / "policy_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "layout": layout,
                "method": "depi",
                "policy_kind": "depi_deployment",
                "official_source_commit": OFFICIAL_SOURCE_COMMIT,
                "runs": runs,
                "training_lineage": lineage,
                "deployment_parameter_count": 1,
                "m1_final_evaluations": m1_rows,
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_final_m1_is_hash_bound_to_every_evaluated_deployment(tmp_path: Path) -> None:
    source_results = {}
    manifest_paths = []
    for layout in ("test_time_simple", "test_time_wide"):
        root = tmp_path / layout
        root.mkdir()
        manifest_path = _write_depi_manifest_with_final_m1(root, layout=layout)
        manifest = _load_policy_manifest(manifest_path, expected_layout=layout)
        assert all(row["passed"] is True for row in manifest["m1_final_evaluations"])
        raw = root / "official-raw"
        raw.mkdir()
        (raw / "run_identity.json").write_text(
            json.dumps(
                {
                    "policy_manifest": {
                        "path": str(manifest_path),
                        "sha256": sha256_path(manifest_path),
                        "content": manifest,
                    }
                }
            ),
            encoding="utf-8",
        )
        source_results[f"{layout}:depi"] = {
            "path": str(raw),
            "sha256": sha256_path(raw),
        }
        manifest_paths.append(manifest_path)
    official = {"source_results": source_results}
    assert _validate_final_m1_from_official_summary(official)

    first_manifest = json.loads(manifest_paths[0].read_text(encoding="utf-8"))
    first_m1 = Path(first_manifest["m1_final_evaluations"][0]["path"])
    first_m1.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="M1 source hash mismatch"):
        _validate_final_m1_from_official_summary(official)
