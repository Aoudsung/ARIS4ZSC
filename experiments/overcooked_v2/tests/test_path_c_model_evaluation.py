from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.path_c.evaluation.pairing import (
    PairingSpec,
    development_pairings,
    execute_formal_pairings,
    execute_pairings,
    validate_formal_pairings,
)
from src.path_c.evaluation.summary import summarize_rows
from src.path_c.evaluation.response_contrast import (
    RESPONSE_CONTRAST_ROWS_SCHEMA_VERSION,
    execute_response_contrast_pairing,
    summarize_response_contrast_rows,
)
from src.path_c.evaluation.standard import (
    STANDARD_ROWS_SCHEMA_VERSION,
    PopulationManifest,
    StandardPolicyEntry,
    execute_standard_pairing,
    standard_episode_seed,
    standard_pairings,
    summarize_project_xp_difference,
    summarize_standard_rows,
    validate_standard_summary_payload,
)
from src.path_c.training.metrics import JsonlLedger


def test_development_schedule_has_self_pairing_and_both_partner_seats() -> None:
    """Retain the historical familiar-partner schedule only for artifact audit."""

    pairings = development_pairings("adapted", ("p0", "p1", "p2", "p3"))
    assert len(pairings) == 9
    assert sum(item.policy_0_id == item.policy_1_id for item in pairings) == 1
    for partner in ("p0", "p1", "p2", "p3"):
        assert any(item.policy_0_id == "adapted" and item.policy_1_id == partner for item in pairings)
        assert any(item.policy_0_id == partner and item.policy_1_id == "adapted" for item in pairings)


def test_backbone_and_condition_pairings_share_structural_episode_seeds() -> None:
    partners = ("p0", "p1", "p2", "p3")
    backbone = development_pairings(
        "official_backbone_seed_100",
        partners,
        focal_policy_source="frozen_official_backbone",
    )
    first = development_pairings("adapted_decision_focused", partners)
    second = development_pairings("adapted_no_probe", partners)
    assert [item.randomization_id for item in backbone] == [
        item.randomization_id for item in first
    ] == [item.randomization_id for item in second]
    from src.path_c.evaluation.pairing import canonical_episode_seed

    for index in range(9):
        assert canonical_episode_seed(
            10103, str(backbone[index].randomization_id), 7
        ) == canonical_episode_seed(10103, str(first[index].randomization_id), 7)

    from experiments.overcooked_v2.model_dock.evaluation_runtime import (
        BACKBONE_EVALUATION_EPISODES_PER_PAIRING,
    )

    assert BACKBONE_EVALUATION_EPISODES_PER_PAIRING == 64
    assert len(backbone) * BACKBONE_EVALUATION_EPISODES_PER_PAIRING == 576
    assert all(
        "adaptation_checkpoint"
        not in {pairing.policy_0_source, pairing.policy_1_source}
        for pairing in backbone
    )


def test_pairing_executor_writes_one_raw_row_per_episode_and_seat(tmp_path: Path) -> None:
    pairings = development_pairings("adapted", ("p0", "p1", "p2", "p3"))

    def evaluate(pairing, seeds):
        return [
            {
                "raw_return": float(index + (pairing.policy_0_id == "adapted")),
                "probe_count": index % 2,
                "safe_candidate_opportunity_count": 2,
                "maximum_probe_budget": (
                    40
                    if pairing.policy_0_source == pairing.policy_1_source
                    == "adaptation_checkpoint"
                    else 20
                ),
                "correct_delivery_count": 1,
                "wrong_delivery_count": 0,
                "indicator_cost": 0.25,
            }
            for index, unused_seed in enumerate(seeds)
        ]

    rows = execute_pairings(
        pairings,
        episodes_per_pairing=3,
        evaluation_seed=10103,
        evaluate_pairing=evaluate,
        output_path=tmp_path / "rows.jsonl",
    )
    assert len(rows) == 27
    assert (tmp_path / "rows.jsonl").read_text(encoding="utf-8").count("\n") == 27
    assert all(set(row["seat_assignment"]) == {"seat_0", "seat_1"} for row in rows)
    assert {(row["pairing_id"], row["episode_index"]) for row in rows} == {
        (pairing.pairing_id, index) for pairing in pairings for index in range(3)
    }


def test_summary_is_recomputed_after_raw_row_change() -> None:
    pairing = PairingSpec(
        pairing_id="a__b",
        policy_0_id="a",
        policy_1_id="b",
        policy_0_source="adaptation_checkpoint",
        policy_1_source="frozen_official_partner",
    )
    rows = execute_pairings(
        (pairing,),
        episodes_per_pairing=2,
        evaluation_seed=3,
        evaluate_pairing=lambda unused_pairing, unused_seeds: (
            {
                "raw_return": 1.0,
                "probe_count": 0,
                "safe_candidate_opportunity_count": 4,
                "maximum_probe_budget": 20,
            },
            {
                "raw_return": 3.0,
                "probe_count": 2,
                "safe_candidate_opportunity_count": 5,
                "maximum_probe_budget": 20,
            },
        ),
    )
    first = summarize_rows(rows)
    changed = [dict(row) for row in rows]
    changed[1]["raw_return"] = 7.0
    second = summarize_rows(changed)
    assert first["overall_mean_raw_return"] == 2.0
    assert second["overall_mean_raw_return"] == 4.0
    assert first["pairings"]["a__b"]["correct_delivery_count"] == 0
    assert first["total_safe_candidate_opportunity_count"] == 9
    assert first["total_maximum_probe_budget"] == 40


def test_pairing_rows_compute_registered_rates_and_frozen_not_applicable() -> None:
    adapted = PairingSpec(
        pairing_id="adapted__partner",
        policy_0_id="adapted",
        policy_1_id="partner",
        policy_0_source="adaptation_checkpoint",
        policy_1_source="frozen_official_partner",
    )
    adapted_row = execute_pairings(
        (adapted,),
        episodes_per_pairing=1,
        evaluation_seed=5,
        evaluate_pairing=lambda unused_pairing, unused_seeds: (
            {
                "raw_return": 0.0,
                "probe_count": 3,
                "safe_candidate_opportunity_count": 12,
                "maximum_probe_budget": 20,
            },
        ),
    )[0]
    assert adapted_row["probe_trigger_rate"] == 0.25
    assert adapted_row["probe_budget_usage_rate"] == 0.15

    self_pairing = development_pairings(
        "adapted", ("p0", "p1", "p2", "p3")
    )[0]
    self_row = execute_pairings(
        (self_pairing,),
        episodes_per_pairing=1,
        evaluation_seed=5,
        evaluate_pairing=lambda unused_pairing, unused_seeds: (
            {
                "raw_return": 0.0,
                "probe_count": 5,
                "safe_candidate_opportunity_count": 25,
                "maximum_probe_budget": 40,
            },
        ),
    )[0]
    assert self_row["maximum_probe_budget"] == 40
    assert self_row["probe_budget_usage_rate"] == 0.125

    frozen = PairingSpec(
        pairing_id="backbone__partner",
        policy_0_id="backbone",
        policy_1_id="partner",
        policy_0_source="frozen_official_backbone",
        policy_1_source="frozen_official_partner",
    )
    frozen_row = execute_pairings(
        (frozen,),
        episodes_per_pairing=1,
        evaluation_seed=5,
        evaluate_pairing=lambda unused_pairing, unused_seeds: (
            {
                "raw_return": 0.0,
                "probe_count": 0,
                "safe_candidate_opportunity_count": 0,
                "maximum_probe_budget": 0,
            },
        ),
    )[0]
    assert frozen_row["probe_trigger_rate"] is None
    assert frozen_row["probe_budget_usage_rate"] is None


def test_formal_executor_accepts_ten_self_and_ninety_directed_cross_pairings(
    monkeypatch,
) -> None:
    policy_ids = tuple(f"policy_{index}" for index in range(10))
    pairings = tuple(
        PairingSpec(
            pairing_id=f"{left}__{right}",
            policy_0_id=left,
            policy_1_id=right,
            policy_0_source="adaptation_checkpoint",
            policy_1_source="adaptation_checkpoint",
        )
        for left in policy_ids
        for right in policy_ids
    )
    validated = validate_formal_pairings(pairings)
    assert len(validated) == 100
    assert sum(item.policy_0_id == item.policy_1_id for item in validated) == 10
    assert sum(item.policy_0_id != item.policy_1_id for item in validated) == 90

    captured = {}

    def fake_execute(values, **kwargs):
        captured["pairing_count"] = len(values)
        captured.update(kwargs)
        return []

    monkeypatch.setattr("src.path_c.evaluation.pairing.execute_pairings", fake_execute)
    assert execute_formal_pairings(
        pairings,
        evaluation_seed=10103,
        evaluate_pairing=lambda unused_pairing, unused_seeds: (),
    ) == []
    assert captured["pairing_count"] == 100
    assert captured["episodes_per_pairing"] == 500


def test_decision_rows_use_only_registered_score_component_names(tmp_path: Path) -> None:
    from experiments.overcooked_v2.model_dock.stage_runtime import _write_rollout_records

    batch = SimpleNamespace(
        episode_ids=np.asarray([[0]]),
        episode_steps=np.asarray([[0]]),
        base_actions=np.asarray([[0]]),
        actions=np.asarray([[1]]),
        probed=np.asarray([[True]]),
        actor_owned_action=np.asarray([[False]]),
        candidate_actions=np.asarray([[1]]),
        decision_scores=np.asarray([[0.25]]),
        information_scores=np.asarray([[0.5]]),
        has_safe_candidate=np.asarray([[True]]),
        budget_remaining=np.asarray([[19]]),
        chosen_j_use=np.asarray([[2.0]]),
        chosen_j_mask=np.asarray([[1.0]]),
        value_base=np.asarray([[0.5]]),
        value_mask=np.asarray([[1.25]]),
        chosen_s_seq=np.asarray([[0.75]]),
        chosen_response_information=np.asarray([[0.5]]),
        partner_indices=np.asarray([[0]]),
        ego_seats=np.asarray([[0]]),
        episode_completed=np.asarray([[False]]),
        completed_episode_returns=np.asarray([[np.nan]]),
        completed_episode_probe_counts=np.asarray([[0]]),
        correct_deliveries=np.asarray([[0]]),
        wrong_deliveries=np.asarray([[0]]),
        indicator_costs=np.asarray([[0.0]]),
    )
    partner_pool = (SimpleNamespace(training_run_id="partner_0"),)

    def write(controller: str, name: str) -> dict:
        decisions_path = tmp_path / f"{name}_decisions.jsonl"
        _write_rollout_records(
            decision_ledger=JsonlLedger(decisions_path, truncate=True),
            episode_ledger=JsonlLedger(tmp_path / f"{name}_episodes.jsonl", truncate=True),
            batch=batch,
            controller=controller,
            partner_pool=partner_pool,
            maximum_probe_budget=20,
        )
        return json.loads(decisions_path.read_text(encoding="utf-8"))

    sequential = write(
        "registered_response_sequential_branch_v1", "sequential"
    )
    assert set(sequential["score_components"]) == {
        "j_use",
        "j_mask",
        "v_base",
        "v_mask",
        "s_seq",
    }
    information = write("generic_response_information", "information")
    assert set(information["score_components"]) == {"response_information"}
    prohibited = {"score", "sequential_score", "controller_score"}
    assert prohibited.isdisjoint(sequential["score_components"])
    assert prohibited.isdisjoint(information["score_components"])


def test_completed_episode_opportunities_are_recounted_and_reset() -> None:
    from experiments.overcooked_v2.model_dock.stage_runtime import (
        _completed_safe_opportunity_counts,
    )

    counts = _completed_safe_opportunity_counts(
        episode_ids=np.asarray(((0,), (0,), (1,), (1,))),
        episode_steps=np.asarray(((0,), (1,), (0,), (1,))),
        has_safe_candidate=np.asarray(((True,), (False,), (True,), (True,))),
        episode_completed=np.asarray(((False,), (True,), (False,), (True,))),
    )
    assert counts == {(1, 0): 1, (3, 0): 2}

    with pytest.raises(RuntimeError, match="step zero"):
        _completed_safe_opportunity_counts(
            episode_ids=np.asarray(((5,),)),
            episode_steps=np.asarray(((1,),)),
            has_safe_candidate=np.asarray(((True,),)),
            episode_completed=np.asarray(((True,),)),
        )


def test_frozen_evaluation_action_has_no_response_shape_placeholder() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from experiments.overcooked_v2.model_dock.evaluation_runtime import (
        _SideState,
        _frozen_action,
    )

    class Adapter:
        @staticmethod
        def apply_actor(params, carry, observations, episode_start):
            del params, observations, episode_start
            return carry + 1, jnp.zeros((carry.shape[0], 3), dtype=jnp.float32)

    action, next_side, probed, response, opportunity = _frozen_action(
        policy=SimpleNamespace(params={}, network_adapter=Adapter()),
        side=_SideState(
            carry=jnp.zeros((2, 4), dtype=jnp.float32),
            log_belief=jnp.zeros((2, 7), dtype=jnp.float32),
            budget_remaining=jnp.ones((2,), dtype=jnp.int32),
        ),
        observations=jnp.zeros((2, 5), dtype=jnp.float32),
        episode_start=jnp.zeros((2,), dtype=jnp.bool_),
        random_key=jax.random.PRNGKey(5),
    )
    assert action.shape == (2,)
    assert next_side.carry.shape == (2, 4)
    assert not np.asarray(probed).any()
    assert response is None
    assert not np.asarray(opportunity).any()


class _FakeOuterUnits:
    seed_root = 2_026_072_101
    sha256 = "e" * 64

    def __init__(self) -> None:
        self.units = tuple(
            SimpleNamespace(
                seeds=SimpleNamespace(evaluation_seed=80_000 + index)
            )
            for index in range(10)
        )

    def unit(self, index: int):
        return self.units[index]


def _standard_manifest(
    population_id: str = "decision_focused",
    *,
    outer_units: _FakeOuterUnits | None = None,
) -> SimpleNamespace:
    source_type = (
        "official_backbone"
        if population_id == "pre_adaptation_backbone"
        else "adaptation_checkpoint"
    )
    return SimpleNamespace(
        sha256=f"{1 + sum(ord(value) for value in population_id):064x}"[-64:],
        population_id=population_id,
        condition_id=None if source_type == "official_backbone" else population_id,
        source_type=source_type,
        layout="test_time_simple",
        episodes_per_pairing=500,
        outer_units=outer_units or _FakeOuterUnits(),
        policies=tuple(
            StandardPolicyEntry(
                outer_unit_id=index,
                policy_id=f"outer_unit_{index:02d}",
                source_type=source_type,
                checkpoint_path=Path(f"/formal/{population_id}/outer_unit_{index:02d}"),
                checkpoint_manifest_sha256=f"{index + 1:064x}",
            )
            for index in range(10)
        ),
    )


def test_standard_matrix_has_ten_sp_and_ninety_directed_xp_cells() -> None:
    manifest = _standard_manifest()
    pairings = standard_pairings(manifest)
    assert len(pairings) == 100
    assert sum(pairing.split == "sp" for pairing in pairings) == 10
    assert sum(pairing.split == "xp" for pairing in pairings) == 90
    assert {
        (pairing.outer_unit_0, pairing.outer_unit_1) for pairing in pairings
    } == {(left, right) for left in range(10) for right in range(10)}
    assert all(
        pairing.policy_0.checkpoint_path != pairing.policy_1.checkpoint_path
        for pairing in pairings
        if pairing.split == "xp"
    )


def test_formal_xp_runtime_keeps_each_side_parameters_state_and_belief_separate() -> None:
    from experiments.overcooked_v2.model_dock.formal_evaluation_runtime import (
        FormalPopulationRuntime,
    )

    source = inspect.getsource(FormalPopulationRuntime._compile_adapted_evaluator)
    assert "left_params" in source and "right_params" in source
    assert "params=left_params" in source and "side=current.left" in source
    assert "params=right_params" in source and "side=current.right" in source
    assert source.count("carry=self.model.initial_carry(count)") == 2
    assert source.count("log_belief=uniform_log_belief(count, 4)") == 2


def test_adapted_population_rejects_an_upstream_training_policy(
    tmp_path: Path,
) -> None:
    from src.path_c.evaluation.standard import _load_policy_entry

    checkpoint = tmp_path / "training_partner"
    checkpoint.mkdir()
    checkpoint_manifest = checkpoint / "manifest.json"
    checkpoint_manifest.write_text("{}\n", encoding="utf-8")
    source = SimpleNamespace(checkpoint_path=checkpoint, checkpoint_history=())
    unit = SimpleNamespace(sources=(source,))
    outer_units = SimpleNamespace(
        units=(unit,),
        unit=lambda unused_index: unit,
    )
    payload = {
        "outer_unit_id": 0,
        "policy_id": "outer_unit_00",
        "checkpoint_path": str(checkpoint),
        "checkpoint_manifest_sha256": hashlib.sha256(
            checkpoint_manifest.read_bytes()
        ).hexdigest(),
        "calibration_summary_path": str(tmp_path / "calibration.json"),
        "calibration_summary_sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="training partner or upstream backbone"):
        _load_policy_entry(
            payload,
            index=0,
            source_type="adaptation_checkpoint",
            condition_id="decision_focused",
            controller="registered_response_sequential_branch_v1",
            outer_units=outer_units,
            base_dir=tmp_path,
        )


def test_checkpoint_resolved_probe_fields_restore_public_yaml_nesting() -> None:
    from src.path_c.evaluation.standard import (
        _normalize_checkpoint_resolved_config,
    )

    payload = {
        "schema_version": "path_c_model_v2",
        "probe": {
            "candidate_window": 100,
            "budget_per_episode": 20,
            "safety_rule": "block_interact_facing_visible_goal_v1",
            "threshold_source": "calibration",
            "decision_null_quantile": 0.95,
            "information_quantile": 0.80,
            "manual_threshold": None,
            "belief_probability_floor": 1.0e-8,
        },
    }
    normalized = _normalize_checkpoint_resolved_config(payload)
    assert normalized["probe"] == {
        "candidate_window": 100,
        "budget_per_episode": 20,
        "safety": {"rule": "block_interact_facing_visible_goal_v1"},
        "threshold": {
            "source": "calibration",
            "decision_null_quantile": 0.95,
            "information_quantile": 0.80,
        },
        "belief_probability_floor": 1.0e-8,
    }
    assert payload["probe"]["threshold_source"] == "calibration"


def test_standard_cell_has_500_unique_condition_independent_episode_seeds() -> None:
    outer_units = _FakeOuterUnits()
    populations = tuple(
        _standard_manifest(name, outer_units=outer_units)
        for name in (
            "pre_adaptation_backbone",
            "decision_focused",
            "no_probe",
            "random_safe_probe",
            "generic_response_information",
        )
    )
    reference = standard_pairings(populations[0])[17]
    expected = [
        standard_episode_seed(
            manifest=populations[0],
            outer_unit_0=reference.outer_unit_0,
            outer_unit_1=reference.outer_unit_1,
            episode_index=index,
        )
        for index in range(500)
    ]
    assert len(set(expected)) == 500
    for manifest in populations[1:]:
        assert [
            standard_episode_seed(
                manifest=manifest,
                outer_unit_0=reference.outer_unit_0,
                outer_unit_1=reference.outer_unit_1,
                episode_index=index,
            )
            for index in range(500)
        ] == expected


def test_standard_summary_uses_pairing_means_and_separates_sp_from_xp() -> None:
    manifest = _standard_manifest()
    rows = []
    for pairing in standard_pairings(manifest):
        pairing_return = (
            float(pairing.outer_unit_0)
            if pairing.split == "sp"
            else float(pairing.outer_unit_0 - pairing.outer_unit_1)
        )
        rows.extend(
            execute_standard_pairing(
                manifest,
                pairing,
                evaluate_pairing=lambda unused_pairing, seeds, value=pairing_return: [
                    {
                        "raw_return": value,
                        "correct_delivery_count": 1,
                        "wrong_delivery_count": 0,
                        "probe_count": 0,
                        "safe_candidate_opportunity_count": 0,
                        "maximum_probe_budget": 0,
                    }
                    for unused_seed in seeds
                ],
            )
        )
    assert len(rows) == 50_000
    assert all(row["schema_version"] == STANDARD_ROWS_SCHEMA_VERSION for row in rows)
    summary = summarize_standard_rows(rows)
    assert summary["sp_pairing_count"] == 10
    assert summary["xp_pairing_count"] == 90
    assert summary["sp_raw_row_count"] == 5_000
    assert summary["xp_raw_row_count"] == 45_000
    assert summary["sp_mean_raw_return"] == pytest.approx(4.5)
    assert summary["sp_pairing_standard_deviation"] == pytest.approx(
        np.std(np.arange(10, dtype=np.float64), ddof=0)
    )
    xp_values = np.asarray(
        [left - right for left in range(10) for right in range(10) if left != right],
        dtype=np.float64,
    )
    assert summary["xp_mean_raw_return"] == pytest.approx(float(xp_values.mean()))
    assert summary["xp_pairing_standard_deviation"] == pytest.approx(
        float(xp_values.std(ddof=0))
    )
    assert summary["sp_minus_xp"] == pytest.approx(4.5)
    assert "overall_mean_raw_return" not in summary
    assert all("standard_error" not in key for key in summary)

    contaminated = dict(summary)
    contaminated["overall_mean_raw_return"] = 1.0
    with pytest.raises(ValueError, match="mixed means"):
        validate_standard_summary_payload(contaminated)

    no_probe_rows = [
        {
            **row,
            "population_id": "no_probe",
            "population_manifest_sha256": "f" * 64,
            "condition_id": "no_probe",
        }
        for row in rows
    ]
    comparison = summarize_project_xp_difference(
        rows,
        no_probe_rows,
        bootstrap_seed=123,
        bootstrap_samples=4,
    )
    assert comparison["mean_xp_difference"] == 0.0
    assert comparison["outer_unit_node_resampling_interval_95_percent"] == [
        0.0,
        0.0,
    ]
    assert comparison["not_an_official_sp_xp_summary"] is True


def test_training_pipeline_exposes_no_familiar_partner_evaluation_stage() -> None:
    from experiments.overcooked_v2.model_dock.stage_runtime import PathCStageRuntime
    from src.path_c.pipeline.run import FAMILY_POOL_STAGES, LEGACY_STAGES

    runtime = object.__new__(PathCStageRuntime)
    runtime.config = SimpleNamespace(is_family_pool=False)
    assert LEGACY_STAGES == ("pool_check", "prefit", "calibration", "adaptation")
    assert set(runtime.stage_runners()) == set(LEGACY_STAGES)
    runtime.config = SimpleNamespace(is_family_pool=True)
    assert set(runtime.stage_runners()) == set(FAMILY_POOL_STAGES)
    assert "evaluation" not in runtime.stage_runners()
    assert "backbone_evaluation" not in runtime.stage_runners()


def test_formal_evaluator_does_not_import_historical_standard_module() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    paths = (
        repository_root / "src" / "path_c" / "evaluation" / "standard.py",
        repository_root
        / "experiments"
        / "overcooked_v2"
        / "model_dock"
        / "formal_evaluation_runtime.py",
        repository_root
        / "experiments"
        / "overcooked_v2"
        / "scripts"
        / "evaluate_path_c_model.py",
        repository_root
        / "experiments"
        / "overcooked_v2"
        / "model_dock"
        / "response_contrast_runtime.py",
    )
    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert (
            "from experiments.overcooked_v2.path_c_standard_evaluation" not in source
        )
        assert (
            "import experiments.overcooked_v2.path_c_standard_evaluation" not in source
        )
        assert "import torch" not in source
        assert "from torch" not in source


def test_response_contrast_pairs_three_branches_and_recomputes_effect_identity() -> None:
    units = tuple(
        SimpleNamespace(seeds=SimpleNamespace(evaluation_seed=70_000 + index))
        for index in range(10)
    )
    manifest = SimpleNamespace(
        population_id="decision_focused",
        sha256="1" * 64,
        outer_units=SimpleNamespace(
            sha256="2" * 64,
            seed_root=2_026_072_302,
            unit=lambda index: units[index],
        ),
        layout="test_time_simple",
        episodes_per_pairing=500,
    )
    pairing = SimpleNamespace(
        split="xp",
        pairing_id="outer_unit_00__outer_unit_01",
        outer_unit_0=0,
        outer_unit_1=1,
    )

    def evaluate(unused_pairing, seeds):
        del unused_pairing
        assert len(set(seeds)) == 500
        return [
            {
                "triggered": True,
                "first_probe_step": index % 100,
                "executed_environment_steps": 1_200,
                "unique_environment_steps": 1_199,
                "branches": {
                    branch: {
                        "raw_return": value,
                        "correct_delivery_count": 1,
                        "wrong_delivery_count": 0,
                        "probe_count": 1,
                        "safe_candidate_opportunity_count": 2,
                        "maximum_probe_budget": 40,
                        "indicator_cost": 0.0,
                    }
                    for branch, value in (
                        ("A1", 2.0),
                        ("A2-mask", 1.0),
                        ("A2-use", 4.0),
                    )
                },
            }
            for index in range(500)
        ]

    rows = execute_response_contrast_pairing(
        manifest,
        pairing,
        evaluate_pairing=evaluate,
    )
    assert len(rows) == 1_500
    assert {row["schema_version"] for row in rows} == {
        RESPONSE_CONTRAST_ROWS_SCHEMA_VERSION
    }
    summary = summarize_response_contrast_rows(rows)
    assert summary["delta_response"] == pytest.approx(3.0)
    assert summary["delta_cost"] == pytest.approx(1.0)
    assert summary["delta_net"] == pytest.approx(2.0)
    assert summary["identity_residual"] == pytest.approx(0.0)
    assert summary["not_an_official_sp_xp_summary"] is True


def test_response_contrast_requires_identical_complete_trajectory_without_trigger() -> None:
    base = {
        "schema_version": RESPONSE_CONTRAST_ROWS_SCHEMA_VERSION,
        "split": "xp",
        "matched_block_id": "block",
        "pairing_id": "outer_unit_00__outer_unit_01",
        "branch": "A1",
        "triggered": False,
        "executed_environment_steps": 1_200,
        "unique_environment_steps": 400,
        "raw_return": 3.0,
    }
    rows = [
        {**base, "branch": branch}
        for branch in ("A1", "A2-mask", "A2-use")
    ]
    assert summarize_response_contrast_rows(rows)["delta_net"] == 0.0
    rows[-1]["raw_return"] = 4.0
    with pytest.raises(ValueError, match="unequal complete trajectories"):
        summarize_response_contrast_rows(rows)
