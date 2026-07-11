from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from experiments.overcooked_v2.path_c_protocol import (
    ActingEpisodeLog,
    ActingEpisodeSpec,
    ActingStepLog,
    BaselineSelection,
    ProbeBudgetController,
    ProbeBudgetState,
    PolicyAction,
    ReturnProbeBudgetCurve,
    ReturnProbeBudgetPoint,
    build_return_probe_budget_curve,
    baseline_selection_payload,
    estimate_audit_cost,
    estimate_locked_primary_endpoint,
    matched_policy_metrics,
    select_strongest_baseline_on_design,
    simulate_cluster_operating_characteristics,
    write_baseline_selection_artifact,
)


_RETURN_POINT_LEDGER_SHA256 = "d" * 64


def _curve(
    policy_name: str,
    auc: float,
    *,
    identity_group: str = "identity-0",
    layout_group: str = "layout-control",
    seed_group: str = "seed-0",
) -> ReturnProbeBudgetCurve:
    return ReturnProbeBudgetCurve(
        policy_name=policy_name,
        split_group_id=f"group-{identity_group}",
        mechanism="mechanism-a",
        identity_group=identity_group,
        style_group=f"style-{identity_group}",
        layout_group=layout_group,
        seed_group=seed_group,
        seed=0,
        budgets=(0, 1, 2),
        normalized_net_returns=(auc, auc, auc),
        auc=float(auc),
        normalization_lower=0.0,
        normalization_upper=10.0,
        probe_cost_per_use=1.0,
    )


def _selection() -> BaselineSelection:
    return select_strongest_baseline_on_design(
        [_curve("strong", 0.2)],
        candidate_order=("strong",),
        split_role="design",
        return_point_ledger_sha256=_RETURN_POINT_LEDGER_SHA256,
        design_split_group_ids=("group-identity-0",),
    )


def test_primary_curve_is_normalized_net_return_over_probe_budget():
    points = [
        ReturnProbeBudgetPoint(
            policy_name="probing_ego",
            split_group_id="held-out-group",
            mechanism="mechanism-a",
            identity_group="held-out-identity",
            style_group="held-out-style",
            layout_group="layout-control",
            seed_group="seed-0",
            seed=0,
            budget=budget,
            raw_return=5.0,
            probes_used=budget,
            realized_probe_cost=float(budget),
            probe_cost_per_use=1.0,
            episode_environment_steps=100 + budget,
            training_environment_steps=1000,
            gradient_updates=20,
            evaluation_environment_step_limit=400,
            evaluation_schedule_id="frozen-schedule",
        )
        for budget in (0, 1, 2)
    ]
    curve = build_return_probe_budget_curve(
        points,
        budget_grid=(0, 1, 2),
        normalization_lower=0.0,
        normalization_upper=10.0,
    )
    assert curve.policy_name == "probing_ego"
    assert curve.budgets == (0, 1, 2)
    assert curve.normalized_net_returns == pytest.approx((0.5, 0.4, 0.3))
    assert curve.auc == pytest.approx(0.4)
    assert points[-1].net_return == pytest.approx(3.0)


def test_curve_requires_every_frozen_budget_and_a_matched_schedule():
    points = [
        ReturnProbeBudgetPoint(
            policy_name="probing_ego",
            split_group_id="identity-0-group",
            mechanism="mechanism-a",
            identity_group="identity-0",
            style_group="identity-0-style",
            layout_group="layout-control",
            seed_group="seed-0",
            seed=0,
            budget=0,
            raw_return=1.0,
            probes_used=0,
            realized_probe_cost=0.0,
            probe_cost_per_use=1.0,
            episode_environment_steps=100,
            training_environment_steps=1000,
            gradient_updates=20,
            evaluation_environment_step_limit=400,
            evaluation_schedule_id="schedule-a",
        )
    ]
    with pytest.raises(ValueError, match="missing budget"):
        build_return_probe_budget_curve(
            points,
            budget_grid=(0, 1),
            normalization_lower=0.0,
            normalization_upper=2.0,
        )

    second = ReturnProbeBudgetPoint(
        policy_name="probing_ego",
        split_group_id="identity-0-group",
        mechanism="mechanism-a",
        identity_group="identity-0",
        style_group="identity-0-style",
        layout_group="layout-control",
        seed_group="seed-0",
        seed=0,
        budget=1,
        raw_return=1.0,
        probes_used=0,
        realized_probe_cost=0.0,
        probe_cost_per_use=1.0,
        episode_environment_steps=101,
        training_environment_steps=1000,
        gradient_updates=20,
        evaluation_environment_step_limit=400,
        evaluation_schedule_id="schedule-b",
    )
    with pytest.raises(ValueError, match="matched evaluation/compute schedule"):
        build_return_probe_budget_curve(
            [*points, second],
            budget_grid=(0, 1),
            normalization_lower=0.0,
            normalization_upper=2.0,
        )


def test_acting_episode_log_hash_binds_the_complete_accounting_ledger():
    budget_before = ProbeBudgetState(budget=1, probes_used=0, cost_per_probe=0.25)
    action = PolicyAction(
        action_id=1,
        is_probe=True,
        propensity=1.0,
        candidate_action_ids=(0, 1),
        candidate_scores=(0.0, 1.0),
        estimated_probe_cost=0.25,
        policy_kind="test",
    )
    budget_after = budget_before.consume(is_probe=True)
    step = ActingStepLog(
        decision_index=0,
        action=action,
        budget_before=budget_before,
        budget_after=budget_after,
        raw_reward=2.0,
        realized_probe_cost=0.25,
        environment_steps=3,
        terminated=True,
        truncated=False,
    )
    episode = ActingEpisodeLog(
        policy_name="probing_ego",
        spec=ActingEpisodeSpec(
            split_role="design",
            episode_uid="design:probing_ego:identity-0:1",
            split_group_id="design-group-0",
            mechanism="mechanism-a",
            identity_group="identity-0",
            style_group="style-0",
            layout_group="layout-control",
            seed_group="seed-0",
            seed=7,
            probe_budget=1,
            training_environment_steps=1000,
            gradient_updates=20,
            evaluation_environment_step_limit=3,
            evaluation_schedule_id="frozen-schedule",
        ),
        probe_cost_per_use=0.25,
        steps=(step,),
        raw_return=2.0,
        realized_probe_cost=0.25,
        environment_steps=3,
        final_budget_state=budget_after,
        policy_metrics={
            "environment_steps": 3,
            "probe_count": 1,
            "realized_probe_cost": 0.25,
        },
    )
    payload = episode.canonical_payload()
    assert payload["schema_version"] == "path_c_acting_episode_log_v1"
    assert payload["steps"][0]["action"]["is_probe"] is True
    assert payload["final_budget_state"]["probes_used"] == 1
    expected_sha256 = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    assert episode.sha256 == expected_sha256


def test_strongest_baseline_is_selected_once_on_design_only():
    curves = [
        _curve("history", 0.55, identity_group="design-identity-0"),
        _curve("history", 0.65, identity_group="design-identity-1"),
        _curve("belief", 0.75, identity_group="design-identity-0"),
        _curve("belief", 0.65, identity_group="design-identity-1"),
    ]
    selection = select_strongest_baseline_on_design(
        curves,
        candidate_order=("history", "belief"),
        split_role="design",
        return_point_ledger_sha256=_RETURN_POINT_LEDGER_SHA256,
        design_split_group_ids=(
            "group-design-identity-0",
            "group-design-identity-1",
        ),
    )
    assert selection.selected_baseline == "belief"
    assert selection.selection_role == "design"
    assert selection.design_auc_by_baseline == pytest.approx(
        {"history": 0.60, "belief": 0.70}
    )
    assert len(selection.sha256) == 64

    with pytest.raises(ValueError, match="only on the design split"):
        select_strongest_baseline_on_design(
            curves,
            candidate_order=("history", "belief"),
            split_role="locked_audit",
            return_point_ledger_sha256=_RETURN_POINT_LEDGER_SHA256,
            design_split_group_ids=(
                "group-design-identity-0",
                "group-design-identity-1",
            ),
        )


def test_design_baseline_tie_uses_frozen_candidate_order():
    selection = select_strongest_baseline_on_design(
        [_curve("first", 0.7), _curve("second", 0.7)],
        candidate_order=("first", "second"),
        split_role="design",
        return_point_ledger_sha256=_RETURN_POINT_LEDGER_SHA256,
        design_split_group_ids=("group-identity-0",),
    )
    assert selection.selected_baseline == "first"


def test_design_selection_rejects_unmatched_identity_support():
    with pytest.raises(ValueError, match="identical identity/layout/seed groups"):
        select_strongest_baseline_on_design(
            [
                _curve("first", 0.7, identity_group="identity-a"),
                _curve("second", 0.8, identity_group="identity-b"),
            ],
            candidate_order=("first", "second"),
            split_role="design",
            return_point_ledger_sha256=_RETURN_POINT_LEDGER_SHA256,
            design_split_group_ids=("group-identity-a", "group-identity-b"),
        )


def test_design_selection_artifact_binds_semantic_and_file_hashes_separately(tmp_path):
    selection = select_strongest_baseline_on_design(
        [_curve("first", 0.7), _curve("second", 0.6)],
        candidate_order=("first", "second"),
        split_role="design",
        return_point_ledger_sha256=_RETURN_POINT_LEDGER_SHA256,
        design_split_group_ids=("group-identity-0",),
    )
    path, digest = write_baseline_selection_artifact(
        tmp_path / "baseline-selection.json",
        selection,
        preregistration_sha256="a" * 64,
        resolved_path_c_sha256="b" * 64,
        semantic_bindings={"code_commit": "c" * 40},
    )
    payload = baseline_selection_payload(selection)
    assert payload["selected_baseline"] == "first"
    assert payload["sha256"] == selection.sha256
    assert payload["probe_budget_grid"] == [0, 1, 2]
    assert payload["normalization_rule"] == "affine_without_clipping"
    assert payload["probe_cost_per_use"] == pytest.approx(1.0)
    assert set(payload["source_curve_set_sha256_by_baseline"]) == {
        "first",
        "second",
    }
    assert payload["return_point_ledger_sha256"] == _RETURN_POINT_LEDGER_SHA256
    assert len(digest) == 64
    assert path.read_bytes()


def test_locked_primary_is_paired_by_held_out_identity_cluster():
    selection = _selection()
    method_curves = []
    baseline_curves = []
    for index in range(4):
        common = {
            "identity_group": f"held-out-identity-{index}",
            "layout_group": f"layout-template-{index}",
            "seed_group": f"seed-{index}",
        }
        method_curves.append(_curve("probing_ego", 0.35, **common))
        baseline_curves.append(_curve("strong", 0.20, **common))

    result = estimate_locked_primary_endpoint(
        method_curves,
        baseline_curves,
        baseline_selection=selection,
        locked_return_point_ledger_sha256="e" * 64,
        split_role="locked_audit",
        primary_split_group_ids=tuple(
            curve.split_group_id for curve in method_curves
        ),
        preregistered_margin=0.10,
        confidence_level=0.95,
        bootstrap_iterations=100,
        seed=3,
    )
    assert result.endpoint_name == (
        "cross_identity_normalized_net_return_probe_budget_auc_difference"
    )
    assert result.cluster_unit == "identity_group"
    assert result.clusters == 4
    assert result.identity_shift_only is True
    assert result.probe_budget_grid == (0, 1, 2)
    assert result.normalization_rule == "affine_without_clipping"
    assert result.probe_cost_per_use == pytest.approx(1.0)
    assert result.estimate == pytest.approx(0.15)
    assert result.confidence_interval == pytest.approx((0.15, 0.15))
    assert result.primary_effective is True


def test_locked_primary_rejects_layout_confounding():
    selection = _selection()
    method = [_curve("probing_ego", 0.4, layout_group="layout-a")]
    baseline = [_curve("strong", 0.2, layout_group="layout-b")]
    with pytest.raises(ValueError, match="must not confound layout shift"):
        estimate_locked_primary_endpoint(
            method,
            baseline,
            baseline_selection=selection,
            locked_return_point_ledger_sha256="e" * 64,
            split_role="locked_audit",
            primary_split_group_ids=(method[0].split_group_id,),
            preregistered_margin=0.0,
            confidence_level=0.95,
            bootstrap_iterations=10,
            seed=0,
        )


def test_probe_budget_controller_is_monotone_and_fails_when_exhausted():
    controller = ProbeBudgetController(budget=1, cost_per_probe=0.25)
    action = PolicyAction(
        action_id=1,
        is_probe=True,
        propensity=1.0,
        candidate_action_ids=(0, 1),
        candidate_scores=(0.0, 1.0),
        estimated_probe_cost=0.25,
        policy_kind="test",
    )
    assert controller.state.remaining == 1
    assert controller.record(action).remaining == 0
    assert controller.state.cumulative_cost == pytest.approx(0.25)
    with pytest.raises(ValueError, match="exhausting its budget"):
        controller.record(action)


def test_policy_fairness_report_checks_interaction_and_update_budgets():
    common = {
        "training_environment_steps": 1000,
        "gradient_updates": 200,
        "evaluation_environment_step_limit": 400,
        "evaluation_schedule_id": "schedule-v1",
        "probe_budget_grid_sha256": "a" * 64,
        "probe_cost_per_use": 0.25,
        "action_support_sha256": "b" * 64,
        "trainable_parameters": 10,
        "training_flops": 100,
        "wall_clock_seconds": 1.0,
        "inference_latency_ms": 0.1,
    }
    report = matched_policy_metrics(
        {"probing_ego": dict(common), "strong": dict(common)},
        required_policies=("probing_ego", "strong"),
    )
    assert report["fairness_conformant"] is True

    changed = dict(common)
    changed["gradient_updates"] = 201
    report = matched_policy_metrics(
        {"probing_ego": dict(common), "strong": changed},
        required_policies=("probing_ego", "strong"),
    )
    assert report["fairness_conformant"] is False
    assert "gradient_updates" in report["mismatches"]


def test_audit_cost_estimator_accounts_for_outer_and_inner_replicas():
    estimate = estimate_audit_cost(
        audit_units=2,
        probes=3,
        outer_replicas_M=5,
        inner_forks_L_inner=7,
        horizon_T_probe=11,
        max_primitive_steps=13,
        maximum_primitive_step_budget=(2 * 3 + 1 * 2) * 5 * 7 * 11 * 13,
        design_audit_units=1,
        design_only_probes=2,
    )
    assert estimate["core_continuations"] == 2 * 3 * 5 * 7
    assert estimate["design_extension_continuations"] == 1 * 2 * 5 * 7
    assert estimate["continuations"] == (2 * 3 + 1 * 2) * 5 * 7
    assert estimate["option_steps"] == (2 * 3 + 1 * 2) * 5 * 7 * 11
    assert estimate["primitive_step_upper_bound"] == (
        (2 * 3 + 1 * 2) * 5 * 7 * 11 * 13
    )
    assert estimate["within_budget"] is True


def test_positive_and_null_power_are_simulated_at_cluster_level():
    result = simulate_cluster_operating_characteristics(
        positive_cluster_effects=(0.2, 0.2, 0.2, 0.2),
        null_cluster_effects=(0.0, 0.0, 0.0, 0.0),
        clusters_per_trial=4,
        simulation_repetitions=8,
        bootstrap_iterations=16,
        alpha=0.05,
        efficacy_margin=0.1,
        equivalence_margin=0.05,
        seed=7,
    )
    assert result["cluster_unit"] == "identity_group"
    assert result["positive"]["effect"] == pytest.approx(0.2)
    assert result["positive"]["confidence_level"] == pytest.approx(0.95)
    assert result["positive"]["confidence_interval"] == pytest.approx((0.2, 0.2))
    assert result["positive"]["power"] == pytest.approx(1.0)
    assert result["null"]["effect"] == pytest.approx(0.0)
    assert result["null"]["confidence_level"] == pytest.approx(0.95)
    assert result["null"]["confidence_interval"] == pytest.approx((0.0, 0.0))
    tost = result["null"]["two_one_sided_tests"]
    assert tost["alpha_per_test"] == pytest.approx(0.05)
    assert tost["confidence_level"] == pytest.approx(0.90)
    assert tost["confidence_interval"] == pytest.approx((0.0, 0.0))
    assert tost["lower_bound_above_negative_margin"] is True
    assert tost["upper_bound_below_positive_margin"] is True
    assert tost["equivalent"] is True
    assert result["null"]["power"] == pytest.approx(1.0)
    assert result["joint"]["power"] == pytest.approx(1.0)
    assert result["positive_power"] == pytest.approx(1.0)
    assert result["null_equivalence_power"] == pytest.approx(1.0)
    assert result["joint_power"] == pytest.approx(1.0)


def test_cluster_power_rejects_alpha_that_cannot_define_tost_interval():
    with pytest.raises(ValueError, match="TOST"):
        simulate_cluster_operating_characteristics(
            positive_cluster_effects=(0.2, 0.2),
            null_cluster_effects=(0.0, 0.0),
            clusters_per_trial=2,
            simulation_repetitions=2,
            bootstrap_iterations=2,
            alpha=0.5,
            efficacy_margin=0.1,
            equivalence_margin=0.05,
            seed=7,
        )


def test_training_readout_aggregates_identity_group_effects(monkeypatch):
    from experiments.overcooked_v2.scripts import diag_d1_train

    def exact_interval(statistic, keys):
        value = statistic(np.arange(np.asarray(keys).size))
        return {"lo": value, "hi": value}

    monkeypatch.setattr(diag_d1_train, "bootstrap_ci", exact_interval)
    report = diag_d1_train._path_c_advantage_block(
        {
            "probing_ego": np.zeros(4, dtype=np.float64),
            "full_history_rnn": np.asarray((0.2, 0.2, 0.4, 0.4)),
        },
        np.asarray(("episode-0", "episode-1", "episode-2", "episode-3")),
        threshold=0.1,
        required_baselines=("full_history_rnn",),
        power_cluster_keys=np.asarray(("identity-b", "identity-b", "identity-a", "identity-a")),
    )
    assert report["power_cluster_unit"] == "identity_group"
    assert report["power_reference_baseline"] == "full_history_rnn"
    assert report["power_cluster_ids"] == ["identity-a", "identity-b"]
    assert report["power_cluster_effects"] == pytest.approx((0.4, 0.2))


def test_training_c4_emits_effect_ci_tost_and_joint_power():
    from experiments.overcooked_v2.scripts.diag_d1_train import (
        _path_c_power_null_measurement,
    )

    result = _path_c_power_null_measurement(
        {
            "path_c_rv_readout": {
                "signal": {
                    "power_cluster_effects": [0.2, 0.2, 0.2, 0.2],
                    "power_cluster_ids": ["p0", "p1", "p2", "p3"],
                    "power_reference_baseline": "full_history_rnn",
                }
            },
            "path_c_terminal_axis_null_readout": {
                "signal": {
                    "power_cluster_effects": [0.0, 0.0, 0.0, 0.0],
                    "power_cluster_ids": ["n0", "n1", "n2", "n3"],
                    "power_reference_baseline": "full_history_rnn",
                }
            },
        },
        {
            "_preregistration_status": "frozen",
            "positive_and_null_cluster_simulation_required": True,
            "joint_operating_characteristics_required": True,
            "equivalence_test": "two_one_sided_tests",
            "secondary_multiplicity": "hierarchical_gatekeeping",
            "selection_role": "design",
            "primary_alpha": 0.05,
            "cluster_unit": "identity_group",
            "clusters_per_trial": 4,
            "simulation_repetitions": 8,
            "bootstrap_iterations": 16,
            "seed": 7,
            "equivalence_margin": 0.05,
            "synthetic_power_min_advantage": 0.1,
            "minimum_positive_power": 0.8,
            "minimum_null_equivalence_power": 0.8,
            "minimum_joint_power": 0.8,
        },
    )
    assert result["available"] is True
    assert result["effect"] == pytest.approx({"positive": 0.2, "null": 0.0})
    assert result["confidence_intervals"]["positive"] == pytest.approx((0.2, 0.2))
    assert result["confidence_intervals"]["null"] == pytest.approx((0.0, 0.0))
    assert result["tost"]["confidence_level"] == pytest.approx(0.90)
    assert result["tost"]["equivalent"] is True
    assert result["positive"]["power"] == pytest.approx(1.0)
    assert result["null"]["power"] == pytest.approx(1.0)
    assert result["joint"]["power"] == pytest.approx(1.0)
    assert result["power_targets_met"]["all"] is True
    assert result["joint_cluster_power_simulation"]["joint_power"] == pytest.approx(1.0)
    assert result["decision_eligible"] is False
