from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import pytest

from src.path_c.evaluation import (
    DEPLOYMENT_MODES,
    EpisodeRow,
    Pairing,
    ResponseContrastRow,
    equal_frequency_bins,
    effect_components,
    spearman_rank_correlation,
    standard_episode_seed,
    standard_pairings,
    summarize_standard_rows,
    summarize_counterfactual_trigger_values,
    validate_development_response_contrast_rows,
    validate_development_rows,
    validate_counterfactual_continuation_index,
    validate_response_contrast_rows,
    validate_standard_rows,
)
from src.path_c.experiment import (
    Population,
    PopulationEntry,
    load_population,
    write_population,
)


def _episode(mode: str, left: int, right: int, index: int) -> EpisodeRow:
    return EpisodeRow(
        population="literal_population",
        layout="test_time_simple",
        deployment_mode=mode,
        split="sp" if left == right else "xp",
        pairing_id=f"{left:02d}_to_{right:02d}",
        left_outer_unit_id=left,
        right_outer_unit_id=right,
        episode_index=index,
        episode_seed=standard_episode_seed(
            evaluation_seed=123,
            layout="test_time_simple",
            left_outer_unit_id=left,
            right_outer_unit_id=right,
            episode_index=index,
        ),
        environment_steps=400,
        raw_return=float(10 * left + right),
        correct_delivery_count=left,
        wrong_delivery_count=right,
        indicator_activation_count=0,
        positive_policy_mediated_effect_count=0,
        cumulative_kl=0.0,
        reference_action_deviation_count=0,
        mean_value_class_count=1.0,
        mean_belief_entropy=0.0,
        mean_predicted_next_policy_tv=0.0,
        response_code_counts=(400, 0),
    )


def test_standard_matrix_has_ten_self_and_ninety_directed_cross_pairings() -> None:
    pairings = standard_pairings()
    assert len(pairings) == 4 * 100
    for mode in DEPLOYMENT_MODES:
        selected = [item for item in pairings if item.deployment_mode == mode]
        assert sum(item.split == "sp" for item in selected) == 10
        assert sum(item.split == "xp" for item in selected) == 90
        assert {(item.left_outer_unit_id, item.right_outer_unit_id) for item in selected} == {
            (left, right) for left in range(10) for right in range(10)
        }


def test_five_hundred_unique_episodes_and_matched_mode_seeds() -> None:
    rows = tuple(
        _episode("posterior_use", left, right, index)
        for left in range(10)
        for right in range(10)
        for index in range(500)
    )
    validate_standard_rows(rows, deployment_modes=("posterior_use",))
    for left, right, index in ((0, 0, 0), (2, 8, 317), (9, 1, 499)):
        seeds = {
            standard_episode_seed(
                evaluation_seed=123,
                layout="test_time_simple",
                left_outer_unit_id=left,
                right_outer_unit_id=right,
                episode_index=index,
            )
            for _ in DEPLOYMENT_MODES
        }
        assert len(seeds) == 1


def test_development_self_pairing_is_complete_and_seed_matched() -> None:
    rows = tuple(
        _episode(mode, 0, 0, index)
        for mode in DEPLOYMENT_MODES
        for index in range(2)
    )
    validate_development_rows(rows, episodes_per_mode=2)
    assert {
        row.episode_seed for row in rows if row.episode_index == 1
    } == {rows[1].episode_seed}


def test_summary_uses_pairing_means_and_pairing_standard_deviation() -> None:
    rows = tuple(
        _episode("posterior_use", left, right, index)
        for left in range(10)
        for right in range(10)
        for index in range(500)
    )
    summary = summarize_standard_rows(rows)["deployment_modes"]["posterior_use"]
    sp_values = [float(11 * index) for index in range(10)]
    xp_values = [
        float(10 * left + right)
        for left in range(10)
        for right in range(10)
        if left != right
    ]
    np.testing.assert_allclose(summary["sp"]["mean_raw_return"], mean(sp_values))
    np.testing.assert_allclose(
        summary["sp"]["pairing_standard_deviation"], stdev(sp_values)
    )
    np.testing.assert_allclose(summary["xp"]["mean_raw_return"], mean(xp_values))
    np.testing.assert_allclose(
        summary["xp"]["pairing_standard_deviation"], stdev(xp_values)
    )


def _contrast_row(*, left: int, right: int, index: int, triggered: bool) -> ResponseContrastRow:
    common = dict(
        pairing_id=f"{left:02d}_to_{right:02d}",
        episode_index=index,
        episode_seed=standard_episode_seed(
            evaluation_seed=17,
            layout="test_time_simple",
            left_outer_unit_id=left,
            right_outer_unit_id=right,
            episode_index=index,
        ),
        triggered=triggered,
        trigger_step=12 if triggered else None,
        trigger_tolerance=0.01 if triggered else None,
        predicted_response_effect=0.2 if triggered else None,
        predicted_policy_cost=0.1 if triggered else None,
        predicted_net_effect=0.1 if triggered else None,
        predicted_regularized_net_effect=0.08 if triggered else None,
        predicted_policy_total_variation=0.03 if triggered else None,
        predicted_policy_mediated_effect=0.06 if triggered else None,
        predicted_policy_gain_lower_score=0.04 if triggered else None,
        predicted_policy_gain_uncertainty=0.02 if triggered else None,
        predicted_next_policy_total_variation=0.04 if triggered else None,
        maximum_action_net_value=0.4 if triggered else None,
        maximum_action_policy_mediated_gain=0.08 if triggered else None,
        maximum_action_predicted_gain_lower_score=0.05 if triggered else None,
        executed_action_net_value=0.2 if triggered else None,
        executed_action_response_value=0.3 if triggered else None,
        executed_action_policy_mediated_gain=0.06 if triggered else None,
        executed_action_predicted_gain_lower_score=0.04 if triggered else None,
        executed_action_policy_gain_uncertainty=0.02 if triggered else None,
        executed_action_expected_next_policy_tv=0.04 if triggered else None,
        executed_action=5 if triggered else None,
        maximum_net_action=5 if triggered else None,
        post_response_belief_l1=0.6 if triggered else None,
        first_left_action_difference_step=13 if triggered else None,
        first_observation_difference_step=14 if triggered else None,
        first_response_code_difference_step=12 if triggered else None,
        first_reward_difference_step=20 if triggered else None,
        left_action_difference_count=3 if triggered else None,
        observation_difference_count=2 if triggered else None,
        response_code_difference_count=1 if triggered else None,
        reward_difference_count=1 if triggered else None,
        environment_steps=1_200,
        a1_raw_return=7.0 if triggered else 2.0,
        a1_correct_delivery_count=1 if triggered else 0,
        a1_wrong_delivery_count=0,
        a1_indicator_activation_count=2 if triggered else 0,
        a2_mask_raw_return=3.0 if triggered else 2.0,
        a2_mask_correct_delivery_count=1 if triggered else 0,
        a2_mask_wrong_delivery_count=1 if triggered else 0,
        a2_mask_indicator_activation_count=3 if triggered else 0,
        a2_use_raw_return=11.0 if triggered else 2.0,
        a2_use_correct_delivery_count=2 if triggered else 0,
        a2_use_wrong_delivery_count=0,
        a2_use_indicator_activation_count=2 if triggered else 0,
    )
    return ResponseContrastRow(**common)


def test_response_mask_effects_follow_three_branch_identity() -> None:
    row = _contrast_row(left=1, right=7, index=4, triggered=True)
    assert effect_components(row) == {
        "delta_response": 8.0,
        "delta_cost": 4.0,
        "delta_net": 4.0,
    }


def test_response_bins_use_the_executed_action_lower_score() -> None:
    rows = []
    for index in range(10):
        row = _contrast_row(left=0, right=0, index=index, triggered=True)
        rows.append(
            replace(
                row,
                predicted_policy_gain_lower_score=float(index),
                executed_action_predicted_gain_lower_score=float(9 - index),
                a2_use_raw_return=float(index),
                a2_mask_raw_return=0.0,
            )
        )
    bins = equal_frequency_bins(rows, bin_count=2)
    assert bins[0]["minimum_executed_action_predicted_gain_lower_score"] == 0.0
    assert bins[0]["maximum_executed_action_predicted_gain_lower_score"] == 4.0
    assert bins[0]["effects"]["delta_response"] == pytest.approx(7.0)


def test_counterfactual_summary_never_uses_post_response_gain_for_coverage() -> None:
    rows = (
        {
            "trigger_id": "a",
            "executed_action_predicted_gain_lower_score": 1.0,
            "empirical_pre_response_discounted_gain": 0.5,
            "empirical_post_response_gain": 100.0,
            "use_action_rank_correlation": 1.0,
            "mask_action_rank_correlation": -1.0,
            "use_optimal_action_match": True,
            "mask_optimal_action_match": False,
        },
        {
            "trigger_id": "b",
            "executed_action_predicted_gain_lower_score": 0.0,
            "empirical_pre_response_discounted_gain": 0.25,
            "empirical_post_response_gain": -100.0,
            "use_action_rank_correlation": 0.5,
            "mask_action_rank_correlation": 0.5,
            "use_optimal_action_match": True,
            "mask_optimal_action_match": True,
        },
    )
    summary = summarize_counterfactual_trigger_values(rows)
    assert summary["pre_response_coverage"] == 0.5
    assert summary["mean_empirical_post_response_gain"] == 0.0
    assert spearman_rank_correlation(
        (3.0, 1.0, 2.0), (30.0, 10.0, 20.0)
    ) == pytest.approx(1.0)


def test_response_reconstruction_rejects_any_saved_field_difference() -> None:
    from experiments.overcooked_v2.counterfactual_audit_app import (
        validate_response_replay,
    )

    expected = _contrast_row(left=0, right=0, index=0, triggered=True)
    validate_response_replay((expected,), (expected,))
    changed = replace(expected, a2_use_raw_return=expected.a2_use_raw_return + 1.0)
    with pytest.raises(RuntimeError, match="a2_use_raw_return"):
        validate_response_replay((expected,), (changed,))


def test_counterfactual_index_pairs_six_actions_two_branches_and_replicas() -> None:
    replicas = 128
    rows = [
        {
            "estimand": "pre_response",
            "forced_action": 5,
            "replica_index": replica,
            "branch": branch,
        }
        for replica in range(replicas)
        for branch in ("use", "mask")
    ]
    rows.extend(
        {
            "estimand": "post_response",
            "forced_action": action,
            "replica_index": replica,
            "branch": branch,
        }
        for action in range(6)
        for replica in range(replicas)
        for branch in ("use", "mask")
    )
    validate_counterfactual_continuation_index(rows, replicas=replicas)
    with pytest.raises(ValueError, match="incomplete"):
        validate_counterfactual_continuation_index(rows[:-1], replicas=replicas)


def test_response_contrast_requires_complete_directed_xp_rows() -> None:
    rows = tuple(
        _contrast_row(left=left, right=right, index=index, triggered=False)
        for left in range(10)
        for right in range(10)
        if left != right
        for index in range(500)
    )
    validate_response_contrast_rows(
        rows,
        evaluation_seed=17,
        layout="test_time_simple",
    )


def test_development_response_contrast_accepts_one_complete_self_pair() -> None:
    rows = tuple(
        _contrast_row(left=0, right=0, index=index, triggered=bool(index % 2))
        for index in range(2)
    )
    validate_development_response_contrast_rows(
        rows,
        evaluation_seed=17,
        layout="test_time_simple",
        episodes_per_pairing=2,
    )


def test_population_manifest_contains_run_directories_only(tmp_path: Path) -> None:
    population = Population(
        name="posterior_use",
        layout="test_time_simple",
        evaluation_kind="standard_matrix",
        entries=tuple(
            PopulationEntry(index, tmp_path / f"run-{index}") for index in range(10)
        ),
    )
    path = write_population(tmp_path / "population.json", population)
    assert load_population(path) == population


def test_development_population_contains_only_outer_unit_zero(tmp_path: Path) -> None:
    population = Population(
        name="seed-100-development",
        layout="test_time_simple",
        evaluation_kind="development_diagnostic",
        entries=(PopulationEntry(0, tmp_path / "run-0"),),
    )
    path = write_population(tmp_path / "development.json", population)
    assert load_population(path) == population


def test_directed_pairing_does_not_duplicate_seat_swap() -> None:
    forward = Pairing("posterior_use", "xp", 2, 7)
    reverse = Pairing("posterior_use", "xp", 7, 2)
    assert forward.pairing_id == "02_to_07"
    assert reverse.pairing_id == "07_to_02"
    assert forward != reverse


def test_fixed_partner_panel_selects_final_checkpoints_and_matches_mode_seeds(
    tmp_path: Path,
) -> None:
    from experiments.overcooked_v2.partner_panel_app import (
        panel_episode_seed,
        select_panel_checkpoints,
    )

    checkpoints = tuple(tmp_path / f"checkpoint-{index}" for index in range(12))
    selected = select_panel_checkpoints(checkpoints, final_only=True)
    assert selected == tuple(path.resolve() for path in checkpoints[2::3])
    seeds = {
        panel_episode_seed(
            evaluation_seed=91,
            layout="test_time_simple",
            ego_outer_unit_id=0,
            partner_index=2,
            episode_index=19,
        )
        for _mode in DEPLOYMENT_MODES
    }
    assert len(seeds) == 1


def test_slot_semantics_summary_uses_recorded_phase_partner_and_slot() -> None:
    from experiments.overcooked_v2.counterfactual_audit_app import (
        _summarize_slot_semantics,
    )

    rows = [
        {
            "source": "self_pairing",
            "partner_index": None,
            "dominant_use_slot": 0,
            "task_phase_before_action": "fetch_plate",
            "use_slot_pairwise_l2": [2.0],
            "use_slot_optimal_actions": [1, 2],
            "use_slot_centered_q": [[0.0, 1.0], [1.0, 0.0]],
        },
        {
            "source": "fixed_partner_00",
            "partner_index": 0,
            "dominant_use_slot": 1,
            "task_phase_before_action": "final_delivery",
            "use_slot_pairwise_l2": [4.0],
            "use_slot_optimal_actions": [2, 2],
            "use_slot_centered_q": [[2.0, 0.0], [0.0, 2.0]],
        },
    ]
    summary = _summarize_slot_semantics(rows)
    assert summary["state_count"] == 2
    assert summary["mean_between_slot_centered_q_distance"] == 3.0
    assert summary["states_with_different_slot_optimal_actions"] == 1
    assert summary["dominant_slot_by_task_phase"] == {
        "fetch_plate": {"0": 1},
        "final_delivery": {"1": 1},
    }
    assert summary["dominant_slot_by_partner"] == {
        "None": {"0": 1},
        "0": {"1": 1},
    }


def test_same_action_a1_reuses_the_exact_a2_mask_branch() -> None:
    from types import SimpleNamespace
    from experiments.overcooked_v2.counterfactual_audit_app import (
        AuditWorld,
        TriggerIdentity,
        _panel_contrast_row,
    )

    zeros = np.asarray([0], dtype=np.int32)
    world = AuditWorld(
        environment_state=zeros,
        observations=np.zeros((1, 2, 1), dtype=np.float32),
        ego_state=zeros,
        partner_state=zeros,
        raw_return=np.asarray([10.0], dtype=np.float32),
        discounted_return=np.asarray([0.0], dtype=np.float32),
        correct_delivery_count=np.asarray([1], dtype=np.int32),
        wrong_delivery_count=zeros,
        indicator_activation_count=zeros,
        last_ego_action=zeros,
        last_response_code=zeros,
        last_reward=np.asarray([0.0], dtype=np.float32),
    )
    capture = SimpleNamespace(
        pre_world=world,
        trigger_action=np.asarray([2], dtype=np.int32),
    )
    continuations = [
        {
            "estimand": "pre_response",
            "replica_index": 0,
            "branch": "mask",
            "remaining_raw_return": 5.0,
            "remaining_correct_deliveries": 2,
            "remaining_wrong_deliveries": 0,
            "remaining_indicator_activations": 0,
        },
        {
            "estimand": "pre_response",
            "replica_index": 0,
            "branch": "use",
            "remaining_raw_return": 8.0,
            "remaining_correct_deliveries": 3,
            "remaining_wrong_deliveries": 0,
            "remaining_indicator_activations": 0,
        },
    ]
    row = _panel_contrast_row(
        identity=TriggerIdentity(
            source="fixed_partner_00",
            partner_index=0,
            episode_index=0,
            episode_seed=17,
            trigger_step=9,
        ),
        capture=capture,
        lane=0,
        pre_continuation_rows=continuations,
        a1_action=2,
        a1_world=None,
    )
    assert row["a1"] == row["a2_mask"]
    assert row["delta_cost"] == 0.0
    assert row["delta_net"] == row["delta_response"] == 3.0

    with pytest.raises(RuntimeError, match="distinct A1 action"):
        _panel_contrast_row(
            identity=TriggerIdentity(
                source="fixed_partner_00",
                partner_index=0,
                episode_index=0,
                episode_seed=17,
                trigger_step=9,
            ),
            capture=capture,
            lane=0,
            pre_continuation_rows=continuations,
            a1_action=3,
            a1_world=None,
        )
