from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import pytest

from src.path_c.evaluation import (
    DEPLOYMENT_MODES,
    EpisodeRow,
    Pairing,
    Population,
    PopulationEntry,
    ResponseContrastRow,
    effect_components,
    load_population,
    standard_episode_seed,
    standard_pairings,
    summarize_standard_rows,
    validate_response_contrast_rows,
    validate_standard_rows,
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
        positive_predicted_response_effect_count=0,
        cumulative_kl=0.0,
        reference_action_deviation_count=0,
        mean_value_class_count=1.0,
        mean_belief_entropy=0.0,
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
            for unused_mode in DEPLOYMENT_MODES
        }
        assert len(seeds) == 1


def test_official_summary_uses_pairing_means_and_pairing_standard_deviation() -> None:
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
    np.testing.assert_allclose(summary["sp"]["pairing_standard_deviation"], stdev(sp_values))
    np.testing.assert_allclose(summary["xp"]["mean_raw_return"], mean(xp_values))
    np.testing.assert_allclose(summary["xp"]["pairing_standard_deviation"], stdev(xp_values))
    np.testing.assert_allclose(
        summary["sp_minus_xp"], mean(sp_values) - mean(xp_values)
    )


def test_response_mask_effects_follow_literal_three_branch_identity() -> None:
    row = ResponseContrastRow(
        pairing_id="01_to_07",
        episode_index=4,
        episode_seed=10,
        triggered=True,
        trigger_step=12,
        trigger_tolerance=0.01,
        predicted_response_effect=0.2,
        predicted_policy_cost=0.1,
        predicted_net_effect=0.1,
        predicted_regularized_net_effect=0.08,
        predicted_policy_total_variation=0.03,
        maximum_action_net_value=0.4,
        executed_action_net_value=0.2,
        executed_action_response_value=0.3,
        executed_action=5,
        maximum_net_action=5,
        post_response_belief_l1=0.6,
        first_left_action_difference_step=13,
        first_observation_difference_step=14,
        first_response_code_difference_step=12,
        first_reward_difference_step=20,
        left_action_difference_count=3,
        observation_difference_count=2,
        response_code_difference_count=1,
        reward_difference_count=1,
        environment_steps=1_200,
        a1_raw_return=7.0,
        a1_correct_delivery_count=1,
        a1_wrong_delivery_count=0,
        a1_indicator_activation_count=2,
        a2_mask_raw_return=3.0,
        a2_mask_correct_delivery_count=1,
        a2_mask_wrong_delivery_count=1,
        a2_mask_indicator_activation_count=3,
        a2_use_raw_return=11.0,
        a2_use_correct_delivery_count=2,
        a2_use_wrong_delivery_count=0,
        a2_use_indicator_activation_count=2,
    )
    assert effect_components(row) == {
        "delta_response": 8.0,
        "delta_cost": 4.0,
        "delta_net": 4.0,
    }


def test_directed_pairing_does_not_duplicate_a_second_seat_swap() -> None:
    forward = Pairing("posterior_use", "xp", 2, 7)
    reverse = Pairing("posterior_use", "xp", 7, 2)
    assert forward.pairing_id == "02_to_07"
    assert reverse.pairing_id == "07_to_02"
    assert forward != reverse


def test_population_manifest_has_ten_distinct_public_policy_entries(
    tmp_path: Path,
) -> None:
    population = Population(
        name="posterior_use",
        layout="test_time_simple",
        evaluation_kind="standard_matrix",
        entries=tuple(
            PopulationEntry(
                outer_unit_id=index,
                checkpoint_path=tmp_path / f"trained_{index}",
                reference_checkpoint_path=tmp_path / f"reference_{index}",
            )
            for index in range(10)
        ),
    )
    path = write_population(tmp_path / "population.json", population)
    assert load_population(path) == population

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["unused_identity"] = "ignored-by-no-consumer"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fields"):
        load_population(path)


def test_response_contrast_requires_complete_directed_xp_rows() -> None:
    rows = tuple(
        ResponseContrastRow(
            pairing_id=f"{left:02d}_to_{right:02d}",
            episode_index=index,
            episode_seed=standard_episode_seed(
                evaluation_seed=17,
                layout="test_time_simple",
                left_outer_unit_id=left,
                right_outer_unit_id=right,
                episode_index=index,
            ),
            triggered=False,
            trigger_step=None,
            trigger_tolerance=None,
            predicted_response_effect=None,
            predicted_policy_cost=None,
            predicted_net_effect=None,
            predicted_regularized_net_effect=None,
            predicted_policy_total_variation=None,
            maximum_action_net_value=None,
            executed_action_net_value=None,
            executed_action_response_value=None,
            executed_action=None,
            maximum_net_action=None,
            post_response_belief_l1=None,
            first_left_action_difference_step=None,
            first_observation_difference_step=None,
            first_response_code_difference_step=None,
            first_reward_difference_step=None,
            left_action_difference_count=None,
            observation_difference_count=None,
            response_code_difference_count=None,
            reward_difference_count=None,
            environment_steps=1_200,
            a1_raw_return=2.0,
            a1_correct_delivery_count=0,
            a1_wrong_delivery_count=0,
            a1_indicator_activation_count=0,
            a2_mask_raw_return=2.0,
            a2_mask_correct_delivery_count=0,
            a2_mask_wrong_delivery_count=0,
            a2_mask_indicator_activation_count=0,
            a2_use_raw_return=2.0,
            a2_use_correct_delivery_count=0,
            a2_use_wrong_delivery_count=0,
            a2_use_indicator_activation_count=0,
        )
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
