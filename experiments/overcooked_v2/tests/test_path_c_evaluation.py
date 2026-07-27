from __future__ import annotations

from pathlib import Path
from statistics import mean, stdev

import numpy as np
import pytest

from src.path_c.evaluation import (
    DEPLOYMENT_MODES,
    EpisodeRow,
    Pairing,
    ResponseContrastRow,
    effect_components,
    standard_episode_seed,
    standard_pairings,
    summarize_development_rows,
    summarize_standard_rows,
    validate_development_response_contrast_rows,
    validate_development_rows,
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


def test_development_diagnostic_uses_one_matched_self_pairing() -> None:
    rows = tuple(
        _episode(mode, 0, 0, index)
        for mode in DEPLOYMENT_MODES
        for index in range(3)
    )
    validate_development_rows(rows, episodes_per_mode=3)
    summary = summarize_development_rows(rows)
    assert summary["run_kind"] == "development"
    assert summary["scientific_readout_allowed"] is False
    assert set(summary["deployment_modes"]) == set(DEPLOYMENT_MODES)


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
        predicted_next_policy_total_variation=0.04 if triggered else None,
        maximum_action_net_value=0.4 if triggered else None,
        maximum_action_policy_mediated_gain=0.08 if triggered else None,
        executed_action_net_value=0.2 if triggered else None,
        executed_action_response_value=0.3 if triggered else None,
        executed_action_policy_mediated_gain=0.06 if triggered else None,
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


def test_development_response_contrast_uses_the_same_self_pairing_seed() -> None:
    rows = tuple(
        _contrast_row(left=0, right=0, index=index, triggered=False)
        for index in range(3)
    )
    validate_development_response_contrast_rows(
        rows,
        evaluation_seed=17,
        layout="test_time_simple",
        episodes_per_pairing=3,
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


def test_development_population_contains_only_outer_unit_zero(
    tmp_path: Path,
) -> None:
    population = Population(
        name="seed-100-development",
        layout="test_time_simple",
        evaluation_kind="development_diagnostic",
        entries=(PopulationEntry(0, tmp_path / "run-0"),),
    )
    path = write_population(tmp_path / "development-population.json", population)
    assert load_population(path) == population


def test_directed_pairing_does_not_duplicate_seat_swap() -> None:
    forward = Pairing("posterior_use", "xp", 2, 7)
    reverse = Pairing("posterior_use", "xp", 7, 2)
    assert forward.pairing_id == "02_to_07"
    assert reverse.pairing_id == "07_to_02"
    assert forward != reverse
