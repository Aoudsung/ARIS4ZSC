from __future__ import annotations

from collections import Counter
import inspect

import pytest

from src.path_c.vqbc.evaluation import standard_episode_seed, standard_pairings
from src.path_c.vqbc.response_contrast import (
    RESPONSE_CONTRAST_SCHEMA_VERSION,
    VQBCResponseContrastRow,
    effect_components,
    first_positive_trigger,
    summarize_response_contrast,
)


def test_four_modes_each_have_ten_self_and_ninety_directed_cross_pairings() -> None:
    pairings = standard_pairings()
    counts = Counter((pairing.deployment_mode, pairing.split) for pairing in pairings)
    for mode in (
        "posterior_use",
        "prior_only",
        "reference_only",
        "generic_response_information",
    ):
        assert counts[(mode, "sp")] == 10
        assert counts[(mode, "xp")] == 90


def test_episode_seed_is_shared_across_deployment_modes() -> None:
    seeds = {
        standard_episode_seed(
            population_id="ten-independent-vqbc-policies",
            layout="test_time_simple",
            left_outer_unit_id=2,
            right_outer_unit_id=7,
            episode_index=19,
        )
        for unused_mode in range(4)
    }
    assert len(seeds) == 1


def _row(index: int, score: float) -> VQBCResponseContrastRow:
    return VQBCResponseContrastRow(
        schema_version=RESPONSE_CONTRAST_SCHEMA_VERSION,
        pairing_id="00_to_01",
        episode_index=index,
        episode_seed=index + 100,
        triggered=True,
        trigger_step=index,
        trigger_tolerance=1.0e-5,
        predicted_response_effect=score + 0.5,
        predicted_policy_cost=0.5,
        predicted_net_effect=score,
        predicted_regularized_net_effect=score - 0.1,
        predicted_policy_total_variation=0.2,
        maximum_action_net_value=score + 1.0,
        executed_action_net_value=score - 0.2,
        executed_action_response_value=score + 0.1,
        executed_action=2,
        maximum_net_action=1,
        post_response_belief_l1=0.4,
        first_left_action_difference_step=index,
        first_observation_difference_step=index,
        first_response_code_difference_step=index,
        first_reward_difference_step=index,
        left_action_difference_count=1,
        observation_difference_count=1,
        response_code_difference_count=1,
        reward_difference_count=1,
        a1_raw_return=2.0,
        a2_mask_raw_return=1.0,
        a2_use_raw_return=4.0,
    )


def test_response_contrast_recomputes_effects_and_bins_policy_level_prediction() -> None:
    row = _row(0, 0.1)
    assert effect_components(row) == {
        "A2-use_minus_A2-mask": 3.0,
        "A1_minus_A2-mask": 1.0,
        "A2-use_minus_A1": 2.0,
    }
    summary = summarize_response_contrast(
        [_row(index, float(index + 1)) for index in range(10)]
    )
    assert len(summary["equal_frequency_bins"]) == 10
    assert summary["highest_bin_primary_effect"] == 3.0
    assert summary["mean_predicted_policy_total_variation"] == pytest.approx(0.2)


def test_response_contrast_trigger_uses_policy_effect_not_action_maximum() -> None:
    step, value = first_positive_trigger(
        [-0.1, 2.0e-7, 0.3, 1.0],
        [1.0e-6, 1.0e-6, 1.0e-6, 1.0e-6],
    )
    assert step == 2
    assert value == pytest.approx(0.3)


def test_untriggered_response_contrast_requires_identical_branches() -> None:
    with pytest.raises(ValueError, match="identical"):
        VQBCResponseContrastRow(
            schema_version=RESPONSE_CONTRAST_SCHEMA_VERSION,
            pairing_id="00_to_01",
            episode_index=0,
            episode_seed=1,
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
            a1_raw_return=1.0,
            a2_mask_raw_return=1.0,
            a2_use_raw_return=2.0,
        )


def test_untriggered_development_contrast_reports_without_bins() -> None:
    rows = [
        VQBCResponseContrastRow(
            schema_version=RESPONSE_CONTRAST_SCHEMA_VERSION,
            pairing_id="development_self_play",
            episode_index=index,
            episode_seed=index + 1,
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
            a1_raw_return=1.0,
            a2_mask_raw_return=1.0,
            a2_use_raw_return=1.0,
        )
        for index in range(10)
    ]
    summary = summarize_response_contrast(rows)
    assert summary["trigger_count"] == 0
    assert summary["equal_frequency_bins"] == []
    assert summary["binning_status"] == "fewer_than_ten_triggered_episodes"


def test_a1_and_a2_mask_both_block_the_first_response() -> None:
    from experiments.overcooked_v2.model_dock import vqbc_response_contrast_runtime

    source = inspect.getsource(
        vqbc_response_contrast_runtime._make_contrast_batch_runner
    )
    # Both branches pass newly_triggered to the response mask at the fork.
    assert source.count("left_mask_response=newly_triggered") >= 2
    assert "mask_execution_logits" in inspect.getsource(
        vqbc_response_contrast_runtime._a1_step_from_use
    )
