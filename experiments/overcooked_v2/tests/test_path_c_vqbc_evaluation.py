from __future__ import annotations

from collections import Counter

import pytest

from src.path_c.vqbc.evaluation import (
    standard_episode_seed,
    standard_pairings,
)
from src.path_c.vqbc.response_contrast import (
    RESPONSE_CONTRAST_SCHEMA_VERSION,
    VQBCResponseContrastRow,
    effect_components,
    first_positive_trigger,
    summarize_response_contrast,
)


def test_four_modes_each_have_ten_self_and_ninety_directed_cross_pairings() -> None:
    pairings = standard_pairings()
    counts = Counter(
        (pairing.deployment_mode, pairing.split) for pairing in pairings
    )
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
        predicted_information_net_value=score,
        a1_raw_return=2.0,
        a2_mask_raw_return=1.0,
        a2_use_raw_return=4.0,
    )


def test_response_contrast_recomputes_three_effects_and_ten_bins() -> None:
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


def test_response_contrast_triggers_at_first_positive_information_net_value() -> None:
    step, value = first_positive_trigger(
        [
            [-1.0] * 6,
            [0.0] * 6,
            [-0.2, 0.3, -0.1, 0.0, 0.1, -0.4],
            [1.0] * 6,
        ]
    )
    assert step == 2
    assert value == pytest.approx(0.3)


def test_untriggered_response_contrast_requires_identical_branches() -> None:
    with pytest.raises(ValueError, match="byte-equivalent"):
        VQBCResponseContrastRow(
            schema_version=RESPONSE_CONTRAST_SCHEMA_VERSION,
            pairing_id="00_to_01",
            episode_index=0,
            episode_seed=1,
            triggered=False,
            trigger_step=None,
            predicted_information_net_value=None,
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
            predicted_information_net_value=None,
            a1_raw_return=1.0,
            a2_mask_raw_return=1.0,
            a2_use_raw_return=1.0,
        )
        for index in range(10)
    ]
    summary = summarize_response_contrast(rows)
    assert summary["trigger_count"] == 0
    assert summary["equal_frequency_bins"] == []
    assert summary["binning_status"] == (
        "fewer_than_ten_triggered_episodes"
    )
