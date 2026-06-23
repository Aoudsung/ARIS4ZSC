import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from toy_factor_game.run_symbolic_pilot import (  # noqa: E402
    DEFAULT_DELTA_ACTION,
    DEFAULT_DELTA_GAP,
    DEFAULT_DELTA_MI,
    DEFAULT_DELTA_OBS,
    DEFAULT_DELTA_RETURN,
    DEFAULT_DELTA_VALUE,
    DEFAULT_MIN_CASES,
    ROLLOUT_DEPTH,
    SYMBOLIC_SCHEMA,
    case_to_row,
    write_scenario_debug_csv,
    synthesize_diagnostic_cases,
    tiered_validation,
)


def test_symbolic_case_synthesizer_outputs_hard_filtered_cases():
    accepted, _rejected = synthesize_diagnostic_cases(
        max_steps=50,
        depth=ROLLOUT_DEPTH,
        likelihood_error=0.05,
        delta_gap=DEFAULT_DELTA_GAP,
        delta_action=DEFAULT_DELTA_ACTION,
        delta_obs=DEFAULT_DELTA_OBS,
        delta_return=DEFAULT_DELTA_RETURN,
        delta_mi=DEFAULT_DELTA_MI,
        delta_value=DEFAULT_DELTA_VALUE,
    )

    assert len(accepted) >= DEFAULT_MIN_CASES
    for case in accepted:
        assert case.passed_filters
        assert case.oracle_passive_gap is not None and case.oracle_passive_gap >= DEFAULT_DELTA_GAP
        assert case.best_response_flip
        assert case.action_gap is not None and case.action_gap >= DEFAULT_DELTA_ACTION
        assert case.observation_separation is not None and case.observation_separation >= DEFAULT_DELTA_OBS
        assert case.best_diag_return_gain is not None and case.best_diag_return_gain >= DEFAULT_DELTA_RETURN


def test_symbolic_high_mi_low_value_cases_satisfy_distractor_inequalities():
    accepted, _rejected = synthesize_diagnostic_cases(
        max_steps=50,
        depth=ROLLOUT_DEPTH,
        likelihood_error=0.05,
        delta_gap=DEFAULT_DELTA_GAP,
        delta_action=DEFAULT_DELTA_ACTION,
        delta_obs=DEFAULT_DELTA_OBS,
        delta_return=DEFAULT_DELTA_RETURN,
        delta_mi=DEFAULT_DELTA_MI,
        delta_value=DEFAULT_DELTA_VALUE,
    )
    distractors = [case for case in accepted if case.high_mi_low_value_distractor]

    assert distractors
    for case in distractors:
        assert case.mi_gap is not None and case.mi_gap >= DEFAULT_DELTA_MI
        assert case.delta_info_gap is not None and case.delta_info_gap >= DEFAULT_DELTA_VALUE
        assert case.distractor_return_gap is not None and case.distractor_return_gap >= DEFAULT_DELTA_RETURN


def test_symbolic_tier0_failure_short_circuits_method_tiers():
    validation = tiered_validation(
        summary={
            "oracle": {"episode_reward_mean": 1.0},
            "passive": {"episode_reward_mean": 1.0},
            "gtvoi": {"episode_reward_mean": 1.0, "diagnostic_cost_mean": 0.0},
            "mi": {"episode_reward_mean": 1.0, "diagnostic_cost_mean": 0.0},
            "random": {"episode_reward_mean": 0.0},
        },
        rows=[],
        case_rows=[],
        methods=["gtvoi", "mi", "passive", "random", "oracle"],
        min_cases=DEFAULT_MIN_CASES,
        delta_gap=DEFAULT_DELTA_GAP,
        delta_action=DEFAULT_DELTA_ACTION,
        delta_return=DEFAULT_DELTA_RETURN,
    )

    assert validation["overall_status"] == "FAIL_CASE_CONSTRUCTION"
    assert validation["tier0_case_validity"]["status"] == "FAIL_CASE_CONSTRUCTION"
    assert validation["tier1_oracle_sanity"]["status"] == "NOT_EVALUATED"
    assert validation["tier2_diagnostic_policy"]["status"] == "NOT_EVALUATED"


def test_symbolic_case_rows_include_phase3_metadata():
    accepted, _rejected = synthesize_diagnostic_cases(
        max_steps=50,
        depth=ROLLOUT_DEPTH,
        likelihood_error=0.05,
        delta_gap=DEFAULT_DELTA_GAP,
        delta_action=DEFAULT_DELTA_ACTION,
        delta_obs=DEFAULT_DELTA_OBS,
        delta_return=DEFAULT_DELTA_RETURN,
        delta_mi=DEFAULT_DELTA_MI,
        delta_value=DEFAULT_DELTA_VALUE,
    )
    row = case_to_row(accepted[0])

    assert SYMBOLIC_SCHEMA == "symbolic_sanity_v4_phase3"
    assert row["case_id"]
    assert row["oracle_passive_gap"] is not None
    assert row["best_diag_option"]
    assert row["passed_filters"] == 1


def test_symbolic_scenario_debug_csv_contains_requested_columns(tmp_path):
    accepted, _rejected = synthesize_diagnostic_cases(
        max_steps=50,
        depth=ROLLOUT_DEPTH,
        likelihood_error=0.05,
        delta_gap=DEFAULT_DELTA_GAP,
        delta_action=DEFAULT_DELTA_ACTION,
        delta_obs=DEFAULT_DELTA_OBS,
        delta_return=DEFAULT_DELTA_RETURN,
        delta_mi=DEFAULT_DELTA_MI,
        delta_value=DEFAULT_DELTA_VALUE,
    )
    case = accepted[0]
    case_rows = [case_to_row(case)]
    rows = [
        {
            "case_id": case.case_id,
            "case_type": case.case_type,
            "scenario": case.scenario.name,
            "method": "oracle",
            "convention": "0-0-0",
            "seed": 0,
            "episode_reward": 1.0,
            "regret_to_oracle": 0.0,
            "first_action": case.oracle_first_actions[0],
            "oracle_first_actions": ",".join(case.oracle_first_actions),
            "passive_first_action": case.passive_first_action,
            "max_delta_info": 0.2,
            "mean_delta_info": 0.1,
            "mean_mi_gain": 0.05,
            "future_return_delta_info_corr": 0.1,
            "future_return_mi_corr": 0.0,
            "diagnostic_cost": 0.0,
            "diagnostic_count": 0,
            "reward_after_first_diagnostic": None,
        },
        {
            "case_id": case.case_id,
            "case_type": case.case_type,
            "scenario": case.scenario.name,
            "method": "passive",
            "convention": "0-0-0",
            "seed": 0,
            "episode_reward": 0.5,
            "regret_to_oracle": 0.5,
            "first_action": case.passive_first_action,
            "oracle_first_actions": ",".join(case.oracle_first_actions),
            "passive_first_action": case.passive_first_action,
            "max_delta_info": 0.1,
            "mean_delta_info": 0.05,
            "mean_mi_gain": 0.05,
            "future_return_delta_info_corr": None,
            "future_return_mi_corr": None,
            "diagnostic_cost": 0.0,
            "diagnostic_count": 0,
            "reward_after_first_diagnostic": None,
        },
    ]
    path = tmp_path / "scenario_debug.csv"
    write_scenario_debug_csv(rows, case_rows, path)
    header = path.read_text().splitlines()[0].split(",")

    assert "case_id" in header
    assert "oracle_passive_gap_realized" in header
    assert "first_action_matches_oracle_rate" in header
    assert "best_diag_return_gain" in header
