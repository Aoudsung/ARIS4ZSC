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
    DEFAULT_MAX_DIAGNOSTIC_COST,
    DEFAULT_MIN_CASES,
    DEFAULT_ORACLE_MARGIN,
    DiagnosticCase,
    ROLLOUT_DEPTH,
    SYMBOLIC_SCHEMA,
    case_to_row,
    evaluate_candidate_case,
    scenario_convention_pairs,
    candidate_scenarios,
    write_scenario_debug_csv,
    write_scenario_debug_views,
    synthesize_diagnostic_cases,
    tiered_validation,
)


def test_symbolic_case_synthesizer_outputs_hard_filtered_cases():
    accepted, rejected = synthesize_diagnostic_cases(
        max_steps=50,
        depth=ROLLOUT_DEPTH,
        likelihood_error=0.05,
        delta_gap=DEFAULT_DELTA_GAP,
        delta_action=DEFAULT_DELTA_ACTION,
        delta_obs=DEFAULT_DELTA_OBS,
        delta_return=DEFAULT_DELTA_RETURN,
        delta_mi=DEFAULT_DELTA_MI,
        delta_value=DEFAULT_DELTA_VALUE,
        max_diagnostic_cost=DEFAULT_MAX_DIAGNOSTIC_COST,
    )

    for case in accepted:
        assert case.passed_filters
        assert case.oracle_passive_gap is not None and case.oracle_passive_gap >= DEFAULT_DELTA_GAP
        assert case.best_response_flip
        assert case.action_gap is not None and case.action_gap >= DEFAULT_DELTA_ACTION
        assert case.observation_separation is not None and case.observation_separation >= DEFAULT_DELTA_OBS
        assert case.best_diag_return_gain is not None and case.best_diag_return_gain >= DEFAULT_DELTA_RETURN
        assert case.diagnostic_opportunity_cost is not None
        assert case.diagnostic_opportunity_cost <= DEFAULT_MAX_DIAGNOSTIC_COST
    if not accepted:
        assert rejected


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
        max_diagnostic_cost=DEFAULT_MAX_DIAGNOSTIC_COST,
    )
    for case in [case for case in accepted if case.high_mi_low_value_distractor]:
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
        oracle_margin=DEFAULT_ORACLE_MARGIN,
    )

    assert validation["overall_status"] == "NO_DIAGNOSTIC_CRITICAL_CASES_FOUND"
    assert validation["tier0_case_validity"]["status"] == "NO_DIAGNOSTIC_CRITICAL_CASES_FOUND"
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
        max_diagnostic_cost=DEFAULT_MAX_DIAGNOSTIC_COST,
    )
    case = accepted[0] if accepted else DiagnosticCase(
        case_id="synthetic_case",
        case_type="diagnostic_critical",
        scenario=None,
        conventions=((0, 0, 0), (1, 0, 0)),
        oracle_passive_gap=DEFAULT_DELTA_GAP,
        best_diag_option="wait_at_bottleneck",
        best_diag_return_gain=DEFAULT_DELTA_RETURN,
        passed_filters=True,
    )
    row = case_to_row(case)

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
        max_diagnostic_cost=DEFAULT_MAX_DIAGNOSTIC_COST,
    )
    case = accepted[0] if accepted else DiagnosticCase(
        case_id="synthetic_case",
        case_type="diagnostic_critical",
        scenario=None,
        conventions=((0, 0, 0), (1, 0, 0)),
        oracle_passive_gap=DEFAULT_DELTA_GAP,
        oracle_first_actions=("wait_at_bottleneck", "cross_corridor"),
        passive_first_action="wait_at_bottleneck",
        best_response_flip=True,
        action_gap=DEFAULT_DELTA_ACTION,
        best_diag_option="wait_at_bottleneck",
        observation_separation=DEFAULT_DELTA_OBS,
        diagnostic_opportunity_cost=0.0,
        best_diag_return_gain=DEFAULT_DELTA_RETURN,
        passed_filters=True,
    )
    case_rows = [case_to_row(case)]
    rows = [
        {
            "case_id": case.case_id,
            "case_type": case.case_type,
            "scenario": case.scenario.name if case.scenario is not None else "synthetic",
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
            "max_mi": 0.08,
            "future_return_delta_info_corr": 0.1,
            "future_return_mi_corr": 0.0,
            "future_reward_gain_after_high_delta_info": 0.2,
            "oracle_gap_after_high_delta_info": 0.0,
            "high_delta_info_count": 1,
            "diagnostic_cost": 0.0,
            "diagnostic_count": 0,
            "reward_after_first_diagnostic": None,
        },
        {
            "case_id": case.case_id,
            "case_type": case.case_type,
            "scenario": case.scenario.name if case.scenario is not None else "synthetic",
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
            "max_mi": 0.06,
            "future_return_delta_info_corr": None,
            "future_return_mi_corr": None,
            "future_reward_gain_after_high_delta_info": 0.0,
            "oracle_gap_after_high_delta_info": 0.5,
            "high_delta_info_count": 0,
            "diagnostic_cost": 0.0,
            "diagnostic_count": 0,
            "reward_after_first_diagnostic": None,
        },
    ]
    path = tmp_path / "scenario_debug.csv"
    debug_rows = write_scenario_debug_csv(rows, case_rows, path)
    views = write_scenario_debug_views(debug_rows, tmp_path, top_k=3)
    header = path.read_text().splitlines()[0].split(",")

    assert "case_id" in header
    assert "oracle_passive_gap_realized" in header
    assert "first_action_matches_oracle_rate" in header
    assert "best_diag_return_gain" in header
    assert "max_mi" in header
    assert "future_reward_gain_after_high_delta_info" in header
    assert set(views) == {
        "top_oracle_passive_gap",
        "top_max_delta_info",
        "top_delta_info_but_low_return",
        "top_mi_but_low_return",
    }
    for metadata in views.values():
        assert (tmp_path / metadata["file"]).exists()


def test_symbolic_tier1_requires_oracle_margin():
    case_rows = [
        {
            "case_id": "case",
            "passed_filters": 1,
            "oracle_passive_gap": DEFAULT_DELTA_GAP,
            "action_gap": DEFAULT_DELTA_ACTION,
            "best_diag_return_gain": DEFAULT_DELTA_RETURN,
            "best_response_flip": 1,
        }
        for _ in range(DEFAULT_MIN_CASES)
    ]
    validation = tiered_validation(
        summary={
            "oracle": {"episode_reward_mean": 1.0},
            "passive": {"episode_reward_mean": 1.0},
            "gtvoi": {"episode_reward_mean": 1.0, "diagnostic_cost_mean": 0.0},
            "mi": {"episode_reward_mean": 1.0, "diagnostic_cost_mean": 0.0},
            "random": {"episode_reward_mean": 0.0},
        },
        rows=[],
        case_rows=case_rows,
        methods=["gtvoi", "mi", "passive", "random", "oracle"],
        min_cases=DEFAULT_MIN_CASES,
        delta_gap=DEFAULT_DELTA_GAP,
        delta_action=DEFAULT_DELTA_ACTION,
        delta_return=DEFAULT_DELTA_RETURN,
        oracle_margin=DEFAULT_ORACLE_MARGIN,
    )

    assert not validation["tier1_oracle_sanity"]["checks"]["oracle_ge_passive_plus_margin"]
    assert validation["tier1_oracle_sanity"]["status"] == "FAIL"


def test_symbolic_case_filter_uses_two_sided_action_gap():
    scenario = next(s for s in candidate_scenarios() if "distractor" in s.name)
    pair = scenario_convention_pairs(scenario)[0]
    case = evaluate_candidate_case(
        scenario=scenario,
        conventions=pair,
        case_idx=0,
        max_steps=50,
        depth=ROLLOUT_DEPTH,
        likelihood_error=0.05,
        delta_gap=DEFAULT_DELTA_GAP,
        delta_action=10_000.0,
        delta_obs=DEFAULT_DELTA_OBS,
        delta_return=DEFAULT_DELTA_RETURN,
        delta_mi=DEFAULT_DELTA_MI,
        delta_value=DEFAULT_DELTA_VALUE,
        max_diagnostic_cost=DEFAULT_MAX_DIAGNOSTIC_COST,
    )

    assert not case.passed_filters
    assert case.failure_reason in {"action_gap", "oracle_passive_gap"}


def test_symbolic_diagnostic_cost_filter_rejects_expensive_options():
    scenario = next(s for s in candidate_scenarios() if "distractor" in s.name)
    pair = scenario_convention_pairs(scenario)[0]
    case = evaluate_candidate_case(
        scenario=scenario,
        conventions=pair,
        case_idx=0,
        max_steps=50,
        depth=ROLLOUT_DEPTH,
        likelihood_error=0.05,
        delta_gap=DEFAULT_DELTA_GAP,
        delta_action=DEFAULT_DELTA_ACTION,
        delta_obs=DEFAULT_DELTA_OBS,
        delta_return=DEFAULT_DELTA_RETURN,
        delta_mi=DEFAULT_DELTA_MI,
        delta_value=DEFAULT_DELTA_VALUE,
        max_diagnostic_cost=0.0,
    )

    assert not case.passed_filters
    assert case.failure_reason is not None


def test_symbolic_high_delta_low_return_artifact_is_labeled():
    scenario = next(s for s in candidate_scenarios() if "distractor" in s.name)
    pair = scenario_convention_pairs(scenario)[0]
    case = evaluate_candidate_case(
        scenario=scenario,
        conventions=pair,
        case_idx=0,
        max_steps=50,
        depth=ROLLOUT_DEPTH,
        likelihood_error=0.05,
        delta_gap=DEFAULT_DELTA_GAP,
        delta_action=DEFAULT_DELTA_ACTION,
        delta_obs=DEFAULT_DELTA_OBS,
        delta_return=10_000.0,
        delta_mi=DEFAULT_DELTA_MI,
        delta_value=DEFAULT_DELTA_VALUE,
        max_diagnostic_cost=DEFAULT_MAX_DIAGNOSTIC_COST,
    )

    assert not case.passed_filters
    assert case.failure_reason is not None
    if case.failure_reason == "HIGH_DELTA_INFO_LOW_RETURN_ARTIFACT":
        assert case.best_diag_delta_info is not None
