import unittest

from experiments.serd.fixture_env import CONTROL_FAMILIES, make_fixture_records
from experiments.serd.jaxmarl_overcooked_v2_adapter import (
    action_divergence as v2_action_divergence,
    control_joint as v2_control_joint,
    semantic_joint as v2_semantic_joint,
)
from experiments.serd.overcooked_ai_adapter import (
    OvercookedSmokeConfig,
    _action_divergence,
    _control_joint,
    _semantic_joint,
    _scripted_branches,
)
from experiments.serd.serd_core import (
    BranchRecord,
    normalized_pair_distance,
    require_family_coverage,
    standardized_mean_difference,
    summarize_family_serd,
    summarize_pre_h_balance,
    summarize_worst_serd,
)


class SerdCoreTest(unittest.TestCase):
    def test_pair_distance_rejects_mismatched_keys(self):
        with self.assertRaises(ValueError):
            normalized_pair_distance({"a": 1.0}, {"b": 1.0})

    def test_fixture_produces_family_and_worst_summaries(self):
        semantic, controls = make_fixture_records(probes=16, seed=7)
        family = summarize_family_serd(
            semantic,
            controls,
            epsilon_shock=0.05,
            epsilon_phi=0.05,
            delta_serd=0.05,
        )
        self.assertEqual(require_family_coverage(family, CONTROL_FAMILIES), [])
        worst = summarize_worst_serd(family, delta_serd=0.05)
        by_policy = {item.policy: item for item in worst}
        self.assertEqual(by_policy["pecan_fixture"].classification, "survival")
        self.assertNotEqual(by_policy["fcp_fixture"].classification, "survival")

    def test_invalid_match_is_excluded(self):
        semantic = [
            BranchRecord(
                probe_id="p0",
                policy="pi",
                domain="d",
                disruption="e",
                family="semantic",
                no_shock_return=10.0,
                branch_return=8.0,
                shock_magnitude=1.0,
                phi_pre_h={"x": 0.0},
            )
        ]
        controls = [
            BranchRecord(
                probe_id="p0",
                policy="pi",
                domain="d",
                disruption="e",
                family="random_lag",
                no_shock_return=10.0,
                branch_return=7.0,
                shock_magnitude=2.0,
                phi_pre_h={"x": 0.0},
            )
        ]
        family = summarize_family_serd(
            semantic,
            controls,
            epsilon_shock=0.05,
            epsilon_phi=0.05,
            delta_serd=0.05,
        )
        self.assertEqual(family, [])

    def test_overcooked_smoke_scripts_have_matched_shock_magnitudes(self):
        config = OvercookedSmokeConfig(horizon=8, shock_horizon=1)
        no_shock, branches = _scripted_branches(config.horizon)
        semantic_magnitude = _action_divergence(
            no_shock, branches["semantic"], config.shock_horizon
        )
        for family in CONTROL_FAMILIES:
            self.assertEqual(
                _action_divergence(no_shock, branches[family], config.shock_horizon),
                semantic_magnitude,
            )

    def test_pre_h_balance_reports_zero_for_same_state_branching(self):
        semantic = [
            BranchRecord(
                probe_id="p0",
                policy="pi",
                domain="d",
                disruption="missed_handoff",
                family="semantic",
                no_shock_return=10.0,
                branch_return=8.0,
                shock_magnitude=0.5,
                phi_pre_h={"x": 1.0, "y": 2.0},
            ),
            BranchRecord(
                probe_id="p1",
                policy="pi",
                domain="d",
                disruption="missed_handoff",
                family="semantic",
                no_shock_return=10.0,
                branch_return=8.0,
                shock_magnitude=0.5,
                phi_pre_h={"x": 2.0, "y": 3.0},
            ),
        ]
        controls = [
            BranchRecord(
                probe_id=record.probe_id,
                policy=record.policy,
                domain=record.domain,
                disruption=record.disruption,
                family="random_lag",
                no_shock_return=10.0,
                branch_return=7.0,
                shock_magnitude=0.5,
                phi_pre_h=dict(record.phi_pre_h),
            )
            for record in semantic
        ]
        balance = summarize_pre_h_balance(
            semantic,
            controls,
            epsilon_shock=0.001,
            epsilon_phi=0.0,
        )
        self.assertEqual({row.covariate for row in balance}, {"x", "y"})
        self.assertTrue(all(row.smd == 0.0 for row in balance))
        self.assertEqual(standardized_mean_difference([1.0, 2.0], [1.0, 2.0]), 0.0)

    def test_policy_interventions_keep_one_action_shock(self):
        base_joint = ("east", "west")
        for disruption in ("missed_handoff", "route_block", "hesitation"):
            semantic_joint = _semantic_joint(base_joint, disruption, probe_index=0)
            semantic_magnitude = _action_divergence([base_joint], [semantic_joint], 1)
            self.assertEqual(semantic_magnitude, 0.5)
            for family in CONTROL_FAMILIES:
                control_joint = _control_joint(
                    base_joint,
                    disruption,
                    family,
                    probe_index=0,
                )
                self.assertEqual(
                    _action_divergence([base_joint], [control_joint], 1),
                    semantic_magnitude,
                )

    def test_overcooked_v2_interventions_keep_one_action_shock(self):
        base_joint = ("right", "left")
        for disruption in ("missed_handoff", "route_block", "hesitation"):
            semantic_joint = v2_semantic_joint(base_joint, disruption, probe_index=0)
            semantic_magnitude = v2_action_divergence([base_joint], [semantic_joint], 1)
            self.assertEqual(semantic_magnitude, 0.5)
            for family in CONTROL_FAMILIES:
                control_joint = v2_control_joint(
                    base_joint,
                    disruption,
                    family,
                    probe_index=0,
                )
                self.assertEqual(
                    v2_action_divergence([base_joint], [control_joint], 1),
                    semantic_magnitude,
                )


if __name__ == "__main__":
    unittest.main()
