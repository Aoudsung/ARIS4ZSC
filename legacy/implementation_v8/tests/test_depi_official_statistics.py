from __future__ import annotations

import numpy as np
import pytest

from src.path_c.official_statistics import (
    common_partner_summary,
    official_node_bootstrap,
    official_pairings,
    official_scoreboard_summary,
    official_two_method_bootstrap,
    registered_superiority_gate,
    rows_to_official_cube,
)


def test_official_pairing_and_episode_counts_are_exact() -> None:
    pairings = official_pairings()
    assert len(pairings) == 100
    assert len([pair for pair in pairings if pair[0] != pair[1]]) == 90
    assert len([pair for pair in pairings if pair[0] == pair[1]]) == 10
    rows = [
        {
            "left_run_index": left,
            "right_run_index": right,
            "episode_index": episode,
            "raw_return": float(100 * left + 10 * right + episode),
        }
        for left, right in pairings
        for episode in range(500)
    ]
    assert len(rows) == 50_000
    cube = rows_to_official_cube(rows)
    assert cube.shape == (10, 10, 500)


def test_official_sp_xp_aggregation_uses_cells_not_a_gap_sd() -> None:
    cube = np.zeros((10, 10, 500), dtype=np.float64)
    for left in range(10):
        for right in range(10):
            cube[left, right] = 100.0 + left if left == right else float(left - right)
    summary = official_scoreboard_summary(cube)
    assert summary["sp_mean"] == pytest.approx(104.5)
    assert summary["xp_mean"] == pytest.approx(0.0)
    assert summary["gap_point"] == pytest.approx(104.5)
    assert "gap_population_sd" not in summary


def test_node_bootstrap_compares_delta_to_replica_wise_best_baseline() -> None:
    cubes = {
        "depi": np.full((10, 10, 500), 5.0),
        "sp": np.full((10, 10, 500), 1.0),
        "state-augmented": np.full((10, 10, 500), 2.0),
        "op": np.full((10, 10, 500), 3.0),
        "fcp": np.full((10, 10, 500), 4.0),
    }
    result = official_node_bootstrap(
        cubes,
        target_method="depi",
        baseline_methods=("sp", "state-augmented", "op", "fcp"),
        fcp_method="fcp",
        replicates=20,
        seed=7,
    )
    assert result["inference_mode"] == "independent_run"
    assert result["depi_vs_best_baseline"]["one_sided_lcb"] == pytest.approx(1.0)
    assert result["depi_sp_minus_fcp_sp"]["one_sided_lcb"] == pytest.approx(1.0)


def test_superiority_requires_both_positive_lcb_and_registered_material_effect() -> None:
    insufficient = registered_superiority_gate(
        {"one_sided_lcb": 5.0, "point_reference": 19.0},
        lcb_threshold=0.0,
        minimum_effect=20.0,
        minimum_effect_rule="point_estimate",
    )
    assert insufficient == {
        "passed": False,
        "superiority_lcb_passed": True,
        "minimum_effect_passed": False,
    }
    sufficient = registered_superiority_gate(
        {"one_sided_lcb": 1.0, "point_reference": 20.0},
        lcb_threshold=0.0,
        minimum_effect=20.0,
        minimum_effect_rule="point_estimate",
    )
    assert sufficient["passed"]


def test_capacity_bootstrap_uses_independent_nodes_and_common_episode_indexes() -> None:
    result = official_two_method_bootstrap(
        np.full((10, 10, 500), 7.0),
        np.full((10, 10, 500), 5.0),
        left_name="depi",
        right_name="ippo-large",
        replicates=20,
        seed=11,
    )
    assert result["point_reference"] == pytest.approx(2.0)
    assert result["one_sided_lcb"] == pytest.approx(2.0)


def test_common_summary_is_mechanism_balanced_and_delta_only_negative_transfer() -> None:
    rows = []
    for ego in range(10):
        for mechanism in range(4):
            for partner in range(4):
                for role in range(2):
                    for episode in range(500):
                        rows.append(
                            {
                                "ego_run_id": f"ego-{ego}",
                                "partner_run_id": f"partner-{mechanism}-{partner}",
                                "partner_mechanism": f"mechanism-{mechanism}",
                                "ego_role": role,
                                "episode_index": episode,
                                "raw_return": float(mechanism),
                            }
                        )
    summary = common_partner_summary(rows)
    assert summary["mean_common_partner_return"] == pytest.approx(1.5)
    assert summary["negative_transfer_rate"] is None
