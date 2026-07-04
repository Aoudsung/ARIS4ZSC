"""Regressions for scripts/aggregate_e1rev.py (PM-7 aggregation, FIX_PLAN F5).

Rules under test (SWEEP_SUMMARY / METHOD_LOCK sec18.12.3):
  * LDS-C1 consumer-side defense: non-final metrics.json HARD-fails aggregation;
  * guard-fail seeds count in the terminal-competence DENOMINATOR but are
    excluded from performance means;
  * eval mode is fail-closed: a missing preregistered field is a schema error,
    and null is never imputed as zero (LDS-C4) — with ONE preregistered
    exception: ego_serve_share may be null ("nobody delivered correctly" ⇒
    share undefined), which is excluded from the CI with n_undefined recorded.

The script only needs numpy, so these tests run everywhere (no env stack).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments/overcooked_v2/scripts/aggregate_e1rev.py"
)
_spec = importlib.util.spec_from_file_location("aggregate_e1rev", _SCRIPT)
agg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agg)


def _write_run(
    root: Path,
    method: str,
    seed: int,
    *,
    final=True,
    run_status="ok",
    updates=5000,
    guard="pass",
    selected="checkpoint.pt",
    ccr=0.8,
    sole=3,
    ret=40.0,
):
    run_dir = (
        root
        / f"e1rev_{method}_s{seed}"
        / "asymm_advantages"
        / method
        / "full_support"
        / f"seed{seed}"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "final": final,
                "run_status": run_status,
                "updates_done": updates,
                "free_rider_guard": guard,
                "checkpoint_selection": {"deployable_checkpoint": selected},
                "selected_ego_correct_completion_rate": ccr,
                "selected_ego_sole_correct_delivery_count": sole,
                "best_greedy_return": ret,
            }
        )
    )


def test_train_mode_happy_path_tables(tmp_path: Path):
    _write_run(tmp_path, "aris_bellman", 0, ccr=0.4, sole=2, ret=46.2)
    _write_run(tmp_path, "aris_bellman", 1, ccr=0.8, sole=9, ret=52.5)
    doc = agg.aggregate_train(str(tmp_path), "e1rev", 5000)
    arm = doc["arms"]["aris_bellman"]
    assert arm["n_seeds"] == 2
    assert arm["guard_pass_count"] == 2
    assert arm["terminal_competence_rate"] == 1.0
    assert arm["mean_selected_ego_correct_completion_rate_guardpass"] == pytest.approx(0.6)


def test_nonfinal_run_hard_fails_aggregation(tmp_path: Path):
    """LDS-C1: a periodic (non-final) metrics.json must poison the WHOLE table."""
    _write_run(tmp_path, "aris_bellman", 0)
    _write_run(tmp_path, "base_only", 0, final=False, run_status=None, updates=1200)
    with pytest.raises(agg.AggregationError, match="final"):
        agg.aggregate_train(str(tmp_path), "e1rev", 5000)


def test_wrong_update_count_hard_fails(tmp_path: Path):
    _write_run(tmp_path, "aris_bellman", 0, updates=4999)
    with pytest.raises(agg.AggregationError, match="updates_done"):
        agg.aggregate_train(str(tmp_path), "e1rev", 5000)


def test_guard_fail_in_denominator_but_not_in_means(tmp_path: Path):
    """sec18.12.3 co-primary: guard-fail seeds shrink competence rate, and their
    (None) performance fields never contaminate the guard-pass means."""
    _write_run(tmp_path, "aris_bellman", 0, ccr=0.6, sole=3, ret=30.0)
    _write_run(
        tmp_path, "aris_bellman", 1,
        guard="fail", selected=None, ccr=None, sole=None, ret=0.0,
    )
    doc = agg.aggregate_train(str(tmp_path), "e1rev", 5000)
    arm = doc["arms"]["aris_bellman"]
    assert arm["n_seeds"] == 2  # denominator includes the guard-fail seed
    assert arm["guard_pass_count"] == 1
    assert arm["terminal_competence_rate"] == 0.5
    assert arm["mean_selected_ego_correct_completion_rate_guardpass"] == pytest.approx(0.6)


def test_duplicate_method_seed_hard_fails(tmp_path: Path):
    """The orchestrator data-loss incident (19b653d) must be detectable: two
    artifacts claiming the same (method, seed) is corruption, not data."""
    _write_run(tmp_path, "aris_bellman", 0)
    # second artifact for the same run id, nested under a different subdir name
    dup = tmp_path / "e1rev_aris_bellman_s0" / "retry" / "seed0"
    dup.mkdir(parents=True)
    (dup / "metrics.json").write_text(
        json.dumps(
            {
                "final": True,
                "run_status": "ok",
                "updates_done": 5000,
                "free_rider_guard": "pass",
                "checkpoint_selection": {"deployable_checkpoint": "checkpoint.pt"},
            }
        )
    )
    with pytest.raises(agg.AggregationError, match="Duplicate"):
        agg.aggregate_train(str(tmp_path), "e1rev", 5000)


def test_empty_deployable_string_is_not_competent(tmp_path: Path):
    """Truthiness, not is-not-None: '' must not count as a deployable ckpt."""
    _write_run(tmp_path, "aris_bellman", 0, selected="")
    doc = agg.aggregate_train(str(tmp_path), "e1rev", 5000)
    assert doc["arms"]["aris_bellman"]["terminal_competence_rate"] == 0.0


def _write_eval(path: Path, *, drop_field=False, null_field=False, null_share=False):
    aggregate = {
        "ego_correct_completion_rate": 0.7,
        "team_correct_delivery_throughput_per_episode": 1.4,
        "ego_correct_delivery_throughput_per_episode": 0.6,
        "ego_serve_share": None if null_share else 0.4,
    }
    if drop_field:
        del aggregate["ego_correct_completion_rate"]
    if null_field:
        aggregate["ego_correct_completion_rate"] = None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"results": [{"aggregate": aggregate}]}))


def test_eval_mode_bootstrap_tables(tmp_path: Path):
    for seed in (0, 1, 2):
        _write_eval(tmp_path / f"e1rev_aris_bellman_s{seed}_eval.json")
    doc = agg.aggregate_eval(str(tmp_path / "*.json"), "e1rev")
    arm = doc["arms"]["aris_bellman"]
    assert arm["n_runs"] == 3
    head = arm["ego_correct_completion_rate"]
    assert head["mean"] == pytest.approx(0.7)
    assert head["ci95_lo"] <= head["mean"] <= head["ci95_hi"]


def test_eval_mode_missing_field_hard_fails(tmp_path: Path):
    _write_eval(tmp_path / "e1rev_aris_bellman_s0_eval.json", drop_field=True)
    with pytest.raises(agg.AggregationError, match="ego_correct_completion_rate"):
        agg.aggregate_eval(str(tmp_path / "*.json"), "e1rev")


def test_eval_mode_null_field_hard_fails(tmp_path: Path):
    """LDS-C4: null is 'unmeasured', and unmeasured must never average as 0."""
    _write_eval(tmp_path / "e1rev_aris_bellman_s0_eval.json", null_field=True)
    with pytest.raises(agg.AggregationError, match="null"):
        agg.aggregate_eval(str(tmp_path / "*.json"), "e1rev")


def test_eval_mode_null_serve_share_is_tolerated_not_zeroed(tmp_path: Path):
    """ego_serve_share=None is a legitimate 'undefined' (no correct delivery by
    anyone): excluded from the CI with n_undefined recorded — never imputed."""
    _write_eval(tmp_path / "e1rev_aris_bellman_s0_eval.json")
    _write_eval(tmp_path / "e1rev_aris_bellman_s1_eval.json", null_share=True)
    doc = agg.aggregate_eval(str(tmp_path / "*.json"), "e1rev")
    share = doc["arms"]["aris_bellman"]["ego_serve_share"]
    assert share["n"] == 1
    assert share["n_undefined"] == 1
    assert share["mean"] == pytest.approx(0.4)
    # renderer must handle the None-bearing arm without crashing
    assert "serve share" in agg._render_markdown(doc)


def test_eval_mode_duplicate_run_hard_fails(tmp_path: Path):
    _write_eval(tmp_path / "a" / "e1rev_aris_bellman_s0_eval.json")
    _write_eval(tmp_path / "b" / "e1rev_aris_bellman_s0_eval.json")
    with pytest.raises(agg.AggregationError, match="Duplicate"):
        agg.aggregate_eval(str(tmp_path / "**" / "*.json"), "e1rev")
