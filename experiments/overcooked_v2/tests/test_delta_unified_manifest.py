from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace


def _row(checkpoint: str, *, run_id: str, role: str, parent: str, mechanism: str):
    return {
        "run_id": run_id,
        "role": role,
        "checkpoint": checkpoint,
        "parent_training_run_id": parent,
        "generation_mechanism": mechanism,
        "checkpoint_stage": 1.0,
        "hyperparameter_family": "test",
        "seed": 1,
        "seed_index": 0,
        "jax_prng_key": [0, 1],
        "owner_seed_index": None,
        "co_training_group_id": None,
        "partner_type_id": None,
    }


def test_manifest_builder_resolves_and_reloads_lineage(tmp_path: Path) -> None:
    from experiments.overcooked_v2.delta_manifest_app import build_partner_manifest
    from src.delta_zsc.manifest import load_partner_manifest

    support = tmp_path / "support.ckpt"
    calibration = tmp_path / "calibration.ckpt"
    confirmatory = tmp_path / "confirmatory.ckpt"
    support.write_bytes(b"support")
    calibration.write_bytes(b"calibration")
    confirmatory.write_bytes(b"confirmatory")
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "version": 2,
                "layout": "test_time_simple",
                "runs": [
                    _row(
                        support.name,
                        run_id="support",
                        role="development_support",
                        parent="support-parent",
                        mechanism="rnn-sp",
                    ),
                    _row(
                        calibration.name,
                        run_id="calibration",
                        role="calibration",
                        parent="calibration-parent",
                        mechanism="state-augmented",
                    ),
                    _row(
                        confirmatory.name,
                        run_id="confirmatory",
                        role="confirmatory",
                        parent="confirmatory-parent",
                        mechanism="fcp",
                    ),
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "manifest.json"
    build_partner_manifest(
        SimpleNamespace(
            plan=str(plan),
            output=str(output),
            expected_layout="test_time_simple",
        )
    )
    manifest = load_partner_manifest(
        output, expected_layout="test_time_simple", verify_files=True
    )
    assert len(manifest.runs) == 3
    assert manifest.runs[0].checkpoint == support.resolve()
    stored = json.loads(output.read_text(encoding="utf-8"))
    assert stored["runs"][0]["checkpoint"] == support.name

    moved = tmp_path.parent / f"{tmp_path.name}-moved"
    shutil.move(str(tmp_path), moved)
    moved_manifest = load_partner_manifest(
        moved / "manifest.json", expected_layout="test_time_simple", verify_files=True
    )
    assert moved_manifest.runs[0].checkpoint == (moved / support.name).resolve()


def test_official_parent_throughput_is_counted_once_per_lineage(tmp_path: Path) -> None:
    import pytest

    from experiments.overcooked_v2.training_app import _upstream_partner_cost

    parent = tmp_path / "sp_parent_seed201_30m"
    checkpoint_root = parent / "official_hydra" / "runs" / "job" / "run_0"
    checkpoints = [
        checkpoint_root / name for name in ("ckpt_0", "ckpt_1", "ckpt_final")
    ]
    for checkpoint in checkpoints:
        checkpoint.mkdir(parents=True)
    (parent / "throughput.txt").write_text(
        "status=0\nenvironment_steps=30000000\nwall_seconds=1800\n",
        encoding="utf-8",
    )
    config = parent / "official_hydra" / ".hydra" / "config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        "model:\n  TOTAL_TIMESTEPS: 30000000.0\nSEED: 201\nNUM_SEEDS: 1\n",
        encoding="utf-8",
    )
    members = tuple(
        SimpleNamespace(parent_training_run_id="sp-parent-201", checkpoint=value)
        for value in checkpoints
    )
    steps, gpu_hours, wall_hours, records = _upstream_partner_cost(
        members, formal=True
    )
    assert steps == 30_000_000
    assert gpu_hours == pytest.approx(0.5)
    assert wall_hours == pytest.approx(0.5)
    assert len(records) == 1
    assert records[0]["status"] == "counted"
    assert records[0]["source_kind"] == "official_throughput"
    assert records[0]["configured_seed"] == 201
    assert records[0]["source"] == str((parent / "throughput.txt").resolve())


def test_official_parent_throughput_must_match_frozen_config(tmp_path: Path) -> None:
    from experiments.overcooked_v2.training_app import _upstream_partner_cost

    parent = tmp_path / "op_parent"
    checkpoint = parent / "official_hydra" / "runs" / "job" / "run_0" / "ckpt_final"
    checkpoint.mkdir(parents=True)
    (parent / "throughput.txt").write_text(
        "status=0\nenvironment_steps=50000000\nwall_seconds=3600\n",
        encoding="utf-8",
    )
    config = parent / "official_hydra" / ".hydra" / "config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        "model:\n  TOTAL_TIMESTEPS: 49000000.0\nSEED: 202\nNUM_SEEDS: 1\n",
        encoding="utf-8",
    )
    member = SimpleNamespace(
        parent_training_run_id="op-parent-202", checkpoint=checkpoint
    )
    try:
        _upstream_partner_cost((member,), formal=False)
    except ValueError as error:
        assert "step mismatch" in str(error)
    else:
        raise AssertionError("Mismatched Official cost evidence was accepted.")


def test_resumed_resource_ledger_reconciles_parent_cost() -> None:
    from experiments.overcooked_v2.training_app import (
        _reconcile_upstream_resource_cost,
    )
    from src.delta_zsc.resources import ResourceLedger

    legacy = ResourceLedger(
        ego_policy_steps=98_304,
        training_gpu_hours=0.25,
        upstream_partner_steps=0,
    )
    corrected = ResourceLedger.from_mapping(
        _reconcile_upstream_resource_cost(
            legacy.to_mapping(),
            steps=80_000_000,
            gpu_hours=3.3891666666666667,
            wall_clock_hours=3.3891666666666667,
        )
    )
    assert corrected.ego_policy_steps == 98_304
    assert corrected.training_gpu_hours == 0.25
    assert corrected.upstream_partner_steps == 80_000_000
    assert corrected.shared_gpu_hours == 3.3891666666666667
    assert corrected.shared_wall_clock_hours == 3.3891666666666667
