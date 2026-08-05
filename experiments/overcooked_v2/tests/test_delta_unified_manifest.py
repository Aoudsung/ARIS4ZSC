from __future__ import annotations

import json
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
