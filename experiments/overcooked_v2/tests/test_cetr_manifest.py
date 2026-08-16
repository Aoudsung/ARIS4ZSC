from __future__ import annotations

import json
from pathlib import Path

import pytest


def _row(
    checkpoint: str,
    *,
    run_id: str,
    role: str,
    parent: str,
    mechanism: str,
    stage: float = 1.0,
    owner_seed_index: int | None = None,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "role": role,
        "checkpoint": checkpoint,
        "parent_training_run_id": parent,
        "generation_mechanism": mechanism,
        "checkpoint_stage": stage,
        "hyperparameter_family": "test",
        "seed": 1,
        "seed_index": 0,
        "jax_prng_key": [0, 1],
        "owner_seed_index": owner_seed_index,
        "co_training_group_id": None,
        "partner_type_id": None,
    }


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps({"version": 2, "layout": "test_time_simple", "runs": rows}),
        encoding="utf-8",
    )


def test_four_mechanism_manifest_loads_and_normalizes(tmp_path: Path) -> None:
    from src.cetr_zsc.manifest import load_partner_manifest, normalized_mechanism

    rows = []
    for index, mechanism in enumerate(("rnn-sp", "rnn-op", "sa", "fcp")):
        checkpoint = tmp_path / f"checkpoint-{index}.pkl"
        checkpoint.write_bytes(b"fixture")
        rows.append(
            _row(
                checkpoint.name,
                run_id=f"run-{index}",
                role="development_support",
                parent=f"parent-{index}",
                mechanism=mechanism,
            )
        )
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, rows)

    manifest = load_partner_manifest(
        manifest_path,
        expected_layout="test_time_simple",
        verify_files=True,
    )
    assert len(manifest.runs) == 4
    assert manifest.runs[0].checkpoint == (tmp_path / "checkpoint-0.pkl").resolve()
    assert normalized_mechanism("state-augmented") == "sa"
    assert set(normalized_mechanism(row.generation_mechanism) for row in manifest.runs) == {
        "fcp",
        "op",
        "sa",
        "sp",
    }


def test_manifest_rejects_parent_overlap_across_roles() -> None:
    from src.cetr_zsc.manifest import PartnerManifest, PartnerRun, validate_partner_manifest

    common = dict(
        checkpoint=Path("fixture"),
        parent_training_run_id="shared-parent",
        generation_mechanism="sp",
        checkpoint_stage=1.0,
        hyperparameter_family="test",
        seed=1,
    )
    manifest = PartnerManifest(
        layout="test_time_simple",
        runs=(
            PartnerRun(
                run_id="support",
                role="development_support",
                **common,
            ),
            PartnerRun(
                run_id="calibration",
                role="calibration",
                **common,
            ),
        ),
    )
    with pytest.raises(ValueError, match="lineage overlaps"):
        validate_partner_manifest(manifest)


def test_manifest_schema_rejects_missing_top_level_field(tmp_path: Path) -> None:
    from src.cetr_zsc.manifest import load_partner_manifest

    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"version": 2, "runs": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="top-level schema"):
        load_partner_manifest(path, verify_files=False)


def test_formal_manifest_missing_mechanism_is_rejected(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from src.cetr_zsc.manifest import load_partner_manifest
    from src.cetr_zsc.partners import build_training_partner_pool

    checkpoint = tmp_path / "checkpoint.pkl"
    checkpoint.write_bytes(b"fixture")
    path = tmp_path / "manifest.json"
    _write_manifest(
        path,
        [
            _row(
                checkpoint.name,
                run_id="sp-run",
                role="development_support",
                parent="sp-parent",
                mechanism="sp",
            )
        ],
    )
    manifest = load_partner_manifest(path, verify_files=True)
    config = SimpleNamespace(
        run_kind="formal",
        partner_pool=SimpleNamespace(checkpoint_stages=(0.0, 0.5, 1.0)),
        evaluation=SimpleNamespace(minimum_partner_runs_per_mechanism=1),
    )
    with pytest.raises(ValueError, match="all four mechanisms"):
        build_training_partner_pool(config, manifest)
