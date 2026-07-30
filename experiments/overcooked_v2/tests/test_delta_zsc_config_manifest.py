from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from src.path_c.experiment import (
    CONFIG_VERSION,
    MANIFEST_VERSION,
    METHOD_VERSION,
    PartnerManifest,
    PartnerRun,
    RUN_BUDGETS,
    load_config,
    load_partner_manifest,
    validate_partner_manifest,
)


ROOT = Path(__file__).resolve().parents[3]
CONFIGS = ROOT / "experiments" / "overcooked_v2" / "configs"


def _run(index: int, role: str, *, parent: str | None = None) -> PartnerRun:
    return PartnerRun(
        run_id=f"run-{index}",
        role=role,
        checkpoint=Path(f"/tmp/checkpoint-{index}"),
        checkpoint_sha256=f"{index:064x}",
        parent_training_run_id=parent or f"parent-{index}",
        generation_mechanism=f"mechanism-{index % 2}",
        seed=index,
        co_training_group_id=f"group-{index}",
        partner_type_id=None,
    )


def _valid_manifest() -> PartnerManifest:
    roles = (
        "frozen_external_train",
        "frozen_external_train",
        "calibration",
        "calibration",
        "confirmatory",
        "confirmatory",
    )
    return PartnerManifest(
        layout="test_time_simple",
        runs=tuple(_run(index + 1, role) for index, role in enumerate(roles)),
    )


def test_registered_versions_and_run_budgets() -> None:
    assert CONFIG_VERSION == 5
    assert MANIFEST_VERSION == 2
    assert METHOD_VERSION == "delta_zsc_v5_decision_equivalent_bayes_r1"
    assert RUN_BUDGETS["mechanical"].num_envs == 4
    assert RUN_BUDGETS["development"].environment_steps == 1_228_800
    assert RUN_BUDGETS["formal"].environment_steps == 11_000_000


@pytest.mark.parametrize(
    ("filename", "run_kind", "expected_envs"),
    (
        ("delta_zsc_simple_development.yaml", "development", 32),
        ("delta_zsc_wide_development.yaml", "mechanical", 4),
        ("delta_zsc_simple_formal.yaml", "formal", 250),
        ("delta_zsc_wide_formal.yaml", "formal", 250),
    ),
)
def test_v5_configs_load_with_registered_budget(
    filename: str, run_kind: str, expected_envs: int
) -> None:
    config = load_config(CONFIGS / filename, run_kind=run_kind)
    assert config.environment.num_envs == expected_envs
    assert config.training.rollout_length == config.environment.episode_steps
    assert len(config.fingerprint) == 64


def test_config_rejects_unknown_fields(tmp_path: Path) -> None:
    import yaml

    source = CONFIGS / "delta_zsc_simple_development.yaml"
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    candidate = tmp_path / "invalid.yaml"
    candidate.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        load_config(candidate, run_kind="mechanical")


def test_manifest_accepts_run_disjoint_roles() -> None:
    validate_partner_manifest(_valid_manifest())


def test_manifest_rejects_checkpoint_parent_and_group_leakage() -> None:
    manifest = _valid_manifest()
    duplicate_hash = replace(
        manifest.runs[-1], checkpoint_sha256=manifest.runs[0].checkpoint_sha256
    )
    with pytest.raises(ValueError, match="checkpoint overlap"):
        validate_partner_manifest(
            replace(manifest, runs=(*manifest.runs[:-1], duplicate_hash))
        )

    shared_parent = replace(
        manifest.runs[2], parent_training_run_id=manifest.runs[0].parent_training_run_id
    )
    with pytest.raises(ValueError, match="parent runs overlap"):
        validate_partner_manifest(
            replace(manifest, runs=(manifest.runs[0], manifest.runs[1], shared_parent, *manifest.runs[3:]))
        )

    shared_group = replace(
        manifest.runs[2], co_training_group_id=manifest.runs[0].co_training_group_id
    )
    with pytest.raises(ValueError, match="co-training group"):
        validate_partner_manifest(
            replace(manifest, runs=(manifest.runs[0], manifest.runs[1], shared_group, *manifest.runs[3:]))
        )


def test_manifest_file_hash_is_verified(tmp_path: Path) -> None:
    manifest = _valid_manifest()
    runs = []
    for index, run in enumerate(manifest.runs):
        checkpoint = tmp_path / f"checkpoint-{index}.bin"
        checkpoint.write_bytes(f"checkpoint-{index}".encode("utf-8"))
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        runs.append(replace(run, checkpoint=checkpoint, checkpoint_sha256=digest))
    materialized = replace(manifest, runs=tuple(runs))
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(materialized.to_mapping(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    loaded = load_partner_manifest(path, expected_layout="test_time_simple")
    assert loaded.to_mapping() == materialized.to_mapping()

    runs[0].checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        load_partner_manifest(path, expected_layout="test_time_simple")
