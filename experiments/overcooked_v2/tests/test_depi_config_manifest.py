"""Fail-closed DEPI configuration, seed, and lineage tests."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from src.path_c.experiment import (
    CHECKPOINT_SCHEMA_VERSION,
    CONFIG_VERSION,
    MANIFEST_VERSION,
    METHOD_VERSION,
    OFFICIAL_PROTOCOL_VERSION,
    PartnerManifest,
    PartnerRun,
    RUN_BUDGETS,
    load_config,
    load_partner_manifest,
    official_training_key,
    validate_config,
    validate_partner_manifest,
)
from src.path_c.response_targets import official_partner_observation_planes
from src.path_c.storage import sha256_path


ROOT = Path(__file__).resolve().parents[3]
CONFIGS = ROOT / "experiments" / "overcooked_v2" / "configs"


def test_registered_identity_and_budgets_are_current() -> None:
    assert CONFIG_VERSION == 15
    assert CHECKPOINT_SCHEMA_VERSION == 5
    assert MANIFEST_VERSION == 3
    assert METHOD_VERSION == "depi_exact_filter_decision_supervision_v5"
    assert OFFICIAL_PROTOCOL_VERSION == "overcooked_v2_iclr2025_5ce1707_v1"
    assert RUN_BUDGETS["mechanical"].environment_steps == 1_024
    assert RUN_BUDGETS["development"].environment_steps == 1_228_800
    assert RUN_BUDGETS["formal"].environment_steps == 29_949_952


@pytest.mark.parametrize(
    ("filename", "run_kind", "environment_count"),
    (
        ("depi_simple_development.yaml", "development", 32),
        ("depi_wide_development.yaml", "development", 32),
        ("depi_simple_mechanical_e2e.yaml", "mechanical", 4),
        ("depi_simple_formal.yaml", "formal", 256),
        ("depi_wide_formal.yaml", "formal", 256),
    ),
)
def test_every_active_config_loads_under_its_registered_budget(
    filename: str, run_kind: str, environment_count: int
) -> None:
    config = load_config(CONFIGS / filename, run_kind=run_kind)
    assert config.environment.num_envs == environment_count
    assert config.method_variant == "b2"
    assert config.model.protocol_components == 4
    assert config.partner_pool.enabled
    assert config.partner_pool.heuristic_family_test_only
    assert len(config.fingerprint) == 64


def test_formal_decision_calibration_and_statistics_registration() -> None:
    config = load_config(CONFIGS / "depi_simple_formal.yaml", run_kind="formal")
    assert (config.anchors.fit_replicas, config.anchors.evaluation_replicas) == (4, 8)
    assert config.anchors.continuation_horizon == 128
    assert config.loss_v2.decision_policy_weight == 0.25
    assert config.loss_v2.combined_policy_kl_threshold == 0.04
    assert config.posterior_calibration.primary_unit == "partner_run"
    assert config.posterior_calibration.secondary_unit == "episode"
    assert config.evaluation.inference_mode in {"independent_run", "paired_run"}
    assert config.evaluation.superiority_lcb_threshold == 0.0
    assert config.evaluation.minimum_effect == 20.0
    assert config.evaluation.minimum_effect_rule in {
        "point_estimate",
        "lower_confidence_bound",
    }


def test_b3_and_nonregistered_formal_component_count_fail_closed() -> None:
    config = load_config(CONFIGS / "depi_simple_formal.yaml", run_kind="formal")
    with pytest.raises(NotImplementedError, match="B3"):
        validate_config(replace(config, method_variant="b3"))
    with pytest.raises(ValueError, match="K=4"):
        validate_config(
            replace(config, model=replace(config.model, protocol_components=2))
        )


def test_official_plane_contract_and_training_keys_are_exact() -> None:
    planes = official_partner_observation_planes(39)
    assert planes.visibility_channel == 10
    assert planes.direction_channels == (11, 12, 13, 14)
    assert planes.inventory_channels == (15, 16, 17, 18, 19)
    assert len({official_training_key(index) for index in range(10)}) == 10
    with pytest.raises(ValueError, match="0..9"):
        official_training_key(-1)


def _run(index: int, role: str, *, parent: str | None = None) -> PartnerRun:
    return PartnerRun(
        run_id=f"run-{index}",
        role=role,
        checkpoint=Path(f"/tmp/depi-checkpoint-{index}"),
        checkpoint_sha256=f"{index:064x}",
        parent_training_run_id=parent or f"parent-{index}",
        generation_mechanism="fixture",
        checkpoint_stage=1.0,
        hyperparameter_family="default",
        seed=index,
        seed_index=None,
        jax_prng_key=(900_000 + index, 800_000 + index),
        owner_seed_index=None,
        co_training_group_id=f"group-{index}",
        partner_type_id=None,
    )


def _valid_manifest() -> PartnerManifest:
    roles = (
        "owner_source",
        "development_support",
        "development_support",
        "comparator_fit",
        "comparator_fit",
        "comparator_validation",
        "comparator_validation",
        "calibration",
        "calibration",
        "confirmatory",
        "confirmatory",
    )
    return PartnerManifest(
        layout="test_time_simple",
        runs=tuple(_run(index + 1, role) for index, role in enumerate(roles)),
    )


def test_manifest_accepts_disjoint_roles_and_rejects_parent_leakage() -> None:
    manifest = _valid_manifest()
    validate_partner_manifest(manifest)
    leaked = replace(
        manifest,
        runs=(
            *manifest.runs[:-1],
            replace(
                manifest.runs[-1],
                parent_training_run_id=manifest.runs[1].parent_training_run_id,
            ),
        )
    )
    with pytest.raises(ValueError, match="overlap"):
        validate_partner_manifest(leaked)


def test_manifest_loader_verifies_the_checkpoint_fingerprint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_bytes(b"depi")
    run = replace(
        _run(1, "owner_source"),
        checkpoint=checkpoint,
        checkpoint_sha256=sha256_path(checkpoint),
    )
    payload = PartnerManifest("test_time_simple", (run,)).to_mapping()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_partner_manifest(
        path, expected_layout="test_time_simple", verify_files=True
    )
    assert loaded.runs[0].checkpoint_sha256 == sha256_path(checkpoint)
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        load_partner_manifest(path, expected_layout="test_time_simple", verify_files=True)
