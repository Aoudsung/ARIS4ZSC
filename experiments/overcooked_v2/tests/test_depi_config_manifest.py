"""Fail-closed DEPI configuration, seed, and lineage tests."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from experiments.overcooked_v2.deployment import DEPLOYMENT_BUNDLE_VERSION
from experiments.overcooked_v2.development_matrix_app import (
    _b2_auxiliary_transition_budget,
    _variant_config,
)

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
    validate_seed_training_manifest,
)
from src.path_c.response_targets import official_partner_observation_planes
from src.path_c.storage import sha256_path


ROOT = Path(__file__).resolve().parents[3]
CONFIGS = ROOT / "experiments" / "overcooked_v2" / "configs"


def test_registered_identity_and_budgets_are_current() -> None:
    assert CONFIG_VERSION == 18
    assert CHECKPOINT_SCHEMA_VERSION == 8
    assert MANIFEST_VERSION == 4
    assert METHOD_VERSION == "depi_decision_consistent_evidence_gated_filter_v8"
    assert OFFICIAL_PROTOCOL_VERSION == "overcooked_v2_iclr2025_5ce1707_v1"
    assert RUN_BUDGETS["mechanical"].environment_steps == 1_024
    assert RUN_BUDGETS["development"].environment_steps == 1_228_800
    assert RUN_BUDGETS["formal"].environment_steps == 29_949_952


def test_readme_identity_cannot_drift_from_code_authority() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"METHOD_VERSION = {METHOD_VERSION}" in readme
    assert f"CONFIG_VERSION = {CONFIG_VERSION}" in readme
    assert f"CHECKPOINT_SCHEMA_VERSION = {CHECKPOINT_SCHEMA_VERSION}" in readme
    assert f"MANIFEST_VERSION = {MANIFEST_VERSION}" in readme
    assert f"deployment bundle schema：`{DEPLOYMENT_BUNDLE_VERSION}`" in readme


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
    assert config.model.instant_partner_dim == 32
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


def test_total_budget_controls_replace_b2_auxiliary_cost_with_exact_ppo_steps(
    tmp_path: Path,
) -> None:
    source = CONFIGS / "depi_simple_development.yaml"
    r0_path = tmp_path / "r0_extra.yaml"
    b0_path = tmp_path / "b0_extra.yaml"
    b1_path = tmp_path / "b1_extra.yaml"
    _variant_config(source, r0_path, "r0_extra", 4)
    _variant_config(source, b0_path, "b0_extra", 4)
    _variant_config(source, b1_path, "b1_extra", 4)
    r0 = load_config(r0_path, run_kind="development")
    b0 = load_config(b0_path, run_kind="development")
    b1 = load_config(b1_path, run_kind="development")
    assert r0.method_variant == "r0"
    assert b0.method_variant == "b0"
    assert b1.method_variant == "b1"
    import yaml

    expected_extra = _b2_auxiliary_transition_budget(
        yaml.safe_load(source.read_text(encoding="utf-8"))
    )
    assert expected_extra == 2_804_096
    assert r0.training.extra_ppo_environment_steps == expected_extra
    assert b0.training.extra_ppo_environment_steps == expected_extra
    assert b1.training.extra_ppo_environment_steps == expected_extra
    assert (
        r0.training.environment_steps
        == b0.training.environment_steps
        == b1.training.environment_steps
        == 4_032_896
    )
    assert r0.training.environment_steps % (
        r0.environment.num_envs * r0.training.rollout_length
    ) == 2_432


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


def test_validate_seed_training_manifest_formal_full_fixture() -> None:
    """Exercise the complete formal-only validator, including config use."""

    rows: list[PartnerRun] = []
    serial = 1

    def add(
        *,
        role: str,
        mechanism: str,
        parent: str,
        stage: float = 1.0,
        family: str = "default",
        seed_index: int | None = None,
        owner: int | None = None,
    ) -> None:
        nonlocal serial
        key = (
            official_training_key(seed_index)
            if seed_index is not None
            else (700_000 + serial, 800_000 + serial)
        )
        rows.append(
            PartnerRun(
                run_id=f"formal-{serial}",
                role=role,
                checkpoint=Path(f"/tmp/formal-checkpoint-{serial}"),
                checkpoint_sha256=f"{serial:064x}",
                parent_training_run_id=parent,
                generation_mechanism=mechanism,
                checkpoint_stage=stage,
                hyperparameter_family=family,
                seed=serial,
                seed_index=seed_index,
                jax_prng_key=key,
                owner_seed_index=owner,
                co_training_group_id=f"formal-group-{role}-{parent}",
                partner_type_id=None,
            )
        )
        serial += 1

    add(
        role="owner_source",
        mechanism="rnn-sp",
        parent="owner-parent-0",
        seed_index=0,
        owner=0,
    )
    for mechanism in ("rnn-sp", "rnn-op"):
        for parent_index in range(10):
            parent = f"support-{mechanism}-{parent_index}"
            for stage in (0.0, 0.5, 1.0):
                add(
                    role="development_support",
                    mechanism=mechanism,
                    parent=parent,
                    stage=stage,
                    seed_index=parent_index,
                )
    for width_index in range(2):
        add(
            role="development_support",
            mechanism="rnn-op",
            parent=f"support-op-width-{width_index}",
            family=f"width-{width_index}",
            seed_index=8 + width_index,
        )
    for role in ("comparator_fit", "comparator_validation"):
        for index in range(8):
            add(
                role=role,
                mechanism="sp" if index < 4 else "op",
                parent=f"{role}-parent-{index}",
            )
    for mechanism in ("sp", "op", "sa", "fcp"):
        for index in range(5):
            add(
                role="calibration",
                mechanism=mechanism,
                parent=f"calibration-{mechanism}-{index}",
            )
    for index in range(2):
        add(
            role="confirmatory",
            mechanism="sp" if index == 0 else "op",
            parent=f"confirmatory-{index}",
        )
    config = load_config(CONFIGS / "depi_simple_formal.yaml", run_kind="formal")
    validate_seed_training_manifest(
        PartnerManifest(layout="test_time_simple", runs=tuple(rows)),
        config=config,
        owner_seed_index=0,
        formal=True,
    )


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
