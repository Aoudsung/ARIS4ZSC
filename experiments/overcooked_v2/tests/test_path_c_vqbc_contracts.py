from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from src.path_c.vqbc.checkpoint import assert_reference_ownership
from src.path_c.vqbc.config import (
    VQBC_DEPLOYMENT_MODES,
    VQBCConfig,
    VQBCFormalTemplate,
)
from src.path_c.vqbc.objectives import (
    assert_training_batch_has_no_audit_labels,
)
from src.path_c.vqbc.pipeline import run_vqbc_pool_check
from src.path_c.vqbc.types import CheckpointMetadataV3


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"


def _development_payload() -> dict:
    return yaml.safe_load(
        (
            CONFIG_ROOT / "path_c_vqbc_v4_development_simple.yaml"
        ).read_text(encoding="utf-8")
    )


def test_v4_development_config_uses_fixed_model_and_complete_rollouts() -> None:
    config = VQBCConfig.from_mapping(
        _development_payload(), base_dir=CONFIG_ROOT
    )
    assert config.schema_version == "path_c_model_v4_1"
    assert config.run_kind == "development"
    assert config.model.slot_count == 8
    assert config.model.response_count == 16
    assert config.training.environment_steps == 1_228_800
    assert config.training.minibatches_per_epoch == 8
    assert config.rollout_count == 96
    assert config.evaluation.deployment_modes == VQBC_DEPLOYMENT_MODES
    assert config.partner_sampling.include_frozen_current_policy


@pytest.mark.parametrize(
    "legacy_field",
    [
        "condition_id",
        "controller",
        "probe",
        "prefit",
        "calibration",
        "training_mode",
    ],
)
def test_v4_config_rejects_legacy_control_fields(legacy_field: str) -> None:
    payload = _development_payload()
    payload[legacy_field] = {}
    with pytest.raises(ValueError, match="unknown"):
        VQBCConfig.from_mapping(payload, base_dir=CONFIG_ROOT)


def test_v4_1_config_rejects_retired_shared_slot_decoder_fields() -> None:
    for field in ("slot_embedding_dim", "response_embedding_dim"):
        payload = _development_payload()
        payload["model"][field] = 16
        with pytest.raises(ValueError, match="unknown"):
            VQBCConfig.from_mapping(payload, base_dir=CONFIG_ROOT)



def test_v4_config_rejects_actor_learning_rate_and_prototype_routing() -> None:
    payload = _development_payload()
    payload["training"]["actor_learning_rate"] = 1.0e-4
    with pytest.raises(ValueError, match="unknown"):
        VQBCConfig.from_mapping(payload, base_dir=CONFIG_ROOT)
    payload = _development_payload()
    payload["partner_sampling"]["prototype_routes"] = [0, 1, 2, 3]
    with pytest.raises(ValueError, match="unknown"):
        VQBCConfig.from_mapping(payload, base_dir=CONFIG_ROOT)


def test_source_free_formal_templates_bind_registered_budgets() -> None:
    for name, layout in (
        ("path_c_vqbc_v4_formal_simple.yaml", "test_time_simple"),
        ("path_c_vqbc_v4_formal_wide.yaml", "test_time_wide"),
    ):
        path = CONFIG_ROOT / name
        template = VQBCFormalTemplate.from_mapping(
            yaml.safe_load(path.read_text(encoding="utf-8")),
            base_dir=path.parent,
        )
        assert template.environment.layout == layout
        assert template.environment.num_envs == 250
        assert template.training.environment_steps == 11_000_000
        assert template.training.minibatches_per_epoch == 50


def test_training_batch_contract_rejects_every_audit_identity() -> None:
    assert_training_batch_has_no_audit_labels(
        {"observations": object(), "actions": object()}
    )
    for field in (
        "family_id",
        "seed",
        "checkpoint_index",
        "training_run_id",
        "partner_member_index",
    ):
        with pytest.raises(ValueError, match="audit labels"):
            assert_training_batch_has_no_audit_labels(
                {"observations": object(), field: object()}
            )


def _metadata(config: VQBCConfig) -> CheckpointMetadataV3:
    reference = config.backbone_init
    return CheckpointMetadataV3(
        schema_version="path_c_model_checkpoint_metadata_v3",
        run_kind=config.run_kind,
        scientific_readout_allowed=False,
        outer_unit_id=None,
        reference_checkpoint_path=str(reference.checkpoint_path),
        reference_training_run_id=reference.training_run_id,
        reference_weights_sha256=reference.flax_weights_sha256,
        reference_launch_config_path=str(reference.launch_config_path),
        reference_launch_config_sha256=reference.launch_config_sha256,
        config_sha256="0" * 64,
        model_weights_sha256="1" * 64,
        state_sha256="2" * 64,
        effective_environment_steps=0,
        completed_episodes=0,
        update_count=0,
    )


def test_checkpoint_reference_ownership_rejects_global_or_foreign_anchor() -> None:
    config = VQBCConfig.from_mapping(
        _development_payload(), base_dir=CONFIG_ROOT
    )
    metadata = _metadata(config)
    assert_reference_ownership(
        metadata, config.backbone_init, expected_outer_unit_id=None
    )
    foreign = replace(
        config.backbone_init, training_run_id="foreign-reference"
    )
    with pytest.raises(ValueError, match="own official reference"):
        assert_reference_ownership(
            metadata, foreign, expected_outer_unit_id=None
        )


def test_pool_check_binds_live_reference_output_to_the_unit(
    tmp_path: Path,
) -> None:
    config = replace(
        VQBCConfig.from_mapping(
            _development_payload(), base_dir=CONFIG_ROOT
        ),
        output_root=tmp_path,
    )

    def validate(reference: object, unused_launch: Path) -> dict:
        del unused_launch
        return {
            "training_run_id": reference.training_run_id,
            "flax_weights_sha256": reference.flax_weights_sha256,
        }

    report = run_vqbc_pool_check(
        config,
        validate_checkpoint=validate,
        reference_call_check=lambda: {
            "reference_training_run_id": (
                config.backbone_init.training_run_id
            ),
            "reference_weights_sha256": (
                config.backbone_init.flax_weights_sha256
            ),
            "carry_exact": True,
            "reference_output_sha256": "a" * 64,
        },
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["reference_call"]["carry_exact"] is True
    assert payload["reference_training_run_id"] == (
        config.backbone_init.training_run_id
    )


def test_checkpoint_metadata_round_trip_keeps_reference_contract() -> None:
    config = VQBCConfig.from_mapping(
        _development_payload(), base_dir=CONFIG_ROOT
    )
    metadata = _metadata(config)
    assert CheckpointMetadataV3.from_mapping(
        metadata.to_mapping()
    ) == metadata


def test_archived_source_mirror_requires_the_registered_exact_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("jaxmarl")
    from experiments.overcooked_v2.path_c_official_artifact import (
        _source_dependency_closure,
    )

    live_path = (
        tmp_path
        / "live"
        / "experiments"
        / "overcooked_v2"
        / "official"
        / "adapter.py"
    )
    mirror_path = (
        tmp_path
        / "mirror"
        / "experiments"
        / "overcooked_v2"
        / "official"
        / "adapter.py"
    )
    live_path.parent.mkdir(parents=True)
    mirror_path.parent.mkdir(parents=True)
    live_path.write_text("current source\n", encoding="utf-8")
    mirror_path.write_text("registered source\n", encoding="utf-8")
    registered_hash = hashlib.sha256(mirror_path.read_bytes()).hexdigest()
    dependency = [{"path": str(live_path), "sha256": registered_hash}]

    with pytest.raises(ValueError, match="closure hash mismatch"):
        _source_dependency_closure(dependency)
    monkeypatch.setenv(
        "PATH_C_OFFICIAL_SOURCE_MIRROR_ROOT",
        str(tmp_path / "mirror"),
    )
    closure = _source_dependency_closure(dependency)
    assert closure["files"][str(live_path.resolve())] == registered_hash
