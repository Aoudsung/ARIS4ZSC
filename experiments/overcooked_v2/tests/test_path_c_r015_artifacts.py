from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from experiments.overcooked_v2 import path_c_r015_artifacts as artifacts
from experiments.overcooked_v2 import path_c_r015_runtime as formal_runtime


def test_atomic_bundle_reentry_accepts_only_identical_files(tmp_path: Path) -> None:
    target = tmp_path / "bundle"
    expected = {"a.json": b"{}\n", "nested/b.json": b'{"x":1}\n'}
    artifacts._atomic_write_bundle(target, expected)
    artifacts._atomic_write_bundle(target, expected)
    assert (target / "a.json").read_bytes() == expected["a.json"]
    with pytest.raises(FileExistsError, match="different content"):
        artifacts._atomic_write_bundle(target, {"a.json": b'{"changed":true}\n'})


def test_probe_registry_digest_is_order_independent_but_payload_keeps_order() -> None:
    preregistration = {
        "experiment": {
            "registered_probe_scripts": [
                {"probe_id": "stay", "primitive_actions": ["stay"]},
                {"probe_id": "up", "primitive_actions": ["up"]},
            ]
        }
    }
    registry = artifacts._probe_registry(preregistration)
    assert registry["schema_version"] == "path_c_r015_probe_registry_v1"
    assert registry["maximum_script_length"] == 2
    assert registry["observed_maximum_script_length"] == 1
    assert [value["probe_id"] for value in registry["probe_scripts"]] == [
        "stay",
        "up",
    ]
    expected_semantics = {
        "schema_version": "path_c_r015_probe_registry_semantics_v1",
        "probe_scripts": [
            {"probe_id": "stay", "primitive_actions": ["stay"]},
            {"probe_id": "up", "primitive_actions": ["up"]},
        ],
    }
    assert registry["semantic_sha256"] == artifacts._canonical_sha256(
        expected_semantics
    )


def test_bundle_validation_rejects_pending_and_changed_artifact(tmp_path: Path) -> None:
    artifact_payloads = {}
    inventory = {}
    for name, filename in artifacts._OUTPUT_FILENAMES.items():
        path = tmp_path / filename
        path.write_text(
            json.dumps({"schema_version": f"fixture_{name}"}) + "\n",
            encoding="utf-8",
        )
        artifact_payloads[name] = path
        inventory[name] = {
            "path": str(path),
            "sha256": artifacts._file_sha256(path),
            "schema_version": f"fixture_{name}",
        }
    updated = tmp_path / "freeze-inputs.json"
    updated.write_text('{"schema_version":"path_c_r015_freeze_manifest_inputs_v2"}\n')
    manifest = {
        "schema_version": artifacts.MACHINE_ARTIFACT_BUNDLE_SCHEMA,
        "scientific_readout_allowed": False,
        "formal_data_read": False,
        "formal_round_count": 2500,
        "formal_prototype_count": 4,
        "formal_sampling_entry_count": 10000,
        "artifacts": inventory,
        "implementation_sources": {
            "fixture": {
                "path": str(updated),
                "sha256": artifacts._file_sha256(updated),
            }
        },
        "updated_freeze_inputs": {
            "path": str(updated),
            "sha256": artifacts._file_sha256(updated),
        },
    }
    manifest_path = tmp_path / "bundle.json"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    assert artifacts.validate_r015_machine_artifact_bundle(manifest_path)["valid"]
    artifact_payloads["official_history_filter"].write_text(
        '{"schema_version":"fixture","status":"pending"}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="hash changed"):
        artifacts.validate_r015_machine_artifact_bundle(manifest_path)


def test_bundle_validation_rejects_pending_generated_freeze_inputs(
    tmp_path: Path,
) -> None:
    inventory = {}
    for name, filename in artifacts._OUTPUT_FILENAMES.items():
        path = tmp_path / filename
        path.write_text(
            json.dumps({"schema_version": f"fixture_{name}"}) + "\n",
            encoding="utf-8",
        )
        inventory[name] = {
            "path": str(path),
            "sha256": artifacts._file_sha256(path),
            "schema_version": f"fixture_{name}",
        }
    updated = tmp_path / "freeze-inputs.json"
    updated.write_text(
        json.dumps(
            {
                "schema_version": artifacts.UPDATED_FREEZE_INPUT_SCHEMA,
                "status": "pending",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": artifacts.MACHINE_ARTIFACT_BUNDLE_SCHEMA,
        "scientific_readout_allowed": False,
        "formal_data_read": False,
        "formal_round_count": 2500,
        "formal_prototype_count": 4,
        "formal_sampling_entry_count": 10000,
        "artifacts": inventory,
        "implementation_sources": {
            "fixture": {
                "path": str(updated),
                "sha256": artifacts._file_sha256(updated),
            }
        },
        "updated_freeze_inputs": {
            "path": str(updated),
            "sha256": artifacts._file_sha256(updated),
        },
    }
    manifest_path = tmp_path / "bundle.json"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="freeze inputs are incomplete"):
        artifacts.validate_r015_machine_artifact_bundle(manifest_path)


def test_machine_artifact_builder_calls_registered_formal_schedule() -> None:
    source = Path(artifacts.__file__).read_text(encoding="utf-8")
    assert "build_r015_formal_sampling_schedule(" in source
    assert "n_rounds=FORMAL_ROUND_COUNT" in source
    assert "FORMAL_ROUND_COUNT = 2500" in source
    assert "FORMAL_PROTOTYPE_COUNT = 4" in source
    assert '"formal_data_read": False' in source


def test_formal_sampling_schedule_is_exactly_four_by_2500() -> None:
    prototypes = ("p0", "p1", "p2", "p3")
    schedule = dict(
        formal_runtime.build_r015_formal_sampling_schedule(
            prototype_ids=prototypes,
            root_seed=8115,
        )
    )
    entries = schedule["entries"]
    artifacts._validate_formal_sampling_schedule(
        schedule,
        prototype_ids=prototypes,
    )
    entries[-1] = dict(entries[0])
    with pytest.raises(ValueError, match="duplicate coordinate"):
        artifacts._validate_formal_sampling_schedule(
            schedule,
            prototype_ids=prototypes,
        )


def test_replay_backend_contract_rejects_generic_summary_backend(
    tmp_path: Path,
) -> None:
    entrypoints = tmp_path / "entrypoints.py"
    entrypoints.write_text(
        "\n".join(
            [
                "def verify_r015_probe_decision(value, context): pass",
                "def verify_r015_safety_comparison(value, context): pass",
                "def verify_r015_trace_manifest(value, context): pass",
            ]
        ),
        encoding="utf-8",
    )
    backend = tmp_path / "backend.py"
    backend.write_text(
        "\n".join(
            [
                "def replay_probe_decision(self, value, context): pass",
                "def replay_safety_comparison(self, value, context): pass",
                "def replay_trace(self, value, context): pass",
                "path_c_r015_trace_replay_verification_v1 = None",
            ]
        ),
        encoding="utf-8",
    )
    sources = {
        "replay_verifier_entrypoints": entrypoints,
        "full_horizon_implementation": backend,
    }
    with pytest.raises(ValueError, match="exact formal verification schemas"):
        artifacts._validate_replay_backend_contract(sources)

    backend.write_text(
        backend.read_text(encoding="utf-8")
        + "\n"
        + "\n".join(artifacts.ARTIFACT_SCHEMAS.values()),
        encoding="utf-8",
    )
    artifacts._validate_replay_backend_contract(sources)


def test_resolved_pilot_protocol_rejects_pending_or_other_selection(
    tmp_path: Path,
) -> None:
    selection = tmp_path / "selection.json"
    selection.write_text("{}\n", encoding="utf-8")
    support = tmp_path / "support.yaml"
    support.write_text("status: complete\n", encoding="utf-8")
    protocol_path = tmp_path / "resolved-pilot.yaml"
    protocol = {
        "schema_version": "path_c_r015_pilot_protocol_v1",
        "status": "resolved_for_authorized_pilot",
        "scientific_readout_allowed": False,
        "frozen_inputs": {
            "preregistration": str(tmp_path / "prefilled_preregistration.yaml"),
            "design_selection_report": str(selection),
            "signed_freeze_manifest": str(
                tmp_path / "pilot_frozen_manifest.json"
            ),
            "support_registration": str(support),
        },
        "outputs": {
            "report_path": str(tmp_path / "pilot_wiring_report.json")
        },
        "execution_boundary": {
            "current_execution_authorized": False,
            "formal_audit_included": False,
        },
    }
    protocol_path.write_text(
        yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8"
    )
    artifacts._validate_resolved_pilot_protocol(
        protocol_path=protocol_path,
        design_selection_report_path=selection,
        support_registration_path=support,
    )

    protocol["outputs"]["report_path"] = "pending_pilot_run"
    protocol_path.write_text(
        yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="pending value"):
        artifacts._validate_resolved_pilot_protocol(
            protocol_path=protocol_path,
            design_selection_report_path=selection,
            support_registration_path=support,
        )

    protocol["outputs"]["report_path"] = str(
        tmp_path / "pilot_wiring_report.json"
    )
    protocol["frozen_inputs"]["design_selection_report"] = str(
        tmp_path / "other-selection.json"
    )
    protocol_path.write_text(
        yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="another design selection"):
        artifacts._validate_resolved_pilot_protocol(
            protocol_path=protocol_path,
            design_selection_report_path=selection,
            support_registration_path=support,
        )


def test_checkpoint_binding_keeps_freeze_input_schema_and_recomputes_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_sha256 = "a" * 64
    weights_sha256 = "b" * 64
    (tmp_path / "config.yaml").write_text("fixture: true\n", encoding="utf-8")
    entries = []
    for seed, role in (
        (100, "ego"),
        (101, "partner"),
        (102, "partner"),
        (201, "partner"),
        (202, "partner"),
    ):
        checkpoint = tmp_path / f"checkpoint-{seed}"
        checkpoint.mkdir()
        manifest = tmp_path / f"manifest-{seed}.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": "path_c_official_training_artifact_v2",
                    "seed": seed,
                    "training_run_id": f"{seed:064x}",
                    "training_config_path": str(tmp_path / "config.yaml"),
                    "checkpoint": {
                        "path": str(checkpoint),
                        "format": "orbax_pytree_v1",
                        "parameter_tree_path": ["params"],
                        "checkpoint_sha256": checkpoint_sha256,
                        "model_weights_hash_domain": (
                            artifacts.FLAX_WEIGHTS_HASH_DOMAIN
                        ),
                        "model_weights_sha256": weights_sha256,
                    },
                }
            ),
            encoding="utf-8",
        )
        entries.append(
            {
                "artifact_id": f"artifact-{seed}",
                "role": role,
                "training_seed": seed,
                "path": str(checkpoint),
                "format": "orbax_pytree_v1",
                "parameter_tree_path": ["params"],
                "training_manifest_path": str(manifest),
            }
        )
    monkeypatch.setattr(artifacts, "load_flax_parameter_tree", lambda *a, **k: {})
    monkeypatch.setattr(
        artifacts,
        "validate_official_artifact_manifest",
        lambda *a, **k: {
            "family_id": "fixture_family",
            "snapshot_environment_steps": 30_000_000,
        },
    )
    monkeypatch.setattr(
        artifacts, "checkpoint_artifact_sha256", lambda path: checkpoint_sha256
    )
    monkeypatch.setattr(artifacts, "flax_weights_sha256", lambda params: weights_sha256)

    bound = artifacts._checkpoint_by_seed({"checkpoints": entries})
    assert set(bound) == {100, 101, 102, 201, 202}
    assert bound[100]["checkpoint_sha256"] == checkpoint_sha256
    assert bound[100]["model_weights_sha256"] == weights_sha256
    assert set(entries[0]) == {
        "artifact_id",
        "role",
        "training_seed",
        "path",
        "format",
        "parameter_tree_path",
        "training_manifest_path",
    }
