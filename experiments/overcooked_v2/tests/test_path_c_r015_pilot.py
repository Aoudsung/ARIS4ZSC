"""R015 试点恢复、真实重放调用与九项核查的静态测试定义。"""

from __future__ import annotations

import copy
import inspect
import json

import pytest
import yaml

from experiments.overcooked_v2 import path_c_r015_pilot as pilot
from experiments.overcooked_v2.path_c_r015_controller import (
    canonical_sha256,
    derive_controller_key,
)


def test_pilot_uses_real_executor_registered_replay_and_full_freeze_validation():
    source = inspect.getsource(pilot.run_r015_pilot)
    assert "executor.run_paired_block(**arguments)" in source
    assert "build_r015_replay_backend" in source
    assert "validate_authorized_pilot_freeze_manifest" in source
    assert "_verify_pilot_input_bindings" in source
    assert "CallableR015ReplayBackendV1" not in source
    assert "_independent_block_replayers(arguments)" in source


def test_direct_pilot_rejects_the_unresolved_template_before_backend_loading(
    tmp_path,
):
    """配置模板不是可运行协议；只有冻结阶段填完路径后才能进入试点。"""

    protocol_path = tmp_path / "pilot_template.yaml"
    protocol_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "path_c_r015_pilot_protocol_v1",
                "status": "template_not_valid_for_runs",
                "scientific_readout_allowed": False,
                "execution_boundary": {
                    "current_execution_authorized": False,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    inputs = []
    for name in ("design", "selection", "freeze"):
        path = tmp_path / f"{name}.yaml"
        path.write_text("{}\n", encoding="utf-8")
        inputs.append(path)
    with pytest.raises(ValueError, match="concrete inputs are unresolved"):
        pilot.run_r015_pilot(
            {
                "_execution_authorized": True,
                "_authorization_reference": "signed-test-authorization",
                "_config_path": str(protocol_path),
                "_design_protocol_path": str(inputs[0]),
                "_design_selection_report_path": str(inputs[1]),
                "_freeze_manifest_path": str(inputs[2]),
                "_output_dir": str(tmp_path / "pilot-output"),
            }
        )


def test_resolved_pilot_protocol_binds_all_runtime_inputs_and_report_path():
    """试点必须同时服从选择、清单、支持登记、预登记与报告路径。"""

    source = inspect.getsource(pilot.run_r015_pilot)
    for field in (
        "frozen_inputs.design_selection_report",
        "frozen_inputs.signed_freeze_manifest",
        "frozen_inputs.support_registration",
        "frozen_inputs.preregistration",
        "outputs.report_path",
    ):
        assert field in source
    assert "declared_selection != selection_path" in source
    assert "declared_manifest != freeze_manifest_path" in source
    assert 'declared_report != output_dir / "pilot_wiring_report.json"' in source
    assert "declared_preregistration.is_file()" in source
    assert "preregistration_binding.get(" in source
    assert "_file_sha256(declared_preregistration)" in source
    assert "declared_support != support_path" in source
    assert "_verify_pilot_input_bindings(" in source


def test_pilot_input_bindings_reject_protocol_support_or_prototype_drift(tmp_path):
    paths = {
        name: tmp_path / f"{name}.yaml"
        for name in ("design", "pilot", "selection", "support")
    }
    for name, path in paths.items():
        path.write_text(f"name: {name}\n", encoding="utf-8")
    prototypes = ("prototype-1", "prototype-2", "prototype-3", "prototype-4")
    manifest = {
        "protocol_artifacts": {
            "design_data_protocol": {
                "path": str(paths["design"].resolve()),
                "sha256": pilot._file_sha256(paths["design"]),
            },
            "pilot_protocol": {
                "path": str(paths["pilot"].resolve()),
                "sha256": pilot._file_sha256(paths["pilot"]),
            },
        },
        "design_selection": {
            "report_path": str(paths["selection"].resolve()),
            "report_sha256": pilot._file_sha256(paths["selection"]),
        },
        "partner_support": {
            "registration_path": str(paths["support"].resolve()),
            "registration_sha256": pilot._file_sha256(paths["support"]),
            "candidate_ids": list(prototypes),
        },
    }
    arguments = {
        "freeze_manifest": manifest,
        "pilot_protocol_path": paths["pilot"],
        "design_protocol_path": paths["design"],
        "selection_path": paths["selection"],
        "support_registration_path": paths["support"],
        "prototype_ids": prototypes,
    }
    pilot._verify_pilot_input_bindings(**arguments)

    changed = copy.deepcopy(manifest)
    changed["partner_support"]["candidate_ids"] = list(reversed(prototypes))
    with pytest.raises(ValueError, match="prototype order"):
        pilot._verify_pilot_input_bindings(
            **{**arguments, "freeze_manifest": changed}
        )
    replacement = tmp_path / "replacement-pilot.yaml"
    replacement.write_text("replacement: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="pilot protocol"):
        pilot._verify_pilot_input_bindings(
            **{**arguments, "pilot_protocol_path": replacement}
        )


def test_atomic_block_records_are_the_resume_source_of_truth(tmp_path):
    block = {
        "schema_version": pilot.PILOT_BLOCK_SCHEMA,
        "scientific_readout_allowed": False,
        "audit_unit_id": "a" * 64,
    }
    pilot._write_atomic_block_record(tmp_path, ordinal=1, block=block)
    assert pilot._load_atomic_block_records(tmp_path) == [block]
    with pytest.raises(FileExistsError, match="overwrite"):
        pilot._write_atomic_block_record(tmp_path, ordinal=1, block=block)

    path = pilot._block_record_path(tmp_path, 1)
    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["block"]["audit_unit_id"] = "b" * 64
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        pilot._load_atomic_block_records(tmp_path)


def test_run_binding_rejects_source_config_or_seed_drift(tmp_path):
    binding = {
        "schema_version": pilot.PILOT_RUN_BINDING_SCHEMA,
        "scientific_readout_allowed": False,
        "source_sha256": {"pilot": "a" * 64},
    }
    pilot._bind_or_verify_run(tmp_path, binding)
    pilot._bind_or_verify_run(tmp_path, copy.deepcopy(binding))
    changed = copy.deepcopy(binding)
    changed["source_sha256"]["pilot"] = "b" * 64
    with pytest.raises(ValueError, match="changed"):
        pilot._bind_or_verify_run(tmp_path, changed)


def _key_only_block():
    audit_unit_id = "1" * 64
    episode_seed = 17
    prototype_id = "prototype-1"
    root = canonical_sha256(
        ["r015_paired_block_v1", audit_unit_id, episode_seed, prototype_id]
    )
    steps = []
    for environment_step in range(400):
        branch_key = derive_controller_key(
            root,
            "actual_episode",
            environment_step,
        )
        steps.append(
            {
                "environment_random_key": derive_controller_key(
                    branch_key,
                    "future_environment",
                    0,
                )
            }
        )
    return {
        "audit_unit_id": audit_unit_id,
        "episode_seed": episode_seed,
        "partner_prototype_id": prototype_id,
        "probe_step": None,
        "groups": {group: {"environment_steps": steps} for group in pilot.R015_FORMAL_GROUPS},
    }


def test_random_key_check_recomputes_every_recorded_key():
    block = _key_only_block()
    assert pilot._random_keys_verified(block) is True
    block["groups"]["A2-use"]["environment_steps"][7] = {
        "environment_random_key": "f" * 64
    }
    assert pilot._random_keys_verified(block) is False


def test_shared_firing_check_recomputes_safety_result_and_group_binding():
    no_fire = {
        "probe_fired": False,
        "probe_step": None,
        "selected_probe_id": None,
        "consultations": [{"selected_for_safety_probe_id": None}],
        "safety_comparisons": [],
        "groups": {group: {} for group in pilot.R015_FORMAL_GROUPS},
    }
    assert pilot._shared_firing_indicator_verified(no_fire) is True

    fire = {
        "probe_fired": True,
        "probe_step": 6,
        "selected_probe_id": "interact",
        "consultations": [{"selected_for_safety_probe_id": "interact"}],
        "safety_comparisons": [
            {
                "positive_posterior_support": True,
                "compatible_hidden_state_reconstructed": True,
                "wrong_delivery_count": 0,
            }
            for _ in range(4)
        ],
        "groups": {
            "A1": {},
            "A2-mask": {"selected_probe_id": "interact"},
            "A2-use": {"selected_probe_id": "interact"},
        },
    }
    assert pilot._shared_firing_indicator_verified(fire) is True
    fire["safety_comparisons"][0]["wrong_delivery_count"] = 1
    assert pilot._shared_firing_indicator_verified(fire) is False


def test_firing_count_isolation_exercises_the_formal_count_only_parser():
    payload = {
        "schema_version": "path_c_r015_firing_count_checkpoint_v1",
        "experiment_id": "R015",
        "checkpoint_round_count": 20,
        "firing_counts_by_prototype": {f"prototype-{index}": 0 for index in range(4)},
    }
    assert pilot._firing_count_input_isolation_verified(payload) is True


def test_zero_probe_explanation_requires_an_observed_reason_count():
    block = {
        "probe_fired": False,
        "zero_probe_interpretability_counts": {
            "candidate_evaluation_count": 120,
            "safety_rejection_count": 0,
            "nonpositive_score_count": 19,
            "window_expiration_count": 1,
        },
    }
    assert pilot._zero_probe_explanation_verified(block) is True
    block["zero_probe_interpretability_counts"]["nonpositive_score_count"] = 0
    block["zero_probe_interpretability_counts"]["window_expiration_count"] = 0
    assert pilot._zero_probe_explanation_verified(block) is False
