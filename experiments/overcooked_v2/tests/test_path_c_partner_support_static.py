"""Official-trainer support contracts for opportunity audit R015.

Most cases create only tiny synthetic files.  The explicitly named Type-A case
is reserved for the authorized remote GPU wiring check and is not a scientific
readout.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pytest
import yaml

from experiments.overcooked_v2.path_c_official_artifact import (
    FLAX_WEIGHTS_HASH_DOMAIN,
    OFFICIAL_ACTION_RULE_ID,
    OFFICIAL_EFFECTIVE_STEP_CONTRACTS,
    OFFICIAL_FAMILY_DEFINITIONS,
    OFFICIAL_FAMILY_HASHES,
    OFFICIAL_OP_FAMILY_ID,
    OFFICIAL_SOURCE_CLOSURE_SCHEMA_VERSION,
    OFFICIAL_SP_FAMILY_ID,
    build_official_artifact_manifest,
    flax_weights_sha256,
    validate_official_launch_config,
    validate_official_artifact_manifest,
)
from experiments.overcooked_v2.path_c_official_evidence import (
    generate_official_r015_support_report,
)
from experiments.overcooked_v2.official.overcooked_v2_experiments_adapter import (
    OFFICIAL_ACTION_ORDER,
    OFFICIAL_CHECKPOINT_FORMAT,
    OFFICIAL_PARAMETER_TREE_PATH,
    _completed_episode_count_from_metrics,
    compose_official_training_config,
    evaluate_official_reference_acceptance,
    official_source_dependency_records,
    reference_configuration_comparison,
    run_type_a_round_trip,
    summarize_type_a_production_cost,
    validate_official_reference_acceptance_report,
    validate_reference_acceptance_contract,
    verify_other_play_symmetry,
)
from experiments.overcooked_v2.path_c_flax_policy import (
    CallableOfficialFlaxNetworkAdapter,
    OfficialFlaxPolicy,
)
from experiments.overcooked_v2.path_c_pool_admission import (
    R015_ADMISSION_IMPLEMENTATION_DEPENDENCIES,
    R015PartnerSupportSpec,
    configured_r015_support_status,
    load_r015_partner_support_config,
    select_r015_support_members,
    validate_r015_support_report_evidence,
)
from experiments.overcooked_v2.path_c_r015_runtime import (
    R015OfficialPolicyRouterV1,
)
from experiments.overcooked_v2.path_c_standard_training import (
    canonical_mapping_sha256,
    partner_training_run_id,
    repository_source_dependency_closure,
)
from experiments.overcooked_v2.scripts.run_path_c_official_op_training import (
    other_play_reference_comparison,
)


def test_training_adapter_is_exactly_restored_and_has_no_r015_imports():
    adapter_path = (
        Path(__file__).resolve().parents[1]
        / "official"
        / "overcooked_v2_experiments_adapter.py"
    )
    source = adapter_path.read_bytes()
    assert hashlib.sha256(source).hexdigest() == (
        "b139587a571bd187078f4a251b1427d912efdd62e42847649c719fc5293f11c8"
    )
    tree = ast.parse(source.decode("utf-8"), filename=str(adapter_path))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not any(
        name.startswith("experiments.overcooked_v2.path_c_r015")
        for name in imported
    )


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
SUPPORT_CONFIG = CONFIG_DIR / "path_c_r015_partner_support_simple.yaml"
LAUNCH_CONFIGS = (
    CONFIG_DIR / "path_c_official_sp_simple_seed100.yaml",
    CONFIG_DIR / "path_c_official_sp_simple_seed101.yaml",
    CONFIG_DIR / "path_c_official_sp_simple_seed102.yaml",
    CONFIG_DIR / "path_c_official_op_simple_seed201.yaml",
    CONFIG_DIR / "path_c_official_op_simple_seed202.yaml",
)
REFERENCE_ACCEPTANCE_CONFIG = (
    CONFIG_DIR / "path_c_official_sp_simple_seed999_acceptance.yaml"
)


def test_official_completed_episode_count_is_recovered_from_averaged_metric() -> None:
    assert _completed_episode_count_from_metrics(
        [0.0, 1.0 / (256 * 256), 2.0 / (256 * 256)],
        num_steps=256,
        num_envs=256,
    ) == 3
    with pytest.raises(RuntimeError, match="integer count"):
        _completed_episode_count_from_metrics(
            [0.5 / (256 * 256)],
            num_steps=256,
            num_envs=256,
        )


def test_official_launch_registrations_bind_signed_variants_and_runtime() -> None:
    payloads = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in LAUNCH_CONFIGS]
    assert [payload["seed"] for payload in payloads] == [100, 101, 102, 201, 202]
    assert [payload["experiment_variant"] for payload in payloads] == [
        "rnn-sp",
        "rnn-sp",
        "rnn-sp",
        "rnn-op",
        "rnn-op",
    ]
    assert all(payload["TOTAL_TIMESTEPS"] == 3.0e7 for payload in payloads)
    assert all(payload["WANDB_MODE"] == "disabled" for payload in payloads)
    assert all(
        payload["environment_kwargs"]["indicate_successful_delivery"] is True
        and payload["runtime"]["JAX_PLATFORMS"] == "cuda,cpu"
        and payload["registration_status"] == "static_registered_not_frozen"
        and payload["scientific_readout_allowed"] is False
        for payload in payloads
    )
    assert all(
        payload["environment_kwargs"]
        == {
            "layout": "test_time_simple",
            "max_steps": 400,
            "observation_type": "DEFAULT",
            "agent_view_size": 2,
            "negative_rewards": True,
            "random_agent_positions": True,
            "sample_recipe_on_delivery": True,
            "indicate_successful_delivery": True,
            "force_path_planning": False,
            "random_reset": False,
        }
        for payload in payloads
    )
    assert all(
        payload["wiring_verification"]["status"] == "verified"
        and payload["type_a_acceptance"]["status"]
        == "amended_pending_registered_reference_run"
        and payload["type_a_acceptance"]["reference_acceptance_report"].endswith(
            "/reference_acceptance_report.json"
        )
        and payload["checkpoint"]["format"] == OFFICIAL_CHECKPOINT_FORMAT
        and payload["checkpoint"]["parameter_tree_path"]
        == list(OFFICIAL_PARAMETER_TREE_PATH)
        and payload["model"]["action_order"] == list(OFFICIAL_ACTION_ORDER)
        for payload in payloads
    )
    assert {
        payload["effective_environment_steps_contract"][
            "effective_environment_steps"
        ]
        for payload in payloads
    } == {29_949_952, 29_999_104}
    assert all(
        payload["effective_environment_steps_contract"]
        == OFFICIAL_EFFECTIVE_STEP_CONTRACTS[payload["experiment_variant"]]
        for payload in payloads
    )
    assert "pending_wiring_verification" not in "\n".join(
        path.read_text(encoding="utf-8") for path in LAUNCH_CONFIGS
    )


def test_official_hydra_composition_and_other_play_symmetry_are_real() -> None:
    sp_launch = yaml.safe_load(LAUNCH_CONFIGS[1].read_text(encoding="utf-8"))
    op_launch = yaml.safe_load(LAUNCH_CONFIGS[3].read_text(encoding="utf-8"))
    sp = compose_official_training_config(sp_launch)
    op = compose_official_training_config(op_launch)
    assert sp["model"]["NUM_ENVS"] == 256
    assert sp["model"]["NUM_STEPS"] == 256
    assert sp["model"]["UPDATE_EPOCHS"] == 4
    assert sp["model"]["NUM_MINIBATCHES"] == 64
    assert sp["model"]["TOTAL_TIMESTEPS"] == 30_000_000
    assert sp["model"]["REW_SHAPING_HORIZON"] == 15_000_000
    assert "op_ingredient_permutations" not in sp["env"]["ENV_KWARGS"]
    assert op["env"]["ENV_KWARGS"]["op_ingredient_permutations"] == [0, 1]
    assert op["model"]["TOTAL_TIMESTEPS"] == 30_000_000
    assert op["model"]["REW_SHAPING_HORIZON"] == 15_000_000
    assert op["model"]["NUM_ENVS"] == 64
    assert op["model"]["ENT_COEF"] == 0.02
    assert (
        int(sp["model"]["TOTAL_TIMESTEPS"])
        // sp["model"]["NUM_ENVS"]
        // sp["model"]["NUM_STEPS"]
        * sp["model"]["NUM_ENVS"]
        * sp["model"]["NUM_STEPS"]
    ) == 29_949_952
    assert (
        int(op["model"]["TOTAL_TIMESTEPS"])
        // op["model"]["NUM_ENVS"]
        // op["model"]["NUM_STEPS"]
        * op["model"]["NUM_ENVS"]
        * op["model"]["NUM_STEPS"]
    ) == 29_999_104
    symmetry = verify_other_play_symmetry(op)
    assert symmetry["both_symmetries_seen_per_agent"] is True
    assert symmetry["independent_agent_permutations_seen"] is True
    comparison = other_play_reference_comparison(op)
    assert comparison["matches_accepted_reference"] is True
    assert comparison["checks"]["registered_other_play_configuration"] is True


def test_official_source_closure_and_registered_reference_are_identical() -> None:
    launch = yaml.safe_load(LAUNCH_CONFIGS[1].read_text(encoding="utf-8"))
    resolved = compose_official_training_config(launch)
    comparison = reference_configuration_comparison(resolved)
    assert comparison["matches_accepted_reference"] is True
    assert comparison["material_differences"] == []
    assert comparison["reference_id"] == (
        "overcooked_v2_experiments_actor_critic_rnn_v1"
    )
    assert comparison["accepted_reference_source"] == comparison[
        "candidate_network_source"
    ]
    closure = official_source_dependency_records("rnn-sp")
    assert closure
    assert any(item["path"].endswith("/ppo/main.py") for item in closure)
    assert any(item["path"].endswith("/ppo/models/cnn.py") for item in closure)
    assert any(item["path"].endswith("/eval/utils.py") for item in closure)
    assert any(
        item["path"].endswith("/path_c_official_artifact.py") for item in closure
    )
    assert all(Path(item["path"]).is_file() for item in closure)
    assert all(len(item["sha256"]) == 64 for item in closure)


def test_seed999_reference_acceptance_contract_and_boundaries_are_frozen() -> None:
    launch = yaml.safe_load(
        REFERENCE_ACCEPTANCE_CONFIG.read_text(encoding="utf-8")
    )
    validate_official_launch_config(launch)
    contract = validate_reference_acceptance_contract(launch)
    assert contract["seed"] == 999
    assert contract["minimum_final_quarter_mean_raw_return"] == 100.0
    assert contract["minimum_final_to_peak_quarter_ratio"] == 0.9

    values = [40.0] * 114 + [80.0] * 114 + [120.0] * 114 + [110.0] * 115
    report = evaluate_official_reference_acceptance(
        values,
        launch_config=launch,
    )
    assert report["quarter_boundaries_zero_based"] == [0, 114, 228, 342, 457]
    assert report["quarter_means"] == [40.0, 80.0, 120.0, 110.0]
    assert report["final_to_peak_quarter_ratio"] == pytest.approx(11.0 / 12.0)
    assert report["acceptance_pass"] is True
    assert report["scientific_readout_allowed"] is False


def test_seed999_reference_acceptance_requires_performance_and_no_collapse() -> None:
    launch = yaml.safe_load(
        REFERENCE_ACCEPTANCE_CONFIG.read_text(encoding="utf-8")
    )
    collapse = [40.0] * 114 + [80.0] * 114 + [120.0] * 114 + [107.0] * 115
    collapse_report = evaluate_official_reference_acceptance(
        collapse,
        launch_config=launch,
    )
    assert collapse_report["performance_threshold_pass"] is True
    assert collapse_report["no_collapse_threshold_pass"] is False
    assert collapse_report["acceptance_pass"] is False

    low_return = [40.0] * 114 + [80.0] * 114 + [100.0] * 114 + [99.0] * 115
    low_return_report = evaluate_official_reference_acceptance(
        low_return,
        launch_config=launch,
    )
    assert low_return_report["performance_threshold_pass"] is False
    assert low_return_report["acceptance_pass"] is False
    with pytest.raises(ValueError, match="every update"):
        evaluate_official_reference_acceptance(
            low_return[:-1],
            launch_config=launch,
        )


def test_seed999_report_is_recomputed_before_production(tmp_path: Path) -> None:
    launch = yaml.safe_load(
        REFERENCE_ACCEPTANCE_CONFIG.read_text(encoding="utf-8")
    )
    values = [40.0] * 114 + [80.0] * 114 + [120.0] * 114 + [110.0] * 115
    report = evaluate_official_reference_acceptance(values, launch_config=launch)
    report.update(
        {
            "launch_config_path": str(REFERENCE_ACCEPTANCE_CONFIG.resolve()),
            "launch_config_sha256": hashlib.sha256(
                REFERENCE_ACCEPTANCE_CONFIG.read_bytes()
            ).hexdigest(),
            "reference_acceptance_contract": dict(
                validate_reference_acceptance_contract(launch)
            ),
            "completed_environment_steps": 29_949_952,
            "returned_episode_returns_by_update": values,
            "source_dependencies": official_source_dependency_records("rnn-sp"),
            "reference_comparison": {
                "reference_id": "overcooked_v2_experiments_actor_critic_rnn_v1",
                "candidate_network_source_sha256": (
                    "b97c92133ed073e780b643e8e4b0c8fc27f4963ab99c2d8bcf0f099a5a78a1d0"
                ),
            },
        }
    )
    report_path = tmp_path / "reference_acceptance_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    validated = validate_official_reference_acceptance_report(
        REFERENCE_ACCEPTANCE_CONFIG,
        report_path,
    )
    assert validated["acceptance_pass"] is True

    report["final_quarter_mean_raw_return"] = 99.0
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="final_quarter_mean_raw_return"):
        validate_official_reference_acceptance_report(
            REFERENCE_ACCEPTANCE_CONFIG,
            report_path,
        )


def test_type_a_cost_projection_uses_three_sp_and_two_other_play_runs() -> None:
    summary = summarize_type_a_production_cost(
        {
            "experiment_variant": "rnn-sp",
            "conservative_full_run_gpu_hours": 0.7,
        },
        {
            "experiment_variant": "rnn-op",
            "conservative_full_run_gpu_hours": 1.2,
        },
    )
    assert summary["predicted_total_gpu_hours"] == pytest.approx(4.5)
    assert summary["effective_environment_steps_contract"] == 149_848_064
    assert summary["acceptance_plus_production_gpu_hours"] == pytest.approx(5.1)
    assert summary["acceptance_plus_production_within_hard_stop"] is True
    assert summary["within_start_ceiling"] is True
    assert summary["hard_stop_gpu_hours"] == 6.0


def test_official_gpu_checkpoint_policy_and_full_episode_round_trip(
    tmp_path: Path,
) -> None:
    report = run_type_a_round_trip(
        LAUNCH_CONFIGS[1],
        tmp_path,
    )
    assert report["scientific_readout_allowed"] is False
    assert report["jax_version"] == "0.4.38"
    assert report["jaxlib_version"] == "0.4.38"
    assert report["jax_backend"] == "gpu"
    assert report["model_weights_sha256_before_save"] == report[
        "model_weights_sha256_after_restore"
    ]
    assert report["official_wrapper_action_equal"] is True
    assert report["official_wrapper_recurrent_state_equal"] is True
    assert report["official_wrapper_logits_equal"] is True
    assert report["official_full_episode_actions_equal"] is True
    assert report["official_full_episode_metrics_equal"] is True
    assert report["episode_steps"] == 400
    assert sum(report["episode_metrics"]["policy_0_action_counts"]) == 400
    assert sum(report["episode_metrics"]["policy_1_action_counts"]) == 400
    assert report["success_indicator_count"] == report["episode_metrics"][
        "correct_delivery_count"
    ]


def test_r015_static_registration_has_two_official_families_and_disjoint_ego() -> None:
    spec = R015PartnerSupportSpec.from_mapping(
        load_r015_partner_support_config(SUPPORT_CONFIG)
    )
    assert {family.family_id for family in spec.families} == {
        OFFICIAL_SP_FAMILY_ID,
        OFFICIAL_OP_FAMILY_ID,
    }
    assert {family.action_rule for family in spec.families} == {
        OFFICIAL_ACTION_RULE_ID
    }
    by_family = {
        family.family_id: [
            candidate
            for candidate in spec.candidates
            if candidate.family_id == family.family_id
        ]
        for family in spec.families
    }
    assert {candidate.training_seed for candidate in by_family[OFFICIAL_SP_FAMILY_ID]} == {
        101,
        102,
    }
    assert {candidate.training_seed for candidate in by_family[OFFICIAL_OP_FAMILY_ID]} == {
        201,
        202,
    }
    assert [candidate.expected_environment_steps for candidate in spec.candidates] == [
        29_949_952,
        29_949_952,
        29_999_104,
        29_999_104,
    ]
    assert 100 not in {candidate.training_seed for candidate in spec.candidates}
    assert (
        OFFICIAL_FAMILY_DEFINITIONS[OFFICIAL_SP_FAMILY_ID]["convention_generation"]
        != OFFICIAL_FAMILY_DEFINITIONS[OFFICIAL_OP_FAMILY_ID][
            "convention_generation"
        ]
    )


def test_r015_missing_official_outputs_remain_configured_and_pending() -> None:
    spec = R015PartnerSupportSpec.from_mapping(
        load_r015_partner_support_config(SUPPORT_CONFIG)
    )
    status = configured_r015_support_status(spec)
    assert status["registration_status"] == "configured"
    assert status["support_status"] == "pending"
    assert status["support_complete"] is False
    assert status["members"] == {}
    dependencies = repository_source_dependency_closure(
        R015_ADMISSION_IMPLEMENTATION_DEPENDENCIES
    )
    assert status["admission_implementation_dependencies"] == dependencies
    assert status["admission_implementation_sha256"] == canonical_mapping_sha256(
        dependencies
    )


def _manual_flax_hash(params: dict[str, object]) -> str:
    digest = hashlib.sha256(FLAX_WEIGHTS_HASH_DOMAIN.encode("ascii") + b"\x00")

    def framed(payload: bytes) -> None:
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    leaves = [
        (("Dense_0", "bias"), np.asarray(params["Dense_0"]["bias"])),
        (("Dense_0", "kernel"), np.asarray(params["Dense_0"]["kernel"])),
        (("GRU_0", "kernel"), np.asarray(params["GRU_0"]["kernel"])),
    ]
    for path, value in leaves:
        array = np.ascontiguousarray(value)
        framed(json.dumps(list(path), separators=(",", ":")).encode("utf-8"))
        framed(str(array.dtype).encode("ascii"))
        framed(",".join(str(item) for item in array.shape).encode("ascii"))
        framed(array.tobytes(order="C"))
    return digest.hexdigest()


def test_flax_weight_hash_matches_manual_fixture_and_is_sensitive() -> None:
    params = {
        "GRU_0": {"kernel": np.asarray([[1, 2]], dtype=np.int16)},
        "Dense_0": {
            "kernel": np.asarray([[1.0], [2.0]], dtype=np.float32),
            "bias": np.asarray([0.5], dtype=np.float32),
        },
    }
    assert flax_weights_sha256(params) == _manual_flax_hash(params)
    reordered = {"Dense_0": params["Dense_0"], "GRU_0": params["GRU_0"]}
    assert flax_weights_sha256(reordered) == flax_weights_sha256(params)

    changed_bytes = copy.deepcopy(params)
    changed_bytes["Dense_0"]["bias"][0] = 0.75
    assert flax_weights_sha256(changed_bytes) != flax_weights_sha256(params)

    changed_dtype = copy.deepcopy(params)
    changed_dtype["GRU_0"]["kernel"] = changed_dtype["GRU_0"]["kernel"].astype(
        np.int32
    )
    assert flax_weights_sha256(changed_dtype) != flax_weights_sha256(params)

    changed_shape = copy.deepcopy(params)
    changed_shape["Dense_0"]["kernel"] = changed_shape["Dense_0"][
        "kernel"
    ].reshape(1, 2)
    assert flax_weights_sha256(changed_shape) != flax_weights_sha256(params)

    changed_path = copy.deepcopy(params)
    changed_path["GRU_renamed"] = changed_path.pop("GRU_0")
    assert flax_weights_sha256(changed_path) != flax_weights_sha256(params)


def test_flax_weight_hash_excludes_checkpoint_metadata() -> None:
    params = {"params": {"kernel": np.asarray([1.0, 2.0], dtype=np.float32)}}
    left_checkpoint = {"params": params, "metadata": {"label": "left"}}
    right_checkpoint = {"params": params, "metadata": {"label": "right"}}
    assert flax_weights_sha256(left_checkpoint["params"]) == flax_weights_sha256(
        right_checkpoint["params"]
    )


def _materialize_official_artifact(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    launch = yaml.safe_load(LAUNCH_CONFIGS[1].read_text(encoding="utf-8"))
    checkpoint_path = tmp_path / "checkpoint.msgpack"
    checkpoint_path.write_bytes(b"synthetic checkpoint container")
    launch["environment_config"] = str(
        (CONFIG_DIR / "ocv2_test_time_simple_standard.yaml").resolve()
    )
    launch["checkpoint"] = {
        "expected_path": str(checkpoint_path),
        "format": "flax_msgpack_file_v1",
        "parameter_tree_path": ["params"],
        "native_save_mechanism": "verified_fixture",
    }
    launch["model"] = {
        "model_class": "OfficialFlaxRecurrentActor",
        "network_definition": "official.fixture.ActorCriticRNN",
        "recurrent_state_initializer": "official.fixture.initialize_carry",
        "apply_interface": "official_rnn_actor_apply_v1",
        "action_rule": OFFICIAL_ACTION_RULE_ID,
        "action_order": list(OFFICIAL_ACTION_ORDER),
    }
    launch["wiring_verification"] = {
        "status": "verified",
        "official_entrypoint_requires_repository_adapter": True,
    }
    launch_path = tmp_path / "launch.yaml"
    launch_path.write_text(yaml.safe_dump(launch, sort_keys=False), encoding="utf-8")

    source_path = tmp_path / "official_source.py"
    source_path.write_text("OFFICIAL_FIXTURE = True\n", encoding="utf-8")
    params = {
        "params": {
            "Dense_0": {
                "kernel": np.asarray([[1.0, 2.0]], dtype=np.float32),
                "bias": np.asarray([0.0, 0.0], dtype=np.float32),
            }
        }
    }
    descriptor = {
        "schema_version": "path_c_official_run_descriptor_v1",
        "run_status": "completed",
        "experiment_variant": "rnn-sp",
        "layout": "test_time_simple",
        "seed": 101,
        "effective_environment_steps": 29_949_952,
        "environment_kwargs": launch["environment_kwargs"],
        "architecture": launch["model"],
        "checkpoint": {
            "path": str(checkpoint_path),
            "format": "flax_msgpack_file_v1",
            "parameter_tree_path": ["params"],
        },
        "training_source_dependencies": [
            {
                "path": str(source_path),
                "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            }
        ],
    }
    descriptor_path = tmp_path / "run_descriptor.json"
    descriptor_path.write_text(json.dumps(descriptor), encoding="utf-8")
    manifest = build_official_artifact_manifest(
        launch_path,
        descriptor_path,
        params=params["params"],
    )
    manifest_path = tmp_path / "official_artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return launch_path, manifest_path, checkpoint_path, source_path, params["params"]


def test_official_artifact_accepts_complete_binding_and_rejects_missing_field(
    tmp_path: Path,
) -> None:
    launch, manifest, checkpoint, _, params = _materialize_official_artifact(tmp_path)
    validated = validate_official_artifact_manifest(
        launch,
        manifest,
        expected_checkpoint_path=checkpoint,
        params=params,
    )
    assert validated["artifact_verified"] is True
    assert validated["model_weights_hash_domain"] == FLAX_WEIGHTS_HASH_DOMAIN
    assert validated["family_id"] == OFFICIAL_SP_FAMILY_ID
    assert validated["snapshot_environment_steps"] == 29_949_952
    assert json.loads(manifest.read_text(encoding="utf-8"))[
        "nominal_total_timesteps"
    ] == 30_000_000

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload.pop("architecture")
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="architecture"):
        validate_official_artifact_manifest(
            launch,
            manifest,
            expected_checkpoint_path=checkpoint,
            params=params,
        )

    launch, manifest, checkpoint, _, params = _materialize_official_artifact(
        tmp_path / "readout"
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["scientific_readout_allowed"] = True
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot authorize scientific readout"):
        validate_official_artifact_manifest(
            launch,
            manifest,
            expected_checkpoint_path=checkpoint,
            params=params,
        )


def test_official_artifact_rejects_architecture_and_dependency_tampering(
    tmp_path: Path,
) -> None:
    launch, manifest, checkpoint, source, params = _materialize_official_artifact(
        tmp_path
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["architecture"]["apply_interface"] = "tampered"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="architecture"):
        validate_official_artifact_manifest(
            launch,
            manifest,
            expected_checkpoint_path=checkpoint,
            params=params,
        )


def test_official_artifact_rejects_config_path_hash_and_format_tampering(
    tmp_path: Path,
) -> None:
    launch, manifest, checkpoint, _, params = _materialize_official_artifact(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["training_config_sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="training configuration binding"):
        validate_official_artifact_manifest(
            launch,
            manifest,
            expected_checkpoint_path=checkpoint,
            params=params,
        )

    launch, manifest, checkpoint, _, params = _materialize_official_artifact(
        tmp_path / "format"
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["checkpoint"]["format"] = "orbax_pytree_directory_v1"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint format changed"):
        validate_official_artifact_manifest(
            launch,
            manifest,
            expected_checkpoint_path=checkpoint,
            params=params,
        )

    launch, manifest, checkpoint, _, params = _materialize_official_artifact(
        tmp_path / "path"
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["environment_config_path"] = str(tmp_path / "another-environment.yaml")
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="environment configuration binding"):
        validate_official_artifact_manifest(
            launch,
            manifest,
            expected_checkpoint_path=checkpoint,
            params=params,
        )

    launch, manifest, checkpoint, source, params = _materialize_official_artifact(
        tmp_path / "dependency"
    )
    source.write_text("OFFICIAL_FIXTURE = False\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dependency closure hash mismatch"):
        validate_official_artifact_manifest(
            launch,
            manifest,
            expected_checkpoint_path=checkpoint,
            params=params,
        )


def _synthetic_selector_inputs(
    spec: R015PartnerSupportSpec,
) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for candidate in spec.candidates:
        family = spec.family(candidate.family_id)
        dependencies = {
            "schema_version": OFFICIAL_SOURCE_CLOSURE_SCHEMA_VERSION,
            "files": {
                f"/remote/{candidate.family_id}/trainer.py": hashlib.sha256(
                    candidate.family_id.encode("utf-8")
                ).hexdigest()
            },
        }
        implementation_sha256 = canonical_mapping_sha256(dependencies)
        config_sha256 = hashlib.sha256(
            f"config:{candidate.candidate_id}".encode("utf-8")
        ).hexdigest()
        architecture = {
            "model_class": family.model_class,
            "action_rule": family.action_rule,
        }
        results[candidate.candidate_id] = {
            "artifact_verified": True,
            "family_id": candidate.family_id,
            "family_spec_sha256": OFFICIAL_FAMILY_HASHES[candidate.family_id],
            "training_seed": candidate.training_seed,
            "snapshot_environment_steps": candidate.expected_environment_steps,
            "checkpoint_path": str(candidate.checkpoint_path),
            "training_config_path": str(candidate.training_config_path),
            "training_manifest_path": str(candidate.training_manifest_path),
            "checkpoint_sha256": hashlib.sha256(
                f"checkpoint:{candidate.candidate_id}".encode("utf-8")
            ).hexdigest(),
            "model_weights_sha256": hashlib.sha256(
                f"weights:{candidate.candidate_id}".encode("utf-8")
            ).hexdigest(),
            "training_config_sha256": config_sha256,
            "training_manifest_sha256": hashlib.sha256(
                f"manifest:{candidate.candidate_id}".encode("utf-8")
            ).hexdigest(),
            "environment_config_sha256": spec.environment_config_sha256,
            "training_implementation_dependencies": dependencies,
            "training_implementation_sha256": implementation_sha256,
            "training_run_id": partner_training_run_id(
                family_spec_sha256=family.family_spec_sha256,
                training_seed=candidate.training_seed,
                training_config_sha256=config_sha256,
                environment_config_sha256=spec.environment_config_sha256,
                training_implementation_sha256=implementation_sha256,
            ),
            "architecture": architecture,
            "architecture_sha256": canonical_mapping_sha256(architecture),
            "admitted": True,
        }
    return results


def test_r015_selector_rejects_same_flax_weights_in_two_candidates() -> None:
    spec = R015PartnerSupportSpec.from_mapping(
        load_r015_partner_support_config(SUPPORT_CONFIG)
    )
    results = _synthetic_selector_inputs(spec)
    first, second = (candidate.candidate_id for candidate in spec.candidates[:2])
    results[second]["model_weights_sha256"] = results[first]["model_weights_sha256"]
    with pytest.raises(ValueError, match="four distinct model weight states"):
        select_r015_support_members(spec, results)


def test_r015_all_or_nothing_support_remains_empty_when_one_candidate_fails() -> None:
    spec = R015PartnerSupportSpec.from_mapping(
        load_r015_partner_support_config(SUPPORT_CONFIG)
    )
    results = _synthetic_selector_inputs(spec)
    results[spec.candidates[0].candidate_id]["admitted"] = False
    selected = select_r015_support_members(spec, results)
    assert selected["support_complete"] is False
    assert selected["support_status"] == "not_admitted"
    assert selected["members"] == {}


def test_r015_registration_rejects_duplicate_run_or_checkpoint_path() -> None:
    payload = yaml.safe_load(SUPPORT_CONFIG.read_text(encoding="utf-8"))
    duplicate_seed = copy.deepcopy(payload)
    duplicate_seed["candidates"][1]["training_seed"] = duplicate_seed[
        "candidates"
    ][0]["training_seed"]
    with pytest.raises(ValueError, match="independent training runs"):
        R015PartnerSupportSpec.from_mapping(duplicate_seed)

    duplicate_path = copy.deepcopy(payload)
    duplicate_path["candidates"][3]["checkpoint_path"] = duplicate_path[
        "candidates"
    ][2]["checkpoint_path"]
    with pytest.raises(ValueError, match="checkpoints must be unique"):
        R015PartnerSupportSpec.from_mapping(duplicate_path)


class _FixtureOfficialRolloutBackend:
    def evaluate_pairing(
        self,
        policy_0_candidate_id: str,
        policy_1_candidate_id: str,
        *,
        canonical_episode_seeds: Sequence[int],
    ) -> Sequence[Mapping[str, Any]]:
        del policy_0_candidate_id, policy_1_candidate_id
        return [
            {
                "raw_episode_return": 20.0,
                "correct_delivery_count": 1,
                "wrong_delivery_count": 0,
                "indicator_activation_count": 0,
                "ambiguous_reward_step_count": 0,
                "policy_0_action_counts": [400, 0, 0, 0, 0, 0],
                "policy_1_action_counts": [0, 400, 0, 0, 0, 0],
            }
            for _ in canonical_episode_seeds
        ]


def _fixture_official_provenance_validator(spec, candidate, *, reported=None):
    family = spec.family(candidate.family_id)
    dependency_path = f"/remote/{candidate.family_id}/trainer.py"
    dependencies = {
        "schema_version": OFFICIAL_SOURCE_CLOSURE_SCHEMA_VERSION,
        "files": {
            dependency_path: hashlib.sha256(
                candidate.family_id.encode("utf-8")
            ).hexdigest()
        },
    }
    implementation_sha256 = canonical_mapping_sha256(dependencies)
    config_sha256 = hashlib.sha256(
        f"config:{candidate.candidate_id}".encode("utf-8")
    ).hexdigest()
    architecture = {
        "model_class": family.model_class,
        "action_rule": family.action_rule,
    }
    computed = {
        "artifact_verified": True,
        "family_id": candidate.family_id,
        "family_spec_sha256": family.family_spec_sha256,
        "training_seed": candidate.training_seed,
        "snapshot_environment_steps": candidate.expected_environment_steps,
        "checkpoint_path": str(candidate.checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(
            f"checkpoint:{candidate.candidate_id}".encode("utf-8")
        ).hexdigest(),
        "model_weights_sha256": hashlib.sha256(
            f"weights:{candidate.candidate_id}".encode("utf-8")
        ).hexdigest(),
        "model_weights_hash_domain": FLAX_WEIGHTS_HASH_DOMAIN,
        "training_config_path": str(candidate.training_config_path),
        "training_config_sha256": config_sha256,
        "training_manifest_path": str(candidate.training_manifest_path),
        "training_manifest_sha256": hashlib.sha256(
            f"manifest:{candidate.candidate_id}".encode("utf-8")
        ).hexdigest(),
        "environment_config_sha256": spec.environment_config_sha256,
        "training_implementation_dependencies": dependencies,
        "training_implementation_sha256": implementation_sha256,
        "training_run_id": partner_training_run_id(
            family_spec_sha256=family.family_spec_sha256,
            training_seed=candidate.training_seed,
            training_config_sha256=config_sha256,
            environment_config_sha256=spec.environment_config_sha256,
            training_implementation_sha256=implementation_sha256,
        ),
        "architecture": architecture,
        "architecture_sha256": canonical_mapping_sha256(architecture),
        "layout": "test_time_simple",
    }
    if reported is not None:
        for field, expected in computed.items():
            if field.endswith("_path"):
                assert Path(str(reported[field])).resolve() == Path(
                    str(expected)
                ).resolve()
            else:
                assert reported[field] == expected
    return computed


def test_official_evidence_generator_keeps_existing_1600_row_schema(
    tmp_path: Path,
) -> None:
    spec = replace(
        R015PartnerSupportSpec.from_mapping(
            load_r015_partner_support_config(SUPPORT_CONFIG)
        ),
        output_dir=tmp_path,
    )
    report = generate_official_r015_support_report(
        spec,
        _FixtureOfficialRolloutBackend(),
        provenance_validator=_fixture_official_provenance_validator,
    )
    assert report["schema_version"] == "path_c_r015_partner_support_report_v2"
    assert report["support_complete"] is True
    assert report["episode_evidence"]["row_count"] == 1_600
    evidence_rows = (tmp_path / "episode_returns.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(evidence_rows) == 1_600
    assert {
        json.loads(row)["schema_version"] for row in evidence_rows
    } == {"path_c_r015_partner_support_episode_v1"}
    members = validate_r015_support_report_evidence(
        spec,
        report,
        candidate_provenance_validator=_fixture_official_provenance_validator,
    )
    assert len(members) == 4


def test_official_flax_policy_hash_action_rule_and_cook_seat_router() -> None:
    import jax
    import jax.numpy as jnp

    params = {"params": {"kernel": np.asarray([1.0], dtype=np.float32)}}

    def initial_state(batch_size: int):
        return jnp.zeros((batch_size, 2), dtype=jnp.float32)

    def apply_actor(params, recurrent_state, observation, episode_start):
        del params, observation
        next_state = recurrent_state + jnp.asarray(episode_start)[..., None]
        logits = jnp.zeros((*recurrent_state.shape[:-1], 6), dtype=jnp.float32)
        return next_state, logits

    policy = OfficialFlaxPolicy(
        params=params,
        network=CallableOfficialFlaxNetworkAdapter(
            initial_state_fn=initial_state,
            apply_actor_fn=apply_actor,
        ),
        expected_model_weights_sha256=flax_weights_sha256(params),
    )
    with pytest.raises(ValueError, match="do not match their registered hash"):
        OfficialFlaxPolicy(
            params=params,
            network=policy.network,
            expected_model_weights_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="action rule changed"):
        OfficialFlaxPolicy(
            params=params,
            network=policy.network,
            expected_model_weights_sha256=flax_weights_sha256(params),
            action_rule="greedy",
        )
    state = policy.initial_state(1)
    step = policy.act(
        jnp.zeros((1, 5, 5, 39), dtype=jnp.float32),
        jnp.ones((1,), dtype=bool),
        state,
        jax.random.PRNGKey(0),
    )
    assert step.logits.shape == (1, 6)
    assert step.next_recurrent_state.shape == (1, 2)
    router = R015OfficialPolicyRouterV1(
        ego_policy=policy,
        partner_policy=policy,
    )
    assert router.ego_agent_id == "agent_1"
    assert router.partner_agent_id == "agent_0"
    with pytest.raises(ValueError, match="ego cook at agent_1"):
        R015OfficialPolicyRouterV1(
            ego_policy=policy,
            partner_policy=policy,
            ego_agent_id="agent_0",
            partner_agent_id="agent_1",
        )
