"""生成 R015 冻结所需的机器可读制品。

本模块只把已经选择的设计参数、实际实现文件、配置和 checkpoint 身份写成内容寻址
清单。它不运行环境、不读取正式结果，也不改变预登记状态。一个“制品包”是同一目录中
一次性写入的全部清单；目录已经存在时，只接受逐字节相同的重入调用。
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import yaml

from experiments.overcooked_v2.path_c_official_artifact import (
    FLAX_WEIGHTS_HASH_DOMAIN,
    checkpoint_artifact_sha256,
    flax_weights_sha256,
    load_flax_parameter_tree,
    validate_official_artifact_manifest,
)
from experiments.overcooked_v2.path_c_response_summary import ResponseSummarySpecV1
from experiments.overcooked_v2.path_c_r015_controller import (
    R015_FILTER_ALGORITHM_ID_V2,
    R015_RESAMPLING_ALGORITHM_ID_V2,
    continuation_controller_manifest_payload,
    sequential_controller_manifest_payload,
)
from experiments.overcooked_v2.path_c_r015_design import parse_delivery_reward
from experiments.overcooked_v2.path_c_r015_runtime import (
    R015_FORMAL_REPLACEMENT_ATTEMPT_COUNT,
    R015_FORMAL_REPLACEMENT_SEED_DERIVATION_ID,
    R015_FORMAL_ROUND_SAMPLING_CONTRACT,
    R015_FORMAL_SAMPLING_SCHEDULE_SCHEMA,
)


MACHINE_ARTIFACT_BUNDLE_SCHEMA = "path_c_r015_machine_artifact_bundle_v1"
UPDATED_FREEZE_INPUT_SCHEMA = "path_c_r015_freeze_manifest_inputs_v2"
FORMAL_ROUND_COUNT = 2500
FORMAL_PROTOTYPE_COUNT = 4

ARTIFACT_SCHEMAS = {
    "decision_evidence_verifier": "path_c_r015_decision_evidence_verification_v1",
    "safety_branch_evidence_verifier": (
        "path_c_r015_safety_branch_verification_v2"
    ),
    "trace_replay_verifier": "path_c_r015_trace_replay_result_v2",
}

_HEX = frozenset("0123456789abcdef")
_OUTPUT_FILENAMES = {
    "ego_evidence_contract": "ego_evidence_contract.json",
    "formal_sampling_schedule": "formal_sampling_schedule.json",
    "response_projection": "response_projection_manifest.json",
    "response_vocabulary": "response_vocabulary_manifest.json",
    "probe_registry": "probe_registry_manifest.json",
    "official_history_filter": "official_history_filter_manifest.json",
    "continuation_controller": "continuation_controller_manifest.json",
    "continuation_planner": "continuation_planner_manifest.json",
    "decision_evidence_verifier": "decision_evidence_verifier_manifest.json",
    "return_bound_derivation": "return_bound_machine_manifest.json",
    "wrong_delivery_detector": "wrong_delivery_detector_manifest.json",
    "planning_stability_report": "planning_stability_report.json",
    "random_key_derivation": "random_key_derivation_manifest.json",
    "safety_branch_evidence_verifier": "safety_verifier_manifest.json",
    "trace_replay_verifier": "trace_replay_verifier_manifest.json",
}


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_mapping(path: str | Path) -> Mapping[str, Any]:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    payload = (
        json.loads(text)
        if target.suffix.lower() == ".json"
        else yaml.safe_load(text)
    )
    if not isinstance(payload, Mapping):
        raise TypeError(f"R015 文件必须是映射：{target}")
    return payload


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping.")
    return value


def _require_sequence(value: Any, *, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field} must be a sequence.")
    return value


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _contains_pending(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_pending(child) for child in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_pending(child) for child in value)
    return isinstance(value, str) and "pending" in value.lower()


def _resolved_file(path: str | Path, *, field: str) -> Path:
    target = Path(path).expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(f"R015 {field} 文件不存在：{target}")
    return target


def _binding(path: str | Path) -> Mapping[str, str]:
    target = _resolved_file(path, field="binding")
    return {"path": str(target), "sha256": _file_sha256(target)}


def _payload_binding(
    *, output_directory: Path, name: str, payload: Mapping[str, Any]
) -> Mapping[str, str]:
    path = output_directory / _OUTPUT_FILENAMES[name]
    return {"path": str(path), "sha256": _bytes_sha256(_json_bytes(payload))}


def _source_paths(
    repository_root: Path,
    freeze_inputs: Mapping[str, Any],
) -> Mapping[str, Path]:
    """解析真实实现路径；调用者只能覆盖已经登记的路径名称。"""

    defaults: dict[str, Path] = {
        "environment_config": repository_root
        / "experiments/overcooked_v2/configs/path_c_official_sp_simple_seed100.yaml",
        "response_summary_implementation": repository_root
        / "experiments/overcooked_v2/path_c_response_summary.py",
        "response_vocabulary_registration": repository_root
        / "experiments/overcooked_v2/path_c_response_probe.py",
        "controller_implementation": repository_root
        / "experiments/overcooked_v2/path_c_r015_controller.py",
        "filter_implementation": repository_root
        / "experiments/overcooked_v2/official/r015_runtime_bridge.py",
        "full_horizon_implementation": repository_root
        / "experiments/overcooked_v2/path_c_r015_full_horizon.py",
        "replay_verifier_entrypoints": repository_root
        / "experiments/overcooked_v2/path_c_r015_runtime.py",
        "verifier_implementation": repository_root
        / "experiments/overcooked_v2/path_c_r015.py",
        "audit_spec": repository_root
        / "idea-stage/refine-logs/PATH_C_OPPORTUNITY_AUDIT_SPEC.md",
        "return_bound_proof": repository_root
        / "idea-stage/refine-logs/R015_RETURN_BOUND_DERIVATION.md",
        "pilot_protocol": repository_root
        / "experiments/overcooked_v2/configs/path_c_r015_pilot.yaml",
        "machine_artifact_builder": repository_root
        / "experiments/overcooked_v2/path_c_r015_artifacts.py",
        "machine_artifact_builder_script": repository_root
        / "experiments/overcooked_v2/scripts/build_r015_machine_artifacts.py",
    }
    overrides = freeze_inputs.get("machine_artifact_source_paths", {})
    if not isinstance(overrides, Mapping):
        raise TypeError("machine_artifact_source_paths must be a mapping.")
    unknown = set(overrides).difference(defaults)
    if unknown:
        raise ValueError(
            "R015 machine artifact source override has unknown field(s): "
            + ", ".join(sorted(str(value) for value in unknown))
        )
    defaults.update({str(name): Path(str(path)) for name, path in overrides.items()})
    return {
        name: _resolved_file(path, field=f"machine_artifact_source_paths.{name}")
        for name, path in defaults.items()
    }


def _validate_replay_backend_contract(sources: Mapping[str, Path]) -> None:
    """拒绝把只返回通用摘要的重放后端登记成正式核验器。"""

    entrypoint_source = sources["replay_verifier_entrypoints"].read_text(
        encoding="utf-8"
    )
    for callable_name in (
        "verify_r015_probe_decision",
        "verify_r015_safety_comparison",
        "verify_r015_trace_manifest",
    ):
        if f"def {callable_name}(" not in entrypoint_source:
            raise ValueError(
                f"R015 replay verifier entrypoint is missing: {callable_name}"
            )
    backend_source = sources["full_horizon_implementation"].read_text(
        encoding="utf-8"
    )
    required_backend_markers = {
        *(ARTIFACT_SCHEMAS.values()),
        "def replay_probe_decision(",
        "def replay_safety_comparison(",
        "def replay_trace(",
    }
    missing = sorted(
        marker for marker in required_backend_markers if marker not in backend_source
    )
    if missing:
        raise ValueError(
            "R015 replay backend cannot be frozen before it emits the exact formal "
            "verification schemas: "
            + ", ".join(missing)
        )


def _declared_path(owner: Path, value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"R015 {field} path is missing.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = owner.parent / path
    return path.resolve()


def _validate_resolved_pilot_protocol(
    *,
    protocol_path: Path,
    design_selection_report_path: Path,
    support_registration_path: Path,
) -> Mapping[str, Any]:
    """冻结时只接受已经填入本次选择和输出位置的试点协议。"""

    protocol = _load_mapping(protocol_path)
    if protocol.get("schema_version") != "path_c_r015_pilot_protocol_v1" or (
        protocol.get("status") != "resolved_for_authorized_pilot"
    ) or protocol.get("scientific_readout_allowed") is not False:
        raise ValueError("R015 pilot protocol was not resolved before freeze.")
    if _contains_pending(protocol):
        raise ValueError("R015 resolved pilot protocol retains a pending value.")
    frozen_inputs = _require_mapping(
        protocol.get("frozen_inputs"), field="pilot_protocol.frozen_inputs"
    )
    if set(frozen_inputs) != {
        "preregistration",
        "design_selection_report",
        "signed_freeze_manifest",
        "support_registration",
    }:
        raise ValueError("R015 resolved pilot protocol has the wrong frozen inputs.")
    if _declared_path(
        protocol_path,
        frozen_inputs["design_selection_report"],
        field="pilot design selection",
    ) != design_selection_report_path.resolve():
        raise ValueError("R015 pilot protocol binds another design selection.")
    if _declared_path(
        protocol_path,
        frozen_inputs["support_registration"],
        field="pilot support registration",
    ) != support_registration_path.resolve():
        raise ValueError("R015 pilot protocol binds another support registration.")
    preregistration_path = _declared_path(
        protocol_path,
        frozen_inputs["preregistration"],
        field="pilot preregistration",
    )
    frozen_manifest_path = _declared_path(
        protocol_path,
        frozen_inputs["signed_freeze_manifest"],
        field="pilot signed freeze manifest",
    )
    if preregistration_path.name != "prefilled_preregistration.yaml" or (
        frozen_manifest_path.name != "pilot_frozen_manifest.json"
    ):
        raise ValueError("R015 pilot protocol points to the wrong freeze outputs.")
    outputs = _require_mapping(
        protocol.get("outputs"), field="pilot_protocol.outputs"
    )
    report_path = _declared_path(
        protocol_path, outputs.get("report_path"), field="pilot report"
    )
    if report_path.name != "pilot_wiring_report.json":
        raise ValueError("R015 pilot protocol has the wrong report output.")
    execution = _require_mapping(
        protocol.get("execution_boundary"), field="pilot_protocol.execution_boundary"
    )
    if execution.get("current_execution_authorized") is not False or (
        execution.get("formal_audit_included") is not False
    ):
        raise ValueError("R015 pilot protocol changed its execution boundary.")
    return protocol


def _selected_design(
    freeze_inputs: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    report_path = _resolved_file(
        freeze_inputs.get("design_selection_report_path", ""),
        field="design_selection_report_path",
    )
    report = _load_mapping(report_path)
    if report.get("schema_version") != "path_c_r015_design_selection_report_v2" or (
        report.get("status") != "selected"
    ):
        raise ValueError("R015 machine artifacts require a selected v2 design report.")
    filter_selection = _require_mapping(
        report.get("filter_selection"), field="filter_selection"
    )
    planning_selection = _require_mapping(
        report.get("planning_selection"), field="planning_selection"
    )
    selected_filter = _require_mapping(
        filter_selection.get("selected_candidate"), field="selected_candidate"
    )
    if filter_selection.get("status") != "selected" or planning_selection.get(
        "status"
    ) != "selected":
        raise ValueError("R015 design report does not contain both mechanical selections.")
    if selected_filter.get("filter_algorithm_id") != R015_FILTER_ALGORITHM_ID_V2 or (
        selected_filter.get("resampling_algorithm")
        != R015_RESAMPLING_ALGORITHM_ID_V2
    ):
        raise ValueError("R015 machine artifacts refuse the retired filter contract.")
    return report, selected_filter, planning_selection


def _support_members(
    freeze_inputs: Mapping[str, Any],
    prototype_ids: Sequence[str],
) -> Mapping[str, Mapping[str, Any]]:
    report = _load_mapping(
        _resolved_file(
            freeze_inputs.get("support_report_path", ""), field="support_report_path"
        )
    )
    if report.get("support_complete") is not True:
        raise ValueError("R015 partner support is not complete.")
    raw_members = report.get("members", report.get("candidates"))
    members = _require_mapping(raw_members, field="support report members")
    if set(members) != set(prototype_ids):
        raise ValueError("R015 support report differs from the four continuation prototypes.")
    return {
        prototype_id: _require_mapping(
            members[prototype_id], field=f"support member {prototype_id}"
        )
        for prototype_id in prototype_ids
    }


def _checkpoint_by_seed(
    freeze_inputs: Mapping[str, Any],
) -> Mapping[int, Mapping[str, Any]]:
    checkpoints = _require_sequence(freeze_inputs.get("checkpoints"), field="checkpoints")
    by_seed: dict[int, Mapping[str, Any]] = {}
    for raw in checkpoints:
        checkpoint = _require_mapping(raw, field="checkpoint")
        required = {
            "artifact_id",
            "role",
            "training_seed",
            "path",
            "format",
            "parameter_tree_path",
            "training_manifest_path",
        }
        if set(checkpoint) != required:
            raise ValueError("R015 checkpoint input has the wrong fields.")
        seed = int(checkpoint.get("training_seed"))
        if seed in by_seed:
            raise ValueError("R015 checkpoint seed is duplicated.")
        expected_role = "ego" if seed == 100 else "partner"
        if checkpoint.get("role") != expected_role:
            raise ValueError("R015 checkpoint role differs from the signed set.")
        checkpoint_path = Path(str(checkpoint["path"])).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"R015 checkpoint does not exist: {checkpoint_path}")
        manifest_path = _resolved_file(
            checkpoint["training_manifest_path"],
            field=f"checkpoint {seed} training_manifest_path",
        )
        manifest = _load_mapping(manifest_path)
        if manifest.get("schema_version") != "path_c_official_training_artifact_v2":
            raise ValueError("R015 checkpoint requires an official v2 training manifest.")
        manifest_checkpoint = _require_mapping(
            manifest.get("checkpoint"), field=f"checkpoint {seed} manifest checkpoint"
        )
        training_config_path = _resolved_file(
            manifest.get("training_config_path", ""),
            field=f"checkpoint {seed} training_config_path",
        )
        parameter_tree_path = tuple(
            str(value)
            for value in _require_sequence(
                checkpoint["parameter_tree_path"],
                field=f"checkpoint {seed} parameter_tree_path",
            )
        )
        params = load_flax_parameter_tree(
            checkpoint_path,
            checkpoint_format=str(checkpoint["format"]),
            parameter_tree_path=parameter_tree_path,
        )
        verified = validate_official_artifact_manifest(
            manifest.get("training_config_path"),
            manifest_path,
            expected_checkpoint_path=checkpoint_path,
            params=params,
        )
        if not isinstance(verified, Mapping):
            raise TypeError("R015 official artifact verification returned another schema.")
        checkpoint_sha256 = checkpoint_artifact_sha256(checkpoint_path)
        model_weights_sha256 = flax_weights_sha256(params)
        if Path(str(manifest_checkpoint.get("path"))).resolve() != checkpoint_path:
            raise ValueError("R015 training manifest binds another checkpoint path.")
        if manifest_checkpoint.get("format") != checkpoint["format"] or tuple(
            manifest_checkpoint.get("parameter_tree_path", ())
        ) != parameter_tree_path:
            raise ValueError("R015 training manifest checkpoint interface changed.")
        if manifest_checkpoint.get("checkpoint_sha256") != checkpoint_sha256 or (
            manifest_checkpoint.get("model_weights_sha256") != model_weights_sha256
        ) or manifest_checkpoint.get("model_weights_hash_domain") != (
            FLAX_WEIGHTS_HASH_DOMAIN
        ):
            raise ValueError("R015 training manifest checkpoint hashes do not match.")
        if manifest.get("seed") != seed or not _is_sha256(
            manifest.get("training_run_id")
        ):
            raise ValueError("R015 training manifest run identity does not match.")
        by_seed[seed] = {
            **copy.deepcopy(dict(checkpoint)),
            "family_id": str(verified["family_id"]),
            "effective_environment_steps": int(
                verified["snapshot_environment_steps"]
            ),
            "checkpoint_sha256": checkpoint_sha256,
            "model_weights_hash_domain": FLAX_WEIGHTS_HASH_DOMAIN,
            "model_weights_sha256": model_weights_sha256,
            "training_manifest_sha256": _file_sha256(manifest_path),
            "training_config_path": str(training_config_path),
            "training_config_sha256": _file_sha256(training_config_path),
            "training_run_id": str(manifest["training_run_id"]),
        }
    if set(by_seed) != {100, 101, 102, 201, 202}:
        raise ValueError("R015 machine artifacts require the signed five checkpoints.")
    return by_seed


def _probe_registry(raw_preregistration: Mapping[str, Any]) -> Mapping[str, Any]:
    experiment = _require_mapping(
        raw_preregistration.get("experiment"), field="experiment"
    )
    scripts = [
        {
            "probe_id": str(_require_mapping(value, field="probe script")["probe_id"]),
            "primitive_actions": list(
                _require_sequence(
                    _require_mapping(value, field="probe script").get(
                        "primitive_actions"
                    ),
                    field="probe primitive_actions",
                )
            ),
        }
        for value in _require_sequence(
            experiment.get("registered_probe_scripts"),
            field="registered_probe_scripts",
        )
    ]
    semantics = {
        "schema_version": "path_c_r015_probe_registry_semantics_v1",
        "probe_scripts": sorted(scripts, key=lambda value: value["probe_id"]),
    }
    return {
        "schema_version": "path_c_r015_probe_registry_v1",
        "probe_scripts": scripts,
        "maximum_script_length": 2,
        "observed_maximum_script_length": max(
            len(value["primitive_actions"]) for value in scripts
        ),
        "semantic_sha256": _canonical_sha256(semantics),
    }


def _validate_formal_sampling_schedule(
    schedule: Mapping[str, Any],
    *,
    prototype_ids: Sequence[str],
) -> None:
    """确认正式清单恰含四原型乘 2,500 轮，且每个坐标唯一。"""

    prototypes = tuple(str(value) for value in prototype_ids)
    formal_root_seed = schedule.get("formal_root_seed")
    if schedule.get("schema_version") != R015_FORMAL_SAMPLING_SCHEDULE_SCHEMA or (
        schedule.get("experiment_id") != "R015"
    ) or schedule.get("frozen_before_formal_data") is not True or (
        schedule.get("selection_uses_outcomes") is not False
    ) or (
        schedule.get("n_rounds") != FORMAL_ROUND_COUNT
    ) or schedule.get("round_sampling_contract") != (
        R015_FORMAL_ROUND_SAMPLING_CONTRACT
    ) or schedule.get("mechanical_replacement_attempts_per_coordinate") != (
        R015_FORMAL_REPLACEMENT_ATTEMPT_COUNT
    ) or schedule.get("replacement_seed_selection_uses_outcomes") is not False or (
        schedule.get("mechanical_replacement_attempt_indices") != "0_through_31"
    ) or schedule.get("replacement_episode_seed_derivation_id") != (
        R015_FORMAL_REPLACEMENT_SEED_DERIVATION_ID
    ) or schedule.get("replacement_episode_seed_collision_policy") != (
        "allow_value_collisions_without_redraw"
    ) or (
        schedule.get("audit_unit_changes_across_replacement_attempts") is not False
    ) or schedule.get("mechanical_attempt_exhaustion_rule") != (
        "terminate_entire_formal_audit"
    ) or schedule.get("round_vectors_generated_independently") is not True or (
        schedule.get("ego_agent_id") != "agent_1"
    ) or schedule.get("partner_agent_id") != "agent_0" or (
        schedule.get("firing_prefixes_independent_across_prototypes") is not True
    ) or schedule.get("cross_prototype_shared_randomness_before_firing") is not False or (
        isinstance(formal_root_seed, bool)
        or not isinstance(formal_root_seed, int)
        or formal_root_seed < 0
    )
    ):
        raise ValueError("R015 formal sampling schedule has the wrong contract.")
    rows = _require_sequence(schedule.get("entries"), field="formal schedule entries")
    expected_coordinates = {
        (round_index, prototype_id)
        for round_index in range(1, FORMAL_ROUND_COUNT + 1)
        for prototype_id in prototypes
    }
    observed_coordinates: set[tuple[int, str]] = set()
    audit_unit_ids: set[str] = set()
    for raw_row in rows:
        row = _require_mapping(raw_row, field="formal sampling row")
        if set(row) != {
            "round_index",
            "audit_unit_id",
            "prototype_id",
            "episode_seeds_by_attempt_index",
            "ego_position",
        }:
            raise ValueError("R015 formal sampling row has the wrong fields.")
        coordinate = (int(row["round_index"]), str(row["prototype_id"]))
        audit_unit_id = str(row["audit_unit_id"])
        raw_episode_seeds = _require_sequence(
            row.get("episode_seeds_by_attempt_index"),
            field="formal sampling replacement episode seeds",
        )
        if len(raw_episode_seeds) != R015_FORMAL_REPLACEMENT_ATTEMPT_COUNT:
            raise ValueError("R015 formal sampling row has the wrong replacement count.")
        row_episode_seeds = tuple(int(value) for value in raw_episode_seeds)
        if any(value < 0 for value in row_episode_seeds):
            raise ValueError("R015 formal sampling row has invalid replacement seeds.")
        if row.get("ego_position") != 1 or not _is_sha256(audit_unit_id):
            raise ValueError("R015 formal sampling row changed its cook seat or identity.")
        prototype_id = str(row["prototype_id"])
        if prototype_id not in prototypes:
            raise ValueError("R015 formal sampling row uses an unknown prototype.")
        prototype_index = prototypes.index(prototype_id)
        expected_audit_unit_id = _canonical_sha256(
            [
                "r015_formal_audit_unit_v4",
                "r015_formal_sampling_schedule_v4",
                formal_root_seed,
                int(row["round_index"]),
                prototype_index,
                prototype_id,
            ]
        )
        if audit_unit_id != expected_audit_unit_id:
            raise ValueError(
                "R015 formal sampling audit unit was not derived from its coordinates."
            )
        expected_episode_seeds = tuple(
            int(
                _canonical_sha256(
                    [
                        "r015_formal_prefrozen_replacement_seed_v4",
                        formal_root_seed,
                        int(row["round_index"]),
                        prototype_index,
                        prototype_id,
                        attempt_index,
                    ]
                )[-8:],
                16,
            )
            for attempt_index in range(R015_FORMAL_REPLACEMENT_ATTEMPT_COUNT)
        )
        if row_episode_seeds != expected_episode_seeds:
            raise ValueError("R015 formal sampling row was not derived from its coordinates.")
        if coordinate in observed_coordinates or audit_unit_id in audit_unit_ids:
            raise ValueError("R015 formal sampling schedule contains a duplicate coordinate.")
        observed_coordinates.add(coordinate)
        audit_unit_ids.add(audit_unit_id)
    if observed_coordinates != expected_coordinates:
        raise ValueError("R015 formal sampling schedule is not four prototypes by 2,500.")


def _atomic_write_bundle(
    output_directory: Path,
    files: Mapping[str, bytes],
) -> None:
    """同一文件系统内一次发布整个目录；重入只接受逐字节相同内容。"""

    target = output_directory.expanduser().resolve()
    if target.exists():
        if not target.is_dir():
            raise FileExistsError(f"R015 artifact target is not a directory: {target}")
        observed = {
            str(path.relative_to(target)): path.read_bytes()
            for path in target.rglob("*")
            if path.is_file()
        }
        if observed != dict(files):
            raise FileExistsError(
                "R015 artifact directory exists with different content; refusing overwrite."
            )
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.tmp-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    try:
        for relative_name, payload in sorted(files.items()):
            destination = staging / relative_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        staging.replace(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_r015_machine_artifact_bundle(
    *,
    freeze_inputs: Mapping[str, Any],
    preregistration_path: str | Path,
    output_directory: str | Path,
    repository_root: str | Path,
    formal_root_seed: int,
) -> Mapping[str, Any]:
    """生成冻结清单需要的全部机器制品，并返回更新后的冻结输入。

    `formal_root_seed` 是正式数据前必须给出的非负整数。函数固定生成 2,500 轮、
    每轮四原型的 10,000 个坐标，并直接调用正式运行模块中的登记日程函数。
    """

    if freeze_inputs.get("schema_version") != UPDATED_FREEZE_INPUT_SCHEMA:
        raise ValueError("R015 freeze inputs have the wrong schema.")
    if isinstance(formal_root_seed, bool) or not isinstance(formal_root_seed, int) or (
        formal_root_seed < 0
    ):
        raise ValueError("R015 formal_root_seed must be a non-negative integer.")
    root = Path(repository_root).expanduser().resolve()
    preregistration_file = _resolved_file(
        preregistration_path, field="preregistration_path"
    )
    preregistration = _load_mapping(preregistration_file)
    if preregistration.get("schema_version") != "path_c_r015_preregistration_v2" or (
        preregistration.get("scientific_readout_allowed") is not False
    ):
        raise ValueError("R015 machine artifacts require the unfrozen v2 preregistration.")
    experiment = _require_mapping(
        preregistration.get("experiment"), field="experiment"
    )
    if experiment.get("freeze_status") != "static_registered_not_frozen":
        raise ValueError("R015 preregistration has already changed its static freeze state.")
    controller = _require_mapping(
        preregistration.get("controller"), field="controller"
    )
    filtering = _require_mapping(
        controller.get("hidden_state_filter"), field="hidden_state_filter"
    )
    continuation = _require_mapping(
        controller.get("continuation"), field="continuation"
    )
    information = _require_mapping(
        preregistration.get("information"), field="information"
    )
    statistics = _require_mapping(
        preregistration.get("statistics"), field="statistics"
    )
    report, selected_filter, planning_selection = _selected_design(freeze_inputs)
    selected_branch_count = int(planning_selection["selected_branch_count"])
    prototype_ids = tuple(
        str(value)
        for value in _require_sequence(
            continuation.get("prototype_member_ids"),
            field="continuation.prototype_member_ids",
        )
    )
    if len(prototype_ids) != FORMAL_PROTOTYPE_COUNT or len(set(prototype_ids)) != 4:
        raise ValueError("R015 continuation library must contain four prototypes.")
    support_members = _support_members(freeze_inputs, prototype_ids)
    checkpoints = _checkpoint_by_seed(freeze_inputs)
    sources = _source_paths(root, freeze_inputs)
    _validate_replay_backend_contract(sources)
    design_selection_report_path = _resolved_file(
        freeze_inputs["design_selection_report_path"],
        field="design_selection_report_path",
    )
    support_registration_path = _resolved_file(
        freeze_inputs.get("support_registration_path", ""),
        field="support_registration_path",
    )
    _validate_resolved_pilot_protocol(
        protocol_path=sources["pilot_protocol"],
        design_selection_report_path=design_selection_report_path,
        support_registration_path=support_registration_path,
    )
    output = Path(output_directory).expanduser().resolve()

    environment_sources_raw = _require_mapping(
        freeze_inputs.get("environment_sources"), field="environment_sources"
    )
    if set(environment_sources_raw) != {"settings", "overcooked", "layouts", "common"}:
        raise ValueError("R015 environment source closure is incomplete.")
    environment_sources: dict[str, Mapping[str, str]] = {}
    for name, raw_binding in environment_sources_raw.items():
        if isinstance(raw_binding, Mapping):
            if set(raw_binding) != {"path", "sha256"}:
                raise ValueError(
                    f"R015 environment_sources.{name} binding has wrong fields."
                )
            current = _binding(raw_binding.get("path", ""))
            if raw_binding.get("sha256") != current["sha256"]:
                raise ValueError(f"R015 environment source changed: {name}")
        else:
            current = _binding(raw_binding)
        environment_sources[str(name)] = current
    environment_config_binding = _binding(sources["environment_config"])
    if environment_config_binding["sha256"] != checkpoints[100].get(
        "training_config_sha256"
    ):
        raise ValueError("R015 environment config differs from the ego training config.")
    environment_source_binding = environment_sources["overcooked"]
    delivery_reward_readback = parse_delivery_reward(
        environment_sources["settings"]["path"]
    )
    if float(delivery_reward_readback["value"]) != float(
        statistics["delivery_reward"]
    ):
        raise ValueError("R015 delivery reward differs from the environment source.")

    response_summary = _require_mapping(
        controller.get("response_summary"), field="response_summary"
    )
    response_spec = ResponseSummarySpecV1(
        response_classes=tuple(
            str(value)
            for value in _require_sequence(
                response_summary.get("response_classes"), field="response_classes"
            )
        ),
        latency_bin_upper_bounds=tuple(
            int(value)
            for value in _require_sequence(
                response_summary.get("latency_bin_upper_bounds"),
                field="latency_bin_upper_bounds",
            )
        ),
    )
    response_vocabulary = {
        "schema_version": "path_c_r015_response_vocabulary_v1",
        "response_summary_schema_version": response_spec.schema_version,
        "response_classes": list(response_spec.response_classes),
        "latency_bin_upper_bounds": list(response_spec.latency_bin_upper_bounds),
        "vocabulary": list(response_spec.vocabulary),
        "token_ids": response_spec.token_ids,
        "response_summary_sha256": response_spec.sha256,
        "implementation_sources": [
            _binding(sources["response_summary_implementation"]),
            _binding(sources["response_vocabulary_registration"]),
        ],
    }
    probe_registry = _probe_registry(preregistration)
    ego_evidence_contract = {
        "schema_version": "path_c_r015_ego_evidence_contract_v1",
        "controller_input": information.get("controller_input"),
        "allowed_controller_fields": list(
            _require_sequence(
                information.get("allowed_controller_fields"),
                field="allowed_controller_fields",
            )
        ),
        "forbidden_controller_fields": list(
            _require_sequence(
                information.get("forbidden_controller_fields"),
                field="forbidden_controller_fields",
            )
        ),
        "implementation_path": str(sources["controller_implementation"]),
        "implementation_sha256": _file_sha256(sources["controller_implementation"]),
    }

    projection_config = _require_mapping(
        information.get("current_probe_response_projection"),
        field="current_probe_response_projection",
    )
    projection_graph = {
        "common_history": list(projection_config.get("preserved_fields", ())),
        "current_response": "B_use_only",
        "masked_routes": list(projection_config.get("mask_targets", ())),
    }
    alias_clearlist = ["y", "current_response", "current_response_token"]
    recurrent_rule = {
        "masked_current_response_written": False,
        "parallel_member_states_advance_from_common_history": True,
    }
    projection_hash_fields = {
        "implementation_sha256": _file_sha256(sources["controller_implementation"]),
        "derived_field_dependency_graph_sha256": _canonical_sha256(projection_graph),
        "explicit_alias_clearlist_sha256": _canonical_sha256(alias_clearlist),
        "recurrent_state_write_rule_sha256": _canonical_sha256(recurrent_rule),
    }
    projection_semantics = {
        "name": projection_config.get("name"),
        "mask_targets": sorted(str(value) for value in projection_config["mask_targets"]),
        "preserved_fields": sorted(
            str(value) for value in projection_config["preserved_fields"]
        ),
        "use_only_route": projection_config.get("use_only_route"),
        **projection_hash_fields,
    }
    response_projection = {
        "schema_version": "path_c_r015_response_projection_v1",
        "name": projection_config.get("name"),
        "mask_targets": list(projection_config["mask_targets"]),
        "preserved_fields": list(projection_config["preserved_fields"]),
        "use_only_route": projection_config.get("use_only_route"),
        "implementation_path": str(sources["controller_implementation"]),
        "derived_field_dependency_graph": projection_graph,
        "explicit_alias_clearlist": alias_clearlist,
        "recurrent_state_write_rule": recurrent_rule,
        **projection_hash_fields,
        "semantic_sha256": _canonical_sha256(projection_semantics),
    }

    filter_parameters = {
        "prototype_prior": dict(filtering["prototype_prior"]),
        "particles_per_prototype": int(selected_filter["particles_per_prototype"]),
        "filter_mode": filtering.get("mode"),
        "initialization_rule": filtering.get("initialization_rule"),
        "initialization_key_source": filtering.get("initialization_key_source"),
        "transition_update_rule": filtering.get("transition_update_rule"),
        "ess_rule": filtering.get("ess_rule"),
        "zero_support_condition": filtering.get("zero_support_condition"),
        "resampling_algorithm": selected_filter.get("resampling_algorithm"),
        "resampling_interval_environment_steps": selected_filter.get(
            "resampling_interval_environment_steps"
        ),
        "resampling_timing": selected_filter.get("resampling_timing"),
        "resampling_ess_fraction_threshold": selected_filter.get(
            "resampling_ess_fraction_threshold"
        ),
        "zero_support_action": filtering.get("zero_support_action"),
        "device_execution_id": filtering.get("device_execution_id"),
        "device_key_contract": filtering.get("device_key_contract"),
        "microbatch_schedule_id": filtering.get("microbatch_schedule_id"),
        "parent_particle_slot_target_per_microbatch": filtering.get(
            "parent_particle_slot_target_per_microbatch"
        ),
        "group_partner_network_by_prototype": filtering.get(
            "group_partner_network_by_prototype"
        ),
        "continuation_member_forwards_in_filter_candidate_stage": filtering.get(
            "continuation_member_forwards_in_filter_candidate_stage"
        ),
    }
    response_vocabulary_binding = _payload_binding(
        output_directory=output,
        name="response_vocabulary",
        payload=response_vocabulary,
    )
    official_history_filter = {
        "schema_version": "path_c_r015_official_history_filter_v2",
        "controller_input": "official_local_history_only",
        "allowed_controller_fields": list(information["allowed_controller_fields"]),
        "forbidden_controller_fields": list(
            information["forbidden_controller_fields"]
        ),
        "full_state_fields_exposed": False,
        "partner_identity_exposed": False,
        "filter_parameters": filter_parameters,
        "fully_adapted_contract": {
            "opening_conditioned_on_official_observation": True,
            "partner_action_count_marginalized": 6,
            "environment_outcomes_affecting_official_match_marginalized": True,
            "conditional_successor_sampled_after_marginal_likelihood": True,
            "single_action_hard_rejection_allowed": False,
            "effective_sample_size_uses_pre_resample_parent_weights": True,
        },
        "response_vocabulary_sha256": response_vocabulary_binding["sha256"],
        "implementation_path": str(sources["filter_implementation"]),
        "implementation_sha256": _file_sha256(sources["filter_implementation"]),
    }

    continuation_members: dict[str, Mapping[str, Any]] = {}
    baseline_id = str(continuation["baseline_member_id"])
    if baseline_id != checkpoints[100]["artifact_id"]:
        raise ValueError("R015 baseline member differs from the seed-100 artifact.")
    continuation_members[baseline_id] = {
        "role": "baseline",
        "checkpoint_sha256": checkpoints[100]["checkpoint_sha256"],
        "model_weights_sha256": checkpoints[100]["model_weights_sha256"],
    }
    for prototype_id in prototype_ids:
        member = support_members[prototype_id]
        training_seed = int(member.get("training_seed", -1))
        if training_seed not in checkpoints or checkpoints[training_seed].get(
            "artifact_id"
        ) != prototype_id or any(
            member.get(field) != checkpoints[training_seed].get(field)
            for field in (
                "family_id",
                "checkpoint_sha256",
                "model_weights_sha256",
                "training_run_id",
            )
        ) or int(member.get("snapshot_environment_steps", -1)) != int(
            checkpoints[training_seed]["effective_environment_steps"]
        ):
            raise ValueError(
                f"R015 support member differs from checkpoint input: {prototype_id}"
            )
        continuation_members[prototype_id] = {
            "role": "prototype",
            "checkpoint_sha256": member["checkpoint_sha256"],
            "model_weights_sha256": member["model_weights_sha256"],
        }
    continuation_controller = continuation_controller_manifest_payload(
        prototype_ids=prototype_ids,
        baseline_member_id=baseline_id,
        checkpoint_bindings={
            member_id: {
                "checkpoint_sha256": str(member["checkpoint_sha256"]),
                "model_weights_sha256": str(member["model_weights_sha256"]),
            }
            for member_id, member in continuation_members.items()
        },
        implementation_path=str(sources["controller_implementation"]),
        implementation_sha256=_file_sha256(sources["controller_implementation"]),
    )

    random_key_derivation = {
        "schema_version": "path_c_r015_random_key_derivation_v1",
        "coordinates": [
            "audit_unit_id",
            "partner_prototype_id",
            "episode_seed",
            "purpose",
            "environment_step",
            "branch_index",
        ],
        "purpose_names": [
            "actual_episode",
            "value_planning",
            "score_estimation",
            "safety_validation",
        ],
        "derivation_algorithm": "sha256_canonical_json_v1",
        "key_payload_schema_version": "path_c_r015_random_key_v1",
        "independent_purpose_namespaces": True,
        "branch_index_changes_key": True,
        "implementation_path": str(sources["verifier_implementation"]),
        "implementation_sha256": _file_sha256(sources["verifier_implementation"]),
    }
    wrong_delivery_detector = {
        "schema_version": "path_c_r015_wrong_delivery_detector_v1",
        "event_name": "wrong_delivery",
        "callable_name": "_wrong_delivery_event",
        "recipe_indicator_cost_is_safety_event": False,
        "recipe_indicator_cost_in_raw_return": True,
        "environment_source_sha256": environment_source_binding["sha256"],
        "environment_source_closure": copy.deepcopy(environment_sources),
        "implementation_path": str(sources["full_horizon_implementation"]),
        "implementation_sha256": _file_sha256(
            sources["full_horizon_implementation"]
        ),
    }
    return_bound = {
        "schema_version": "path_c_r015_return_bound_derivation_v2",
        "proved": True,
        "reward_accounting": "undiscounted_raw_team_reward",
        "horizon_environment_steps": 400,
        "delivery_reward": float(statistics["delivery_reward"]),
        "delivery_reward_source_sha256": delivery_reward_readback["source_sha256"],
        "return_lower_bound": float(statistics["return_lower_bound"]),
        "return_upper_bound": float(statistics["return_upper_bound"]),
        "paired_difference_absolute_bound": float(
            statistics["paired_difference_absolute_bound"]
        ),
        "paired_difference_width": float(statistics["paired_difference_width"]),
        "lockstep_invariant_required": True,
        "arbitrary_suffix_carry_in_excluded": False,
        "minimum_probe_step": 1,
        "maximum_probe_step": 100,
        "environment_config_sha256": environment_config_binding["sha256"],
        "environment_source_sha256": environment_source_binding["sha256"],
        "environment_source_closure": copy.deepcopy(environment_sources),
        "implementation_path": str(sources["return_bound_proof"]),
        "implementation_sha256": _file_sha256(sources["return_bound_proof"]),
    }

    from experiments.overcooked_v2.path_c_r015_runtime import (
        build_r015_formal_sampling_schedule,
    )

    formal_sampling_schedule = {
        **build_r015_formal_sampling_schedule(
            prototype_ids=prototype_ids,
            root_seed=formal_root_seed,
            n_rounds=FORMAL_ROUND_COUNT,
        ),
        "formal_root_seed": formal_root_seed,
    }
    _validate_formal_sampling_schedule(
        formal_sampling_schedule,
        prototype_ids=prototype_ids,
    )

    payloads: dict[str, Mapping[str, Any]] = {
        "ego_evidence_contract": ego_evidence_contract,
        "formal_sampling_schedule": formal_sampling_schedule,
        "response_projection": response_projection,
        "response_vocabulary": response_vocabulary,
        "probe_registry": probe_registry,
        "official_history_filter": official_history_filter,
        "continuation_controller": continuation_controller,
        "return_bound_derivation": return_bound,
        "wrong_delivery_detector": wrong_delivery_detector,
        "random_key_derivation": random_key_derivation,
    }
    bindings = {
        name: _payload_binding(output_directory=output, name=name, payload=payload)
        for name, payload in payloads.items()
    }
    continuation_planner = sequential_controller_manifest_payload(
        implementation_path=str(sources["controller_implementation"]),
        implementation_sha256=_file_sha256(sources["controller_implementation"]),
        response_projection_sha256=bindings["response_projection"]["sha256"],
        official_history_filter_sha256=bindings["official_history_filter"]["sha256"],
        probe_registry_semantic_sha256=str(probe_registry["semantic_sha256"]),
        random_key_derivation_sha256=bindings["random_key_derivation"]["sha256"],
        continuation_controller_sha256=bindings["continuation_controller"]["sha256"],
        replay_backend_implementation_path=str(sources["full_horizon_implementation"]),
        replay_backend_implementation_sha256=_file_sha256(
            sources["full_horizon_implementation"]
        ),
    )
    continuation_planner = {
        **continuation_planner,
        "branches_per_candidate": selected_branch_count,
        "grid_evaluation_rule_id": "lazy_ascending_first_pass",
        "device_execution_contract": {
            "branch_head_belief_updates_once": True,
            "remaining_horizon_uses_compiled_scan": True,
            "future_random_keys_stored_stepwise": False,
        },
    }
    payloads["continuation_planner"] = continuation_planner
    bindings["continuation_planner"] = _payload_binding(
        output_directory=output,
        name="continuation_planner",
        payload=continuation_planner,
    )

    support_registration_binding = _binding(support_registration_path)
    support_report_binding = _binding(freeze_inputs.get("support_report_path", ""))
    verifier_common = {
        "environment_config_sha256": environment_config_binding["sha256"],
        "environment_source_sha256": environment_source_binding["sha256"],
        "official_history_filter_sha256": bindings["official_history_filter"][
            "sha256"
        ],
        "response_projection_sha256": bindings["response_projection"]["sha256"],
        "ego_evidence_contract_sha256": bindings["ego_evidence_contract"]["sha256"],
        "random_key_derivation_sha256": bindings["random_key_derivation"]["sha256"],
        "probe_registry_sha256": bindings["probe_registry"]["sha256"],
        "probe_registry_semantic_sha256": probe_registry["semantic_sha256"],
        "support_registration_sha256": support_registration_binding["sha256"],
        "support_report_sha256": support_report_binding["sha256"],
        "implementation_path": str(sources["replay_verifier_entrypoints"]),
        "implementation_sha256": _file_sha256(
            sources["replay_verifier_entrypoints"]
        ),
        "replay_backend_implementation_path": str(
            sources["full_horizon_implementation"]
        ),
        "replay_backend_implementation_sha256": _file_sha256(
            sources["full_horizon_implementation"]
        ),
    }
    decision_verifier = {
        "schema_version": "path_c_r015_decision_evidence_verifier_v1",
        "callable": "verify_r015_probe_decision",
        "required_result_schema": ARTIFACT_SCHEMAS[
            "decision_evidence_verifier"
        ],
        "contract": {
            "replay_every_consultation": True,
            "use_official_local_history_only": True,
            "recompute_complete_probe_registry": True,
            "recompute_eligibility_and_static_safety": True,
            "recompute_j_use_j_mask_v_base_v_mask": True,
            "recompute_score_and_selected_probe": True,
            "validate_first_positive_stop": True,
            "validate_registered_random_keys": True,
        },
        **verifier_common,
        "continuation_planner_sha256": bindings["continuation_planner"]["sha256"],
        "wrong_delivery_detector_sha256": bindings["wrong_delivery_detector"][
            "sha256"
        ],
    }
    safety_verifier = {
        "schema_version": "path_c_r015_safety_branch_evidence_verifier_v2",
        "callable": "verify_r015_safety_comparison",
        "required_result_schema": ARTIFACT_SCHEMAS[
            "safety_branch_evidence_verifier"
        ],
        "contract": {
            "replay_every_branch_in_frozen_environment": True,
            "recompute_wrong_delivery_per_branch": True,
            "recompute_branch_count": True,
            "validate_registered_random_keys": True,
            "validate_positive_posterior_support": True,
            "validate_compatible_hidden_state": True,
            "reconstruct_from_official_history_and_support_checkpoint": True,
            "execute_selected_registered_probe_script": True,
            "return_branch_evidence_and_result_hashes": True,
        },
        **verifier_common,
        "wrong_delivery_detector_sha256": bindings["wrong_delivery_detector"][
            "sha256"
        ],
        "response_vocabulary_sha256": bindings["response_vocabulary"]["sha256"],
    }
    trace_verifier = {
        "schema_version": "path_c_r015_trace_replay_verifier_manifest_v2",
        "callable_name": "verify_r015_trace_manifest",
        "required_result_schema": ARTIFACT_SCHEMAS[
            "trace_replay_verifier"
        ],
        "contract": {
            "full_environment_replay": True,
            "environment_steps": 400,
            "validate_actions": True,
            "validate_official_observations": True,
            "validate_projected_controller_inputs": True,
            "validate_raw_rewards": True,
            "validate_done_boundary": True,
            "replay_frozen_controller_every_step": True,
            "recompute_every_ego_action": True,
            "recompute_every_partner_action": True,
            "validate_recurrent_state_writes": True,
            "validate_partner_recurrent_state_writes": True,
            "validate_a1_selected_action_and_registered_script": True,
            "validate_a2_pre_probe_action_identity": True,
            "validate_a2_mask_and_use_belief_routes": True,
            "validate_a2_continuation_controller_actions": True,
        },
        **verifier_common,
        "ego_checkpoint_sha256": checkpoints[100]["checkpoint_sha256"],
        "continuation_planner_sha256": bindings["continuation_planner"]["sha256"],
        "response_vocabulary_sha256": bindings["response_vocabulary"]["sha256"],
    }
    selected_planning_result = next(
        _require_mapping(value, field="planning ordered result")
        for value in _require_sequence(
            planning_selection.get("ordered_results"), field="planning ordered_results"
        )
        if int(_require_mapping(value, field="planning result")["branch_count"])
        == selected_branch_count
    )
    planning_stability = {
        "schema_version": "path_c_r015_planning_stability_report_v1",
        "branch_count": selected_branch_count,
        "doubled_branch_count": 2 * selected_branch_count,
        "action_ranking_agreement": float(
            selected_planning_result["exact_decision_agreement"]
        ),
        "independent_design_data": True,
        "formal_data_included": False,
        "passed": True,
        "design_selection_report_path": str(
            _resolved_file(
                freeze_inputs["design_selection_report_path"],
                field="design_selection_report_path",
            )
        ),
        "design_selection_report_sha256": _file_sha256(
            freeze_inputs["design_selection_report_path"]
        ),
        "continuation_planner_sha256": bindings["continuation_planner"]["sha256"],
        "probe_registry_semantic_sha256": probe_registry["semantic_sha256"],
    }
    for name, payload in (
        ("decision_evidence_verifier", decision_verifier),
        ("safety_branch_evidence_verifier", safety_verifier),
        ("trace_replay_verifier", trace_verifier),
        ("planning_stability_report", planning_stability),
    ):
        payloads[name] = payload
        bindings[name] = _payload_binding(
            output_directory=output, name=name, payload=payload
        )

    if any(_contains_pending(payload) for payload in payloads.values()):
        raise ValueError("R015 generated machine artifact contains a pending value.")
    if len(formal_sampling_schedule.get("entries", ())) != (
        FORMAL_ROUND_COUNT * FORMAL_PROTOTYPE_COUNT
    ):
        raise RuntimeError("R015 formal sampling schedule is not 4 x 2500.")

    protocol_path = _resolved_file(report["protocol_path"], field="design protocol")
    updated_inputs = copy.deepcopy(dict(freeze_inputs))
    updated_inputs.pop("machine_artifact_source_paths", None)
    updated_inputs["settings_path"] = environment_sources["settings"]["path"]
    updated_inputs["environment_sources"] = environment_sources
    updated_inputs["semantic_artifacts"] = {
        "response_vocabulary": bindings["response_vocabulary"],
        "probe_registry": bindings["probe_registry"],
        "ego_history_contract": bindings["ego_evidence_contract"],
        "response_projection": bindings["response_projection"],
        "filter_implementation": _binding(sources["filter_implementation"]),
        "planner_implementation": _binding(sources["controller_implementation"]),
        "continuation_controller": bindings["continuation_controller"],
    }
    updated_inputs["execution_artifacts"] = {
        "production_runtime_bridge": _binding(sources["filter_implementation"]),
        "full_horizon_executor": _binding(sources["full_horizon_implementation"]),
        "replay_backend": _binding(sources["full_horizon_implementation"]),
    }
    updated_inputs["preregistration_artifacts"] = {
        "environment_config": environment_config_binding,
        "environment_source": environment_source_binding,
        **bindings,
    }
    updated_inputs["protocol_artifacts"] = {
        "design_data_protocol": _binding(protocol_path),
        "pilot_protocol": _binding(sources["pilot_protocol"]),
    }
    updated_inputs["text_artifacts"] = {
        "audit_spec": _binding(sources["audit_spec"]),
        "preregistration": _binding(preregistration_file),
        "return_bound_derivation": _binding(sources["return_bound_proof"]),
    }
    if _contains_pending(updated_inputs):
        raise ValueError("R015 generated freeze inputs retain a pending value.")
    updated_inputs_binding_path = output / "freeze_manifest_inputs.generated.json"
    updated_inputs_bytes = _json_bytes(updated_inputs)

    artifact_inventory = {
        name: {
            **bindings[name],
            "schema_version": payloads[name].get("schema_version"),
        }
        for name in sorted(payloads)
    }
    bundle_manifest = {
        "schema_version": MACHINE_ARTIFACT_BUNDLE_SCHEMA,
        "scientific_readout_allowed": False,
        "formal_data_read": False,
        "formal_root_seed": formal_root_seed,
        "formal_round_count": FORMAL_ROUND_COUNT,
        "formal_prototype_count": FORMAL_PROTOTYPE_COUNT,
        "formal_sampling_entry_count": FORMAL_ROUND_COUNT * FORMAL_PROTOTYPE_COUNT,
        "artifacts": artifact_inventory,
        "implementation_sources": {
            name: _binding(path) for name, path in sorted(sources.items())
        },
        "updated_freeze_inputs": {
            "path": str(updated_inputs_binding_path),
            "sha256": _bytes_sha256(updated_inputs_bytes),
        },
    }
    if _contains_pending(bundle_manifest):
        raise ValueError("R015 machine artifact bundle contains a pending value.")
    files = {
        _OUTPUT_FILENAMES[name]: _json_bytes(payload)
        for name, payload in payloads.items()
    }
    files[updated_inputs_binding_path.name] = updated_inputs_bytes
    files["bundle_manifest.json"] = _json_bytes(bundle_manifest)
    _atomic_write_bundle(output, files)
    return {
        "bundle_manifest": bundle_manifest,
        "bundle_manifest_path": str(output / "bundle_manifest.json"),
        "updated_freeze_inputs": updated_inputs,
        "updated_freeze_inputs_path": str(updated_inputs_binding_path),
    }


def validate_r015_machine_artifact_bundle(
    bundle_manifest_path: str | Path,
) -> Mapping[str, Any]:
    """只按清单重算文件摘要；不执行被绑定的项目代码。"""

    path = _resolved_file(bundle_manifest_path, field="bundle_manifest")
    manifest = _load_mapping(path)
    if manifest.get("schema_version") != MACHINE_ARTIFACT_BUNDLE_SCHEMA or (
        manifest.get("scientific_readout_allowed") is not False
    ) or manifest.get("formal_data_read") is not False:
        raise ValueError("R015 machine artifact bundle has the wrong identity.")
    if manifest.get("formal_round_count") != FORMAL_ROUND_COUNT or manifest.get(
        "formal_prototype_count"
    ) != FORMAL_PROTOTYPE_COUNT or manifest.get("formal_sampling_entry_count") != (
        FORMAL_ROUND_COUNT * FORMAL_PROTOTYPE_COUNT
    ):
        raise ValueError("R015 machine artifact bundle changed the formal sample size.")
    artifacts = _require_mapping(manifest.get("artifacts"), field="artifacts")
    if set(artifacts) != set(_OUTPUT_FILENAMES):
        raise ValueError("R015 machine artifact bundle is incomplete.")
    for name, raw in artifacts.items():
        binding = _require_mapping(raw, field=f"artifact {name}")
        if _file_sha256(_resolved_file(binding.get("path", ""), field=name)) != (
            binding.get("sha256")
        ):
            raise ValueError(f"R015 machine artifact hash changed: {name}")
        if _contains_pending(_load_mapping(binding["path"])):
            raise ValueError(f"R015 machine artifact contains pending: {name}")
    implementation_sources = _require_mapping(
        manifest.get("implementation_sources"), field="implementation_sources"
    )
    if not implementation_sources:
        raise ValueError("R015 machine artifact bundle lacks implementation sources.")
    for name, raw in implementation_sources.items():
        binding = _require_mapping(raw, field=f"implementation source {name}")
        if set(binding) != {"path", "sha256"} or _file_sha256(
            _resolved_file(binding.get("path", ""), field=str(name))
        ) != binding.get("sha256"):
            raise ValueError(f"R015 implementation source hash changed: {name}")
    updated = _require_mapping(
        manifest.get("updated_freeze_inputs"), field="updated_freeze_inputs"
    )
    if _file_sha256(updated["path"]) != updated.get("sha256"):
        raise ValueError("R015 generated freeze inputs hash changed.")
    updated_payload = _load_mapping(updated["path"])
    if updated_payload.get("schema_version") != UPDATED_FREEZE_INPUT_SCHEMA or (
        _contains_pending(updated_payload)
    ):
        raise ValueError("R015 generated freeze inputs are incomplete.")
    return {
        "schema_version": "path_c_r015_machine_artifact_bundle_validation_v1",
        "valid": True,
        "bundle_manifest_sha256": _file_sha256(path),
        "artifact_count": len(artifacts),
        "formal_sampling_entry_count": FORMAL_ROUND_COUNT * FORMAL_PROTOTYPE_COUNT,
    }
