"""R015 从设计数据到正式轮次的统一、可恢复执行入口。

这个模块只协调已经登记的真实实现。它不把阶段检查替换成模拟结果，也不从命令行
自行授予执行权限。每一阶段完成后先原子保存产物，再把路径和 SHA-256 写入统一状态；
恢复时只有摘要仍一致的已完成阶段才会被跳过。
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import yaml

from experiments.overcooked_v2.path_c_r015_controller import canonical_sha256
from experiments.overcooked_v2.path_c_r015_design import (
    FREEZE_ASSEMBLY_REPORT_SCHEMA,
    FREEZE_INPUT_SCHEMA,
    apply_preregistration_fill_values,
    authorize_pilot_freeze_manifest,
    bind_completed_pilot_report,
    bind_completed_pilot_report_to_preregistration,
    build_pending_freeze_manifest,
    file_sha256,
    freeze_inputs_with_preregistration_path,
    load_mapping,
    load_r015_design_protocol,
    preregistration_fill_values,
    run_r015_design,
    sign_ready_freeze_manifest,
    validate_prefilled_preregistration,
    validate_ready_freeze_manifest,
)
from experiments.overcooked_v2.path_c_r015_pilot import (
    _verify_backend_freeze_identity,
    run_r015_pilot,
)
from experiments.overcooked_v2.path_c_r015_runtime import (
    R015_FORMAL_REPLACEMENT_ATTEMPT_COUNT,
    R015_FORMAL_REPLACEMENT_SEED_DERIVATION_ID,
    R015_FORMAL_ROUND_SAMPLING_CONTRACT,
    R015_FORMAL_SAMPLING_SCHEDULE_SCHEMA,
    R015FormalBlockProducer,
    R015FormalBlockProductionResultV1,
    R015FormalBlockRequestV1,
    run_r015_formal,
)


R015_PIPELINE_CONFIG_SCHEMA = "path_c_r015_pipeline_v1"
R015_PIPELINE_STATE_SCHEMA = "path_c_r015_pipeline_state_v1"
R015_PIPELINE_RUN_BINDING_SCHEMA = "path_c_r015_pipeline_run_binding_v1"
R015_PIPELINE_RESULT_SCHEMA = "path_c_r015_pipeline_result_v1"
R015_PIPELINE_STAGE_ORDER = ("design", "freeze", "pilot", "formal")
_HEX = frozenset("0123456789abcdef")


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"R015 统一入口要求 {field} 是映射。")
    return value


def _require_nonempty(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"R015 统一入口缺少 {field}。")
    return text


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"R015 统一入口不能序列化 {type(value)!r}。")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            _jsonable(payload),
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_yaml_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(payload), handle, sort_keys=False, allow_unicode=True)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _resolve_path(raw: Any, *, base: Path, field: str) -> Path:
    text = _require_nonempty(raw, field=field)
    path = Path(text)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _bound_file(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise ValueError(f"R015 统一入口缺少绑定文件：{path}")
    return {"path": str(path), "sha256": file_sha256(path)}


def _single_manifest_dependency(
    dependencies: Mapping[str, Any], *, suffix: str, field: str
) -> str:
    """从训练清单的源码闭包中取一个精确匹配的真实文件。"""

    normalized_suffix = suffix.replace("\\", "/")
    matches = [
        str(path)
        for path in dependencies
        if str(path).replace("\\", "/").endswith(normalized_suffix)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"R015 主体训练清单没有唯一绑定 {field}：{normalized_suffix}"
        )
    return matches[0]


def materialize_r015_freeze_inputs(
    *,
    design_protocol_path: str | Path,
    partner_support_registration_path: str | Path,
    design_selection_report_path: str | Path,
) -> Mapping[str, Any]:
    """从已登记协议和五份训练清单构造冻结阶段的真实输入。

    这个函数不读取设计结果，也不加载 checkpoint。设计选择路径只是提前登记其
    唯一输出位置；冻结阶段仍会从该文件重算两项机械选择。checkpoint 格式、参数树
    路径和环境源码均从实际训练清单读取，调用方不再手工抄写一份易漂移的中间文件。
    """

    from experiments.overcooked_v2.path_c_pool_admission import (
        load_r015_partner_support_config,
    )

    protocol_path = Path(design_protocol_path).expanduser().resolve()
    registration_path = Path(partner_support_registration_path).expanduser().resolve()
    protocol = load_r015_design_protocol(protocol_path)
    support = _require_mapping(protocol.get("support"), field="design support")
    declared_registration = _path_declared_by_file(
        owner_path=protocol_path,
        raw_path=support.get("partner_support_registration"),
        field="design support registration",
    )
    if declared_registration != registration_path:
        raise ValueError("R015 冻结输入的伙伴支持登记与设计协议不一致。")
    support_config = load_r015_partner_support_config(registration_path)
    raw_candidates = support_config.get("candidates")
    if not isinstance(raw_candidates, Sequence) or isinstance(
        raw_candidates, (str, bytes, bytearray)
    ) or len(raw_candidates) != 4:
        raise ValueError("R015 冻结输入需要支持登记中的四个伙伴候选。")

    ego = dict(_require_mapping(support.get("ego_candidate"), field="ego candidate"))
    candidates = [
        (ego, protocol_path),
        *(
            (
                dict(_require_mapping(item, field="partner candidate")),
                registration_path,
            )
            for item in raw_candidates
        ),
    ]
    checkpoints: list[Mapping[str, Any]] = []
    ego_manifest: Mapping[str, Any] | None = None
    for candidate, declaration_path in candidates:
        seed = int(candidate.get("training_seed", -1))
        manifest_path = _resolve_path(
            candidate.get("training_manifest_path"),
            base=declaration_path.parent,
            field=f"seed {seed} training_manifest_path",
        )
        manifest = load_mapping(manifest_path)
        if manifest.get("schema_version") != "path_c_official_training_artifact_v2" or (
            int(manifest.get("seed", -1)) != seed
        ):
            raise ValueError(f"R015 seed {seed} 训练清单身份不正确。")
        manifest_checkpoint = _require_mapping(
            manifest.get("checkpoint"), field=f"seed {seed} checkpoint"
        )
        checkpoint_path = _resolve_path(
            candidate.get("checkpoint_path"),
            base=declaration_path.parent,
            field=f"seed {seed} checkpoint_path",
        )
        if Path(str(manifest_checkpoint.get("path", ""))).resolve() != checkpoint_path:
            raise ValueError(f"R015 seed {seed} 候选与训练清单指向不同 checkpoint。")
        parameter_tree_path = manifest_checkpoint.get("parameter_tree_path")
        if not isinstance(parameter_tree_path, Sequence) or isinstance(
            parameter_tree_path, (str, bytes, bytearray)
        ) or not parameter_tree_path:
            raise ValueError(f"R015 seed {seed} 训练清单缺少参数树路径。")
        checkpoints.append(
            {
                "artifact_id": str(candidate["candidate_id"]),
                "role": "ego" if seed == 100 else "partner",
                "training_seed": seed,
                "path": str(checkpoint_path),
                "format": _require_nonempty(
                    manifest_checkpoint.get("format"),
                    field=f"seed {seed} checkpoint format",
                ),
                "parameter_tree_path": [str(value) for value in parameter_tree_path],
                "training_manifest_path": str(manifest_path),
            }
        )
        if seed == 100:
            ego_manifest = manifest
    if {int(item["training_seed"]) for item in checkpoints} != {
        100,
        101,
        102,
        201,
        202,
    } or ego_manifest is None:
        raise ValueError("R015 冻结输入不是登记的主体 seed 100 加四伙伴集合。")

    source_closure = _require_mapping(
        ego_manifest.get("training_implementation_dependencies"),
        field="ego training source closure",
    )
    dependency_files = _require_mapping(
        source_closure.get("files"), field="ego training source files"
    )
    environment_sources = {
        name: _single_manifest_dependency(
            dependency_files,
            suffix=f"/jaxmarl/environments/overcooked_v2/{filename}",
            field=f"environment_sources.{name}",
        )
        for name, filename in (
            ("settings", "settings.py"),
            ("overcooked", "overcooked.py"),
            ("layouts", "layouts.py"),
            ("common", "common.py"),
        )
    }

    def design_declared_path(field: str) -> Path:
        return _path_declared_by_file(
            owner_path=protocol_path,
            raw_path=support.get(field),
            field=f"design support {field}",
        )

    return {
        "schema_version": FREEZE_INPUT_SCHEMA,
        "design_selection_report_path": str(
            Path(design_selection_report_path).expanduser().resolve()
        ),
        "settings_path": environment_sources["settings"],
        "environment_sources": environment_sources,
        "checkpoints": checkpoints,
        "support_registration_path": str(registration_path),
        "support_report_path": str(design_declared_path("partner_support_report")),
        "support_episode_evidence_path": str(
            design_declared_path("partner_support_episode_evidence")
        ),
    }


def _pipeline_config_binding(
    path: Path, config: Mapping[str, Any]
) -> Mapping[str, Any]:
    """绑定执行配置，同时允许以后只补入独立的正式运行授权。"""

    stable = copy.deepcopy(dict(config))
    stable.pop("authorization", None)
    return {
        "path": str(path),
        "stable_content_sha256": canonical_sha256(stable),
        "inline_authorization_excluded": True,
    }


def _artifact(path: str | Path) -> Mapping[str, Any]:
    return _bound_file(Path(path).resolve())


def _source_paths() -> tuple[Path, ...]:
    package = Path(__file__).resolve().parent
    return (
        Path(__file__).resolve(),
        package / "batched_rollout.py",
        package / "env_adapter.py",
        package / "path_c_backbone_ppo.py",
        package / "path_c_r015.py",
        package / "path_c_r015_controller.py",
        package / "path_c_r015_design.py",
        package / "path_c_r015_pilot.py",
        package / "path_c_r015_runtime.py",
        package / "path_c_r015_full_horizon.py",
        package / "path_c_r015_artifacts.py",
        package / "path_c_flax_policy.py",
        package / "path_c_official_artifact.py",
        package / "path_c_official_evidence.py",
        package / "path_c_pool_admission.py",
        package / "path_c_response_probe.py",
        package / "path_c_response_summary.py",
        package / "path_c_seed.py",
        package / "path_c_sequence.py",
        package / "path_c_standard.py",
        package / "path_c_standard_diagnostics.py",
        package / "path_c_standard_training.py",
        package / "official" / "overcooked_v2_experiments_adapter.py",
        package / "official" / "r015_runtime_bridge.py",
        package / "scripts" / "run_path_c_r015_pipeline.py",
        package / "scripts" / "run_path_c_r015_formal.py",
    )


def load_r015_pipeline_config(path: str | Path) -> Mapping[str, Any]:
    """读取统一入口配置；配置只声明权限，不会产生权限。"""

    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload = _require_mapping(payload, field="pipeline config")
    if payload.get("schema_version") != R015_PIPELINE_CONFIG_SCHEMA:
        raise ValueError("R015 统一入口配置版本不受支持。")
    _require_mapping(payload.get("paths"), field="paths")
    if "authorization" not in payload and "authorization_file" not in payload:
        raise ValueError("R015 统一入口没有声明授权来源。")
    return payload


def _authorization_mapping(
    config: Mapping[str, Any], *, config_path: Path
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    raw_path = config.get("authorization_file")
    if raw_path is None:
        authorization = _require_mapping(
            config.get("authorization"), field="authorization"
        )
        return authorization, {
            "kind": "inline",
            "sha256": canonical_sha256(authorization),
        }
    path = _resolve_path(
        raw_path,
        base=config_path.parent,
        field="authorization_file",
    )
    authorization = load_mapping(path)
    return authorization, {"kind": "file", **_bound_file(path)}


def _require_stage_authorization(
    authorization: Mapping[str, Any], stage: str
) -> Mapping[str, Any]:
    item = _require_mapping(authorization.get(stage), field=f"authorization.{stage}")
    if item.get("authorized") is not True:
        raise PermissionError(f"R015 {stage} 阶段没有显式执行授权。")
    _require_nonempty(item.get("reference"), field=f"authorization.{stage}.reference")
    if stage in {"freeze", "formal"}:
        _require_nonempty(
            item.get("authorization_quote"),
            field=f"authorization.{stage}.authorization_quote",
        )
    return item


def _pipeline_paths(
    config: Mapping[str, Any], *, config_path: Path, through: str
) -> Mapping[str, Path]:
    raw = _require_mapping(config.get("paths"), field="paths")
    required = {
        "output_root",
        "design_protocol",
        "partner_support_registration",
    }
    if R015_PIPELINE_STAGE_ORDER.index(through) >= 1:
        required.update(
            {"freeze_inputs", "preregistration_template", "pilot_protocol"}
        )
    if through == "formal":
        required.add("formal_runtime_config")
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError("R015 统一入口缺少路径：" + ", ".join(missing))
    resolved = {
        name: _resolve_path(value, base=config_path.parent, field=f"paths.{name}")
        for name, value in raw.items()
    }
    if "formal_preregistration" not in resolved:
        resolved["formal_preregistration"] = (
            resolved["output_root"]
            / "freeze"
            / "formal_frozen_preregistration.yaml"
        )
    return resolved


def _path_declared_by_file(
    *, owner_path: Path, raw_path: Any, field: str
) -> Path:
    return _resolve_path(raw_path, base=owner_path.parent, field=field)


def _validate_explicit_support_registration(
    *, paths: Mapping[str, Path], through: str
) -> Mapping[str, Any]:
    """确认逐项传入的伙伴支持登记与各阶段现有文件指向同一文件。"""

    explicit = paths.get("partner_support_registration")
    if explicit is None:
        raise ValueError("R015 统一入口缺少显式伙伴支持登记路径。")
    explicit = explicit.resolve()
    expected = _bound_file(explicit)
    declarations: list[tuple[str, Path]] = []

    design_path = paths["design_protocol"]
    design = load_mapping(design_path)
    design_support = _require_mapping(design.get("support"), field="design support")
    declarations.append(
        (
            "design_protocol.support.partner_support_registration",
            _path_declared_by_file(
                owner_path=design_path,
                raw_path=design_support.get("partner_support_registration"),
                field="design support registration",
            ),
        )
    )

    target_index = R015_PIPELINE_STAGE_ORDER.index(through)
    if target_index >= R015_PIPELINE_STAGE_ORDER.index("freeze"):
        freeze_path = paths["freeze_inputs"]
        freeze_inputs = load_mapping(freeze_path)
        declarations.append(
            (
                "freeze_inputs.support_registration_path",
                _path_declared_by_file(
                    owner_path=freeze_path,
                    raw_path=freeze_inputs.get("support_registration_path"),
                    field="freeze support registration",
                ),
            )
        )
        preregistration_path = paths["preregistration_template"]
        preregistration = load_mapping(preregistration_path)
        preregistration_support = _require_mapping(
            preregistration.get("partner_support"),
            field="preregistration partner_support",
        )
        declarations.append(
            (
                "preregistration.partner_support.registration_path",
                _path_declared_by_file(
                    owner_path=preregistration_path,
                    raw_path=preregistration_support.get("registration_path"),
                    field="preregistration support registration",
                ),
            )
        )
    if target_index >= R015_PIPELINE_STAGE_ORDER.index("freeze"):
        pilot_path = paths["pilot_protocol"]
        pilot = load_mapping(pilot_path)
        frozen_inputs = _require_mapping(
            pilot.get("frozen_inputs"), field="pilot frozen_inputs"
        )
        declarations.append(
            (
                "pilot_protocol.frozen_inputs.support_registration",
                _path_declared_by_file(
                    owner_path=pilot_path,
                    raw_path=frozen_inputs.get("support_registration"),
                    field="pilot support registration",
                ),
            )
        )
    if through == "formal":
        runtime_path = paths["formal_runtime_config"]
        runtime = load_mapping(runtime_path)
        runtime_support = _require_mapping(
            runtime.get("support"), field="formal runtime support"
        )
        declarations.append(
            (
                "formal_runtime.support.partner_support_registration",
                _path_declared_by_file(
                    owner_path=runtime_path,
                    raw_path=runtime_support.get("partner_support_registration"),
                    field="formal runtime support registration",
                ),
            )
        )

    disagreements = [name for name, path in declarations if path.resolve() != explicit]
    if disagreements:
        raise ValueError(
            "R015 逐项传入的伙伴支持登记与下列现有文件不一致："
            + ", ".join(disagreements)
        )
    return expected


def _checkpoint_declaration_sha256(freeze_inputs_path: Path | None) -> str | None:
    if freeze_inputs_path is None or not freeze_inputs_path.is_file():
        return None
    payload = load_mapping(freeze_inputs_path)
    if payload.get("schema_version") != FREEZE_INPUT_SCHEMA:
        raise ValueError("R015 统一入口的冻结输入版本不受支持。")
    checkpoints = payload.get("checkpoints")
    if not isinstance(checkpoints, Sequence) or isinstance(
        checkpoints, (str, bytes, bytearray)
    ) or len(checkpoints) != 5:
        raise ValueError("R015 冻结输入没有五个 checkpoint 声明。")
    return canonical_sha256(checkpoints)


def _build_run_binding(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> Mapping[str, Any]:
    stable_inputs: dict[str, Mapping[str, Any]] = {
        "pipeline_config": _pipeline_config_binding(config_path, config),
        "design_protocol": _bound_file(paths["design_protocol"]),
    }
    for name in (
        "freeze_inputs",
        "preregistration_template",
        "pilot_protocol",
        "formal_runtime_config",
        "partner_support_registration",
    ):
        path = paths.get(name)
        if path is not None and path.is_file():
            stable_inputs[name] = _bound_file(path)
    source_bindings = {
        str(path): file_sha256(path) for path in _source_paths() if path.is_file()
    }
    declared_formal_path = paths.get("formal_preregistration")
    return {
        "schema_version": R015_PIPELINE_RUN_BINDING_SCHEMA,
        "stable_inputs": stable_inputs,
        "source_sha256": source_bindings,
        "checkpoint_declaration_sha256": _checkpoint_declaration_sha256(
            paths.get("freeze_inputs")
        ),
        "declared_formal_preregistration_path": (
            str(declared_formal_path) if declared_formal_path is not None else None
        ),
        "output_root": str(paths["output_root"]),
        "scientific_readout_authorized_by_binding": False,
    }


def _binding_sha256(payload: Mapping[str, Any]) -> str:
    return canonical_sha256(payload)


def _initial_state(run_binding: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "schema_version": R015_PIPELINE_STATE_SCHEMA,
        "run_binding": copy.deepcopy(dict(run_binding)),
        "run_binding_sha256": _binding_sha256(run_binding),
        "stages": {
            stage: {"status": "not_started"} for stage in R015_PIPELINE_STAGE_ORDER
        },
        "terminal_stage": None,
        "formal_effect_value_read": False,
    }


def _load_or_create_state(
    *, state_path: Path, run_binding: Mapping[str, Any], resume: bool
) -> dict[str, Any]:
    if state_path.exists():
        if not resume:
            raise FileExistsError(
                "R015 统一状态已存在；只有显式 --resume 才能恢复。"
            )
        state = dict(load_mapping(state_path))
        if state.get("schema_version") != R015_PIPELINE_STATE_SCHEMA or (
            state.get("run_binding_sha256") != _binding_sha256(run_binding)
        ) or state.get("run_binding") != run_binding:
            raise ValueError("R015 恢复时配置、源码或稳定输入摘要发生变化。")
        return state
    if resume:
        raise FileNotFoundError("R015 --resume 找不到已有统一状态。")
    state = dict(_initial_state(run_binding))
    _write_json_atomic(state_path, state)
    return state


def _bind_formal_root_seed(
    *, state_path: Path, state: dict[str, Any], formal_root_seed: int
) -> None:
    """在预试点冻结开始前一次性绑定正式抽样根 seed。"""

    existing = state.get("formal_root_seed")
    if existing is not None and existing != formal_root_seed:
        raise ValueError("R015 恢复时 formal_root_seed 与预试点绑定值不同。")
    if existing is None:
        state["formal_root_seed"] = formal_root_seed
        state["formal_sampling_registered_before_pilot"] = True
        _write_json_atomic(state_path, state)


def _artifact_map_valid(artifacts: Any) -> bool:
    if not isinstance(artifacts, Mapping) or not artifacts:
        return False
    for binding in artifacts.values():
        if not isinstance(binding, Mapping):
            return False
        path = Path(str(binding.get("path", "")))
        if not path.is_file() or not _is_sha256(binding.get("sha256")) or (
            file_sha256(path) != binding["sha256"]
        ):
            return False
    return True


def _stage_is_complete(
    state: Mapping[str, Any], *, stage: str, input_binding_sha256: str
) -> bool:
    item = _require_mapping(
        _require_mapping(state.get("stages"), field="state.stages").get(stage),
        field=f"state.stages.{stage}",
    )
    if item.get("status") != "completed":
        return False
    if item.get("input_binding_sha256") != input_binding_sha256:
        raise ValueError(f"R015 {stage} 已完成阶段的输入摘要发生变化。")
    if not _artifact_map_valid(item.get("artifacts")):
        raise ValueError(f"R015 {stage} 已完成阶段的产物发生变化。")
    return True


def _set_stage_status(
    *,
    state_path: Path,
    state: dict[str, Any],
    stage: str,
    status: str,
    input_binding_sha256: str,
    artifacts: Mapping[str, Mapping[str, Any]] | None = None,
    result_summary: Mapping[str, Any] | None = None,
) -> None:
    item: dict[str, Any] = {
        "status": status,
        "input_binding_sha256": input_binding_sha256,
    }
    if artifacts is not None:
        item["artifacts"] = copy.deepcopy(dict(artifacts))
    if result_summary is not None:
        item["result_summary"] = copy.deepcopy(dict(result_summary))
    state["stages"][stage] = item
    _write_json_atomic(state_path, state)


def _record_stage_failure(
    *,
    state_path: Path,
    state: dict[str, Any],
    stage: str,
    input_binding_sha256: str,
    error: Exception,
) -> None:
    """只记录机械失败类型和消息；阶段内部产物仍由各入口自行恢复。"""

    _set_stage_status(
        state_path=state_path,
        state=state,
        stage=stage,
        status="failed",
        input_binding_sha256=input_binding_sha256,
        result_summary={
            "error_type": type(error).__name__,
            "error_message": str(error),
        },
    )


def _stage_artifacts(state: Mapping[str, Any], stage: str) -> Mapping[str, Any]:
    return _require_mapping(
        _require_mapping(
            _require_mapping(state.get("stages"), field="state.stages").get(stage),
            field=f"state.stages.{stage}",
        ).get("artifacts"),
        field=f"state.stages.{stage}.artifacts",
    )


def _stage_result_summary(
    state: Mapping[str, Any], stage: str
) -> Mapping[str, Any]:
    item = _require_mapping(
        _require_mapping(state.get("stages"), field="state.stages").get(stage),
        field=f"state.stages.{stage}",
    )
    return _require_mapping(
        item.get("result_summary"), field=f"state.stages.{stage}.result_summary"
    )


def _run_design_stage(
    *, paths: Mapping[str, Path], output_root: Path
) -> tuple[Mapping[str, Any], Mapping[str, Mapping[str, Any]]]:
    protocol = dict(load_r015_design_protocol(paths["design_protocol"]))
    design_output = output_root / "design"
    protocol.update(
        {
            "_protocol_path": str(paths["design_protocol"]),
            "_output_dir": str(design_output),
            "_execution_authorized": True,
        }
    )
    result = run_r015_design(protocol)
    selection_path = Path(str(result.get("selection_report_path", ""))).resolve()
    artifacts = {"design_selection_report": _artifact(selection_path)}
    histories_path = design_output / "design_histories.jsonl"
    if histories_path.is_file():
        artifacts["design_histories"] = _artifact(histories_path)
    return result, artifacts


def _manifest_without_preregistration_text(
    manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    result = copy.deepcopy(dict(manifest))
    text = dict(_require_mapping(result.get("text_artifacts"), field="text_artifacts"))
    text.pop("preregistration", None)
    result["text_artifacts"] = text
    return result


def _run_freeze_stage(
    *,
    paths: Mapping[str, Path],
    output_root: Path,
    selection_path: Path,
    authorization_quote: str,
    formal_root_seed: int,
) -> tuple[Mapping[str, Any], Mapping[str, Mapping[str, Any]]]:
    from experiments.overcooked_v2.path_c_r015_artifacts import (
        build_r015_machine_artifact_bundle,
        validate_r015_machine_artifact_bundle,
    )

    freeze_dir = output_root / "freeze"
    inputs = dict(load_mapping(paths["freeze_inputs"]))
    if inputs.get("schema_version") != FREEZE_INPUT_SCHEMA:
        raise ValueError("R015 统一入口收到错误版本的冻结输入。")
    inputs["design_selection_report_path"] = str(selection_path)
    pilot_template = load_mapping(paths["pilot_protocol"])
    if pilot_template.get("schema_version") != "path_c_r015_pilot_protocol_v1" or (
        pilot_template.get("scientific_readout_allowed") is not False
    ) or _require_mapping(
        pilot_template.get("execution_boundary"), field="pilot execution_boundary"
    ).get("current_execution_authorized") is not False:
        raise ValueError("R015 试点协议模板的诚实标记发生变化。")
    support_registration_path = _resolve_path(
        inputs.get("support_registration_path"),
        base=paths["freeze_inputs"].parent,
        field="freeze_inputs.support_registration_path",
    )
    resolved_pilot_protocol = copy.deepcopy(dict(pilot_template))
    resolved_pilot_protocol["status"] = "resolved_for_authorized_pilot"
    resolved_pilot_protocol["frozen_inputs"] = {
        "preregistration": str(freeze_dir / "prefilled_preregistration.yaml"),
        "design_selection_report": str(selection_path),
        "signed_freeze_manifest": str(freeze_dir / "pilot_frozen_manifest.json"),
        "support_registration": str(support_registration_path),
    }
    resolved_outputs = dict(
        _require_mapping(
            resolved_pilot_protocol.get("outputs"), field="pilot outputs"
        )
    )
    resolved_outputs["report_path"] = str(
        output_root / "pilot" / "pilot_wiring_report.json"
    )
    resolved_pilot_protocol["outputs"] = resolved_outputs
    resolved_pilot_path = freeze_dir / "resolved_pilot_protocol.yaml"
    _write_yaml_atomic(resolved_pilot_path, resolved_pilot_protocol)
    source_overrides = dict(inputs.get("machine_artifact_source_paths", {}))
    source_overrides["pilot_protocol"] = str(resolved_pilot_path)
    inputs["machine_artifact_source_paths"] = source_overrides
    if isinstance(formal_root_seed, bool) or not isinstance(
        formal_root_seed, int
    ) or formal_root_seed < 0:
        raise ValueError("formal_root_seed 必须是显式非负整数。")
    bundle = build_r015_machine_artifact_bundle(
        freeze_inputs=inputs,
        preregistration_path=paths["preregistration_template"],
        output_directory=freeze_dir / "machine_artifacts",
        repository_root=Path(__file__).resolve().parents[2],
        formal_root_seed=formal_root_seed,
    )
    inputs = dict(
        _require_mapping(
            bundle.get("updated_freeze_inputs"),
            field="machine artifact updated_freeze_inputs",
        )
    )
    resolved_inputs_path = Path(
        _require_nonempty(
            bundle.get("updated_freeze_inputs_path"),
            field="machine artifact updated_freeze_inputs_path",
        )
    ).resolve()
    bundle_manifest_path = Path(
        _require_nonempty(
            bundle.get("bundle_manifest_path"),
            field="machine artifact bundle_manifest_path",
        )
    ).resolve()
    bundle_validation = validate_r015_machine_artifact_bundle(bundle_manifest_path)
    bundle_manifest = _require_mapping(
        bundle.get("bundle_manifest"), field="machine artifact bundle_manifest"
    )
    bundle_artifacts = _require_mapping(
        bundle_manifest.get("artifacts"), field="machine artifact artifacts"
    )
    schedule_binding = _require_mapping(
        bundle_artifacts.get("formal_sampling_schedule"),
        field="formal sampling schedule",
    )
    schedule_path = Path(
        _require_nonempty(
            schedule_binding.get("path"), field="formal sampling schedule path"
        )
    ).resolve()
    if file_sha256(schedule_path) != schedule_binding.get("sha256"):
        raise ValueError("R015 正式抽样日程在预试点冻结前发生变化。")
    schedule = load_mapping(schedule_path)
    if schedule.get("formal_root_seed") != formal_root_seed or (
        schedule.get("schema_version") != R015_FORMAL_SAMPLING_SCHEDULE_SCHEMA
    ) or schedule.get("round_sampling_contract") != (
        R015_FORMAL_ROUND_SAMPLING_CONTRACT
    ) or schedule.get("mechanical_replacement_attempts_per_coordinate") != (
        R015_FORMAL_REPLACEMENT_ATTEMPT_COUNT
    ) or schedule.get("replacement_episode_seed_derivation_id") != (
        R015_FORMAL_REPLACEMENT_SEED_DERIVATION_ID
    ) or schedule.get("replacement_episode_seed_collision_policy") != (
        "allow_value_collisions_without_redraw"
    ):
        raise ValueError("R015 正式抽样日程没有绑定显式根 seed。")

    first_manifest = build_pending_freeze_manifest(inputs)
    first_path = freeze_dir / "first_pass_manifest.json"
    _write_json_atomic(first_path, first_manifest)
    fill_values = preregistration_fill_values(first_manifest)
    fill_path = freeze_dir / "preregistration_fill_values.yaml"
    _write_yaml_atomic(fill_path, fill_values)

    preregistration_template = load_mapping(paths["preregistration_template"])
    filled = apply_preregistration_fill_values(preregistration_template, fill_values)
    prefilled_validation = validate_prefilled_preregistration(filled)
    filled_path = freeze_dir / "prefilled_preregistration.yaml"
    _write_yaml_atomic(filled_path, filled)

    second_inputs = freeze_inputs_with_preregistration_path(inputs, filled_path)
    second_manifest_unsigned = build_pending_freeze_manifest(second_inputs)
    if _manifest_without_preregistration_text(first_manifest) != (
        _manifest_without_preregistration_text(second_manifest_unsigned)
    ):
        raise ValueError("R015 第二遍冻结装配改变了预登记文本以外的对象。")
    ready = sign_ready_freeze_manifest(
        second_manifest_unsigned,
        authorization_quote=authorization_quote,
    )
    ready_validation = validate_ready_freeze_manifest(ready)
    ready_path = freeze_dir / "ready_manifest.json"
    _write_json_atomic(ready_path, ready)
    frozen = authorize_pilot_freeze_manifest(
        ready,
        authorization_quote=authorization_quote,
    )
    frozen_path = freeze_dir / "pilot_frozen_manifest.json"
    _write_json_atomic(frozen_path, frozen)
    frozen_schedule = _require_mapping(
        _require_mapping(
            frozen.get("preregistration_artifacts"),
            field="pilot frozen preregistration_artifacts",
        ).get("formal_sampling_schedule"),
        field="pilot frozen formal_sampling_schedule",
    )
    if frozen_schedule.get("path") != str(schedule_path) or (
        frozen_schedule.get("sha256") != schedule_binding.get("sha256")
    ):
        raise ValueError("R015 试点前冻结清单没有绑定正式抽样日程。")

    assembly = {
        "schema_version": FREEZE_ASSEMBLY_REPORT_SCHEMA,
        "scientific_readout_allowed": False,
        "first_pass_manifest": _artifact(first_path),
        "preregistration_fill_values": _artifact(fill_path),
        "filled_preregistration": {
            **_artifact(filled_path),
            "validation": prefilled_validation,
        },
        "ready_manifest": {
            **_artifact(ready_path),
            "validation": ready_validation,
        },
        "pilot_frozen_manifest": _artifact(frozen_path),
        "checkpoint_identity_sha256": canonical_sha256(
            [
                {
                    "path": item.get("path"),
                    "checkpoint_sha256": item.get("checkpoint_sha256"),
                    "model_weights_sha256": item.get("model_weights_sha256"),
                    "training_run_id": item.get("training_run_id"),
                }
                for item in frozen["checkpoints"]
            ]
        ),
        "type_b_authorization_quote_sha256": hashlib.sha256(
            authorization_quote.encode("utf-8")
        ).hexdigest(),
        "formal_readout_enabled": False,
        "formal_sampling_registered_before_pilot": True,
        "formal_root_seed": formal_root_seed,
        "formal_sampling_schedule": _artifact(schedule_path),
        "machine_artifact_bundle_validation": bundle_validation,
    }
    assembly_path = freeze_dir / "freeze_assembly_report.json"
    _write_json_atomic(assembly_path, assembly)
    artifacts = {
        "resolved_freeze_inputs": _artifact(resolved_inputs_path),
        "first_pass_manifest": _artifact(first_path),
        "preregistration_fill_values": _artifact(fill_path),
        "prefilled_preregistration": _artifact(filled_path),
        "ready_manifest": _artifact(ready_path),
        "pilot_frozen_manifest": _artifact(frozen_path),
        "freeze_assembly_report": _artifact(assembly_path),
        "formal_sampling_schedule": _artifact(schedule_path),
        "resolved_pilot_protocol": _artifact(resolved_pilot_path),
    }
    artifacts["machine_artifact_bundle"] = _artifact(bundle_manifest_path)
    return assembly, artifacts


def _run_pilot_stage(
    *,
    paths: Mapping[str, Path],
    output_root: Path,
    selection_path: Path,
    frozen_manifest_path: Path,
    prefilled_preregistration_path: Path,
    resolved_pilot_protocol_path: Path,
    authorization_reference: str,
) -> tuple[Mapping[str, Any], Mapping[str, Mapping[str, Any]]]:
    pilot_output = output_root / "pilot"
    result = run_r015_pilot(
        {
            "_config_path": str(resolved_pilot_protocol_path),
            "_design_protocol_path": str(paths["design_protocol"]),
            "_design_selection_report_path": str(selection_path),
            "_freeze_manifest_path": str(frozen_manifest_path),
            "_output_dir": str(pilot_output),
            "_execution_authorized": True,
            "_authorization_reference": authorization_reference,
        }
    )
    if result.get("scientific_readout_allowed") is not False or (
        result.get("formal_interval_includes_pilot") is not False
    ):
        raise ValueError("R015 试点产物错误地开启了科学读取或进入正式区间。")
    report_path = Path(str(result.get("report_path", ""))).resolve()
    frozen_manifest = load_mapping(frozen_manifest_path)
    completed_manifest = bind_completed_pilot_report(
        frozen_manifest,
        pilot_report_path=report_path,
    )
    completed_manifest_path = output_root / "freeze" / "pilot_bound_manifest.json"
    prefilled = load_mapping(prefilled_preregistration_path)
    pilot_bound_preregistration = bind_completed_pilot_report_to_preregistration(
        prefilled,
        completed_manifest,
    )
    if completed_manifest.get("scientific_readout_allowed") is not False or (
        pilot_bound_preregistration.get("scientific_readout_allowed") is not False
    ):
        raise ValueError("R015 试点完成后不得开启正式科学读取。")
    pilot_bound_path = output_root / "freeze" / "pilot_bound_preregistration.yaml"
    _write_yaml_atomic(pilot_bound_path, pilot_bound_preregistration)
    if load_mapping(pilot_bound_path) != pilot_bound_preregistration:
        raise ValueError("R015 试点回填预登记回读不一致。")
    # completed manifest 是这一对文件的完成发布标记，必须最后写。
    _write_json_atomic(completed_manifest_path, completed_manifest)
    return result, {
        "pilot_report": _artifact(report_path),
        "pilot_blocks": _artifact(pilot_output / "pilot_blocks.jsonl"),
        "pilot_completion": _artifact(pilot_output / "pilot_complete.json"),
        "pilot_bound_manifest": _artifact(completed_manifest_path),
        "pilot_bound_preregistration": _artifact(pilot_bound_path),
    }


class _OfficialR015FormalBlockProducer:
    """把官方生产后端的完整回合执行器接到正式块接口。"""

    def __init__(
        self,
        backend: Any,
        preregistration: Any,
        *,
        support_spec: Any,
        ego_candidate: Mapping[str, Any],
        frozen_manifest_path: Path,
        preregistration_path: Path,
    ) -> None:
        method = getattr(
            getattr(backend, "full_horizon_executor", None),
            "produce_formal_paired_block",
            None,
        )
        if not callable(method):
            raise RuntimeError("R015 官方完整回合执行器没有正式 v2 块生产入口。")
        self.backend = backend
        self.preregistration = preregistration
        self.support_spec = support_spec
        self.ego_candidate = copy.deepcopy(dict(ego_candidate))
        self.frozen_manifest_path = frozen_manifest_path.resolve()
        self.preregistration_path = preregistration_path.resolve()
        self._frozen_manifest_sha256 = file_sha256(self.frozen_manifest_path)
        self._preregistration_sha256 = file_sha256(self.preregistration_path)
        self._verify_formal_freeze_identity(refresh_files=True)
        self.backend.full_horizon_executor.formal_freeze_identity_verifier = (
            self.verify_before_independent_replay
        )

    def _verify_formal_freeze_identity(self, *, refresh_files: bool) -> None:
        if file_sha256(self.frozen_manifest_path) != self._frozen_manifest_sha256 or (
            file_sha256(self.preregistration_path) != self._preregistration_sha256
        ):
            raise ValueError("R015 formal freeze files changed after producer startup.")
        manifest = load_mapping(self.frozen_manifest_path)
        preregistration = load_mapping(self.preregistration_path)
        if manifest.get("freeze_status") != "frozen" or manifest.get(
            "scientific_readout_allowed"
        ) is not True or preregistration.get("status") != "frozen" or (
            preregistration.get("scientific_readout_allowed") is not True
        ):
            raise ValueError("R015 formal producer requires the final frozen pair.")
        if manifest.get("formal_preregistration_canonical_sha256") != (
            canonical_sha256(preregistration)
        ):
            raise ValueError("R015 formal manifest binds another preregistration.")
        _verify_backend_freeze_identity(
            backend=self.backend,
            support_spec=self.support_spec,
            ego_candidate=self.ego_candidate,
            freeze_manifest=manifest,
            refresh_files=refresh_files,
        )

    def verify_before_independent_replay(self) -> None:
        """每块独立重放前再次绑定实际内存参数和最终封存文件。"""

        self._verify_formal_freeze_identity(refresh_files=False)

    def produce_formal_paired_block(
        self,
        request: R015FormalBlockRequestV1,
    ) -> R015FormalBlockProductionResultV1:
        self._verify_formal_freeze_identity(refresh_files=False)
        result = self.backend.full_horizon_executor.produce_formal_paired_block(
            request=request,
            preregistration=self.preregistration,
        )
        if isinstance(result, R015FormalBlockProductionResultV1):
            return result
        if not isinstance(result, Mapping):
            raise TypeError("R015 正式 v2 块生产入口返回了错误类型。")
        if result.get("schema_version") != "path_c_r015_paired_block_v2":
            raise ValueError("R015 正式入口拒绝非 v2 配对块。")
        return R015FormalBlockProductionResultV1(
            mechanically_valid=True,
            paired_block=result,
        )


@runtime_checkable
class R015FormalProducerFactory(Protocol):
    """从正式预登记和官方运行配置构建真实正式块生产器。"""

    def __call__(
        self,
        *,
        runtime_config: Mapping[str, Any],
        runtime_config_path: Path,
        preregistration_path: Path,
        frozen_manifest_path: Path,
    ) -> R015FormalBlockProducer:
        ...


def _support_registration_path(
    runtime_config: Mapping[str, Any], *, runtime_config_path: Path
) -> Path:
    support = _require_mapping(runtime_config.get("support"), field="support")
    path = Path(
        _require_nonempty(
            support.get("partner_support_registration"),
            field="support.partner_support_registration",
        )
    )
    return path.resolve() if path.is_absolute() else (runtime_config_path.parent / path).resolve()


def build_official_r015_formal_producer(
    *,
    runtime_config: Mapping[str, Any],
    runtime_config_path: Path,
    preregistration_path: Path,
    frozen_manifest_path: Path,
) -> R015FormalBlockProducer:
    """加载五个真实 checkpoint，并要求完整回合执行器已经提供正式块实现。"""

    from experiments.overcooked_v2.official.r015_runtime_bridge import (
        _resolve_r015_candidate_artifact_paths,
        build_official_r015_production_backend,
    )
    from experiments.overcooked_v2.path_c_pool_admission import (
        R015PartnerSupportSpec,
        load_r015_partner_support_config,
    )
    from experiments.overcooked_v2.path_c_r015 import load_r015_preregistration

    support_path = _support_registration_path(
        runtime_config,
        runtime_config_path=runtime_config_path,
    )
    support_spec = R015PartnerSupportSpec.from_mapping(
        load_r015_partner_support_config(support_path)
    )
    preregistration = load_r015_preregistration(preregistration_path)
    preregistration._require_frozen_support()
    backend = build_official_r015_production_backend(
        runtime_config,
        support_spec,
        config_base_path=runtime_config_path.parent,
    )
    support = _require_mapping(runtime_config.get("support"), field="support")
    ego_candidate = _resolve_r015_candidate_artifact_paths(
        _require_mapping(
            support.get("ego_candidate"), field="support.ego_candidate"
        ),
        config_base_path=runtime_config_path.parent,
    )
    return _OfficialR015FormalBlockProducer(
        backend,
        preregistration,
        support_spec=support_spec,
        ego_candidate=ego_candidate,
        frozen_manifest_path=frozen_manifest_path,
        preregistration_path=preregistration_path,
    )


def build_r015_formal_runtime_binding_sha256(
    runtime_config_path: str | Path,
    *,
    pipeline_run_binding_sha256: str | None = None,
    additional_source_paths: Sequence[str | Path] = (),
) -> str:
    """绑定正式运行配置、实现源码和可选的统一入口运行摘要。"""

    config_path = Path(runtime_config_path).resolve()
    if pipeline_run_binding_sha256 is not None and not _is_sha256(
        pipeline_run_binding_sha256
    ):
        raise ValueError("R015 统一入口运行摘要不是 SHA-256。")
    sources = list(_source_paths())
    sources.extend(Path(path).resolve() for path in additional_source_paths)
    unique_sources = tuple(dict.fromkeys(sources))
    return canonical_sha256(
        {
            "runtime_config_sha256": file_sha256(config_path),
            "pipeline_run_binding_sha256": pipeline_run_binding_sha256,
            "source_sha256": {
                str(path): file_sha256(path)
                for path in unique_sources
                if path.is_file()
            },
        }
    )


def _formal_output_paths(config: Mapping[str, Any], output_root: Path) -> Mapping[str, Any]:
    formal = _require_mapping(config.get("formal", {}), field="formal")
    output_dir = output_root / "formal"
    dataset = _resolve_path(
        formal.get("dataset_path", output_dir / "formal_dataset.json"),
        base=output_root,
        field="formal.dataset_path",
    )
    view = _resolve_path(
        formal.get("view_record_path", output_dir / "formal_view_record.json"),
        base=output_root,
        field="formal.view_record_path",
    )
    raw_counts = formal.get(
        "firing_count_checkpoint_paths",
        [
            output_dir / f"firing_count_round_{round_index}.json"
            for round_index in (200, 400, 800, 1600, 2500)
        ],
    )
    if not isinstance(raw_counts, Sequence) or isinstance(
        raw_counts, (str, bytes, bytearray)
    ) or len(raw_counts) != 5:
        raise ValueError("R015 正式阶段必须声明五个触发次数文件路径。")
    count_paths = tuple(
        _resolve_path(value, base=output_root, field="formal firing count path")
        for value in raw_counts
    )
    return {
        "output_directory": output_dir,
        "dataset_path": dataset,
        "view_record_path": view,
        "firing_count_checkpoint_paths": count_paths,
    }


def _run_formal_stage(
    *,
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
    output_root: Path,
    completed_manifest_path: Path,
    pilot_bound_preregistration_path: Path,
    formal_authorization_quote: str,
    pipeline_run_binding_sha256: str,
    formal_producer_factory: R015FormalProducerFactory,
) -> tuple[Mapping[str, Any], Mapping[str, Mapping[str, Any]]]:
    # 最终封存是独立的科学授权动作；统一入口只调用冻结模块的唯一实现。
    from experiments.overcooked_v2.path_c_r015_design import (
        authorize_formal_freeze_after_pilot,
    )

    formal_paths = _formal_output_paths(config, output_root)
    completed_manifest = load_mapping(completed_manifest_path)
    pilot_bound_preregistration = load_mapping(pilot_bound_preregistration_path)
    formal_freeze_dir = output_root / "freeze"
    formal_manifest_path = formal_freeze_dir / "formal_frozen_manifest.json"
    generated_preregistration_path = (
        formal_freeze_dir / "formal_frozen_preregistration.yaml"
    )
    already_published = (
        formal_manifest_path.is_file()
        and generated_preregistration_path.is_file()
    )
    if already_published:
        formal_manifest = load_mapping(formal_manifest_path)
        formal_preregistration = load_mapping(generated_preregistration_path)
        formal_authorization = _require_mapping(
            formal_manifest.get("formal_freeze_authorization"),
            field="formal_freeze_authorization",
        )
        if formal_authorization.get("authorization_quote") != (
            formal_authorization_quote
        ) or formal_manifest.get("formal_preregistration_canonical_sha256") != (
            canonical_sha256(formal_preregistration)
        ) or formal_preregistration.get("formal_data") != {
            "dataset_path": str(formal_paths["dataset_path"]),
            "firing_count_checkpoint_paths": [
                str(path)
                for path in formal_paths["firing_count_checkpoint_paths"]
            ],
            "view_consumption_record_path": str(formal_paths["view_record_path"]),
        }:
            raise ValueError("R015 恢复时最终正式封存产物发生变化。")
    else:
        expected_manifest, expected_preregistration = (
            authorize_formal_freeze_after_pilot(
                completed_manifest,
                pilot_bound_preregistration,
                authorization_quote=formal_authorization_quote,
                formal_dataset_path=formal_paths["dataset_path"],
                firing_count_checkpoint_paths=formal_paths[
                    "firing_count_checkpoint_paths"
                ],
                formal_view_record_path=formal_paths["view_record_path"],
            )
        )
        if formal_manifest_path.is_file() and load_mapping(
            formal_manifest_path
        ) != expected_manifest:
            raise ValueError("R015 已写出的正式清单与恢复重算结果不同。")
        if generated_preregistration_path.is_file() and load_mapping(
            generated_preregistration_path
        ) != expected_preregistration:
            raise ValueError("R015 已写出的正式预登记与恢复重算结果不同。")
        formal_manifest = expected_manifest
        formal_preregistration = expected_preregistration
        if not generated_preregistration_path.is_file():
            _write_yaml_atomic(generated_preregistration_path, formal_preregistration)
        if canonical_sha256(load_mapping(generated_preregistration_path)) != (
            formal_manifest["formal_preregistration_canonical_sha256"]
        ):
            raise ValueError("R015 正式预登记写入后的规范摘要不一致。")
        # formal manifest 是最终封存文件对的完成发布标记，必须最后写。
        _write_json_atomic(formal_manifest_path, formal_manifest)

    declared_preregistration_path = paths["formal_preregistration"]
    if declared_preregistration_path.resolve() != generated_preregistration_path.resolve():
        if declared_preregistration_path.is_file() and (
            load_mapping(declared_preregistration_path) != formal_preregistration
        ):
            raise ValueError(
                "R015 声明的正式预登记与 pilot 后最终封存结果不一致。"
            )
        if not declared_preregistration_path.is_file():
            _write_yaml_atomic(
                declared_preregistration_path, formal_preregistration
            )
        preregistration_path = declared_preregistration_path
    else:
        preregistration_path = generated_preregistration_path

    runtime_config = load_mapping(paths["formal_runtime_config"])
    producer = formal_producer_factory(
        runtime_config=runtime_config,
        runtime_config_path=paths["formal_runtime_config"],
        preregistration_path=preregistration_path,
        frozen_manifest_path=formal_manifest_path,
    )
    if not isinstance(producer, R015FormalBlockProducer):
        raise TypeError("R015 正式 producer factory 没有返回正式块生产器。")
    runtime_binding = build_r015_formal_runtime_binding_sha256(
        paths["formal_runtime_config"],
        pipeline_run_binding_sha256=pipeline_run_binding_sha256,
    )
    result = run_r015_formal(
        preregistration_path,
        producer=producer,
        output_directory=formal_paths["output_directory"],
        runtime_binding_sha256=runtime_binding,
        maximum_mechanical_attempts=32,
    )
    terminal_path = formal_paths["output_directory"] / "formal_terminal_result.json"
    return result, {
        "formal_frozen_manifest": _artifact(formal_manifest_path),
        "formal_frozen_preregistration": _artifact(preregistration_path),
        "formal_terminal_result": _artifact(terminal_path),
    }


def _result(
    *, state: Mapping[str, Any], state_path: Path, through: str
) -> Mapping[str, Any]:
    stages = _require_mapping(state.get("stages"), field="state.stages")
    completed = [
        stage
        for stage in R015_PIPELINE_STAGE_ORDER
        if _require_mapping(stages.get(stage), field=f"state.stages.{stage}").get(
            "status"
        )
        == "completed"
    ]
    terminal_stage = state.get("terminal_stage")
    return {
        "schema_version": R015_PIPELINE_RESULT_SCHEMA,
        "status": (
            "stopped_by_registered_mechanical_result"
            if terminal_stage is not None
            else "completed_through_requested_stage"
        ),
        "requested_through": through,
        "completed_stages": completed,
        "terminal_stage": terminal_stage,
        "state_path": str(state_path),
        "state_sha256": file_sha256(state_path),
        "stage_artifacts": {
            stage: copy.deepcopy(dict(stages[stage].get("artifacts", {})))
            for stage in R015_PIPELINE_STAGE_ORDER
        },
    }


def run_r015_pipeline(
    config: Mapping[str, Any],
    *,
    config_path: str | Path,
    through: str,
    resume: bool,
    execution_authorized: bool,
    formal_root_seed: int | None = None,
    formal_producer_factory: R015FormalProducerFactory = (
        build_official_r015_formal_producer
    ),
) -> Mapping[str, Any]:
    """依次运行设计、两遍冻结、试点和正式轮次，并可从完整阶段恢复。

    返回值固定含 schema_version、status、completed_stages、terminal_stage、
    state_path 和 stage_artifacts，调用方不需要再拼接四个独立入口的结果。
    """

    if config.get("schema_version") != R015_PIPELINE_CONFIG_SCHEMA:
        raise ValueError("R015 统一入口配置版本不受支持。")
    if through not in R015_PIPELINE_STAGE_ORDER:
        raise ValueError("R015 --through 必须是 design、freeze、pilot 或 formal。")
    if execution_authorized is not True:
        raise PermissionError("R015 统一入口默认不执行；必须显式确认本次授权。")
    source_path = Path(config_path).resolve()
    paths = _pipeline_paths(config, config_path=source_path, through=through)
    _validate_explicit_support_registration(paths=paths, through=through)
    output_root = paths["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    authorization, authorization_binding = _authorization_mapping(
        config,
        config_path=source_path,
    )
    target_index = R015_PIPELINE_STAGE_ORDER.index(through)
    if target_index >= R015_PIPELINE_STAGE_ORDER.index("freeze") and (
        isinstance(formal_root_seed, bool)
        or not isinstance(formal_root_seed, int)
        or formal_root_seed < 0
    ):
        raise ValueError("R015 freeze 及后续阶段要求显式 formal_root_seed。")
    stage_authorizations = {
        stage: _require_stage_authorization(authorization, stage)
        for stage in R015_PIPELINE_STAGE_ORDER[: target_index + 1]
    }

    run_binding = _build_run_binding(
        config_path=source_path,
        config=config,
        paths=paths,
    )
    state_path = output_root / "pipeline_state.json"
    state = _load_or_create_state(
        state_path=state_path,
        run_binding=run_binding,
        resume=resume,
    )
    if target_index >= R015_PIPELINE_STAGE_ORDER.index("freeze"):
        _bind_formal_root_seed(
            state_path=state_path,
            state=state,
            formal_root_seed=formal_root_seed,
        )
    state["requested_through"] = through
    state["authorization_binding"] = authorization_binding
    _write_json_atomic(state_path, state)

    design_input = canonical_sha256(
        [_binding_sha256(run_binding), stage_authorizations["design"], "design"]
    )
    if not _stage_is_complete(
        state, stage="design", input_binding_sha256=design_input
    ):
        _set_stage_status(
            state_path=state_path,
            state=state,
            stage="design",
            status="running",
            input_binding_sha256=design_input,
        )
        try:
            design_result, artifacts = _run_design_stage(
                paths=paths,
                output_root=output_root,
            )
        except Exception as error:
            _record_stage_failure(
                state_path=state_path,
                state=state,
                stage="design",
                input_binding_sha256=design_input,
                error=error,
            )
            raise
        design_status = str(design_result.get("status", ""))
        _set_stage_status(
            state_path=state_path,
            state=state,
            stage="design",
            status="completed",
            input_binding_sha256=design_input,
            artifacts=artifacts,
            result_summary={"status": design_status},
        )
    design_status = str(_stage_result_summary(state, "design").get("status", ""))
    if design_status != "selected":
        if state.get("terminal_stage") not in (None, "design"):
            raise ValueError("R015 统一状态包含互相矛盾的终止阶段。")
        state["terminal_stage"] = "design"
        state["terminal_reason"] = design_status
        _write_json_atomic(state_path, state)
        return _result(state=state, state_path=state_path, through=through)
    if state.get("terminal_stage") is not None:
        raise ValueError("R015 已选设计结果不能带有既存终止标记。")
    if target_index == 0:
        return _result(state=state, state_path=state_path, through=through)

    design_artifacts = _stage_artifacts(state, "design")
    selection_path = Path(design_artifacts["design_selection_report"]["path"])
    freeze_authorization = stage_authorizations["freeze"]
    freeze_input = canonical_sha256(
        [
            _binding_sha256(run_binding),
            design_artifacts,
            freeze_authorization,
            formal_root_seed,
            "freeze",
        ]
    )
    if not _stage_is_complete(
        state, stage="freeze", input_binding_sha256=freeze_input
    ):
        _set_stage_status(
            state_path=state_path,
            state=state,
            stage="freeze",
            status="running",
            input_binding_sha256=freeze_input,
        )
        try:
            freeze_result, artifacts = _run_freeze_stage(
                paths=paths,
                output_root=output_root,
                selection_path=selection_path,
                authorization_quote=_require_nonempty(
                    freeze_authorization.get("authorization_quote"),
                    field="authorization.freeze.authorization_quote",
                ),
                formal_root_seed=formal_root_seed,
            )
        except Exception as error:
            _record_stage_failure(
                state_path=state_path,
                state=state,
                stage="freeze",
                input_binding_sha256=freeze_input,
                error=error,
            )
            raise
        _set_stage_status(
            state_path=state_path,
            state=state,
            stage="freeze",
            status="completed",
            input_binding_sha256=freeze_input,
            artifacts=artifacts,
            result_summary={
                "schema_version": freeze_result.get("schema_version"),
                "formal_readout_enabled": False,
            },
        )
    if target_index == 1:
        return _result(state=state, state_path=state_path, through=through)

    freeze_artifacts = _stage_artifacts(state, "freeze")
    pilot_authorization = stage_authorizations["pilot"]
    pilot_input = canonical_sha256(
        [
            _binding_sha256(run_binding),
            design_artifacts,
            freeze_artifacts,
            pilot_authorization,
            "pilot",
        ]
    )
    if not _stage_is_complete(
        state, stage="pilot", input_binding_sha256=pilot_input
    ):
        _set_stage_status(
            state_path=state_path,
            state=state,
            stage="pilot",
            status="running",
            input_binding_sha256=pilot_input,
        )
        try:
            pilot_result, artifacts = _run_pilot_stage(
                paths=paths,
                output_root=output_root,
                selection_path=selection_path,
                frozen_manifest_path=Path(
                    freeze_artifacts["pilot_frozen_manifest"]["path"]
                ),
                prefilled_preregistration_path=Path(
                    freeze_artifacts["prefilled_preregistration"]["path"]
                ),
                resolved_pilot_protocol_path=Path(
                    freeze_artifacts["resolved_pilot_protocol"]["path"]
                ),
                authorization_reference=_require_nonempty(
                    pilot_authorization.get("reference"),
                    field="authorization.pilot.reference",
                ),
            )
            wiring = _require_mapping(
                pilot_result.get("wiring_checks"), field="pilot wiring_checks"
            )
            if len(wiring) != 9 or any(
                value is not True for value in wiring.values()
            ):
                raise ValueError("R015 试点没有通过全部九项接线检查。")
        except Exception as error:
            _record_stage_failure(
                state_path=state_path,
                state=state,
                stage="pilot",
                input_binding_sha256=pilot_input,
                error=error,
            )
            raise
        _set_stage_status(
            state_path=state_path,
            state=state,
            stage="pilot",
            status="completed",
            input_binding_sha256=pilot_input,
            artifacts=artifacts,
            result_summary={"wiring_checks": copy.deepcopy(dict(wiring))},
        )
    if target_index == 2:
        return _result(state=state, state_path=state_path, through=through)

    pilot_artifacts = _stage_artifacts(state, "pilot")
    formal_authorization = stage_authorizations["formal"]
    formal_input = canonical_sha256(
        [
            _binding_sha256(run_binding),
            pilot_artifacts,
            formal_authorization,
            _bound_file(paths["formal_runtime_config"]),
            "formal",
        ]
    )
    if not _stage_is_complete(
        state, stage="formal", input_binding_sha256=formal_input
    ):
        _set_stage_status(
            state_path=state_path,
            state=state,
            stage="formal",
            status="running",
            input_binding_sha256=formal_input,
        )
        try:
            formal_result, artifacts = _run_formal_stage(
                config=config,
                paths=paths,
                output_root=output_root,
                completed_manifest_path=Path(
                    pilot_artifacts["pilot_bound_manifest"]["path"]
                ),
                pilot_bound_preregistration_path=Path(
                    pilot_artifacts["pilot_bound_preregistration"]["path"]
                ),
                formal_authorization_quote=_require_nonempty(
                    formal_authorization.get("authorization_quote"),
                    field="authorization.formal.authorization_quote",
                ),
                pipeline_run_binding_sha256=_binding_sha256(run_binding),
                formal_producer_factory=formal_producer_factory,
            )
        except Exception as error:
            _record_stage_failure(
                state_path=state_path,
                state=state,
                stage="formal",
                input_binding_sha256=formal_input,
                error=error,
            )
            raise
        state["formal_effect_value_read"] = bool(
            formal_result.get("effect_value_read", False)
        )
        _set_stage_status(
            state_path=state_path,
            state=state,
            stage="formal",
            status="completed",
            input_binding_sha256=formal_input,
            artifacts=artifacts,
            result_summary={
                "termination_kind": formal_result.get("termination_kind"),
                "effect_value_read": formal_result.get("effect_value_read"),
            },
        )
    return _result(state=state, state_path=state_path, through=through)


__all__ = [
    "R015FormalProducerFactory",
    "R015_PIPELINE_CONFIG_SCHEMA",
    "R015_PIPELINE_RESULT_SCHEMA",
    "R015_PIPELINE_RUN_BINDING_SCHEMA",
    "R015_PIPELINE_STAGE_ORDER",
    "R015_PIPELINE_STATE_SCHEMA",
    "build_official_r015_formal_producer",
    "build_r015_formal_runtime_binding_sha256",
    "load_r015_pipeline_config",
    "materialize_r015_freeze_inputs",
    "run_r015_pipeline",
]
