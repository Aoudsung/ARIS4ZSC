"""R015 的 80 块接线试点入口；试点永不进入正式统计。"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from experiments.overcooked_v2.path_c_r015_controller import (
    OfficialHistoryV1,
    canonical_sha256,
    derive_controller_key,
)
from experiments.overcooked_v2.path_c_r015_runtime import (
    R015_FORMAL_GROUPS,
    validate_r015_lockstep_groups,
    verify_r015_probe_decision,
    verify_r015_safety_comparison,
    verify_r015_trace_manifest,
)


PILOT_BLOCK_SCHEMA = "path_c_r015_executed_paired_block_v2"
PILOT_REPORT_SCHEMA = "path_c_r015_pilot_wiring_report_v2"
PILOT_RUN_BINDING_SCHEMA = "path_c_r015_pilot_run_binding_v1"
PILOT_COMPLETION_SCHEMA = "path_c_r015_pilot_completion_v1"
PILOT_BLOCK_RECORD_SCHEMA = "path_c_r015_pilot_atomic_block_record_v1"
_HEX = frozenset("0123456789abcdef")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"R015 pilot evidence contains unsupported value {type(value)!r}.")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    if not path.exists():
        return []
    result = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"R015 pilot JSONL has an incomplete line {line_number}."
            ) from error
        if not isinstance(payload, Mapping):
            raise ValueError("R015 pilot JSONL records must be mappings.")
        result.append(payload)
    return result


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        _jsonable(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl_atomic(path: Path, payloads: Sequence[Mapping[str, Any]]) -> None:
    """从已原子完成的块记录重建只含完整行的顺序账本。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(
                json.dumps(
                    _jsonable(payload),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _block_record_path(output_dir: Path, ordinal: int) -> Path:
    return output_dir / "pilot_block_records" / f"block_{ordinal:04d}.json"


def _load_atomic_block_records(output_dir: Path) -> list[Mapping[str, Any]]:
    """只读取已原子改名完成且序号连续的试点块。"""

    record_dir = output_dir / "pilot_block_records"
    if not record_dir.exists():
        return []
    paths = sorted(record_dir.glob("block_*.json"))
    blocks: list[Mapping[str, Any]] = []
    for ordinal, path in enumerate(paths, start=1):
        if path != _block_record_path(output_dir, ordinal):
            raise ValueError("R015 pilot atomic block records contain a gap.")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping) or payload.get("schema_version") != (
            PILOT_BLOCK_RECORD_SCHEMA
        ) or int(payload.get("block_ordinal", -1)) != ordinal:
            raise ValueError("R015 pilot atomic block record has the wrong schema.")
        block = payload.get("block")
        if not isinstance(block, Mapping) or canonical_sha256(block) != payload.get(
            "block_sha256"
        ):
            raise ValueError("R015 pilot atomic block record changed after completion.")
        blocks.append(block)
    return blocks


def _write_atomic_block_record(
    output_dir: Path,
    *,
    ordinal: int,
    block: Mapping[str, Any],
) -> None:
    path = _block_record_path(output_dir, ordinal)
    if path.exists():
        raise FileExistsError("R015 pilot refuses to overwrite a completed block.")
    _write_json_atomic(
        path,
        {
            "schema_version": PILOT_BLOCK_RECORD_SCHEMA,
            "scientific_readout_allowed": False,
            "block_ordinal": ordinal,
            "block_sha256": canonical_sha256(block),
            "block": block,
        },
    )


def _source_binding() -> Mapping[str, str]:
    """绑定试点直接调用的执行、策略桥和重放实现。"""

    from experiments.overcooked_v2 import path_c_r015_full_horizon
    from experiments.overcooked_v2 import path_c_r015_runtime
    from experiments.overcooked_v2.official import r015_runtime_bridge

    paths = {
        "pilot": Path(__file__).resolve(),
        "full_horizon_executor": Path(path_c_r015_full_horizon.__file__).resolve(),
        "runtime_verifier": Path(path_c_r015_runtime.__file__).resolve(),
        "official_runtime_bridge": Path(r015_runtime_bridge.__file__).resolve(),
    }
    return {name: _file_sha256(path) for name, path in paths.items()}


def _bind_or_verify_run(output_dir: Path, binding: Mapping[str, Any]) -> None:
    path = output_dir / "pilot_run_binding.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if canonical_sha256(existing) != canonical_sha256(binding):
            raise ValueError(
                "R015 pilot resume changed source, config, checkpoint binding, or seeds."
            )
        return
    if (output_dir / "pilot_blocks.jsonl").exists() or (
        output_dir / "pilot_block_records"
    ).exists():
        raise ValueError("R015 pilot evidence exists without its immutable run binding.")
    _write_json_atomic(path, binding)


def _verify_backend_freeze_identity(
    *,
    backend: Any,
    support_spec: Any,
    ego_candidate: Mapping[str, Any],
    freeze_manifest: Mapping[str, Any],
    refresh_files: bool = True,
) -> None:
    """核对实际加载的五成员库就是冻结清单中的五个 checkpoint。"""

    verifier = getattr(backend, "verify_frozen_checkpoint_identities", None)
    if not callable(verifier):
        raise TypeError("R015 production backend lacks frozen identity verification.")
    verified = verifier(freeze_manifest, refresh_files=refresh_files)

    expected = {
        candidate.candidate_id: {
            "training_seed": int(candidate.training_seed),
            "checkpoint_path": Path(candidate.checkpoint_path).resolve(),
        }
        for candidate in support_spec.candidates
    }
    ego_id = str(ego_candidate.get("candidate_id", ""))
    expected[ego_id] = {
        "training_seed": int(ego_candidate["training_seed"]),
        "checkpoint_path": Path(str(ego_candidate["checkpoint_path"])).resolve(),
    }
    raw_checkpoints = freeze_manifest.get("checkpoints")
    if not isinstance(raw_checkpoints, Sequence) or isinstance(
        raw_checkpoints, (str, bytes)
    ):
        raise ValueError("R015 freeze manifest lacks five checkpoint bindings.")
    frozen = {
        int(item.get("training_seed", -1)): item
        for item in raw_checkpoints
        if isinstance(item, Mapping)
    }
    if set(backend.policies) != set(expected) or set(verified) != set(expected) or (
        len(frozen) != 5
    ):
        raise ValueError("R015 pilot backend differs from the frozen five-member library.")
    for candidate_id, identity in expected.items():
        record = frozen.get(identity["training_seed"])
        actual = verified.get(candidate_id)
        if not isinstance(record, Mapping):
            raise ValueError("R015 pilot freeze manifest omits a loaded checkpoint.")
        if not isinstance(actual, Mapping) or int(
            actual.get("training_seed", -1)
        ) != identity["training_seed"] or Path(
            str(actual.get("path", ""))
        ).resolve() != identity["checkpoint_path"] or int(
            record.get("training_seed", -1)
        ) != identity["training_seed"] or Path(
            str(record.get("path", ""))
        ).resolve() != identity["checkpoint_path"] or not _is_sha256(
            record.get("checkpoint_sha256")
        ) or not _is_sha256(record.get("model_weights_sha256")) or not _is_sha256(
            record.get("training_run_id")
        ):
            raise ValueError("R015 pilot checkpoint identity differs from the freeze manifest.")


def _verify_pilot_input_bindings(
    *,
    freeze_manifest: Mapping[str, Any],
    pilot_protocol_path: Path,
    design_protocol_path: Path,
    selection_path: Path,
    support_registration_path: Path,
    prototype_ids: Sequence[str],
) -> None:
    """核对试点实际读取的协议、选择报告和支持登记均来自冻结清单。"""

    protocol_artifacts = freeze_manifest.get("protocol_artifacts")
    partner_support = freeze_manifest.get("partner_support")
    design_selection = freeze_manifest.get("design_selection")
    if not all(
        isinstance(value, Mapping)
        for value in (protocol_artifacts, partner_support, design_selection)
    ):
        raise ValueError("R015 pilot freeze manifest lacks bound inputs.")
    for raw_binding, actual_path, label in (
        (
            protocol_artifacts.get("design_data_protocol"),
            design_protocol_path,
            "design protocol",
        ),
        (
            protocol_artifacts.get("pilot_protocol"),
            pilot_protocol_path,
            "pilot protocol",
        ),
    ):
        if not isinstance(raw_binding, Mapping) or Path(
            str(raw_binding.get("path", ""))
        ).resolve() != actual_path.resolve() or raw_binding.get("sha256") != (
            _file_sha256(actual_path)
        ):
            raise ValueError(f"R015 pilot {label} differs from its freeze manifest.")
    if Path(str(design_selection.get("report_path", ""))).resolve() != (
        selection_path.resolve()
    ) or design_selection.get("report_sha256") != _file_sha256(selection_path):
        raise ValueError("R015 pilot selection differs from its freeze manifest.")
    if Path(str(partner_support.get("registration_path", ""))).resolve() != (
        support_registration_path.resolve()
    ) or partner_support.get("registration_sha256") != _file_sha256(
        support_registration_path
    ):
        raise ValueError(
            "R015 pilot support registration differs from its freeze manifest."
        )
    frozen_prototypes = partner_support.get("candidate_ids")
    if not isinstance(frozen_prototypes, Sequence) or isinstance(
        frozen_prototypes, (str, bytes)
    ) or tuple(str(value) for value in frozen_prototypes) != tuple(prototype_ids):
        raise ValueError("R015 pilot prototype order differs from its freeze manifest.")


def _spread(values: Sequence[float]) -> Mapping[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("R015 pilot spread requires finite paired differences.")
    return {
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "median": float(np.median(array)),
        "standard_deviation": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
    }


def _independent_block_replayers(arguments: Mapping[str, Any]):
    """一次独立重跑供轨迹、决策和安全三类核验共同读取。"""

    cache: dict[str, Mapping[str, Any]] = {}

    def recomputed(executor: Any) -> Mapping[str, Any]:
        if "block" not in cache:
            value = executor.run_paired_block(**copy.deepcopy(dict(arguments)))
            if not isinstance(value, Mapping) or value.get("schema_version") != (
                PILOT_BLOCK_SCHEMA
            ):
                raise TypeError("R015 pilot replay returned another block schema.")
            cache["block"] = value
        return cache["block"]

    def trace(executor: Any, context: Mapping[str, Any]) -> Mapping[str, Any]:
        group = str(context.get("replay_group", ""))
        if group not in R015_FORMAL_GROUPS:
            raise ValueError("R015 pilot trace replay lacks its formal group.")
        return recomputed(executor)["groups"][group]

    def decision(executor: Any, context: Mapping[str, Any]) -> Mapping[str, Any]:
        del context
        return {"consultations": recomputed(executor)["consultations"]}

    def safety(executor: Any, context: Mapping[str, Any]) -> Mapping[str, Any]:
        del context
        return {"comparisons": recomputed(executor)["safety_comparisons"]}

    return trace, decision, safety


def _information_isolation_verified(block: Mapping[str, Any]) -> bool:
    """从控制器实际输入重建官方历史，不信任空的自报违规列表。"""

    try:
        for group in R015_FORMAL_GROUPS:
            payload = block["groups"][group]
            if payload.get("forbidden_read_events") != []:
                return False
            events = payload.get("official_history_events")
            if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
                return False
            OfficialHistoryV1(tuple(events))
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _response_pairing_verified(block: Mapping[str, Any]) -> bool:
    groups = block.get("groups")
    if not isinstance(groups, Mapping):
        return False
    if block.get("probe_fired") is not True:
        return all(
            "current_probe_response" not in groups[group]
            and "belief_current_response_extra" not in groups[group]
            for group in R015_FORMAL_GROUPS
        )
    try:
        masked = groups["A2-mask"]
        used = groups["A2-use"]
        response = used["current_probe_response"]
        return (
            masked["current_probe_response"] == response
            and masked["belief_current_response_extra"] is None
            and used["belief_current_response_extra"] == response
            and masked["continuation_direct_response"] is None
            and used["continuation_direct_response"] is None
            and masked["recurrent_direct_response"] is None
            and used["recurrent_direct_response"] is None
            and masked["non_belief_response_aliases"] == []
            and used["non_belief_response_aliases"] == []
        )
    except (KeyError, TypeError):
        return False


def _shared_firing_indicator_verified(block: Mapping[str, Any]) -> bool:
    """由共同咨询证据重算块级触发标记，并核对两条探查轨迹。"""

    consultations = block.get("consultations")
    groups = block.get("groups")
    if not isinstance(consultations, Sequence) or not isinstance(groups, Mapping):
        return False
    selected = [
        item.get("selected_for_safety_probe_id")
        for item in consultations
        if isinstance(item, Mapping)
        and item.get("selected_for_safety_probe_id") is not None
    ]
    comparisons = block.get("safety_comparisons")
    if not isinstance(comparisons, Sequence):
        return False
    accepted = bool(selected) and len(comparisons) == 4 and all(
        isinstance(item, Mapping)
        and item.get("positive_posterior_support") is True
        and item.get("compatible_hidden_state_reconstructed") is True
        and int(item.get("wrong_delivery_count", -1)) == 0
        for item in comparisons
    )
    fired = block.get("probe_fired") is True
    if fired is not accepted:
        return False
    if fired:
        probe_id = block.get("selected_probe_id")
        probe_step = block.get("probe_step")
        return (
            isinstance(probe_step, int)
            and not isinstance(probe_step, bool)
            and 1 <= probe_step <= 100
            and len(selected) == 1
            and selected[0] == probe_id
            and groups["A2-mask"].get("selected_probe_id") == probe_id
            and groups["A2-use"].get("selected_probe_id") == probe_id
            and "selected_probe_id" not in groups["A1"]
        )
    return (
        block.get("probe_step") is None
        and block.get("selected_probe_id") is None
        and len(selected) <= 1
        and all("selected_probe_id" not in groups[group] for group in R015_FORMAL_GROUPS)
    )


def _random_keys_verified(block: Mapping[str, Any]) -> bool:
    try:
        trajectories = {
            group: block["groups"][group]["environment_steps"]
            for group in R015_FORMAL_GROUPS
        }
        if any(len(value) != 400 for value in trajectories.values()):
            return False
        root_key = canonical_sha256(
            [
                "r015_paired_block_v2",
                block["audit_unit_id"],
                int(block["episode_seed"]),
                block["partner_prototype_id"],
            ]
        )
        probe_step = block.get("probe_step")
        for step in range(400):
            keys = {
                trajectories[group][step]["environment_random_key"]
                for group in R015_FORMAL_GROUPS
            }
            if len(keys) != 1 or not _is_sha256(next(iter(keys))):
                return False
            if probe_step is None or step < int(probe_step):
                branch_key = derive_controller_key(
                    root_key,
                    "actual_episode",
                    step,
                )
                key_offset = 0
            else:
                branch_key = derive_controller_key(
                    root_key,
                    "post_fire",
                    int(probe_step),
                )
                key_offset = step - int(probe_step)
            expected = derive_controller_key(
                branch_key,
                "future_environment",
                key_offset,
            )
            if next(iter(keys)) != expected:
                return False
    except (KeyError, TypeError):
        return False
    return True


def _online_filter_verified(block: Mapping[str, Any]) -> bool:
    """核对真实三组轨迹逐步更新后验；零支持会在执行时直接关闭。"""

    try:
        for group in R015_FORMAL_GROUPS:
            for step in block["groups"][group]["environment_steps"]:
                if step.get("belief_update_mode") != "online_each_environment_step":
                    return False
                if not _is_sha256(step.get("belief_before_sha256")) or not _is_sha256(
                    step.get("belief_after_sha256")
                ):
                    return False
    except (KeyError, TypeError):
        return False
    return True


def _zero_probe_explanation_verified(block: Mapping[str, Any]) -> bool:
    counts = block.get("zero_probe_interpretability_counts")
    if not isinstance(counts, Mapping) or set(counts) != {
        "candidate_evaluation_count",
        "safety_rejection_count",
        "nonpositive_score_count",
        "window_expiration_count",
    }:
        return False
    if any(isinstance(value, bool) or int(value) < 0 for value in counts.values()):
        return False
    if int(counts["candidate_evaluation_count"]) <= 0:
        return False
    if block.get("probe_fired") is True:
        return True
    return sum(
        int(counts[key])
        for key in (
            "safety_rejection_count",
            "nonpositive_score_count",
            "window_expiration_count",
        )
    ) > 0


def _firing_count_input_isolation_verified(payload: Mapping[str, Any]) -> bool:
    """实际调用正式计数解析器，确认加入任一配对差就会被拒绝。"""

    from experiments.overcooked_v2.path_c_r015 import (
        evaluate_r015_firing_count_checkpoint,
    )

    contaminated = dict(payload)
    contaminated["delta_net"] = 0.0
    try:
        evaluate_r015_firing_count_checkpoint(
            contaminated,
            statistics=None,  # 字段检查必须先于任何统计对象读取。
            expected_prototype_ids=tuple(
                payload["firing_counts_by_prototype"]
            ),
        )
    except ValueError as error:
        return "count-only input fields" in str(error)
    return False


def run_r015_pilot(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """运行四原型各 20 块并要求九项接线检查全部通过。"""

    if config.get("_execution_authorized") is not True:
        raise PermissionError("R015 pilot execution requires the recorded authorization.")
    authorization_reference = str(config.get("_authorization_reference", "")).strip()
    if not authorization_reference:
        raise PermissionError("R015 pilot requires its recorded authorization reference.")
    config_path = Path(str(config.get("_config_path", ""))).resolve()
    output_dir = Path(str(config.get("_output_dir", ""))).resolve()
    design_protocol_path = Path(
        str(config.get("_design_protocol_path", ""))
    ).resolve()
    selection_path = Path(
        str(config.get("_design_selection_report_path", ""))
    ).resolve()
    freeze_manifest_path = Path(
        str(config.get("_freeze_manifest_path", ""))
    ).resolve()
    if not all(
        path.is_file()
        for path in (
            config_path,
            design_protocol_path,
            selection_path,
            freeze_manifest_path,
        )
    ) or not str(config.get("_output_dir", "")):
        raise ValueError("R015 pilot requires its four bound input paths.")
    pilot_protocol = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(pilot_protocol, Mapping) or pilot_protocol.get(
        "schema_version"
    ) != "path_c_r015_pilot_protocol_v1":
        raise ValueError("R015 pilot protocol has the wrong schema.")
    if pilot_protocol.get("scientific_readout_allowed") is not False or (
        pilot_protocol.get("execution_boundary", {}).get(
            "current_execution_authorized"
        )
        is not False
    ):
        raise ValueError("R015 pilot honesty markers changed.")
    if pilot_protocol.get("status") != "resolved_for_authorized_pilot":
        raise ValueError("R015 pilot refuses a protocol whose concrete inputs are unresolved.")
    frozen_inputs = pilot_protocol.get("frozen_inputs")
    outputs = pilot_protocol.get("outputs")
    wiring_checks = pilot_protocol.get("wiring_checks")
    if not all(
        isinstance(value, Mapping)
        for value in (frozen_inputs, outputs, wiring_checks)
    ) or any(
        "pending" in str(value).lower()
        for value in (frozen_inputs, outputs)
    ):
        raise ValueError("R015 pilot protocol retains an unresolved input or output.")

    def protocol_path(raw: Any, *, field: str) -> Path:
        text = str(raw or "").strip()
        if not text:
            raise ValueError(f"R015 pilot protocol lacks {field}.")
        path = Path(text)
        return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()

    declared_selection = protocol_path(
        frozen_inputs.get("design_selection_report"),
        field="frozen_inputs.design_selection_report",
    )
    declared_manifest = protocol_path(
        frozen_inputs.get("signed_freeze_manifest"),
        field="frozen_inputs.signed_freeze_manifest",
    )
    declared_support = protocol_path(
        frozen_inputs.get("support_registration"),
        field="frozen_inputs.support_registration",
    )
    declared_preregistration = protocol_path(
        frozen_inputs.get("preregistration"),
        field="frozen_inputs.preregistration",
    )
    declared_report = protocol_path(
        outputs.get("report_path"), field="outputs.report_path"
    )
    if declared_selection != selection_path or declared_manifest != freeze_manifest_path:
        raise ValueError("R015 pilot runtime paths differ from its resolved protocol.")
    if declared_report != output_dir / "pilot_wiring_report.json":
        raise ValueError("R015 pilot report path differs from its resolved protocol.")
    if not declared_preregistration.is_file():
        raise ValueError("R015 pilot resolved preregistration does not exist.")
    if set(wiring_checks) != {
        "information_isolation",
        "filter_normalization_and_repeat_stability",
        "response_use_pairing",
        "random_key_naming",
        "zero_probe_interpretability",
        "artifact_counts_read_back",
        "lockstep_structural_zero",
        "shared_firing_indicator",
        "firing_count_input_isolation",
    } or any(value != "required" for value in wiring_checks.values()):
        raise ValueError("R015 pilot protocol changed its nine required wiring checks.")

    from experiments.overcooked_v2.official.r015_runtime_bridge import (
        build_official_r015_production_backend,
    )
    from experiments.overcooked_v2.path_c_pool_admission import (
        R015PartnerSupportSpec,
        load_r015_partner_support_config,
    )
    from experiments.overcooked_v2.path_c_r015_design import (
        load_mapping,
        load_r015_design_protocol,
        validate_authorized_pilot_freeze_manifest,
    )
    from experiments.overcooked_v2.path_c_r015_full_horizon import (
        build_r015_replay_backend,
    )

    design_protocol = load_r015_design_protocol(design_protocol_path)
    selection = load_mapping(selection_path)
    if selection.get("schema_version") != "path_c_r015_design_selection_report_v2" or (
        selection.get("status") != "selected"
    ) or selection.get("scientific_readout_allowed") is not False:
        raise ValueError("R015 pilot requires a successful design selection report.")
    freeze_manifest = load_mapping(freeze_manifest_path)
    validate_authorized_pilot_freeze_manifest(freeze_manifest)
    preregistration_binding = freeze_manifest.get("text_artifacts", {}).get(
        "preregistration"
    )
    if not isinstance(preregistration_binding, Mapping) or Path(
        str(preregistration_binding.get("path", ""))
    ).resolve() != declared_preregistration or preregistration_binding.get(
        "sha256"
    ) != _file_sha256(declared_preregistration):
        raise ValueError(
            "R015 pilot preregistration differs from its resolved protocol or freeze manifest."
        )
    selected_filter = selection["filter_selection"]["selected_candidate"]
    planning_branches = int(
        selection["planning_selection"]["selected_branch_count"]
    )
    particles = int(selected_filter["particles_per_prototype"])
    timing = str(selected_filter["resampling_timing"])

    support = design_protocol["support"]
    support_path = Path(str(support["partner_support_registration"]))
    if not support_path.is_absolute():
        support_path = (design_protocol_path.parent / support_path).resolve()
    if declared_support != support_path:
        raise ValueError("R015 pilot support path differs from its resolved protocol.")
    support_config = load_r015_partner_support_config(support_path)
    support_spec = R015PartnerSupportSpec.from_mapping(support_config)
    prototype_ids = tuple(candidate.candidate_id for candidate in support_spec.candidates)
    _verify_pilot_input_bindings(
        freeze_manifest=freeze_manifest,
        pilot_protocol_path=config_path,
        design_protocol_path=design_protocol_path,
        selection_path=selection_path,
        support_registration_path=support_path,
        prototype_ids=prototype_ids,
    )
    backend = build_official_r015_production_backend(
        design_protocol,
        support_spec,
        config_base_path=design_protocol_path.parent,
    )
    _verify_backend_freeze_identity(
        backend=backend,
        support_spec=support_spec,
        ego_candidate=support["ego_candidate"],
        freeze_manifest=freeze_manifest,
    )
    executor = backend.full_horizon_executor
    replay_backend = build_r015_replay_backend(
        {"_r015_production_backend": backend}
    )

    sampling = pilot_protocol["sampling"]
    role_isolation = design_protocol.get("role_isolation")
    if not isinstance(role_isolation, Mapping) or (
        str(sampling.get("episode_seed_namespace"))
        != str(role_isolation.get("pilot_episode_namespace"))
    ) or int(sampling.get("root_seed", -1)) != int(
        role_isolation.get("pilot_root_seed", -2)
    ) or str(sampling.get("design_episode_namespace_forbidden")) != str(
        role_isolation.get("design_episode_namespace")
    ) or str(sampling.get("formal_episode_namespace_forbidden")) != str(
        role_isolation.get("formal_episode_namespace")
    ) or int(sampling.get("admission_evaluation_seed_root_forbidden", -1)) != int(
        role_isolation.get("admission_evaluation_seed_root", -2)
    ):
        raise ValueError("R015 pilot seed roles differ from the frozen design protocol.")
    expected_coordinates = [
        (prototype_id, block_index)
        for block_index in range(1, 21)
        for prototype_id in prototype_ids
    ]
    if len(expected_coordinates) != 80 or len(prototype_ids) != 4:
        raise ValueError("R015 pilot requires four prototypes and exactly 80 blocks.")
    if int(sampling.get("paired_blocks_per_prototype", -1)) != 20 or int(
        sampling.get("total_paired_blocks", -1)
    ) != 80:
        raise ValueError("R015 pilot sampling slots changed.")
    episode_seeds = [
        int(
            canonical_sha256(
                [
                    sampling["episode_seed_namespace"],
                    int(sampling["root_seed"]),
                    prototype_id,
                    block_index,
                ]
            )[-8:],
            16,
        )
        for prototype_id, block_index in expected_coordinates
    ]
    if len(set(episode_seeds)) != 80 or int(
        sampling["admission_evaluation_seed_root_forbidden"]
    ) in set(episode_seeds):
        raise ValueError("R015 pilot realized seeds are not role-isolated.")
    if len(
        {
            str(sampling["episode_seed_namespace"]),
            str(sampling["design_episode_namespace_forbidden"]),
            str(sampling["formal_episode_namespace_forbidden"]),
        }
    ) != 3:
        raise ValueError("R015 pilot random-number namespaces overlap.")

    run_binding = {
        "schema_version": PILOT_RUN_BINDING_SCHEMA,
        "scientific_readout_allowed": False,
        "authorization_reference": authorization_reference,
        "pilot_protocol_path": str(config_path),
        "pilot_protocol_sha256": _file_sha256(config_path),
        "design_protocol_path": str(design_protocol_path),
        "design_protocol_sha256": _file_sha256(design_protocol_path),
        "design_selection_report_path": str(selection_path),
        "design_selection_report_sha256": _file_sha256(selection_path),
        "freeze_manifest_path": str(freeze_manifest_path),
        "freeze_manifest_sha256": _file_sha256(freeze_manifest_path),
        "support_registration_path": str(support_path),
        "support_registration_sha256": _file_sha256(support_path),
        "prototype_ids": list(prototype_ids),
        "particles_per_prototype": particles,
        "resampling_timing": timing,
        "planning_branches": planning_branches,
        "sampling": copy.deepcopy(dict(sampling)),
        "realized_episode_seeds_sha256": canonical_sha256(episode_seeds),
        "source_sha256": _source_binding(),
    }
    _bind_or_verify_run(output_dir, run_binding)

    blocks_path = output_dir / "pilot_blocks.jsonl"
    blocks = _load_atomic_block_records(output_dir)
    if blocks_path.exists():
        try:
            ledger = _load_jsonl(blocks_path)
        except ValueError as error:
            if "incomplete line" not in str(error):
                raise
            ledger = []
        if canonical_sha256(ledger) != canonical_sha256(blocks):
            _write_jsonl_atomic(blocks_path, blocks)
    elif blocks:
        _write_jsonl_atomic(blocks_path, blocks)
    existing_coordinates = [
        (str(item.get("partner_prototype_id")), int(item.get("block_index", -1)))
        for item in blocks
    ]
    if existing_coordinates != expected_coordinates[: len(existing_coordinates)]:
        raise ValueError("R015 pilot resume records changed the frozen order.")
    completion_path = output_dir / "pilot_complete.json"
    report_path = output_dir / "pilot_wiring_report.json"
    if completion_path.exists():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if not isinstance(completion, Mapping) or completion.get(
            "schema_version"
        ) != PILOT_COMPLETION_SCHEMA or len(blocks) != 80:
            raise ValueError("R015 pilot completion marker is inconsistent.")
        if not report_path.is_file() or _file_sha256(report_path) != completion.get(
            "report_sha256"
        ) or _file_sha256(blocks_path) != completion.get("pilot_blocks_sha256"):
            raise ValueError("R015 pilot completed evidence changed after sealing.")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(report, Mapping) or report.get("schema_version") != (
            PILOT_REPORT_SCHEMA
        ):
            raise ValueError("R015 pilot sealed report has the wrong schema.")
        return {**dict(report), "report_path": str(report_path)}
    start_time = time.monotonic()
    for block_ordinal, (prototype_id, block_index) in enumerate(
        expected_coordinates[len(blocks) :],
        start=len(blocks) + 1,
    ):
        episode_seed_digest = canonical_sha256(
            [
                sampling["episode_seed_namespace"],
                int(sampling["root_seed"]),
                prototype_id,
                block_index,
            ]
        )
        episode_seed = int(episode_seed_digest[-8:], 16)
        filter_initialization_key = canonical_sha256(
            [
                "r015_pilot_filter_initialization_v2",
                sampling["episode_seed_namespace"],
                int(sampling["root_seed"]),
                block_ordinal,
            ]
        )
        audit_unit_id = canonical_sha256(
            ["r015_pilot_block_v1", prototype_id, block_index, episode_seed]
        )
        arguments = {
            "partner_prototype_id": prototype_id,
            "episode_seed": episode_seed,
            "audit_unit_id": audit_unit_id,
            "filter_initialization_key": filter_initialization_key,
            "particles_per_prototype": particles,
            "resampling_timing": timing,
            "planning_branches": planning_branches,
        }
        produced_started = time.monotonic()
        produced = executor.run_paired_block(**arguments)
        produced_wall_seconds = max(time.monotonic() - produced_started, 1.0e-12)
        if not isinstance(produced, Mapping) or produced.get("schema_version") != (
            PILOT_BLOCK_SCHEMA
        ):
            raise TypeError("R015 pilot executor returned another block schema.")
        trace_recompute, decision_recompute, safety_recompute = (
            _independent_block_replayers(arguments)
        )
        block = copy.deepcopy(dict(produced))
        replay_started = time.monotonic()
        for group in R015_FORMAL_GROUPS:
            verification = verify_r015_trace_manifest(
                produced["groups"][group],
                {
                    "_r015_replay_backend": replay_backend,
                    "_r015_trace_recompute": trace_recompute,
                    "replay_group": group,
                },
            )
            block["groups"][group]["replay_verification"] = verification
        decision_payload = {"consultations": produced["consultations"]}
        safety_payload = {"comparisons": produced["safety_comparisons"]}
        block["probe_decision_replay"] = verify_r015_probe_decision(
            decision_payload,
            {
                "_r015_replay_backend": replay_backend,
                "_r015_probe_decision_recompute": decision_recompute,
            },
        )
        block["safety_replay"] = verify_r015_safety_comparison(
            safety_payload,
            {
                "_r015_replay_backend": replay_backend,
                "_r015_safety_comparison_recompute": safety_recompute,
            },
        )
        replay_wall_seconds = max(time.monotonic() - replay_started, 1.0e-12)
        block["block_index"] = block_index
        block["filter_initialization_key"] = filter_initialization_key
        block["particles_per_prototype"] = particles
        block["resampling_timing"] = timing
        block["planning_branches"] = planning_branches
        block["measured_production_wall_seconds"] = produced_wall_seconds
        block["measured_independent_replay_wall_seconds"] = replay_wall_seconds
        safety_true_transitions = sum(
            int(
                comparison.get("throughput", {}).get(
                    "true_environment_transitions", 0
                )
            )
            for comparison in produced["safety_comparisons"]
        )
        block_true_transitions = (
            int(produced["planning_real_environment_transitions"])
            + int(
                produced.get("execution_throughput", {}).get(
                    "true_environment_transitions", 0
                )
            )
            + safety_true_transitions
        )
        block["measured_true_environment_transitions"] = block_true_transitions
        block["measured_real_transition_throughput_per_second"] = (
            block_true_transitions / produced_wall_seconds
        )
        _write_atomic_block_record(
            output_dir,
            ordinal=block_ordinal,
            block=block,
        )
        _append_jsonl(blocks_path, block)
        blocks.append(block)

    if len(blocks) != 80:
        raise ValueError("R015 pilot did not complete exactly 80 blocks.")
    elapsed = max(time.monotonic() - start_time, 1.0e-12)
    counts = {prototype_id: 0 for prototype_id in prototype_ids}
    spreads = {
        "use_minus_reference": [],
        "use_minus_mask": [],
        "reference_minus_mask": [],
    }
    fork_steps = []
    branch_head_particle_steps = []
    paired_suffix_batched_lane_time_steps = []
    compiled_batch_calls = []
    active_batch_sizes = []
    block_wall_seconds = []
    replay_wall_seconds = []
    block_true_transitions = []
    block_throughput = []
    zero_counts = {
        "candidate_evaluation_count": 0,
        "safety_rejection_count": 0,
        "nonpositive_score_count": 0,
        "window_expiration_count": 0,
    }
    for block_ordinal, block in enumerate(blocks, start=1):
        prototype_id, block_index = expected_coordinates[block_ordinal - 1]
        expected_seed = episode_seeds[block_ordinal - 1]
        expected_audit_unit_id = canonical_sha256(
            ["r015_pilot_block_v1", prototype_id, block_index, expected_seed]
        )
        expected_filter_key = canonical_sha256(
            [
                "r015_pilot_filter_initialization_v2",
                sampling["episode_seed_namespace"],
                int(sampling["root_seed"]),
                block_ordinal,
            ]
        )
        if block.get("schema_version") != PILOT_BLOCK_SCHEMA or (
            block.get("scientific_readout_allowed") is not False
        ) or str(block.get("partner_prototype_id")) != prototype_id or int(
            block.get("block_index", -1)
        ) != block_index or int(block.get("episode_seed", -1)) != expected_seed or (
            block.get("audit_unit_id") != expected_audit_unit_id
        ) or block.get("filter_initialization_key") != expected_filter_key or int(
            block.get("particles_per_prototype", -1)
        ) != particles or str(block.get("resampling_timing")) != timing or int(
            block.get("planning_branches", -1)
        ) != planning_branches:
            raise ValueError("R015 pilot completed block changed its frozen coordinate.")
        counts[prototype_id] += 1
        differences = block["lockstep_verification"]["differences"]
        spreads["use_minus_reference"].append(float(differences["delta_net"]))
        spreads["use_minus_mask"].append(float(differences["delta_response"]))
        spreads["reference_minus_mask"].append(float(differences["delta_cost"]))
        fork_steps.append(int(block["planning_fork_environment_steps"]))
        branch_head_particle_steps.append(
            int(block["planning_branch_head_particle_transitions"])
        )
        paired_suffix_batched_lane_time_steps.append(
            int(block["planning_paired_suffix_batched_lane_time_steps"])
        )
        compiled_batch_calls.append(int(block["planning_compiled_batch_calls"]))
        active_batch_sizes.append(
            [int(value) for value in block["planning_active_batch_sizes"]]
        )
        block_wall_seconds.append(float(block["measured_production_wall_seconds"]))
        replay_wall_seconds.append(
            float(block["measured_independent_replay_wall_seconds"])
        )
        block_true_transitions.append(
            int(block["measured_true_environment_transitions"])
        )
        block_throughput.append(
            float(block["measured_real_transition_throughput_per_second"])
        )
        for key in zero_counts:
            zero_counts[key] += int(block["zero_probe_interpretability_counts"][key])
    count_input_example = {
        "schema_version": "path_c_r015_firing_count_checkpoint_v1",
        "experiment_id": "R015",
        "checkpoint_round_count": 20,
        "firing_counts_by_prototype": {
            prototype_id: sum(
                int(block["probe_fired"])
                for block in blocks
                if block["partner_prototype_id"] == prototype_id
            )
            for prototype_id in prototype_ids
        },
    }
    filter_selected_result = next(
        result
        for result in selection["filter_selection"]["ordered_results"]
        if result["particles_per_prototype"] == particles
        and result["resampling_timing"] == timing
    )
    lockstep_recomputed = []
    for block in blocks:
        recomputed = validate_r015_lockstep_groups(
            block["groups"],
            probe_step=block["probe_step"],
            selected_probe_actions=(
                (str(block["selected_probe_id"]),)
                if block["probe_fired"]
                else ()
            ),
        )
        lockstep_recomputed.append(
            recomputed.get("verified") is True
            and canonical_sha256(recomputed)
            == canonical_sha256(block["lockstep_verification"])
        )
    all_replays_verified = all(
        block["groups"][group]["replay_verification"].get("verified") is True
        and block["probe_decision_replay"].get("verified") is True
        and block["safety_replay"].get("verified") is True
        for block in blocks
        for group in R015_FORMAL_GROUPS
    )
    wiring_checks = {
        "information_isolation": all(
            _information_isolation_verified(block) for block in blocks
        ),
        "filter_normalization_and_repeat_stability": (
            filter_selected_result["passed"] is True
            and all(_online_filter_verified(block) for block in blocks)
        ),
        "response_use_pairing": all(
            _response_pairing_verified(block) for block in blocks
        ),
        "random_key_naming": (
            len({block["audit_unit_id"] for block in blocks}) == 80
            and all(_random_keys_verified(block) for block in blocks)
        ),
        "zero_probe_interpretability": all(
            _zero_probe_explanation_verified(block) for block in blocks
        ),
        "artifact_counts_read_back": counts
        == {prototype_id: 20 for prototype_id in prototype_ids}
        and all_replays_verified,
        "lockstep_structural_zero": all(lockstep_recomputed),
        "shared_firing_indicator": all(
            _shared_firing_indicator_verified(block) for block in blocks
        ),
        "firing_count_input_isolation": _firing_count_input_isolation_verified(
            count_input_example
        ),
    }
    if any(value is not True for value in wiring_checks.values()):
        raise ValueError("R015 pilot failed at least one registered wiring check.")
    total_fork_steps = sum(fork_steps)
    total_production_wall_seconds = sum(block_wall_seconds)
    total_replay_wall_seconds = sum(replay_wall_seconds)
    report = {
        "schema_version": PILOT_REPORT_SCHEMA,
        "scientific_readout_allowed": False,
        "pilot_data": True,
        "formal_interval_includes_pilot": False,
        "scientific_effect_used_for_round_budget": False,
        "empirical_variance_used_for_round_budget": False,
        "selected_filter": copy.deepcopy(dict(selected_filter)),
        "selected_planning_branch_count": planning_branches,
        "partner_prototype_ids": list(prototype_ids),
        "completed_blocks_per_prototype": counts,
        "wiring_checks": wiring_checks,
        "descriptive_paired_difference_spread": {
            key: _spread(values) for key, values in spreads.items()
        },
        "planning_fork_environment_steps_per_block": fork_steps,
        "planning_fork_environment_steps_total": total_fork_steps,
        "planning_real_environment_transitions_per_block": fork_steps,
        "planning_real_environment_transitions_total": total_fork_steps,
        "planning_branch_head_particle_transitions_per_block": (
            branch_head_particle_steps
        ),
        "planning_branch_head_particle_transitions_total": sum(
            branch_head_particle_steps
        ),
        "planning_paired_suffix_batched_lane_time_steps_per_block": (
            paired_suffix_batched_lane_time_steps
        ),
        "planning_paired_suffix_batched_lane_time_steps_total": sum(
            paired_suffix_batched_lane_time_steps
        ),
        "planning_compiled_batch_calls_per_block": compiled_batch_calls,
        "planning_active_batch_sizes_per_block": active_batch_sizes,
        "production_wall_seconds_per_block": block_wall_seconds,
        "independent_replay_wall_seconds_per_block": replay_wall_seconds,
        "production_wall_seconds_total": total_production_wall_seconds,
        "independent_replay_wall_seconds_total": total_replay_wall_seconds,
        "true_environment_transitions_per_block": block_true_transitions,
        "true_environment_transitions_total": sum(block_true_transitions),
        "real_transition_throughput_per_block": block_throughput,
        "measured_total_throughput_environment_transitions_per_second": (
            sum(block_true_transitions)
            / max(total_production_wall_seconds, 1.0e-12)
        ),
        "measured_wall_seconds_this_invocation": elapsed,
        "zero_probe_interpretability_counts": zero_counts,
        "pilot_blocks_path": str(blocks_path),
        "pilot_blocks_sha256": _file_sha256(blocks_path),
        "pilot_run_binding_path": str(output_dir / "pilot_run_binding.json"),
        "pilot_run_binding_sha256": _file_sha256(
            output_dir / "pilot_run_binding.json"
        ),
    }
    _write_json_atomic(report_path, report)
    completion = {
        "schema_version": PILOT_COMPLETION_SCHEMA,
        "scientific_readout_allowed": False,
        "completed_block_count": 80,
        "run_binding_sha256": canonical_sha256(run_binding),
        "pilot_blocks_path": str(blocks_path),
        "pilot_blocks_sha256": _file_sha256(blocks_path),
        "report_path": str(report_path),
        "report_sha256": _file_sha256(report_path),
        "formal_audit_started": False,
    }
    _write_json_atomic(completion_path, completion)
    return {**report, "report_path": str(report_path)}


__all__ = ["run_r015_pilot"]
