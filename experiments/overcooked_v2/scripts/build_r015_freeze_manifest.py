#!/usr/bin/env python3
"""从真实 R015 产物完成两遍冻结清单装配；不运行试点或正式审计。"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from experiments.overcooked_v2.path_c_r015_design import (
    FREEZE_ASSEMBLY_REPORT_SCHEMA,
    apply_preregistration_fill_values,
    authorize_formal_freeze_after_pilot,
    authorize_pilot_freeze_manifest,
    bind_completed_pilot_report,
    bind_completed_pilot_report_to_preregistration,
    build_pending_freeze_manifest,
    canonical_sha256,
    file_sha256,
    freeze_inputs_with_preregistration_path,
    load_mapping,
    preregistration_fill_values,
    sign_ready_freeze_manifest,
    validate_prefilled_preregistration,
    validate_ready_freeze_manifest,
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _authorization_quote(arguments: argparse.Namespace) -> str:
    if arguments.authorization_quote is not None:
        return str(arguments.authorization_quote).strip()
    if arguments.authorization_quote_file is not None:
        return arguments.authorization_quote_file.read_text(encoding="utf-8").strip()
    raise ValueError("R015 freeze authorization requires a nonempty signed quote.")


def _verify_first_pass_preregistration_identity(
    inputs: Mapping[str, Any], preregistration_path: Path
) -> None:
    text_artifacts = inputs.get("text_artifacts")
    if not isinstance(text_artifacts, Mapping):
        raise ValueError("R015 freeze inputs lack text_artifacts.")
    raw_binding = text_artifacts.get("preregistration")
    if isinstance(raw_binding, Mapping):
        if set(raw_binding) != {"path", "sha256"}:
            raise ValueError("R015 preregistration binding has the wrong fields.")
        bound_path = Path(str(raw_binding["path"])).resolve()
        bound_sha256 = raw_binding.get("sha256")
    else:
        bound_path = Path(str(raw_binding)).resolve()
        bound_sha256 = file_sha256(bound_path)
    actual_path = preregistration_path.resolve()
    if bound_path != actual_path or bound_sha256 != file_sha256(actual_path):
        raise ValueError(
            "R015 --preregistration differs from the first-pass input binding."
        )


def _without_preregistration_text_binding(
    manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    result = copy.deepcopy(dict(manifest))
    text_artifacts = dict(result["text_artifacts"])
    text_artifacts.pop("preregistration")
    result["text_artifacts"] = text_artifacts
    return result


def _require_two_pass_outputs(arguments: argparse.Namespace) -> None:
    required = {
        "preregistration": arguments.preregistration,
        "preregistration_fill_values_output": (
            arguments.preregistration_fill_values_output
        ),
        "filled_preregistration_output": arguments.filled_preregistration_output,
        "ready_manifest_output": arguments.ready_manifest_output,
        "assembly_report_output": arguments.assembly_report_output,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(
            "Two-pass R015 assembly lacks output argument(s): " + ", ".join(missing)
        )
    if arguments.authorize_pilot_freeze and arguments.frozen_manifest_output is None:
        raise ValueError("An authorized pilot freeze requires --frozen-manifest-output.")


def _run_two_pass(arguments: argparse.Namespace) -> None:
    _require_two_pass_outputs(arguments)
    quote = _authorization_quote(arguments)
    inputs = load_mapping(arguments.inputs)
    _verify_first_pass_preregistration_identity(inputs, arguments.preregistration)

    first_manifest = build_pending_freeze_manifest(inputs)
    _write_json(arguments.output, first_manifest)
    fill_values = preregistration_fill_values(first_manifest)
    _write_yaml(arguments.preregistration_fill_values_output, fill_values)

    preregistration = load_mapping(arguments.preregistration)
    filled_preregistration = apply_preregistration_fill_values(
        preregistration,
        fill_values,
    )
    prefilled_validation = validate_prefilled_preregistration(filled_preregistration)
    _write_yaml(arguments.filled_preregistration_output, filled_preregistration)

    second_inputs = freeze_inputs_with_preregistration_path(
        inputs,
        arguments.filled_preregistration_output,
    )
    second_manifest_unsigned = build_pending_freeze_manifest(second_inputs)
    if _without_preregistration_text_binding(first_manifest) != (
        _without_preregistration_text_binding(second_manifest_unsigned)
    ):
        raise ValueError(
            "R015 second pass changed an object other than the filled preregistration."
        )
    ready_manifest = sign_ready_freeze_manifest(
        second_manifest_unsigned,
        authorization_quote=quote,
    )
    ready_validation = validate_ready_freeze_manifest(ready_manifest)
    _write_json(arguments.ready_manifest_output, ready_manifest)

    assembly_report = {
        "schema_version": FREEZE_ASSEMBLY_REPORT_SCHEMA,
        "scientific_readout_allowed": False,
        "first_pass_manifest": {
            "path": str(arguments.output.resolve()),
            "sha256": file_sha256(arguments.output),
        },
        "preregistration_fill_values": {
            "path": str(arguments.preregistration_fill_values_output.resolve()),
            "sha256": file_sha256(arguments.preregistration_fill_values_output),
        },
        "filled_preregistration": {
            "path": str(arguments.filled_preregistration_output.resolve()),
            "sha256": file_sha256(arguments.filled_preregistration_output),
            "validation": prefilled_validation,
        },
        "ready_manifest": {
            "path": str(arguments.ready_manifest_output.resolve()),
            "sha256": file_sha256(arguments.ready_manifest_output),
            "validation": ready_validation,
        },
        "type_b_authorization_quote_sha256": ready_manifest["type_b_signoff"][
            "authorization_quote_sha256"
        ],
        "pilot_wiring_report_status": "not_run_in_registered_sequence",
        "preregistration_formal_readout_enabled": False,
    }
    if arguments.authorize_pilot_freeze:
        frozen_manifest = authorize_pilot_freeze_manifest(
            ready_manifest,
            authorization_quote=quote,
        )
        _write_json(arguments.frozen_manifest_output, frozen_manifest)
        assembly_report["authorized_pilot_freeze_manifest"] = {
            "path": str(arguments.frozen_manifest_output.resolve()),
            "sha256": file_sha256(arguments.frozen_manifest_output),
        }
    _write_json(arguments.assembly_report_output, assembly_report)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "装配 R015 冻结清单。默认只产出第一遍待签清单；--two-pass 会填充预登记、"
            "重算其摘要并产出第二遍待授权清单。"
        )
    )
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--two-pass", action="store_true")
    parser.add_argument("--preregistration", type=Path)
    parser.add_argument("--preregistration-fill-values-output", type=Path)
    parser.add_argument("--filled-preregistration-output", type=Path)
    parser.add_argument("--ready-manifest-output", type=Path)
    parser.add_argument("--assembly-report-output", type=Path)
    authorization = parser.add_mutually_exclusive_group()
    authorization.add_argument("--authorization-quote")
    authorization.add_argument("--authorization-quote-file", type=Path)
    parser.add_argument(
        "--authorize-pilot-freeze",
        action="store_true",
        help=(
            "显式把第二遍清单翻为只供已登记试点使用的 frozen；预登记和科学读取仍不翻转。"
        ),
    )
    parser.add_argument("--frozen-manifest-output", type=Path)
    mode.add_argument(
        "--validate-ready",
        action="store_true",
        help="核验一个待授权翻转清单，并把核验结果写入 --output。",
    )
    mode.add_argument(
        "--bind-pilot-report",
        type=Path,
        help="把已完成且九项检查全通过的试点报告补入 --inputs 指向的 frozen 清单。",
    )
    parser.add_argument(
        "--pilot-preregistration-output",
        type=Path,
        help="与 --bind-pilot-report 同用，把试点报告绑定补入仍关闭科学读取的预登记。",
    )
    mode.add_argument("--authorize-formal-freeze", action="store_true")
    parser.add_argument("--formal-dataset-path", type=Path)
    parser.add_argument(
        "--firing-count-checkpoint-path",
        action="append",
        type=Path,
        dest="firing_count_checkpoint_paths",
    )
    parser.add_argument("--formal-view-record-path", type=Path)
    parser.add_argument("--formal-preregistration-output", type=Path)
    arguments = parser.parse_args()

    if arguments.authorize_pilot_freeze and not arguments.two_pass:
        raise ValueError("--authorize-pilot-freeze requires --two-pass.")
    if arguments.frozen_manifest_output is not None and not (
        arguments.two_pass and arguments.authorize_pilot_freeze
    ):
        raise ValueError(
            "--frozen-manifest-output requires --two-pass and --authorize-pilot-freeze."
        )
    if arguments.pilot_preregistration_output is not None and (
        arguments.bind_pilot_report is None
    ):
        raise ValueError("--pilot-preregistration-output requires --bind-pilot-report.")
    formal_only = (
        arguments.formal_dataset_path,
        arguments.firing_count_checkpoint_paths,
        arguments.formal_view_record_path,
        arguments.formal_preregistration_output,
    )
    if any(value is not None for value in formal_only) and not (
        arguments.authorize_formal_freeze
    ):
        raise ValueError("Formal output arguments require --authorize-formal-freeze.")

    if arguments.authorize_formal_freeze:
        required = {
            "preregistration": arguments.preregistration,
            "formal_dataset_path": arguments.formal_dataset_path,
            "firing_count_checkpoint_paths": arguments.firing_count_checkpoint_paths,
            "formal_view_record_path": arguments.formal_view_record_path,
            "formal_preregistration_output": arguments.formal_preregistration_output,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                "Formal R015 freeze lacks argument(s): " + ", ".join(missing)
            )
        quote = _authorization_quote(arguments)
        completed_manifest = load_mapping(arguments.inputs)
        pilot_bound_preregistration = load_mapping(arguments.preregistration)
        frozen_manifest, frozen_preregistration = authorize_formal_freeze_after_pilot(
            completed_manifest,
            pilot_bound_preregistration,
            authorization_quote=quote,
            formal_dataset_path=arguments.formal_dataset_path,
            firing_count_checkpoint_paths=arguments.firing_count_checkpoint_paths,
            formal_view_record_path=arguments.formal_view_record_path,
        )
        _write_yaml(arguments.formal_preregistration_output, frozen_preregistration)
        written_preregistration = load_mapping(arguments.formal_preregistration_output)
        if canonical_sha256(written_preregistration) != frozen_manifest.get(
            "formal_preregistration_canonical_sha256"
        ):
            raise ValueError(
                "R015 formal preregistration serialization changed its frozen content."
            )
        _write_json(arguments.output, frozen_manifest)
        return
    if arguments.validate_ready:
        manifest = load_mapping(arguments.inputs)
        _write_json(arguments.output, validate_ready_freeze_manifest(manifest))
        return
    if arguments.bind_pilot_report is not None:
        if (
            arguments.pilot_preregistration_output is not None
            and arguments.preregistration is None
        ):
            raise ValueError("Pilot preregistration output requires --preregistration.")
        manifest = load_mapping(arguments.inputs)
        bound = bind_completed_pilot_report(
            manifest,
            pilot_report_path=arguments.bind_pilot_report,
        )
        if arguments.pilot_preregistration_output is not None:
            preregistration = load_mapping(arguments.preregistration)
            pilot_bound_preregistration = (
                bind_completed_pilot_report_to_preregistration(
                    preregistration,
                    bound,
                )
            )
            _write_yaml(
                arguments.pilot_preregistration_output,
                pilot_bound_preregistration,
            )
            if canonical_sha256(
                load_mapping(arguments.pilot_preregistration_output)
            ) != canonical_sha256(pilot_bound_preregistration):
                raise ValueError(
                    "R015 pilot preregistration serialization changed its binding."
                )
        _write_json(arguments.output, bound)
        return
    if arguments.two_pass:
        _run_two_pass(arguments)
        return

    inputs = load_mapping(arguments.inputs)
    manifest = build_pending_freeze_manifest(inputs)
    _write_json(arguments.output, manifest)
    if arguments.preregistration_fill_values_output is not None:
        _write_yaml(
            arguments.preregistration_fill_values_output,
            preregistration_fill_values(manifest),
        )


if __name__ == "__main__":
    main()
