#!/usr/bin/env python3
"""用一个命令串行运行 R015 设计、冻结、试点和正式轮次。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from experiments.overcooked_v2.path_c_r015_design import load_r015_design_protocol
from experiments.overcooked_v2.path_c_r015_pipeline import (
    R015_PIPELINE_STAGE_ORDER,
    load_r015_pipeline_config,
    materialize_r015_freeze_inputs,
    run_r015_pipeline,
)


CONFIG_DIRECTORY = Path(__file__).resolve().parents[1] / "configs"
DEFAULT_DESIGN_PROTOCOL = CONFIG_DIRECTORY / "path_c_r015_design_data_protocol.yaml"
DEFAULT_PREREGISTRATION = CONFIG_DIRECTORY / "path_c_r015_preregistration.yaml"
DEFAULT_PARTNER_SUPPORT = CONFIG_DIRECTORY / "path_c_r015_partner_support_simple.yaml"
DEFAULT_PILOT_PROTOCOL = CONFIG_DIRECTORY / "path_c_r015_pilot.yaml"


def _write_invocation_config(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    if path.exists():
        existing = yaml.safe_load(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("R015 恢复目录已绑定另一组统一入口路径。")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_derived_freeze_inputs(path: Path, payload: dict[str, object]) -> None:
    """原子保存自动导出的冻结输入；恢复时拒绝静默改写。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("R015 恢复目录中的自动冻结输入已经改变。")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _direct_config(arguments: argparse.Namespace) -> Path:
    if arguments.output_root is None or arguments.authorization_file is None:
        missing = []
        if arguments.output_root is None:
            missing.append("output_root")
        if arguments.authorization_file is None:
            missing.append("authorization_file")
        raise ValueError("R015 直接入口缺少路径：" + ", ".join(missing))
    output_root = arguments.output_root.resolve()
    design_protocol = (
        arguments.design_protocol or DEFAULT_DESIGN_PROTOCOL
    ).resolve()
    partner_support = (
        arguments.partner_support_registration or DEFAULT_PARTNER_SUPPORT
    ).resolve()
    preregistration = (
        arguments.preregistration_template or DEFAULT_PREREGISTRATION
    ).resolve()
    pilot_protocol = (
        arguments.pilot_protocol or DEFAULT_PILOT_PROTOCOL
    ).resolve()
    formal_runtime_config = (
        arguments.formal_runtime_config.resolve()
        if arguments.formal_runtime_config is not None
        else output_root / "pipeline_formal_runtime.generated.yaml"
    )
    if arguments.formal_runtime_config is None:
        # 正式运行读取的主体、四伙伴和环境合同与设计协议相同。第一次调用就把
        # 加载器已解析的绝对 checkpoint 路径保存下来；这样先运行设计、以后再补入
        # 后续授权时，统一运行绑定不会因为新增一份运行配置而改变。
        runtime_payload = dict(load_r015_design_protocol(design_protocol))
        runtime_support = dict(runtime_payload["support"])
        runtime_support["partner_support_registration"] = str(partner_support)
        runtime_payload["support"] = runtime_support
        _write_invocation_config(formal_runtime_config, runtime_payload)
    freeze_inputs = (
        arguments.freeze_inputs.resolve()
        if arguments.freeze_inputs is not None
        else output_root / "pipeline_freeze_inputs.generated.json"
    )
    if arguments.freeze_inputs is None:
        # 冻结输入只由已有登记、训练清单和未来设计选择报告的固定路径构成，
        # 不读取设计结果。因此第一次调用即可绑定，后续阶段无需改写调用配置。
        derived = materialize_r015_freeze_inputs(
            design_protocol_path=design_protocol,
            partner_support_registration_path=partner_support,
            design_selection_report_path=(
                output_root / "design" / "design_selection_report.json"
            ),
        )
        _write_derived_freeze_inputs(freeze_inputs, dict(derived))
    direct = {
        "design_protocol": design_protocol,
        "freeze_inputs": freeze_inputs,
        "preregistration_template": preregistration,
        "partner_support_registration": partner_support,
        "pilot_protocol": pilot_protocol,
        "formal_runtime_config": formal_runtime_config,
        "output_root": output_root,
    }
    paths = {
        name: str(value.resolve())
        for name, value in direct.items()
        if value is not None
    }
    payload: dict[str, object] = {
        "schema_version": "path_c_r015_pipeline_v1",
        "paths": paths,
        "authorization_file": str(arguments.authorization_file.resolve()),
    }
    config_path = output_root / "pipeline_invocation_config.yaml"
    _write_invocation_config(config_path, payload)
    return config_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "运行统一且可恢复的 R015 流水线。默认不执行；授权必须同时存在于"
            "配置或授权文件，并由本次命令显式确认。"
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="可选统一配置；省略时使用下面的现有配置和产物路径。",
    )
    parser.add_argument(
        "--design-protocol",
        type=Path,
        help="省略时使用仓库内登记的 R015 设计协议。",
    )
    parser.add_argument(
        "--freeze-inputs",
        type=Path,
        help=(
            "可选人工准备的冻结输入；省略时从设计协议、支持登记和五份训练清单"
            "自动生成。"
        ),
    )
    parser.add_argument(
        "--preregistration-template",
        type=Path,
        help="省略时使用仓库内登记的 R015 预登记模板。",
    )
    parser.add_argument(
        "--partner-support-registration",
        type=Path,
        help=(
            "逐项入口使用的现有伙伴支持登记路径；会与设计协议、冻结输入、"
            "试点协议、预登记和正式运行配置逐项核对。"
        ),
    )
    parser.add_argument(
        "--pilot-protocol",
        type=Path,
        help="省略时使用仓库内登记的 80 块试点协议。",
    )
    parser.add_argument(
        "--formal-runtime-config",
        type=Path,
        help="省略时复用已登记设计协议中的同一主体、伙伴和环境配置。",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--authorization-file", type=Path)
    parser.add_argument(
        "--through",
        choices=R015_PIPELINE_STAGE_ORDER,
        required=True,
        help="本次最多运行到 design、freeze、pilot 或 formal。",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从摘要仍一致的最后一个完整阶段恢复。",
    )
    parser.add_argument(
        "--formal-root-seed",
        type=int,
        help=(
            "正式抽样清单的数据前根 seed；运行到 freeze、pilot 或 formal 时必填，"
            "design-only 不要求。"
        ),
    )
    parser.add_argument(
        "--authorized",
        action="store_true",
        help="确认本次命令可使用配置中逐阶段记录的授权；本参数本身不新增授权。",
    )
    arguments = parser.parse_args()
    if not arguments.authorized:
        raise PermissionError("R015 统一入口默认不执行；请显式传入 --authorized。")
    if arguments.through != "design" and arguments.formal_root_seed is None:
        raise ValueError("运行到 freeze 及后续阶段必须显式传入 --formal-root-seed。")
    if os.environ.get("JAX_PLATFORMS") != "cuda,cpu":
        raise RuntimeError("R015 统一入口要求 JAX_PLATFORMS=cuda,cpu。")
    direct_values = (
        arguments.design_protocol,
        arguments.freeze_inputs,
        arguments.preregistration_template,
        arguments.partner_support_registration,
        arguments.pilot_protocol,
        arguments.formal_runtime_config,
        arguments.output_root,
        arguments.authorization_file,
    )
    if arguments.config is not None and any(
        value is not None for value in direct_values
    ):
        raise ValueError("--config 不能与逐项路径同时使用。")
    config_path = (
        arguments.config.resolve()
        if arguments.config is not None
        else _direct_config(arguments)
    )
    result = run_r015_pipeline(
        load_r015_pipeline_config(config_path),
        config_path=config_path,
        through=arguments.through,
        resume=arguments.resume,
        execution_authorized=True,
        formal_root_seed=arguments.formal_root_seed,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
