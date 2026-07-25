#!/usr/bin/env python3
"""经 CUDA 12 包装器启动可恢复的 R015 80 块完整接线试点。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from experiments.overcooked_v2.path_c_r015_pilot import run_r015_pilot


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "加载冻结五 checkpoint 后端，运行、独立重放并原子保存 R015 80 块试点。"
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--design-protocol", required=True, type=Path)
    parser.add_argument("--design-selection-report", required=True, type=Path)
    parser.add_argument("--freeze-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--authorization-reference",
        default="user_message_2026-07-15_r015_design_freeze_pilot",
        help="写入不可变运行绑定的既有授权引用；不改变试点或正式统计。",
    )
    parser.add_argument("--authorized", action="store_true")
    arguments = parser.parse_args()
    if not arguments.authorized:
        raise PermissionError("R015 pilot launcher requires --authorized.")
    if os.environ.get("JAX_PLATFORMS") != "cuda,cpu":
        raise RuntimeError("R015 pilot requires JAX_PLATFORMS=cuda,cpu.")
    result = run_r015_pilot(
        {
            "_config_path": str(arguments.config.resolve()),
            "_design_protocol_path": str(arguments.design_protocol.resolve()),
            "_design_selection_report_path": str(
                arguments.design_selection_report.resolve()
            ),
            "_freeze_manifest_path": str(arguments.freeze_manifest.resolve()),
            "_output_dir": str(arguments.output_dir.resolve()),
            "_execution_authorized": True,
            "_authorization_reference": arguments.authorization_reference,
        }
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
