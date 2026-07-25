#!/usr/bin/env python3
"""经冻结抽样清单运行 R015 正式轮次；不复用设计或试点数据。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Mapping

import yaml

from experiments.overcooked_v2.path_c_r015_pipeline import (
    build_official_r015_formal_producer,
    build_r015_formal_runtime_binding_sha256,
)
from experiments.overcooked_v2.path_c_r015_runtime import (
    run_r015_formal,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="运行冻结的 R015 正式轮次。")
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument(
        "--frozen-manifest",
        type=Path,
        help="最终冻结清单；省略时读取预登记同目录的 formal_frozen_manifest.json。",
    )
    parser.add_argument("--runtime-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--authorized", action="store_true")
    arguments = parser.parse_args()
    if not arguments.authorized:
        raise PermissionError("R015 formal launcher requires --authorized.")
    if os.environ.get("JAX_PLATFORMS") != "cuda,cpu":
        raise RuntimeError("R015 formal execution requires JAX_PLATFORMS=cuda,cpu.")

    preregistration_path = arguments.preregistration.resolve()
    frozen_manifest_path = (
        arguments.frozen_manifest.resolve()
        if arguments.frozen_manifest is not None
        else preregistration_path.parent / "formal_frozen_manifest.json"
    )
    runtime_config_path = arguments.runtime_config.resolve()
    raw_config = yaml.safe_load(runtime_config_path.read_text(encoding="utf-8"))
    if not isinstance(raw_config, Mapping):
        raise ValueError("R015 formal runtime config must contain one mapping.")

    producer = build_official_r015_formal_producer(
        runtime_config=raw_config,
        runtime_config_path=runtime_config_path,
        preregistration_path=preregistration_path,
        frozen_manifest_path=frozen_manifest_path,
    )
    runtime_binding_sha256 = build_r015_formal_runtime_binding_sha256(
        runtime_config_path,
        additional_source_paths=(Path(__file__).resolve(),),
    )
    result = run_r015_formal(
        preregistration_path,
        producer=producer,
        output_directory=arguments.output_dir.resolve(),
        runtime_binding_sha256=runtime_binding_sha256,
        maximum_mechanical_attempts=32,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
