#!/usr/bin/env python3
"""一次性生成 R015 冻结机器制品；不运行环境或正式数据。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from experiments.overcooked_v2.path_c_r015_artifacts import (
    build_r015_machine_artifact_bundle,
    validate_r015_machine_artifact_bundle,
)


def _load_mapping(path: Path):
    text = path.read_text(encoding="utf-8")
    value = (
        json.loads(text)
        if path.suffix.lower() == ".json"
        else yaml.safe_load(text)
    )
    if not isinstance(value, dict):
        raise TypeError("R015 freeze inputs must be a mapping.")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="生成并核验 R015 过滤、规划、重放、安全和正式抽样机器清单。"
    )
    parser.add_argument("--freeze-inputs", type=Path)
    parser.add_argument("--preregistration", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--formal-root-seed", type=int)
    parser.add_argument("--validate", type=Path)
    arguments = parser.parse_args()
    if arguments.validate is not None:
        print(
            json.dumps(
                validate_r015_machine_artifact_bundle(arguments.validate),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return
    required = {
        "freeze-inputs": arguments.freeze_inputs,
        "preregistration": arguments.preregistration,
        "output-directory": arguments.output_directory,
        "repository-root": arguments.repository_root,
        "formal-root-seed": arguments.formal_root_seed,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(
            "R015 machine artifact arguments missing: " + ", ".join(missing)
        )
    result = build_r015_machine_artifact_bundle(
        freeze_inputs=_load_mapping(arguments.freeze_inputs),
        preregistration_path=arguments.preregistration,
        output_directory=arguments.output_directory,
        repository_root=arguments.repository_root,
        formal_root_seed=arguments.formal_root_seed,
    )
    print(
        json.dumps(
            {
                "bundle_manifest_path": result["bundle_manifest_path"],
                "updated_freeze_inputs_path": result["updated_freeze_inputs_path"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
