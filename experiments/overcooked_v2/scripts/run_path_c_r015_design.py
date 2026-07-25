#!/usr/bin/env python3
"""经 CUDA 12 包装器启动 R015 设计数据；默认配置本身仍未授权。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from experiments.overcooked_v2.path_c_r015_design import (
    load_r015_design_protocol,
    run_r015_design,
    run_r015_filter_device_equivalence,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="运行已授权的 R015 设计数据链。")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--filter-candidate-particles", type=int)
    parser.add_argument("--filter-candidate-timing")
    parser.add_argument("--filter-device-equivalence-output", type=Path)
    parser.add_argument("--filter-device-diagnostic-steps", type=int)
    parser.add_argument("--authorized", action="store_true")
    arguments = parser.parse_args()
    if not arguments.authorized:
        raise PermissionError("R015 design launcher requires --authorized.")
    if os.environ.get("JAX_PLATFORMS") != "cuda,cpu":
        raise RuntimeError("R015 design requires JAX_PLATFORMS=cuda,cpu.")
    config = dict(load_r015_design_protocol(arguments.config))
    config["_protocol_path"] = str(arguments.config.resolve())
    config["_output_dir"] = str(arguments.output_dir.resolve())
    config["_execution_authorized"] = True
    filter_arguments = (
        arguments.filter_candidate_particles,
        arguments.filter_candidate_timing,
    )
    if (filter_arguments[0] is None) is not (filter_arguments[1] is None):
        raise ValueError("R015 filter-only launch requires particles and timing together.")
    if filter_arguments[0] is not None:
        config["_filter_candidate_only"] = filter_arguments
    if arguments.filter_device_equivalence_output is not None:
        if filter_arguments[0] is not None:
            raise ValueError("Equivalence mode fixes the first registered candidate.")
        config["_equivalence_output_path"] = str(
            arguments.filter_device_equivalence_output.resolve()
        )
        if arguments.filter_device_diagnostic_steps is not None:
            config["_filter_device_diagnostic_steps"] = (
                arguments.filter_device_diagnostic_steps
            )
        result = run_r015_filter_device_equivalence(config)
    elif arguments.filter_device_diagnostic_steps is not None:
        raise ValueError("Filter diagnostics require an explicit output path.")
    else:
        result = run_r015_design(config)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
