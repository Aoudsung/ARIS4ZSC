#!/usr/bin/env python3
"""执行登记的官方 Other-Play 训练，并保留 seed 999 验收的原始源码绑定。"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Mapping

from experiments.overcooked_v2.official import (
    overcooked_v2_experiments_adapter as official_adapter,
)


_BASE_REFERENCE_COMPARISON = official_adapter.reference_configuration_comparison


def other_play_reference_comparison(
    resolved_config: Mapping[str, Any],
) -> dict[str, Any]:
    """核对同一官方网络，同时保留 Other-Play 登记的 64 个并行环境。"""

    model = resolved_config.get("model")
    environment = resolved_config.get("env")
    if not isinstance(model, Mapping) or not isinstance(environment, Mapping):
        raise ValueError("Other-Play resolved configuration lacks model or environment.")
    kwargs = environment.get("ENV_KWARGS")
    if not isinstance(kwargs, Mapping):
        raise ValueError("Other-Play resolved configuration lacks environment kwargs.")
    registered_other_play = (
        model.get("NUM_ENVS") == 64
        and list(kwargs.get("op_ingredient_permutations", ())) == [0, 1]
    )
    comparison_config = copy.deepcopy(dict(resolved_config))
    comparison_config["model"]["NUM_ENVS"] = 256
    comparison = _BASE_REFERENCE_COMPARISON(comparison_config)
    comparison["checks"]["registered_other_play_configuration"] = (
        registered_other_play
    )
    comparison["matches_accepted_reference"] = all(
        comparison["checks"].values()
    )
    if not registered_other_play:
        comparison["material_differences"].append(
            {
                "field": "registered_other_play_configuration",
                "accepted_reference": "64 environments with independent ingredient 0/1 renaming",
                "candidate": "different",
            }
        )
    return comparison


def run_registered_other_play_training(
    launch_config_path: str | Path,
) -> dict[str, Any]:
    """Use the common production runner with the variant-correct reference check."""

    original_comparison = official_adapter.reference_configuration_comparison
    original_dependencies = official_adapter.official_source_dependency_records
    this_file = Path(__file__).resolve()

    def dependencies(experiment_variant: str) -> list[dict[str, str]]:
        records = list(original_dependencies(experiment_variant))
        if experiment_variant == "rnn-op":
            records.append(
                {
                    "path": str(this_file),
                    "sha256": official_adapter._file_sha256(this_file),
                }
            )
            records.sort(key=lambda item: item["path"])
        return records

    official_adapter.reference_configuration_comparison = (
        other_play_reference_comparison
    )
    official_adapter.official_source_dependency_records = dependencies
    try:
        return official_adapter.run_official_production_training(
            Path(launch_config_path).resolve()
        )
    finally:
        official_adapter.reference_configuration_comparison = original_comparison
        official_adapter.official_source_dependency_records = original_dependencies


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    result = run_registered_other_play_training(arguments.config)
    print(f"output_dir={result['output_dir']}")
    print(f"completed_environment_steps={result['completed_environment_steps']}")
    print(f"completed_episode_count={result['completed_episode_count']}")
    print(f"metric_record_count={result['metric_record_count']}")
    print(f"checkpoint_sha256={result['checkpoint_sha256']}")
    print(f"model_weights_sha256={result['model_weights_sha256']}")
    print(f"training_run_id={result['training_run_id']}")


if __name__ == "__main__":
    main()
