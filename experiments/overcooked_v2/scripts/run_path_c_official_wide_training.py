#!/usr/bin/env python3
"""Execute one registered official Test Time Wide training run."""

from __future__ import annotations

import argparse

from experiments.overcooked_v2.official.overcooked_v2_experiments_wide_adapter import (
    run_official_wide_production_training,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    result = run_official_wide_production_training(arguments.config)
    print(f"output_dir={result['output_dir']}")
    print(f"completed_environment_steps={result['completed_environment_steps']}")
    print(f"completed_episode_count={result['completed_episode_count']}")
    print(f"metric_record_count={result['metric_record_count']}")
    print(f"checkpoint_sha256={result['checkpoint_sha256']}")
    print(f"model_weights_sha256={result['model_weights_sha256']}")
    print(f"training_run_id={result['training_run_id']}")


if __name__ == "__main__":
    main()
