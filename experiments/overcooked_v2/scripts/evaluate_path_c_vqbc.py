#!/usr/bin/env python3
"""Run four matched standard matrices for one ten-policy VQBC population."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.overcooked_v2.model_dock.vqbc_evaluation_runtime import (
    run_development_evaluation,
    run_standard_evaluation,
)
from src.path_c.vqbc.evaluation import load_population


def main() -> None:
    parser = argparse.ArgumentParser()
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--population-manifest", type=Path)
    inputs.add_argument("--development-config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if args.population_manifest is not None:
        if args.checkpoint is not None or args.output_root is not None:
            parser.error(
                "Population evaluation does not accept development arguments."
            )
        run_standard_evaluation(
            load_population(args.population_manifest.resolve())
        )
        return
    if args.checkpoint is None:
        parser.error("Development evaluation requires --checkpoint.")
    run_development_evaluation(
        resolved_config_path=args.development_config.resolve(),
        checkpoint_path=args.checkpoint.resolve(),
        output_root=(
            None if args.output_root is None else args.output_root.resolve()
        ),
    )


if __name__ == "__main__":
    main()
