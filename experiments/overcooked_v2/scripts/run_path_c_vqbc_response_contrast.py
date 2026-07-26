#!/usr/bin/env python3
"""Run the matched VQBC response-use and response-mask contrast."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.overcooked_v2.model_dock.vqbc_response_contrast_runtime import (
    run_development_response_contrast,
    run_response_contrast,
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
                "Population contrast does not accept development arguments."
            )
        run_response_contrast(
            load_population(args.population_manifest.resolve())
        )
        return
    if args.checkpoint is None:
        parser.error("Development contrast requires --checkpoint.")
    run_development_response_contrast(
        resolved_config_path=args.development_config.resolve(),
        checkpoint_path=args.checkpoint.resolve(),
        output_root=(
            None if args.output_root is None else args.output_root.resolve()
        ),
    )


if __name__ == "__main__":
    main()
