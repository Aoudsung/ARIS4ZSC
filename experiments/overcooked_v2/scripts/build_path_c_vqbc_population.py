#!/usr/bin/env python3
"""Build the checked ten-policy manifest used by VQBC evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.path_c.vqbc.evaluation import build_population_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--population-id", required=True)
    parser.add_argument(
        "--layout",
        choices=("test_time_simple", "test_time_wide"),
        required=True,
    )
    parser.add_argument(
        "--resolved-config",
        type=Path,
        action="append",
        required=True,
        help="Repeat exactly ten times in outer-unit order zero through nine.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        required=True,
        help="Repeat exactly ten times in outer-unit order zero through nine.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    build_population_manifest(
        args.manifest,
        population_id=args.population_id,
        layout=args.layout,
        resolved_config_paths=args.resolved_config,
        checkpoint_paths=args.checkpoint,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
