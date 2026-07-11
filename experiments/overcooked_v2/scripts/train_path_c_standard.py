#!/usr/bin/env python3
"""Train one standard-path seed from an explicit configuration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.overcooked_v2.path_c_standard_training import (
    StandardPathCTrainer,
    load_standard_training_config,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke-seed", type=int)
    parser.add_argument("--smoke-output-dir")
    args = parser.parse_args()
    config = load_standard_training_config(args.config)
    if args.smoke_seed is not None or args.smoke_output_dir is not None:
        if config.get("run_kind") != "smoke":
            raise ValueError("Command-line seed/output overrides are smoke-only.")
        if args.smoke_seed is None or args.smoke_output_dir is None:
            raise ValueError("Smoke seed and output directory must be supplied together.")
        config["seed"] = int(args.smoke_seed)
        config["output_dir"] = str(args.smoke_output_dir)
    manifest = StandardPathCTrainer(config).run()
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
