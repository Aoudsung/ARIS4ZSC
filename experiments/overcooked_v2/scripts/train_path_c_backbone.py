#!/usr/bin/env python3
"""Run the Path C recurrent self-play backbone from a resolved YAML config."""

from __future__ import annotations

import argparse
import json

from experiments.overcooked_v2.path_c_backbone_ppo import run_backbone_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    print(json.dumps(run_backbone_training(arguments.config), sort_keys=True))


if __name__ == "__main__":
    main()
