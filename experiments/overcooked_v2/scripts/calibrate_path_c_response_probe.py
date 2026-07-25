#!/usr/bin/env python3
"""Collect no-intervention partner-response scores and choose one threshold."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.overcooked_v2.path_c_adaptation import (
    run_adaptation_training,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    summary = run_adaptation_training(arguments.config)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
