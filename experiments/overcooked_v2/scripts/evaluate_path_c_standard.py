#!/usr/bin/env python3
"""Evaluate standard Path C checkpoints with the published pairing protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.overcooked_v2.path_c_standard_evaluation import (
    run_standard_evaluation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    summary = run_standard_evaluation(args.config)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
