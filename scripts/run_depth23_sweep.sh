#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=src python -m tcaso_critic.cli sweep \
  --config configs/sweep_grounded_two_rooms_depth23.yaml \
  --out reports/grounded_two_rooms_depth23
