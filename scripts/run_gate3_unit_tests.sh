#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pytest -q \
  tests/test_canonical.py \
  tests/test_invariants.py \
  tests/test_vectorized_graph_and_diagnostics.py \
  "$@"
