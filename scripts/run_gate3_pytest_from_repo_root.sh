#!/usr/bin/env bash
set -euo pipefail
# Usage from the ARIS4ZSC repository root:
#   ./scripts/run_gate3_pytest_from_repo_root.sh
PKG_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHONPATH="$PKG_DIR/src${PYTHONPATH:+:$PYTHONPATH}" python -m pytest -q \
  "$PKG_DIR/tests/test_canonical.py" \
  "$PKG_DIR/tests/test_invariants.py" \
  "$PKG_DIR/tests/test_vectorized_graph_and_diagnostics.py" \
  "$@"
