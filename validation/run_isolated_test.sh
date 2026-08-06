#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 experiments/overcooked_v2/tests/test_delta_*.py" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
TEST_FILE="$1"
test -f "${TEST_FILE}"

export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# The repository does not depend on ambient pytest plugins. Disabling plugin
# autoload prevents unrelated tracing/coverage plugins from retaining JAX
# executables or terminating a process after a successful test.
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

mapfile -t TESTS < <(python - "${TEST_FILE}" <<'PY'
import ast
from pathlib import Path
import sys
source = Path(sys.argv[1])
module = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
for node in module.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
        print(node.name)
PY
)
if [[ ${#TESTS[@]} -eq 0 ]]; then
  echo "no tests found in ${TEST_FILE}" >&2
  exit 3
fi

for test_name in "${TESTS[@]}"; do
  echo "[isolated] ${TEST_FILE}::${test_name}"
  python -m pytest -q "${TEST_FILE}::${test_name}"
done
