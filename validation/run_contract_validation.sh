#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

python -m compileall -q src/delta_zsc experiments/overcooked_v2 validation

python - <<'PY_COMPAT'
import ast
from pathlib import Path

for root in (Path("src/delta_zsc"), Path("experiments/overcooked_v2"), Path("validation")):
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        ast.parse(
            path.read_text(encoding="utf-8"),
            filename=str(path),
            feature_version=(3, 10),
        )
PY_COMPAT

python - <<'PY'
from pathlib import Path
import ast
import re

from src.delta_zsc.config import (
    CHECKPOINT_SCHEMA_VERSION,
    CONFIG_VERSION,
    MANIFEST_VERSION,
    METHOD_VERSION,
    load_config,
)
from src.delta_zsc.semantic_initializer import SEMANTIC_INITIALIZER_SCHEMA_VERSION

paths = sorted(Path("experiments/overcooked_v2/configs").glob("delta_unified_*.yaml"))
assert len(paths) == 9, paths
for path in paths:
    run_kind = path.stem.rsplit("_", 1)[-1]
    if run_kind == "collector":
        run_kind = "development"
    config = load_config(path, run_kind=run_kind)
    assert config.version == CONFIG_VERSION
    print(path, config.version, config.method_variant, config.method)

pyproject_source = Path("pyproject.toml").read_text(encoding="utf-8")
dependency_match = re.search(
    r"(?ms)^dependencies\s*=\s*(\[.*?^\])", pyproject_source
)
assert dependency_match is not None
project_dependencies = set(ast.literal_eval(dependency_match.group(1)))
version_match = re.search(
    r'(?m)^version\s*=\s*"([^"]+)"', pyproject_source
)
assert version_match is not None
requirements = {
    line.strip()
    for line in Path("requirements.txt").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.lstrip().startswith("#")
}
assert requirements == project_dependencies, (
    sorted(project_dependencies - requirements),
    sorted(requirements - project_dependencies),
)
identity = __import__("json").loads(Path("ARTIFACT_IDENTITY.json").read_text(encoding="utf-8"))
assert identity["active_method_version"] == METHOD_VERSION
assert identity["config_version"] == CONFIG_VERSION
assert identity["checkpoint_schema_version"] == CHECKPOINT_SCHEMA_VERSION
assert identity["manifest_version"] == MANIFEST_VERSION
assert identity["semantic_initializer_schema_version"] == SEMANTIC_INITIALIZER_SCHEMA_VERSION
assert identity["package_version"] == version_match.group(1)
assert Path("IMPLEMENTATION_MATRIX.md").is_file()
PY

python -m experiments.overcooked_v2.delta_zsc --help >/dev/null
for command in \
  build-partner-manifest validate-partner-manifest train cuda-preflight \
  build-policy-manifest evaluate summarize-evaluations summarize-population-matrices \
  build-semantic-initializer posterior-diagnostics belief-intervention run-development-matrix \
  evaluate-development-matrix summarize-development-matrix \
  train-baseline train-official-parent run-upstream resource-report formal-claim; do
  python -m experiments.overcooked_v2.delta_zsc "${command}" --help >/dev/null
done

python - <<'PY'
from pathlib import Path
import yaml
payload = yaml.safe_load(Path(".github/workflows/delta-unified-ci.yml").read_text(encoding="utf-8"))
assert {"test", "cuda-preflight"} <= set(payload["jobs"])
PY

test ! -d src/path_c
test ! -d analysis
test ! -f src/delta_zsc/transition.py
! grep -R -n -E \
  'pair_comparator|separation_margin|context_dropout|capability_consistency_weight|posterior_decision_weight|decision_policy_weight|gradient_routing|decision_regret|response_cross_log_likelihood' \
  src/delta_zsc
! find experiments/overcooked_v2 -maxdepth 1 -type f -name '*.py' -print0 \
  | xargs -0 grep -n 'src\.path_c'

python - <<'PY_PATHS'
from pathlib import Path
import re

credential_patterns = (
    re.compile(r"BEGIN (?:RSA|OPENSSH|EC|DSA) PRIVATE KEY"),
    re.compile(r"gh[opusr]_[A-Za-z0-9_]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
)
for path in Path(".").rglob("*"):
    if not path.is_file() or {".git", ".venv", "__pycache__", ".pytest_cache"} & set(path.parts):
        continue
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        continue
    for pattern in credential_patterns:
        if pattern.search(text):
            raise AssertionError(f"credential pattern {pattern.pattern!r} in {path}")
    if "legacy" in path.parts or path.as_posix() == "validation/run_contract_validation.sh":
        continue
    for marker in ("/Users/", "/mnt/workspace/", "/mnt/data/"):
        if marker in text:
            raise AssertionError(f"local absolute path {marker!r} in {path}")
PY_PATHS

git diff --check
python validation/generate_voi_diagnostics.py >/dev/null

echo "CONTRACT_VALIDATION_COMPLETE"
