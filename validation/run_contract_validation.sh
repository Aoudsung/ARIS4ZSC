#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

python -m compileall -q src/delta_zsc experiments/overcooked_v2

python - <<'PY'
from pathlib import Path
from src.delta_zsc.config import load_config

for path in sorted(Path("experiments/overcooked_v2/configs").glob("delta_unified_*.yaml")):
    run_kind = path.stem.rsplit("_", 1)[-1]
    config = load_config(path, run_kind=run_kind)
    print(path, config.fingerprint)
PY

python -m experiments.overcooked_v2.delta_zsc --help >/dev/null
for command in \
  build-partner-manifest validate-partner-manifest train cuda-preflight \
  build-policy-manifest evaluate summarize-evaluations \
  posterior-diagnostics belief-intervention run-development-matrix \
  evaluate-development-matrix summarize-development-matrix \
  train-baseline resource-report formal-claim; do
  python -m experiments.overcooked_v2.delta_zsc "${command}" --help >/dev/null
done

python - <<'PY'
from pathlib import Path
import yaml
payload = yaml.safe_load(Path(".github/workflows/delta-unified-ci.yml").read_text())
assert {"test", "cuda-preflight"} <= set(payload["jobs"])
PY

test ! -d src/path_c
test ! -d analysis
! grep -R -n -E \
  'pair_comparator|separation_margin|context_dropout|capability_consistency_weight|posterior_decision_weight|decision_policy_weight|gradient_routing|decision_regret|response_cross_log_likelihood' \
  src/delta_zsc
! find experiments/overcooked_v2 -maxdepth 1 -type f -name '*.py' -print0 \
  | xargs -0 grep -n 'src\.path_c'

python validation/generate_voi_diagnostics.py >/dev/null

echo "CONTRACT_VALIDATION_COMPLETE"
