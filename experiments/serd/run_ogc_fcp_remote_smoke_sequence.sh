#!/usr/bin/env bash
# Remote-only OGC/FCP smoke sequence for SERD M3.
#
# Intended execution location:
#   /apps/users/cxw/ARIS4ZSC on the SSH backend.
#
# This script runs inventory -> checkpoint-load -> eval-smoke using the
# remote OGC checkout. It does not emit SERD BranchRecord rows and is not
# claim-bearing evidence.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apps/users/cxw/ARIS4ZSC}"
REMOTE_PY="${REMOTE_PY:-/apps/users/cxw/venvs/ogc-py310/bin/python}"
OGC_SRC="${OGC_SRC:-/apps/users/cxw/ZSC_coordinator/external/OGC/src}"
POPULATION_JSON="${POPULATION_JSON:-populations/fcp/Overcooked-CounterCircuit6_9/population.json}"
OUT_DIR="${OUT_DIR:-/apps/users/cxw/ARIS4ZSC/results/serd_ogc_fcp_remote_smoke}"
AGENT_ID="${AGENT_ID:-1}"
BAND="${BAND:-low}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-checkpoint}"
ENV_NAMES="${ENV_NAMES:-Overcooked-CounterCircuit6_9}"
N_EPISODES="${N_EPISODES:-1}"
AGENT_IDXS="${AGENT_IDXS:-0}"
TIMEOUT_SEC="${TIMEOUT_SEC:-300}"
RESULTS_FNAME="${RESULTS_FNAME:-ogc_fcp_eval_smoke}"

cd "${PROJECT_ROOT}"
mkdir -p "${OUT_DIR}"

echo "[M3-FCP] inventory"
"${REMOTE_PY}" -m experiments.serd.run_ogc_fcp_remote_smoke \
  --mode inventory \
  --ogc-src "${OGC_SRC}" \
  --population-json "${POPULATION_JSON}" \
  --agent-id "${AGENT_ID}" \
  --band "${BAND}" \
  --output-json "${OUT_DIR}/inventory.json"

echo "[M3-FCP] checkpoint-load"
"${REMOTE_PY}" -m experiments.serd.run_ogc_fcp_remote_smoke \
  --mode checkpoint-load \
  --ogc-src "${OGC_SRC}" \
  --population-json "${POPULATION_JSON}" \
  --agent-id "${AGENT_ID}" \
  --band "${BAND}" \
  --output-json "${OUT_DIR}/checkpoint_load.json"

echo "[M3-FCP] eval-smoke"
eval_args=()
if [[ -n "${EGO_XPID:-}" ]]; then
  eval_args+=(--ego-xpid "${EGO_XPID}")
fi

"${REMOTE_PY}" -m experiments.serd.run_ogc_fcp_remote_smoke \
  --mode eval-smoke \
  --ogc-src "${OGC_SRC}" \
  --population-json "${POPULATION_JSON}" \
  --agent-id "${AGENT_ID}" \
  --band "${BAND}" \
  --checkpoint-name "${CHECKPOINT_NAME}" \
  --env-names "${ENV_NAMES}" \
  --n-episodes "${N_EPISODES}" \
  --agent-idxs "${AGENT_IDXS}" \
  --timeout-sec "${TIMEOUT_SEC}" \
  --results-path "${OUT_DIR}" \
  --results-fname "${RESULTS_FNAME}" \
  --output-json "${OUT_DIR}/eval_smoke.json" \
  "${eval_args[@]}"

echo "[M3-FCP] wrote:"
echo "  ${OUT_DIR}/inventory.json"
echo "  ${OUT_DIR}/checkpoint_load.json"
echo "  ${OUT_DIR}/eval_smoke.json"
echo "  ${OUT_DIR}/${RESULTS_FNAME}.csv"
