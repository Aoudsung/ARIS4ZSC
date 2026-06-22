#!/usr/bin/env bash
set -euo pipefail

# Remote-only ToMZSC PECAN target-domain reproduction for M3.
# Run from /apps/users/cxw/ARIS4ZSC on the SSH backend.

TGSSA_ROOT="${TGSSA_ROOT:-/apps/users/cxw/Document/CodeSpace/Selfs/TG-SSA}"
TOMZSC_REPO="${TOMZSC_REPO:-${TGSSA_ROOT}/external/ToMZSC}"
OUT_ROOT="${OUT_ROOT:-/apps/users/cxw/ARIS4ZSC/results/serd_pecan_target_domain_reproduction}"
LAYOUT="${LAYOUT:-counter_circuit}"
ENV_NAME="overcooked_${LAYOUT}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1048576}"
NUM_TEAMMATES="${NUM_TEAMMATES:-3}"
NUM_ENVS="${NUM_ENVS:-16}"
NUM_STEPS="${NUM_STEPS:-16}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-4}"
NUM_EPOCHS="${NUM_EPOCHS:-4}"
HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
NUM_LAYERS="${NUM_LAYERS:-1}"
EVAL_ITERS="${EVAL_ITERS:-10}"
PYTHON_BIN="${PYTHON_BIN:-${TGSSA_ROOT}/.venv/bin/python}"
REUSE_EXISTING_CHECKPOINTS="${REUSE_EXISTING_CHECKPOINTS:-0}"
PATCH_COUNTER_CIRCUIT_EVAL="${PATCH_COUNTER_CIRCUIT_EVAL:-1}"

# The remote system ptxas is CUDA 11.8, while the TG-SSA venv ships CUDA 12.9
# ptxas. Point XLA at the venv CUDA data dir so JAX does not pick the older
# system assembler.
CUDA_NVCC_DIR="${CUDA_NVCC_DIR:-${TGSSA_ROOT}/.venv/lib/python3.10/site-packages/nvidia/cuda_nvcc}"
export PATH="${CUDA_NVCC_DIR}/bin:${PATH}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_cuda_data_dir=${CUDA_NVCC_DIR}}"
export JAX_PLATFORM_NAME="${JAX_PLATFORM_NAME:-gpu}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

LOG_DIR="${OUT_ROOT}/logs"
SAVE_TEAMMATES="${OUT_ROOT}/save_teammates"
SAVE_PECAN="${OUT_ROOT}/save_pecan"
CLUSTER_DIR="${OUT_ROOT}/clusters"
TEAMMATE_ALG="teammate_pool_counter_circuit"
PECAN_ALG="pecan_1m"
EVAL_ALG="pecan_1m_eval_smoke"
TEAMMATE_DIR="${SAVE_TEAMMATES}/${ENV_NAME}/${TEAMMATE_ALG}_${ENV_NAME}"
PECAN_DIR="${SAVE_PECAN}/${ENV_NAME}/${PECAN_ALG}_${ENV_NAME}"
PECAN_CLUSTER_LABELS="${CLUSTER_DIR}/pecan_clusters.json"
PATCHED_TOMZSC_REPO="${OUT_ROOT}/_patched_tomzsc"
EVAL_REPO="${TOMZSC_REPO}"
if [[ "${PATCH_COUNTER_CIRCUIT_EVAL}" == "1" ]]; then
  EVAL_REPO="${PATCHED_TOMZSC_REPO}"
fi
EVAL_JSON="${EVAL_REPO}/overcooked_cache/eval/${ENV_NAME}/${EVAL_ALG}.json"

mkdir -p "${LOG_DIR}" "${OUT_ROOT}" "${CLUSTER_DIR}"

run_logged() {
  local log_path="$1"
  shift
  {
    printf '$'
    printf ' %q' "$@"
    printf '\n\n'
    "$@"
  } >"${log_path}" 2>&1
}

write_json() {
  local path="$1"
  shift
  "${PYTHON_BIN}" - "$path" "$@" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(sys.argv[2])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

cat >"${OUT_ROOT}/training_command.txt" <<EOF
TGSSA_ROOT=${TGSSA_ROOT}
TOMZSC_REPO=${TOMZSC_REPO}
LAYOUT=${LAYOUT}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS}
NUM_TEAMMATES=${NUM_TEAMMATES}
NUM_ENVS=${NUM_ENVS}
NUM_STEPS=${NUM_STEPS}
NUM_MINIBATCHES=${NUM_MINIBATCHES}
NUM_EPOCHS=${NUM_EPOCHS}
HIDDEN_SIZE=${HIDDEN_SIZE}
NUM_LAYERS=${NUM_LAYERS}
CUDA_NVCC_DIR=${CUDA_NVCC_DIR}
XLA_FLAGS=${XLA_FLAGS}
JAX_PLATFORM_NAME=${JAX_PLATFORM_NAME}
XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE}
REUSE_EXISTING_CHECKPOINTS=${REUSE_EXISTING_CHECKPOINTS}
PATCH_COUNTER_CIRCUIT_EVAL=${PATCH_COUNTER_CIRCUIT_EVAL}

1. Prepare official ToMZSC checkout:
${PYTHON_BIN} -m scripts.prepare_tomzsc_external --repo ${TOMZSC_REPO} --json-output ${OUT_ROOT}/tomzsc_prepare_counter_circuit.json

2. Train target-domain teammate pool:
${PYTHON_BIN} overcooked_train_rl.py +alg=overcooked_teammate alg.ENV_KWARGS.layout=${LAYOUT} alg.TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS} alg.NUM_ENVS=${NUM_ENVS} alg.NUM_STEPS=${NUM_STEPS} alg.NUM_MINIBATCHES=${NUM_MINIBATCHES} alg.NUM_EPOCHS=${NUM_EPOCHS} alg.HIDDEN_SIZE=${HIDDEN_SIZE} alg.NUM_LAYERS=${NUM_LAYERS} alg.TEST_DURING_TRAINING=False WANDB_MODE=disabled NUM_SEEDS=${NUM_TEAMMATES} SAVE_PATH=${SAVE_TEAMMATES} ALG_NAME=${TEAMMATE_ALG}

3. Train target-domain PECAN response:
${PYTHON_BIN} overcooked_train_rl.py +alg=overcooked_response alg.ENV_KWARGS.layout=${LAYOUT} alg.TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS} alg.NUM_ENVS=${NUM_ENVS} alg.NUM_STEPS=${NUM_STEPS} alg.NUM_MINIBATCHES=${NUM_MINIBATCHES} alg.NUM_EPOCHS=${NUM_EPOCHS} alg.HIDDEN_SIZE=${HIDDEN_SIZE} alg.NUM_LAYERS=${NUM_LAYERS} alg.TEST_DURING_TRAINING=False alg.TEAMMATE_DIR=${TEAMMATE_DIR} alg.PECAN=True WANDB_MODE=disabled NUM_SEEDS=1 SAVE_PATH=${SAVE_PECAN} ALG_NAME=${PECAN_ALG}

4. Evaluate/load-smoke PECAN response:
${PYTHON_BIN} overcooked_eval.py +alg=overcooked_eval alg.ENV_KWARGS.layout=${LAYOUT} alg.TEAMMATE_DIR=${TEAMMATE_DIR} alg.EGO_DIR=${PECAN_DIR} alg.METHOD=pecan alg.CLUSTER_LABELS=${PECAN_CLUSTER_LABELS} +alg.PECAN=True alg.HIDDEN_SIZE=${HIDDEN_SIZE} alg.NUM_LAYERS=${NUM_LAYERS} +alg.EVAL_ITERS=${EVAL_ITERS} ALG_NAME=${EVAL_ALG} WANDB_MODE=disabled
EOF

prepare_eval_repo() {
  if [[ "${PATCH_COUNTER_CIRCUIT_EVAL}" != "1" ]]; then
    return
  fi
  if [[ ! -d "${PATCHED_TOMZSC_REPO}" ]]; then
    cp -a "${TOMZSC_REPO}" "${PATCHED_TOMZSC_REPO}"
  fi
  "${PYTHON_BIN}" - "${PATCHED_TOMZSC_REPO}/ctom.py" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
old = """            my_interact = timestep.state.agent_interact[aidx]
            other_interact = timestep.state.agent_interact[1-aidx]
"""
new = """            agent_interact = getattr(
                timestep.state,
                "agent_interact",
                jnp.zeros((env.num_agents,), dtype=jnp.int32),
            )
            my_interact = agent_interact[aidx]
            other_interact = agent_interact[1-aidx]
"""
if old not in text and new not in text:
    raise SystemExit("ctom.py patch anchor not found")
if old in text:
    text = text.replace(old, new)
path.write_text(text, encoding="utf-8")
PY
}

write_json "${OUT_ROOT}/reproduction_manifest.json" "$(
  printf '{"status":"running","claim_bearing":false,"executed_training":true,"started_at":"%s","source_path":"%s","repo":"%s","layout":"%s","target_domain":"counter_circuit","requested_algorithm":{"ALG_NAME":"pecan_1m","PECAN":true},"seed_list":[154],"num_teammates":%s,"total_timesteps":%s,"hardware_backend":"remote zsc","teammate_dir":"%s","pecan_dir":"%s"}' \
    "${STARTED_AT}" "${TGSSA_ROOT}" "${TOMZSC_REPO}" "${LAYOUT}" "${NUM_TEAMMATES}" "${TOTAL_TIMESTEPS}" "${TEAMMATE_DIR}" "${PECAN_DIR}"
)"

(
  cd "${TGSSA_ROOT}"
  run_logged "${LOG_DIR}/00_prepare.log" \
    "${PYTHON_BIN}" -m scripts.prepare_tomzsc_external \
    --repo "${TOMZSC_REPO}" \
    --json-output "${OUT_ROOT}/tomzsc_prepare_counter_circuit.json"
)

(
  cd "${TOMZSC_REPO}"
  if [[ "${REUSE_EXISTING_CHECKPOINTS}" != "1" ]]; then
    run_logged "${LOG_DIR}/01_train_teammates.log" \
      "${PYTHON_BIN}" overcooked_train_rl.py \
      +alg=overcooked_teammate \
      "alg.ENV_KWARGS.layout=${LAYOUT}" \
      "alg.TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS}" \
      "alg.NUM_ENVS=${NUM_ENVS}" \
      "alg.NUM_STEPS=${NUM_STEPS}" \
      "alg.NUM_MINIBATCHES=${NUM_MINIBATCHES}" \
      "alg.NUM_EPOCHS=${NUM_EPOCHS}" \
      "alg.HIDDEN_SIZE=${HIDDEN_SIZE}" \
      "alg.NUM_LAYERS=${NUM_LAYERS}" \
      "alg.TEST_DURING_TRAINING=False" \
      "WANDB_MODE=disabled" \
      "NUM_SEEDS=${NUM_TEAMMATES}" \
      "SAVE_PATH=${SAVE_TEAMMATES}" \
      "ALG_NAME=${TEAMMATE_ALG}"

    run_logged "${LOG_DIR}/02_train_pecan.log" \
      "${PYTHON_BIN}" overcooked_train_rl.py \
      +alg=overcooked_response \
      "alg.ENV_KWARGS.layout=${LAYOUT}" \
      "alg.TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS}" \
      "alg.NUM_ENVS=${NUM_ENVS}" \
      "alg.NUM_STEPS=${NUM_STEPS}" \
      "alg.NUM_MINIBATCHES=${NUM_MINIBATCHES}" \
      "alg.NUM_EPOCHS=${NUM_EPOCHS}" \
      "alg.HIDDEN_SIZE=${HIDDEN_SIZE}" \
      "alg.NUM_LAYERS=${NUM_LAYERS}" \
      "alg.TEST_DURING_TRAINING=False" \
      "alg.TEAMMATE_DIR=${TEAMMATE_DIR}" \
      "alg.PECAN=True" \
      "WANDB_MODE=disabled" \
      "NUM_SEEDS=1" \
      "SAVE_PATH=${SAVE_PECAN}" \
      "ALG_NAME=${PECAN_ALG}"
  else
    printf 'Reusing existing checkpoints under %s and %s\n' "${TEAMMATE_DIR}" "${PECAN_DIR}" \
      >"${LOG_DIR}/01_02_reuse_existing_checkpoints.log"
  fi

  "${PYTHON_BIN}" - "${PECAN_CLUSTER_LABELS}" "${NUM_TEAMMATES}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
num_teammates = int(sys.argv[2])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps([0] * num_teammates) + "\n", encoding="utf-8")
PY

  prepare_eval_repo
  cd "${EVAL_REPO}"
  run_logged "${LOG_DIR}/03_eval_pecan.log" \
    "${PYTHON_BIN}" overcooked_eval.py \
    +alg=overcooked_eval \
    "alg.ENV_KWARGS.layout=${LAYOUT}" \
    "alg.TEAMMATE_DIR=${TEAMMATE_DIR}" \
    "alg.EGO_DIR=${PECAN_DIR}" \
    "alg.METHOD=pecan" \
    "alg.CLUSTER_LABELS=${PECAN_CLUSTER_LABELS}" \
    "+alg.PECAN=True" \
    "alg.HIDDEN_SIZE=${HIDDEN_SIZE}" \
    "alg.NUM_LAYERS=${NUM_LAYERS}" \
    "+alg.EVAL_ITERS=${EVAL_ITERS}" \
    "ALG_NAME=${EVAL_ALG}" \
    "WANDB_MODE=disabled"
)

FINISHED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

{
  find "${TEAMMATE_DIR}" "${PECAN_DIR}" -maxdepth 1 -type f -name '*.safetensors' -print0 2>/dev/null \
    | xargs -0 -r sha256sum
} >"${OUT_ROOT}/checkpoint_hashes.txt"

{
  find "${LOG_DIR}" -type f -maxdepth 1 -print0 \
    | xargs -0 -r sha256sum
} >"${OUT_ROOT}/training_logs_manifest.txt"

"${PYTHON_BIN}" - "${OUT_ROOT}" "${TEAMMATE_DIR}" "${PECAN_DIR}" "${EVAL_JSON}" "${LAYOUT}" <<'PY'
import json
import os
import sys
from pathlib import Path

out = Path(sys.argv[1])
teammate_dir = Path(sys.argv[2])
pecan_dir = Path(sys.argv[3])
eval_json = Path(sys.argv[4])
layout = sys.argv[5]

teammates = sorted(teammate_dir.glob("*.safetensors"))
pecan = sorted(pecan_dir.glob("*.safetensors"))

load_status = {
    "status": "checkpoint_load_passed" if pecan else "checkpoint_load_failed_missing_checkpoint",
    "claim_bearing": False,
    "teammate_dir": str(teammate_dir),
    "teammate_safetensors": len(teammates),
    "pecan_dir": str(pecan_dir),
    "pecan_safetensors": len(pecan),
    "layout": layout,
    "config_evidence": {"ALG_NAME": "pecan_1m", "PECAN": True},
}
if pecan:
    load_status["sample_checkpoint"] = str(pecan[0])
out.joinpath("load_smoke.json").write_text(json.dumps(load_status, indent=2, sort_keys=True) + "\n", encoding="utf-8")

eval_status = {
    "status": "eval_smoke_passed" if eval_json.exists() else "eval_smoke_failed_missing_eval_json",
    "claim_bearing": False,
    "eval_json": str(eval_json),
    "layout": layout,
}
if eval_json.exists():
    try:
        payload = json.loads(eval_json.read_text(encoding="utf-8"))
        eval_status["keys"] = sorted(payload.keys())
        eval_status["total_returns_preview"] = payload.get("total_returns", [])[:10]
    except Exception as exc:
        eval_status["status"] = "eval_smoke_failed_json_parse"
        eval_status["error"] = repr(exc)
out.joinpath("eval_smoke.json").write_text(json.dumps(eval_status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

cat >"${OUT_ROOT}/adapter_compatibility.md" <<EOF
# PECAN Adapter Compatibility

Status: PECAN_TARGET_REPRODUCTION_ACCEPTED_NOT_SERD if load and eval smoke
files both pass; otherwise inspect the failure logs.

- Source route: official ToMZSC/JaxMARL Overcooked response policy.
- Target domain: ${LAYOUT} / ${ENV_NAME}.
- Config evidence: ALG_NAME=${PECAN_ALG}, alg.PECAN=True.
- Eval smoke route: ${EVAL_REPO}.
- Eval shim: PATCH_COUNTER_CIRCUIT_EVAL=${PATCH_COUNTER_CIRCUIT_EVAL}. When
  enabled, the result-local ToMZSC copy treats missing agent_interact state as
  zero interaction for rollout-smoke compatibility with ${LAYOUT}. This is an
  adapter risk, not a paper claim.
- Same-state SERD implication: adapter must branch through the JaxMARL
  Overcooked state and PECAN recurrent policy state. This reproduction only
  establishes a target-domain PECAN checkpoint candidate; it is not SERD.
EOF

cat >"${OUT_ROOT}/deviation_report.md" <<EOF
# PECAN Reproduction Deviation Report

- Domain changed from prior cramped-room evidence to locked target domain
  ${LAYOUT}.
- Budget uses ${TOTAL_TIMESTEPS} timesteps and the reduced official server
  hyperparameter shape already used by TG-SSA ToMZSC 1M reproductions.
- W&B is disabled for non-secret local artifact generation.
- Official ToMZSC eval assumes state.agent_interact; ${LAYOUT} on this
  backend lacks that field. Eval smoke therefore uses a result-local ctom shim
  when PATCH_COUNTER_CIRCUIT_EVAL=1.
- This is a reproduction candidate and not a paper-quality or SERD claim.
EOF

cat >"${OUT_ROOT}/operator_note.md" <<EOF
# Operator Note

- Started: ${STARTED_AT}
- Finished: ${FINISHED_AT}
- Host route: remote zsc
- Source root: ${TGSSA_ROOT}
- ToMZSC repo: ${TOMZSC_REPO}
- Output root: ${OUT_ROOT}
- No credentials or one-time passwords were written.
- Claim boundary: target-domain PECAN checkpoint reproduction candidate only;
  not SERD and not paper evidence.
EOF

STATUS="PECAN_TARGET_REPRODUCTION_ACCEPTED_NOT_SERD"
if ! grep -q 'checkpoint_load_passed' "${OUT_ROOT}/load_smoke.json"; then
  STATUS="BLOCKED_PECAN_LOAD_FAILURE"
fi
if ! grep -q 'eval_smoke_passed' "${OUT_ROOT}/eval_smoke.json"; then
  STATUS="BLOCKED_PECAN_ROLLOUT_FAILURE"
fi

write_json "${OUT_ROOT}/reproduction_manifest.json" "$(
  printf '{"status":"%s","claim_bearing":false,"executed_training":true,"started_at":"%s","finished_at":"%s","source_path":"%s","repo":"%s","layout":"%s","target_domain":"counter_circuit","requested_algorithm":{"ALG_NAME":"pecan_1m","PECAN":true},"seed_list":[154],"num_teammates":%s,"total_timesteps":%s,"hardware_backend":"remote zsc","teammate_dir":"%s","pecan_dir":"%s","eval_json":"%s"}' \
    "${STATUS}" "${STARTED_AT}" "${FINISHED_AT}" "${TGSSA_ROOT}" "${TOMZSC_REPO}" "${LAYOUT}" "${NUM_TEAMMATES}" "${TOTAL_TIMESTEPS}" "${TEAMMATE_DIR}" "${PECAN_DIR}" "${EVAL_JSON}"
)"

printf '%s\n' "${STATUS}"
