#!/usr/bin/env bash
# Remote JAX/CUDA 12 guard for zsc-customer.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=5 bash experiments/overcooked_v2/scripts/with_jax_cuda12.sh \
#     .venv/bin/python experiments/overcooked_v2/scripts/check_jax_cuda.py
#
# The wrapper keeps JAX from picking the system CUDA 11.8 ptxas by forcing the
# virtualenv-provided CUDA 12 ptxas to the front of PATH and by setting XLA's
# CUDA data directory to the matching package root.
set -euo pipefail

fail() {
  echo "with_jax_cuda12: $*" >&2
  exit 2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." >/dev/null 2>&1 && pwd)"

if [[ -n "${VENV:-}" ]]; then
  VENV_DIR="$VENV"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  VENV_DIR="$REPO_ROOT/.venv"
elif [[ -x "$REPO_ROOT/.venv_tgssa/bin/python" ]]; then
  VENV_DIR="$REPO_ROOT/.venv_tgssa"
else
  fail "no usable virtualenv found at .venv or .venv_tgssa"
fi

case "$VENV_DIR" in
  /*) ;;
  *) VENV_DIR="$REPO_ROOT/$VENV_DIR" ;;
esac
[[ -x "$VENV_DIR/bin/python" ]] || fail "virtualenv python is not executable: $VENV_DIR/bin/python"
VENV_REAL="$(cd "$VENV_DIR" >/dev/null 2>&1 && pwd -P)"

PTXAS=""
for base in "$VENV_REAL" "$REPO_ROOT/.venv_tgssa" "$REPO_ROOT/.venv"; do
  [[ -d "$base" ]] || continue
  candidate="$(find "$base" -path '*/nvidia/cuda_nvcc/bin/ptxas' -type f -perm -111 -print -quit 2>/dev/null || true)"
  if [[ -n "$candidate" ]]; then
    PTXAS="$candidate"
    break
  fi
done
[[ -n "$PTXAS" ]] || fail "virtualenv CUDA 12 ptxas was not found under $VENV_DIR"

PTXAS_BIN="$(cd "$(dirname "$PTXAS")" >/dev/null 2>&1 && pwd -P)"
CUDA_NVCC_ROOT="$(cd "$PTXAS_BIN/.." >/dev/null 2>&1 && pwd -P)"
PTXAS_VERSION="$("$PTXAS" --version 2>&1 || true)"
if [[ "$PTXAS_VERSION" != *"release 12."* ]]; then
  echo "$PTXAS_VERSION" >&2
  fail "ptxas must be CUDA 12; refusing to run with $PTXAS"
fi

export PATH="$PTXAS_BIN:$PATH"
case " ${XLA_FLAGS:-} " in
  *" --xla_gpu_cuda_data_dir=$CUDA_NVCC_ROOT "*) ;;
  *" --xla_gpu_cuda_data_dir="*) fail "XLA_FLAGS already points at a different CUDA data dir: $XLA_FLAGS" ;;
  *) export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_NVCC_ROOT${XLA_FLAGS:+ $XLA_FLAGS}" ;;
esac

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=5
fi
VISIBLE_NO_SPACES="${CUDA_VISIBLE_DEVICES//[[:space:]]/}"
if [[ ",$VISIBLE_NO_SPACES," == *",3,"* && "${ALLOW_ECC_GPU3:-0}" != "1" ]]; then
  fail "GPU3 has recorded ECC errors; set a different CUDA_VISIBLE_DEVICES value"
fi

# JAX selects the CUDA client with "cuda"; default_backend() still reports "gpu".
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

case ":${PYTHONPATH:-}:" in
  *":$REPO_ROOT:"*) ;;
  *) export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" ;;
esac

if [[ $# -eq 0 ]]; then
  echo "REPO_ROOT=$REPO_ROOT"
  echo "VENV_DIR=$VENV_DIR"
  echo "ptxas=$PTXAS"
  "$PTXAS" --version
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "JAX_PLATFORMS=$JAX_PLATFORMS"
  echo "XLA_FLAGS=$XLA_FLAGS"
  exit 0
fi

cd "$REPO_ROOT"
exec "$@"
