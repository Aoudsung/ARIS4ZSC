#!/usr/bin/env bash
# Link-A round-2 GPU continuation: chunks + D1 gate already done on disk.
# Runs the 4 remaining stages: tune/readout/mode_cert on GPU (pure-torch models),
# VOI on CPU (JAX rollouts). Designed to be launched under setsid+nohup so an SSH
# control-socket drop cannot SIGHUP it (root cause of the round-1 phase-1 failure).
#   setsid nohup bash diag_link_a_round2_gpu_continue.sh <anchor> <out_dir> <gpu> > log 2>&1 &
set -uo pipefail
ANCHOR=${1:?anchor checkpoint}
OUT=${2:?output dir}
GPU=${3:?gpu index}
PY=.venv/bin/python
export JAX_PLATFORMS=cpu

fail() { echo "STAGE_FAILED: $1 (rc=$2)"; echo "ROUND2_GPU_FAILED"; exit "$2"; }

echo "== [1/4] D1 tune on cuda:$GPU =="
CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 $PY \
  experiments/overcooked_v2/scripts/diag_d1_train.py \
  --chunks "$OUT/chunks" --out "$OUT/d1" --stage tune --device cuda || fail tune $?

echo "== [2/4] D1 readout on cuda:$GPU =="
CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 $PY \
  experiments/overcooked_v2/scripts/diag_d1_train.py \
  --chunks "$OUT/chunks" --out "$OUT/d1" --stage readout --device cuda || fail readout $?

echo "== [3/4] mode identifiability C-4 on cuda:$GPU =="
CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 $PY \
  experiments/overcooked_v2/scripts/diag_link_a_mode_cert.py \
  --chunks "$OUT/chunks" --out "$OUT/mode_cert.json" --device cuda || fail mode_cert $?

echo "== [4/4] VOI probe C-5 on CPU (JAX rollouts) =="
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY \
  experiments/overcooked_v2/scripts/latent_v3_voi_probe.py \
  --checkpoint "$ANCHOR" --out "$OUT/voi.json" || fail voi $?

echo "ROUND2_GPU_DONE -> golden review + build_latent_v3_certificate.py next"
