#!/usr/bin/env bash
# Link-A certificate round 2 runner (phase 1: generation + D1 + mode-cert + VOI).
# Golden family review and the certificate build run AFTER a human/agent golden
# check, so this script stops before build_latent_v3_certificate.py.
#   bash experiments/overcooked_v2/scripts/diag_link_a_round2.sh <anchor_ckpt> <out_dir> [jobs]
set -euo pipefail
ANCHOR=${1:?anchor checkpoint}
OUT=${2:?output dir}
JOBS=${3:-24}
PY=.venv/bin/python
export JAX_PLATFORMS=cpu

echo "== [1/5] generation (72 units: 18 partners x 4 egos) =="
EPS_SCALE=${EPS_SCALE:-2} bash experiments/overcooked_v2/scripts/diag_link_a_generate_all.sh \
  "$ANCHOR" "$OUT/chunks" "$JOBS"

echo "== [2/5] D1 gate =="
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 $PY experiments/overcooked_v2/scripts/diag_d1_train.py \
  --chunks "$OUT/chunks" --out "$OUT/d1" --stage gate

echo "== [3/5] D1 tune + readout =="
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 $PY experiments/overcooked_v2/scripts/diag_d1_train.py \
  --chunks "$OUT/chunks" --out "$OUT/d1" --stage tune
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 $PY experiments/overcooked_v2/scripts/diag_d1_train.py \
  --chunks "$OUT/chunks" --out "$OUT/d1" --stage readout

echo "== [4/5] mode identifiability (C-4) =="
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 $PY experiments/overcooked_v2/scripts/diag_link_a_mode_cert.py \
  --chunks "$OUT/chunks" --out "$OUT/mode_cert.json"

echo "== [5/5] VOI probe (C-5, pooled + reactive arm) =="
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY \
  experiments/overcooked_v2/scripts/latent_v3_voi_probe.py \
  --checkpoint "$ANCHOR" --out "$OUT/voi.json"

echo "ROUND2_PHASE1_DONE -> golden review + build_latent_v3_certificate.py next"
