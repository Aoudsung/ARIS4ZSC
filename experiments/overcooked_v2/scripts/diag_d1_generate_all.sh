#!/usr/bin/env bash
# D1 generation fan-out (prereg D1 §2). Run from CPR_REPO root on the remote host.
#   bash experiments/overcooked_v2/scripts/diag_d1_generate_all.sh <anchor_ckpt> <out_dir> [jobs]
# Scale note: per-unit episode counts follow the prereg floors (train 100 / dev 50 /
# blind 50 per ego); if the sufficiency gate fails, rerun with EPS_SCALE=2.
set -euo pipefail
export D1_ANCHOR=${1:?anchor checkpoint}
export D1_OUT=${2:?output dir}
JOBS=${3:-24}
EPS_SCALE=${EPS_SCALE:-1}
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export D1_PY=.venv/bin/python
mkdir -p "$D1_OUT"
rm -f "$D1_OUT/failures.txt"

TRAIN_PARTNERS=(ingredient-near-yield ingredient-far-yield server-left-claim
  server-right-claim bottleneck-yield-terminal-yield bottleneck-push-terminal-claim)
DEV_PARTNERS=(heldout-handoff-alternate-yield heldout-resource-server-claim)
BLIND_PARTNERS=(blind-dish-yield blind-prep-alternate-yield blind-dish-claim
  blind-server-nearest-claim-push blind-bottleneck-alternate-neutral
  blind-ingredient-near-neutral)
EGOS=(argmax fullchain random)

worklist=$(mktemp)
i=0
for ego in "${EGOS[@]}"; do
  for p in "${TRAIN_PARTNERS[@]}"; do
    echo "role_conditioned_v2 $p $ego $((100 * EPS_SCALE)) $((11000 + i * 1000))" >>"$worklist"
    i=$((i + 1))
  done
  for p in "${DEV_PARTNERS[@]}"; do
    echo "role_conditioned_v2 $p $ego $((50 * EPS_SCALE)) $((11000 + i * 1000))" >>"$worklist"
    i=$((i + 1))
  done
  for p in "${BLIND_PARTNERS[@]}"; do
    echo "blind_v1 $p $ego $((50 * EPS_SCALE)) $((11000 + i * 1000))" >>"$worklist"
    i=$((i + 1))
  done
done
total=$(wc -l <"$worklist")
echo "work units: $total -> $D1_OUT (jobs=$JOBS, eps_scale=$EPS_SCALE)"

set +e
xargs -P "$JOBS" -L1 bash -c '
  pset="$0"; partner="$1"; ego="$2"; eps="$3"; seed="$4"
  "$D1_PY" experiments/overcooked_v2/scripts/diag_d1_dataset.py \
    --anchor_checkpoint "$D1_ANCHOR" \
    --partner_set "$pset" --partner "$partner" --ego "$ego" \
    --episodes "$eps" --seed "$seed" --max-episode-options 40 \
    --out "$D1_OUT/d1_${partner}_${ego}.npz" \
    >"$D1_OUT/log_${partner}_${ego}.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "FAILED $partner $ego rc=$rc" >>"$D1_OUT/failures.txt"
  fi
  exit $rc
' <"$worklist"
xargs_rc=$?
set -e

done_n=$(find "$D1_OUT" -maxdepth 1 -name '*.npz' -type f | wc -l)
echo "chunks: $done_n/$total (xargs rc=$xargs_rc)"
if [ -s "$D1_OUT/failures.txt" ]; then
  echo "FAILURES:"
  cat "$D1_OUT/failures.txt"
  exit 1
fi
[ "$done_n" -eq "$total" ] || { echo "chunk count mismatch"; exit 1; }
