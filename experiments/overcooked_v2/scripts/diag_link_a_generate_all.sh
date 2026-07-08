#!/usr/bin/env bash
# Link-A latent_v3 certificate dataset fan-out. Run from CPR_REPO root on zsc-customer.
#   bash experiments/overcooked_v2/scripts/diag_link_a_generate_all.sh <anchor_ckpt> <out_dir> [jobs]
set -euo pipefail

export D1_ANCHOR=${1:?anchor checkpoint}
export D1_OUT=${2:?output dir}
JOBS=${3:-24}
EPS_SCALE=${EPS_SCALE:-2}
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export D1_PY=${D1_PY:-.venv/bin/python}

mkdir -p "$D1_OUT"
rm -f "$D1_OUT/failures.txt"

TRAIN_PARTNERS=(
  latent-ingnear-patience2
  latent-ingnear-escalate2
  latent-ingfar-patience4
  latent-ingfar-titfortat1
  latent-prepnear-block12
  latent-prepnear-static-yield
  latent-prepfar-block18
  latent-prepfar-static-claim
  latent-bneck-titfortat2
  latent-bneck-escalate3
  latent-flex-patience6
  latent-flex-block25
)

BLIND_CERT_PARTNERS=(
  blind-cert-ingnear-titfortat3
  blind-cert-ingfar-escalate2
  blind-cert-prepnear-patience5
  blind-cert-prepfar-block20
  blind-cert-bneck-patience3
  blind-cert-flex-block16
)

# prepchain added round 2: reaction-family triggers (escalate/tit-for-tat punish
# paths) require a deferring ego to appear in the data + golden traces at all.
EGOS=(argmax fullchain random prepchain)

worklist=$(mktemp)
i=0
for ego in "${EGOS[@]}"; do
  for p in "${TRAIN_PARTNERS[@]}"; do
    echo "latent_v3_dev $p $ego $((100 * EPS_SCALE)) $((21000 + i * 1000))" >>"$worklist"
    i=$((i + 1))
  done
  for p in "${BLIND_CERT_PARTNERS[@]}"; do
    echo "latent_v3_dev $p $ego $((50 * EPS_SCALE)) $((21000 + i * 1000))" >>"$worklist"
    i=$((i + 1))
  done
done

total=$(wc -l <"$worklist")
echo "Link-A work units: $total -> $D1_OUT (jobs=$JOBS, eps_scale=$EPS_SCALE)"

set +e
xargs -P "$JOBS" -L1 bash -c '
  pset="$0"; partner="$1"; ego="$2"; eps="$3"; seed="$4"
  "$D1_PY" experiments/overcooked_v2/scripts/diag_d1_dataset.py \
    --anchor_checkpoint "$D1_ANCHOR" \
    --partner_set "$pset" --partner "$partner" --ego "$ego" \
    --episodes "$eps" --seed "$seed" --max-episode-options 60 \
    --out "$D1_OUT/linka_${partner}_${ego}.npz" --golden \
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

