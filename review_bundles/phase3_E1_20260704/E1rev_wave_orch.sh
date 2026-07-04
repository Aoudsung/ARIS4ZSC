#!/bin/bash
cd /apps/users/cxw/Document/CodeSpace/Selfs/ARIS4ZSC-phase1-4d680ab
CFG=experiments/overcooked_v2/configs/ocv2_step4_asymm_role_v2_e1rev.yaml
PRF=results_phase2/E1_preflight.json
PY=/apps/users/cxw/Document/CodeSpace/Selfs/CPR_REPO/.venv/bin/python
TAG=e1rev
mkdir -p results_phase3 logs_phase3
run_one() {
  local method=$1
  local seed=$2
  local gpu=$3
  local out=results_phase3/${TAG}_${method}_s${seed}
  local log=logs_phase3/${TAG}_train_${method}_s${seed}.log
  local done_marker=$out/asymm_advantages/${method}/full_support/seed${seed}/metrics.json
  if [ -f "$done_marker" ]; then echo "[skip] ${method} s${seed}"; return 0; fi
  rm -rf "$out"
  echo "[start] ${method} s${seed} GPU=${gpu} -> ${out}"
  CUDA_VISIBLE_DEVICES=$gpu JAX_PLATFORMS=cpu PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 timeout 5400 $PY \
    experiments/overcooked_v2/train_aris.py --config $CFG --preflight_path $PRF \
    --graph_variant full_support --method $method --seed $seed --output_dir $out > $log 2>&1
  echo "[done ] ${method} s${seed} exit=$? -> ${done_marker}"
}
methods=(aris_bellman base_only global_gru flat_factor partner_id_q)
seeds=(0 1 2 3 4)
queue=()
for m in "${methods[@]}"; do for s in "${seeds[@]}"; do queue+=("$m:$s"); done; done
GPUS=(4 5 6)
i=0
while [ $i -lt ${#queue[@]} ]; do
  pids=()
  for g in "${GPUS[@]}"; do
    [ $i -ge ${#queue[@]} ] && break
    IFS=':' read -r m s <<< "${queue[$i]}"
    ( run_one $m $s $g ) &
    pids+=($!); i=$((i+1))
  done
  for p in "${pids[@]}"; do wait $p; done
done
echo "[E1REV ALL DONE]"
