#!/bin/bash
# E1-rev stage-1 held-out eval orchestrator (EXPERIMENT_CHAIN_PLAN 补充C §14.2).
# Runs from the WAVE dir (checkpoint-relative graph paths resolve) with CODE
# from the w1fix archive (window-1 fixes: hard reward-scale gate, cache schema
# v3, throughput fields) via PYTHONPATH + absolute script path.
#
# LDS-C1 producer-side fix baked in:
#   * SUCCESS marker written ONLY on exit==0 AND non-empty output JSON;
#   * resume skips ONLY on the marker (never on output-file existence);
#   * no destructive cleanup of shared dirs.
# Bash-gotcha discipline (19b653d post-mortem): one `local` per line.
set -u
cd /apps/users/cxw/Document/CodeSpace/Selfs/ARIS4ZSC-phase1-4d680ab || exit 1
W1FIX=/apps/users/cxw/Document/CodeSpace/Selfs/ARIS4ZSC-w2fix-108bb89  # S28 fix included
PY=/apps/users/cxw/Document/CodeSpace/Selfs/CPR_REPO/.venv/bin/python
EVAL_SEED=0          # FIXED across all checkpoints (baseline-cache reuse, §14.2)
EPISODES=25          # stage-1 screening depth
PARTNERS="heldout-handoff-alternate-yield,heldout-resource-server-claim"
CACHE=results_phase3_eval/.baseline_cache
mkdir -p results_phase3_eval logs_phase3_eval

run_one() {
  local method=$1
  local seed=$2
  local gpu=$3
  local ckpt="results_phase3/e1rev_${method}_s${seed}/asymm_advantages/${method}/full_support/seed${seed}/checkpoint.pt"
  local out="results_phase3_eval/e1rev_${method}_s${seed}_stage1.json"
  local marker="${out}.SUCCESS"
  local log="logs_phase3_eval/eval_${method}_s${seed}.log"
  if [ -f "$marker" ]; then echo "[skip-ok] ${method} s${seed}"; return 0; fi
  if [ ! -f "$ckpt" ]; then echo "[no-ckpt] ${method} s${seed} (not deployable — expected for guard-fail seeds)"; return 0; fi
  echo "[start] ${method} s${seed} GPU=${gpu}"
  CUDA_VISIBLE_DEVICES=$gpu JAX_PLATFORMS=cpu PYTHONPATH=$W1FIX PYTHONDONTWRITEBYTECODE=1 \
    timeout 10800 "$PY" "$W1FIX/experiments/overcooked_v2/evaluate_aris.py" \
      --checkpoint "$ckpt" --graph_variants full_support \
      --partners "$PARTNERS" --episodes "$EPISODES" --seed "$EVAL_SEED" \
      --baseline_cache_dir "$CACHE" --fast \
      --output "$out" > "$log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ] && [ -s "$out" ]; then
    : > "$marker"
    echo "[done ] ${method} s${seed} exit=0 -> ${out}"
  else
    echo "[FAIL ] ${method} s${seed} exit=${rc} (no marker; see ${log})"
  fi
}

methods=(aris_bellman base_only global_gru flat_factor partner_id_q)
seeds=(0 1 2 3 4)
gpus=(4 5 6)
queue=()
for m in "${methods[@]}"; do for s in "${seeds[@]}"; do queue+=("$m:$s"); done; done

i=0
declare -A slot_pid=()
while [ $i -lt ${#queue[@]} ] || [ ${#slot_pid[@]} -gt 0 ]; do
  for g in "${gpus[@]}"; do
    if [ -n "${slot_pid[$g]:-}" ]; then
      if ! kill -0 "${slot_pid[$g]}" 2>/dev/null; then
        wait "${slot_pid[$g]}" 2>/dev/null
        unset "slot_pid[$g]"
      fi
    fi
    if [ -z "${slot_pid[$g]:-}" ] && [ $i -lt ${#queue[@]} ]; then
      IFS=: read -r m s <<< "${queue[$i]}"
      i=$((i+1))
      ( run_one "$m" "$s" "$g" ) &
      slot_pid[$g]=$!
    fi
  done
  sleep 20
done
echo "[E1REV_EVAL_STAGE1 ALL DONE]"
