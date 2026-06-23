#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

SEEDS="42,200,201,202,203"
OUTPUT_DIR="results/toy"
GPUS="0,2,3,4,7"
JOBS_PER_GPU=6
N_EPISODES=8000
BATCH_SIZE=16
HIDDEN_DIM=128
MAX_STEPS=50
N_PER_CONV=5
MAX_MEMORY_USED_MB=20000
PYTHON_BIN=".venv/bin/python"
DRY_RUN=0
FORCE=0

usage() {
  cat >&2 <<'EOF'
usage: run_neural_campaign.sh [options]

Options:
  --seeds CSV                 Default: 42,200,201,202,203
  --output_dir DIR            Default: results/toy
  --gpus CSV                  Default: 0,2,3,4,7
  --jobs_per_gpu N            Default: 6
  --n_episodes N              Default: 8000
  --batch_size N              Default: 16
  --hidden_dim N              Default: 128
  --max_steps N               Default: 50
  --n_per_conv N              Default: 5
  --max_memory_used_mb N      Default: 20000
  --python_bin PATH           Default: .venv/bin/python
  --force                     Retrain even when model.pt and results.json exist
  --dry_run                   Print the deduplicated train/eval plan only
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds) SEEDS="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --jobs_per_gpu) JOBS_PER_GPU="$2"; shift 2 ;;
    --n_episodes) N_EPISODES="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --hidden_dim) HIDDEN_DIM="$2"; shift 2 ;;
    --max_steps) MAX_STEPS="$2"; shift 2 ;;
    --n_per_conv) N_PER_CONV="$2"; shift 2 ;;
    --max_memory_used_mb) MAX_MEMORY_USED_MB="$2"; shift 2 ;;
    --python_bin) PYTHON_BIN="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --dry_run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

require_positive_int() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[0-9]+$ ]] || [[ "$value" -le 0 ]]; then
    echo "$name must be a positive integer; got '$value'" >&2
    exit 2
  fi
}

require_nonnegative_int() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    echo "$name must be a non-negative integer; got '$value'" >&2
    exit 2
  fi
}

require_positive_int "--jobs_per_gpu" "$JOBS_PER_GPU"
require_positive_int "--n_episodes" "$N_EPISODES"
require_positive_int "--batch_size" "$BATCH_SIZE"
require_positive_int "--hidden_dim" "$HIDDEN_DIM"
require_positive_int "--max_steps" "$MAX_STEPS"
require_positive_int "--n_per_conv" "$N_PER_CONV"
require_nonnegative_int "--max_memory_used_mb" "$MAX_MEMORY_USED_MB"

IFS=',' read -r -a SEED_LIST <<< "$SEEDS"
IFS=',' read -r -a REQUESTED_GPUS <<< "$GPUS"
if [[ "${#SEED_LIST[@]}" -eq 0 || -z "${SEED_LIST[0]}" ]]; then
  echo "--seeds must contain at least one seed" >&2
  exit 2
fi
if [[ "${#REQUESTED_GPUS[@]}" -eq 0 || -z "${REQUESTED_GPUS[0]}" ]]; then
  echo "--gpus must contain at least one GPU index" >&2
  exit 2
fi

TRAIN_METHODS=(
  base_only aris_bellman flat_latent global_gru oracle_belief_factorq oracle_belief_flatq
  base_only aris_bellman flat_latent global_gru oracle_belief_factorq oracle_belief_flatq
  aris_bellman aris_bellman aris_bellman aris_bellman aris_bellman aris_bellman
)
TRAIN_GRAPHS=(
  full_support full_support full_support full_support full_support full_support
  overcomplete overcomplete overcomplete overcomplete overcomplete overcomplete
  shuffled_routes shuffled_relevance random_same_size overcomplete_minus_low_ce minus_critical complete_option_graph
)

if [[ "${#TRAIN_METHODS[@]}" -ne 18 || "${#TRAIN_GRAPHS[@]}" -ne 18 ]]; then
  echo "internal error: expected 18 deduplicated train jobs" >&2
  exit 2
fi

select_usable_gpus() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found; cannot dynamically select GPUs" >&2
    exit 2
  fi
  local smi
  smi="$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits)"
  local usable=()
  for requested in "${REQUESTED_GPUS[@]}"; do
    local trimmed="${requested//[[:space:]]/}"
    local line
    line="$(printf '%s\n' "$smi" | awk -F',' -v gpu="$trimmed" '$1+0 == gpu+0 {print; exit}')"
    if [[ -z "$line" ]]; then
      echo "requested GPU $trimmed not reported by nvidia-smi" >&2
      exit 2
    fi
    local used
    used="$(printf '%s\n' "$line" | awk -F',' '{gsub(/ /, "", $2); print $2+0}')"
    if [[ "$used" -le "$MAX_MEMORY_USED_MB" ]]; then
      usable+=("$trimmed")
    else
      echo "Skipping GPU $trimmed: memory.used=${used}MiB > ${MAX_MEMORY_USED_MB}MiB" >&2
    fi
  done
  if [[ "${#usable[@]}" -eq 0 ]]; then
    echo "no usable GPUs in requested set '$GPUS' under memory threshold ${MAX_MEMORY_USED_MB}MiB" >&2
    exit 2
  fi
  printf '%s\n' "${usable[@]}"
}

mapfile -t USABLE_GPUS < <(select_usable_gpus)
MAX_PARALLEL=$(( ${#USABLE_GPUS[@]} * JOBS_PER_GPU ))
if [[ "$MAX_PARALLEL" -le 0 ]]; then
  echo "usable GPU slots must be positive" >&2
  exit 2
fi

model_dir_for() {
  local seed="$1"
  local method="$2"
  local graph="$3"
  printf '%s/seed%s/%s/%s' "$OUTPUT_DIR" "$seed" "$method" "$graph"
}

running_job_pids() {
  local seed="$1"
  local method="$2"
  local graph="$3"
  ps -eo pid=,cmd= | awk \
    -v seed="$seed" \
    -v method="$method" \
    -v graph="$graph" \
    -v output_dir="$OUTPUT_DIR" '
      $0 ~ /experiments\/toy_factor_game\/train.py/ &&
      index($0, "--seed " seed) &&
      index($0, "--method " method) &&
      index($0, "--graph_variant " graph) &&
      index($0, "--output_dir " output_dir) {
        print $1
      }'
}

train_cmd_for() {
  local seed="$1"
  local method="$2"
  local graph="$3"
  printf '%q ' "$PYTHON_BIN" experiments/toy_factor_game/train.py \
    --seed "$seed" \
    --n_episodes "$N_EPISODES" \
    --hidden_dim "$HIDDEN_DIM" \
    --method "$method" \
    --graph_variant "$graph" \
    --max_steps "$MAX_STEPS" \
    --batch_size "$BATCH_SIZE" \
    --output_dir "$OUTPUT_DIR"
}

eval_cmd_for() {
  local seed="$1"
  printf '%q ' "$PYTHON_BIN" experiments/toy_factor_game/evaluate.py \
    --seed "$seed" \
    --output_dir "$OUTPUT_DIR" \
    --experiments 1,3,4 \
    --methods base_only,aris_bellman,flat_latent,global_gru,oracle_belief_factorq,oracle_belief_flatq,random_policy \
    --exp1_graph_variants full_support,overcomplete \
    --graph_variants full_support,overcomplete,overcomplete_minus_low_ce,minus_critical,random_same_size,complete_option_graph,shuffled_routes,shuffled_relevance \
    --hidden_dim "$HIDDEN_DIM" \
    --n_per_conv "$N_PER_CONV" \
    --max_steps "$MAX_STEPS"
}

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "DRY RUN"
  echo "Requested GPUs: $GPUS"
  echo "Usable GPUs: ${USABLE_GPUS[*]}"
  echo "Slots: $MAX_PARALLEL (${#USABLE_GPUS[@]} GPUs x $JOBS_PER_GPU jobs/GPU)"
  for seed in "${SEED_LIST[@]}"; do
    seed="${seed//[[:space:]]/}"
    [[ -z "$seed" ]] && continue
    echo "== seed $seed =="
    for idx in "${!TRAIN_METHODS[@]}"; do
      method="${TRAIN_METHODS[$idx]}"
      graph="${TRAIN_GRAPHS[$idx]}"
      gpu="${USABLE_GPUS[$(( idx % ${#USABLE_GPUS[@]} ))]}"
      out_dir="$(model_dir_for "$seed" "$method" "$graph")"
      action="TRAIN"
      if [[ "$FORCE" -eq 0 && -f "$out_dir/model.pt" && -f "$out_dir/results.json" ]]; then
        action="SKIP_EXISTING"
      fi
      running_pids="$(running_job_pids "$seed" "$method" "$graph" | paste -sd ',' -)"
      if [[ -n "$running_pids" ]]; then
        action="RUNNING_CONFLICT"
      fi
      echo -e "${action}\tseed=${seed}\tgpu=${gpu}\tmethod=${method}\tgraph=${graph}\tout=${out_dir}"
      if [[ "$action" == "RUNNING_CONFLICT" ]]; then
        echo "  running_pids=${running_pids}"
      fi
      echo "  CUDA_VISIBLE_DEVICES=$gpu $(train_cmd_for "$seed" "$method" "$graph")"
    done
    echo "EVAL seed=${seed}"
    echo "  $(eval_cmd_for "$seed")"
  done
  exit 0
fi

mkdir -p "$OUTPUT_DIR/logs"
JOBS_TSV="$OUTPUT_DIR/jobs.tsv"
PIDS_TSV="$OUTPUT_DIR/pids.tsv"
STATUS_TSV="$OUTPUT_DIR/status.tsv"
SEED_STATUS_TSV="$OUTPUT_DIR/seed_status.tsv"
STATUS_DIR="$OUTPUT_DIR/.campaign_status"
mkdir -p "$STATUS_DIR"
: > "$JOBS_TSV"
: > "$PIDS_TSV"
: > "$STATUS_TSV"
: > "$SEED_STATUS_TSV"
printf 'seed\tmethod\tgraph_variant\tgpu\toutput_dir\tlog\tcommand\n' > "$JOBS_TSV"
printf 'seed\tmethod\tgraph_variant\tgpu\tpid\toutput_dir\tlog\n' > "$PIDS_TSV"
printf 'seed\tmethod\tgraph_variant\tgpu\tstatus\texit_code\toutput_dir\tlog\n' > "$STATUS_TSV"
printf 'seed\ttrain_status\teval_status\teval_results\tlog\n' > "$SEED_STATUS_TSV"

append_status_files() {
  find "$STATUS_DIR" -type f -name '*.tsv' -print0 | sort -z | while IFS= read -r -d '' file; do
    cat "$file" >> "$STATUS_TSV"
    rm -f "$file"
  done
}

write_campaign_summary() {
  "$PYTHON_BIN" - "$OUTPUT_DIR" "$JOBS_TSV" "$STATUS_TSV" "$SEED_STATUS_TSV" "$MAX_PARALLEL" "${USABLE_GPUS[*]}" <<'PY'
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

output_dir = Path(sys.argv[1])
jobs_tsv = Path(sys.argv[2])
status_tsv = Path(sys.argv[3])
seed_status_tsv = Path(sys.argv[4])
max_parallel = int(sys.argv[5])
usable_gpus = sys.argv[6].split()

def read_tsv(path):
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))

statuses = read_tsv(status_tsv)
seed_statuses = read_tsv(seed_status_tsv)
by_seed = defaultdict(list)
for row in statuses:
    by_seed[row["seed"]].append(row)

summary = {
    "output_dir": str(output_dir),
    "usable_gpus": usable_gpus,
    "max_parallel": max_parallel,
    "n_jobs_recorded": len(read_tsv(jobs_tsv)),
    "n_status_rows": len(statuses),
    "seeds": [],
}
for seed_row in seed_statuses:
    seed = seed_row["seed"]
    counter = Counter(row["status"] for row in by_seed.get(seed, []))
    summary["seeds"].append({
        "seed": int(seed),
        "train_status": seed_row["train_status"],
        "eval_status": seed_row["eval_status"],
        "eval_results": seed_row["eval_results"],
        "log": seed_row["log"],
        "job_status_counts": dict(counter),
    })

with (output_dir / "campaign_summary.json").open("w") as f:
    json.dump(summary, f, indent=2, sort_keys=True)
PY
}

launch_train_job() {
  local seed="$1"
  local method="$2"
  local graph="$3"
  local gpu="$4"
  local log="$5"
  local out_dir="$6"
  local status_file="$7"
  local cmd
  cmd="$(train_cmd_for "$seed" "$method" "$graph")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$seed" "$method" "$graph" "$gpu" "$out_dir" "$log" "$cmd" >> "$JOBS_TSV"
  (
    set +e
    CUDA_VISIBLE_DEVICES="$gpu" bash -lc "$cmd" > "$log" 2>&1
    rc=$?
    status="done"
    if [[ "$rc" -ne 0 ]]; then
      status="failed"
    elif [[ ! -f "$out_dir/model.pt" || ! -f "$out_dir/results.json" ]]; then
      status="failed_missing_artifact"
      rc=20
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$seed" "$method" "$graph" "$gpu" "$status" "$rc" "$out_dir" "$log" > "$status_file"
    exit "$rc"
  ) &
  local pid=$!
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$seed" "$method" "$graph" "$gpu" "$pid" "$out_dir" "$log" >> "$PIDS_TSV"
}

run_seed() {
  local seed="$1"
  local log_dir="$OUTPUT_DIR/logs/seed${seed}"
  mkdir -p "$log_dir"
  local running=0
  local seed_failed=0
  local launched=0

  echo "== Seed $seed: training deduplicated checkpoint matrix =="
  for idx in "${!TRAIN_METHODS[@]}"; do
    local method="${TRAIN_METHODS[$idx]}"
    local graph="${TRAIN_GRAPHS[$idx]}"
    local gpu="${USABLE_GPUS[$(( launched % ${#USABLE_GPUS[@]} ))]}"
    local out_dir
    out_dir="$(model_dir_for "$seed" "$method" "$graph")"
    local log="$log_dir/${method}__${graph}.log"
    local status_file="$STATUS_DIR/seed${seed}__${method}__${graph}.tsv"
    local running_pids
    running_pids="$(running_job_pids "$seed" "$method" "$graph" | paste -sd ',' -)"

    if [[ -n "$running_pids" ]]; then
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$seed" "$method" "$graph" "-" "$out_dir" "$log" "running-conflict-pids=${running_pids}" >> "$JOBS_TSV"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$seed" "$method" "$graph" "-" "running_conflict" "30" "$out_dir" "$log" >> "$STATUS_TSV"
      echo "Running job conflict for seed=$seed method=$method graph=$graph output_dir=$OUTPUT_DIR pids=$running_pids" >&2
      seed_failed=1
      break
    fi

    if [[ "$FORCE" -eq 0 && -f "$out_dir/model.pt" && -f "$out_dir/results.json" ]]; then
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$seed" "$method" "$graph" "-" "$out_dir" "$log" "skip-existing" >> "$JOBS_TSV"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$seed" "$method" "$graph" "-" "skipped_existing" "0" "$out_dir" "$log" >> "$STATUS_TSV"
      continue
    fi

    launch_train_job "$seed" "$method" "$graph" "$gpu" "$log" "$out_dir" "$status_file"
    launched=$(( launched + 1 ))
    running=$(( running + 1 ))

    if [[ "$running" -ge "$MAX_PARALLEL" ]]; then
      if ! wait -n; then
        seed_failed=1
      fi
      append_status_files
      running=$(( running - 1 ))
      if [[ "$seed_failed" -ne 0 ]]; then
        break
      fi
    fi
  done

  while [[ "$running" -gt 0 ]]; do
    if ! wait -n; then
      seed_failed=1
    fi
    append_status_files
    running=$(( running - 1 ))
  done
  append_status_files

  local failed_count
  failed_count="$(awk -F'\t' -v seed="$seed" 'NR>1 && $1==seed && $5 ~ /^failed/ {n++} END {print n+0}' "$STATUS_TSV")"
  if [[ "$seed_failed" -ne 0 || "$failed_count" -ne 0 ]]; then
    printf '%s\t%s\t%s\t%s\t%s\n' "$seed" "failed" "not_run" "" "" >> "$SEED_STATUS_TSV"
    write_campaign_summary
    echo "Seed $seed failed during training; evaluation skipped" >&2
    return 1
  fi

  echo "== Seed $seed: evaluation Exp 1/3/4 =="
  local eval_log="$log_dir/evaluate_exp134.log"
  local eval_results="$OUTPUT_DIR/eval_results_seed${seed}.json"
  local eval_cmd
  eval_cmd="$(eval_cmd_for "$seed")"
  set +e
  bash -lc "$eval_cmd" > "$eval_log" 2>&1
  local eval_rc=$?
  set -e
  if [[ "$eval_rc" -ne 0 || ! -f "$eval_results" ]]; then
    printf '%s\t%s\t%s\t%s\t%s\n' "$seed" "done" "failed" "$eval_results" "$eval_log" >> "$SEED_STATUS_TSV"
    write_campaign_summary
    echo "Seed $seed evaluation failed; see $eval_log" >&2
    return 1
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' "$seed" "done" "done" "$eval_results" "$eval_log" >> "$SEED_STATUS_TSV"
  write_campaign_summary
}

echo "Requested GPUs: $GPUS"
echo "Usable GPUs: ${USABLE_GPUS[*]}"
echo "Max parallel jobs: $MAX_PARALLEL"

overall_rc=0
for raw_seed in "${SEED_LIST[@]}"; do
  seed="${raw_seed//[[:space:]]/}"
  [[ -z "$seed" ]] && continue
  if ! run_seed "$seed"; then
    overall_rc=1
    break
  fi
done

write_campaign_summary
exit "$overall_rc"
