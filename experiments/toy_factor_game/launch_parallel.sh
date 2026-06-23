#!/usr/bin/env bash
set -euo pipefail

TASK="${1:-}"
if [[ -z "$TASK" ]]; then
  echo "usage: $0 exp1|exp3|exp4 [--seed N] [--output_dir DIR] [--hidden_dim N] [--max_steps N] [--n_episodes N] [--gpus CSV] [--jobs_per_gpu N] [--batch_size N] [--dry_run]" >&2
  exit 2
fi
shift

SEED=42
OUTPUT_DIR="results/toy"
HIDDEN_DIM=128
MAX_STEPS=50
N_EPISODES=8000
GPUS="0"
JOBS_PER_GPU=3
BATCH_SIZE=16
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed) SEED="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --hidden_dim) HIDDEN_DIM="$2"; shift 2 ;;
    --max_steps) MAX_STEPS="$2"; shift 2 ;;
    --n_episodes) N_EPISODES="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --jobs_per_gpu) JOBS_PER_GPU="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --dry_run) DRY_RUN=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

IFS=',' read -r -a GPU_LIST <<< "$GPUS"
MAX_PARALLEL=$(( ${#GPU_LIST[@]} * JOBS_PER_GPU ))
if [[ "$MAX_PARALLEL" -le 0 ]]; then
  echo "--gpus and --jobs_per_gpu must define at least one worker" >&2
  exit 2
fi

case "$TASK" in
  exp1)
    GRAPH_VARIANTS=(full_support full_support full_support full_support full_support full_support overcomplete overcomplete overcomplete overcomplete overcomplete overcomplete)
    METHODS=(base_only aris_bellman flat_latent global_gru oracle_belief_factorq oracle_belief_flatq base_only aris_bellman flat_latent global_gru oracle_belief_factorq oracle_belief_flatq)
    ;;
  exp3)
    GRAPH_VARIANTS=(full_support full_support full_support full_support full_support full_support)
    METHODS=(base_only aris_bellman flat_latent global_gru oracle_belief_factorq oracle_belief_flatq)
    ;;
  exp4)
    GRAPH_VARIANTS=(full_support overcomplete overcomplete_minus_noncritical minus_critical random_same_size complete_option_graph shuffled_routes shuffled_relevance)
    METHODS=(aris_bellman aris_bellman aris_bellman aris_bellman aris_bellman aris_bellman aris_bellman aris_bellman)
    ;;
  *)
    echo "unknown task: $TASK; expected exp1, exp3, or exp4" >&2
    exit 2
    ;;
esac

running=0
for idx in "${!GRAPH_VARIANTS[@]}"; do
  gpu="${GPU_LIST[$(( idx % ${#GPU_LIST[@]} ))]}"
  graph="${GRAPH_VARIANTS[$idx]}"
  method="${METHODS[$idx]}"
  cmd=(
    python experiments/toy_factor_game/train.py
    --seed "$SEED"
    --n_episodes "$N_EPISODES"
    --hidden_dim "$HIDDEN_DIM"
    --method "$method"
    --graph_variant "$graph"
    --max_steps "$MAX_STEPS"
    --batch_size "$BATCH_SIZE"
    --output_dir "$OUTPUT_DIR"
  )
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "CUDA_VISIBLE_DEVICES=$gpu ${cmd[*]}"
    continue
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" &
  running=$(( running + 1 ))
  if [[ "$running" -ge "$MAX_PARALLEL" ]]; then
    wait -n
    running=$(( running - 1 ))
  fi
done

if [[ "$DRY_RUN" -eq 0 ]]; then
  wait
fi
