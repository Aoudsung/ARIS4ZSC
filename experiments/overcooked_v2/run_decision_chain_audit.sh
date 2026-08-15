#!/usr/bin/env bash
set -euo pipefail

repo=${1:?repository path}
run_root=${2:?registered run root}
partner_manifest=${3:?partner manifest}
gpu=${4:?CUDA device index}
output=${5:-"$run_root/decision_chain_audit"}

python="$repo/.venv/bin/python"
config="$repo/experiments/overcooked_v2/configs/delta_unified_simple_development.yaml"
policy_manifest="$run_root/exploratory_h2_support_eval/policy_manifests/delta_active.json"
active_evaluation="$run_root/exploratory_h2_support_eval/evaluations/delta_active"
cuda_bin="$repo/.venv/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin"

mkdir -p "$output/evaluations" "$output/logs"
cd "$repo"

run_cuda() {
  env \
    PATH="$cuda_bin:$PATH" \
    CUDA_VISIBLE_DEVICES="$gpu" \
    JAX_PLATFORMS=cuda \
    JAX_DEFAULT_MATMUL_PRECISION=highest \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    PYTHONUNBUFFERED=1 \
    "$@"
}

pids=()
for mode in reference_only residual passive; do
  run_cuda "$python" -m experiments.overcooked_v2.delta_zsc evaluate \
    --config "$config" \
    --run-kind development \
    --evaluation-mode common_partner \
    --partner-manifest "$partner_manifest" \
    --partner-role development_support \
    --seed 0 \
    --execution-mode "$mode" \
    --policy-manifest "$policy_manifest" \
    --output "$output/evaluations/$mode" \
    > "$output/logs/evaluate_$mode.log" 2>&1 &
  pids+=("$!")
done

run_cuda "$python" -m experiments.overcooked_v2.delta_zsc decision-chain-audit \
  --config "$config" \
  --run-kind development \
  --partner-manifest "$partner_manifest" \
  --partner-role development_support \
  --deployment "$run_root/development_matrix/runs/k-4/delta_active/seed-0/final_deployment" \
  --deployment "$run_root/development_matrix/runs/k-4/delta_active/seed-1/final_deployment" \
  --deployment "$run_root/development_matrix/runs/k-4/delta_active/seed-2/final_deployment" \
  --deployment "$run_root/development_matrix/runs/k-4/delta_active/seed-3/final_deployment" \
  --deployment "$run_root/development_matrix/runs/k-4/delta_active/seed-4/final_deployment" \
  --seed 0 \
  --trajectory-steps 256 \
  --anchor-states 240 \
  --output "$output/chains" \
  > "$output/logs/decision_chain_audit.log" 2>&1 &
pids+=("$!")

for pid in "${pids[@]}"; do
  wait "$pid"
done

"$python" -m experiments.overcooked_v2.delta_zsc summarize-decision-chain-audit \
  --audit "$output/chains/decision_chain_audit.json" \
  --evaluation "reference_only=$output/evaluations/reference_only" \
  --evaluation "residual=$output/evaluations/residual" \
  --evaluation "passive=$output/evaluations/passive" \
  --evaluation "active=$active_evaluation" \
  --bootstrap-replicates 9999 \
  --seed 0 \
  --output "$output/decision_chain_audit_report.md"
