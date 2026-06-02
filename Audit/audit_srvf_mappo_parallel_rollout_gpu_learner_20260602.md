# Audit: SRVF-MAPPO Parallel Rollout GPU Learner

Date: 2026-06-02

## Scope

This audit covers the high-resource training path added for classic
Overcooked SRVF-MAPPO.

## Correspondence Check

- The change does not modify V0, source IRF labels, SRVF heads, belief update,
  alpha calibration, MAPPO loss, source likelihood, partner split, or evaluation.
- Parallel workers collect source-partner `RolloutBatch` fragments only.
- The main process concatenates rollout fragments and performs all optimizer
  steps on the configured learner device.
- Worker GOAT partner policies default to CPU to avoid multiplying GPU memory
  use across workers.
- Target partners remain evaluation-only.
- Existing formal run
  `runs/srvf_mappo_classic_formal_method_only_20260602_134826` is not stopped
  or mutated by this code path.

## Server Validation

- `ruff check raob tests`: passed.
- `compileall -q raob tests`: passed.
- `pytest -q`: `7 passed`.
- Small parallel smoke completed:
  `runs/srvf_mappo_parallel_small_smoke_20260602_144932`.
- High-resource smoke completed:
  `runs/srvf_mappo_parallel_18w_smoke_20260602_145007`.
- High-resource smoke produced `worker_count=18`, `gpu_learner_epochs=4`,
  `resource_mode=parallel_rollout_gpu_learner`, `rows_per_sec`,
  `collector_elapsed_sec`, and `learner_elapsed_sec` in the monitor logs.
