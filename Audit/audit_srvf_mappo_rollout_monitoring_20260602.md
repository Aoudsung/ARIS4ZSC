# Audit: SRVF-MAPPO Rollout Monitoring

Date: 2026-06-02

## Scope

This audit covers the process-monitoring addition for SRVF-MAPPO classic
Overcooked training.

## Correspondence Check

- The change adds monitoring fields to `RolloutBatch` without changing the
  tensors used by MAPPO, SRVF source likelihood, V0, belief updates, or
  evaluation.
- Reward monitor values come only from source-partner training rollouts.
- Target rewards, target action labels, target partner ids, and target identity
  are not introduced into training-time logs.
- `train_metrics_seed*.jsonl` now includes compact reward/action/timing fields.
- `rollout_monitor_seed*.jsonl` records full source rollout diagnostics,
  including by-partner reward, alpha, action distribution, and timing.
- Existing formal run
  `runs/srvf_mappo_classic_formal_method_only_20260602_134826` is intentionally
  not modified or restarted; this monitor applies to later runs.

## Server Validation

- `ruff check raob tests`: passed.
- `compileall -q raob tests`: passed.
- `pytest -q`: `5 passed`.
- Fresh classic smoke run completed:
  `runs/srvf_mappo_classic_monitor_smoke_20260602_142554`.
- Smoke produced `rollout_monitor_seed11.jsonl` with reward, by-partner,
  action-distribution, alpha/fallback, and timing fields.
