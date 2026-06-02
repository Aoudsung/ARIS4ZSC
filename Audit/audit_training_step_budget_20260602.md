# Audit: Formal Classic Training Step Budget

Date: 2026-06-02

## Scope

- Align formal classic Overcooked training budget with the paper-style `1e8` environment-step axis.
- Preserve SRVF-MAPPO method logic, rollout contents, source/target split, and loss definitions.

## Method-Design Check

- `Method_design.md` defines SRVF-MAPPO as MAPPO on a V0-anchored posterior-predictive belief MDP.
- The changed code only affects how long formal training runs and how progress is reported.
- No changes were made to V0 fitting, SRVF heads, belief updates, alpha calibration, PPO losses, source IRF targets, or target leakage guards.

## Implementation Correspondence

- `formal-classic` now defaults to `--target-env-steps 100000000` and `--updates 0`.
- With `--updates 0`, each seed trains until actual collected rollout rows satisfy `cumulative_env_steps >= target_env_steps`.
- Explicit `--updates N` keeps fixed-update mode for smoke/debug runs.
- `--smoke` forces `updates=2` and disables the target-step stopping condition.
- Metrics, monitor JSONL, status JSON, eval JSON, summary JSON, and report output now include step-budget progress fields.

## Expected Default Scale

With the current high-resource default:

```text
worker_count = 18
rollout_episodes_per_partner = 1
horizon = 400
rollout_rows/update ~= 7200
target_env_steps = 100000000
estimated_updates/seed ~= 13889
```

The implementation does not hard-code this estimate; it uses actual rollout rows.

## Validation

Server validation completed:

- `PYTHONPATH=. .venv/bin/python -m ruff check raob tests`: passed.
- `PYTHONPATH=. .venv/bin/python -m compileall -q raob tests`: passed.
- `PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q`: 13 passed.
- Small server smoke with `--target-env-steps 14400`: passed.
  - Run: `runs/srvf_mappo_step_budget_smoke_20260602`.
  - Final `cumulative_env_steps`: `14400`.
  - Final `target_env_steps`: `14400`.
  - Final `env_step_progress`: `1.0`.
  - Final `estimated_updates_remaining`: `0`.
  - Updates completed: `18`.
  - Smoke used `workers=2`, so actual rollout rows were `800` per update.

## Audit Result

Passed. The code now aligns formal classic training with a per-seed environment-step budget without changing SRVF-MAPPO method logic.
