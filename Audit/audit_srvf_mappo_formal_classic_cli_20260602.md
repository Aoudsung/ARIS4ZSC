# Audit: SRVF-MAPPO Formal Classic CLI

Date: 2026-06-02

## Scope

This audit covers the formal classic Overcooked training/evaluation entrypoint
added to `raob/srvf_mappo.py` after the single-file migration.

## Method Correspondence

- `python -m raob.srvf_mappo formal-classic` trains only SRVF-MAPPO; no baseline
  training path is introduced.
- V0 data collection uses source partners only and records raw-reward discounted
  returns over public chart states.
- Source `IRFTable` collection computes
  `r + gamma * V0(g_next) - V0(g)` and centered action residuals.
- `RolloutBatch` is collected from classic Overcooked source-partner rollouts;
  belief updates use only `(g, ego_action, Delta z)`.
- `SourceBatch` is derived only from source IRF rows and source beta factors.
- Target partners are used for closed-loop evaluation and offline response-only
  posterior diagnostics; target rewards, target action labels, target identity,
  and target partner ids are not used for training or alpha calibration.
- Checkpoints, V0 metrics, source/target IRF tables, per-seed evaluation JSON,
  `summary.json`, `status.json`, and `report.md` are written under a fresh run
  directory.

## Server Validation

- `ruff check raob tests`: passed.
- `compileall -q raob tests`: passed.
- `pytest -q`: `5 passed`.
- `python -m raob.srvf_mappo self-test`: passed, including gradient isolation.
- Tiny real classic smoke completed:
  `runs/srvf_mappo_classic_cli_smoke_20260602_134629`.

## Formal Run Status

- Formal method-only classic performance run launched on the server:
  `runs/srvf_mappo_classic_formal_method_only_20260602_134826`.
- Server PID: `1226665`.
- Initial seed-11 training metric was written with `rollout_rows=3600` at
  update `0`; no NaN/Inf was observed in the logged losses.
- The run was still active at the time of this audit note, with `status.json`
  reporting stage `train_eval_seed`.

The completed method result must be read from that run directory's
`summary.json` and `report.md` after `status.json` becomes `complete`.
