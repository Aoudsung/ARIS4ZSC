# Audit: SRVF-MAPPO Single-File Migration

Date: 2026-06-02

## Scope

This audit covers the migration from the obsolete V0-SRVF / SRVF-IRF code path
to the single-file `raob/srvf_mappo.py` implementation based on
`Method_design.md`.

## Correspondence Check

- Core object `xi=(M,beta)` is represented by `SRVFBelief` beta posterior and
  empirical-Bayes `alpha`.
- Posterior-predictive score is implemented as
  `A0(g,a) + alpha * U(g,a)^T mu` in `SRVFBelief.score_from_posterior`.
- `alpha=0` gives population fallback through `a0`; `alpha=1` gives raw SRVF.
- MAPPO actor logits use continuation logits plus `srvf_score / tau`.
- Policy loss is separated from SRVF source likelihood in `UnifiedLoss`.
- `detach_srvf_in_actor=True` blocks policy gradients from updating SRVF heads.
- Source likelihood consumes `SourceBatch`.
- Classic rollout collection emits `RolloutBatch` and updates belief only with
  `(g, action, Delta z)`.
- Classic Overcooked wrapper supplies tensor helpers for actor, critic, and
  public chart inputs.

## Removed Obsolete Paths

- Removed old `raob/srvf_irf` finite-reservoir checkpoint/posterior/metrics path.
- Removed OGC/JAX benchmark files.
- Removed old SRVF-IRF config.
- Replaced README and package metadata with SRVF-MAPPO classic-only framing.

## Open Validation Items

- Server `compileall` passed for `raob tests`.
- Server pytest passed: `4 passed in 2.11s`.
- Server ruff passed: `All checks passed!`.
- Server classic Overcooked smoke confirmed real wrapper emits a nonempty
  `RolloutBatch`: `rows=2`, `obs=(2,520)`, `global=(2,1040)`, `g=(2,520)`,
  `phase=(2,0)`.
- `SourceBatch` conversion is covered by synthetic pytest; real source IRF table
  collection remains the next benchmark-data step.
- No local runtime validation was executed, in compliance with AGENTS.md.
