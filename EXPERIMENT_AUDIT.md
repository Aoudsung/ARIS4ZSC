# Experiment Audit Report

**Date**: 2026-06-27
**Auditor**: GPT-5.5 (Codex CLI, xhigh reasoning, cross-model, read-only)
**Project**: ARIS-Bellman OvercookedV2 (ARIS4ZSC)
**Trace**: `.aris/traces/experiment-audit/20260627_run01/` (raw prompt + output)

## Overall Verdict: FAIL
## Integrity Status: fail

> The FAIL is driven by **stale/internally-inconsistent result artifacts** (produced by
> PRE-P1 code), **not by fabricated or phantom-DONE claims** — the tracker and dashboard
> honestly mark formal experiments as NOT STARTED. The just-implemented P1 honesty fixes
> address the root causes for FUTURE runs, but the existing result JSONs predate them.

## Checks

### A. Ground Truth Provenance: WARN
- This is an RL task: "ground truth" = environment return / reference baselines (no supervised labels).
- Raw `mean_return` is real simulation return (`evaluate_aris.py:460-471,517`); random baseline is rollout-generated (`evaluate_aris.py:730-756`); external reference requires checkpoints (`:759-806`).
- The within-run "reference closure" is derived from the evaluated outputs (best variant), i.e. a proxy. Current P1 code labels it `within_run_relative_return_not_reference_gap_closure` (`evaluate_aris.py:822-855`).
- **Code-vs-results mismatch**: PRE-P1 eval JSONs still carry self-reference `raw_value: 1.0`/`status: ok` (`eval_aris_bellman_seed0.json:3975-3979`); P1 code now suppresses that and sets `status: within_run_reference_variant` (`evaluate_aris.py:838-840`).
- No official external benchmark eval script (custom env rollout).

### B. Score Normalization: WARN
- No normalization by max/min/mean of the model's own prediction tensor found.
- Reference-closure denominator is a return gap, not prediction stats (`diagnostics.py:53-76`); raw returns reported in matrix (`step4_matrix_5kupd_fulldiag_s0.json:3-118`).
- Suspicious self-referential `mean_within_run_relative_return_raw: 1.0` in PRE-P1 eval files (each file evaluates one variant and normalizes against itself). Now labeled as within-run proxy in code, but old files remain misleading if read as oracle-gap closure.

### C. Result File Existence / Claim Match: FAIL  ← decisive
- All listed files exist (10 eval JSONs, 24 metrics JSONs).
- **Provenance conflict**: `step4_matrix_5kupd_fulldiag_s0.json` records `full_diagnostics: false` (`:162-183`) while filename/manifest imply full diagnostics and `eval/eval_results.json` says `full_diagnostics: true` (`:1-6`). Gates G1/G3 FAIL (`:137-154`).
- P1 orchestration would now copy manifest provenance into the matrix + assert seed/path consistency (`run_step4_microtrain.py:511-555`); the existing 5k matrix predates that guard.
- NOT a phantom-DONE: tracker/dashboard mark formal experiments not started (`EXPERIMENT_TRACKER.md:8-15`, `PROJECT_DASHBOARD.md:45-46`). The failure is stale/inconsistent provenance + failed gates.

### D. Dead Code Detection: WARN
- Diagnostics (`delta_info`, `mi`, `diagnostic_cost`, belief-swap) are called and serialized (`evaluate_aris.py:568-579`; `eval_aris_bellman_seed0.json:3942-3954`).
- PRE-P1 eval files contain only q-proxy factor deletion (`eval_aris_bellman_seed0.json:7939-7941`); no listed artifact contains **rollout** factor-deletion return drops, though the P1 rollout path now exists (`evaluate_aris.py:137-151,658-693`).
- Minor: `_finite_summary` defined but uncalled (`evaluate_aris.py:1032`).

### E. Scope Assessment: WARN
- Pilot scale: one layout (`configs/ocv2_step4.yaml:1`), mostly seed 0 for the 5000-update artifacts, 3 eval episodes/partner (`step4_matrix_5kupd_fulldiag_s0.json:162-183`), plus an older 3-seed 1000-update verify matrix.
- Plan requires stronger formal scope (`EXPERIMENT_PLAN.md:186-196`); dashboard says formal not started (`PROJECT_DASHBOARD.md:30,45-46`). Docs mostly avoid "robust/formal/comprehensive" completion claims — good.

### F. Evaluation Type: simulation_only / self_supervised_proxy (PASS)
- raw returns / random baseline / stability probe → `simulation_only`; within-run relative return + CE construction → `self_supervised_proxy`. No `real_gt` (no supervised labels), no human eval. Correctly classified for an RL task.

## Action Items
1. **Regenerate result artifacts with P1 code** (and after the P0 stability fix) so the matrix carries consistent provenance (`full_diagnostics` read from the eval manifest) and the within-run proxy is honestly labeled. The current 5k "fulldiag" matrix is stale/inconsistent — do not cite it.
2. **Do not read PRE-P1 eval JSONs' `raw_value: 1.0` / `mean_within_run_relative_return_raw: 1.0` as oracle/reference-gap closure** — they are within-run self-reference. Mark or quarantine these files.
3. Produce **rollout** factor-deletion return-drop artifacts (not just q-proxy) for any causal-deletion claim.
4. Remove or wire `_finite_summary` (`evaluate_aris.py:1032`).
5. Keep scope language pilot-honest until ≥5 seeds × ≥5 diagnostic layouts (formal) are run.

## Claim Impact (all 5 are Type-B; formal experiments NOT STARTED)
- **C1** ZSC = Bellman control over value-sufficient factor beliefs: **unsupported** (5000-update greedy policy collapsed to noop; provenance conflict).
- **C2** Factor beliefs suffice / partner identity unnecessary: **not tested**.
- **C3** Diagnostic behavior emerges from Bellman control: **not tested cleanly** (diagnostics computed on a degenerate policy).
- **C4** Support graph is load-bearing: **contradicted in this smoke run** (shuffled_relevance is the only non-collapsed aris variant — backwards).
- **C5** Δ_info predicts reward better than MI: **not tested cleanly** (degenerate policy + within-run proxy).
