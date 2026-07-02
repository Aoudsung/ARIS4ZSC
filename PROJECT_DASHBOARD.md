# PROJECT_DASHBOARD.md

**ARIS-Bellman for Zero-Shot Coordination — pipeline status & decisive-results tracker.**
Last updated: 2026-06-26 · Read this first for "where am I" (30 seconds).

> Required entrypoint per [CLAUDE.md](CLAUDE.md). Execution rules live in
> [OPERATING_CONSTRAINTS.md](OPERATING_CONSTRAINTS.md); this file is *status*, not *permission*.

---

## Terminology (authoritative)

- **ARIS** = the harness (Auto-claude-code Research In Sleep). The tooling under `.claude/skills/`.
- **ARIS-Bellman** = the research method. Factor-local Bellman control for ZSC.
  Code in `src/aris_bellman/` + `experiments/overcooked_v2/`.

This project (`ARIS4ZSC`) = *using ARIS the harness to develop the ARIS-Bellman method.*

---

## 1. Project identity

| | |
|---|---|
| **Title** | Zero-Shot Coordination via Bellman Control over Value-Sufficient Interaction Factor Beliefs |
| **Proposal** | v4 (ARIS-Bellman restructure), 2026-06-23 — [FINAL_PROPOSAL.md](idea-stage/refine-logs/FINAL_PROPOSAL.md) |
| **Thesis** | Partner diversity supplies priors; test-time coordination requires Bellman control over the small set of *value-sufficient interaction factors* that determine the current best response. |
| **Benchmark** | JaxMARL **OvercookedV2** (test-time protocol formation) + toy_factor_game (regression) |
| **Target tier** | ICLR / NeurIPS / ICML class |
| **Compute budget** | 2,200–4,500 GPU-hours (pilot: 500–1,000); used so far: ~0 formal |

---

## 2. Pipeline status — CURRENT STAGE

**Stage: W1.5 (experiment-bridge) — pre-experiment infrastructure hardening.**

The OvercookedV2 adapter, CE pipeline, support-graph builder, and a "Step-4
micro-train" orchestration exist and are being **correctness-hardened** before any
formal experiment runs. Recent work is gates/semantics, not results:
event semantics, CE scheduling, **reward-scale consistency gate**, per-variant eval
split, parallel/​single-GPU eval, preflight-as-hard-gate (see
[README_FIXES_20260624.md](README_FIXES_20260624.md) and `git log`: P0–P6 fixes).

**Formal Experiments 1–5: NOT STARTED.** Next milestone = first accepted-preflight
formal run on a diagnostic layout.

---

## 3. Decisive results — claims ↔ experiments ↔ status

Five claims (proposal §11) map to five experiments (proposal §7). All Type-B
(cross-model + human acquittal required; see
[OPERATING_CONSTRAINTS.md](OPERATING_CONSTRAINTS.md) §4).

| # | Claim (must be supported) | Experiment | Status |
|---|---------------------------|------------|--------|
| 1 | ZSC = Bellman control over value-sufficient factor beliefs; high `Δ_info` actions predict reward / oracle-gap closure, high MI alone does not | Exp 1 — Bellman diagnostic value | ⬜ NOT STARTED |
| 2 | Factor-local beliefs generalize compositionally beyond flat partner latents | Exp 2 — factor graph vs flat latent | ⬜ NOT STARTED |
| 3 | Factor beliefs are value-sufficient for control (belief-swap / factor-deletion are causal) | Exp 3 — value-sufficiency ablation | ⬜ NOT STARTED |
| 4 | The support graph is load-bearing, not decorative capacity | Exp 4 — support-graph robustness | ⬜ NOT STARTED |
| 5 | Factor beliefs track adaptive protocol formation, degrade gracefully | Exp 5 — adaptive-partner stress test | ⬜ NOT STARTED |
| — | Human partner study | Human study (Month 4) | ⬜ NOT STARTED |

Source of truth for run status: [EXPERIMENT_TRACKER.md](idea-stage/refine-logs/EXPERIMENT_TRACKER.md)
(execution checklist) + `EXPERIMENT_LOG.md` (results record — **not yet created**;
ARIS convention, see §6).

---

## 4. Readiness gates (what unlocks each downstream phase)

| Phase / skill | Locked until… |
|---------------|---------------|
| Formal Exp 1–5 runs | preflight accepted for the target layout(s) + invariant gate green (`python3 .aris/tools/aris_bellman_fidelity_gate.py` exits 0) |
| `/result-to-claim` | an experiment has produced a real `EXPERIMENT_LOG.md` result file |
| `/auto-review-loop` (W2) | ≥1 decisive result supported by cross-model verdict |
| `/paper-writing` (W3) | `NARRATIVE_REPORT.md` exists + main claims supported |
| `/rebuttal`, resubmit, camera-ready | external reviews / acceptance received |

Do not start a locked phase. Crossing a gate requires recorded evidence, not inference.

---

## 5. Method invariants (the load-bearing constraints)

These are *correctness* constraints of ARIS-Bellman. Violating them invalidates the
science, not just the run. Full lists: [OvercookedV2_plan.md](artifacts/OvercookedV2_plan.md)
§21 (22 "what not to do"), §15 (preflight); proposal §11 (non-claims), §12 (impl alignment).

Core set — now enforced mechanically by [`.aris/tools/aris_bellman_fidelity_gate.py`](.aris/tools/aris_bellman_fidelity_gate.py) (checks I1–I9; static, runs in-boundary; ✅ green on current tree):

- **No G-TVOI / MI selector** in the deployed method. `Δ_info` and MI are *post-hoc
  diagnostics only*. Action selection is pure Bellman `argmax_ω Q(s,b,ω)`.
- **Single TD loss.** No response-prediction / calibration / sparsity / supervised
  factor-label / probe-selection losses in the main method.
- **CE is preprocessing.** Never estimate CE inside the training loop.
- **Reward-scale consistency** across preflight, CE local returns, and training target.
- **Preflight is a hard gate** before formal training (`--preflight_path`).
- **Articulation-point bottlenecks**, not `degree ≤ 2`.
- **Factor deletion removes 3 things**: latent state + evidence route + action relevance.
- **No true-factor oracle and no factor-accuracy as a main V2 metric.** Use
  control-grounded reference-gap closure.

---

## 6. Artifact map & known divergences

| Artifact | Location | Notes |
|----------|----------|-------|
| Final proposal | `idea-stage/refine-logs/FINAL_PROPOSAL.md` | ✅ |
| Experiment plan | `idea-stage/refine-logs/EXPERIMENT_PLAN.md` | ✅ |
| Experiment tracker | `idea-stage/refine-logs/EXPERIMENT_TRACKER.md` | ✅ (status checklist) |
| Experiment log (results) | `idea-stage/refine-logs/EXPERIMENT_LOG.md` | ⬜ missing — create on first result |
| Findings log | `findings.md` | ⬜ missing — ARIS convention for debug/decision log |
| Migration plan | `artifacts/OvercookedV2_plan.md` | ✅ |

**Divergence to reconcile (low priority):** ARIS convention is `refine-logs/` at
project root (sibling of `idea-stage/`); this project nests it as
`idea-stage/refine-logs/`. Downstream skills that hardcode `refine-logs/…` may not
find these. Decide: move, or symlink, or pin the path in `.aris/config.json`.

---

## 7. Entry points

- Execution rules → [OPERATING_CONSTRAINTS.md](OPERATING_CONSTRAINTS.md)
- Agent coordination / Codex scope → [AGENTS.md](AGENTS.md)
- Remote contract → [CUSTOMER.md](CUSTOMER.md)
- Project posture (machine-readable) → [`.aris/config.json`](.aris/config.json)
