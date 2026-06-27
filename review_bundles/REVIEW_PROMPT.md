# Deep Review Prompt — ARIS-Bellman OvercookedV2 (code + training results vs proposal)

> Paste this whole prompt to the reviewer (a strong cross-model LLM, e.g. GPT-5.x at
> high reasoning, or a human expert). Give it the bundle
> `ocv2_train_s0_review_*.zip` AND the proposal files
> `idea-stage/refine-logs/FINAL_PROPOSAL.md` + `artifacts/OvercookedV2_plan.md`.

---

## 0. Your role and standard

You are a **senior multi-agent RL researcher** serving simultaneously as (a) a top-tier
(ICML/NeurIPS) **reviewer** and (b) a **method-fidelity auditor**. You are performing a
**Type-B review**: not "did it run / compile" (machine-checkable, Type-A), but
"**is the method faithfully implemented, are the metrics sound, and are the scientific
claims actually supported by this evidence**". Be adversarial: your goal is to **falsify**
the claims and find the ways this could be wrong, fooling itself, or diverging from its own
proposal. Do **not** rubber-stamp. Every finding must cite concrete evidence as
`path:line` (code) or the specific metric/file (results). Where you cannot verify something
from the bundle, say so explicitly and list what you'd need.

## 1. What you are reviewing

A research bundle produced from a 1-seed × 5000-update verification micro-train of
**ARIS-Bellman** on JaxMARL **OvercookedV2** (`cramped_room`), plus the source code that
produced it:

- `README.md`, `TRAINING_ANALYSIS.md` — orientation + the team's own metrics analysis (treat
  as a claim to be checked, not ground truth).
- `results/` — 24 per-run `metrics.json` (6 methods × 4 graph variants × seed 0), the
  training log, `train_results.json`, and `reference_1000update_eval_matrix.json` (a prior
  1000-update **greedy** eval matrix + gates for comparison).
- `ce/` — the interaction support graph used for training (`graph.json`, `ce_refined.meta.json`,
  `ce_matrix.npy`, `ce_refined.npy`).
- `code/` — `src/aris_bellman/` (method: `td.py`, `factor_belief.py`, `factor_q.py`,
  `specs.py`, `replay.py`, `metrics.py`) and `experiments/overcooked_v2/*.py` (env adapter,
  `event_extractor.py`, `ce_sampler.py`, `train_aris.py`, `evaluate_aris.py`,
  `graph_builder.py`, `options.py`, `layout_parser.py`, `evidence_router.py`, ...), plus
  `configs/ocv2_step4.yaml`, the `scripts/`, and `tests/`.
- `verification/` — `FIDELITY_GATE.{md,json}` (the team's own I1–I9 gate), the gate script,
  `FIX_PLAN_9ISSUES.md`, and git refs.
- `status/` — `PROJECT_DASHBOARD.md`, `OPERATING_CONSTRAINTS.md`.

Authoritative contract = the **proposal**: `FINAL_PROPOSAL.md` (method + theory + claims) and
`OvercookedV2_plan.md` (migration spec + "what not to do"). If either is missing, request it
before judging fidelity.

## 2. The proposal in brief — the contract the code must honor

**Thesis.** ZSC = Bellman control over a small set of **value-sufficient interaction factors**
(latent local coordination modes), NOT partner identity. The agent acts by ordinary Bellman
control `ω_t = argmax_ω Q(s_t, b_t, ω)` with a **factor-local Q decomposition**
`Q(s,b,ω) = Q_base(s,ω) + Σ_{f∈Rel(ω)} A_f(s,ω,b_f)`, where each option reads only its
relevant factors. **There is no separate probe selector.** A task-valid option becomes
"diagnostic" only when its Bellman value is high because it refines value-relevant beliefs.

**Five claims (must be supported, all Type-B):**
1. ZSC can be formulated as Bellman control over value-sufficient interaction-factor beliefs.
2. Local factor beliefs suffice for many coordination decisions; full partner identity is unnecessary.
3. Diagnostic behavior **emerges from Bellman control** (not a separate selector).
4. Factor-local evidence routing + factor-local Q relevance make graph structure **load-bearing**.
5. **Bellman diagnostic value (Δ_info) predicts reward improvement better than raw MI** (Corollary 2).

**Six non-claims** (do not penalize the work for not doing these, but flag if the code/results
secretly depend on them): not all conventions recoverable; not all layouts have cheap probes;
factor modes need not match human labels; Δ_info/MI are NOT learned selectors; LLM text not
required; no universal guarantees for arbitrary adaptive partners.

**Implementation-alignment requirements (proposal §12):** single TD loss `L_ARIS` (no
response/calibration/sparsity/probe losses); `argmax_ω Q` selection; G-TVOI/MI **post-hoc only**;
factor deletion removes **latent state + evidence route + action relevance** (all three);
diagnostic cost measured vs `Q_base`/uniform-belief, not per-step option cost; completion metrics
terminate on task completion (or report post-completion separately).

**Method invariants I1–I9** (the bundle's `FIDELITY_GATE` claims all GREEN — re-derive
independently, do not trust the gate): I1 no MI/probe selector in deploy path; I2 single TD loss;
I3 pure Bellman argmax; I4 CE is preprocessing (never inside the training loop); I5 reward-scale
single-source across preflight/CE/eval/TD target; I6 articulation-point bottlenecks (not degree≤2);
I7 preflight hard gate; I8 no factor-accuracy as a primary metric; I9 factor deletion removes all
three things.

**The 12 "what not to do" (plan §21):** (1) no toy `env_factor_id` in V2; (2) no factor accuracy
as a main metric; (3) no G-TVOI/MI selectors; (4) no CE estimation inside training; (5) no
full-JAX migration first; (6) don't train main model on complete option graph by default; (7) don't
use only random partners without scripted protocol sanity; (8) don't use full-episode return for
local CE; (9) don't assume partner macro-option is observed for neural partners; (10) no degree≤2
bottleneck detection; (11) no true-factor oracle in main V2 results; (12) no layout selection
without preflight CE/reference-gap checks.

## 3. Review axes — be exhaustive; cite `path:line`

**A. Method–proposal fidelity.** Does `code/` actually implement the proposal's method?
- Is selection truly `argmax_ω Q` (`train_aris.py` `_select_option`) with no info-gain/selector branch?
- Is the Q-function the factor-local decomposition `Q_base + Σ_{Rel(ω)} A_f` (`factor_q.py`)? Does
  each option read **only** its relevant factors (verify `Rel(ω)`/relevance masking is enforced, not
  cosmetic)?
- Is the belief factor-local with **routed** per-factor evidence (`factor_belief.py`,
  `evidence_router.py`), with no global-history shortcut?
- Is training a **single** TD loss (`td.py`, `train_aris.py`) with no auxiliary
  response/calibration/sparsity/probe terms?
- Is CE strictly **preprocessing** (`ce_sampler.py`), never estimated in the training loop?
- Does factor deletion remove **all three** (latent + route + relevance) (`graph_builder.py`)?

**B. Invariants I1–I9.** Independently confirm or refute each. The gate is a static grep-level
check; look for ways it could pass while the invariant is *semantically* violated (e.g., I5 is
"structural single-source" — verify the reward scale is actually numerically consistent across
preflight, CE local returns, eval, and the TD target, given `ce/ce_refined.meta.json`,
`configs/ocv2_step4.yaml`, and the code).

**C. Anti-patterns.** Check all 12 §21 items against the code/results. Flag any violation.

**D. Results ↔ claims (the core of the review).** For **each of the 5 claims**, give a verdict:
`SUPPORTED / PARTIALLY / UNSUPPORTED / CONTRADICTED / NOT-TESTED-HERE`, with the specific
evidence. In particular:
- The training metrics show **TD loss does not decrease in any of the 24 runs** (`td_loss_decreased=N`),
  several diverging (~70–625). Assess: is this benign (reward-scale growth) or genuine Q-divergence?
  What does it imply for the validity of `argmax Q` selection and for claims 1/3/4?
- **CRITICAL — the 5000-update greedy eval reverses the 1000-update result.** Prior 1000-update greedy
  eval: aris_bellman/full_support = **4.381** (best, all gates PASS). New 5000-update greedy
  full-diagnostic eval (`results/.../step4_matrix_5kupd_fulldiag_s0.json`): aris_bellman/full_support =
  **−0.400** (collapsed, std ±0.000), BELOW base_only 5.36, flat_factor 5.37, global_gru 4.92, and
  random 4.37 → **G1 and G3 FAIL**. aris_bellman/minus_high_ce = −2.0, overcomplete = −0.42, but
  shuffled_relevance = **5.34** (the only non-collapsed aris cell — backwards from the proposal, which
  predicts shuffling relevance should DEGRADE). partner_id_q (the other high-TD-divergence method) also
  collapsed to −0.4. Assess rigorously: is this **Q-divergence / overtraining** (consistent with the
  TD-loss blow-up to ~70–625 in `TRAINING_ANALYSIS.md`)? A ±0.000 std at −0.4 suggests a degenerate
  fixed policy (e.g., noop, only the cost floor). What does a 4.381→−0.400 reversal between 1000 and
  5000 updates imply for **claim 1** (is `argmax Q` control even valid when Q diverges) and for the
  G2/G4 "PASS" (passing only because full_support itself collapsed — is that a real causal signal or an
  artifact of comparing two broken policies)?
- `served_soup_count = 0` for **every** method in training (incl. random with 21 deliveries). Does the
  task actually close? Could the reward/return or the `serve_soup` termination be mis-measured in a way
  that confounds all comparisons (`event_extractor.py`, `option_termination.py`, `options.py`)?
- Claim 5 (Δ_info > MI as a reward predictor): the **full diagnostics** (G-TVOI/MI/belief-swap/
  factor-deletion) ARE now available in the eval JSONs (`results/eval_*.json` from the full-diagnostic
  run). Use them — but note the underlying aris policy may be degenerate (see above), so interpret
  Δ_info/MI correlations with that caveat. State what a clean (non-collapsed) run would need.

**E. Experimental rigor & the smoke/formal boundary.** This is a **1-seed, 5000-update micro-train on
the `cramped_room` smoke layout**, not a formal Exp 1–5 run (`PROJECT_DASHBOARD.md` §2–3 mark formal
experiments NOT STARTED; the plan requires diagnostic-layout preflight + ≥5 seeds × ≥5 layouts for main
results). Judge: which conclusions are legitimately drawable now vs which are over-reach? Is the gate
set (G1–G4 in the orchestrator) a *debug* gate or a *formal* causal-ablation gate (the proposal's third
key figure wants full_support/overcomplete/minus_critical/random_same_size/shuffled_routes/complete)?

**F. Metric & objective soundness.** Verify reward-scale consistency end-to-end
(`ce_refined.meta.json` cost_coef=0.02/shaped=1.0 vs `configs/ocv2_step4.yaml` training.* vs the TD
target in `td.py`/`train_aris.py`). Check the consistency gate that rejects mismatched graphs
(`train_aris.py` `_enforce_graph_objective_metadata`) — is it sound, or could it pass a stale graph?
Are gate comparisons (mean-only, overlapping std at 1 seed) statistically meaningful?

**G. Regressive / over-defensive code smells.** This project **explicitly forbids** (i) regressive
changes that weaken a real signal and (ii) over-defensive changes that mask failures with
silent fallbacks/try-except/stubs. Scan `code/` (esp. the recently changed `event_extractor.py`,
`ce_sampler.py`, `train_aris.py`, `evaluate_aris.py`, `graph_builder.py`) and `FIX_PLAN_9ISSUES.md`.
Flag any silent fallback, swallowed exception, default that hides a missing input, or diagnostic that
returns zero/skips instead of failing loudly.

**H. Red-flag triage & falsification.** List the single most likely way each headline conclusion is
**wrong**. What one experiment/inspection would most cheaply confirm or kill each red flag?

## 4. Output format

1. **Verdict header**: overall fidelity (FAITHFUL / DRIFT / BROKEN) + overall claim-support
   (CONVINCING / PRELIMINARY / UNSUPPORTED) in one line each, with one-sentence justification.
2. **Per-claim table**: claim # → verdict → key evidence (`path:line` or metric) → what's missing.
3. **Findings** grouped by axis A–H. Each finding: `[CRITICAL|MAJOR|MINOR]` + evidence (`path:line`)
   + why it matters + concrete fix. CRITICAL = invalidates the science or an invariant; MAJOR =
   threatens a claim or rigor; MINOR = hygiene.
4. **Invariant ledger**: I1–I9 each PASS/CONCERN/FAIL with your independent evidence (not the gate's).
5. **Anti-pattern ledger**: §21 1–12 each OK/VIOLATED.
6. **Top 5 actions** before this can count as formal-experiment evidence, ranked by impact.
7. **What you could not verify** from the bundle, and exactly what you'd need.

Be specific, terse, and evidence-bound. Prefer "I checked X at `file:line`, it does/doesn't do Y"
over generic advice. Distinguish clearly between *implementation bugs*, *scientific-validity gaps*,
and *scope/over-reach*. Do not accept the bundle's own README/TRAINING_ANALYSIS or FIDELITY_GATE as
proof — re-derive.
