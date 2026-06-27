# P0 & P1 Revision Plan (provenance-honest)

## Provenance verdict
- **P1 (measurement / behavioral / provenance honesty): YES — derived from the two reviews.**
  Every item below was raised with file:line by round-1 and/or round-2.
- **P0 (the divergence *mechanism*): NO — that was my own code-reading hypothesis, not from the
  reviews, and on re-analysis it was incomplete.** The reviews gave only *symptoms* (TD→1e15,
  `td_loss_decreased=false`, noop collapse, shuffled_relevance survives) and a *generic* "stabilize
  Bellman / checkpoint sweep / Huber/Double-Q". Therefore P0 is re-analyzed below and framed as
  **diagnose-then-fix**, not a prescribed knob.

---

## P0 — Training divergence / greedy collapse  (METHOD/NUMERICAL; diagnose-first)

### Re-analysis (grounded in code, corrected)
- Belief is bounded: `factor_belief.py:169` returns `masked_softmax` (per-factor distribution),
  so `centered_belief ∈ [−1,1]` (`factor_q.py:206-207`). → NOT a belief blow-up.
- Gradients are clipped (`grad_clip_norm: 5.0`, applied in `train_aris._td_update`). → NOT a
  simple gradient explosion.
- Target is detached, uses target net + `max_next`, **no Huber, no target/reward clipping**
  (`td.py:35-56`). The Q head is `Q = q_base + Σ_f A_f` with `A_f = (residual_weight·centered_belief)`
  and `residual_weight` **unbounded** (`factor_q.py:241-254`).
- Most consistent cause: **classic TD / deadly-triad divergence** (bootstrapped max-target + MSE +
  off-policy replay) that grad-clipping slows but does not stop; the **unbounded summed advantage**
  gives the net capacity to express huge Q and makes severity **graph-structure-dependent**
  (minus_high_ce→1.66e15, overcomplete→3.9e5, full→69, shuffled_relevance→0.76). The 1000→5000
  reversal (4.381→−0.40) + ±0.000-std noop collapse are the downstream symptom.

### P0 step 1 — CONFIRM the cause (instrumentation; cheap; do before any fix)
Add lightweight per-update logging to `train_aris._td_update` (behind a `--debug_stability` flag):
`q_pred` mean/abs-max, `target` mean/abs-max, `max_next` abs-max, `adv.sum` abs-max and per-factor
`A_f` abs-max (from `factor_q`), grad-norm pre/post clip, belief entropy. Run a short single-job
trace (aris_bellman/full_support and /minus_high_ce, ~1000–2000 updates) and plot the curves +
a **checkpoint sweep** (save+greedy-eval every ~500 updates: greedy return, option histogram,
forced-noop rate). Expected if the hypothesis holds: `adv.sum` and `target` grow monotonically,
greedy return peaks early then collapses to noop.

### P0 step 2 — FIX (only after step 1 confirms; each change must preserve proposal fidelity)
Candidate stabilizers, smallest-blast-radius first (cross-model reviewed for I1–I9 + §12 fidelity):
1. **Huber (smooth-L1) TD loss** instead of MSE in `td.py` — still a *single* TD loss (I2 intact).
2. **Target / advantage magnitude control**: normalize `Σ_f A_f` by the number of *relevant*
   factors per option (keeps factor-local decomposition; removes the "more factors → bigger Q"
   amplifier) and/or bound per-factor `A_f` (soft, e.g. tanh-scaled) — must NOT collapse the
   load-bearing relevance signal; verify against the graph-ablation ordering.
3. **Double-Q** (decouple action selection / evaluation in the target) — standard deadly-triad fix.
4. **Checkpoint selection by greedy validation** (early-stop / keep-best) — never report the
   last checkpoint blindly.
5. Only if 1–4 insufficient: lr / target_update_interval sweep (tuning is the *last* resort).
Acceptance: TD loss stable (non-increasing trend), greedy full_support no longer noop-collapses,
and the graph-ablation ordering becomes interpretable (full ≥ shuffled_relevance, not reversed).

### Fidelity guard
No change may add an auxiliary loss (I2), a selector (I1/I3), move CE into training (I4), or weaken
the factor-local relevance routing (claim 4). Huber + advantage-normalization + Double-Q preserve
"single TD loss, argmax-Q, factor-local Q". Any advantage bounding must be checked to not flatten
relevance.

---

## P1 — Measurement / behavioral / provenance honesty  (review-grounded; Codex-implementable)

All items are "make it fail loud / measure correctly", i.e. REMOVING over-defensive masking — the
opposite of regressive/over-defensive. File:line from the reviews.

1. **Diagnostics must not silently zero.** `evaluate_aris._DIAG_SKIP` (shape-mismatch → 0.0) and
   `_mean_or_zero` / weighted-summary `no_values` (empty → 0.0): return explicit status / NaN, and
   in formal eval **raise** unless `--allow_diag_skip`. (`evaluate_aris.py:478-498,966-989`)
2. **Factor-deletion = rollout Δreturn for formal eval.** Default `--factor_deletion_episodes > 0`;
   report `rollout_factor_mask` and the `q_proxy_factor_mask` separately, never conflate.
   (`evaluate_aris.py:563-619,1053-1068`; eval JSON `:7939-7948`)
3. **Stop labeling within-run relative return as reference/oracle gap.** Rename the field
   (`within_run_relative_return`), require an explicit oracle/reference file for any oracle-gap
   metric; do not emit `raw_value:1.0` for a variant vs itself. (`evaluate_aris.py:82-86,748-777`)
4. **Random baseline = direct equal eval.** Run `random_policy` × partners × episodes into the same
   matrix (same protocol), not n=1 extracted from `reference_baselines`.
   (`run_step4_microtrain.py:295-312`)
5. **Matrix provenance.** Matrix phase must not record a `full_diagnostics` it didn't run — read it
   from `eval_results.json` (fixes the true/false conflict); embed the exact eval command/args; fail
   on manifest mismatch. (`results/eval/eval_results.json:1-6` vs `step4_matrix...json:171-182`)
6. **Content-hash provenance (strengthen T1/I5).** Stamp hashes of CE matrix, replay, graph,
   option-library, layout-parse, partner-pool, reward-config, git-commit into graph metadata + eval
   output; `train_aris._enforce_graph_objective_metadata` checks hashes, not only scalars.
   (`train_aris.py:559-651`)
7. **Forced-noop / no-valid visibility.** Count forced-noop & no-valid-option events
   (`train_aris.py:972-977,1338-1354`; `evaluate_aris.py:531-560`); formal eval fails if nonzero
   outside terminal states (catches the collapse instead of scoring it).
8. **Partner-option evidence.** `_option_dist_value` missing→0 (`evidence_router.py:422-428`):
   report observed/inferred/missing rate; hard-gate formal neural-partner runs.
9. **Shaped-reward.** `_shaped_reward_for_agent` sum-all fallback (`train_aris.py:1152-1158`):
   require `agent_0` key in formal runs (or an explicit shared-shaping flag).
10. **Checkpoint-load.** `_checkpoint_loads` blanket `except Exception` (`train_aris.py:1276-1281`):
    catch specific types, log type+message.
11. **Completion accounting (behavioral correctness).** Unify `delivery_event` / sparse-reward /
    `completed` / `served_soup` / `serve_soup` termination into one canonical completion definition;
    split ego/partner/correct/wrong delivery; add trace unit tests for `serve_soup`.
    (`option_termination.py:90-96`; `evaluate_aris.py:420-428,352-355`;
    `metrics/random_policy__full_support__seed0.json` delivery=21 but served_soup=0)

### Out of P0&P1 scope (tracked, not now)
- Preflight threshold (plan ≥8 valid options vs accepted 7) → governance (P-gov).
- True `minus_critical` + full ablation suite (random_same_size/shuffled_routes/complete) +
  multi-seed × diagnostic-layout → formal scope (P3). Both blocked on P0 anyway (no point running
  ablations on a divergent learner).

## Sequencing
P0-step1 (instrument+confirm) ∥ P1 (Codex implements honesty fixes) → P0-step2 (stabilize, reviewed)
→ short 1-seed re-run on the fixed learner → only then P3 formal scope.
