# REVIEW_BRIEF — ARIS-Bellman 根因裁定 · 异模型族对抗性评审

**To:** codex (GPT-5.5, xhigh) · **From:** Claude (static audit synthesis) · **Date:** 2026-07-02
**Repo:** `/Users/aoudsung/Documents/ARIS4ZSC` · **Branch:** `rc-rootcause-fix` (working tree has UNCOMMITTED changes — audit was on the working tree, not HEAD)
**Plan of record:** [ROOTCAUSE_REVIEW_PLAN.md](../../ROOTCAUSE_REVIEW_PLAN.md) · **Output → ** `CODEX_OUTPUT.md` in this folder

---

## 0. YOUR JOB (read this first)

Five parallel **static** audits (all authored by Claude subagents) concluded:

> **Adjudication under review:** The core scientific claim of ARIS-Bellman — that ZSC is Bellman
> control over *value-sufficient interaction-factor beliefs* — **has never been tested**. Every
> headline negative result to date (baselines matching ARIS; held-out failure; "standard data
> gives no discriminative path") is an **artifact** of four causes: **P1** an oracle evidence leak,
> **P2** a degenerate partner substrate, **P3** a CE sentinel-zero, **P4** a memory-less belief.
> The method is therefore **neither validated nor refuted**.

You are a **different model family**, deployed precisely to break the same-family blind spots of
that synthesis. **Do not rubber-stamp it.** Your task is to **independently verify each finding from
`file:line`** and to **actively try to refute the top-line verdict**. Where you cannot reproduce a
claim by reading the cited code, do not confirm it.

### Anti-anchoring protocol (please follow the order)
1. Read **§1 (adjudication questions)** and **§2 (evidence anchor table)** first.
2. For each anchor, **open the cited file yourself** (line numbers may have drifted — every anchor
   also carries a grep-able string; trust the string over the number). Form **your own** verdict.
3. **Only after** writing your per-finding verdicts and your counter-hypothesis analyses, read
   **Appendix A** (our classification/severity/ranking). Then note where you disagree.
4. Default to skepticism. A finding is `CONFIRMED` only if you reproduced the behavior from code.

Two configs are in scope: `configs/ocv2_step4_asymm.yaml` (the G2/parity history) and
`configs/ocv2_step4_asymm_role_v1.yaml` (the **current working-tree active arm**:
`partner_set: role_conditioned_v1`, `sparse_credit: role_contrib_team`). Findings note which.

---

## 1. Adjudication questions (argue BOTH sides before judging)

For each, first construct the **strongest case for the counter-hypothesis (X′)**, then judge, then
give a confidence. These five drive the verdict; treat them as the point of the review.

### Q-A — "Core claim is UNTESTED (not refuted)."
- **Counter A′:** some existing result subset **already** suffices to *refute or support* the method.
  If a legitimately-run comparison exists where ARIS's factored belief cannot have helped yet it
  matched/beat baselines (or vice-versa), the "untested" verdict is wrong.
- Consider: the asymm G2 parity (all arms → completion 1.0) and the A1/A2/A7 dissociation
  (`analysis/03_ablations_A1_A7.txt`). Does the dissociation *alone* establish anything ARIS-specific
  independent of P1–P4?

### Q-B — "Baseline parity is a MECHANICAL consequence of the oracle leak (P1)."
- **Counter B′ (the key test):** if every arm receives the same oracle intent channel, ARIS's
  factor-local relevance routing should still add a measurable edge — and **it apparently did once**:
  the (now-invalidated-for-other-reasons) contested-layout result at [METHOD_LOCK.md:231-240](../../METHOD_LOCK.md)
  shows `aris_bellman 10.0` throughput vs `flat_factor 5.0` (**2×**) and `base_only 0.0`, all
  consuming the same routed evidence. **Reconcile this with B.** Either (i) the routing genuinely
  helps even with the oracle present (⇒ P1 does not fully explain parity, and the asymm parity has a
  *different* cause — e.g. task collapses to single-agent competence), or (ii) that 2× is an artifact
  of the synthetic `cr-*` partners, not of routing. Which, and what code/evidence decides it?
- This is the single most important question for the strength of the P1 pillar.

### Q-C — "serve_soup CE=0.000 on asymm is a min_weight SENTINEL, so METHOD_LOCK §11's 'asymm has no terminal interaction externality' is unsafe."
- **Counter C′:** 17 reachable serve-attempts under random-option play is *itself* evidence about the
  layout; more samples would still give CE≈0 because on asymm the terminal role is geometric, not
  negotiated. Is the sentinel argument (support 17 < min_weight 20 ⇒ never estimated) **airtight**,
  or does the estimand (D4: random-valid-option occupancy) conflate "few samples" with "no
  externality" in a way that cuts *both* ways?

### Q-D — "The belief cannot accumulate across decisions (4-step window, zero-init each selection), so the documented 'accumulating failure trace' mechanism is structurally absent and is the main cause of held-out perseveration."
- **Counter D′:** window size is a config knob; in-distribution success proves 4 steps suffice; the
  held-out failure (sec13: 20 serve attempts / 0 success) is explained by **training-distribution
  narrowness**, not by the window. Distinguish these two explanations from code + the recorded traces.

### Q-E — "Our classification / severity / fix-ordering (Appendix A) is correct."
- **Counter E′:** independently re-rank. Flag anything we **over-stated, mis-classified, or missed**.

### Q-F — *(advisory, near-Type-B, final call is human)* After P1 de-oracling + P4 belief persistence, is **TD-only + purely behavioral evidence** in principle sufficient to learn factor inference? Give the argument and a discriminating test design. Do **not** bless the method.

---

## 2. Evidence anchor table (open each; verify from code)

Schema: `ID | claim | anchor(file:line + grep string) | code fact | how to falsify`.
All line numbers verified against the working tree on 2026-07-02; if drifted, grep the string.

### Pillars

**P1 — Ground-truth partner intent injected into the evidence stream (train/eval/CE).** IMPL, high.
- `experiments/overcooked_v2/partner_pool.py:210-213` — grep `option_confidence=1.0` /
  `option_dist=_one_hot`. `ScriptedProtocolPartner.act()` returns its own currently-executing option
  as a one-hot with confidence 1.0.
- `experiments/overcooked_v2/option_executor.py:57-63` — grep `partner_action.option_id`; and the
  layer's own contract at `:13` grep `must never read`. The shared train/eval/CE substrate feeds that
  option id / dist straight into `extract_event`.
- `experiments/overcooked_v2/evidence_router.py` — grep `partner_terminal_option` (`:77`, written at
  `:249` `routed[...] = 1.0`), `partner_option_is_i` (`:32`), `partner_option_confidence` (`:36`),
  `partner_option_known` (`:58`). ≥11 channels are direct reads of the partner's private intent;
  channel `partner_terminal_option` is a readout of the exact latent mode the belief must infer.
- Dead alternative: `experiments/overcooked_v2/option_inferencer.py` (`PartnerOptionInferencer`) is
  instantiated **only** in `experiments/overcooked_v2/tests/test_ocv2_p0_p1_fixes.py:1152,1166`
  (grep confirmed) — wired into neither train, eval, nor CE.
- **Falsify:** find any train/eval/CE path where the routed `partner_option_*` channels are zeroed or
  behavior-derived rather than taken from `partner_action`. Or show a config flag that disables them.
- **Why it matters:** all four arms (base_only/global_gru/flat_factor/aris) consume the identical
  routed tensor (`train_aris.py` `_state_repr`, grep `GlobalGRUQNetwork`), so the factored belief
  confers no information advantage ⇒ **parity is the expected outcome** (feeds Q-B).

**P2 — Geometry bonuses swamp role bonuses ⇒ partners behaviorally identical (HEAD scale).** SUB, high.
- `experiments/overcooked_v2/partner_pool.py:233` `_protocol_score`; role bonus at `:424`
  (grep `4000.0 if condition else -1000.0` — **working tree**; was `+4/-1` at HEAD); recovery bonuses
  **not rescaled**: `:277` and `:283` grep `score += 100.0` (clear/drop); terminal tiers `:471-479`
  grep `-30000.0` / `8000.0` / `5000.0`; blocking predicate `:405` `_blocking_critical_cell`.
- **Two versions:** METHOD_LOCK sec15/16's "identical partners" describes **HEAD** (`git show HEAD:…`
  — role `+4/-1`, clear `+100`). The **working tree** rescaled role→`±4000/±1000` + terminal tiers,
  which fixes the swamp for role partners but **left recovery terms at +100 / useless-fetch at -10**.
- **Falsify (HEAD):** find any role/positional/bottleneck term combination that flips the argmax when
  `clear_interaction_cell` scores ~97.6 and the best competitor ~6. **Falsify (working tree):** find a
  state where an in-role option (+4000) loses to `clear` (-1000+100) — i.e. whether the un-rescaled
  recovery can still livelock a role partner (W1).
- Corroboration: `analysis/partner_differentiation_probe.md` (1/6 unique on tight layouts, 4/6 asymm).

**P3 — serve_soup CE=0.000 (asymm) is a below-support sentinel, not a measurement.** EST+IMPL, high.
- `experiments/overcooked_v2/ce_sampler.py:593` matrix init to zeros; `:615` grep
  `if weight_sum < min_weight:` → `continue` leaves the cell at **0.0** (skipped ≡ measured-zero).
  Default `min_weight=20.0` (`:588`), **hardcoded** in the canonical pipeline at
  `scripts/run_ce_pipeline.py:210` grep `min_weight=20.0`.
- Support IS computed but **not persisted**: `weight_sum` exists in-loop (`:615`) yet only aggregate
  `min_weight` reaches the sidecar (`:677`), never per-pair support.
- `refine_empirical_ce` cannot rescue: `_top_pairs` filters `> 0.0` (grep in `ce_sampler.py`), so
  sentinel cells are never bootstrapped; refine mean over positive estimates only ⇒ survivorship bias.
- **Recorded proof to re-read:** `logs/cr_ce_regen.log` (asymm: serve attempts=17 < 20 ⇒ every serve
  cell skipped; then `GraphCoverageError: … required option 8:serve_soup`) vs
  `logs/ce_regen/ce_cramped_probe.log` (cramped: serve attempts=91 ⇒ CE=8.8967). If those log paths
  are absent in this checkout, the numbers are quoted in [METHOD_LOCK.md:210-229](../../METHOD_LOCK.md).
- **Falsify:** show the 0.000 is a genuine estimate (a serve pair with weight_sum ≥ 20 that resolved
  to ~0), not a skip. Or argue occupancy (D4) makes the distinction moot.

**P4 — Belief is a 4-step sliding window re-encoded from zero each selection; no cross-decision state.** IMPL-vs-claim, high.
- `configs/ocv2_step4_asymm_role_v1.yaml:199` (and `ocv2_step4_asymm.yaml:173`) grep `evidence_window: 4`.
- `src/aris_bellman/factor_belief.py:53` `initial_hidden` returns `torch.zeros(...)`; `encode_history`
  (`:103`) defaults `initial_hidden` to those zeros (`:114-116`); `forward` (`:156`) is called per
  option selection with no persistent hidden threaded in (grep `encode_history` in `train_aris.py` /
  `evaluate_aris.py` `_current_belief` / `_state_repr`).
- `src/aris_bellman/replay.py:11` `EvidenceBuffer`, `window` (`:22`), ring-buffer `append` (`:32`),
  fixed-size `snapshot` (`:46`). Stores fixed snapshots ⇒ persistent-hidden training would need a
  different replay design (structural, not just a knob).
- Claimed mechanism it contradicts: `evidence_router.py` grep `accumulates a per-mode failure trace`.
- **Falsify:** find where belief hidden state survives across option decisions within an episode, or
  where the window spans enough decisions for a failure trace to compound.

**P5 — Reward/exploration/replay conditioned on partner's ground-truth mode; eval return same.** claim-layer, high.
- `configs/ocv2_step4_asymm_role_v1.yaml:99` grep `sparse_credit: role_contrib_team` (active);
  `experiments/overcooked_v2/sparse_credit.py` grep `role_contrib_team` — ego delivery worth full vs
  0.3× team reward keyed on partner `terminal_policy`.
- `train_aris.py` grep `terminal_policy` (reward call ~`:1409`, exploration switch, `_seed_role_replay`).
- eval headline `reward_sum` uses the same `_training_reward` (`evaluate_aris.py` grep `_training_reward`).
- **Falsify:** show the deployment argmax reads the oracle (it should NOT — this is INTENDED curriculum,
  the concern is claim-framing, not a deployment leak). Confirm the mode↔role correlation is
  oracle-manufactured in ≥2 of {reward, exploration, replay}.

### Secondary findings (compact — full anchors in the audit reports; verify the starred ones)

| ID | claim | key anchor |
|---|---|---|
| ★S1 | `ego_option_terminated_failed` misses `"max_steps"`; `"option_invalid"` never produced | failure set `train_aris.py:1482` `{budget_exhausted,env_max_steps,option_invalid}` vs `option_termination.py:154` returns `"max_steps"` |
| S2 | belief GRU consumes trailing zero-padding as evidence | `replay.py:38-46` partial fill; `factor_belief.py` `encode_history` no length mask |
| S3 | failure re-routes last event twice (duplicated window step) | `train_aris.py:~1483`; `evaluate_aris.py:~639` |
| S4 | `pot_became_cooked` ≡ `pot_became_ready` (duplicate channel) | `event_extractor.py` grep `_pot_became_ready` calls `_pot_became_cooked` |
| ★S6 | bare `ce_sampler.py collect` CLI does not exclude held-out partners | `ce_sampler.py:~1111` `make_training_partners(...)` full registry; filter only in `run_ce_pipeline.py:118-140` |
| S7 | batched CE `compute_local_returns` collapses to ~horizon-1 (interleaved episodes) | `ce_sampler.py:412-430` append vs `:158-161` break — sequential path unaffected (all artifacts sequential) |
| S8 | replay coverage gate counts partner deliveries, not ego-side estimability | `ce_sampler.py:541-548` |
| S9 | skipped-vs-measured zero not recorded; refine survivorship bias | `ce_sampler.py:669-671`, sidecar `:677` |
| S10 | γ=0.99 / horizon=5 hardcoded, config `local_return_horizon_options` ignored by pipeline | `run_ce_pipeline.py:158-159` vs `layout_diagnostics.py:41` |
| S11 | `--sparse_ce_support` invisible to objective gate metadata | `ce_sampler.py:383-384`; metadata `run_ce_pipeline.py:61-76` |
| ★S16 | global_gru diagnostics shape crash | `evaluate_aris.py:716-724` `RuntimeError("Diagnostic belief/mode-mask shape mismatch")` (gru state `[1,F,T,64]` ≠ `[1,F,M]`) |
| ★S17 | `--allow_diag_skip` short-circuits ALL integrity gates; used for every arm in recorded runs | `evaluate_aris.py` early-return in `_validate_eval_integrity` (grep `allow_diag_skip`, `:359`); runner scripts in bundle passed it to all methods |
| ★S18 | `reward_scale_verified` permanently false (provenance uses full registry vs train subset) | `evaluate_aris.py:120,1549` `make_training_partners(...)` vs train stamps train-subset pool |
| S19 | `diagnose_traces.py` non-functional (3 defects) but §12.4 plans mechanism evidence on it | evidence_dim=6 vs 64; `ctx.evidence_router` missing; reads `info["event"]` never returned |
| S20 | `completion` counts partner/wrong deliveries and saturates on 1 delivery | `evaluate_aris.py:461-462`; `event_extractor.py:150` `delivery_event = ego OR partner OR env_correct` |
| S21 | eval seed inert (`random_reset:false`) + same seed sequence per partner ⇒ byte-identical across indistinct partners | `evaluate_aris.py:323,348,388`; `env_adapter.py:76-82` |
| S22 | asymm cfg: ARIS Q tanh-bounded but baselines unbounded (un-matched, baseline-favoring) | `ocv2_step4_asymm.yaml` value_bound lacks `apply_to_all_methods`; **fixed** in role_v1 `:187` |
| S23 | `train_partners` unset ⇒ silent fallback to ALL partners (train + selection + CE) | `train_aris.py:782-783`; `run_ce_pipeline.py:141-142` |
| S24 | missing graph provenance downgraded to warning (legacy graphs bypass pool check) | `train_aris.py:1129-1135` |

### Working-tree partner-library redesign (role_conditioned_v1 — UNCOMMITTED)

| ID | claim | key anchor |
|---|---|---|
| ★W1 | recovery terms (+100 clear/drop, −10 fetch) NOT rescaled with role→±4000 ⇒ in-role option beats clear ⇒ possible livelock | `partner_pool.py:277,283` (+100) vs `:424` (±4000) |
| ★W2 | `heldout-yield-terminal-claim` ≈ near-duplicate of trained `bottleneck-push-terminal-claim`; it is the v4 analysis's headline claim partner | `partner_pool.py:91-164`; `scripts/parse_role_v1_v4.py:11` `CLAIM_PARTNER=` |
| W3 | yield-tier internal ordering ⇒ yield partners camp bottlenecks instead of prepping | `partner_pool.py:467-483` (+5000 prep/support ties) + `_bottleneck_bonus` |
| W4 | `alternate` bottleneck policy doesn't alternate (`self.elapsed` stale at selection) | `partner_pool.py:285` uses `self.elapsed`; reset at `:197-202` |

### Design-layer (proposal-grounding audit)

| ID | claim | source |
|---|---|---|
| D1 | partner-differentiation precondition never stated as checkable; `partner-induced return variance` in plan §15 metrics but absent from `accept_layout()` | FINAL_PROPOSAL §5.1/§6 Assumption 1; OvercookedV2_plan §15 |
| D2 | `reference_base_gap` gate is pre-training only; a curriculum lifting base_only to ceiling defeats it | OvercookedV2_plan §15 `accept_layout()` |
| D3 | closed mode set K_f ⇒ only recombination generalization; Exp 2 "held-out families" expectation contradicts non-claims 1/6 | FINAL_PROPOSAL §5.9, §11 |
| D4 | CE estimand conflates measurability with occupancy (uniform-valid-option behavior policy) | ce_sampler behavior policy; §11 interpretation |

### Exonerations (verified CORRECT — please try to break these too)

E1 single TD loss / target / double-Q / dynamic next-option mask (`src/aris_bellman/td.py`,
`train_aris.py` capture at end-of-option, sign `masked_fill(~mask,-1e9)`).
E2 relevance routing + factor deletion removes latent+route+relevance (`graph_builder.py`,
`diagnostics.py:197-227`). E3 variants genuinely distinct; parity NOT wiring-faked
(`train_aris.py` `_state_repr`; eval reads `method` from checkpoint). E4 no partner-ID into ARIS
(popped; only `partner_id_q` embeds it). E5 checkpoint selection train-partners only.
E6 split enforced in current configs. E7 sparse-credit attribution leak-proof on coincident deliveries.
E8 synthetic `cr-*` revert clean. E9 A1/A2/A7 dissociation valid *for the ARIS controller*.

---

## 3. Certainty tiers

- **Arithmetic / structural — verify by reading, no execution:** P2 (score magnitudes), P3 (min_weight
  skip logic), P4 (window + zero-init), S1, S16, S18, S20, S22, W1–W2.
- **Needs an execution probe — design it, do NOT run it (execution-gated):** Q-C (does serve CE cross
  min_weight with more episodes / lower min_weight?), Q-B (does de-oracled ARIS separate from
  baselines?), Q-D (does persistent belief change held-out perseveration?), P5 downstream effect on
  learned policy. For each, specify sample size, decision threshold, expected branches.

---

## 4. Scope boundary

This review adjudicates **code-vs-claim fidelity** and the **artifact-safety of past conclusions**
(Type-A–leaning). The go/no-go on the science itself is **Type-B** → your verdict + a human
checkpoint (OPERATING_CONSTRAINTS §4). **Do not declare the method sound or unsound.** Q-F is advisory
only. If you find the "untested" verdict itself is wrong (Q-A′/Q-B′), say so plainly — that is the
most valuable possible output.

---

## 5. Must-answer checklist

1. Per-finding verdict for **every** ID in §2 incl. exonerations E1–E9 (`CONFIRMED` / `REFUTED` /
   `REFINED` + your `file:line` + correction).
2. Counter-hypothesis analyses **Q-A′ … Q-D′** (strongest case → judgment → confidence). Q-B is
   mandatory and decisive.
3. Which items in our synthesis are **over-stated, mis-classified, or wrong** (Q-E′).
4. Your **own** must-fix ranking + diff against our ordering (Appendix A / plan §2.3) + reasons.
5. **New findings** we missed (`NEW-1…`, same schema as §2).
6. Q-F advisory (argument + discriminating test; no verdict on the method).
7. **Probe designs** for every §3 execution-gated question.

---

## 6. Manifest (read these in-repo)

**Code (primary):**
`experiments/overcooked_v2/{partner_pool.py, option_executor.py, event_extractor.py,
evidence_router.py, option_inferencer.py, ce_sampler.py, scripts/run_ce_pipeline.py, train_aris.py,
evaluate_aris.py, layout_diagnostics.py, diagnose_traces.py, option_termination.py, options.py}` ·
`src/aris_bellman/{factor_belief.py, factor_q.py, replay.py, td.py, graph_builder.py, diagnostics.py,
specs.py}`

**Configs:** `experiments/overcooked_v2/configs/{ocv2_step4_asymm.yaml, ocv2_step4_asymm_role_v1.yaml}`

**Docs / evidence:** [METHOD_LOCK.md](../../METHOD_LOCK.md) (§8, §11, §15–16, §231-240) ·
[FIDELITY_GATE.md](../../FIDELITY_GATE.md) · [RC_REWARD_CREDIT_FIX.md](../../RC_REWARD_CREDIT_FIX.md) ·
`idea-stage/refine-logs/FINAL_PROPOSAL.md` (§5.1, §5.9, §6, §11) · `artifacts/OvercookedV2_plan.md`
(§15) · this bundle's `analysis/` (`03_ablations_A1_A7.txt`, `07_matched_baselines_train.txt`,
`partner_differentiation_probe.md`, `invalidation_notice.md`) and `logs/` (CE regen logs for P3).

---

## 7. Return format (write to `CODEX_OUTPUT.md`)

```
## Verdicts
ID | verdict | your evidence (file:line) | correction | your severity | your rank
## Counter-hypothesis analyses
Q-A′ … Q-D′ : strongest case → judgment → confidence
## New findings
NEW-1 | claim | anchor | code fact | how to falsify
## Revised fix ordering
(diff vs plan §2.3 + reasons)
## Probe designs
(per §3 question: probe, sample size, decision threshold, expected branches)
## Q-F advisory
```

---

## Appendix A — our classification (read ONLY after forming your own verdicts)

Full ID index with our classification (IMPL=implementation bug · SUB=substrate design flaw ·
EST=estimator limitation · MTH=method design limitation · GOV=governance) and severity is in
[ROOTCAUSE_REVIEW_PLAN.md](../../ROOTCAUSE_REVIEW_PLAN.md) Appendix A. Our top-line: P1+P2+P3+P4 make
the core claim untested; P5 makes in-distribution "adaptation" ≠ genuine inference; the free-riding
bug (already fixed 2026-06-29) was the only *previously-closed* implementation defect. Our fix order:
**de-oracle (P1) → belief persistence (P4) → CE support sidecar + re-measure (P3) → partner lib
(W1-W3) → curriculum de-oracle (P5) → eval integrity (S16-S19)**. Tell us where we are wrong.
