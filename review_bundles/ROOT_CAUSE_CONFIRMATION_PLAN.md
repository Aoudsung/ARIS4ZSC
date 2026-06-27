# Root-Cause Confirmation Plan

Scope: confirm (or refute) the root cause behind each problem raised in the manual
review of `ocv2_diag_s0_20260627_204650.zip`. Every test below is **falsifiable**:
it predicts a *different observable signature* per candidate cause and ends in a
decision rule. The goal is to know **what to actually fix**, not to add more band-aids.

All findings cited below were re-verified against the bundle's own
`raw/results/ocv2_stability/u*/.../metrics.json`.

---

## 0. Problems (from review, verified)

| # | Problem (review wording) | Verified evidence |
|---|---|---|
| P1 | Bellman/Q learning still collapses; Huber+Double-Q insufficient; checkpoint-selection only *masks* it | u1000 greedy_validation 500→5.39, **1000→−0.4** with td_last=**0.972**; u5000 …2000→5.33, **3500–5000 all −0.4**, td_last=**5.6e7** |
| P2 | return is shaped/local progress, not task completion; ego never completes | u5000 completion=0; served_soup=4 but **ego_delivery=0, partner_delivery=4** |
| P3 | graph not proven load-bearing | formal variant suite not run on current code; stale step4 had shuffled_relevance > full_support |
| P4 | Claim 5 (Δ_info > MI) no clean evidence | stability eval is `--fast`; Δ_info/MI = `no_values` |
| P5 | provenance improved (stability) but package not self-consistent (step4 stale, pre-P1) | step4 `full_diagnostics` conflict; resolved_config lacks td_loss/double_q keys |

## 1. Dependency structure (why ordering matters)

```
        RC-1 (collapse mechanism) ──┬──► RC-3 (graph load-bearing)   [meaningless on a collapsed policy]
                                    └──► RC-4 (Δ_info vs MI)          [needs a non-collapsed, diagnostic-taking policy]
RC-2 (completion reachability)  ── independent, cheap, run in parallel
RC-0 (provenance regen)         ── cross-cutting precondition for citing ANY result
```

**RC-3 and RC-4 are gated on RC-1.** Ablating the graph or measuring diagnostic value on a
policy that deterministically noops produces noise, not evidence. Confirm/repair P1 first.

---

## RC-1 — Confirm the collapse mechanism  *(decisive; do first)*

**Target:** P1. The review's sharpest finding is that collapse appears at **u1000 with TD
loss = 0.97 (small)** — i.e. it is **not only** the deadly-triad loss-scale explosion
(that 5.6e7 only shows up at 3000+). There is an earlier, independent **argmax-Q ranking
collapse**: the greedy policy comes to rank `noop` highest in every state → all-noop →
−0.4 cost-floor.

**The single decisive measurement:** track, per checkpoint, the value gap
`Δ = Q(s,b,noop) − max_{ω≠noop} Q(s,b,ω)` on a fixed canonical-state set.
The crux question: **does Δ cross 0 (noop starts winning) BEFORE the TD loss explodes?**
- If yes → ranking collapse is the *primary* root cause; the loss-scale blow-up is
  downstream/secondary. Huber/Double-Q (loss-scale fixes) cannot help by construction.
- If Δ crosses 0 only *after / together with* the TD explosion → loss-scale divergence is
  primary and a stronger TD-stabilizer is the right lever.

### Candidate causes and their distinct signatures
| Hyp | Mechanism | Confirming signature (in the dump) |
|---|---|---|
| H1a-i | `q_base` head collapses for options | per-option `q_base` drifts ≤ `q_base(noop)` while `Σ_f A_f` stays ~flat |
| H1a-ii | summed factor-advantage `Σ_f A_f` drifts negative for options (unbounded residual) | `q_base` ~flat but `Σ_f A_f(option)` ≪ 0; one/few factors dominate the sum |
| H1b | target over-pessimism / bootstrap pull-down | TD **target** for option-transitions systematically < noop-transitions; gap widens with updates |
| H1c | **reward/cost mis-spec — noop is genuinely optimal** | Monte-Carlo option net-return (shaped − cost) ≤ noop return at canonical states *(then collapse is correct behavior for a bad objective, not a learning bug)* |
| H1d | replay self-reinforcement after first noop | replay option-kind histogram → noop-dominated right after Δ first crosses 0 |
| H1e | belief degenerates → A_f uninformative | mean belief entropy → 0 (or A_f variance → 0) coincident with collapse |

### Instrumentation (one run; no retrain of the whole matrix)
One instrumented seed-0 / full_support / 5000-update run that **saves every 500-update
checkpoint** and at each logs, on a **fixed canonical-state set**:
1. per option ω: `Q_total`, `q_base`, `Σ_f A_f`, **per-factor `A_f` breakdown**, argmax, noop-rank, and Δ above.
   - hook: call `FactorQ.forward` (`src/aris_bellman/factor_q.py`) returning the base/advantage decomposition on the canonical batch.
2. TD **target** values for option- vs noop-transitions in the update batch (`src/aris_bellman/td.py` target branch).
3. Monte-Carlo option net-return (shaped − `cost_coef`·cost) by rolling each option from each canonical state (tests H1c — *the cheapest way to rule reward-design in or out*).
4. mean belief entropy + per-factor `A_f` variance (H1e).
5. replay option-kind histogram (H1d).

**Canonical-state set:** initial reset state + ~5 states sampled from the *best*
(update-2000) checkpoint rollout, chosen to span the decision points where plate/serve
*should* beat noop: empty pot, partially-filled pot, full/cooking pot, soup-ready,
plate-in-hand. Serialize once, reuse across checkpoints (same states → comparable Δ).

### Decision rule → fix
- Δ crosses 0 pre-explosion **and** H1a-ii signature → **advantage-sum is unbounded** → enable/clip `advantage_norm` (already a flag, default off) and/or bound the residual; re-confirm Δ stays < 0.
- H1a-i → base-head/target problem → revisit target construction (RC-1.2).
- H1c holds (MC noop ≥ options) → **objective is mis-specified** (cost too high vs shaped/sparse) → fix `cost_coef`/`shaped_reward_coef`/sparse-delivery scale; this is *not* a learning bug and no stabilizer will fix it.
- H1b → target over-pessimism → check n-step option discounting + cost-in-target + target-update interval.
- H1d → exploration/replay → keep ε higher / prioritized or on-policy correction.
- H1e → belief uninformative → inspect `factor_belief` update.

**Also produces (review step 1):** the explicit `checkpoint_final` vs `checkpoint_best`
eval table (best update, final update, best/final greedy return, td_last, noop_frac,
completion) — final is already known to be −0.4 from `greedy_validation`; this makes the
masking explicit and quantified.

**Boundary:** instrumentation script = in-boundary (static, I write it). Running the
instrumented 5000-update training on cxw2 = remote execution → **needs authorization**.

---

## RC-2 — Confirm completion reachability  *(cheap; parallel)*

**Target:** P2. Distinguish **(a) counter/termination bug** from **(b) learning/value
failure** for `completion=0` and `ego_delivery=0`.

**Test:** drive a **scripted optimal protocol** (deliberate fetch→cook→plate→serve by the
ego agent, ignoring the learned Q) and observe whether `serve_soup` option success,
`ego_delivery_event_count`, and `completion_rate` fire.
- Also log, over a *non-collapsed* checkpoint rollout, whether plate_soup/pick_plate/
  serve_soup are ever in the **valid-option mask** (option preconditions) vs ever
  **argmax-selected**.

### Decision rule
| Observation | Root cause |
|---|---|
| scripted run completes + counters fire | counters OK → completion=0 is a **value/learning** failure (same root as RC-1: serve options never out-rank noop). No event/termination fix needed. |
| scripted run does *not* complete, or counters stay 0 under optimal play | **termination/event/attribution bug** → fix `option_termination.py` (serve_soup success), `event_extractor.py` (ego vs partner delivery attribution) |
| serve options never in valid mask | **option-generation/precondition bug** → fix option library / preconditions |

**Boundary:** scripted-probe script = in-boundary; running it on cxw2 = remote → needs authorization. (No training; seconds to run.)

---

## RC-3 — Confirm graph load-bearing  *(gated on RC-1)*

**Target:** P3 / Claim 4. Only meaningful once RC-1 yields a **non-collapsed training
protocol** (e.g. the stable-window checkpoint, or RC-1's fix applied). Then run the
**formal** variant suite with current code:
`full_support, minus_critical, random_same_size, shuffled_routes, shuffled_relevance,
complete_option_graph` (mark `minus_high_ce` **debug-only**, not formal).

**Confirming comparison:** on return **and** completion **and** diagnostics, with the
*same* non-collapsed checkpoint-selection protocol and **held-out eval partners/layouts**:
- full_support ≥ {minus_critical, shuffled_relevance, random_same_size} on all three → **graph load-bearing (Claim 4 supported)**.
- shuffled_relevance ≥ full_support (the stale step4 signature) persisting on a
  non-collapsed policy → **graph NOT load-bearing (Claim 4 refuted)** — a real negative
  result, reported as such.

**Boundary:** remote, multi-run → needs authorization; do **after** RC-1.

---

## RC-4 — Confirm Δ_info vs MI  *(gated on RC-1)*

**Target:** P4 / Claim 5. Needs a non-collapsed policy that *takes diagnostic actions*.
Run **full** diagnostics (not `--fast`) on the non-collapsed checkpoint; emit the
**transition-level table**: `t, option, Δ_info, MI, next-k return, completion Δ,
oracle-gap before/after`.

**Confirming test:** does `Δ_info` predict next-k reward improvement **better than** `MI`
(rank correlation / partial regression, raw transitions not just aggregate means)?
- Δ_info significantly more predictive → Claim 5 supported.
- not better / MI ≥ Δ_info → Claim 5 unsupported. Either way reported honestly.

**Boundary:** remote, after RC-1.

---

## RC-0 — Provenance regeneration  *(cross-cutting precondition)*

**Target:** P5. Quarantine the stale pre-P1 `ocv2_step4`; regenerate **all** cited
artifacts in one current-code pipeline run writing manifests + content hashes
(`provenance.py`), `full_diagnostics` read from the eval manifest. No result is citable
until its provenance is consistent with the current commit.

**Boundary:** remote, bundled with RC-3/RC-4 runs.

---

## Execution gating (what I need from you)

In-boundary now (no authorization needed): I write the **RC-1 instrumentation hook** and
the **RC-2 scripted-probe script**, plus a canonical-state serializer.

Needs your authorization (remote execution on cxw2):
1. **RC-1 instrumented 5000-update run** ← decisive; ~5 min + instrumentation overhead.
2. **RC-2 scripted-probe** ← cheap; seconds.
3. RC-3 / RC-4 / RC-0 ← only after RC-1 identifies the fix and a non-collapsed protocol exists.

Recommended authorization: **RC-1 + RC-2 first** (confirm the two root causes), then decide
RC-3/4/0 based on RC-1's outcome.
