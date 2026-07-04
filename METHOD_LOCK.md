# METHOD_LOCK — ARIS-Bellman OvercookedV2 ZSC (asymm_advantages)

Status: **method-locking + validation phase** (no further method interventions).
Frozen candidate: `aris_bellman_g2_coverage_constrained_progression_seed_v1`
(immutable at `frozen/…_v1/` on the remote). All comparisons are against it.

Dev-heldout partners (now USED for debugging, no longer pristine):
`bottleneck-yield`, `flexible-balanced`. A fresh **blind** split is created only
after the final method is locked (see §7). Do not tune on the blind split.

---

## 1. Frozen causal story

```
actor-specific reward credit fixed free-riding
  → terminal-yield curriculum created ego-owned terminal-stage data
  → raw top-K CE support failed: not value-sufficient (serve_soup excluded)
  → coverage-constrained CE restored serve_soup belief access
  → ARIS learned ego-owned serving (held-out completion 1.0)
  → removing serve_soup support destroys ego serving
```

**Specific claim (do NOT weaken to "a larger graph helps" — A2 refutes that):**
> CE support selection must be **value-sufficient and task-stage complete**;
> raw top-K by CE magnitude is insufficient.

## 2. The dissociation (the load-bearing result)

| Variant | Graph | Outcome (train-proxy + held-out) |
|---|---|---|
| **G2 / A3 / A4 / A6** | coverage-constrained, serve present | **succeed** — 5/5 (G2) + A3/A4/A6 3/3, held-out **1.0**, ego=20/partner=0 |
| **A1** raw top-16 | serve crowded out | **build-gate FAIL** (`lacks serve_soup`), 3/3 |
| **A2** raw top-24 | serve present but no coverage structure | **free-rides**, `ego_sole=0`, partner>ego, 3/3 |
| **A7** minus-serve | coverage-constrained, serve factor deleted | **free-rides**, `ego_sole=0`, partner>ego, 3/3 |

A1 (selection too small) + A2 (capacity ≠ coverage) + A7 (factor deletion) together
establish the coverage-constrained support graph + serve_soup factor as causally
load-bearing — not decorative.

## 3. Necessary vs removable (this layout/split)

```
NECESSARY / load-bearing:
  actor-specific sparse credit (ego_delivery)
  coverage-constrained CE support graph
  serve_soup per-option / per-kind coverage
  dynamic next-option TD mask
  terminal-yield TRAIN PARTNER            (A9 pending: tests the partner itself)

REMOVABLE (each dropped, held-out stayed 1.0):
  Bellman seed replay            (A3 ✓)
  directed exploration           (A4 ✓)
  3× terminal-yield upweight     (A6 ✓ — only the upweight, not the partner)

RESOLVING (train-proxy pass; held-out eval queued to confirm):
  progression shaping            (A5 PASS 3/3 -> REMOVABLE; ego 24-26 >> partner 5-14)
  shaped-CE vs sparse/task-CE    (G3 PASS 3/3 -> sparse-CE support WORKS; cleaner)
  CE-collection stochasticity    (§9.2 regen-CE PASS -> selector robust to CE seed)

UNRESOLVED:
  baseline parity                (§9.8 trains pass guard on TRAIN; held-out eval is decisive)
  terminal-yield PARTNER         (A9 — trains running)
  blind held-out                 (§7 — after lock)
```

Implied simplified final method (pending held-out confirmation of A5/G3/A9):
`actor-credit + coverage-constrained SPARSE/task-CE graph + serve coverage +
dynamic next-option TD mask + terminal-yield partner` — NO progression shaping,
NO seed replay, NO directed exploration, NO yield-upweight.

## 4. G2-lite (locked simplified candidate)

`ocv2_step4_asymm_G2lite.yaml` = G2 minus seed-replay, minus directed-exploration,
yield-upweight → 1×. Keeps everything in "NECESSARY" above + progression shaping
(pending A5). This is the combined-removal test and the basis for the final rerun.

## 5. Branch-by-decision-rule (after pending jobs)

```
A5 passes  → drop progression shaping from main method (major simplification)
A5 fails   → keep it, report as terminal-stage learning stabilizer; rely on G3 to
             keep graph support independent of shaping
G3 passes  → make sparse/task-CE the default support builder (cleanest)
G3 fails   → keep shaped-CE support, report support-score sensitivity (A7 still
             shows the selected serve factor is load-bearing)
A9 passes  → terminal-yield partner also removable (cleaner still)
A9 fails   → terminal-yield is a necessary train-only terminal-ownership curriculum
baselines match G2 → narrow the claim (curriculum/reward/option engineering, not
             ARIS-specific). baselines fail → strong ARIS claim holds.
regen-CE passes → selector robust to CE stochasticity.
regen-CE fails  → support-estimation reliability problem (more CE episodes /
             reserved coverage), NOT a Bellman-control failure; do not touch policy.
```

## 6. Final clean rerun (after A5/G3/A9 decide)

```
final method: 5 seeds × (≥2 CE seeds) × 50–100 eval episodes/partner, CIs reported.
  no seed replay (A3), no directed exploration (A4), yield 1× (A6),
  progression per A5, support-CE per G3, terminal-yield per A9.
20 ep/partner was a rapid robustness read — NOT the final table.
```

## 7. Blind split (create AFTER lock; do not tune)

```
blind_1: terminal-yield-like, different bottleneck protocol
blind_2: flexible-balanced variant, changed resource priority
blind_3: dish/serve timing variant
blind_4: stochastic/adaptive local-mode variant (if available)
report: dev-heldout {bottleneck-yield, flexible-balanced} vs blind-heldout {new}
If blind fails: label current as dev success, start v2 — do not patch silently.
```

## 8. SPLIT RECLASSIFICATION (2026-07-01) — parity failure = split too easy

Held-out batch: **base_only + flat_factor reach dev-heldout completion 1.0
(ego=20/partner=0) = ARIS.** The dev-heldout partners leave ALL serving to the ego,
so the task collapses to single-agent terminal competence — a no-factor controller
solves it. The completion result is curriculum/layout-driven, **NOT ARIS-specific.**
(`global_gru` eval crashed on a diagnostics shape bug — unknown, needs `--allow_diag_skip`.)
**A9** (no terminal-yield partner) fails hard (ego=0/partner=32-46) → terminal-yield is
a **necessary train-only terminal-ownership curriculum**.

Reclassify, do NOT discard:
- **Table 1 (dev sanity split):** ARIS = baselines = 1.0. Validates the reward-credit
  fix, terminal competence, coverage-constrained graph repair, serve gate, no
  free-riding. NOT the main ZSC benchmark.
- **Table 2 (discriminative split — TO BUILD):** must require factor-local inference;
  ARIS should beat base_only/global_gru/flat_factor. Tests the core claim.

The A1/A2/A7 dissociation still shows the graph is causal *for the ARIS controller* —
but base_only needs no graph and still completes, so task-level ARIS superiority is
**unproven on this split**. Next work is BENCHMARK CONSTRUCTION, not ARIS changes.

## 8b. G2-lite FINAL (frozen engineering candidate — no more method changes)

```
keep:    actor-specific sparse credit; coverage-constrained CE graph (+ serve/plate/
         pick coverage); dynamic next-option TD mask; terminal-yield TRAIN partner (A9)
remove:  seed replay (A3); directed exploration (A4); 3x yield upweight (A6);
         progression shaping (A5 pass)
support: task/sparse CE. The A5 no-progression CE == sparse support (no shaping bonus
         in the return), and it carries progression=disabled metadata, so it also
         matches a progression-off training config — avoiding the objective-gate
         mismatch that a shaped-CE(metadata=on) + progression-off config would hit.
```

## 9. Correlation-reversal discriminative benchmark (queue §12.4-8)

Factors: **F_bottleneck {yield, push} × F_serving {ego_serves, partner_serves}.**
Partners added to `partner_pool.py` (`terminal_policy="serve"` = partner owns terminal):
```
TRAIN diagonal (factors correlated):   cr-yield-egoserve, cr-push-partnerserve
HELD-OUT off-diagonal (reversed):      cr-yield-partnerserve, cr-push-egoserve
```

**DISCRIMINABILITY PREFLIGHT (scripted oracle vs fixed ego; diagnose_split.py) — RESULT:**
```
Completion SATURATES (every ego gets >=1 delivery) -> use THROUGHPUT, not completion
  (matches sec7). Off-diagonal throughput (deliveries):
    cr-yield-partnerserve: oracle=7.9  fixed=1.0   gap 6.9x   (serving axis discriminates)
    cr-push-egoserve     : oracle=4.8  fixed=5.0   ~0         (bottleneck axis inert for
                                                               throughput; still a valid
                                                               observable spurious CUE for
                                                               serving, correlated on the
                                                               train diagonal)
VERDICT: split IS solvable by adaptation and a fixed convention fails on serve-partners
  -> PROCEED with THROUGHPUT as the eval metric. Note: base_only sees the partner in its
  obs, so it MAY react without belief; the training experiment tests belief vs reactivity.
```

CR CAMPAIGN (in flight): sparse CE regen on the 2 diagonal partners -> train
{aris_bellman, base_only, global_gru, flat_factor} x3 on the diagonal -> eval off-diagonal
throughput. Guard `require_ego_delivery_selection` OFF (ego must be free to prep when the
partner serves; checkpoint chosen by greedy return).

If the preflight passes:
```
regen sparse/task CE on the DIAGONAL train partners (coverage: serve + bottleneck);
train {base_only, global_gru, flat_factor, ARIS-G2lite, oracle} on diagonal;
eval off-diagonal with metrics BEYOND completion: time-to-complete, collisions,
  wrong-role stalls, ego/partner split, first diagnostic action, belief-entropy drop,
  Delta_info, Q-rank of the convention-dependent option;
factor deletion (serve, bottleneck separately) + shuffled relevance.
```

Decision rule (queue §13):
```
base==flat==ARIS on Table 2   -> split still not testing ARIS, or relevance adds nothing
ARIS>base but not >flat        -> factor beliefs matter, relevance routing does not
ARIS>flat under reversal       -> factor-local relevance routing SUPPORTED (the claim)
ARIS>global_gru under recomb   -> compositional local factors > monolithic modeling
only oracle wins               -> belief/evidence routing insufficient
```

## 10. §10 gate status
```
[x] G2-lite >=4/5 seeds, dev completion >=0.8, ego dominates   (Table 1 sanity)
[x] A1 raw top-16 fails; A2 no match; A7 minus-serve degrades  (graph causal FOR ARIS)
[!] matched baselines do NOT match under same curriculum        FAILED on Table 1
                                                                (split non-discriminative)
[ ] discriminative Table 2 built + passes preflight
[ ] ARIS > baselines on Table 2
[ ] blind-heldout above baselines
```
Verdict: **system validity established; the CORE ARIS CLAIM is UNTESTED until Table 2.**

## 11. LAYOUT-ARTIFACT finding + contested-layout pivot (2026-07-01)

The CR benchmark on asymm_advantages could not build: **serve_soup CE = 0.000** (and
plate_soup = 0.000). In asymm_advantages roles are **spatially fixed** (the ego is the
natural server by layout access), so "who serves" is geometry, not negotiated coordination
— NOT an interaction externality ARIS can capture. The only CE-detectable factors are
bottleneck-spatial (max 0.599) and those are throughput-navigable (auto-pathing). => The
persistent baseline parity is a **layout artifact**, not a method failure. ARIS's factor
mechanism captures interaction externalities; asymm_advantages has none at the terminal
stage.

PROBE on the CONTESTED **cramped_room** (symmetric, both agents contest pot/serve):
```
                 asymm_advantages   cramped_room
serve_soup CE         0.000            8.98      (top factor serve_soup<->clear_cell 8.4)
plate_soup CE         0.000            0.64
deliver_to_pot CE     0.305            8.98
```
=> On cramped_room, "who serves" IS a value-sufficient interaction factor. This is the
layout ARIS should be tested on. (forced_coord is a stronger fallback — mandatory handoff.)

CRAMPED BENCHMARK RESULT (3 seeds, 20ep, per-partner adaptation) — FIRST POSITIVE ARIS SIGNAL:
```
method        yield-partner(ego serves)  serve-partner(ego preps)  mean throughput
aris_bellman  ego=13.3 prt=0  (serves)   ego=0 prt=6.7  (preps)    10.0   <-- ADAPTS
flat_factor   ego=6.7  prt=0             ego=0 prt=3.3             5.0
global_gru    ego=0    prt=0             ego=0 prt=1.3            0.67
base_only     ego=0    prt=0             ego=0 prt=0              0.0    <-- fails both
```
ARIS >> base_only/global_gru => partner-belief adaptation HELPS (core claim supported on a
contested layout). ARIS >> flat_factor (2x) => factor-local RELEVANCE ROUTING matters.
Retroactively explains asymm parity: that layout's coordination wasn't a CE factor.

CAVEATS (do not overclaim): IN-DISTRIBUTION (train==eval partners) -> shows ARIS LEARNS to
adapt, not yet that it GENERALIZES; need a HELD-OUT novel-partner test for the ZSC claim.
3 seeds/20ep, aris seed1 collapsed to always-prep (2/3 adapt) -> need more seeds+episodes.
Single layout -> confirm on forced_coord/coord_ring.

## 12. Next tier (contested-layout validation)
```
1. Robustness: cramped_room, aris seeds 0-4 (+more), 50-100 ep; explain/repair seed1.
2. HELD-OUT novel partners (the ZSC test): design serving sub-protocols (S0-S3, sec6),
   train on a subset, eval on held-out compositions. ARIS should transfer; base/gru fail.
3. 2nd contested layout: forced_coord (mandatory handoff) and/or coord_ring, same suite.
4. Diagnostics (sec6) NOW meaningful (ARIS uses the serve factor): diagnose_traces.py ->
   serve Q-rank when valid, factor-advantage decomposition, belief-swap ΔQ on the serve
   factor, Δ_info vs MI. Contrast ARIS-adapt vs base_only-fail vs flat_factor-partial.
5. sec10 gate on the contested layout; then the honest write-up (asymm = Table 1 sanity;
   cramped = Table 2 discriminative where ARIS wins).
```

## sec13. HELD-OUT novel-partner test (2026-07-01) — narrow-training failure, not mechanism failure

Evaluated cramped checkpoints on 4 ORIGINAL-library partners (`ingredient-near/far`,
`dish-server`, `server-left`) never in cramped training. Result: ALL methods 0-1.7
throughput. IDENTICAL deterministic behavior across all 20 episodes/partner (greedy +
same env seed).

DIAGNOSIS from option_kind_stats on ARIS/novel `ingredient-near`:
```
option              attempts  success   |  in-dist(cr-yield-egoserve)
fetch_ingredient      100      100      |    106  106     (identical)
deliver_to_pot        100       80      |     80   60
pick_plate             40       40      |     40   40     (identical)
plate_soup             20       20      |     20   20     (identical)
serve_soup             20        0      |     34   20   <<-- BLOCKED
```
ARIS ATTEMPTS serve 20x but 0 succeed — the novel partner physically blocks the serve
station. On the in-dist partner (which has terminal_policy="yield" -> partner explicitly
avoids serve), 20/34 serve attempts succeed. So ARIS learned "serve when the partner
yields at terminal" — a policy that WORKS but doesn't cover the "partner neither yields
nor serves" mode that ingredient-near/dish-server represent.

=> The mechanism IS working (serve factor active in-dist). The failure is TRAINING
DISTRIBUTION NARROWNESS: 2 training partners can't teach 3+ behavioral modes. Standard
ZSC narrow-train pathology, not an ARIS-specific failure.

BUT: the in-distribution win alone is not a ZSC claim. To salvage the ZSC claim, either:
(a) BROADEN training — cramped train on 4+ diverse partners (yield/serve/neither/switch),
    eval on held-out modes. Standard ZSC recipe.
(b) NARROW claim — "value-sufficient CE support selection + factor-local relevance
    routing generate correct terminal-role adaptation for the partner modes seen in
    training, but the specific 2-partner curriculum here does not transfer to novel
    behavioral modes. Broader curricula are needed for full ZSC."

Option (a) is one training cycle away (~10 min per method x 3 seeds); (b) is defensible
and honest today.

## sec14. Compliance action: synthetic-partner ban (2026-07-01)

User directive: prohibit testing on synthetic/non-standard datasets. Actions:
- Reverted all `cr-*` scripted partners + `terminal_policy="serve"` + switch_ys/sy
  logic from `partner_pool.py`. Back to 7 STANDARD library partners.
- Killed the broad-training run mid-flight (was using synthetic partners).
- Kept the `ego_option_terminated_failed` evidence channel — general improvement,
  no synthetic-data dependency; fires on any option timeout against any partner.
- Rebuilt the ZSC experiment using ONLY standard partners on standard cramped_room.
  Train: {terminal-yield, dish-server, bottleneck-yield, ingredient-near}
  Held-out: {ingredient-far, server-left, flexible-balanced}
  Methods: aris_bellman + base_only + global_gru + flat_factor, 3 seeds.

## sec15. Identical-eval finding — NOT an eval bug, a partner-scoring artifact

Standard-partner held-out eval on cramped_room produced BYTE-IDENTICAL per-partner
results. Root cause found empirically:
```
Init state on cramped_room, score for each valid option:
  ingredient-near : clear_interaction_cell=98.8, fetch_ingredient=5.9, ...
  ingredient-far  : clear_interaction_cell=98.8, fetch_ingredient=5.9, ...
  dish-server     : clear_interaction_cell=98.8, fetch_ingredient=0.9, ...
  server-left     : clear_interaction_cell=98.8, fetch_ingredient=0.9, ...
  bottleneck-yield: clear_interaction_cell=99.8, fetch_ingredient=1.9, ...
  flexible-balanced: clear_interaction_cell=99.8, fetch_ingredient=1.9, ...
  terminal-yield  : clear_interaction_cell=106.8, fetch_ingredient=5.9, ...
```
Every library partner picks `clear_interaction_cell` because the +100 bonus from
`_blocking_critical_cell()` (partner_pool.py) swamps the +/-1 role bonuses. In
cramped_room's tight 6-cell space the partner starts blocking a critical cell so
this fires immediately for all 7 partners.

=> On cramped_room, library partners are BEHAVIORALLY IDENTICAL from the ego's
perspective. cramped_room + library partners is NOT a valid ZSC discriminator.
The prior "aris_bellman 10.0 vs baselines 0-5" cramped_room result used synthetic
cr-* partners (invalidated by the user's synthetic-data ban), and library partners
there give the same trajectory for everyone. The whole cramped-room ARIS-vs-
baselines narrative should be discarded.

On asymm_advantages, library partners DO differ (larger layout, partner not stuck
blocking): the G2 result had ARIS + baselines all reach ~1.0 completion under the
same curriculum. That's the honest finding for asymm_advantages: no discriminative
gap for ARIS-specific mechanism on this layout/split with only standard partners.

Path forward with ONLY standard partners + standard layouts:
- Try `forced_coord` (mandatory handoff — likely differentiates partners more)
- Try `coord_ring` (larger, forces partner interaction)
- On asymm_advantages, try a HARDER train/held-out split (smaller train subset)

## sec16. Partner-differentiation probe across standard layouts (2026-07-01)

Ran 60-step primitive-action trace per standard library partner, on 5 standard
layouts. Count of unique partner behaviors (out of 6 tested):
```
asymm_advantages    4/6  (ing-near/far/dish/server collapse; yield-family differ)
cramped_room        1/6  (all identical — +100 clear_interaction_cell bonus fires)
forced_coord        1/6  (all identical)
coord_ring          1/6  (all identical)
two_rooms           1/6  (all identical)
```
Cause: `_protocol_score` gives +/-1 role bonuses but +100 for clear_critical_cell,
+3 for bottleneck yield, +/-100 for terminal yield. On tight layouts the +100 cell-
clearing bonus fires for every partner from the initial state → identical
trajectories. Only asymm_advantages gives partners room to act; even there, only
the YIELD-family partners (terminal-yield, bottleneck-yield, flexible-balanced)
differentiate via the +3 bottleneck bonus. Role bonuses (+/-1) are too small to
matter anywhere.

**Consequence: with ONLY standard partners + ONLY standard layouts, there is no
path to a discriminative ARIS-specific ZSC test on OvercookedV2.** The 4 role-
based partners are behaviorally indistinguishable on every standard layout, and
the 3 yield-based partners only differ on asymm_advantages, where baselines
already match ARIS at 1.0 completion under matched curriculum (G2 §9.8 result).

## HONEST SCIENTIFIC CONCLUSION (standard-data only)

Validated on standard data:
- Actor-specific sparse credit fix (removes free-riding on shared delivery reward)
- Value-sufficient CE support selection (A1 raw top-K fails, A2 capacity insufficient,
  A7 minus-serve collapses)
- System-level: ARIS + baselines reach 1.0 completion on the standard
  asymm_advantages dev-heldout under the same curriculum

NOT demonstrated on standard data:
- ARIS-specific ZSC generalization gap vs base_only/global_gru/flat_factor on any
  standard OvercookedV2 layout with standard library partners

To test ARIS's ZSC claim requires either:
(a) A different environment with partners that intrinsically differentiate on
    value-critical dimensions (not role bonuses swamped by geometry bonuses)
(b) A revision of the scripted-partner scoring so role bonuses can dominate
    geometry — but that's a partner-library redesign, likely outside "standard".

## sec10 gate status (contested layout)
```
[x] ARIS learns the task on a contested layout (throughput 10 vs baselines <=5)
[x] matched baselines do NOT match under same curriculum (base 0, gru 0.67, flat 5 vs ARIS 10)
[x] ARIS > flat_factor => relevance routing matters
[ ] HELD-OUT novel-partner generalization (ZSC) — NOT yet tested (in-distribution so far)
[ ] robustness (>=4/5 seeds; seed1 collapsed) + 2nd layout + 50-100 ep
[ ] diagnostics mechanism evidence
```

---

## sec17. Root-cause repair implementation lock entry (2026-07-02, static-only)

Status: **implementation repaired statically; remote verification and scientific adjudication pending**.

This entry is append-only. It does not rewrite or erase earlier contaminated conclusions. It records the repair boundary accepted for the next clean test pass.

1. P1 / NEW-G3 are repaired in the main path: scripted partners no longer emit true `option_id`, `option_dist`, or confidence into decision evidence. Train/eval/CE share a behavior-inferred partner-option semantic. Any true scripted label, if reintroduced later, must be diagnostic-only and must not enter factor evidence.
2. P4 / S1 / S2 / S3 are repaired in the main ARIS/flat path: belief hidden state persists across option decisions, replay stores evidence masks/lengths, zero padding is masked, and option failure evidence uses a boundary annotation rather than duplicating the last primitive event.
3. P3 / S8 / S9 / S10 / S11 / D4 are repaired in the estimator path: CE artifacts carry support sidecars with `weight_sum`, `estimable_mask`, `skipped_mask`, and `measured_zero_mask`; gamma/horizon/min_weight/support objective are metadata; unsupported CE cells are not interpretable as zero externality.
4. P5 is repaired in the main method path: true partner terminal policy is not consumed by reward, exploration, seeded replay, or eval-return conditioning. Role/terminal-conditioned curricula may exist only as explicitly marked oracle-ablation or benchmark-v2 diagnostic paths.
5. S17 / S20 / NEW-2 are repaired in eval/checkpoint infrastructure: `allow_diag_skip` cannot bypass formal hard integrity checks, headline completion is ego-owned correct delivery, and a free-riding checkpoint cannot be published as deployable `checkpoint.pt` before ego-delivery eligibility passes.
6. P2 / W / D substrate issues remain under human governance. `role_conditioned_v1` is not locked as a standard benchmark by this repair. `role_conditioned_v2_candidate` is at most a candidate benchmark-v2 substrate pending a partner-differentiation certificate and Type-B human approval.
7. Historical results remain artifact-suspect. The repaired code makes the core claim eligible for clean testing; it does not validate or refute ARIS-Bellman. NEW-4 is explicitly quarantined here: CODEX_IMPL_SPEC v1-v4 method-layer iterations and the remote numbers cited there were produced before this repair, without preregistration, under unresolved role-v1 governance and contaminated P1/P5 paths. They are diagnostic-only and cannot support or refute any main claim.

Forbidden post-lock interpretations:
- Do not cite pre-repair results as validation/refutation of the core ZSC claim.
- Do not cite CODEX_IMPL_SPEC v1-v4 remote numbers, ego/prt flips, RMR comparisons, or role-v1 reruns as anything stronger than diagnostic-only NEW-4 material.
- Do not say P1 mathematically forced ARIS/flat parity; the correct interpretation is oracle-evidence contamination of the black-box inference claim.
- Do not infer “no externality” from CE=0 without `estimable_mask=true`, `skipped_mask=false`, and sufficient support.
- Do not treat role/terminal-policy-conditioned curriculum as the main black-box method.
- Do not treat team delivery or partner delivery as ego-owned success.

Next valid claim transition requires: remote gated verification of I10-I18, the registered four-arm de-oracled rerun, CE support probe, belief-persistence ablation, partner-differentiation certificate if benchmark-v2 is used, and Type-B human decision.

---

## sec18. Phase-0 decisions + preregistration (2026-07-02, pre-Phase-1 lock)

Append-only. Records the two governance decisions closing Phase 0 of
[EXPERIMENT_CHAIN_PLAN.md](EXPERIMENT_CHAIN_PLAN.md), and PREREGISTERS the
decision rules for R2.1/R2.2/E1/E2/E3. Per plan §8: results are read out by
these tables verbatim; post-hoc reinterpretation is forbidden.

### 18.1 G0.1 — partner-substrate governance (D-B, decided)

`role_conditioned_v2` is admitted **only as a controlled mechanism-diagnostic
substrate** ("benchmark-v2-diagnostic"), conditional on passing the R2.1
partner-differentiation certificate. Constraints:

1. The paper's headline ZSC generalization claim may NOT rest solely on
   scripted role_conditioned_v2 partners.
2. Claim-level ZSC evidence comes from an FCP/MEP trained-partner population
   (EXPERIMENT_PLAN original design). Sequencing: the FCP/MEP investment
   (~400-800 GPU-h) is GATED on E1 — any preregistered ARIS signal in E1
   unlocks it; a clean four-arm null after artifact checks defers it.
3. Framing rule: scripted v2 = controlled mechanism analysis (E1-E6);
   FCP population = ZSC benchmark evidence (E7 / final tables). Both labeled
   as such in any write-up. This respects the 7/1 synthetic-data directive's
   intent: no headline claim rests on self-designed partners.

### 18.2 G0.2 — free-rider guard in role_v1 formal configs (D-A, decided)

`require_ego_delivery_selection` flipped to **true** in
`ocv2_step4_asymm_role_v1.yaml` and `ocv2_step4_asymm_role_v1_novb.yaml`.
Rationale: train partners include yield-family → a correct adaptive policy
necessarily has ego-sole deliveries in mixed-partner greedy validation, so the
guard cannot reject good policies; it rejects exactly the historically observed
always-prep collapse (sec11 seed1). The written-waiver alternative was
rejected: contrib_team pays prep-only policies on claim-partner deliveries, so
mean-return selection alone cannot be proven to filter the collapse.
General rule (ledger): any config whose train set contains ≥1 partner mode
where the ego should serve keeps this guard ON.

### 18.3 Fidelity-gate note

The mechanical gate now implements I1–I17 (17/17 PASS on this tree;
`FIDELITY_GATE.{md,json}` are tool-generated from now on). I18 (decisive-rerun
archive + Type-B human checkpoint) is process-level, not statically checkable:
it is tracked by EXPERIMENT_CHAIN_PLAN Phases 3–5 and this file's dated
entries, not by the gate.

### 18.4 PREREGISTRATION — R2.1 partner-differentiation certificate

Procedure: per candidate layout × partner set: fixed seeds + randomized
starts; 60–100 primitive-step open-loop traces AND full episodes with a
competent scripted ego (NOT noop-ego); pairwise trajectory/action/return
divergence on value-critical phases; scripted-oracle completability; verify
partner-only play does not saturate the task.

| Outcome | Preregistered conclusion |
|---|---|
| ≥2 distinguishable behavior modes on each of ≥2 value-critical factors | PASS — substrate admitted for E1-E6 |
| Distinguishable on a factor subset only | Narrow: re-cut train/held-out along the distinguishable subset; record the narrowing |
| Collapse (indistinguishable) | FAIL — STOP; no ARIS-vs-baseline claim on this substrate; escalate to FCP/MEP line |

### 18.5 PREREGISTRATION — R2.2 asymm CE support probe

500–1000 episodes/train-partner; artifact must carry per-pair weight_sum +
all masks + CI. Citation rule: a zero is interpretable only with
`estimable_mask=true`.

| Outcome | Preregistered conclusion |
|---|---|
| serve CE measured-nonzero with support ≥ min_weight | sec11's "asymm has no terminal externality" is REFUTED; asymm re-admitted as discriminative-layout candidate |
| serve CE measured-zero with support ≥ min_weight | geometry explanation SUPPORTED; asymm remains Table-1 sanity only |
| support still < min_weight | inconclusive — report support, raise collection budget or targeted starts; NO conclusion |

### 18.6 PREREGISTRATION — E1 de-oracled four-arm rerun (decisive)

Arms: aris_bellman / flat_factor / global_gru / base_only (+partner_id_q as
oracle upper reference, excluded from claims). 5 seeds × 50–100 eval
episodes/partner; train on train split, eval dev-heldout; ALL hard integrity
flags true or the run is inadmissible. Headline: ego_correct_completion_rate +
throughput; secondary: time-to-complete, ego/partner split, wrong-delivery,
first-diagnostic-action timing.

| Outcome (CI-separated) | Preregistered conclusion |
|---|---|
| ARIS > flat > base | factor-local relevance routing SUPPORTED (Claims 2/4 discriminative evidence) |
| ARIS ≈ flat > base | beliefs useful, routing adds nothing → narrow to weak Claim 2 |
| all four ≈ | run artifact-suspect checklist; if clean → record "no ARIS-specific advantage on this substrate" (honest null); do NOT retro-blame P1 |
| only ARIS collapses | treat as de-oracling regression; fix, rerun; no scientific conclusion |
| partner_id_q >> ARIS | quantify the inference gap as upper-bound distance; analysis only |

FCP/MEP gate: any of rows 1-2 → unlock the population line (18.1). Row 3 after
clean artifact checks → defer FCP/MEP, reassess method.

### 18.7 PREREGISTRATION — E2 oracle-channel ablation + E3 belief-persistence ablation

E2: same checkpoints, eval twice (behavior-inferred channels vs zeroed).
| Both arms drop hard when zeroed | evidence channels carry the load (expected) |
| ARIS retains separation when zeroed | ARIS advantage does NOT come from partner-option inference — report honestly, revisit mechanism claim |

E3: window-4 / window-8 / persistent-hidden (main), 3 seeds each.
| Only persistent improves repeat-failure held-out cases | P4 mechanism claim SUPPORTED |
| No difference | held-out failure was substrate/distribution — mechanism claim stays unproven; do not claim accumulation |


### 18.8 Amendments (2026-07-03, post-Phase-1 feasibility calibration; append-only)

1. **sec18.6 power precheck (amends the "all four ≈" row):** with 5 seeds only large
   effects (d≳1.8 @ 80% power) are detectable. Before reading "all four ≈" as a null,
   apply: **CI overlap with consistent ordering across seeds → run +5 seeds once and
   re-adjudicate; only a second overlap permits the null reading.** CI
   operationalization: seed-level bootstrap 95% (10k resamples) or Mann-Whitney U.
2. **sec18.7 E2 zeroed-mode integrity note:** the zeroed-channel ablation eval must
   record `evidence_policy=behavior_inferred_v1_zeroed_ablation` (explicit mode) so
   the S17 hard gate can admit it without weakening the formal-path check.
3. **Formal-path rule (Phase-1 finding R1-A):** formal runs consume the pipeline
   `graph.json` via the `graph_path` branch (unconditional coverage gate + stamped
   provenance). The `ce_path` branch with `require_task_stage_coverage=false` is
   smoke-only and must never appear in a formal config.

## sec18.9 R2.1 supplemental preregistration — throughput lens + layout sweep (2026-07-03)

Append-only. Extends the R2.1 preregistration (sec18.4) after the 2026-07-03 result
revealed that the completion-rate lens saturates on asymm_advantages: three of the
six admissibility cells were rejected purely because a random ego eventually
completed (rand=1.0) or a partner completed alone (ponly=1.0). This DOES NOT flip
sec18.4 — it operationalizes the throughput lens that sec9 already established as
the discriminative metric, so we do not "shop" for the friendlier reading.

### 18.9.1 Justification (why throughput, why now)
- Preregistered lens history: sec9 CR preflight already switched from completion
  to throughput ("Completion SATURATES → use THROUGHPUT"); the R2.1 completion
  thresholds are the legacy substrate-certificate defaults, not the sec9 lens.
- Behavioral vs value cleavage: the R2.1 differentiation probe PASSED (27/28 pairs
  distinguishable, 2/2 axes). What the completion certificate rejected is a value
  cleavage, which throughput can register while completion cannot.
- Non-shopping guard: this throughput lens is applied UNIFORMLY across all
  candidate layouts before E1 base is chosen — it is not tightened for a layout
  after it fails.

### 18.9.2 Throughput admissibility metrics (per (layout, partner))
Compute over N seeds × E episodes:
- `fsm_throughput`   : mean(ego_deliv + partner_deliv) under competent ego, all seeds
- `rand_throughput`  : same, random ego
- `ponly_throughput` : same, ego = noop
- `ego_serve_share`  : mean(ego_deliv / (ego_deliv + partner_deliv)) when denom > 0
- `fsm_ttfs`         : time-to-first-serve under fsm (option-steps; +inf if none)

### 18.9.3 Admissibility predicate (preregistered, no per-layout override)
A (layout, partner) cell is ADMITTED iff ALL hold:
- `fsm_throughput  >= --fsm-tp-min`   (default 0.8 — a skilled ego actually serves)
- `rand_throughput <= --rand-tp-max`  (default 0.4 — random ego does not)
- `ponly_throughput <= --ponly-tp-max`(default 0.4 — partner does not solo)
- `fsm_throughput - rand_throughput >= --gap-min` (default 0.5 — skill gap exists)

Partners whose completion-lens admission was rejected only via ceiling saturation
(ingredient-*: rand_completion=1.0 with rand_throughput low; server-*:
ponly_completion=1.0 with ponly_throughput low) become admissible under this
predicate when the throughput reading really does show a gap.

### 18.9.4 Layout sweep decision table (preregistered)
Sweep `{asymm_advantages, cramped_room, forced_coord, coord_ring}` × role_conditioned_v2.

| Sweep outcome (throughput lens) | Preregistered E1 base decision |
|---|---|
| cramped_room dominates both axes (serving-CE precondition from sec11 + admissible held-out) | cramped_room = E1 base; asymm → Table-1 sanity |
| only forced_coord / coord_ring admissible (bottleneck/handoff as protocol, not path) | move E1 to that layout; document as "coordination-protocol not navigation" |
| asymm throughput-admits ≥ 2 held-out on the serving axis | asymm stays; replace or drop heldout-resource-server-claim (which stays degenerate) |
| any layout: fsm_throughput < 0.8 AND rand_throughput < 0.4 on all partners | probe budget too small → escalate max-options / episodes, do NOT flip verdict |
| all layouts admit 0–1 held-out cells under throughput | scripted-v2 substrate exhausted → G0.1 pre-decided exit: FCP/MEP population line |

### 18.9.5 FSM-ego probe hygiene (mandatory)
Certificates must average over ≥ 3 seeds per (layout, partner, policy). Single-seed
deterministic-FSM zeros (e.g., R2.1's ingredient-far-yield fsm=0.0 vs rand=1.0) are
geometry-lock artifacts, not real failures. Report `n_seeds` in every artifact.

### 18.9.6 Held-out membership: not decided by the sweep
`heldout-resource-server-claim` was ponly=1.0 on asymm; it is not "solved" by this
supplement. Whether to replace it (v2.1) or drop it stays a governance decision
(G0.1 benchmark-v2-diagnostic scope). This sec only decides the layout base of E1.

### 18.9.7 Non-goals
- Does not override sec18.4 completion admissibility for narrower purposes
  (curriculum sanity, Table-1); those still use completion.
- Does not change the R2.2 CE support probe (sec18.5) — R2.2 remains
  archival-obligation for P3 sentinel and CE estimand discipline. If asymm is
  demoted, R2.2 collapses in scope but should still be run on whichever layout E1
  chooses, since CE support underlies P3 for ALL bases.

## sec18.10 R2.1 probe methodology revision — E1 as decisive admissibility test (2026-07-03)

Append-only. sec18.9 preregistered a throughput lens on top of the completion
lens; the 2026-07-03 sweep showed both lenses fail on v2 partners for a subtler
reason: `rand_throughput > fsm_throughput` on multiple (asymm/cramped) cells,
which is physically impossible for a competent oracle ego. Root cause: the
FSM ego (state-based inventory pipeline) is not role-aware; v2 partners are
role-aware (yield/claim/handoff); FSM contests the terminal with partners
whose role expects them to serve, dropping throughput below random-ego (which
sometimes stays out of the way).

The failure is METHODOLOGICAL, not substrate: the R2.1 differentiation probe
PASSED cleanly (2/2 axes, 27/28 pairs distinguishable), and the cramped×v2
sanity showed `ponly_throughput=0` across all partners (v2 partners are NOT
partner-solo on cramped, unlike asymm/server-*). The substrate has both
diversity AND ego-necessity signals; only the FSM-based admissibility gate
cannot register them.

### 18.10.1 Decision (user-approved, 2026-07-03)
- ABANDON the FSM-oracle admissibility gate for v2 partners. Do NOT design a
  role-aware oracle probe — it would need to read `partner.protocol.terminal_policy`
  and thereby violate P5 in spirit even if isolated to the probe.
- USE E1 ITSELF as the decisive admissibility test. This is admitted because
  Phase-1 fixed the eval-integrity gates (I10-I17) that previously masked
  substrate quality, and the four arms + partner_id_q reference now form a
  self-diagnostic instrument.
- The R2.1 differentiation certificate (already PASSED) remains the ONLY R2.1
  precondition for E1.

### 18.10.2 E1 admissibility read (preregistered, added to sec18.6)
E1 outputs a substrate admissibility verdict as a BY-PRODUCT of its own
readings, using the arms and the partner_id_q reference already in the
preregistered design:

| E1 signal | Substrate admissibility conclusion |
|---|---|
| base_only reaches `ego_correct_completion_rate >= 0.9` on held-out | Substrate is NOT ZSC-discriminative (base solves without any belief/routing). Record as diagnostic; do not claim ARIS advantage regardless of relative ordering. |
| partner_id_q strongly separates from ARIS/flat/base | An oracle information gap exists → belief-inference is genuinely required → substrate is admissible for ARIS-claim reading. |
| partner_id_q ≈ ARIS ≈ flat ≈ base | Substrate provides no information gap → not admissible for the inference-claim reading, but sec18.6 rows still apply to routing/simplicity claims. |
| Any arm exhibits `oracle_source_count > 0` or `reward_scale_verified=false` | ineligible; do not read (I10-I17 gate enforces this). |

The four-arm read replaces the FSM-oracle admissibility read of sec18.4/18.9.
Both prior R2.1 certificates stay archived as historical context; they do NOT
gate E1.

### 18.10.3 R2.2 scope collapse
sec18.5 asymm CE support probe is REDUCED to an archival P3-sentinel run
(≤300 ep/partner, one seed, ≈45 min) executed AFTER E1 to close the "is
serve_soup CE below-support-sentinel vs measured-zero" question permanently.
It is NOT a fork gate anymore. If E1 renders a verdict, the fork is
already resolved.

### 18.10.4 What did not change
- The R2.1 differentiation gate must still PASS before E1 (it did).
- All E1 integrity flags (I10-I17) still hard-block reading. Nothing in this
  revision weakens those.
- sec18.9 stays on file as a completed but methodologically superseded gate.

## sec18.11 Pre-E1 innovation preregistration (2026-07-04, user-directed)

Append-only. User directive: correctness as precondition, innovation as goal,
ICLR as the bar. Recorded BEFORE any E1 data exists — this is the only honest
window for method-layer changes; once E1 produces numbers, the method layer
freezes until read-out completes.

Direction (full plan: ICLR_UPGRADE_PLAN.md; ledger IDs U1/U2):
1. U1 (from D6): interventional/active factor discovery — targeted-start
   collection + forced-option interventional CE (refine_interventional_ce is
   in-tree, unwired) with per-factor support certificates. Motivated by the
   measured impossibility: mutual-exclusion coordination factors carry
   exactly-zero passive joint support (12000-row evidence).
2. U2 (from S27): persistent protocol-mode belief (FactorModeFilter) — the
   inference latent moves from transient partner options to episode-persistent
   factor modes, aligning the implementation with the proposal's own §5.2
   theory and eliminating the support-freeze bug class by construction. The
   transient-option inferencer (S27-fixed) remains as a diagnostic + ablation
   arm.
3. E1 gains preregistered arms: aris(option-infer) vs aris(mode-filter), and
   passive-CE graph vs interventional-CE graph. Read-out additions are in
   ICLR_UPGRADE_PLAN §5; sec18.6/18.10.2 readings stay authoritative.
4. FCP/MEP spike starts EARLY (no longer gated on E1 signal) — amends the
   sec18.1 gating: the K2 feasibility spike (1-2 days) is ungated; the full
   population investment still awaits the spike verdict + user approval.
5. Obligations before the story is committed: /novelty-check on U1 and U2;
   formalization attempt of the U1 exclusion-support proposition; honest
   demotion paths are preregistered in ICLR_UPGRADE_PLAN §7.

Forbidden: reading any E1/E2/E3 result against a rule not written down before
the run; tuning U1/U2 designs on decisive-run outcomes.

## sec18.12 E1-rev preregistration — oracle-FREE terminal-competence scaffolds (2026-07-04, SIGNED-OFF by user same day)

Append-only. Drafted while the E1 (no-scaffold) wave is still running; readings
below are preregistered BEFORE any E1 eval is read.

### 18.12.1 Trigger (measured, not speculative)
All completed E1 training runs (aris_bellman seeds 0-4, base_only seed 0) show
ego_correct_delivery ∈ {0,1} vs partner 6-12; pilot guard verdict
`free_riding_partner_serves_while_ego_idle`; no deployable checkpoint. Root
cause analysis (EXPERIMENT_LOG 2026-07-04): (1) contrib_team pays prep against
the 3/6 claim partners → free-riding optimum; (2) terminal chain
pick_plate→plate_soup→serve is exploration-starved (measured 52/16 rows per
12000 under random play); (3) dense prep shaping (coef 1.0) is a broad local
optimum. Historical continuity: identical to RC-era "ego never serves once in
5000 updates" (RC_REWARD_CREDIT_FIX §4b); G2 solved it with scaffolds; P5
correctly removed the oracle-conditioned ones, but role_v1 config ALSO disabled
the two ego-kind-only (P5-clean) scaffolds. E1(no-scaffold) is measuring that
vacuum.

### 18.12.2 E1-rev change set (option (c) — P5-clean, no partner-truth access)
1. `terminal_progress_shaping.enabled: true` with ego-kind-only bonuses
   (pick_plate / plate_soup / serve): conditions ONLY on the ego's own option
   kinds and ego inventory (P5-audited ego-local; precedent: enabled in
   ocv2_step4_asymm.yaml formal config).
2. `terminal_exploration.enabled: true` (ego-kind exploration bias; P5-audited:
   `_sample_exploration_option` has no partner-policy parameter).
3. `sparse_credit: contrib_team` UNCHANGED (keeps role semantics: prepping for
   a serving partner remains rewardable).
4. Everything else identical to E1 (same graph, same partners, same seeds
   0-4, same budget 5000 updates).
Bonus values: copy the asymm.yaml formal-config values verbatim (no tuning
against E1 outcomes — they predate E1).

### 18.12.3 Preregistered readings (E1-rev)
| Outcome | Reading |
|---|---|
| ego terminal competence appears (guard passes, deployable checkpoints exist) for ≥3/5 seeds in ≥2 arms | E1-rev becomes the decisive run; apply sec18.6 + sec18.10.2 tables to E1-rev eval |
| competence appears in some arms but not others (e.g. aris yes, base no or vice versa) | THIS IS ITSELF a discriminative signal — read per sec18.6 rows with "terminal competence rate" as a co-primary metric alongside ego_correct_completion_rate |
| still no arm reaches competence | scaffolds insufficient at this budget → escalate budget (10k updates) once; if still zero, record "asymm×v2 terminal chain unlearnable under P5-clean training at this scale" as an honest negative and shift the discriminative question to non-terminal factors (bottleneck axis) |
| E1(no-scaffold) vs E1-rev comparison | archived as the SCAFFOLD ABLATION — quantifies how much of terminal competence is scaffold-driven vs method-driven; feeds the paper's honest-reporting section (curriculum-dependence was a G2-era criticism; now measured) |

### 18.12.4 Guards
- E1(no-scaffold) wave runs to completion and is archived as baseline record;
  its eval is NOT read against sec18.6 (no deployable checkpoints ⇒ nothing to
  read); its training metrics ARE the scaffold-ablation baseline.
- No further reward/exploration tuning after E1-rev results are seen (one
  budget escalation preregistered above is the only allowed knob).
- All integrity gates unchanged (I10-I17); scaffolds are ego-local by
  construction and do not touch the evidence path.
