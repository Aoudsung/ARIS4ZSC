# Option-Executor Behavior — Findings & Resulting Problems (RC-2b)

Scope: the OvercookedV2 **macro-option executor** — how a selected option turns into primitive
actions and terminates — and the behaviors that make task **completion structurally 0** on
cramped_room. All findings are evidence-driven (per-step traces + oracle runs), not inferred.

Context: RC-1 (training value divergence) is fixed and verified. RC-2b asks *why no policy ever
completes the task*. This document records what the **executor** does, observed on a state-aware
optimal oracle (so the learned policy is not the variable). It is a findings doc, not a fix plan.

Code under study: `experiments/overcooked_v2/options.py` (`OCV2OptionLibrary`),
the option-step loop in `experiments/overcooked_v2/evaluate_aris.py:_execute_eval_option`,
and `experiments/overcooked_v2/partner_pool.py` (the partner uses the *same* executor).

Evidence artifacts (cxw2 → local `results/ocv2_rc1_bounded/`): `rc2b_reachability_oracle.json`,
`rc2b_reach_patience0.json`, `rc2b_reach_fsmfix.json`, `rc2b_trace.json`, plus the inline traces quoted below.
Diagnostic scripts: `scripts/rc2b_reachability.py`, `rc2b_layout_scan.py`, `rc2b_trace.py`, `rc2b_inspect.py`.

---

## 0. How the executor works (mechanism)

A selected option drives the ego one primitive step at a time:
- `primitive_action` (`options.py:72-97`): if the agent is **adjacent to the target entity** and
  facing it → `interact`; if adjacent but not facing → turn toward it; else **navigate one step**
  toward the closest stand cell.
- `_interaction_action` (`options.py:277-294`): adjacency (`|dx|+|dy|==1`) + facing check; correct.
- `_next_cell_toward` (`options.py:320-344`): pick the adjacent passable cell with the smallest
  shortest-path distance to the target **that is not the partner's current cell**
  (`blocked={partner_pos}`). **Deterministic; single-agent; only the partner's *current* cell is avoided.**
- Each entity option's step budget is `max_option_steps` (`options.py:220`); the option ends on its
  terminal event or when the budget is exhausted.
- The **partner uses the identical executor** (`partner_pool.py:102` calls `option_library.primitive_action(state, 1, …)`).

Layout under study (cramped_room, from `rc2b_inspect`): **6 passable cells**; every interaction
entity has **exactly one** adjacent passable stand cell:
`pot (2,0)→stand (2,1)`, `plate_pile (1,3)→stand (1,2)`, `ingredient piles→(1,1)/(3,1)`, `delivery (3,3)→(3,2)`.

---

## 1. Observed behaviors (each with evidence)

### B1 — Greedy, deterministic, single-agent navigation → simultaneous-move collisions / intermittent livelock
Both agents plan only around the partner's **current** cell and move **deterministically**. On a
simultaneous step the partner can move into the very cell the ego planned, so the move collides and
both bounce; next step the deterministic choice repeats → transient livelock. Evidence (full-pipeline
trace, contesting the pot stand (2,1)):
```
deliver  agent(1,1) partner(2,1) act4   4->4 blk=F   ← partner sits on pot stand
deliver  agent(1,1) partner(2,1) act4   4->4 blk=F   ← ego cannot advance
deliver  agent(2,1) partner(1,2) act3   4->4 blk=T   ← collision as they cross
deliver  agent(2,1) partner(3,2) act5(interact) 4->0 ← finally delivers once partner clears
```
Net effect: options that *should* take ~3 steps take 5+, and many stall for several steps. It is not
a hard deadlock by itself, but it sharply lowers throughput in tight space.

### B2 — Sequencing trap: pre-fetch a 4th ingredient → hands occupied → cannot `pick_plate`
`fetch_ingredient` stays valid (`_task_precondition`, `options.py:364-372`, via
`_ingredient_has_future_sink`) **even when the only pot is already full and cooking**. The agent
fetches an extra ingredient; its hands are now occupied; but `pick_plate` requires **empty hands**
(`options.py:373-388`, `is_empty_inventory`), and the agent has no deliver target (pot full) and no
recovery, so it idles forever holding the ingredient. Evidence (pot fills then traps the agent):
```
deliver → pot c8 -> c12C  (3 ingredients, COOKING)
fetch   → inv 0 -> 4      (grabs a 4th ingredient while the pot is cooking)
clear_interaction_cell × ∞   agent stuck inv=4, pot c12C -> c14R (READY) but never plated
```
This is a property of the **option set + preconditions** (a fixed kind-priority or a naive policy
walks straight into it). It is recoverable in principle (`drop_item_to_counter` exists) but nothing
triggers the recovery. *(Mitigated in the oracle FSM by "finish current soup before pre-fetch + drop
to recover"; the underlying precondition over-permissiveness remains.)*

### B3 — Single-stand-cell chokepoint + partner occupancy → option can never complete
After B2 is mitigated, `pick_plate` is finally attempted but succeeds **0/36**. The plate pile (1,3)
has exactly one stand cell (1,2); the scripted partner (dish-server) **permanently occupies (1,2)**.
The ego can never stand there, so the **cooked, ready** soup is never plated. Evidence (40+ steps,
across 3 pick_plate options):
```
pick_plate  agent(2,1) partner(1,2) target(1,3) blk=T  pots[c12C]
pick_plate  agent(2,1) partner(1,2) target(1,3) blk=T  pots[c14R]   ← soup READY, partner never leaves (1,2)
... (identical every step; ego blocked every step; pick_plate times out) ...
```

### B4 — When the target is blocked, the option burns its whole budget in place
`_next_cell_toward` returns a colliding move (or none); the executor has **no "give up / step
aside / re-plan"**. The option thrashes for all `max_option_steps`, contributing nothing. The
opposite over-correction is also a hazard: an aggressive "terminate after N no-progress steps"
patience (`block_patience=3` in the F2 prototype) **kills recoverable options** that were merely
stalled by B1 — it cut deliveries to ~2/episode and produced 360+ premature terminations. So the
executor lacks a *calibrated* blocked-handling policy: it either thrashes (patience off) or
over-terminates (patience too tight).

### B5 — The executor is shared, so the partner has the same failure modes
Because `partner_pool` drives the partner through `option_library.primitive_action`, the partner
also navigates greedily/deterministically and can **camp a chokepoint** (B3) or contend (B1). There
is **no yield / step-aside protocol on either side**, so chokepoint contention has no resolution path.

---

## 2. Resulting problems (impact chain)

| # | Problem | Caused by | Evidence |
|---|---|---|---|
| P1 | **completion = 0 for every method and every partner**, although the pot fills, cooks, and reaches READY | B2 → B3 (+B1 throughput) | oracle completion 0 (all 6 partners) with `pots c12C→c14R`; pick_plate 0/36 |
| P2 | "return" only ever measures **ingredient-farming** (fetch→deliver), never task success | the pipeline cannot pass plating | return ~5.3 with completion 0; `served_soup` only via partner, `ego_delivery=0` |
| P3 | The benchmark, as configured, **does not test learnable coordination** — it is a structural chokepoint deadlock with no yield mechanism, so **no policy** (learned or optimal) can complete it | B3 + B5 | state-aware optimal oracle also completes 0 |
| P4 | Downstream claim experiments are **untestable on this setup** (graph load-bearing, Δ_info vs MI need a completing, coordinating policy) | P1–P3 | RC-3/RC-4 gated in the plan |
| P5 | Earlier "completion=0 / farming under a diverging policy" conclusions were a **downstream symptom** of P1, not the root | misattribution before RC-2b | prior smoke matrices on cramped_room |

The F4 reachability gate correctly **rejects** such `(layout, partner)` instances — that rejection is
principled given B1–B5, not blind layout-picking.

---

## 3. Distilled root cause

> The macro-option executor uses **deterministic, single-agent greedy navigation that only avoids
> the partner's current cell and has no yield/step-aside or blocked-recovery**, and the option
> preconditions allow a **pre-fetch sequencing trap**. On layouts where interaction entities have a
> **single stand cell** and a fixed partner **occupies/contests** it, this produces an unbreakable
> chokepoint deadlock: the soup cooks but can never be plated/served → completion 0, for the optimal
> oracle and therefore for any learned policy.

Three independent contributors; all three present on cramped_room:
1. **Layout** — single-stand-cell interaction chokepoints (no alternative approach).
2. **Partner** — permanently camps/contests the chokepoint (shared-executor behavior, B5).
3. **Executor** — no yield/step-aside, no calibrated blocked-handling, plus the precondition pre-fetch trap.

## 4. Open sub-questions (for the fix phase; not resolved here)
- **Why does the partner permanently occupy (1,2)?** Is `ScriptedProtocolPartner` itself deadlocked
  / parked by its protocol (a partner bug → fixable, salvages tight layouts), or is it intended
  behavior (→ exclude via the reachability gate)?
- Does a calibrated yield/step-aside in the **shared** executor (both agents) resolve the mutual
  deadlock without changing option semantics or method fidelity?
- Which standard layouts have **≥2 stand cells** per interaction entity (structurally robust to a
  camping partner) — i.e. the principled admittable set from the layout scan.
