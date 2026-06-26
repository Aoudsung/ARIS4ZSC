# Fix Plan — 9 Blocking/Scope Issues (OvercookedV2 ARIS pipeline)

Status: DRAFT for Codex review. Grounded against current `main` (89d357f).
Constraint: static edits only; running tests/experiments and CE regeneration are
execution-gated handoffs (remote SSH already authorized for Step 4).

Guard rails enforced in every fix below:
- Must NOT break the in-flight run7 verification micro-train.
- Must NOT introduce regressive (回退式) or over-defensive (过防御式) changes.
- Event-semantics fixes must align with OvercookedV2 dynamic-object bit encoding.

---

## Problem 1 — `_pot_became_full()` fires on first ingredient, not on full pot
Severity: HIGH. File: `event_extractor.py:256-270`.

Verified root cause: current predicate is
`not has_ingredient_bits(before) and has_ingredient_bits(after)` → "empty → any
ingredient", inflating coverage / task-progress / CE stage coverage.

Fix (use the EXISTING helper, no new import):
`state_utils.is_pot_full()` already exists (`state_utils.py:92` =
`ingredient_count_py(contents) >= MAX_INGREDIENTS`). So:

```python
from .state_utils import is_pot_full  # add to existing import block

def _pot_became_full(pot_changed_cells, changed_cells, before_values, after_values) -> bool:
    for before, after in _changed_pot_values(pot_changed_cells, changed_cells, before_values, after_values):
        if not is_pot_full(before) and is_pot_full(after):
            return True
    return False
```

Encoding note (verified): COOKED/PLATE are the low 2 bits; `ingredient_count_py`
shifts them out, so a full pot that later cooks does NOT re-fire (before already
full → `not is_pot_full(before)` is False). Transition fires exactly once.

Test (`tests/test_event_extractor.py`, execution-gated):
- before=0x0, after=0x4 (1 ingredient) → False
- before=0x4, after=0x14 (2) → False
- before=0x14, after=0x54 (3 = MAX) → True
(Use real bit patterns matching MAX_INGREDIENTS=3; exact constants resolved against
`jaxmarl.environments.overcooked_v2.common` at test-write time.)

---

## Problem 2 — `delivery_event` counts "drop plated soup on counter" as delivery
Severity: HIGH. File: `event_extractor.py:111-119, 209-210`.

Verified root cause: `_delivered_soup` only checks
`interacted and is_plated_cooked_soup(before) and is_empty_inventory(after)` — no
check that the interacted target is a delivery cell.

Verified primitives available (no signature change needed):
- `state_utils.agent_facing_pos(state, agent_id)` (`state_utils.py:182`).
- Static grid is `state.grid[..., 0]`; delivery cell is `StaticObject.GOAL`
  (confirmed by `layout_parser.py:63`).

Fix: add a delivery-target check using the PRE-step state (interaction resolves
against pre-step orientation):

```python
# state_utils.py
def cell_is_delivery(state, pos) -> bool:
    grid = np.asarray(state.grid)
    x, y = pos
    if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]:
        return int(grid[y, x, 0]) == int(StaticObject.GOAL)
    return False
```

```python
# event_extractor.py — replace delivery_event assignment
def _delivery_target(prev_state, agent_id) -> bool:
    return cell_is_delivery(prev_state, agent_facing_pos(prev_state, agent_id))

delivery_event = (
    _delivered_soup(ego_inventory_before, ego_inventory_after, ego_interacted)
    and _delivery_target(prev_state, 0)
) or (
    _delivered_soup(partner_inventory_before, partner_inventory_after, partner_interacted)
    and _delivery_target(prev_state, 1)
)
```

Optional secondary signal (mirrors the existing `_explicit_wrong_delivery`
pattern, only if key present — NOT a guess-driven default): if `info` exposes a
verified delivery key, prefer it; else use the facing-cell check above. To avoid
guessing unverified key names, the facing-cell check is the PRIMARY signal; the
info path is added only after confirming the actual OvercookedV2 info schema on
remote. (Do not invent `correct_delivery`/`successful_delivery` keys unverified.)

Proposal-safety: tightens a false-positive; does not weaken any real delivery
detection (a true delivery always faces a GOAL cell).

---

## Problem 3 — `_plate_picked` also fires for soup pickup
Severity: MEDIUM-HIGH. File: `event_extractor.py:304-306`.

Verified root cause: `has_plate(after)` is True for plated soup too, so
plate-pickup is conflated with soup/plate_soup transitions.

Fix: add `is_plain_plate` and use it.

```python
# state_utils.py
def is_plain_plate(inv: int) -> bool:
    return has_plate(inv) and not has_ingredient_bits(inv) and not is_cooked(inv)
```

```python
# event_extractor.py
def _plate_picked(*inventories: int) -> bool:
    pairs = ((inventories[0], inventories[1]), (inventories[2], inventories[3]))
    return any(not is_plain_plate(b) and is_plain_plate(a) for b, a in pairs)
```

`_soup_picked` unchanged (already
`not is_plated_cooked_soup(before) and is_plated_cooked_soup(after)`).

Test: empty→plate True; plate→plated_soup False; empty→plated_cooked_soup False.

---

## Problem 4 — `run_trace_diagnostic.py` recreates `OptionRuntime` every primitive step
Severity: MEDIUM (corrupts deadlock-fix verification only). File:
`scripts/run_trace_diagnostic.py:108`.

Verified root cause: `rt = OptionRuntime(...)` is INSIDE `for step_i in
range(opt.max_steps)` (line 108), resetting bottleneck runtime state each step.
Training/eval create runtime at option level (correct); only this diagnostic is
wrong.

Fix: move construction above the step loop (after `pos_start` at line 90):

```python
pos_start = get_agent_pos(state, 0)
rt = OptionRuntime(option_id=oid, start_pos=pos_start)
for step_i in range(opt.max_steps):
    ...
    terminated, reason = option_terminated(opt, prev_state, state, event, 0, step_i + 1, rt)
```

Proposal-safety: aligns diagnostic with the training runtime lifecycle; no
behavior change to training/eval.

---

## Problem 5 — batched CE collection over-collects (triangular schedule) + duplicate episode_ids
Severity: HIGH (only the batched efficiency path). File:
`ce_sampler.py:171, 224, 356-358, 377` and `_EnvSlot`.

Verified root cause: slot `i` is seeded with `episodes_remaining = episodes - i`
and `done = i >= episodes`. For episodes=100,batch=64 each partner runs
sum_{i=0}^{63}(100-i)=4384 episodes (~44×), and `episode_id = base + (episodes -
remaining)` collides across slots.

Fix: replace per-slot countdown with a single global episode queue per partner.

```python
# remove episodes_remaining from _EnvSlot; add nothing else.
next_episode_idx = 0  # per-partner, before the while loop

def _assign_episode(slot, idx):
    slot.episode_id = episode_base + idx
    slot.t_option = 0
    slot.needs_new_option = True

# init: give the first min(batch_size, episodes) slots one episode each
for i, slot in enumerate(slots):
    if i < episodes:
        _assign_episode(slot, i); slot.done = False
    else:
        slot.done = True
next_episode_idx = min(batch_size, episodes)

# on episode end (replaces lines 355-368):
if done_i or slot.t_option >= option_limit:
    if next_episode_idx < episodes:
        idx = next_episode_idx; next_episode_idx += 1
        _assign_episode(slot, idx)
        new_seed = int(rng.integers(0, 2**31 - 1))
        reset_indices.append(i); reset_seeds.append(new_seed)
        if hasattr(slot.partner, "reset"): slot.partner.reset(new_seed)
    else:
        slot.done = True; active_count -= 1
```

Also fix the progress print (line 377) to use a real counter, and ensure total
unique `episode_id` per partner == `episodes`.

Test (`tests/test_ce_sampler_batched.py`, execution-gated):
`rows = collect_option_replay_batched(... episodes=5, batch_size=3 ...)`;
assert per-partner `len({r.episode_id for r in rows}) == 5`.

Proposal-safety: makes batched path produce the SAME episode budget the
sequential path already produces; corrects CE statistical weighting.

NOTE: the committed CE artifact was built by the SEQUENTIAL `collect_option_replay`
(run_ce_pipeline.py:48), so this bug did NOT affect run7. It must be fixed before
the batched path is used for any formal CE.

---

## Problem 6 — `run_ce_pipeline.py` hardcodes wrong reward scale (CE/train objective mismatch)
Severity: HIGH. File: `scripts/run_ce_pipeline.py:59-60`.

Verified root cause: `cost_coef=1.0, shaped_reward_coef=0.0` hardcoded, while
`configs/ocv2_step4.yaml` training uses `cost_coef=0.02, shaped_reward_coef=1.0`.
CE local returns are cost-dominated; training TD is shaped-reward-driven.

Verified consequence (NEW, important): remote `outputs/p0_verify/graph.json`
metadata records only `{graph_variant, eta, max_factors, full_max_factors}` — NO
reward scale. The committed `ce_refined.npy` (Jun 26 00:41) was produced by this
script with the buggy scale. **The in-flight run7 training is built on a CE graph
whose factor selection used the wrong objective.** Therefore run7 verification
gates are on a questionable graph.

Fix (script):
1. Load `configs/ocv2_step4.yaml`; read `cost_coef`, `shaped_reward_coef`,
   `cost_per_step` from `training`.
2. Pass them into `collect_option_replay(...)`.
3. Write them + `reward_scale_source: "config.training"` into the saved replay
   metadata AND `graph.json` metadata.

Execution consequence (HANDOFF, gated): after the script fix, Step 2-3 CE must be
REGENERATED with config-consistent scale, and Step 4 RE-RUN on the regenerated
graph for the verification to be valid. This is the correct (non-defensive)
resolution. Flag for user authorization; do not silently keep the stale graph.

Optional: `ce_sampler.collect` CLI defaults → `--cost_coef 0.02
--shaped_reward_coef 1.0`, or require a `--config` for formal CE.

---

## Problem 7 — coverage gate prints FAIL but still saves the formal graph
Severity: HIGH (over-defensive script behavior). File:
`scripts/run_ce_pipeline.py:74-80, 143-151`.

Verified root cause: both `replay_coverage_gate` (74-79) and
`validate_task_stage_coverage` (143-148) are caught-and-printed; `graph.json` is
saved unconditionally at line 150-151 regardless of gate outcome.

Fix: make the gate blocking.

```python
import argparse
# argparse: --require_task_stage_coverage (default True), --allow_incomplete_graph (default False)

coverage_ok = True
try:
    validate_task_stage_coverage(graph)
    print("PASS: Task stage coverage validated")
except RuntimeError as exc:
    coverage_ok = False
    print(f"FAIL: {exc}")

if coverage_ok:
    graph_path = output_dir / "graph.json"
    graph_path.write_text(json.dumps(graph.to_json_dict(), indent=2), encoding="utf-8")
else:
    if args.allow_incomplete_graph:
        debug_path = output_dir / "debug_incomplete_graph.json"
        d = graph.to_json_dict(); d.setdefault("metadata", {}).update(
            {"formal_graph": False, "coverage_gate": "failed"})
        debug_path.write_text(json.dumps(d, indent=2), encoding="utf-8")
        print(f"Saved DEBUG-ONLY graph to {debug_path}")
    sys.exit(1)  # non-zero; do NOT write formal graph.json
```

Apply the same blocking semantics to the earlier `replay_coverage_gate`.
Proposal-safety: this REMOVES over-defensive logging-only behavior (the user's
explicit anti-pattern), turning a logged warning into a real gate.

---

## Problem 8 — configs hardcode `layout: cramped_room` (smoke, not formal benchmark)
Severity: MEDIUM. Files: `configs/ocv2_debug.yaml:1`, `configs/ocv2_step4.yaml:1`.

Verified: only two configs exist, both `cramped_room`. Proposal preflight layout
set (plan lines 1470-1472, 1533-1536): `test_time_simple, test_time_wide,
grounded_coord_simple, grounded_coord_ring, cramped_room_v2`.

Fix (ADDITIVE — do NOT mutate the running pipeline's config):
1. Keep cramped_room as the smoke config (optionally copy to
   `ocv2_smoke_cramped_room.yaml`); the running run7 depends on `ocv2_step4.yaml`,
   leave it intact until verification completes.
2. Add `ocv2_main.yaml` with `layouts.candidates:` = the proposal preflight set and
   NO single hardcoded `layout`.
3. Formal scripts must select a preflight-ACCEPTED layout (via
   `layout_diagnostics.accept_layout`), not default to cramped_room.

Scope note: this is a FORMAL-benchmark config, consumed only by the formal
milestone (M-series), which is execution-gated. It does not alter run7.

---

## Problem 9 — `run_step4_microtrain.py` is a verification matrix, not the formal V4 matrix
Severity: MEDIUM. File: `scripts/run_step4_microtrain.py`.

Verified gaps vs proposal (plan lines 1175-1182): missing `shuffled_routes`,
`random_same_size`, `minus_critical`; uses debug `minus_high_ce`/`overcomplete`;
trains every method on every variant (wasteful); gates are debug gates
(full > overcomplete/minus_high_ce), not formal causality gates.

Critical context: run_step4 is the APPROVED verification micro-train (cramped_room,
debug variants), and the previously-removed variants failed for real reasons
(`minus_critical` needs validation criticality scores — graph_builder.py raises;
`random_same_size` can miss required options and fail coverage). Rewriting
run_step4 now would (a) break run7 and (b) re-trigger those failures.

Fix (ADDITIVE formal matrix; defer execution to formal milestone):
1. Do NOT mutate run_step4 verification gates while run7 is in flight.
2. Add a SEPARATE formal matrix (new script `run_formal_matrix.py` or a
   `--mode formal` flag) with two explicit blocks:
   - Method comparison: {base_only, flat_factor, global_gru, partner_id_q,
     aris_bellman} × full_support only (skip graph variants for non-belief
     methods — fixes the wasteful all-variants training).
   - Graph causality: aris_bellman × {full_support, shuffled_routes,
     shuffled_relevance, random_same_size, minus_critical, overcomplete}.
3. Prerequisites that must land FIRST (separate, scoped items — not silent
   stubs). NOTE (verified): the BUILDERS already exist —
   `graph_builder.shuffled_routes_graph` (line 470) and `random_same_size_graph`
   (line 416); `minus_critical_graph` (line 351) exists but raises without
   `criticality_scores` (has a `ce_highest_debug` fallback at line 411). So the
   real missing work is:
   - Generate VALIDATION criticality scores (plan line 1185, validation return
     drop) so `minus_critical` uses real scores, not the debug fallback.
   - Make `random_same_size` coverage-safe (current sampler can miss required
     options and fail the task-stage gate) — a real fix, not a defensive skip.
   - Formal wiring + validation of `shuffled_routes` into the matrix.
4. Replace debug gates with formal V4 gates (method superiority at full_support;
   causal degradation for shuffled_routes/shuffled_relevance/random_same_size vs
   full_support; minus_critical drop). 

Scope note: formal matrix is execution-gated (multi-variant, possibly multi-layout)
and belongs to the formal milestone, AFTER the verification stage and the
problem-6 CE regeneration. This item is design + scaffolding now; execution later.

---

## Cross-cutting consequences & sequencing

1. Problems 1,2,3 change event semantics → any coverage/CE built before the fix is
   stale. Combined with Problem 6, the formal CE must be regenerated once 1-3,5,6
   land. (Execution-gated handoff.)
2. Problem 6 specifically implies run7's CE graph is objective-inconsistent →
   recommend: finish current run7 for a baseline reading, but treat its gates as
   provisional; regenerate CE + re-run Step 4 after fixes for the authoritative
   result.
3. Problems 1-7 are static code edits (safe now). Problems 8-9 are formal-milestone
   design/scaffolding; their execution stays gated.
4. New tests (problems 1,3,5) are written as code but RUNNING them is execution-
   gated (local-exec constraint) → handoff or remote-authorized run.

## Codex review tightenings (accepted — verdict: PASS on all 9)

T1. P6 metadata becomes a HARD GATE, not just a record: `train_aris` / `evaluate_aris`
    must REJECT a graph whose metadata is missing or mismatched on `cost_coef`,
    `shaped_reward_coef`, `cost_per_step`, `layout`, and an added
    `event_semantics_version` (bumped when P1-P3 land). This converts the
    objective-consistency check into an enforced precondition (not over-defensive:
    a mismatch is a genuine correctness violation).

T2. CE output paths must be VERSIONED/explicit. `run_ce_pipeline.py` hardcodes
    `outputs/p0_verify`, which is dangerous around the in-flight run7. New CE runs
    write to a new versioned path (e.g. `outputs/p1_verify_<scale>/`); never clobber
    the run7 graph in place.

T3. P8 formal multi-layout must NOT reuse a single stale `graph.ce_path` across
    layouts. Require per-layout CE/replay + strict layout-metadata checks at load.

T4. P9 builders already exist (see above); the gating work is criticality-score
    generation + coverage-safe random control + formal wiring/gates.

Implementation order (Codex-recommended):
1) Freeze run7 surface (no overwrite of remote `outputs/p0_verify/*`, no mutation of
   the running script/config). 2) P1-P3 + tests. 3) P4. 4) P5. 5) P6+P7+T1+T2
   together. 6) [after exec authorization] regenerate CE → new versioned path →
   re-run Step 4. 7) P8 scaffolding (+T3). 8) P9 (criticality scores + coverage-safe
   random + wiring), then formal matrix.

## T1 GAP CLOSURE (post-implementation verification finding)

Problem: `_enforce_graph_objective_metadata` (train_aris.py, runs after `_build_graph`)
requires reward-scale metadata ON `graph.metadata`. The `graph_path` flow carries it
(stamped `graph.json`), but the `ce_path` flow used by `run_step4`/`ocv2_step4.yaml`
builds each variant via `build_graph_variant` from the RAW `ce_refined.npy`, which has
no metadata. Result: after CE regeneration, a ce_path multi-variant Step 4 re-run would
raise at the gate for every variant even though the CE is correct.

Root cause: reward-scale provenance is not propagated from CE generation to the
ce_path-built variant. Verified: `build_graph_variant` metadata = {graph_variant, eta,
max_factors, full_max_factors} only (no reward scale).

Fix (CE sidecar — minimal, reuses existing enforcement; no change to the gate logic):

Producer — `scripts/run_ce_pipeline.py`: immediately after
`np.save(refined_path, refined)` (line ~155), write a sidecar next to the matrix:
`sidecar = output_dir / "ce_refined.meta.json"` containing the already-built
`reward_metadata` dict (layout, cost_coef, cost_per_step, shaped_reward_coef,
reward_scale_source, event_semantics_version). This point is only reached after the
replay coverage gate passed, so a sidecar implies a valid CE matrix.

Consumer — `train_aris._build_graph` ce_path branch (after the formal_experiment
metadata stamp, lines ~542-546, before the zero-factor check): derive
`sidecar = Path(ce_path).parent / (Path(ce_path).stem + ".meta.json")`. If it exists,
load it and merge into `graph.metadata` (`{**(graph.metadata or {}), **sidecar_meta}`).
If absent, do nothing → the existing `_enforce_graph_objective_metadata` raises with the
"regenerate CE" message (correct for legacy `outputs/p0_verify/ce_refined.npy`).

Why this is correct and not over-defensive:
- Enforcement stays the single source of truth; we only PROPAGATE provenance.
- Legacy CE (no sidecar) still correctly fails the gate → forces regeneration.
- Regenerated CE (sidecar matches config) passes for ALL variants → multi-variant
  ablation re-run works via ce_path.
- No silent fallback: a missing sidecar does not fabricate metadata.

Guard rails: do NOT modify `run_step4_microtrain.py`; do NOT touch `outputs/`; do NOT
run anything. No edits to `_enforce_graph_objective_metadata` / `_graph_objective_metadata_status`.

Test (write only, execution-gated): unit test that a graph whose metadata is merged
from a sidecar dict matching a config passes `_graph_objective_metadata_status[...
reward_scale_verified] is True`, and a graph without it is False.

## P2 / P6 FULL CLOSURE (remote-verified facts)

Remote env verified at
/apps/users/cxw/Document/CodeSpace/Selfs/TG-SSA/external/JaxMARL/jaxmarl/environments/overcooked_v2/overcooked.py:
- `step_env` info dict = ONLY `{"shaped_reward": ...}` (overcooked.py:210). There is NO
  `correct_delivery`/`successful_delivery` info key → the planned `_delivery_from_info()`
  is not implementable, and the existing `_explicit_wrong_delivery(info)` (reads
  `info["wrong_delivery"]`) NEVER fires → `correct_delivery` currently == `delivery_event`
  (latent bug: wrong deliveries counted as correct).
- The AUTHORITATIVE correct-delivery signal lives on STATE: `state.new_correct_delivery`
  (set during step, overcooked.py:1181; consumed in obs at 640-649).

### P2 fix (event_extractor.py)
Reorder + use the authoritative state signal:
1. heuristic_delivery = (ego `_delivered_soup` AND `_delivery_target(prev_state,0)`) OR
   (partner `_delivered_soup` AND `_delivery_target(prev_state,1)`)  [already implemented]
2. `correct_delivery = bool(np.asarray(next_state.new_correct_delivery).item())`  [authoritative]
3. `delivery_event = bool(heuristic_delivery or correct_delivery)`  [superset; never misses a
   correct delivery, still catches wrong-recipe drops on GOAL]
4. `wrong_delivery_event = bool(delivery_event and not correct_delivery)`
5. Remove the now-dead `_explicit_wrong_delivery` + its `info` usage (verify no other refs).
Semantics: delivery_event = any dish delivered to GOAL; correct_delivery = reward-earning
correct-recipe delivery (authoritative); wrong_delivery_event = delivered to GOAL but wrong.
Tests (extend tests/test_event_extractor.py + its `_state` stub to carry
`new_correct_delivery`): correct-recipe@GOAL → delivery_event True, correct_delivery True,
wrong_delivery_event False; wrong-recipe@GOAL (new_correct_delivery False) → delivery_event
True, correct_delivery False, wrong_delivery_event True; counter(WALL) drop → delivery_event
False.

### P6 fix (ce_sampler.py collect CLI)
Verified: `collect` defaults `--cost_coef 1.0 --shaped_reward_coef 0.0` (ce_sampler.py:1047-1049);
`_cmd_collect` passes them to the collector AND stamps them into the metadata sidecar; ce_sampler
does NOT import yaml. run_ce_pipeline does not use this CLI (imports collect_option_replay
directly), so changing the CLI is safe.
1. `import yaml`; `from experiments.overcooked_v2.event_extractor import EVENT_SEMANTICS_VERSION`.
2. Add `--config` to the collect subparser; change `--cost_coef/--shaped_reward_coef/--cost_per_step`
   defaults to None.
3. In `_cmd_collect`, resolve reward scale BEFORE collecting:
   - if `args.config`: load yaml, read `training.{cost_coef,shaped_reward_coef,cost_per_step}`;
     reward_scale_source = "config.training".
   - else: if any of the three is None → raise ValueError("pass --config or all of
     --cost_coef/--shaped_reward_coef/--cost_per_step for formal CE"); else use explicit values,
     reward_scale_source = "cli_explicit".
4. Use the resolved values in collect_kwargs and in the metadata sidecar, plus add
   `reward_scale_source` and `event_semantics_version` to that metadata. No silent old-scale path.

Guard rails: do NOT touch run_step4_microtrain.py / outputs/; do NOT run anything; no
over-defensive fallbacks (direct `next_state.new_correct_delivery` access — it is confirmed
present; a future env lacking it should error, not be masked).

## Anti-pattern self-check
- No fix weakens a real signal (no regressive changes).
- Problem 7 explicitly REMOVES over-defensive logging-only behavior.
- No silent fallbacks/stubs introduced; missing builders (P9) are called out as
  real work, not masked.
