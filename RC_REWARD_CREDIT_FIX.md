# RC Reward-Credit Fix — Static Handoff

**Branch:** `rc-rootcause-fix` · **Date:** 2026-06-29 · **Status:** static edits applied, NOT executed.
**Source:** `~/Downloads/Review&Plan.md` (root-cause analysis + minimal fix plan).

> Execution boundary (OPERATING_CONSTRAINTS.md): these are STATIC edits only. CE
> regeneration, training, eval, and the regression test are **execution-gated** —
> run remote per [CUSTOMER.md](CUSTOMER.md) or after an explicit local relaxation.
> Next gated steps: (1) cross-model **codex** review of this diff (CODE_REVIEW=true),
> (2) remote CE-regen → micro-train → held-out eval.

---

## 1. Root cause (confirmed against HEAD)

OvercookedV2's **+delivery sparse reward is shared** (`rewards["agent_0"]` is the
team reward). `train_aris._training_reward`, `ce_sampler` (both rollout paths),
and `evaluate_aris` all credited that shared reward to the **ego** option's TD
target — so the ego was paid the delivery reward when the **partner** served.
On the train/held-out split where train partners finish recipes, "fetch / deliver
ingredient / wait while the partner plates & serves" became a high-return ego
option; on held-out partners that do not bail it out, the same ego farms
ingredients to the pot and never completes (`ingredient_delivered_to_pot=60`,
`ego_delivery_count=0`, `completion_rate=0`).

Every code location cited in the plan was re-verified at HEAD (the plan's
`code/...` paths map to repo-root `experiments/...`; line numbers had drifted).
The actor-specific event fields the fix needs (`ego_correct_delivery`,
`ego_delivery_event`, `ego_wrong_delivery_event`, …) already exist in
`event_extractor.OCV2Event` (EVENT_SEMANTICS_VERSION=2).

## 2. What changed (single source of truth + 4 producer sites)

**New helper — `experiments/overcooked_v2/event_extractor.py`:**
- `actor_sparse_reward(team_sparse, event, *, mode, ego_delivery_reward, ego_wrong_delivery_penalty)`
- `sparse_credit_params(training_cfg)` → resolves kwargs from `config.training`; absent key ⇒ legacy `team`.

Modes:
| mode | behaviour |
|---|---|
| `team` (default) | legacy passthrough — returns the shared team sparse reward. Reproduces pre-fix runs **exactly** (no-op extraction). |
| `ego_delivery` (**asymm default**) | credit the **real env** sparse reward to the ego only on steps where the ego is the **sole** deliverer (`ego_delivery_event and not partner_delivery_event`): correct ⇒ +reward, wrong ⇒ −penalty (`env.negative_rewards`); a partner delivery ⇒ 0. No inferred constant; stays on the `value_bound.vmax` scale. The sole-deliverer gate closes a residual leak on multi-goal layouts (asymm has 2 goal tiles) where a coincident ego+partner delivery would otherwise sum into `team_sparse`. |
| `ego_correct_delivery` | the plan's literal mode: explicit `ego_delivery_reward` (20.0) / `ego_wrong_delivery_penalty` (−20.0) constants. |

**Deviation from the plan (deliberate, documented):** the plan proposed the
constant-based `ego_correct_delivery` (20/−20 "inferred from replay"). The asymm
config defaults to **`ego_delivery`** instead — it credits the environment's own
delivery reward (no guessed constant), which the project's own discipline prefers
("do not invent unverified constants", FIX_PLAN_9ISSUES P2). The literal mode is
fully implemented and selectable by one config line if you want the plan verbatim.

**Wiring (all route through the helper):**
- `train_aris._training_reward(step, config, agent_key, event)` — now takes the event; used by **both** train (`_execute_option`) and eval (`evaluate_aris._execute_eval_option`, which imports it).
- `ce_sampler._rollout_option` (sequential) and `collect_option_replay_batched` — new `credit_params` kwarg (default `None` ⇒ team), applied at the per-step accumulation.
- CE consumers (`compute_local_returns`, `refine_interventional_ce`, `layout_diagnostics._partner_return_stats`, `aris_td_loss`, `_transition_training_return`) read `reward_sum` and **propagate automatically** — no change needed.

**Objective-consistency gate (anti-stale-graph, T1 pattern):**
- `run_ce_pipeline.py` and `ce_sampler._cmd_collect` record `sparse_credit` in the CE reward metadata (→ `graph.json` metadata **and** `ce_refined.meta.json` sidecar).
- `train_aris._expected_graph_objective_metadata` adds `sparse_credit`; `_graph_objective_metadata_status` treats an **absent** key as legacy `team` (so pre-fix graphs still load for team runs) but **rejects** an `ego_delivery` config against a stale team/absent graph — forcing CE regeneration with a clear "regenerate CE" error.

**Ego-terminal-aware checkpoint selection + free-rider guard (`training.require_ego_delivery_selection`, asymm = true):**
- `_run_greedy_validation` returns `ego_correct_delivery_count`, **`ego_sole_correct_delivery_count`** (leak-proof), `partner_correct_delivery_count`, `completion_rate`.
- **Selection is gated, not just guarded:** under the flag a greedy checkpoint is *eligible* to become "best" only if `ego_sole_correct_delivery_count > 0`, so a higher-return but non-serving (shaped-farming / free-riding) checkpoint **cannot be selected**. Without the flag, legacy `mean_return` selection is preserved (non-regressive).
- The verdict (`pass`/`fail`/`not_required`/`not_evaluated`) is computed once after training and written to `metrics.checkpoint_selection.free_rider_guard`; a `fail` (greedy ran but no serving checkpoint) raises **after** metrics+checkpoints persist, with a **split diagnosis**: `free_riding_partner_serves_while_ego_idle` (partner served, `max_partner_correct_delivery_seen>0`) vs `no_terminal_stage_shaped_farming_or_stall`.

**Diagnostic label fix:** dynamic-budget exhaustion relabelled `running` → `budget_exhausted` in train/eval/**and batched** CE rollout loops (`option_success` classifies both identically — purely cosmetic for option stats).

**Round-1 adversarial-review hardening:**
- *Guard wiring:* `_normalize_training_stability_config` **raises** if `require_ego_delivery_selection: true` but greedy validation is disabled — the opted-in guard can no longer silently no-op (reads the effective normalized condition).
- *Coincident-delivery leak:* both ego modes credit only when the ego is the **sole** deliverer that step (no partner-share re-leak on multi-goal layouts).

**Round-2 review-response (maps to reviewer findings A–F):**
- **A** — asymm config `graph.ce_path`/`replay_path` repointed to the new versioned `outputs/asymm_ce_egocredit/`, so the stale team CE cannot be silently consumed; regen still required (execution-gated).
- **B** — selection made ego-terminal-aware (above); the guard now *selects* an ego-serving checkpoint rather than only aborting, and the failure message distinguishes free-riding from shaped-farming.
- **C** — added the actor-local `OCV2Event.ego_sole_correct_delivery` (`ego_delivery_event and not partner_delivery_event and correct_delivery`); the guard/metrics use it instead of the shared-`correct_delivery`-polluted `ego_correct_delivery`.
- **D** — CE/objective metadata now record & enforce `ego_delivery_reward`/`ego_wrong_delivery_penalty` **only** in `ego_correct_delivery` mode (no false reject for `team`/`ego_delivery`).
- **E** — batched CE now caps option duration at the dynamic `option_budget` and labels `budget_exhausted`, matching sequential/train/eval (off the formal critical path — `run_ce_pipeline` uses the sequential collector — but now consistent).
- **F** — `actor_sparse_reward`/`sparse_credit_params` moved to the dependency-light `sparse_credit.py` (no JaxMARL); `event_extractor` re-exports them; the regression test imports the module directly so it **runs** in a JaxMARL-free env (with an `importorskip`-guarded re-export identity check).

**Regression test (write-only):** `tests/test_actor_sparse_credit.py` pins the decisive failure mode — a partner-delivery step gives the ego **0** credit in both ego modes, full credit on an ego sole-delivery, `team` is exact passthrough, and **coincident ego+partner deliveries credit 0** (no re-leak). Imports the JaxMARL-free `sparse_credit` module.

### Files touched
```
experiments/overcooked_v2/sparse_credit.py        (NEW — JaxMARL-free credit helper, single source)
experiments/overcooked_v2/event_extractor.py      (re-export helper; + ego_sole_correct_delivery field)
experiments/overcooked_v2/train_aris.py           (reward, gate+constants, ego-aware selection+guard, label, config validation)
experiments/overcooked_v2/evaluate_aris.py        (reward call, label)
experiments/overcooked_v2/ce_sampler.py           (both rollout paths, metadata+constants, label, batched budget, sole field)
experiments/overcooked_v2/scripts/run_ce_pipeline.py (sparse_credit metadata + constants + collect)
experiments/overcooked_v2/configs/ocv2_step4_asymm.yaml (sparse_credit, guard, ego-credit CE path)
tests/test_actor_sparse_credit.py                 (NEW — write-only regression)
```
> NOTE: `ocv2_step4_asymm.yaml` also carries **pre-existing uncommitted hunks** (`replay_path`,
> `diagnostics:`) and `scripts/rc2_reachability.py` / `scripts/rc2b_layout_scan.py` are
> **pre-existing edits unrelated to this fix** — keep them out of the fix commits (see §5).

## 3. Preservation checks (substrate must not solve away the phenomenon)

- **Phase A substrate certificate unchanged:** preflight / `layout_diagnostics`
  CE collection is left on `team` credit (no `credit_params`), so the
  substrate-admissibility certificate is not perturbed by this fix.
- **The fix only corrects credit:** it does not hand the ego the terminal policy
  nor make the task trivially solvable. Base_only must still *learn* plate/serve;
  the ARIS-vs-baseline scientific question is preserved.
- **`team` mode is bit-identical** to pre-fix behaviour (no-op extraction), so any
  pre-fix result can be reproduced.

## 4. Execution-gated verification plan (run remote per CUSTOMER.md)

1. **Regression test:** `pytest -q tests/test_actor_sparse_credit.py` — now runs JaxMARL-free (the `sparse_credit` import has no env deps); the re-export identity case skips without JaxMARL.
2. **Regenerate asymm CE** with the updated config to the new versioned path (config already points there; never clobber `outputs/asymm_ce` — T2):
   `python experiments/overcooked_v2/scripts/run_ce_pipeline.py --config .../ocv2_step4_asymm.yaml --output_dir outputs/asymm_ce_egocredit/ …`
   Confirm `graph.json`/`ce_refined.meta.json` metadata has `sparse_credit: ego_delivery`.
3. **Smoke micro-train** (cheap: ~500–1000 updates on the regenerated graph). PASS iff: greedy validation shows **`ego_sole_correct_delivery_count > 0`** (the leak-proof count the selection gate uses); `plate_soup`/`serve_soup` appear in option stats; `checkpoint_selection.free_rider_guard == "pass"`; objective gate did not reject (proves the regenerated graph's `sparse_credit` matches). A `fail` will carry `free_rider_diagnosis` (free-riding vs shaped-farming) to triage.
4. **Held-out eval** (`bottleneck-yield,flexible-balanced`, ~10 episodes): `ingredient_delivered_to_pot` should no longer be 60/plate=0/serve=0; `completion_rate` should move above the random reference.
5. **Stage-gated metric / Q-audit** (plan §5.3–5.4) — optional follow-up.

## 4b. Remote verification RESULTS (2026-06-29, zsc-customer)

Ran the full chain on the remote (`CPR_REPO`, JAX on CPU + torch GPU; see [[reference-remote-jax-cpu]]).
Smoke config: `ocv2_step4_asymm.yaml` as-is (1000 updates, aris_bellman, seed 0, full_support).

**Verified — the primary fix works (free-riding eliminated):**
- Regression test `tests/test_actor_sparse_credit.py`: **8/8 pass** in the remote venv.
- CE regen → `outputs/asymm_ce_egocredit/`: coverage gate **passed**; `graph.json` + `ce_refined.meta.json` record **`sparse_credit: ego_delivery`**. Old `outputs/asymm_ce` has no `sparse_credit` key → correctly rejectable.
- Objective gate **accepted** the regenerated ego-credit graph (only a benign provenance-hash warning).
- **Decisive credit-fix evidence** (vs the original failure run):

  | signal | OLD (buggy) | NOW (fixed) |
  |---|---|---|
  | training partner serves / ego serves | 25 / 0 | 9 / 0 |
  | **greedy `best_greedy_return`** | **+33.99 (free-rider selected)** | **None; greedy `mean_return = −0.84`** |
  | free-rider guard | n/a | **fail → `no_terminal_stage`** (`max_partner_correct_delivery_seen=0`) |
  | held-out completion / ego / partner | 0 / 0 / 0 | 0 / 0 / 0 |
  | held-out `mean_return` | +7.74 (farming, `ingredient→pot=60`) | **−0.84 (wait/stall, `wait_duration_after_arrival=200`)** |

  The 9 partner serves in training credited the ego **0** (old: ≈+20 each → +33.99 free-rider checkpoint). Greedy return went **negative** — the free-riding optimum is gone. The guard correctly diagnosed `no_terminal_stage` (not free-riding).

**Not yet achieved (the now-binding constraint, research not bug):** the ego did **not** learn to plate/serve in the 1000-update smoke — greedy policy is a degenerate **wait/stall** (held-out all options end `wait_duration_after_arrival`, `mean_return=−0.84`, completion 0). This is the plan's **contributing cause #2** (dense `shaped_reward_coef=1.0` / low `cost_coef=0.02`) and/or insufficient budget/exploration — surfaced correctly by the guard, **not** a defect in the fix.

**Recommended next step:** address contributing cause #2 — re-tune shaping/cost (and/or a terminal-stage incentive), increase training budget + exploration, then re-verify (target: greedy `ego_sole_correct_delivery_count > 0`, held-out completion > 0). This is a Type-B direction → codex + human acceptance before any decisive claim.

Remote artifacts: `outputs/asymm_ce_egocredit/`, `results_rcfix/asymm_egocredit/.../{metrics.json,checkpoint.pt}`, `results_rcfix/asymm_egocredit/eval_heldout.json`, logs in `logs_rcfix/`.

## 5. Suggested attributable commits (do not push; user-gated)

Memory feedback: *incremental attributable commits* — one mechanism per commit.
```
1. refactor: actor_sparse_reward single-source helper in dependency-light sparse_credit.py
   + thread team-default through train/CE/eval + objective-metadata gate
   (NO-OP: team mode == pre-fix behaviour; helper re-exported from event_extractor)
2. fix(RC): asymm config sparse_credit=ego_delivery + ego-credit CE path (the fix)
3. feat: ego_sole_correct_delivery event field (leak-proof actor-local correct delivery)
4. feat: ego-terminal-aware checkpoint selection + free-rider guard (+ config validation,
   split free-riding/shaped-farming diagnosis)
5. fix: relabel dynamic-budget option termination running->budget_exhausted (train/eval/CE,
   incl. batched-CE option-budget cap)
6. feat: enforce ego_correct_delivery reward constants in objective metadata
7. test: actor sparse-credit regression (partner serve / coincident => ego 0 credit) [write-only]
```
Golden-trace verification of commit 1's no-op claim is execution-gated (run
`scripts/executor_golden_trace.py` remote); the no-op-ness is otherwise verifiable
by inspection (`actor_sparse_reward(x, event, mode="team") == x`).

## 6. Deferred (scoped follow-ups, NOT in this patch)

- **TD next-action mask** (plan secondary): store `valid_options(next_state,0)` in
  `OptionTransition` (`src/aris_bellman/specs.py`) and pass it as `option_mask_next`
  to `aris_td_loss` instead of the static graph mask (`train_aris._graph_tensors`).
  Touches the dataclass → replay stacking → TD batch; isolate as its own commit.
- **Dense-shaping / cost re-tune** (plan contributing cause #2): with terminal
  credit corrected, revisit whether `shaped_reward_coef=1.0` / `cost_coef=0.02`
  still let "farm ingredients" carry positive return. Defer until step 3/4 numbers exist;
  the guard's `no_terminal_stage_shaped_farming_or_stall` diagnosis flags this case.
- **Actor-local correctness from the env** (review C, full fix): `ego_sole_correct_delivery`
  conservatively zeros on coincident deliveries because the env exposes only a shared
  `new_correct_delivery` bool. A true per-actor correctness signal would need an
  env/event-layer change; deferred as it only affects rare same-step double deliveries.
- **Train-only non-serving validation partner** (plan #3): **PROMOTED to the next required step.**
  The 2-arm re-verify (see `RESULTS_SUMMARY.md`) shows neither longer training+exploration (Arm A)
  nor lower shaping (Arm B) makes the ego serve — it never serves once in 5000 updates because the
  `dish-server`/`server-left` train partners monopolize the terminal stage. The ego needs a
  train condition where it must serve itself. The guard detects this (`no_terminal_stage`) but
  cannot fix it; the non-serving partner is the fix.
- **`partner_id_q` 6-ID-embed-vs-4-train mismatch** (plan): a baseline limitation, not the ARIS defect.

> Reviewer findings A–F were all addressed in this patch (see §2 "Round-2 review-response"); E
> (batched-CE budget) is fixed but off the formal critical path since `run_ce_pipeline` uses the
> sequential collector.
