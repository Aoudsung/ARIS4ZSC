# Implementation spec v2 — role_conditioned_v1 ARIS-fitness fixes

Executor: codex. Reviewer: Claude (post-execution audit).

This spec addresses 5 root-cause failures diagnosed in v1's role_conditioned_v1 run:
ARIS belief_swap_delta ≈ 0 on the claim partner despite `observed_dist_rate=1.0`;
graph relevance holes on task-critical options; value_bound only on ARIS; eval reward
missing ContributionLedger; role_replay chain asymmetric difficulty.

Constraint recap (unchanged from v1): single TD loss; action selection by
`argmax Q(s,b,ω)`; no probe selector; no imitation loss; no explicit fallback controller.
`role_contrib_team` credit is a training-time signal (partner identity known at
training only); eval never sees the reward, only greedy Q.

Repo: `/Users/aoudsung/Documents/ARIS4ZSC`. Do NOT run any training/CE/eval — code only.

---

## 0. Design invariants

- **Tier 1** = bug/consistency fixes (must apply before comparison is trustworthy).
- **Tier 2** = mechanism-relevant fixes (address the diagnosed root causes).
- Every new config key is read via `cfg.get(..., DEFAULT)`. Absent → v1 behavior.
- Backward-compat: `standard7` runs must still work byte-identically.
- Ablation toggles: each Tier 2 fix has a config flag so we can attribute credit
  post-hoc.

---

## 1. Tier-1-A. Consistent value bound across methods (v1 Fix 4)

**Problem**: `value_bound` (vmax=30, base_bound=15, adv_bound=15) is threaded only
into `ArisBellmanQNetwork`. Other methods have unlimited Q output. On role_v1 the
observed greedy return reaches 50-70 → ARIS's Q head is bounded well below the
empirical value range.

**Fix**: apply the same `value_bound` config to ALL Q-network branches.

**File**: `experiments/overcooked_v2/train_aris.py`

Locate `_build_q_network(method, obs_dim, graph, config)` (around line 1150-1200).
For every branch (`aris_bellman`, `base_only`, `flat_factor`, `global_gru`,
`partner_id_q`), thread the same `value_bound / vmax / base_bound / adv_bound`
kwargs, OR add a NEW config key `apply_value_bound_to_all_methods: bool` that
gates whether the bound is applied to non-ARIS methods.

Recommended: gate behind a config flag so v1 comparability is preserved when
needed.

```yaml
# config addition
training:
  value_bound:
    enabled: true
    apply_to_all_methods: true   # v2 default. If false, v1 behavior (ARIS-only).
    vmax: 30.0
    base_bound: 15.0
    adv_bound: 15.0
```

For methods without a native "base + advantage" split (`base_only`, `global_gru`,
`partner_id_q`, `flat_factor`), interpret the bound as a hard tanh clip on the
final Q output at magnitude `vmax`:

```python
q_out = vmax * torch.tanh(q_raw / vmax)
```

Only apply when `training.value_bound.enabled=True` AND
`training.value_bound.apply_to_all_methods=True`.

Deliverable: every non-ARIS Q network reads and applies the same clip when
enabled. Verify via `python -c "import torch; ..."` unit-style test in check §12.

---

## 2. Tier-1-B. Eval reward parity with contrib_team (v1 Fix 5)

**Problem**: `evaluate_aris.py:542` calls `_training_reward(step, ctx.config, "agent_0", event)`
without `ego_contributed`, so under `contrib_team` partner deliveries return 0 in
eval reward → partner-serving methods' `mean_return` is understated. The role
outcome counts (ego_delivery_count, partner_delivery_count, role_match_rate) are
NOT affected because they read env-event fields directly. But return-based
metrics are corrupted.

**Fix**: thread `ContributionLedger` through `_execute_eval_option` mirroring
`train_aris._execute_option`. Same signature pattern, same per-episode reset.

**File**: `experiments/overcooked_v2/evaluate_aris.py`

### 2.1 Import at top
```python
from experiments.overcooked_v2.reward_design import ContributionLedger
```

### 2.2 Extend `_execute_eval_option` signature
Add ONE keyword-only param (default None), same pattern as train:
```python
def _execute_eval_option(
    ctx, env, obs, partner, router, graph, evidence_buffer,
    option_id, collect_diagnostics, allow_diag_skip, rng,
    *,
    contribution_ledger: ContributionLedger | None = None,
) -> tuple[float, bool, dict[str, np.ndarray], dict[str, Any]]:
```

### 2.3 Inside `_execute_eval_option` primitive-step loop
BEFORE the reward line, update ledger + query:
```python
        if contribution_ledger is not None:
            contribution_ledger.update(event, ego_option_kind=str(opt.kind))
        ego_contributed = False
        if contribution_ledger is not None:
            ego_contributed = contribution_ledger.query_and_reset_on_delivery(event)
        reward_sum += _training_reward(
            step, ctx.config, "agent_0", event,
            ego_contributed=ego_contributed,
        )
```

### 2.4 Create ledger per episode in `_run_episode`
Where `_run_episode` (around line 340-400) resets env at episode start:
```python
    contribution_ledger = ContributionLedger.from_config(
        ctx.config.get("training", {})
    )
```
And pass it into every `_execute_eval_option` call within that episode.

Deliverable: `contrib_team` runs' eval reward now includes partner-delivery credit
when ego contributed. `role_match_rate` unchanged.

---

## 3. Tier-2-A. Per-id + broader graph coverage (v1 Fix 2)

**Problem**: v1's `graph.json` shows `pick_plate id=7` and 11/12
`drop_item_to_counter` options have NO relevant factor. Config only asserted
`{pick_plate: min_factors: 1}` — enough to cover ONE pick_plate id, not all. ARIS's
factor-local relevance mask then blocks belief influence on those options entirely.

**Fix**: extend `required_option_id_coverage` in the config AND (if needed)
strengthen `_coverage_constrained_pairs` to enforce per-id for the listed kinds.

### 3.1 Config extension (immediate)

**File**: `configs/ocv2_step4_asymm_role_v1.yaml`

Replace the current `required_option_id_coverage`:
```yaml
graph:
  required_option_kind_coverage:
    pick_plate: {min_factors: 1}
    plate_soup: {min_factors: 1}
    serve_soup: {min_factors: 1}
    clear_interaction_cell: {min_factors: 1}
  required_option_id_coverage:
    pick_plate:                {min_per_valid_option: 1}
    plate_soup:                {min_per_valid_option: 1}
    serve_soup:                {min_per_valid_option: 1}
    clear_interaction_cell:    {min_per_valid_option: 1}
    deliver_ingredient_to_pot: {min_per_valid_option: 1}
```

Do NOT require per-id coverage for `drop_item_to_counter` (12 counters — would
crowd out CE fill). Instead the graph relies on per-kind coverage plus the CE
tail.

### 3.2 Verify `_coverage_constrained_pairs` already supports `min_per_valid_option`
for these kinds — it should (v1 already handled `serve_soup`). If not, extend
the loop in `graph_builder.py::_coverage_constrained_pairs` so any kind listed
under `required_option_id_coverage` gets per-id reservation before CE fill.

### 3.3 Failure behavior
If any required option id has NO CE candidate above eta, currently
`GraphCoverageError` is raised. Add a fallback: LOWER eta on a per-id basis to
find at least one candidate (walk down: 0.05 → 0.01 → 0.001 → 0.0). Record
`selected_by = "mandatory_per_option_id_low_eta"` with the actual eta used, so
provenance is auditable. This prevents build failures on rare-CE options while
keeping the strong CE candidates preferred.

Deliverable: new CE build for role_v1 has every task-critical option covered by
at least one factor. Verify by dumping `graph.json` and counting factors per
option after training.

---

## 4. Tier-2-B. role_contrib_team credit (v1 Fix 1)

**Problem**: `contrib_team` rewards `ego terminal delivery` and
`partner terminal delivery + ego contribution` equally. Under a claim partner
this means ARIS is not penalized for taking terminal; it just needs "some
positive reward path." ARIS's ego-terminal prior wins.

**Fix**: add a role-conditioned credit mode that scales down (not zero) ego
terminal delivery when the partner is a claim-role partner. This is a
training-time signal (partner_terminal_policy accessible at train time).

### 4.1 New mode in `sparse_credit.py`

Extend `SPARSE_CREDIT_MODES`:
```python
SPARSE_CREDIT_MODES = (
    "team", "ego_delivery", "ego_correct_delivery",
    "contrib_team", "role_contrib_team",
)
```

Extend `actor_sparse_reward` with TWO new kwarg-only params:
```python
def actor_sparse_reward(
    team_sparse: float,
    event: Any,
    *,
    mode: str = DEFAULT_SPARSE_CREDIT_MODE,
    ego_delivery_reward: float = 20.0,
    ego_wrong_delivery_penalty: float = -20.0,
    ego_contributed: bool = False,
    contrib_scale: float = 1.0,
    partner_terminal_policy: str | None = None,
    ego_terminal_penalty_under_claim: float = 0.3,  # scale factor, not sign
) -> float:
```

Add branch AFTER `contrib_team`:
```python
    if mode == "role_contrib_team":
        ego_sole_delivery = bool(getattr(event, "ego_delivery_event", False)) and not bool(
            getattr(event, "partner_delivery_event", False)
        )
        partner_delivery = bool(getattr(event, "partner_delivery_event", False))
        # yield partner: ego should serve. reward ego, zero partner.
        if partner_terminal_policy == "yield":
            if ego_sole_delivery:
                return float(team_sparse)
            return 0.0
        # claim partner: partner should serve; ego terminal scaled down; partner+contrib full.
        if partner_terminal_policy == "claim":
            if partner_delivery and bool(ego_contributed):
                return float(contrib_scale) * float(team_sparse)
            if ego_sole_delivery:
                return float(ego_terminal_penalty_under_claim) * float(team_sparse)
            return 0.0
        # unknown / None policy: fall back to contrib_team semantics.
        if ego_sole_delivery:
            return float(team_sparse)
        if partner_delivery and bool(ego_contributed):
            return float(contrib_scale) * float(team_sparse)
        return 0.0
```

Extend `sparse_credit_params` return dict:
```python
    "partner_terminal_policy": None,  # populated at call site
    "ego_terminal_penalty_under_claim": float(
        (cfg.get("role_contrib_team") or {}).get("ego_terminal_penalty_under_claim", 0.3)
    ),
```

### 4.2 Thread partner policy into `_training_reward`

**File**: `experiments/overcooked_v2/train_aris.py`

Extend `_training_reward` signature:
```python
def _training_reward(
    step, config, agent_key, event,
    *,
    ego_contributed: bool = False,
    partner_terminal_policy: str | None = None,
) -> float:
```

Inside, pass `partner_terminal_policy` through to `actor_sparse_reward`:
```python
    sparse = actor_sparse_reward(
        team_sparse,
        event,
        ego_contributed=ego_contributed,
        partner_terminal_policy=partner_terminal_policy,
        **{k: v for k, v in sparse_credit_params(config.get("training")).items()
           if k != "partner_terminal_policy"},
    )
```

### 4.3 Extract partner policy at option-execution site

In `_execute_option`, at function entry read partner protocol ONCE:
```python
    partner_terminal_policy = getattr(
        getattr(partner, "protocol", None), "terminal_policy", None
    )
```

Then in the `_training_reward` call inside the primitive loop:
```python
        reward_sum += _training_reward(
            step, config, "agent_0", event,
            ego_contributed=ego_contributed,
            partner_terminal_policy=partner_terminal_policy,
        )
```

Do the same in every training/replay/seed path (grep for `_training_reward(` and
add the kwarg; when partner not in scope, pass `None`).

### 4.4 Config
```yaml
training:
  sparse_credit: role_contrib_team
  contrib_team:
    contrib_scale: 1.0
  role_contrib_team:
    ego_terminal_penalty_under_claim: 0.3
```

### 4.5 Objective metadata gate
Add `role_contrib_team.ego_terminal_penalty_under_claim` to the expected
metadata when `mode == "role_contrib_team"`; extend
`_expected_graph_objective_metadata` in `train_aris.py:_expected_graph_objective_metadata`
following the `contrib_team` pattern.

Deliverable: under `role_contrib_team`, ego is not fully rewarded for serving
against a claim partner. All methods now feel the role signal directly through
reward. ARIS's argument then reduces to "does the mechanism help ON TOP OF this
signal?"

### 4.6 Ablation
This IS the toggle for Fix 1. `sparse_credit: contrib_team` vs `role_contrib_team`
gives the ablation. Both must be preserved.

---

## 5. Tier-2-C. Eval-parity ledger propagation to `_execute_eval_option`

Already covered in §2 (Tier-1-B). Ensures returns reflect the training objective.

---

## 6. Deferred (Tier 3) — role_context in ARIS Q (v1 Fix 3)

**Not in this spec.** Rationale: adding a global `role_context` to ARIS's base Q
dilutes the "purely factor-local" claim. If Tier 1 + Tier 2 close the ARIS gap,
Fix 3 is unnecessary. If not, the honest conclusion is that the factor-local
mechanism itself doesn't add value on this task, and Fix 3 is an admission
rather than a defense.

Postpone until we see v2 numbers.

---

## 7. Additional integrity checks

### 7.1 Verify `_training_reward` signature callers updated

After §4, grep every call site of `_training_reward(` in the repo and confirm
they either pass `partner_terminal_policy` (train paths where partner is in
scope) or explicitly pass `None` (contexts without partner ID). No silent
default when partner is available.

### 7.2 Verify `_execute_eval_option` callers pass the ledger

After §2, every call site of `_execute_eval_option(` MUST pass
`contribution_ledger=ledger` where a per-episode ledger is created at
`env.reset()`. Otherwise the eval reward parity is only partial.

### 7.3 Backward compat under `standard7`

Run partner-differentiation probe (spec v1 §12.12) on `standard7`. If the
result changes vs v1 (was 4/6 unique behaviors), a regression happened. Root
cause: probably `_terminal_policy_bonus` accidentally affects standard7 partners
(none of which have `terminal_policy` set except `terminal-yield`).

Confirm via:
```python
python -c "
from experiments.overcooked_v2.partner_pool import make_training_partners, TRAINING_PROTOCOLS
lib = ...  # OCV2OptionLibrary on asymm_advantages
partners = make_training_partners(lib, partner_set='standard7')
# ... print 60-step primitive sequences and assert 4/6 unique
"
```

### 7.4 Rerun v1's §12.11 assertions

The 8 role_v1 partner names/terminal_policy/curriculum_group sequences must be
byte-identical to v1 spec expected output. No re-ordering allowed.

---

## 8. Deliverables (single-commit or one-PR set)

- [ ] `sparse_credit.py`: extend modes + new `role_contrib_team` branch + new params
- [ ] `train_aris.py`:
      - extend `_training_reward` signature (partner_terminal_policy)
      - thread partner_terminal_policy from `_execute_option` and all seed/replay paths
      - `_build_q_network` applies value_bound to all methods when
        `apply_value_bound_to_all_methods=true`
      - `_expected_graph_objective_metadata` includes
        `role_contrib_team.ego_terminal_penalty_under_claim` when mode matches
- [ ] `evaluate_aris.py`:
      - `_execute_eval_option` adds `contribution_ledger` kwarg
      - `_run_episode` creates per-episode ledger and threads it
- [ ] `graph_builder.py` (only if `_coverage_constrained_pairs` doesn't already
      handle multiple kinds in `required_option_id_coverage`):
      - loop over each kind's per-id requirement, use per-eta fallback for hard cases
- [ ] `configs/ocv2_step4_asymm_role_v1.yaml`:
      - `sparse_credit: role_contrib_team`
      - `role_contrib_team.ego_terminal_penalty_under_claim: 0.3`
      - `value_bound.apply_to_all_methods: true`
      - extended `required_option_id_coverage` per §3.1

- [ ] Every `_training_reward(` call site updated (grep + verify)
- [ ] Every `_execute_eval_option(` call site passes ledger
- [ ] `python -m py_compile` on every modified file — must all pass
- [ ] Partner-differentiation probe on `standard7` remains 4/6 unique (regression
      check)
- [ ] Report git diff --stat + per-file 1-line summary + any deviation from spec

No training / CE / eval runs — code only.

---

## 9. Expected result pattern after v2 fixes

Reproduce v1 pipeline on role_v1 config (CE regen + 12 trains + 12 evals). Expected:

- **All methods' role_match_rate ↑** because `role_contrib_team` directly rewards
  role match. This is a strong training signal; every method should learn.
- **ARIS closes the gap to flat_factor** because graph coverage no longer starves
  belief on task-critical options.
- **base_only remains reactive** — no belief structure — but does OK because reward
  encodes the role signal directly.
- **global_gru still strong** — recurrence + role reward is the easy path.

**The critical question v2 answers**: does ARIS's factor-local relevance routing
add value ON TOP OF role-conditioned reward + proper graph coverage + fair value
bounds?

- If ARIS ≥ flat_factor ≥ global_gru: mechanism helps.
- If ARIS ≈ flat_factor and both < global_gru: recurrence beats structure here.
- If ARIS < flat_factor: relevance mask is still limiting even with coverage —
  Fix 3 (role_context) becomes the next candidate.
- If all methods equal and high: reward encoded the answer; no method-specific
  advantage on this task.

Each outcome is scientifically informative; v2's job is to make the comparison
fair, not to guarantee ARIS wins.
