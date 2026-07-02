# Implementation spec — role_conditioned_v1 benchmark

Executor: codex. Reviewer: Claude (post-execution audit).

This spec is the source of truth. Do not deviate from any signature, field name,
or numerical value below. If a change conflicts with existing code, resolve by
preserving the spec — never invent alternatives.

Repo: `/Users/aoudsung/Documents/ARIS4ZSC`
All file paths below are repo-relative.

---

## Objective

Add `role_conditioned_v1` partner benchmark and matching training objective
(`contrib_team` credit + contribution ledger) as a versioned NEW benchmark
alongside the existing `standard7` library. The existing standard7 code path
must remain byte-compatible so old configs, CE, and results still run.

---

## 0. Coding constraints

- All added imports must be at top of the file after existing imports.
- No emojis. No decorative comments. Only comments explaining a non-obvious
  invariant (why, not what).
- No print statements outside `if __name__ == "__main__"` blocks or existing
  logger patterns.
- Every new function must have full type annotations on parameters + return.
- All new config keys are read via `cfg.get(..., DEFAULT)` — no KeyError paths.
- After edits, run: `python -m py_compile <file>` on every modified file. Report
  any syntax error.
- Do not delete or rename existing public functions/classes unless the spec
  explicitly says so.

---

## 1. `experiments/overcooked_v2/partner_pool.py`

### 1.1 Add kind sets after existing imports, before `ProtocolSpec`

```python
TERMINAL_KINDS = frozenset({"pick_plate", "plate_soup", "serve_soup"})
PREP_KINDS = frozenset({"fetch_ingredient", "deliver_ingredient_to_pot"})
SUPPORT_KINDS = frozenset({
    "drop_item_to_counter",
    "clear_interaction_cell",
    "wait_at_bottleneck",
    "cross_bottleneck",
})
```

### 1.2 Extend `ProtocolSpec`

Add ONE field to the existing dataclass (do not remove any existing field):

```python
    curriculum_group: str | None = None
```

### 1.3 Rename `TRAINING_PROTOCOLS` -> `STANDARD7_PROTOCOLS`

Rename the existing tuple constant `TRAINING_PROTOCOLS` to `STANDARD7_PROTOCOLS`.
Preserve all 7 entries verbatim (ingredient-near, ingredient-far, dish-server,
server-left, bottleneck-yield, flexible-balanced, terminal-yield).

Add backward-compat alias immediately after:
```python
TRAINING_PROTOCOLS = STANDARD7_PROTOCOLS  # backward-compat alias
```

### 1.4 Add `ROLE_CONDITIONED_V1_PROTOCOLS`

Add this tuple constant after `STANDARD7_PROTOCOLS`:

```python
ROLE_CONDITIONED_V1_PROTOCOLS: tuple[tuple[str, ProtocolSpec], ...] = (
    (
        "ingredient-near-yield",
        ProtocolSpec(
            role="ingredient_person",
            pot_preference="near",
            terminal_policy="yield",
            curriculum_group="terminal_yield",
        ),
    ),
    (
        "ingredient-far-yield",
        ProtocolSpec(
            role="ingredient_person",
            pot_preference="far",
            terminal_policy="yield",
            curriculum_group="terminal_yield",
        ),
    ),
    (
        "server-left-claim",
        ProtocolSpec(
            role="server",
            delivery_preference="left",
            terminal_policy="claim",
            curriculum_group="terminal_claim",
        ),
    ),
    (
        "server-right-claim",
        ProtocolSpec(
            role="server",
            delivery_preference="right",
            terminal_policy="claim",
            curriculum_group="terminal_claim",
        ),
    ),
    (
        "bottleneck-yield-terminal-yield",
        ProtocolSpec(
            role="flexible",
            bottleneck_policy="yield",
            terminal_policy="yield",
            curriculum_group="terminal_yield",
        ),
    ),
    (
        "bottleneck-push-terminal-claim",
        ProtocolSpec(
            role="flexible",
            bottleneck_policy="push",
            terminal_policy="claim",
            curriculum_group="terminal_claim",
        ),
    ),
    (
        "heldout-yield-terminal-claim",
        ProtocolSpec(
            role="flexible",
            bottleneck_policy="yield",
            terminal_policy="claim",
            curriculum_group="terminal_claim",
        ),
    ),
    (
        "heldout-push-terminal-yield",
        ProtocolSpec(
            role="flexible",
            bottleneck_policy="push",
            terminal_policy="yield",
            curriculum_group="terminal_yield",
        ),
    ),
)
```

### 1.5 Add `PARTNER_REGISTRIES`

Immediately after `ROLE_CONDITIONED_V1_PROTOCOLS`:

```python
PARTNER_REGISTRIES: dict[str, tuple[tuple[str, ProtocolSpec], ...]] = {
    "standard7": STANDARD7_PROTOCOLS,
    "role_conditioned_v1": ROLE_CONDITIONED_V1_PROTOCOLS,
}
```

### 1.6 Update `make_training_partners`

Change signature to:
```python
def make_training_partners(
    option_library: Any,
    partner_set: str = "standard7",
) -> list[ScriptedProtocolPartner]:
```

Body: pick protocols from `PARTNER_REGISTRIES[partner_set]` (KeyError-safe:
raise `ValueError` with valid choices if unknown). Iterate as before but from
the selected protocol tuple.

### 1.7 Add `_terminal_policy_bonus` helper

Place after existing `_bottleneck_bonus` helper (before `_path_length_penalty`):

```python
def _terminal_policy_bonus(opt: OptionSpec, policy: str | None) -> float:
    """Lexicographic role-identity tier (magnitude 10^3 - 10^4)."""
    if policy == "yield":
        if opt.kind in TERMINAL_KINDS:
            return -30000.0
        if opt.kind in PREP_KINDS or opt.kind in SUPPORT_KINDS:
            return 5000.0
        return 0.0
    if policy == "claim":
        if opt.kind in TERMINAL_KINDS:
            return 8000.0
        if opt.kind in PREP_KINDS:
            return 1000.0
        if opt.kind == "wait_at_bottleneck":
            return -2000.0
        return 0.0
    return 0.0
```

### 1.8 Rescale `_role_bonus`

Modify existing `_role_bonus`:
```python
def _role_bonus(condition: bool) -> float:
    """Lexicographic role-identity tier (magnitude 10^3)."""
    return 4000.0 if condition else -1000.0
```

### 1.9 Modify `_protocol_score`

In the existing `_protocol_score`, REPLACE the entire block that begins with:
```python
        if self.protocol.terminal_policy == "yield" and opt.kind in {
            "pick_plate",
            "plate_soup",
            "serve_soup",
        }:
            score -= 100.0
        if self.protocol.terminal_policy == "yield" and opt.kind in {
            "clear_interaction_cell",
            "wait_at_bottleneck",
        }:
            score += 3.0
```
with:
```python
        score += _terminal_policy_bonus(opt, self.protocol.terminal_policy)
```

Do NOT touch other blocks in `_protocol_score`.

### 1.10 Do NOT reintroduce cr-* partners, `terminal_policy="serve"`,
`switch_ys`/`switch_sy`, or `total_steps`. These stay reverted.

---

## 2. `experiments/overcooked_v2/sparse_credit.py`

### 2.1 Extend `SPARSE_CREDIT_MODES` (or equivalent tuple/constant)

Find the tuple/set of accepted modes and add `"contrib_team"` at the end.

### 2.2 Extend `actor_sparse_reward` signature

Add ONE new keyword-only parameter, keeping ALL existing parameters:
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
) -> float:
```

### 2.3 Add `contrib_team` branch in `actor_sparse_reward`

Insert this branch AFTER the existing `ego_correct_delivery` branch, BEFORE
the final `raise ValueError` (or equivalent unrecognized-mode error):

```python
    if mode == "contrib_team":
        ego_sole_delivery = bool(getattr(event, "ego_delivery_event", False)) and not bool(
            getattr(event, "partner_delivery_event", False)
        )
        partner_delivery = bool(getattr(event, "partner_delivery_event", False))
        if ego_sole_delivery:
            return float(team_sparse)
        if partner_delivery and bool(ego_contributed):
            return float(contrib_scale) * float(team_sparse)
        return 0.0
```

### 2.4 Extend `sparse_credit_params`

Where the function reads `training_cfg` and returns a dict, ADD these keys to
the returned dict (do not remove existing keys):
```python
    "contrib_scale": float(
        (cfg.get("contrib_team") or {}).get("contrib_scale", 1.0)
    ),
```

The `mode` key must still be returned as-is (accept "contrib_team" as a value).

---

## 3. `experiments/overcooked_v2/reward_design.py`

### 3.1 Add `ContributionLedger` dataclass at the end of the module

```python
CONTRIB_OPTION_KINDS: frozenset[str] = frozenset({
    "deliver_ingredient_to_pot",
    "clear_interaction_cell",
    "drop_item_to_counter",
    "fetch_ingredient",
})


@dataclass
class ContributionLedger:
    """Delivery-delimited ledger of ego contribution to the current dish cycle.

    Set on any support-kind option that produced a visible env effect (pot state
    changed OR object pickup/drop). Reset when a delivery event fires (natural
    dish-cycle boundary — no time hyperparameter).
    """
    ego_contributed: bool = False

    @classmethod
    def from_config(cls, training_cfg: dict[str, Any] | None) -> "ContributionLedger":
        return cls()

    def reset(self) -> None:
        self.ego_contributed = False

    def update(self, event: Any, ego_option_kind: str) -> None:
        if str(ego_option_kind) not in CONTRIB_OPTION_KINDS:
            return
        visible_effect = bool(getattr(event, "pot_changed", False)) or bool(
            getattr(event, "object_pickup_or_drop", False)
        )
        if visible_effect:
            self.ego_contributed = True

    def query_and_reset_on_delivery(self, event: Any) -> bool:
        """Return current contribution flag; reset if this step was a delivery.
        Call this AT the reward-computation site so the value used for reward
        matches the ledger state accumulated up to (and including) the delivery."""
        result = bool(self.ego_contributed)
        if bool(getattr(event, "delivery_event", False)):
            self.ego_contributed = False
        return result
```

Required imports at top of file (add if missing):
- `from dataclasses import dataclass`
- `from typing import Any`

Do NOT remove `terminal_progress_bonus` / `terminal_progress_params` — they
remain for backward-compat (config can disable them).

---

## 4. `experiments/overcooked_v2/train_aris.py`

### 4.1 Import additions

At top with other `overcooked_v2` imports:
```python
from experiments.overcooked_v2.reward_design import ContributionLedger
from experiments.overcooked_v2.partner_pool import PARTNER_REGISTRIES
```

### 4.2 Wire `partner_set` through `make_training_partners`

Every call to `make_training_partners(option_lib)` in this file MUST become:
```python
make_training_partners(
    option_lib,
    partner_set=str(config.get("training", {}).get("partner_set", "standard7")),
)
```
(or `training_cfg.get(...)` if `training_cfg` is the local variable name).

### 4.3 Remove ego-delivery checkpoint gating

Find the checkpoint-selection block that consults `require_ego_delivery_selection`
and `_free_rider_guard_verdict` to decide `_eligible` and `deployable_checkpoint`.
Modify so:

- `deployable_checkpoint` = `selected_checkpoint` (always the greedy-best), unconditionally.
- `free_rider_guard` field: KEEP computing it but only record as diagnostic —
  do not gate `deployable_checkpoint` on it.
- `run_status = "ok"` unconditionally at end of training. Remove the branch that
  sets `run_status = "no_deployable_checkpoint"` and raises/returns early.

Do NOT delete `_free_rider_guard_verdict` / `_free_rider_diagnosis` functions.
They remain callable for diagnostics.

### 4.4 Thread `ContributionLedger` through the training rollout

In `_execute_option`, extend signature (keyword-only param, default None):
```python
def _execute_option(
    env: OCV2Adapter,
    obs: dict[str, np.ndarray],
    partner: Any,
    option_lib: OCV2OptionLibrary,
    router: OCV2EvidenceRouter,
    evidence_buffer: EvidenceBuffer,
    option_id: int,
    graph: GraphSpec,
    rng: np.random.Generator,
    config: dict[str, Any],
    *,
    contribution_ledger: ContributionLedger | None = None,
) -> tuple[OptionTransition, bool, dict[str, np.ndarray]]:
```

Inside the primitive-step loop, immediately AFTER computing `event` and BEFORE
computing reward, update the ledger:
```python
        if contribution_ledger is not None:
            contribution_ledger.update(event, ego_option_kind=str(opt.kind))
```

Change the reward call from:
```python
        reward_sum += _training_reward(step, config, "agent_0", event)
```
to:
```python
        ego_contributed = False
        if contribution_ledger is not None:
            ego_contributed = contribution_ledger.query_and_reset_on_delivery(event)
        reward_sum += _training_reward(
            step, config, "agent_0", event,
            ego_contributed=ego_contributed,
        )
```

### 4.5 Extend `_training_reward` signature

Change signature to:
```python
def _training_reward(
    step: Any,
    config: dict[str, Any],
    agent_key: str,
    event: Any,
    *,
    ego_contributed: bool = False,
) -> float:
```

In the call to `actor_sparse_reward` inside `_training_reward`, pass
`ego_contributed=ego_contributed`.

### 4.6 Every caller of `_execute_option` in train_aris.py must create + pass
a fresh `ContributionLedger` per training episode (create at env.reset() time,
reset per episode). Search for every `_execute_option(` invocation and add
`contribution_ledger=ledger` where `ledger` is the current episode's ledger.

If a caller is in a scripted-terminal-seed path (`_maybe_seed_terminal_replay`
or `_pretrain_from_seed_replay`), create a fresh ledger per seed episode there
too.

### 4.7 Every caller of `_training_reward` in train_aris.py must be updated

Grep for `_training_reward(` and add `ego_contributed=` where appropriate.
Callers outside the option loop (e.g. greedy validation) should pass
`ego_contributed=False` (safe default) unless a ledger is available in scope.

### 4.8 Role-aware exploration

Locate `_sample_exploration_option`. Extend signature with a keyword-only param:
```python
    partner_terminal_policy: str | None = None,
```

Body: read `config.training.role_exploration`. If `enabled` (default False),
look up the sub-dict by `partner_terminal_policy` value (`"yield"` or `"claim"`);
fall back to `role_exploration.default` if not found; fall back to legacy
`terminal_exploration` block if role_exploration absent (backward compat).

Use `preferred_kinds` from the selected sub-dict.

Update every caller of `_sample_exploration_option` in train_aris.py to pass
`partner_terminal_policy=getattr(getattr(current_partner, "protocol", None), "terminal_policy", None)`.

Do NOT delete the existing terminal_exploration code path; keep it as the
fallback so standard7 configs still work.

### 4.9 Role-balanced partner sampling

Locate `_select_train_partners` (or equivalent function that builds the training
partner list). Where it currently sets `sampling_weight` on each partner from
`training.partner_sampling`, ALSO set:

```python
    default_group = getattr(getattr(partner, "protocol", None), "curriculum_group", None)
    setattr(
        partner,
        "sampling_group",
        str(groups_cfg.get(partner.name, default_group or partner.name)),
    )
```

where `groups_cfg = train_cfg.get("partner_groups", {}) or {}`.

Locate `_sample_partner`. Rewrite body to:
```python
def _sample_partner(partners: list[Any], rng: np.random.Generator) -> Any:
    grouped: dict[str, list[Any]] = {}
    for partner in partners:
        group = str(getattr(partner, "sampling_group", getattr(partner, "name", "default")))
        grouped.setdefault(group, []).append(partner)

    if len(grouped) < len(partners):
        # Multiple groups -> sample group uniformly, then partner by weight within group.
        group_names = sorted(grouped)
        group = group_names[int(rng.integers(0, len(group_names)))]
        candidates = grouped[group]
    else:
        candidates = partners

    weights = np.asarray(
        [max(0.0, float(getattr(p, "sampling_weight", 1.0))) for p in candidates],
        dtype=np.float64,
    )
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("partner_sampling weights must be finite and have positive sum.")
    probs = weights / total
    idx = int(rng.choice(np.arange(len(candidates)), p=probs))
    return candidates[idx]
```

If there is a currently-existing `_sample_partner` with different semantics,
preserve behavior for the single-group case (which was the pre-existing
default), and only add the new multi-group branch.

### 4.10 Role-balanced replay seed (optional support)

Locate `_maybe_seed_terminal_replay`. Add a branch: if `config.training.role_replay_seed`
is present, iterate its `chains` list instead of the legacy single-partner form.
Each chain dict has keys: `name`, `partner`, `target_actor` ("ego"|"partner"),
`target_deliveries`, `ego_priority_kinds`.

For a chain with `target_actor="ego"`, count `ego_sole_correct_delivery` per the
existing counter. For `target_actor="partner"`, count `partner_correct_delivery`.

Preserve the legacy `terminal_replay_seed` code path as fallback (do not delete).

If the implementation would be extensive, gate this behind
`if training_cfg.get("role_replay_seed", {}).get("enabled", False):` and use
the existing scripted-terminal helpers; report per-chain counts in the same
metrics dict.

Emit a `logger` warning if both `terminal_replay_seed.enabled=True` and
`role_replay_seed.enabled=True` — use role_replay_seed and skip the legacy one.

---

## 5. `experiments/overcooked_v2/evidence_router.py`

### 5.1 Extend `EVIDENCE_INDEX`

Add these entries at the END of the EVIDENCE_INDEX dict (before the closing `}`):
```python
    "ego_terminal_option": 49,
    "partner_terminal_option": 50,
    "partner_prep_option": 51,
    "partner_support_option": 52,
    "partner_bottleneck_wait": 53,
    "partner_bottleneck_cross": 54,
```

Do NOT modify existing indices. Keep the existing `ego_option_terminated_failed: 48`
that was added earlier.

`D_EVID = 64` remains unchanged.

### 5.2 Add kind sets

Near top of file after imports, mirror the sets from partner_pool.py (import them
directly to avoid drift):
```python
from experiments.overcooked_v2.partner_pool import (
    TERMINAL_KINDS,
    PREP_KINDS,
    SUPPORT_KINDS,
)
```

### 5.3 Factor-local routing in `route()`

Inside the existing `for factor_idx, factor in enumerate(self.graph.factors):`
loop, AFTER the existing per-factor evidence assignments and AFTER the
`ego_option_terminated_failed` block, add:

```python
            # Factor-local semantic evidence: each kind-tagged channel fires ONLY
            # for factors whose option pair touches an option of that kind.
            options_seq = self.graph.options
            i_kind = str(options_seq[int(factor.option_i)].kind)
            j_kind = str(options_seq[int(factor.option_j)].kind)
            factor_touches_terminal = i_kind in TERMINAL_KINDS or j_kind in TERMINAL_KINDS
            factor_touches_prep = i_kind in PREP_KINDS or j_kind in PREP_KINDS
            factor_touches_support = i_kind in SUPPORT_KINDS or j_kind in SUPPORT_KINDS
            factor_touches_bottleneck = "bottleneck" in i_kind or "bottleneck" in j_kind

            ego_kind = None
            if ego_option_id is not None and 0 <= int(ego_option_id) < len(options_seq):
                ego_kind = str(options_seq[int(ego_option_id)].kind)
            partner_kind = None
            if current_partner_option is not None and 0 <= int(current_partner_option) < len(options_seq):
                partner_kind = str(options_seq[int(current_partner_option)].kind)

            if factor_touches_terminal and ego_kind in TERMINAL_KINDS:
                routed[factor_idx, EVIDENCE_INDEX["ego_terminal_option"]] = 1.0
            if factor_touches_terminal and partner_kind in TERMINAL_KINDS:
                routed[factor_idx, EVIDENCE_INDEX["partner_terminal_option"]] = 1.0
            if factor_touches_prep and partner_kind in PREP_KINDS:
                routed[factor_idx, EVIDENCE_INDEX["partner_prep_option"]] = 1.0
            if factor_touches_support and partner_kind in SUPPORT_KINDS:
                routed[factor_idx, EVIDENCE_INDEX["partner_support_option"]] = 1.0
            if factor_touches_bottleneck and partner_kind == "wait_at_bottleneck":
                routed[factor_idx, EVIDENCE_INDEX["partner_bottleneck_wait"]] = 1.0
            if factor_touches_bottleneck and partner_kind == "cross_bottleneck":
                routed[factor_idx, EVIDENCE_INDEX["partner_bottleneck_cross"]] = 1.0
```

Do NOT create a global "terminal_role_conflict" or "terminal_role_complement"
signal. The routing must remain strictly factor-local for the ARIS vs
flat_factor distinction.

---

## 6. `experiments/overcooked_v2/graph_builder.py`

### 6.1 Add ROLE_CONTRAST_KIND_PAIRS

Near the top of the file, after existing GRAPH_VARIANTS / kind constants:
```python
ROLE_CONTRAST_KIND_PAIRS: tuple[tuple[str, str], ...] = (
    ("serve_soup", "wait_at_bottleneck"),
    ("serve_soup", "clear_interaction_cell"),
    ("serve_soup", "deliver_ingredient_to_pot"),
    ("plate_soup", "deliver_ingredient_to_pot"),
    ("pick_plate", "fetch_ingredient"),
)
```

### 6.2 Insert role-contrast reservation in `_coverage_constrained_pairs`

In `_coverage_constrained_pairs`, AFTER the existing per-option-id coverage step
(step 1) and per-kind coverage step (step 2), BEFORE the CE fill step,
insert role-contrast reservation:

```python
    # Step 2.5: role-contrast reservation. For each configured kind-pair, reserve
    # at least one factor whose option pair matches the kinds (in either order).
    # Structural prior for terminal role switching; not a gate.
    role_pairs_cfg = tuple(
        tuple(p) for p in (
            (selection_cfg or {}).get("required_role_contrast_pairs")
            or ROLE_CONTRAST_KIND_PAIRS
        )
    )
    for a_kind, b_kind in role_pairs_cfg:
        if len(selected) >= max_factors:
            break
        already = any(
            {kind(i), kind(j)} == {str(a_kind), str(b_kind)}
            for _s, i, j in selected
        )
        if already:
            continue
        best = next(
            (
                p for p in all_pairs
                if (p[1], p[2]) not in keys
                and {kind(p[1]), kind(p[2])} == {str(a_kind), str(b_kind)}
            ),
            None,
        )
        if best is not None:
            add(best, "mandatory_role_contrast", enforce_div=False)
```

`all_pairs`, `keys`, `selected`, `kind()`, and `add()` are local names already
present in `_coverage_constrained_pairs`. If any of those specific local-name
targets are named differently in this codebase, adapt to the actual names but
preserve the semantic: reserve one factor per pair (in either kind-order)
before CE fill, with provenance `"mandatory_role_contrast"`.

Provenance tag `"mandatory_role_contrast"` must be recorded in the same
metadata slot as other selected_by tags.

### 6.3 Do NOT change other selection paths

`top_k`, `full_support_graph`, `random_same_size_graph`, `minus_serve_soup_graph`,
etc. remain unchanged. Role-contrast reservation runs only when the coverage-
constrained path is invoked.

---

## 7. `experiments/overcooked_v2/ce_sampler.py`

### 7.1 Thread ContributionLedger through both rollout paths

In `_rollout_option` (sequential path) and `collect_option_replay_batched`
(batched path), add a per-episode `ContributionLedger` and use it for the
`actor_sparse_reward` call.

For `_rollout_option`:
- At the start of the function (before the primitive-step loop), instantiate:
  ```python
  ledger = ContributionLedger()
  ```
- In the primitive-step loop, before the reward call, update the ledger:
  ```python
  ledger.update(event, ego_option_kind=str(opt.kind))
  ```
- Change the `actor_sparse_reward` call to pass:
  ```python
  ego_contributed=ledger.query_and_reset_on_delivery(event),
  ```

For `collect_option_replay_batched`:
- The batched path processes multiple env slots. Maintain a **list of ledgers**
  indexed by slot: `slot_ledgers: list[ContributionLedger] = [ContributionLedger() for _ in ...]`
- When an env slot resets (new episode), reset the corresponding ledger:
  `slot_ledgers[i].reset()`
- Before the reward call for each slot's event, call:
  `slot_ledgers[i].update(event_i, ego_option_kind=str(current_option_kind_i))`
- Pass `ego_contributed=slot_ledgers[i].query_and_reset_on_delivery(event_i)` to
  `actor_sparse_reward`.

Add the import at top of file:
```python
from experiments.overcooked_v2.reward_design import ContributionLedger
```

### 7.2 Preserve legacy modes

Do NOT remove the existing `exclude_terminal_progress_from_reward_sum` gate.
The ledger applies only when the sparse credit mode is `contrib_team`
(other modes ignore `ego_contributed`).

---

## 8. `experiments/overcooked_v2/scripts/run_ce_pipeline.py`

### 8.1 Extend reward_metadata dict

Where `reward_metadata` is built (near top of `main`), ADD keys:
```python
    "partner_set": str(train_cfg.get("partner_set", "standard7")),
    "contribution_credit": {
        "contrib_scale": float(
            (train_cfg.get("contrib_team") or {}).get("contrib_scale", 1.0)
        ),
    },
```

Keep all existing keys (`layout`, `cost_coef`, `shaped_reward_coef`,
`sparse_credit`, `terminal_progress_shaping`, etc.).

### 8.2 Pass partner_set to make_training_partners

Find the call to `make_training_partners(lib)` inside the CE main and change to:
```python
make_training_partners(
    lib,
    partner_set=str(train_cfg.get("partner_set", "standard7")),
)
```

---

## 9. `experiments/overcooked_v2/train_aris.py` — objective-metadata gate

### 9.1 Extend expected metadata dict

In `_expected_graph_objective_metadata` (around the file), ADD to the returned
dict:
```python
    "partner_set": str(training.get("partner_set", "standard7")),
```

And gate on `contrib_team` mode by including the assist scale (only when the
mode is `contrib_team`):
```python
    if str(params.get("mode", "")) == "contrib_team":
        expected["contribution_credit"] = {
            "contrib_scale": float(
                (training.get("contrib_team") or {}).get("contrib_scale", 1.0)
            ),
        }
```

The gate comparison logic then rejects a CE where the stored `partner_set`
mismatches the training `partner_set` (i.e. can't train `role_conditioned_v1`
on a `standard7` CE, and vice versa).

---

## 10. `experiments/overcooked_v2/evaluate_aris.py`

### 10.1 Add role_match metric per result

In the outer eval loop where each result dict is built (around
`aggregate, episodes = _evaluate_partner(...)` and the subsequent
`result = {...}` assembly), locate the point where `aggregate` is finalized.
Right after the aggregate is complete, add:

```python
    partner_protocol = None
    try:
        partners_map = {p.name: p for p in make_training_partners(
            ctx.option_lib,
            partner_set=str(ctx.config.get("training", {}).get("partner_set", "standard7")),
        )}
        p_obj = partners_map.get(partner_name)
        proto_obj = getattr(p_obj, "protocol", None) if p_obj is not None else None
        if proto_obj is not None:
            partner_protocol = {
                "role": getattr(proto_obj, "role", None),
                "bottleneck_policy": getattr(proto_obj, "bottleneck_policy", None),
                "terminal_policy": getattr(proto_obj, "terminal_policy", None),
                "curriculum_group": getattr(proto_obj, "curriculum_group", None),
            }
    except Exception:
        partner_protocol = None
    aggregate["partner_protocol"] = partner_protocol

    terminal_policy = (partner_protocol or {}).get("terminal_policy")
    ego_deliv = int(aggregate.get("ego_correct_delivery_count", 0) or 0)
    prt_deliv = int(aggregate.get("partner_correct_delivery_count", 0) or 0)
    total_deliv = int(aggregate.get("correct_delivery_count", ego_deliv + prt_deliv) or 0)
    if terminal_policy == "yield":
        role_match = ego_deliv
    elif terminal_policy == "claim":
        role_match = prt_deliv
    else:
        role_match = total_deliv
    aggregate["role_match_delivery_count"] = int(role_match)
    aggregate["role_match_rate"] = float(role_match / max(1, int(args.episodes)))
```

### 10.2 Add summary field

Where the top-level `summary` dict is built at the end of `evaluate()`, ADD:
```python
    def _mean_or_zero(vals):
        return float(np.mean(vals)) if vals else 0.0

    summary["mean_role_match_rate"] = _mean_or_zero([
        float(r["aggregate"].get("role_match_rate", 0.0)) for r in results
    ])
    summary["mean_ego_delivery_count"] = _mean_or_zero([
        float(r["aggregate"].get("ego_correct_delivery_count", 0.0)) for r in results
    ])
    summary["mean_partner_delivery_count"] = _mean_or_zero([
        float(r["aggregate"].get("partner_correct_delivery_count", 0.0)) for r in results
    ])
```

If a `_mean_or_zero` (or `_mean_or_nan`) helper already exists in the file,
reuse it. Do NOT introduce a duplicate helper.

### 10.3 Thread partner_set to make_training_partners

Every existing `make_training_partners(ctx.option_lib)` in evaluate_aris.py
must pass `partner_set=str(ctx.config.get("training", {}).get("partner_set", "standard7"))`.
Same for `make_training_partners(option_lib)` in helper functions like
`_resolve_partner_names` (thread `option_lib` and `partner_set` through).

---

## 11. Config: new file `configs/ocv2_step4_asymm_role_v1.yaml`

Create a new YAML file. Use the existing `ocv2_step4_asymm.yaml` as a base
(copy every non-training section verbatim: `env`, `layouts`, `graph` etc.),
then REPLACE the `training` block with:

```yaml
training:
  partner_set: role_conditioned_v1

  train_partners:
    - ingredient-near-yield
    - ingredient-far-yield
    - server-left-claim
    - server-right-claim
    - bottleneck-yield-terminal-yield
    - bottleneck-push-terminal-claim

  partner_groups:
    ingredient-near-yield: terminal_yield
    ingredient-far-yield: terminal_yield
    bottleneck-yield-terminal-yield: terminal_yield
    server-left-claim: terminal_claim
    server-right-claim: terminal_claim
    bottleneck-push-terminal-claim: terminal_claim

  partner_sampling: {}

  sparse_credit: contrib_team

  contrib_team:
    contrib_scale: 1.0

  terminal_progress_shaping:
    enabled: false
    ego_plate_pick_bonus: 0.0
    ego_plate_soup_bonus: 0.0
    ego_serve_bonus: 0.0
    max_bonus_per_step: null

  role_exploration:
    enabled: true
    yield:
      bias_start: 0.90
      bias_end: 0.35
      anneal_updates: 2500
      preferred_kinds:
        - serve_soup
        - plate_soup
        - pick_plate
    claim:
      bias_start: 0.90
      bias_end: 0.35
      anneal_updates: 2500
      preferred_kinds:
        - deliver_ingredient_to_pot
        - fetch_ingredient
        - clear_interaction_cell
        - wait_at_bottleneck
        - drop_item_to_counter

  terminal_exploration:
    enabled: false

  terminal_replay_seed:
    enabled: false

  role_replay_seed:
    enabled: true
    chains:
      - name: ego_terminal_against_yield
        partner: ingredient-near-yield
        target_actor: ego
        target_deliveries: 32
        ego_priority_kinds:
          - serve_soup
          - plate_soup
          - pick_plate
          - deliver_ingredient_to_pot
          - fetch_ingredient
          - clear_interaction_cell
          - wait_at_bottleneck
      - name: support_partner_terminal_claim
        partner: server-left-claim
        target_actor: partner
        target_deliveries: 32
        ego_priority_kinds:
          - deliver_ingredient_to_pot
          - fetch_ingredient
          - clear_interaction_cell
          - wait_at_bottleneck
          - drop_item_to_counter
    seed_updates: 600
    seed_batch_size: 8

  require_ego_delivery_selection: false
  require_preflight_accepted: true

  # Preserve everything else from base config (learning_rate, gamma, cost_coef,
  # cost_per_step, shaped_reward_coef, value_bound, hidden_dim, obs_encoder,
  # num_partners, evidence_window, target_update_interval, grad_clip_norm,
  # epsilon_start, epsilon_end, max_episode_options, log_interval,
  # checkpoint_every, checkpoint_eval_episodes, select_best_by,
  # updates_per_transition, batch_size, warmup_transitions, replay_size,
  # td_loss, huber_delta, double_q, advantage_norm, total_updates).
  # Copy these values verbatim from ocv2_step4_asymm.yaml.
```

Also update the `graph` block for this config: leave `ce_path` and `replay_path`
pointing to NEW paths — `outputs/asymm_ce_role_v1/ce_refined.npy` and
`outputs/asymm_ce_role_v1/replay.npz` respectively. Keep the coverage-constrained
selection block as-is (serve_soup / plate_soup / pick_plate coverage).

---

## 12. Testing after edits

Run these checks and report results:

1. `python -m py_compile experiments/overcooked_v2/partner_pool.py`
2. `python -m py_compile experiments/overcooked_v2/sparse_credit.py`
3. `python -m py_compile experiments/overcooked_v2/reward_design.py`
4. `python -m py_compile experiments/overcooked_v2/evidence_router.py`
5. `python -m py_compile experiments/overcooked_v2/train_aris.py`
6. `python -m py_compile experiments/overcooked_v2/evaluate_aris.py`
7. `python -m py_compile experiments/overcooked_v2/ce_sampler.py`
8. `python -m py_compile experiments/overcooked_v2/graph_builder.py`
9. `python -m py_compile experiments/overcooked_v2/scripts/run_ce_pipeline.py`
10. `python -c "from experiments.overcooked_v2.partner_pool import PARTNER_REGISTRIES; print(list(PARTNER_REGISTRIES.keys()))"`
11. `python -c "from experiments.overcooked_v2.partner_pool import make_training_partners; from experiments.overcooked_v2.env_adapter import OCV2Adapter; from experiments.overcooked_v2.layout_parser import parse_layout; from experiments.overcooked_v2.options import OCV2OptionLibrary; env=OCV2Adapter('asymm_advantages',max_steps=400,observation_type='default',force_path_planning=True); lg=parse_layout(env.env,'asymm_advantages'); lib=OCV2OptionLibrary(lg,max_option_steps=16,strict_preconditions=True,dynamic_budget=True); ps=make_training_partners(lib,partner_set='role_conditioned_v1'); print([p.name for p in ps]); print([getattr(p.protocol,'terminal_policy',None) for p in ps]); print([getattr(p.protocol,'curriculum_group',None) for p in ps])"`

Expected output for check 11 (order-preserving):
```
['ingredient-near-yield', 'ingredient-far-yield', 'server-left-claim', 'server-right-claim', 'bottleneck-yield-terminal-yield', 'bottleneck-push-terminal-claim', 'heldout-yield-terminal-claim', 'heldout-push-terminal-yield']
['yield', 'yield', 'claim', 'claim', 'yield', 'claim', 'claim', 'yield']
['terminal_yield', 'terminal_yield', 'terminal_claim', 'terminal_claim', 'terminal_yield', 'terminal_claim', 'terminal_claim', 'terminal_yield']
```

12. Partner-differentiation probe on asymm_advantages:
```python
import numpy as np
from experiments.overcooked_v2.env_adapter import OCV2Adapter
from experiments.overcooked_v2.layout_parser import parse_layout
from experiments.overcooked_v2.options import OCV2OptionLibrary
from experiments.overcooked_v2.partner_pool import make_training_partners

env = OCV2Adapter("asymm_advantages", max_steps=400, observation_type="default", force_path_planning=True)
lg = parse_layout(env.env, "asymm_advantages")
lib = OCV2OptionLibrary(lg, max_option_steps=16, strict_preconditions=True, dynamic_budget=True)
partners = make_training_partners(lib, partner_set="role_conditioned_v1")
seqs = {}
for p in partners:
    env.reset(0); p.reset(0); rng = np.random.default_rng(0)
    acts = []
    for _ in range(60):
        pa = p.act(None, env.state, rng)
        acts.append(int(pa.primitive_action))
        env.step(0, pa.primitive_action)
    seqs[p.name] = tuple(acts)
print("unique behaviors:", len(set(seqs.values())), "/", len(partners))
for n, s in seqs.items():
    print(n, s[:12])
```
Expected: `unique behaviors: >= 6 / 8` (must beat standard7's 4/6 on asymm).

Report each check's output. If any fails, do NOT try to fix it silently — stop
and report the failure with the exact error.

---

## 13. Deliverables checklist for codex

- [ ] All 8 source files modified per §1-§10.
- [ ] New config `configs/ocv2_step4_asymm_role_v1.yaml` created per §11.
- [ ] All 12 checks in §12 pass with the expected outputs.
- [ ] `git diff --stat` output attached in the completion report.
- [ ] For each file, a brief 1-line summary of what was changed.
- [ ] Any deviation from this spec (even minor) explicitly called out.

Do NOT run any training / CE regen / eval. Codex's job is code-only. Runs will
happen later under Claude's coordination.
