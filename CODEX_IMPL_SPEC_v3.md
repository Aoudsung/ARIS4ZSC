# Implementation spec v3 — actor-asymmetric relevance fix (corrected)

Executor: codex. Reviewer: Claude (post-execution audit).

Addresses one structural bug diagnosed in v2 results: `_relevant_options_for_factor`
treats CE column indices as ego-action ids, but CE is `CE[ego_option, partner_option]`.
This routes ARIS's relevance mask to partner-only options that ego never selects,
starving the factor-local Q of belief influence on the correct ego actions.

Evidence from v2 replay:
```
option 6 pick_plate  : ego_count=2302, partner_weight=0.0     (ego side)
option 7 pick_plate  : ego_count=0,    partner_weight=974.6   (partner side)
option 8 serve_soup  : ego_count=327,  partner_weight=0.0     (ego side)
option 9 serve_soup  : ego_count=0,    partner_weight=2242.9  (partner side)
```

Under legacy relevance, factor `(2:deliver, 9:partner_serve)` routes to
option 9 — ego never picks that. Under v3, kind-based projection routes to
ego-selectable options of matching kind (e.g., option 8 = ego serve).

Matches v1 diagnostic: ARIS `belief_swap_delta.mean_abs_maxq_delta ≈ 0`
vs `flat_factor ≈ 0.07`.

Constraint recap: single TD loss; `argmax Q`; no selector/imitation. CE artifact
does not need regeneration (row/col semantics unchanged); only the relevance
mapping changes. Reuse `outputs/asymm_ce_role_v1_v2/`.

Repo: `/Users/aoudsung/Documents/ARIS4ZSC`. Code only — no training runs.

---

## 0. Design invariants (corrected)

- CE matrix and replay stay v2 as-is. Only relevance mapping changes.
- Backward compat: when `relevance_semantics` is absent or set to `"legacy_id_pair"`,
  behavior is preserved byte-for-byte. `standard7` pipeline unaffected.
- v3 gated behind `graph.relevance_semantics: ego_kind_projection`.
- **HARD FAIL**: if `relevance_semantics=ego_kind_projection` and `ego_selectable`
  can't be derived, raise. Never silently fall back to legacy — that produced a
  looks-like-v3-actually-legacy graph in the initial draft.
- **route_map stays legacy** (see §1.3.2). Only `graph.relevance` is projected.
  route_map drives evidence_router entity/region lookup — projecting it would
  strip partner-side spatial evidence.

---

## 1. `experiments/overcooked_v2/graph_builder.py`

### 1.1 Derive ego-selectable options from replay ROWS (not option_kind_stats)

Add to graph_builder.py (replay metadata is a JSON string in the current codebase;
`option_kind_stats` only has `attempt_count/success_count/timeout_count`, so it
CANNOT be used to derive ego-selectable ids — read replay rows directly):

```python
def _load_ego_selectable_from_replay(replay_path: str | None) -> frozenset[int] | None:
    """Read ego_option per replay row; ego-selectable = the set of ids that
    appear as ego_option in at least one row. Returns None if replay is absent
    or has no rows with ego_option; caller must decide policy (raise / fallback)."""
    if not replay_path:
        return None
    import json
    import numpy as np
    from pathlib import Path

    p = Path(replay_path)
    if not p.exists():
        return None

    ids: set[int] = set()
    try:
        with np.load(p, allow_pickle=False) as npz:
            if "rows" not in npz.files:
                return None
            for raw in npz["rows"]:
                try:
                    row = json.loads(str(raw))
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
                if "ego_option" in row:
                    ids.add(int(row["ego_option"]))
    except Exception:
        return None

    return frozenset(ids) if ids else None
```

Notes:
- Use `allow_pickle=False` — the replay writer stores JSON strings, not pickles.
- Silently skip rows that fail to parse (defensive).
- Returns None when absent or empty; callers decide.

### 1.2 Kind-based ego-projected relevance helper

Add:

```python
def _ego_kind_relevant_options(
    factor: FactorSpec,
    options: list[OptionSpec],
    ego_selectable: frozenset[int],
) -> list[int]:
    """v3 relevance: project CE factor to EGO-selectable options by kind.
    factor.option_i is ego-side (CE row), factor.option_j is partner-side (CE col).
    Both are considered as KIND references. The resulting relevance set is:
    every ego-selectable option whose kind matches option_i.kind or option_j.kind,
    plus any ego-selectable option matched by the factor's entity/region tags
    (preserves the entity/region generalization already used by legacy)."""
    if not (0 <= int(factor.option_i) < len(options)):
        return []
    if not (0 <= int(factor.option_j) < len(options)):
        return []
    kind_i = str(options[int(factor.option_i)].kind)
    kind_j = str(options[int(factor.option_j)].kind)
    target_kinds = {kind_i, kind_j}

    entity_ids = set(factor.entity_ids)
    region_ids = set(factor.region_ids)

    relevant: set[int] = set()
    for opt in options:
        oid = int(opt.id)
        if oid not in ego_selectable:
            continue
        if str(opt.kind) in target_kinds:
            relevant.add(oid)
        elif entity_ids.intersection(opt.entity_ids):
            relevant.add(oid)
        elif region_ids.intersection(opt.region_ids):
            relevant.add(oid)
    return sorted(relevant)
```

### 1.3 Extend `make_graph_spec` — split relevance vs route_map paths

Change signature to accept the two new kwargs; the body applies v3 semantics ONLY
to `graph.relevance`, and route_map continues using legacy semantics.

```python
def make_graph_spec(
    layout_name: str,
    options: list[OptionSpec],
    factors: list[FactorSpec],
    *,
    route_source_factors: list[FactorSpec] | None = None,
    relevance_source_factors: list[FactorSpec] | None = None,
    metadata: dict[str, Any] | None = None,
    ego_selectable: frozenset[int] | None = None,
    relevance_semantics: str = "legacy_id_pair",
) -> GraphSpec:
```

Modify the per-factor construction loop:

```python
    def _rel_options_for_q(f: FactorSpec) -> list[int]:
        if relevance_semantics == "ego_kind_projection":
            if ego_selectable is None:
                raise ValueError(
                    "make_graph_spec: relevance_semantics='ego_kind_projection' "
                    "requires ego_selectable set (must not be None)."
                )
            return _ego_kind_relevant_options(f, options, ego_selectable)
        if relevance_semantics == "legacy_id_pair":
            return _relevant_options_for_factor(f, options)
        raise ValueError(
            f"Unknown relevance_semantics {relevance_semantics!r}; expected "
            "'legacy_id_pair' or 'ego_kind_projection'."
        )

    for idx, factor in enumerate(factors):
        relevance_factor = (
            relevance_source_factors[idx] if relevance_source_factors is not None else factor
        )
        # Q-side relevance (may be projected to ego kinds).
        for option_id in _rel_options_for_q(relevance_factor):
            relevance[idx, option_id] = True

        # route_map (evidence-router entity/region routing) ALWAYS uses legacy
        # semantics: option_j is the partner-side option, but the entity/region
        # it references is the correct spatial target for that evidence. Do NOT
        # project route_map, or partner's serve-target evidence gets stripped.
        route_factor = route_source_factors[idx] if route_source_factors is not None else factor
        route_map[idx] = tuple(_relevant_options_for_factor(route_factor, options))
```

### 1.4 Thread through builders

Add `ego_selectable` + `relevance_semantics` kwargs (both with sensible legacy
defaults) to `build_support_graph`, `full_support_graph`, `overcomplete_graph`,
`minus_high_ce_graph`, `minus_serve_soup_graph`, `random_same_size_graph`,
`complete_option_graph`, `shuffled_routes_graph`, `shuffled_relevance_graph`,
and `build_graph_variant`. Pass through to `make_graph_spec` at every callsite.

Default all newly added kwargs to `ego_selectable=None` and
`relevance_semantics="legacy_id_pair"` so v1/v2/standard7 callers see no
behavior change if they don't pass the params.

### 1.5 Record semantics in graph metadata

In every `make_graph_spec` return path, ADD to metadata:
```python
metadata["relevance_semantics"] = str(relevance_semantics)
if ego_selectable is not None:
    metadata["ego_selectable_count"] = int(len(ego_selectable))
    metadata["ego_selectable_ids"] = sorted(int(x) for x in ego_selectable)
```

---

## 2. `experiments/overcooked_v2/train_aris.py`

### 2.1 Import + import surface

Add:
```python
from experiments.overcooked_v2.graph_builder import (
    build_graph_variant,
    validate_task_stage_coverage,
    _load_ego_selectable_from_replay,   # v3
)
```

### 2.2 In `_build_graph` — derive + hard-fail policy

Locate `_build_graph` (around line 800). Right before `build_graph_variant(...)`
is called (around line 823), add:

```python
    relevance_semantics = str(graph_cfg.get("relevance_semantics", "legacy_id_pair"))
    ego_selectable_v3: frozenset[int] | None = None
    if relevance_semantics == "ego_kind_projection":
        ego_selectable_v3 = _load_ego_selectable_from_replay(graph_cfg.get("replay_path"))
        if ego_selectable_v3 is None:
            raise RuntimeError(
                "graph.relevance_semantics='ego_kind_projection' requires replay "
                "rows with ego_option field. Loaded replay_path="
                f"{graph_cfg.get('replay_path')!r} does not provide any. Either "
                "regenerate the CE replay (which stores per-row ego_option) or "
                "switch relevance_semantics back to 'legacy_id_pair'."
            )
```

Then thread both into `build_graph_variant`:

```python
    graph = build_graph_variant(
        args.graph_variant,
        layout_graph.layout_name,
        option_lib.options,
        ce_matrix,
        eta=float(graph_cfg.get("ce_eta", 0.0)),
        max_factors=max_factors,
        full_max_factors=int(graph_cfg.get("full_max_factors", max_factors)),
        overcomplete_extra_factors=int(graph_cfg.get("overcomplete_extra_factors", 0)),
        mode_config=graph_cfg.get("modes"),
        seed=args.seed,
        require_task_stage_coverage=bool(graph_cfg.get("require_task_stage_coverage", True)),
        selection_cfg=graph_cfg,
        ego_selectable=ego_selectable_v3,
        relevance_semantics=relevance_semantics,
    )
```

### 2.3 Objective-metadata gate

The v3 relevance semantics is part of the model's structural objective; a graph
built under v3 should not be reused for a v2/legacy training config and vice
versa. Extend `_expected_graph_objective_metadata` to include the CONFIG value:

```python
    expected["relevance_semantics"] = str(
        (config.get("graph") or {}).get("relevance_semantics", "legacy_id_pair")
    )
```

Extend `_graph_objective_metadata_status` to compare, with a legacy-default
lookup on the observed side so pre-v3 graphs don't get rejected:

```python
        elif key == "relevance_semantics":
            observed = metadata.get(key, "legacy_id_pair")
            matches = str(observed) == str(expected_value)
```

This preserves backward compat: a `standard7` graph produced before v3 has no
`relevance_semantics` key in its metadata, but the observed default is
`"legacy_id_pair"` — matching the expected default from a `standard7` config.

---

## 3. `configs/ocv2_step4_asymm_role_v1.yaml`

Add ONE key under `graph`:

```yaml
graph:
  # ...existing keys...
  relevance_semantics: ego_kind_projection   # v3 fix; legacy_id_pair for v1/v2 provenance
```

Preserve `ce_path` and `replay_path` pointing at `outputs/asymm_ce_role_v1_v2/`
(CE artifact is REUSED — only graph rebuild is required).

---

## 4. Testing

### 4.1 py_compile

Compile every modified file:
```
graph_builder.py, train_aris.py
```
Both must pass without any warning.

### 4.2 Manual relevance sanity — inclusion/exclusion (not exact set)

```python
python -c "
import numpy as np
from experiments.overcooked_v2.env_adapter import OCV2Adapter
from experiments.overcooked_v2.layout_parser import parse_layout
from experiments.overcooked_v2.options import OCV2OptionLibrary
from experiments.overcooked_v2.graph_builder import (
    build_graph_variant, _load_ego_selectable_from_replay,
)

env = OCV2Adapter('asymm_advantages', max_steps=400,
                  observation_type='default', force_path_planning=True)
lg = parse_layout(env.env, 'asymm_advantages')
lib = OCV2OptionLibrary(lg, max_option_steps=16,
                        strict_preconditions=True, dynamic_budget=True)
ce = np.load('outputs/asymm_ce_role_v1_v2/ce_refined.npy')
ego_sel = _load_ego_selectable_from_replay('outputs/asymm_ce_role_v1_v2/replay.npz')
print('ego_selectable size:', len(ego_sel) if ego_sel else 'None')
assert ego_sel is not None, 'ego_selectable derivation failed'

graph = build_graph_variant(
    'full_support', 'asymm_advantages', lib.options, ce,
    eta=0.05, max_factors=16, full_max_factors=16, mode_config=None,
    seed=0, require_task_stage_coverage=True,
    selection_cfg={
        'selection': 'coverage_constrained_ce',
        'required_option_kind_coverage': {
            'pick_plate': {'min_factors': 1},
            'plate_soup': {'min_factors': 1},
            'serve_soup': {'min_factors': 1},
            'clear_interaction_cell': {'min_factors': 1},
        },
        'required_option_id_coverage': {
            'pick_plate': {'min_per_valid_option': 1},
            'plate_soup': {'min_per_valid_option': 1},
            'serve_soup': {'min_per_valid_option': 1},
            'clear_interaction_cell': {'min_per_valid_option': 1},
            'deliver_ingredient_to_pot': {'min_per_valid_option': 1},
        },
    },
    ego_selectable=ego_sel, relevance_semantics='ego_kind_projection',
)

n_opts = len(lib.options)
partner_only = {int(opt.id) for opt in lib.options if int(opt.id) not in ego_sel}
print('partner-only option ids:', sorted(partner_only))

# Global leak check: no factor may route to a partner-only ego action.
leaks = []
for f in range(graph.num_factors):
    for oid in np.flatnonzero(graph.relevance[f]):
        if int(oid) in partner_only:
            leaks.append((f, int(oid), lib.options[int(oid)].kind))
print('relevance leaks to partner-only ids (empty means v3 correct):', leaks)
assert not leaks, 'v3 relevance still routes to partner-only options'

# F1-style check: find a factor with option pair (2:deliver, 9:serve_partner)
# and verify option 8 (ego serve) is in relevance and option 9 is NOT.
serve_partner = 9
serve_ego = 8
target_pair_factors = [
    f for f, fs in enumerate(graph.factors)
    if {int(fs.option_i), int(fs.option_j)} == {2, 9}
]
if target_pair_factors:
    for f in target_pair_factors:
        rel = set(int(x) for x in np.flatnonzero(graph.relevance[f]))
        assert serve_ego in rel, f'factor {f}: ego_serve(8) missing from relevance {rel}'
        assert serve_partner not in rel, f'factor {f}: partner_serve(9) present in relevance {rel}'
        print(f'F{f} (2,9) relevance includes 8: OK, excludes 9: OK, full set: {sorted(rel)}')

# Metadata records semantics
assert graph.metadata.get('relevance_semantics') == 'ego_kind_projection'
print('graph.metadata.relevance_semantics=', graph.metadata.get('relevance_semantics'))
print('graph.metadata.ego_selectable_count=', graph.metadata.get('ego_selectable_count'))
print('ALL CHECKS PASSED')
"
```

Requirements:
- `ego_selectable size` > 0 (must be derived).
- `relevance leaks to partner-only ids` = `[]` (no leak).
- For any factor whose (option_i, option_j) == {2, 9}: option 8 must be present
  in relevance; option 9 must NOT be present. Full set may include other ids
  (e.g., 3:plate_soup via pot entity intersection — that's OK).

### 4.3 Backward-compat: standard7 partner-diff probe

Run the v1 §12.12 pattern with `partner_set=standard7` (no
`relevance_semantics` in that config), assert result unchanged at 4/6 unique
behaviors on asymm_advantages. Because the default is `legacy_id_pair`, this
must be byte-identical to prior behavior.

### 4.4 Hard-fail path

Explicit test: if `relevance_semantics=ego_kind_projection` is set but
`replay_path` is missing / has no `ego_option` rows, `_build_graph` must raise
`RuntimeError` with the message specified in §2.2. Do NOT silently fall back
to legacy semantics.

---

## 5. Deliverables checklist

- [ ] `graph_builder.py`:
      - Add `_load_ego_selectable_from_replay`
      - Add `_ego_kind_relevant_options`
      - Extend `make_graph_spec` signature (`ego_selectable`, `relevance_semantics`)
      - Extend `build_support_graph` / `full_support_graph` / `overcomplete_graph` /
        `minus_high_ce_graph` / `minus_serve_soup_graph` /
        `random_same_size_graph` / `complete_option_graph` /
        `shuffled_routes_graph` / `shuffled_relevance_graph` / `build_graph_variant`
      - Split relevance vs route_map so route_map stays legacy
      - Record `relevance_semantics` + `ego_selectable_count` in graph metadata
      - Raise on `ego_kind_projection` without `ego_selectable`
- [ ] `train_aris.py`:
      - Import `_load_ego_selectable_from_replay`
      - `_build_graph`: derive ego_selectable + hard-fail policy under
        `ego_kind_projection`; thread into `build_graph_variant`
      - `_expected_graph_objective_metadata`: add `relevance_semantics` under
        `graph.` key
      - `_graph_objective_metadata_status`: legacy default `"legacy_id_pair"`
        for missing key so pre-v3 graphs still load
- [ ] `configs/ocv2_step4_asymm_role_v1.yaml`:
      - `graph.relevance_semantics: ego_kind_projection`
- [ ] py_compile both modified source files
- [ ] Manual §4.2 sanity check produces `leaks=[]` AND F1 style factor(s)
      have 8 ∈ relevance, 9 ∉ relevance
- [ ] standard7 partner-diff probe still 4/6 unique
- [ ] `git diff --stat` + per-file 1-line summary + any deviation from spec

**No training / CE / eval runs.** After v3 patch, we'll rerun train+eval using
the existing `outputs/asymm_ce_role_v1_v2/` CE (no CE regen), writing into
`results_role_v1_v3/`.

---

## 7. CE-side partner_terminal_policy threading (path B; new in v3)

Rationale: `_expected_graph_objective_metadata` requires CE metadata to reflect
the `role_contrib_team` objective (see §2.3). Currently ce_sampler calls
`actor_sparse_reward(mode="role_contrib_team", partner_terminal_policy=None, ...)`
which falls through to the unknown-policy fallback (equivalent to contrib_team
semantics). Under path B, CE reward numerics must reflect true role_contrib_team
semantics, so ce_sampler must extract each partner's terminal_policy per row.

### 7.1 Sequential path — `_rollout_option`

**File**: `experiments/overcooked_v2/ce_sampler.py`

At the top of `_rollout_option` (after `opt = option_lib.options[option_id]`
or equivalent), extract:

```python
    partner_terminal_policy = getattr(
        getattr(partner, "protocol", None), "terminal_policy", None
    )
```

Then in the primitive-step loop, modify the `actor_sparse_reward` call to
pass `partner_terminal_policy=partner_terminal_policy`:

```python
        reward_sum += actor_sparse_reward(
            float(step.rewards.get("agent_0", 0.0)),
            event,
            partner_terminal_policy=partner_terminal_policy,
            **{k: v for k, v in _credit.items() if k != "partner_terminal_policy"},
        )
```

The dict-filter is required because `sparse_credit_params()` returns
`partner_terminal_policy: None` in the dict, and we'd double-supply otherwise.

### 7.2 Batched path — `collect_option_replay_batched`

**File**: `experiments/overcooked_v2/ce_sampler.py`

Each `slot.partner` has its own protocol. In the primitive-step processing
where slots' events are batched (around the existing loop that calls
`actor_sparse_reward` per slot), extract per-slot policy:

```python
        partner_terminal_policy_i = getattr(
            getattr(slot.partner, "protocol", None), "terminal_policy", None
        )
        slot.reward_sum += actor_sparse_reward(
            reward_i,
            event,
            partner_terminal_policy=partner_terminal_policy_i,
            **{k: v for k, v in _credit.items() if k != "partner_terminal_policy"},
        )
```

For rows the batched path writes with terminal_progress bonus, only the
sparse credit call needs the policy — the terminal_progress bonus is
actor-local and doesn't depend on partner policy.

### 7.3 No API change to `actor_sparse_reward`

The function already accepts `partner_terminal_policy` as a kwarg (added in v2
§4.1). No signature change needed here — only the callsites in ce_sampler.

---

## 8. `run_ce_pipeline.py` — write role_contrib_team block into CE metadata

**File**: `experiments/overcooked_v2/scripts/run_ce_pipeline.py`

Locate `reward_metadata` construction. After the existing block that adds
`contribution_credit`:
```python
    reward_metadata = {
        ...
        "contribution_credit": {
            "contrib_scale": float(
                (train_cfg.get("contrib_team") or {}).get("contrib_scale", 1.0)
            ),
        },
        ...
    }
```

ADD (inside the `reward_metadata` dict OR immediately after via
`reward_metadata["role_contrib_team"] = ...`):

```python
    if credit_params["mode"] == "role_contrib_team":
        reward_metadata["role_contrib_team"] = {
            "ego_terminal_penalty_under_claim": float(
                (train_cfg.get("role_contrib_team") or {}).get(
                    "ego_terminal_penalty_under_claim", 0.3
                )
            ),
        }
```

Notes:
- Guard on mode so contrib_team runs don't get a stray block.
- Value must match what `sparse_credit_params()` reports as
  `ego_terminal_penalty_under_claim` — both read the same config path.
- After this, CE artifact metadata will contain `role_contrib_team` block →
  training-side `_enforce_graph_objective_metadata` passes → v2 crash resolved.

### 8.1 Metadata gate side (§2.3 refinement)

Given §7 + §8, `_expected_graph_objective_metadata` legitimately requires
`role_contrib_team` block under `role_contrib_team` mode. Do NOT relax this
gate to lenient — with the CE now correctly writing the block, the gate stays
strict and provides real cross-side consistency guarantee. Codex should
verify that the v2-spec gate code (added in v2 §4.5) is still present and
correct, with the observed side reading:

```python
    elif key == "role_contrib_team":
        observed = metadata.get(key, {})
        matches = (
            observed
            and isinstance(observed, dict)
            and float(observed.get("ego_terminal_penalty_under_claim", -999.0))
            == float(expected_value.get("ego_terminal_penalty_under_claim"))
        )
```

If the current v2 gate uses `json.dumps(sort_keys=True)` comparison (like
`terminal_progress_shaping`), that's fine — leave as is. The point is the
comparison must not be structural-inequality-tolerant.

---

## 9. Testing additions (extend §4)

### 9.1 CE side unit-check after §7 + §8

After codex applies §7 + §8, regenerate CE (this IS a required regen under
path B):
```
python experiments/overcooked_v2/scripts/run_ce_pipeline.py \
  --config experiments/overcooked_v2/configs/ocv2_step4_asymm_role_v1.yaml \
  --output_dir outputs/asymm_ce_role_v1_v3 --seed 42 --episodes-per-partner 100
```

After it finishes, verify:
```bash
python -c "
import json
m = json.load(open('outputs/asymm_ce_role_v1_v3/ce_refined.meta.json'))
print('sparse_credit:', m['sparse_credit'])
print('contribution_credit:', m.get('contribution_credit'))
print('role_contrib_team:', m.get('role_contrib_team'))
assert m['sparse_credit'] == 'role_contrib_team'
assert m.get('role_contrib_team', {}).get('ego_terminal_penalty_under_claim') == 0.3
print('CE metadata correct')
"
```

Then run the §4.2 sanity check but against the v3 CE directory:
- `outputs/asymm_ce_role_v1_v3/ce_refined.npy`
- `outputs/asymm_ce_role_v1_v3/replay.npz`

### 9.2 CE regen writes to v3 output dir; config updated

The config `graph.ce_path` and `graph.replay_path` currently point at
`outputs/asymm_ce_role_v1_v2/`. Codex must update these to
`outputs/asymm_ce_role_v1_v3/` in `configs/ocv2_step4_asymm_role_v1.yaml`.
(V2 CE artifact stays intact for provenance.)

---

## 10. Deliverables checklist (updated)

- [ ] All of §5 (relevance fix items)
- [ ] `ce_sampler.py`:
      - `_rollout_option`: extract `partner_terminal_policy` from partner protocol;
        pass to `actor_sparse_reward` (dict-filter to avoid double-supply)
      - `collect_option_replay_batched`: same, per slot
- [ ] `run_ce_pipeline.py`:
      - `reward_metadata` includes `role_contrib_team` block when mode matches
- [ ] `configs/ocv2_step4_asymm_role_v1.yaml`:
      - `graph.ce_path`: `outputs/asymm_ce_role_v1_v3/ce_refined.npy`
      - `graph.replay_path`: `outputs/asymm_ce_role_v1_v3/replay.npz`
      - `graph.relevance_semantics: ego_kind_projection` (from §3)
- [ ] py_compile ce_sampler.py, run_ce_pipeline.py, graph_builder.py, train_aris.py
- [ ] Regen CE into `outputs/asymm_ce_role_v1_v3/` and verify metadata has both
      `contribution_credit` and `role_contrib_team` blocks
- [ ] §4.2 sanity check produces `leaks=[]` on the v3 CE
- [ ] standard7 partner-diff probe still 4/6 unique
- [ ] `git diff --stat` + per-file 1-line summary

---

## 6. Expected result direction (probabilistic, not a promise)

- ARIS `belief_swap_delta.mean_abs_maxq_delta` should rise from ~0 toward a
  non-zero value comparable to `flat_factor` (~0.03-0.07). If it remains ~0
  after v3, the issue is not the relevance mask — investigate
  `FactorLocalQNetwork.forward()` and belief-update dynamics next.
- ARIS `role_match_rate` on the claim partner should rise from 0.80 toward
  1.5+ (baseline range under proper belief routing).
- Baselines (`base_only`, `global_gru`, `flat_factor`) do NOT use the relevance
  mask; their scores should be unchanged. Any change indicates unintended
  side-effect.
