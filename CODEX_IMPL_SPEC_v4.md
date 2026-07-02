# Implementation spec v4 — complement-aware relevance + diagnostic fix

Executor: codex. Reviewer: Claude (post-execution audit).

v3 proved the actor-asymmetric CE→relevance bug was the primary cause of ARIS's
claim-side failure. v3 flipped ARIS from `ego=33/prt=16` (抢 terminal) to
`ego=2/prt=31.3` (让 partner 主导). Remaining gap vs `global_gru`/`flat_factor`
(rmr 1.28 vs 1.67 / 1.50) is **support throughput**, not role identification.

v4 targets four specific residual weaknesses:

```
1. v3 ego_kind_projection is TOO NARROW. Factor (2:deliver, 9:partner_serve)
   only routes to {2,3,4,8} — the ego needs SUPPORT actions (fetch, clear,
   drop, wait, cross) to actually help the claim partner deliver more.
2. swap_dQ = 0 in v3 is NOT valid evidence that belief doesn't reach Q.
   The current swap likely gets overwritten by evidence recomputation.
   Diagnostic must be reworked before drawing that conclusion.
3. Don't push ego_terminal_penalty_under_claim below 0.3. ARIS is already too
   conservative (ego=2). Deeper penalty makes throughput worse, not better.
4. Need option-kind breakdown per partner to see where ARIS wastes cycles
   (noop / wait / failed vs fetch / deliver / clear / drop).
```

Constraint recap (unchanged): single TD loss; `argmax Q`; no selector/imitation.
CE reused — no regen. Only graph rebuild + retrain + eval.

Repo: `/Users/aoudsung/Documents/ARIS4ZSC`. Code only.

---

## 0. Design invariants

- CE artifact stays as v3 (`outputs/asymm_ce_role_v1_v3/`). Only relevance mask changes.
- Backward compat: `standard7` byte-identical. v3 configs (`ego_kind_projection`)
  and v1/v2/legacy (`legacy_id_pair`) still work.
- v4 gated behind `graph.relevance_semantics: ego_complement_projection`. Hard-fail
  if invoked without `ego_selectable`.
- All diagnostic additions are non-invasive (new methods, new eval fields);
  legacy training/eval paths unchanged.

---

## 1. `experiments/overcooked_v2/graph_builder.py` — complement projection

### 1.1 Add kind category groups

At top of module (after existing imports, before existing constants):

```python
# Complement-projection kind categories. Terminal actions and support/prep
# actions form the "task chain" the ego navigates around the partner.
_V4_TERMINAL_KINDS: frozenset[str] = frozenset({
    "pick_plate", "plate_soup", "serve_soup",
})
_V4_PREP_KINDS: frozenset[str] = frozenset({
    "fetch_ingredient", "deliver_ingredient_to_pot",
})
_V4_SUPPORT_KINDS: frozenset[str] = frozenset({
    "drop_item_to_counter",
    "clear_interaction_cell",
    "wait_at_bottleneck",
    "cross_bottleneck",
})
```

Import from `partner_pool` where those sets already exist for TERMINAL/PREP/SUPPORT
if the semantics match exactly (partner_pool.py already exposes
`TERMINAL_KINDS/PREP_KINDS/SUPPORT_KINDS`). If they match, IMPORT rather than
duplicate:

```python
from experiments.overcooked_v2.partner_pool import (
    TERMINAL_KINDS as _V4_TERMINAL_KINDS,
    PREP_KINDS as _V4_PREP_KINDS,
    SUPPORT_KINDS as _V4_SUPPORT_KINDS,
)
```

Prefer the import to avoid drift.

### 1.2 Add `_ego_complement_relevant_options`

Place after `_ego_kind_relevant_options` (which stays in place for backward compat
with `ego_kind_projection`):

```python
def _ego_complement_relevant_options(
    factor: FactorSpec,
    options: list[OptionSpec],
    ego_selectable: frozenset[int],
) -> list[int]:
    """v4 relevance: complement projection.

    Given factor (option_i, option_j) — CE row = ego action, CE col = partner
    action — determine the ego-side task-chain category the factor is about
    and route relevance to every ego-selectable option in that category set.

    Categorization rule:
      - If either kind is TERMINAL -> route to TERMINAL + PREP + SUPPORT
      - Else if either kind is PREP -> route to PREP + TERMINAL
      - Else if either kind is SUPPORT -> route to SUPPORT + PREP
      - Else -> fall back to v3 kind-based projection

    Rationale: on a claim partner (partner terminal), the ego needs support
    (fetch, clear) and prep (deliver_to_pot) to keep partner deliveries going,
    not just to refrain from serve_soup. v3 kind projection only routed serve
    factors to {2,3,4,8}, leaving 12/17/19/22/23/25 (drop / clear / wait /
    cross) unreachable through this factor.

    NEVER routes to partner-only ids (ego_selectable filter).
    NEVER routes to noop (excluded from categories).
    """
    if not (0 <= int(factor.option_i) < len(options)):
        return []
    if not (0 <= int(factor.option_j) < len(options)):
        return []

    kind_i = str(options[int(factor.option_i)].kind)
    kind_j = str(options[int(factor.option_j)].kind)
    kinds_in_factor = {kind_i, kind_j}

    # Pick the broadest applicable category set (TERMINAL wins first).
    if kinds_in_factor & _V4_TERMINAL_KINDS:
        target_kinds = _V4_TERMINAL_KINDS | _V4_PREP_KINDS | _V4_SUPPORT_KINDS
    elif kinds_in_factor & _V4_PREP_KINDS:
        target_kinds = _V4_PREP_KINDS | _V4_TERMINAL_KINDS
    elif kinds_in_factor & _V4_SUPPORT_KINDS:
        target_kinds = _V4_SUPPORT_KINDS | _V4_PREP_KINDS
    else:
        # Fallback: replicate v3 ego_kind_projection behavior (kind equality only).
        return _ego_kind_relevant_options(factor, options, ego_selectable)

    entity_ids = set(factor.entity_ids)
    region_ids = set(factor.region_ids)

    relevant: set[int] = set()
    for opt in options:
        oid = int(opt.id)
        if oid not in ego_selectable:
            continue
        if str(opt.kind) == "noop":
            continue
        if str(opt.kind) in target_kinds:
            relevant.add(oid)
        elif entity_ids.intersection(opt.entity_ids):
            relevant.add(oid)
        elif region_ids.intersection(opt.region_ids):
            relevant.add(oid)
    return sorted(relevant)
```

### 1.3 Extend `make_graph_spec` to dispatch on new semantics

Add a branch in the dispatch inside `_rel_options_for_q`:

```python
    def _rel_options_for_q(factor: FactorSpec) -> list[int]:
        if relevance_semantics == "ego_complement_projection":
            if ego_selectable is None:
                raise ValueError(
                    "make_graph_spec: relevance_semantics='ego_complement_projection' "
                    "requires ego_selectable set (must not be None)."
                )
            return _ego_complement_relevant_options(factor, options, ego_selectable)
        if relevance_semantics == "ego_kind_projection":
            if ego_selectable is None:
                raise ValueError(
                    "make_graph_spec: relevance_semantics='ego_kind_projection' "
                    "requires ego_selectable set (must not be None)."
                )
            return _ego_kind_relevant_options(factor, options, ego_selectable)
        if relevance_semantics == "legacy_id_pair":
            return _relevant_options_for_factor(factor, options)
        raise ValueError(
            f"Unknown relevance_semantics {relevance_semantics!r}; expected "
            "'legacy_id_pair', 'ego_kind_projection', or 'ego_complement_projection'."
        )
```

Route_map continues to use `_relevant_options_for_factor` (legacy) — unchanged
from v3.

### 1.4 Metadata records v4 semantics

No change beyond §1.5 of v3 — the metadata already records
`relevance_semantics: str(...)`. New string value flows through.

### 1.5 train_aris.py already dispatches by string

`_build_graph` (train_aris.py:817-828 in v3) already reads
`relevance_semantics = str(graph_cfg.get(...))` and hard-fails when
`ego_selectable` is None. No code change needed in train_aris — the new
`ego_complement_projection` value is dispatched by graph_builder automatically.

Verify the hard-fail message covers the new value too (the current message
mentions only `ego_kind_projection`); update to be neutral:

**File**: `experiments/overcooked_v2/train_aris.py` around line 819-827:

Change:
```python
    if relevance_semantics == "ego_kind_projection":
        ego_selectable_v3 = _load_ego_selectable_from_replay(graph_cfg.get("replay_path"))
        if ego_selectable_v3 is None:
            raise RuntimeError(
                "graph.relevance_semantics='ego_kind_projection' requires ..."
            )
```

To:
```python
    if relevance_semantics in {"ego_kind_projection", "ego_complement_projection"}:
        ego_selectable_v3 = _load_ego_selectable_from_replay(graph_cfg.get("replay_path"))
        if ego_selectable_v3 is None:
            raise RuntimeError(
                f"graph.relevance_semantics={relevance_semantics!r} requires "
                "replay rows with ego_option field. Loaded replay_path="
                f"{graph_cfg.get('replay_path')!r} does not provide any. Either "
                "regenerate the CE replay (which stores per-row ego_option) or "
                "switch relevance_semantics back to 'legacy_id_pair'."
            )
```

---

## 2. `configs/ocv2_step4_asymm_role_v1.yaml` — enable v4 semantics

Change ONE line:

```yaml
graph:
  # ... other keys ...
  relevance_semantics: ego_complement_projection   # v4 (was ego_kind_projection in v3)
```

Optionally add `value_bound.apply_to_all_methods: true` (already present per v2)
— verify still set.

**IMPORTANT**: leave `ce_path`/`replay_path` pointing at
`outputs/asymm_ce_role_v1_v3/`. CE is REUSED. Only graph rebuild required.

Also add a new sub-block for a switch-off ablation (see §4):

```yaml
graph:
  # ...
  relevance_semantics: ego_complement_projection

training:
  # ...
  value_bound:
    enabled: true
    apply_to_all_methods: true
    vmax: 30.0
    base_bound: 15.0
    adv_bound: 15.0
    adv_unit: 1.0
    reward_scale: 1.0
```

Leave value_bound enabled by default for v4; §5 covers the ablation config.

---

## 3. Belief-override diagnostic — fix swap_dQ

v3's `belief_swap_delta` returned 0 across ARIS seeds even though ARIS behavior
changed massively. The current swap replaces belief mid-forward, but evidence
buffer may recompute belief inside the same forward pass, silently overwriting
the override. Confirm + fix by adding a bypass-mode forward.

### 3.1 Investigate `FactorLocalQNetwork.forward`

Locate `FactorLocalQNetwork.forward` (in `src/aris_bellman/factor_q.py` or
`experiments/overcooked_v2/model.py` — search for the class). Determine
whether the belief input is used directly OR recomputed from evidence
mid-forward.

If direct — the swap should work. Then the 0 is a real signal. But add the
`recompute_belief=False` API anyway to make the invariant explicit.

If evidence overrides — expose a `bypass_evidence: bool = False` kwarg that
skips evidence-based recomputation when True.

### 3.2 New API: `forward_with_belief_override`

Add on the ARIS Q network class (whichever handles belief):

```python
def forward_with_belief_override(
    self,
    obs: torch.Tensor,
    belief_override: torch.Tensor,
    *,
    graph_kwargs: dict[str, Any] | None = None,
    bypass_evidence_recompute: bool = True,
) -> torch.Tensor:
    """Forward pass using an EXTERNALLY provided belief tensor. When
    bypass_evidence_recompute=True (default), skip any evidence-buffer belief
    recomputation inside forward — the passed belief is used verbatim.

    Semantic: this is a diagnostic-only path. Training/eval must continue to
    use the normal forward. The purpose is to answer 'does the trained Q head
    actually condition on belief?' independently of whatever the belief buffer
    computes at inference time."""
    graph_kwargs = graph_kwargs or {}
    if bypass_evidence_recompute:
        # Feed belief straight through the model's normal-forward pipeline WITHOUT
        # invoking the belief-computation path (evidence -> belief). Whatever code
        # path produces Q from (obs, belief, graph_kwargs) is what we call here.
        # If the normal forward's belief arg IS already treated as the final belief
        # tensor (no re-derivation), just call normal forward:
        return self.forward(obs, belief_override, **graph_kwargs)
    return self.forward(obs, belief_override, **graph_kwargs)
```

If the normal forward path DOES re-derive belief, split the internal method:

```python
def _forward_from_belief(
    self, obs: torch.Tensor, belief: torch.Tensor, **graph_kwargs
) -> torch.Tensor:
    """Q head + factor advantages given a fixed belief tensor. No evidence."""
    # existing forward internal logic that runs AFTER belief is available
    ...

def forward(self, obs, belief_or_evidence, **graph_kwargs):
    belief = self._compute_belief(belief_or_evidence)  # or pass-through
    return self._forward_from_belief(obs, belief, **graph_kwargs)

def forward_with_belief_override(self, obs, belief_override, **graph_kwargs):
    return self._forward_from_belief(obs, belief_override, **graph_kwargs)
```

Choose whichever fits the actual code structure. Preserve `forward` public
signature and semantics — do not break existing callers.

### 3.3 New eval fields — belief influence decomposition

In `evaluate_aris.py`, extend the per-option diagnostic computation. Add a new
private helper near `_option_diagnostics` or wherever `belief_swap_delta` is
currently computed. For each option decision:

```python
def _belief_influence_decomposition(
    ctx, obs_tensor, belief_actual, graph_kwargs,
    q_actual: torch.Tensor,
) -> dict[str, float]:
    """Decompose belief's influence on Q by comparing forward outputs under
    three counterfactuals. Runs only for ARIS-family methods (belief input);
    baselines return zeros."""
    if not hasattr(ctx.q_net, "forward_with_belief_override"):
        return {
            "belief_zero_delta": 0.0,
            "belief_uniform_delta": 0.0,
            "relevance_zero_delta": 0.0,
        }
    with torch.no_grad():
        # 1. All-zero belief
        belief_zero = torch.zeros_like(belief_actual)
        q_zero = ctx.q_net.forward_with_belief_override(
            obs_tensor, belief_zero, graph_kwargs=graph_kwargs,
        ).squeeze(0)
        # 2. Uniform belief (over valid modes)
        mode_mask = graph_kwargs.get("mode_mask")  # [B, F, M]
        if mode_mask is not None:
            mm = mode_mask.to(dtype=belief_actual.dtype)
            n_valid = mm.sum(dim=-1, keepdim=True).clamp(min=1.0)
            belief_unif = mm / n_valid
        else:
            belief_unif = torch.ones_like(belief_actual) / belief_actual.shape[-1]
        q_unif = ctx.q_net.forward_with_belief_override(
            obs_tensor, belief_unif, graph_kwargs=graph_kwargs,
        ).squeeze(0)
        # 3. All-zero relevance mask (probe factor-Q contribution)
        graph_kwargs_no_rel = dict(graph_kwargs)
        rm = graph_kwargs_no_rel.get("relevance_mask")
        if rm is not None:
            graph_kwargs_no_rel["relevance_mask"] = torch.zeros_like(rm)
        q_no_rel = ctx.q_net.forward_with_belief_override(
            obs_tensor, belief_actual, graph_kwargs=graph_kwargs_no_rel,
        ).squeeze(0)
    return {
        "belief_zero_delta": float((q_actual - q_zero).abs().max().item()),
        "belief_uniform_delta": float((q_actual - q_unif).abs().max().item()),
        "relevance_zero_delta": float((q_actual - q_no_rel).abs().max().item()),
    }
```

Then in the eval loop where the current `belief_swap_delta` is computed (search
for the callsite in `evaluate_aris.py`), also call this decomposition and
accumulate its values per episode + per aggregate. Add to `aggregate`:

```python
aggregate["belief_influence"] = {
    "mean_belief_zero_delta": ...,
    "mean_belief_uniform_delta": ...,
    "mean_relevance_zero_delta": ...,
}
```

Non-ARIS methods will show zeros — that's expected.

Interpretation rules for post-v4 reading:
```
if belief_zero_delta > 0.05 : ARIS Q genuinely conditions on belief.
if belief_zero_delta ~ 0 and behavior differs across seeds:
    belief is NOT the driver; training-time weight change explains delta.
if relevance_zero_delta > 0.05 : relevance-masked factor advantages contribute.
```

---

## 4. Claim-side option breakdown per-partner

`evaluate_aris.py` already computes `option_kind_stats` per aggregate, but not
split by partner. Under v4 we need per-partner breakdown to compare where ARIS
vs `global_gru` vs `flat_factor` spend cycles on the claim partner.

The per-partner split is ALREADY in place — each result in `eval_heldout.json`
has its own `aggregate.option_kind_stats`. What's missing is a parser convenience
in the report script.

**No code change to evaluate_aris.py required for this.** Instead, extend the
parser (`parse_role_v1_v4.py` — new file) to print, for the claim partner:

```
method       total_opts  fetch  deliver  pick  plate  serve  clear  drop  wait  cross  noop  timeout%
aris_bellman ...          ...    ...      ...   ...    ...    ...    ...   ...   ...    ...   ...
global_gru   ...          ...    ...      ...   ...    ...    ...    ...   ...   ...    ...   ...
```

This is a report-side improvement, not a code-behavior change.

---

## 5. Value bound ablation config

Per user's §5, verify the v2 config setting `apply_to_all_methods: true` is
still active AND add a v4 ablation config `ocv2_step4_asymm_role_v1_novb.yaml`
identical to the main config except:

```yaml
training:
  value_bound:
    enabled: false
```

This gives us a matched-baseline check: if disabling value_bound closes the
ARIS gap further, the bound (even with apply_to_all=true) was still a limiter.

Include only ARIS + one baseline (say `flat_factor`) in the ablation run to
save compute; not the full 4-method sweep.

---

## 6. Testing after edits

### 6.1 py_compile

```
graph_builder.py, train_aris.py (if edited for §1.5 message)
evaluate_aris.py (§3.3)
src/aris_bellman/factor_q.py or model.py (§3.2)
```

All must pass.

### 6.2 Sanity check: v4 relevance projection

```python
python -c "
import numpy as np
from experiments.overcooked_v2.env_adapter import OCV2Adapter
from experiments.overcooked_v2.layout_parser import parse_layout
from experiments.overcooked_v2.options import OCV2OptionLibrary
from experiments.overcooked_v2.graph_builder import (
    build_graph_variant, _load_ego_selectable_from_replay,
)

env = OCV2Adapter('asymm_advantages', max_steps=400, observation_type='default', force_path_planning=True)
lg = parse_layout(env.env, 'asymm_advantages')
lib = OCV2OptionLibrary(lg, max_option_steps=16, strict_preconditions=True, dynamic_budget=True)
ce = np.load('outputs/asymm_ce_role_v1_v3/ce_refined.npy')
ego_sel = _load_ego_selectable_from_replay('outputs/asymm_ce_role_v1_v3/replay.npz')
assert ego_sel is not None

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
    ego_selectable=ego_sel, relevance_semantics='ego_complement_projection',
)

# Leak check (must still hold)
partner_only = {int(opt.id) for opt in lib.options if int(opt.id) not in ego_sel}
leaks = []
for f in range(graph.num_factors):
    for oid in np.flatnonzero(graph.relevance[f]):
        if int(oid) in partner_only:
            leaks.append((f, int(oid), lib.options[int(oid)].kind))
print('leaks (must be empty):', leaks)
assert not leaks

# Complement check: for factors touching partner terminal (option_j==9 or ==8),
# relevance must include AT LEAST ONE support-kind ego option AND AT LEAST ONE
# terminal-kind ego option.
SUPPORT_KINDS = {'clear_interaction_cell','wait_at_bottleneck','cross_bottleneck','drop_item_to_counter'}
TERMINAL_KINDS = {'pick_plate','plate_soup','serve_soup'}
terminal_factors_checked = 0
for f, fs in enumerate(graph.factors):
    partner_kind = str(lib.options[int(fs.option_j)].kind)
    ego_kind = str(lib.options[int(fs.option_i)].kind)
    if not ({partner_kind, ego_kind} & TERMINAL_KINDS):
        continue
    rel_kinds = {str(lib.options[int(oid)].kind) for oid in np.flatnonzero(graph.relevance[f])}
    has_support = bool(rel_kinds & SUPPORT_KINDS)
    has_terminal = bool(rel_kinds & TERMINAL_KINDS)
    if not (has_support and has_terminal):
        print(f'FACTOR {f}: touches terminal but rel_kinds={sorted(rel_kinds)} — missing support or terminal')
        continue
    terminal_factors_checked += 1
print('terminal-touching factors with full support+terminal ego coverage:', terminal_factors_checked)
assert terminal_factors_checked > 0, 'no terminal-adjacent factor gets complement projection'

# F1-style factor (2, 9): must include support kinds now, not just deliver+serve
target_pairs = [f for f, fs in enumerate(graph.factors) if {int(fs.option_i), int(fs.option_j)} == {2, 9}]
if target_pairs:
    for f in target_pairs:
        rel = {int(x) for x in np.flatnonzero(graph.relevance[f])}
        rel_kinds = {str(lib.options[oid].kind) for oid in rel}
        print(f'F{f} (2,9) rel kinds under complement projection:', sorted(rel_kinds))
        assert 9 not in rel, f'partner_serve id 9 still in relevance {rel}'
        assert SUPPORT_KINDS & rel_kinds, f'no support kind in relevance {rel_kinds}'

# Metadata records v4 semantics
assert graph.metadata['relevance_semantics'] == 'ego_complement_projection'
print('v4 SANITY PASSED')
"
```

Expected:
- `leaks: []`
- terminal-touching factors: all pass the "has support AND has terminal" check
- F1 (2,9) rel_kinds now includes support kinds (clear/wait/cross/drop) plus prep+terminal

### 6.3 Standard7 backward compat

Same probe as v3 §4.3 — must remain 4/6 unique on asymm_advantages.

### 6.4 Diagnostic sanity

Small runtime check: call
`forward_with_belief_override(bypass_evidence_recompute=True)` with a zeros
belief and confirm Q output differs from the actual-belief Q — indicating the
forward pass uses the belief argument.

---

## 7. Deliverables checklist

- [ ] `graph_builder.py`:
      - kind-category imports (TERMINAL/PREP/SUPPORT from partner_pool)
      - `_ego_complement_relevant_options`
      - `make_graph_spec` dispatches on new value
- [ ] `train_aris.py`:
      - `_build_graph` hard-fail message covers `ego_complement_projection`
- [ ] `evaluate_aris.py`:
      - Add `_belief_influence_decomposition`
      - Add `aggregate["belief_influence"]` fields
- [ ] `src/aris_bellman/factor_q.py` or `experiments/overcooked_v2/model.py`:
      - Add `forward_with_belief_override(bypass_evidence_recompute=True)`
- [ ] `configs/ocv2_step4_asymm_role_v1.yaml`:
      - `graph.relevance_semantics: ego_complement_projection`
      - `value_bound.apply_to_all_methods: true` (verify already present)
- [ ] `configs/ocv2_step4_asymm_role_v1_novb.yaml`: v4 config with value_bound
      disabled (ablation)
- [ ] py_compile all modified
- [ ] §6.2 v4 sanity check: leaks=[], terminal factors get complement coverage,
      F1 (2,9) includes support kinds
- [ ] §6.3 standard7 probe: 4/6 unique
- [ ] §6.4 belief-override diagnostic sanity
- [ ] `git diff --stat` + per-file 1-line summary

**No training / CE / eval runs.** After v4 patch, pipeline:
1. Rebuild graphs (train stage runs graph_builder each time, so nothing to
   pre-build) — just retrain 12 runs on the v3 CE
2. Eval 12 runs with new `belief_influence` fields
3. Optionally run ARIS + flat_factor on the `novb` config for value_bound ablation

---

## 8. Expected result direction (probabilistic)

**Primary target**: ARIS claim-side ego_deliv rises from 2.0 back toward 5-15
(matching or exceeding the throughput of global_gru's 13.3), while
partner_deliv stays high. Total throughput on claim partner should rise
from ARIS's current 33.3 toward 40-60 (matching/exceeding global_gru's 60).

**Secondary target**: `belief_zero_delta > 0.05` for ARIS. If the new
diagnostic still returns 0, we have direct evidence that ARIS's trained Q
doesn't use belief at inference — a deeper mechanism issue.

**Falsification**: if v4 relevance widening doesn't change ARIS throughput,
the remaining gap is in the Q network's ability to exploit the wider relevance
mask — factor-Q architecture concern, not relevance geometry.

**Baseline preservation**: `global_gru`/`flat_factor`/`base_only` scores
should not change materially (they don't use relevance mask). Small drifts OK
from training stochasticity, but no systematic shift.
