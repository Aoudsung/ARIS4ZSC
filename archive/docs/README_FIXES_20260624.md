# OvercookedV2 ARIS-Bellman Root-Fix Bundle — 2026-06-24

This bundle applies root-cause fixes to the OvercookedV2 adapter implementation. The changes deliberately avoid regression-style rollbacks and defensive bypasses.

## Root-cause fixes applied

1. **Preflight is a hard formal-training gate.** `train_aris.py` now requires an accepted preflight report through `--preflight_path` / `preflight.path`. Rejected layouts cannot be bypassed through the trainer; engineering smoke must be run through a separate smoke harness.
2. **CE local returns now match the training reward scale.** `compute_local_returns()` uses `reward_sum + shaped_reward_coef * shaped_reward_sum - cost_coef * realized_cost`.
3. **Noop is excluded from interaction-factor support.** `graph_builder.py` excludes noop option pairs from full, overcomplete, complete, random, shuffled, and derived graph variants.
4. **Task-progress metrics are recorded in training.** `OptionTransition` now carries an event summary; `train_aris.py` aggregates ingredient pickup, pot fill, soup ready, plate pickup, soup pickup, delivery, wrong delivery, collision/block, recipe/button events, noop rate, max-step rate, and task-progress events.
5. **Preflight and layout diagnostics use the same cost/shaped reward scale as CE/training.** `layout_diagnostics.py` forwards `cost_coef` and `shaped_reward_coef` and filters CE stats using non-noop factorable pairs.
6. **Existing P0/P1 fixes are preserved.** Episode-level fixed partners, route-map-aware evidence routing, factor-local prior-centered residuals, bottleneck option runtime semantics, formal graph construction requirements, no G-TVOI/MI selector in training, and criticality-score-only `minus_critical` remain in place.

## Verification performed in this environment

```bash
python -m py_compile $(find src/aris_bellman experiments/overcooked_v2 -name '*.py')
pytest -q experiments/overcooked_v2/tests/test_ocv2_static_invariants.py
```

`py_compile` passed for all ARIS/OvercookedV2 files. The static invariant tests passed locally. Full JaxMARL-dependent tests were not run in this container because `jaxmarl` is not installed here.

## Formal run contract

Formal training now requires:

```bash
python experiments/overcooked_v2/layout_diagnostics.py \
  --config experiments/overcooked_v2/configs/ocv2_debug.yaml \
  --output results/ocv2/preflight.json

python experiments/overcooked_v2/ce_sampler.py collect ...
python experiments/overcooked_v2/ce_sampler.py estimate ...
python experiments/overcooked_v2/graph_builder.py ...

python experiments/overcooked_v2/train_aris.py \
  --config experiments/overcooked_v2/configs/ocv2_debug.yaml \
  --preflight_path results/ocv2/preflight.json \
  --graph_variant full_support \
  --method aris_bellman \
  --seed 0
```

Do not use this trainer for rejected-layout smoke runs. A rejected preflight means the layout/run is not diagnostic-critical for ARIS-Bellman.
