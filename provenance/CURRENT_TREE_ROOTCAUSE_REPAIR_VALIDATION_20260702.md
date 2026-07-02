# Current-Tree Root-Cause Repair Validation -- 2026-07-02

Branch: `codex/rootcause-current-repair-20260702`

Baseline commit: `01b1a93 wip: pre-fix baseline`

Scope: static checks plus remote non-experiment validation only. No local project
tests/execution, training, evaluation, CE generation, LaTeX, reruns, or experiment
jobs were run.

## Local Static Checks

- `git diff --check`: PASS
- Raw partner option oracle grep over `experiments/overcooked_v2 src/aris_bellman`:
  PASS, no matches for `partner_action.option_id`, `partner_action.option_dist`,
  `partner_action.confidence`, `raw_partner_action.option*`, or `pa.option*`.
- Terminal-policy grep over main train/eval/CE/executor paths:
  remaining matches are limited to partner scripted behavior generation
  (`partner_pool.py`) and the gated/ablation-only `sparse_credit.py` helper.
  Main reward/eval/CE call sites strip `partner_terminal_policy`, and
  `train_aris.py` rejects role-conditioned reward/exploration/replay unless the
  run is explicitly marked `oracle_role_conditioned_ablation=true`.

## Remote Scratch

Host alias: `zsc-customer`

Remote path:
`/apps/users/cxw/Document/CodeSpace/Selfs/ARIS4ZSC-current-rootcause-repair-20260702-codex`

Python:
`/apps/users/cxw/Document/CodeSpace/Selfs/CPR_REPO/.venv/bin/python`

The upload excluded `.git`, `.codex_local_backup`, bulk `review_bundles`,
`outputs`, `results`, `wandb`, and Python cache directories.

## Remote Parse

Command class: AST/YAML/JSON parse only.

Result:

```text
AST_PARSE_OK 72 python files
YAML_PARSE_OK 7 yaml files
JSON_PARSE_OK 6 json files
```

## Remote Pytest

Command:

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 \
  /apps/users/cxw/Document/CodeSpace/Selfs/CPR_REPO/.venv/bin/python -m pytest -q \
  tests/test_event_extractor.py \
  tests/test_ce_sampler_batched.py \
  tests/test_graph_objective_metadata.py \
  experiments/overcooked_v2/tests/test_ocv2_static_invariants.py \
  experiments/overcooked_v2/tests/test_ocv2_p0_p1_fixes.py
```

Result:

```text
54 passed
```

Only warning: `jaxopt` package deprecation warning from the remote environment.

## Interpretation Boundary

This validation supports a Type-A implementation-fidelity handoff for P1/P3/P4/P5,
S17/S20, and NEW-2. It does not validate or refute the ARIS-Bellman scientific
claim. CODEX_IMPL_SPEC v1-v4 outputs remain NEW-4 diagnostic-only material.
