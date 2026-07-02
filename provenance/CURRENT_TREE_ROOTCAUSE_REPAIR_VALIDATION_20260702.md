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

## Reviewer Follow-Up

Reviewer subagent `Sartre` found four blockers after commit `d1633ae`; the
follow-up diff addresses them before final acceptance:

- P3 support masking now closes both producer and consumer sides. `run_ce_pipeline.py`
  writes support-masked `ce_refined.npy` for formal graph/training consumption and
  keeps `ce_refined_unmasked.npy` for audit. `train_aris.py` also reads
  `ce_refined.meta.json` before graph construction and masks `ce_path` again from
  `estimable_mask`, so stale unmasked `.npy` files cannot influence factor selection
  when a support sidecar is present.
- P3 support parameters are enforced, not only recorded. The graph objective gate
  compares config-derived CE `gamma`, `horizon_options`, `min_weight`,
  `reward_objective`, and `support_objective` against both top-level metadata and
  the nested CE support audit.
- P4 persistence remains trainable. Replay stores the hidden state at the start of
  the visible evidence window; TD state construction re-encodes the evidence window
  from that detached base, so the recurrent filter receives TD gradients while
  preserving cross-window history.
- NEW-2 stale deployment artifacts are removed. If the run does not select a
  deployable `checkpoint.pt`, old `checkpoint.pt` / `checkpoint_best.pt` files in a
  reused output directory are unlinked and the removal is recorded in metrics.

After these changes the same remote parse and pytest commands above were rerun with
the same results:

```text
AST_PARSE_OK 72 python files
YAML_PARSE_OK 7 yaml files
JSON_PARSE_OK 6 json files
54 passed
```

## Interpretation Boundary

This validation supports a Type-A implementation-fidelity handoff for P1/P3/P4/P5,
S17/S20, and NEW-2. It does not validate or refute the ARIS-Bellman scientific
claim. CODEX_IMPL_SPEC v1-v4 outputs remain NEW-4 diagnostic-only material.
