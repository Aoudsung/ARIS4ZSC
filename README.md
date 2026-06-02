# SRVF-MAPPO for Classic Overcooked

This repository implements the method specified in `Method_design.md`:
**V0-anchored posterior-predictive SRVF-MAPPO**.

The implementation boundary is explicit:

- the main implementation lives in `raob/srvf_mappo.py`;
- `RolloutBatch` is the MAPPO rollout interface;
- `SourceBatch` is the source IRF likelihood interface;
- SRVF belief updates use only `(g, action, Delta z)`;
- conservative `alpha` is a source-calibrated trust coefficient, not a target-return oracle;
- fallback is population-anchored MAPPO, not random action;
- classic Overcooked-AI is the active benchmark path.

## Install

```bash
pip install -e ".[dev]"
```

## Core Module

`raob/srvf_mappo.py` contains:

```text
RolloutBatch
SourceBatch
IRFTable
NeuralSRVFHeads
SRVFBelief
MAPPOActorCritic
UnifiedLoss
initialize_source_beta
source_table_to_batch
iter_source_batches
collect_classic_rollout_batch
gradient_audit
```

The module can run its own synthetic self-tests, but per project policy those
checks should be executed on the server, not locally.

## Classic Overcooked Dependency

Classic Overcooked is configured on the server from:

```text
external/goat_overcooked/mapbt/envs/overcooked/overcooked_berkeley
```

This is the GOAT repository's Berkeley Overcooked-AI submodule, not
OvercookedV2.

## Execution Constraint

Per `AGENTS.md`, do not run tests, training, evaluation, diagnostics, or runtime
commands locally. Execute validation and formal runs on the server path specified
in `guidance.md`.

Server validation should use:

```bash
cd /apps/users/cxw/Document/CodeSpace/Selfs/RAOB_ZSC
PYTHONPATH=. .venv/bin/python -m compileall -q raob tests
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
```

## Formal Classic Training Budget

`formal-classic` defaults to the paper-style training budget:

```bash
PYTHONPATH=. .venv/bin/python -m raob.srvf_mappo formal-classic \
  --device cuda \
  --policy-device cuda \
  --worker-policy-device cpu \
  --workers 18 \
  --torch-num-threads 1
```

By default, each seed trains until `cumulative_env_steps >= 100000000`.
The loop counts actual rollout rows, so the stopping condition is independent
of early episode termination or rollout batch-size changes. Passing
`--updates N` switches back to fixed-update mode for smoke/debug runs.

## Status

The current implementation exposes the real classic Overcooked-AI rollout
collector, source IRF table construction, SRVF-MAPPO training, and held-out
evaluation through `python -m raob.srvf_mappo formal-classic`.
