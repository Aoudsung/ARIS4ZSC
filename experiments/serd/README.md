# SERD Experiment Bridge

This package contains the first executable bridge layer for:

**SERD: Same-Trajectory Decomposition of Semantic Error Recovery in Zero-Shot Human-AI Coordination**

It is not the full Overcooked/JaxMARL experiment. It is the environment-agnostic metric and sanity harness that the real adapter must call.

## Current Components

- `serd_core.py`: core `SERD_f`, `SERD_worst`, matching, CI, and collapse/survival classification.
- `fixture_env.py`: deterministic branch fixture for CPU sanity checks.
- `run_sanity.py`: CLI that emits JSON and CSV sanity results.
- `overcooked_ai_adapter.py`: MDP-only Overcooked-AI `counter_circuit` branch-record smoke adapter.
- `run_overcooked_ai_smoke.py`: CLI that emits scripted Overcooked-AI branch records and SERD summaries.
- `run_overcooked_ai_m1.py`: CLI that emits policy-driven handcoded Overcooked-AI M1 probes plus preH balance.
- `jaxmarl_overcooked_v2_adapter.py`: narrow-import JaxMARL OvercookedV2 `test_time_simple` branch-record adapter.
- `run_overcooked_v2_m2.py`: CLI that emits policy-driven handcoded OvercookedV2 M2 probes plus preH balance.
- `run_ogc_fcp_remote_smoke.py`: remote-only OGC/FCP source smoke runner for M3 inventory, checkpoint-load, and tiny OGC eval smoke gates.
- `run_ogc_fcp_remote_smoke_sequence.sh`: remote-only wrapper that runs the M3 OGC/FCP inventory, checkpoint-load, and tiny eval-smoke gates in order.
- `run_tomzsc_pecan_counter_circuit_reproduction.sh`: remote-only ToMZSC/JaxMARL PECAN target-domain reproduction script for M3.
- `ogc_fcp_serd_adapter.py`: native OGC FCP BranchRecord adapter implementation, without a claim-bearing M4 bundle runner yet.
- `run_m4_positive_control.py`: deterministic fixture M4 bundle writer used only to prove the metric/bundle path can produce positive `SERD_worst`; it is not FCP, PECAN, or human-coordination evidence.

## Sanity Commands

```bash
python3 -m unittest tests/test_serd_core.py
python3 -m experiments.serd.run_sanity \
  --probes 32 \
  --seed 42 \
  --output-json results/serd_sanity/sanity.json \
  --output-csv results/serd_sanity/family_serd.csv
```

## Overcooked-AI M1 Smoke Command

This command verifies that the SERD adapter can emit `BranchRecord` rows from
the real Overcooked-AI MDP transition logic on `counter_circuit`. It still uses
scripted policies, so it is not evidence about FCP, PECAN, or human coordination.

```bash
git clone --depth 1 https://github.com/HumanCompatibleAI/overcooked_ai.git /private/tmp/overcooked_ai
python3 -m experiments.serd.run_overcooked_ai_smoke \
  --overcooked-src /private/tmp/overcooked_ai/src \
  --layout counter_circuit \
  --probes 8 \
  --horizon 8 \
  --output-json results/serd_overcooked_ai_smoke/smoke.json \
  --branch-csv results/serd_overcooked_ai_smoke/branch_records.csv \
  --family-csv results/serd_overcooked_ai_smoke/family_serd.csv
```

The adapter fails visibly if neither an installed `overcooked-ai` package nor
`--overcooked-src` is available.

## Overcooked-AI M1 Policy-Driven Adapter Command

This command exercises the M1 structural gate on `counter_circuit`: 50 probes
per disruption, complete control-family coverage, and a preH balance table. It
uses a deterministic handcoded policy, so it is still not evidence about FCP,
PECAN, or the final SERD claim.

```bash
python3 -m experiments.serd.run_overcooked_ai_m1 \
  --overcooked-src /private/tmp/overcooked_ai/src \
  --layout counter_circuit \
  --probes-per-disruption 50 \
  --rollout-horizon 20 \
  --warmup-horizon 20 \
  --output-json results/serd_overcooked_ai_m1/m1.json \
  --branch-csv results/serd_overcooked_ai_m1/branch_records.csv \
  --family-csv results/serd_overcooked_ai_m1/family_serd.csv \
  --worst-csv results/serd_overcooked_ai_m1/worst_serd.csv \
  --balance-csv results/serd_overcooked_ai_m1/pre_h_balance.csv
```

## Real Adapter Contract

## JaxMARL OvercookedV2 M2 Policy-Driven Adapter Command

This command exercises the M2 structural gate on OvercookedV2
`test_time_simple`: 50 probes per disruption, complete control-family
coverage, same-seed reset/rollout determinism checks, and a preH balance table.
It uses a deterministic handcoded policy, so it is still not evidence about FCP,
PECAN, or the final SERD claim.

The adapter intentionally loads only the official OvercookedV2 source files
instead of the public `jaxmarl.make` entrypoint, because the public package
imports unrelated environments and optional visualization/physics dependencies.

```bash
git clone --depth 1 https://github.com/FLAIROx/JaxMARL.git /private/tmp/JaxMARL
/private/tmp/aris4zsc_jaxmarl_venv/bin/python -m experiments.serd.run_overcooked_v2_m2 \
  --jaxmarl-src /private/tmp/JaxMARL \
  --layout test_time_simple \
  --probes-per-disruption 50 \
  --rollout-horizon 20 \
  --warmup-horizon 20 \
  --output-json results/serd_overcooked_v2_m2/m2.json \
  --branch-csv results/serd_overcooked_v2_m2/branch_records.csv \
  --family-csv results/serd_overcooked_v2_m2/family_serd.csv \
  --worst-csv results/serd_overcooked_v2_m2/worst_serd.csv \
  --balance-csv results/serd_overcooked_v2_m2/pre_h_balance.csv
```

Current local venv used for this project: `/private/tmp/aris4zsc_jaxmarl_venv`.
It must provide `jax==0.4.38`, `jaxlib==0.4.38`, `chex`, and `flax`.

## M4 Output Bundle Contract

Future claim-bearing M4 runs must satisfy the static contract in:

```text
experiments/serd/m4_output_contract.schema.json
```

The contract mirrors `refine-logs/M4_SERD_PILOT_ACCEPTANCE.md`: a standalone
`worst_serd.csv` is required, canonical `BranchRecord` columns must be used,
all required control families must be present unless a written scope change
exists, and provenance must link each policy/domain pair back to M3 acceptance.
This schema is not a validation result and does not unlock Workflow 2.

Current status as of 2026-06-18 15:55 CST:

- M3 FCP smoke is accepted as `SMOKE_PASSED_NOT_SERD`.
- M3 PECAN target-domain reproduction is accepted as
  `PECAN_TARGET_REPRODUCTION_ACCEPTED_NOT_SERD`.
- M3->M4 transition is `M3_TO_M4_FULL_SCOPE_READY`.
- M4 FCP and PECAN pilot bundles have been accepted for diagnostic review.
- Workflow 2 Round 2 verified documentation cleanup and identified metric-floor
  positive-control evidence as the next blocker before paper-writing.

## M4 Positive-Control Bundle Command

This command is intended for the remote backend under the active no-local-
execution constraint. It writes a full M4-shaped fixture bundle where
`pecan_fixture` is expected to have positive `SERD_worst`.

```bash
cd /apps/users/cxw/ARIS4ZSC
python3 -m experiments.serd.run_m4_positive_control \
  --probes 32 \
  --seed 42 \
  --output-dir results/serd_m4_positive_control/fixture_counter_circuit_20260618_1650
```

This is a metric-floor positive control only. It must not be cited as FCP,
PECAN, human-coordination, survival, superiority, or policy evidence.

## Remote OGC/FCP M3 Smoke Commands

These commands are intended for the SSH backend with the OGC checkout and
`ogc-py310` environment. They are not local tests and should not be run on the
local workspace.

Inventory only, no OGC import:

```bash
cd /apps/users/cxw/ARIS4ZSC
/apps/users/cxw/venvs/ogc-py310/bin/python -m experiments.serd.run_ogc_fcp_remote_smoke \
  --mode inventory \
  --ogc-src /apps/users/cxw/ZSC_coordinator/external/OGC/src \
  --population-json populations/fcp/Overcooked-CounterCircuit6_9/population.json \
  --output-json results/serd_ogc_fcp_remote_smoke/inventory.json
```

Checkpoint/config structure load, no rollout:

```bash
cd /apps/users/cxw/ARIS4ZSC
/apps/users/cxw/venvs/ogc-py310/bin/python -m experiments.serd.run_ogc_fcp_remote_smoke \
  --mode checkpoint-load \
  --ogc-src /apps/users/cxw/ZSC_coordinator/external/OGC/src \
  --population-json populations/fcp/Overcooked-CounterCircuit6_9/population.json \
  --agent-id 1 \
  --band low \
  --output-json results/serd_ogc_fcp_remote_smoke/checkpoint_load.json
```

Tiny OGC rollout/API smoke. By default this infers `--ego-xpid` from the
selected population checkpoint's sibling `xpid.txt`; pass `--ego-xpid` only
when evaluating a different ego policy.

```bash
cd /apps/users/cxw/ARIS4ZSC
/apps/users/cxw/venvs/ogc-py310/bin/python -m experiments.serd.run_ogc_fcp_remote_smoke \
  --mode eval-smoke \
  --ogc-src /apps/users/cxw/ZSC_coordinator/external/OGC/src \
  --population-json populations/fcp/Overcooked-CounterCircuit6_9/population.json \
  --agent-id 1 \
  --band low \
  --checkpoint-name checkpoint \
  --env-names Overcooked-CounterCircuit6_9 \
  --n-episodes 1 \
  --agent-idxs 0 \
  --results-path /apps/users/cxw/ARIS4ZSC/results/serd_ogc_fcp_remote_smoke \
  --results-fname ogc_fcp_eval_smoke \
  --output-json results/serd_ogc_fcp_remote_smoke/eval_smoke.json
```

One-command remote sequence for the default candidate:

```bash
cd /apps/users/cxw/ARIS4ZSC
bash experiments/serd/run_ogc_fcp_remote_smoke_sequence.sh
```

The sequence accepts environment overrides such as `PROJECT_ROOT`, `REMOTE_PY`,
`OGC_SRC`, `POPULATION_JSON`, `OUT_DIR`, `AGENT_ID`, `BAND`, `EGO_XPID`,
`ENV_NAMES`, `N_EPISODES`, and `TIMEOUT_SEC`. If `EGO_XPID` is omitted, the
Python runner still infers it from the selected population checkpoint's
`xpid.txt`.

Passing these smoke gates only establishes that the OGC population/checkpoint
path is usable. It does not emit SERD `BranchRecord` rows and is not evidence
for FCP, PECAN, human coordination, or SERD survival.

The remote runner exits nonzero for incomplete inventory, checkpoint-load
failure, nonzero OGC eval return code, or a missing eval CSV. Treat
`eval_smoke_passed` as valid only together with `output_csv_exists: true`.
After a remote run, use `refine-logs/M3_REMOTE_FCP_SMOKE_ACCEPTANCE.md` to map
the JSON/CSV outputs to tracker status and next-step decisions.

## Remote ToMZSC PECAN M3 Reproduction Command

This command is intended for the SSH backend. It trains/reuses a target-domain
`counter_circuit` teammate pool and `pecan_1m` response policy, then writes the
non-claim-bearing M3 evidence bundle.

```bash
cd /apps/users/cxw/ARIS4ZSC
bash experiments/serd/run_tomzsc_pecan_counter_circuit_reproduction.sh
```

The final accepted rerun used:

```bash
REUSE_EXISTING_CHECKPOINTS=1 PATCH_COUNTER_CIRCUIT_EVAL=1 \
  bash experiments/serd/run_tomzsc_pecan_counter_circuit_reproduction.sh
```

`PATCH_COUNTER_CIRCUIT_EVAL=1` creates a result-local ToMZSC copy for eval
smoke because official ToMZSC eval assumes `state.agent_interact`, which this
`counter_circuit` backend lacks. The shim treats missing `agent_interact` as
zero interaction. This is a documented adapter risk and is not M4 evidence.

## Real Adapter Contract

The Overcooked-AI/JaxMARL adapter should emit `BranchRecord` values:

- same `probe_id` for semantic and control branches;
- same policy/domain/disruption identifiers;
- `no_shock_return`, `branch_return`, and `shock_magnitude`;
- `phi_pre_h` containing only pre-shock and H-step covariates.

Do not include K-step recovery-window covariates in `phi_pre_h`.

Current limitation: the M1 and M2 adapters use handcoded policies. FCP/PECAN
policy loading remains a separate M3 milestone.
