# DELTA-ZSC v5 validation report

Date: 2026-08-10

## Scope

This report validates source contracts for
`delta_belief_conditioned_raw_return_pairwise_crn_v5` and the implemented
`test_time_wide` workflow. It does not claim benchmark performance. Earlier
CUDA pilots are not v5 evidence.

## Artifact identity

- Config schema: 3.
- Checkpoint schema: 5.
- Deployment bundle schema: 4.
- Partner and policy manifest schema: 2.
- Semantic initializer schema: 1.
- Package: `aris4zsc==0.6.0`.
- Official source: the registered ICLR 2025 revision.

## Local environment

The isolated regression ran on CPU with Python 3.13.1, JAX/JAXLIB 0.4.38 and
NumPy 2.5.1. The formal environment remains Python 3.10 with the registered
JAX and Official packages. Local CPU execution validates contracts only; it is
not a substitute for the real CUDA execution.

## Executed acceptance paths

### Static and command contracts

`validation/run_contract_validation.sh` covers:

- active source compilation and Python 3.10 grammar compatibility;
- all seven registered configurations across both layouts;
- the root CLI and all 19 subcommand help paths;
- workflow parsing and the single active namespace;
- absence of active retired implementation imports and identifiers;
- dependency equality between `pyproject.toml` and `requirements.txt`;
- credential-pattern and local-path scans;
- regenerated exact 66-outcome delayed-VOI diagnostics.

### Isolated pytest regression

`validation/run_all_isolated_tests.py` runs every top-level test in a fresh,
bounded CPU process. The final run completed in 328.753525 seconds:

- discovered: 82;
- executed: 82;
- passed: 82;
- failed: 0;
- timed out: 0.

Per-file distribution:

| Test file | Count |
|---|---:|
| `test_delta_belief_value.py` | 19 |
| `test_delta_unified_cli.py` | 4 |
| `test_delta_unified_core.py` | 15 |
| `test_delta_unified_manifest.py` | 4 |
| `test_delta_unified_repository.py` | 7 |
| `test_delta_unified_runner_storage.py` | 6 |
| `test_delta_unified_training.py` | 12 |
| `test_delta_unified_voi.py` | 4 |
| `test_delta_v4_semantics.py` | 11 |
| **Total** | **82** |

The machine-readable per-test record is
`validation/ISOLATED_TEST_RESULTS.json`.

## `test_time_wide` layout contracts established locally

The executed tests and static checks establish:

- the Official local observation for the `test_time_wide` layout is handled as
  `5x5x43`, with task
  blocks derived from ingredient count rather than Simple-only offsets;
- the registered partner probabilities remain uniform throughout training;
- development extra-budget controls include pilot, current and successor
  continuation cost;
- development produces exactly 55 commands and maps each seed to one support
  SP initializer shared by every paired variant;
- ordinary formal training accepts exactly seed indexes `0..9`, while the CUDA
  engineering command alone may use `-1`;
- formal shape uses 256 environments and Floyd exact without-replacement
  anchor selection;
- calibration, development coverage and confirmatory parent/co-training
  lineages are disjoint at the manifest boundary;
- Official policy manifests preserve ordered run paths and training lineage;
- the same evaluator supports a paper population cube `(10,10,500)` rooted at
  42 and a common-partner matrix `(10,16,2,500)` rooted at 0;
- raw rows carry actual two-word environment keys, and final summaries compare
  exact schedules across methods;
- OP evaluation explicitly uses non-permuted observations;
- formal evaluation requires one visible GPU and records peak memory against
  the registered 40,000 MiB limit;
- single-layout summaries do not invoke the full claim builder.

The server-side execution adapter completed the full-budget ten-run Official SP
population for the `test_time_wide` layout from one registered root-key split
on two L40s. It produced
all ordered 0.0/0.5/final checkpoints without changing the run keys. A
post-training Orbax restore failed because the saved arrays carried no
checkpoint-side sharding metadata; metadata-directed NumPy restoration fixed
that boundary, and continuation reused all completed checkpoints without
retraining them.

The final-code rerun of the two-seed SP-only pilot completed six development
runs, one K=4 initializer and three matched-key held-out evaluations. Each arm
contains 800 unique raw-return rows. Means were 24.625 (`base`), -2.825
(`response_only`) and 0.300 (`delta_active`). Paired seed differences were
`delta_active-response_only=(1.35, 4.90)`,
`response_only-base=(-15.95, -38.95)` and
`delta_active-base=(-14.60, -34.05)`. Active DELTA improved on response-only in
both seeds, but both semantic arms remained below base. These are descriptive
development observations, not H1/H2/H3 or SOTA evidence.

## Exact-VOI synthetic acceptance

`validation/voi_synthetic_diagnostics.json` records:

- exactly 66 normalized outcomes;
- maximum component outcome-mass error below `1e-6`;
- uninformative delayed response with zero VOI and information gain up to
  float32 error;
- decision-revealing response with positive VOI and information gain;
- component-identifying but decision-irrelevant response with positive
  information gain and zero VOI up to float32 error.

The implementation uses finite enumeration: no quadrature count, value clamp
or information-gain reward enters control.

## Not established by local validation

The following remain server work and are not represented as passed:

1. Official OP support, development-coverage, full confirmatory, baseline and
   FCP populations for the `test_time_wide` layout;
2. the full fitted K=2/4/8 semantic-initializer set for that layout beyond the
   pilot K=4 artifact;
3. real 256-lane CUDA compilation, memory and throughput on the target L40;
4. the 55-run development matrix for the `test_time_wide` layout;
5. ten final DELTA deployments for that layout;
6. five paper population cubes and six common-partner matrices;
7. the `test_time_wide`-layout H1/H2/H3 components or any full
   H1/H2/H3/SOTA statement.

Those questions are answered only by the server artifacts and raw returns.
