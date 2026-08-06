# DELTA-ZSC v4 validation artifacts

## Authoritative local regression

Run every discovered test function in a fresh bounded process:

```bash
python validation/run_all_isolated_tests.py
```

The runner writes incremental and final evidence to:

```text
validation/ISOLATED_TEST_RESULTS.json
validation/LOCAL_TEST_RESULTS.txt
```

Each test receives an isolated CPU JAX process with ambient pytest plugins disabled. This avoids retaining heterogeneous XLA executable caches across the full suite and prevents unrelated plugins from changing the result.

The final release result is:

```text
complete=true
discovered=50
executed=50
passed=50
failed=0
timed_out=0
```

`validation/run_isolated_test.sh <test-file>` remains available for file-scoped diagnosis; it also runs each top-level test function in a separate process.

## Contract validation

Run:

```bash
bash validation/run_contract_validation.sh
```

The script reproduces the non-benchmark acceptance suite:

- active source compilation and Python 3.10 grammar compatibility;
- all six version-3 configuration loads;
- top-level CLI and all 16 subcommand help paths;
- workflow YAML parsing;
- active/legacy namespace separation;
- absence of active `src/delta_zsc/transition.py`;
- rejection of retired patch-chain mechanisms;
- credential-pattern and local absolute-path scans;
- exact dependency equality between `pyproject.toml` and `requirements.txt`;
- deterministic 66-outcome delayed-VOI diagnostics.

## Synthetic exact-VOI diagnostics

`validation/generate_voi_diagnostics.py` writes
`validation/voi_synthetic_diagnostics.json`. It checks:

- exactly 66 normalized outcomes per probe/component;
- uninformative delayed response has zero VOI and information gain up to float32 error;
- a decision-revealing response has positive VOI and information gain;
- component-identifying but successor-decision-irrelevant information has positive information gain and zero VOI up to float32 error.

## Evidence boundary

These artifacts certify local implementation contracts only. They do not establish:

- real Official partner-checkpoint compatibility under the pinned CUDA runtime;
- a fitted lineage-disjoint semantic initializer;
- CUDA memory or throughput;
- v4 development return;
- formal H1/H2/H3;
- a SOTA claim.

Those require the registered empirical protocol after this source release.
