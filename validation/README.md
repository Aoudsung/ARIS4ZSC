# Validation artifacts

`run_contract_validation.sh` reproduces the non-benchmark acceptance suite:
active compilation, all registered configuration loads, every CLI help path,
workflow YAML parsing, active/legacy namespace checks, and the synthetic VOI
diagnostic.

JAX test files are deliberately executed one at a time through
`run_isolated_test.sh`. This mirrors CI and prevents a CPU worker from retaining
multiple large compilation caches:

```bash
validation/run_isolated_test.sh experiments/overcooked_v2/tests/test_delta_unified_voi.py
validation/run_isolated_test.sh experiments/overcooked_v2/tests/test_delta_unified_core.py
validation/run_isolated_test.sh experiments/overcooked_v2/tests/test_delta_unified_training.py
validation/run_isolated_test.sh experiments/overcooked_v2/tests/test_delta_unified_runner_storage.py
validation/run_isolated_test.sh experiments/overcooked_v2/tests/test_delta_unified_manifest.py
validation/run_isolated_test.sh experiments/overcooked_v2/tests/test_delta_unified_repository.py
```

`generate_voi_diagnostics.py` writes `voi_synthetic_diagnostics.json`. Its three
registered cases distinguish decision value from mere identifiability:

- no response information: VOI = 0 and information gain = 0;
- response reveals a decision-relevant component: VOI > 0;
- response reveals component identity but all components have the same action
  ordering/value: information gain > 0 while VOI = 0.

The checked-in JSON is marked pending until the v3 generator is executed in the
registered environment; the historical v2 local result is not v3 acceptance.

These are implementation acceptance artifacts, not benchmark results. The real
Official/CUDA preflight requires the pinned Python 3.10 environment, a
lineage-bound partner manifest, real checkpoints, and a self-hosted GPU.
