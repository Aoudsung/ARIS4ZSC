# CETR-ZSC validation entry points

## Isolated local regression

The repository runner discovers every top-level test function in
`experiments/overcooked_v2/tests/test_cetr_*.py` and runs each in a fresh
bounded CPU process:

```bash
python validation/run_all_isolated_tests.py
```

The runner writes `ISOLATED_TEST_RESULTS.json` and
`LOCAL_TEST_RESULTS.txt`. These generated outputs are not source-tree evidence
and are absent until a user intentionally runs the runner.

## Contract validation

The former version-specific contract script was removed because it only
validated retired implementation paths, configurations, diagnostics, and CLI
commands. The active contracts are protected by
`experiments/overcooked_v2/tests/test_cetr_repository.py` and the CETR CI
workflow.

## Evidence boundary

Source-level checks certify repository wiring only. They do not establish
Official upstream assets, fitted initializers, real CUDA memory or throughput,
development return, formal return, or any performance claim.
