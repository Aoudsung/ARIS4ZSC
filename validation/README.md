# DELTA-ZSC v5 validation artifacts

## Isolated local regression

Run every discovered test function in a fresh bounded CPU process:

```bash
python validation/run_all_isolated_tests.py
```

The runner writes `ISOLATED_TEST_RESULTS.json` and
`LOCAL_TEST_RESULTS.txt`. The current result is:

```text
complete=true
discovered=82
executed=82
passed=82
failed=0
timed_out=0
```

## Contract validation

Run:

```bash
bash validation/run_contract_validation.sh
```

It checks active-source compilation, Python 3.10 grammar, nine registered
configs, the root CLI and 19 subcommands, workflow parsing, namespace and
dependency contracts, credential/local-path scans, source formatting and the
deterministic 66-outcome delayed-VOI diagnostic.

## Evidence boundary

These files certify local implementation contracts only. They do not establish
Official Wide upstream assets, fitted initializers, real CUDA memory or
throughput, development return, formal Wide return, H1/H2/H3 or SOTA.
