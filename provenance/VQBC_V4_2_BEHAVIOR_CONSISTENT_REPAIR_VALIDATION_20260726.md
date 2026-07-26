# VQBC V4.2 behavior-consistent repair validation

**Date:** 2026-07-26
**Scope:** static/local validation of the V4.2 source revision
**Scientific readout:** prohibited

## Repair identity

This revision replaces V4.1's unconstrained next-action continuation with a belief-conditioned raw-return continuation evaluated under the same KL-regularized policy used by runtime. It also repairs the use/mask physical-world weighting, preserves slot posterior, records stochastic-policy effects, and fixes the matched A1/A2 branch semantics.

## Local environment

Available: Python, JAX, PyYAML, pytest.
Unavailable: Flax, Optax, JaxMARL. Full model/optimizer/checkpoint/environment tests therefore remain remote requirements.

## Executed checks

The following command was executed from the repository root with `PYTHONPATH=.`:

```text
pytest -q \
  experiments/overcooked_v2/tests/test_path_c_vqbc_math.py \
  experiments/overcooked_v2/tests/test_path_c_vqbc_training.py \
  experiments/overcooked_v2/tests/test_path_c_vqbc_contracts.py \
  experiments/overcooked_v2/tests/test_path_c_vqbc_evaluation.py
```

Result:

- collected: 49
- passed: 42
- skipped: 7
- failed: 0
- errors: 0

Skip boundary:

- six tests require Flax/Flax Core;
- one startup-contract test requires JaxMARL.

Additional checks completed:

- bytecode compilation of all VQBC source and three OvercookedV2 VQBC runtimes;
- import of all delayed-dependency VQBC modules;
- parsing of the development config and both source-free formal templates;
- module-registry implementation-file and exact test-ID validation;
- rejection of V4.1 config/checkpoint namespaces;
- executed-response consistency: outcome likelihood, stale-belief targets, and the E-step consume the rollout-recorded response code, while recoded targets supervise only the response encoder/codebook;
- `git diff --check`.

Local test artifacts:

- log SHA-256: `53e1dd2974d73d2d748de0fb1959de52803b2eb9f3bda1199f856920443295f7`
- JUnit XML SHA-256: `1d4870c756b877f7496074be3b5e6152b81683ed6aae280ce0dc06d6f98e6182`

No benchmark result, remote test result, or scientific mechanism result is claimed by this document. The implementation remains `implemented`, not `tested`, until the registered remote CUDA environment completes the full Flax/Optax/JaxMARL suite and the same-budget seed-100 development rerun.
