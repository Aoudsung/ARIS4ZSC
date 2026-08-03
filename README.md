# ARIS4ZSC — DELTA-ZSC V6

This branch contains the implementation candidate for **DELTA-ZSC-E2E**, an
end-to-end Bayes-coordination agent for OvercookedV2 zero-shot coordination.

```text
METHOD_VERSION = delta_zsc_v6_end_to_end_bayes_coordination
CONFIG_VERSION = 9
MANIFEST_VERSION = 2
OFFICIAL_PROTOCOL_VERSION = overcooked_v2_iclr2025_5ce1707_v1
```

## Status and evidence boundary

V6 is an implementation candidate. Unit tests, mechanical runs and CUDA
preflights can establish only that the registered computation is executable and
reproducible. They do not establish ZSC effectiveness. Scientific conclusions
require the frozen ten-run Simple and Wide Official matrices, the common-partner
scoreboards and complete resource ledgers.

V4.4 remains reproducible on `codex/path-c-simplification`; its reports are in
[`docs/legacy/v44`](docs/legacy/v44/README.md). V5 r2/r3 remains available in git
history before the V6 replacement commit. V5 checkpoints, optimizers, replay,
generator and calibration artifacts are deliberately incompatible with V6.

## Active method

The deployable policy is one fixed-size path:

```text
legal observation/action/done history
    -> Official CNN + task GRU
    -> diagonal-Gaussian partner belief
    -> one continuously belief-conditioned actor
    -> stochastic environment action
```

The deployment bundle additionally retains one shaped-return value head, twin
raw-return Q heads and the structured partner-response decoder. Training adds a
live and EMA continuous partner generator, EMA policy/Q targets and a softly
weighted counterfactual replay. None of those training-only objects is used as a
partner-ID router.

The primary method has no base/residual split, hard gate, deployment tier,
fallback policy, qualification-controlled loss, target-policy epoch or generator
admission state. C0–C5 labels survive only in the post-training `audit-signals`
report and cannot affect training, checkpoint selection, deployment or formal
samples.

The binding design is
[`docs/theory/DELTA_ZSC_COMPLETE_THEORY_AND_DESIGN.md`](docs/theory/DELTA_ZSC_COMPLETE_THEORY_AND_DESIGN.md).
The fixed benchmark and claim protocol is
[`docs/FORMAL_EXPERIMENT_PROTOCOL.md`](docs/FORMAL_EXPERIMENT_PROTOCOL.md).

## Runtime

Use Python 3.10 and the exact Official source commit
`5ce1707cf31c1c115e6f6ba96db7bc9cc80a850e`. JaxMARL and
`overcooked_v2_experiments` must both come from that clean source tree.

```bash
git clone https://github.com/overcookedv2/experiments.git /path/to/official
git -C /path/to/official checkout 5ce1707cf31c1c115e6f6ba96db7bc9cc80a850e
python -m pip install -e .
python -m pip install --no-deps \
  -e /path/to/official/JaxMARL \
  -e /path/to/official/experiments
python -m compileall -q src/path_c experiments/overcooked_v2
pytest -q experiments/overcooked_v2/tests/test_delta_zsc_*.py
```

Formal and registered preflight workers fail closed unless JAX reports a single
CUDA GPU, `JAX_PLATFORMS=cuda`, healthy dispatcher registration and peak memory
below 40,000 MiB. CPU tests do not substitute for this CUDA acceptance.

## Commands

The V6 method path is:

```text
upstream
build-partner-manifest
validate-manifest
train
evaluate-official
summarize-official
evaluate-common
audit-signals
```

Engineering and optional commands include `mechanical-e2e`, `cuda-preflight`,
`calibrate-safety`, `evaluate-common-br-prox`, baseline reproduction and resource
reporting. `calibrate-safety` creates `DELTA-ZSC-E2E+Safety`; it never replaces or
modifies the primary `DELTA-ZSC-E2E` artifact.

Example mechanical run:

```bash
python -m experiments.overcooked_v2.path_c mechanical-e2e \
  --config experiments/overcooked_v2/configs/delta_zsc_simple_mechanical_e2e.yaml \
  --partner-manifest manifests/simple-engineering.json \
  --require-cuda \
  --output runs/mechanical/v6-e2e
```

Formal-shape one-update preflight:

```bash
python -m experiments.overcooked_v2.path_c cuda-preflight \
  --config experiments/overcooked_v2/configs/delta_zsc_simple_formal.yaml \
  --partner-manifest manifests/simple-engineering.json \
  --seed-index -1 \
  --output runs/preflight/v6-simple
```

Index `-1` is reserved for engineering and is never one of Official seed indices
0–9. After a clean V6 commit is frozen, every formal seed trains and exports the
same final end-to-end algorithm. A numerical failure is reported; it is not
restarted under a changed method and is never replaced with an SP checkpoint.

## Registered per-run simulator budget

For a formal 457-update run, excluding upstream partner training:

| Source | Attempted transitions |
|---|---:|
| Ego PPO | 29,949,952 |
| Ego source initialization | 65,536 |
| Generator training | 5,849,600 |
| Counterfactual continuation | 5,701,632 |
| Matched-code legal probes | 14,848 |
| Generator initialization/holdout | 25,600 |
| Total | 41,607,168 |

The runtime ledger is authoritative and separately records upstream cost,
GPU-hours, peak memory, deployable parameters and training-only parameters.

## Security

Never commit credentials, private keys, run artifacts or checkpoints.
`SSH_Document.md` and `.DS_Store` are explicitly ignored. Server credentials are
operational material outside this repository.
