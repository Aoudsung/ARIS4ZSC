# ARIS4ZSC — DELTA-ZSC V5

This branch contains the implementation candidate for **DELTA-ZSC: Decision-Equivalent Latent Teammate Adaptation for Zero-Shot Coordination**.

The active method version is:

```text
delta_zsc_v5_decision_equivalent_bayes_r2_official
```

## Status

V5 is an implementation candidate, not a validated research result. The repository provides the model, training, calibration, deployment, and confirmatory-evaluation paths, but no successful end-to-end V5 training run or scientific performance claim is included.

The V4.4 implementation and its negative-mechanism evidence remain available unchanged on [`codex/path-c-simplification`](https://github.com/Aoudsung/ARIS4ZSC/tree/codex/path-c-simplification). Archived V4.4 reports in this branch are under [`docs/legacy/v44`](docs/legacy/v44/README.md).

## Active design

The deployment path has fixed capacity with respect to the number of partners:

```text
legal interaction history
    -> continuous partner-belief posterior
    -> one shared belief-conditioned actor and critic
    -> calibrated conditional residual or robust-base fallback
    -> environment action and posterior update
```

The V5 path does not use discrete critic slots, a response codebook, target-Q use/mask self-distillation, or a per-partner actor library. Counterfactual supervision comes from paired simulator continuations with fit and evaluation replicas kept separate.

The complete mathematical design and proof obligations are documented in [`docs/theory/DELTA_ZSC_COMPLETE_THEORY_AND_DESIGN.md`](docs/theory/DELTA_ZSC_COMPLETE_THEORY_AND_DESIGN.md).

## Repository layout

- `src/path_c/`: active DELTA-ZSC V5 package.
- `experiments/overcooked_v2/`: official environment adapter and V5 command applications.
- `experiments/overcooked_v2/configs/`: Simple/Wide development and formal configurations.
- `experiments/overcooked_v2/tests/`: V5 unit, invariant, and mechanical smoke tests.
- `examples/partner_manifest_plan.template.json`: development-only schema example;
  see `examples/README.md` before constructing formal manifests.
- `docs/legacy/v44/`: archived V4.4 reports and design records.

## Installation and validation

The registered runtime is the Official repository's documented Python 3.10 environment with the pinned JAX/Flax/Optax versions in `pyproject.toml`.

```bash
git clone https://github.com/overcookedv2/experiments.git /path/to/overcookedv2-official
git -C /path/to/overcookedv2-official checkout 5ce1707cf31c1c115e6f6ba96db7bc9cc80a850e
python -m pip install -e .
python -m pip install --no-deps \
  -e /path/to/overcookedv2-official/JaxMARL \
  -e /path/to/overcookedv2-official/experiments
python -m compileall -q src/path_c experiments/overcooked_v2
pytest -q experiments/overcooked_v2/tests/test_delta_zsc_*.py
python -m experiments.overcooked_v2.path_c --help
```

The editable Official checkout is required because the wheel built by the
fixed commit omits its Hydra configuration tree. Runtime validation checks the
checkout commit, cleanliness, source provenance, and the configuration tree.

Static or unit-test success does not establish ZSC effectiveness. Scientific conclusions require frozen, run-disjoint partner manifests, all ten runs on both layouts, complete calibration, both registered scoreboards, the capacity control, resource accounting, and preregistered ablations.

## Formal evidence protocol

The benchmark runtime is fixed to OvercookedV2 Official Experiment commit [`5ce1707`](https://github.com/overcookedv2/experiments/tree/5ce1707cf31c1c115e6f6ba96db7bc9cc80a850e). JaxMARL and `overcooked_v2_experiments` must both be installed from that exact clean source. Formal runs fail closed if provenance differs.

DELTA keeps the full method specified by the theory document: continuous belief, one shared actor/critic, real-return counterfactual anchors, decision-equivalence training, regret-potential shaping, continuous partner generation, and the run-block conformal hard gate. The four published baselines keep their own Official recipes; training pipelines are not artificially made identical.

Two independent scoreboards are mandatory:

1. **Official-Protocol Scoreboard:** reproduce SP, State-Augmented, OP and FCP, then evaluate each method and DELTA with the Official ordered 10×10 matrix, 500 episodes per cell, common episode keys, stochastic final policies, and raw 400-step return.
2. **Common-Partner Scoreboard:** all five methods face the same 16 fresh partners (four independent runs from each of SP, State-Augmented, OP and FCP), in both roles with common episode keys. Local empirical BR-Prox is reported only as a one-action-deviation audit.

IPPO-Large is a separate capacity-matched control, not an Official Table 2 baseline. The complete frozen process and claim boundaries are in [`docs/FORMAL_EXPERIMENT_PROTOCOL.md`](docs/FORMAL_EXPERIMENT_PROTOCOL.md).

## Commands

```bash
python -m experiments.overcooked_v2.path_c build-partner-manifest \
  --plan examples/partner_manifest_plan.json \
  --output manifests/partners.json

python -m experiments.overcooked_v2.path_c validate-manifest \
  --config experiments/overcooked_v2/configs/delta_zsc_simple_development.yaml \
  --partner-manifest manifests/partners.json \
  --run-kind development

python -m experiments.overcooked_v2.path_c upstream ...
python -m experiments.overcooked_v2.path_c train-official-baseline ...
python -m experiments.overcooked_v2.path_c train ...
python -m experiments.overcooked_v2.path_c calibrate ...
python -m experiments.overcooked_v2.path_c build-delta-policy-manifest ...
python -m experiments.overcooked_v2.path_c evaluate-official ...
python -m experiments.overcooked_v2.path_c summarize-official ...
python -m experiments.overcooked_v2.path_c evaluate-common-br-prox ...
python -m experiments.overcooked_v2.path_c evaluate-common ...
python -m experiments.overcooked_v2.path_c summarize-capacity-control ...
python -m experiments.overcooked_v2.path_c summarize-resources ...
python -m experiments.overcooked_v2.path_c build-formal-claim-report ...
```

Training, calibration, and confirmatory partners must be checkpoint-, parent-run-, and co-training-group-disjoint. Manifest validation treats any violation as an error.

Formal baseline and DELTA runs use `split(PRNGKey(42), 10)`, final checkpoints only, and no result-dependent seed deletion or hyperparameter changes. Each DELTA run owns its SP/OP support, generator state, anchors and at least 19 calibration parent runs; none is shared across outer runs. The current formal anchor budget executes 57 triggers × 128 worlds × 6 actions × 96 replicas × 400 steps = **1,680,998,400 attempted branch transitions per DELTA run**. This cost is additional to the 29,949,952 main PPO steps and is never described as budget matched.

## Security

Do not commit local server credentials, private keys, run artifacts, or checkpoints. `SSH_Document.md` is explicitly ignored because it is local operational material and must never enter this public repository.
