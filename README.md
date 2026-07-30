# ARIS4ZSC — DELTA-ZSC V5

This branch contains the implementation candidate for **DELTA-ZSC: Decision-Equivalent Latent Teammate Adaptation for Zero-Shot Coordination**.

The active method version is:

```text
delta_zsc_v5_decision_equivalent_bayes_r1
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
- `examples/partner_manifest_plan.template.json`: immutable partner-lineage manifest template.
- `docs/legacy/v44/`: archived V4.4 reports and design records.

## Installation and validation

The registered runtime is Python 3.11 with the pinned JAX/Flax/Optax versions in `pyproject.toml`.

```bash
python -m pip install -e .
python -m compileall -q src/path_c experiments/overcooked_v2
pytest -q experiments/overcooked_v2/tests/test_delta_zsc_*.py
python -m experiments.overcooked_v2.path_c --help
```

Static or unit-test success does not establish ZSC effectiveness. Scientific conclusions require frozen, run-disjoint partner manifests and complete training, calibration, and confirmatory evaluation.

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
python -m experiments.overcooked_v2.path_c train ...
python -m experiments.overcooked_v2.path_c calibrate ...
python -m experiments.overcooked_v2.path_c evaluate ...
```

Training, calibration, and confirmatory partners must be checkpoint-, parent-run-, and co-training-group-disjoint. Manifest validation treats any violation as an error.

## Security

Do not commit local server credentials, private keys, run artifacts, or checkpoints. `SSH_Document.md` is explicitly ignored because it is local operational material and must never enter this public repository.
