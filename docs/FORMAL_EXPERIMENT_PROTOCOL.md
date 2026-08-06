# FORMAL_EXPERIMENT_PROTOCOL — Frozen confirmatory contract

`authoritative: true`

## 1. Environment

- layouts: `test_time_simple`, `test_time_wide`;
- episode length: 400;
- action count: 6;
- view radius: 2;
- negative rewards: enabled;
- random initial positions: enabled;
- recipe resampling after delivery: enabled;
- successful-delivery indicator: enabled;
- Official source identity: fixed in `src/delta_zsc/config.py`.

## 2. Confirmatory DELTA

```yaml
method_variant: delta_active
method:
  latent_components: 4
  continuation_horizon: 128
  adaptation_kl_budget: 0.04
```

Active VOI exactly enumerates the registered 66-outcome compact response
marginal. There is no VOI sample-count setting, quadrature diagnostic, or
non-negative control clamp.

## 3. Base PPO

- ordinary environment steps: 29,949,952;
- environments: 256;
- rollout length: 256;
- update epochs: 4;
- minibatches: 64;
- learning rate: 0.00025;
- warm-up fraction: 0.05;
- cosine annealing: enabled;
- gradient norm: 0.25;
- gamma: 0.99;
- GAE lambda: 0.95;
- clip/value-clip: 0.2;
- entropy coefficient: 0.01;
- value coefficient: 0.5;
- Adam epsilon: 1e-5.

## 4. Sparse decision observations

- anchor interval: 1,048,576 ordinary environment steps;
- states per trigger: 16;
- fit replicas: 4;
- evaluation replicas: 8;
- continuation horizon: 128;
- six forced ego actions;
- base policy for first-action branches and all continuation actions;
- matched CRN across action branches;
- fit/evaluation replica split fixed before model scoring;
- no replay across outer updates.

## 5. Partner distributions

Training support contains independent SP and OP parent runs, each with progress
checkpoints 0, 0.5, and 1.0. Sampling is uniform over mechanism, then
hyperparameter family, stage, and run. Formal support requires at least ten
independent parents per mechanism.

Calibration and confirmatory panels are parent- and co-training-lineage disjoint
from support and from each other. The manifest is lineage-bound and owner-free.
Heuristics remain test-only.

## 6. Runs and evaluation

- training runs: seed indexes 0..9;
- engineering seed: -1, excluded from scientific summaries;
- episodes per ego/partner/role pairing: 500;
- both ego roles: required;
- bootstrap replicates: 9,999;
- one-sided alpha: 0.05;
- minimum ego runs: 10;
- minimum independent partner runs per mechanism: 4;
- material effect: 20 raw-return points.

Simple and Wide are tested separately and combined only by intersection.

## 7. Acceptance artifacts

Every formal run must preserve:

- resolved config;
- method/source identity;
- lineage-bound partner manifest;
- complete checkpoint descriptor;
- per-update metrics and resource ledger;
- final deployment bundle;
- raw evaluation rows;
- posterior diagnostic artifact;
- belief-intervention artifact;
- final three-claim report.

A CUDA preflight is an execution acceptance test, not a scientific result. It
must exercise one real update, an anchor trigger, separate base/latent updates,
checkpoint restore, deployment export, and finite VOI/KL diagnostics.
