# FORMAL_EXPERIMENT_PROTOCOL — Frozen DELTA-ZSC v4 contract

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

## 2. Confirmatory method

```yaml
version: 3
method_variant: delta_active
method:
  latent_components: 4
  continuation_horizon: 128
  adaptation_kl_budget: 0.04
```

The latent is episode-static. Formal training requires a layout-specific,
lineage-bound spectral-simplex initializer artifact. The initializer uses no
partner labels and is part of run identity.

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
- policy/value clip: 0.2;
- entropy coefficient: 0.01;
- value coefficient: 0.5;
- Adam epsilon: 1e-5.

## 4. Sparse current and successor decision observations

- anchor interval: 1,048,576 ordinary environment steps;
- states per trigger: 16;
- fit replicas: 8;
- evaluation replicas: 8;
- continuation horizon: 128;
- current forced actions: 6;
- active probe actions: 6;
- forced post-response actions per probe: 6;
- one unforced collection-time-base bridge per probe replica;
- probe- and bridge-transition rewards excluded from the successor target;
- matched CRN across action alternatives;
- base policy for unforced continuation actions;
- fit/evaluation split fixed before scoring;
- no replay across outer updates;
- full five-dimensional measurement covariance;
- nested `lax.map` bounded-memory execution.

Eight fit replicas satisfy the minimum six needed for a potentially full-rank
five-dimensional sample covariance.

## 5. Delayed response and exact active value

The active target is the legal two-transition window
`o[t+1] -> o[t+2]`, conditioned on probe `a[t]`; `a[t+1]` is used only to remove
its direct ego effect. Either terminal transition invalidates the window.

VOI exactly enumerates the 66 delayed compact outcomes and uses the
probe-conditioned `t+2` decision matrix. Its incremental value is discounted
by `gamma^2`. There is no sample-count field, quadrature estimate, non-negative
clamp, or information-gain reward.

## 6. Partner distributions

Training support contains independent SP and OP parents with checkpoints at
0.0, 0.5, and 1.0. Sampling is uniform over mechanism, family, stage, and run.
Formal support requires at least ten independent parents per mechanism.

Initializer calibration, posterior calibration, and confirmatory panels are
parent- and co-training-lineage disjoint from support and from each other.
Partner manifests are owner-free. Heuristics remain test-only.

## 7. Runs and evaluation

- DELTA training seed indexes: 0..9;
- engineering seed: -1, excluded from science;
- episodes per ego/partner/role pairing: 500;
- both ego roles: required;
- bootstrap replicates: 9,999;
- one-sided alpha: 0.05;
- minimum ego runs: 10;
- minimum independent partner runs per mechanism: 4;
- material effect: 20 raw-return points.

Simple and Wide are inferred separately and combined only by intersection.

## 8. Required artifacts

Every formal run preserves:

- resolved config and method/source identity;
- semantic initializer NPZ/JSON and source metadata;
- lineage-bound partner manifest;
- complete checkpoint descriptor;
- per-update shared/semantic/decision metrics and resource ledger;
- final deployment bundle;
- final current/successor decision and active-policy audit;
- raw evaluation rows;
- posterior diagnostic artifact;
- belief-intervention artifact;
- final three-claim report.

## 9. Execution acceptance

A CUDA preflight is engineering evidence only. It must exercise:

- real Official reset/step and frozen partner checkpoints;
- semantic initializer loading;
- one legal delayed response window;
- one current and successor anchor trigger;
- separate finite latent and PPO updates;
- checkpoint save/restore and deployment export;
- finite exact VOI, action-wise VOI spread, mirror KL, and posterior metrics.

No formal claim is eligible until the ten-run frozen matrices and all lineage
checks are complete.
