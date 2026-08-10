# FORMAL_EXPERIMENT_PROTOCOL — Registered DELTA-ZSC v5 contract

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
- local observation shape: `5x5x39` on Simple and `5x5x43` on Wide;
- Official source identity: registered in `src/delta_zsc/config.py`.

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
- environments: 128;
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

Formal DELTA seed `s` uses development-support SP parent `s` as its base-policy
initializer. Development variants use the same mapping for seeds `0..4`; all
variants sharing a seed receive the same initializer. The engineering
collector and CUDA execution use parent 0.

Initializer calibration, development coverage, and confirmatory panels are
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

The paper-compatible population estimand is a directed `(10,10,500)` cube for
SP, State-Augmented, OP, FCP and DELTA-active. It uses root seed 42, the
Official split into SP/cross branches, and the fixed ordered-cell enumeration.
The common-partner estimand is `(10,16,2,500)` for DELTA-active and all five
registered baselines, using root seed 0. Both paths save the actual two-word
environment keys and use non-permuted OP observations.

Running Wide alone reports only Wide components of the hypotheses. It does not
invoke `formal-claim` or establish full H1/H2/H3 or SOTA.

`grounded_coord_ring` may use the same formal training budget and paper
population estimator for a direct comparison with its published Table 2 row.
That single-layout extension is reported separately and does not enter the
registered Simple/Wide H1/H2/H3 claim.

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

- one visible CUDA GPU, the formal 128-environment shape and peak memory below
  40,000 MiB;
- real Official reset/step and fixed partner checkpoints;
- semantic initializer loading;
- one legal delayed response window;
- one current and successor anchor trigger;
- separate finite latent and PPO updates;
- checkpoint save/restore and deployment export;
- finite exact VOI, action-wise VOI spread, mirror KL, and posterior metrics.

Anchor-index sampling uses exact without-replacement Floyd sampling, so the
formal shape does not materialize a full random sort of all 32,768 rollout
positions. The engineering run verifies that shape on the actual server GPU.

The upstream assets, initializer, engineering execution, development matrix,
formal runs and evaluations execute as one dependency graph. Diagnostic or
return values are reported but never release, block or alter a later job.
Formal claims require complete ten-run matrices and lineage records on both
layouts.
