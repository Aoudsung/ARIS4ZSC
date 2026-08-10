# DELTA-ZSC v5 `test_time_wide` layout execution revision summary

## Active identity

- Upstream source: `Aoudsung/ARIS4ZSC`, branch `zsc_v8`.
- Active method:
  `delta_belief_conditioned_raw_return_pairwise_crn_v5`.
- Configuration schema: 3.
- Checkpoint schema: 5.
- Deployment bundle schema: 4.
- Partner-manifest schema: 2.
- Package version: `0.6.0`.
- Active namespace: `src/delta_zsc/`.
- Active CLI: `python -m experiments.overcooked_v2.delta_zsc`.
- Official benchmark source: the registered ICLR 2025 revision.

v5 DELTA checkpoints, optimizer state, semantic-initializer artifacts and
deployment bundles are incompatible with v4 and earlier artifacts. Existing Official parent
checkpoints and manifest-v2 lineage records remain usable.

## Root cause closed by the implementation

The v3 failure was not an observation-index failure. The aligned interface
target contained convention information, but the model learned a pooled
response instead of partner-semantic components. A physical-time transition
then either erased sparse evidence or, when made sticky, accumulated a
partner-independent likelihood bias. Immediate one-step response also lacked a
causal teammate reaction to the current probe.

v5 changes the estimand and parameterization rather than adding an anti-collapse
penalty:

1. the latent is constant inside an episode and resets only at a true boundary;
2. high-frequency occurrence factors are shared and cannot alter component
   responsibilities;
3. only conditional semantic factors produce Bayes likelihood ratios;
4. response emissions are pooled baselines plus K-centered residuals with
   direct embedding skips, while decision value is learned directly as a
   belief-conditioned dueling raw-return critic;
5. the event residual is initialized by an unlabeled, lineage-bound
   spectral-simplex artifact;
6. every rollout supplies a raw-reward TD(lambda) target and sparse anchors
   calibrate pairwise CRN action differences without a tunable loss weight;
7. active information is delayed until the teammate has had one reaction step;
8. successor decision values are conditioned on the probe and evaluated at
   t+2 after one base-policy bridge;
9. the active controller uses an exact 66-outcome finite Bayesian value and a
   two-step discount.

No partner identity, SP/OP label, component-separation loss, entropy gate,
component-specific actor, or component-specific critic is introduced.

## Final method path

```text
legal immediate response
    -> shared occurrence score (prediction only)
    -> component-semantic likelihood
    -> episode-static posterior
    -> belief-conditioned raw-return action value
    -> pairwise-CRN-calibrated action ordering
    -> passive KL mirror update

candidate probe at t
    -> collection-time base bridge at t+1
    -> delayed response o[t+1] -> o[t+2]
    -> exact 66-outcome Bayes update
    -> probe-conditioned t+2 all-action value
    -> gamma^2 VOI
    -> active KL mirror update
```

The active probe commits exactly one base bridge before a new posterior-dependent
action can be taken. The CRN estimator uses the same timing and excludes probe
and bridge rewards from the successor target.

## Optimization contract

One outer transaction is:

```text
collect D_n and sparse CRN anchors C_n with base omega_n
    -> update latent Theta_n using D_n, C_n and stopgrad(omega_n)
    -> update base omega_n with PPO minibatches from D_n
```

`base_params` and `latent_params` have separate parameter trees, optimizers and
Adam moments. PPO replay is statically base-only. The latent transaction is:

\[
L_{latent}=L_{shared\ NLL}+L_{semantic\ NLL}
+L_{raw\ TD}+L_{pairwise\ CRN}+L_{successor},
\]

where every present term has fixed coefficient one and the response proper
scores are divided by their own observation counts.

## Calibration and provenance

Development and formal semantic variants require a fitted initializer. The
initializer:

- uses legal response contexts collected by a completed base policy;
- fits a cross-fitted pooled event predictor;
- forms normalized episode residuals;
- uses SVD plus a centered regular simplex;
- records layout, protocol, registered source revision, component count and
  lineages;
- rejects SP/OP-label-derived construction and calibration/training parent
  overlap.

K=2, K=4 and K=8 artifacts can be generated from the same calibration panel.

## Validation status

The source package passes the complete isolated local CPU contract suite,
configuration/CLI validation, namespace checks and exact-VOI synthetic
acceptance. The exact counts and environment are recorded in
`VALIDATION_REPORT.md` and `validation/ISOLATED_TEST_RESULTS.json`.

This package does not yet contain a completed v5 CUDA run, paired development
matrix, formal H1/H2/H3 result, or SOTA claim. The prior v3 CUDA results are
historical failure diagnostics only.

Detailed traceability is in `IMPLEMENTATION_MATRIX.md`.

## `test_time_wide` layout execution closure

The `test_time_wide` implementation now includes:

- layout-derived extraction for the Official `5x5x43` observation;
- uniform registered partner sampling for the complete training run;
- exact development extra-budget accounting including pilot continuations;
- a seed-index SP initializer mapping shared by paired variants;
- the 128-environment formal shape and exact bounded-state Floyd
  anchor sampling that removes the prior L40 full-sort compile blocker;
- formal seed `0..9` enforcement with `-1` reserved for the CUDA engineering
  command;
- disjoint calibration, development-coverage and confirmatory lineage;
- version-2 Official policy manifests carrying training lineage;
- one static Official upstream DAG for support, panels, four baseline
  populations and the required FCP source populations;
- memory-bounded Official population execution that retains the single
  root-key split while scanning one run per visible device, so the registered
  ten-run population fits on one L40 without changing any run seed or budget;
- metadata-directed Orbax restore for Official arrays without checkpoint-side
  sharding metadata, plus direct continuation from already completed run
  aliases after a post-training process interruption;
- a paper-compatible `(10,10,500)` population evaluator rooted at 42 and a
  common-partner `(10,16,2,500)` evaluator rooted at 0;
- exact cross-method environment-key schedule comparison, non-permuted OP
  observations and formal evaluation memory accounting.

All full-workflow roots and panel allocations are selected before execution.
The implemented publication workflow remains available, but the completed
server schedule covered only a smaller SP-only signal pilot: two paired seeds
of `base`, `response_only` and `delta_active`, measured on two parent-disjoint
SP partners. It did not invoke `formal-claim` or state complete H1/H2/H3 or
SOTA.

The final-code rerun produced held-out means of 24.625 (`base`), -2.825
(`response_only`) and 0.300 (`delta_active`). The paired
`delta_active-response_only` differences were 1.35 and 4.90;
`response_only-base` was -15.95 and -38.95, and `delta_active-base` was -14.60
and -34.05. These are descriptive two-seed development results on the
`test_time_wide` layout only.
