# DELTA-ZSC v4 revision summary

## Active identity

- Upstream source: `Aoudsung/ARIS4ZSC`, branch `zsc_v8`, commit
  `15e90b1be0d50ef99df0fa5837312d70fa913643`.
- Local source-archive baseline: `018dd8202abd06e2a685872406c8d4a6e1804e69`.
- Active method:
  `delta_belief_conditioned_raw_return_pairwise_crn_v5`.
- Configuration schema: 3.
- Checkpoint/deployment schema: 4.
- Partner-manifest schema: 2.
- Package version: `0.6.0`.
- Active namespace: `src/delta_zsc/`.
- Active CLI: `python -m experiments.overcooked_v2.delta_zsc`.
- Official benchmark source: `5ce1707cf31c1c115e6f6ba96db7bc9cc80a850e`.

v4 DELTA checkpoints, optimizer state, semantic-initializer artifacts and
deployment bundles are incompatible with v3. Existing Official SP/OP parent
checkpoints and manifest-v2 lineage records remain usable.

## Root cause closed by the implementation

The v3 failure was not an observation-index failure. The aligned interface
target contained convention information, but the model learned a pooled
response instead of partner-semantic components. A physical-time transition
then either erased sparse evidence or, when made sticky, accumulated a
partner-independent likelihood bias. Immediate one-step response also lacked a
causal teammate reaction to the current probe.

v4 changes the estimand and parameterization rather than adding an anti-collapse
penalty:

1. the latent is constant inside an episode and resets only at a true boundary;
2. high-frequency occurrence factors are shared and cannot alter component
   responsibilities;
3. only conditional semantic factors produce Bayes likelihood ratios;
4. response and decision emissions are pooled baselines plus K-centered
   residuals with direct embedding skips and standard fan-in residual
   initialization;
5. the event residual is initialized by an unlabeled, lineage-bound
   spectral-simplex artifact;
6. shared response, semantic response and decision channels are separately
   mean-normalized with fixed coefficient one;
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
    -> current shared decision baseline + centered component residual
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
Adam moments. PPO replay is statically base-only. The latent objective is:

\[
L_{latent}=L_{shared}+L_{semantic}+L_{decision},
\]

where each present channel is divided by its own observation count.

## Calibration and provenance

Development and formal semantic variants require a fitted initializer. The
initializer:

- uses legal response contexts collected by a completed base policy;
- fits a cross-fitted pooled event predictor;
- forms normalized episode residuals;
- uses SVD plus a centered regular simplex;
- records layout, protocol, source commit, component count and lineages;
- rejects SP/OP-label-derived construction and calibration/training parent
  overlap.

K=2, K=4 and K=8 artifacts can be generated from the same calibration panel.

## Validation status

The source package passes the complete isolated local CPU contract suite,
configuration/CLI validation, namespace checks and exact-VOI synthetic
acceptance. The exact counts and environment are recorded in
`VALIDATION_REPORT.md` and `validation/ISOLATED_TEST_RESULTS.json`.

This package does not contain a completed v4 CUDA run, paired development
matrix, formal H1/H2/H3 result, or SOTA claim. The prior v3 CUDA results are
historical failure diagnostics only.

Detailed traceability is in `IMPLEMENTATION_MATRIX.md`.
