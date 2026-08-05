# Final DELTA-ZSC revision summary

## Artifact identity

- Source reconstruction base: `149f588d733be48722eea3bff844922f1ed885cc`
- Active method: `delta_joint_response_decision_mirror_voi_v2`
- Active namespace: `src/delta_zsc/`
- Active CLI: `python -m experiments.overcooked_v2.delta_zsc`
- Official benchmark source: `5ce1707cf31c1c115e6f6ba96db7bc9cc80a850e`

This revision replaces the active DEPI v8 patch chain rather than adding another
layer to it. The complete retired implementation remains available under
`legacy/implementation_v8/`, but cannot be imported by package discovery or the
active experiment applications.

## Root-cause closure

| Previously observed root cause | Final correction |
|---|---|
| Eight slots behaved as eight value estimators, not eight coordination modes | A single exchangeable categorical latent state jointly parameterizes one response emission and one decision emission. Components have no separate actors or critics. |
| Value targets were derived from the model's own target-network Q values | Sparse labels are real simulator returns from all-action common-random-number continuations. |
| Response prediction could change a posterior without changing useful action ordering | Response and CRN action-return contrasts are two measurement channels of the same latent component. The latent score is decision-relevant by construction. |
| PPO, response, critic, comparator, and actor objectives shared parameters and Adam moments | `base_params` and `latent_params` have disjoint trees, optimizers, objectives, and static replay paths. |
| Auxiliary adaptation actor and entropy dynamics could collapse the executed policy | Task competence is learned only by base PPO. Adaptation is the analytic solution of a KL-constrained mirror-improvement problem. |
| Training labels and replay features could refer to different continuation policies | Each outer update commits the latent transaction against the exact collection-time base tree before any PPO minibatch changes that tree. |
| Fixed scripts did not pressure a general partner-conditional function | Training support is a manifest-bound, mechanism/family/stage/run-stratified population of independent SP/OP parents and progress checkpoints; calibration and confirmatory lineages are disjoint. |
| Earlier "active" terms were entropy or cross-likelihood proxies | Active DELTA integrates complete learned response outcomes, performs an all-component likelihood calculation and exact Bayes update for every outcome, and values the posterior through latent-conditioned task returns. |
| Old proposal, implementation, evaluation and documentation described different methods | One versioned configuration authority, one active namespace, one CLI, one deployment schema, and one authoritative document index now define the method. |

## Completed VOI v2

For current response posterior `b_t` and candidate probe action `a`, the
implementation:

1. predicts the next component prior `b_bar = b_t T`;
2. sums source components exactly;
3. sums binary visibility exactly;
4. integrates visible relative position, direction, and factorized inventory
   with a deterministic multidimensional Halton rule;
5. sums inventory-change exactly whenever it is legally observable;
6. scores every generated complete response under every latent component;
7. performs a normalized categorical Bayes update;
8. evaluates the posterior-optimal latent-conditioned action value;
9. subtracts the prior-optimal value;
10. reports raw VOI, non-negative control VOI, expected information gain, and
    the nested `S/2` versus `S` quadrature difference.

The control path adds `gamma * max(raw_voi, 0)` to posterior expected action
returns and then solves the registered KL-constrained mirror update. Information
gain and quadrature difference are diagnostics only; neither gates nor rescales
the policy.

The public VOI API accepts either a shared `[..., K, A]` decision matrix or a
probe-conditioned `[..., P, K, A]` matrix. The registered OvercookedV2 model
uses the shared matrix as an explicitly bounded one-response local-stationarity
surrogate; it does not claim exact long-horizon Bayes-adaptive planning.

## Training boundary

One outer update is an alternating estimator transaction:

```text
collect D_n and CRN anchors C_n with base parameters omega_n
    -> update latent parameters Theta_n using stopgrad(omega_n), D_n, C_n
    -> update omega_n with on-policy PPO minibatches from D_n
```

PPO replay uses `compute_latent=False, execute_adaptation=False`. Latent replay
and anchor continuation use `compute_latent=True,
execute_adaptation=False`. Frozen deployment uses both flags. These are static
code paths, not learned gates or research-stage blockers.

## Final active code boundary

```text
src/delta_zsc/
  base_policy.py             task-only recurrent competence
  behavior_statistics.py     analytic Beta posteriors
  transition.py              learned row-stochastic mode dynamics
  response_model.py          complete factorized teammate response
  decision_model.py          CRN return contrasts and covariance score
  belief_filter.py           response-only categorical Bayes filter
  latent_model.py            shared response/decision latent semantics
  bayes_voi.py               completed VOI v2
  mirror_policy.py           analytic KL adaptation
  losses.py                  PPO and latent proper scores
  training.py                separate alternating transactions
  anchors.py                 sparse real all-action continuations
  runner.py                  legal-history rollout
  manifest.py / partners.py  lineage and sampling
  storage.py / resources.py  identity and accounting
```

The active experiment layer contains only manifest construction, training,
deployment, Official baselines/evaluation, posterior diagnostics, belief
intervention, development matrices, resource reporting, and the closed formal
claim synthesis.

## Paper-level claim structure

The repository pre-registers only three ordered confirmatory claims:

1. `H1`: frozen active DELTA exceeds every registered same-protocol baseline on
   both Simple and Wide, with a positive one-sided run-level lower bound and at
   least 20 raw-return points of material effect;
2. `H2`: passive DELTA exceeds response-only under paired seeds on both layouts;
3. `H3`: in a same-world intervention, the correct legal-history belief has
   positive empirical decision value over a task-matched shuffled belief.

The active-versus-passive VOI increment is a pre-registered secondary result.
Posterior entropy, information gain, response NLL, and quadrature error cannot
be promoted into substitute performance claims.

## Evidence boundary

The package contains implementation and CPU-side acceptance evidence, not
formal benchmark results. It therefore makes no SOTA claim. Formal conclusions
require the pinned Python 3.10/Official/CUDA environment, real hash-bound partner
checkpoints, ten ego runs, both roles, 500 episodes per pairing, and the frozen
Simple/Wide statistical protocol.
