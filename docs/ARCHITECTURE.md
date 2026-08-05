# ARCHITECTURE — Equation-to-code map and ownership boundary

| Scientific object | Active implementation |
|---|---|
| task-only recurrent state | `src/delta_zsc/base_policy.py` |
| current teammate geometry | `src/delta_zsc/observation.py`, `base_policy.py` |
| analytic behavior posterior | `behavior_statistics.py` |
| row-stochastic latent dynamics | `transition.py` |
| complete response emission | `response_model.py` |
| CRN decision emission | `decision_model.py` |
| response-only exact filter | `belief_filter.py`, `latent_model.py` |
| KL mirror policy | `mirror_policy.py` |
| Rao-Blackwellized Halton/Bayes VOI | `bayes_voi.py` |
| sparse all-action continuations | `anchors.py` |
| PPO and latent scores | `losses.py` |
| separate optimizer transactions | `training.py` |
| legal rollout state machine | `runner.py` |
| partner sampling and lineage | `partners.py`, `manifest.py` |
| checkpoint identity | `storage.py` |
| deployment bundle | `experiments/overcooked_v2/deployment.py` |
| Official evaluator adapter | `official_policy.py` |
| end-to-end lifecycle | `training_app.py`, `evaluation_app.py` |
| same-world belief intervention | `intervention_app.py` |
| three-claim synthesis | `formal_claim_app.py` |

## Parameter ownership

```text
base_params
  task encoder, task GRU, instantaneous geometry encoder, actor, value
  <- PPO only

latent_params
  transition logits, component embeddings, response emission, decision emission
  <- one response/decision predictive score only
```

The parameter trees and Adam states are separate. No loss-name routing table can
silently transfer gradients between them. Within each outer update the latent
transaction consumes the exact collection-time base tree before PPO commits any
base-policy minibatch update, preserving the CRN continuation estimand.

## Runtime state

`PolicyState` contains only:

- base task carry;
- categorical response belief;
- analytic behavior counts;
- previous legal observation;
- previous ego action;
- episode-start flag.

Partner IDs, counterfactual returns, comparator state, codebooks, calibration
models, and paper diagnostics are absent.

## Static execution modes

The same `DeltaModel` exposes three static paths:

```text
base-only PPO replay
  compute_latent = false
  execute_adaptation = false

latent estimation / CRN continuation
  compute_latent = true
  execute_adaptation = false

frozen deployment
  compute_latent = true
  execute_adaptation = true
```

This division is a code boundary, not an adaptive gate. It keeps PPO compilation
small and guarantees that data collection and anchor continuations are generated
by the registered base policy.

## Legacy boundary

The active source tree contains no `src/path_c/` package and no retired DEPI
application entry points. The complete superseded package, experiment apps,
configs, workflow and tests are preserved under `legacy/implementation_v8/`.
The pinned Official adapter imports protocol constants from
`src/delta_zsc/config.py`; it cannot transitively pull an old model or optimizer
back into the active execution graph.
