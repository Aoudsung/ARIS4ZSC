# ARCHITECTURE — DELTA-ZSC v4 equation-to-code map

`authoritative: true`

## 1. Scientific object map

| Object | Active implementation |
|---|---|
| task recurrent actor-critic | `src/delta_zsc/base_policy.py` |
| direct/interface response extraction | `src/delta_zsc/observation.py` |
| legal Beta history statistics | `src/delta_zsc/behavior_statistics.py` |
| episode-static prior and Bayes correction | `src/delta_zsc/belief_filter.py` |
| shared occurrence / centered semantic emissions | `src/delta_zsc/response_model.py` |
| unlabeled spectral-simplex artifact | `src/delta_zsc/semantic_initializer.py` |
| current and successor decision emissions | `src/delta_zsc/decision_model.py` |
| unified legal state transition | `src/delta_zsc/latent_model.py`, `model.py` |
| exact delayed 66-outcome VOI | `src/delta_zsc/bayes_voi.py` |
| analytic KL mirror policy | `src/delta_zsc/mirror_policy.py` |
| current/successor CRN anchors | `src/delta_zsc/anchors.py` |
| channel-normalized losses | `src/delta_zsc/losses.py` |
| optimizer transaction | `src/delta_zsc/training.py` |
| vector rollout and sparse snapshots | `src/delta_zsc/runner.py` |
| training, audit, deployment export | `experiments/overcooked_v2/training_app.py` |
| initializer and posterior diagnostics | `experiments/overcooked_v2/calibration_app.py` |
| one active CLI | `experiments/overcooked_v2/delta_zsc.py` |

There is deliberately no active `transition.py`: physical-time latent dynamics
were removed in v4.

## 2. Parameter ownership

`base_params` own:

- task frame encoder;
- task GRU;
- instantaneous partner encoder;
- base actor logits;
- base value function.

`latent_params` own:

- shared component embeddings;
- immediate response model;
- delayed probe-response model;
- current and successor decision model.

The trees are disjoint and have separate Adam states. PPO receives a
stop-gradient latent tree and executes `compute_latent=False`. The latent loss
receives a stop-gradient base tree.

## 3. Runtime state

`PolicyState` contains only:

- task recurrent carry;
- episode-static categorical belief;
- six Beta statistics;
- previous local observation;
- previous ego action;
- episode-start flag.

It contains no partner ID, counterfactual return, hidden environment state,
future response, semantic initializer metadata, or decision anchor.

## 4. Immediate step order

`DeltaModel.step` executes:

1. base task update on current observation;
2. episode-static prior reset/persistence;
3. immediate response extraction from stored previous observation and action;
4. shared and semantic response prediction;
5. semantic-only Bayes correction;
6. legal statistics update;
7. current decision prediction;
8. optional delayed probe-response and successor decision prediction;
9. exact VOI for `delta_active`;
10. passive or active mirror policy;
11. storage of the current observation for the next legal response.

The executed action and terminal flag are inserted only after the environment
transition by `observe_after_transition`.

## 5. Response model topology

Both immediate and delayed models have:

```text
full frame -> frame MLP
legal behavior features + ego/probe action embedding
    -> shared context trunk
        -> shared occurrence heads
        -> component trunk(shared context, component embedding)
            -> context residual
component embedding -> direct residual skip
shared semantic logits + centered(context + skip + initializer bias)
```

Immediate semantic heads cover position, direction, inventory, and event.
Delayed semantic prediction contains the event head required by active VOI.

## 6. Decision topology

Current decision:

```text
task features + instantaneous partner + behavior
    -> shared trunk -> shared mean and shared variance
shared trunk + component embedding
    -> component trunk -> centered context residual
component embedding -> direct mean residual skip
```

Successor decision adds a probe-action embedding and a separate shared/component
trunk. Both return `[...,K,A]` means and shared component-broadcast variance.

## 7. Training data alignment

A rollout stores `T+1` legal observations and `T` terminal-aware response-next
frames. Immediate response target `t` is:

```text
observations[t] -> response_next_observations[t] under actions[t]
```

Delayed probe target `t` is:

```text
probe input: observations[t], actions[t]
response: response_next_observations[t] -> response_next_observations[t+1]
alignment/exclusion action: actions[t+1]
valid: not dones[t] and not dones[t+1]
```

This indexing is encoded in `RolloutBatch` and tested directly.

## 8. Anchor memory boundary

Current anchors materialize `N x A x R` worlds. Active anchors traverse the
probe axis, one unforced base bridge, and the post-response decision-action axis
with nested `lax.map`; only `N x R` worlds are live. Returned arrays are still
complete:

- current fit/evaluation means `[N,A]`;
- current replicas `[N,A,R]`;
- successor means `[N,P,A]`;
- successor replicas `[N,P,A,R]`;
- covariance `[N,5,5]` and `[N,P,5,5]`;
- validity masks.

## 9. Static execution paths

- PPO replay: `compute_latent=False`, `execute_adaptation=False`;
- response/decision MLE replay: `compute_latent=True`,
  `execute_adaptation=False`;
- current anchors: forced measured action, then base policy;
- active successor anchors: forced probe, one base bridge, forced `t+2`
  decision action, then base policy;
- evaluation/deployment: `compute_latent=True`,
  `execute_adaptation=True`.

These are compile-time paths, not learned gates.

## 10. Artifact and schema boundary

- method: `delta_episode_static_centered_residual_delayed_exact_voi_v4`;
- config schema: 3;
- checkpoint schema: 4;
- deployment bundle: 3;
- manifest schema: 2;
- semantic initializer schema: 1.

Training identity includes the resolved initializer mapping and source path.
Development/formal semantic variants reject a missing initializer and validate
its method version, layout, K, calibration role, event count, and parent-lineage
disjointness from the DELTA training support. Mechanical checks and pure
base/history collection runs may use the deterministic fallback. v3 DELTA
state is not loaded.

## 11. Active and historical boundaries

Only `src/delta_zsc/` and the applications listed by the repository test are
active. `legacy/implementation_v8/`, `docs/legacy/`, and retired workflows are
read-only history. Active package discovery excludes retired namespaces.
