# EVALUATION_SPEC — DELTA-ZSC v4 development and inference

`authoritative: true`

## 1. Independent units

Ego training run and partner parent run are independent generalization nodes.
Episodes are repeated measurements within a pairing and are never bootstrapped
as independent ZSC samples. Mechanism and checkpoint stage remain registered
strata.

## 2. Pre-training semantic initialization

Each layout has a lineage-bound calibration panel disjoint from training
support and confirmatory partners. The spectral-simplex initializer is built
without partner labels. The artifact records event/episode counts, singular
values, source run IDs, and a parent-disjoint conditional oracle diagnostic.
Formal training must bind the initializer artifact in run identity.

The oracle diagnostic may use SP/OP labels only to measure whether mechanism
adds held-out event prediction after legal state/history conditioning. It never
changes the initializer, model, optimizer, or policy.

## 3. Development matrix

At `K=4`, five paired seeds run:

- `history_rnn`;
- `base`;
- `response_only`;
- `delta_passive`;
- `delta_active`;
- `history_rnn_extra`;
- `base_extra`.

Core variants receive equal ordinary PPO interaction. Passive and active DELTA
share the sparse current-anchor budget; active additionally pays the registered
probe-successor continuation cost. Extra controls spend the corresponding
simulator budget on ordinary PPO and collect no privileged labels.

For `K=2` and `K=8`, run passive and active DELTA only. This is a bounded
sensitivity analysis, not an open sweep.

Primary development contrasts are:

1. `delta_passive - response_only`;
2. `delta_active - delta_passive`;
3. `delta_active - base` and `delta_active - history_rnn`;
4. active DELTA against total-interaction controls.

## 4. Reproducibility and pairing

Paired cells must record:

- base initialization identity;
- environment, partner, anchor-index, and minibatch root seeds;
- semantic initializer identity;
- action/reward/partner-member stream hashes where applicable;
- resolved config and source method identity.

PPO-side development metrics are descriptive unless replicated across paired
runs. Posterior/mechanism claims must be supported by independent partner-run
aggregation rather than a single final anchor batch.

## 5. Confirmatory performance

Frozen `delta_active` is evaluated against Official SP, OP, State-Augmented,
FCP, and a capacity-matched IPPO-Large when lineage and resource artifacts are
available. Every ego run is crossed with every confirmatory partner run in both
roles.

Raw episode rows are reduced to an ego-run x partner-run matrix. The bootstrap
independently resamples ego and partner nodes. H1 uses an intersection-union
rule: on each layout, every baseline contrast needs a one-sided 95% lower bound
above zero and a point estimate of at least 20.

## 6. H2 and H3

H2 uses paired development seeds for `delta_passive - response_only`; both
layouts require a 95% interval whose lower endpoint is above zero.

H3 is a same-world belief intervention. Environment state, partner, base logits,
current decision matrix, CRN outcomes, and model parameters are fixed. Only the
legal belief is replaced by a task-matched belief from another partner run. A
crossed ego/partner bootstrap evaluates empirical continuation value.

## 7. Active mechanism evidence

Report for every layout and partner stratum:

- exact delayed-response VOI mean, max, min, and negative fraction;
- expected information gain;
- per-state `max_a-min_a` VOI and information-gain spread;
- active/passive probability total variation;
- active/passive and base/active greedy disagreement;
- current and successor decision agreement/regret;
- empirical continuation value of active versus passive choices when measured.

A positive constant VOI shared by all actions is not active probing evidence.
Information gain cannot substitute for task VOI.

## 8. Posterior and semantic diagnostics

The calibration/posterior application reports:

- factor-level shared and semantic NLL/counts;
- interface/recipe coverage and positive rates;
- conditional event NLL and `OTHER/MULTI` rate;
- component event JS;
- response-induced filter KL versus the episode-static prior;
- belief entropy, partner separation, and phase drift;
- pooled SP/OP event distribution and total variation;
- initializer singular values and conditional oracle gain.

No diagnostic threshold gates benchmark evaluation. These measurements localize
model error and prevent a partner-independent global winner from being
misreported as inference.

## 9. Resource accounting

Every method reports:

- ordinary ego-policy simulator steps;
- current-anchor continuation steps;
- probe, base-bridge, and `A x A` post-response continuation steps;
- calibration/initializer steps;
- upstream partner-training steps;
- shared and marginal GPU/wall time;
- peak memory;
- deployable and training-only parameter counts;
- batch-one inference latency.

The full successor matrix is computed with bounded-memory `lax.map`, but all
simulator transitions are still charged.

## 10. Prohibited analysis

- selecting partners or stages after viewing DELTA return;
- treating episodes as independent bootstrap nodes;
- using mechanism labels to train the semantic initializer or posterior;
- reporting posterior sharpness without partner separation;
- replacing raw return with response NLL or VOI;
- promoting a null active increment into an unregistered claim;
- combining layouts so one masks failure on the other;
- reusing v3 DELTA checkpoints or results as v4 runs.
