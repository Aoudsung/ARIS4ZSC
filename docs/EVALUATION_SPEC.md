# EVALUATION_SPEC — Development, confirmatory inference, and diagnostics

`authoritative: true`

## 1. Independent units

The independent generalization nodes are ego training run and partner training
run. Episodes are repeated measurements within a pairing and are never treated
as independent ZSC samples. Mechanisms are preserved as registered strata.

## 2. Development matrix

At `K=4`, five paired seeds run:

- `history_rnn`;
- `base`;
- `response_only`;
- `delta_passive`;
- `delta_active`;
- `history_rnn_extra`;
- `base_extra`.

The five core variants use equal ordinary PPO interaction. Passive and active
DELTA share the same sparse anchor budget. The two `*_extra` controls spend that
exact simulator budget on additional PPO interaction and collect no anchors.

For `K=2` and `K=8`, only passive and active DELTA are run. Together with the
main `K=4` runs, this is a bounded sensitivity analysis rather than a broad
hyperparameter sweep.

Primary development contrasts:

1. `delta_passive - response_only`: decision-emission contribution.
2. `delta_active - delta_passive`: active VOI increment.
3. `delta_active - base` and `delta_active - history_rnn`.
4. active DELTA against total-interaction controls.

## 3. Confirmatory performance

Frozen `delta_active` is compared with every registered same-protocol baseline
on both roles and every confirmatory partner run. Baselines include Official
SP, OP, State-Augmented, FCP, and a capacity-matched IPPO-Large when its lineage
and resource artifact are supplied.

For each method, raw episode rows are reduced to an ego-run x partner-run matrix.
The bootstrap independently resamples ego and partner nodes. H1 uses an
intersection-union rule: every baseline contrast must have one-sided 95% lower
bound above zero and point estimate at least 20.

## 4. H2 and H3

H2 uses paired development seeds for
`delta_passive - response_only`; both layouts must have a 95% interval whose
lower endpoint is above zero.

H3 performs a same-world intervention. Environment state, partner, CRN returns,
base logits, and learned action-return matrix remain fixed; only the legal-history
belief is replaced by a task-matched belief from another partner run. A crossed
ego/partner bootstrap tests whether the correct belief has positive empirical
decision value.

Claims are evaluated in the closed order H1 -> H2 -> H3.

## 5. Active-VOI evidence

The active increment is reported on both layouts regardless of sign. It remains
a pre-registered secondary result. Mechanism diagnostics include:

- raw and non-negative VOI;
- expected information gain;
- nested Halton quadrature error;
- action-wise VOI dispersion;
- achieved adaptation KL;
- active/passive action disagreement;
- empirical continuation value at belief-intervention anchors.

Information gain cannot substitute for task VOI.

## 6. Posterior-predictive diagnostics

The calibration application is diagnostic only. It reports response NLL,
component-mixture gain, posterior entropy, and held-out structure by independent
partner run. It does not tune a conformal gate or decide whether training may
continue.

## 7. Resource accounting

Every method reports:

- ordinary ego-policy simulator steps;
- sparse anchor continuation steps;
- upstream partner-training steps;
- shared and marginal GPU/wall time;
- peak memory;
- deployable and training-only parameter counts;
- batch-one inference latency.

The main table includes marginal, shared, amortized, and fully loaded views. No
baseline is charged for privileged anchors it did not collect; total-interaction
controls provide the causal budget comparison.

## 8. Prohibited analysis

- selecting partner runs after viewing DELTA performance;
- treating episodes as independent bootstrap nodes;
- choosing the strongest baseline only after excluding inconvenient methods;
- replacing raw return with response NLL as the primary endpoint;
- promoting calibration or active-VOI diagnostics into new claims after results;
- combining Simple and Wide so that one layout masks failure on the other.
