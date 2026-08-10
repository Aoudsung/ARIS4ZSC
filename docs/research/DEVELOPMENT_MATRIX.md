# DEVELOPMENT_MATRIX — Registered DELTA-ZSC v5 matrix

`authoritative: true`

## Main K=4 block

Five paired seeds (`0..4`) are used for every row and layout. Each DELTA row is
bound to the same layout-specific residual-panel initializer root; the runner
selects the K-specific `k-2`, `k-4`, or `k-8` artifact for that row. Thus K
sensitivity changes only the simplex cardinality, not the calibration data.
The matrix command therefore requires `--semantic-initializer`; latent-bearing
development cells do not silently fall back to an unfitted simplex.

| Variant | Episode-static response latent | Current decision | Delayed response + successor decision | KL adaptation | Anchor cost |
|---|---:|---:|---:|---:|---:|
| history_rnn | no | no | no | no | 0 |
| base | no | no | no | no | 0 |
| response_only | yes | no | no | no | 0 |
| delta_passive | yes | yes | no | yes | current anchors |
| delta_active | yes | yes | yes | yes | current + successor anchors |
| history_rnn_extra | no | no | no | no | reallocated to PPO |
| base_extra | no | no | no | no | reallocated to PPO |

The core rows have equal ordinary PPO interaction. Extra controls add the exact
charged pilot, current and successor continuation transition count to PPO and
receive no counterfactual labels.

The main block is `7 variants x 5 seeds = 35` runs. K sensitivity adds
`2 variants x 2 additional K values x 5 seeds = 20` runs, for exactly 55 runs
per layout.

## K sensitivity

Only passive and active DELTA are run for `K=2` and `K=8`; the main `K=4` runs
are reused. Initializer construction uses the same residual data but projects a
simplex with the registered K.

Development seed `s` initializes every variant and K value from the same
development-support SP parent `s`. Partner sampling remains uniform over the
registered mechanism, family, stage and run probabilities throughout each
training run.

## Pre-specified contrasts

1. `delta_passive - response_only` — decision-emission contribution.
2. `delta_active - delta_passive` — delayed active-response contribution.
3. `delta_active - base`.
4. `delta_active - history_rnn`.
5. `delta_active - base_extra`.
6. `delta_active - history_rnn_extra`.

## Required paired diagnostics

- base action/reward and partner-stream identity where variants should share
  collection behavior;
- initializer identity and singular values;
- component event JS and partner-separation L1;
- response-induced filter KL and phase drift;
- current/successor agreement, regret, and component disagreement;
- VOI and information-gain action spread;
- active/passive policy TV;
- report-only semantic/decision gradient norms and cosine on the shared
  component embeddings.

High posterior sharpness with low partner separation is a failed global winner,
not successful inference. Positive information gain with zero VOI spread is not
active probing. Low decision loss without held-out return gain localizes failure
to policy conversion or distribution shift.

These diagnostics are interpreted after the registered matrix is complete.
None changes the variants, losses, budgets, partner distribution or scheduling
of downstream jobs.
