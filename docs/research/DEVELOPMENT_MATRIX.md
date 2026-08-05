# DEVELOPMENT_MATRIX — Registered feasible matrix

`authoritative: true`

## Main K=4 block

Five paired seeds (`0..4`) are used for every row and each layout.

| Variant | Response latent | Decision channel | KL adaptation | Active VOI | Anchor cost |
|---|---:|---:|---:|---:|---:|
| history_rnn | no | no | no | no | 0 |
| base | no | no | no | no | 0 |
| response_only | yes | no | no | no | 0 |
| delta_passive | yes | yes | yes | no | registered |
| delta_active | yes | yes | yes | yes | registered |
| history_rnn_extra | no | no | no | no | reallocated to PPO |
| base_extra | no | no | no | no | reallocated to PPO |

The first five rows have equal ordinary PPO interaction. Passive and active use
identical anchor observations. The two extra controls add exactly the anchor
continuation transition count to PPO and collect no privileged labels.
Deployable parameter capacity is checked within each paired block.

## K sensitivity

Only passive and active DELTA are run for `K=2` and `K=8`. The main `K=4` runs
are reused, giving 10 additional training runs per layout rather than another
full matrix.

## Pre-specified contrasts

1. `delta_passive - response_only` — decision-emission contribution.
2. `delta_active - delta_passive` — active response VOI.
3. `delta_active - base`.
4. `delta_active - history_rnn`.
5. `delta_active - base_extra`.
6. `delta_active - history_rnn_extra`.

The matrix application mechanically validates interaction, anchor and capacity
alignment before writing its artifact.

## Diagnostic interpretation

- High information gain with zero VOI means identifiable but decision-irrelevant
  response structure.
- High VOI with large nested quadrature error means numerical resolution is
  insufficient; this is reported, not silently gated.
- Low decision NLL but no XP gain points to policy conversion or distribution
  mismatch.
- Response improvement without decision improvement falsifies the claim that
  generic teammate prediction is sufficient.
