# DELTA-ZSC v5 status dashboard

Date: 2026-08-10

## Active source status

| Item | Status | Evidence |
|---|---|---|
| Method identity | implemented | config 3, checkpoint schema 5, deployment schema 4, package 0.6.0 |
| Episode-static response posterior | implemented and locally tested | filter persistence/reset tests |
| Belief-conditioned raw-return critic | implemented and locally tested | dense TD(lambda), pairwise-CRN and mirror-policy tests |
| Delayed response and exact 66-outcome VOI | implemented and locally tested | delayed-window and synthetic VOI tests |
| Semantic initializer K=2/4/8 | implemented and locally tested | initializer/provenance tests |
| `test_time_wide` `5x5x43` observation | implemented and locally tested | layout-derived channel-block tests |
| Uniform registered partner sampling | implemented and locally tested | sampler and training-wiring tests |
| Config-selected development matrix | implemented | 55 cells per layout with seed-index SP initialization and full pilot/current/successor cost |
| Formal CUDA shape | implemented | 256 lanes, Floyd anchor sampling, `<40,000 MiB` execution rule |
| Official upstream DAG | implemented | config-selected layout with support/panels/baselines/FCP populations and lineage ledgers |
| Paper population evaluator | implemented and locally tested | `(10,10,500)`, root 42, directed SP/XP cells |
| Common-partner evaluator | implemented and locally tested | `(10,16,2,500)`, root 0, five baselines plus DELTA |

## Validation status

- Active source compilation and contract validation: passed.
- Seven registered configurations are present.
- Root CLI plus 19 subcommands are present.
- Isolated CPU regression: 82/82 passed with no failure or timeout; details are
  recorded in `VALIDATION_REPORT.md`.
- No local source check is scientific performance evidence.

## Empirical status

| Work | Status |
|---|---|
| `test_time_wide` SP-only ten-run parent population | completed on the isolated server copy |
| Two-seed `base/response_only/delta_active` pilot | completed: six development runs |
| Held-out two-parent SP evaluation | completed: 800 episodes per arm with matched keys |
| Full Official upstream populations for this layout | deferred; not part of the current pilot |
| Fitted semantic initializer for this layout | pilot K=4 artifact completed; full K=2/4/8 set deferred |
| Real 256-lane CUDA execution | not yet executed |
| 55-run development matrix for this layout | not yet executed |
| Ten-run DELTA population for this layout | not yet executed |
| Paper/common-partner evaluations for this layout | not yet executed |
| Layout-side H1/H2/H3 components | not generated |
| Full H1/H2/H3 or SOTA claim | unavailable: `test_time_simple` was not run |

## Completed pilot path

The completed server pilot is deliberately smaller than the implemented full
workflow. It trained one ten-run Official SP population, preassigned six parent
runs to training, two to the unlabeled K=4 initializer and two to held-out
measurement, then ran paired seeds 0 and 1 for `base`, `response_only` and
`delta_active`. All three arms used the same two held-out SP parents, roles and
environment keys.

The final-code rerun produced held-out raw-return means of `base=24.625`,
`response_only=-2.825`, and `delta_active=0.300`. Paired seed differences were
`delta_active-response_only=(1.35, 4.90)`,
`response_only-base=(-15.95, -38.95)`, and
`delta_active-base=(-14.60, -34.05)`. Active DELTA improved on response-only in
both seeds, while both semantic arms remained substantially below base.

The 55-run matrix, OP/SA/FCP populations, formal DELTA population, paper cube,
16-partner common evaluation and claim/resource synthesis were not part of this
pilot and were not started. The pilot did not invoke `formal-claim`.

These are two-seed SP-only development observations, not registered H2,
Table 2, H1/H2/H3 or SOTA evidence.
