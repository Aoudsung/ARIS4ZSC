# DELTA-ZSC v4 status dashboard

Date: 2026-08-06

## Active source status

| Item | Status | Evidence |
|---|---|---|
| Method identity | complete | config 3, checkpoint schema 4, package 0.6.0 |
| Episode-static latent | implemented and locally tested | `belief_filter.py`; core filter test |
| Shared occurrence / component semantic split | implemented and locally tested | `response_model.py`; posterior-isolation tests |
| Centered response and decision residuals | implemented and locally tested | response/decision centering test |
| Spectral-simplex initializer | implemented and locally tested | initializer/provenance tests |
| Three separately normalized proper-score channels | implemented and locally tested | v4 objective test |
| Delayed causal probe response | implemented and locally tested | delayed-window tests |
| Probe-conditioned t+2 CRN decision target | implemented and locally tested | successor-anchor tests |
| Exact 66-outcome delayed VOI | implemented and locally tested | VOI tests and synthetic artifact |
| One committed base bridge after an active probe | implemented and locally tested | active bridge test |
| Final action-selectivity audit | implemented and locally tested | final-audit broadcasting regression |
| Versioned deployment/checkpoint round trip | implemented and locally tested | storage/deployment test |

## Validation status

- Active source compilation: passed.
- Six registered configurations: passed.
- Root CLI plus 16 subcommand help paths: passed.
- Namespace/workflow/credential contracts: passed.
- Isolated local CPU pytest suite: `50/50` passed, zero failures, zero
  timeouts; per-test evidence is recorded in
  `validation/ISOLATED_TEST_RESULTS.json`.
- Exact delayed-VOI synthetic acceptance: passed.

## Empirical status

| Stage | Status |
|---|---|
| Real fitted initializer from calibration lineages | not generated in this source package |
| Real Official-checkpoint v4 mechanical run | not executed |
| CUDA memory/throughput acceptance | not executed |
| Paired five-seed development matrix | not executed |
| Formal ten-run Simple/Wide matrix | not executed |
| H1/H2/H3 evidence | not generated |
| SOTA claim | closed |

The three-seed v3 CUDA pilot remains historical failure evidence. Its return,
posterior and VOI measurements cannot be reported as v4 results.

## Next executable scientific path

1. Build lineage-disjoint fitted semantic initializers for Simple and Wide at
   K=2/4/8.
2. Run the one-update CUDA mechanical path and archive its resource ledger.
3. Execute the bounded paired development matrix.
4. Inspect partner separation, filter KL, component event JS, successor action
   ordering, VOI action spread and active/passive policy TV together with raw
   return.
5. Proceed to formal training only under the registered protocol; no local
   source test acts as a performance gate or substitute result.
