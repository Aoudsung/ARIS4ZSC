# CETR-ZSC

This repository contains the active CETR-ZSC (constrained episodic tail-robust
zero-shot coordination) implementation for the OvercookedV2 Test-Time Protocol
Formation benchmark.

The active method identity is defined only by
[`src/cetr_zsc/config.py`](src/cetr_zsc/config.py):

- `METHOD_VERSION = constrained_episodic_tail_robust_zsc_v1`
- active configuration contract: `version: 6`
- active package root: `src/cetr_zsc/`
- active CLI entry point: `experiments/overcooked_v2/cetr_zsc.py`

Registered budgets, seeds, schemas, and sample sizes are not duplicated here;
[`src/cetr_zsc/config.py`](src/cetr_zsc/config.py) is the sole authority.

## Method in one page

CETR trains one partner-agnostic recurrent actor, initialized from a
seed-matched Official-SP actor and then trained as one complete actor tree. The
external target is parent-level lower-half CVaR over completed raw episodic
returns. Its self-play performance is an explicit reference-derived constraint,
implemented with a dual variable rather than a hand-tuned tolerance or loss
weight.

Training uses the complete undiscounted raw return-to-go. External parents are
assigned deterministically with full fold×role coverage on every update, and
checkpoint stages rotate by update and lane slot. Parent tail weights are
computed from completed episode returns with double cross-fitting; the complete
`EpisodeBatch` is normalized exactly once before minibatch slicing, with no
observed-subset re-normalization. Self-play uses the same current policy on both
sides with independent recurrent carries, so its gradient is the bilateral
self-composition gradient. A scalar value baseline exists only to reduce
policy-gradient variance.

Deployment is deliberately small:

```text
local observation and recurrent history
    -> one CNN-to-GRU actor
    -> action
```

Partner IDs, algorithm labels, checkpoint metadata, parent groups, co-training
lineage, `q`, the SP dual, hidden simulator state, future responses, and
counterfactual returns are not deployment inputs. Deployment bundle version 7
contains only the actor parameter subtree; the reference artifact and training
manifest are same-directory provenance records, including `provenance.json`,
for audit binding rather than runtime use.

## Evaluation boundary

The primary question is held-out external-partner robustness. The confirmatory
panel is lineage-disjoint from development support in both parent and
co-training lineage. The primary report contains external mean, external
lower-half CVaR, and self-play minus the reference-derived target. The `10×10`
population matrix is supplementary and does not replace the confirmatory
estimand.

The decisive comparison is Official-SP, Official-OP, Official-FCP, the retired
V6 DELTA-active result as a historical comparator, and CETR-ZSC. Evaluation
summaries are descriptive; the unique `claim` entry point compares CETR against
exactly `{"fcp"}` using crossed ego-run/parent-lineage bootstrap and paired
seed-index self-play differences. GO, NO-GO, and INCONCLUSIVE are defined in
`docs/SCIENTIFIC_SPEC.md` and `docs/RESEARCH_PLAN.md`. No performance claim
exists until the registered formal runs and raw evaluation artifacts are
complete. **CETR 尚无训练结果。**

## Repository contracts

- [`docs/PROTOCOL_INDEX.md`](docs/PROTOCOL_INDEX.md) maps questions to the only
  authoritative documents.
- [`docs/METHOD_SPEC.md`](docs/METHOD_SPEC.md) defines the objective, contracts,
  training transaction, and legal information boundary.
- [`docs/SCIENTIFIC_SPEC.md`](docs/SCIENTIFIC_SPEC.md) defines the registered
  problem, panels, endpoints, and decision rules.
- [`docs/THEORY.md`](docs/THEORY.md) separates exact algebra from approximation
  and non-claims.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) maps equations to CETR source
  files and application entry points.
- [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md) defines panel disjointness and
  the decisive experiment.
- [`docs/status/DASHBOARD.md`](docs/status/DASHBOARD.md) records implementation
  and evidence status.
- [`IMPLEMENTATION_MATRIX.md`](IMPLEMENTATION_MATRIX.md) traces requirements to
  code.

V6 DELTA and DEPI are retired and removed from the active tree. Their prior code,
contracts, and results are historical only and are preserved by git history; no
historical artifact defines CETR.
