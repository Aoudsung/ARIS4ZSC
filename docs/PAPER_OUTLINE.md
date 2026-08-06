# PAPER_OUTLINE — DELTA-ZSC v4

## Working title

**Decision-Relevant Episode-Static Adaptation for Zero-Shot Coordination**

## Central claim

Teammate prediction becomes useful for zero-shot coordination only when pooled
response regularities are separated from partner-semantic residuals, legal
history selects those residuals, and the same latent semantics predict
counterfactual action ordering.

## Contributions

1. **Episode-static response-decision latent.** The persistence assumption is
   matched to a fixed teammate per episode, with no physical-time transition
   that erases sparse evidence.
2. **Shared occurrence and centered semantic residuals.** High-frequency
   no-change factors are predicted by shared heads, while conditional geometry,
   event convention, and action-value corrections use zero-mean component
   residuals with direct embedding paths.
3. **Unlabeled spectral-simplex initialization.** Cross-fitted episode residuals
   initialize exchangeable event semantics without partner IDs or SP/OP labels.
4. **Delayed decision-relevant active value.** A separate two-step response head
   models the first window in which a teammate can react to a probe; exact
   66-outcome Bayes updates are valued by a probe-conditioned successor CRN
   matrix.
5. **Causal and statistically valid evaluation.** Crossed run-level inference,
   total-interaction controls, same-world belief intervention, semantic
   diagnostics, and fully loaded resource accounting distinguish mechanism from
   mere posterior sharpness.

## Narrative

1. v3 extracted a real convention signal but an unconstrained mixture learned a
   pooled predictor; a sticky posterior amplified a partner-independent winner.
2. The failure identifies two separations that the model must encode:
   pooled occurrence versus component semantics, and current task value versus
   component decision residual.
3. v4 imposes these separations by parameterization rather than partner labels,
   entropy gates, or multiple actors/critics.
4. Passive DELTA tests whether legal semantic history improves decisions.
5. Active DELTA tests whether a causally delayed response is action-selective
   and worth the probe opportunity.
6. Simple/Wide matrices, current/successor audits, and belief interventions test
   the full chain.

## Main figures

1. v3 failure diagnosis and v4 episode-static graphical model.
2. Shared occurrence head versus centered semantic residual head.
3. Cross-fitted spectral residuals and simplex initialization.
4. Immediate posterior -> current decision residual -> KL mirror policy.
5. Probe `a_t` -> delayed response `o_{t+1}->o_{t+2}` -> exact 66 outcomes ->
   successor matrix -> active value.
6. Simple/Wide cross-play matrices and same-world belief intervention.

## Main tables

1. Confirmatory all-baseline raw-return comparison with crossed-node intervals.
2. Nested development variants and total-interaction controls.
3. Semantic/predictive mechanism audit: event JS, partner separation, filter KL,
   current/successor decision regret, VOI spread, policy TV.
4. H1/H2/H3 closed-hierarchy decision.
5. Marginal, shared, amortized, and fully loaded resources.

## Limitation statement

DELTA v4 uses a finite exchangeable episode-static latent and values one delayed
response window under the registered base continuation policy. It is not a
partner-identity model or full Bayes-adaptive planner. Spectral directions are
initialization coordinates, not discovered ground-truth protocols. Exactness
applies to the finite 66-outcome learned marginal, not to the environment's
complete future trajectory distribution.
