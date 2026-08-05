# PAPER_OUTLINE — DELTA-ZSC

## Working title

**Decision-Relevant Bayesian Adaptation for Zero-Shot Coordination**

## Central claim

Predicting teammate behavior is insufficient unless the inferred uncertainty is
trained to preserve counterfactual action ordering and converted into a bounded
shared-policy update.

## Contributions

1. **Unified response-decision latent model.** Legal teammate responses and
   sparse CRN action-return contrasts train one exchangeable latent state,
   without partner identities or per-type policies.
2. **Analytic bounded adaptation.** A KL-constrained mirror policy converts
   posterior action values into a single shared actor without an auxiliary
   adaptation network.
3. **Decision-relevant active response value.** Complete response outcomes are
   integrated by deterministic low-discrepancy quadrature and exact Bayes
   updates; identity information with no decision value contributes zero.
4. **Causal and statistically valid evaluation.** Same-world belief
   interventions, crossed run-level bootstrap, total-interaction controls, and
   fully loaded resource accounting support the mechanism claim.

## Main narrative

1. Existing ZSC methods often entangle task learning, partner recognition and
   adaptation losses.
2. The scientifically relevant object is not partner identity but the action
   ordering implied by legal history.
3. DELTA factorizes task competence and latent estimation, then combines them
   analytically.
4. Passive DELTA tests decision-relevant inference; active DELTA tests whether
   the next response itself has task value.
5. Simple/Wide results establish performance, the response-only comparison
   establishes the decision channel, and same-world interventions establish
   causal belief value.

## Main figures

1. Unified graphical model and train/deployment information boundary.
2. Response-only posterior -> decision emission -> KL mirror policy.
3. Active VOI computation: exact binary marginalization + Halton categorical
   quadrature -> all-component likelihood -> Bayes posterior ->
   posterior-optimal decision.
4. Simple/Wide cross-play matrices and all-baseline contrasts.
5. Same-world belief intervention and VOI/IG/quadrature diagnostics.

## Main tables

1. Confirmatory raw-return comparison with crossed-node intervals.
2. Nested development ablations and total-interaction controls.
3. H1/H2/H3 closed-hierarchy decision.
4. Marginal and fully loaded compute/parameter/latency accounting.

## Limitation statement

The registered active term is a one-response decision-equivalence VOI using a
local-stationarity surrogate. It is not full Bayes-adaptive planning. The paper
reports the nested quadrature error and states the `2 epsilon_drift` approximation
bound rather than presenting the term as exact long-horizon value.
