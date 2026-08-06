# SCIENTIFIC_SPEC — Decision-relevant adaptation in zero-shot coordination

`authoritative: true`

## 1. Substrate

The registered substrate is OvercookedV2 Test-Time Protocol Formation with
400-step episodes, six ego actions, view radius two, random initial positions,
negative rewards, recipe resampling after delivery, and successful-delivery
indication. Simple and Wide are distinct confirmatory layouts and must both be
reported.

## 2. Scientific problem

A previously unseen teammate may produce legal local histories that imply a
different action ordering for the ego agent. The problem is not to classify the
teammate's training algorithm. It is to estimate, from legal history alone, the
smallest uncertainty state needed to improve decisions without sacrificing
base task competence.

Formally, let `H_t` be legal local history and `z_t` an exchangeable latent
coordination mode. DELTA asks whether a response-only posterior

\[
b_t(z)=p(z_t=z\mid H_t)
\]

can support a decision-relevant action-return estimate

\[
\bar Q_t(a)=\sum_z b_t(z)Q_z(x_t,r_t,u_t,a)
\]

that improves zero-shot cross-play through a single shared policy.

## 3. Legal deployment information

The deployed policy may use only:

- the ego's local observation history;
- the ego's previous executed actions;
- episode boundaries;
- analytic statistics computed from observable teammate events;
- the learned response likelihood and its categorical posterior.

It may not use partner run ID, training algorithm, checkpoint index, family
label, hidden simulator state, future response, counterfactual return, or any
training-only lineage metadata.

Counterfactual all-action continuations are privileged training observations.
They supervise the meaning of latent components but are never inserted into the
online posterior state.

## 4. Unified latent semantics

A latent component is defined only through two conditional observation
channels:

1. the distribution of an observable teammate response;
2. the distribution of centered CRN action-return contrasts.

Components are exchangeable. A component is defined by two conditional
distributions: the teammate response it predicts and the action-return
contrast it implies; it need not map to a human-interpretable protocol.
A component is scientifically useful only when legal response evidence
changes a decision-relevant posterior mixture.

The observable response retains direct teammate visibility and, when visible,
position, direction, inventory, and inventory change. It also contains a
two-candidate aligned world-interface event and independently covered recipe
change. Alignment availability is component-shared; ambiguous alignment and
cross-episode transitions add no interface evidence.

## 5. Primary hypotheses

### H1 — performance

Frozen `delta_active` exceeds every registered same-protocol baseline on both
Simple and Wide. For each baseline, the one-sided crossed-node bootstrap lower
bound must be positive and the point estimate must be at least one correct
delivery, 20 raw-return points.

### H2 — decision-emission contribution

At fixed `K=4`, partner distribution, base PPO budget, network capacity, seed,
and evaluation panel:

\[
J(\text{delta_passive}) > J(\text{response_only})
\]

on both layouts. This isolates the sparse counterfactual decision observation
from response prediction alone.

### H3 — causal value of the legal-history belief

In the same source world and with the same learned action-return matrix, the
policy formed from the correct belief must select higher empirical continuation
value than a task-matched shuffled belief:

\[
\mathbb E[(\pi_{b}-\pi_{\tilde b})^T G_{\text{source}}] > 0
\]

on both layouts.

The claims follow a closed hierarchy H1 -> H2 -> H3. All measurements remain
visible even when a later claim is not eligible.

## 6. Secondary question: active response value

`delta_active - delta_passive` estimates whether choosing actions partly for the
decision value of the next teammate response improves cross-play. This contrast
is pre-registered and fully reported, but it is secondary: the paper does not
promote a null active increment into a new mechanism after observing results.

## 7. Required negative controls

- `history_rnn`: tests whether generic recurrent capacity explains performance.
- `base`: task competence without history-based latent adaptation.
- `response_only`: response prediction without decision-conditioned action change.
- `history_rnn_extra` and `base_extra`: spend the exact anchor simulator cost on
  additional ordinary PPO interaction.
- `K in {2,4,8}`: bounded capacity sensitivity, not an open sweep.
- informative-but-decision-irrelevant synthetic VOI test: information gain may
  be positive while decision VOI must remain zero.

## 8. Non-claims

The repository does not claim:

- recovery of true partner identity or a unique true protocol;
- exact solution of the full Bayes-adaptive POMDP;
- global policy improvement from an approximate action-return model;
- calibration merely because posterior entropy changes;
- SOTA performance without frozen raw Simple/Wide matrices;
- independence of the task recurrent state from every indirect teammate effect.

The active VOI exactly enumerates the 66 outcomes of the compact
`(visibility, interface availability, change, event)` marginal of that complete
response. It remains a registered one-response decision-equivalence approximation.
Its local-stationarity scope and error bound are explicit in `THEORY.md`.
