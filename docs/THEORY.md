# THEORY — Finite guarantees and explicit limits

`authoritative: true`

## 1. Exact filter relative to the learned model

Given a normalized previous belief, row-stochastic transition `T`, and finite
component response likelihoods, the update in `belief_filter.py` is exactly the
categorical Chapman-Kolmogorov prediction followed by Bayes correction. The
result is normalized and non-negative. This is an implementation theorem about
the registered latent model, not a claim that its components are true human
protocols.

## 2. Proper composite latent score

The response and CRN decision channels are conditionally independent given the
same component. Their summed negative log likelihood is a proper composite
predictive score for the registered factorization. Normalization by `N_y+N_A`
changes estimator scale, not its optimum. No arbitrary auxiliary-loss weights
are required.

## 3. Mirror-policy solution

For fixed base policy `pi_0`, finite action values `Q`, and KL radius
`delta>0`, the constrained objective

\[
\max_\pi \langle\pi,Q\rangle
\quad\text{s.t.}\quad D_{KL}(\pi\Vert\pi_0)\le\delta
\]

has the exponential-tilting solution

\[
\pi_\eta(a)\propto\pi_0(a)e^{Q(a)/\eta}.
\]

KL decreases monotonically with `eta`; bisection therefore returns the unique
boundary solution when the constraint is active. The code tests the achieved KL
numerically.

This guarantees improvement only for the supplied model-based objective. A
real-return guarantee additionally requires accurate action values.

## 4. Non-negativity of exact decision VOI

Let

\[
V(b)=\max_a\sum_kb_k\mu_k(a).
\]

`V` is the maximum of linear functions and is therefore convex. A Bayes
posterior is a martingale, so

\[
\mathbb E_y[b^{y}]=\bar b.
\]

Jensen's inequality gives

\[
\mathbb E_y[V(b^y)]-V(\bar b)\ge0.
\]

The implementation records the raw deterministic quadrature estimate and uses
its non-negative part for control because finite quadrature can violate the
inequality slightly.

## 5. Decision relevance, not identity information

If all components induce the same action-value vector, then `V(b)` is independent
of `b`; hence VOI is exactly zero even when the response perfectly identifies
the component. More generally, information that only separates components with
identical optimal decision value has zero decision VOI. This property is covered
by a synthetic test where information gain is positive but VOI is zero.

## 6. Deterministic quadrature consistency

For the finite discrete response model, inverse-CDF integration with a
low-discrepancy sequence converges to the response expectation as sample count
increases. DELTA Rao-Blackwellizes both binary factors: visibility is summed
exactly, and inventory-change is summed exactly whenever it is legally
observable. Source components are also summed exactly. Only the remaining
position/direction/inventory integral is approximated. The `S/2` versus `S`
prefix difference is a convergence diagnostic, not a probabilistic confidence
interval or a formal error bound.

## 7. Scope of the local-stationarity surrogate

The true one-step Bayes-adaptive value may use a response- and probe-dependent
future utility matrix `mu^{a,y}`. The standard model uses a shared local matrix
`mu`. Suppose

\[
\sup_{a,y,k,a'}|\mu^{a,y}_k(a')-\mu_k(a')|\le\epsilon_{drift}.
\]

For any belief, the corresponding optimal values differ by at most
`epsilon_drift`; applying this to both posterior and prior terms yields

\[
|\mathrm{VOI}_{true}(a)-\mathrm{VOI}_{local}(a)|
\le 2\epsilon_{drift}.
\]

Thus the active term is well-founded when decision-equivalent action ordering
changes slowly over the one-response horizon. The repository does not infer
that condition from entropy and does not claim exact long-horizon planning.
The public VOI API already accepts probe-conditioned future utilities for a
future extension that supplies them with valid training observations.

## 8. Error decomposition for adapted performance

The difference between the ideal and implemented adapted objective can be
decomposed into:

1. response-model error;
2. posterior filtering error inherited from that model;
3. decision-emission error;
4. local-stationarity error;
5. deterministic quadrature error;
6. KL projection restriction.

The architecture exposes diagnostics for response NLL, held-out decision
ordering, posterior entropy, raw/clamped VOI, information gain, quadrature
prefix error, and achieved KL. These measurements diagnose failure but do not
become additional training gates.

## 9. What is not proved

No theorem in this repository establishes:

- universal ZSC generalization to arbitrary partners;
- global return improvement under misspecified emissions;
- semantic identifiability of latent components beyond permutation;
- exact recovery of teammate intent;
- exact solution of the full partially observable stochastic game;
- SOTA performance without confirmatory raw results.
