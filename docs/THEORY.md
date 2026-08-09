# THEORY — DELTA-ZSC v4 guarantees, identifiability, and limits

`authoritative: true`

## 1. Exact episode-static filtering

Given a normalized prior `b`, finite semantic log likelihoods `ell_k`, and no
episode boundary, `belief_filter.py` returns

\[
\operatorname{softmax}(\log b+\ell).
\]

At an episode boundary it replaces `b` by the uniform prior before correction
and suppresses the fictitious cross-episode response. The implementation is an
exact categorical Bayes update for the registered episode-static model. It is
not a claim that a component equals a unique human protocol.

## 2. Shared factors cannot alter posterior odds

Suppose the complete response likelihood factorizes as

\[
p(y\mid k)=p_{sh}(y^{occ})p_{sem}(y^{sem}\mid k).
\]

Then

\[
\frac{b'(i)}{b'(j)}=
\frac{b(i)p_{sem}(y\mid i)}{b(j)p_{sem}(y\mid j)},
\]

because `p_sh` cancels. v4 enforces this cancellation structurally by removing
the component axis from occurrence heads and by passing only semantic log
probabilities to `filter_update`. Therefore high-frequency partner-independent
no-change events cannot create a component winner.

## 3. Centered residual decomposition

For component logits `L_k=L_0+Delta_k` with `sum_k Delta_k=0`, the component
mean is exactly `L_0`. Shared prediction and specialization are identifiable as
separate parameter roles: changing all components equally cannot be represented
by the residual branch, and changing only the pooled baseline cannot create a
posterior likelihood ratio.

Centering does not guarantee useful specialization by itself. It removes the
specific pooled-offset degeneracy and supplies first-order component paths. The
semantic predictive score still decides whether residual differences persist.

## 4. Spectral-simplex initializer properties

The initializer is centered because both the regular simplex vertices and the
final bias are zero mean over components. It is label-free because its inputs
are only episode-grouped event residuals after cross-fitted pooled prediction.
Every frame coordinate participates through a fixed Rademacher projection whose
dimension and seed are stored in the artifact; behavior and action features are
uncompressed. Partner IDs and SP/OP labels are absent from the construction
function and artifact contract.

SVD chooses directions of greatest unexplained episode-level event variation;
it does not assert that those directions are true partner identities. The
simplex gives all components equal norm and pairwise symmetric starting
geometry, avoiding a privileged random winner.

## 5. Three proper predictive channels

Each channel term is a mean negative log probability of observations generated
under the registered conditional model. A fixed sum of proper scores remains a
proper composite score for those marginals. Separate normalization changes
channel scale but not the optimum of an individual channel and prevents sample
frequency from becoming an implicit coefficient.

Because the three marginals share parameters and a latent variable, their
optima can conflict. The final anchor audit computes exact semantic and
decision gradient norms and their cosine on the shared component embeddings.
These diagnostics reveal conflict without gating training or tripling the full
optimizer backward pass.

## 6. Decision contrast likelihood

Centered six-action returns lie in the five-dimensional subspace orthogonal to
the all-ones vector. The Helmert basis is orthonormal on that subspace, so
projecting targets, means, and covariance loses only the unidentifiable common
offset. With at least six fit replicas, the sample covariance can be full rank
five before numerical jitter. The shared model variance keeps component
likelihood differences tied to predicted means rather than component-specific
uncertainty.

## 7. CRN successor estimand

For a probe `a` and post-response action `a'`, the registered target is the
discounted return beginning at `t+2`, where `a'` is forced after one unforced
collection-time-base bridge. Rewards on both the probe and bridge transitions
are excluded. Matched roots and step IDs make noise common across probe and
decision alternatives. Sequential `lax.map` changes only execution memory,
not the random variables or estimator.

The successor predictor conditions on the pre-probe legal state and probe. It
therefore models an expectation over the stochastic probe successor, the
unforced bridge action, the teammate reaction, and environment transition; it
is not a deterministic simulator-state value oracle.

## 8. Delayed response causal timing

In the simultaneous-action environment, the partner action at time `t` cannot
condition on ego action `a_t`. The earliest policy reaction is the partner
action selected at `t+1`. Therefore the delayed target compares the
intermediate and delayed observations. Removing the second ego action's direct
physical effect leaves a legal observable response attributable to the joint
successor dynamics under the registered continuation distribution.

This target is causal with respect to probe timing but remains observational:
other state changes and partner stochasticity are integrated by the learned
conditional distribution.

## 9. Exact 66-outcome normalization

The delayed compact outcome distribution contains:

- two `availability=0` outcomes;
- two `availability=1, change=0` outcomes;
- sixty-two `availability=1, change=1` event outcomes.

Shared Bernoulli factors are broadcast across components; the 31-class event is
normalized per component. Summing the 66 exponentiated log probabilities equals
one for every probe and component up to floating-point error.

## 10. Non-negativity of exact decision VOI

For fixed probe-conditioned successor matrix `mu^a`,

\[
V^a(b)=\max_{a'}\sum_k b(k)\mu^a_k(a')
\]

is convex in `b`. A Bayes posterior is a martingale under the predictive outcome
distribution, so Jensen's inequality gives

\[
\mathbb E_y[V^a(b^{a,y})]-V^a(b)\ge0.
\]

The code sums all 66 outcomes exactly. Negative values can therefore only be
floating-point artifacts or malformed inputs; they are reported, not modified.

## 11. Information is not decision value

If every component has the same successor action-value vector, `V^a` is
independent of belief and VOI is zero even if the response identifies the
component perfectly. Conversely, action-independent response information can
produce the same positive VOI for every probe and therefore no active policy
change. v4 consequently reports action-wise VOI spread, information-gain
spread, and active/passive policy total variation in addition to their means.

## 12. Mirror-policy guarantee

For finite action values and `delta>0`, maximizing expected supplied value under
`D_KL(pi||pi0)<=delta` has the exponential-tilt solution. Bisection finds the
active boundary when required. This guarantees optimality only for the supplied
model-based values. Real-return improvement requires accurate response and
decision models.

## 13. Identifiability boundary

Finite mixtures are permutation-invariant. Centering and simplex initialization
remove a harmful symmetric fixed point but do not establish unique semantic
labels. Scientific usefulness requires all of the following empirical links:

1. semantic component distributions differ;
2. legal histories select different mixtures for different partners;
3. component decision residuals induce different action orderings;
4. the correct belief has higher same-world continuation value than a shuffled
   belief.

Posterior sharpness alone is insufficient; a partner-independent global winner
fails conditions 2 and 4.

## 14. Approximation boundary

v5 is not a full Bayes-adaptive POMDP solver. The delayed head predicts one
reaction window, and successor values integrate one registered base bridge plus
the horizon-`H` continuation. VOI values only the first decision at `t+2` after
that response, and its incremental control contribution is discounted by
`gamma^2`. It does not recursively price all future information.

The successor state entering VOI is a *learned prediction*
`chi(x_t, b_t, a_t, y)` of the `t+2` encoder features, not the simulated world.
Its error is therefore method error, and it is reported rather than assumed
away: the audit records the successor model's held-out feature error against
the "nothing moves in two steps" identity baseline, and the gap between the
critic evaluated at the predicted landing state and at the realised one. A
successor model that cannot beat the identity baseline contributes nothing that
VOI could use, and the reported numbers say so directly.

The method also assumes the episode-static latent is an adequate summary of
partner-relevant convention uncertainty. Continuous within-partner adaptation
that cannot be represented by legal history features and `K` exchangeable
residual modes remains model error.

## 15. Explicit non-claims

The implementation does not prove:

- recovery of partner identity or training algorithm;
- unique latent semantics;
- calibrated posterior from entropy alone;
- real-return improvement from mirror adaptation without accurate values --
  the same-world mirror improvement against the held-out anchor replicas is the
  measurement that bears on this, and it is a reported diagnostic, not a claim;
- SOTA performance without formal frozen matrices;
- exact long-horizon active planning;
- independence of behavior statistics from partner history.
