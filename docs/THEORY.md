# THEORY — CETR-ZSC guarantees, approximations, and limits

`authoritative: true`

This document separates algebraic properties of the CETR contract from
finite-sample approximations and empirical claims. It introduces no new
experiment evidence. Registered numerical fields are owned by
[`src/cetr_zsc/config.py`](../src/cetr_zsc/config.py).

## 1. Parent-level lower-half risk

For fixed parent estimates `\widehat J_g` and nominal masses `p_{0,g}`, the
finite optimization problem

\[
\min_{q\in\Delta}\sum_g q_g\widehat J_g
\quad\text{s.t.}\quad
0\le q_g\le 2p_{0,g}
\]

is a linear program. Sorting parent estimates in ascending order and filling the
available mass under each cap gives an optimizer. Under a uniform nominal mass
and an even number of independent parents, its value is the mean of the worst
half of those parents. This is the algebraic meaning of the lower-half CVaR
contract; it is not a claim about an infinite population.

The cap prevents all adversarial mass from concentrating on one parent. It also
means that a single noisy parent cannot have unlimited influence. Neither fact
proves that the resulting policy will be robust on an unseen distribution.

## 2. Cross-fitting boundary

Double cross-fitting uses completed returns from one fold to determine tail
weights for the other fold. Therefore an episode's own observed return is not
used both to select its tail membership and to supply its own weighted update.
This blocks the direct self-selection path in the finite batch estimator.

Cross-fitting does not make the estimate automatically unbiased under arbitrary
adaptive sampling, does not remove parent-level dependence, and does not create
new independent data. Its guarantee is an ordering property of the estimator,
not a generalization theorem.

## 3. Complete episodic return and fixed advantages

The method's target is the complete undiscounted raw return-to-go of one complete
episode. Once collected, the return-to-go and scalar-baseline advantage are fixed
for the whole PPO transaction. Shared normalization preserves a common scale
across self-play and external samples; it does not prove that PPO follows the
exact gradient of the population lower-tail objective.

The scalar value baseline is a control variate for policy-gradient variance. It
has no deployment role and no authority to change the action distribution. A
well-fitted baseline therefore cannot by itself establish better coordination.

## 4. Self-play composition gradient

If both agents in a self-play episode use `\pi_\theta` and have independent
recurrent carries, differentiating the joint trajectory likelihood includes both
sides' log-probability terms. The resulting estimator is the bilateral gradient
of the self-composition return `J(\pi_\theta,\pi_\theta)` for the sampled
trajectory distribution, subject to the usual policy-gradient regularity
conditions. A one-sided snapshot update would estimate a different problem; that
former V6 construction is retired and is not part of CETR.

This identity does not imply low-variance gradients, global optimization, or
self-play improvement after an arbitrary PPO step.

## 5. Lagrangian and dual update

For a fixed policy, the constrained objective has Lagrangian

\[
\mathcal L(\theta,\lambda)=
\rho_{\mathrm{ext}}(\theta)+
\lambda(J_{\mathrm{SP}}(\theta)-\tau_{\mathrm{SP}}),
\quad \lambda\ge0.
\]

The projected update

\[
\lambda^+=[\lambda+\eta(\tau_{\mathrm{SP}}-\widehat J_{\mathrm{SP}})]_+
\]

increases pressure when measured self-play is below target and leaves the dual
variable nonnegative. Reusing the actor learning-rate schedule defines the
registered dual step. This is a standard primal-dual construction, not a proof
of convergence for the non-convex recurrent PPO problem and not a guarantee that
finite-run self-play satisfies the constraint.

## 6. Reference-derived target

The target `\tau_SP` is a measured return of a seed-matched Official-SP reference
under the registered evaluation protocol. It is not a manually selected
threshold and has no artificial tolerance. This removes a hidden tuning degree
of freedom from the method definition, but the estimate still has measurement
error and does not guarantee that a trainable actor can attain it.

The reference initializes the actor, derives the target, and supplies a frozen
audit baseline. It is not a deployment ensemble or a policy correction.

## 7. Legal information and deployment

The deployment state is a recurrent carry derived from the local observation and
ego action history, together with the episode boundary. The actor therefore
implements a partner-agnostic mapping from legal local history to actions.
Training-only parent groups, `q`, cross-fitting folds, `lambda`, reference
metadata, and counterfactual returns are not in this state. This is an
information-boundary property of the architecture, not evidence that local
history is uninformative.

The lineage-disjoint panel is an evaluation design. It does not become a runtime
feature and does not prove universal out-of-distribution robustness.

## 8. What is exact, what is approximate

Exact or structural within the contract:

- the finite lower-half risk optimizer for supplied parent return estimates;
- the bilateral form of the self-play likelihood gradient;
- the nonnegative projection in the dual update;
- the absence of training-only group, tail, and lineage fields from deployment;
- the single actor and scalar-baseline parameter ownership described in the
  method contract.

Approximate or empirical:

- parent returns and `q` are estimated from finite completed episodes;
- cross-fitting reduces estimator reuse but does not remove sampling error;
- PPO and the recurrent function approximation only approximately optimize the
  constrained objective;
- the measured reference target is a finite estimate of reference self-play;
- held-out external mean and lower-half CVaR estimate performance on the
  registered confirmatory panel, not every possible partner;
- the population `10×10` matrix estimates a supplementary ordered population
  object and is not the external-parent estimand.

## 9. Explicit non-claims

CETR does not prove:

- recovery of partner identity, algorithm, mechanism, or hidden state;
- unique semantic protocols or a latent partner taxonomy;
- convergence or global optimality of the primal-dual PPO procedure;
- that a lower-half training objective guarantees every individual partner;
- that reference initialization guarantees self-play non-inferiority;
- that panel-level improvement is SOTA or universal robustness;
- that a population matrix validates the held-out external estimand;
- any performance result before the formal confirmatory artifacts exist.

The former V6 posterior, continuation-anchor, mirror, and VOI machinery was
retired. No theorem or evidence in this document depends on those deleted
mechanisms.
