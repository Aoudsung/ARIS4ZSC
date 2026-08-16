# METHOD_SPEC — CETR-ZSC method contract

`authoritative: true`

## 1. Identity and scope

CETR-ZSC means constrained episodic tail-robust zero-shot coordination. Its
active method identity is
`constrained_episodic_tail_robust_zsc_v1`, and its registered configuration is
owned by [`src/cetr_zsc/config.py`](../src/cetr_zsc/config.py). The resolved
configuration contract records `version: 6`; all other registered budgets,
seeds, schema fields, and sample sizes must be read from that module rather than
copied into this document.

The method learns one partner-agnostic recurrent actor. It does not infer a
partner label, select an execution mode, or route deployment through a
partner-specific controller. The scientific objective is a constrained
external-partner tail objective defined on completed episodes.

## 2. Notation and legal history

For episode `e`, let `o_t` be the complete local observation and `h_t` the
actor's recurrent carry. The deployable policy is

\[
\pi_\theta(a_t\mid o_t,h_t).
\]

The legal local history contains only observations exposed to the ego, executed
ego actions, recurrent state derived from that history, and episode boundaries.
A completed episode has raw rewards `r_{e,0},\ldots,r_{e,T-1}` and return

\[
G_e=\sum_{t=0}^{T-1}r_{e,t},
\qquad
R_{e,t}=\sum_{u=t}^{T-1}r_{e,u}.
\]

The training estimand is the complete undiscounted raw episodic return. Under the
registered Overcooked protocol this is one complete 400-step episode per lane;
the exact environment and run values remain defined by `config.py`.

A partner parent is a frozen external training or evaluation source. All parent
labels, mechanism labels, checkpoint stages, and co-training lineage are
training and evaluation bookkeeping. They are not inputs to `\pi_\theta`.

## 3. Primary constrained objective

Let `g` index independent parent groups. A group contains all registered
checkpoint stages for one parent; stages are sampled within the group and do not
create additional independent parents. Let `J_g(\theta)` be the expected
completed-episode raw return against parent group `g`.

The nominal parent distribution is

\[
p_{0,g}=p_{0,\text{mechanism}(g)}\,p_{0,g\mid\text{mechanism}(g)},
\]

with equal mechanism mass over SP, OP, SA, and FCP and equal parent mass within
each mechanism, as registered in `config.py`. The external uncertainty set is
the parent-level lower-half CVaR set

\[
\mathcal Q_{1/2}(p_0)=
\left\{q\in\Delta:
0\le q_g\le 2p_{0,g}\right\}.
\]

The external objective is

\[
\rho_{\mathrm{ext}}(\theta)=
\min_{q\in\mathcal Q_{1/2}(p_0)}
\sum_g q_g J_g(\theta).
\]

The cap and tail mass are method identity constants, not YAML fields, sweep
parameters, or values selected from development returns. When the nominal parent
mass is uniform and the parent count is even, this objective is the mean return
of the worst half of independent parents. It is a lower-half tail objective, not
a hard minimum and not a population cross-play surrogate.

CETR solves the constrained problem

\[
\max_\theta\ \rho_{\mathrm{ext}}(\theta)
\quad\text{subject to}\quad
J_{\mathrm{SP}}(\theta)\ge\tau_{\mathrm{SP}}.
\]

The confirmatory endpoint is the same external-partner estimand on a
lineage-disjoint held-out panel. A population matrix is supplementary and does
not replace this primary problem.

## 4. Identity contract I: reference-derived self-play constraint

For training seed `s`, the constraint target is derived from the measured,
seed-matched Official-SP reference:

\[
\tau_{\mathrm{SP}}^{(s)}=
\widehat J_{\mathrm{SP}}(\pi_{\mathrm{ref}}^{(s)}).
\]

The reference is evaluated with the registered environment and evaluation key
schedule. The `cetr_reference_sp` artifact is version 2: its `source_checkpoint`
is a resolved absolute path, and training requires that path to be textually
identical to the resolved `--sp-initializer` path. There is no artificial
tolerance band, hand-tuned threshold, SP margin, or configured SP loss weight.
The reference has exactly three roles:

1. initialize the single trainable actor with a seed-matched Official-SP
   checkpoint;
2. provide the measured `\tau_SP` target;
3. remain a frozen audit baseline for the constraint comparison.

The reference is not a deployment branch, is not an online teacher, and does not
supply an action correction after training. The target is a measured contract,
not a promise that the constraint will be satisfied before the formal result is
run.

## 5. Identity contract II: parent-level lower-half tail risk

For a completed batch, estimate each parent return from its raw episodic
returns. The optimizer of the finite linear program is obtained by sorting
parent estimates in ascending order and filling mass under `q_g <= 2p_{0,g}`
until total mass is one. This produces the lower-half tail weights without a
new objective or a learned partner score.

External lane assignment is deterministic and covers every registered support
parent on every update. The formal support is the four-mechanism × four
independent-parent panel registered in `config.py`, yielding sixteen parent
groups and forty-eight stage members; each parent exposes all registered
checkpoint stages, which remain one parent group. Development and mechanical
runs use the corresponding one-parent-per-mechanism support from the same
configuration authority. For each parent, the four external lanes are the
fold-by-role cells `A0`, `A1`, `B0`, and `B1`; the stage slot is rotated by
`(update_index + lane_slot) mod 3`. Parent selection is not random and there is
no observed-subset re-normalization.

The external risk calculation is deliberately separated from the PPO surrogate.
The PPO actor loss is an estimator using already-computed risk weights; its
numerical value is not itself the definition of the worst-parent return. A
single noisy low-return episode must not determine its own tail membership and
its update direction. Therefore the external batch is split into two folds:

- fold A completed returns determine `q^A`, which weights only fold B's PPO
gradient;
- fold B completed returns determine `q^B`, which weights only fold A's PPO
gradient.

The monitored cross-fitted tail value is
`0.5 * (q^A · J_hat^B + q^B · J_hat^A)`, with each fold's weights evaluated
against the other fold's parent return estimates. This double cross-fitting is
part of the method contract. It is not an optional analysis or a post-hoc
variance correction.

## 6. Actor, baseline, and training signal

The actor is one Official-isomorphic CNN-to-GRU recurrent network followed by an
actor head. It starts from a seed-matched Official-SP initialization and the
whole actor tree is trainable. There is no separate partner encoder, latent
component bank, posterior, mirror controller, or partner-conditioned policy.

A scalar value baseline is trained only as a variance-reduction device for the
policy-gradient estimator. It is not a deployment critic and it never changes
the action logits. The policy signal is the complete undiscounted raw
return-to-go `R_{e,t}`. Advantages are collected from the current policy and
current scalar baseline, fixed once for the update, and reused unchanged across
all PPO minibatches and epochs. `prepare_episode_batch` performs exactly one
shared normalization on the complete `EpisodeBatch`, before minibatch
scheduling and slicing. Group-wise, parent-wise, mechanism-wise, and
minibatch-wise normalization are forbidden because they would erase the
relative scale supplied by the parent risk weights and the SP dual.

## 7. Self-play gradient and dual constraint

A self-play episode executes the current policy on both sides with independent
recurrent carries. Its policy-gradient contribution contains both agents'
log-probability terms, so it is the gradient of the self-composition objective
`J_SP(\theta)=J(\pi_\theta,\pi_\theta)`, rather than a one-sided update against a
stop-gradient snapshot.

The Lagrangian is

\[
\mathcal L(\theta,\lambda)=
\rho_{\mathrm{ext}}(\theta)+
\lambda\bigl(J_{\mathrm{SP}}(\theta)-\tau_{\mathrm{SP}}\bigr),
\qquad \lambda\ge 0.
\]

The training dual update is

\[
\lambda_{n+1}=
\left[\lambda_n+\eta_n
  \bigl(\tau_{\mathrm{SP}}-\widehat J_{\mathrm{SP},n}\bigr)\right]_+,
\qquad \lambda_0=0.
\]

`\eta_n` reuses the actor optimizer learning-rate schedule. There is no upper
cap, independent dual learning rate, tolerance, or hand-written auxiliary
coefficient. The dual variable is a training state variable and never enters
the deployment state.

## 8. Training transaction

Each update is one closed primal-dual transaction:

1. freeze the current actor and scalar baseline for collection;
2. collect complete episodes, with deterministic self-play and fully covered
   external lanes arranged as required by the registered panel design;
3. compute complete raw episodic returns;
4. compute the closed-form cross-fitted parent tail-weight sets from those raw
   returns;
5. prepare the complete `EpisodeBatch`, compute/fix return-to-go advantages, and
   apply exactly one shared normalization before any minibatch split;
6. apply one standard clipped PPO primal update using the external risk weights
   and the self-play constraint contribution;
7. apply one dual update from the measured self-play batch;
8. record the resolved identity and continue end-to-end with the updated actor.

There is no pretraining/freeze/gate phase, no best-checkpoint selection, and no
restart for a poor score. The SP-to-external lane ratio is a sampling design
specified by the registered configuration, not an additional scientific
parameter.

## 9. Deployment boundary and legal information

The only deployment computation is

\[
(o_t,h_t)\ \longrightarrow\ \pi_\theta\ \longrightarrow\ a_t.
\]

The actor may read its own local observation and recurrent history, including
episode boundaries and actions it previously executed. It may not read partner
run ID, algorithm, checkpoint index, family label, mechanism, parent group,
co-training lineage, hidden simulator state, future response, counterfactual
return, `q`, `\lambda`, or any training-only manifest field. Parent groups, tail
weights, cross-fitting folds, the reference target, and the dual variable are
optimization-time variables and are absent from the deployment bundle and
runtime state.

The scalar value baseline is also absent from action selection. The deployment
bundle is version 7 and contains only the actor parameter subtree needed by the
runtime. The reference-SP artifact and training manifest are not deployment
inputs; they are stored as provenance records in the same bundle directory,
including `provenance.json`, for audit binding only. Deployment does not contain
a posterior update, a partner classifier, an active probe, a VOI path, or a
second execution policy.

## 10. Retired boundary

The former V6 DELTA mechanisms are historical failure modes, not active
alternatives to this contract. The response latent and posterior, belief-
conditioned decision critic, continuation anchors, mirror/VOI controller,
group-wise minimax loss, and one-sided snapshot self-play update were retired or
deleted. They are not imported, wrapped, or silently reproduced by CETR. V6 and
DEPI artifacts are version-incompatible and preserved only in git history.
