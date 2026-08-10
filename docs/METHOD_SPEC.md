# METHOD_SPEC — Unified DELTA-ZSC v5 algorithm

`authoritative: true`

## 1. Variables and legal history

An episode is indexed by `e`; its latent coordination mode is
`z_e in {1,...,K}`. The teammate checkpoint is fixed inside the episode and is
resampled only at a real terminal boundary. `H_t` contains only deployable
local observations, executed ego actions, episode boundaries, and deterministic
statistics computed from that history.

Training has three observation channels:

1. immediate shared and semantic response observations;
2. delayed probe-response observations;
3. sparse CRN current and probe-successor all-action returns.

Counterfactual returns and partner lineage never enter deployment state.

## 2. Base task policy

`base_params` contain one recurrent task actor-critic. Explicit teammate
semantic channels are masked from the recurrent task input; a separate
memoryless encoder reads current teammate geometry. `history_rnn` disables this
history mask as a capacity control.

All training rollout actions and unforced anchor transitions use the
collection-time base policy. Current anchors force their measured action.
Active anchors force the probe, execute one unforced base bridge while the
teammate reacts, then force the measured post-response action at `t+2`. PPO is
therefore on-policy with respect to the base policy and independent of latent
parameters.

## 3. Legal behavior statistics

Six Beta(1,1) statistics are maintained per episode:

- visibility;
- visible movement;
- visible inventory occupancy;
- visible inventory change;
- aligned interface change;
- recipe change.

The feature map exposes posterior mean and bounded log-precision. Statistics
reset at episode start and otherwise accumulate without decay. They are legal
history features; they are not a substitute posterior and are never trained.

## 4. Episode-static latent prior

There is no physical-time transition matrix. The prior used at time `t` is

\[
\bar b_t(k)=
\begin{cases}
1/K,&\text{episode start},\\
b_{t-1}(k),&\text{otherwise}.
\end{cases}
\]

This matches the registered data-generating process: the partner member is
constant throughout a 400-step episode and changes only after `done`.

## 5. Immediate response extraction

The direct response reads partner visibility and, when visible, relative
position, direction, inventory, and visible-at-both-ends inventory change.

The interface extractor uses two candidate frame alignments: successful ego
movement and stationary ego. Channel blocks are derived from the ingredient
count encoded by the Official observation shape. Simple uses static/dynamic/
recipe starts `20/29/34` in a 39-channel frame; Wide uses `22/32/38` in a
43-channel frame. Static channels must match on exactly one candidate overlap.
`stay/interact` use only the stationary candidate. For
`interact`, the ego-facing cell is removed before dynamic comparison. Ambiguous
or failed alignment is unavailable. Cross-episode transitions are masked.

The derived dynamic block produces:

- interface availability `M`;
- change indicator `C`;
- 31-class event `E` when `M=C=1`.

The event support is `3 facilities x 2 appeared/disappeared x 5 objects`, plus
`OTHER/MULTI`. Pot ingredient count changes and cooked-state changes have the
registered structured encoding. A goal event requires a real dynamic-plane
change. Recipe change uses an independent same-cell coverage mask.

## 6. Shared occurrence / component semantic factorization

The complete immediate likelihood is

\[
p(y_t\mid z_e,H_t,a_{t-1})=
 p_{sh}(y_t^{occ}\mid H_t,a_{t-1})
 p_{sem}(y_t^{sem}\mid z_e,H_t,a_{t-1}).
\]

### 6.1 Shared occurrence factors

The following heads have no component axis:

- visibility;
- inventory-change occurrence when visible at both ends;
- interface availability;
- interface change/no-change when available;
- recipe change/no-change when covered.

They are proper predictive observations and contribute to `L_shared`, but
cannot change normalized component responsibilities.

### 6.2 Component semantic factors

The following conditional factors have a component axis:

- position, direction, and factorized inventory when visible;
- interface event type when an available change occurred.

Only their summed log likelihood is passed to the Bayes filter.

## 7. Centered residual response emission

Every semantic head has a pooled baseline and a centered component residual:

\[
L_k(x)=L_0(x)+\Delta L_k(x),\qquad
\sum_k\Delta L_k(x)=0.
\]

A residual contains:

- a context projection from a component-conditioned hidden state;
- a direct projection from the component embedding;
- for the event head, a trainable centered initializer bias.

Residual projections use standard fan-in scale. Shared pooled output heads keep
small near-zero initialization. Centering prevents the residual branch from
changing the pooled mean and prevents a component-independent offset from
masquerading as specialization.

The pooled event baseline remains factorized into facility, direction, object,
and `OTHER/MULTI`, then normalized jointly over 31 classes. The component
residual is a full 31-class vector so it can express convention contrasts such
as ingredient-class swaps.

## 8. Unlabeled spectral-simplex initialization

For legal event samples, let `X` contain the full current frame, behavior
features, and ego action. Every frame coordinate enters a fixed, label-free
Rademacher projection; the projected frame is concatenated with the uncompressed
behavior/action features. A deterministic five-fold episode-disjoint pooled
softmax model then estimates `q0(E|X)`. The fixed projection is calibration
instrumentation recorded in the initializer artifact, not a learned DELTA
mechanism or a source of partner labels. For episode `e`,

\[
r_e=\frac{1}{\sqrt{n_e}}
\sum_{t\in e,C_t=1}
[\operatorname{onehot}(E_t)-q_0(E_t\mid X_t)].
\]

SVD supplies up to `K-1` leading residual directions. A regular centered
`K`-vertex simplex is projected onto those directions and scaled to fixed
logit norm 0.5. Deterministic orthogonal completion handles low rank.

The initializer uses no partner ID, mechanism label, return, or hidden state.
A parent-disjoint SP/OP conditional oracle may be measured in the same
calibration application, but it is diagnostic-only and is not read by the
initializer or training code.

## 9. Response-only episode-static filter

Let `ell_sem_t(k)` be the summed conditional semantic log likelihood. The legal
posterior is

\[
b_t(k)=\frac{\bar b_t(k)e^{\ell^{sem}_t(k)}}
{\sum_j\bar b_t(j)e^{\ell^{sem}_t(j)}}.
\]

If no semantic factor is observed, every `ell_sem_t(k)=0` and the posterior is
unchanged. Shared occurrence factors are excluded exactly, not approximately.
Decision and delayed-probe observations never update online belief.

## 10. Belief-conditioned action value

The decision side estimates one quantity: the raw-return action value under the
current posterior,

\[
Q_\psi(x_t,b_t,a)=V_\psi(x_t,b_t)+A_\psi(x_t,b_t,a),\qquad
\sum_a \pi_0(a\mid x_t,b_t)\,A_\psi(x_t,b_t,a)=0.
\]

The value head carries the state's level and the advantage carries only the
action contrast the mirror step consumes. `E` advantage heads share a trunk and
differ in initialisation; their spread is *reported*, never fitted, because a
variance the objective can shrink measures nothing.

Component-conditional values are the same function at each one-hot posterior,
\(Q_\psi(x_t,e_k,\cdot)\). There is no separately parameterised component head.
The previous formulation predicted `K` centered component means under a Gaussian
mixture likelihood; measured with fixed parameters, holding both residual
branches at zero changed the fitted training NLL by 0.082 of 7.55, so the
decomposition was not identified by the data. Each anchor observes one real
partner's return vector, never `K` of them, and no cross-component paired
counterfactual exists to recover it from.

Two channels train \(\psi\), both on raw task reward so that they estimate the
same quantity:

1. **TD(lambda) on every rollout step.** Every step supplies
   \((x_t,b_t,a_t,r_t,x_{t+1},b_{t+1})\). The bootstrap comes from a Polyak
   target copy \(\bar\psi \leftarrow \tau\psi + (1-\tau)\bar\psi\), so the
   critic the mirror reads does not chase its own estimation noise. Shaped
   reward is excluded: the PPO critic keeps it because it is training task
   competence, but this target must match the anchor contrasts, which accumulate
   raw reward.
2. **Pairwise CRN contrast calibration.** See section 11.

## 11. Pairwise CRN action contrasts

The anchor target is the same-replica action *difference*

\[
\hat\Delta_{ab}=\tfrac1R\sum_r\bigl(G^{(r)}_a-G^{(r)}_b\bigr),
\]

with the standard error of that difference. Replica `r` of action `a` and
replica `r` of action `b` share one CRN draw, so the partner, the environment
noise and the continuation draw cancel before averaging.

Estimating `A` means separately and letting a density explain the vector could
not separate best from second best on a single anchor: median best-second margin
0.0004 against median replica standard error 0.0031, no anchor above two
standard errors, and an exact tie in every row -- so `argmax` was picking a
winner by index order.

Contrasts enter as a precision-weighted regression on the critic's predicted
differences. The weight is \(1/(\sigma^2_{ab}+\bar\sigma^2)\) where
\(\bar\sigma^2\) is the batch's mean contrast variance. Plain inverse-variance
weighting is wrong here: under CRN, two actions whose continuations re-merge
give bit-identical returns and therefore zero sample variance, which would hand
"these two actions are exactly equal" near-unbounded weight. Adding the pooled
variance bounds the weight above, leaves well-measured pairs at
\(1/\sigma^2_{ab}\) asymptotically, and introduces no tunable constant.

Nothing is thresholded away. Unresolvable pairs still train "these two actions
are close", which is the honest content of the measurement.

## 11a. Two-tier anchor measurement

Every candidate world receives a cheap low-replica pilot pass; only the states
whose largest pairwise SNR is highest are measured at the registered replica
budget. Selection is on the pilot's own replicas and the retained states are
measured again under fresh keys, so the contrasts that reach the loss are not
the ones that won the selection. What selection biases is which states the
decision head trains on -- deliberately, as the task-stage stratification also
does -- not the value measured at them.

Candidate indexes are drawn exactly without replacement by Floyd sampling.
Its state scales with the requested anchor count rather than the full
`rollout_length x environment_count` grid. At the formal `256x256` shape this
avoids the full random sort that exceeded the L40 shared-memory limit while
preserving the same uniform subset distribution.

## 11b. Probe-conditioned successor state

For each forced probe `a`, the CRN target is constructed as follows:

1. clone the sparse pre-action world;
2. force probe `a` using matched randomness across probe alternatives;
3. discard the reward on this probe transition;
4. if nonterminal, execute one unforced base-policy bridge with matched
   randomness and discard its reward;
5. if the bridge is nonterminal, clone the `t+2` world and force each decision
   action `a'` with matched randomness;
6. continue for `H` reward-bearing steps under the base policy;
7. split fit and evaluation replicas before scoring.

Probe and post-response decision actions are traversed by nested `jax.lax.map`,
so only `anchor_count x replica_count` worlds are live at once.

Alongside this, a learned successor model predicts the `t+2` encoder features

\[
\chi(x_t,b_t,a_t,y)\;\longrightarrow\;(\text{task},\ \text{instant})_{t+2},
\]

conditioned on the delayed response outcome `y`. It is supervised by the
encoder's own `t+2` features, taken stop-gradiented, and by the critic's TD
target at that state. Active VOI then evaluates \(Q_\psi\) at the predicted
landing state rather than at the current one: pricing a probe at the current
state charges it two steps of delay and credits it with none of the position
those steps buy.

**Registered approximation.** The model is trained on the *observed* `y`, but at
VOI time the successor is evaluated once per probe at the expected outcome
\(\bar y_a = E[y \mid a]\), not once per enumerated outcome:

\[
E_y\bigl[Q_\psi(\chi(y))\bigr]\;\approx\;Q_\psi(\chi(\bar y_a)).
\]

Enumerating a landing state per outcome means a
`[time, lane, probe, 66, component]` tensor -- 52 million critic rows at the
registered formal shape, tens of gigabytes, which the final audit materialises
across every timestep. The quantity is well defined; the materialisation is
not. What the approximation assumes is that the landing state varies smoothly
in the response, **not** that the response is unimportant: the posterior VOI
prices is still integrated over all sixty-six outcomes exactly.

## 12. Delayed probe-response emission

Immediate response remains the passive-filter target. Active probing uses a
separate target:

\[
Y^{probe}_t = g(o_{t+1},o_{t+2},a_{t+1}).
\]

`a_t` is the probe input. The teammate can first react to `a_t` when choosing
its action at the intermediate state. The extractor therefore compares
`o[t+1] -> o[t+2]` and uses `a[t+1]` only to align frames and remove the direct
physical effect of the ego continuation action. The window is invalid if either
intervening transition is terminal.

The delayed head predicts shared visibility/availability/change and a
component-semantic 31-class event. It is trained only for `delta_active` and is
never inserted into deployment history after the fact.

## 13. Response proper scores and direct decision objectives

Immediate and delayed shared observations form one shared proper-score channel;
immediate and delayed semantic observations form one semantic proper-score
channel:

\[
L_{sh}=-\frac{\sum\log p_{sh}}{N_{sh}},\qquad
L_{sem}=-\frac{\sum\log p_{sem}}{N_{sem}}.
\]

Decision value is supervised directly by raw-reward TD(lambda) on every rollout
step and by measured pairwise CRN differences on anchor updates. Active DELTA
also fits the two-step successor feature and its value consistency. The
registered latent transaction is

\[
L_{latent}=L_{sh}+L_{sem}+L_{raw\ TD}+L_{pairwise\ CRN}+L_{successor}.
\]

Every present term has fixed coefficient one; an inapplicable term contributes
zero. No tunable auxiliary weight, entropy term, separation loss, pseudo-label,
or partner classifier is introduced. The optimizer reports the composite and
parameter-group gradient norms. Exact semantic/decision norms and cosine are
computed report-only in the final anchor audit; they never rescale the
objective or replicate the complete training backward graph.

## 14. Alternating estimator transaction

For rollout `D_n`, anchors `C_n`, base tree `omega_n`, and latent tree
`Theta_n`:

\[
\Theta_{n+1}=\operatorname{MLE}(\Theta_n;D_n,C_n,\operatorname{stopgrad}\omega_n),
\]

\[
\omega_{n+1}=\operatorname{PPO}(\omega_n;D_n).
\]

The latent transaction is committed before any PPO minibatch changes
`omega_n`. Anchor labels and base features therefore refer to the same
collection-time policy.

## 15. Passive mirror adaptation

The posterior expected current action value is

\[
\bar Q_t(a)=\sum_k b_t(k)\mu_k(a).
\]

DELTA solves

\[
\max_\pi\langle\pi,\bar Q_t\rangle
\quad\text{s.t.}\quad
D_{KL}(\pi\Vert\pi_0)\le\delta.
\]

The solution is `pi_eta(a) proportional pi_0(a) exp(Q(a)/eta)` with deterministic
bisection on `eta`.

## 16. Exact delayed-response active VOI

For probe `a`, the compact delayed outcome is
`(visibility, availability, change, event)`. Conditional geometry and recipe
are marginalized out. The exact support contains 66 outcomes.

For every outcome `y`,

\[
b^{a,y}(k)\propto b_t(k)p_\phi(y\mid k,H_t,a).
\]

Using the probe-conditioned successor matrix,

\[
V^a(q)=\max_{a'}\sum_k q(k)\mu^a_k(a'),
\]

\[
VOI(a)=\sum_y p(y\mid a)V^a(b^{a,y})-V^a(b_t).
\]

The active current value is `Q_current(a) + gamma^2 * VOI(a)`. Exact enumeration
has no sample count or quadrature field. Small negative values are reported as
floating-point diagnostics and are never clamped. Information gain is reported
but never added to control reward.

## 17. Variants

- `history_rnn`: generic recurrent capacity control;
- `base`: shared PPO task policy only;
- `response_only`: shared/semantic immediate response prediction and legal
  posterior, no decision adaptation;
- `delta_passive`: belief-conditioned critic and passive robust mirror adaptation;
- `delta_active`: all v5 channels, successor decision, and delayed exact VOI.

Only `K`, `H`, and `delta` are registered scientific method fields. Network
widths, optimizer settings, fixed initializer norm, replica counts, and budgets
are engineering or measurement settings.

## 18. Registered partner and initializer binding

Training samples the partner manifest uniformly over mechanism, family, stage
and run for the complete run; observed performance never changes those
probabilities. Formal seed `s in 0..9` initializes the base policy from
development-support SP parent `s`. Development seed `s in 0..4` uses the same
mapping, and every variant at that seed receives the same parent parameters.
The initializer collector and the 256-lane CUDA engineering run use parent 0.

These paths and their parent/co-training lineage are recorded in run identity.
They do not become deployment inputs. The upstream jobs, initializer,
engineering execution, development matrix, formal training and evaluation form
one dependency graph; no diagnostic value or development return controls
whether a downstream dependency is scheduled.
