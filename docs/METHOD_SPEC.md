# METHOD_SPEC — Unified DELTA-ZSC v4 algorithm

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
movement and stationary ego. Static channels 20:29 must match on exactly one
candidate overlap. `stay/interact` use only the stationary candidate. For
`interact`, the ego-facing cell is removed before dynamic comparison. Ambiguous
or failed alignment is unavailable. Cross-episode transitions are masked.

Dynamic channels 29:34 produce:

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

## 10. Current decision emission

Each component predicts centered all-action returns using

\[
\mu_k(x)=\mu_0(x)+\Delta\mu_k(x),\qquad
\sum_k\Delta\mu_k(x)=0.
\]

Both the shared mean and residual are centered across actions. Model variance
is predicted once by the shared branch and broadcast across components, so a
component cannot win the mixture by altering only its uncertainty.

Action returns are scored in the orthonormal five-dimensional Helmert contrast
subspace. The Gaussian covariance is the projected shared model variance plus
the full CRN measurement covariance.

## 11. Probe-conditioned successor decision emission

For each forced probe `a`, a second decision head predicts

\[
\mu^a_k(a')
\]

for every candidate decision action `a'` at `t+2`. Its input is the legal
pre-probe state and a learned probe-action embedding. It integrates the probe
transition and one unforced collection-time-base bridge, after which the
delayed response is observable.

The CRN target is constructed as follows:

1. clone the sparse pre-action world;
2. force probe `a` using matched randomness across probe alternatives;
3. discard the reward on this probe transition;
4. if nonterminal, execute one unforced base-policy bridge with matched
   randomness and discard its reward;
5. if the bridge is nonterminal, clone the `t+2` world and force each decision
   action `a'` with matched randomness;
6. continue for `H` reward-bearing steps under the base policy;
7. split fit and evaluation replicas before scoring.

Probe and post-response decision actions are traversed by nested `jax.lax.map`, so only
`anchor_count x replica_count` worlds are live at once. The estimator is the
same full `A x A` matrix without the materialized `A^2` memory multiplier.

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

## 13. Three channel-normalized proper scores

Immediate and delayed shared observations form one shared channel; immediate
and delayed semantic observations form one semantic channel; current and
successor CRN observations form one decision channel:

\[
L_{sh}=-\frac{\sum\log p_{sh}}{N_{sh}},\qquad
L_{sem}=-\frac{\sum\log p_{sem}}{N_{sem}},\qquad
L_{dec}=-\frac{\sum\log p_{dec}}{N_{dec}}.
\]

The registered latent objective is

\[
L_{latent}=L_{sh}+L_{sem}+L_{dec}.
\]

Every present channel has fixed coefficient one. A missing channel contributes
zero. No tunable auxiliary weight, entropy term, separation loss, pseudo-label,
or partner classifier is introduced. The optimizer reports the composite and
parameter-group gradient norms. Exact semantic/decision channel norms and
cosine are computed report-only on the shared component embeddings in the final
anchor audit; they never rescale the objective or replicate the complete
training backward graph.

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
- `delta_passive`: current decision emission and passive mirror adaptation;
- `delta_active`: all v4 channels, successor decision, and delayed exact VOI.

Only `K`, `H`, and `delta` are registered scientific method fields. Network
widths, optimizer settings, fixed initializer norm, replica counts, and budgets
are engineering or measurement settings.
