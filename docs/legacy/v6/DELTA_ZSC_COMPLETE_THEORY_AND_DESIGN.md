# DELTA-ZSC V6 End-to-End Bayes Coordination (legacy)

This is the binding design for the active branch. Code, configuration,
checkpoints and deployment artifacts must agree with it in one commit.

## 1. Registered identity and immutable benchmark boundary

```text
METHOD_VERSION = delta_zsc_v6_end_to_end_bayes_coordination
CONFIG_VERSION = 9
MANIFEST_VERSION = 2
OFFICIAL_PROTOCOL_VERSION = overcooked_v2_iclr2025_5ce1707_v1
OFFICIAL_SOURCE_COMMIT = 5ce1707cf31c1c115e6f6ba96db7bc9cc80a850e
```

The fixed Official environment, evaluator, episode keys, stochastic action
sampling, 400-step raw return and ordered 10×10 matrix define the performance
experiment. V6 may use a different training algorithm, but at test time the
policy may read only the current Official observation, `done`, its own recurrent
state and the action key.

V5 checkpoints and training-only artifacts are not compatible with V6. The only
objects that are fixed during normal V6 learning are the Official protocol,
run-disjoint data partitions, EMA target networks and the mechanically final
checkpoint. No scientific statistic selects a checkpoint or seed.

Formal CUDA execution remains float32 and fixes JAX matmul precision to
`highest`; this avoids batch-shape-dependent TF32 differences in the bounded
decision-regret chunks without introducing mixed precision or changing a model
hyperparameter.

Every partner-manifest entry records the real two-word JAX PRNG key together
with its checkpoint hash, parent run and co-training lineage. Missing key
provenance is invalid even for non-Official initialization or support partners.

## 2. Scientific object

Let (H_t=(o_0,a_0,d_0,\ldots,o_t)) be the legal ego history. Reward is absent
because the Official policy interface does not provide it. The model learns

\[
q_\phi(z_t\mid H_t)=\mathcal N(\mu_t,\operatorname{diag}\sigma_t^2)
\]

and a single shared policy

\[
\pi_\theta(a_t\mid x_t,b_t),\qquad
b_t=[\mu_t,\log\sigma_t,\bar H(q_t)].
\]

The purpose of (z_t) is not to classify partner identity. It is to retain the
history information that changes the value ordering of ego actions. A single
Gaussian is the registered first model; multimodality is not assumed without
evidence.

## 3. Deployable model

### 3.1 Task history

The task path is the Official CNN, LayerNorm and GRU-128 backbone. It consumes the
current observation, previous ego action and previous episode boundary. It
receives gradients from PPO, shaped value and raw-Q, but never from response
reconstruction.

### 3.2 Partner belief

The belief cell consumes the previous and current legal observations, previous
ego action, previous episode boundary and its carry. It outputs eight means and
eight log standard deviations clipped to `[-5, 2]`. Prior mean and log standard
deviation are zero. Normalized uncertainty is the mean log-standard-deviation
position in the registered interval.

The behavior actor uses only the deterministic 17-dimensional posterior summary.
The raw-Q regret calculation may draw 16 reparameterized particles from the same
posterior.

### 3.3 One actor and three value objects

There is one low-rank continuously conditioned actor. There is no base actor,
residual actor, hard gate or fallback route. The critic exposes semantically
different outputs:

\[
V_\nu^{\rm shaped}(x,b),\quad
Q_{\psi_1}^{\rm raw}(x,b,a),\quad
Q_{\psi_2}^{\rm raw}(x,b,a).
\]

PPO value fits the shaped training return. Both Q heads fit raw simulator return.
The twin heads are numerical estimators, not partner categories.

### 3.4 Context dropout

At each time/lane a dedicated random domain records a Boolean mask. The behavior
summary is replaced with the prior summary with probability

\[
p_{drop}(s)=0.30-0.20s/S.
\]

PPO replay reuses the recorded mask exactly. Deployment always uses zero dropout.
This trains generalist and context-sensitive behavior inside the same actor.

### 3.5 Structured response

The response decoder predicts only partner-visible semantics: visibility,
relative position including not-visible, visible direction, visible inventory and
visible interaction change. Losses are BCE, categorical cross-entropy and
class-balanced BCE. It does not predict the full observation, reward or done.
Interaction change compares the visible partner's semantic inventory factors at
its previous and current positions; movement alone is never labeled as an
interaction.
Its head gradients cannot update task, actor, shaped value or raw-Q.

## 4. End-to-end objectives

Every objective exists from the first update. Missing data produces a recorded
zero gradient; no qualification statistic disables a loss.

\[
\mathcal L=\mathcal L_{PPO}
+\mathcal L_Q+\mathcal L_{CF}+0.1\mathcal L_{DE}
+\mathcal L_{resp}+0.001\mathcal L_{IB}
+0.25\mathcal L_{Q\pi}+0.1\mathcal L_{robust}.
\]

Official PPO retains four epochs, 64 minibatches, clip 0.2, GAE 0.95, entropy
0.01 and its warmup-plus-cosine learning-rate schedule. Policy gradient reaches
belief with registered weight 0.1. Advantages and value targets are computed
once from rollout behavior values, then held fixed across all four epochs; the
value loss is the Official clipped squared-error maximum. Approximate KL is a
reported diagnostic and cannot skip a registered minibatch.

### 4.1 Dense raw-Q

Every ordinary transition contributes recurrent Retrace with raw reward,
recorded behavior probability, terminal truncation and λ=0.95. Chunk endpoints use
the EMA actor probability and conservative EMA twin-Q expected value. Live heads
fit one stop-gradient target. EMA updates once per outer update with coefficient
0.005.

### 4.2 Smooth Q-to-policy improvement

\[
w_t=\sigma(\operatorname{gap}(Q)-1)
\exp[-\operatorname{mean}|Q_1-Q_2|/5].
\]

The policy target is `softmax(stopgrad(min(Q1,Q2)))` at temperature 1. A flat or
disagreeing Q automatically supplies little weight without a binary gate.

### 4.3 Generalist regularization

The same actor is evaluated with the full summary and the prior summary. The
normalized posterior uncertainty weights

\[
D_{KL}(\pi(\cdot\mid x,b)\,\|\,\pi(\cdot\mid x,b_0)).
\]

Thus uncertain beliefs remain near the learned generalist behavior without a
separate base parameter tree.

### 4.4 Information bottleneck

The belief KL to `N(0,I)` uses 0.1 nat free bits per latent dimension. It limits
checkpoint fingerprint memorization but never decides whether adaptation is
allowed.

## 5. Counterfactual supervision and replay

Every 16 outer updates, including update zero, collect 32 ordinary states and 16
post-evidence matched-code pairs (32 further states). State selection is uniform
within time/source strata and never uses regret or a qualification score.

Each of 64 worlds forces six first actions, four common-random-number replicas and
128 continuation steps. A matched pair first runs two hidden-code branches for 16
steps under paired ego/environment randomness, so only legal visible history can
separate beliefs. Identical pre-evidence histories must produce identical
posteriors.

Each branch return is its 128-step raw simulator-return prefix. If the branch is
still nonterminal, it adds the detached endpoint value under the collection-time
EMA actor and conservative `min(Q1,Q2)` target. Terminal endpoints contribute
exactly zero. Thus an anchor is a versioned truncated-return target, not a claim
that 128 steps form a complete episode; its stored target fingerprint makes
policy drift explicit and replay weighting continuously retires stale labels.

The centered versioned label is

\[
A_{CF}(h,a)=\bar G(h,a)-\frac16\sum_{a'}\bar G(h,a').
\]

Replay capacity is 512 states. Each row stores collection logits/update/target
fingerprint, return sum, squared sum and replica count, matched-pair ID, source and
lineage. Old data has continuous weight

\[
w_i=\operatorname{clip}_{[10^{-3},1]}
\exp[-D_{KL}(\pi_{now}\|\pi_i)/0.05]\exp[-age_i/64],
\]

then batch-mean normalization. No epoch boundary clears replay. Each outer update
draws four 64-row stratified minibatches while preserving matched pairs.

Decision-equivalence supervision is continuous:

\[
L_{DE}=\left(\|\mu_i-\mu_j\|_2-
\operatorname{sg}\frac{\|A_i-A_j\|_2}{s_A+\epsilon}\right)^2,
\]

where (s_A) has EMA decay 0.99 and initial value 1. There are no code-region
labels, clustering gates or artificial parity groups. Both Euclidean distances
use `sqrt(sum(square(delta)) + 1e-12)` so an exactly collapsed or masked pair has
a finite zero gradient rather than the undefined derivative of an unfloored norm.

## 6. Decision-regret shaping

With 16 posterior particles and EMA raw-Q,

\[
R_t=\mathbb E_z\max_a\bar Q(x_t,z,a)-
\max_a\mathbb E_z\bar Q(x_t,z,a).
\]

It is divided by an action-range EMA (decay 0.99, initial 1, epsilon `1e-3`). The
detached shaping is

\[
F_t=\operatorname{clip}(\tilde R_t-\gamma\tilde R_{t+1},-1,1),
\]

with terminal next potential exactly zero. Its weight is

\[
0.1\,\sigma((s/S-0.20)/0.05).
\]

This is a continuous registered schedule, not a C5 switch. The target path cannot
directly update belief or Q.

## 7. Belief gradient contract

PPO, raw-Q, CF, response, decision equivalence, information bottleneck, Q-policy
and robust objectives all differentiate from one parameter snapshot. Head
optimizers mask belief leaves. Belief gradients are EMA-norm normalized (decay
0.99), then instantaneously capped so that each objective's normalized L2 norm
cannot exceed its registered weight. They are processed by PCGrad in exactly that
order and applied once per outer update by an independent optimizer. The cap
covers the EMA adaptation window, so a newly spiking response gradient cannot
erase or dominate decision gradients merely through scale.

The run records every raw loss, raw gradient norm, pre-cap EMA-normalized norm,
post-cap normalized norm, pairwise cosine and final combined norm.

## 8. Continuous partner generator

Training has one live generator and one EMA copy. Four run-disjoint SP/OP/SA/FCP
source policies provide balanced logit distillation initialization; they do not
define deployed categories. Every outer update collects 32 complete 400-step
generator episodes against the updated, frozen-for-collection ego and performs
Official-shaped recurrent PPO with four epochs and eight environment minibatches.

Imitation decreases and decision BR-diversity increases continuously with

\[
\rho(s)=0.75\min(s/(0.30S),1).
\]

Smoothness weight is 0.05. A competence dual uses raw-return CVaR20, learning rate
0.01 and range `[0,10]`; CVaR EMAs use decay 0.95. It never admits, rejects,
rolls back or snapshots a candidate.

Partner sampling is

\[
c=\sigma((CVaR_G-CVaR_{external})/20),\quad p_G=\rho c.
\]

Live and EMA generators each receive `p_G/2`; external partners receive `1-p_G`.
The distribution changes continuously with progress and competence.

## 9. Fixed outer-update order

Every update executes:

1. soft partner mixture;
2. 256-environment × 256-step ego rollout;
3. EMA decision-regret shaping;
4. PPO head updates;
5. dense raw-Q Retrace;
6. scheduled anchor collection or replay only;
7. four raw-Q/CF/DE replay updates;
8. two structured-response updates;
9. one combined belief update;
10. 32 complete generator episodes and generator PPO;
11. EMA, gradient statistics, competence dual and resource ledger;
12. mechanical checkpointing.

Only NaN/Inf, CUDA OOM, checkpoint corruption or Official/run-identity mismatch
may stop the run. Signal magnitudes cannot select another algorithm.

## 10. State, deployment and retrospective audits

The V6 `TrainState` binds online/target parameters, five optimizer states,
live/EMA generator, competence and CVaR statistics, replay and its sampling
counter, belief-gradient/action-range/advantage EMAs, all optimizer counters,
runner/random state, update count and resource ledger. Resume identity
fingerprints every object capable of changing the next update.

The primary artifact `DELTA-ZSC-E2E` always exports the final task encoder,
belief encoder, single actor, critic and response decoder. It never loads an
owner checkpoint as a substitute. `DELTA-ZSC-E2E+Safety` is an optional separate
post-training wrapper and cannot enter the primary method matrix.

C0–C5 names are retrospective readouts only: task competence, partner decision
heterogeneity, raw-Q ranking, online belief recovery, belief-conditioned control
and regret/information-value association. They do not write model state, select
checkpoints or alter samples.

## 11. Exact formal auxiliary budget

For 457 updates, anchors trigger 29 times. Continuations are
`29×64×6×4×128 = 5,701,632`; legal probes are
`29×16×2×16 = 14,848`. Generator training is
`457×32×400 = 5,849,600`. With 29,949,952 ego PPO steps, 65,536 ego
initialization steps and 25,600 generator initialization/holdout steps, the fixed
per-run total excluding upstream partners is 41,607,168 attempted transitions.

These counts are costs, not performance evidence. The runtime ledger remains the
final authority.
