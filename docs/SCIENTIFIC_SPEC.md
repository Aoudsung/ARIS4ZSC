# SCIENTIFIC_SPEC — Decision-relevant episode-static adaptation

`authoritative: true`

## 1. Registered substrate

The substrate is OvercookedV2 Test-Time Protocol Formation with 400-step
episodes, six ego actions, view radius two, negative rewards, random initial
positions, recipe resampling after delivery, and successful-delivery indication.
`test_time_simple` and `test_time_wide` are separate layouts; neither may hide
failure on the other. The pinned local observation has 39 channels on Simple
and 43 channels on Wide.

## 2. Scientific problem

A previously unseen teammate may express a convention that changes the ego's
best action. The problem is not to identify the teammate's training algorithm.
It is to learn, from legal interaction history alone, an uncertainty state that
supports better decisions while retaining one shared task policy.

Because the teammate checkpoint is constant within an episode, DELTA v5 represents
an exchangeable episode-level latent `z_e`. The legal posterior is

\[
b_t(z)=p(z_e=z\mid H_t),
\]

and its current decision value is learned directly from that posterior:

\[
Q_\psi(x_t,b_t,a).
\]

For active DELTA, candidate probes also receive the value of a delayed partner
response under a probe-conditioned successor decision matrix.

## 3. Legal deployment information

Deployment may use only:

- local observation history;
- previously executed ego actions;
- episode boundaries;
- deterministic behavior statistics computed from observable responses;
- learned shared/semantic response models;
- the episode-static categorical posterior;
- the learned belief-conditioned action value and successor feature model.

Deployment may not use partner run ID, SP/OP label, checkpoint stage, training
family, hidden simulator state, future observations, counterfactual returns, or
manifest lineage.

## 4. Unified latent semantics

A component is exchangeable and is defined by:

1. a conditional immediate semantic response distribution;
2. for active DELTA, a conditional delayed probe-response distribution.

Shared occurrence heads model pooled visibility/change frequencies but do not
define component semantics and cannot alter posterior odds.

A component is scientifically useful only when legal semantic evidence selects
it differently across partners and the resulting posterior changes action
ordering in the belief-conditioned raw-return critic. Posterior entropy
reduction alone is not evidence of adaptation.

## 5. Primary hypotheses

### H1 — final-checkpoint performance

On both Simple and Wide, final-checkpoint `delta_active` exceeds every registered
same-protocol baseline. For each contrast, the one-sided crossed-node bootstrap
lower bound must be positive and the point estimate must be at least one
correct delivery, 20 raw-return points.

### H2 — decision-emission contribution

At fixed `K=4`, partner distribution, base budget, architecture, seed, and
panel:

\[
J(\text{delta_passive})>J(\text{response_only})
\]

on both layouts. This isolates current decision supervision and mirror
adaptation from response prediction alone.

### H3 — causal value of the legal belief

Holding source world, base logits, learned decision matrix, and CRN outcomes
fixed, the correct legal-history belief must produce higher empirical
continuation value than a task-matched shuffled belief:

\[
\mathbb E[(\pi_b-\pi_{\tilde b})^TG_{source}]>0
\]

on both layouts.

Claims are evaluated in the closed order `H1 -> H2 -> H3`.

## 6. Secondary active question

`delta_active - delta_passive` measures whether delayed action-selective
response value improves zero-shot cross-play. It is pre-registered and reported
regardless of sign. Mean VOI or information gain cannot establish active
control; action-wise VOI spread and active/passive policy divergence are
required mechanism evidence.

## 7. Required controls

- `history_rnn`: generic recurrent-history capacity;
- `base`: task competence without latent adaptation;
- `response_only`: legal response posterior without decision adaptation;
- `history_rnn_extra` and `base_extra`: spend anchor simulator cost on ordinary
  PPO interaction;
- `K in {2,4,8}`: bounded capacity sensitivity;
- synthetic uninformative, decision-revealing, and
  identifiable-but-decision-irrelevant exact-VOI cases;
- parent-disjoint conditional oracle diagnostic for residual event information;
- shared-occurrence posterior-independence test;
- episode-static reset/persistence test.

## 8. Required mechanism measurements

Every v5 study reports:

- immediate and delayed shared/semantic NLL and counts;
- component event Jensen-Shannon separation;
- posterior entropy and response-induced filter KL;
- belief separation by partner run/mechanism and within-episode phase drift;
- current and successor top-action agreement, regret, pairwise sign agreement,
  and critic-ensemble action disagreement;
- exact VOI, information gain, their action-wise spread, and negative numerical
  fraction;
- active/passive policy total variation and greedy disagreement;
- report-only semantic/decision component-embedding gradient norms and cosine;
- semantic initializer singular values, source lineage, and conditional oracle
  gain.

None is a substitute for raw return.

## 9. Non-claims

The project does not claim:

- recovery of true partner identity or a unique protocol taxonomy;
- exact long-horizon Bayes-adaptive planning;
- calibration from a sharp posterior alone;
- causal partner labels from spectral directions;
- guaranteed real-return improvement from approximate decision values;
- SOTA performance before complete ten-seed Simple/Wide evaluation;
- that behavior statistics contain no partner information;
- that v4 or earlier development results are evidence for v5.

## 10. Single-layout result boundary

Each configured layout reports two separate estimands. The
paper-compatible population matrix contains ten final policies crossed as a
directed `(10,10,500)` raw-return cube. The common-partner comparison contains
ten egos, sixteen independent confirmatory partners, both ego roles and 500
episodes per pairing. These objects are never substituted for one another.

A completed run for one layout may report that layout's component of H1, H2
and H3. It cannot
close the both-layout hypotheses, invoke the repository's full claim builder,
or support a SOTA statement without the corresponding Simple result.
