# RESEARCH_PLAN — DELTA-ZSC v6 registered research plan

`authoritative: true`

This document consolidates the former `RESEARCH_PROGRAM`,
`FORMAL_EXPERIMENT_PROTOCOL`, `EVALUATION_SPEC`, `research/DEVELOPMENT_MATRIX`,
`research/STATISTICAL_PREREGISTRATION` and `PAPER_OUTLINE` documents. Nothing
was re-registered in the merge; duplicated passages were kept once and are
cross-referenced.

## 1. Registered formal contract

### 1.1 Environment

- layouts: `test_time_simple`, `test_time_wide`;
- episode length: 400;
- action count: 6;
- view radius: 2;
- negative rewards: enabled;
- random initial positions: enabled;
- recipe resampling after delivery: enabled;
- successful-delivery indicator: enabled;
- local observation shape: `5x5x39` on Simple and `5x5x43` on Wide;
- Official source identity: registered in `src/delta_zsc/config.py`.

### 1.2 Confirmatory method

```yaml
version: 4
method_variant: delta_active
method:
  latent_components: 4
  continuation_horizon: 128
  adaptation_kl_budget: 0.04
```

The latent is episode-static. Formal training requires a layout-specific,
lineage-bound spectral-simplex initializer artifact. The initializer uses no
partner labels and is part of run identity.

Every formal run additionally requires `--sp-initializer` at the matching
`run-<seed>/ckpt_final`. The embedded Official actor is immutable, the residual
head starts at exactly zero, and no previous DELTA checkpoint is compatible.

### 1.3 Base PPO

- ordinary environment steps: 29,949,952;
- environments: 128;
- rollout length: 256;
- update epochs: 4;
- minibatches: 64;
- learning rate: 0.00025;
- warm-up fraction: 0.05;
- cosine annealing: enabled;
- gradient norm: 0.25;
- gamma: 0.99;
- GAE lambda: 0.95;
- policy/value clip: 0.2;
- entropy coefficient: 0.01;
- value coefficient: 0.5;
- Adam epsilon: 1e-5.

The 128 lanes are fixed as 64 current-policy self-play plus 64 frozen
cross-play lanes. Every one of the 64 minibatches contains one lane from each
group. Advantages are normalized per group; actor/value use the worse group and
entropy uses the lower group entropy. Only the residual/value subtree is
optimized.

### 1.4 Sparse current and successor decision observations

- anchor interval: 1,048,576 ordinary environment steps;
- states per trigger: 16;
- fit replicas: 8;
- evaluation replicas: 8;
- continuation horizon: 128;
- current forced actions: 6;
- active probe actions: 6;
- forced post-response actions per probe: 6;
- one unforced collection-time-base bridge per probe replica;
- probe- and bridge-transition rewards excluded from the successor target;
- matched CRN across action alternatives;
- base policy for unforced continuation actions;
- fit/evaluation split fixed before scoring;
- no replay across outer updates;
- full five-dimensional measurement covariance;
- nested `lax.map` bounded-memory execution.

All anchor candidates, successor observations and contrast indexes are drawn
from the frozen cross-play half. Every response, successor and raw-value latent
update likewise slices to cross-play before replay.

Eight fit replicas satisfy the minimum six needed for a potentially full-rank
five-dimensional sample covariance.

### 1.5 Delayed response and exact active value

The active target is the legal two-transition window
`o[t+1] -> o[t+2]`, conditioned on probe `a[t]`; `a[t+1]` is used only to remove
its direct ego effect. Either terminal transition invalidates the window.

VOI exactly enumerates the 66 delayed compact outcomes and uses the
probe-conditioned `t+2` decision matrix. Its incremental value is discounted
by `gamma^2`. There is no sample-count field, quadrature estimate, non-negative
clamp, or information-gain reward.

### 1.6 Partner distributions

Training support contains independent SP and OP parents with checkpoints at
0.0, 0.5, and 1.0. Sampling is uniform over mechanism, family, stage, and run.
Formal support requires at least ten independent parents per mechanism.

Formal DELTA seed `s` uses development-support SP parent `s` as its base-policy
initializer. Development variants use the same mapping for seeds `0..4`; all
variants sharing a seed receive the same initializer. The engineering
collector and CUDA execution use parent 0.

The immutable references are the ten registered final Official SP checkpoints
in the development-support panel, matched to DELTA by seed index. Their root
seed and the number of checkpoints produced by the same Official invocation are
generation details, not properties of the trained actor. Each reference must
still match the registered SP architecture, hyperparameters, layout and full
training budget. For each layout, the dependency chain reuses these references,
then trains the `base` initializer collector (Simple and Wide each have a
checked-in `*_initializer_collector.yaml` config), fits the K-specific semantic
initializer from the calibration panel, and runs the formal-shape CUDA
preflight. A mechanical checkpoint cannot substitute for a full-budget
reference.

Initializer calibration, development coverage, and confirmatory panels are
parent- and co-training-lineage disjoint from support and from each other.
Partner manifests are owner-free. Heuristics remain test-only.

The State-Augmented development-coverage and confirmatory partner sources each
train ten runs because the Official ordered-population state pass requires
`run_count**2` to be divisible by its ten fixed minibatches. For these two
partner-source populations only, the pipeline uses 128 rather than the Official
256 parallel environments to fit the Wide job on one L40. Total timesteps and
the learning-rate schedule definition remain fixed, so this doubles optimizer
updates and is a registered Official-protocol deviation. The population
identity and training artifact record the effective environment count, while
only runs `0..3` enter each final four-partner panel. The registered
State-Augmented baseline remains on the unmodified Official 256-environment
path.

### 1.7 Runs and evaluation

- DELTA training seed indexes: 0..9;
- engineering seed: -1, excluded from science;
- episodes per ego/partner/role pairing: 500;
- both ego roles: required;
- bootstrap replicates: 9,999;
- one-sided alpha: 0.05;
- minimum ego runs: 10;
- minimum independent partner runs per mechanism: 4;
- material effect: 20 raw-return points.

Simple and Wide are inferred separately and combined only by intersection.
The two formal estimands (paper-compatible population and common-partner) are
defined in §2.3.

### 1.8 Required artifacts

Every formal run preserves:

- resolved config and method/source identity;
- semantic initializer NPZ/JSON and source metadata;
- lineage-bound partner manifest;
- complete checkpoint descriptor;
- per-update shared/semantic/decision metrics and resource ledger;
- final deployment bundle;
- raw evaluation mode (`reference_only`, `residual`, `passive`, or `active`) in
  every row, identity and summary;
- final current/successor decision and active-policy audit;
- raw evaluation rows;
- posterior diagnostic artifact;
- belief-intervention artifact;
- final three-claim report.

### 1.9 Execution acceptance

A CUDA preflight is engineering evidence only. It must exercise:

- one visible CUDA GPU, the formal 128-environment shape and peak memory below
  40,000 MiB;
- real Official reset/step and fixed partner checkpoints;
- semantic initializer loading;
- one legal delayed response window;
- one current and successor anchor trigger;
- separate finite latent and PPO updates;
- checkpoint save/restore and deployment export;
- finite exact VOI, action-wise VOI spread, mirror KL, and posterior metrics.

Anchor-index sampling uses exact without-replacement Floyd sampling, so the
formal shape does not materialize a full random sort of all 32,768 rollout
positions. The engineering run verifies that shape on the actual server GPU.

The upstream assets, initializer, engineering execution, development matrix,
formal runs and evaluations execute as one dependency graph. Diagnostic or
return values are reported but never release, block or alter a later job.
Formal claims require complete ten-run matrices and lineage records on both
layouts.

## 2. Evaluation and inference

### 2.1 Independent units

Ego training run and partner parent run are independent generalization nodes.
Episodes are repeated measurements within a pairing and are never bootstrapped
as independent ZSC samples. Mechanism and checkpoint stage remain registered
strata.

### 2.2 Pre-training semantic initialization

Each layout has a lineage-bound calibration panel disjoint from training
support and confirmatory partners. The spectral-simplex initializer is built
without partner labels. The artifact records event/episode counts, singular
values, source run IDs, and a parent-disjoint conditional oracle diagnostic.
Formal training must bind the initializer artifact in run identity.

The oracle diagnostic may use SP/OP labels only to measure whether mechanism
adds held-out event prediction after legal state/history conditioning. It never
changes the initializer, model, optimizer, or policy.

### 2.3 Two formal evaluation estimands

The paper-compatible population estimator evaluates SP, State-Augmented, OP,
FCP and DELTA-active separately. For each method, the same ordered ten-run
population is used on the left and right of a `(10,10,500)` raw-return cube.
The ten diagonal cells define SP and the ninety directed off-diagonal cells
define XP. Root seed 42 is split into Official SP/cross branches before fixed
cell enumeration. Its output extends Table 2; the public paper does not expose
the aggregation that produced its printed error terms, so locally named
dispersion measures are reported separately.

The common-partner estimator evaluates final-checkpoint `delta_active` against
Official SP, OP, State-Augmented, FCP, and a capacity-matched IPPO-Large. Every
ten-run ego population is crossed with the same sixteen confirmatory partner
runs in both roles for 500 episodes, producing `(10,16,2,500)` rows per method.
Root seed 0 and the exact key schedule are shared across all six methods.

Raw episode rows are reduced to an ego-run x partner-run matrix. The bootstrap
independently resamples ego and partner nodes. H1 uses an intersection-union
rule: on each layout, every baseline contrast needs a one-sided 95% lower bound
above zero and a point estimate of at least 20.

Both evaluators use the same Official environment and policy adapters and
explicitly disable OP ingredient permutation. The two estimands are never
combined or substituted for one another. A result from either layout alone
cannot close the both-layout hypotheses or invoke the full claim builder.

The paper-compatible estimator also supports `grounded_coord_ring` as a
single-layout Table 2 extension. Its published reference row is selected by
layout; it does not expand the registered Simple/Wide formal claim.

### 2.4 H2 and H3 operational definitions

H2 uses paired development seeds for `delta_passive - response_only`; both
layouts require a 95% interval whose lower endpoint is above zero.

H3 is a same-world belief intervention. Environment state, partner, base logits,
current decision matrix, CRN outcomes, and model parameters are fixed. Only the
legal belief is replaced by a task-matched belief from another partner run. A
crossed ego/partner bootstrap evaluates empirical continuation value.

The statistical decision rules for H1/H2/H3 are registered in §4.

### 2.5 Active mechanism evidence

Report for every layout and partner stratum:

- exact delayed-response VOI mean, max, min, and negative fraction;
- expected information gain;
- per-state `max_a-min_a` VOI and information-gain spread;
- active/passive probability total variation;
- active/passive and base/active greedy disagreement;
- current and successor decision agreement/regret;
- empirical continuation value of active versus passive choices when measured.

A positive constant VOI shared by all actions is not active probing evidence.
Information gain cannot substitute for task VOI.

### 2.6 Posterior and semantic diagnostics

The calibration/posterior application reports:

- factor-level shared and semantic NLL/counts;
- interface/recipe coverage and positive rates;
- conditional event NLL and `OTHER/MULTI` rate;
- component event JS;
- response-induced filter KL versus the episode-static prior;
- belief entropy, partner separation, and phase drift;
- pooled SP/OP event distribution and total variation;
- initializer singular values and conditional oracle gain.

No diagnostic result controls execution of benchmark evaluation or any later
job. These measurements localize model error and prevent a
partner-independent global winner from being misreported as inference.

### 2.7 Resource accounting

Every method reports:

- ordinary ego-policy simulator steps;
- current-anchor continuation steps;
- probe, base-bridge, and `A x A` post-response continuation steps;
- calibration/initializer steps;
- upstream partner-training steps;
- shared and marginal GPU/wall time;
- peak memory;
- deployable and training-only parameter counts;
- batch-one inference latency.

The full successor matrix is computed with bounded-memory `lax.map`, but all
simulator transitions are still charged.

### 2.8 Prohibited analysis

- selecting partners or stages after viewing DELTA return;
- treating episodes as independent bootstrap nodes;
- using mechanism labels to train the semantic initializer or posterior;
- reporting posterior sharpness without partner separation;
- replacing raw return with response NLL or VOI;
- promoting a null active increment into an unregistered claim;
- combining layouts so one masks failure on the other;
- reusing v5 or earlier DELTA checkpoints or results as v6 runs.

### 2.9 Frozen policy-surface diagnostics

Every v6 deployment evaluation identifies one of four modes:
`reference_only`, `residual`, `passive`, or `active`. The selected mode is
written to each raw episode row, the run identity and every summary. Passive
mode advances no active-probe pending state. These modes decompose an already
trained policy; they are not additional selected training variants.

## 3. Development matrix

### 3.1 Main K=4 block

Five paired seeds (`0..4`) are used for every row and layout. Each DELTA row is
bound to the same layout-specific residual-panel initializer root; the runner
selects the K-specific `k-2`, `k-4`, or `k-8` artifact for that row. Thus K
sensitivity changes only the simplex cardinality, not the calibration data.
The matrix command therefore requires `--semantic-initializer`; latent-bearing
development cells do not silently fall back to an unfitted simplex.

| Variant | Episode-static response latent | Current decision | Delayed response + successor decision | KL adaptation | Anchor cost |
|---|---:|---:|---:|---:|---:|
| base | no | no | no | no | 0 |
| response_only | yes | no | no | no | 0 |
| delta_passive | yes | yes | no | yes | current anchors |
| delta_active | yes | yes | yes | yes | current + successor anchors |
| base_extra | no | no | no | no | reallocated to PPO |

The core rows have equal ordinary PPO interaction. Extra controls add the exact
charged pilot, current and successor continuation transition count to PPO and
receive no counterfactual labels.

The main block is `5 variants x 5 seeds = 25` runs. K sensitivity adds
`2 variants x 2 additional K values x 5 seeds = 20` runs, for exactly 45 runs
per layout.

### 3.2 K sensitivity

Only passive and active DELTA are run for `K=2` and `K=8`; the main `K=4` runs
are reused. Initializer construction uses the same residual data but projects a
simplex with the registered K. This is a bounded sensitivity analysis, not an
open sweep.

Development seed `s` initializes every variant and K value from the same
development-support SP parent `s`. Partner sampling remains uniform over the
registered mechanism, family, stage and run probabilities throughout each
training run.

Every development rollout has 16 self-play and 16 frozen cross-play lanes.
Eight paired minibatches each contain two lanes from each group. Latent and CRN
channels read only cross-play lanes. The immutable reference is shared across
paired cells for seed `s`; only the residual/value and applicable latent
subtrees train.

The four frozen evaluation modes (`reference_only`, `residual`, `passive`,
`active`) are reported for `delta_active`. They diagnose which layer changes
return and do not create extra trained variants.

### 3.3 Reproducibility and pairing

Paired cells must record:

- base initialization identity;
- environment, partner, anchor-index, and minibatch root seeds;
- semantic initializer identity;
- observed action/reward/partner-member streams where pairing requires them;
- the actual two-word environment keys for every evaluation episode;
- resolved config and source method identity.

PPO-side development metrics are descriptive unless replicated across paired
runs. Posterior/mechanism claims must be supported by independent partner-run
aggregation rather than a single final anchor batch.

### 3.4 Pre-specified contrasts

1. `delta_passive - response_only` — decision-emission contribution.
2. `delta_active - delta_passive` — delayed active-response contribution.
3. `delta_active - base`.
4. `delta_active - base_extra`.

### 3.5 Required paired diagnostics

- base action/reward and partner-stream identity where variants should share
  collection behavior;
- initializer identity and singular values;
- component event JS and partner-separation L1;
- response-induced filter KL and phase drift;
- current/successor agreement, regret, and component disagreement;
- VOI and information-gain action spread;
- active/passive policy TV;
- response-component gradient norm and verified zero decision gradient into
  the detached grounding coordinate.

High posterior sharpness with low partner separation is a failed global winner,
not successful inference. Positive information gain with zero VOI spread is not
active probing. Low decision loss without held-out return gain localizes failure
to policy conversion or distribution shift.

These diagnostics are interpreted after the registered matrix is complete.
None changes the variants, losses, budgets, partner distribution or scheduling
of downstream jobs.

The pre-specified v6 development targets are mean SP at least 128.1,
every-seed SP at least 113.8, and mean XP at least 33.2. They are execution
targets, not evidence and not formal hypothesis replacements.

## 4. Statistical preregistration

### 4.1 Registered before confirmatory evaluation

- v6 method/source identity and schemas;
- required seed-matched Official-SP reference and zero residual initialization;
- fixed half self-play / half frozen cross-play training lanes and paired
  minimax PPO;
- XP-only latent and CRN-anchor estimation;
- Simple/Wide configs and semantic initializer artifacts;
- training, initializer-calibration, posterior-calibration, and confirmatory
  partner manifests;
- seed indexes 0..9;
- all registered baseline labels;
- 500 episodes per pairing and both roles;
- raw return as primary endpoint;
- crossed ego-run/partner-run bootstrap with 9,999 replicates;
- one-sided alpha 0.05;
- material effect 20 points;
- closed hierarchy `H1 -> H2 -> H3`;
- active-minus-passive as secondary;
- exact delayed 66-outcome active marginal and probe-successor estimand.

### 4.2 H1

For every layout and baseline `m`, estimate

\[
\Delta_m=J(\text{delta-active})-J(m).
\]

H1 passes only when every contrast has one-sided 95% lower bound above zero and
point estimate at least 20. This is an intersection-union rule; no baseline is
removed after results.

### 4.3 H2

Using paired development seed means, estimate

\[
J(\text{delta-passive})-J(\text{response-only}).
\]

Both layouts require a 95% interval lower endpoint above zero. The closed claim
hierarchy interprets H2 after H1, but H2 execution and reporting do not wait on
or change in response to H1.

### 4.4 H3

At same-world anchors, compare source continuation value of correct-belief and
task-matched shuffled-belief mirror policies. Bootstrap ego and partner nodes.
Both layouts require a one-sided lower bound above zero. The closed claim
hierarchy interprets H3 after H1 and H2, but all intervention jobs execute on
the preassigned schedule regardless of those results.

### 4.5 Secondary active result

Always report `delta_active - delta_passive`, exact VOI, information gain,
action-wise spreads, active/passive policy TV, and successor decision quality.
No minimum active effect is imposed after results, and a null result is not
redefined as success.

### 4.6 Diagnostic outputs

Always report initializer singular values and conditional oracle gain,
shared/semantic/decision scores and counts, component JS, partner separation,
filter KL, posterior entropy, phase drift, current/successor regret, gradient
alignment, exact-VOI numerical diagnostics, K sensitivity, negative transfer,
and full resources. None may replace the primary endpoint.

### 4.7 Missingness and failures

A missing run is not replaced by another seed. Infrastructure continuation
keeps the same recorded identity. Exclusion requires pre-existing mechanical invalidity such as an
unreadable checkpoint, lineage mismatch, non-finite state, malformed initializer,
or incomplete raw artifact. Poor performance, uniform belief, or zero active
increment are never exclusion criteria.

## 5. Execution program

The objective is benchmark superiority supported by a falsifiable mechanism
chain. Engineering acceptance certifies experiment validity; it is not a
reason to postpone end-to-end runs.

### 5.1 Source and semantic-initializer execution

For the recorded v6 source revision:

1. run compilation, configuration, CLI, repository-boundary, core, VOI,
   semantic, training, runner/storage, and manifest tests;
2. generate the v6 synthetic exact-VOI artifact;
3. train the layout's `base` initializer collector from Official SP parent 0,
   then build the K-specific initializer artifacts from the registered
   calibration panel;
4. verify initializer centering, singular values, unlabeled construction, and
   parent-disjoint conditional oracle artifact;
5. execute one real CUDA update at the formal vectorized shape with the K=4
   initializer, current and successor anchors, save/restore, deployment export,
   and final audit.

The preflight must also demonstrate a seed-matched immutable reference,
zero-residual identity, 50/50 mixed training lanes, paired SP/XP minibatches,
XP-only latent/anchor data and a final reference-relative KL bound.

### 5.2 Paired development matrix

Run the registered five-seed Simple/Wide matrix (§3) without adding new losses
or post-hoc variants. Read results in this causal order:

1. full-frame base task competence and reproducibility;
2. semantic response specialization and partner separation;
3. response-only versus passive DELTA;
4. current and successor decision quality;
5. passive versus active DELTA;
6. total-interaction controls and bounded K sensitivity;
7. resource and numerical diagnostics.

Interpret the mechanism with all of the following evidence together:

- component event distributions are non-identical;
- beliefs differ by partner more than by episode phase alone;
- legal-belief changes alter the shared critic's action ordering;
- same-world belief intervention has positive empirical value;
- active VOI has action-wise spread and changes the passive policy when its
  secondary contribution is claimed.

These are interpretation criteria. Their values do not change losses,
variants, budgets or downstream execution.

### 5.3 Formal execution

Record source, configs, initializer artifacts, manifests and seeds in every run
identity. Train ten `delta_active` runs per layout and all registered baselines
under the Official protocol. Produce the paper population cubes, full ego x
partner x role matrices, resource ledgers, posterior diagnostics, final
current/successor audits and belief interventions.

Execution follows data dependencies continuously. Diagnostics are reported in
parallel with performance evidence and never act as stage-release decisions.

### 5.4 Paper decision

The strongest paper closes:

- H1: material superiority to every baseline on both layouts;
- H2: decision supervision adds value beyond response inference;
- H3: the legal belief has same-world causal decision value.

The active increment is secondary. A null active increment is reported without
inventing a replacement mechanism. A negative passive result sends diagnosis
back to the corresponding response, decision, or mirror-policy link.

### 5.5 Revision discipline

A future modification enters the active method only when it repairs one of:

1. legal semantic response -> partner-relevant posterior;
2. posterior -> component-conditioned action ordering;
3. action ordering -> higher held-out return;
4. probe -> action-selective delayed information value.

It must retain one shared actor-critic, avoid partner-label supervision, and add
no arbitrary anti-collapse gate or loss without a separately identified
estimand.

## 6. Paper plan

### 6.1 Working title

**Decision-Relevant Episode-Static Adaptation for Zero-Shot Coordination**

### 6.2 Central claim

Teammate prediction becomes useful for zero-shot coordination only when pooled
response regularities are separated from partner-semantic residuals, legal
history selects those residuals, and the resulting legal belief changes a
raw-return action ordering calibrated by real counterfactual continuations.

### 6.3 Contributions

1. **Episode-static response latent.** The persistence assumption is
   matched to a fixed teammate per episode, with no physical-time transition
   that erases sparse evidence.
2. **Shared occurrence and centered semantic residuals.** High-frequency
   no-change factors are predicted by shared heads, while conditional geometry,
   and event convention use zero-mean component residuals with direct embedding
   paths.
3. **Unlabeled spectral-simplex initialization.** Cross-fitted episode residuals
   initialize exchangeable event semantics without partner IDs or SP/OP labels.
4. **Belief-conditioned raw-return adaptation.** Dense raw-reward TD(lambda)
   and sparse pairwise CRN differences calibrate one shared dueling critic;
   mirror improvement consumes its legal-belief action ordering.
5. **Delayed decision-relevant active value.** A separate two-step response head
   models the first window in which a teammate can react to a probe; exact
   66-outcome Bayes updates are valued by a probe-conditioned successor CRN
   matrix.
6. **Causal and statistically valid evaluation.** Crossed run-level inference,
   total-interaction controls, same-world belief intervention, semantic
   diagnostics, and fully loaded resource accounting distinguish mechanism from
   mere posterior sharpness.

### 6.4 Narrative

1. v3 extracted a real convention signal but an unconstrained mixture learned a
   pooled predictor; a sticky posterior amplified a partner-independent winner.
2. The failure identifies two separations that the model must encode:
   pooled occurrence versus component semantics, and response inference versus
   directly observed belief-conditioned raw return.
3. v6 imposes these separations by parameterization rather than partner labels,
   entropy gates, or multiple actors/critics.
4. Passive DELTA tests whether legal semantic history improves decisions.
5. Active DELTA tests whether a causally delayed response is action-selective
   and worth the probe opportunity.
6. Simple/Wide matrices, current/successor audits, and belief interventions test
   the full chain.

### 6.5 Main figures

1. v3/v4/v5 failure diagnosis and v6 episode-static residual system.
2. Shared occurrence head versus centered semantic residual head.
3. Cross-fitted spectral residuals and simplex initialization.
4. Immediate posterior -> belief-conditioned raw-return critic -> KL mirror policy.
5. Probe `a_t` -> delayed response `o_{t+1}->o_{t+2}` -> exact 66 outcomes ->
   successor matrix -> active value.
6. Simple/Wide cross-play matrices and same-world belief intervention.

### 6.6 Main tables

1. Confirmatory all-baseline raw-return comparison with crossed-node intervals.
2. Nested development variants and total-interaction controls.
3. Semantic/predictive mechanism audit: event JS, partner separation, filter KL,
   current/successor decision regret, VOI spread, policy TV.
4. H1/H2/H3 closed-hierarchy decision.
5. Marginal, shared, amortized, and fully loaded resources.

### 6.7 Limitation statement

DELTA v6 uses a finite exchangeable episode-static latent and values one delayed
response window under the registered base continuation policy. It is not a
partner-identity model or full Bayes-adaptive planner. Spectral directions are
initialization coordinates, not discovered ground-truth protocols. Exactness
applies to the finite 66-outcome learned marginal, not to the environment's
complete future trajectory distribution.
