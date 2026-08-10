# DELTA-ZSC v5: Belief-Conditioned Raw-Return Adaptation

This repository contains the unified DELTA-ZSC v5 implementation for the
OvercookedV2 Test-Time Protocol Formation benchmark.

Active identity:

- `METHOD_VERSION = delta_belief_conditioned_raw_return_pairwise_crn_v5`
- `CONFIG_VERSION = 3`
- `CHECKPOINT_SCHEMA_VERSION = 5`
- `MANIFEST_VERSION = 2`
- package version: `0.6.0`
- active namespace: `src/delta_zsc/`
- active CLI: `python -m experiments.overcooked_v2.delta_zsc`

v5 checkpoints, optimizer state, semantic-initializer metadata, and deployment
bundles are intentionally incompatible with v4 and earlier DELTA artifacts.
Official parent checkpoints and version-2 partner manifests remain valid.
Earlier CUDA runs remain failure-diagnostic evidence only and cannot be
resumed or included in v5 scientific summaries.

The retired DEPI v8 implementation is preserved under
`legacy/implementation_v8/`; historical documents are under `docs/legacy/`.
Neither path defines the active method.

## Scientific object

DELTA does not train one actor or critic per partner type. It trains one shared
base actor-critic and one exchangeable latent model. Inside an episode, the
teammate checkpoint is fixed, so the coordination latent is episode-static:

```text
legal local history
    -> shared occurrence prediction
    -> component-semantic response likelihood
    -> episode-static Bayes posterior b_t(z)
    -> belief-conditioned raw-return value Q(x_t,b_t,a)
    -> dense TD(lambda) fit + sparse pairwise-CRN action calibration
    -> KL-constrained mirror policy
    -> optional delayed-response Bayesian VOI
```

The latent components are not partner IDs or algorithm labels. A component is
useful only when legal semantic response evidence selects it differently across
partners and the resulting belief changes action ordering in the shared value
model.

## Active v5 design

v5 retains the episode-static response model introduced after the v3 failure
and replaces the unidentifiable component-wise return mixture with the quantity
the trajectories actually observe: raw return conditioned on legal belief.
A trainable physical-time
transition erased sparse evidence, high-frequency no-change factors could
create a partner-independent component winner, near-symmetric output heads
learned a pooled response instead of component semantics, and immediate
one-step response did not contain a teammate reaction to the current probe.

The active implementation therefore makes six coupled changes.

1. **Episode-static latent.** There is no learned physical-time transition.
   The posterior persists identically within an episode and resets to the
   uniform prior only at a true episode boundary.
2. **Shared occurrence / component semantic factorization.** Visibility,
   inventory-change occurrence, interface availability/change, and recipe
   change are predicted and scored by shared heads with no component axis.
   Only conditional geometry and structured interface event type can change
   the posterior.
3. **Centered response residuals and a belief-conditioned critic.** Semantic
   response heads are `shared baseline + zero-mean component residual`.
   Decision value is a dueling raw-return function of task state, current
   partner evidence and the complete posterior, with an ensemble spread used
   only as a diagnostic.
4. **Unlabeled spectral-simplex initialization.** A fixed label-free
   Rademacher projection makes every frame coordinate available to a
   cross-fitted pooled event model without introducing a privileged label or
   an unbounded calibration matrix. Its predictions are removed from
   episode-level event counts, leading residual
   directions are found by SVD, and a centered regular simplex initializes the
   event residuals. SP/OP labels and partner IDs are never used to construct the
   initializer. A parent-disjoint label oracle is emitted only as a diagnostic.
5. **Direct decision supervision.** Every rollout step supplies a raw-reward
   TD(lambda) target. Sparse all-action continuations calibrate pairwise action
   differences with their measured CRN uncertainty. No objective weight or
   effect threshold is introduced.
6. **Delayed active estimand.** Passive filtering keeps the legal immediate
   response. Active VOI uses a separate two-step response head for
   `o[t+1] -> o[t+2]`, after the teammate has had one opportunity to react to
   probe `a[t]`. The direct effect of the second ego action is removed. The ego
   action at `t+1` is the fixed collection-time base bridge; VOI is valued by
   an all-action matrix at `t+2`, after the delayed response is observable.

## Response contract

The immediate response factorizes as

\[
p(y_t\mid z,H_t,a_{t-1})
=p_{\rm shared}(y_t^{\rm occ}\mid H_t,a_{t-1})
 p_{\rm sem}(y_t^{\rm sem}\mid z,H_t,a_{t-1}).
\]

Shared occurrence factors are:

- visibility;
- visible-at-both-ends inventory-change occurrence;
- aligned-interface availability;
- aligned-interface change/no-change;
- independently covered recipe change/no-change.

Component-semantic factors are:

- relative position, direction, and factorized inventory when visible;
- one of 31 structured interface events when an aligned interface change
  occurs: `3 facilities x 2 appeared/disappeared x 5 objects`, plus
  `OTHER/MULTI`.

The alignment, interaction exclusion, recipe coverage, and 31-class event
encoding retain the exact v3 observation contract. Shared factors are still
trained and audited, but they cannot alter normalized component responsibilities.

## Episode-static filter

For episode start indicator `d_t`,

\[
\bar b_t =
\begin{cases}
\operatorname{Uniform}(K), & d_t=1,\\
b_{t-1}, & d_t=0,
\end{cases}
\]

and

\[
b_t(k)=\frac{\bar b_t(k)p_{\rm sem}(y_t\mid k,H_t)}
{\sum_j\bar b_t(j)p_{\rm sem}(y_t\mid j,H_t)}.
\]

Shared occurrence log probability is absent from this Bayes ratio by
construction.

## Centered component semantics

For any semantic logit vector or action-value vector,

\[
L_k(x)=L_0(x)+\Delta L_k(x),\qquad
\sum_k\Delta L_k(x)=0.
\]

The shared branch learns pooled task regularities. The residual branch learns
only departures from that pooled prediction. A direct embedding-to-output path
prevents the component difference from being multiplied away by a near-zero
shared output map.

Decision value is not decomposed into separately fitted component heads. The
same critic is evaluated at the current posterior for control and at one-hot
component beliefs when exact delayed-response VOI needs component-conditional
utilities.

## Exact delayed active VOI

The delayed probe head models the compact response

\[
Y^{probe}=(V,M,C,E)
\]

under the collection-time base continuation policy. It exactly enumerates 66
outcomes per probe:

- `M=0`: two visibility outcomes;
- `M=1,C=0`: two visibility outcomes;
- `M=1,C=1`: `2 x 31 = 62` visibility/event outcomes.

For each candidate probe `a`, the decision head predicts a post-response matrix
`mu^a[k,a_decision]` at `t+2`. The exact finite value is

\[
\operatorname{VOI}(a)=
\sum_y p(y\mid a)
\max_{a'}\sum_k b^{a,y}(k)\mu^a(k,a')
-
\max_{a'}\sum_k b_t(k)\mu^a(k,a').
\]

The current control value is

\[
Q^{active}_t(a)=Q_\psi(x_t,b_t,a)
+\gamma^2\operatorname{VOI}(a).
\]

The successor CRN target excludes rewards on both the forced probe and the
unforced base-policy bridge. The posterior-dependent action is forced only at
`t+2`, so `gamma^2` has the correct temporal meaning. The finite 66-outcome
sum uses no sampling count, quadrature estimate, VOI clamp, or
information-gain reward.

## Training boundary

`base_params` are trained only by recurrent on-policy PPO. Every rollout and
counterfactual continuation is generated by the collection-time base policy.
`latent_params` are trained first in each outer transaction, before PPO changes
the base tree:

```text
collect rollout D_n and sparse CRN anchors C_n with base omega_n
    -> update latent Theta_n against D_n, C_n and stopgrad(omega_n)
    -> update base omega_n with PPO minibatches from D_n
```

The latent transaction is

\[
L_{latent}=L_{shared\ NLL}+L_{semantic\ NLL}
+L_{raw\ TD}+L_{pairwise\ CRN}+L_{successor},
\]

with fixed coefficient one for each present term. Response scores are normalized
by their own actual observation counts. `response_only` omits decision terms;
`delta_passive` uses dense raw-reward targets and current all-action anchors;
`delta_active` additionally uses delayed response and successor measurements.

## Semantic initializer workflow

A development or formal v5 run that trains semantic latent parameters
(`response_only`, `delta_passive`, or `delta_active`) requires a versioned,
fitted initializer artifact. The
`--deployment` argument is a previously completed *development collection*
policy for the same layout (normally the `base` variant); it is used only to
generate legal, unlabeled response contexts and does not create a circular
dependency on a formal DELTA run. A single calibration pass can build the
registered K=2/4/8 artifacts from the same residual panel:

```bash
python -m experiments.overcooked_v2.delta_zsc build-semantic-initializer \
  --config experiments/overcooked_v2/configs/delta_unified_simple_development.yaml \
  --run-kind development \
  --partner-manifest /path/to/partner_manifest.json \
  --deployment /path/to/base_collection_deployment \
  --partner-role calibration \
  --component-count 2 \
  --component-count 4 \
  --component-count 8 \
  --output /path/to/simple_initializer
```

With multiple component counts the command writes each paired artifact below
`k-2/`, `k-4/`, and `k-8/`, plus the shared summary and resource ledger:

- `semantic_component_initializer.npz`;
- `semantic_component_initializer.json`;
- `semantic_initializer_summary.json`;
- `resource_ledger.json`.

Training then receives `--semantic-initializer /path/to/simple_initializer`.
Only mechanical checks and pure `base`/`history_rnn` collection runs may use
the deterministic orthogonal-simplex fallback. Scientific development and
formal semantic variants reject a missing, mismatched, label-derived, or
training-lineage-overlapping initializer.

## Main CLI

```bash
python -m experiments.overcooked_v2.delta_zsc --help
```

The CLI has 19 subcommands covering partner-manifest construction, Official
upstream assets, semantic initialization, CUDA preflight, training, development
matrices, both evaluation estimands, posterior diagnostics, belief
intervention, baselines, resource accounting, and claim assembly.

## Layout-selected execution

The `test_time_wide` layout reads the Official `5x5x43` local observation, samples the
registered support manifest uniformly over mechanism, family, stage and run,
and uses 256 formal environments. Exact without-replacement anchor selection
uses a bounded-state Floyd sampler, so the registered 256-lane shape no longer
requires a full 65,536-element random sort on L40 hardware.

`run-upstream` reads the layout from its config and constructs the support, calibration,
development-coverage, confirmatory, baseline and FCP-source populations with
preassigned root seeds and lineage records. Development and formal seed `s`
start from support SP parent `s`; all variants at one development seed use the
same parent. Execution is one continuous dependency graph. Diagnostic or return
values do not decide whether later jobs run.
IPPO-Large is not an upstream asset: train it with `train-baseline` after the
corresponding DELTA deployments exist.

Evaluation preserves two different objects for each layout:

- the paper-compatible `(10,10,500)` directed population cube, rooted at 42;
- the common-partner `(10,16,2,500)` matrix, rooted at 0.

Both use final checkpoints, raw team return, actual two-word environment keys
and non-permuted OP observations. Results from one layout alone do not establish
full H1/H2/H3 or SOTA.

## Validation boundary

Local regression and synthetic diagnostics establish implementation contracts,
not benchmark performance. v5 has completed one two-seed, SP-only development
pilot on the `test_time_wide` layout. In the final-code rerun,
`delta_active-response_only` was +1.35 and +4.90 across the two seeds, while
`response_only-base` was -15.95 and -38.95 and `delta_active-base` was -14.60
and -34.05. The pilot did not include `delta_passive`, so it is not registered
H2 or a paper population matrix. The formal ten-seed protocol across both
layouts has not been run, so H1/H2/H3 and SOTA remain unestablished.

See:

- `docs/PROTOCOL_INDEX.md` for authoritative documents;
- `IMPLEMENTATION_MATRIX.md` for requirement-to-code/test traceability;
- `VALIDATION_REPORT.md` for executed local checks;
- `validation/` for reproducible contract scripts.
