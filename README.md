# DELTA-ZSC v4: Episode-Static Decision-Relevant Adaptation

This repository contains the unified DELTA-ZSC v4 implementation for the
OvercookedV2 Test-Time Protocol Formation benchmark.

Active identity:

- `METHOD_VERSION = delta_belief_conditioned_raw_return_pairwise_crn_v5`
- `CONFIG_VERSION = 3`
- `CHECKPOINT_SCHEMA_VERSION = 4`
- `MANIFEST_VERSION = 2`
- package version: `0.6.0`
- active namespace: `src/delta_zsc/`
- active CLI: `python -m experiments.overcooked_v2.delta_zsc`

v4 checkpoints, optimizer state, semantic-initializer metadata, and deployment
bundles are intentionally incompatible with v3 DELTA artifacts. Official SP/OP
parent checkpoints and version-2 partner manifests remain valid. The v3 CUDA
runs remain failure-diagnostic evidence only and cannot be resumed or included
in v4 scientific summaries.

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
    -> shared decision baseline + centered component decision residual
    -> KL-constrained mirror policy
    -> optional delayed-response Bayesian VOI
```

The latent components are not partner IDs or algorithm labels. A component is
useful only when it jointly predicts a legal semantic response difference and a
different action-value residual.

## What changed in v4

v4 repairs the failure mode measured in v3: a trainable physical-time
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
3. **Centered residual emissions.** Every semantic response and decision head
   is `shared baseline + zero-mean component residual`. Residuals receive both
   a context path and a direct component-embedding skip with standard fan-in
   initialization. Decision variance is shared across components.
4. **Unlabeled spectral-simplex initialization.** A fixed label-free
   Rademacher projection makes every frame coordinate available to a
   cross-fitted pooled event model without introducing a privileged label or
   an unbounded calibration matrix. Its predictions are removed from
   episode-level event counts, leading residual
   directions are found by SVD, and a centered regular simplex initializes the
   event residuals. SP/OP labels and partner IDs are never used to construct the
   initializer. A parent-disjoint label oracle is emitted only as a diagnostic.
5. **Channel-normalized proper score.** Shared response, semantic response, and
   decision observations each contribute their own mean negative log
   likelihood with fixed coefficient one. Rollout length and anchor frequency
   therefore cannot silently become loss weights.
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

The same decomposition is used for current and probe-successor decision
emissions:

\[
\mu_k(x,a)=\mu_0(x,a)+\Delta\mu_k(x,a),\qquad
\sum_k\Delta\mu_k(x,a)=0.
\]

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
Q^{active}_t(a)=\sum_k b_t(k)\mu^{current}_k(a)
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

The latent objective is

\[
L_{latent}=L_{shared}+L_{semantic}+L_{decision},
\]

where each term is independently normalized by its own actual observation
count. `response_only` omits decision observations; `delta_passive` uses current
all-action anchors; `delta_active` additionally uses delayed response and
probe-successor anchors.

## Semantic initializer workflow

A development or formal v4 run that trains semantic latent parameters
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

The CLI covers partner-manifest construction, semantic initialization, CUDA
preflight, training, development matrices, evaluation, posterior diagnostics,
belief intervention, baselines, resource accounting, and formal-claim assembly.

## Validation boundary

Local regression and synthetic diagnostics establish implementation contracts,
not benchmark performance. v4 has not been trained on the formal ten-seed
Simple/Wide protocol in this source package. The old v3 pilot remains evidence
for why v4 was required, not evidence that v4 improves return.

See:

- `docs/PROTOCOL_INDEX.md` for authoritative documents;
- `IMPLEMENTATION_MATRIX.md` for requirement-to-code/test traceability;
- `VALIDATION_REPORT.md` for executed local checks;
- `validation/` for reproducible contract scripts.
