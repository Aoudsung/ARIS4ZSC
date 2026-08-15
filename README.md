# DELTA-ZSC v6: Immutable Competence and Decision-Grounded Residual Adaptation

This repository contains the active DELTA-ZSC v6 implementation for the
OvercookedV2 Test-Time Protocol Formation benchmark.

Active identity:

- `METHOD_VERSION = delta_self_consistent_decision_grounded_residual_v6`
- `CONFIG_VERSION = 4`
- `CHECKPOINT_SCHEMA_VERSION = 6`
- deployment bundle version: `5`
- partner/policy manifest version: `2`
- package version: `0.6.0`
- active namespace: `src/delta_zsc/`
- active CLI: `python -m experiments.overcooked_v2.delta_zsc`

v6 checkpoints, optimizer state, resolved configs and deployment bundles are
intentionally incompatible with prior DELTA versions. Official parent
checkpoints and version-2 partner manifests remain valid. The retired DEPI v8
implementation and all historical documents have been removed from the tree;
nothing historical defines the active method.

## Why v6 exists

The observed failure was not in deployment mirror/VOI. `delta_active` achieved
SP 21.5 and XP 22.7, while disabling mirror/VOI still gave SP 21.22 and XP
22.50. The base actor itself had collapsed. In v5, the Official SP actor was
only an initializer, all of its parameters were subsequently rewritten by PPO,
training never included current-policy self-composition, and tiny pooled
minibatches allowed modest XP changes to hide severe SP damage.

v6 changes that training boundary while retaining the episode-static response
posterior, raw-return critic, CRN contrasts, successor model and exact delayed
66-outcome VOI.

```text
full legal local history
    -> immutable seed-matched Official-SP reference actor
    -> zero-initialized trainable residual actor
    -> reference-relative KL projection
    -> optional passive mirror or active delayed-response VOI/mirror
    -> final reference-relative KL projection
```

## Active design

1. **Immutable competence branch.** The Official task encoder, GRU, actor trunk
   and actor head are embedded in `base_params["reference"]`. They receive no
   Adam state and no gradient. `--sp-initializer` is required and must point to
   the matching `run-<seed>/ckpt_final`; engineering seed `-1` uses `run-0`.
   Every variant supplies the complete legal observation to this branch,
   exactly preserving the Official actor's input distribution and recurrent
   state semantics.
2. **Zero residual and shared value branch.** PPO updates only the instantaneous
   partner encoder, residual actor and value function. At update zero the
   residual is exactly zero, so the policy is exactly the embedded Official SP
   actor. Belief innovation is confidence-gated; task and current geometry can
   still learn a partner-agnostic robust residual when the posterior is
   uninformative.
3. **Fixed self/cross-play lanes.** Every training rollout uses the first half
   of lanes for current-policy self-play and the second half for frozen
   cross-play. The self partner has a separate recurrent state and is a
   stop-gradient collection-time snapshot. Generic diagnostic rollouts remain
   frozen-partner-only.
4. **Paired minimax PPO.** Development uses 32 lanes/8 minibatches and formal
   uses 128 lanes/64 minibatches. Every minibatch contains equal SP and XP
   lanes. Advantages normalize per group; actor/value use the worse group and
   entropy uses the smaller group entropy.
5. **XP-only latent estimation.** Response, successor and raw-return latent
   updates read only frozen cross-play lanes. Sparse anchor candidates and CRN
   contrast calibration are also restricted to and remapped within that half.
   Dynamic self snapshots therefore do not violate the episode-static partner
   assumption.
6. **Structural decision grounding.** The existing belief-conditioned critic
   receives the detached posterior-weighted full response-component embedding.
   The existing raw TD and pairwise CRN scores train only the critic. No new
   loss, coefficient, partner classifier or gradient into response semantics is
   introduced.
7. **One KL envelope.** Residual, passive mirror and active VOI/mirror outputs
   are finally projected into the registered KL ball around the immutable
   Official reference. The reported `adaptation_kl` is the actual final
   reference-relative divergence.

The `method` block still contains exactly three scientific scalars:

```yaml
method:
  latent_components: 4
  continuation_horizon: 128
  adaptation_kl_budget: 0.04
```

## Diagnostic execution modes

Evaluation exposes four fixed policy surfaces without retraining:

- `reference_only`: immutable Official actor;
- `residual`: reference plus trained residual, after reference projection;
- `passive`: residual plus current-value mirror, with no active probe state;
- `active`: complete delayed-response VOI/mirror controller.

Raw evaluation rows, run identity and summaries record the selected mode.

## Evidence boundary

CPU compilation and contract tests establish engineering consistency only. A
real Simple 128-lane v6 CUDA preflight has passed on one L40 with 1,436.18 MiB
recorded peak device memory. This remains engineering evidence: the Simple
development matrix is running, and there is no completed development result,
formal H1/H2/H3 result or SOTA evidence yet. The active execution chain reuses
the ten registered Official SP references, builds the layout-specific base
collector and semantic initializers, runs the registered CUDA execution, and
then runs the paired development matrix. These are artifact dependencies, not
diagnostic release gates: no intermediate score blocks or changes a scheduled
downstream run. The proposed development targets—mean SP at least 128.1,
every-seed SP at least 113.8, and mean XP at least 33.2—remain unachieved
targets, not results.

See `docs/PROTOCOL_INDEX.md` for the authoritative specification set and
`REVISION_V6.md` for the change rationale.
