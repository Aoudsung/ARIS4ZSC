# Retired DELTA-ZSC v6 revision (historical)

This document describes a retired method. It is preserved as historical
context only; CETR identity, contracts, and evidence are defined elsewhere.

## Observed failure

The measured `delta_active` result was SP 21.5 / XP 22.7. Disabling mirror and
VOI changed it only to SP 21.22 / XP 22.50. This localizes the collapse to the
trained base actor rather than to deployment adaptation. The prior system used
Official SP only as initialization, let heterogeneous-partner PPO rewrite the
complete actor, omitted current-policy self-play, pooled SP/XP optimization,
and gave response coordinates no structural role in the decision critic.

## Revision

v6 (`delta_self_consistent_decision_grounded_residual_v6`) makes four linked
changes:

- embeds a seed-matched Official-SP actor as an immutable reference, preserves
  its complete legal observation input for every variant, and trains a
  zero-initialized residual/value subtree only;
- trains on fixed half self-play / half frozen cross-play lanes with independent
  recurrent states and paired minimax PPO minibatches;
- limits every latent, successor and CRN-anchor channel to frozen cross-play
  lanes;
- feeds a detached posterior-weighted full response-component embedding into
  the existing belief-conditioned critic, trained by the existing TD/CRN
  scores.

Residual and mirror movement share the registered `adaptation_kl_budget`; a
final projection guarantees the executed policy remains inside that forward-KL
ball around the immutable reference. No new scientific scalar, auxiliary loss,
partner label or decision actor was added.

## Compatibility

- config schema: 4;
- checkpoint schema: 6;
- deployment bundle: 5;
- evaluation schema: 3;
- partner/policy manifests: 2.

Earlier DELTA checkpoints, optimizer/replay state and deployment bundles are
rejected. Official parent checkpoints and manifest-v2 lineage remain usable.

Before any v6 development or formal run, the redundant `history_rnn` and
`history_rnn_extra` rows were removed. Once every immutable reference consumes
the exact Official frame, those rows are behaviorally identical to `base` and
`base_extra`; retaining them would spend budget on duplicate controls.

The 2026-08-12 upstream amendment also makes State-Augmented panel construction
executable on Wide: each development-coverage and confirmatory source trains
the minimum valid ten-run population, with 128 environments recorded in its
identity and artifact, and contributes only runs `0..3` to the final panel.
This doubles optimizer updates at fixed timesteps relative to Official
State-Augmented training. It does not change the registered baseline, which
continues to use 256 environments.

The 2026-08-14 execution audit closed the bootstrap dependency for Simple by
adding its missing `base` initializer-collector config. The registered chain
reuses the ten full-budget Official SP final checkpoints already present in the
development-support panel, uses parent 0 to train a v6 collector, fits K=2/4/8
semantic initializers from calibration data, and loads K=4 in the 128-lane CUDA
preflight. Reference eligibility depends on the Official SP recipe and final
training budget, not on the population's root seed or invocation size.

## Evidence status

The implementation has complete CPU-side contract evidence, and the real
Simple 128-lane CUDA preflight passed on one L40 with 1,436.18 MiB recorded
peak device memory. Earlier v5 pilots remain failure diagnostics and cannot be
reported as v6 performance. The Simple development matrix is running; no
completed v6 development, formal or SOTA result currently exists.
