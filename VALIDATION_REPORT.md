# DELTA-ZSC v6 validation report

Date: 2026-08-14

## Scope

This report covers source-level engineering validation for
`delta_self_consistent_decision_grounded_residual_v6`. It does not establish
SP, XP, H1/H2/H3 or benchmark superiority.

## Active artifact identity

- config schema: 4;
- checkpoint schema: 6;
- deployment bundle: 5;
- evaluation schema: 3;
- partner/policy manifest: 2;
- semantic initializer: 1;
- Official source revision: registered in `src/delta_zsc/config.py`.

## Locally established

The CPU suite covers compilation, configuration and repository boundaries;
immutable-reference/zero-residual ownership;
full-frame sequence equivalence to the actual Official Flax actor, including
episode resets and action probabilities; paired SP/XP minibatches;
group-normalized minimax PPO; independent self-partner recurrence; frozen-only
generic rollouts; XP-only anchor index handling; detached full-embedding critic
grounding; checkpoint/deployment rejection; exact 66-outcome VOI; and the
existing legal-information, seed, lineage and estimator-order contracts.

The final exact counts and command outcomes are regenerated during the final
validation run and recorded in `ARTIFACT_IDENTITY.json` and the validation
artifacts.

## Server-established engineering execution

The Simple formal-shape CUDA preflight completed on one L40 with 128 lanes. It
executed 32,768 ordinary PPO steps, 1,477,632 anchor continuation steps and a
fresh 1,510,400-step final decision audit. The recorded peak device memory was
1,436.18 MiB, below the registered 40,000 MiB ceiling. This establishes CUDA
execution and persistence only; it is not performance evidence.

## Not established

- a completed v6 paired Simple or Wide development matrix;
- any achieved SP or XP target;
- ten-seed formal deployments or evaluations;
- H1, H2, H3 or a SOTA claim.

Prior v5 pilots remain useful only for diagnosing the base-actor collapse and
cannot be included as v6 evidence.
