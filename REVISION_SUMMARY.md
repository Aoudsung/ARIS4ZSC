# DELTA-ZSC v6 revision summary

The active method is
`delta_self_consistent_decision_grounded_residual_v6` with config schema 4,
checkpoint schema 6 and deployment bundle 5. `REVISION_V5.md` preserves the
previous method's historical rationale; `REVISION_V6.md` is the detailed
active revision note.

The revision responds to a base-actor collapse: active DELTA measured SP 21.5
and XP 22.7, and removing mirror/VOI left SP 21.22 and XP 22.50. v6 therefore
changes actor ownership and training distribution rather than adding another
deployment heuristic:

- immutable seed-matched Official-SP reference with its complete legal
  observation input preserved;
- zero-initialized residual/value subtree as the only PPO owner;
- fixed 50/50 self-play and frozen cross-play lanes with independent recurrent
  state;
- paired group-normalized minimax PPO;
- XP-only response/successor/value and CRN-anchor estimation;
- full detached response-coordinate grounding in the existing critic;
- final reference-relative KL projection after residual and mirror/VOI.

No new loss, scientific scalar, partner label or component-specific actor was
introduced. The three-field method block and latent-before-PPO estimator order
remain unchanged.

The State-Augmented coverage and confirmatory partner sources now train ten
runs at 128 environments, record that deviation in their identities, and still
contribute only four runs to each final panel. The Official State-Augmented
baseline remains at 256 environments.

CPU source validation is engineering evidence only. No v6 CUDA preflight,
development matrix, formal hypothesis result or SOTA claim currently exists.
Earlier results are retained strictly as failure diagnostics.
