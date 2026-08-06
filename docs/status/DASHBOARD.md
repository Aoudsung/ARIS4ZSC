# Unified DELTA-ZSC status dashboard

Updated: 2026-08-06. This page reports implementation status only; it does not
turn engineering validation into scientific evidence.

| Track | Status | Meaning |
|---|---|---|
| Unified method | implemented | one response-decision latent model, response-only filter, separate base/latent optimizers, analytic KL adaptation, and deterministic Bayesian VOI |
| VOI | implemented and exercised on CUDA | exact 66-outcome compact marginal completed three development runs and final audits; values remain near zero in the current pilot |
| Active code boundary | implemented | only `src/delta_zsc/` and the unified CLI remain in the active tree; the full DEPI v8 source/apps/configs/workflow/tests and historical analysis are archived under `legacy/implementation_v8/`; old decision/evidence ledgers are under `docs/legacy/v8/` |
| Local CPU regression | pending for v3 | the prior v2 result is superseded; no Python or tests were run while applying this revision |
| End-to-end path | passed for the three-seed pilot | revised response extraction, exact VOI, training, checkpoint/deployment export and Official-environment evaluation all completed on one A10 |
| Registered Python 3.10 install | executed remotely | Python 3.10.19, JAX 0.4.38 and one CUDA device were verified on `ali_ZSC` |
| Real Official/CUDA execution | three-seed development pilot completed | real SP/OP pilot checkpoints and Official reset/step were used; no numerical update failed |
| Development evidence | pilot generated; registered matrix absent | Simple development-support mean raw return is 6.6778 over 3,600 episodes; a non-independent one-seed SP comparator is 5.6167; posterior separation and decision quality remain poor |
| Formal evidence | not generated | no H1/H2/H3 or SOTA claim is made by the repository alone |

The empirical priority is to explain and correct the near-uniform posterior,
weak decision agreement, OP-partner failure and role asymmetry before spending
compute on the paired development matrix. The pilot artifacts are under
`/mnt/workspace/ARIS4ZSC_v5/runs/engineering/SP_OP_results/delta_interface_v3_3seed_20260806`.
