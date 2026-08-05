# Unified DELTA-ZSC status dashboard

Updated: 2026-08-05. This page reports implementation status only; it does not
turn engineering validation into scientific evidence.

| Track | Status | Meaning |
|---|---|---|
| Unified method | implemented | one response-decision latent model, response-only filter, separate base/latent optimizers, analytic KL adaptation, and deterministic Bayesian VOI |
| VOI | implemented and core-tested | exact visibility/event marginalization, Halton integration of remaining response factors, all-component likelihood, exact Bayes update, decision value, information-gain diagnostic, and nested-prefix error |
| Active code boundary | implemented | only `src/delta_zsc/` and the unified CLI remain in the active tree; the full DEPI v8 source/apps/configs/workflow/tests and historical analysis are archived under `legacy/implementation_v8/`; old decision/evidence ledgers are under `docs/legacy/v8/` |
| Local CPU regression | passed | all active files compile; 26 tests across six isolated test files pass in the available local JAX runtime |
| Mocked end-to-end path | passed | rollout, base PPO update, latent update, sparse anchors, checkpoint/deployment round trip and KL/VOI outputs are finite |
| Registered Python 3.10 install | specified, not executed in this local runtime | `pyproject.toml` and CI pin the formal environment |
| Real Official/CUDA preflight | workflow ready, not executed here | requires the pinned external benchmark packages, real checkpoints, lineage-bound manifest and self-hosted GPU |
| Development evidence | not generated | no paired five-seed Simple/Wide matrix is included in this source package |
| Formal evidence | not generated | no H1/H2/H3 or SOTA claim is made by the repository alone |

The next executable action is the real CUDA mechanical preflight followed by
the paired development matrix. These are direct end-to-end runs, not additional
method gates.
