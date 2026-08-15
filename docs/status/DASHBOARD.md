# DELTA-ZSC v6 status dashboard

Date: 2026-08-14

| Item | Status |
|---|---|
| Method/config/checkpoint/deployment identity | implemented: v6 / 4 / 6 / 5 |
| Full-frame immutable seed-matched Official-SP reference | implemented and sequence-equivalence tested |
| Zero residual and trainable value branch | implemented and CPU-tested |
| Fixed 50/50 self/cross-play training lanes | implemented and CPU-tested |
| Paired group-normalized minimax PPO | implemented and CPU-tested |
| XP-only latent, successor and anchor channels | implemented and CPU-tested |
| Detached full response-coordinate critic grounding | implemented and CPU-tested |
| Final reference-relative KL projection | implemented and CPU-tested |
| Reference/residual/passive/active evaluation modes | implemented |
| Episode-static response posterior and exact 66-outcome VOI | retained and CPU-tested |
| Real 128-lane v6 CUDA preflight | passed on one L40; peak 1,436.18 MiB |
| Paired v6 development matrix | Simple 45-cell run in progress |
| Formal v6 Simple/Wide runs | not run |
| H1/H2/H3 or SOTA claim | unavailable |

Earlier v5 return measurements diagnose why the base actor had to be revised;
they are not v6 performance evidence. Proposed SP/XP targets are unachieved.
The completed CUDA preflight is engineering execution evidence only; it does
not contribute a return result to any hypothesis.
