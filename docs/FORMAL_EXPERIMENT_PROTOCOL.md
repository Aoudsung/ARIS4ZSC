# DELTA-ZSC V6 Formal Experiment Protocol

This protocol is subordinate to the binding
[`DELTA_ZSC_COMPLETE_THEORY_AND_DESIGN.md`](theory/DELTA_ZSC_COMPLETE_THEORY_AND_DESIGN.md)
and may not redefine the method.

## Registered identities

```text
DELTA method: delta_zsc_v6_end_to_end_bayes_coordination
Primary artifact: DELTA-ZSC-E2E
Config: 9
Manifest: 2
Official protocol: overcooked_v2_iclr2025_5ce1707_v1
Official source: 5ce1707cf31c1c115e6f6ba96db7bc9cc80a850e
```

Both layouts use the fixed Official OvercookedV2 environment contract: view size
2, negative rewards, random agent positions, recipe resampling after delivery,
delivery indication, six actions and 400 steps. Formal workers must expose exactly
one CUDA device and must fail closed on a CPU fallback, protocol mismatch, NaN/Inf,
OOM or checkpoint corruption.

The fixed Official `ippo.py` contains logging callbacks which JAX 0.4.38 may try to
place on a CPU device. During CUDA-only upstream tracing, the adapter replaces only
those logging side effects with no-ops and records that suppression. It does not
change tensors, random keys, losses, gradients, optimizer state or checkpoints.

## Training definitions

SP, State-Augmented, OP and FCP use their unchanged Official recipes. OP remains
50M nominal timesteps and 64 environments. SP and State-Augmented remain 30M and
256 environments. FCP retains its registered population lineage and distinct PPO
settings. The methods intentionally do not share one training algorithm.

Every V6 seed runs one fixed end-to-end algorithm for all 457 outer updates. No
C0–C5 statistic, generator score, replay statistic, safety calibration or learning
curve may enable or disable a loss, alter the partner mixture, choose a checkpoint,
substitute an owner policy or change the deployed parameter tree. A numerical
failure is reported as a failure; the seed is not replaced.

Official seed indices 0–9 are the ten keys produced by
`split(PRNGKey(42), 10)`, not integer PRNG seeds. Engineering validation uses the
reserved index -1 and must finish before the V6 commit is frozen. Formal seed
learning curves cannot be used to revise code or configuration after the freeze.

The V6 primary deployment always contains the mechanically final task encoder,
single-Gaussian belief, single belief-conditioned actor, shaped value, twin raw-Q
and structured response parameters. It is named `DELTA-ZSC-E2E`. The optional
`DELTA-ZSC-E2E+Safety` artifact is produced only by `calibrate-safety`; it is an
ablation and cannot replace or enter the primary ten-seed matrix.

All `4 × 64` PPO minibatches execute in their registered order. GAE targets use
the behavior values recorded during rollout, Official value clipping uses the
clipped/unclipped squared-error maximum, and approximate KL is diagnostic only;
it never changes the optimizer-step count.

## Per-run resource ledger

For one formal 457-update V6 run, excluding upstream source-partner training, the
registered attempted-transition budget is:

| Source | Attempted transitions |
|---|---:|
| Ego PPO | 29,949,952 |
| Ego source-policy initialization | 65,536 |
| Generator training | 5,849,600 |
| Counterfactual continuation | 5,701,632 |
| Matched-code legal probes | 14,848 |
| Generator initialization and holdout | 25,600 |
| **Total** | **41,607,168** |

There are exactly 29 anchor triggers, including update zero. Each trigger contains
64 states, six actions, four replicas and 128 continuation steps, for 196,608
continuation transitions. Its 16 matched pairs additionally use 512 legal probe
steps per trigger. The runtime ledger is authoritative and separately reports
upstream partner cost, calibration, evaluation, GPU-hours, peak memory, deployable
parameters and training-only parameters. Auxiliary costs must never be hidden in
the 29,949,952 ego PPO steps.

## Official scoreboard

For every method and layout, ten final artifacts form ten diagonal SP cells and
ninety ordered off-diagonal XP cells. Each cell runs 500 complete stochastic
episodes using the same vector of 500 environment keys. Each policy owns its
recurrent carry, carry resets per episode, and only the complete 400-step raw
`agent_0` return is accumulated. Intermediate checkpoint selection is forbidden.

The point estimates are

\[
J_{SP}=\frac1{10}\sum_i\bar R_{ii},\qquad
J_{XP}=\frac1{90}\sum_{i\ne j}\bar R_{ij},\qquad
Gap=J_{SP}-J_{XP}.
\]

The public code does not register a Table-2 Gap standard-deviation formula, so the
repository does not invent one. A 9,999-replicate run-node bootstrap may be
reported only as supplemental inference and must preserve row/column node
dependence and paired episode-key resampling.

## Common-partner scoreboard

All methods additionally face the same fresh SP, State-Augmented, OP and FCP
partners in both roles with common episode keys. These partners are excluded from
all V6 initialization, training, generator, anchor and optional safety-calibration
sets and from every FCP training population. This scoreboard distinguishes
method-internal convention compatibility from external zero-shot coordination.

## Mechanism audits and claim boundary

`audit-signals` computes the former C0–C5 quantities only after training. They are
read-only measurements of initial/final task competence, partner decision
heterogeneity, held-out raw-Q quality, legal-history recovery, belief-conditioned
control and regret calibration. They cannot write to a checkpoint, deployment,
partner manifest or formal result.

Unit tests, CUDA mechanical runs, formal-shape preflight and mechanism audits prove
only mechanical execution, identity integrity and the stated measurements. ZSC
claims require the frozen Simple and Wide Official matrices, the common-partner
scoreboards, raw-return data and complete resource disclosure.
