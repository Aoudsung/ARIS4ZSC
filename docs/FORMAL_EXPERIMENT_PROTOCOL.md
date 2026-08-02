# DELTA-ZSC r3 Formal Experiment Protocol

This protocol is subordinate to
[`DELTA_ZSC_COMPLETE_THEORY_AND_DESIGN.md`](theory/DELTA_ZSC_COMPLETE_THEORY_AND_DESIGN.md)
and may not redefine the method.

## Registered identities

```text
DELTA method: delta_zsc_v5_decision_equivalent_bayes_r3_signal_contract
Config: 7
Manifest: 2
Official protocol: overcooked_v2_iclr2025_5ce1707_v1
Official source: 5ce1707cf31c1c115e6f6ba96db7bc9cc80a850e
```

Both layouts use the fixed Official OvercookedV2 environment contract: view size
2, negative rewards, random agent positions, recipe resampling after delivery,
delivery indication, six actions and 400 steps. Formal workers must use one CUDA
device; no CPU fallback is permitted.

CUDA-only workers retain `JAX_PLATFORMS=cuda` and a GPU default backend. The
fixed Official `ippo.py` contains `jax.debug.print` and
`jax.debug.callback(wandb.log, ...)` logging side effects that JAX 0.4.38 tries
to place on a CPU device. During CUDA-only Official upstream tracing, the local
adapter replaces only these two debug callbacks with no-ops and restores them
immediately afterward. This does not alter tensors, PRNG keys, losses,
gradients, optimizer state, or checkpoints, and the suppression is recorded in
`upstream_summary.json`.

## Training recipes and accounting

SP, State-Augmented, OP and FCP use their unchanged Official Hydra recipes. The
methods intentionally do not share a single training algorithm. OP remains 50M
nominal timesteps and 64 environments; SP and State-Augmented remain 30M and 256
environments; FCP uses its registered population lineage and distinct PPO settings.

DELTA uses 457 updates of 256 environments × 256 steps, for 29,949,952 ego PPO
steps. It additionally discloses owner/base distillation collection, source policy
training, generator episodes, training anchors, audit anchors, calibration,
GPU-hours and peak memory. The maximum r3 anchor budgets are 24,291,328 training
transitions and 88,477,696 independent audit transitions per run; actual execution
is qualification-dependent and is read from the ledger.

Each DELTA seed owns mutually lineage-disjoint resources: one owner-SP source;
SP/OP/SA/FCP generator initialization sources; four development runs per mechanism;
and five calibration runs per mechanism. Official seed indices 0–9 are the outputs
of `split(PRNGKey(42), 10)`, not integer seeds. Engineering validation uses index -1.

## Signal-contract deployment

Every formal seed exports one artifact. C0 failure exports exact owner-SP fallback;
C0-only exports robust base; C3/C4 without C5 exports calibrated passive DELTA;
C5 exports calibrated full active DELTA. Qualification failure never deletes or
replaces a seed. Full DELTA uses the frozen run-block conformal/support gate;
`DELTA-r3 always-on` is an explicit ablation and is forbidden in the formal Full
configuration. Fewer than 19 valid calibration blocks forces base abstention.

## Official scoreboard

For every method and layout, ten final artifacts form ten diagonal SP cells and
ninety ordered off-diagonal XP cells. Each cell runs 500 complete stochastic
episodes with the same 500 environment keys. Carry resets per episode. Only raw
`agent_0` return is accumulated. No intermediate checkpoint selection is allowed.

The reported point estimates are

\[
J_{SP}=\frac1{10}\sum_i\bar R_{ii},\qquad
J_{XP}=\frac1{90}\sum_{i\ne j}\bar R_{ij},\qquad
Gap=J_{SP}-J_{XP}.
\]

The public code does not register a Table-2 Gap standard-deviation formula, so this
repository does not invent one. A 9,999-replicate run-node bootstrap may be reported
only as supplemental DELTA inference.

## Common-partner scoreboard

All methods additionally face the same fresh SP/State-Augmented/OP/FCP partners in
both roles with common episode keys. These partners are excluded from every DELTA
source, development, anchor and calibration set and from every FCP population.
This scoreboard distinguishes method-internal convention compatibility from external
zero-shot coordination.

## Freeze and claim rules

The mechanical engineering seed must complete state-machine path coverage, fallback,
calibration, deployment and resume; CUDA formal-shape preflight must then pass. After
the atomic freeze commit, Official seed 0–9 learning curves cannot change code or
configuration. All ten artifacts enter evaluation. Mechanical checks and C0–C5
qualification are not performance results; scientific claims require the frozen
Official and common-partner scoreboards plus complete resource disclosure.
