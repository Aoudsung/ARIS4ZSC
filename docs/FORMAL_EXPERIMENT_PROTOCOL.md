# DELTA-ZSC V5 formal experiment protocol

This file is an implementation guide, not experimental evidence. The mathematical authority is [`DELTA_ZSC_COMPLETE_THEORY_AND_DESIGN.md`](theory/DELTA_ZSC_COMPLETE_THEORY_AND_DESIGN.md); this protocol may constrain execution but may not remove or weaken any mechanism required by that document.

## Frozen authorities

- DELTA method: `delta_zsc_v5_decision_equivalent_bayes_r2_official`.
- Config/manifest: versions 5 and 2.
- Official runtime: OvercookedV2 experiments and JaxMARL commit `5ce1707cf31c1c115e6f6ba96db7bc9cc80a850e`.
- Python runtime: 3.10, matching the fixed Official repository instructions.
- Layouts: `test_time_simple` and `test_time_wide`.
- Environment: view size 2, negative rewards, random positions, recipe resampling on delivery, successful-delivery indication, six actions, 400 steps.

Formal software provenance is part of every run identity. An editable Official source is accepted only when it is a clean checkout at the exact fixed commit. The DELTA worktree must likewise be clean and committed: every identity records the full repository SHA, installed-distribution lock, Official SHA, runtime, backend and devices, so an uncommitted implementation is not an admissible formal run. Resume is allowed only with an identical run identity and restores model, optimizer, RNG, runner and generator state.

Formal outputs therefore belong in the ignored `runs/`, `results/` or
`reports/` trees (or outside the checkout). Writing artifacts into a new,
non-ignored source-tree path intentionally makes the next formal command reject
the worktree until those artifacts are moved or explicitly ignored.

## Native training recipes

Training mechanisms are intentionally method-specific. SP, State-Augmented, OP and FCP execute their unmodified Official Hydra recipes. FCP must additionally disclose its complete population lineage and formation ledger. Official FCP retains `NUM_SEEDS=1` because the fixed implementation derives its ten training runs from ten population subdirectories; each population contains eight independently trained SP parents at three checkpoints. OP remains 50M and 64 environments. The paper and fixed source do not register the inner population root keys, so this repository refuses to invent them: FCP requires an explicit frozen population, full lineage and full resource ledger.

| Method | Nominal main PPO | Actual main PPO | Other simulator cost |
|---|---:|---:|---|
| SP | 30M | 29,949,952 | none |
| State-Augmented | 30M | 29,949,952 | 4,000,000 registered state-collection steps |
| OP | 50M | 49,987,584 | symmetry transformations |
| FCP | 30M | 29,949,952 | complete population formation cost |
| DELTA | 30M | 29,949,952 | owned SP/OP support, anchors, support collection and conformal calibration |

Every method has ten final policies per layout. The outer keys are the ten exact outputs of `jax.random.split(jax.random.PRNGKey(42), 10)`, not integer seeds 0–9. DELTA derives ego, generator, snapshot, anchor and calibration domains by fixed `fold_in` tags. No DELTA outer run shares a frozen training checkpoint, parent run, generator snapshot, anchor trajectory, calibration parent, optimizer or teacher state with another outer run.

DELTA uses the Official PPO constants and 256×256 trajectory chunks. Recurrent state crosses chunk boundaries and resets only at a real episode boundary. Exactly 128 environment lanes control agent 0 and 128 control agent 1. Its training reward is recorded as separate raw, Official-shaped and decision-regret-potential components. The Official shaping coefficient reaches zero at 15M environment steps. There is no KL early stopping.

The DELTA-specific values are one frozen implementation instance, not a claim that development evidence established an optimal latent dimension, generator distribution, anchor budget or loss weighting. This experiment tests that exact instance and cannot prove or disprove the complete continuous decision-equivalence method family.

The formal counterfactual schedule has 457 main updates and triggers every eight updates, hence 57 collections. A collection contains 64 ordinary worlds and 32 matched-code pairs (64 intervened worlds), for 128 worlds total. It executes 6 actions, 32 fit replicas, 64 evaluation replicas and a 400-step fixed continuation. Attempted transitions per DELTA run are therefore:

```text
57 × 128 × 6 × 96 × 400 = 1,680,998,400
```

Deterministic microbatch size is selected only from registered complete-world candidates. Keys depend on anchor, action, replica, time and RNG domain, never batch order. Failure to fit the smallest candidate stops the run; budgets are not silently reduced.

The run-block conformal gate remains part of the full method. It uses at least 19 independent calibration parent runs per outer DELTA run at α=0.05 and a frozen support gate. Calibration parents are disjoint from training and common-panel partners.

## Scoreboard 1: Official protocol

For every method and layout, evaluate the ten final policies as 90 ordered off-diagonal pairings plus ten diagonal pairings. Every cell receives the same vector of 500 keys derived from repository-convention `PRNGKey(0)`. Both policies are stochastic and maintain separate recurrent states reset per episode. The only score is the complete raw `agent_0` return over 400 steps.

The policy surface is exactly Official `compute_action(obs, done, hstate, key)` plus `init_hstate`. It has no transition-reward hook. To avoid privileged train/test mismatch, the deployable DELTA belief receives a zero explicit reward channel during both training and evaluation; reward remains available only to PPO targets and training-only response/teacher losses. Delivery evidence available through the registered official observation (`indicate_successful_delivery=true`) remains legal. Partner identity, generator code, environment state, partner parameters/carry, branches and future return are never exposed.

The report contains the 50,000 episode rows, 10×10 cell matrix, ten diagonal SP-cell means, 90 ordered XP-cell means and point Gap. Population standard deviations are reported for SP and XP cells. No Gap standard deviation is invented.

Formal comparison uses 9,999 registered training-node bootstrap replicates. Within a replicate each method resamples its own ten run nodes, uses the same node weights for rows and columns, and all methods share the episode-index resample. The comparator is the best of SP, State-Augmented, OP and FCP within each replicate. DELTA passes only if the one-sided 95% XP lower bound is positive on both layouts and its SP is non-inferior to FCP within one Official correct-delivery reward (20 raw points).

## Scoreboard 2: common unseen partners

The frozen panel has 16 completely fresh partners: four independent SP, four State-Augmented, four OP and four FCP runs. No panel checkpoint, parent or co-training group may occur in any ego method's training lineage. Every one of the ten egos of every method faces every partner in both roles for 500 common-key episodes.

Report mechanism-balanced mean return, worst mechanism, 10% partner-CVaR, deliveries and empirical local BR-Prox. Negative transfer is defined only for DELTA because only DELTA has its registered paired robust-base branch. BR-Prox uses independent fit/evaluation replicas and is explicitly limited to one forced action followed by the frozen deployed continuation; it is not an unrestricted best response.

The common-score comparison uses ego-node and mechanism-stratified partner-node resampling with shared episode indexes. The one-sided 95% lower bound of DELTA minus the replicate-wise best baseline must be positive on both layouts. The worst-mechanism margin is reported as a point estimate; no numeric pass threshold is invented because none was preregistered.

## Capacity, resources and claim boundary

IPPO-Large changes only the shared Official RNN width. Its integer width is selected mechanically to minimize absolute deployment-parameter mismatch to DELTA and is frozen before training. It is an extended capacity control, never relabeled as a Table 2 baseline.

Every artifact reports ego steps, partner/pretraining steps, counterfactual steps, state collection, calibration, evaluation, GPU-hours, peak memory, deployment/training-only parameters and inference latency. Training simulator totals exclude evaluation and never hide auxiliary computation inside the 30M ego budget.

`gpu_hours` means GPU-device hours (`wall time × devices actually used`), not wall-clock hours. DELTA, its calibration, and the one-key upstream trainer use one unsharded default GPU; the fixed multi-device Official baseline process charges every visible GPU it uses. `inference_latency_ms` is the median of 100 synchronized, compiled, stochastic batch-one policy steps after 20 warm-up steps; it includes the recurrent update and action sample but excludes compilation and the environment transition. DELTA resource inputs must include every training ledger and its matching calibration ledger. Upstream SP/OP ledgers must not be added again because each DELTA training ledger already charges its owned support-parent training cost.

Training-time real-return branches are charged to `counterfactual_steps`. Prefix rollouts and all-action branches executed only to fit the frozen conformal/support artifact are charged to `calibration_steps`; the run metadata preserves both calibration sub-counts.

No mechanism claim is unlocked by unit tests or by the two scoreboards alone. The final claim report additionally requires the capacity-control gate and preregistered full-vs-ablation evidence for continuous belief, real-return anchors, decision equivalence, regret potential, conformal gate and continuous generator on both layouts. Every ablation must have positive return-difference lower bound, the predicted intermediate direction and no larger training simulator budget than full DELTA.
