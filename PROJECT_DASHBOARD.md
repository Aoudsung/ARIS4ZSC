# PROJECT_DASHBOARD.md

**ARIS-Bellman for Zero-Shot Coordination — pipeline status & decisive-results tracker.**
Last updated: 2026-07-11 · Read this first for "where am I" (30 seconds).

> Required entrypoint per [CLAUDE.md](CLAUDE.md). Execution rules live in
> [OPERATING_CONSTRAINTS.md](OPERATING_CONSTRAINTS.md); this file is *status*, not *permission*.
>
> 治理精简 2026-07-08：本文件的门已按 docs/status/GOVERNANCE_CUTLIST.md 处置；加/减门须过 OPERATING_CONSTRAINTS.md §7 退休阀门。

---

## Terminology (authoritative)

- **ARIS** = the harness (Auto-claude-code Research In Sleep). The tooling under `.claude/skills/`.
- **ARIS-Bellman** = the research method. Factor-local Bellman control for ZSC.
  Code in `src/aris_bellman/` + `experiments/overcooked_v2/`.
- **SP / XP** = self-play within one independent training run / cross-play between
  different independent training runs.

This project (`ARIS4ZSC`) = *using ARIS the harness to develop the ARIS-Bellman method.*

---

## 1. Project identity

| | |
|---|---|
| **Title** | Zero-Shot Coordination via Bellman Control over Value-Sufficient Interaction Factor Beliefs |
| **Current line** | Path C = active value probing for value-sufficient residual partner abstractions. It is the active replacement line after the earlier asymmetric-layout + role-conditioned v2 partner line and Link-A substrate certificates failed to produce a publishable positive claim. |
| **Current proposal** | [PATH_C_PROPOSAL.md](idea-stage/refine-logs/PATH_C_PROPOSAL.md), [PATH_C_THEORY.md](idea-stage/refine-logs/PATH_C_THEORY.md), [PATH_C_MODULE_DESIGN.md](idea-stage/refine-logs/PATH_C_MODULE_DESIGN.md), [PATH_C_EXECUTION_PLAN.md](idea-stage/refine-logs/PATH_C_EXECUTION_PLAN.md), [PATH_C_PREREGISTRATION.md](idea-stage/refine-logs/PATH_C_PREREGISTRATION.md)；实现状态由 `experiments/overcooked_v2/configs/module_registry.yaml` 管理。 |
| **Thesis** | The agent should recover only the residual partner abstraction that changes control beyond public state, and should do so through value-driven probes while still training the ego representation only with temporal-difference value loss. |
| **Benchmark** | JaxMARL **OvercookedV2** (test-time protocol formation) + toy_factor_game (regression) |
| **Target tier** | ICLR / NeurIPS / ICML class |
| **Compute budget** | 尚未冻结。训练步数、梯度更新数、独立身份组数、外层样本数和审计步数仍为空；在 calibration 吞吐与功效结果产生前不填写 GPU-hour。Path C 当前只有软件 Type-A 运行，没有科学实验结果。 |

---

## 2. Pipeline status — CURRENT STAGE

**Stage: Path C public-benchmark migration; R010 smoke passed, then a static code review fixed the probe gate and hot paths — remote evidence re-binding pending before R020 configuration freeze.**

The earlier asymmetric-layout + role-conditioned v2 partner line reached a
negative blind-test readout, and the Link-A substrate certificates reached a
second negative result on 2026-07-09:
the terminal handoff substrate still made partner tendency readable from public
state, so waiting and reacting captured nearly all oracle value. Those results
are evidence for redirecting the project, not positive support for the original
ARIS-Bellman claims.

Path C is now the active line. The third-version interfaces and the P1–P5 code-review
findings have been revised and bound to code commit
`fc647fb00255c5bfc58253a38fa145cb8864afd7`. On GPU 5, the remote CUDA JAX suite
passes 290 tests with no skips; the log SHA-256 is
`6b30b0cd623ebe650f3ea084de2dafb8cea64aef7bb8413587ab70c6b17fbc85` and the
JUnit report SHA-256 is
`d40b43bc7084c8ce916bbc537fd7a6919da28b8b3a760ef552b197b3199f97ab`.
The formal 101-row module-registry software report SHA-256 is
`61f669b0c980f454a138eadd22ed3a846621bb94e86e8bec568023b546af8bf4` and it
binds module-registry SHA-256
`2c22792a0a16318f63ee2efa6dc9adcd542c9a78377c65f650e8d17577e4adb7`.
The preregistration remains an inadmissible template with unfilled numeric and
semantic-hash slots. No Path C training, data generation, instrument result, primary
result, or scientific readout is recorded.

R003's minimal software semantics are complete after a fail-closed CUDA JAX
compatibility scan on GPU 5.
All four proposal candidate layouts expose 96-dimensional public observations, but
their option counts are 28, 28, 26 and 27 and their ordered option identifiers differ.
Across all 23 registered JaxMARL layouts, 22 satisfy the two-agent evidence contract;
the largest group sharing one exact observation-and-option contract has size 1, below
the four role-isolated layouts required by the proposal. The bound Type-A artifact is
`.codex_remote_validation/path_c_r003_all_layout_semantics_bound_fc647fb_20260711.json`
with SHA-256
`6268916dc37a5571e024224e919e596e3d80e1346cd3c86d74266ee8db4798a1`.
The current single `EgoEvidenceSpecV1` vocabulary and raw numeric probe scripts would
therefore change action meaning across layouts. The project did not add a cross-layout
action layer. R003 consequently fixed `asymm_advantages` as an internal single-layout
software target; that choice remains a valid Type-A compatibility result but is no longer
the paper's main benchmark layout.

The resulting code object is commit
`b6f32578837dd3b5146c355500b911400cb42f78`. GPU 5 then passed 108 targeted tests
and the broader 339-test suite with zero failures and zero skips. The broader JUnit
report SHA-256 is
`89116841aad9d63a2c1e7f4a6641b0f17f7b44e27979fc967ce2ebfa9a3e3390`.
The old R010 input fixes `asymm_advantages`, a 96-dimensional public observation,
28 options, the 30-theta `path_c_synthetic` partner registry and the existing random-key
schedule. Its remote artifact SHA-256 is
`1a548d0800d4052b45920cc746184af3de280004fff686c96af52b07ce9c2406`.
It has not been run, and no locked data was accessed. The planned R010 instrument smoke
is now paused because it does not test compatibility with published benchmark results.

The experiment direction was corrected on 2026-07-11. The main paper comparison now
uses the published ICLR 2025 OvercookedV2 Test Time protocol: `test_time_simple` is the
primary layout and `test_time_wide` is the prespecified replication. The standard metric
is mean cross-play (XP) episode return, where independently trained policies are paired
at test time. The official scale is 10 independent seeds, 90 directed cross-seed pairings
covering both player positions, and 500 episodes per pairing. Self-play return and the
self-play-minus-XP gap are co-reported. A literature check through 2026-07-11 found no
later published result using the exact same Test Time Simple/Wide protocol. Fictitious
Co-Play is therefore the best directly comparable published reference found, with XP
`6±29` on Test Time Simple and `23±40` on Test Time Wide; it is not labelled a
protocol-independent global best result.
The official repository publishes code and configuration but no release or downloadable
baseline checkpoint, so published numbers are the current external reference.

The R005 static gap check completed on 2026-07-11 without running any code. Loading the
two Test Time layouts is not a blocker (both are registered in jaxmarl v0.1.0, and R003
already reset all 22 two-agent layouts). Among environment kwargs, only
`indicate_successful_delivery` requires an `OCV2Adapter` code change; view size, random
agent positions, 400 steps and path-planning flags are config-only. The real blockers are
six protocol-semantic gaps: privileged global observation (the 96-dim featurizer reads the
full grid and `state.recipe`), an option action space whose primitive expansion reads full
simulator state, fixed-ego-versus-scripted-partner training with no self-play, reshaped
return accounting instead of raw episode returns, no pairing-matrix/role-swap evaluation,
and a gradient-update budget with no 30M-env-step path. The fixed revision: build a
parallel standard path (official partial observation → recurrent TD ensemble Q → primitive
actions), train each seed self-contained with a within-run self-play partner pool (the same
partner-formation class as Fictitious Co-Play, still a single TD loss), and evaluate with a
pairing-matrix driver producing official SP/XP records; the legacy option/CE/featurizer/
scripted-partner stack is retained only as the mechanism-instrument and ablation line.
Details: `idea-stage/refine-logs/EXPERIMENT_PLAN.md` §9.

The parallel standard path was then implemented statically on 2026-07-11. It adds the
delivery-indicator adapter argument and slot-neutral stepping; exact Simple/Wide
environment files; a convolutional local-observation encoder feeding the existing
recurrent ensemble value network over six primitive actions; seed-contained self-play
pool formation followed by Path C training; and a separate checkpoint-pairing evaluator
that persists raw 400-step episode returns and summarizes 10 self-play plus 90 directed
cross-play pairings. The smoke configuration is explicitly marked as ineligible for
scientific readout and requires the CUDA JAX backend. No local or remote code was run,
and the working tree has not been committed, so this implementation is not yet a
software compatibility result.

R010 passed remotely on 2026-07-11 using GPU 0 and the CUDA 12 JAX wrapper. The
targeted regression contains 56 passed tests with no failures or skips. The smoke
artifact reports `jax_backend=gpu`, device `cuda:0`, two independent training seeds,
2 self-play pairings, 2 directed cross-play pairings, 2 episodes per pairing and all
8 raw-return rows. Each seed completed 1,600 environment steps, split evenly between
self-play pool formation and Path C training. The run also corrected a static shape
assumption: with the delivery indicator enabled, Simple observations are 5×5×39 and
Wide observations are 5×5×43. The final smoke remains ineligible for scientific
readout; its zero returns are not evidence about performance.

Three legacy modules remain deliberately `planned`. The static revision now includes a
cross-fitted ecological return estimator that excludes the target episode outcome,
a content-addressed secondary-profile recomputation path, a restorable OCV2 adapter,
restorable partner controllers, and a split-manifest-bound numeric seed chain from
collection through evaluation. Exact posterior inference now binds the registered
generation controllers, enumerates hidden option choices over the complete primitive
action and public-state path, and uses the same option distribution as generation. The
current unresolved work is different from the old registry wording: freeze the still-open
formal training numbers and layout-specific model configurations before R020. Only Path C itself must be trained
for the main table; the internal recurrent and belief systems are ablations. This project
is therefore not yet benchmark-ready.

The instrument correction is binding: a single saved simulator state estimates only
the response law conditional on that realized hidden state. Every outer replicate must
instead draw the complete hidden state `U=(theta, execution_state)` independently from
the posterior given the registered history. `M` counts those complete-state draws;
`L_inner` only estimates future continuation noise. Exact enumeration supplies the
posterior from which complete states are sampled; an analytically weighted mixture is
secondary unless it receives a matching concentration proof. Exact mode cannot prune
positive posterior mass.

The response kernel uses the finite vocabulary from `ResponseSummarySpecV1`; all
history-based agents share `EgoEvidenceSpecV1`; and train, design, calibration and
locked-audit roles are group-disjoint. Replay uses original random keys, while audit
forks use fresh named random streams. An identical immutable snapshot and fork
coordinate must reproduce the same output without mutating the source snapshot.

---

## 3. Decisive results — claims ↔ experiments ↔ status

The old five-claim table is superseded by Path C. It remains historically useful,
but none of its claims has recorded support suitable for paper writing.

Path C's key object is `W_C`: the public-context residual value quotient, meaning
the smallest partner abstraction that changes control after public state is
already known. The primary retained code is the normalized advantage decision code;
subtracting a learned public-state baseline is only a secondary diagnostic. `F_C`
means a value-irrelevant identity or style fingerprint that may predict raw behavior
but must not explain the retained value representation.
All claim-level readouts are Type-B (cross-model + human acquittal required; see
[OPERATING_CONSTRAINTS.md](OPERATING_CONSTRAINTS.md) §4).

| Required result | Question | Status |
|---|---|---|
| Software conformance | Do sequence, evidence, response, posterior, random-key and artifact semantics match their frozen versions? | 🟢 CODE COMMIT + CUDA JAX 290-TEST REPORT BOUND; FROZEN PREREGISTRATION STILL MISSING |
| Published-protocol compatibility | Can Path C produce 10 independent policies and the official `test_time_simple`/`test_time_wide` SP/XP, all-pairing and role-swapped records? | 🟡 R010 CUDA JAX SMOKE PASS 2026-07-11 (small-scale SP/XP and role-swap wiring verified), then same-day static review confirmed all eight protocol clauses but fixed a degenerate probe-gate default and hot-path waste; remote regression + smoke must re-bind evidence to the fixed code before R020 |
| Standard benchmark performance | What are Path C's mean XP episode returns on both Test Time layouts relative to the published FCP references `6±29` and `23±40`? | ⬜ NOT RUN |
| Probe ablation | Under the same Path C checkpoint and interaction cost, does value-directed probing outperform random and no probe? | ⬜ NOT RUN |
| Instrument and mechanism validity | Do the belief-kernel, positive/null and fingerprint checks support the scoped representation interpretation? | ⬜ NOT RUN; DOES NOT BLOCK STANDARD XP PERFORMANCE |

Source of truth for run status: [EXPERIMENT_TRACKER.md](idea-stage/refine-logs/EXPERIMENT_TRACKER.md)
(execution checklist) + `docs/status/EXPERIMENT_LOG.md` (results record — active;
ARIS convention, see §6).

---

## 4. Readiness gates (what unlocks each downstream phase)

| Phase / skill | Locked until… |
|---------------|---------------|
| Path C Phase A — static build and audit | Unlocked for static maintenance only: file edits, code inspection, configuration drafting, and preregistration drafting. No local tests or runs. |
| Path C standard benchmark smoke | ✅ R010 passed remotely on CUDA JAX; smoke artifacts are Type-A only. |
| Path C main training and XP evaluation | Standard smoke passes; official Test Time configuration and 10-seed/30M-step/500-episode schedule are fixed; explicit remote-run authorization recorded. Each run reads back its effective data budget. Instrument validity is not a start condition. |
| Path C mechanism evaluation | The relevant posterior/reset and instrument checks have recorded evidence. These checks gate mechanism wording only. |
| `/auto-review-loop` (W2) | ≥1 decisive result supported by cross-model verdict |
| `/paper-writing` (W3) | `NARRATIVE_REPORT.md` exists + main claims supported |

Do not start a locked phase. Crossing a gate requires recorded evidence, not inference.

---

## 5. Method invariants (the load-bearing constraints)

These are *correctness* constraints of ARIS-Bellman. Violating them invalidates the
science, not just the run. Full lists: [OvercookedV2_plan.md](artifacts/OvercookedV2_plan.md)
§21 (22 "what not to do"), §15 (preflight); proposal §11 (non-claims), §12 (impl alignment).

Core set — checked mechanically by [`.aris/tools/aris_bellman_fidelity_gate.py`](.aris/tools/aris_bellman_fidelity_gate.py) (checks I1–I9; static, runs in-boundary; ✅ green on current tree). Per [OPERATING_CONSTRAINTS.md](OPERATING_CONSTRAINTS.md) §7 this gate is a **pre-claim / pre-deploy audit**: require it green before reading a claim or deploying changed method code, not before every run start.

- **No mutual-information training objective.** Path C training probes may use
  preregistered normalized-advantage ensemble disagreement, with propensity, support,
  budget and cost logged. It does not add an information-gain loss. Ordinary
  exploitation remains Bellman action-value maximization; direct-information probing
  is a design-only diagnostic baseline.
- **Single TD loss.** No response-prediction / calibration / sparsity / supervised
  factor-label / probe-selection losses in the main method.
- **Shared history contract.** Main and internal history ablations receive the same
  `EgoEvidenceSpecV1`; identity, mechanism and style never enter it.
- **Primary endpoint first.** The decisive comparison is mean XP episode return on
  `test_time_simple` and `test_time_wide`, using the published protocol. SP and the
  SP-minus-XP gap are co-reported. The probe-budget curve is a supporting ablation.
- **Complete-state outer draws.** Each belief-kernel outer replicate independently
  samples the complete hidden state. Inner continuations do not increase the outer
  sample count.
- **Finite canonical response.** The theorem-level kernel uses one token from the
  frozen `ResponseSummarySpecV1`; structured multi-label outputs are secondary.
- **Instrument cost is scoped.** The static audit-cost artifact applies to belief-kernel
  instrument rollouts. It does not gate standard Test Time training or XP evaluation.
- **Decision order is fixed.** First align and run the published benchmark protocol;
  then run Path C ablations; then use the instrument to justify mechanism wording.
- **Summaries are recomputed.** Standard SP/XP statistics are rebuilt from raw episode,
  seed-pair and player-position records. Instrument alpha/beta is separately rebuilt
  from outer-cluster token cells.
- **Dataset roles are derived.** Version-3 collection accepts a frozen split group,
  derives its role from `SplitManifestV1`, and the evaluator parses Parquet rows to
  recompute episode, transition and group coverage rather than trusting manifest totals.
- **CE is preprocessing.** Never estimate CE inside the training loop.
- **Reward-scale consistency** across preflight, CE local returns, and training target.
- **Articulation-point bottlenecks**, not `degree ≤ 2`. This is a pre-claim audit for
  the Exp-4 support-graph claim — check it before reading that claim, not as a start gate.
- **Factor deletion removes 3 things**: latent state + evidence route + action
  relevance. This is a pre-ablation audit — run it before the Exp-3 value-sufficiency
  ablation and before reading its claim, not as a start gate.
- **No oracle labels in deployable agents.** Identity, mechanism, style and true value
  classes never enter deployable evidence or training. Synthetic registry truth is
  allowed only for instrument controls, value-class labels and explicitly separated
  oracle diagnostics; it is not a deployable benchmark input.

Preflight is the one hard *start* gate; it now lives with the Formal Exp 1–5 gate in
§4 (see [OPERATING_CONSTRAINTS.md](OPERATING_CONSTRAINTS.md) §2 for `--preflight_path`).

---

## 6. Artifact map & known divergences

| Artifact | Location | Notes |
|----------|----------|-------|
| Final proposal | `idea-stage/refine-logs/FINAL_PROPOSAL.md` | ✅ |
| Experiment plan | `idea-stage/refine-logs/EXPERIMENT_PLAN.md` | ✅ |
| Experiment tracker | `idea-stage/refine-logs/EXPERIMENT_TRACKER.md` | ✅ (status checklist) |
| Experiment log (results) | `docs/status/EXPERIMENT_LOG.md` | ✅ active |
| Path C proposal | `idea-stage/refine-logs/PATH_C_PROPOSAL.md` | ✅ active current line |
| Path C execution plan | `idea-stage/refine-logs/PATH_C_EXECUTION_PLAN.md` | ✅ active current plan |
| Path C preregistration | `idea-stage/refine-logs/PATH_C_PREREGISTRATION.md` | 🟡 third-version template; numeric and semantic hashes not frozen |
| Path C module design | `idea-stage/refine-logs/PATH_C_MODULE_DESIGN.md` | ✅ third-version static map; remote full suite passed |
| Path C module registry | `experiments/overcooked_v2/configs/module_registry.yaml` | ✅ remote suite exit 0; entries remain implemented or planned until a final commit and per-test archived rows exist |
| Migration plan | `artifacts/OvercookedV2_plan.md` | ✅ |

**Divergence to reconcile (low priority):** ARIS convention is `refine-logs/` at
project root (sibling of `idea-stage/`); this project nests it as
`idea-stage/refine-logs/`. Downstream skills that hardcode `refine-logs/…` may not
find these. Decide: move, or symlink, or pin the path in `.aris/config.json`.

### Path C module traceability

The machine-readable source is `experiments/overcooked_v2/configs/module_registry.yaml`.
The P1–P5 code object is bound to commit
`fc647fb00255c5bfc58253a38fa145cb8864afd7`; its remote CUDA JAX run passed 290
tests with no skips and archived a JUnit report hash. Modules with complete registered
test coverage are therefore `tested`. C2, C5 and D1 retain their historical `planned`
labels because the registry has not yet been migrated to the public-benchmark protocol.
Nothing is `frozen`.

The registry identifiers `C2_PRIMARY_RETURN_BUDGET_AUC` and
`C5_ACTING_BASELINE_BENCHMARK` describe the superseded internal-baseline plan.
Their `planned` status no longer means that five self-built baselines must be trained
before the public benchmark. A later code revision must either rename them or scope
them explicitly to ablations, and add the standard Test Time SP/XP evaluator.

<!-- PATH_C_MODULE_TRACEABILITY:BEGIN -->
| Module ID | Status |
|---|---|
| A1_CONFIG_BINDING | tested |
| A2_PRIME_RECURRENT_SEQUENCE | tested |
| A3_SEQUENCE_TEMPORAL_DIFFERENCE | tested |
| A4_NORMALIZED_ADVANTAGE_PROBE | tested |
| A5_BASE_RESIDUAL_SECONDARY | tested |
| B1_SYNTHETIC_FACTORIAL | tested |
| B2_RESPONSE_AND_STORAGE | tested |
| B3_ECOLOGICAL_VALUE_CLASSES | tested |
| C1_RESPONSE_READOUT_SECONDARY | tested |
| C2_PRIMARY_RETURN_BUDGET_AUC | planned |
| C3_RETAINED_DECISION_CODE | tested |
| C4_POWER_AND_SECONDARY_REPORT | tested |
| C5_ACTING_BASELINE_BENCHMARK | planned |
| C6_TWO_STAGE_DECISION | tested |
| I1_IMMUTABLE_OCV2_SNAPSHOT | tested |
| I2_EXACT_SHARED_POSTERIOR | tested |
| I3_FULL_STATE_OUTER_SAMPLING | tested |
| I4_EXACT_HISTORY_MATCHING | tested |
| I5_FROZEN_AUDIT_BATTERY | tested |
| I6_SIMULTANEOUS_KERNEL_BOUNDS | tested |
| I7_VALIDITY_CONTROLS | tested |
| I8_SPLIT_AND_CROSS_FITTING | tested |
| D1_ARTIFACT_CONTRACT | planned |
| D2_CONFORMANCE_TEST_DEFINITIONS | tested |
<!-- PATH_C_MODULE_TRACEABILITY:END -->

---

## 7. Entry points

- Execution rules → [OPERATING_CONSTRAINTS.md](OPERATING_CONSTRAINTS.md)
- Agent coordination / Codex scope → [AGENTS.md](AGENTS.md)
- Remote contract → [CUSTOMER.md](CUSTOMER.md)
- Project posture (machine-readable) → [`.aris/config.json`](.aris/config.json)

---

## 8. Historical Root-Cause Repair Status — 2026-07-02

This section records the closed repair line that preceded Path C. It is not the
current stage. Historical OvercookedV2 results remain diagnostic only under
`FINDINGS_LEDGER` P1/P2/P3/P4/P5/S17/S20/NEW-2/NEW-3/NEW-4. Later blind-test and
Link-A readouts closed the asymmetric-layout + role-conditioned v2 partner line
as a positive-paper path;
their negative results now motivate Path C.

| Claim / artifact | current status | blocking / dependency IDs | allowed interpretation |
|---|---|---|---|
| Oracle-free factor evidence | static repair complete, remote pending | P1, NEW-G3, I10 | clean-test candidate after diff review |
| Accumulated belief / failure traces | static repair complete, ablation pending | P4, S1-S3, I11 | mechanism claim pending ablation |
| CE support graph claims | static repair complete, CE probe pending | P3, S8-S11, D4-D5, I12 | support-relative only until CE probe |
| Black-box main-method objective | static repair complete, objective gate pending | P5, I13 | no role/terminal-policy main-claim evidence |
| Formal eval / checkpoint path | static repair complete, gated rerun pending | S17, S20, NEW-2, I14-I18 | may be used for preregistered rerun after diff review |
| `role_conditioned_v2_candidate` | BENCHMARK-CANDIDATE | P2, W1-W7, D1-D5 | inspect/certify only; not approved benchmark |

Dashboard rule (master quarantine-release): no result may move a claim from pending to supported/refuted — and no historical OvercookedV2 headline result may be read as validation or refutation — until the relevant ledger IDs above are either closed by remote verification or explicitly waived by Type-B human decision with reason. The former standalone "Core ZSC ARIS-vs-baseline claim" and "Historical headline OvercookedV2 results" rows fold into this rule plus the section intro (which already records "the core claim is not validated or refuted" and "historical results remain diagnostic only under FINDINGS_LEDGER P1/P2/P3/P4/P5/S17/S20/NEW-2/NEW-3/NEW-4"); the core claim's dependency IDs (P1, P3, P4, P5, S17, S20, NEW-2) are distributed across the mechanism rows above.

---

## 已归档 / 已合并（2026-07-08，依据 docs/status/GOVERNANCE_CUTLIST.md）

治理精简处置留档。以下门已从在架门面移除，仅在此记录来龙去脉；载重残留（若有）已注明去向。

- **归档 · §4 `/result-to-claim` 解锁门**：原为「某个实验已产出真实 `EXPERIMENT_LOG.md` 结果文件才解锁 `/result-to-claim`」。归档理由：重言——没有结果文件本就无从做 claim，此门未挡过任何有记录的失败。载重残留「读数须从产物回读、不得凭 config 意图」已由数据充分性纪律（[OPERATING_CONSTRAINTS.md](OPERATING_CONSTRAINTS.md) §6）覆盖。
- **归档 · §4 rebuttal / resubmit / camera-ready 解锁门**：原为「收到外部评审 / 录用通知才解锁」。归档理由：逻辑前提，不护任何失败模式——没有外部评审本就无从反驳。相关执行边界仍由 [OPERATING_CONSTRAINTS.md](OPERATING_CONSTRAINTS.md) §1 保。
- **合并 · §5 「Preflight is a hard gate」静态不变量**：折进 §4 的 Formal Exp 1–5 启动门（同源同义，对应 S27——无效布局污染 12000 行 replay，事后无法补救）。preflight 仍是唯一的硬启动门；`--preflight_path` 细节见 [OPERATING_CONSTRAINTS.md](OPERATING_CONSTRAINTS.md) §2。
- **合并 · §8 表「Historical headline OvercookedV2 results = QUARANTINED」行**：折进 §8 主隔离规则与段首说明。历史结果仅诊断、既不证实也不证伪的口径保留在段首句与主隔离规则中（依赖 ID P1/P2/P3/P4/P5/S17/S20/NEW-2/NEW-3/NEW-4）。
- **合并 · §8 表「Core ZSC ARIS-vs-baseline claim = testable only after remote gates」行**：折进 §8 主隔离规则（claim 表↔不变量↔fidelity 门三处重述塌成一处）。「核心 claim 尚无科学结论」的口径保留在段首句；其依赖 ID（P1、P3、P4、P5、S17、S20、NEW-2）已分布在各机制行。
