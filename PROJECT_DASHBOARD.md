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

**Stage: Path C third-version static implementation review.**

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

Three modules remain deliberately `planned`. The static revision now includes a
cross-fitted ecological return estimator that excludes the target episode outcome,
a content-addressed secondary-profile recomputation path, a restorable OCV2 adapter,
restorable partner controllers, and a split-manifest-bound numeric seed chain from
collection through evaluation. Exact posterior inference now binds the registered
generation controllers, enumerates hidden option choices over the complete primitive
action and public-state path, and uses the same option distribution as generation. The
unresolved work is producing admissible trained acting baseline artifacts, completing
the return-budget primary dependency chain, and freezing the final claim-critical
artifact contract against a real commit. Weak or missing acting artifacts fail closed,
locked-audit ledgers are inaccessible while the primary result is marked not run, and
secondary mappings cannot create or veto a decision. This project is therefore not
benchmark-ready.

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
| Instrument validity | Do the preregistered checks pass and does `beta_lower > alpha_upper` using cell-specific simultaneous bounds? | ⬜ NOT RUN |
| Design/calibration freeze | Were summary, battery, strongest deployable baseline, margins and sample sizes fixed without locked-audit access? | ⬜ NOT FROZEN |
| Locked primary efficacy | Is the lower confidence bound of cross-identity normalized net-return versus probe-budget area-under-the-curve difference greater than the frozen margin? | ⬜ NOT RUN |
| Secondary mechanisms | Do finite-token response, normalized-advantage representation, fingerprint leakage, power/null and ecological cross-fitted value-bin analyses support the scoped interpretation? | ⬜ NOT RUN |

Source of truth for run status: [EXPERIMENT_TRACKER.md](idea-stage/refine-logs/EXPERIMENT_TRACKER.md)
(execution checklist) + `docs/status/EXPERIMENT_LOG.md` (results record — active;
ARIS convention, see §6).

---

## 4. Readiness gates (what unlocks each downstream phase)

| Phase / skill | Locked until… |
|---------------|---------------|
| Path C Phase A — static build and audit | Unlocked for static maintenance only: file edits, code inspection, configuration drafting, and preregistration drafting. No local tests or runs. |
| Path C software and instrument validation | Static implementation reviewed, tests authorized and actually passed, preregistration fully frozen, static cost estimate accepted, and explicit remote-run authorization recorded. Each run must read back effective data budget from artifacts. |
| Path C locked primary evaluation | Software conformance and instrument validity have recorded evidence; design/calibration choices and hashes are frozen before locked-audit access. |
| Path C secondary mechanism evaluation | The locked primary confidence-interval lower bound strictly exceeds the preregistered margin, and Type-B review agrees the scoped continuation is justified. |
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
- **Shared history contract.** Main and deployable history baselines receive the same
  `EgoEvidenceSpecV1`; identity, mechanism and style never enter it.
- **Primary endpoint first.** The decisive comparison is cross-identity normalized
  net-return versus probe-budget area under the curve against the strongest baseline
  selected on the design split. Response prediction is secondary.
- **Complete-state outer draws.** Each belief-kernel outer replicate independently
  samples the complete hidden state. Inner continuations do not increase the outer
  sample count.
- **Finite canonical response.** The theorem-level kernel uses one token from the
  frozen `ResponseSummarySpecV1`; structured multi-label outputs are secondary.
- **Cost must pass before launch.** The static audit-cost artifact reads all dimensions
  and the maximum primitive-step budget from the frozen preregistration. A cost above
  that budget fails before any rollout starts.
- **Decision order is fixed.** Software conformance and the cost artifact are checked
  first, followed by instrument validity, the frozen design baseline, locked primary
  efficacy, and only then secondary mechanism analyses.
- **Summaries are recomputed.** Software conformance binds archived module-registry
  test rows; instrument alpha/beta is rebuilt from outer-cluster token cells; design
  and locked primary statistics are rebuilt from separate complete episode ledgers.
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
test coverage are therefore `tested`. C2, C5 and D1 remain `planned` because they
require training, design selection and frozen semantic/numeric bindings. Nothing is
`frozen`.

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
