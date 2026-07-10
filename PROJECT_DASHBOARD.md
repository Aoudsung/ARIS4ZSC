# PROJECT_DASHBOARD.md

**ARIS-Bellman for Zero-Shot Coordination — pipeline status & decisive-results tracker.**
Last updated: 2026-07-10 · Read this first for "where am I" (30 seconds).

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
| **Current proposal** | [PATH_C_PROPOSAL.md](idea-stage/refine-logs/PATH_C_PROPOSAL.md) (V2, belief-resampled audit §4.5), [PATH_C_THEORY.md](idea-stage/refine-logs/PATH_C_THEORY.md) (supersedes removed PATH_C_T1_PROOF.md), [PATH_C_EXECUTION_PLAN.md](idea-stage/refine-logs/PATH_C_EXECUTION_PLAN.md), [PATH_C_PREREGISTRATION.md](idea-stage/refine-logs/PATH_C_PREREGISTRATION.md) |
| **Thesis** | The agent should recover only the residual partner abstraction that changes control beyond public state, and should do so through value-driven probes while still training the ego representation only with temporal-difference value loss. |
| **Benchmark** | JaxMARL **OvercookedV2** (test-time protocol formation) + toy_factor_game (regression) |
| **Target tier** | ICLR / NeurIPS / ICML class |
| **Compute budget** | 2,200–4,500 GPU-hours (pilot: 500–1,000). Path C has no recorded run yet in this tracker; historical remote runs are logged in [docs/status/EXPERIMENT_LOG.md](docs/status/EXPERIMENT_LOG.md). |

---

## 2. Pipeline status — CURRENT STAGE

**Stage: Path C Phase A — static takeover, implementation audit, and
preregistration freeze.**

The earlier asymmetric-layout + role-conditioned v2 partner line reached a
negative blind-test readout, and the Link-A substrate certificates reached a
second negative result on 2026-07-09:
the terminal handoff substrate still made partner tendency readable from public
state, so waiting and reacting captured nearly all oracle value. Those results
are evidence for redirecting the project, not positive support for the original
ARIS-Bellman claims.

Path C is now the active line. The immediate work is static: make the Path C code
path default-off, verify the planned readouts and hard baselines, and freeze the
go/no-go preregistration. No Path C training, data generation, or scientific
readout is recorded yet. Phase B execution remains locked until the
preregistration is frozen and the user explicitly authorizes remote execution.

2026-07-10 instrument correction (three external review rounds): forking one saved
simulator checkpoint only estimates the response law of the *realized* hidden state,
not the belief-averaged kernel the certificate needs. Kernel readouts must use
belief-resampled posterior-state replicas — Tier 1 = exact enumeration + forward
replay over the finite partner registry; Tier 2 = grouping full-state snapshots by
exact ego-observable history. Spec: [PATH_C_PROPOSAL.md](idea-stage/refine-logs/PATH_C_PROPOSAL.md)
§4.5 + [PATH_C_THEORY.md](idea-stage/refine-logs/PATH_C_THEORY.md) §7–§8; freeze items in
[PATH_C_PREREGISTRATION.md](idea-stage/refine-logs/PATH_C_PREREGISTRATION.md) §J and
`configs/path_c_preregistration.yaml` (`belief_kernel_audit`). E0 must build this
instrument before any kernel readout; the existing pooled-cell kernel audit in
`path_c_evaluation.py` is the coarse distributional fallback, not the estimator.

---

## 3. Decisive results — claims ↔ experiments ↔ status

The old five-claim table is superseded by Path C. It remains historically useful,
but none of its claims has recorded support suitable for paper writing.

Path C's key object is `W_C`: the public-context residual value quotient, meaning
the smallest partner abstraction that changes control after public state is
already known. `F_C` means a value-irrelevant identity or style fingerprint that
may predict raw behavior but must not explain the retained value representation.
All claim-level readouts are Type-B (cross-model + human acquittal required; see
[OPERATING_CONSTRAINTS.md](OPERATING_CONSTRAINTS.md) §4).

| Check | Question | Status |
|---|---|---|
| Signal over hard baselines | Does the Path C representation predict the value-relevant probe response better than public-state, full-history recurrent, identity, and Bayesian/HMM belief-filter baselines at matched budget? | ⬜ NOT RUN |
| Cross-identity transfer | Does the same signal survive when identity, surface style, seed, and layout style are disjoint but the mechanism is the same? | ⬜ NOT RUN |
| Fingerprint exclusion | Is the retained representation unable to rely on `F_C`? | ⬜ NOT RUN |
| Power and null | Does the measurement recover known `W_C` on the synthetic positive and return empty on a terminal-axis null at the same budget? | ⬜ NOT RUN |
| Go/no-go | Do the frozen Path C checks jointly pass before any wider experiment is started? | ⬜ BLOCKED ON PREREGISTRATION + AUTHORIZATION |

Source of truth for run status: [EXPERIMENT_TRACKER.md](idea-stage/refine-logs/EXPERIMENT_TRACKER.md)
(execution checklist) + `docs/status/EXPERIMENT_LOG.md` (results record — active;
ARIS convention, see §6).

---

## 4. Readiness gates (what unlocks each downstream phase)

| Phase / skill | Locked until… |
|---------------|---------------|
| Path C Phase A — static build and audit | Unlocked for static maintenance only: file edits, code inspection, configuration drafting, and preregistration drafting. No local tests or runs. |
| Path C Phase B — go/no-go execution | Path C preregistration frozen, static checks reviewed, and explicit user authorization for remote execution. Each run must read back effective data budget from artifacts. |
| Path C Phase C — full paper-facing evaluation | Phase B returns GO under the frozen rule and Type-B readout agrees that continuing is justified. |
| `/auto-review-loop` (W2) | ≥1 decisive result supported by cross-model verdict |
| `/paper-writing` (W3) | `NARRATIVE_REPORT.md` exists + main claims supported |

Do not start a locked phase. Crossing a gate requires recorded evidence, not inference.

---

## 5. Method invariants (the load-bearing constraints)

These are *correctness* constraints of ARIS-Bellman. Violating them invalidates the
science, not just the run. Full lists: [OvercookedV2_plan.md](artifacts/OvercookedV2_plan.md)
§21 (22 "what not to do"), §15 (preflight); proposal §11 (non-claims), §12 (impl alignment).

Core set — checked mechanically by [`.aris/tools/aris_bellman_fidelity_gate.py`](.aris/tools/aris_bellman_fidelity_gate.py) (checks I1–I9; static, runs in-boundary; ✅ green on current tree). Per [OPERATING_CONSTRAINTS.md](OPERATING_CONSTRAINTS.md) §7 this gate is a **pre-claim / pre-deploy audit**: require it green before reading a claim or deploying changed method code, not before every run start.

- **No G-TVOI / MI selector** in the deployed method. `Δ_info` and MI are *post-hoc
  diagnostics only*. Action selection is pure Bellman `argmax_ω Q(s,b,ω)`.
- **Single TD loss.** No response-prediction / calibration / sparsity / supervised
  factor-label / probe-selection losses in the main method.
- **CE is preprocessing.** Never estimate CE inside the training loop.
- **Reward-scale consistency** across preflight, CE local returns, and training target.
- **Articulation-point bottlenecks**, not `degree ≤ 2`. This is a pre-claim audit for
  the Exp-4 support-graph claim — check it before reading that claim, not as a start gate.
- **Factor deletion removes 3 things**: latent state + evidence route + action
  relevance. This is a pre-ablation audit — run it before the Exp-3 value-sufficiency
  ablation and before reading its claim, not as a start gate.
- **No true-factor oracle and no factor-accuracy as a main V2 metric.** Use
  control-grounded reference-gap closure.

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
| Path C preregistration | `idea-stage/refine-logs/PATH_C_PREREGISTRATION.md` | 🟡 draft, numeric fields pending freeze |
| Path C module design | `idea-stage/refine-logs/PATH_C_MODULE_DESIGN.md` | 🟡 draft / static build map |
| Findings log | `findings.md` | ⬜ missing — ARIS convention for debug/decision log |
| Migration plan | `artifacts/OvercookedV2_plan.md` | ✅ |

**Divergence to reconcile (low priority):** ARIS convention is `refine-logs/` at
project root (sibling of `idea-stage/`); this project nests it as
`idea-stage/refine-logs/`. Downstream skills that hardcode `refine-logs/…` may not
find these. Decide: move, or symlink, or pin the path in `.aris/config.json`.

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
