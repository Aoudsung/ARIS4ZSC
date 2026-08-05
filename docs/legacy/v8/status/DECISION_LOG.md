# 决策日志（Decision Log）

> **Active-method note (2026-08-05):** entries below are historical and append-only. The current method identity and executable contract are defined only by `docs/PROTOCOL_INDEX.md` and `src/delta_zsc/config.py`; superseded DEPI entries do not override unified DELTA-ZSC.

本日志仅记录项目的**资源与方向决策**（投入、停机、复活、基线口径、治理修订等人类裁决），按时间顺序追加。

**重要边界：本日志中的任何条目都不作为科学论点的证据。** 科学证据只以 `EVIDENCE_LEDGER.md`（证据台账）中登记的实验、审计与统计读数为准。人类裁决可以决定资源投向，但不能证明任何科学主张成立。

---

## D1 — 2026-08-03 人类裁决（复活 Θ3/V6、SOTA 基线与治理修订）

- **日期**：2026-08-03
- **性质**：资源与方向决策（非科学证据）
- **来源**：抄录自 `docs/status/EVIDENCE_LEDGER.md` 既有条目"2026-08-03 人类裁决（后于本日论点裁决）"（台账末条，原文保留未动），并见台账同日新增条目中的迁移指针。

**原文抄录**：

> 人类裁决（后于本日论点裁决）：一、核心原则：在 OvercookedV2 benchmark 中超越 SOTA 的 performance 是唯一核心目标，所有中间过程服务于推进该目标，不设拖延性门槛；方法轨是唯一关键路径，理论/诊断轨为并行旁路、无阻塞权。二、复活 Θ3/V6：推翻本日裁决中 Θ3 的传递性门控连带出局，Θ3（V6 端到端 Bayes 协调为载体）复活为方法载体论点；S1 停机判据的效力限定为仅对 Θ1，"不再投入方法训练"不外溢到 Θ3/方法轨。复议依据：传递性门控越界＋S1 个体级读数（去 identity oracle 84.22）支持个体推断路线。三、SOTA 基线：仅以基准论文 Gessler et al.（arXiv:2503.17821）Table 2 已发表数值为准：Test Time Simple 最优 FCP 6±29，Test Time Wide 最优 FCP 23±40（10 seeds、每格 500 episodes）；此为论文作者自训 checkpoint 数值，口径偏移风险如实披露，但按本裁决不作为立项门槛，判决式仍要求 J_XP 显著高于注册基线数值，注册专页见 `docs/research/SOTA_BASELINE.md`。四、配套治理修订：晋升门精简（保留 commit 冻结与预测/基线登记，删除主论点裁决与 S1/S2 前置），S3 整级降级为 M1 式单 seed preflight（门槛值跑前入台账，失败换载体不停目标），S5 进入条件改为开发矩阵通过＋commit 冻结＋SOTA 基线已注册，论文形态恢复含 L4；本条同时作为 FORMAL_EXPERIMENT_PROTOCOL 判决处补充注册基线的修订动因登记。证据：本条与本次提交 diff。

**迁移说明**：按 2026-08-03 外部六线评审 §11 及 §14 建议（评审归档于 `docs/research/REVIEW_AND_SUGGESTION_2026.md`），该裁决条目自本日起作为决策记录迁移至本日志；台账中原行保留不删不改，但不再以该裁决内容作为任何科学论点的论证依据。另注：同日台账新增条目已登记项目统一重构为 DEPI（Decision-Equivalent Protocol Inference）的方向决策，与本裁决的资源投向（方法轨为唯一关键路径）兼容，后续执行以 DEPI 重构计划为准。

---

## D2 — 2026-08-03 用户对重构计划 F/C 节两项悬而未决问题的科学裁定

- **日期**：2026-08-03
- **性质**：用户对统一重构（DEPI）计划 F 节（SOTA 判决统计程序）与 C 节（latent 信念语义）两项悬而未决问题的裁定
- **来源**：用户直接裁定；落地文件为 `docs/research/STATISTICAL_PREREGISTRATION.md`（裁定记录）、`docs/research/SOTA_BASELINE.md`（生效说明追加段）、本日志与 `docs/status/EVIDENCE_LEDGER.md` 登记条目。

**裁定一（F 节，SOTA 判决路径）：采用路径 1。**

- 内容：在固定 Official commit 上重训/获取 FCP、OP、SA 的 run-level 节点，己方与基线在同 episode keys 上评估，做配对/双样本层级推断（统计单位为训练运行，run 级）；已发表 Table 2 数值（Simple 6±29、Wide 23±40）降为外部 sanity check，不参与判决；"显著超过 SOTA"措辞仅在配对推断支持下使用。
- 后果：FCP 种群重训（约 24 亿步/布局）成为判决批的前置成本项，须按 DEVELOPMENT_MATRIX 第四节全口径记账并跑前登记预算；判决式执行口径以预注册路径 1 为准，路径 2 降级为备用（仅在路径 1 资源不可行并经新裁定后启用）。

**裁定二（C 节，latent 信念语义）：采用补全 Bayes 语义。**

- 内容：为后验补生成模型、似然与 held-out 校准协议（proper scoring rule），保留 Bayes 叙事，而非改称 context embedding。
- 后果：METHOD_SPEC（方法规格）必须定义生成语义与校准——生成模型与似然的形式化定义、后验推导、held-out 校准协议与 proper scoring rule 指标成为方法规格的必备组件；未完成该补全前，不得以"已闭合的 Bayes 信念"名义引用 latent 后验。

**备注**：本条为科学程序裁定（决定判决程序与方法语义的合规口径），其本身仍不作为任何科学论点成立的证据；具体程序条款见 `docs/research/STATISTICAL_PREREGISTRATION.md` 第五节"裁定记录"。

---

## D3 — 2026-08-04 generator 开放前置条件登记与部署白名单复审结论

- **日期**：2026-08-04
- **性质**：资源与方向决策（非科学证据）
- **来源**：任务 #13 返场复审（L_separation 梯度通路修复批次）附带的遗留事项登记；相关实现原语见 `src/path_c/generator_training.py`（`GeneratorCodeArchive`/`empty_code_archive`/`archive_insert`）与 `src/path_c/experiment.py`（`validate_config` 对 `partner_generator.enabled` 的拒绝条款）。

**登记内容**：学习式 partner generator 维持禁用（§7.3 静态宽 partner pool 为默认）。日后若 SA/FCP 上游源到位、拟开放 generator，须先满足以下两项前置条件：

1. **GeneratorCodeArchive 运行时接线**：§7.2 代码档案当前仅有原语与单元测试（`empty_code_archive`/`archive_insert`，容量 256），尚未接入 `TrainState`；接入会改变 checkpoint 身份，须按 CONFIG_VERSION 递增规则一并登记后再行。
2. **update_generator 冻结后空跑的算力浪费修复**：generator 禁用期间其扫描路径仍可能被编译/空跑；建议改为在 `lax.cond` 分支内执行或以 progress 短路，避免无效算力计入资源台账。

**部署白名单复审结论**：本批次复审确认部署白名单维持现状——既有的 5 个子树与三对象（task/capability/protocol）+ actor 的部署图一致，无需扩展。

**备注**：本条仅登记开放条件与白名单现状，不构成对 generator 路线的科学主张；开放时点由后续资源裁决决定。

---

## D4 — 2026-08-04 审查后方法边界与部署白名单更正

- **日期**：2026-08-04
- **性质**：实现/主张边界决策（非科学证据）
- **来源**：《当前仍存在的问题》及同轮代码、权威规格修订。

**登记内容**：近期路线保留固定 sticky transition，将主张收缩为注册离散 response model 下的
decision-equivalent online inference，不声称 B3/主动协议形成。component 统一改称
exchangeable response regimes。为保留合法即时控制，部署图新增无 carry 的 instant-partner
encoder；因此 D3 所述“五个子树白名单维持现状”已被新方法身份取代。当前 deployment schema
白名单以 `METHOD_SPEC.md` 为准，包含 task、instant、capability、component、actor、critic、
response decoder 七个子树。

**备注**：本条只更正后续执行口径，不证明性能、校准或机制结论。

---

## 后续条目结构（预留）

后续每条决策按以下格式追加，只增不改：

- **D\<n\> — \<日期\> \<决策标题\>**
  - **日期**：YYYY-MM-DD
  - **性质**：资源与方向决策（非科学证据）
  - **来源**：裁决出处（如台账条目指针、会议纪要等）
  - **原文抄录**：决策原文（引用块）
  - **迁移说明/备注**：与台账的交叉指针及适用范围（可选）
