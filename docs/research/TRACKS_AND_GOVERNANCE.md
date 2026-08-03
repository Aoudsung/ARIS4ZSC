# 双轨治理

职责：定义探索轨与确认轨的规则和轨间晋升门。两条轨道分开住，审计和好奇互不打扰。
骨架沿用 legacy [RESEARCH_PIPELINE](../legacy/v44/RESEARCH_PIPELINE.md) 的五阶段流水线与文档三层制。

## 探索轨（S1 到 S4）

- 预算上限：单级实验预算在启动前写进[仪表盘](../status/DASHBOARD.md)，超支需先在台账登记理由。
- 允许改假设。探索轨的产物不是结论，是判据的输入。
- 单页报告。一次实验的报告一页封顶，超出的细节进产物文件，不进报告。
- 失败必须入台账。负结果与意外现象同等重要，只追加，不改写。
- 好奇心旁支额度（可选，非强制）：每个阶段可用至多约两成的时间花在注册阶梯之外的问题上，旁支产出同样入台账；不影响方法轨关键路径时才可动用。
- 所有探索轨产物标注 `scientific_readout_allowed: false`，不得用作有效性主张。

## 确认轨（S5）

- 继承全部现有冻结纪律：注册身份、fail-closed 停机条件、seed 规则、claim boundary。完整定义见 [PROTOCOL_INDEX](../PROTOCOL_INDEX.md) 收录的协议文档。
- 这些纪律只约束 S5。探索轨不受其限。
- 冻结后的学习曲线不得反过来修订代码或配置。数值失败按失败报告，不换方法重启。

## 晋升门

探索轨结果想进确认轨，必须全部满足：

1. 与本轨 claim 相关的预测已登记：理论轨见 [THEORY_PREDICTIONS](THEORY_PREDICTIONS.md)（容差已注册）；方法轨（Θ3）须已注册 SOTA 基线数值，见 [SOTA_BASELINE](SOTA_BASELINE.md)。
2. commit 冻结，仓库测试全绿。

任一条不满足，确认轨不启动。没有例外通道。

旁路轨（理论/诊断）对方法轨无阻塞权：其通过与否不构成方法轨晋升的前置条件，也不得拖延方法轨向超越 SOTA 目标的推进。

## 文档三层制

- 活仪表盘：[DASHBOARD](../status/DASHBOARD.md)，不超过 150 行，只写当前阶段、判据状态、唯一下一步。
- 只追加台账：[EVIDENCE_LEDGER](../status/EVIDENCE_LEDGER.md)，结论被推翻时追加新条目，不改旧条目。
- 单次运行报告一页上限，归档进 `docs/status/` 或按主题进 `docs/research/`。
- 退休即归档：不再维护的文档移入 `docs/legacy/`，原样冻结，断链不修。

## 修订段（2026-08-04，统一重构 I 节，文档重构任务 #16；只追加，不改写上文）

**晋升条件治理（冲突冻结）。** 开发矩阵的组件取舍与晋升规则自此以 [DEVELOPMENT_MATRIX](DEVELOPMENT_MATRIX.md) 为唯一权威口径：取舍由配对增量决定，不设制度性停机门槛，任何一级读数为负不触发计划停机。本文件上文"晋升门"一节中与该口径冲突的旧晋升/停机条款（包括门条件隐含的停机语义）予以**冻结**；仍生效的最低条件仅为：commit 冻结且仓库测试全绿、与本轨 claim 相关的预测/基线已登记（理论轨 [THEORY_PREDICTIONS](THEORY_PREDICTIONS.md)、方法轨 SOTA 基线 [SOTA_BASELINE](SOTA_BASELINE.md)）。冻结条款如需恢复，须追加新修订段并先入[证据台账](../status/EVIDENCE_LEDGER.md)，不得静默复活。

**探索轨读数与论文依赖的调和规则。** S1–S4 探索轨产物一律标注 `scientific_readout_allowed: false`，不得直接用作论文有效性主张；与论文表图/论证的依赖冲突按以下规则调和：探索轨读数须经**正式重跑**（确认轨口径，跑前注册）或按**预注册程序复核**后方可入论文；调和规则的完整登记处在 [EVALUATION_SPEC](../EVALUATION_SPEC.md) 第七节，两途皆未通过的读数只保留为台账条目，不入论文。
