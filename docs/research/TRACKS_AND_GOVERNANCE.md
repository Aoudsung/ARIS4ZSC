# 双轨治理

职责：定义探索轨与确认轨的规则和轨间晋升门。两条轨道分开住，审计和好奇互不打扰。
骨架沿用 legacy [RESEARCH_PIPELINE](../legacy/v44/RESEARCH_PIPELINE.md) 的五阶段流水线与文档三层制。

## 探索轨（S1 到 S4）

- 预算上限：单级实验预算在启动前写进[仪表盘](../status/DASHBOARD.md)，超支需先在台账登记理由。
- 允许改假设。探索轨的产物不是结论，是判据的输入。
- 单页报告。一次实验的报告一页封顶，超出的细节进产物文件，不进报告。
- 失败必须入台账。负结果与意外现象同等重要，只追加，不改写。
- 好奇心旁支额度：每个阶段允许约两成的时间花在注册阶梯之外的问题上，旁支产出同样入台账。
- 所有探索轨产物标注 `scientific_readout_allowed: false`，不得用作有效性主张。

## 确认轨（S5）

- 继承全部现有冻结纪律：注册身份、fail-closed 停机条件、seed 规则、claim boundary。完整定义见 [PROTOCOL_INDEX](../PROTOCOL_INDEX.md) 收录的协议文档。
- 这些纪律只约束 S5。探索轨不受其限。
- 冻结后的学习曲线不得反过来修订代码或配置。数值失败按失败报告，不换方法重启。

## 晋升门

探索轨结果想进确认轨，必须全部满足：

1. 主论点已完成裁决并写入[证据台账](../status/EVIDENCE_LEDGER.md)。
2. 该论点的全部前置阶梯级通过（见 [EXPERIMENT_LADDER](EXPERIMENT_LADDER.md) 各级通过判据）。
3. 全部可证伪预测已登记，容差已注册（见 [THEORY_PREDICTIONS](THEORY_PREDICTIONS.md)）。
4. commit 冻结，仓库测试全绿。

任一条不满足，确认轨不启动。没有例外通道。

## 文档三层制

- 活仪表盘：[DASHBOARD](../status/DASHBOARD.md)，不超过 150 行，只写当前阶段、判据状态、唯一下一步。
- 只追加台账：[EVIDENCE_LEDGER](../status/EVIDENCE_LEDGER.md)，结论被推翻时追加新条目，不改旧条目。
- 单次运行报告一页上限，归档进 `docs/status/` 或按主题进 `docs/research/`。
- 退休即归档：不再维护的文档移入 `docs/legacy/`，原样冻结，断链不修。
