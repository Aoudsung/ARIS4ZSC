# DEPI 规范索引

本页是 active 文档的唯一入口。若链接目标与旧研究笔记冲突，以标记为
`authoritative: true` 的文件为准。

## 权威合同

1. [`SCIENTIFIC_SPEC.md`](SCIENTIFIC_SPEC.md)：问题定义、L1–L5 合法信息边界、estimand、
   因果链与 falsifiers。
2. [`METHOD_SPEC.md`](METHOD_SPEC.md)：R0/B0–B2 嵌套结构、机制消融、capability 防坍缩、
   component 干预诊断、exact filter、anchor、M1、calibration 和部署状态。
3. [`EVALUATION_SPEC.md`](EVALUATION_SPEC.md)：Official/Common-Partner 矩阵、统计单位、门限
   与机制控制。
4. [`FORMAL_EXPERIMENT_PROTOCOL.md`](FORMAL_EXPERIMENT_PROTOCOL.md)：正式软件、硬件、seed、
   预算、执行顺序、失败政策和 freeze 合同。

## 理论与研究执行

- [`THEORY.md`](THEORY.md)：可由当前结构证明的有限保证及不作出的保证。
- [`research/DEVELOPMENT_MATRIX.md`](research/DEVELOPMENT_MATRIX.md)：嵌套 core、总预算对照、
  六项机制消融、K sensitivity 与 raw-evaluator/component-diagnostic artifact 合同。
- [`research/STATISTICAL_PREREGISTRATION.md`](research/STATISTICAL_PREREGISTRATION.md)：正式
  inference 与 claim gate 的预注册摘要。
- [`RESEARCH_PROGRAM.md`](RESEARCH_PROGRAM.md)：当前阶段、下一决策点和停止规则；只链接注册
  数字，不复制它们。

## 活状态与审计

- [`status/DASHBOARD.md`](status/DASHBOARD.md)：当前实施/证据状态，不代表科学结果。
- [`status/EVIDENCE_LEDGER.md`](status/EVIDENCE_LEDGER.md)：append-only 证据账本。
- [`status/DECISION_LOG.md`](status/DECISION_LOG.md)：append-only 决策记录。

## 历史材料

已被当前方法取代的理论、研究计划和阶段文档位于 [`legacy/README.md`](legacy/README.md)。它们
只用于 provenance，不能作为实现、配置、统计或 claim 的依据。
