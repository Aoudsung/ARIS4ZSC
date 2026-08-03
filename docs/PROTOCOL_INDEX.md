# 确认轨协议附卷索引

职责：一页索引，编目全部契约类文档。这些文档管的是"怎么做才有效"，不管"研究什么问题"。研究问题见 [RESEARCH_PROGRAM](RESEARCH_PROGRAM.md)。
纪律：契约文档的修订必须与代码、配置在同一提交内保持一致，修订理由入台账。字节级哈希锁已于 2026-08-03 废除，见[证据台账](status/EVIDENCE_LEDGER.md)。
阶梯级编号对应 [EXPERIMENT_LADDER](research/EXPERIMENT_LADDER.md)：S1 只读面板，S2 受控相图，S3 单 seed preflight（已整级降级），S4 探索性矩阵，S5 正式冻结矩阵。

## 收录文档

| 文档 | 性质 | 生效阶梯级 | 说明 |
|---|---|---|---|
| [theory/DELTA_ZSC_COMPLETE_THEORY_AND_DESIGN.md](theory/DELTA_ZSC_COMPLETE_THEORY_AND_DESIGN.md) | 绑定设计 | S5 方法定义 | 方法架构、损失、预算、执行顺序以它为准。确认轨 commit 冻结后不得再改，冻结前可正常修订 |
| [FORMAL_EXPERIMENT_PROTOCOL.md](FORMAL_EXPERIMENT_PROTOCOL.md) | 正式实验协议 | S5 矩阵、记分板、统计 | Official 10×10 矩阵、共同伙伴记分板、资源台账的定义，从属于绑定设计 |
| [theory/DELTA_ZSC_FOUNDATIONAL_THEORY_AND_PROOFS.md](theory/DELTA_ZSC_FOUNDATIONAL_THEORY_AND_PROOFS.md) | 方法内部引理 | 方法自洽性论证 | 不可识别性下界、决策充分统计量、regret 性质。工程规范以绑定设计为准 |
| [legacy/v44/design/PATH_C_PREREGISTRATION.md](legacy/v44/design/PATH_C_PREREGISTRATION.md) | 历史归档 | 不生效 | V4.4 时代的预注册统计纪律，只作追溯参考 |

## 与新研究层的关系

契约层的内容不再出现在研究层文档里，只允许链接引用。预算数字、注册身份、seed 规则等事实只存在于上表文档中。研究层文档复制这些数字属于违规，会造成多源漂移。

## 修订流程（留档备查）

将来如需修订契约文档：先在台账登记动因，在同一提交内同步更新文档、引用链接和受影响的代码与配置，最后重跑仓库测试确认全绿。确认轨 commit 冻结之后的任何方法修订都属于违规。

## 修订段（2026-08-04，统一重构 I 节，文档重构任务 #16；只追加，不改写上文）

**权威文件层级登记。** 文档收敛为四份权威文件，构成权威层；其余 `docs/research/` 与 `docs/theory/` 文件降为支撑附录与台账，与权威层冲突时以权威层为准：

| 层级 | 文件 | 职责 |
|---|---|---|
| 权威层 1 | [SCIENTIFIC_SPEC](SCIENTIFIC_SPEC.md) | 问题、合法信息、中心主张、estimand、可证伪条件 |
| 权威层 2 | [METHOD_SPEC](METHOD_SPEC.md) | 模型、数据流、损失、partner curriculum |
| 权威层 3 | [EVALUATION_SPEC](EVALUATION_SPEC.md) | 伙伴划分、baseline、统计、资源 |
| 权威层 4 | [THEORY](THEORY.md) | 一般结果、特例与适用边界 |
| 支撑附录 | `docs/research/` 其余文件（DEVELOPMENT_MATRIX、STATISTICAL_PREREGISTRATION、SOTA_BASELINE、THEORY_PREDICTIONS、TRAJECTORY_AND_ESTIMATION_SPEC 等）与 `docs/theory/` 两文件 | 被权威层交叉引用的细则、预测台账与证明全文；其中 DEVELOPMENT_MATRIX 与 STATISTICAL_PREREGISTRATION 分别为矩阵与判决统计的细则权威，由 EVALUATION_SPEC 指向 |

**命名统一决定。** 分支名统一为 `delta-zsc-v5`（`agent/delta-zsc-v5`）；方法名统一为 **DEPI**（Decision-Equivalent Protocol Inference）——原"V6"称谓废止用于当前方法，仅历史台账条目内的 V6 字样按只追加纪律保留；文档、代码提交信息与论文措辞一律用 DEPI。

