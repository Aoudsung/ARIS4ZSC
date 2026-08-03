# 研究仪表盘

活文档，上限 150 行。只写三样：当前阶段，判据状态，唯一下一步。
历史结论去[证据台账](EVIDENCE_LEDGER.md)，研究内容去 [RESEARCH_PROGRAM](../RESEARCH_PROGRAM.md)。

## 当前阶段

探索轨，论点已裁决（2026-08-03）。主论点 Θ2（信息时序定理生态验证），次论点 Θ5（分解链诊断）。Θ1 判死（迁移差 −83.60，停机效力仅限 Θ1），Θ3 复活为方法载体论点（人类复议推翻传递性门控连带出局），Θ4 挂起；论文形态恢复含 L4（方法 SOTA），DELTA-ZSC-E2E 作为 SOTA 竞争者出场，SOTA 注册基线见 [SOTA_BASELINE](../research/SOTA_BASELINE.md)。快照基线 commit `6b9f598`，后续成果见 commit `eeca646` 及之后。

## 判据状态

| 判据 | 状态 |
|---|---|
| 文献矩阵生态位图谱定稿 | 初稿完成，待复核 |
| 顶会标准方案（PAPER_STANDARD） | 已就位，T2/T3/T4 已形式化 |
| S2 理论工具验证 | 通过，180/180 |
| S1 pilot | 通过，行数与标记完整 |
| S1 全量面板（200,000 行） | 完成，读数入台账 |
| S1 迁移值分析 | 完成，迁移差 −83.60，Θ1 判死 |
| 论点裁决 | 完成：主 Θ2、次 Θ5，候选集冻结 |
| 生态 (Δ, κ, TV) 估计（H2 验证） | 完成：类型 κ̂=0.137，个体 κ̂ 均值 0.099，Δ̂=27.3 |
| E5 插入式 Bayes 路由器对撞（H3/P4） | 未开始 |
| V6 价值排序 preflight（M1，Spearman 门槛） | 激活：随 Θ3 复活恢复，整级降级为 preflight（门槛值跑前入台账；失败则换载体，不停止目标） |

## 唯一下一步

方法轨关键路径（唯一关键路径）：M1 preflight——V6 单 seed 机械跑，价值排序 preflight 门槛（held-out 延续动作排序 Spearman 与预测覆盖率）跑前入台账；失败则换载体，不停止超越 SOTA 的目标。见 [EXPERIMENT_LADDER](../research/EXPERIMENT_LADDER.md) S3。

并行旁路（对方法轨无阻塞权）：

- E5：插入式 Bayes 路由器对撞（H3/P4，实测回收值对理论值 Δ̂·TV̂(t)/2，判据 [0.5,1.5]），见 [TRAJECTORY_AND_ESTIMATION_SPEC](../research/TRAJECTORY_AND_ESTIMATION_SPEC.md) §5。
- 旁路诊断写表：Θ5 面板读数回填（S1 全量面板与 E0 结构分析入表图设计）。
