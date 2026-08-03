# 研究总纲

本仓库文档体系的入口。2026-08-03 文档重构后，体系顶层是研究方案，契约内容降级为确认轨协议附卷。

## 这个研究想干什么

一句话：搞清楚零样本协调的差距到底在哪一环，然后按证据决定往哪押注。

过去的文档体系把工程契约放在顶层：注册身份、门控、冻结协议写满了三份主文档，真正的科学问题压在 legacy 归档里。重构把顺序翻了过来。现在顶层是候选论点、判决性实验和理论预测，契约退到附卷，只管"S5 正式实验怎么做才有效"。

中心论点状态（2026-08-03 裁决与人类复议）：主论点 Θ2（信息时序定理生态验证），次论点 Θ5（分解链诊断）；**Θ3 复活为方法载体论点**（V6 端到端 Bayes 协调为载体，复议依据见 [THESIS_CANDIDATES](research/THESIS_CANDIDATES.md) 末节）。核心原则：在 OvercookedV2 benchmark 中超越 SOTA 的 performance 是唯一核心目标，方法轨是唯一关键路径，理论/诊断轨为并行旁路、无阻塞权。

## 当前阶段与唯一下一步

探索轨，论点已裁决。唯一下一步：方法轨关键路径——执行 M1 preflight（V6 单 seed，价值排序 preflight 门槛值跑前入台账；失败则换载体，不停止目标）。并行旁路（不阻塞方法轨）：E5 插入式 Bayes 路由器对撞（H3/P4，见 [TRAJECTORY_AND_ESTIMATION_SPEC](research/TRAJECTORY_AND_ESTIMATION_SPEC.md) §5）、旁路诊断写表（Θ5 面板读数回填）。详见[仪表盘](status/DASHBOARD.md)。

## 研究层文档地图

| 文档 | 一句话定位 |
|---|---|
| [research/THESIS_CANDIDATES](research/THESIS_CANDIDATES.md) | 五个候选论点、falsifier、四维评估、裁决留空 |
| [research/RESEARCH_THESIS](research/RESEARCH_THESIS.md) | 中心论点，初版只含已排除假说，裁决后展开 |
| [research/LITERATURE_MATRIX](research/LITERATURE_MATRIX.md) | 文献卡片与生态位图谱 |
| [research/EXPERIMENT_LADDER](research/EXPERIMENT_LADDER.md) | S1 到 S5 判决性实验阶梯 |
| [research/THEORY_PREDICTIONS](research/THEORY_PREDICTIONS.md) | 理论的定量预测表与失配条款 |
| [research/PAPER_STANDARD](research/PAPER_STANDARD.md) | 顶会标准：主张层级、定理清单、基线注册、表图设计与审稿防御 |
| [research/SOTA_BASELINE](research/SOTA_BASELINE.md) | SOTA 基线注册专页：布局映射、已发表数值、判决式与口径偏移披露 |
| [research/TRACKS_AND_GOVERNANCE](research/TRACKS_AND_GOVERNANCE.md) | 探索轨与确认轨规则、晋升门 |
| [status/DASHBOARD](status/DASHBOARD.md) | 活仪表盘，当前阶段与唯一下一步 |
| [status/EVIDENCE_LEDGER](status/EVIDENCE_LEDGER.md) | 只追加证据台账 |

## 确认轨冻结协议（语义降级声明）

以下三份文档的定位从"体系顶层设计"改写为**确认轨冻结协议**：它们定义 S5 正式实验的方法、矩阵与统计纪律，不再承担研究方案的职能。确认轨 commit 冻结前可正常修订，冻结后不得再改。

- [theory/DELTA_ZSC_COMPLETE_THEORY_AND_DESIGN.md](theory/DELTA_ZSC_COMPLETE_THEORY_AND_DESIGN.md)，确认轨绑定设计。
- [FORMAL_EXPERIMENT_PROTOCOL.md](FORMAL_EXPERIMENT_PROTOCOL.md)，S5 矩阵、记分板与 claim boundary。
- [theory/DELTA_ZSC_FOUNDATIONAL_THEORY_AND_PROOFS.md](theory/DELTA_ZSC_FOUNDATIONAL_THEORY_AND_PROOFS.md)，方法内部引理。

完整编目与解冻流程见 [PROTOCOL_INDEX](PROTOCOL_INDEX.md)。

## 归档层

- [legacy/v44](legacy/v44/README.md)，V4.4 研究线全量归档，只读，断链不修。本方案的理论资产（信息价值分解链、TV 匹配界、历史需求定理）与三条负面证据都出自这里，引用时带证据指针。
- [legacy/v5_r3](legacy/v5_r3/README.md)，根级旧副本归档，非权威。

## 阅读顺序建议

第一次读：本文档，然后 THESIS_CANDIDATES，然后 EXPERIMENT_LADDER。
要动手做实验：EXPERIMENT_LADDER 找到所在级，对照 TRACKS_AND_GOVERNANCE 的轨道规则。
要改方法：先读 PROTOCOL_INDEX 确认哪些文件碰不得。
