# 证据台账（只追加）

每条：日期、结论、证据指针。结论被推翻时追加新条目，不改旧条目。
首批条目摘录自 legacy 源出台账 [legacy/v44/status/FINDINGS_LEDGER.md](../legacy/v44/status/FINDINGS_LEDGER.md)，原始证据细节以源出台账为准，摘录不改原文。

---

## 自 legacy 移植条目

- **2026-07-09** Link-A 隐藏性证书两轮结构性判负，旧方法线在 OvercookedV2 asymm 基底上死亡，项目转向 Path C。证据：源出台账第 1 条。

- **2026-07-14** 自写 torch 训练器同预算只得 19 分（官方 IPPO 复现约 130），统一根因为训练栈缺陷；此后正式产物一律用官方训练器。证据：源出台账第 2 条。

- **2026-07-28** V4.4 开发训练保持自我配对能力（四模式均约 168），但保守预测收益零转化：47 触发 → 12 动作差异 → 0 奖励差异；固定伙伴方向不可重复。证据：源出台账第 3 条，`legacy/v44/status/PATH_C_V4_4_SEED100_DEVELOPMENT_RESULTS.md`。

- **2026-07-29** 冻结 checkpoint 反事实审计（1.01 亿环境步、225 触发态 × 128 replica）：预测下界覆盖率 29.33%（登记 95%），预测与经验动作排序 Spearman +0.013，高分组经验收益低于低分组。V4.4 预测器不携带真实动作价值排序。证据：源出台账第 4 条。

- **2026-07-29** 官方策略库 5×4 交叉矩阵：最佳固定策略跨伙伴均值 +19.82（SP-100），逐伙伴最优选择均值约 139；差距是策略模式选择差距。证据：源出台账第 5 条，`legacy/v44/status/PATH_C_V4_4_COUNTERFACTUAL_AND_LIBRARY_MATRIX_REPORT.md`。

- **2026-07-29** 代码级根因裁决：V4.4 八个槽是同一共享执行策略的八个 critic，outcome 模型以 target 网络自身 Q 为标签（自蒸馏），学到的是"同一策略的多个自洽价值估计"而非"伙伴条件控制价值"。证据：源出台账第 6 条。

- **2026-07-29** 研究问题重述：本基底是隐藏离散语境问题，语境为伙伴惯例，单步不可读，回合级价值约 120 分；原提案在动作粒度上定价信息，而价值全部位于惯例粒度。证据：源出台账第 7 条。

- **2026-07-29** V4.4 artifact-only 真实回报读出：225 触发状态 64/64 拆分、四折 LOPO、10,000 次伙伴分层 bootstrap，oracle lift −0.0135 [−0.1649,+0.1399]，历史 probe lift −0.3597 [−0.9326,+0.0852]，区间均跨 0，登记裁决 INCONCLUSIVE。119.65 表观差距完全来自同 checkpoint 对角格。run-disjoint 兼容性机会仍未测量。证据：源出台账第 8 条，`legacy/v44/status/PATH_C_V4_4_ARTIFACT_VALUE_READOUT_REPORT.md`。

- **2026-07-30** V4.4 读出完整性复核：9 个确定性产物逐字节一致，SHA-256 清单不变，独立重算与机器摘要一致，独立 reviewer 裁决 PASS。结论不扩大证据范围。证据：源出台账第 9 条，`legacy/v44/EXPERIMENT_AUDIT.md`。

## 本台账新增条目

- **2026-08-03** 文档重构前置：工作区 72 个未提交改动提交为快照 commit `6b9f598`（agent/delta-zsc-v5 分支），作为研究方案文档重构的回滚基线。证据：`git log --oneline -2`。

- **2026-08-03** 根级孤儿文档归档：仓库根级 2148 行旧版理论文档（V5 r3，不在任何 git 仓库内）复制至 `docs/legacy/v5_r3/`，附非权威声明，原文件保留。证据：`docs/legacy/v5_r3/README.md`。

- **2026-08-03** 文档重构完成：研究层文档（RESEARCH_PROGRAM、research/ 六份、status/ 两份、PROTOCOL_INDEX）全部新建；README 导航追加。论点裁决留空，待 S1 与文献复核。证据：本次提交 diff。

- **2026-08-03** 字节级哈希锁废除：按用户决策，SHA 逐字节校验属不必要措施。已移除 `test_delta_zsc_repository.py` 中的 `test_registered_design_document_is_byte_exact` 与注册哈希常量，理论文档恢复为可正常修订（确认轨 commit 冻结后仍不得改方法）。仓库其余 8 项测试全绿。证据：本次提交 diff。

- **2026-08-03** T2/T3/T4 定理形式化完成：TV 精确式、历史样本复杂度、路由紧上界等式写入 FOUNDATIONAL §10，必要性下界修正为标准 Le Cam 二点形式，常数复核与 legacy 草案一致。证据：`docs/theory/DELTA_ZSC_FOUNDATIONAL_THEORY_AND_PROOFS.md` §10，commit `eeca646`。

- **2026-08-03** S2 受控相图验证全过：180 个参数点（服务器执行，commit `4e146bb`）上 T1 链 180/180、T2 精确式 180/180（最大残差 8.9e-16）、T3 边界 180/180，窗口外无净收益成立。理论工具验证通过，生态估计协议解锁。探索轨读数，`scientific_readout_allowed: false`。证据：服务器 `runs/exploration/s2_phase/summary.json`。

- **2026-08-03** S1 面板盘点与 pilot：服务器现存上游只有 SP 与 OP 各 10 个完整 seed（ckpt_final），SA/FCP 在 V6 树上不可用；第一版面板限定 sp/op 两类型，入台账说明。Pilot（3 seed/类型，36 配对 × 5 回合）行数与标记完整，探索性读数 Γ_compat≈69.3，污染对照 identity 均值 146.7 vs run-disjoint 14.0，复现历史污染形态。回合数太少，不构成结论。证据：服务器 `runs/exploration/s1_panel/pilot/summary.json`。

- **2026-08-03** S1 放量预算登记：全量面板取 sp/op 各 10 seed，20×20 配对共 400 格 × 500 回合 × 400 步 = 80,000,000 环境步。证据：本条目。

- **2026-08-03** S1 执行方式改多卡分片：单 GPU 串行方案实测 GPU 利用率近零且只用 1/8 卡，已停止（commit `032f93f` 给面板工具加了分片与合并支持），改为 GPU 2/4/5/6 四分片并行，产物入 `runs/exploration/s1_panel/full/shard_{0..3}`，完成后跑 `--merge-shards` 出最终汇总。证据：服务器 shard 日志与 nvidia-smi 读数。
