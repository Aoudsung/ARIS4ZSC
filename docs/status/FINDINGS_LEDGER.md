# 结论台账（只追加）

每条：日期、结论、证据指针。结论被推翻时追加新条目，不改旧条目。
旧台账（asymm-role-v2 线及更早）随 commit `6b09cd7` 退休，可从 git 历史取回
（`git show 6b09cd7^:docs/status/FINDINGS_LEDGER.md`）。

---

- **2026-07-09** Link-A 隐藏性证书两轮结构性判负，本方法线在 OvercookedV2
  asymm 基底上死亡；项目转向 Path C。证据：git 历史中的裁决记录（旧台账）。

- **2026-07-14** 自写 torch 训练器同预算只得 19 分（官方 IPPO 复现约 130），
  巩固失败的统一根因是训练栈缺陷；此后正式产物一律用官方训练器。证据：旧台账
  R015 根因裁决。

- **2026-07-28** V4.4 开发训练保持自我配对能力（四模式均约 168），但保守预测
  收益零转化：47 触发 → 12 动作差异 → 0 奖励差异；固定伙伴方向不可重复。证据：
  `docs/status/PATH_C_V4_4_SEED100_DEVELOPMENT_RESULTS.md` §1–8，commit `b360562`。

- **2026-07-29** 冻结 checkpoint 反事实审计（1.01 亿环境步、225 触发态 × 128
  replica）：预测下界覆盖率 29.33%（登记 95%）、预测/经验动作排序 Spearman
  +0.013、高分组经验收益低于低分组。V4.4 预测器不携带真实动作价值排序。证据：
  同报告 §9，commit `594384b`。

- **2026-07-29** 官方策略库 5×4 交叉矩阵：最佳固定策略跨伙伴均值 +19.82
  （SP-100），逐伙伴最优选择均值约 139；V4.4 全系统（约 +21）≈ 其冻结骨干。
  差距是策略模式选择差距，单一固定策略结构上不可跨越。证据：
  `docs/status/PATH_C_V4_4_COUNTERFACTUAL_AND_LIBRARY_MATRIX_REPORT.md`，
  commit `87f680e`。

- **2026-07-29** 代码级根因裁决（九项声明逐条对源码核实）：八个槽是同一共享
  执行策略的八个 critic（`retrace_targets` 单一 `target_execution_probabilities`）；
  outcome 模型以 target 网络自身 Q 为标签（自蒸馏，从未见真实分支回报）；
  "LCB" 是相关误差下的 μ−1σ，非置信下界；责任度是槽对自身 bootstrap 目标的
  自洽性。V4.4 学到的是"同一策略的多个自洽价值估计"，不是"伙伴条件控制价值"。
  修复循环按停机判据关闭，不再产生 V4.5 小修。证据：本台账上两条 + `src/path_c`
  源码（training.py `retrace_targets`/`_branch_targets`/
  `episode_responsibility_evidence`，method.py `bellman_control_values`）。

- **2026-07-29** 研究问题重述：本基底是隐藏离散语境问题——语境为伙伴惯例，
  单步不可读（D1），跨回合表达，回合级价值约 120 分；原提案在动作粒度上定价
  信息，而价值全部位于惯例粒度。下一测量为在线可识别性探针（见
  `PROJECT_DASHBOARD.md` 下一步）。证据：交叉矩阵 + D1 证书 + 审计三者合并推导。

- **2026-07-29** 工作流裁决：确立五阶段研究流水线与文档三层制，取代逐版本修复
  循环。证据：`docs/RESEARCH_PIPELINE.md`。

- **2026-07-29** V4.4 artifact-only 真实回报读出：225 个触发状态的 128 个 replica
  固定拆为 64 fit / 64 evaluation，四折 LOPO 与 10,000 次伙伴分层 bootstrap 得到
  `L_oracle = −0.0135 [−0.1649,+0.1399]`、主历史 `L_probe =
  −0.3597 [−0.9326,+0.0852]`、`tau_response = +0.0499
  [−0.1795,+0.2715]`。三者区间均跨 0，登记裁决为 INCONCLUSIVE：当前注册回应、
  触发与六动作候选分布既无正价值证书，也无限定范围内的无机会证书。V4.4 保持冻结，
  不启动新 RL；该结果不改变 `119.65` 完全来自同 checkpoint 对角格的结论。证据：
  `docs/status/PATH_C_V4_4_ARTIFACT_VALUE_READOUT_REPORT.md`。本条同时取代上文把
  `119.65` 解释为可捕获策略模式差距或惯例粒度价值的两项旧叙述；run-disjoint
  兼容性机会仍未测量。

- **2026-07-30** V4.4 读出完整性复核：正式运行与独立复跑的 9 个确定性产物逐字节
  一致，五个冻结输入及四个实现文件的运行前后 SHA-256 均不变；从逐状态产物独立复算
  伙伴等权点估计和 10,000 次 bootstrap 分位数，与机器摘要完全一致。独立 reviewer
  裁决 PASS，evaluation type 为 `simulation_only`；该结论不扩大证据范围，区间仍条件于
  当前四伙伴面板和固定 continuation 样本。证据：`EXPERIMENT_AUDIT.md`、
  `EXPERIMENT_AUDIT.json`。
