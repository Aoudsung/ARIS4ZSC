# REVIEW_BRIEF — 全代码库潜伏缺陷扫荡（latent-defect sweep）

**Date:** 2026-07-04 · **Branch:** `codex/rootcause-current-repair-20260702` @ `179307f`
**触发**：ego-delivery≈0 事故链暴露的不是单个 bug，而是**若干缺陷类**（吸收态、配置派生漂移、
接线缺口、哨兵混叠、指标语义……）。在进入决定性 held-out eval 与后续 E2/E3/R2.2/U1/U2 之前，
用 codex 对整个代码库做一次**按缺陷类锚定的深度扫荡**，把同类潜伏缺陷在咬人之前找出来。
**性质**：纯静态评审（read-only）。不触发任何执行；不改方法层（METHOD_LOCK sec18.11 冻结令）。

---

## 1. 事故驱动的缺陷类分类法（每类锚定一个已实测的真实缺陷）

| 类 | 锚定事故（已确认/已修） | 一般化模式 |
|---|---|---|
| **C1 吸收态/不可恢复状态** | S27：推断器乘性 Bayes 更新 `belief∝belief×lik`，精确 0 质量永不复活；`reset()` 冻结 t=0 有效集 → plate/serve 永久不可推断（`930c37b` 修） | 任何迭代更新中某状态一旦到 0/极值即锁死；t=0 计算的 mask/支撑集被全程沿用；EMA/衰减到底后无恢复路径 |
| **C2 配置派生漂移** | role_v1 派生自 asymm.yaml 时连带关掉两个 **P5-clean** 脚手架（terminal_progress_shaping / terminal_exploration）→ E1 ego≈0 真空（sec18.12.1） | 派生 config 静默翻转/丢弃承重键；代码默认值≠正式 config 值；死配置键（定义但从未被读）；键改名后旧键静默失效 |
| **C3 配置接线缺口** | `support_mix` 在 CE 采集路径未接线（5 处构造点用默认值，codex [A]，`c1e112b` 修） | 参数在 config 定义、在某构造点读取，但其他构造点没传；多个工厂函数默认值发散 |
| **C4 哨兵值混叠** | P3：CE 不可估 cell 记 0.0，与"测得为零"不可区分（support sidecar 修） | 0.0/None/-1/NaN 哨兵与合法测量混叠；None→0 静默强转；NaN 被 clip/max 静默吞掉 |
| **C5 指标语义** | F9：train 侧 completion 混计 team 送餐，free-riding 被团队回报掩盖（guard 才暴露） | 指标可被"错误的执行者"满足；分母静默剔除失败 run；train-partner 指标被当泛化读 |
| **C6 实验条件接线** | `relevance_semantics: ego_complement_projection` 进了 E1 config（诊断-only 值），被门拦下（`ee24ccd`） | 臂/消融开关：日志写的是 intent，不是生效值；base_only/global_gru/flat_factor 声称关闭的机制仍有残留路径；train 与 eval 默认值不一致 |
| **C7 信息泄漏** | P1：oracle 伙伴选项真值泄入证据路径；P5：oracle 条件化课程（均已修） | 伙伴特权信息经侧通道到达 ego：shaping 项、探索偏置、guard 反馈、图元数据、obs 特征、partner_id_q 参照臂共享代码路径 |
| **C8 状态残留/重置** | （S27 的近邻：推断器须每 episode 重建才安全） | 对象跨 episode/seed 复用未完全 reset；持久信念跨 episode 泄漏（E3 语义被破坏）；缓存 key 太粗导致跨条件串值 |
| **C9 编排/产物完整性** | 编排器单 `local` 语句吞参 → 输出目录互覆 + cross-wipe 数据丢失（`19b653d`） | 目录碰撞；无属主检查的 rm -rf；非原子写；断点续跑读到 stale 产物；并行 eval 竞态；缓存 key 遗漏影响结果的输入 |
| **C10 数值/单位** | reward-scale 门的由来（历史 reward 缩放不一致）；graph/config 目标元数据漂移两次被门拦 | 窗口 off-by-one；env-step vs option-step 混用；归一化分母错；γ/horizon 跨模块不一致；浮点相等比较 |
| **C11 门/守卫盲区** | objective 门两次正确拦截（sparse_ce_support 漂移）——反问：**哪些承重字段不在门里？** | 门只比对承重字段的子集；守卫存在空洞满足（vacuous pass）路径；门读 logged intent 而非生效对象 |
| **C12 自由狩猎** | — | 分类法之外、但会污染决定性判读或后续阶段的任何缺陷 |

## 2. 三个 pass 的划分（每 pass 一次 `codex exec`，互相独立，可并行）

**全库覆盖承诺**：每个 .py / .yaml 文件被指派给恰好一个 pass（`train_aris.py`/`evaluate_aris.py`
太大且横跨子系统，三个 pass 各用**不同透镜**读它们——透镜在各 prompt 里指明）。

| Pass | 主题 | 缺陷类 | 独占文件（另加两大文件的对应透镜） |
|---|---|---|---|
| **A** | 证据/推断/信念子系统 | C1 C7 C8 (+C12) | option_inferencer, evidence_router, option_executor, partner_option_classifier, event_extractor, option_termination, options, factor_belief, obs_featurizer, obs_encoder, state_utils, specs, replay；tests: s27/provenance/event_extractor |
| **B** | 配置/接线/门/臂 | C2 C3 C6 C11 (+C12) | 全部 9 个 configs（谱系 diff）, partner_pool, sparse_credit, reward_design, provenance, run_ce_pipeline, run_step4_microtrain, `.aris/tools/aris_bellman_fidelity_gate.py`；tests: graph_objective/e2_e3_cache |
| **C** | 指标/数值/CE 估计/产物 | C4 C5 C9 C10 (+C12) | metrics, ce_sampler, graph_builder, td, factor_q, batched_rollout(S7 隔离核查), env_adapter, layout_parser, diagnostics×3, executor_golden_trace, rc* 脚本, run_stability_probe, run_trace_diagnostic, parse_role_v1_v4；tests: sparse_credit/ce_batched/symbolic/toy |

每个 pass 末尾附**事前验尸（pre-mortem）**：逐行走查即将执行的链路（§4），给出
CERTIFIED-CLEAN 或发现清单。

## 3. 运行方式（用户在终端执行；建议 xhigh reasoning）

```bash
cd /Users/aoudsung/Documents/ARIS4ZSC
B=review_bundles/latent_defect_sweep_20260704
codex exec "$(cat $B/PASS_A_PROMPT.txt)" | tee $B/PASS_A_OUTPUT.txt
codex exec "$(cat $B/PASS_B_PROMPT.txt)" | tee $B/PASS_B_OUTPUT.txt
codex exec "$(cat $B/PASS_C_PROMPT.txt)" | tee $B/PASS_C_OUTPUT.txt
```
三个 pass read-only、互相独立，可开三个终端并行；wave 仍在跑也不冲突（本地静态分析）。

## 4. 事前验尸目标（三个 pass 分摊；这是"防止后续流程出错"的直接落点）

1. **stage-1 held-out eval 路径**（最近的下一步）：evaluate_aris 在 `mode=inferred` +
   `require_inferred` + 基线缓存启用 + 固定 eval seed 下的完整代码路径 → Pass A（推断侧）
   + Pass C（缓存/指标侧）。
2. **E2 zeroed 消融 eval 路径**：zeroed 模式下门放行逻辑、`zeroed_count` 记账、
   oracle_source==0 硬约束不被弱化 → Pass A。
3. **E3 belief_persistence off 路径**：off 是否 bit-复现 pre-P4（无残留持久态）→ Pass A。
4. **held-out eval 的 config 派生时刻**（尚未写的 config —— C2 类的高危时点）：
   从 e1rev.yaml 派生 eval 用法时哪些键承重、哪些绝不能带入 → Pass B 产出
   **派生检查单**（这是 Pass B 的必交产物，不是可选项）。
5. **25-run 汇总与 deployable checkpoint 选择**：guard-fail 的 run 在汇总里的处理路径；
   checkpoint_selection 的 eligibility 记录 → Pass C。

## 5. 输出契约（三个 prompt 内嵌同一契约；此处为权威版本）

- 每条发现：`[LDS-<pass><序号>] [severity] [class] file:line` + 六段
  （CODE 原文引用 / DEFECT 机制 / FAILURE SCENARIO 具体状态→错误行为→污染哪个链路步骤 /
  BLAST RADIUS 影响哪些臂・指标・已完成结果 / TEST GAP 哪个现有测试本该抓到、为何没抓到 /
  FIX SKETCH 最小修复方向——**不实现**）。
- Severity 与链路绑定：**[A]** 污染决定性 held-out 判读或静默作废已完成 wave 结果；
  **[B]** 污染后续既定步骤（E2/E3/R2.2/U1/U2/多臂 E1）；**[C]** 潜伏但不在既定路径上。
- 每类必交 **COVERAGE 声明**（读过的文件/函数、跑过的 grep 模式、排除了什么）；
  该类无发现 → 显式 `NONE-FOUND` + coverage，不许沉默跳过。
- 已裁决项不重报：S1–S27、P1–P5、D1–D7、W 系、F8–F13、NEW 系、S7（已知 defer，红线隔离）。
  但**允许**报告"已裁决项的修复不完整/引入新缺陷"。

## 6. 结果回流（评审 → 台账 → 修复，沿用既有纪律）

1. 用户跑完三个 pass → 输出落盘本 bundle。
2. Claude 解析 + 去重 + 逐条裁决：CONFIRMED（进 FINDINGS_LEDGER，分配正式 ID，进修复队列）/
   REFUTED（附反证）/ DEFERRED（记录、排期）。分歧项列给用户裁决。
3. **修复窗口纪律**（关键，防走偏）：
   - **窗口一（eval 前，仅限）**：eval 路径上的 [A] 级"测量污染"修复——即不修会让决定性
     判读本身失真的缺陷。修复走 codex diff 复评 + 保存性检查（"基底修复不得替方法解题"
     纪律），一机制一 commit + 回归测试。
   - **窗口二（判读后）**：其余全部（训练路径、[B]、[C]）。训练路径的 [A] 若成立，意味着
     wave 结果可信度受损——那是**裁决问题**（可能触发重跑），由用户 Type-B 决定，
     不允许静默修完继续用旧结果。
4. TEST GAP 段直接转化为每个 CONFIRMED 修复的回归测试要求。

## 7. 非目标（写给评审者，也写给未来的自己）

- 不评方法好坏、不提方法改进（U1/U2 已有专门轨道）；只找"实现/配置与既定意图不符"。
- 不重开已裁决的设计争论（D 系）；design-limitation ≠ defect。
- 不要求跑任何东西；所有论证基于代码阅读 + 既有产物文本。
