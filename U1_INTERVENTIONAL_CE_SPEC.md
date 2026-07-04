# U1_INTERVENTIONAL_CE_SPEC — 干预式因子发现（设计规格，送 codex 评审）

**Status:** DESIGN — 未实现；**已通过 codex 评审为 GO-WITH-CHANGES**（本轮修订版含所有必需修改）
**Ledger:** U1（预注册 METHOD_LOCK sec18.11；计划 ICLR_UPGRADE_PLAN §2-C1）
**Date:** 2026-07-04

## 0. 动机（实测证据）

被动 CE（均匀随机 ego × 脚本伙伴占用加权二阶交互）对**互斥型协调因子**结构性失明：
- 12000 行 asymm×v2 replay：伙伴-终端列联合支持度**精确 0**（S27 修复前）；
- S27 修复后预计恢复部分列支持，但互斥对角 cell（ego=serve × partner=serve）的联合占用
  仍受碰撞率上界约束——**交互本身摧毁测量交互所需的共现**；
- 随机 ego 的终端占用饥饿独立存在（52/12000 行 ego-plate）。

**C1 命题（形式化义务，/proof-writer 待跑）**：对互斥因子 ω，被动联合支持度
≤ collision_rate × T；干预式采样以预算下界保证支持度。

## 1. 设计总览：两段式"被动筛选 + 干预确证"

```
Phase 2  被动采集(现状,不变) → 支持度审计
Phase 2b 干预 top-up(新):
         对 required-coverage 中 支持度<min_weight 的选项 ω:
           K 个 episode, ego 策略 = 任务推进基线(FSM) + do(ω when valid)
           行打 tag: collection_policy="interventional_topup:opt<ω>"
Phase 3  估计: CE_passive(被动行,现状) + CE_int(top-up 行,分开估计)
Phase 4  建图: 因子候选 = passive-estimable ∪ interventional-estimable
         每因子带 support certificate {source, passive_ws, interventional_ws}
```

**估计量诚实性（核心设计约束，codex 已确认 KEEP SEPARATE）**：干预行改变采样分布 ⇒ CE_int
与 CE_passive 是**不同估计量**，绝不混合平均或 importance-weight 成单估计量（motivating cells
的 passive support≈0 ⇒ positivity 失败、方差无界）。审计 sidecar 增列 `interventional_weight_sum`
/ `passive_weight_sum`；graph.json 每因子记 `estimator_source ∈ {passive, interventional}`
与 `support_certificate`；论文按来源分列报告。可选 overlap-only importance 诊断作补充分析，
不替代双估计量设计。

## 2. 精确改动点

### 2.1 采集侧（`ce_sampler.py`）
- `collect_option_replay(...)` 加参：
  `option_bias: dict | None = None`，形如 `{"target_option_ids": [3,5,8,9], "bias_prob": 1.0,
  "base_policy": "fsm" | "uniform"}`。
  钩子 = 现有 `_sample_valid_option(option_lib, env.state, 0, rng)` 调用点：
  target 中任一在当前 valid 集 → 以 bias_prob 选它（多目标按 valid 顺序首个）；否则
  base_policy（fsm= 复用 `rc2b_layout_scan.fsm` 的任务推进管线，使 ego 能把状态推进到
  终端阶段——解决"随机 ego 连汤都煮不出"的占用饥饿）。
- **`OptionReplayRow` 加字段 `collection_policy: str = "uniform_random"`**（默认不变；
  replay.npz 元数据向后兼容——`_row_from_json_dict`（ce_sampler.py:1199 附近）显式将旧文件
  缺字段读作 `"uniform_random"`；对干预行 tag 为 `"interventional_topup:opt<ω>"`）。
- **P1/P5 边界（codex Q4 修订）**：偏置只作用于 EGO（我方 agent，无 oracle 问题）；伙伴仍是
  黑箱脚本；伙伴权重仍走 S27 修复后的行为推断器（含 codex-[A] 的 config 接线）。
  **明文限制**：FSM 与 targeted-start 代码只可读 `env.state`（公共状态）、`layout` 与
  `option_lib.valid_options`；**禁止**访问 `partner.name / partner.protocol / partner.role /
  partner.terminal_policy` 或任何其派生。**新 I10 gate 项（加入 fidelity gate 工具）**：
  对 `top-up` 相关代码文件 grep 上述禁止 token。

### 2.2 管线侧（`run_ce_pipeline.py`）
- 新 CLI：`--interventional_topup`（默认关）、`--topup_episodes_per_option K=40`、
  `--topup_base_policy fsm`。
- 流程：被动采集 → **`passive_rows = partition_replay_rows(rows, "uniform_random")`** →
  `estimate_empirical_ce_with_support(passive_rows, ...)` → 找
  `required_option_{id,kind}_coverage` 中 `passive_ws < min_weight` 的选项集 Ω⁻ →
  对每 ω∈Ω⁻ 跑 top-up 采集（tag 行）→ **`int_rows = partition_replay_rows(rows, "interventional_topup:*")`** →
  对 int_rows **单独**跑 `estimate_empirical_ce_with_support(int_rows, ...)` 得 CE_int + audit_int
  → 建图（源感知选择，见 §2.3）。
- **`refine_empirical_ce` 与所有 estimator/refine 入口必须显式接受行分区结果**（codex Q1 修订）；
  统一入口 `estimate_empirical_ce_with_support(rows, ...)` 增前置 assert：所有 rows 的
  `collection_policy` 属同一分区（不同分区混合直接 raise）。
- 元数据：`ce_refined.meta.json` 增 `interventional` 块（codex Q2 修订，完整字段）：
  `{topup_options: [...], episodes_per_option, base_policy: "fsm"|"uniform",
    target_option, bias_prob, valid_hit_count, fallback_count, partner_set,
    support_objective, start_state_policy, audit_int}`；sidecar 分列 `passive_weight_sum`
  / `interventional_weight_sum`；replay_metadata 同时存 `passive_row_count` / `int_row_count`。
  目标一致性门（train 侧）校验新块存在性与参数匹配。

### 2.3 建图侧（`graph_builder.py`）— **源感知选择**（codex Q3 修订）
- `_coverage_constrained_pairs` 的候选**不做 flat union**，而是构造成 source-aware pair list：
  `candidates = [(ce, i, j, source) for source in {passive, interventional}]`，按 `-ce` 排序。
- **保持现有 mandatory coverage 排序**（先 required_option_id_coverage → required_option_kind_
  coverage → mandatory_role_contrast → ce_fill）；但在**每个 mandatory 类别里**加 source quotas
  或 tie-breaking，避免大 interventional 候选池吃掉 `max_factors` 预算并挤占 passive 候选
  （"union only widens"在 max_factors 约束下为**假**——codex 明确证伪）。
- 每个 selected factor 记：`estimator_source ∈ {passive, interventional}`（选中来源），
  **同时**记 `passive_ws` 与 `interventional_ws` 两者（哪怕另一源不足 min_weight，也存 ws=0
  或 support_certificate 说明——诚实报告可估性）。
- 覆盖检查语义不变（required option 必须被某候选 touch），错误消息升级为区分
  "被动不可估 + 干预后仍不可估"（真正的 STOP 信号）。
- graph.json 因子 metadata 加 `support_certificate: {passive: {ws, estimable}, interventional: {ws, estimable}}`。

### 2.4 可选二期（本批不实现，仅接口预留）
`refine_interventional_ce(..., intervention_runner=...)` 的 runner 实现（pair 级强制对照）。
top-up 已满足 E1 需要；runner 属 C1 完整故事的加强件，E1 后视需要实施。

## 3. 不变量影响（I1–I17 + 新 I18-CE）
- I4（CE=预处理）**加强**：干预行必须 preprocessing-only；训练/eval 时 `partition_replay_rows`
  在训练读 replay 时若混入 interventional 行 = FAIL（防止 U1 数据泄入 online replay）。
- I10（无 oracle 证据）**加强**：新增 grep tripwire 覆盖 top-up 代码文件。
- I12（支持度 sidecar）**加强**：双审计列（passive + interventional）。
- **新 I18-CE（fidelity gate 工具入门后加入）**：每因子必须有 `estimator_source ∈
  {passive, interventional}`；ce_refined.meta.json 必须含 `interventional` 块（若 topup 启用）
  或 `interventional_topup_enabled: false`（若未启用，此时 interventional_ws=0 all cells，
  由 gate 校验一致性）；混合平均 CE_passive/CE_int 的代码路径（例如未经 partition 的
  `estimate_empirical_ce_with_support(rows_mixed, ...)`）= FAIL。

## 4. 测试计划（write-only，codex 要求扩展）
1. 偏置采样器：target 有效时以 bias_prob 命中；无效时回退 base_policy；tag 正确。
2. 双估计量隔离（**mixed-replay leak test，codex 强制**）：合成 replay 混放（uniform_random +
   interventional_topup:opt3 + interventional_topup:opt5）→ `partition_replay_rows` 返回三个
   独立集；`estimate_empirical_ce_with_support` 拒绝混合输入（raise on cross-partition）。
3. **选择器 budget/crowding test（codex Q3 强制）**：合成一个大 interventional 候选池 +
   有限 passive 候选，验证 mandatory coverage 与 source quota 在 max_factors 约束下不被
   crowding；A1 教训（raw top-K 排挤 serve）不重演。
4. **Metadata parity test**：调用 pipeline，验证 `ce_refined.meta.json.interventional` 块含
   codex Q2 所列全部字段；replay_metadata 有 passive_row_count / int_row_count；graph.json
   每因子 support_certificate 结构完整。
5. 覆盖门语义：仅被动不可估 → 走 top-up 后建图成功；两者都不可估 → 报升级版错误。
6. 向后兼容：无 `--interventional_topup` 时输出与现状 byte-等价（golden）；旧 replay.npz
   缺 `collection_policy` 字段 → 默认 `uniform_random`。
7. **P1/P5 gate（新 I10 项）**：静态检查 top-up 相关代码文件不含 `partner.name /
   partner.protocol / partner.role / partner.terminal_policy` 引用（grep-based tripwire）。

## 5. 预注册读出（并入 sec18 系列，实施前入档 METHOD_LOCK）
| 结果 | 结论 |
|---|---|
| top-up 后终端因子可估且 CE 显著>eta | C1 获得实证支持（被动盲区 + 干预可见）——headline 证据 |
| top-up 后可估但 CE≈0（有支持的测得零） | asymm×v2 终端交互真实微弱——诚实记录，布局降级理由升级为"测得"而非"哨兵" |
| top-up 后仍不可估（支持度仍不足） | 采集预算/偏置设计问题——升级 K、修 FSM 基线，不得下科学结论 |
| E1: interventional-CE 图 vs passive-CE 图 臂对比 | 图差异 + 下游回报差 = C1 的价值直读（ICLR_UPGRADE_PLAN §5） |

## 6. 回滚与提交
- 全部改动 flag-gated（默认关 ⇒ 现状 bit-不变）；一机制一提交：
  ① row-tag + 偏置采样器 ② 管线 top-up 段 ③ 建图 union+certificate ④ 测试。
- 回滚 = 关 flag。

## 7. 请 codex 重点审
1. 双估计量分离是否有泄漏路径（任何把 top-up 行混入 CE_passive 的入口）？
2. FSM 基线作为 ego 采集策略是否引入与训练分布的系统性偏差需要在 meta 中声明？
3. 覆盖门 union 语义是否弱化了 A1 教训（raw top-K 排挤 serve）——union 只扩不缩，应无此险，请证伪。
4. `option_bias` 的 P1/P5 边界论证是否有漏洞。
