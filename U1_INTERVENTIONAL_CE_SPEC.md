# U1_INTERVENTIONAL_CE_SPEC — 干预式因子发现（设计规格，送 codex 评审）

**Status:** DESIGN — 未实现；与 S27 修复 diff 同批送 codex 评审后实施
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

**估计量诚实性（核心设计约束）**：干预行改变采样分布 ⇒ CE_int 与 CE_passive 是**不同
估计量**，绝不混合平均。审计 sidecar 增列 `interventional_weight_sum` / `passive_weight_sum`；
graph.json 每因子记 `estimator_source ∈ {passive, interventional, both}`；论文按来源分列报告。

## 2. 精确改动点

### 2.1 采集侧（`ce_sampler.py`）
- `collect_option_replay(...)` 加参：
  `option_bias: dict | None = None`，形如 `{"target_option_ids": [3,5,8,9], "bias_prob": 1.0,
  "base_policy": "fsm" | "uniform"}`。
  钩子 = 现有 `_sample_valid_option(option_lib, env.state, 0, rng)` 调用点：
  target 中任一在当前 valid 集 → 以 bias_prob 选它（多目标按 valid 顺序首个）；否则
  base_policy（fsm= 复用 `rc2b_layout_scan.fsm` 的任务推进管线，使 ego 能把状态推进到
  终端阶段——解决"随机 ego 连汤都煮不出"的占用饥饿）。
- `OptionReplayRow` 加字段 `collection_policy: str = "uniform_random"`（默认不变；
  replay.npz 元数据向后兼容——旧文件缺字段读作默认）。
- **P1/P5 边界**：偏置只作用于 EGO（我方 agent，无 oracle 问题）；伙伴仍是黑箱脚本；
  伙伴权重仍走 S27 修复后的行为推断器。FSM 只读公共状态。

### 2.2 管线侧（`run_ce_pipeline.py`）
- 新 CLI：`--interventional_topup`（默认关）、`--topup_episodes_per_option K=40`、
  `--topup_base_policy fsm`。
- 流程：被动采集 → `estimate_empirical_ce_with_support` → 找
  `required_option_{id,kind}_coverage` 中 `passive_ws < min_weight` 的选项集 Ω⁻ →
  对每 ω∈Ω⁻ 跑 top-up 采集（tag 行）→ 对 top-up 行**单独**跑
  `estimate_empirical_ce_with_support` 得 CE_int + audit_int → 合并候选（union，源打标）。
- 元数据：`ce_refined.meta.json` 增 `interventional` 块 {topup_options, episodes, base_policy,
  audit_int}; sidecar 存双审计。目标一致性门（train 侧）校验新块存在性与参数匹配。

### 2.3 建图侧（`graph_builder.py`）
- `_coverage_constrained_pairs` 的候选来源从单矩阵改为
  `candidates = pairs(CE_passive, eta) ∪ pairs(CE_int, eta_int)`（eta_int 独立 config，
  默认=eta）；每候选携带 `estimator_source`。
- 覆盖检查语义不变（required option 必须被某候选 touch），但错误消息升级为区分
  "被动不可估 + 干预后仍不可估"（真正的 STOP 信号）。
- graph.json 因子 metadata 加 `support_certificate`。

### 2.4 可选二期（本批不实现，仅接口预留）
`refine_interventional_ce(..., intervention_runner=...)` 的 runner 实现（pair 级强制对照）。
top-up 已满足 E1 需要；runner 属 C1 完整故事的加强件，E1 后视需要实施。

## 3. 不变量影响（I1–I17）
- I4（CE=预处理）不变：干预发生在预处理阶段，训练环内无 CE。
- I10（无 oracle 证据）不变：偏置只碰 ego。
- I12（支持度 sidecar）**加强**：双审计列。
- 新增候选 **I18-CE**（实施后入 gate 工具）：`estimator_source` 必须存在于每因子；
  混合平均 CE_passive/CE_int 的代码路径 = FAIL。

## 4. 测试计划（write-only）
1. 偏置采样器：target 有效时以 bias_prob 命中；无效时回退 base_policy；tag 正确。
2. 双估计量隔离：合成 replay（被动行 + top-up 行混放）→ CE_passive 只用被动行、
   CE_int 只用 top-up 行（行级 tag 过滤正确）。
3. 覆盖门语义：仅被动不可估 → 走 top-up 后建图成功；两者都不可估 → 报升级版错误。
4. 向后兼容：无 `--interventional_topup` 时輸出与现状 byte-等价（golden）。

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
