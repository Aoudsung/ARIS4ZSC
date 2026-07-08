# ICLR_UPGRADE_PLAN — 在正确性纪律内提升方法创新性

**Status:** PLAN（方向已由用户指令确立：正确性为前提、创新性为目标、ICLR 为标尺）
**Date:** 2026-07-04 · **Branch:** `codex/rootcause-current-repair-20260702`
**关键窗口声明**：E1 尚未产生任何数据。**现在是方法创新的唯一诚实窗口**——所有方法层改动
在看到决定性结果之前预注册，不构成 post-hoc 漂移；E1 一旦出数，方法层即冻结至读出完成。

---

## 1. 现有工作的贡献力盘点（诚实评估）

| 资产 | 对 ICLR 的贡献力 |
|---|---|
| P1–P5 修复、完整性门、support sidecar、评估基建 | **不可发表**（correctness infrastructure）——但是可复现性/诚实性叙事的支撑，进附录 |
| A1/A2/A7 图因果消融（对 ARIS 控制器） | 二级证据（mechanism ablation），非 headline |
| R2.1 差异化/可容许性证书框架 | 二级贡献（benchmark 构造方法论），可进附录或 workshop |
| 因子-局部信念 + 价值充分支持图 + 纯 Bellman（原提案核心） | **主张尚未测试**；作为方法框架其新颖性中等——需要下述两个升级点把它从"又一个 partner-modeling 变体"提升为有独特科学内容的工作 |

**结论**：不升级就去投，最可能的评审意见是"engineering-heavy, scripted partners,
incremental over opponent-modeling"。升级点必须来自**真实的科学内容**——恰好，我们
自己的失败诊断提供了两个。

## 2. 两个由失败证据自然涌现的创新点

### C1 互斥型协调因子的被动估计不可能性 + 干预式因子发现（源自 D6）

**科学观察（已有实测支持）**：协调中最重要的角色因子（who-serves / who-yields）在功能上
**反相关**——恰恰因为两个体互斥地执行，(ego=ω, partner=ω) 的联合占用质量被交互本身摧毁。
我们的数据：12000 行 replay 中伙伴-终端列联合质量**精确为 0**（S27 修复前）且即便修复后
对角互斥 cell 的质量仍受碰撞率上界约束。

**可形式化命题（草案，需 proof-writer 打磨）**：对互斥因子 ω（同一时刻至多一个 agent 执行），
被动占用加权二阶交互估计的联合支持度 ≤ 碰撞率 × 轨迹长度 → 在任意固定 min_weight 下随
状态空间增大而不可估计；而干预式采样（targeted starts / forced-option rollouts）以预算下界
保证支持度。**"你最想测的协调因子恰是被动数据里最测不到的"**——干净、可辩护、有普适性。

**方法组件**：干预式因子发现（Active Factor Discovery）——
(a) targeted-start 采集：从终端阶段状态初始化（汤已熟）制造双方终端行为的支持度；
(b) forced-option 干预：受控地强制 ego 执行候选 ω，对比不同伙伴行为上下文下的回报差
（`refine_interventional_ce` 已在代码库中存在但未接线——接线 + 元数据化）。
CE 从"被动预处理"升级为"被动筛选 + 干预确证"两段式，支持图的每个因子带支持度证书。

### C2 持久协议模式信念——推断对象从瞬时选项改为持久模式（源自 S27）

**科学观察**：S27 暴露的不只是一个 bug——**瞬时选项是错误的潜变量**。提案自己的理论
（§5.2）说潜变量是因子模式 z_f（episode-持久的协议），而实现推断的是伙伴的瞬时选项再
路由成因子证据。瞬时选项支撑集随状态变化（S27 冻结类 bug 的温床）、生命周期只有几步、
与模式的关系要靠下游 GRU 重新学。

**方法组件**：FactorModeFilter——信念原生地活在因子模式层：可观测行为统计（选项类占用
剖面、事件计数、响应时延）→ 每因子充分统计量 → 模式后验。模式 episode-持久 ⇒ 支撑集
冻结类缺陷**按构造不存在**；与 P4 的持久隐态信念天然对齐；ZSC 故事更连贯（新伙伴 = 已见
模式的新组合，模式层信念组合式迁移）。瞬时选项推断器降级为诊断工具。

**现成的判别基建**：E2（置零 partner_option 通道）已实现——它本来测"推断通道贡献多大"，
在升级后的设计里变成"模式层信念是否不再需要选项标签瓶颈"的直接对照臂。

### C3（辅助，不新增工作）诊断行为涌现
E6（Δ_info vs MI 事后回归）+ 持久模式信念下的探测行为分析——提案 Claim 3 的证据线，
升级后更可信（模式层信念让"探测哪个因子"可解释）。

## 3. 文献定位与查新义务

主要竞品线：OBL（off-belief learning）、FCP/TrajeDi/MEP（population diversity）、
LILI/latent-intention opponent modeling、ToM-style belief agents。**差异化立足点**：
(i) 因子-局部分解 + 价值充分性选择（非 flat latent）；(ii) C1 的不可能性观察 + 干预式发现
（未见直接对应物）；(iii) 单 TD 损失、无辅助目标的信念塑形。
**硬性义务**：story 定稿前必须跑 `/novelty-check`（对 C1/C2 各一次）——若 C1 命题已有
近似先例，立足点收缩到"协调因子发现"的实证域，主张相应降级。**查新先于写作。**

## 4. 分层执行（全部走既有纪律：台账 → codex diff 评审 → 门 → 预注册 → 远程验证）

```
tier-1 正确性（进行中）：
  T1.1 S27 支撑注入修复 + 回归测试            [代码已写, 静态绿; 远程验证排队(等SSH)]
  T1.2 修复后重采 CE (~4h 过夜)               [replay 的 partner dist 已污染, 必须重采]
  T1.3 E1-as-diagnostic 照 sec18.10 跑         [同时给 tier-2 提供 baseline 数据]
tier-2 方法创新（设计 → 评审 → 实现）：
  T2.1 C1: targeted-start 采集 + interventional CE 接线 + 支持度证书    [~2-4 天]
  T2.2 C2: FactorModeFilter 设计规格 → codex 评审 → 实现 + E1 加臂     [~4-6 天]
  T2.3 C1 命题形式化（/proof-writer）+ /novelty-check ×2               [并行, ~2 天]
tier-3 基准强化：
  T3.1 FCP/MEP spike 提前启动（不再等 E1 信号）— ICLR 评审几乎必问 scripted
       partners; K2 风险(基建缺口)提前退役                              [1-2 天 spike]
  T3.2 headline 评估在 FCP 群体上, scripted v2 降为机制分析(G0.1 框架不变)
```

## 5. E1 重构为多臂预注册对比（sec18.6 保持，加臂）

```
臂: base_only / global_gru / flat_factor / aris(option-infer, tier-1修复版)
    / aris(mode-filter, tier-2) / partner_id_q(oracle上界参照)
预注册新增读出:
  aris(mode) > aris(option) > flat   → 模式层信念是真实贡献(C2 支持)
  aris(mode) ≈ aris(option)          → C2 收缩为工程简化, 如实报告
  干预式CE图 vs 被动CE图 (同臂对比)   → C1 的价值直读(因子集差异 + 下游回报差)
```

> **读出口径修正（2026-07-05）**：上块中 "aris(mode) > aris(option) > flat → C2 支持"
> 的措辞已被取代——单一排序结果只读作"候选机制贡献"；C2 headline 需容量匹配归因 +
> 非角色伙伴 + 语义门同时支持。**读出口径的唯一权威来源 = U2_ROLE_BELIEF_PROPOSAL
> （rev3）§5**，签收后随 METHOD_LOCK sec18.14 冻结；本块保留作历史记录，不再更新。

## 6. 时间线（至 ICLR deadline ~9 月下旬）

```
7月上旬   tier-1 收尾 + E1-diagnostic + T2/T3 设计与评审
7月中下旬 T2.1/T2.2 实现落地 + FCP 群体训练启动 + C1 命题成稿
8月       E1 多臂正式跑(scripted v2 机制线 + FCP headline 线) + E2-E7
9月上旬   终表(5 seeds × 2 CE seeds × 50-100ep, CI) + 盲测 split + 写作
9月下旬   提交(先过 /paper-claim-audit + /citation-audit + kill-argument)
预算: 现余 GPU 时充足(至今 <100 正式); FCP 线 400-800 GPU 时在批准预算内
```

## 7. 诚实护栏（创新不豁免任何纪律）

1. 所有 tier-2/3 机制在 E1 出数**之前**预注册（本文档 + METHOD_LOCK sec18.11 即为记录）。
2. 读出仍按预注册表逐字执行；若 mode-filter 臂不敌 option-infer 臂，如实报告并降级 C2。
3. C1 命题若证明不成立/查新撞车，如实收缩——立足点降为实证观察 + 估计器工程。
4. 每个机制单独提交、单独消融（incremental-attributable-commits 纪律）。
5. I10–I17 完整性门、伪影自检清单、Type-B 双签全部不变。
```
