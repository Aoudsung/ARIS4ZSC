# E1-rev2 预注册草案 — 群体压力干预（待用户 Type-B 签收后入 METHOD_LOCK sec18.14）

**Date:** 2026-07-05 · **状态：草案，未签收，未运行**
**依据**：sec18.13.1 判决（POLICY-DEFAULT-WAIT）+ 训练伙伴探针矩阵（EXPERIMENT_LOG 2026-07-05）。
**上游机制定式**：ARIS 策略默认让位；信念仅对识别出的分布内伙伴解锁接管；OOD → 默认 → 对
yielder 死锁。修复目标 = **让默认行为变得鲁棒**（群体压力），同时**保住条件化让位**（方法的
真实长处，server-claim 行团队回报证明）。

## 1. 干预内容（单一变量：训练群体组成；其余全部冻结）

train_partners 6 → **8**：现有 6 个不动，**新增注册表中已存在、非 held-out 的两个 yield 变体**
（零新伙伴设计，held-out 定义不动）：
- `bottleneck-yield`（bottleneck_policy=yield）
- `terminal-yield`（bottleneck+terminal 双 yield）

理由（可证伪）：探针矩阵显示系统性薄弱轴 = **bottleneck-中介的 yield 交互**（s2：
near-yield=1.00 vs bottleneck-yield-terminal-yield=0.00）。新增两伙伴把该轴的训练暴露
从 1/6 提到 3/8，同时把"ego 必须接管终端"的episode占比从 3/6 提到 5/8。

**明令冻结**：reward 全部参数（sparse_credit/contrib_scale/shaping/exploration）逐字不动
（sec18.12.4 禁令）；CE 图需按新群体重采重建（S23/I17：CE 只用 train 伙伴 ⇒ 群体变即图变，
这是流程要求非调参）；seeds 0-4、budget 5000 updates、eval 协议与 stage-1 逐字同。

## 2. 臂与规模（最小充分对照）

| 臂 | 目的 | 规模 |
|---|---|---|
| aris_bellman @ pop8 | 干预主体 | 5 seeds |
| base_only @ pop8 | 公平对照（排除"任何臂在 pop8 都会变"混淆） | 5 seeds |

10 train runs（~4-5h, 3 槽）+ CE 重采重建（~4h 过夜档）+ stage-1 协议 eval（~2h，缓存命中）。
global_gru/flat_factor 不重训（stage-1 已证其 pop6 下 held-out ≈1.0，作为固定参照线引用）。

## 3. 预注册读出（sec18.13.2 双行胜利条件为主判据）

| 结果 | 结论 |
|---|---|
| aris@pop8 双行达标：alternate-yield egoCCR>0 于 ≥3/5 seeds **且** server-claim partner 吞吐不降级（相对 pop6 aris） | 群体压力修复默认行为且保住条件化让位 → **角色自适应主张成立**，进入多臂正式跑（8 月线）以 pop8 为正式群体 |
| aris@pop8 yield 行达标但 server-claim 行退化为硬抢（partner 吞吐 ↓） | 修复以牺牲条件化为代价 → 记录"群体压力消除条件化"，方法主张收窄；U2 线检验"结构化信念能否在 pop8 下保住让位" |
| aris@pop8 yield 行仍 ≈0 而 base@pop8 正常 | 群体压力不足以改变 ARIS 默认 → 升级为 FCP/MEP 群体线（T3.2），本臂记诚实负结果 |
| base@pop8 在任一 held-out 行显著劣化 | 干预本身破坏基底可比性 → ARTIFACT-SUSPECT，先过伪影自检再解读任何 aris 行 |
| aris@pop8 两行全 1.0 且 serve share ≈1.0（变成 base 式硬抢） | 不满足胜利条件第二行的精神；如实记录，不宣称修复成功 |

headline 仍为 sec18.6 的 ego_correct_completion_rate；上表为 sec18.13.2 预注册次级判据的
操作化。CI 口径沿用 §9.2（seed 级 bootstrap 95%）。

## 4. 门与纪律

- 全部完整性硬门不变（I10-I17、reward-scale 硬门、evidence 精确匹配门）。
- CE 重建走 run_ce_pipeline 正式路径（顺序采集；LDS-B4 修复应在重建前落地——CE 脚本读
  config 的 sparse_ce_support，窗口二本就排在"CE 重建前"）。
- 编排器用 LDS-C1-safe 模板（成功 marker）；launch 前 codex 过一遍脚本。
- 结果只按本表读出；两行胜利条件不得事后放宽。
- U2/U1 创新轨道不受本实验门控（sec18.11），并行推进；但 U2 的"死锁解药"叙事已按
  zeroed 证据撤回（EXPERIMENT_LOG 2026-07-05），U2 的检验场改为"pop8 下谁更好地利用
  群体多样性"。

## 5. 预算

CE ~4h（过夜）+ train 10 runs ~5h + eval ~2h ⇒ **1 个墙钟天内闭环**。GPU 时 ~15-20。
