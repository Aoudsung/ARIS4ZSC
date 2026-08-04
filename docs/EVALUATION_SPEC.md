# EVALUATION_SPEC：DEPI 评估与统计规格

`authoritative: true`

本文件固定性能结论、机制归因和统计判定的边界。实现常量和配置字段由
[`src/path_c/experiment.py`](../src/path_c/experiment.py) 单点校验；报告不得以聚合表替代原始
run-level artifact，也不得在看见结果后更换统计单位或阈值。

## 1. 两类结论必须分开

1. **benchmark 结论**：DEPI 在固定 Official protocol 和 Common-Partner panel 上的 held-out
   XP 表现。无论机制门是否通过，该结果都必须完整报告。
2. **机制结论**：性能提升来自合法历史中的 capability/protocol 推断及其决策作用。只有开发
   增量、校准、可识别性、可恢复价值、容量和资源门在两个布局全部通过时才能作此归因。

测试通过、latent 聚类、内部 Q、entropy、单个 seed 或单个布局均不构成科学结论。

## 2. Official 主评估

- 布局：`test_time_simple`、`test_time_wide`；
- ego：10 个正式 seed index，固定为 0–9；
- 每个 ego/partner/layout/role pairing：500 个 400-step episodes；
- 指标：未重塑的 simulator return；双方角色都评估；
- root evaluation seed：0；每个 pairing 的 episode-key schedule 必须写入并校验 hash；
- 原始节点必须保留 ego run、partner run、layout、role、episode returns、checkpoint hash、
  manifest hash 和 key-schedule lineage。

Official summary 只能消费完整的原始节点。缺失、重复、跨布局、跨 checkpoint 或 key schedule
不一致均 fail closed。SP 结果只作为正常 comparator，不得替代任何数值失败的 DEPI seed。

## 3. Common-Partner panel

Common-Partner 是机制族留出测试，不能与训练、comparator fit、posterior calibration 的 parent
runs 重叠。每个布局包含 SP、state-augmented、OP、FCP 四种机制，各至少 4 个独立 partner
runs，并在双方角色上评估；另报告 greedy courier 和 stationary helper 两个 deterministic
Official-plane heuristic。正式支持固定为 10 个独立 ego runs、至少 4 个机制、每机制至少
4 个独立 partner runs。

BR-Prox 使用同一真实状态的 all-action continuation，必须单独报告 empirical action-value
regret；它不能由 critic prediction、actor logits 或 policy entropy替代。

## 4. 正式统计单位与门

主配置固定 `inference_mode: independent_run`。partner-run/ego-run 节点是推断单位，episode
只在节点内聚合；不得把 episode 当独立样本扩大有效样本量。正式 scoreboard 使用 9,999 次
node bootstrap 和单侧 95% 下置信界。

对注册 superiority contrast，同时要求：

```text
one-sided LCB > 0
point estimate >= 20 raw-return points
```

20 分等于一次正确交付的注册物质效应。当前 `minimum_effect_rule` 固定为
`point_estimate`；若未来改用下置信界，必须在看见新结果前同步修改权威规格、配置、测试和
证据账本。任何 post-hoc partner 筛选、seed 替换、布局合并或统计模式切换均无效。

## 5. B0–B2 开发矩阵

开发矩阵对 B0、B1、B2 和 K={2,4,8} 全组合执行。每个 K/seed 内三种方法必须具有相同：

- 静态伙伴 sampler artifact；
- simulator-transition 训练预算；
- deployable parameter capacity；
- 训练与评估随机键域；
- Official 环境、rollout 和评估定义。

同 seed、同 evaluation key schedule 做 paired contrast，使用 9,999 次 seed-block bootstrap
给出 99% 区间。K=4 是正式主设置；机制主张要求 K=4 的 `B1-B0` 和 `B2-B1` 两个区间下界
均大于零。B3 当前为 `not_implemented`，不能进入矩阵或报告为零增益层。

## 6. posterior calibration

每个布局使用与训练隔离的 fresh SP/OP/SA/FCP parent runs：每族 5 个、每 run 64 episodes，
共至少 20 个 run blocks。primary unit 为 partner run，secondary unit 为 episode；9,999 次
bootstrap 必须先按 run、再按 run 内 episode 重采样。

正式 gate 同时要求：

- posterior-predictive joint NLL 分别优于 uniform-mixture 和 no-history baseline 至少
  0.02 nats/step；
- position 与 direction 的 90% highest-probability-set coverage 均在 [0.85, 0.95]；
- event Brier 不高于 prior baseline 的 0.90 倍。

评分必须复用 deployment exact filter 的共享 joint likelihood；component index accuracy、
pooled episode p-value 和只保留位置 coverage 的版本均禁止。

## 7. 机制控制

前四项在每个布局均需独立 schema artifact：

- task leakage：partner-plane 扰动不改变 task pathway，held-out run probe 不得越过注册门；
- history shuffle：同 checkpoint、当前状态、伙伴、role 和 CRN keys 下替换合法 history carry；
- `swap-u` / `swap-c`：task-state matched 的跨-run context swap，使用真实 all-action
  continuation 判定方向；
- recoverable value：同 checkpoint 和 keys 报 G1 legal-history、G2 shuffled-history、
  G3 state-only、G4 oracle-continuation，并在 `G4-G2 >= 20` 的 signal states 上报告恢复率；

M1 则必须在每个 formal ego run 的最后 policy update 之后重新计算，报告
posterior path 与三个独立 bootstrap members 的 tie-aware Spearman、top-action
agreement 和 empirical value regret。该 artifact 的 model fingerprint 必须与导出
deployment 一致，两个布局各 10 个 seed 的路径和 SHA-256 由 DEPI policy manifest
绑定；formal claim builder 必须从 Official raw identity 重新校验这些节点。

这些控制失败时 benchmark 仍需报告，但 formal claim report 必须保持
`mechanism_claims_unlocked=false`。

## 8. 容量、资源与产物

容量控制报告 deployable parameters、training-only parameters 和 comparator 容量；资源报告
至少包含 training simulator steps、anchor continuation steps、evaluation steps、GPU hours、
peak memory 和失败/恢复历史。所有 summary 必须保存输入路径与 SHA-256，并校验 method、布局、
policy manifest、final-checkpoint M1 和 partner manifest lineage。没有真实 artifact 时不得用手写 JSON、空数组、
占位布尔值或文档声明代替。
