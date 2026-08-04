# EVALUATION_SPEC：DEPI 评估与统计规格

`authoritative: true`

本文件固定性能、组件增量和机制读数的统计边界。实现常量由
[`src/path_c/experiment.py`](../src/path_c/experiment.py) 单点校验；summary 不得替代原始
run/episode artifact。

## 1. 结论分层

- benchmark、Common-Partner、filter architecture、decision supervision、decision-supervision
  cost efficiency、predictive
  calibration、history dependence、source-world context value、recoverable value 和 capacity
  是独立主张。
- 每项只由自己的注册证据决定。任何局部失败不得隐藏 XP，也不得撤销另一项已成立结论。
- `mechanism_claims_unlocked` 仅保留为 aggregate diagnostic，不是论文主张总开关。

测试通过、latent 聚类、内部 Q、entropy、单 seed 或单布局都不是科学结论。

## 2. Official 主评估

- 布局：`test_time_simple`、`test_time_wide`；
- ego：正式 seed index 0–9；
- 每个 ego/partner/layout/role pairing：500 个 400-step episodes；
- 指标：未重塑 simulator return；两种角色均评估；
- root seed 与 key schedule 由正式合同固定并写入 hash；
- raw 节点保存 ego/partner run、layout、role、episode returns、checkpoint/manifest hash 和
  key lineage。

Official summary 只消费完整 raw nodes。缺失、重复、跨布局、跨 checkpoint 或 key 不一致都
fail closed。正式数值失败按原样保留。

## 3. Common-Partner 与 BR-Prox

Common-Partner parents 必须与 support、comparator 和 calibration lineage-disjoint。每布局
覆盖 SP、state-augmented、OP、FCP 四种机制及两个 deterministic Official-plane heuristics，
并在双方角色评估。

BR-Prox 使用同一真实状态的 all-action continuation，报告 empirical action-value regret；
critic、logits 或 entropy 不能替代。

## 4. 正式统计单位

主模式为 independent run。ego-run/partner-run 是推断单位，episode 仅在节点内聚合。正式
scoreboard 使用注册次数的 node bootstrap 和单侧 LCB。superiority 同时要求 LCB 大于零与
注册物质效应；不得 post-hoc 选 seed、伙伴、布局或统计模式。

## 5. 开发矩阵与双重预算

固定 paired seed indexes 0--9，按分层资源设计执行：

```text
K=4 main: R0, B0, B1, B2, R0-extra, B0-extra, B1-extra
K=4 mechanism: deterministic-context, decision-only, Q-only, actor-only,
               no-separation, no-capability, response-only-posterior
K=2/8 sensitivity: B1, B2
```

所有格共享 partner sampler、deployable capacity、训练/评估 key domains、Official 环境和
评估定义。core 四格共享主 PPO transition budget；B2 的 anchor/probe 成本额外列账。
`R0-extra/B0-extra/B1-extra` 不采 anchors，而把 B2 的同一额外 transition 数用于普通 PPO；其 total
training simulator steps 必须与 B2 精确相等。

主嵌套 contrasts 为 `B1-B0` 和 `B2-B1`；`B0-R0` 只比较结构基座与 full-history reference。
必须同时报告 `B2-R0-extra`、`B2-B0-extra`、`B2-B1-extra` 以及 B2 对七项机制消融的 paired
contrast，避免把额外数据成本或较简单机制误写为算法增益。

每个注册 K/variant 格必须由 `evaluate-development-matrix` 生成 raw episode parquet、run identity、
deployment hashes、config fingerprint、key schedule 和 resource ledger。summary 只接受这些
evaluation directories，并从 raw rows 重算 per-seed 双角色 XP 与 99% paired intervals。
自由格式 `depi_development_scores.json`、训练曲线或手写 `xp_mean` 一律无效。

每个 K 的 B2 必须覆盖十个 final-policy component diagnostic artifacts；所有 seed 在同一个由
comparator validation histories 冻结的只读 panel 上重放。summary 在 `[anchor,K,action]` 张量
上重新执行 permutation alignment，报告跨 seed stability，并验证 minimum utilization、
effective count、dominant fraction、usage entropy、pairwise response/action divergence 和 actor
intervention 均来自该共享 panel。

K=4 是主设置。`B1-B0` 下界是否为正只决定 filter architecture claim；`B2-B1` 下界与 final
M1 只决定 decision-supervision claim。B3 不进入矩阵。

## 6. posterior-predictive calibration

每布局使用共享且 lineage-disjoint 的 SP/OP/SA/FCP panel。primary unit 为 partner run，
secondary unit 为 episode；bootstrap 先抽 run，再抽 run 内 episode。

gate 使用 run-level interval，而非聚合点估计：

- model-minus-uniform NLL 的 UCB 不高于 `-0.02`；
- model-minus-no-history NLL 的 UCB 不高于 `-0.02`；
- position/direction 的 90% HPS coverage 95% interval 完整包含于注册 coverage band；
- event Brier 相对注册 prior baseline 的 contrast UCB 不高于零。

event gate 要求总正事件至少 100、每个 partner family 至少 20；不足必须为 `not_estimable`。
另报 event prevalence interval、positive/negative Brier 与 reliability curve。

评分复用 deployment filter/joint likelihood。component index accuracy 与 pooled episode p-value
禁止进入 gate。

## 7. 机制控制

每个布局生成独立 schema-3 raw/summary artifacts：

- **task excess leakage**：按 episode group-held-out CV，比较 learned representation 与固定
  task-state planes 的 balanced accuracy；excess 不得超过注册阈值。
- **protocol-state transplant**：同 checkpoint、source world/partner/role/keys 下替换合法
  recurrent context，报告 ego-run paired drop 及 99% 区间。
- **context sensitivity**：报告原始/交换 `c` 的 action-distribution TV，作描述性分段效应。
- **source-world context value**：在 source all-action returns 上计算
  `(pi_correct-pi_swapped)·G_source`，以 ego-run bootstrap LCB 判门。
- **recoverable value**：G4 名称固定为 fit-selected cross-fitted proxy，不是 oracle upper
  bound；报告 top-action selection stability。只有 `LCB(G4-G2)` 达注册阈值才计算 rho。
- **final M1**：必须使用 final-policy fresh anchors、重新初始化的 bootstrap members 和不重叠
  replica index domains；所有 policy fingerprints 与 deployment 一致。

transplant 只调用 `decision_from_frozen_context(x_s,r_s,u,c)`，不得把 donor recurrent hidden 与
source previous observation/action 拼成 hybrid state。区间同时给出 fixed-panel ego bootstrap 与
ego/partner/donor crossed bootstrap。

G4 偶尔低于 G1 不构成 schema 失败，也不作 pass gate。task representation 在 carry
transplant 下的恒等不变只作描述性/结构测试，不重复算一项经验门。

## 8. 容量、资源与来源闭合

资源报告至少分列：ego PPO、owner initialization、upstream partner、comparator history 与
continuation、reference ego、counterfactual continuation、matched probe、final M1、component
diagnostic、mechanism evaluation、calibration、evaluation、GPU hours、peak memory、inference
latency、deployable 与 training-only parameters。每个方法同时报告 marginal、shared、按实际
reuse count amortized 与 fully-loaded reproduction cost。失败/恢复历史也必须保留。

三个同期适应 proxy 必须在 manifest 级验证与 DEPI 相同的训练 pool、ego interaction budget 和
deployable capacity，并使用同一 Common-Partner panel、角色、episode keys 与 raw return。结果
只能称仓内 proxy 对照，不能外推为对已发表方法的复现或领域 SOTA 比较。

所有 summary 必须验证输入路径与 SHA、method、布局、deployment、config、partner panel、
episode count/roles/keys 和资源账本。formal claim builder 应再次从 raw sources 重算可重算
布尔值；不存在真实 artifact 时，不得以空数组、手写 JSON 或占位 pass 替代。
