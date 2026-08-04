# DEPI 开发矩阵执行规格

本文件是 [`METHOD_SPEC.md`](../METHOD_SPEC.md) 与
[`EVALUATION_SPEC.md`](../EVALUATION_SPEC.md) 的执行说明；注册数字以权威文档、配置和
fail-closed 代码为准。

## 1. 问题

- `B0-R0`：task-only recurrence + instantaneous partner branch 能否与强 full-history RNN
  竞争？该差值不是组件严格增量。
- `B1-B0`：打开 capability/exact response filter 后是否有正增量？
- `B2-B1`：真实 continuation 的 decision supervision 是否有额外正增量？
- `B2-R0/B0/B1-extra`：相同总 simulator cost 下，监督是否优于更多普通 PPO 数据？
- K sensitivity：以上结论是否只依赖一个 latent component count？
- 七项机制消融：deterministic context、decision-only、Q-only、actor-only、no-separation、
  no-capability、response-only-posterior 是否排除了更简单解释？

B3 未实现，不进入任何格。

## 2. 格与共同控制

K=4/seed 执行 R0、B0、B1、B2、R0-extra、B0-extra、B1-extra，以及 deterministic-context、
decision-only、Q-only、actor-only、no-separation、no-capability、response-only-posterior。
K=2/8 只执行 B1/B2 sensitivity。所有格共享 partner sampler、deployable capacity、named key
domains、Official 环境和 evaluator。

core 四格的主 PPO transitions 相同；B2 continuation/probe 额外列账。R0/B0/B1 不采集后
丢弃 anchors。extra 两格用普通 PPO 精确替换 B2 的 auxiliary transition cost，并与 B2 的
total training simulator steps 相等。tail 不足标准 rollout 时使用注册 fixed-shape tail
kernel，不能向上取整成本。

## 3. 训练

先用独立 comparator-fit/validation parents 和固定 Official reference ego checkpoint 运行
`collect-pair-comparator-source`，再以生成的严格 source 运行 `fit-pair-comparator`。随后
`run-development-matrix` 接收 development config、partner manifest、冻结 comparator、固定
seed indexes 0--9 和输出目录。命令按注册分层矩阵生成 config 并调用同一训练
入口。

`development_matrix.json` 每格绑定 run identity、config path/hash/fingerprint、partner pool
hash、resource ledger、PPO/auxiliary/total steps、deployable capacity 和 episode-key domains。
写 artifact 前逐 seed fail-closed 验证双重预算、容量、sampler 和 keys。

## 4. raw evaluator

对每个 K/variant 运行一次 `evaluate-development-matrix`。它加载该格所有 deployment，使用
固定 Official evaluator 对全部有序 policy pairs 运行注册 episodes，并保存：

- `episode_returns.parquet`：left/right seed、两种角色、有序 pairing、episode index、raw
  return、key-schedule hash；
- `development_evaluation.json`：matrix/config/deployment hashes、episode count、schedule、
  raw parquet hash；
- run identity 与 resource ledger。

不得提交自由格式 score JSON。

## 5. summary

`summarize-development-matrix` 只接受完整 evaluation directories。它重新验证 deployment/raw
hash、episode 数、双方角色与 schedule，并从 episode rows 内部重算每 seed XP。对每个 K
生成 `B0-R0`、`B1-B0`、`B2-B1`、`B2-B0`、`B2-R0-extra`、`B2-B0-extra`、`B2-B1-extra` 和
B2 对七项机制消融的 paired 99% interval。B2 的 final component diagnostics 在同一只读
comparator validation history panel 上生成 `[anchor,K,action]` 张量，对每个 K 执行
permutation-aligned cross-seed stability 重算。

formal claim builder 再次回溯这些 raw rows 并重算 summary；手写均值、boolean 或区间不能
解锁任何主张。负结果、跨零、缺格和失败 seed 必须保留。
