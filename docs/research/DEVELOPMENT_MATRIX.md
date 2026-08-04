# B0–B2 开发矩阵执行规格

本文件是 [`METHOD_SPEC.md`](../METHOD_SPEC.md) 与
[`EVALUATION_SPEC.md`](../EVALUATION_SPEC.md) 的执行说明，不复制其中注册的版本、seed、预算、
K 集合或统计次数。精确数值只由权威文档、development config 和 CLI fail-closed 校验定义。

## 1. 要回答的问题

- B1−B0：结构化 task/protocol 隔离与 exact response filter 是否带来正的 held-out XP 增量？
- B2−B1：真实 continuation 的 decision supervision 是否带来额外正增量？
- component-count sensitivity：上述结论是否只依赖单一 latent 容量？

B3 不是可执行层。`build_b3_active_voi` 必须抛错，matrix artifact 必须写
`b3_status: not_implemented`。

## 2. 共同控制

每个 component count 和 seed 内，B0/B1/B2 必须共享：

- 同一个 source config、partner manifest 和 family-uniform sampler artifact；
- 同一训练 simulator-transition 预算和 anchor 采集预算；
- 同一 named random-key domains 与后续 evaluation episode-key schedule；
- 相同 deployable parameter capacity；B0/B1 的不用模块保留为 inert parameters；
- 相同 Official environment、rollout、checkpoint 和 evaluation procedure。

runner 在写 matrix artifact 前逐项比较 hash、资源账本与容量；任一不匹配立即失败。

## 3. 执行

使用 `run-development-matrix`，提供 development config、经校验的 partner manifest、唯一且
非空的 seed indexes 和输出目录。若不显式给 component counts，命令运行权威方法规格中注册的
完整 sensitivity set；若显式给出，必须恰好覆盖该集合一次。

命令为每一格生成只改变 `method_variant` 与 `protocol_components` 的 config，调用同一训练入口，
并输出 `development_matrix.json`。每格必须有 run identity、config fingerprint、partner-pool
hash、resource ledger、deployable capacity 和 episode-key domains。

## 4. score artifact

训练结束后的 held-out XP 评估写一个独立 `depi_development_scores` artifact。每行只含：

```text
variant
protocol_components
seed_index
xp_mean
evaluation_key_schedule_sha256
```

行集合必须与 matrix 全覆盖且无重复；同一 component count/seed 的三种 variant 必须共享
evaluation-key hash。不得从训练曲线、best checkpoint 或不同 episode schedules 拼接 score。

## 5. paired summary 与裁决

`summarize-development-matrix` 对每个 component count 生成 B1−B0、B2−B1、B2−B0 的 seed-level
paired increments，并按 [`EVALUATION_SPEC.md`](../EVALUATION_SPEC.md) 的注册 bootstrap 规则
给出区间。正式主 component count 的两个逐级增量区间下界都为正，才解锁开发增量门。

summary 同时报告 per-seed XP、每 variant 总训练 transitions、GPU hours 和所有输入 SHA-256。
负结果、区间跨零、缺格或 key mismatch 必须保留；不得新增 seed、换 component count 或改成
非配对统计来修复结论。
