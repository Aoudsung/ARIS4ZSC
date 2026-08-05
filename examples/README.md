# Partner manifest plan 示例

[`partner_manifest_plan.template.json`](partner_manifest_plan.template.json) 只是 schema 形状示例，
不是可直接用于 development 或 formal 的完整伙伴池。它故意只展示少量角色和机制；正式所需
run 数量、stage、width variants、隔离关系和 key 规则由
[`../docs/FORMAL_EXPERIMENT_PROTOCOL.md`](../docs/FORMAL_EXPERIMENT_PROTOCOL.md) 与配置校验定义。

每个 run 条目包含：

- `run_id`：artifact 内唯一标识；
- `role`：`owner_source`、`development_support`、`comparator_fit`、
  `comparator_validation`、`calibration` 或 `confirmatory`；
- `checkpoint`：真实 checkpoint 的绝对路径；
- `parent_training_run_id`：用于验证 parent-run disjointness；
- `generation_mechanism` 与 `hyperparameter_family`：构成静态池的 family；
- `checkpoint_stage`：同 parent run 的训练阶段；
- `seed`、`seed_index`、`jax_prng_key`：可审计随机 lineage；
- `owner_seed_index`：只声明资源归属，不进入 policy；
- `co_training_group_id`、`partner_type_id`：可选 lineage 元数据，同样禁止进入 policy。

把模板复制为计划文件并替换所有 `/ABSOLUTE/PATH/...` 占位符后，应通过
`build-partner-manifest` 读取 upstream run identities、解析真实 checkpoint、计算 SHA-256 并
生成 manifest；随后用 `validate-manifest` 按布局、run kind 和 owner seed 验证。不要手工把模板
改名为正式 manifest，也不要使用跳过 hash 的选项运行 formal 工作流。

greedy courier 与 stationary helper 没有 checkpoint，不写入训练 manifest；它们由评估代码
构造，只属于 Common-Partner 的 test-only heuristic family。
