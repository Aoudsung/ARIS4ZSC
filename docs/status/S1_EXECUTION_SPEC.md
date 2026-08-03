# S1 机会面板执行规格（注册版）

状态：探索轨产物，`scientific_readout_allowed: false`。
职责：S1 只读机会面板的执行规格。判决对象是 Θ1（run-disjoint 兼容性机会），定义见 [EXPERIMENT_LADDER](../research/EXPERIMENT_LADDER.md) S1 与 [THESIS_CANDIDATES](../research/THESIS_CANDIDATES.md)。
工具：`experiments/overcooked_v2/panel_opportunity_app.py`。

## 范围与分期

第一版（本规格）：全回合交叉面板。模式库策略作为 agent_0，伙伴策略作为 agent_1，固定角色，完整回合原始回报。直接测得 Γ_compat 的全回合版本与身份重合污染对照。
第二版（另行注册）：共同前缀状态快照加多模式配对延续，测同前缀的 `V_Z − V_fix` 与历史/状态分级参照。第一版通过后才启动。

## 面板维度

- 伙伴类型：SP、OP、SA（State-Augmented）、FCP，全部使用服务器现存官方 checkpoint，不训练任何新策略。
- 每类型独立运行数：以服务器清单为准。执行前在服务器侧盘点各类型现存 checkpoint 目录，把结果写入面板 manifest（每个 entry 的 runs 列表）并入台账。某类型可用 run 数小于 2 时，该类型在伙伴侧与模式侧只能二选一并入台账说明。
- run-disjoint 分配：同一配对中，模式侧 run 与伙伴侧 run 不得共享训练运行 ID；同类型自配对（模式侧与伙伴侧来自同一类型）允许，但必须来自不同 run，并单独打 `same_type_disjoint` 标记。
- 同 checkpoint 自匹配格单独打 `identity_match` 标记，用于复现并量化身份重合污染（对照台账 2026-07-29 的 119.65 条目）。

## 环境与回合协议

环境契约一字不改地使用 Official 协议：view size 2、负奖励、随机初始位置、食谱交付后重采样、交付指示、400 步。回合数默认取正式协议注册值，pilot 可降。episode key 从正式评估根种子派生，同一伙伴在同一模式组内复用相同 key 向量（配对设计）。

## 读数与判决

| 读数 | 定义 | 判决用途 |
|---|---|---|
| 逐格均值 | 每配对每回合原始回报 | 全部后续统计的原始行 |
| V_fix | 跨伙伴取最佳单一模式 run 的均值 | Γ_compat 的被减数 |
| V_Z | 逐伙伴类型取最佳模式后对类型平均 | Γ_compat 的减数来源 |
| Γ_compat | V_Z − V_fix | Θ1 的主读数 |
| 迁移值 | 开发类型上冻结的"类型到最佳模式"映射在留出类型上的均值减 V_fix | Θ1 的迁移条件 |
| 污染对照 | identity_match 格均值减 run-disjoint 格均值 | 复现历史污染效应 |

统计单位是训练运行。区间用运行级重采样，方法沿用正式协议的 bootstrap 纪律（不复制其数字）。

停机判据：Γ_compat 点估计不高于 δ_min 或区间跨零，判 Θ1 无机会，按 EXPERIMENT_LADDER S1 停机条款处理，不启动第二版。

## δ_min 预注册

δ_min 取一次正确交付的原始回报增量，即代码注册的 `OFFICIAL_CORRECT_DELIVERY_REWARD` 常量值，执行时从 `src.path_c.experiment` 读取并记入作业日志。不另行设定。

## 预算纪律

- pilot：每配对回合数降至注册值的百分之一量级，核对行数、配对数与标记完整性。
- pilot 通过后才放量；放量倍数与预计环境步先写入台账。
- 作业 fail-closed：评估器返回非有限值、行数不符或清单校验失败即停机。

## 产物

- 逐回合原始行（parquet）：模式类型、模式 run、伙伴类型、伙伴 run、标记、episode 索引、原始回报。
- 汇总 JSON：逐格均值、Γ_compat、迁移值、污染对照、运行级区间。
- 一页报告入 `docs/status/`，主读数入 [EVIDENCE_LEDGER](../status/EVIDENCE_LEDGER.md)。
