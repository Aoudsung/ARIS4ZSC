# Path C 项目状态

最后更新：2026-07-28。

## 当前实现

- 方法版本：`path_c_v4_4_retrace_calibrated_control_r1`。
- 状态：seed-100 开发训练和评估已完成；冻结 checkpoint 的反事实任务价值审计已完成，最终 69 项远端测试全部通过。
- 活跃源码：`src/path_c/`。
- OvercookedV2 接入：`experiments/overcooked_v2/official_adapter.py`。
- 唯一入口：`python -m experiments.overcooked_v2.path_c`。
- 布局配置：`path_c_simple.yaml` 与 `path_c_wide.yaml`，schema version 4。
- 十单元正式训练：关闭。

## 冻结 checkpoint 反事实任务价值审计裁决

V4.4 第 1,228,800 环境步 checkpoint 已完成只读审计。47 个自我配对触发点和 178 个固定
伙伴触发点均使用 128 组共同随机数重放，完整保存 403,200 条 continuation。审计实际执行
101,149,120 个环境步；原训练、评估、固定伙伴产物和 checkpoint 均未改写。

主要结果是：预测下界分数覆盖率只有 29.33%，低于登记的 95%；预测与经验动作排序的平均
Spearman 等级相关系数为 +0.01287；高分一半的经验回应前收益为 −0.01687，低分一半为
+0.00527。自我配对能力仍为 168.48，但当前预测器没有把更有价值的触发状态排到更高位置。

八项正式训练解除条件中，动作价值重复估计、严格正的排序相关、多个固定伙伴出现正的重复
回应前收益以及能力保持四项满足；其中排序相关接近 0，不能解释为强相关。预测下界覆盖率和
高分组收益两项未满足；Other-Play seed 202 的负效应根因及槽语义两项证据不足。正式十单元
训练继续关闭。

完整结果、口径和证据哈希见
[`PATH_C_V4_4_SEED100_DEVELOPMENT_RESULTS.md`](docs/status/PATH_C_V4_4_SEED100_DEVELOPMENT_RESULTS.md#9-冻结-checkpoint-的反事实任务价值审计)。
最终远端清单摘要为 `bf4416fa2a323769882f4ad19a6f9937e627c1e44ba15db97707e756d18d25ac`。

## V4.4 seed-100 开发裁决

V4.4 在 Test Time Simple 完成 1,228,800 个训练环境步、3,072 个训练回合和 96 次更新。四种部署模式各完成 500 个匹配自我配对回合；回应屏蔽完成 500 个匹配回合；四个固定伙伴与四种部署模式共完成 8,000 个回合。全部产物仍为开发诊断，`scientific_readout_allowed: false`。

完整的原始行重算、训练过程、固定伙伴结果和证据哈希见
[`PATH_C_V4_4_SEED100_DEVELOPMENT_RESULTS.md`](docs/status/PATH_C_V4_4_SEED100_DEVELOPMENT_RESULTS.md)。

训练后的自我配对能力保持在参考策略量级：`posterior_use`、`prior_only`、`reference_only` 和 `generic_response_information` 的平均原始回报分别为 168.48、168.76、168.20 和 168.20。

保守下界（lower-confidence bound，LCB）尚未转化为任务收益：

- 500 个回应屏蔽回合中触发 47 个，触发率为 9.4%；
- 47 个触发回合中只有 12 个出现动作轨迹差异，没有回合出现奖励差异；
- 回应收益、任务成本和净收益在全部回合、触发回合及按实际执行动作 LCB 重算的十个等频分箱中均为 0；
- 触发时实际执行动作 LCB 全部严格高于数值容差，范围为 `4.33e-5` 至 `1.05e-2`，因此结果不是由触发条件未执行造成的。

固定伙伴面板中的 `posterior_use - prior_only` 匹配回报差依次为：

- Self-Play seed 101：+2.12，回合级标准误 0.98；
- Self-Play seed 102：+0.12，回合级标准误 0.24；
- Other-Play seed 201：+2.64，回合级标准误 1.73；
- Other-Play seed 202：−2.32，回合级标准误 2.01。

该方向没有在四个伙伴间重复；只有第一个伙伴的描述性常态近似区间不跨 0。一个训练单元和回合级区间不能作为独立训练重复。正式十单元门因此继续关闭。

当前自报回应屏蔽摘要按策略平均 LCB 分箱，而实际触发使用执行动作 LCB。原始行完整保留，本次裁决使用后者独立重算；后续应修正摘要字段名称与分箱输入，但不得追溯改写本次原始产物。

## V4.3 开发裁决

V4.3 完成 49 项远端测试、1,228,800 环境步训练、2,000 个四模式评估回合和 500 组回应屏蔽对照。它修复了“回应只能产生动作无关数值平移”的问题：230/500 个对照回合产生动作轨迹差异，触发点 next-policy TV 均值为 0.0545。

科学目标仍未成立：

- `posterior_use`、`prior_only`、`reference_only`、`generic_response_information` 平均回报分别为 168.24、167.96、168.20、168.44；
- 回应屏蔽效应为 −0.08；
- 498/500 回合最终回报完全相同；
- 2 个回合因回应使用而下降，0 个回合改善；
- 训练伙伴回报未随训练改善。

该结果是开发诊断，`scientific_readout_allowed: false`。

## V4.4 修订对象

```text
full-episode Retrace
    + episode-coherent expert exploration
    + outcome uncertainty calibration
    + LCB policy-mediated trigger
    + fixed-partner development panel
```

具体变化：

- 一步 Bellman target 替换为完整回合 off-policy Retrace；
- 训练时每个回合固定一个 latent expert，用其动作价值构造一致的支持 policy；
- 保留少量均匀 floor，保证六动作均有非零支持；
- use/mask next-Q 均值与方差共同构造保守控制值；
- 回应屏蔽触发与分箱使用实际执行动作的 policy-gain LCB，而非未校准均值；
- 新增单 checkpoint × 固定 SP/OP 伙伴 panel；
- TD-only responsibility、持续 posterior、KL 执行器和完整 E-step 记录保持不变。

## 语义主干

```text
layout config + run kind
    → registered budget
    → one training unit
    → one TrainState
    → one run_identity.json
    → one population or partner panel
    → standard / response-contrast / panel evaluation
```

## 下一步

唯一下一步是修正回应屏蔽摘要的分箱字段，并沿 47 个已触发回合追查为何保守预测仅在 12 个回合改变动作、且没有一次到达奖励。该控制链问题解决前不增加训练 seed 或预算。正式训练只在以下事实同时成立后讨论：

1. Retrace target和重要性比率有限、可读，并能把后段奖励传播到更早状态；
2. coherent exploration覆盖替代动作且不破坏参考能力；
3. 高 policy-gain LCB 分箱中的 realized response effect呈正向校准；
4. 固定伙伴 panel中至少部分伙伴出现 `posterior_use > prior_only`；
5. 回应屏蔽中出现正任务效应，而非仅出现动作差异；
6. reference/self-play能力保持。
