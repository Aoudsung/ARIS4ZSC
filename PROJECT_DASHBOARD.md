# Path C 项目状态

最后更新：2026-07-28。

## 当前实现

- 方法版本：`path_c_v4_4_retrace_calibrated_control_r1`。
- 状态：完整静态修订已形成；尚未在锁定远端环境运行 V4.4 训练或评估。
- 活跃源码：`src/path_c/`。
- OvercookedV2 接入：`experiments/overcooked_v2/official_adapter.py`。
- 唯一入口：`python -m experiments.overcooked_v2.path_c`。
- 布局配置：`path_c_simple.yaml` 与 `path_c_wide.yaml`，schema version 4。
- 十单元正式训练：关闭。

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

在新目录完成 V4.4 的远端机械验证，然后重复相同 seed-100、1,228,800 步开发单元。正式训练只在以下事实同时成立后讨论：

1. Retrace target和重要性比率有限、可读，并能把后段奖励传播到更早状态；
2. coherent exploration覆盖替代动作且不破坏参考能力；
3. 高 policy-gain LCB 分箱中的 realized response effect呈正向校准；
4. 固定伙伴 panel中至少部分伙伴出现 `posterior_use > prior_only`；
5. 回应屏蔽中出现正任务效应，而非仅出现动作差异；
6. reference/self-play能力保持。
