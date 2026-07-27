# Path C 项目状态

最后更新：2026-07-27。

## 当前实现

- 方法版本：`path_c_v4_2_control_memory_r1`。
- 状态：`implemented`；本代码包未在当前环境执行远端依赖测试、训练或评估。
- 活跃源码：`src/path_c/`。
- OvercookedV2 接入：`experiments/overcooked_v2/official_adapter.py`。
- 唯一入口：`python -m experiments.overcooked_v2.path_c`。
- 布局配置：`path_c_simple.yaml` 与 `path_c_wide.yaml`。

## 语义主干

```text
layout config + run kind
    -> one registered budget
    -> one training unit
    -> one TrainState
    -> one run_identity.json
    -> one population manifest
    -> standard or response-contrast evaluation
```

`run_identity.json` 只防止真实的跨实验混用：训练恢复时绑定配置、seed、outer unit、reference 与伙伴支持；评估时绑定配置、population 与评估 seed。仓库不维护第二套 registry、canonical projection、内容哈希链或 fallback。

## 当前模型

- 冻结官方策略作为动作先验；可训练官方循环参数只接收 Bellman 梯度。
- 独立 control memory 在 GRU 前融合上一 ego 动作与上一原始团队回报，使这些证据能够跨多步传播。
- 无标签 latent Bellman experts、持久 slot posterior、动态动作价值等价类。
- 单一回应模型、即时回报模型和 behavior-consistent use/mask continuation。
- 实际执行使用相对官方策略的 KL 正则分布。
- 标准矩阵与 A1/A2-mask/A2-use 回应屏蔽对照分开报告。

## 运行边界

- `development`：32 environments、1,228,800 环境步、8 minibatches。
- `formal`：250 environments、11,000,000 环境步、50 minibatches。
- 两种预算由 `src/path_c/experiment.py` 固定，不能仅通过 CLI 标签伪造正式运行。
- 旧 checkpoint 与旧评估目录不得恢复到该方法版本。

## 下一步

在锁定依赖的远端新目录依次完成：六类测试、官方接口对照、一个完整环境回合、一次训练更新、真实 `TrainState` Orbax 保存恢复、一个微型标准配对和一个回应屏蔽配对。机械接线全部通过后，才决定是否重复 seed-100 开发训练。
