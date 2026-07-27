# Path C 项目状态

最后更新：2026-07-27。

## 当前实现

- 方法版本：`path_c_v4_3_executable_response_value_r1`。
- 状态：完整代码修订已形成；尚未在锁定远端环境运行 V4.3 训练或评估。
- 活跃源码：`src/path_c/`。
- OvercookedV2 接入：`experiments/overcooked_v2/official_adapter.py`。
- 唯一入口：`python -m experiments.overcooked_v2.path_c`。
- 布局配置：`path_c_simple.yaml` 与 `path_c_wide.yaml`。
- 十单元正式训练：关闭。

## 上一版本的开发裁决

V4.2 control-memory r1 已完成 42 项远端测试、1,228,800 环境步训练和匹配开发评估。回应使后验发生变化，但 use/mask 策略总变差均值仅 `1.85e-5`，500 个回应屏蔽回合的回应效应、任务成本和净效应均为 0。该结果属于开发诊断，`scientific_readout_allowed: false`。

## V4.3 修订对象

```text
TD-only latent assignment
    → response-conditioned next-action value vectors
    → exact KL-consistent next policies
    → shift-invariant policy-mediated gain
    → supported training behavior
```

具体变化：

- latent responsibility 只读取整回合 TD evidence；
- response/reward/next-Q likelihood 不再参与槽分配；
- outcome head 预测 use/mask belief 下的下一 Q action vector，以及下一 reference logits；
- use/mask 通过同一 KL 执行算子形成下一策略；
- 回应屏蔽触发要求实际执行动作具有正 policy-mediated gain，并达到 next-policy TV floor；
- 训练行为固定加入 0.1 的均匀动作支持；
- KL 温度每个 rollout 通过 24 次 log-space bisection求解；
- response code 使用 centered-advantage delta 的均值与标准差；
- 完整保存每个 epoch/lane/slot 的 responsibility 与全部证据分量。

## 语义主干

```text
layout config + run kind
    → registered budget
    → one training unit
    → one TrainState
    → one run_identity.json
    → one population manifest
    → standard or response-contrast evaluation
```

## 下一步

在新目录完成 V4.3 的远端机械验证，然后重复相同 seed-100、1,228,800 步开发单元。正式训练只在以下条件同时满足后讨论：

1. TD responsibility 可读且不由 response likelihood 主导；
2. predicted next-policy TV 明显高于数值噪声；
3. 高 policy-mediated gain 分箱出现真实动作与奖励差异；
4. `posterior_use` 相对 `prior_only` 的差异到达任务回报层；
5. reference/self-play 能力保持。
