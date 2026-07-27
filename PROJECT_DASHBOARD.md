# Path C 项目状态

最后更新：2026-07-27。

## 当前实现

- 方法版本：`path_c_v4_2_control_memory_r1`。
- 状态：远端软件验证和一个 Test Time Simple 开发训练单元已经完成；正式训练仍未授权启动。
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

2026-07-27 已在隔离远端目录完成六类测试、官方接口对照、完整环境回合、真实训练更新、
`TrainState` 的 Orbax 保存恢复、微型真实配对以及 seed-100 参考策略对应的完整开发训练与评估。
最终为 42 项测试通过、0 失败、0 错误、0 跳过；开发训练完成 1,228,800 个环境步、
3,072 个完整回合和 96 次更新。

四种部署模式各完成 500 个匹配 seed 的单策略自配对诊断。`posterior_use`、`prior_only`、
`reference_only` 和 `generic_response_information` 的平均原始团队回报依次为 168.28、
168.20、168.20 和 168.20。该诊断不是标准零样本协作（zero-shot coordination）矩阵。
回应屏蔽对照完成 500 个匹配回合；全部触发，但回应使用效应、任务成本和净效应的实际平均值
均为 0。最终 rollout 的 use/mask 策略总变差均值为 `1.85e-5`，回应改变后验却很少改变动作，
所以当前机制仍未通过正式训练前的开发判据。

下一项唯一优先工作是根据完整决策行定位为何 control memory 已接收历史证据但 use/mask 动作
分布仍几乎相同；在该问题解决前不启动十单元正式训练。独立审计位于远端
`.codex_remote_validation/path_c_control_memory_r1_20260727/development_simple_seed100/independent_audit.json`。
