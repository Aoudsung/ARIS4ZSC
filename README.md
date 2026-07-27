# ARIS4ZSC — Path C

当前活动实现是 **Path C V4.3 executable-response-value r1**。研究问题是：在 OvercookedV2 的零样本协作中，伙伴回应能否通过改变可执行的后续动作分布产生额外任务价值。

方法标识：

```text
path_c_v4_3_executable_response_value_r1
```

## 1. 代码组织

代码沿唯一业务数据流组织：

```text
官方局部观测与循环特征
    → Bellman-trained control memory
    → latent Q experts 与持续 slot posterior
    → response / reward / next-action-value model
    → 与真实 KL 执行器一致的 J_use / J_mask
    → 训练期 support behavior 或部署 policy
    → environment transition
    → TD-only slot responsibility 与 outcome 更新
```

活动模块：

- `src/path_c/experiment.py`：配置、开发/正式预算、训练单元和 population。
- `src/path_c/method.py`：Bayes 更新、value class、可执行回应价值、KL 策略、温度求解和码本。
- `src/path_c/model.py`：control memory、独立 Q experts、action-vector outcome model 和回应编码器。
- `src/path_c/training.py`：Bellman target、TD-only responsibility、outcome target、损失、优化器和 target network。
- `src/path_c/runner.py`：环境、主体、伙伴、训练 support behavior 和完整轨迹。
- `src/path_c/evaluation.py`：标准矩阵、回应屏蔽行结构、验证和统计。
- `src/path_c/storage.py`：单一 `run_identity.json`、Orbax checkpoint 和无损记录。
- `experiments/overcooked_v2/official_adapter.py`：锁定的官方网络、训练器和 JaxMARL 环境适配。
- `experiments/overcooked_v2/training_app.py`：一个完整训练用例。
- `experiments/overcooked_v2/deployment.py`：训练 checkpoint 到可执行 policy 的唯一装配。
- `experiments/overcooked_v2/standard_evaluation_app.py`：10×10 SP/XP 矩阵。
- `experiments/overcooked_v2/response_contrast_app.py`：A1/A2-mask/A2-use 回应屏蔽对照。
- `experiments/overcooked_v2/path_c.py`：唯一 CLI 入口。

## 2. V4.3 的统一修订

V4.2 control-memory r1 的开发结果显示：回应会改变 slot posterior，但 use/mask 策略总变差均值仅约 `1.85e-5`，500 个回应屏蔽回合没有任务效应。V4.3 同时修复以下闭锁：

1. **slot responsibility 只由整回合 Bellman/TD evidence 决定。** Response、reward 和 outcome likelihood 只训练对应模型并写入审计记录，不再定义 latent slot。
2. **outcome model 预测回应条件的下一动作价值向量。** 不再用两个标量 continuation target 自我确认零差异。
3. **内部规划和实际执行使用同一 KL policy。** use/mask 都先构造回应条件的下一动作分布，再计算原始回报 continuation。
4. **触发使用平移不变的 policy-mediated gain。** 六个动作共同上移不能再伪装成可执行回应价值。
5. **训练行为具有固定动作支持。** 训练时使用 `(1-ε)π + ε Uniform`，登记 `ε=0.1`；评估仍使用原始部署 policy。
6. **KL 温度由 batchwise 二分精确求解。** 删除在 96 次更新中几乎不移动的慢速 dual step。
7. **回应码编码动作相对价值变化。** 目标签名为 current-to-next centered-advantage delta 的均值与离散度，共 `2 × action_count` 维。
8. **责任度与证据分量完整保存。** 每个 update、epoch、environment lane 和 slot 都写入责任、TD energy、outcome energies 和 bootstrap availability。

## 3. 命令入口

```bash
python -m experiments.overcooked_v2.path_c upstream ...
python -m experiments.overcooked_v2.path_c build-units ...
python -m experiments.overcooked_v2.path_c train ...
python -m experiments.overcooked_v2.path_c build-population ...
python -m experiments.overcooked_v2.path_c evaluate ...
```

`--run-kind development|formal` 直接选择代码内登记预算：

- development：32 environments、1,228,800 environment steps、8 minibatches；
- formal：250 environments、11,000,000 environment steps、50 minibatches。

V4.3 修改了模型输出、response code、训练 batch 和 checkpoint tree。旧 V4.2 checkpoint 不得恢复到 V4.3，必须使用新输出目录。

## 4. 实验身份

每个输出目录只有一个 `run_identity.json`。训练身份绑定配置、seed、outer unit、reference 和伙伴支持；评估身份绑定配置、population 和评估 seed。恢复只比较这一份直接可读的事实，不维护第二套 registry、canonical projection、内容哈希链或 fallback。

## 5. 验证顺序

```bash
python -m pip install -e .
pytest -q
```

随后在新目录依次完成：

1. 真实 `TrainState` Orbax 保存与恢复；
2. 一个完整 400 步环境回合；
3. 一次完整训练更新；
4. 一个微型标准配对和一个回应屏蔽配对；
5. 同预算 seed-100 开发复跑。

十单元正式训练保持关闭，直至 TD-only slots、next-policy TV、policy-mediated gain 和真实 A2-use/A2-mask 回报差同时通过开发判据。
