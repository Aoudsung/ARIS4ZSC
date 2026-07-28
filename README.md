# ARIS4ZSC — Path C

当前活动实现是 **Path C V4.4 retrace-calibrated-control r1**。研究问题是：在 OvercookedV2 零样本协作中，伙伴回应能否通过改变可执行的后续动作分布，产生可重复的正任务价值。

方法标识：

```text
path_c_v4_4_retrace_calibrated_control_r1
```

## 1. 当前证据边界

V4.3 的 Test Time Simple 开发单元已完整结束：49 项远端测试通过，训练完成 1,228,800 个环境步、3,072 个完整回合和 96 次更新。V4.3 已使回应能够改变真实动作轨迹，但回应屏蔽效应为负且没有任何回合获得回报改善：500 个对照回合中 230 个出现动作轨迹差异，只有 2 个出现回报差异，二者均为下降。该结果证明“回应可以进入控制”，尚未证明“回应可以改善控制”。全部读数保持 `run_kind: development` 和 `scientific_readout_allowed: false`。

V4.4 是针对该结果的统一根因修订，尚未在锁定远端环境运行。十单元正式训练继续关闭。

## 2. 唯一业务数据流

```text
官方局部观测与循环特征
    → Bellman-trained control memory
    → latent Q experts 与持续 slot posterior
    → response / reward / next-action-value model
    → uncertainty-calibrated J_use / J_mask
    → deployment policy 或训练期 coherent support behavior
    → environment transition
    → full-episode Retrace target
    → TD-only slot responsibility 与 outcome 更新
```

活动模块：

- `src/path_c/experiment.py`：配置、开发/正式预算、训练单元和 population。
- `src/path_c/method.py`：Bayes 更新、value class、KL 策略、保守回应价值、温度求解和码本。
- `src/path_c/model.py`：control memory、独立 Q experts、action-vector outcome model 和回应编码器。
- `src/path_c/training.py`：Retrace target、TD-only responsibility、outcome target、损失、优化器和 target network。
- `src/path_c/runner.py`：环境、主体、伙伴、训练期 coherent support behavior 和完整轨迹。
- `src/path_c/evaluation.py`：标准矩阵、回应屏蔽行结构、验证和统计。
- `src/path_c/storage.py`：单一 `run_identity.json`、Orbax checkpoint 和无损记录。
- `experiments/overcooked_v2/official_adapter.py`：锁定官方网络、训练器和 JaxMARL 环境接口。
- `experiments/overcooked_v2/training_app.py`：一个完整训练用例及每轮 E-step 记录。
- `experiments/overcooked_v2/deployment.py`：训练 checkpoint 到部署 policy 的唯一装配。
- `experiments/overcooked_v2/standard_evaluation_app.py`：10×10 SP/XP 矩阵。
- `experiments/overcooked_v2/response_contrast_app.py`：A1/A2-mask/A2-use 回应屏蔽对照。
- `experiments/overcooked_v2/partner_panel_app.py`：单一训练 ego 对固定官方伙伴的开发 panel。
- `experiments/overcooked_v2/path_c.py`：唯一 CLI 入口。

## 3. V4.4 的统一修订

V4.3 的失败并非后验或执行器没有接线。主要问题是：稀疏任务奖励仍通过一步目标传播，训练期均匀随机支持缺少整回合一致性，且预测均值没有对 outcome 误差进行风险校准。V4.4 同时修订以下对象：

1. **完整回合 Retrace。** Bellman target 使用真实 behavior probability、目标部署 policy 和截断重要性比率，在 400 步完整回合内向前传播稀疏奖励。
2. **整回合一致的专家支持行为。** 每个环境回合固定采样一个 `(estimator, slot)` 假设，训练行为在部署 policy、该专家诱导 policy 和小幅均匀支持之间混合；评估仍严格使用部署 policy。
3. **结果模型的不确定性进入控制。** 下一动作 Q 模型继续输出均值和 log standard deviation；部署、触发和分箱使用登记的 lower-confidence values，而不是只使用未校准均值。
4. **回应屏蔽触发采用 LCB policy gain。** 仅当实际执行动作的平移不变 policy-mediated gain 下置信界为正，并达到 next-policy TV floor 时触发。
5. **固定伙伴开发 panel。** 一个训练 checkpoint 分别与登记的 SP/OP 冻结伙伴运行四种部署模式，区分“自我配对已饱和”和“对外部伙伴仍无控制价值”。
6. **TD-only slot semantics 保持不变。** Response、reward、next-Q 和 next-reference 误差只训练 outcome model并进入审计，不参与 latent responsibility。
7. **完整记录保持不变。** 每个 update、epoch、lane 和 slot 的 responsibility、TD energy、outcome energies、bootstrap availability、重要性比率和 trace coefficient均完整保存。

登记参数位于两份布局配置中：

```text
uncertainty_penalty = 1.0
retrace_lambda = 0.9
importance_ratio_clip = 1.0
behavior_exploration_mix = 0.25
behavior_exploration_temperature = 0.5
behavior_uniform_floor = 0.02
```

## 4. 命令入口

```bash
python -m experiments.overcooked_v2.path_c upstream ...
python -m experiments.overcooked_v2.path_c build-units ...
python -m experiments.overcooked_v2.path_c train ...
python -m experiments.overcooked_v2.path_c build-population ...
python -m experiments.overcooked_v2.path_c evaluate ...
python -m experiments.overcooked_v2.path_c evaluate-panel ...
```

训练入口的 `--run-kind mechanical|development|formal` 选择代码内登记预算：

- mechanical：4 个并行环境、1,600 个环境步、1 个小批次，只用于一次完整更新和恢复检查；
- development：32 environments、1,228,800 environment steps、8 minibatches；
- formal：250 environments、11,000,000 environment steps、50 minibatches。

上游训练和科研评估仍只接受 `development|formal`；微型固定伙伴 panel 可在 `mechanical` 模式下显式指定回合数。

V4.4 改变了配置 schema、模型目标、训练 batch、checkpoint tree 和评估行。V4.3 checkpoint 不得恢复到 V4.4，必须使用新输出目录。

## 5. 实验身份

每个输出目录只有一份 `run_identity.json`。训练身份绑定配置、seed、outer unit、reference 和伙伴支持；评估身份绑定配置、population 和评估 seed。该文件只阻止跨实验混合，不构造额外 registry、canonical projection、内容哈希链或 fallback。

## 6. 验证顺序

```bash
python -m pip install -e .
pytest -q
```

随后在新目录依次完成：

1. 一个完整训练更新及真实 `TrainState` Orbax 保存/恢复；
2. 一个微型固定伙伴 panel；
3. 同预算 seed-100 开发复跑；
4. 固定伙伴 panel 与回应屏蔽校准。

十单元正式训练保持关闭，直至 Retrace credit、LCB policy gain、真实回应屏蔽收益和固定伙伴泛化同时通过开发判据。
