# ARIS4ZSC — Path C

本仓库当前只保留 Path C V4.2：研究伙伴回应能否通过改变后续动作选择，提高 OvercookedV2 中的零样本协作表现。

## 代码结构

代码按真实业务数据流组织，而不是按“防护层”组织：

```text
官方局部观测与循环特征
    → control memory（上一动作、上一团队回报进入循环状态）
    → latent Bellman experts 与 slot posterior
    → response / reward / continuation model
    → KL-regularized execution policy
    → environment rollout
    → Bellman 与 outcome 更新
```

核心文件：

- `src/path_c/experiment.py`：配置、开发/正式预算、训练单元和 population 结构。
- `src/path_c/method.py`：Bayes 更新、价值类、回应价值与执行策略等纯数学函数。
- `src/path_c/model.py`：control memory、Q experts、outcome model 与回应编码器。
- `src/path_c/training.py`：目标、责任度、损失、优化器和 target network 更新。
- `src/path_c/runner.py`：训练时的环境、主体与伙伴状态推进。
- `src/path_c/evaluation.py`：标准矩阵与回应屏蔽的行结构和统计。
- `src/path_c/storage.py`：一个 run identity、Orbax checkpoint 和无损记录。
- `experiments/overcooked_v2/official_adapter.py`：锁定版本官方网络、训练器与环境适配。
- `experiments/overcooked_v2/upstream_app.py`：锁定配方的官方SP/OP训练。
- `experiments/overcooked_v2/training_app.py`：Path C训练用例。
- `experiments/overcooked_v2/deployment.py`：训练checkpoint到可执行policy的唯一装配。
- `experiments/overcooked_v2/standard_evaluation_app.py`：标准10×10 SP/XP评估。
- `experiments/overcooked_v2/response_contrast_app.py`：A1/A2-mask/A2-use回应屏蔽评估。
- `experiments/overcooked_v2/evaluation_app.py`：按population类型选择评估用例。
- `experiments/overcooked_v2/manifest_app.py`：从已完成run目录构造训练单元与population。
- `experiments/overcooked_v2/path_c.py`：唯一 CLI 入口。

## 命令

```bash
python -m experiments.overcooked_v2.path_c upstream ...
python -m experiments.overcooked_v2.path_c train ...
python -m experiments.overcooked_v2.path_c evaluate ...
python -m experiments.overcooked_v2.path_c build-units ...
python -m experiments.overcooked_v2.path_c build-population ...
```

`--run-kind development|formal` 不是标签。它直接选择代码内登记的预算：

- development：32 environments、1,228,800 environment steps、8 minibatches；
- formal：250 environments、11,000,000 environment steps、50 minibatches。

## 实验身份

每个输出目录只有一个 `run_identity.json`。它直接记录配置、seed、outer unit、reference、partner support、population 和方法版本。恢复或续跑只比较这一份事实，不维护额外 registry、canonical projection、哈希链或 fallback。

训练 population 只登记训练 run 目录；reference 从训练 run identity 读取，避免把一个模型与另一个 reference 误配。

## 开发方式

1. 先写清本次修改要改变的可观测行为。
2. 阅读对应完整调用链，再修改唯一事实来源。
3. 修改过程中只运行最接近该行为的测试。
4. 完成一个完整功能切片后，再运行六类集成测试。
5. 任何错误直接暴露；不加入 fallback、静默裁剪、默认替代或 `nan_to_num`。
6. 原始记录完整保存；摘要只能从原始行重算。

当前分支状态仍为 `implemented`。在锁定依赖的远端环境完成模型、训练、环境、恢复和微型评估检查前，不应启动新的开发训练或正式十单元训练。
