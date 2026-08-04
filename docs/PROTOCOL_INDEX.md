# Unified DELTA-ZSC 规范索引

`authoritative: true`

本文件是 `agent/delta-zsc-unified` 分支的唯一规范入口。当前科研方法身份为：

- `METHOD_VERSION = delta_joint_response_decision_bayes_v1`；
- unified config schema = `1`；
- unified checkpoint schema = `1`；
- unified deployment schema = `1`；
- Official OvercookedV2 source commit =
  `5ce1707cf31c1c115e6f6ba96db7bc9cc80a850e`。

## 1. 权威文档

以下六份文档共同定义当前项目，且不存在相互覆盖关系：

1. [`SCIENTIFIC_SPEC.md`](SCIENTIFIC_SPEC.md)
   定义科学问题、合法信息边界、决策等价 estimand、中心命题与可证伪条件。
2. [`METHOD_SPEC.md`](METHOD_SPEC.md)
   定义联合响应—决策潜变量模型、解析行为统计、精确滤波、KL mirror policy、VOI、训练数据流和部署图。
3. [`THEORY.md`](THEORY.md)
   给出模型的概率语义、置换不变性、KL 策略闭式解、VOI 推导、可识别性条件与不保证事项。
4. [`EVALUATION_SPEC.md`](EVALUATION_SPEC.md)
   只注册 H1 性能、H2 decision-emission 增量、H3 belief 因果价值三项论文假设；其余均为诊断。
5. [`FORMAL_EXPERIMENT_PROTOCOL.md`](FORMAL_EXPERIMENT_PROTOCOL.md)
   定义代码冻结、伙伴 lineage、训练、CUDA acceptance、原始数据、统计和资源复现合同。
6. [`MIGRATION_TO_UNIFIED_DELTA.md`](MIGRATION_TO_UNIFIED_DELTA.md)
   列出旧 DEPI 机制的删除、替代、代码迁移与 artifact 不兼容关系。

项目状态只在 [`status/DASHBOARD.md`](status/DASHBOARD.md) 中记录。状态面板不得重写阈值、预算、方法公式或主张。

## 2. 权威代码路径

当前科研方法只由以下路径实现：

```text
src/delta_zsc/
experiments/overcooked_v2/unified_training_app.py
experiments/overcooked_v2/unified_evaluation_app.py
experiments/overcooked_v2/unified_calibration_app.py
experiments/overcooked_v2/unified_causal_app.py
experiments/overcooked_v2/unified_summary_app.py
experiments/overcooked_v2/delta_zsc.py
experiments/overcooked_v2/configs/delta_unified_*.yaml
experiments/overcooked_v2/tests/test_unified_delta_*.py
.github/workflows/delta-unified-ci.yml
```

`src/path_c` 与旧 `experiments/overcooked_v2/path_c.py` 不再定义方法。新代码可以复用其中经过固定 Official commit 验证的环境、伙伴 checkpoint、语义 observation plane、CRN continuation、manifest、资源与存储适配器；任何旧 DEPI actor、belief、comparator、separation、generator、loss routing、claim gate 或配置均为非权威 legacy。

## 3. 单一原理

当前方法只采用一个科学原理：

> 可观测伙伴响应与反事实动作价值是同一潜在协调模式的两类条件观测。训练时二者共同确定模式语义；部署时仅用合法响应滤波，并通过 KL 约束的解析式策略改进将 belief 转化为动作。

该原理导出：

```text
response emission + decision emission
              -> one joint latent likelihood
              -> Bayesian belief
              -> expected decision value and exact myopic VOI
              -> analytic KL mirror policy
```

不得在活动方法中重新引入独立 comparator、separation geometry、learned capability loss、posterior pseudo-label、context dropout、auxiliary actor、information-bonus weight 或 partner generator。

## 4. 方法参数与工程参数

论文方法只有三个选择：

```yaml
method:
  latent_components: K
  continuation_horizon: H
  adaptation_kl_budget: delta
```

- `K` 定义潜变量容量；
- `H` 定义动作决策等价的 continuation estimand；
- `delta` 定义部署适应相对 base policy 的最大 KL。

网络宽度、学习率、rollout 长度、anchor 数、replica 数、bootstrap 次数和 episode 数属于工程或统计设计，不得改写为新的算法组件。

## 5. 版本和不兼容性

旧 DEPI checkpoint、deployment、comparator、anchor replay、calibration、development-matrix 和 formal-claim artifact 与 unified DELTA 不兼容。加载器必须 fail closed，不能自动映射或部分恢复。

统一版本的任何以下变更都要求升级方法或 schema：

- 改变合法信息边界；
- 改变 response/decision emission 的概率分解；
- 改变 transition 或 filter recursion；
- 改变 continuation reward、折扣或 horizon；
- 改变 mirror-policy 约束；
- 改变三项主假设或其统计单位；
- 改变 checkpoint 中影响下一 update 的状态集合。

## 6. 结论边界

代码可导入、单元测试、CPU mechanical run 和 CUDA preflight 只构成工程证据。只有冻结代码、冻结 config、冻结伙伴 lineage 下生成的 raw-backed evaluation 与三假设 summary 才能进入论文结论。

任何失败结果必须保留。不得替换 seed、丢弃伙伴、修改阈值、追加新的 claim、把诊断升级为贡献，或使用旧 DEPI 结果支持 unified DELTA。
