# MIGRATION_TO_UNIFIED_DELTA：从 DEPI 补丁链到统一 DELTA-ZSC

`authoritative: true`

## 1. 迁移动机

旧DEPI的根本问题不是某一个实现bug，而是方法由多轮局部修复累积而成：response prediction、value signature、rank hinge、posterior pseudo-label、actor target、pair comparator、separation geometry、capability semantics、anti-collapse、context dropout和多类gate各自解决一个失败模式，却没有共同概率语义。

这导致：

- 方法配置出现大量loss weights与thresholds；
- 同一latent承担不一致语义；
- gradient ownership需要人工路由；
- 每个新增目标都要求新消融和gate；
- 开发矩阵与资源成本指数增长；
- 方法章节更像审稿意见changelog，而不是从一个原理推导的算法。

Unified DELTA不是`DEPI v9 = v8 + patch`。它以新的method/schema身份彻底替代旧主路径。

## 2. 统一替代原则

旧系统中的所有核心需求由以下单一链路覆盖：

```text
observable response emission
+ privileged decision emission
-> one joint latent likelihood
-> Bayesian belief
-> expected action value + exact myopic VOI
-> analytic KL mirror policy
```

任何旧组件只有在无法由该链路表达时才可能保留。审查结果是：旧训练机制均可删除；只保留固定Official环境、伙伴checkpoint、semantic planes、CRN continuation、manifest、storage与resource adapter作为基础设施。

## 3. 逐项迁移表

| 旧 DEPI 结构 | 根本问题 | Unified DELTA 替代 | 状态 |
|---|---|---|---|
| learned capability GRU | 动态协议/任务进度/身份不可识别 | 四个Beta-Bernoulli合法行为统计 | 删除 |
| capability consistency loss | 零向量退化 | 解析posterior，无训练 | 删除 |
| capability semantic prediction | 与variance/covariance冲突 | 统计量本身即语义 | 删除 |
| capability variance floor | 可由时间噪声满足 | proper Beta posterior | 删除 |
| capability covariance penalty | 强迫真实相关统计解耦 | 不需要 | 删除 |
| fixed sticky `p_stay` | 人工时标，遮挡语义混乱 | learned row-stochastic transition | 删除 |
| evidence-gated identity transition | 丢失negative visibility并混淆missingness/dynamics | factor-wise response mask + 每physical step transition | 删除 |
| channel mean/max response encoder | 无法表示5×5位置 | coordinate-preserving stopped CNN | 替换 |
| zero-component control | OOD干预 | trained base + component residual nested control | 替换 |
| response NLL weight | 与其他目标无统一尺度 | joint marginal likelihood | 删除 |
| twin critic signature loss | 额外Q近似与actor断裂 | probabilistic decision emission | 删除 |
| rank hinge | 人工排序补丁 | decision emission density | 删除 |
| component signature loss | component可合并，需继续补丁 | response+decision共享latent index | 删除 |
| `q^A` pseudo-label | 自指critic责任度 | exact decision-emission likelihood | 删除 |
| posterior-decision KL weight | pseudo-label temperature/weight | joint forward algorithm | 删除 |
| decision actor KL loss | 需要weight/temperature与aux KL | analytic mirror policy | 删除 |
| context dropout | actor可识别dropout，新增regularizer | base/adaptation结构分离 | 删除 |
| auxiliary actor transaction | PPO与aux optimizer/Trust region耦合 | 无适应actor gradient | 删除 |
| gradient routing table | 多目标边界人工维护 | base与latent独立参数树/optimizer | 删除 |
| matched pair comparator | label来自已有returns，重复代理 | decision emission直接消费anchors | 删除 |
| comparator probability thresholds | 未校准、样本少 | 不存在 | 删除 |
| pair separation loss | 几何尺度与task consequence混淆 | joint likelihood自然竞争 | 删除 |
| separation margin | 额外方法参数 | 不存在 | 删除 |
| anchor replay/age/drift weight | continuation label陈旧 | current rollout一次性anchors | 删除 |
| partner generator | 高维OOD code与能力破坏 | 真实frozen broad partner support | 删除 |
| 14个K=4变体 | 每个patch都需消融 | base/response-only/joint/full四层 | 删除 |
| 60-run `p_stay` scan | 扫描人工常量 | transition MLE | 删除 |
| 10类独立claim | 选择性叙事风险 | H1/H2/H3有序层级 | 删除 |
| global mechanism gate | 局部失败锁死所有主张 | 三项独立数值 + hierarchy | 删除 |

## 4. 代码边界

### 4.1 Active method

```text
src/delta_zsc/
  config.py
  types.py
  behavior_statistics.py
  response_model.py
  filtering.py
  model.py
  mirror_policy.py
  voi.py
  training.py
  runner.py
  anchors.py
  deployment.py
```

### 4.2 Active experiment code

```text
experiments/overcooked_v2/
  delta_zsc.py
  unified_training_app.py
  unified_evaluation_app.py
  unified_calibration_app.py
  unified_causal_app.py
  unified_summary_app.py
  configs/delta_unified_*.yaml
  tests/test_unified_delta_*.py
```

### 4.3 Reusable legacy infrastructure

以下 `src/path_c` 功能可被active code调用，但不定义方法：

- Official semantic observation plane parser；
- Official environment adapter；
- frozen partner pool/checkpoint loader；
- CRN counterfactual continuation collector；
- partner manifest与lineage validation；
- parquet/storage/hash helpers；
- resource ledger与CUDA worker validation。

### 4.4 Non-authoritative legacy

以下旧路径即使仍在仓库中，也不得被unified config、training或deployment导入：

-旧 `model.py` / `training.py` 方法图；
- protocol component intervention critic；
- comparator app与anchor matched-pair分类；
- separation terms；
- capability GRU；
-旧 calibration/formal claim builder；
- protocol sensitivity matrix；
- decision coverage proxy；
-旧development matrix；
-旧 `experiments.overcooked_v2.path_c` CLI。

CI通过token gate与import boundary测试保护该边界。

## 5. Artifact不兼容

Unified加载器拒绝：

- DEPI config version 18；
- DEPI checkpoint schema 8；
- DEPI deployment schema 9；
- frozen comparator artifacts；
- DEPI final M1；
-旧 component diagnostics；
-旧 formal claim report；
-旧 training support latents；
-旧 development matrix scores。

迁移不提供参数转换，因为旧actor将latent context作为learned input，而新方法使用base actor加解析mirror adapter；两者参数语义不同。

## 6. 配置迁移

旧：

```yaml
loss_v2:
  signature_weight: ...
  response_weight: ...
  separation_weight: ...
  decision_policy_weight: ...
  capability_*: ...
  component_signature_weight: ...
  posterior_decision_weight: ...
  ...
```

新：

```yaml
method:
  latent_components: 4
  continuation_horizon: 128
  adaptation_kl_budget: 0.04
```

其余section只描述网络、optimizer、数据量和统计设计。

## 7. 实验迁移

旧主矩阵：多个core/extra/mechanism/K/p_stay组合。

新主矩阵：

```text
train: base, response_only, joint
infer: full = joint parameters + analytic VOI
```

只保留：

- H1 full vs strongest external baseline；
- H2 joint vs response-only；
- H3 correct vs shuffled belief；
- full-joint VOI descriptive contrast；
- calibration与resource diagnostics。

## 8. 迁移完成判据

以下全部满足才视为迁移完成：

1. `src/delta_zsc`不含retired token或旧方法import；
2. active configs没有loss weight section；
3. unified checkpoint保存完整runner并通过resume equivalence；
4. base/response-only/joint同seed base parameter fingerprint相同；
5. joint likelihood在真实rollout/anchor上可运行；
6. mirror policy逐状态满足KL；
7. full只复用joint参数，不产生额外训练；
8. CPU tests与CUDA preflight通过；
9.权威文档与代码identity一致；
10. 旧artifact全部fail closed。

## 9. 未来修改规则

发现新失败时，修订顺序固定为：

1. 检查estimand和数据是否合法；
2. 检查joint model是否设定错误；
3. 修改emission/transition probability model；
4. 修改analytic decision operator；
5. 升级method/schema并重新development。

禁止的默认反应：

- 新增loss weight；
- 新增latent geometry penalty；
- 新增hard gate；
- 新增某一failure专用actor/critic；
- 新增不进入deployment概率模型的auxiliary task；
- 在正式结果上post-hoc增加claim。

这一规则用于防止项目重新回到“边发现问题、边加补丁”的工程化循环。
