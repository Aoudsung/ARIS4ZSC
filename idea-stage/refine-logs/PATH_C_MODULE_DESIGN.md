# Path C V4.2 control-memory r1 模块设计

状态：`implemented`。方法标识：`path_c_v4_2_control_memory_r1`。

## 1. 目标

模型只保留一条可执行因果链：

```text
ego history
  -> Bellman-trained control state
  -> latent value hypotheses and slot posterior
  -> behavior-consistent J_use / J_mask
  -> KL-regularized primitive-action policy
  -> environment response
  -> slot-level Bayes update
```

伙伴身份、训练 seed、SP/OP 标签和 checkpoint 编号只允许出现在运行构造记录中，不进入训练张量、Q heads、回应模型、后验或阈值。

## 2. 信息状态

官方 recurrent network接收公开局部观测和回合边界，输出官方循环特征 `f_t`。独立 control memory 在进入其 GRU 前融合：

\[
u_t=\operatorname{LayerNorm}(f_t+E(a_{t-1})+P(r_{t-1})),
\qquad
c_t=\operatorname{GRU}(c_{t-1},u_t).
\]

Q 和 outcome heads 只读取 `c_t`。`E`、`P` 和 control GRU 与官方可训练循环参数一样，仅由 Bellman loss 更新。这样上一动作和上一原始团队回报会进入长期 recurrent state，而不是只作为当前一步旁路。

## 3. 动作价值专家

每个 slot `m` 和 twin estimator `e` 具有独立 dueling head：

\[
Q_{e,m}(c_t,b_t,a)=V_{e,m}(c_t,b_t)+A_{e,m}(c_t,b_t,a)
-\frac{1}{|\mathcal A|}\sum_u A_{e,m}(c_t,b_t,u).
\]

belief 输入停止梯度。固定随机 prior 同样按 estimator 与 slot 独立，且不更新。共享 control representation 只接受 Bellman/action-value 梯度。

## 4. 后验和价值类

slot posterior 按单一 response kernel 更新：

\[
\log b_{t+1}(m)=\log b_t(m)+\log p(y_t\mid c_t,a_t,m)-\log Z.
\]

posterior 始终保存在 slot 层。动态 value class 只用于当前控制统计，不将 class posterior 均匀写回 slot。两个 slots 仅在动作相对价值距离与 twin 不确定性上界共同低于容差时合并。

## 5. 回应和 continuation

回应编码器从 `(o_t,a_t,o_{t+1},d_t)` 生成有限 response code。outcome model在停止梯度的 control feature上预测：

- `p_m(y | c,a)`；
- 即时原始团队回报；
- use controller belief 下的原始 continuation scalar；
- mask controller belief 下的原始 continuation scalar。

use 与 mask 使用相同物理联合权重：

\[
w(m,y\mid c,a)=b(m)p_m(y\mid c,a).
\]

两者只在 continuation controller 接收的 belief 上不同。continuation targets由 target network在部署时同一 KL 执行分布下计算，不使用无约束 `max`。

## 6. 执行策略

\[
\pi(a)\propto \pi_{ref}(a)\exp(J(a)/\alpha).
\]

`posterior_use` 使用持久后验与 `J_use`；`prior_only` 始终使用均匀 slot prior；`reference_only` 直接执行官方策略；`generic_response_information` 使用 generalized Jensen–Shannon response information。两个 KL 温度分别在线更新。

## 7. 训练

整回合 responsibility 使用 Bellman、response、reward 和两种 continuation 证据之和；责任度停止梯度。每个 epoch 重新执行 target-network E-step。bootstrap mask仅作用于独立 experts；target network在每个 minibatch 后 Polyak 更新。

两个优化器的消费者固定：

- Bellman optimizer：官方可训练循环、control memory、learned Q experts；
- outcome optimizer：response/reward/continuation heads和回应编码器。

随机 prior冻结，outcome loss不进入官方循环或control memory。

## 8. 代码所有权

| 文件 | 唯一职责 |
|---|---|
| `src/path_c/experiment.py` | 配置、固定预算、训练单元、population |
| `src/path_c/method.py` | Bayes、value class、J值、KL策略、码本 |
| `src/path_c/model.py` | control memory、Q experts、outcome和response heads |
| `src/path_c/training.py` | targets、responsibility、loss、optimizer、target update |
| `src/path_c/runner.py` | 设备端训练rollout与伙伴状态 |
| `src/path_c/evaluation.py` | 配对、结果行、验证和统计 |
| `src/path_c/storage.py` | 单一run identity、Orbax和无损记录 |
| `official_adapter.py` | 锁定官方软件接口 |
| `training_app.py` | 一个训练用例 |
| `deployment.py` | checkpoint到policy的唯一装配 |
| `standard_evaluation_app.py` | 标准矩阵 |
| `response_contrast_app.py` | 三分支对照 |

## 9. 实验身份

每个输出目录只有一个 `run_identity.json`。训练身份绑定配置、run kind、seed、outer unit、reference和伙伴支持；评估身份绑定配置、population与seed。恢复只比较该对象，不维护额外registry、哈希链或兼容fallback。

## 10. 验收边界

代码通过测试只说明实现接线成立。开发训练用于验证：slot责任度、posterior-to-action敏感性、policy-level预测效应与真实分支效应是否校准。正式十单元训练只有在开发机制读数成立后才启动。
