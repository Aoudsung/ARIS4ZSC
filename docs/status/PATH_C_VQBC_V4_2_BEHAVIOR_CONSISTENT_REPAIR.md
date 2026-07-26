# Path C VQBC V4.2：行为一致的信念条件反事实价值修复

**日期：** 2026-07-26
**状态：** 静态实现完成；本地纯 JAX、配置和评估合同已测试；远端 Flax/Optax/JaxMARL 训练与同一 seed-100 开发复跑尚未执行
**运行权限：** `scientific_readout_allowed: false`
**模型 schema：** `path_c_model_v4_2`

## 1. 修复对象

V4.1 已经打破无标签价值槽对称性，持久槽后验也会进入正常动作。然而开发证据显示，内部预测的回应净值没有被真实执行器承载：回应使槽后验和绝对预测值明显变化，实际 KL 约束动作分布几乎不变，500 个回应屏蔽回合的最终回报逐行相同。

根因并非单独的 `max` 算子，而是同一反事实在模型、目标和运行器中使用了不同语义：

1. `J_use/J_mask` 以无约束下一动作最大值评价回应，部署却使用冻结参考策略上的 KL 正则化分布；
2. 旧 mask 值同时改变 controller belief 和评价者对真实槽的条件权重，未保持相同物理世界分布；
3. Q 专家只依赖 recurrent history，无法在相同官方历史、不同显式 controller belief 下表示 use/mask continuation；
4. outcome head 回归完整 next-Q 向量，误差主要由动作无关 value baseline 主导；
5. 对照按未执行动作的最大分数触发和分箱，A1 在触发步又吸收了本应屏蔽的回应；
6. 零 posterior 质量槽仍可能被计为活动 quotient；
7. 原始回应码账本只保存终局回报，无法定位差异在哪一层消失。

V4.2 将这些问题作为同一个目标—模型—执行器修复处理，不采用降低 KL、提高预算或强制动作切换的症状性方案。

## 2. 统一反事实对象

设当前 slot belief 为 \(b(m)\)，执行动作 \(a\) 后的回应模型为 \(p_m(y\mid h,a)\)，评价者对真实槽和回应的共同物理权重为

\[
\omega(m,y\mid h,a)=b(m)p_m(y\mid h,a).
\]

回应后 controller belief 为

\[
b^{a,y}(m)=\frac{b(m)p_m(y\mid h,a)}{\sum_jb(j)p_j(y\mid h,a)}.
\]

V4.2 的 use 和 mask 分支始终使用同一个 \(\omega(m,y\mid h,a)\)。二者唯一差异是 continuation controller 接收的 belief：

\[
J^e_{\mathrm{use}}(a)=
\sum_m b(m)r_m(h,a)+
\gamma\sum_{m,y}\omega(m,y\mid h,a)
C^e_m(h,a,y,b^{a,y}),
\]

\[
J^e_{\mathrm{mask}}(a)=
\sum_m b(m)r_m(h,a)+
\gamma\sum_{m,y}\omega(m,y\mid h,a)
C^e_m(h,a,y,b).
\]

其中 \(C^e_m\) 直接预测在真实 KL 约束执行策略下的**原始回报 continuation scalar**，不再预测一个随后由无约束 `max` 处理的 next-Q vector。两个 twin estimator 最后按保守最小值聚合。

执行策略仍为统一策略：

\[
\pi_U(a)\propto \pi_{\mathrm{ref}}(a)
\exp(J_{\mathrm{use}}(a)/\alpha),
\qquad
\pi_M(a)\propto \pi_{\mathrm{ref}}(a)
\exp(J_{\mathrm{mask}}(a)/\alpha).
\]

训练 continuation target 使用 target network 在真实下一 observation、相同 reference logits、相同温度和对应 controller belief 下逐元素重算的这两个分布。数学评分、Bellman target、outcome target 和运行时采样因而使用同一个行为算子。

## 3. 代码修订

### 3.1 Belief-conditioned Q experts

`src/path_c/vqbc/model.py`

- 每个 independent twin dueling expert 显式接收 `stop_gradient(slot_log_belief)`；
- belief 只进入 Q/outcome heads，不作为 auxiliary gradient 进入 recurrent backbone；
- backbone 继续只接收 Bellman/action-value gradient；
- 同一 recurrent feature 可以表示 `Q(z,b_use,a)` 与 `Q(z,b_mask,a)`；
- official critic hidden/value 参数仍机械复制到每个 learned expert，belief projection 零初始化以保留初始任务能力。

### 3.2 行为一致 continuation model

原 `next_q_mean[..., estimator, slot, action, response, next_action]` 已删除。OutcomeModel 现在输出：

- slot-conditioned response distribution；
- slot-conditioned immediate reward distribution；
- `continuation_use_mean/log_std[..., estimator, slot, action, response]`；
- `continuation_mask_mean/log_std[..., estimator, slot, action, response]`。

两种 continuation 使用同一组参数，在 use posterior 和 mask prior 的显式 belief features 上求值。相同参数化保证分支差异只能来自 controller belief，而不是两套任意独立网络。

### 3.3 精确 target runtime policy

`src/path_c/vqbc/training.py`、`objectives.py`

- target Q 在实际下一后验下生成 use policy；
- 同一下一 recurrent feature 在回应前 belief 下重新生成 mask policy；
- raw continuation target 为目标 Q 在对应实际执行分布下的期望，不含 KL 惩罚项；
- Bellman target、continuation target 和部署 policy 共用 `vqbc_forward`/`regularized_policy`；
- 增加 one-response-stale Bellman training，使 mask belief 下的 Q 不是未训练外推；
- E-step 的 outcome evidence同时包含 use/mask continuation likelihood。

### 3.4 共同物理权重

`src/path_c/vqbc/policy.py`

`bellman_control_values` 不再让 mask 分支使用未条件化的世界权重。两个分支均按 `b(m) p_m(y|a)` 积分；`updated_belief` 只作为 use controller 输入。该实现直接对应 A2-use/A2-mask 的因果差异。

### 3.5 随机策略级原始效应

V4.2 显式计算：

\[
\Delta_{\mathrm{response}}^{pred}
=\sum_a\pi_U(a)[J_U(a)-J_M(a)],
\]

\[
\Delta_{\mathrm{cost}}^{pred}
=\sum_a[\pi_M(a)-\pi_U(a)]J_M(a),
\]

\[
\Delta_{\mathrm{net}}^{pred}
=\sum_a\pi_U(a)J_U(a)-\sum_a\pi_M(a)J_M(a).
\]

代码测试验证
`predicted_net = predicted_response - predicted_policy_cost`。
另行记录 KL 正则化目标差，但该字段不代替原始环境回报预测。

### 3.6 Response code 与分支 belief 解耦

response code 是 observable transition 的摘要。V4.2 使用 target network 在 canonical uniform belief 下生成控制签名，再构造仅用于训练 response encoder 与更新 codebook 的未来 code target；相同 `(o,a,o')` 不会仅因 use/mask controller belief 不同而得到不同回应码。已经在 rollout 中实际执行并写入账本的 `batch.response_codes` 是该 transition 的不可变运行时证据，outcome likelihood、stale-belief target 和 responsibility E-step 均读取这些实际码，不会被同一 rollout 后更新的 codebook 重新编码。训练和部署仍共用同一个 response encoder 与 codebook。

### 3.7 Quotient 与 posterior support

- slot-level posterior继续持久保存；
- hard quotient 仍只在 advantage-distance 上置信界满足等价阈值时合并；
- diagnostics只统计 posterior mass 不低于登记 floor 的控制类；
- 零质量或近零质量专家不再被报告为稳定活动 value quotient；
- quotient不参与 posterior回写。

### 3.8 回应屏蔽三分支

`vqbc_response_contrast_runtime.py`

- 触发使用 policy-level predicted raw net effect，而不是 `max_a S(a)`；
- A1 从 mask policy采样当前动作，并在触发步屏蔽回应；
- A2-mask执行与A2-use相同的当前动作，仅屏蔽该回应；
- 后续各分支均按自身持久belief和同一运行时policy继续；
- 原始行记录最大动作分数、实际执行动作分数、policy TV、触发后belief L1、首次动作/观测/回应码/奖励分歧步及分歧计数；
- 等频分箱使用 policy-level raw predicted net effect；
- scale-aware float32 tolerance阻止舍入误差触发。

## 4. 版本与恢复边界

V4.2 有意与 V4.1 不兼容：

- config schema：`path_c_model_v4_2`；
- checkpoint manifest：`path_c_flax_checkpoint_v4`；
- checkpoint metadata：`path_c_model_checkpoint_metadata_v4`；
- train-state hash namespace：`path_c_vqbc_train_state_v4_2`；
- population/evaluation/response-contrast row schemas均升级；
- 配置文件更名为 `path_c_vqbc_v4_2_*`；
- V4/V4.1 checkpoint不能resume或deployment到V4.2。

## 5. 本地验证边界

当前容器具有 JAX、PyYAML 和 pytest，缺少 Flax、Optax 与 JaxMARL。已执行：

- 纯 JAX反事实数学、quotient、policy、配置和评估合同；
- 可独立运行的objective、codebook与minibatch测试；
- 全部V4.2文件bytecode编译和模块导入；
- `git diff --check`；
- 配置解析和旧schema拒绝。

依赖 Flax/Optax 的完整模型参数树、梯度路由、optimizer和checkpoint测试，以及真实JaxMARL rollout，必须在注册远端CUDA环境执行后才能把模块从 `implemented` 改为 `tested`。

## 6. 下一次开发复跑的机械条件

保持相同 seed-100、1,228,800 训练步和500个匹配评估seed。十单元正式训练继续关闭，直到同时满足：

1. target continuation使用的动作概率与runtime同一 `(history, belief, alpha)` 下的概率逐元素一致；
2. use/mask在代码与测试中共享同一物理联合权重；
3. belief变化能在固定recurrent feature下改变Q、continuation和正常动作；
4. policy-level predicted raw effect与完整分支raw-return effect呈单调校准；
5. A1与A2-mask在触发步都真正屏蔽回应；
6. 高预测值样本的use/mask policy TV明显高于V4.1的 `0.000741`；
7. 至少部分匹配回合出现回应导致的动作差异、奖励差异和最终回报差异；
8. responsibility有效质量和posterior-supported quotient count不重新退化；
9. `posterior_use` 相对 `prior_only` 与 `reference_only` 的匹配结果完整报告；
10. 全部产物继续保持 `scientific_readout_allowed: false`。

只有上述开发机制成立后，才重新考虑十单元正式训练。
