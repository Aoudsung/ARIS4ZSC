# METHOD_SPEC：DEPI 可执行方法规格

`authoritative: true`

当前身份：`depi_exact_filter_decision_supervision_v5`，配置版本 15，checkpoint schema 5，
deployment bundle schema 6。本文件描述代码中唯一有效的方法；旧 artifact 一律 fail closed。

## 1. B0–B2 共同模型骨架

所有层级使用同一参数容量和同一个 actor/critic：

- task encoder：Official CNN trunk + GRU-128；
- capability encoder：GRU-64，输出 16 维 `u_t`；
- K 个 16 维可学习 component embeddings `m_k`；
- universal actor：输入 `(x_t, concat(u_t,c_t))`；
- dueling critic：同一输入，输出 `V,Q1,Q2`；
- response decoder：按 component 输出联合 response likelihood。

B0 保留完整当前观测进入 task GRU，同时将 actor/critic 的 context 数值清零；其余 context
与 decoder 参数仍保留为容量匹配的 inert 参数。B1/B2 在 task GRU 输入前调用固定 Official
semantic-plane extractor，将 other-agent position、direction 和 inventory planes 全部清零。
代码对不符合固定 Official channel contract 的观测 fail closed。

PolicyState 的完整字段为：`task_carry`、`capability_carry`、`protocol_carry`、
`context_summary`、`previous_observation`、`previous_action`、`episode_start`。
`protocol_carry` 直接保存 `pi_t`，不存在协议 hidden size 或 recognition-network carry。

## 2. capability 与 protocol 时标

CapabilityEncoder 每步从

```text
e_t = (o_t - o_{t-1}, embed(a^ego_{t-1}), episode_start)
```

更新内部 hidden，但发布给 actor/critic 的 `u_t` 仅每 16 个 episode-local steps 更新；
episode start 清零 hidden、published value 和计数。相隔 16 步且属于同 episode 的发布值施加
一致性损失。`evaluate-identifiability` 分别执行 `swap-u` 和 `swap-c` 的真实 continuation
干预，防止把两条路径的语义只留在命名层。

协议状态每步更新。正式 K=4；开发矩阵必须同时报告 K=2、4、8。component 可交换，不得
固定映射到伙伴算法族。Hungarian alignment 仅可用于冻结后的可视化，不能进入训练、
calibration gate 或正式 component accuracy。

## 3. 精确 sticky filter

令 `T[i,j]=P(z_t=j|z_{t-1}=i)`。正式 K=4 时对角为 0.9，非对角为 `0.1/3`；
其他开发 K 使用 `(1-0.9)/(K-1)`。唯一 predictive 实现为：

```text
pi_bar_t = pi_{t-1} @ T
pi_t = softmax(log(pi_bar_t) + ell_t)
```

其中 `ell_t,k` 是共享 response decoder 对刚观测到的合法 response `y_t` 给出的
`log p(y_t | z_t=k, sg(frame_{t-1}), sg(x_{t-1}), u_{t-1}, a^ego_{t-1})`。
transition 只应用一次；更新后不以额外 softmax 修复错误概率质量。episode start 固定为
uniform prior 并忽略跨 episode emission。

训练与 deployment 调用同一个 `sticky_transition_matrix`、
`exact_bayes_filter_step` 和 `component_joint_log_probability`。

## 4. 联合 response likelihood

response target 包括 visibility、`position | visible`、`direction | visible`、可见 inventory
因子和 observation-level event。每个 head 都保留同一个 component 轴 K：

```text
log p_k(y) = log p_k(visibility)
           + visible * [log p_k(position) + log p_k(direction)
                        + sum_f log p_k(inventory_f)]
           + event_valid * log p_k(event)

log p(y|H,a) = logsumexp_k(log pi_k + log p_k(y))
```

只允许一次 component marginalization。position 在不可见时不计分，避免与 visibility
重复计数。运动学 trunk 读 `sg(frame_t),m_k,u_t,a_t`；event trunk 读
`sg(x_t),m_k,u_t,a_t`。可声明 event 仅为可见性改变或双方帧均可见时的 inventory 改变。

## 5. 训练目标与参数所有权

B2 的单次 combined transaction 为：

```text
L = L_PPO + 1.0 * L_decision + 1.0 * L_response
          + 0.1 * L_separation + 0.01 * L_capability_consistency

L_decision = L_Q + 0.25 * KL(p_A || policy)
p_A = softmax(stop_gradient(A) / 1.0)
```

`L_Q` 用 conservative centered action values 拟合 empirical continuation signature，包含
Huber fit 与注册 gap/margin 的排序 hinge。B1 只启用 PPO、response 和 capability
consistency；B0 只启用 PPO。三者都收集同预算 anchor continuation，只有 B2 使用监督。

所有项在当前参数上进入同一个 `jax.value_and_grad` 和一个 optimizer transaction。
每个 loss 获得数值相同、但对非所有者叶子 stop-gradient 的 parameter view：

| 参数子树 | 可更新目标 |
|---|---|
| task encoder | PPO、Q signature、actor decision、owner distillation |
| capability encoder | PPO、response、signature、decision、separation |
| component embeddings | PPO、response、signature、decision、separation |
| actor | PPO、actor decision、owner distillation |
| critic | PPO、Q signature |
| response decoder | response NLL |

owner initialization 只能改变 task encoder 与 actor，并必须通过“参数实际变化且 KL 降低”
测试。不存在独立 response/critic optimizer 或滞后 snapshot gradient。

同一 anchor payload 每个 outer update 只在第一个 optimizer transaction 使用一次；后续
minibatches 只运行 PPO/response。metrics 必须报告 anchor effective sample size、最大使用
次数、辅助梯度范数和实际总权重。

每个 optimizer update 后用 candidate params 在 rollout 上完整重放 `(x,u,c)`，计算行为
log-prob 与更新后 log-prob 的 KL。超过 0.04 只产生 `kl_early_stop` 并停止余下 minibatches；
只有 NaN/Inf 设置 `nonfinite_failure`。下一 outer update 的 epoch 数减一、下限二，干净
update 逐次恢复；有效 epoch 数属于 checkpoint state。

## 6. 合法 anchor 与 comparator

注册 anchor trigger 每 16 outer updates 发生一次。每次采集 32 个普通真实快照和 16 对
matched histories；每条 all-action 标签使用 6 个动作、4 个 fit replicas、8 个独立
evaluation replicas、128 步 continuation。matched pair 先在 task-feature 空间从不同
partner run 的真实快照中最近邻匹配，再各自运行 16 步合法 probe。环境、ego carry、伙伴
真实 carry 和 episode 状态均保留；禁止中途 reset 或 carry 清零。

comparator 的输入是 probe 中可复算的 visibility、relative position、direction、inventory、
observable change 与 ego-action sequence；不含伙伴动作或 run identity。label 来自 empirical
continuation-signature prototypes 的 same/different regime，而不是 run ID。首次 anchor 的
comparator fit、validation 和 DEPI supervision partner runs 三者互斥；模型随后冻结并进入
checkpoint。validation accuracy 以 partner run 为 bootstrap block 报区间。

manifest schema 3 使用独立 `development_support`、`comparator_fit` 和
`comparator_validation` roles；三者的 parent run、checkpoint 与 co-training group 必须两两
不相交。首次 comparator 冻结前，向量环境按 lane index 固定保留 `6/8` support、`1/8` fit、
`1/8` validation lanes。只有 support lanes 的 `ppo_mask=1`；fit/validation lanes 只产生独立
comparator development evidence。comparator 冻结后，它们只在 episode boundary 合法切回
support pool，禁止 mid-episode 换伙伴。普通 anchor 与后续 separation pairs 也只从 support
lanes 采集。

统一类别顺序为 `equivalent, distinct, ambiguous`：

- distinct probability `<=0.55` 且 signature distance `<=1.0`：equivalent；
- distinct probability `>0.70` 且 signature distance `>1.0`：distinct；
- 其余：ambiguous，只记账、不进入 loss。

提前 done、padding 或与 comparator development runs 重叠的 pair 均 `pair_valid=false`，权重
必须为零。无法形成固定数量的 run-disjoint pairs 时 fail closed。

## 7. M1 独立价值诊断

M1 使用三个不同初始化、各自 Adam optimizer 和 bootstrap sampling counter 的 value-signature
readouts。输入包含 anchor 的完整合法 PolicyState carry 与当前观测；fit replicas 训练成员，
evaluation replicas 才用于判定。排序采用 tie-aware Spearman，并同时报告 top-action agreement
与 empirical value regret。

posterior path 和每个 member path 都要求至少 90% anchors 的 Spearman `>=0.8`。M1 是
report-only final-mechanism condition：失败不得中断训练、替换 checkpoint 或隐藏 XP。成员
params、optimizer states、counters、summary 和固定形状 history 均进入 TrainState。

## 8. posterior calibration

独立命令 `calibrate-posterior` 只写 `posterior_calibration.json`，不产生部署 wrapper。
正式校准使用每个 SP/OP/SA/FCP 五个 fresh parent runs、每 run 64 episodes，key domain 与训练
分离并使用注册 offset。primary unit 是 partner run，secondary unit 是 episode；gate 使用
两级 bootstrap 的 run mean，pooled score 仅作描述。

主分数是共享 joint likelihood 的 posterior-predictive NLL；次分数是 event Brier；position
和 direction 都使用 posterior-predictive 90% highest-probability sets 并同时进入 gate。
no-history baseline 保留当前物理 task/frame，固定 uniform posterior、清零 capability 并使用
uniform mixture embedding。component index 不参与评分。

通过条件：模型 NLL 分别优于 uniform mixture 和 no-history baseline 至少 0.02 nats/step；
两项 coverage 都落在 `[0.85,0.95]`；event Brier 不超过 prior baseline 的 0.9 倍。

## 9. 静态伙伴池

正式 B0–B2 不构造生成式伙伴、其 optimizer、archive 或采样状态。唯一构造器
`build_partner_pool(config,manifest,split)` 将 family 定义为
`(mechanism, hyperparameter_family)`，先 family 均匀，再 family 内 stage 均匀，最后 stage
内 run/seed 均匀。每个 member 独立记录 family、mechanism、hyperparameter family、stage、
seed、run 和 checkpoint。

正式训练支持由 SP/OP 各 10 个独立 parent runs 的 `{0.0,0.5,1.0}` checkpoints，以及两个
独立 final-stage OP width variants 构成。两个 deterministic Official-plane heuristics
（greedy courier、stationary helper）无 checkpoint，整个 family 只进入 common-partner 的
family-disjoint test panel，绝不进入训练。

## 10. TrainState、resume 与 deployment

TrainState 必须保存：online/EMA params、PPO optimizer、当前 anchor batch、separation terms、
冻结 comparator、supervision readings、anchor counter/microbatch size、effective epochs、三个
bootstrap members 及 optimizer/counters、M1 summary/history、PPO optimizer step、完整 runner
state、随机域、update/step counters 和资源账本。checkpoint sidecar 对每一项做 fingerprint；
缺字段、版本不符或恢复后 fingerprint 不同均拒绝加载。

deployment 白名单仅包含 task encoder、capability encoder、component embeddings、单一
actor、critic 和 response decoder。response decoder 必须部署，因为 exact online filter
需要它。deployment 不含训练 anchors、comparator、bootstrap members 或机制评估对象。
最后一次 policy update 后必须以当前 params 执行一次 M1，然后才保存最终
checkpoint 并导出 deployment。M1 artifact 记录的 deployable-parameter fingerprint 必须与
deployment bundle 一致；policy manifest 逐 seed 绑定其路径、SHA-256、run id 和判定。

## 11. 可执行层级与应用

- B0：`build_b0_full_history_ppo`；
- B1：`build_b1_protocol_architecture`；
- B2：`build_b2_decision_supervision`；
- B3：`build_b3_active_voi` 始终抛出 `NotImplementedError`。

`run-development-matrix` 自动执行 B0–B2 × K={2,4,8} 并核对 sampler、simulator budget、
seed/key domain 与参数容量；`summarize-development-matrix` 只接受完整 paired score artifact。
机制评估由 `evaluate-identifiability` 和 `evaluate-recoverable-value` 生成独立 schema-2
artifacts；正式 claim report 必须消费这些真实 CRN artifact。
