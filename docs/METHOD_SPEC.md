# METHOD_SPEC：DEPI 可执行方法规格

`authoritative: true`

当前身份：`depi_instant_partner_exact_filter_decision_supervision_v7`，配置版本 17，checkpoint
schema 7，partner manifest schema 4，deployment bundle schema 8。旧 artifact 一律 fail
closed。

## 1. 容量匹配的 R0/B0–B2 骨架

所有可执行层级构造相同参数树：

- task encoder：Official CNN trunk + GRU-128；
- memoryless instant-partner encoder：当前 other-agent planes -> 32 维 `r_t`；
- capability encoder：GRU-64 -> 16 维 `u_t`；
- K 个 16 维 exchangeable response-regime embeddings；
- 一个 actor 和一个 dueling critic，输入 `(x_t,concat(r_t,u_t,c_t))`；
- joint response decoder。

层级的数值接线为：

| 层级 | task GRU 输入 | actor/critic 可见 context | 辅助目标 |
|---|---|---|---|
| R0 | 完整当前观测 | 显式 context 全零 | 无 |
| B0 | 清除显式伙伴 planes | `r_t`，`u,c` 置零 | 无 |
| B1 | 同 B0 | `r_t,u_t,c_t` | response + capability |
| B2 | 同 B1 | 同 B1 | B1 + decision + separation |

因此 B1 是 B0 的严格接线增量，B2 是 B1 的监督增量。R0 是外部强参照，不写成 B0 的组件
前级。所有 inert 参数仍部署以保证容量一致。

`r_t` 的函数没有 carry、previous observation、previous action 或 delta 参数；其无记忆性质由
函数签名保证。task mask 只保证不直接读取 other-agent position/direction/inventory planes，
不阻止伙伴行为的合法世界后果进入 task planes。

PolicyState 保存 `task_carry`、`capability_carry`、`protocol_carry`、`context_summary`、
`previous_observation`、`previous_action`、`episode_start`。`protocol_carry` 就是 `pi_t`，没有
额外 recognition GRU。

## 2. capability 与 exact response filter

CapabilityEncoder 每步读取：

```text
e_t = (o_t-o_{t-1}, embed(a^ego_{t-1}), episode_start)
```

hidden 每步更新，published `u_t` 每 16 个 episode-local steps 更新；episode start 清零。
除相邻发布窗口的 consistency 外，前四个坐标直接预测仅由最近完整合法窗口复算的
visibility、visible movement、visible inventory-holding 与 visible inventory-change rates（线性
映射到 `[-1,1]`）；所有发布坐标按 partner-run group mean 施加跨伙伴 standard-deviation
floor。训练 metrics 报告 semantic prediction loss、variance-floor loss、minimum published
std、publication/partner count、dimension variance 与 mean norm。该目标排除了“仅靠
consistency 即可取全零”的最优解，但
不把其余坐标或整个 `u_t` 宣称为可唯一识别的人类能力；机制评估仍单独报告 `swap-u` 与
`no-capability`。

固定 sticky matrix 的对角为 0.9，其余质量均分。唯一 filter 为：

```text
pi_bar_t = pi_{t-1} @ T
pi_t = softmax(log(pi_bar_t) + ell_t)
c_t = sum_k pi_t,k m_k
```

episode start 使用 uniform prior 并跳过跨 episode emission。transition 不读 task、ego action
或 response，也不学习；所以本节只实现 online response-regime inference，不实现 active
protocol formation。

## 3. joint response likelihood

每个 head 保留同一个 component 轴 K，并只做一次 mixture marginalization：

```text
log p_k(y) = log p_k(visibility)
           + visible * [log p_k(position)+log p_k(direction)+sum_f log p_k(inventory_f)]
           + previous_visible*visible*log p_k(visible_inventory_change)

log p(y|H,a) = logsumexp_k(log pi_k + log p_k(y))
```

visibility transition 不能进入 event。运动学 trunk 读
`(sg[frame_t],m_k,u_t,a_t)`；event trunk 只读 `(m_k,u_t,a_t)`。decoder 不读 task features，
response gradient 也不拥有 task encoder。

component 仅称 exchangeable response regimes。训练与 deployment 复用同一
`sticky_transition_matrix`、`exact_bayes_filter_step`、response target 和 joint likelihood。
对每个 final-policy anchor 还报告 component response 的 pairwise symmetric divergence。

## 4. 每 outer update 的固定训练事务

PPO scan 只计算 PPO，每个 optimizer transaction 后用 rollout 保存的完整行为分布计算精确
categorical KL：

```text
mean_t sum_a pi_old(a|H_t) * [log pi_old(a|H_t)-log pi_new(a|H_t)]
```

不得用 executed-action Monte Carlo 差替代。KL 超阈只缩短当前/下一 PPO exposure；NaN/Inf
才触发数值失败。

所有启用 response、capability 或 decision 目标的层级都在 PPO scan 后固定执行一次 auxiliary
optimizer transaction。无论 PPO early stop 发生在第几个 epoch，辅助目标每个 outer update 的
exposure 都恰好一次；R0/B0 不执行辅助事务。

参数所有权为：

| 参数子树 | 可更新目标 |
|---|---|
| task encoder、instant encoder | PPO、decision/signature、owner distillation |
| capability encoder、component embeddings | PPO、response、decision/signature、separation |
| actor | PPO、actor decision、owner distillation |
| critic | PPO、Q signature |
| response decoder | response |

owner distillation 只能更新 task/instant encoder 与 actor，并必须同时验证参数变化和 KL 下降。

## 5. decision target、置信度与陈旧性

对每个 anchor 的 all-action returns 先 centered，再用：

```text
scale = max(1.4826*MAD(A), 20 raw-return points)
p_A = softmax((A/scale)/temperature)
```

actor loss 是 `KL(p_A||policy)`。confidence 优先使用 fit replicas 的 top-action stability；无
replica tensor 时使用 top-two gap/SE 的 sigmoid。旧 anchor 的权重再乘：

```text
exp(-KL(pi_current || pi_collection)/0.04)
```

metrics 至少报告 target entropy、top-action stability、confidence q10/q50/q90、drift KL、
drift weight 和 effective anchor count。训练 anchors 可在 trigger 间复用，但不能以不衰减权重
无限维持 off-policy 标签。

每个 exchangeable component 的显式 signature 定义为共享 conservative critic 在 one-hot
`z=k`（即 `c=m_k`）干预下的 centered action values：

```text
S_k(x,u,a) = center_a min(Q1(x,r,u,m_k,a), Q2(x,r,u,m_k,a))
sum_k stop_gradient(pi_k) S_k(x,u,a) ~= A_empirical(a)
```

这条直接约束不赋予 component index 唯一语义。final-policy fresh anchors 必须同时导出
utilization、effective count/collapse、pairwise response divergence、pairwise action-signature
divergence、one-hot `z=k` 对 actor ordering/TV 的影响；跨 seed 只允许先做 permutation alignment
再报告稳定性。

## 6. anchor 密度与双重预算公平性

正式 anchor 注册数由正式配置冻结。development 配置缩小每 trigger 的普通状态和 matched
pairs，使每百万 PPO transitions 的 anchor-state 密度与 formal 近似一致，而不是沿用正式
batch 造成八倍监督密度。development 保留与 formal 相同的 fit/evaluation replica 数以维持
标签方差和 cross-fitting 口径，只缩小 state/pair 数；代码按完整 transition 乘积验证密度，
不能只按 replica 单项判断。

R0/B0/B1 不采集并丢弃 B2 continuations。开发矩阵有两种预算口径：

1. core R0/B0/B1/B2 具有相同主 PPO transitions，B2 额外成本如实计入；
2. `R0-extra`、`B0-extra`、`B1-extra` 把 B2 的 continuation + matched-probe transitions 精确换成更多普通
   PPO transitions。最后不足一个标准 rollout 的部分由独立 fixed-shape tail kernel 执行，
   transition ledger 必须精确相等。

## 7. K/方法无关的冻结 comparator

comparator 在 ego matrix 训练前单独采集并拟合：`collect-pair-comparator-source` 使用固定
Official reference ego checkpoint，与 manifest 中互斥的 comparator-fit/validation partner
parents 在真实 simulator 中记录完整 16 步合法历史和 all-action continuations；source artifact
绑定 reference checkpoint、partner manifest、Official commit、config fingerprint、key schedule
与 simulator resource ledger。随后 `fit-pair-comparator` 只接受该严格 schema。状态匹配使用固定 simulator task-only planes，不用
learned task embedding。输入 feature 只含合法可复算 history；label 为：

```text
1[ L2-distance(centered empirical action signatures) > registered threshold ]
```

它不使用 K prototypes、component index、partner run identity 或伙伴动作。fit/validation 至少
各含注册数量的 histories 和八个互斥 partner parents；pair blocks 以 partner run bootstrap，
artifact 绑定 raw source SHA、feature dimensions、fit/validation counts 和 interval。正式 B2
训练必须显式传入冻结 artifact，禁止在 update 0 临时拟合。comparator parents 永不进入 PPO、
decision 或 separation supervision。

matched-pair 分类仍保留 equivalent/distinct/ambiguous；ambiguous、invalid、padding 和 lineage
重叠 pair 权重为零。分类 probability 与 empirical signature distance 必须共同满足注册阈值。

## 8. final-policy M1

最后一次 optimizer update 后：

1. 冻结 deployment params；
2. 用同一 params 重新采集 final rollout/anchors；
3. continuation policy 也固定为同一 params；
4. 重新初始化三个 bootstrap members，只在 final fit replicas 上训练；
5. 只用不重叠 evaluation replica index range 判定；
6. artifact 记录 deployment/collection/continuation 指纹、partner panel hash、key derivation与
   fit/evaluation ranges。

M1 为 report-only。失败不能重启、换 checkpoint 或隐藏 XP。Official policy manifest 必须逐
seed 绑定 M1 路径、SHA 和 deployment fingerprint。

## 9. posterior-predictive calibration

所有 ego seeds 共享一个与训练 lineage-disjoint 的冻结 panel：SP/OP/SA/FCP 各五个 parents。
每个 run 内按 episode、run 间按 partner run 做两级 bootstrap。gate 使用 interval：

- `UCB(NLL_model-NLL_uniform) <= -0.02`；
- `UCB(NLL_model-NLL_no_history) <= -0.02`；
- position/direction coverage interval 与 `[0.85,0.95]` 相交；
- `UCB(Brier_event-0.9*Brier_prior) <= 0`。

该 artifact 名称和论文措辞只能是 posterior-predictive calibration。

## 10. partner sampling

训练分布按以下层级归一化：

```text
p(mechanism) * p(hyperparameter_family|mechanism)
             * p(stage|family) * p(run|stage)
```

SP/OP mechanism 总质量各为 1/2；新增 OP width family 不改变 OP 总质量。heuristic family 只
进入 held-out Common-Partner。manifest schema 4 对 owner、support、comparator fit、
comparator validation、shared calibration 和 confirmatory roles 做 parent/checkpoint/group
互斥校验。

## 11. checkpoint、deployment 与命令

TrainState 保存 online/EMA params、PPO optimizer、anchors、separation terms、冻结 comparator、
aux readings、anchor counter/microbatch、effective epochs、bootstrap state、M1 history、runner、
随机域、step/update counters 和资源账本。schema/fingerprint 不同即拒绝恢复。

deployment 白名单包含 task encoder、instant partner encoder、capability encoder、component
embeddings、actor、critic、response decoder；不含 anchors、comparator 或 bootstrap members。

可执行主层级为 R0/B0/B1/B2；同容量机制消融为 `deterministic_context`、`decision_only`、
`q_only`、`actor_only`、`no_separation`、`no_capability`。B3 始终抛
`NotImplementedError`。六项消融只允许 development，正式配置固定 B2。

开发链为 `collect-pair-comparator-source -> fit-pair-comparator -> run-development-matrix -> evaluate-development-matrix ->
summarize-development-matrix`。summary 只接受 evaluator directories，必须从 raw episode
parquet 重算 XP；自由格式 score JSON 无效。机制 artifact schema 为 3。
开发结论固定使用 seed indexes 0--9；每个 B2 final run 还必须提供 component diagnostic artifact。
`scientific-dry-run` 使用两个 ego、少量 fresh partners 和真实 Official simulator 串起训练、
calibration、development score、identifiability、recoverable value 与 claim-style report；其所有
输出固定 `scientific_readout_allowed=false`，不能进入正式 claim builder。
