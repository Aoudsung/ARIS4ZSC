# SCIENTIFIC_SPEC：DEPI 科学问题与合法信息边界

`authoritative: true`

本文件定义研究对象、合法信息、estimand 和可证伪主张。实现、配置或报告与本文件冲突时，
以本文件为准。

## 1. 当前研究问题与主张边界

固定任务是 OvercookedV2 ICLR 2025 Test-Time Protocol Formation。ego 在部署时面对训练
运行和算法族均留出的伙伴，只能利用自己的合法局部历史在线协调。

当前方法检验的是：

> 在固定 Official 协议下，基于合法历史的 decision-equivalent online response-regime
> inference，以及由真实 all-action continuation 形成的动作监督，是否改善 held-out XP。

固定对称 sticky transition 不受 ego 动作控制，因此当前实现不是主动协议形成动力学。不得
声称 ego 已通过动作塑造 latent transition，也不得把未实现的 B3/VOI 写成当前贡献。可用的
Bayes 措辞限于：

> exact filtering under the registered discrete response model, with held-out
> posterior-predictive calibration.

它不表示恢复了有协议真值的“真实校准后验”。

## 2. 合法信息边界

部署 policy 只可读取：

- L1：当前 ego 局部观测 `o_t`；
- L2：ego 自己的前一步动作；
- L3：`done` / `episode_start`；
- L4：仅由 L1–L3 更新的 ego recurrent carries；
- L5：仅由 L1–L4 累积得到的历史。

禁止输入包括伙伴真实动作、伙伴 run/算法族/代码/lineage、联合观测、全局环境状态、另一
agent 的 carry、未来轨迹，以及由这些量产生的部署路由信号。伙伴 run 标识只可用于离线
分块、采样和统计，不能进入模型或 comparator feature。训练期联合环境快照和伙伴 carry
只可复制真实状态并运行 CRN continuation，不得成为 ego 输入。

## 3. 四个可执行表征对象

- `x_t`：task GRU 表征。B0–B2 的 task GRU 只接收移除了显式 other-agent position、
  direction 和 inventory planes 的当前观测。
- `r_t=f_instant(o_t^partner)`：无记忆即时伙伴通路，只读取当前 other-agent planes，不保存
  carry，负责可见位置、朝向、inventory、挡路和碰撞几何。
- `u_t`：4 维低频发布的 capability/tendency context。hidden 每步只读相邻帧中即时伙伴
  semantic planes 的变化、ego 前一动作和 episode start，actor 可见值每 16 步发布；四个坐标
  全部预测窗口级可复算行为率。anti-collapse 使用不读取 run ID 的 batch variance floor 与
  off-diagonal covariance penalty。必须报告 prediction/variance/covariance/norm/坍缩诊断、
  `swap-u` 和 `no-capability` 增量。
- `pi_t,c_t`：注册离散 response model 下的精确类别滤波结果；
  `c_t=sum_k pi_t,k m_k`。

task isolation 的精确保证只有：task GRU 不直接读取显式 other-agent semantic planes。伙伴
改变食材、锅、柜台、订单、通路等任务世界的合法后果仍会进入 `x_t`，因此不得声称
`I(x_t; partner history)=0` 或“伙伴历史不能出现在 task carry”。

K 个 component 的名称固定为 **exchangeable response regimes**。index 不对应伙伴身份、
算法族或人类可命名协议，也不具有唯一 decision-signature 语义。训练直接约束
`sum_k pi_k S_k(x,u,a)` 拟合 empirical centered continuation signature，并由 fit-replica
signature 构造 permutation-equivariant `q^A_k`，加入 `KL(sg(q^A)||pi_t)`；one-hot `z=k`
干预构造 `S_k`。这建立 decision-consistent posterior coupling，但不赋予 index 真值语义。
component 本身仍只能用
utilization、response/action divergence、one-hot actor effect 和跨 seed 置换对齐诊断描述。

## 4. 决策等价 estimand

对合法历史 `H_t` 和 ego 首动作 `a`，定义：

```text
G(H_t,a) = registered CRN continuation 的折扣 raw simulator return 均值
A(H_t,a) = G(H_t,a) - mean_b G(H_t,b)
```

continuation 从真实环境、ego carry 和伙伴 carry 快照开始，首动作强制为 `a`，之后双方按
冻结 continuation policy 运行。fit 与 evaluation replicas 使用可审计且不重叠的 key
index domains。标签不得包含 critic、学习端点值或手写协议类别。

训练 anchor、冻结 comparator、final M1、BR-Prox、identifiability 和 recoverable-value 必须
共用同一个 continuation contract：`gamma`、horizon、raw reward definition、terminal
handling、continuation-policy fingerprint 及互斥 fit/evaluation key domains。任一字段不同即
fail closed，不能共用 `A(H,a)` 名称或阈值。

两段历史的 centered signatures 相同，只表示它们在注册 horizon/continuation policy 下
decision-equivalent；不表示伙伴类型相同。

## 5. 可识别控制

机制证据分别回答以下问题，不组成一个会抹掉其他结论的巨型门槛：

1. **task excess leakage**：同一个按 episode 分组的 held-out probe，比较 learned `x_t`
   与固定合法 task-state planes 的 partner-run balanced accuracy，只对 excess accuracy 判门。
   partner-plane 扰动下 task recurrence 不变属于结构测试，不作为额外经验主张。
2. **protocol-state transplant**：固定 checkpoint、当前世界、伙伴、role 和 CRN keys，替换由
   合法历史形成的 capability/protocol state，报告 ego-run paired return drop。
3. **context sensitivity**：交换 `c` 后 action distribution 的 total variation，仅说明执行
   策略使用了 context。
4. **source-world context value**：在 source 世界和 source partner 不变时，以真实 all-action
   return 检验原始 context 的动作分布是否比 donor context 具有更高期望价值。不得要求结果
   朝 donor 世界 signature 移动。
5. **recoverable value**：G1 legal history、G2 transplanted history、G3 state-only、G4
   fit-selected cross-fitted proxy。G4 不是数学上界，`G4>=G1` 不作 pass gate。只有 ego-run
   bootstrap 的 `LCB(G4-G2)` 达注册阈值时才报告回收率及区间。
6. **final M1**：最后一次 optimizer update 后冻结 deployment params，重新采集 final-policy
   anchors，重新初始化 bootstrap members，并用独立 fit/evaluation replica index domains
   评估。旧训练 anchor 不能成为 final M1 证据。

## 6. response target 范围

response target 只能从相邻合法局部帧复算：visibility、`position|visible`、
`direction|visible`、visible inventory 和 visible inventory change。

```text
event = visible_inventory_change
event_mask = previous_visible & current_visible
```

进入/离开视野只进入 visibility head，不能在 event 中重复计数。运动学 head 可读 stopped
physical frame；event head 只能读 `(m_k,u_t,a_t^ego,sg[压缩物理协变量])`，不能读 task
features；物理协变量是固定的 stop-gradient channel-wise spatial mean/max 摘要，可涵盖可见
agent/object/workstation feasibility，但不经过 task encoder，也不复制逐格 frame。让位、争抢、
等待或角色意图均不得作为标签。必须同时报告去除 component 的 held-out NLL、event
prevalence、正/负例 Brier 和 reliability curve。

## 7. 方法层级与公平比较

- R0：强 full-history recurrent PPO 外部参照；
- B0：task-only recurrence + memoryless `r_t`，无 actor-visible `u,c`；
- B1：B0 + capability/exact response filter；
- B2：B1 + empirical decision supervision/separation；
- B3：主动 action-conditioned transition/VOI，未实现并 fail closed。

同容量开发消融固定为 deterministic-context、decision-only、Q-only、actor-only、
no-separation、no-capability 与 response-only-posterior；它们与 R0、filter-only(B1)、
extra-rollout controls 和 K
sensitivity 一起用于排除更简单的机制解释，不改变正式 B2 身份。

同期适应方法只以三个诚实的仓内 proxy 呈现：full-history recurrent、recurrent Bayes-filter
和 deterministic latent-context。它们必须绑定同一训练伙伴池、ego interaction budget、
deployable capacity、Common-Partner panel 和 episode keys；不得写成 CooT、RecBayes 或其他
论文方法的忠实复现。未完成同协议同期复现前，不得声称达到 2026 领域 SOTA。

`B1-B0` 和 `B2-B1` 是严格嵌套增量；`R0` 只回答显式结构是否优于普通 recurrence。开发期
同时报告：相同主 PPO transitions 的 core matrix，以及把 B2 anchor/probe 成本全部换成普通
PPO transitions 的 `R0-extra`/`B0-extra`/`B1-extra` 总 simulator budget 对照。基线不得采集后丢弃 B2
anchors 来伪造预算相等。

## 8. 独立主张向量

formal report 必须分别输出：

```text
performance_claim
filter_architecture_claim
decision_supervision_claim
decision_supervision_cost_efficiency_claim
predictive_calibration_claim
history_dependence_claim
context_causal_value_claim
recoverable_value_claim
capacity_explanation_rejected
common_partner_generalization_claim
```

每项只由命名证据决定。兼容字段 `mechanism_claims_unlocked` 仅为 aggregate diagnostic，不能
撤销已独立成立的 benchmark 或局部机制结论。普通 XP 无论机制结果如何都必须完整报告。

## 9. 可证伪条件

- Φ1：K=4 的 `B1-B0` 区间下界不为正，filter architecture 增量主张失败；
- Φ2：`B2-B1` 区间下界不为正或 final M1 失败，decision-supervision 主张失败；
- Φ3：protocol-state transplant 的 paired drop 区间不排除零，history-dependence 主张失败；
- Φ4：source-world context value 区间下界不为正，context-causal-value 主张失败；
- Φ5：learned task probe 超出固定 task-state probe 的注册 excess threshold，隔离归因失败；
- Φ6：posterior-predictive bootstrap contrast/coverage gate 失败，只撤回相应校准主张；
- Φ7：recoverable denominator 下界不足，则回收率标为 not estimable，不强制解释数值比率。

本项目不声称 latent component 有唯一语义、response event 等于意图、G4 是 oracle upper
bound、更低 NLL 必然提高回报，或 B3 已实现。
