# SCIENTIFIC_SPEC：DEPI 科学问题与合法信息边界

`authoritative: true`

本文件定义研究对象、可观测信息、estimand 与可证伪条件。任何实现、配置或报告与
本文件冲突时，以本文件为准。

## 1. 问题与中心主张

固定任务是 OvercookedV2 ICLR 2025 Test-Time Protocol Formation。ego 在部署时面对
训练运行和算法族均留出的伙伴，只能利用自己的合法局部历史在线协调。

待检验的中心主张是：

> 在相同伙伴分布、相同 simulator-transition 预算、相同随机键和容量匹配条件下，
> B2 能把合法历史中的动态协议信息转化为正确的动作价值排序和实际动作改变，并在
> 两个 Official 布局的 held-out XP 上优于 B0/B1 与注册基线。

该主张不是由网络命名、latent 可视化、测试通过或机械可执行性自动成立；它必须同时
满足开发增量、校准、机制干预和正式 benchmark 证据。

## 2. 合法信息边界

部署 policy 只可读取以下信息：

- L1：当前 ego 局部观测 `o_t`；
- L2：ego 自己的前一步动作；
- L3：`done` / `episode_start`；
- L4：由 L1–L3 更新的 ego recurrent carries；
- L5：仅由 L1–L4 累积得到的历史。

禁止输入包括：伙伴真实动作、伙伴 run/算法族/代码/lineage、联合观测、全局环境状态、
另一 agent 的 recurrent carry、未来轨迹或任何由这些量生成的部署路由信号。伙伴 run
标识只可用于离线分块、采样和统计，不能进入模型。模拟器内部联合动作只用于推进世界；
comparator artifact 不得保留伙伴动作。

训练期环境快照和伙伴 carry 只可用于从真实 episode 状态复制 CRN continuation。
它们不得成为 ego 输入。任何违反本节的信息路径直接使结果无效。

## 3. 三个科学对象

- `x_t`：任务状态表征。B1/B2 的 task GRU 只接收清除了全部 other-agent semantic
  planes 的当前观测；因此它不能由伙伴可见序列重建伙伴历史。
- `u_t`：慢时标 capability/tendency。其隐藏状态每步累积合法证据，但 actor 可见值只
  每 16 步发布一次；同 episode 非重叠窗口施加一致性损失。
- `pi_t, c_t`：动态协议状态。`pi_t` 是 K 类后验，完整协议 carry 就是 `pi_t`；
  `c_t = sum_k pi_t,k m_k`。它每步用新增可观察 response 做精确离散滤波。

K 个 component 是可交换的 value-signature bases。它们不代表伙伴身份，也不固定映射
到任何训练算法族。若需解释 component，只能在冻结 anchor set 上按 empirical action
signature 做 permutation-invariant matching；原始 index accuracy 不是科学读数。

## 4. 决策等价 estimand

对合法历史 `H_t` 和每个 ego 首动作 `a`，定义

```text
G(H_t,a) = mean of discounted raw simulator return over registered CRN continuations
A(H_t,a) = G(H_t,a) - mean_b G(H_t,b)
```

continuation 从真实环境、ego carry 和伙伴 carry 的快照开始，首动作强制为 `a`，之后双方
按冻结目标策略运行 128 步，折扣为配置中的 `gamma`。fit 与 evaluation replicas 使用
不同 key 域。标签只等于固定时域内的 raw simulator return；禁止学习价值、critic 或其他
端点估计进入标签。

两段历史在当前任务状态下若诱导相同的 centered action signature，则对本任务决策等价；
若签名不同，则存在 decision-relevant protocol difference。该定义不把 partner run 当作
协议真值。

## 5. 因果链与必要证据

机制归因必须闭合以下链条：

```text
合法历史 -> pi_t/c_t -> empirical action ordering -> actor probability change
          -> real CRN continuation effect -> held-out XP
```

所需控制为：

1. task leakage：固定 task-only planes 改变 partner planes，task carry 必须不变；held-out
   task representation probe 不得显著识别 partner run。
2. history shuffle：固定 checkpoint、当前观测、task carry、布局、episode time、伙伴和
   CRN keys，仅替换 capability/protocol history carry；报告 run-level paired XP drop。
3. context swap：在 task-state matched、不同 partner-run 状态间分别交换 `u` 和 `c`，
   以真实 all-action CRN continuation 衡量方向一致性，不能只比较 logits。
4. recoverable value：同 checkpoint 和 keys 报告 G1 legal-history、G2 shuffled-history、
   G3 state-only、G4 oracle-continuation，以及在 `G4-G2` 达注册信号阈值状态上的回收率。
5. M1：每个 formal ego run 在最后一次 policy update 后、导出 deployment 前，
   对 deployment 同一参数指纹重新评估 posterior path 和三个独立 bootstrap value
   members。十个 seed 的结果必须经 policy manifest 和 SHA-256 进入 formal claim。

普通 XP 无论机制控制是否通过都必须生成和报告；上述控制只决定能否作“提升来自协议
推断”的归因。

## 6. 可观察事件范围

response target 只能由相邻两帧合法局部观测复算：伙伴可见性变化、可见相对位置、方向、
inventory，以及可见 inventory transition。不得把让位、争抢、等待意图或角色意图写成
观察标签。运动学 head 可读 stopped physical frame；event head 可读 `sg[x_t]` 作为物理
可行性条件，但 response loss 不得更新 task encoder。

## 7. 可证伪条件

- Φ1：B1−B0 或 B2−B1 的注册 paired increment 不为正；对应增量主张失败。
- Φ2：history-shuffle drop 不为正或注册区间跨零；合法历史机制主张失败。
- Φ3：真实 continuation 的 `swap-c` 因果一致性不超过注册阈值；协议上下文因果主张失败。
- Φ4：task leakage probe 超阈或 task representation 随 partner-history shuffle 漂移；结构
  归因失败。
- Φ5：held-out posterior calibration gate 失败；“已校准 Bayes posterior”措辞撤回，只能
  称为未校准 categorical context filter。
- Φ6：最终 checkpoint 的 M1 posterior path 或任一独立 bootstrap value member
  未达到注册排序质量，或 M1 证据与 deployment 参数指纹不一致；
  value-signature mechanism 不成立，但训练和 benchmark 报告继续完成。

## 8. 方法层级边界

B0、B1、B2 是当前可执行层级。B3 所需 action-conditioned value of information 尚未实现，
必须显式标记 `not_implemented` 并 fail closed。不得用空配置、零权重或占位输出替代实现。

本项目不声称 latent component 有唯一语义，不声称 response event 等于意图，不声称
posterior calibration 单独保证高回报，也不以内部 Q、entropy 或表征变化代替真实回报。
