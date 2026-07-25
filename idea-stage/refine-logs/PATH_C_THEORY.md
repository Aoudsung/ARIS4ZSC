# 修订理论框架第三版
## Belief-Resampled Probe Kernels、Residual Control Quotient 与有限审计恢复

本文档替代 `PATH_C_THEORY_REVISED.md` 中“单 checkpoint 的 prefix-and-fork 可估计 history-indexed kernel”的有限样本部分。核心修正是区分：

1. **realized-state continuation kernel**：给定一次实际隐藏状态 realization 的回应分布；
2. **belief-averaged intervention kernel**：给定 ego 可见历史、对隐藏状态 posterior 积分后的回应分布。

普通 checkpoint forking 估计前者。理论中的 belief-MDP 对象需要后者。有限审计定理因此必须建立在独立 posterior-state replicas，而不是单 checkpoint 的大量 continuation forks 上。

---

## 1. 受控部分可观测系统

考虑折扣 cooperative POMDP。ego 的可观察历史为 `h∈H`，公共上下文为 `C=C(h)`。

令 `U_t` 表示干预时刻所有对未来 continuation 有影响、但不属于 ego 信息集的状态。它至少可以包含

`U_t=(θ,m_t,o_t^priv,e_t^hid,ξ_t^persist)`，

其中 `θ` 是 episode-static partner mechanism/type，`m_t` 是伙伴内部记忆或适应状态，其余项表示私人观测、隐藏环境变量和持续性外生状态。只写 `x_t=(θ,m_t)` 时，默认它已经包含所有 continuation-relevant 隐变量。

posterior information state 为

`ν_h(du)=P(U_t∈du | H_t=h)`。

对 learned representation，`h` 不是任意更丰富的审计记录，而是
[R015 规范 §1.1](PATH_C_OPPORTUNITY_AUDIT_SPEC.md) 中由 `ego_evidence_contract` 制品绑定的
R015 专用主体可见历史合同：官方局部观测、主体自身动作和内部脚本记录、原始回报与回合
边界，以及只能由这些字段确定导出的掩码、进展和回应记号。可唯一恢复的伙伴动作只是派生
字段；直接读取评估器伙伴动作属于 action-augmented 信息增强诊断。旧代码类
`EgoEvidenceSpecV1` 包含评估器伙伴动作，只能用于历史机制仪器，不是该合同。identity、
mechanism、style、seed 与 layout-style labels 也不属于该信息集。审计使用额外信息时，所得
结论只适用于 instrument，不自动推出 learned representation recovery。

本文把一个具体冻结策略称为伙伴原型；把由相同训练算法、训练目标和约定生成机制产生的
策略集合称为伙伴族。只改变 seed 不产生新的伙伴族。参考支持 `D_ref` 指定定理覆盖的伙伴、
主体可见历史和完整隐藏状态后验集合。进入 `D_ref` 的伙伴族是支持内对象，不是 held-out
伙伴族。

belief-MDP 最优动作价值为

`Q_B^*(h,a)=Q_B^*(C(h),ν_h,a)`。

探针族、探针成本、partner family、策略类与 reference support `D_ref` 均冻结。所有结论只在 `D_ref` 及注册 audit information states 上成立。

### 1.1 官方观测机会审计的夹逼对象

令 `pi_ref` 为冻结的回应屏蔽参照控制器。它拥有与探查控制器相同的动作、候选脚本、窗口、
预算和规划器，也保留同一原始官方历史，但不把当前候选产生的注册回应记号送入专用伙伴信念
更新支路。具体屏蔽必须使用 R015 §2.3 的 `P_mask^(p,t)`：信念接口、共享延续控制器和循环
状态只能读取投影后的共同历史 `x`，当前回应记号及其非官方显式别名不得从旁路进入。原始官方
局部观测仍然保留，所以该比较识别的是注册回应通道的使用效应，不是删除全部伙伴行为证据。
令 `Pi_obs` 为遵守同一约束且动作只依赖 `h` 的控制器类别，定义

`Delta_class_star = sup_{pi in Pi_obs} J(pi) - J(pi_ref)`。

若显式控制器 `pi_L` 只使用 `h`，则因 `pi_L in Pi_obs`，

`Delta_L = J(pi_L) - J(pi_ref) <= Delta_class_star`。

再令 `Pi_full` 为可额外读取完整 `U_t`、其余动作与约束不变的控制器类别。任何主体可见
控制器都可以忽略额外信息，因此 `Pi_obs` 包含于 `Pi_full`。若 `J_full_upper` 是
`sup_{pi in Pi_full} J(pi)` 的已证明上界，则

`Delta_class_star <= J_full_upper - J(pi_ref) = Delta_U`。

所以正确关系是

`Delta_L <= Delta_class_star <= Delta_U`。

左侧只要求显式控制器的信息接口可实现；右侧要求完整信息最优值被精确求出，或规划误差已有
证明上界。告知伙伴身份但仍使用有限前瞻或有限蒙特卡洛的固定控制器只是诊断，不是上界。
若不存在 `J_full_upper`，机会审计可以确立构造性机会，但不能据此否定整个控制器类别。

---

## 2. Residual control quotient

### 定义 1：归一化控制签名

定义

`A^*(h,a)=Q_B^*(h,a)−max_{a'∈A}Q_B^*(h,a')`。

该签名最大分量为 0，保留最优动作集合与所有相对 value gaps，同时消除不影响动作选择的整体 offset。

### 定义 2：control equivalence

对同一公共上下文 `C` 的 histories：

`h ~_V h'  iff  A^*(h,·)=A^*(h',·)`。

对应商记为 `W_C^V`。

### 命题 P0：最粗 control-sufficient quotient

若 residual statistic `T` 对控制签名充分，即存在 `g_C` 使

`A^*(h,·)=g_C(T(h))`，

则 `T` 的划分加细 `W_C^V`；且 `W_C^V` 自身充分。因此 `W_C^V` 是划分序/生成 `σ`-field 意义下最粗的 control-sufficient residual quotient。

**证明。** 由等值类定义直接得到。若 `T` 合并两个不同 advantage signatures，则同一 `T` 值不可能经确定函数同时恢复二者。∎

**范围。** 这是 target characterization，不是 temporal-difference（时序差分，TD）收敛或神经表示恢复定理；最粗信息划分也不等于最少坐标数。

---

## 3. 两类探针 kernel

冻结 `ResponseSummarySpecV1` 与 probe battery `Π_audit`。`R^S` 是该 spec 从原始窗口映射出的唯一 finite categorical token；latency bins、terminal、censored、invalid-script、support-violation 和词表大小 `q` 均在 locked audit 前冻结。structured multi-label outputs 只作 secondary。

### 定义 3：realized-state continuation kernel

对 `u` 与 `π∈Π_audit`：

`G^S_{π,h}(r|u)=P(R^S=r | H_t=h,U_t=u,do(π))`。

它描述在已经固定一个隐藏状态 realization 后，未来随机性造成的 response distribution。

### 定义 4：belief-averaged intervention kernel

理论目标为

`K^S_π(r|h)=P(R^S=r | H_t=h,do(π))`

`          =∫ G^S_{π,h}(r|u) ν_h(du)`。

它是 probe response 的 posterior predictive distribution，也是 belief-MDP information state 对未来回应的可观察含义。

### 命题 P1：单 checkpoint fork 的 estimand

设 checkpoint 在 history `h` 处包含一次 realization

`U_h^*~ν_h`。

从该 checkpoint 运行 `L_inner` 个使用新 continuation randomness 的 forks，经验分布记为 `K_hat^fork_L`。则条件于 `U_h^*`：

`K_hat^fork_L -> G^S_{π,h}(·|U_h^*)` almost surely。

一般不收敛到 `K^S_π(·|h)`。二者对 `ν_h`-几乎所有 `u` 相等的充分必要条件是 `G^S_{π,h}(·|u)` 对 posterior hidden state 几乎处处不变。

**证明。** 条件于 checkpoint hidden state，fork responses iid 来自 `G^S_{π,h}(·|U_h^*)`，由强大数定律得结论。∎

### 备注：边际无偏不等于一致

若把 checkpoint realization 也视为可重复随机抽样，则对任意 response event `B`：

`E[K_hat^fork_L(B)]=K^S_π(B|h)`。

但单 audit checkpoint 的 hidden-draw error 不随 `L_inner` 消失。对 `f(R)∈[0,1]`，若有 `M` 个独立 posterior-state replicas、每个有 `L_inner` 个 inner forks，则

`Var(μ_hat)=σ_between^2/M + σ_within^2/(M L_inner)`，

其中

`σ_between^2=Var_{U~ν_h}(E[f(R)|h,U,π])`，

`σ_within^2=E_{U~ν_h}(Var(f(R)|h,U,π))`。

单 checkpoint 设计对应 `M=1`；即使 `L_inner→∞`，第一项仍然保留。在 informative probe + unresolved belief 的目标场景中，第一项不是可忽略噪声，而是机制敏感性的直接表现。

---

## 4. 正确的 audit sampling object

### 定义 5：registered audit information state

一个 audit unit `I_i` 包含：

- ego-observable history/information state `h_i`；
- public context `C_i` 与固定 ego internal state；
- posterior law `ν_i=P(U_t∈·|H_t=h_i)`；
- 一个能从 `ν_i` 生成与 `h_i` 相容 full simulator state 的 evaluator-side conditional reset mechanism。

自然历史只是构造 `I_i` 的一种方式。synthetic experiments 也可以直接注册可重复的 belief-state generators，而不依赖自然历史精确重现。

### 定义 6：belief-resampled paired-fork protocol

对每个 `I_i`：

1. outer draw：`U_{ij} iid~ν_i`, `j=1,…,M`；
2. state reconstitution：生成与 `h_i` 的 ego-observable decision information 相容的 full checkpoint；
3. probe pairing：从该 outer state 克隆所有 frozen probes；
4. future randomization：每个 continuation 使用新未来随机性；
5. optional inner forks：每个 `(U_{ij},π)` 可运行 `L_inner` 次，仅用于降低 conditional continuation noise。

快照语义分成两种不可混用的运行：replay 使用 `SnapshotV1` 保存的 original keys；audit fork
使用由 `(manifest_seed, unit_id, outer_id, probe_id, inner_id, stream_name)` 导出的 fresh keys，
并把 JAX、NumPy、Python 与 Torch streams 分开命名。runner 接收不可变 snapshot copy 与显式 keys，不共享可变随机状态对象。

估计量为

`K_hat^B_{iπ}=(1/M) Σ_j G_hat_{ijπ}`。

若 `L_inner=1`，每个 `R_{ijπ}` 的边际分布正是 `K^S_π(·|h_i)`。若对同一 `U_{ij}` 配对多个 probes，跨 probe 相关性必须用 cluster-level inference 处理，但不会改变各 probe kernel 的边际正确性。

### 命题 P2：posterior-resampled response 的正确性

在 exact posterior-state reset 下：

`U_{ij}~ν_i`,

`R_{ijπ}|U_{ij}~G^S_{π,h_i}(·|U_{ij})`

蕴含

`R_{ijπ} iid~K^S_π(·|h_i)` across `j`。

**证明。** 对 `U` 积分得到 `K`；outer samples 独立，因此 responses across `j` 独立同分布。∎

---

## 5. Probe quotient 与 population alignment

### 定义 7：belief-kernel probe equivalence

`h ~_K h'` 当且仅当对所有 `π∈Π_audit`：

`K^S_π(·|h)=K^S_π(·|h')`。

对应商记为 `W_C^K`。

注意：这里必须使用 `K`，不能用某个 realized checkpoint 的 `G(·|U^*)` 替代。

### 命题 P3：双向加细

1. 若 **factorization** 成立：`h~_V h' ⇒ h~_K h'`，则 `W_C^V` 的划分加细 `W_C^K`。
2. 若 **separation** 成立：`h not~_V h' ⇒ h not~_K h'`，则 `W_C^K` 的划分加细 `W_C^V`。
3. 二者同时成立时，`W_C^V=W_C^K`，至多相差标签重命名。

**证明。** 由等价关系和划分加细定义直接得到。∎

---

## 6. Mixture-kernel alignment gap

**TV 约定（全文统一）。** `TV(P,Q)=sup_A|P(A)−Q(A)|=(1/2)||P−Q||_1`，取值于 `[0,1]`；本节 `α,β`、引理 L1/L2、定理 T1 及 §8 的 `ρ,ξ` 均按此约定。

在冻结 audit support 上定义

`α = max_w sup_{h,h'∈w} max_π TV(K_π(·|h),K_π(·|h'))`，

`β = min_{w≠w'} inf_{h∈w,h'∈w'} max_π TV(K_π(·|h),K_π(·|h'))`。

这两个量都基于 posterior predictive mixture kernels。

- `α` 测量同一 value class 内 belief-kernel 的不一致；
- `β` 测量不同 value classes 的 probe separation；
- `β−α` 是 population alignment gap。

单-checkpoint realized-state distances 不一致地估计上述量。它们可以作为 posterior-response heterogeneity diagnostic，但不能代入 partition theorem。

---

## 7. 有限审计集恢复定理

### 假设 BR1–BR5

- **BR1（finite audit design）。** 冻结 `m` 个 audit information states、`p` 个 probes，response alphabet 大小为 `q<∞`。
- **BR2（exact posterior reset）。** 每个 audit state 可产生 `M` 个相互独立、hidden state 恰服从 `ν_i` 的 valid replicas。
- **BR3（fresh continuation）。** 对每个 outer replica，未来随机性独立；若 probes 在同一 outer replica 上配对，推断以 outer replica 为 cluster。
- **BR4（uniform mixture gap）。** ground-truth value partition 对 belief kernels 的类内上界为 `α`，类间下界为 `β>α`。
- **BR5（locked decision rule）。** summary、posterior/reset mechanism、audit battery、threshold 与 class labels 的估计程序在 locked audit 前冻结；value signatures 与 kernel estimation 使用交叉拟合或独立数据。

### 引理 L1：belief-kernel 经验分布集中

令每个 `(i,π)` cell 使用 `M` 个独立 posterior-predictive responses：

`K_hat_{iπ}=M^{-1} Σ_{j=1}^M δ_{R_{ijπ}}`。

对单 cell，Bretagnolle–Huber–Carol 多项分布界（L1 范数形式）给出

`P(||K_hat−K||_1 ≥ λ) ≤ 2^q exp(−M λ²/2)`。

对全部 `mp` 个 cells 取 union bound、令单 cell 失败概率为 `δ/(mp)`，得 L1 半径

`λ_M = sqrt(2(q log 2 + log(mp/δ))/M)`；

再按 §6 约定 `TV=(1/2)||·||_1` 折半：

`η_M = λ_M/2 = sqrt((q log 2 + log(mp/δ))/(2M))`。

则以至少 `1−δ` 的概率，所有 cells 同时满足

`TV(K_hat,K)≤η_M`。

（勘误说明 2026-07-10：此前版本从 L1 界直接写出 `TV≤η_M`，未标明 TV 约定与折半步骤；在"TV=||·||_1"的另一约定下该式会差一个因子 2。推导显式后核对：`η_M` 公式、引理 L2、定理 T1 与推论的常数 8 在标准约定下全部正确、无需改动。）

跨 probe 的 paired dependence 不影响逐 cell 边际界和 union bound；但估计 probe contrasts 或 bootstrap confidence interval（置信区间，CI）时必须以 outer replica 为 cluster。

### 引理 L2：pairwise TV error

在 L1 事件上，对任意 `h,h',π`：

`|TV(K_hat_π(h),K_hat_π(h'))−TV(K_π(h),K_π(h'))|≤2η_M`。

### 定理 T1：belief-kernel audit partition recovery

若

`β−α > 4η_M`

并冻结

`τ∈(α+2η_M,β−2η_M)`，

则以至少 `1−δ` 的概率，以下经验判定在所有 registered audit pairs 上恢复真实 value-class partition：

- 若 `max_π TV(K_hat_π(h),K_hat_π(h'))≤τ`，判为同类；
- 否则判为异类。

**证明。** 同类 pair 的经验最大距离不超过 `α+2η_M`；异类 pair 至少有一个 probe 的经验距离不小于 `β−2η_M`。阈值区间非空并严格分开两类。∎

### 推论：外层样本复杂度

使 `η_M<(β−α)/4` 的充分条件为

`M > 8[q log 2 + log(mp/δ)]/(β−α)^2`。

这里 `M` 是独立 posterior-state replicas 数，不是单 checkpoint 的 inner fork 数。

---

## 8. Approximate posterior/reset corollary

生态系统可能只能使用 approximate posterior `q_i` 与 approximate state reconstitution/continuation kernel `G_tilde`。

定义

`K^q_{iπ}=∫ G_{iπ}(·|u) q_i(du)`。

若

`ρ=max_i TV(q_i,ν_i)`，

`ξ=max_{i,π,u} TV(G_tilde_{iπ}(·|u),G_{iπ}(·|u))`，

则 Markov-kernel contraction 与三角不等式给出

`TV(K_tilde^q_{iπ},K_{iπ})≤ρ+ξ`。

若 Monte Carlo error 同时至多 `η_M`，则 uniform cell error budget 为

`ε_cell=η_M+ρ+ξ`。

### 推论 T1-A：approximate belief-kernel recovery

若

`β−α > 4(η_M+ρ+ξ)`，

并选择

`τ∈(α+2ε_cell,β−2ε_cell)`，

则经验 partition 仍恢复 registered audit partition。

若 `ρ` 或 `ξ` 没有可审计上界，则不能调用 exact/approximate recovery theorem；只能报告 model-based posterior predictive fit 或 distributional ecological audit。

### 备注：粒子样本依赖

单个 particle filter 内的 particles 通常不是 independent and identically distributed（独立同分布）posterior draws。若使用 sequential Monte Carlo（序贯蒙特卡洛）或 Markov chain Monte Carlo（马尔可夫链蒙特卡洛），应采用：

- 多个独立 particle systems/chains；
- 明确的 mixing 与 effective sample size（有效样本量，ESS）诊断；
- cluster-robust 或相应依赖集中界。

不得把 nominal particle count 或 inner forks 直接代入 iid multinomial bound。

### 备注：`ρ=ξ=0` 的两个实例（配套提案 §4.5）

对 evaluator 自有的有限脚本伙伴池，有两条获取 posterior-state replica 的精确路线使 `ρ=ξ=0`，从而 recovery 条件回到定理 T1 的精确形式 `β−α>4η_M`，无需动用上面的 `ρ,ξ` 预算：

- **Tier 1（精确枚举 + 前向重放 + 完整状态抽样）：** 前向算法在有限伙伴类型与离散 execution state 上给出精确联合后验（`ρ=0`；option 边界发射为闭式混合，归属歧义处精确边缘化）。每个 outer replicate 从该联合后验有放回且独立地抽取完整 `U=(θ, execution_state)`，再用兼容重构与 fresh fork keys 执行配对 frozen probes（`ξ=0`）。主 kernel 表每个 outer replicate 只保留一行，并在该行内保存全部配对探针和 inner fork。
- **Tier 2（按历史归组匹配 / hash-and-match）：** 用可观测历史给参考生成的完整状态快照归组，同组即 `ν_h` 的精确抽样（`ρ=0`），无需 state reconstruction（`ξ=0`）；代价是 support 限于可复现历史。

上面的 `ρ,ξ>0` 推论 T1-A 只治理**第三条 learned-posterior 路线**（对无闭式似然的黑箱伙伴用近似后验 + 模型化重构）。两条精确路线的具体流程、闭式似然与禁止拼接（never-splice）规则见提案 §4.5。

---

## 9. Finite hidden-state operator formulation

若 hidden state 有 `k` 个值，把所有 `(π,r)` 的 state-conditional response probabilities 堆叠成矩阵/operator `M_C`。对 belief vector `b(h)`：

`k_C(h)=M_C b(h)`，

其中 `k_C(h)` 是堆叠后的 belief-response kernel vector。

### Full column rank 的正确位置

- 若 `M_C` full column rank，则 `b(h)` 可由 exact `k_C(h)` 唯一恢复，因而 `W_C^V` 也可恢复；这是强但构造性的充分条件。
- quotient identification 不要求恢复完整 belief。更弱条件只要求 `M_C` 在不同 value classes 之间单射。
- 可以定义 restricted operator margin，例如

`σ_V(M_C)=inf_{h not~_V h'} ||M_C(b(h)−b(h'))|| / d_V(h,h')`，

其中分母是注册的 value-class distance。`σ_V>0` 给出 quotient-level separation。
- 无论 full rank 还是 restricted rank，都不能补救 one-checkpoint mixture-estimation error；仍需 posterior weighting/resampling。

### 有限 support 的 Rao–Blackwell secondary estimator

若 posterior weights `w_u(h)` 可精确计算，可分别估计 `G_π(·|h,u)` 并混合：

`K_hat_π(·|h)=Σ_u w_u(h) G_hat_π(·|h,u)`。

若 weights 也近似：

`TV(K_hat,K)≤TV(w_hat,w)+Σ_u w_hat_u TV(G_hat_u,G_u)`。

因此普通 state-conditional forks 可以保留，但只能作为 mixture components 的估计器。
该估计器的数据结构不是引理 L1 的独立 categorical response 样本。除非另行证明与其 component sampling 和权重估计完全匹配的集中界，它不能使用 `η_M` 作为主 instrument 的 sampling radius，也不能替代每个 outer replicate 的完整状态独立抽样。

---

## 10. 无 posterior reset 时的可识别对象

若 exact histories 几乎不重复、又不能从 `P(U|h)` 采样，则 pointwise `K_π(·|h)` 不能由一个自然 occurrence 非参数估计。此时有两个诚实替代对象。

### 10.1 Designed belief-state audit generators

直接冻结 `I_i=(C_i,ν_i)`，每次 rollout 独立采样 `U~ν_i`。它们是可重复的 experimental information states。T1 对这些 states 成立；结论只外推到注册 generator support。

### 10.2 Coarsened distributional audit cells

冻结正概率 cell map `G:H→g`，定义

`K_π(·|g)=P(R^S∈· | G(H)=g,do(π))`。

通过新 episode 与随机 probe assignment 可估计该 kernel。它平均了 cell 内 histories 与 hidden states，支持 distributional alignment claim，但不等价于 individual-history quotient recovery，除非另加 cell homogeneity/smoothness 条件。

---

## 11. Approximate control guarantee

### 定理 T2：advantage approximation 到 action regret

设 learned readout `A_hat(h,a)` 满足

`sup_a |A_hat(h,a)−A^*(h,a)|≤ε`。

令 `a_hat∈argmax_a A_hat(h,a)`，则

`Q^*(h,a^*)−Q^*(h,a_hat)≤2ε`。

若该 one-step `Q^*` suboptimality 在策略访问到的所有 histories 上成立，则

`V^*(h)−V^{π_hat}(h)≤2ε/(1−γ)`。

该结果不依赖 probe-kernel estimator，因此不受 round-2 mismatch 影响。

---

## 12. Nuisance refinement 与 conditional leakage

Raw response kernel 可能比 `W_C^V` 更细。nuisance 不应被定义为唯一坐标，而应定义为在给定 `(C,A^*)` 后仍可预测的 identity/style/source information。

必须区分：

- `G(·|h,u)` 的 hidden-state sensitivity：这是 posterior mixing 的组成部分，不自动等于 fingerprint；
- `K(·|h)` 在相同 value class 内随 identity/style 改变：这是 factorization failure 或 nuisance refinement；
- retained representation 中的 conditional identity leakage：这是经验 anti-fingerprint failure。

单-checkpoint `G` 的差异不能直接用来判定 belief-kernel factorization。

---

## 13. 学习算法与理论接口

理论与算法通过以下可审计接口连接：

- advantage/value error；
- return 与 return–probe-budget curve；
- belief-kernel `α,β` 与 alignment gap；
- posterior/reset validity gate；
- conditional leakage；
- operational compression frontier。

不再假设 generic off-policy neural TD 收敛；不再假设坐标剪枝返回 exact minimal quotient；不再声称 ensemble disagreement 必然找到 separating probes。

---

## 14. Active probing 的理论边界

T1 只说明：在 frozen battery 和正确 posterior-predictive samples 下，足够大的 mixture-kernel gap 可被有限样本恢复。它不证明 value-directed probe selection 比 random probing 更快。

要理论化 active design，应另加：

- belief-to-response operator 的可区分性；
- probe cost 与 control regret 约束；
- posterior contraction 或 value uncertainty reduction；
- coverage/positivity；
- adaptive design 的序贯集中条件。

当前 ensemble disagreement 仍作为 empirical acquisition heuristic，通过 random、direct-information、no-probe 和 oracle-design baselines 检验。

---

## 15. Instrument validity

任何 kernel-level alignment claim 必须先通过版本化 instrument validity：

1. synthetic audits 中 posterior-state replicas 的频率/矩与 exact posterior 一致；
2. state reconstitution 保持注册 ego-observable information state；
3. outer replica ESS/independence 达标；
4. CI 与 bootstrap 以 outer replicas 为 cluster；
5. point-mass posterior control 中 ordinary fork 与 belief-resampled estimator 一致；
6. overlapping-posterior control 报告 ordinary fork 的 expected discrepancy 或 failure rate，并验证 belief-resampled estimator 的恢复；不要求一次有限随机样本必然失败；
7. exact mode 无正质量 pruning；approximate route 报告 discarded posterior mass、posterior bias 与 reset bias，无法上界时拒绝 theorem-level claim；
8. generation 与 inference 共用一个 option distribution，且 sparse forward recursion 与 tiny-horizon brute-force enumerator 的 differential test 通过；
9. 每个 cell 先构造 simultaneous confidence bounds，再计算 `alpha_upper=max(within upper confidence bound)` 与 `beta_lower=min(between lower confidence bound)`。仪器有效的必要数值条件是 `beta_lower>alpha_upper`。

---

## 16. 建议主理论表述

> We distinguish the realized-state continuation kernel from the belief-averaged intervention kernel. Ordinary checkpoint forking consistently estimates only the former. On registered resettable information states, independent posterior-state resampling yields samples from the latter. Under a mixture-kernel alignment gap and a locked audit design, the empirical probe partition recovers the registered control partition with finite-sample guarantees. Approximate posterior/reset mechanisms enter through an explicit bias budget; without a bounded budget, ecological kernel results are restricted to distributional or model-based evidence.
