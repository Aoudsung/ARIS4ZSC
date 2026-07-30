# DELTA-ZSC：面向无限伙伴空间的决策等价连续潜变量零样本协作

## Decision-Equivalent Latent Teammate Adaptation for Zero-Shot Coordination

**文档性质：** 完整理论方案、算法规格、证明、训练协议、实验规约与代码重构依据
**目标投稿层级：** ICLR / NeurIPS
**主基准：** JaxMARL OvercookedV2 Test Time Simple / Wide
**第二领域：** Hanabi
**方法约束：** 部署时只有一套共享 actor、一套共享 critic 和一套在线伙伴上下文推断器；参数规模不随伙伴类别或伙伴数量增长。

---

# 摘要

零样本协作要求主体智能体在不访问伙伴身份、参数或训练来源的情况下，与训练时未见过的伙伴直接完成协作。现有 Path C V4.4 使用离散 latent slot、slot-specific Q heads、由 Q 变化量构造的 response code，以及基于目标网络自身 Q 的 use/mask 价值模型。反事实审计已经表明，这套系统虽然能够让回应改变后验和动作，却不能可靠排序真实任务价值；进一步的真实回报读出也无法在当前触发状态、当前 response code 和六动作候选范围内建立正价值证书。该证据否定的是当前离散自蒸馏机制，而不是“在线伙伴上下文”这一总体方向。

本方案提出 **DELTA-ZSC**。核心思想不是为每类伙伴训练一套 actor 或 critic，而是把潜在无限的伙伴策略空间压缩为一个固定维度的、连续的**决策等价潜空间**。模型维护伙伴决策上下文的在线后验 `q_phi(z_t | H_t)`，并使用一个共享的 belief-conditioned actor 与一个共享的 universal critic 进行控制。潜变量不以伙伴身份、训练算法或 checkpoint 为监督，而由三类信号共同约束：伙伴条件响应预测、真实模拟器反事实动作价值、以及价值等价伙伴应在潜空间中合并的 quotient geometry。训练伙伴由一个连续 code-conditioned partner generator 与独立冻结伙伴档案共同构成；生成器不追求表面轨迹差异，而在能力约束下最大化主体最优响应的差异。

为学习决策导向的主动行为，DELTA-ZSC 将当前连续后验下的控制遗憾定义为势函数，并以 potential-based shaping 的形式改善信用分配；该 shaping 在固定势函数下严格保持原始任务最优策略不变。为抑制未见伙伴上的负迁移，方法保留一个共享 robust base policy，并使用 partner-run-block conformal calibration 决定何时启用伙伴条件 residual。整个系统的部署参数规模与伙伴数量无关，不需要测试时梯度更新，也不需要有限策略库路由。

理论部分给出：无结构、不可识别伙伴空间上的 ZSC 不可能性下界；连续决策潜变量作为 Bayes 控制充分统计量的条件；身份细分不变性；近似决策商空间的价值误差界；单一 conditional critic 的存在性；表示、后验和 critic 误差到最终回报遗憾的显式分解；决策遗憾势函数的策略不变性；split-replica 评价的无选择偏差性质；以及 block-conformal 适应门的有限样本覆盖保证。

---

# 1. 研究问题与当前失败的准确边界

## 1.1 旧系统失败的对象

当前 V4.4 的失败不能被解释成“伙伴适应不重要”，其准确含义是：

1. 离散 slot 由自身 Bellman 拟合误差形成，不具备稳定的伙伴决策语义；
2. 所有 slot 使用同一共享执行策略的 target probabilities，因此它们是多个 critic，而不是多个可迁移协作模式；
3. use/mask outcome targets 来自目标网络自身 Q，而非真实 use/mask continuation return；
4. response code 由目标 Q 的 centered-advantage 变化量生成，随后又用于更新 Q 条件后验，形成内生循环；
5. 当前所谓 LCB 是预测均值减固定倍数预测标准差，没有独立的覆盖校准；
6. 真实回报读出只覆盖旧评分器筛出的触发状态，不能代表整个任务机会分布。

因此，旧结果只否定：

```text
离散 TD slots
+ Q-delta response code
+ target-Q 自蒸馏 use/mask value
+ 冻结 reference actor 的局部 logits 偏移
```

它不否定：

```text
连续在线伙伴上下文
+ 单一共享条件 actor/critic
+ 真实回报监督
+ Bayes-adaptive 控制
```

## 1.2 新的单一研究问题

> 在伙伴策略空间不可枚举、伙伴身份和参数不可见的条件下，是否存在一个固定维度的连续决策上下文，使一个共享 actor–critic 能够从合法交互历史中在线推断该上下文，并对训练时未见的伙伴产生近似 Bayes 最优的协作行为？

该问题包含三个不可替代的前提：

1. **结构存在性：** 伙伴对主体最优响应的影响具有低维、可连续参数化的结构；
2. **历史可识别性：** 在关键决策发生前，合法交互历史足以缩小该决策上下文的不确定性；
3. **共享控制可表达性：** 一个固定规模、连续条件化的 actor–critic 能够表示不同伙伴所需的响应。

若任一前提不成立，增加 slot、网络宽度、伙伴 ID 或训练时长都不能创建真正的 ZSC 能力。

---

# 2. 形式化问题

## 2.1 两智能体合作部分可观测随机博弈

考虑两智能体合作随机博弈：

\[
\mathcal G=(\mathcal S,\mathcal A^E,\mathcal A^P,
\mathcal O^E,\mathcal O^P,T,R,\Omega,\gamma),
\]

其中 `E` 是需要零样本适应的主体，`P` 是伙伴。两者共享团队奖励。隐藏环境状态为 `s_t`，联合动作为 `(a_t^E,a_t^P)`，主体只能观察 `o_t^E`。

伙伴策略记为：

\[
\xi\in\Xi,\qquad
 a_t^P\sim \xi(\cdot\mid h_t^P),
\]

其中 `Xi` 可以是无限集合，伙伴可以是随机、循环且历史依赖的策略。伙伴策略在一个回合中固定，但其内部记忆随时间变化。

主体合法历史为：

\[
H_t=(o_0^E,a_0^E,r_0,\ldots,a_{t-1}^E,r_{t-1},o_t^E).
\]

主体策略不得读取伙伴 ID、训练 seed、checkpoint、伙伴参数、伙伴私有观测或完整模拟器状态。

## 2.2 历史 MDP

对任意固定伙伴策略 `xi`，把完整主体历史 `H_t` 视为状态，可以定义边缘化的历史转移核：

\[
P_\xi(dH_{t+1},dr_t\mid H_t,a_t^E),
\]

该核已经对隐藏环境状态、伙伴私有历史和伙伴动作积分。于是每个伙伴策略诱导一个共享状态空间、共享主体动作空间但转移与奖励不同的历史 MDP。

这一定义避免把主体不可见的完整状态错误地代入部署控制器，同时允许理论上精确描述伙伴策略对主体决策的影响。

## 2.3 隐藏决策状态

令：

\[
U_t=(s_t,h_t^P,\xi)
\]

表示影响未来回报、但主体不可直接观察的完整隐藏变量。本方案不要求恢复 `U_t`，而假设存在固定维度映射：

\[
Z_t=\psi(U_t)\in\mathcal Z\subset\mathbb R^d,
\]

使 `Z_t` 保存未来控制所需的信息。`Z_t` 可以随时间变化，因为伙伴内部记忆与隐藏任务状态会变化；它不是静态伙伴身份标签。

同时定义主体当前可观测任务特征：

\[
X_t=f_X(H_t).
\]

## 2.4 决策充分因子假设

**假设 A1（Controlled Decision-Sufficient Factorization）。** 存在固定维度潜空间 `Z` 和条件核 `K_*`，使：

\[
P(r_t,X_{t+1},Z_{t+1}\mid H_t,U_t,a_t^E)
=
K_*(r_t,X_{t+1},Z_{t+1}\mid X_t,Z_t,a_t^E).
\]

也就是说，一旦给定 `(X_t,Z_t)`，完整身份、partner seed 和其他隐藏细节不再影响未来控制分布。

这是方法最核心、也是必须被经验检验的结构假设。它不声称所有合作任务都满足低维结构。

## 2.5 近似因子化

真实任务通常只近似满足 A1。定义近似模型误差：

\[
\sup_{u,h,a}
|r_u(h,a)-r_{\psi(u)}(h,a)|\le \epsilon_r,
\]

\[
\sup_{u,h,a}
\operatorname{TV}
\left(P_u(\cdot\mid h,a),P_{\psi(u)}(\cdot\mid h,a)\right)
\le \epsilon_p.
\]

`epsilon_r` 和 `epsilon_p` 分别衡量潜变量未保留的奖励与转移差异。

---

# 3. 决策等价商空间

## 3.1 动作相对价值签名

在给定历史 `h` 和隐藏上下文 `u` 下，定义主体动作价值：

\[
Q^*(h,a,u).
\]

消除动作无关平移，定义 mean-centered action signature：

\[
\Sigma(h,u,a)
=
Q^*(h,a,u)
-
\frac{1}{|\mathcal A^E|}
\sum_{a'}Q^*(h,a',u).
\]

使用均值中心化而不是样本最大值中心化，是为了让由独立 continuation 样本估计的签名保持无偏。

## 3.2 决策等价关系

在历史 `h` 下：

\[
u\sim_h u'
\quad\Longleftrightarrow\quad
\Sigma(h,u,\cdot)=\Sigma(h,u',\cdot).
\]

如果两个伙伴身份或隐藏状态产生相同的动作相对价值，它们在该历史下对主体控制是等价的。模型应当压缩这类差异，而不是奖励身份可辨识性。

## 3.3 连续决策距离

定义局部经验决策距离：

\[
d_{\mathrm{DE}}(u,u';h)
=
\|\Sigma(h,u,\cdot)-\Sigma(h,u',\cdot)\|_2.
\]

定义跨状态加权距离：

\[
D_{\mathrm{DE}}(u,u')
=
\mathbb E_{h\sim\nu}
[w(h)d_{\mathrm{DE}}(u,u';h)],
\]

其中 `nu` 是固定的状态审计分布，`w(h)` 可以使用动作 margin 或状态重要性，但必须在确认性结果生成前冻结。

`Z` 的几何应近似这个 quotient distance，而不是 checkpoint identity、SP/OP 标签或轨迹欧氏距离。

---

# 4. DELTA-ZSC 总体架构

部署时仅保留：

```text
一个 Task Encoder
一个 Partner Belief Encoder
一个 Belief Set Encoder
一个共享 Universal Actor
一个共享 Universal Dueling Critic
一个共享 Response Decoder
一个冻结的适应置信门
```

没有：

```text
每类伙伴 actor
每类伙伴 critic
离散 slot
slot responsibility
伙伴 ID 分类器
response codebook
策略库路由
测试时梯度更新
```

完整执行链：

```text
官方合法历史 H_t
    ├── Task Encoder → X_t
    └── Partner Belief Encoder → q_phi(Z_t | H_t)
                                      ↓
                             posterior particles
                                      ↓
                        permutation-invariant belief embedding eta_t
                                      ↓
                shared base + low-rank context-residual actor
                                      ↓
                           one shared action distribution
                                      ↓
                              environment transition
                                      ↓
                         response likelihood and belief update
```

---

# 5. 连续伙伴后验

## 5.1 多峰后验

不同伙伴解释在早期可能同时成立。使用固定组件数的混合高斯后验：

\[
q_\phi(z_t\mid H_t)
=
\sum_{m=1}^{M}
\alpha_{t,m}
\mathcal N
\left(
\mu_{t,m},
\operatorname{diag}(\sigma^2_{t,m})
\right).
\]

`M` 是后验近似复杂度，不是伙伴类别数。所有混合组件共享同一个编码网络，组件可以在不同回合表示不同假设。

建议默认：

```text
latent_dim d = 8 或 16
mixture_components M = 4
posterior_particles K = 8 或 16
```

维度必须由 run-disjoint 验证上的有效秩、预测误差和控制回报共同选择，不能解释为固定数量的伙伴类型。

## 5.2 递归更新

构造一步交互证据：

\[
e_t=f_E(o_{t-1}^E,a_{t-1}^E,r_{t-1},o_t^E,d_{t-1}).
\]

递归状态更新：

\[
c_t=\operatorname{GRU}_\phi(c_{t-1},e_t).
\]

由 `c_t` 输出混合权重、均值和方差。episode start 时恢复固定先验。

## 5.3 Belief Set Encoder

从后验采样：

\[
z_t^{(k)}\sim q_\phi(z_t\mid H_t),
\qquad k=1,\dots,K.
\]

使用共享的 permutation-invariant 聚合：

\[
\eta_t=
ho_\eta
\left(
\sum_{k=1}^K w_k\varphi_\eta(z_t^{(k)}),
\sum_{k=1}^K w_k\varphi_\eta(z_t^{(k)})^2,
\mathcal H(w),
S_t^{\mathrm{support}}
\right).
\]

`eta_t` 同时保留后验中心、离散程度、多峰信息和训练支持度。

---

# 6. 单一共享 Actor 与 Critic

## 6.1 低秩上下文调制 Actor

基础策略负责在伙伴信息不足或 OOD 时保持任务能力：

\[
\ell_0(a\mid X_t).
\]

伙伴条件 residual 通过低秩调制生成。对第 `l` 层：

\[
W_l(\eta)
=
W_l^{(0)}
+
U_l\operatorname{diag}(g_l(\eta))V_l^\top,
\]

其中低秩 `r` 固定，参数量不随伙伴数量增长。

最终 logits：

\[
\ell_\theta(a\mid X_t,\eta_t)
=
\ell_0(a\mid X_t)
+
\kappa_t\Delta\ell_\theta(a\mid X_t,\eta_t),
\]

`kappa_t∈[0,1]` 是适应置信门。部署策略：

\[
\pi_\theta(a\mid X_t,\eta_t)
=\operatorname{softmax}(\ell_\theta).
\]

这是一套 actor；base 与 residual 是同一网络的共享组成部分，不是多个伙伴策略。

## 6.2 单一 Universal Dueling Critic

Critic 输入 `(X_t,eta_t)`，一次输出全部主体动作价值：

\[
Q_\omega(X_t,\eta_t,a)
=
V_\omega(X_t,\eta_t)
+A_\omega(X_t,\eta_t,a)
-
\frac1{|A|}\sum_{a'}A_\omega(X_t,\eta_t,a').
\]

它只有一套参数。标准 target network 或 Polyak copy 只用于数值稳定，不承担伙伴语义。

## 6.3 退化后验接口

训练期若已知 privileged teacher latent `z^T`，把 belief 设为：

\[
b=\delta_{z^T}.
\]

通过相同 Belief Set Encoder、相同 actor 和相同 critic 计算 full-information teacher 策略。没有额外 teacher actor 或每类 teacher critic。

---

# 7. 伙伴响应模型

定义共享 response decoder：

\[
p_\omega(y_{t+1},r_t,o_{t+1}^E
\mid X_t,a_t^E,z_t).
\]

主版本直接预测官方可见信息，不再学习 Q-delta codebook。OvercookedV2 中的响应目标可以由官方观测确定地构造：

- 下一官方局部观测；
- 可见伙伴相对位置与方向变化；
- 可见库存变化；
- 可见环境网格变化；
- 成功/失败交互结果；
- 团队原始奖励与 episode boundary；
- 当前不可见标记。

所有字段必须是部署时主体可见历史的函数。显式伙伴原始动作仍只能作为 action-augmented 诊断版本。

响应损失：

\[
\mathcal L_{\mathrm{resp}}
=
-\mathbb E_{z_t\sim q_\phi}
\log p_\omega(y_{t+1},r_t,o_{t+1}^E
\mid X_t,a_t^E,z_t).
\]

该损失负责可识别性，但不单独定义 latent 语义。

---

# 8. Privileged Teacher 与决策充分蒸馏

## 8.1 Teacher latent

对于连续 partner generator，训练期已知生成 code `c`，定义：

\[
z_t^T=e_\chi(c,X_t).
\]

允许 `z_t^T` 随任务状态变化，但不能输入 partner ID 或 checkpoint 编号。

对于没有生成 code 的冻结训练伙伴，可以使用只在训练期运行的 full-trajectory encoder：

\[
z_t^T=e_\chi^{\mathrm{full}}(H_{0:T},t),
\]

其输出仅作为蒸馏目标，不在部署使用。

## 8.2 动作价值蒸馏

学生后验采样产生 `eta_t^S`，teacher 使用退化后验 `eta_t^T`。定义 centered-Q 蒸馏：

\[
\mathcal L_{\mathrm{adv}}
=
\left\|
\operatorname{center}
Q_\omega(X_t,\eta_t^S,\cdot)
-
\operatorname{sg}
\left[
\operatorname{center}
Q_\omega(X_t,\eta_t^T,\cdot)
\right]
\right\|_2^2.
\]

这里 `sg` 表示 stop-gradient。

## 8.3 策略蒸馏

\[
\mathcal L_{\pi\text{-distill}}
=
w_t
D_{\mathrm{KL}}
\left(
\pi_\theta(\cdot\mid X_t,\eta_t^T)
\;\|\;
\pi_\theta(\cdot\mid X_t,\eta_t^S)
\right).
\]

权重由 teacher action margin 决定：

\[
w_t=
Q_T(a_T^*)-
\max_{a\neq a_T^*}Q_T(a).
\]

只有动作选择真正影响价值时，才强迫学生恢复 teacher 决策。

## 8.4 不直接预测生成 code

模型不使用：

\[
\|z-c\|^2
\]

或 partner code 分类损失作为主目标。生成 code 可以包含大量行为细节，恢复 code 会重新退化成身份/风格识别。

---

# 9. 真实反事实动作价值 Anchor

## 9.1 目的

仅靠 TD 和响应预测，模型仍可能形成内部自洽但任务价值错误的表示。必须使用真实模拟器 continuation return 为 latent 与 critic 提供外部坐标系。

## 9.2 Anchor 采样

每隔固定更新，从当前训练轨迹中按预登记规则抽取 anchor states：

- 时间分层；
- 任务阶段分层；
- posterior decision regret 分位数分层；
- partner source 与 code 分层；
- 不根据确认性伙伴结果调整采样。

## 9.3 环境克隆

对每个 anchor：

1. 克隆完整环境状态；
2. 克隆伙伴参数、伙伴 recurrent carry 与随机流 lineage；
3. 克隆主体合法历史；
4. 对每个主体动作 `a∈A^E` 强制执行一步；
5. 后续使用冻结 target actor 继续到固定 horizon 或 episode end；
6. 所有动作分支使用共同原始随机数；
7. fit replicas 与 evaluation replicas 完全分离。

后续 continuation 使用冻结的 privileged target actor `bar pi^T`。因此 anchor 直接估计的是 `Q^{bar pi^T}`，不是不可获得的精确 `Q^*`。若 teacher 满足 `||Q^{bar pi^T}-Q^*||_infty <= epsilon_T`，则 mean-centered signature 的逐动作误差至多 `2 epsilon_T`；该 teacher gap 必须作为独立诊断报告，不能把辅助 anchor 直接称为最优价值真值。

## 9.4 真实目标

对动作 `a` 的 fit continuation mean：

\[
\bar G_{\mathrm{fit}}(h,a)
=
\frac1{N_{\mathrm{fit}}}
\sum_{n=1}^{N_{\mathrm{fit}}}G_n(h,a).
\]

定义无偏 centered signature：

\[
\widehat\Sigma_{\mathrm{cf}}(h,a)
=
\bar G_{\mathrm{fit}}(h,a)
-
\frac1{|A|}
\sum_{a'}\bar G_{\mathrm{fit}}(h,a').
\]

Critic anchor loss：

\[
\mathcal L_{\mathrm{cf}}
=
\sum_a
\operatorname{Huber}
\left(
\operatorname{center}Q_\omega(X_h,\delta_{z^T},a)
-
\operatorname{sg}\widehat\Sigma_{\mathrm{cf}}(h,a)
\right).
\]

真实 continuation return 而非 target Q 是 ground truth。

## 9.5 独立评价

任何由 fit replicas 选择的动作，必须用 evaluation replicas 评价：

\[
\bar G_{\mathrm{eval}}(h,\hat a_{\mathrm{fit}}).
\]

evaluation replicas 不参与网络更新、阈值选择或模型选择。

---

# 10. 决策商空间几何损失

对两个 teacher contexts `z_i^T,z_j^T`，使用真实 anchor signatures：

\[
d_{ij}^{\mathrm{cf}}
=
\mathbb E_h
\left[
\|\widehat\Sigma_i(h)-\widehat\Sigma_j(h)\|_2
\right].
\]

定义三段式 quotient loss：

\[
\mathcal L_{\mathrm{quot}}
=
\begin{cases}
\|z_i^T-z_j^T\|_2^2,
&d_{ij}^{\mathrm{cf}}\le\epsilon_{\mathrm{eq}},\\
[m-\|z_i^T-z_j^T\|_2]_+^2,
&d_{ij}^{\mathrm{cf}}\ge\epsilon_{\mathrm{sep}},\\
(\|z_i^T-z_j^T\|_2-g(d_{ij}^{\mathrm{cf}}))^2,
&\text{otherwise}.
\end{cases}
\]

其中 `g` 是冻结的单调尺度映射。

该损失实现：

- 行为不同但最优响应相同的伙伴被压缩；
- 行为相似但需要不同主体响应的伙伴被分开；
- 潜空间不需要有限类别语义；
- 新伙伴可以落在已见上下文之间。

---

# 11. 信息瓶颈与后验一致性

## 11.1 信息瓶颈

\[
\mathcal L_{\mathrm{IB}}
=
D_{\mathrm{KL}}
\left(q_\phi(z_t\mid H_t)\|p(z_t)\right).
\]

使用 free-bits 或 KL warmup，避免早期 posterior collapse。

## 11.2 Prefix–Full Posterior Consistency

完整训练轨迹 teacher posterior 为 `q_full`，在线 prefix posterior 为 `q_prefix`：

\[
\mathcal L_{\mathrm{cons}}
=
D_{\mathrm{KL}}
\left(
q_{\mathrm{full}}(z_t)
\|q_{\mathrm{prefix}}(z_t)
\right).
\]

该项训练在线 encoder 从有限历史恢复完整轨迹中可确定的决策上下文。

## 11.3 梯度路由

主版本固定：

- actor policy-gradient 不反传到 belief encoder；
- critic real-return、response、counterfactual、distillation、quotient 和 consistency loss 更新 belief encoder；
- decision-regret shaping 对 belief 与 critic 全部 stop-gradient；
- partner ID、partner seed 和训练算法标签不进入任何梯度路径。

这样避免 actor 通过任意改变 latent 来制造易优化的策略捷径。

---

# 12. 决策导向主动行为

## 12.1 连续后验下的决策遗憾

使用退化后验 critic 定义 full-information action value：

\[
Q^{\mathrm{FI}}(X_t,z,a)
=Q_{\bar\omega}(X_t,\delta_z,a),
\]

其中 `bar omega` 是冻结 target critic。

当前 belief 下的决策遗憾：

\[
\mathcal R_{\mathrm{dec}}(X_t,b_t)
=
\mathbb E_{z\sim b_t}
\left[
\max_a Q^{\mathrm{FI}}(X_t,z,a)
\right]
-
\max_a
\mathbb E_{z\sim b_t}
\left[
Q^{\mathrm{FI}}(X_t,z,a)
\right].
\]

性质：

- `R_dec≥0`；
- 身份仍不确定但所有 context 共享同一最优动作时，`R_dec=0`；
- 只有剩余不确定性会改变最优动作时，`R_dec>0`。

## 12.2 Potential-based shaping

定义：

\[
\Phi(X_t,b_t)=-\mathcal R_{\mathrm{dec}}(X_t,b_t).
\]

训练 shaping：

\[
F_t
=
\lambda_{\mathrm{DR}}
\left[
\gamma\Phi(X_{t+1},b_{t+1})-
\Phi(X_t,b_t)
\right]
=
\lambda_{\mathrm{DR}}
\left[
\mathcal R_t-
\gamma\mathcal R_{t+1}
\right].
\]

PPO 使用：

\[
r_t'=r_t+F_t.
\]

终止状态固定 `Phi=0`。由于是势函数差，它只改善信用分配，不改变原始任务的最优策略集合。实现中 `Phi` 由 target critic 与 target belief model 计算，并在一个 PPO 更新块内保持冻结；不同势函数版本产生的 rollout 不得混入同一个 advantage batch。

## 12.3 不使用身份信息奖励

不使用 partner classification accuracy、response entropy 或 mutual information 作为 actor intrinsic reward。这些目标会奖励可辨识但价值无关的伙伴差异。

---

# 13. 连续训练伙伴生成过程

## 13.1 单一 Partner Generator

训练期伙伴由一个共享条件策略生成：

\[
\mu_\psi(a_t^P\mid o_t^P,c),
\qquad
c\sim\mathcal U([-1,1]^{d_c}).
\]

不同 code 通过 FiLM/low-rank modulation 产生连续伙伴行为。参数量不随生成伙伴实例数增长。

## 13.2 混合训练支持

训练伙伴分布为：

\[
P_{\mathrm{train}}
=
\lambda_gP_{\mathrm{generator}}
+
\lambda_sP_{\mathrm{snapshot}}
+
\lambda_fP_{\mathrm{frozen\ external}},
\]

其中：

- current generator 提供连续变化；
- generator snapshot archive 防止遗忘和共适应；
- 独立冻结伙伴防止生成器与 ego 私下形成不可外推协议。

## 13.3 能力约束

对 sampled code `c_j`，使用同一个 shared actor 的 privileged teacher context 计算团队回报：

\[
J_j=J(\pi_\theta(\cdot\mid\delta_{z_j^T}),\mu_\psi(\cdot\mid c_j)).
\]

要求低分位能力：

\[
\operatorname{CVaR}_{\beta}(J_j)\ge R_{\min}.
\]

这防止生成器通过完全随机或恶意伙伴制造虚假多样性。

## 13.4 Best-response diversity

对 code `c_i,c_j` 的真实 counterfactual signatures 定义 RBF kernel：

\[
K_{ij}
=
\exp
\left(
-\frac{
\|S(c_i)-S(c_j)\|_2^2
}{2\sigma_D^2}
\right),
\]

其中 `S(c)` 是跨 anchor states 的真实 centered action return signature。

多样性目标：

\[
\mathcal J_{\mathrm{BRDiv}}
=
\log\det(K+\epsilon I).
\]

生成器优化：

\[
\max_\psi
\mathcal J_{\mathrm{BRDiv}}
-\lambda_{\mathrm{smooth}}\mathcal L_{\mathrm{smooth}}
\quad
\text{s.t.}
\quad
\operatorname{CVaR}_{\beta}(J)\ge R_{\min}.
\]

使用 Lagrangian 实现。生成器更新时冻结 ego actor、critic 与 teacher encoder，避免目标共同漂移。

---

# 14. 适应置信门与 OOD 回退

## 14.1 Base 与 conditional policy

Base policy：

\[
\pi_0(a\mid X_t)=\operatorname{softmax}(\ell_0).
\]

Conditional policy：

\[
\pi_c(a\mid X_t,b_t)
=
\operatorname{softmax}(\ell_0+\Delta\ell).
\]

预测条件适应增益：

\[
\widehat\Delta_t
=
\mathbb E_{a\sim\pi_c}Q_\omega(X_t,b_t,a)
-
\mathbb E_{a\sim\pi_0}Q_\omega(X_t,b_t,a).
\]

## 14.2 Partner-run-block conformal calibration

在与训练运行完全分离的 calibration partner runs 上，对每个 run `j` 定义：

\[
S_j
=
\max_{(h,a)\in\mathcal I_j}
|\widehat Q(h,a)-\bar G_{\mathrm{eval}}(h,a)|.
\]

取 split-conformal 分位数 `q_alpha`。若 evaluation continuation mean 还存在有限 replica 误差，额外加入 bounded-return concentration 半径 `epsilon_MC`。

部署门：

\[
\kappa_t
=
\mathbf 1
\left[
\widehat\Delta_t
>
2(q_\alpha+\epsilon_{\mathrm{MC}})
\right]
\cdot
\mathbf 1[S_t^{\mathrm{support}}\ge\tau].
\]

未通过时严格回退到 base policy。

训练中使用平滑 sigmoid 近似；确认性评估使用冻结硬门。

---

# 15. 统一训练目标

## 15.1 PPO 与真实奖励

\[
\mathcal L_{\mathrm{PPO}}
=
-\mathbb E
\min
\left(
\rho_t\hat A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t
\right).
\]

GAE 使用 shaped reward `r'_t`，但最终评估始终使用原始环境回报。

## 15.2 Critic loss

\[
\mathcal L_V
=
\operatorname{Huber}(V_\omega-\hat V^{\mathrm{GAE}}),
\]

并加入真实反事实 action anchor：

\[
\mathcal L_Q=\mathcal L_{\mathrm{cf}}.
\]

## 15.3 总损失

\[
\begin{aligned}
\mathcal L_{\mathrm{DELTA}}
=&\;
\mathcal L_{\mathrm{PPO}}
+c_V\mathcal L_V
-c_H\mathcal H(\pi)
+\lambda_{\mathrm{cf}}\mathcal L_{\mathrm{cf}}\\
&+\lambda_{\mathrm{resp}}\mathcal L_{\mathrm{resp}}
+\lambda_{\mathrm{adv}}\mathcal L_{\mathrm{adv}}
+\lambda_{\pi}\mathcal L_{\pi\text{-distill}}\\
&+\lambda_{\mathrm{quot}}\mathcal L_{\mathrm{quot}}
+\lambda_{\mathrm{cons}}\mathcal L_{\mathrm{cons}}
+\beta\mathcal L_{\mathrm{IB}}.
\end{aligned}
\]

## 15.4 参数更新分区

| 参数 | 接收的梯度 |
|---|---|
| Task Encoder | PPO、V、CF |
| Belief Encoder | response、V/CF、adv distill、quotient、consistency、IB |
| Actor | PPO、policy distill |
| Critic | V、CF、adv distill |
| Response Decoder | response |
| Teacher Encoder | CF、quotient、teacher return |
| Partner Generator | competence-constrained BR diversity |

---

# 16. 统一训练算法

## Algorithm 1：DELTA-ZSC

```text
输入：环境分布、连续 partner generator、冻结外部伙伴档案
初始化：task encoder、belief encoder、actor、critic、decoder、teacher encoder
初始化：generator snapshot archive、target critic、target belief model

for outer update = 1 ... N:

    # A. 冻结伙伴分布，更新主体
    for ego block step = 1 ... B_ego:
        1. 从 current generator / snapshot archive / frozen archive 采样伙伴
        2. 采样环境、席位和随机种子
        3. 一部分 lane 使用 student posterior；一部分 lane 使用 privileged degenerate posterior
        4. 收集完整 episode 或固定长 recurrent rollout
        5. 用 target critic + detached posterior 计算 decision-regret potential
        6. 计算 shaped reward、GAE、PPO loss 与 value loss
        7. 计算 response、teacher/student、IB 与 consistency losses
        8. 更新 actor、critic、belief encoder 和 decoder

    # B. 真实反事实 anchor
    if outer update mod anchor_interval == 0:
        1. 从最近轨迹按冻结分层规则采样 anchor states
        2. 克隆环境与伙伴状态
        3. 对全部主体动作执行 fit/evaluation paired continuations
        4. 用 fit replicas 更新 CF critic 与 quotient geometry
        5. evaluation replicas 只写入 audit/calibration artifact

    # C. 更新伙伴生成器
    if outer update mod generator_interval == 0:
        1. 冻结 actor、critic、teacher encoder
        2. 采样连续 partner codes
        3. 估计 competence 与真实 best-response signatures
        4. 在 CVaR competence constraint 下最大化 BR-diversity logdet
        5. 更新 generator
        6. 按固定规则保存 generator snapshot 到 archive

    # D. Target 更新
    Polyak update target critic / target belief model

训练结束：
    1. 冻结全部参数
    2. 使用独立 calibration partner runs 计算 conformal radius
    3. 冻结 support threshold 与 hard gate
    4. 只运行一次 confirmatory partner evaluation
```

这是一套统一算法，不是“为每种伙伴分别训练”或“先建策略库再选择”。

---

# 17. 部署算法

## Algorithm 2：未见伙伴在线适应

```text
初始化 belief prior、actor carry、task carry

for t = 0 ... T-1:
    1. 从官方历史更新 task feature X_t
    2. 更新 q_phi(z_t | H_t)
    3. 采样 K 个 posterior particles 并形成 eta_t
    4. 计算 base policy pi_0 与 conditional policy pi_c
    5. 用冻结 critic、conformal radius 和 support score 计算 hard gate kappa_t
    6. 执行 pi = pi_0                      if kappa_t = 0
             pi = pi_c                      if kappa_t = 1
    7. 观察官方下一观测和原始团队奖励
    8. 更新历史
```

部署阶段不使用：

- partner generator；
- teacher encoder；
- partner code；
- simulator clone；
- counterfactual future；
- 参数更新。

---

# 18. 理论结果与详细证明

## 定理 1：无结构且不可识别时 ZSC 的不可能性

**定理。** 考虑两个等先验伙伴上下文 `z_1,z_2`。在决策历史 `h` 前，两者诱导的主体合法历史分布完全相同。存在两个不同动作 `a_1,a_2` 和 `Delta>0`，使：

\[
Q(h,a_1,z_1)-Q(h,a,z_1)\ge\Delta,
\quad\forall a\neq a_1,
\]

\[
Q(h,a_2,z_2)-Q(h,a,z_2)\ge\Delta,
\quad\forall a\neq a_2.
\]

则任何只读取合法历史的随机策略，在 `h` 处的平均决策遗憾至少为 `Delta/2`。

**证明。** 由于两种上下文在 `h` 前不可区分，策略必须在两者下使用同一动作分布 `p(a|h)`。在 `z_1` 下：

\[
Q(h,a_1,z_1)-\mathbb E_{a\sim p}Q(h,a,z_1)
\ge\Delta[1-p(a_1)].
\]

在 `z_2` 下：

\[
Q(h,a_2,z_2)-\mathbb E_{a\sim p}Q(h,a,z_2)
\ge\Delta[1-p(a_2)].
\]

等先验平均：

\[
\operatorname{Regret}
\ge
\frac\Delta2
[2-p(a_1)-p(a_2)].
\]

因为 `a_1 != a_2` 且概率和不超过 1：

\[
p(a_1)+p(a_2)\le1.
\]

所以：

\[
\operatorname{Regret}\ge\frac\Delta2.
\]

证毕。

**含义。** 对任意伙伴都无条件成功是不可能的。方法必须明确结构、覆盖与可识别性假设。

---

## 定理 2：连续 belief 是 Bayes 控制充分统计量

**定理。** 若假设 A1 成立，则：

\[
b_t(dz)=P(Z_t\in dz\mid H_t)
\]

与当前任务特征 `X_t` 一起构成 Bayes-adaptive MDP 的充分状态；存在最优策略：

\[
\pi^*(a\mid X_t,b_t).
\]

**证明。** 给定 `(X_t,b_t,a_t)`，即时期望奖励为：

\[
\bar r(X_t,b_t,a_t)
=
\int r(X_t,z,a_t)b_t(dz).
\]

由 A1，下一观测、奖励和潜变量的联合分布为：

\[
P(dX_{t+1},dr_t,dz_{t+1}\mid X_t,b_t,a_t)
=
\int
K_*(dX_{t+1},dr_t,dz_{t+1}\mid X_t,z_t,a_t)
 b_t(dz_t).
\]

观测 `X_{t+1},r_t` 后，Bayes 规则唯一确定后验：

\[
b_{t+1}=\mathcal B(b_t,X_t,a_t,r_t,X_{t+1}).
\]

因此 `(X_{t+1},b_{t+1})` 的转移分布只依赖 `(X_t,b_t,a_t)`，与更早历史条件独立。于是 `(X_t,b_t)` 是 Markov 状态。

有限时域下，令终止值为 0，Bellman 递推：

\[
V_t^*(X,b)
=
\max_a
\left[
\bar r(X,b,a)
+
\gamma
\mathbb E
V_{t+1}^*(X',b')
\right].
\]

递推仅依赖 `(X,b)`。无限折扣时域下，Bellman 算子是 `gamma` 压缩，存在唯一固定点。故存在只依赖 `(X_t,b_t)` 的最优策略。证毕。

---

## 命题 1：价值无关身份细分不变性

**命题。** 假设所有受控核与奖励只通过 `z=psi(i)` 依赖身份 `i`。把任一身份拆成多个子身份，使子身份具有相同 `z`，并保持总先验质量不变，则任意历史下的 latent posterior、Bayes value、最优策略和决策遗憾均不变。

**证明。** 原身份先验 `p(i)` 通过 `psi` 推送到 latent prior：

\[
p_Z(B)=\sum_{i:\psi(i)\in B}p(i).
\]

拆分后，同一 `z` 下子身份概率之和仍为原概率，因此 pushforward prior 不变。由于同一 `z` 的所有子身份具有相同 likelihood，任意观测序列下 Bayes posterior 的 pushforward measure 仍不变。由定理 2，价值与策略只依赖 `(X,b_Z)`，故不变。决策遗憾也是 `b_Z` 和 `Q(X,z,a)` 的函数，故不变。证毕。

---

## 定理 3：近似决策商空间的价值误差界

**定理。** 奖励满足 `|r|≤R_max`，折扣 `gamma<1`。两个 history MDP 满足：

\[
\sup_{h,a}|r_1(h,a)-r_2(h,a)|\le\epsilon_r,
\]

\[
\sup_{h,a}\operatorname{TV}(P_1(\cdot\mid h,a),P_2(\cdot\mid h,a))
\le\epsilon_p.
\]

则：

\[
\|V_1^*-V_2^*\|_\infty
\le
\frac{\epsilon_r}{1-\gamma}
+
\frac{2\gamma R_{\max}}{(1-\gamma)^2}\epsilon_p.
\]

相同界适用于最优 `Q`。

**证明。** 令：

\[
V_{\max}=\frac{R_{\max}}{1-\gamma},
\qquad
\Delta=\|V_1^*-V_2^*\|_\infty.
\]

对任意 `h`：

\[
\begin{aligned}
|V_1^*(h)-V_2^*(h)|
&=
|T_1V_1^*(h)-T_2V_2^*(h)|\\
&\le
\max_a
\Big(
|r_1-r_2|
+\gamma|\mathbb E_{P_1}V_1^*-\mathbb E_{P_2}V_2^*|
\Big).
\end{aligned}
\]

将期望差拆开：

\[
|\mathbb E_{P_1}V_1^*-\mathbb E_{P_2}V_2^*|
\le
|\mathbb E_{P_1}(V_1^*-V_2^*)|
+
|\mathbb E_{P_1}V_2^*-\mathbb E_{P_2}V_2^*|.
\]

第一项不超过 `Delta`。由于 `|V_2^*|≤V_max`，按 total variation 的对偶界，第二项不超过：

\[
2V_{\max}\epsilon_p.
\]

因此：

\[
\Delta
\le
\epsilon_r+
\gamma\Delta+
2\gamma V_{\max}\epsilon_p.
\]

移项：

\[
\Delta
\le
\frac{\epsilon_r+2\gamma V_{\max}\epsilon_p}{1-\gamma}
=
\frac{\epsilon_r}{1-\gamma}
+
\frac{2\gamma R_{\max}}{(1-\gamma)^2}\epsilon_p.
\]

`Q` 的证明将第一步动作固定后完全相同。证毕。

**含义。** latent 不需要恢复完整伙伴身份；只要奖励与受控转移误差小，控制价值误差就有界。

---

## 定理 4：一套 conditional critic 足以表达连续伙伴族

**定理。** 假设 `X×Z` 为紧集，主体动作集合有限；奖励 `r(X,z,a)` 连续，转移核对 `(X,z)` 满足 Feller 连续性。则 `Q^*(X,z,a)` 连续。对任意 `epsilon>0`，存在一套共享 ReLU 网络 `Q_omega(X,z,a)`，满足：

\[
\sup_{X,z,a}|Q_\omega(X,z,a)-Q^*(X,z,a)|<\epsilon.
\]

**证明。** 从 `Q_0=0` 开始。Bellman 算子：

\[
(TQ)(X,z,a)
=
r(X,z,a)
+
\gamma
\mathbb E
\max_{a'}Q(X',z',a').
\]

有限动作上的最大值保持连续；Feller 连续性保证对连续有界函数的条件期望关于 `(X,z)` 连续。因此 `T` 将连续有界函数映射到连续有界函数。

`T` 在 sup norm 下是 `gamma` 压缩，所以 `Q_n=T^nQ_0` 一致收敛到唯一固定点 `Q^*`。连续函数的一致极限连续，因此 `Q^*` 连续。紧集上的连续函数可由 ReLU 网络一致逼近，故结论成立。证毕。

**边界。** 该定理是条件性存在结果；真正需要实证检验的是低维 `Z` 是否存在，而不是网络是否具有通用逼近能力。

---

## 命题 2：连续后验决策遗憾的性质

定义：

\[
\mathcal R(b)
=
\mathbb E_b\max_aQ(z,a)
-
\max_a\mathbb E_bQ(z,a).
\]

则：

1. `R(b)≥0`；
2. 若存在动作 `a*` 对 `b` 几乎所有 `z` 都是最优动作，则 `R(b)=0`；
3. 对有限动作，若 `R(b)=0`，则至少存在一个动作对 `b` 几乎所有 `z` 都是最优动作；
4. 任意保持 latent pushforward posterior 不变的身份细分不会改变 `R(b)`。

**证明。** 第一条来自：

\[
\max_a\mathbb E_bQ(z,a)
\le
\mathbb E_b\max_aQ(z,a).
\]

第二条：若共同最优动作 `a*` 存在，则：

\[
\mathbb E_b\max_aQ(z,a)
=
\mathbb E_bQ(z,a^*)
=
\max_a\mathbb E_bQ(z,a).
\]

第三条：设 `a_bar` 达到右侧最大值。若在正概率集合上 `a_bar` 非最优，则：

\[
\max_aQ(z,a)-Q(z,a_{bar})>0
\]

在正概率集合上成立，其期望严格大于 0，与 `R=0` 矛盾。因此 `a_bar` 几乎处处最优。第四条直接由 `R` 只依赖 latent posterior 得到。证毕。

---

## 定理 5：表示、后验和 critic 误差到控制遗憾

**定理。** 令真实 Bayes-adaptive optimal critic 为：

\[
Q_B^*(X,b,a),
\]

回报绝对值上界对应 `|Q_B^*|≤V_max`。假设：

1. latent factorization 引起的 uniform Q error 不超过 `epsilon_M`；
2. learned critic 对近似 belief MDP 的 uniform error 不超过 `epsilon_Q`；
3. learned posterior 满足：

\[
\operatorname{TV}(b,\hat b)\le\epsilon_b.
\]

令：

\[
\hat a=
\arg\max_a\widehat Q_B(X,\hat b,a).
\]

则单步 Bayes action regret 满足：

\[
Q_B^*(X,b,a^*)-Q_B^*(X,b,\hat a)
\le
2(\epsilon_M+\epsilon_Q+2V_{\max}\epsilon_b).
\]

若该界对所有可达 belief states 成立，则对贪心策略：

\[
J^*-J^{\hat\pi_{\mathrm{greedy}}}
\le
\frac{2(\epsilon_M+\epsilon_Q+2V_{\max}\epsilon_b)}{1-\gamma}.
\]

对实际随机 actor，若其相对 learned critic 的逐状态优化误差满足：

\[
\epsilon_\pi
=\max_a\widehat Q_B(X,\hat b,a)
-\mathbb E_{a\sim\pi_\theta}
\widehat Q_B(X,\hat b,a),
\]

则：

\[
J^*-J^{\pi_\theta}
\le
\frac{2(\epsilon_M+\epsilon_Q+2V_{\max}\epsilon_b)+\epsilon_\pi}{1-\gamma}.
\]

**证明第一步：Bayes critic 关于 belief 的 Lipschitz 性。** 固定第一动作 `a`。对任意后续历史策略 `pi`，令隐藏 context 为 `z` 时的折扣回报为 `G^pi(z)`，有 `|G^pi(z)|≤V_max`。因此：

\[
|\mathbb E_bG^\pi-\mathbb E_{\hat b}G^\pi|
\le
2V_{\max}\operatorname{TV}(b,\hat b).
\]

对所有固定第一动作为 `a` 的策略取上确界，得到：

\[
|Q_B^*(X,b,a)-Q_B^*(X,\hat b,a)|
\le
2V_{\max}\epsilon_b.
\]

**第二步：总 action-value 误差。** 由三角不等式：

\[
|\widehat Q_B(X,\hat b,a)-Q_B^*(X,b,a)|
\le
\epsilon_M+\epsilon_Q+2V_{\max}\epsilon_b
=:\delta.
\]

**第三步：贪心动作遗憾。** 令 `a*` 为真实最优动作。则：

\[
\begin{aligned}
Q_B^*(a^*)-Q_B^*(\hat a)
&\le
[Q_B^*(a^*)-\widehat Q(a^*)]
+[\widehat Q(a^*)-\widehat Q(\hat a)]\\
&\quad+
[\widehat Q(\hat a)-Q_B^*(\hat a)].
\end{aligned}
\]

中间项不大于 0，因为 `hat a` 对 learned critic 贪心；首尾各不超过 `delta`，故单步 regret 不超过 `2delta`。

**第四步：回报界。** 令：

\[
D(s)=V^*(s)-V^{\hat\pi}(s).
\]

对贪心策略，由单步 Bellman suboptimality：

\[
D(s)
\le
2\delta+
\gamma\mathbb E D(s').
\]

取 sup：

\[
\|D\|_\infty
\le
2\delta+
\gamma\|D\|_\infty,
\]

所以：

\[
\|D\|_\infty
\le
\frac{2\delta}{1-\gamma}.
\]

对随机 actor，先写：

\[
Q_B^*(a^*)-\mathbb E_{a\sim\pi_\theta}Q_B^*(a)
\le 2\delta+\epsilon_\pi,
\]

再应用相同递推，即得随机 actor 的回报界。证毕。

---

## 定理 6：决策遗憾势函数不改变最优策略

**定理。** 在 augmented belief MDP 中，令任意有界势函数 `Phi(s)` 在终止状态为 0。定义 shaped reward：

\[
r'(s,a,s')
=
r(s,a,s')
+
\lambda[\gamma\Phi(s')-\Phi(s)].
\]

则任意策略从固定初始状态的 shaped return 与原 return 只相差常数 `-lambda Phi(s_0)`，因此最优策略集合不变。

**证明。** 对轨迹求和：

\[
\begin{aligned}
G'
&=
\sum_{t=0}^{\infty}\gamma^t
\left[r_t+\lambda(\gamma\Phi_{t+1}-\Phi_t)\right]\\
&=
G+
\lambda
\sum_{t=0}^{\infty}
(\gamma^{t+1}\Phi_{t+1}-\gamma^t\Phi_t).
\end{aligned}
\]

后项望远镜相消：

\[
G'=G-\lambda\Phi_0+
\lambda\lim_{T\to\infty}\gamma^{T+1}\Phi_{T+1}.
\]

由于 `Phi` 有界且 `gamma<1`，极限为 0；有限回合终止势为 0 时也为 0。因此：

\[
G'=G-\lambda\Phi_0.
\]

初始状态固定时，差值与策略无关，故策略排序与最优策略集合不变。证毕。

**应用。** 取 `Phi=-R_dec`，shaping 奖励决策遗憾下降，但不把“辨认伙伴”本身变成新的任务目标。

---

## 命题 3：split-replica 评价消除动作选择偏差

**命题。** 对一个固定 anchor state，fit replicas 与 evaluation replicas 条件独立。令：

\[
\hat a=f(D_{\mathrm{fit}})
\]

是任意由 fit 数据选择的动作。evaluation mean 为：

\[
\widehat G_{\mathrm{eval}}(\hat a).
\]

则：

\[
\mathbb E
\left[
\widehat G_{\mathrm{eval}}(\hat a)
\mid D_{\mathrm{fit}}
\right]
=
G(\hat a),
\]

从而 evaluation mean 对所选动作的真实价值无选择偏差。

**证明。** 条件于 `D_fit` 后，`hat a` 是固定值。evaluation replicas 与 fit 数据独立，且每个 evaluation return 对该动作真实期望无偏，因此条件期望等于 `G(hat a)`。再对 `D_fit` 取全期望得到结论。证毕。

---

## 定理 7：Partner-run-block conformal 适应门

**定理。** 假设 calibration partner runs 与未来测试 partner run 在 run 层面交换。对每个 calibration run `j`，定义 block score：

\[
S_j
=
\max_{(h,a)\in\mathcal I_j}
|\widehat Q(h,a)-\bar G_j(h,a)|.
\]

共有 `n` 个 calibration runs。令：

\[
k=\lceil(n+1)(1-\alpha)\rceil,
\]

`q_alpha` 为 calibration scores 的第 `k` 小顺序统计量；若 `k>n` 则取无穷。则对新的交换 partner run：

\[
P(S_{n+1}\le q_\alpha)\ge1-\alpha.
\]

因此该事件上，新 run 的所有登记状态与动作同时满足经验 continuation coverage。

**证明。** 在交换性下，`S_{n+1}` 在 `n+1` 个 scores 中的秩均匀。其秩不超过 `k` 的概率至少为 `k/(n+1)≥1-alpha`。使用仅由前 `n` 个 score 构造的保守顺序统计量得到结论； ties 使覆盖更保守。由于 score 对 run 内所有登记索引取最大值，事件 `S_{n+1}≤q_alpha` 同时覆盖全部索引。证毕。

若每个 empirical evaluation mean 到真实期望的误差以概率 `1-delta` 同时不超过 `epsilon_MC`，则联合概率至少 `1-alpha-delta` 时：

\[
|\widehat Q-G|\le q_\alpha+\epsilon_{\mathrm{MC}}.
\]

因此若 conditional 与 base policy 的预测差满足：

\[
\widehat V_c-
\widehat V_0
>
2(q_\alpha+\epsilon_{\mathrm{MC}}),
\]

则在覆盖事件上：

\[
V_c>V_0.
\]

证明仅需分别对 `V_c,V_0` 使用误差界并相减。

---

# 19. 可证伪科学假设

## H1：低维决策结构

在 run-disjoint 伙伴上，真实 counterfactual action signatures 的谱满足稳定低有效秩；固定维度 `d` 的 latent model 能在 held-out partner runs 上重建 action-value ordering。

## H2：共享条件策略可表达

在完全未见伙伴上，privileged teacher latent 条件下的单一 shared actor 显著优于 context-free robust base：

\[
J_{\mathrm{oracle\ context}}-J_{\mathrm{base}}>0.
\]

若 H2 不成立，说明单一条件 actor 的表达机制或训练伙伴分布不足。

## H3：合法历史可恢复

在线 posterior actor 回收 oracle context 增益的显著比例：

\[
\rho_{\mathrm{recover}}
=
\frac{J_{\mathrm{online}}-J_{\mathrm{base}}}
{J_{\mathrm{oracle}}-J_{\mathrm{base}}}.
\]

只有分母明确为正时报告该比率；原始回报差是主要终点。

## H4：端到端 ZSC

在 mechanism-disjoint partner families 上，DELTA-ZSC 的平均回报、worst-family 回报和 BR-Prox 均优于强 ZSC 与 meta-RL 基线，且 negative-transfer rate 不高于预登记门槛。

## H5：只利用决策相关信息

在伙伴容易辨认但不改变最优动作的受控条件中：

- decision-regret 接近 0；
- DELTA-ZSC 不产生额外探查成本；
- identity-prediction baseline 仍可能探查；
- actor 输出对身份细分保持不变。

---

# 20. 实验设计

## 20.1 受控有限任务

正交操纵：

1. 伙伴是否行为可区分；
2. 伙伴是否改变主体最优动作；
3. 证据出现时间是否早于不可逆决策；
4. 测试 latent 是否位于训练支持内。

四个核心条件：

| 条件 | 预期 |
|---|---|
| 易区分、价值相关 | online posterior 收缩，actor 适应，回报提升 |
| 易区分、价值无关 | latent 可不同，但 decision regret 与行为差异接近 0 |
| 难区分、价值相关 | oracle 有收益，online 恢复率随证据强度下降 |
| 证据晚于承诺点 | oracle 有收益，online 无端到端收益 |

该任务必须可精确枚举 Bayes value，用于核对定理 1、2、5、6。

## 20.2 OvercookedV2

主布局：Test Time Simple。
冻结复现：Test Time Wide。

训练伙伴支持：

- 连续 generator 的未见 code；
- 独立 generator seeds；
- frozen SP / OP / BR-diverse partners；
- generator snapshot archive。

确认性测试必须包含完全未参与：

- 表示学习；
- partner generator 更新；
- hyperparameter selection；
- conformal calibration；
- stopping rule；

的 partner mechanisms。

## 20.3 三层泛化

1. **Code interpolation：** 同一 generator 未见 code；
2. **Run-disjoint：** 独立初始化、独立 generator training run；
3. **Mechanism-disjoint：** 未见训练算法、目标或约定生成机制。

只有第三层可以支持广义 ZSC 主张。

## 20.4 Hanabi

使用相同：

- 连续 partner posterior；
- shared actor/critic；
- decision-equivalent anchor；
- decision-regret potential；
- run-block calibration。

领域特定 observation/action encoder 可以变化，但方法计算对象不得变化。

---

# 21. Baselines

至少包含：

1. Self-Play / recurrent IPPO；
2. Other-Play；
3. Fictitious Co-Play；
4. BRDiv / BDP 类 diversity 方法；
5. E3T；
6. PACE；
7. PEARL-style probabilistic context actor–critic；
8. VariBAD / Bayes-adaptive recurrent meta-RL；
9. ContraBAR-style contrastive Bayes representation；
10. 普通 recurrent actor，无显式 partner loss；
11. 当前 V4.4 discrete-slot 系统；
12. finite policy-library router；
13. DELTA-ZSC 各项消融。

所有 baseline 必须使用相同训练 partner budget、环境步、网络参数量级和确认性 partner splits。

---

# 22. 主要指标

- Held-out cross-play raw return；
- Worst partner-mechanism return；
- BR-Prox；
- Negative-transfer rate：conditional policy 低于 base 的伙伴比例；
- Oracle-context gain；
- Online-context gain；
- Context recovery ratio；
- Decision regret over time；
- Posterior calibration；
- Conformal empirical coverage；
- 任务交付、错误交付与推理延迟；
- 参数量与训练环境步。

---

# 23. 决定性消融

1. Discrete slots vs continuous posterior；
2. 单点 latent vs uncertainty-aware posterior；
3. 简单 concat vs low-rank modulation；
4. 仅响应预测 vs +真实 CF anchor；
5. 无 quotient geometry；
6. identity/code prediction 替代 decision loss；
7. 无 information bottleneck；
8. 固定伙伴池 vs continuous partner generator；
9. trajectory diversity vs best-response diversity；
10. 无 decision-regret shaping；
11. identity-information shaping vs decision-regret shaping；
12. 无 privileged teacher；
13. 无 conformal gate；
14. base-only；
15. oracle latent；
16. action-augmented observation。

---

# 24. 统计推断

主要独立单位：

- ego training run；
- partner training run；
- partner generation mechanism。

Episode 只是配对均值的重复测量，不能当成 ZSC 泛化样本。

推荐：

1. ego-run 与 partner-run 双向 node bootstrap；
2. mechanism 保持固定分层；
3. 每次重采样重新计算模型均值、worst-family 与 negative-transfer；
4. 9,999 或以上有效 bootstrap；
5. 确认性阈值与随机种子在测试前冻结；
6. calibration partner runs 与 confirmatory partner runs 完全分离。

---

# 25. 预登记门与停机规则

## 25.1 结构门

在训练任何正式 DELTA-ZSC 前，用真实 CF signatures 检查：

- held-out reconstruction error；
- effective rank；
- run-disjoint nearest-neighbor transfer；
- value-equivalent identity collapse。

若固定维度 `d_max` 仍不能在 run-disjoint partners 上恢复动作排序，则“不存在当前分布下的低维可迁移结构”，项目不得继续通过增加网络容量解释。

## 25.2 Oracle-control 门

若 oracle latent shared actor 不优于 base，则不允许把 online 失败归因于 posterior inference。

## 25.3 Online inference 门

只有 oracle gain 明确为正时，才读取 recovery ratio。

## 25.4 外部有效性门

Simple、Wide、mechanism-disjoint panel 和 Hanabi 必须分别报告。不得用一项大收益抵消另一项失败。

---

# 26. 代码重构规格

## 26.1 主目录

```text
src/path_c/
├── experiment.py
├── types.py
├── task_encoder.py
├── belief_encoder.py
├── belief_set_encoder.py
├── universal_actor.py
├── universal_critic.py
├── response_decoder.py
├── teacher_context.py
├── counterfactual_anchor.py
├── decision_geometry.py
├── regret_potential.py
├── partner_generator.py
├── calibration.py
├── training.py
├── runner.py
├── evaluation.py
├── statistics.py
└── storage.py
```

旧系统整体迁移：

```text
src/path_c_v44_legacy/
```

主 CLI 不得导入 legacy 训练模块。

## 26.2 核心类型

```python
class BeliefState(NamedTuple):
    recurrent_carry: Any
    mixture_logits: Any
    means: Any
    log_variances: Any
    support_score: Any

class PolicyState(NamedTuple):
    task_carry: Any
    belief_state: BeliefState
    previous_action: Any
    previous_reward: Any
    episode_start: Any

class ModelOutput(NamedTuple):
    task_features: Any
    belief_embedding: Any
    base_logits: Any
    residual_logits: Any
    gate: Any
    execution_logits: Any
    state_value: Any
    action_values: Any
    response_parameters: Any

class CounterfactualAnchorBatch(NamedTuple):
    legal_histories: Any
    teacher_latents: Any
    fit_returns_by_action: Any
    evaluation_returns_by_action: Any
    partner_run_ids: Any
    replica_split: Any
```

## 26.3 必须删除的主路径对象

```text
slot_count
slot_log_belief
TwinDuelingQ per-slot heads
responsibility
behavior_slot
behavior_estimator
CodebookState
response_code
target_response_signatures
next_q_use_targets
next_q_mask_targets
bellman_control_values
J_use / J_mask 自蒸馏控制
```

## 26.4 Config v5

```yaml
version: 5

model:
  task_hidden_dim: 128
  belief_hidden_dim: 128
  latent_dim: 8
  mixture_components: 4
  posterior_particles: 8
  modulation_rank: 8

loss:
  value_weight: 0.5
  entropy_weight: 0.01
  response_weight: 1.0
  counterfactual_weight: 1.0
  advantage_distill_weight: 0.5
  policy_distill_weight: 0.25
  quotient_weight: 0.25
  consistency_weight: 0.25
  information_bottleneck_weight: 0.001
  decision_regret_weight: 0.1

anchors:
  interval_updates: 8
  states_per_interval: 64
  fit_replicas: 16
  evaluation_replicas: 16
  continuation_horizon: 400

partner_generator:
  code_dim: 8
  generator_update_interval: 16
  competence_threshold: <from preregistration>
  cvar_level: 0.2
  brdiv_weight: 1.0
  snapshot_interval: 32

calibration:
  alpha: 0.05
  block_unit: partner_run
  support_threshold: <fit on calibration only>
```

具体默认值只能作为 development 起点；正式值必须通过 development runs 冻结。

---

# 27. 工程不变量测试

1. 主代码中不存在 slot、responsibility 或 codebook；
2. actor/critic 参数树不含 partner-specific 分支；
3. partner ID、seed、algorithm label 不进入模型输入；
4. teacher latent 仅在训练 lane 使用；
5. evaluation partner run 不进入训练、校准或 early stopping；
6. anchor 分支前环境、伙伴 carry 与 RNG lineage 完全相同；
7. fit/evaluation replicas 无交集；
8. CF targets 来自 simulator raw return，不来自 model Q；
9. decision-regret shaping 对 encoder/critic stop-gradient；
10. terminal potential 固定为 0；
11. conformal scores 按 partner run block 构造；
12. gate 关闭时 execution logits 与 base logits 逐元素相同；
13. 部署 artifact 不包含 generator、teacher 或 simulator state；
14. exact checkpoint / parent run / co-training lineage overlap 使实验立即失败；
15. run-level bootstrap 不把 episode 当成独立伙伴样本。

---

# 28. 预期论文贡献

1. **连续决策商空间。** 把不可枚举伙伴身份压缩为固定维度、由真实动作价值定义的连续 latent，而不是有限 partner types 或策略库。
2. **一套共享 Bayes-adaptive actor–critic。** 参数量不随伙伴数量增长，并显式利用 posterior uncertainty。
3. **真实反事实决策监督。** 用 simulator continuation return 校正 latent 与 critic，消除 target-Q 自蒸馏闭环。
4. **决策导向探索。** 使用 identity-invariant decision regret potential 改善信息获取信用分配，并证明策略不变性。
5. **能力约束的连续伙伴生成。** 训练伙伴多样性由主体 best-response 差异而不是轨迹表面差异定义。
6. **可校准的 OOD 回退。** 通过 partner-run-block conformal gate 控制未见伙伴上的负迁移。
7. **可定位的失败分解。** 将最终失败分成低维结构不存在、历史不可识别、共享策略不可表达、critic 误差和 OOD 覆盖不足。

---

# 29. 局限性

1. 方法依赖低维决策结构；不存在该结构时不能保证泛化；
2. 真实 counterfactual anchors 依赖可克隆模拟器，现实人类伙伴无法直接提供；
3. generator 仍可能与 ego 共适应，因此必须使用独立冻结伙伴和 mechanism-disjoint 测试；
4. conformal coverage 依赖 partner-run exchangeability，不能自动外推到任意现实伙伴；
5. Bayes belief 是近似的，理论界中的 posterior error 需要独立估计；
6. potential shaping 保持最优策略不变，但不保证深度 PPO 的优化过程必然改进；
7. 单一 latent 维度不是普适常数，必须由任务分布决定；
8. Hanabi 与 OvercookedV2 的 observation encoder 不同，跨领域主张只限于共同算法对象而非共享权重。

---

# 30. 相关工作定位

DELTA-ZSC 与以下方向有关但不等同：

- PEARL：概率上下文变量与单一条件 actor–critic；
- VariBAD：belief-conditioned Bayes-adaptive policy 与结构化探索；
- ContraBAR：对比表示作为 Bayes 控制充分统计量；
- UVFA：单一函数在条件变量上的价值泛化；
- BRDiv 与 ZSC-Eval：按 best-response 差异构造与评估伙伴多样性；
- E3T：不依赖大型预训练伙伴群体的端到端 ZSC；
- PACE：通过上下文感知探索加快未知伙伴适应；
- Partner Modelling Emerges：只有伙伴差异真正改变任务分配时，结构化伙伴表示才会出现；
- Cross-Environment Cooperation：跨任务分布训练有助于形成更一般的合作规范；
- OvercookedV2：构造仅靠状态覆盖无法解决、需要在线适应的 ZSC 情景。

DELTA-ZSC 的区别在于：潜变量由**决策等价与真实反事实动作价值**定义；探索奖励采用**策略不变的决策遗憾势函数**；训练伙伴由**能力约束下的连续 best-response diversity**生成；部署适应由**run-block conformal gate**控制。

---

# 31. 参考文献建议

1. Rakelly et al. Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables. ICML 2019.
2. Zintgraf et al. VariBAD: A Very Good Method for Bayes-Adaptive Deep RL via Meta-Learning. ICLR 2020.
3. Choshen and Tamar. ContraBAR: Contrastive Bayes-Adaptive Deep RL. ICML 2023.
4. Schaul et al. Universal Value Function Approximators. ICML 2015.
5. Rahman et al. Generating Teammates for Training Robust Ad Hoc Teamwork Agents via Best-Response Diversity. TMLR 2023.（BRDiv）
6. Wang et al. ZSC-Eval: An Evaluation Toolkit and Benchmark for Multi-agent Zero-shot Coordination. NeurIPS 2024.
7. Yan et al. An Efficient End-to-End Training Approach for Zero-Shot Human-AI Coordination. NeurIPS 2023.
8. Ma et al. Fast Peer Adaptation with Context-aware Exploration. ICML 2024.
9. Mon-Williams et al. Partner Modelling Emerges in Recurrent Agents (But Only When It Matters). NeurIPS 2025.
10. Jha et al. Cross-environment Cooperation Enables Zero-shot Multi-agent Coordination. ICML 2025.
11. Gessler et al. OvercookedV2: Rethinking Overcooked for Zero-Shot Coordination. 2025.
12. Ng, Harada, and Russell. Policy Invariance Under Reward Transformations. ICML 1999.

---

# 32. 最终方法定义

DELTA-ZSC 不是伙伴分类器，也不是有限策略路由器。它是一个固定容量的 amortized Bayes-adaptive controller：

\[
\boxed{
H_t
\rightarrow
q_\phi(z_t\mid H_t)
\rightarrow
\pi_\theta(a_t\mid X_t,q_\phi)
}
\]

其中：

\[
\boxed{
\text{无限伙伴身份}
\rightarrow
\text{连续决策等价商空间}
}
\]

\[
\boxed{
\text{多个 partner-specific actor/critic}
\rightarrow
\text{一套共享低秩条件 actor + 一套共享 universal critic}
}
\]

\[
\boxed{
\text{target-Q 自蒸馏}
\rightarrow
\text{真实共同随机数 continuation return}
}
\]

\[
\boxed{
\text{身份信息探索}
\rightarrow
\text{策略不变的决策遗憾 shaping}
}
\]

\[
\boxed{
\text{固定伙伴类别}
\rightarrow
\text{能力约束的连续 best-response-diverse partner process}
}
\]

只有当低维结构、历史可识别性和共享控制可表达性在 run-disjoint 与 mechanism-disjoint 伙伴上联合成立时，项目才可以声称构建了真正有效且可扩展的 ZSC。
