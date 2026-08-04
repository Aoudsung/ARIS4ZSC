# SCIENTIFIC_SPEC：Unified DELTA-ZSC 科学问题

`authoritative: true`

## 1. 研究对象

研究环境为固定 Official commit 的 OvercookedV2 Test-Time Protocol Formation：两个智能体在局部可观测、有限时域、共享团队回报的任务中协作；ego 在训练时接触一组伙伴，部署时与 lineage-disjoint 的陌生伙伴交互。

核心困难不是识别伙伴的算法名称或训练 seed，而是：

> 在只能读取合法局部交互历史时，判断哪些伙伴响应差异会改变 ego 当前可选动作的长期团队价值，并据此在不破坏基础任务能力的前提下在线调整行为。

## 2. 合法信息边界

部署时 ego 在时刻 `t` 只能读取：

- 当前局部 observation `o_t^ego`；
- 自身过去执行的动作；
- episode boundary；
- 由这些量递归计算的内部状态。

部署图禁止读取：

- 伙伴动作标签；
- 伙伴 ID、算法、seed、checkpoint stage 或 lineage；
- 全局环境状态；
- 另一个智能体的私有 observation；
- 训练期 counterfactual returns；
- comparator、oracle、partner class 或人工协议标签。

训练时允许使用 simulator snapshot 和 common-random-number continuation 产生 privileged decision observation，但该信息只能监督潜变量模型，不能进入 deployment state。

## 3. 四类状态对象

### 3.1 任务状态 `x_t`

`x_t=f_task(H_t)` 是 task-only recurrent representation。task encoder 的直接输入清除显式 other-agent planes；伙伴通过改变锅、物体、订单和通路而产生的合法任务后果仍可进入 `x_t`。

### 3.2 即时伙伴几何 `r_t`

`r_t=f_instant(o_t^partner-planes)` 只使用当前 observation 中可见的伙伴位置、方向和 inventory planes，不含 recurrent carry。它支持碰撞规避、让位和当前操作时机，不承担跨时刻伙伴适应。

### 3.3 解析行为统计 `u_t`

`u_t` 不是 learned latent。它由四个 Beta-Bernoulli posterior 的均值与 precision 构成：

1. 当前帧可见率；
2. 前后均可见时的移动率；
3. 可见时的持物率；
4. 前后均可见时的 inventory-change 率。

该对象只概括可复算的长期行为统计，不需要一致性、语义预测、variance floor 或 covariance loss。

### 3.4 潜在协调模式 `z_t`

`z_t∈{1,...,K}` 是 exchangeable local coordination mode。其唯一语义是：

> 在给定 `(x_t,r_t,u_t)` 后，`z_t` 同时决定伙伴下一步可观测响应的分布，以及 ego 各动作 continuation value 的分布。

`z_t` 不是伙伴身份、算法类别、人类命名策略或固定 episode type。component index 可交换，跨 seed 只能在 permutation 后比较。

部署 belief 为：

\[
b_t(k)=P(z_t=k\mid H_t^{legal}).
\]

## 4. 决策等价 estimand

对合法历史 `H_t` 和 ego 首动作 `a`，定义：

\[
G_H(H_t,a)
=
\mathbb E\left[
\sum_{\tau=0}^{H-1}\gamma^\tau r_{t+\tau}^{raw}
\mid H_t,a_t=a,\pi_0,\pi^{-ego}
\right].
\]

其中：

- `H` 是配置中的 `continuation_horizon`；
- `gamma` 与 PPO discount 一致；
- 第一步强制为 `a`；
- 后续 ego 使用 frozen base policy `pi_0`；
- 伙伴延续其真实 recurrent state；
- 所有动作分支共享 CRN replica roots；
- 不使用 learned endpoint bootstrap。

中心化 action-value signature 为：

\[
A_H(H_t,a)
=
G_H(H_t,a)
-
\frac{1}{|\mathcal A|}
\sum_{a'}G_H(H_t,a').
\]

两个历史在当前任务状态下决策等价，当且仅当其 `A_H` 相同；该定义不要求伙伴身份相同，也不要求下一步运动学相同。

## 5. 单一生成假设

Unified DELTA 假设：在局部时间尺度上，伙伴响应 `y_t` 与 counterfactual decision observation `A_t` 是同一潜在模式的两类条件观测：

\[
p(y_t,A_t,z_t\mid x_t,r_t,u_t,a_{t-1})
=
p(z_t\mid z_{t-1})
  p_\theta(y_t\mid z_t,x_t,r_t,u_t,a_{t-1})
  p_\psi(A_t\mid z_t,x_t,r_t,u_t).
\]

普通 rollout 只观测 `y_t`；anchor state 额外观测带 replica uncertainty 的 `A_t`。两种 evidence 在一个 forward marginal likelihood 中共同确定 `z_t`，不存在后验 pseudo-label、pair comparator 或独立 separation geometry。

## 6. 训练与部署的非对称性

训练时：

- base policy `pi_0(a|x_t,r_t)` 由 PPO 学习；
- latent model 在 base-policy rollout 上学习；
- response evidence 每个有效 transition 都存在；
- decision evidence 只在稀疏 CRN anchors 存在；
- base PPO 与 latent MLE 使用独立参数树、optimizer 和 Adam moments。

部署时：

- 不再有 decision anchors；
- belief 只由合法 response evidence 更新；
- decision emission 提供各模式的 action values；
- 通过解析 KL mirror solution 修正 `pi_0`；
- full 版本进一步由同一 response model 计算 myopic Bayes VOI。

这是一种合法的 privileged-training / observation-only-deployment 设计：训练期 privileged information建立 latent 的决策语义，但部署策略不读取该信息。

## 7. 中心科学命题

### H1：性能命题

在相同 Official 环境、训练伙伴分布、资源披露和 lineage-disjoint confirmatory panel 上，Full DELTA 的 XP 高于最强外部基线，并达到至少一次正确交付的物质效果尺度。

### H2：机制命题

在相同 base policy、数据、模型容量和评估 keys 下，joint response-decision latent model 优于 response-only latent model。该命题直接检验 privileged decision emission 是否是方法的有效创新，而非额外历史网络或参数量。

### H3：因果命题

固定 source world、伙伴、任务表示、行为统计、decision-emission geometry、base policy 和 all-action source returns，只替换 belief 时，正确 belief 的策略相对 task-matched shuffled belief 具有正的 source-world decision value：

\[
\mathbb E
\left[
(\pi_{correct}-\pi_{shuffle})^\top G_{source}
\right] > 0.
\]

## 8. 可证伪条件

以下任一结果均可否定相应主张，且不得通过追加模块修复后继续使用同一注册结果：

- Full 未超过强外部基线：H1失败；
- joint 未超过 response-only：H2失败；
- belief transplant 的 source-world value LCB 不高于0：H3失败；
- decision emission 在 independent evaluation replicas 上不能预测 action ordering：决策语义不成立；
- response component residual 对 held-out predictive log score 无增量：合法 response 无法支持 latent inference；
- belief长期均匀或单一 component 支配且 decision gain 不存在：latent model未成功识别；
- analytic adaptation频繁达到KL边界但无回报增益：decision geometry与真实控制不一致；
- VOI无增量：主动信息价值在该substrate/模型下无效。

## 9. 创新性边界

以下内容单独均不构成新颖贡献：历史编码、categorical belief、Bayesian filter、latent partner representation、context-conditioned policy或多样伙伴训练。

本项目的潜在贡献是三者的统一：

1. **decision-emitting latent coordination model**：真实CRN all-action continuation作为稀疏decision emission，与合法response共享latent cause；
2. **observation-only Bayesian deployment**：训练时建立决策语义，部署时仅用合法伙伴响应滤波；
3. **analytically KL-bounded Bayes adaptation**：从posterior decision value和response model直接导出策略修正与VOI，不训练第二个适应actor，也不使用information-bonus weight。

只有H1、H2、H3形成闭环时，才可以将该组合声明为论文贡献。

## 10. 研究范围

当前研究只针对：

- 两智能体 cooperative OvercookedV2；
- 6个离散动作；
- 400-step episode；
- view radius 2；
- test_time_simple 与 test_time_wide；
- fixed-horizon one-action continuation estimand。

不声称：

- component具有人类可命名语义；
- learned transition是真实伙伴心理状态；
- myopic VOI等价于完整Bayes-adaptive planning；
-结果可直接推广到连续动作、多人或非合作任务；
-超过Official旧基线即等于2026领域SOTA。
