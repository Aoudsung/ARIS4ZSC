# THEORY：Unified DELTA-ZSC 理论、推导与边界

`authoritative: true`

本文件只证明由当前联合概率模型和KL约束优化直接导出的性质。所有结论相对于 `SCIENTIFIC_SPEC.md` 的estimand与 `METHOD_SPEC.md` 的实现成立。

## 1. 联合状态空间模型

令：

- `z_t∈{1,...,K}` 为exchangeable coordination mode；
- `T_{jk}=P(z_{t+1}=k|z_t=j)`；
- `y_{t+1}` 为合法可观测伙伴response；
- `A_t` 为训练anchor上的centered continuation signature；
- `c_t=(x_t,r_t,u_t)` 为已知条件。

模型为：

\[
p(z_{0:T},y_{1:T},A_{\mathcal I}\mid c_{0:T},a_{0:T-1})
=
p(z_0)
\prod_{t=0}^{T-1}
T_{z_tz_{t+1}}
 p_\theta(y_{t+1}\mid z_{t+1},c_t,a_t)
\prod_{t\in\mathcal I}
p_\psi(A_t\mid z_t,c_t).
\]

其中 `I` 是稀疏anchor集合。普通response与privileged decision observation并非两个loss，而是同一joint likelihood中的两类观测。

## 2. 精确forward recursion

给定严格正的row-stochastic `T`、有限emission log likelihood和概率向量 `b_t`，prediction为：

\[
\bar b_{t+1}(k)=\sum_jb_t(j)T_{jk}.
\]

correction为：

\[
b_{t+1}(k)
=
\frac{
\bar b_{t+1}(k)L^y_{t+1,k}(L^A_{t+1,k})^{I_{t+1}}
}
{
\sum_l\bar b_{t+1}(l)L^y_{t+1,l}(L^A_{t+1,l})^{I_{t+1}}
}.
\]

### 命题1：概率闭合

若 `b_t` 和 `T` 各行均为概率分布，且所有likelihood有限为正，则 `b_{t+1}` 严格为正且和为1。

**证明。** `bar b` 是概率向量与row-stochastic matrix的乘积，非负且和为1。严格正的 `T` 使每个分量严格正。与正likelihood相乘后归一化常数有限且为正，因此posterior严格正且归一。□

episode start使用uniform initial prior，不应用虚构的pre-episode transition。rollout内真实episode transition则按同一recursion处理。

## 3. 为什么负visibility必须进入likelihood

response factorization写为：

\[
p(y\mid z)
=p(v\mid z)
[p(pos,dir,inv\mid v=1,z)]^v
[p(event\mid v_{t-1}=v_t=1,z)]^{m_{event}}.
\]

当 `v=0` 时，position/direction/inventory factor被mask，但 `p(v=0|z)`仍进入likelihood。因此“伙伴不可见”是合法负证据，不应把整次filter update设为identity。

transition描述physical-time latent dynamics，与observation missingness无关；即使部分response factor缺失，Chapman–Kolmogorov prediction仍然发生。

## 4. Component置换不变性

对任意permutation matrix `P`，同时变换：

\[
b'_t=b_tP,
\quad
T'=P^TTP,
\]

并对response/decision emission的component axis应用同一permutation，则joint likelihood、marginal evidence、expected action value和VOI保持不变。

### 命题2：index不可识别

component index没有全局语义；模型最多在permutation意义下可识别。

因此禁止：

- 把index映射为SP/OP/SA/FCP；
- 跨seed直接比较component 1；
- 使用component classification accuracy作为主指标。

## 5. 解析行为统计的Bayesian语义

对每个observable Bernoulli event：

\[
p_j\sim\mathrm{Beta}(\alpha_0,\beta_0),
\]

在 `s` 次事件、`f` 次合法非事件后：

\[
p_j\mid H_t
\sim
\mathrm{Beta}(\alpha_0+s,\beta_0+f).
\]

posterior mean与variance分别为：

\[
\mathbb E[p_j]
=
\frac{\alpha_j}{\alpha_j+\beta_j},
\]

\[
\operatorname{Var}(p_j)
=
\frac{\alpha_j\beta_j}
{(\alpha_j+\beta_j)^2(\alpha_j+\beta_j+1)}.
\]

这给出稳定、可解释、随证据量收缩的不确定性。无需learned consistency、variance floor或covariance regularization。

它不保证四个统计量充分描述全部伙伴能力；其角色是提供低维、合法、无训练退化的长期行为条件，而不是声明完整伙伴模型。

## 6. Heteroscedastic decision emission

对action `a`，anchor fit replicas给出sample mean `A_hat_a`和standard error `se_a`。模型定义：

\[
\hat A_a\mid z=k,c
\sim
\mathcal N
\left(
\mu_{k,a}(c),
 s_{k,a}(c)^2+se_a^2
\right).
\]

### 命题3：噪声自动降权

固定residual时，`se_a`增大会增大total variance，降低该observation对mean error的二次惩罚与gradient magnitude。因此高方差anchor自然具有较小影响，不需要独立confidence weight。

该likelihood仍依赖Gaussian近似；若replica return明显重尾或多峰，应将distribution升级为Student-t或显式empirical likelihood，并升级方法版本，而不是增加人工weight。

## 7. 单一joint likelihood消除loss权重

完整log likelihood为：

\[
\log p_\Theta(y,A)
=
\sum_t\log p_\Theta(y_t\mid y_{<t},A_{<t})
+
\sum_{t\in\mathcal I}
\log p_\Theta(A_t\mid y_{\le t},A_{<t}).
\]

response与decision evidence的相对作用由：

- 实际观测数量；
- 概率分布的normalized log density；
- replica uncertainty；

决定，而不是由人为 `lambda_response`、`lambda_decision`决定。

若需要增加anchor信息量，应调整预先注册的数据采集量或改进likelihood，不得通过观察性能后调一个loss weight。

## 8. Latent identifiability条件

联合latent只有在以下条件下可能具有非退化语义：

1. 至少两个component在response或decision emission上不同；
2. 训练分布对这些差异具有positive support；
3. transition与emission不完全等价；
4. 模型容量未允许base distribution独立解释全部数据而residual永远为零；
5. anchors覆盖会改变action ordering的状态。

response与decision共享component index的作用是：若一个partition只改善response却不能一致解释decision emission，其joint likelihood会受到惩罚。反之亦然。

仍存在以下不可识别情况：

- 两个component的两类emission都完全相同；
- 训练伙伴从未访问区分这些component的状态；
- decision anchors只包含同一top action；
- response与decision在数据中统计独立且没有共同latent cause。

这些情况必须由held-out predictive gain、decision prediction和H2检验揭示，不能靠separation loss强行制造component。

## 9. KL mirror policy闭式解

给定base policy `pi_0` 和action value `q_a`，考虑：

\[
\max_{\pi\in\Delta}
\sum_a\pi_aq_a
\quad
\text{s.t.}
D_{KL}(\pi\Vert\pi_0)\le\delta.
\]

Lagrangian为：

\[
\mathcal L(\pi,\eta,\lambda)
=
\sum_a\pi_aq_a
-
\eta\left[
\sum_a\pi_a\log\frac{\pi_a}{\pi_{0,a}}-\delta
\right]
+
\lambda\left(\sum_a\pi_a-1\right).
\]

一阶条件给出：

\[
\pi_\eta(a)
=
\frac{\pi_0(a)e^{q_a/\eta}}
{\sum_{a'}\pi_0(a')e^{q_{a'}/\eta}}.
\]

### 命题4：KL有界

若代码通过单调dual search选择 `eta` 使：

\[
D_{KL}(\pi_\eta\Vert\pi_0)\le\delta,
\]

则部署策略逐状态满足注册KL预算，除数值容差外不会越界。

### 命题5：支持保持

若 `pi_0(a)=0`，则 `pi_eta(a)=0`。因此adapter不能把概率质量放到base policy零支持动作。

### 命题6：极限

- `delta→0` 时，`pi_DELTA→pi_0`；
- `eta→∞` 时，`pi_eta→pi_0`；
- 在允许足够KL且base support包含最优动作时，`eta→0`使质量集中到最大`q_a`动作。

这给出适应强度的单一可解释控制量 `delta`，替代actor auxiliary weight与temperature。

## 10. Expected decision value

posterior平均action value为：

\[
\bar Q_t(a)=\sum_kb_t(k)\mu_{k,a}(c_t).
\]

若decision emission均值是对应continuation estimand的条件期望，则 `bar Q` 是在当前belief下的Bayes action value。

该结论相对于frozen base continuation policy、horizon `H` 与discount `gamma`成立，不是无限时域最优Q。

## 11. VOI非负性

令：

\[
V(b)=\max_a\sum_kb(k)\mu_{k,a}.
\]

对候选probe action `a`，response outcome `y`产生posterior `b^{a,y}`。Bayes plausibility满足：

\[
\mathbb E_y[b^{a,y}]=\bar b.
\]

由于 `V(b)` 是若干线性函数的pointwise maximum，因此是convex function。由Jensen：

\[
\mathbb E_y[V(b^{a,y})]
\ge
V(\mathbb E_y[b^{a,y}])
=V(\bar b).
\]

所以：

\[
VOI(a)
=
\mathbb E_y[V(b^{a,y})]-V(\bar b)
\ge0.
\]

### 推论

若response outcome与 `z` 独立，所有posterior均等于predictive prior，则 `VOI(a)=0`。

若所有component具有相同action-value vector，则belief变化不改变 `V`，同样有 `VOI(a)=0`。

因此VOI只在“动作会产生模式相关响应”且“模式会改变最优后续动作”时为正，这与研究动机直接一致。

## 12. Full action value的单位一致性

full版本使用：

\[
Q^{BA}_t(a)=\bar Q_t(a)+\gamma VOI_t(a).
\]

两项均以discounted raw team return为单位，因此可以直接相加，不存在information-bonus scale。

当前VOI采用current-state decision geometry近似下一时刻价值，是myopic local approximation。它不是完整Bayes-adaptive POMDP求解。

## 13. 训练数据策略与局部改进语义

所有训练rollout与anchors都由base policy `pi_0`产生。故decision emission学习的是：

\[
G_H^{\pi_0}(H_t,a),
\]

即强制首动作后继续使用base policy的局部deviation value。

analytic adapter是围绕 `pi_0` 的trust-region local improvement，而不是学习一个完全独立policy。该设计使：

- PPO保持on-policy；
- anchors与continuation policy一致；
- joint/full共享数据和模型；
- adaptation不能无限偏离其value labels适用的局部区域。

但它不证明多步反复应用adapter后仍严格对应原始one-action continuation labels；KL预算与confirmatory evaluation负责经验检验这一近似。

## 14. 复杂度

每步filter复杂度：

\[
O(K^2+K\cdot C_y),
\]

其中 `C_y` 是response head评分成本。

passive policy复杂度：

\[
O(K|A|+I_{dual}|A|),
\]

其中 `I_dual` 为固定dual search iterations。

myopic VOI对 `Y=3` 个coarse outcomes精确枚举：

\[
O(|A|K|Y|+|A|^2K|Y|).
\]

在 `K≤8, |A|=6, |Y|=3` 时为常数级开销，不训练额外网络。

## 15. 不保证事项

当前理论不保证：

- 模型设定正确；
- MLE在有限非凸网络中找到全局最优；
- latent component具有人类语义；
-训练伙伴覆盖confirmatory decision geometry；
- response predictive gain必然产生XP增益；
- joint一定优于response-only；
- full VOI一定优于joint；
- KL local improvement在model misspecification下单调提高真实return；
- `K=4` 最优；
-固定128-step estimand代表所有任务阶段；
-结果推广到连续动作、多人或非合作环境。

这些均由 `EVALUATION_SPEC.md` 的raw-backed实验检验，而不是通过增加新loss或gate转化为结构保证。
