# DELTA-ZSC V6：理论基础与证明边界

> **修订头（2026-08-03）。** 动因：外部六线评审判定点：主论点 Θ2 的生态理论验证不成立（评审原文 `/Users/aoudsung/Downloads/Review&Suggestion.md`，仓库归档副本 [`docs/research/REVIEW_AND_SUGGESTION_2026.md`](../research/REVIEW_AND_SUGGESTION_2026.md)，指针 §7；统一重构计划 D 节 `Rigor_与_Generality_统一优化_7f793915.md`，只读）。本文件两处修订：
> 1. §10.3 T3 必要性证明中的不等式 \(\log\frac{1+\kappa}{1-\kappa}\le2\kappa\) 错误（取 κ=1/2 时 log 3≈1.0986>1，数值验证证伪），证明链按修正界 \(\mathrm{atanh}(\kappa)\le\kappa/(1-\kappa^2)\) 重建（评审 §7.5）；注册陈述的保守 4κ² 口径经重建后仍然成立，常数不变。
> 2. §10.2 T2 明确降级为“二元等先验、一一对应模式、常数 gap Δ”特例；注册一般情形 estimand 为 value-weighted distinguishability D_V(t)，并声明 E[Δ(H)]·TV/2 的“分别平均再相乘”用法在一般情形不成立（评审 §7.4）。

本文是 V6 的理论附卷；活动工程规范以
[`DELTA_ZSC_COMPLETE_THEORY_AND_DESIGN.md`](./DELTA_ZSC_COMPLETE_THEORY_AND_DESIGN.md)
为准。V5 r2/r3 的完整旧推导保留在本仓库的 git 历史中，不是 V6 运行规范。

## 1. 问题设定

两智能体共享回报，ego 只观察局部观测。伙伴策略 ξ 可以是历史依赖的，且其
身份、参数、训练来源和私有状态均不可见。合法 ego 历史为

\[
H_t=(o_0,a_0,d_0,\ldots,o_t).
\]

对任意固定伙伴 ξ，(H_t) 诱导一个历史 MDP。V6 寻找固定维度统计量
(b_t=b(H_t))，使一个共享策略能近似最优地响应多个伙伴。

## 2. 无结构问题的下界

若两个伙伴在关键决策前诱导完全相同的合法历史分布，却分别要求互斥的唯一
最优动作，则任意合法在线策略在该决策前都无法区分二者。令正确动作收益差为
Δ，伙伴先验各为 1/2，则任意策略至少在一个伙伴上以不小于 1/2 的概率选择错误
动作，因此 Bayes 遗憾下界为 Δ/2。

这说明连续 belief 不能凭空解决不可识别性。有效 ZSC 同时需要：伙伴在任务上
产生决策异质性、异质性在关键时刻前留下合法证据、共享控制器具有足够表达力。

## 3. 决策充分统计量

若对任意伙伴后验 (p(\xi\mid H_t))，条件未来 raw-return 的动作值只通过
(b(H_t)) 变化，即

\[
Q(H_t,a)=\tilde Q(x_t,b(H_t),a),
\]

则 (b) 是 ego 控制的充分统计量；在函数逼近无误差时，基于 (b) 的贪心或
策略改进与基于完整历史的控制等价。V6 以 dense Retrace、真实 all-action
continuation 和完整回报 PPO 共同逼近这一条件，而不把伙伴 ID 当作监督。

## 4. 决策等价不变性

定义 centered action signature

\[
A(h,a;\xi)=G(h,a;\xi)-|\mathcal A|^{-1}\sum_{a'}G(h,a';\xi).
\]

给所有动作值增加同一个常数不改变 (A)，也不改变最优响应。因此用 centered
signature 监督 latent 几何对回报基线平移不变。若两段 post-evidence 合法历史
具有相同 signature，V6 的连续度量目标最小值要求其均值距离为零；若 signature
相差，则目标距离随差异连续增长。该结论不需要离散伙伴类别。

相同 pre-evidence 历史必须得到相同 posterior，这是确定性合法策略接口直接
给出的因果一致性条件。隐藏 code 不能作为分离监督。

## 5. Twin raw-Q 的误差作用

设保守估计 \(\hat Q=\min(Q_1,Q_2)\)，且在关心的状态动作集上

\[
\|\hat Q-Q\|_\infty\le\epsilon_Q.
\]

则由 \(\hat Q\) 选择的动作相对真实最优动作的单步 continuation value 损失至多
(2\epsilon_Q)：

\[
Q(a^*)-Q(\hat a)
\le[Q(a^*)-\hat Q(a^*)]+[\hat Q(\hat a)-Q(\hat a)]
\le2\epsilon_Q.
\]

V6 的 Q-to-policy 权重在动作 gap 小或 twin disagreement 大时连续下降，因此不会
把这一界解释成硬可靠性证书。

## 6. Decision regret

对 posterior particles，

\[
R=\mathbb E_z\max_a Q(z,a)-\max_a\mathbb E_zQ(z,a)\ge0,
\]

因为 max 是凸函数。若所有 posterior hypotheses 共享一个最优动作，则两项相等，
(R=0)。反之，(R>0) 只表明完美上下文可能改变动作选择；它本身不证明信息可从
合法历史获得，也不证明探索动作有正回报。

V6 只将 EMA-Q 的 detached、尺度归一化 (R) 用作有界势差，并用固定连续日程
引入。它不把内部 regret 当成性能结论。其任务价值必须在训练后 C5 审计中与真实
信息价值比较。

## 7. 势函数边界

对固定势函数 \(\Phi(H)=-\tilde R(H)\)，标准 shaping

\[
F_t=\gamma\Phi(H_{t+1})-\Phi(H_t)
=\tilde R_t-\gamma\tilde R_{t+1}
\]

在完整 episode 上望远镜求和，仅留下初始势与 terminal 势；terminal 势置零时，
同一起始历史下策略排序不变。V6 实现进一步将单步差裁剪到 `[-1,1]` 并使用随
训练变化的 EMA 势，因此严格策略不变性只对一个固定 target 快照与无裁剪区间
成立。正式论文不得把动态近似实现描述为无条件定理；它是一个经验信用分配项。

## 8. Soft replay 的语义

一个 anchor 标签属于采集策略的 continuation。V6 不声称旧标签与当前 on-policy
Q 完全一致，而用策略 KL 与 age 的指数权重连续控制偏差。该机制避免策略版本边界
上的突变，但不消除离策略误差。报告必须包含 replay age、policy drift、有效权重
和使用次数。

## 9. 多目标 belief 优化

EMA 范数归一化使各目标的贡献不随原始 loss 单位任意缩放。为了覆盖 EMA
尚未跟上突发梯度的适应窗口，实现在加权后还将每个目标的瞬时 L2 范数限制为不超过
该目标的注册权重。固定顺序 PCGrad 再对每个后续梯度移除其与已接受梯度的负投影；因此
该后续梯度不会直接抵消先前目标。这是一项局部一阶约束，不保证所有目标同时下降，也不构成
收敛证明。完整回报仍是最终裁决。

## 10. 路由定理：匹配界、历史样本复杂度与端到端等式

本节形式化三个定理。它们是 PATH_C 提案草案（legacy §2.2–2.3）的正式化版本，
数值验证由受控相图实验（S2）承担；在生态任务上的定量推论登记于
[`docs/research/THEORY_PREDICTIONS.md`](../research/THEORY_PREDICTIONS.md)。

### 10.1 记号与二惯例静态模型（T2 的设定）

固定共同前缀策略 q 与切换时刻 t。两种惯例 z∈{0,1}，等先验；在该时刻之前的
合法历史 H_t 分别服从分布 P_0^t 与 P_1^t。两种模式 m∈{0,1} 与惯例一一对应：
在惯例 z 下选模式 m=z 的剩余回报为常数 r+，选错为 r−，记 Δ=r+−r−>0。
总变差距离 TV(P,Q)=½‖P−Q‖_1。此模型内：

- V_fix^q(t)=r−+Δ/2（统一选任一模式的期望回报）；
- V_HZ^q(t)=r+（知道惯例即可选对）；
- V_hist^q(t) 为仅读 H_t 的路由器的最优期望回报。

### 10.2 定理 T2（TV 精确式与匹配界）

**陈述。** 在上述二惯例静态模型中，设 p_e^* 为基于 H_t 区分惯例的最小
Bayes 错误率，则

\[
p_e^*=\frac{1-\mathrm{TV}(P_0^t,P_1^t)}{2},
\]

且

\[
V_{HZ}^q(t)-V_{hist}^q(t)=\Delta\,\frac{1-\mathrm{TV}(P_0^t,P_1^t)}{2},
\qquad
V_{hist}^q(t)-V_{fix}^q(t)=\Delta\,\frac{\mathrm{TV}(P_0^t,P_1^t)}{2}.
\]

第一式是任何历史方法不可避免的遗憾，第二式由 Bayes 路由器达到，两者构成匹配
上下界。

**适用范围（2026-08-03 降级声明，评审 §7.4）。** T2 的精确式只在 §10.1 的
二元静态模型内成立：恰好两个等先验惯例、模式与惯例一一对应、所有历史上
正确/错误收益差都是同一个常数 Δ。一般情形（惯例/模式多于两个、先验不等、
收益差随历史 Δ(h) 变化）下，“把 Δ 与 TV 分别平均再相乘”的用法

\[
\mathbb E[\Delta(H)]\cdot\mathrm{TV}(P_0^t,P_1^t)/2
\]

一般不等于真实可恢复价值，不得用于生态任务。一般情形的 estimand 是
value-weighted distinguishability：

\[
D_V(t)=\frac12\int\Delta(h)\,\lvert p_1^t(h)-p_0^t(h)\rvert\,dh,
\]

其中 Δ(h) 是历史 h 处选错模式的收益差，p_z^t 是惯例 z 诱导的合法历史密度。
仅当 Δ(h)≡Δ 为常数时 D_V(t) 退化为 Δ·TV(P_0^t,P_1^t)/2，即上式第二式。生态
预测与对撞口径必须直接估计 D_V(t)（legal-history value-weighted
distinguishability），不得用分别平均的 Δ̂ 与 TV̂ 之积替代。

**证明。** 任意基于 H_t 的检验在等先验两点检验中的错误率为
p_e=½(P_0(判 1)+P_1(判 0))=½∫min(dP_0,dP_1)=½(1−TV)，其中最后一个等号是
Le Cam 恒等式，逐点取 min(dP_0,dP_1) 的检验（即后验较大者）达到它。在静态
收益模型下，路由期望回报为 r+−Δ·p_e，故最小错误率检验（Bayes 路由器）达到
V_hist=r+−Δ(1−TV)/2。代入 V_HZ=r+ 与 V_fix=r−+Δ/2 直接得两式。两式相加恒为
Δ/2，即固定总机会在“可恢复”与“不可恢复”之间的精确分割。证毕。

### 10.3 定理 T3（历史样本复杂度）

**设定。** 在 T2 模型上追加结构：历史中唯一随惯例变化的证据是 n 个条件独立
Bernoulli 回应，惯例 0 的参数为 (1−κ)/2，惯例 1 的参数为 (1+κ)/2，
0<κ≤1/2。目标：相对惯例参考的遗憾不超过 ε，其中 0<ε≤Δ/8。

**陈述。**

- 必要性：任何方法都需要
\[ n\ge\frac{\log(\Delta/(4\varepsilon))}{4\kappa^2}. \]
- 充分性：经验均值阈值检验在
\[ n\ge\frac{2\log(\Delta/\varepsilon)}{\kappa^2} \]
时已经足够。
- 合并：最坏情况历史长度精确到常数阶为 Θ(log(Δ/ε)/κ²)。

**必要性证明。** 由 T2，最优路由的遗憾为 Δ·p_e^*，其中 p_e^*=(1−TV_n)/2 是
n 个证据下两点检验的最小错误率。Le Cam 二点检验下界给出

\[
p_e^*\ge\frac14\exp\big(-\mathrm{KL}(P_0^n\,\|\,P_1^n)\big).
\]

单步 KL 直接计算为

\[
kl=\kappa\log\frac{1+\kappa}{1-\kappa}
=\mathrm{KL}\!\left(\mathrm{Bern}\frac{1+\kappa}{2}\,\Big\|\,\mathrm{Bern}\frac{1-\kappa}{2}\right).
\]

**界的修正（2026-08-03）。** 原证明使用 log((1+κ)/(1−κ))≤2κ，该不等式错误：
取 κ=1/2 时 log 3≈1.0986>1=2κ（python 数值验证证伪）。正确链条：由
atanh(κ)≤κ/(1−κ²)（0≤κ<1，数值验证：κ=1/2 时 0.5493≤0.6667），

\[
\log\frac{1+\kappa}{1-\kappa}=2\,\mathrm{atanh}(\kappa)
\le\frac{2\kappa}{1-\kappa^2}\le\frac83\,\kappa
\qquad(0\le\kappa\le1/2),
\]

最后一步用 κ≤1/2 时 1/(1−κ²)≤4/3。故单步 kl≤(8/3)κ²≤4κ²，
KL(P_0^n‖P_1^n)=n·kl≤(8/3)nκ²≤4nκ²。代入得遗憾下界
(Δ/4)·exp(−(8/3)nκ²)≥(Δ/4)·exp(−4nκ²)。要让任意方法的遗憾不超过 ε，
紧形式需 n≥3log(Δ/(4ε))/(8κ²)；注册形式采用保守的单步 KL≤4κ² 口径，得
n≥log(Δ/(4ε))/(4κ²)，必要性成立。注册陈述不变。证毕。

**充分性证明。** 阈值检验：当样本均值 X̄≥1/2 时判惯例 1，否则判 0。两个
均值距阈值均为 κ/2，Hoeffding 给出每侧错误至多 exp(−nκ²/2)。取
exp(−nκ²/2)≤ε/Δ，即 n≥2log(Δ/ε)/κ²，则 p_e≤ε/Δ，遗憾 Δ·p_e≤ε。证毕。

**常数复核（2026-08-03 重建）。** 下界常数 1/4 来自 Le Cam 两点下界与保守
口径单步 KL≤4κ² 的组合；紧口径为 kl≤(8/3)κ²，可把分母改为 8/3。原文所称
“紧口径 kl≤2κ²、分母 2”依赖被证伪的不等式（κ=1/2 时 kl=0.5493>0.5=2κ²），
一并废止。数值验证（python，2026-08-03）给出单步 KL 的夹逼：

\[
2\kappa^2\le kl\le\frac83\kappa^2,\qquad
kl/\kappa^2\in[2.0000,\,2.1972]\;(\kappa\in(0,1/2]),
\]

即下界分支的每样本 KL(Bern((1±κ)/2)‖·)≥c·κ² 常数 c=2（κ→0⁺ 取下确界），
必要性阶与 Hoeffding 充分性阶一致，故阶为 Θ(log(Δ/ε)/κ²)。上界常数 2 来自
Hoeffding。ε≤Δ/8 保证对数项为正且远离平凡区。注册陈述与 legacy 提案草案的
常数一致（保守 4κ² 口径在修正界下仍然成立）。数值验证时，真实最小历史长度
n_emp 应落在注册下界与上界之间（下界是弱必要条件，上界是构造性充分条件）。

### 10.4 定理 T4（路由紧上界与端到端等式）

**设定。** 回到一般设定（§1 与 legacy 提案 §2.1–2.2）：固定共同前缀 q 与切换
时刻 t，模式库 M={1,…,K}，G_t^q(m) 为从共同前缀真实接管模式 m 的随机剩余
回报，B_t^q 为共同前缀的期望累计回报。

**陈述。** 对任意只读官方历史的路由器 g:H_t→M，

\[
\mathbb E[G_t^q(g(H_t))]\le V_{hist}^q(t),
\]

且逐历史选择条件期望最高的模式达到等号，故该界在给定模式库与观测协议下是紧的。
令学习路由器 ĝ 的预测遗憾 R_pred^q(t)=V_hist^q(t)−E[G_t^q(ĝ(H_t))]≥0，则存在
精确等式

\[
J_{route}^q(t)-J_{sfix}^q(t)=\Gamma_{hist}^q(t)-R_{pred}^q(t),
\]

\[
J_{route}^q(t)-J_{fix}(0)=\Gamma_{hist}^q(t)-R_{pred}^q(t)-C_{wait}^q(t),
\]

其中 J_sfix=B_t+V_fix，J_route=B_t+E[G_t^q(ĝ(H_t))]，Γ_hist=V_hist−V_fix，
C_wait=J_fix(0)−J_sfix 为有符号等待代价。

**证明。** 上界：对每个 H_t，E[G_t^q(g(H_t))|H_t]≤max_m E[G_t^q(m)|H_t]，取
期望即得；取 g^*(H_t)∈argmax_m E[G_t^q(m)|H_t] 达等号。第一个等式：两边同减
B_t 后，左端为 E[G_t^q(ĝ)]−V_fix，右端为 (V_hist−V_fix)−(V_hist−E[G_t^q(ĝ)])，
展开即相等。第二个等式：J_route−J_fix(0)=(J_route−J_sfix)+(J_sfix−J_fix(0))，
代入第一式与 C_wait 定义即得。真实接管已包含在 G_t^q 中，故不得再扣一次切换
成本。证毕。

**推论（生态解释）。** 在 OvercookedV2 等生态任务中，学习路由器只能给
V_hist 提供构造性下界：显著优于 J_sfix 即证明存在可恢复价值；未能优于只说明
当前注册模型没有找到它，不构成无机会证书。无机会证书必须由 T2/T3 的受控上界
或 run-disjoint 面板读数给出。

## 11. 证据边界

下列机械事实不能推出算法有效：后验方差有限、Q head 有 action gap、decision
regret 非零、generator code 产生不同轨迹、单元测试通过或 CUDA 执行成功。

科学结论只来自冻结 V6 的 Official Simple/Wide 10×10 raw-return 矩阵、共同伙伴
面板和完整资源账本。C0–C5 是训练后的解释性测量，不得开启 loss、选择 checkpoint、
替换 seed 或改变正式样本。
