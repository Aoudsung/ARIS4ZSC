# DELTA-ZSC V6：理论基础与证明边界

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

## 10. 证据边界

下列机械事实不能推出算法有效：后验方差有限、Q head 有 action gap、decision
regret 非零、generator code 产生不同轨迹、单元测试通过或 CUDA 执行成功。

科学结论只来自冻结 V6 的 Official Simple/Wide 10×10 raw-return 矩阵、共同伙伴
面板和完整资源账本。C0–C5 是训练后的解释性测量，不得开启 loss、选择 checkpoint、
替换 seed 或改变正式样本。
