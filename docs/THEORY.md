# THEORY：DEPI 的有限理论保证与限制

本文只证明当前实现可由结构与代数直接保证的性质，不把优化成功、泛化或高回报当作定理。
科学对象和可证伪主张以 [`SCIENTIFIC_SPEC.md`](SCIENTIFIC_SPEC.md) 为准，算法细节以
[`METHOD_SPEC.md`](METHOD_SPEC.md) 为准。

## 1. 精确离散滤波

设 `pi` 为 K 维概率向量，sticky transition matrix `T` 每行非负且和为 1，emission
log-likelihood `ell` 有限。预测与更新为

```text
pi_bar = pi @ T
pi_new[k] = exp(log(pi_bar[k]) + ell[k]) / Z
```

因为 `pi_bar[k] >= 0` 且 `sum_k pi_bar[k]=1`，而 sticky matrix 的每个元素严格为正，所以
`pi_bar[k]>0`。有限 `ell` 使每个未归一化权重为正，`Z>0`；故 `pi_new` 非负且总和严格为 1。
这正是一次 Chapman–Kolmogorov prediction 后接一次 Bayes correction。实现若再乘一次 `T`、
按错误轴归一化或把 `pi_new` 当 logits 再 softmax，便不再是该滤波器。

episode start 使用 uniform prior 并跳过跨 episode emission，因此过去 episode 的观测不可能
通过 protocol carry 影响新 episode 的 posterior。数值实现使用 log-domain normalization，
但浮点非有限值仍按正式 failure policy 处理，而非静默重置。

## 2. component 置换对称性

对任意 permutation matrix `P`，同时变换

```text
pi' = pi P
M' = P^T M
T' = P^T T P
ell' = ell P
```

则 `c = pi M = pi' M'`，mixture likelihood 和 actor/critic 输入保持不变。正式 sticky matrix
对所有非对角元素相同，也满足 `T'=T`。因此 component label 不可识别，只有由 empirical
decision signature 定义的等价类可比较。这给出禁止 raw-index accuracy 和训练期标签对齐的
理论依据。

## 3. 结构信息隔离保证

B1/B2 在 task encoder 之前用固定 channel contract 把 other-agent position、direction 和
inventory planes 清零。记该投影为 `P_task(o)`，则任意只在这些被遮蔽 planes 上不同的
`o,o'` 满足 `P_task(o)=P_task(o')`。在相同 task carry 下，确定性 task recurrence 的下一
carry 与输出必相同。

该保证只覆盖显式 semantic planes 和单步计算图；它不自动证明其余物理 planes 与伙伴行为
统计独立，也不证明 capability/protocol 路径语义正确。因此仍需 task leakage probe、history
shuffle 和真实 continuation 控制。B0 故意保留完整当前观测进入 task GRU，用于量化结构隔离
本身的增量。

## 4. 合法历史与时标

CapabilityEncoder 的证据只由连续 ego observations、ego previous action 和 episode-start
标志构成，所以按归纳法，其 hidden 与 published `u` 都是 L1–L5 的函数。exact filter 的
emission target 也只由连续局部 observations 复算，故 posterior `pi` 和 `c` 同样是合法历史
的函数。伙伴动作仅存在于 simulator transition 内，既不进入模型也不进入 comparator artifact。

每 16 步发布 `u` 只是一个结构时标，不足以推出 `u` 必然表示稳定 capability；`swap-u` 与
一致性读数负责检验该解释。每步更新 `pi` 也不推出它必然追踪动态 protocol；校准、`swap-c`
和 recoverable-value 控制负责检验。

## 5. 联合 likelihood 的一致混合

给定 component `z=k`，response target 的条件因子相加得到一个 component-specific
log-likelihood `log p_k(y)`。随后

```text
log p(y|H,a) = logsumexp_k(log pi[k] + log p_k(y))
```

只 marginalize 一次。不可见时屏蔽 position、direction 和 inventory 条件项，visibility
仍计分，因此不会把“不可见”同时作为 visibility 与任意位置标签重复计算。position、direction、
inventory 和 event 全部保留相同 K 轴，避免先对各 head 独立混合后拼成不存在的联合模型。

这只是 proper-likelihood 结构；若模型错设或数据覆盖不足，并不保证 posterior calibration。

## 6. 决策等价与监督

centered empirical signature

```text
A(H,a) = G(H,a) - mean_b G(H,b)
```

消除了对所有动作相同的 return offset。若两段历史的 A 相同，则在注册 continuation horizon
与目标策略下，它们给出相同动作排序和差值，因而对该有限决策问题等价。反之，任一动作差异
证明存在 decision-relevant distinction。该关系不要求也不允许使用 partner identity。

Q-signature fit、排序 hinge 和 actor target KL 使 empirical continuation 标签能够产生决策
梯度；stop-gradient target 防止 actor 反向改变标签。参数所有权保证 response NLL 不更新 task
encoder，Q signature 不更新 response decoder，所有 loss 又在同一次 optimizer transaction 的
同一参数快照上计算。它保证计算语义一致，但不保证非凸优化找到全局最优。

## 7. 因果归因的必要条件

随机化、容量和 key 匹配的 B0–B2 增量可排除已注册的预算与容量混杂；同状态 CRN history
shuffle 和 context swap 可降低环境噪声；G1–G4 可区分“环境中没有可恢复信号”和“模型未恢复
信号”。这些条件共同支持有限范围的机制归因，但仍依赖伙伴 panel、状态匹配质量、continuation
horizon 和统计功效。它们不是对所有未知伙伴的普遍因果定理。

## 8. 明确不作的保证

当前方法不保证：

- component 与人类可命名协议一一对应；
- posterior 在未通过 held-out gate 前已校准；
- response event 等价于意图；
- 更低 response NLL 必然提高回报；
- B2 必然优于 B0/B1 或任何正式基线；
- B3 的主动信息价值已经实现。

这些都是实验问题；反例必须按 [`EVALUATION_SPEC.md`](EVALUATION_SPEC.md) 原样报告。
