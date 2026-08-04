# THEORY：DEPI 的有限理论保证与限制

本文只证明由当前结构和代数直接保证的性质。科学对象以
[`SCIENTIFIC_SPEC.md`](SCIENTIFIC_SPEC.md) 为准，算法以
[`METHOD_SPEC.md`](METHOD_SPEC.md) 为准。

## 1. 精确离散滤波

若 `pi` 是概率向量、sticky matrix `T` 行随机且严格为正、emission log-likelihood `ell`
有限，则：

```text
pi_bar = pi @ T
pi_new = softmax(log(pi_bar)+ell)
```

中 `pi_bar` 严格为正且和为 1，归一化常数有限为正，所以 `pi_new` 仍是严格正的概率向量。
这恰好是一次 Chapman–Kolmogorov prediction 后接一次 Bayes correction。重复乘 T、按错误
轴归一化或把 posterior 再当 logits softmax 都不再是该滤波器。

该定理只相对于注册 response model 成立。固定 T 不依赖 ego action，因此它不能证明主动
protocol formation；held-out calibration 也只能检验 posterior-predictive distribution。

## 2. component 置换对称性

对 permutation matrix `P` 同时变换：

```text
pi'=pi P, M'=P^T M, T'=P^T T P, ell'=ell P
```

则 `pi M=pi'M'`，mixture likelihood 与 actor/critic 输入不变。对称 sticky T 还满足
`T'=T`。因此 component index 不可识别；当前 component 只能称 exchangeable response
regimes，不能声称其 index 是伙伴类型或 value-signature prototype。

## 3. task 与 instant-partner 通路保证

B0–B2 在 task encoder 前应用固定投影 `P_task`，清除显式 other-agent position、direction、
inventory planes。若 `o,o'` 仅在这些 planes 上不同，则同一 task carry 下：

```text
P_task(o)=P_task(o') => next_task_carry(o)=next_task_carry(o')
```

`r_t=f_instant(o_t^partner)` 的函数签名没有 carry 或历史参数，所以它只能是当前伙伴 planes
的函数。两点结合给出“当前几何可用、显式伙伴序列不进入 task recurrence”的结构保证。

这不保证 task carry 与伙伴历史统计独立。伙伴会合法改变未遮蔽的任务世界，故其后果仍可
被 `x_t` 记录；task probe 必须以固定 task-state probe 为 baseline，而不能以 chance 为唯一
失败标准。

## 4. 合法历史与 capability 限制

Capability evidence、protocol emission 和 instant geometry 都只由 L1–L5 构造，按归纳法其
输出仍属于合法 ego history。伙伴动作只存在于 simulator world update，不进入部署图。

每 16 步发布只证明 actor-visible `u` 的数值在窗口内固定，不证明它表示稳定 capability。
零向量仍是 consistency loss 的可行解，但不再是窗口统计 prediction objective 的最优解；
variance floor 又惩罚跨发布低方差。由于这些统计只固定前四坐标且受观测条件限制，
variance/norm/collapse、`swap-u` 和 no-capability 消融仍是必要经验诊断。

## 5. joint likelihood 不重复计数 visibility event

每个 component 先形成完整条件 log-likelihood，再对 K 做一次 logsumexp。position、direction
和 inventory 只在 current visible 时计分；event 只在 previous/current 都 visible 时计入
visible inventory change。进入/离开视野只进入 visibility head。因此同一个 visibility
transition 不会被 visibility 与 event 重复计分。

event head 不读 task features；kinematic head 可读 stopped frame。这防止 response objective
通过 task trunk 建立旁路，但不保证模型设定正确或 component 一定被利用。

## 6. 决策等价、尺度和精确 KL

centered empirical signature 消除了所有动作共享的 return offset。相同 signature 表示在
注册 horizon/continuation policy 下动作排序和差值相同，且不要求 partner identity 相同。

one-hot `z=k` 经共享 critic 得到 `S_k`，训练只要求 posterior mixture `sum_k pi_k S_k`
逼近 empirical signature。这个约束在 component 同时置换时保持不变，所以它建立直接的
decision coupling，但不能打破 index 不可识别性，也不能单独保证各 component 不合并；后者
只能由 pairwise divergence、one-hot intervention 与跨 seed permutation alignment 诊断。

用 MAD 和一次正确交付尺度下限归一化后，actor target 对 raw reward 单位变化更稳定；
top-action stability 和 policy-drift weight 控制 noisy/off-policy anchors，但都不保证非凸优化
成功。

完整行为分布给出精确：

```text
KL(pi_old||pi_new)=sum_a pi_old(a)[log pi_old(a)-log pi_new(a)] >= 0
```

有限精度容差外，该量不会像 executed-action Monte Carlo 差那样因抽样而为负。固定一次
auxiliary transaction 使 response/capability exposure 不随 PPO early stop 改变。

## 7. 预算与嵌套识别

B0 与 B1 具有同一 task/instant 接线，B1 只打开 `u,c`；B2 再打开 decision/separation，故
`B1-B0`、`B2-B1` 分别对应注册增量。R0 的完整 task history 是外部 reference，不能把
`R0->B1` 当单组件差。

core matrix 匹配主 PPO transitions，回答额外监督是否有用；R0-extra/B0-extra/B1-extra 以普通 PPO 精确
替换 B2 的额外 simulator cost，回答相同总交互下是否优于更多数据。两种 estimand 不可混写。

## 8. 因果读数与 G4 边界

source-world context value 比较的是同一实际世界中两种 action distributions 在同一
all-action value vector 上的差，避免要求 donor context 在 source partner 下仍保持 donor
世界语义。它仍依赖 task-state matching、positivity、continuation horizon 和 panel 覆盖，
不是普遍因果定理。

G4 在 noisy fit replicas 上选择动作、在独立 replicas 上评估。cross-fitting 降低同样本选择
偏差，但不能保证 G4≥G1。recoverable ratio 只有在 run-level denominator bootstrap LCB 达
阈值时可估；否则必须输出 not estimable。

## 9. 明确不保证

当前方法不保证 component 有人类可命名语义、posterior 在未通过 gate 前已校准、`u` 在所有
未见分布上都不坍缩、
response NLL 改善必然提高回报、B2 优于 R0/B0/B1/extra controls，或 B3 已实现。所有反例和
失败节点必须按 [`EVALUATION_SPEC.md`](EVALUATION_SPEC.md) 原样报告。
