# METHOD_SPEC：Unified DELTA-ZSC 可执行方法

`authoritative: true`

代码权威路径为 `src/delta_zsc/`。本文件中的每一对象都必须在该路径中有唯一实现；未在本文件出现的 comparator、separation、capability loss、context dropout、auxiliary actor、generator 或 gate 不属于当前方法。

## 1. 总体数据流

```text
                          ┌────────────────────────────┐
current task planes ─────>│ task-only recurrent base  │──> x_t
current partner planes ──>│ memoryless instant branch │──> r_t
                          └─────────────┬──────────────┘
                                        │
legal observable events ──> Beta sufficient statistics u_t
                                        │
previous frame/action + u ─> response emission p_theta(y_t|z_t,...)
                                        │
learned transition T ──────> exact categorical belief b_t
                                        │
training CRN anchors ──────> decision emission p_psi(A_t|z_t,...)
                                        │
                          E_b[Q_k] + gamma * VOI
                                        │
base policy pi_0 ─────────> analytic KL mirror policy pi_DELTA
```

模型只包含一个 base policy/value network 和一个 latent coordination model。不存在每个 component、伙伴或类别独立的 actor/critic。

## 2. Base policy

### 2.1 输入隔离

`BasePolicyModel` 的 recurrent task branch接收清除显式 other-agent position/direction/inventory planes后的 observation。其输出为 `x_t`。

独立的 memoryless instant-partner CNN 接收当前 other-agent planes，输出 `r_t`。该分支没有 previous observation、previous action 或 carry。

### 2.2 Actor与value

base network输出：

\[
\pi_0(a_t\mid x_t,r_t),
\qquad
V_0(x_t,r_t).
\]

base policy只由标准PPO训练。PPO rollout始终由 `pi_0` 生成，包括 `joint` 和 `full` 的训练；因此其importance ratio和GAE保持on-policy。

## 3. 解析行为统计

维护四个独立Beta posterior：

\[
p_j\mid H_t\sim\mathrm{Beta}(\alpha_{t,j},\beta_{t,j}),
\quad j=1,...,4.
\]

每个合法观察更新：

\[
\alpha_{t,j}=\alpha_{t-1,j}+n_{t,j},
\]

\[
\beta_{t,j}=\beta_{t-1,j}+d_{t,j}-n_{t,j}.
\]

其中 `n` 为事件指示，`d` 为该事件是否可观察。episode start恢复 `Beta(1,1)` prior，再加入当前帧可合法观察的证据。

传入latent model的特征为：

\[
u_t=
[\mathbb E p_1,...,\mathbb E p_4,
 \log(1+\alpha_1+\beta_1),...,
 \log(1+\alpha_4+\beta_4)].
\]

该统计无参数、无optimizer、无额外loss。

## 4. Latent coordination model

### 4.1 模式与transition

\[
z_t\in\{1,...,K\}.
\]

transition logits为可训练 `K×K` 矩阵：

\[
T_{ij}=\operatorname{softmax}(\tau_i)_j.
\]

初始化对角略高，但 `T` 完全由joint marginal likelihood学习。没有固定 `p_stay`、evidence-gated identity transition或sensitivity matrix。

### 4.2 Response emission

响应目标 `y_{t+1}` 包含：

- partner visibility；
- visible时的5×5 relative position；
- visible时的direction；
- visible时的factorized inventory；
- 前后均visible时的inventory change。

response model读取：

\[
(o_t, u_t, a_t).
\]

`o_t` 在进入response model前stop-gradient。空间编码使用保留坐标的CNN并flatten，不使用会丢失位置的channel mean/max pooling。

每个head采用嵌套base-residual参数化：

\[
\ell_k^y
=
\ell_{base}^y(o_t,u_t,a_t)
+
\Delta\ell_k^y(o_t,u_t,a_t).
\]

所有residual head零初始化。`include_component_residual=false` 直接得到训练过的base distribution，是合法嵌套control，而非zero-embedding OOD干预。

一个component必须联合解释完整response：

\[
\log p_\theta(y_{t+1}\mid z_{t+1}=k)
=
\log p(v)
+v[\log p(pos)+\log p(dir)+\log p(inv)]
+m_{event}\log p(event).
\]

只在factor可观察时计入其likelihood。visibility=0本身始终是合法证据。

### 4.3 Decision emission

给定stop-gradient base features `(x_t,r_t)` 与解析统计 `u_t`，latent model输出：

\[
\mu_{k,a}(x_t,r_t,u_t),
\qquad
s_{k,a}(x_t,r_t,u_t)>0.
\]

同样使用base-residual结构：

\[
\mu_{k,a}=\mu_{base,a}+\Delta\mu_{k,a}.
\]

CRN anchor给出centered empirical returns `A_hat_t(a)`、每动作replica standard error `se_t(a)`和action mask。其decision emission为：

\[
p_\psi(\hat A_t\mid z_t=k)
=
\prod_a
\mathcal N\left(
\hat A_t(a);
\mu_{k,a},
 s_{k,a}^2+se_t(a)^2
\right)^{m_a}.
\]

由此，noisy anchors自动具有更宽observation variance；不再使用confidence weight、temperature、rank hinge或pseudo-label。

## 5. Joint latent likelihood

对一个base-policy rollout，普通transition只包含response evidence；anchor时刻同时包含decision evidence。

forward recursion：

\[
\bar b_{t+1}(k)
=
\sum_j b_t(j)T_{jk},
\]

\[
L^y_{t+1,k}
=p_\theta(y_{t+1}\mid z_{t+1}=k),
\]

\[
L^A_{t+1,k}
=p_\psi(\hat A_{t+1}\mid z_{t+1}=k).
\]

\[
b_{t+1}(k)
\propto
\bar b_{t+1}(k)
L^y_{t+1,k}
(L^A_{t+1,k})^{I_{anchor}}.
\]

latent objective是唯一的负对数边缘似然：

\[
\mathcal L_{latent}
=-\log p_\Theta(y_{1:T},A_{\mathcal I}).
\]

实现将response observation与decision observation的log evidence直接相加，并按实际观测数量归一化。不存在可调的response-vs-decision loss weight。

### 5.1 Rollout边界

PPO rollout可能从episode中段开始。latent likelihood只使用该rollout中首个observed episode start之后的transition，使forward recursion具有真实uniform initial prior。被排除的前缀仍可用于PPO，但不进入latent MLE。

### 5.2 Anchor生命周期

anchor只属于产生它的当前rollout，并在该rollout的latent update中使用一次。不得跨outer update replay或缓存。因此：

- continuation policy与标签policy一致；
- 不需要age weight或policy-drift fingerprint；
- checkpoint不保存anchor buffer；
- resume不会改变后续监督数据。

## 6. 两个独立估计器

### 6.1 PPO estimator

\[
\omega\leftarrow
\arg\min_\omega \mathcal L_{PPO}(\omega;\Theta_{sg}).
\]

latent parameters在PPO update中stop-gradient。PPO只更新base task encoder、instant-partner encoder、base actor和value。

### 6.2 Latent MLE estimator

\[
\Theta\leftarrow
\arg\min_\Theta \mathcal L_{latent}(\Theta;\omega_{sg}).
\]

base features在latent update中stop-gradient。latent optimizer只更新transition、response emission和decision emission。

两个estimator拥有：

- 独立参数树；
- 独立optimizer；
- 独立Adam moments；
- 无跨目标loss weighting；
- 无gradient routing table。

训练顺序为：rollout → 当前rollout anchor采集 → latent update → PPO minibatch updates。由于rollout由base policy生成，PPO仍严格对应其behavior distribution。

## 7. Online deployment filter

deployment state只保存：

```text
task_carry
Beta behavior posterior
categorical belief b_t
previous legal observation
ego previous action
episode_start
```

在episode首帧：

\[
b_t=\mathrm{Uniform}(K),
\]

且不应用虚构的pre-episode response。

之后每个physical transition：

\[
b_{t+1}(k)
\propto
\left(\sum_jb_t(j)T_{jk}\right)
 p_\theta(y_{t+1}\mid z_{t+1}=k).
\]

即使伙伴不可见，transition继续发生，visibility=0继续作为证据；只有position/direction/inventory等不可观察factor被mask。

## 8. Analytic KL mirror policy

latent decision mean给出：

\[
\bar Q_t(a)=\sum_k b_t(k)\mu_{k,a}(x_t,r_t,u_t).
\]

passive joint版本求解：

\[
\max_\pi
\sum_a\pi(a)\bar Q_t(a)
\quad
\text{s.t.}
D_{KL}(\pi\Vert\pi_0)\le\delta.
\]

等价闭式族：

\[
\pi_\eta(a)
=
\frac{\pi_0(a)\exp(\bar Q_t(a)/\eta)}
{\sum_{a'}\pi_0(a')\exp(\bar Q_t(a')/\eta)}.
\]

代码通过单调一维dual search选择最小满足KL约束的 `eta`。`eta`不是方法超参数；唯一科学选择是 `delta=adaptation_kl_budget`。

base policy概率为零的动作保持零支持；因此analytic adapter不会选择base policy认为绝对不可能的动作。

## 9. Myopic Bayes VOI

full版本从同一response model计算三类精确coarse outcome：

1. invisible；
2. visible且inventory不变；
3. visible且inventory改变。

对每个候选ego action `a`：

\[
p(y\mid b_t,a)=\sum_k\bar b_t(k)p_\theta(y\mid k,a),
\]

\[
b_{t+1}^{a,y}(k)
\propto
\bar b_t(k)p_\theta(y\mid k,a).
\]

使用当前local decision geometry近似下一步最优价值：

\[
VOI_t(a)
=
\mathbb E_y
\left[
\max_{a'}\sum_kb_{t+1}^{a,y}(k)\mu_{k,a'}
\right]
-
\max_{a'}\sum_k\bar b_t(k)\mu_{k,a'}.
\]

VOI与task return同单位，full action value为：

\[
Q_t^{BA}(a)=\bar Q_t(a)+\gamma VOI_t(a).
\]

再将 `Q_t^{BA}` 代入同一个KL mirror solution。没有VOI weight或information bonus coefficient。

## 10. 自然嵌套版本

| Variant | Base PPO | Response latent | Decision emission | KL adaptation | VOI |
|---|---:|---:|---:|---:|---:|
| `base` | ✓ | — | — | — | — |
| `response_only` | ✓ | ✓ | — | — | — |
| `joint` | ✓ | ✓ | ✓ | ✓ | — |
| `full` | ✓ | ✓ | ✓ | ✓ | ✓ |

`joint`与`full`共享完全相同的训练参数；full只在部署时开启由已训练模型解析计算的VOI。

response-only不使用decision emission进行策略适应，因此H2不会把response prediction本身误写为控制增益。

## 11. Partner distribution

训练伙伴只来自manifest role=`development_support`。采样层级固定为：

```text
mechanism uniform
 -> hyperparameter family uniform within mechanism
 -> checkpoint stage uniform within family
 -> parent/run uniform within stage
```

新增一个OP width family不会增加OP总概率。confirmatory、calibration和训练parent必须lineage-disjoint。

训练时不使用生成器。伙伴多样性通过真实frozen checkpoints和checkpoint stages建立，避免在未约束latent code空间外推伙伴行为。

## 12. Checkpoint与deployment

checkpoint保存：

- base/latent parameters；
- 两套optimizer state；
- 完整runner，包括环境、伙伴carry、ego legal state、role和RNG；
- environment steps与update count；
-资源账本。

不保存：anchor、comparator、pair、M1 ensemble或claim gate。

uninterrupted与save/restore continuation必须参数、optimizer、runner、RNG和后续抽样完全等价。

deployment只导出base params、latent params、config、variant和完整fingerprint，不导出optimizer或训练数据。

## 13. 配置合同

方法段恰好包含：

```yaml
method:
  latent_components: 4
  continuation_horizon: 128
  adaptation_kl_budget: 0.04
```

任何新增方法weight必须先证明它由joint probability model或KL constrained optimization不可导出，并升级方法版本；不能以修复某个实验失败为由直接追加。
