我建议将整个 Path C 核心重构为：

# 价值商信念条件 Bellman 控制器

**Value-Quotient Belief-Conditioned Bellman Controller，VQBC**

它不是在当前模型外面再增加模块，而是用一个统一的后验条件动作价值系统，替换目前彼此脱节的：

- shared actor；
- scalar prototype value heads；
- family-label response heads；
- transition feature predictor；
- external belief filter；
- probe override controller；
- prefit/calibration/adaptation 多阶段训练。

------

# 一、必须修复的唯一根因

当前模型的根本错误，可以概括为：

[
\text{用于估计信息价值的 belief}
\quad\not\rightarrow\quad
\text{真正执行任务的 policy}
]

当前 actor 根据 recurrent feature独立产生基础动作：

[
a_t^{\text{base}}\sim\pi_\phi(a\mid z_t),
]

外部 belief 仅用于判断是否覆盖该动作：

[
a_t=
\operatorname{Override}
\bigl(
a_t^{\text{base}},b_t,\hat J_{\text{use}},\hat J_{\text{mask}}
\bigr).
]

执行动作、观察回应后：

# [ b_{t+1}

\operatorname{BayesUpdate}(b_t,y_t),
]

但下一步 actor依旧计算：

[
a_{t+1}^{\text{base}}
\sim
\pi_\phi(a\mid z_{t+1}),
]

而不是：

[
a_{t+1}
\sim
\pi_\phi(a\mid z_{t+1},b_{t+1}).
]

代码上，`PathCFlaxAdaptationModel` 的输入只有 recurrent carry、observation 和 episode boundary；belief、上一动作和上一奖励均未输入模型。 当前 shared actor又明确使用 `stop_gradient(features)`，而四个 prototype value heads只输出标量 (V_k(z))，没有输出不同伙伴假设下的动作价值 (Q_k(z,a))。

rollout中则先从 actor采样 `base_action`，随后才使用外部 belief计算 probe score并覆盖动作；回应产生后，belief虽然更新，但仍然没有进入下一步正常 actor。

这解释了所有实验现象：

- 适应前 XP 为 −69.83；
- 关闭探查已经将 XP 提升到 −1.98；
- 决策导向仅进一步提升到 −0.28；
- 即约 97.5% 的 XP 改善来自普通 population adaptation，而不是回应使用；
- `A2-use − A2-mask` 只有 +0.0156；
- 所有适应策略的 SP 都从 144.78 大幅下降到 56.52–94.01。

这表明当前模型实际学习的是：

[
\text{多伙伴平均策略}
]

而不是：

[
\text{根据伙伴后验切换控制策略}.
]

因此，统一修复必须同时满足：

[
\boxed{
\text{belief更新、信息价值估计、正常动作选择和probe动作选择必须使用同一个动作价值对象}
}
]

------

# 二、统一模型的完整结构

统一模型只保留一个执行闭环：

```text
官方局部历史
    ↓
TD-only recurrent control representation
    ↓
无标签 latent action-value hypotheses
    ↓
当前 value quotient 与 posterior belief
    ↓
同一个模型计算 J_use 与 J_mask
    ↓
一个统一动作分布选择普通动作或信息动作
    ↓
环境回应
    ↓
更新 posterior
    ↓
下一步全部动作直接依赖更新后的 posterior
```

该闭环中不再存在“先采样 base action，再由外部 controller覆盖”的结构。

------

## 2.1 输入历史必须与设计文档一致

当前 recurrent backbone只输入 observation和 episode boundary。 但部分可观测控制至少需要：

[
x_t=
(o_t,a_{t-1},r_{t-1},d_{t-1}),
]

其中：

- (o_t)：官方局部观测；
- (a_{t-1})：ego上一 primitive action；
- (r_{t-1})：上一原始团队奖励；
- (d_{t-1})：回合边界。

统一编码器应为：

[
z_t=f_\theta(z_{t-1},o_t,e(a_{t-1}),r_{t-1},d_{t-1}).
]

这不是增加身份信息。上一动作和奖励本来就是 ego 自己可观察的历史，也是正确构造 belief-state representation所必需的变量。

------

## 2.2 用动作价值替代 scalar critic

当前 backbone只通过 shared scalar value：

[
V(z_t)
]

接受控制梯度。这无法保证表示保留不同动作之间的价值差异。

统一模型必须输出一组无标签 latent action-value hypotheses：

[
Q_m(z_t,a),
\qquad
m=1,\ldots,M.
]

采用 dueling 参数化：

# [ Q_m(z,a)

## V(z) + c_m(z) + A_m(z,a)

\frac{1}{|\mathcal A|}
\sum_{u}A_m(z,u).
]

其中：

- (V(z))：共享任务进展价值；
- (c_m(z))：latent hypothesis的整体 value offset；
- (A_m(z,a))：真正决定动作排序的 residual action advantage。

定义归一化控制签名：

# [ \widetilde A_m(z,a)

Q_m(z,a)-\max_u Q_m(z,u).
]

Path C真正关心的是 (\widetilde A_m)，而不是 (Q_m) 的整体水平。

共享 recurrent representation (z_t) 只能接收这些 (Q_m) 的 Bellman TD gradient。禁止以下梯度进入 (z_t)：

- response prediction；
- latent assignment；
- partner classification；
- reconstruction；
- contrastive objective；
- information-gain objective；
- probe-selection objective。

这样既满足“只有控制价值梯度塑造 representation”的要求，也使 representation真正对应理论中的动作相对价值，而不是当前的 scalar state value。

------

# 三、伙伴结构不能再由四个人工 family label定义

当前模型将伙伴人工分为：

- adapted ego；
- own backbone history；
- foreign self-play history；
- Other-Play history。

训练时真实 `partner_indices` 被 one-hot后直接用于路由 response、value、transition 和 reward heads。

这些类别是训练来源，不是控制价值等价类。它们混合了：

- 训练算法；
- seed；
- checkpoint进度；
- 能力水平；
- convention；
- 当前策略非平稳变化。

统一模型继续使用同一伙伴池收集多样化交互，但必须完全隐藏：

- family ID；
- seed；
- checkpoint index；
- SP/OP label；
- training run ID。

这些字段只允许写入审计账本，不能进入模型、head routing、belief或阈值。

------

## 3.1 无标签 value-slot 形成

设置 (M) 个过完备 value slots：

[
Q_1,\ldots,Q_M.
]

每个 episode或完整 recurrent sequence，对每个 slot计算 Bellman误差：

# [ \mathcal E_{e,m}

\sum_{t\in e}
\rho
\left(
Q_m(z_t,a_t)-y_{t,m}
\right),
]

其中 (\rho) 为 Huber loss，目标为：

# [ y_{t,m}

r_t
+
\gamma(1-d_t)
\sum_u
\pi_{\text{exec}}^{-}(u\mid h_{t+1},b_{t+1})
Q_m^{-}(z_{t+1},u).
]

然后根据 Bellman适配程度计算无标签 responsibility：

# [ q_{e,m}

\operatorname{softmax}*m
\left(
-\mathcal E*{e,m}/\tau
\right).
]

`q` 必须 stop-gradient。共享表示和 Q heads通过：

# [ \mathcal L_Q

\sum_e\sum_m
\operatorname{sg}(q_{e,m})
\mathcal E_{e,m}
]

进行训练。

这意味着 latent slot的语义来自：

> 哪个动作价值函数最能解释这段轨迹的 Bellman结构。

而不是来自：

> 该伙伴属于 SP、OP、哪个 seed或哪个 checkpoint。

为防止所有 slots因完全对称而坍缩，可使用：

- 独立随机初始化；
- episode-level bootstrap masks；
- fixed randomized prior Q heads；
- clipped double-Q target。

这些机制不提供伙伴标签，也不定义人工风格类别，只维持价值后验的可辨识性。

------

# 四、在每个状态动态形成真正的 (W_C)

即便两个 latent slots在 response模式或总体回报上不同，只要其动作相对价值相同，它们就不应被控制器区分。

在每个当前状态 (z_t)，根据：

[
\widetilde A_m(z_t,\cdot)
]

动态形成控制等价类。

若：

## [ \left| \widetilde A_m(z_t,\cdot)

\widetilde A_n(z_t,\cdot)
\right|_\infty
\leq
\epsilon_Q,
]

则将 (m,n) 放入同一个当前 value quotient class：

[
m\sim_{z_t}n.
]

阈值 (\epsilon_Q) 不应是任意风格超参数，而应来自 value estimator的误差界，例如 clipped double-Q差异或 bootstrap confidence radius。

控制器随后只维护 quotient-level posterior：

# [ \beta_t(c)

\sum_{m\in c}b_t(m).
]

这一步非常关键，因为它从结构上保证：

- response model可以观察到行为差异；
- latent slots可以捕获不同轨迹规律；
- 但如果这些差异不改变 action advantage，则它们被合并；
- 这些身份或风格信息不能进入动作选择。

这才是把理论中的 (W_C^V) 变成可执行对象，而不是将四个训练来源类别直接称为伙伴原型。

------

# 五、只保留一个回应模型

当前代码同时包含：

1. `PrototypeResponseHeads`；
2. `PrototypeTransitionHeads` 内部的另一套 response branch。

probe评分使用：

```python
transition_response_probabilities
```

实际 belief更新使用：

```python
response_probabilities
```

两套网络虽然都针对同一个 token训练交叉熵，但参数完全独立，也没有 posterior consistency约束。

这意味着：

[
b_{\text{score}}^{a,y}
\neq
b_{\text{runtime}}^{a,y}.
]

统一模型只能有一个 response kernel：

[
P_\psi(y\mid \operatorname{sg}(z_t),a_t,m).
]

它必须同时用于：

- online belief update；
- (J_{\text{use}})；
- (J_{\text{mask}})；
- generic information baseline；
- response contrast；
- deployment inference。

不能再存在两套概率模型。

------

# 六、回应表示必须保留“会改变动作”的变化

当前名义上有 10 个 token，但正式一步 rollout通常只会产生：

- visible；
- unseen；
- local non-agent change；
- terminal。

这几乎只编码可见性和局部几何变化，不能可靠表达：

- 伙伴向哪个方向移动；
- 是否让路；
- 是否抢占对象；
- 是否拾取、放下或交互；
- 是否改变任务阶段；
- 当前回应后 ego应改变哪一个动作；
- 多步延迟回应。

统一模型应仍使用有限 categorical response，保持理论对象有限，但 response code不应手工压缩成四个粗事件。

更合适的定义是一个**价值保持回应编码器**：

# [ y_t

g_\psi
(o_t,a_t,o_{t+1}),
]

其训练目标不是重构 observation，也不是识别伙伴身份，而是预测下一状态的归一化动作价值：

[
\widetilde A_m^{-}(z_{t+1},\cdot).
]

response encoder及其 decoder只读取 detached表示，其损失不进入 recurrent backbone。

因此 response code保留的是：

> 哪些观测变化预示着下一步动作价值结构发生变化。

而不是：

> 哪些变化最容易识别伙伴身份。

随后，value quotient aggregation进一步消除同一控制类内部的 response fingerprint。

------

# 七、删除 next-feature predictor，直接预测 response-conditioned next action values

当前 transition head预测一个高维 next feature：

[
\hat z_{t+1}^{m,a,y},
]

再将其送入非线性 value head得到 continuation value。

这种设计包含两层根本误差：

[
V\bigl(E[z_{t+1}]\bigr)
\neq
E[V(z_{t+1})],
]

而且当前模型只有 scalar prototype value，无法输出 response后应执行的动作。

统一模型应直接预测：

# [ G_m(z_t,a_t,y,u)

E
\left[
Q_m^{-}(z_{t+1},u)
\mid
z_t,a_t,y,m
\right].
]

其中：

- (a_t)：当前普通任务或探查动作；
- (y)：可能观察到的 response；
- (u)：观察 response后下一步可能执行的动作。

训练目标直接来自 target Q vector：

# [ \mathcal L_G

## \sum_{t,m} \operatorname{sg}(q_{t,m}) \left| G_m(\operatorname{sg}(z_t),a_t,y_t,\cdot)

\operatorname{sg}
\left[
Q_m^{-}(z_{t+1},\cdot)
\right]
\right|^2.
]

它只更新 outcome head，不更新 recurrent representation。

这样便不再预测一个难以解释的 hidden mean，而是直接预测控制器真正需要的 response-conditioned continuation action values。

------

# 八、统一的 (J_{\text{use}})、(J_{\text{mask}}) 和实际执行策略

给定当前 posterior (b_t(m))，对于任意合法 primitive action (a)：

# [ P(y\mid z_t,a,b_t)

\sum_m
b_t(m)
P_\psi(y\mid z_t,a,m).
]

若执行 (a) 后观察到 (y)，posterior为：

# [ b_t^{a,y}(m)

\frac{
b_t(m)P_\psi(y\mid z_t,a,m)
}{
\sum_j b_t(j)P_\psi(y\mid z_t,a,j)
}.
]

令：

# [ \widehat r_b(z,a)

\sum_m b_t(m)\widehat r_m(z,a).
]

回应被使用时：

# [ J_{\text{use}}(a)

\widehat r_b(z,a)
+
\gamma
\sum_y
P(y\mid z,a,b)
\max_u
\sum_m
b^{a,y}(m)
G_m(z,a,y,u).
]

回应被屏蔽时：

# [ J_{\text{mask}}(a)

\widehat r_b(z,a)
+
\gamma
\sum_y
P(y\mid z,a,b)
\max_u
\sum_m
b(m)
G_m(z,a,y,u).
]

最佳不使用当前 response的动作价值为：

# [ V_{\text{mask}}

\max_a J_{\text{mask}}(a).
]

净信息价值为：

# [ S(a)

J_{\text{use}}(a)-V_{\text{mask}}.
]

关键变化在于：

> (\max_u) 所选择的 posterior-optimal continuation action，必须在实际观察到 (y) 后由同一个控制器真正执行。

当前实现只在数学上计算 posterior-weighted value，却没有对应的 posterior-conditioned actor。统一模型将该动作选择直接用于下一步执行，因此内部 estimator与实际 policy一致。

------

# 九、取消 base-action override，使用一个统一行为策略

当前流程先采样 `base_action`，再由 controller决定是否覆盖，并将被覆盖步骤从 actor loss中移除。

这会形成一个参数依赖但梯度不完整的 hybrid policy。

统一模型不再区分：

- actor动作；
- probe动作；
- controller-owned动作。

所有 primitive actions都由同一个行为分布选择：

# [ \pi_{\text{exec}}(a\mid h_t,b_t)

\frac{
\pi_{\text{ref}}(a\mid h_t)
\exp\left(
\underline J_{\text{use}}(a)/\alpha
\right)
}{
\sum_{u}
\pi_{\text{ref}}(u\mid h_t)
\exp\left(
\underline J_{\text{use}}(u)/\alpha
\right)
}.
]

其中：

- (\pi_{\text{ref}}) 是冻结的官方 recurrent self-play policy；
- (\underline J_{\text{use}}) 使用 clipped double-Q或保守下界；
- (\alpha) 控制相对官方任务策略的偏离。

这等价于求解：

## [ \max_\pi E_{a\sim\pi} [J_{\text{use}}(a)]

\alpha
D_{\mathrm{KL}}
(\pi|\pi_{\text{ref}}).
]

因此它不是额外加一个“SP保护 gate”，而是统一控制目标本身：

> 在不无必要地偏离已经具备任务能力的参考策略的前提下，依据 posterior-conditioned action value进行策略改进。

(\alpha) 可以通过一个 KL dual variable在线调节，使平均偏离维持在固定 trust-region半径内。

这一结构直接解决 SP 崩塌：

- 当前 adaptation会重写整个 actor，因此 XP改善伴随 SP从144.78下降到最低56.52；
- 新模型的基础任务策略始终由冻结的 (\pi_{\text{ref}}) 保留；
- Path C只通过价值证明充分的 residual action preference偏离参考策略；
- 当 response没有控制价值时，(\pi_{\text{exec}}) 自动退化到参考策略，而不需要人为关闭模块。

任何普通任务动作都可以因为信息价值而获得更高 (J_{\text{use}})，因此仍然实现“用普通任务动作探查”，但不再需要独立 probe override。

------

# 十、统一训练循环

新的 Path C学习不再使用：

```text
prefit
→ training calibration
→ adaptation
→ deployment calibration
```

当前所谓 1M-step prefit在配置下实际只包含约10次大批量 Adam更新，因为：

# [ 250\text{ environments} \times 400\text{ steps}

100000
]

而 prefit batch size同样为100000；1M环境步只形成10个完整batch。

随后模型又执行约20000次 adaptation optimizer updates，但 threshold保持 prefit后固定值，造成模型尺度持续变化而阈值不变。

统一模型从第一条 rollout开始运行同一个循环：

```text
reference policy produces competent behavior
    ↓
collect complete recurrent trajectories with unlabeled diverse partners
    ↓
compute latent TD responsibilities
    ↓
update recurrent Q representation using Bellman loss
    ↓
update single outcome/response model on detached representation
    ↓
update online posterior
    ↓
recompute the same KL-regularized posterior policy
```

模型初始化时：

- (\pi_{\text{ref}}) 已具有任务能力；
- value residual heads接近零；
- response model尚未形成可靠差异；
- 因而 (\pi_{\text{exec}}\approx\pi_{\text{ref}})。

随着 Bellman专家和 outcome model形成，posterior-conditioned residual才逐渐出现。无需用冷启动模型计算 probe score，也无需通过全局分位数人为决定是否触发。

完整损失只有两个逻辑对象：

# [ \mathcal L

\mathcal L_{\mathrm{Bellman}}
+
\mathcal L_{\mathrm{outcome}}.
]

其中：

# [ \nabla_\theta \mathcal L

\nabla_\theta\mathcal L_{\mathrm{Bellman}},
]

即 recurrent representation只接收 Bellman/action-value gradient。

`outcome` 可以用一个联合概率模型统一表示：

[
p_\psi
\left(
y_t,
r_t,
Q^{-}(z_{t+1},\cdot)
\mid
\operatorname{sg}(z_t),a_t,m
\right),
]

而不是当前五个同权重、量纲不同的独立辅助损失。

------

# 十一、对应代码应整体如何替换

| 当前文件/对象                       | 统一替换                                                     |
| ----------------------------------- | ------------------------------------------------------------ |
| `model/backbone.py`                 | 输入 observation、上一ego动作、上一原始奖励、episode boundary的 recurrent value backbone |
| `model/heads.py::SharedActorCritic` | 冻结官方 reference actor；不再训练 shared actor覆盖基础能力  |
| `PrototypeValueHeads`               | 无标签 dueling latent action-value heads，输出 `[B,M,A]`     |
| `PrototypeResponseHeads`            | 删除                                                         |
| `model/transition.py`               | 单一 response-conditioned outcome model，输出 response distribution、reward和next-Q vector |
| `partner_indices` family routing    | 从训练语义中完全删除，仅保留审计字段                         |
| `belief/update.py`                  | 对无标签 value slots进行 posterior更新，并聚合为动态 value quotient posterior |
| `probe/scores.py`                   | 用同一 response kernel和直接 next-Q计算可执行的 (J_{\text{use}})、(J_{\text{mask}}) |
| `probe/controllers.py`              | 删除 base-action override；改为一个 KL-regularized action distribution |
| `actor_owned_action`                | 删除                                                         |
| `prefit.py`                         | 删除独立 prefit流程，逻辑合并进统一 outcome update           |
| `adaptation.py`                     | 替换为 recurrent latent Bellman update和target-network update |
| calibration quantiles               | 删除，不再用score分位数代替真实控制价值                      |
| `response_contrast_runtime.py`      | 保留，但 mask/use必须作用于每一步 posterior-conditioned action policy |
| formal matrix infrastructure        | 保留                                                         |
| outer-unit、seed、receipt和hash治理 | 保留                                                         |

------

# 十二、该统一方案如何逐项修复当前结果

| 当前表现                              | 根因                                          | VQBC中的直接修复                                          |
| ------------------------------------- | --------------------------------------------- | --------------------------------------------------------- |
| no-probe已获得几乎全部XP提升          | actor学习多伙伴平均折中                       | policy直接条件化于posterior，不再只能平均                 |
| `A2-use−A2-mask≈0`                    | belief不进入普通actor                         | belief进入每一步动作分布                                  |
| SP严重下降                            | adaptation重写整个actor                       | 冻结official reference policy，使用KL正则化value residual |
| decision-focused只比random略好        | probe是外部动作覆盖                           | 信息价值与任务价值进入同一个action value                  |
| family belief语义不稳定               | family是训练来源标签                          | latent slots只由TD residual形成                           |
| response可能识别风格而非控制差异      | 直接在family上做Bayes                         | posterior先经过value quotient聚合                         |
| 两套response概率不一致                | score与belief使用不同head                     | 全系统只使用一个response kernel                           |
| `J_use`不可执行                       | 没有posterior-conditioned continuation policy | 观察response后实际执行同一模型选出的(u^*(y))              |
| next-feature模型偏差大                | (V(E[z'])\neq E[V(z']))                       | 直接预测response-conditioned next action values           |
| 全局threshold随模型漂移失效           | score尺度与固定分位数脱节                     | 无threshold，直接求解regularized value policy             |
| 辅助梯度影响backbone步长              | 全树global clipping                           | Bellman和outcome参数组分别裁剪和优化                      |
| early/mid checkpoints主要产生能力差异 | family混合competence和convention              | value quotient忽略不改变action ranking的offset差异        |

------

# 十三、修订后必须满足的不可妥协性质

统一实现完成后，以下测试必须成立。

### Belief 必须改变正常动作

固定同一个 (z_t)，分别输入两个 posterior：

[
b^{(1)},b^{(2)}.
]

若其 posterior-weighted advantage ranking不同，则：

[
\pi_{\text{exec}}(\cdot\mid z,b^{(1)})
\neq
\pi_{\text{exec}}(\cdot\mid z,b^{(2)}).
]

若 value quotient相同，则动作分布必须相同。

### Mask/use 必须阻断真实控制路径

在 `A2-mask` 中屏蔽当前 response后，不仅 probe controller，而且后续每一步普通动作都不得读取该 response引起的 posterior变化。

### 不允许 family label影响任何预测

改变：

- seed；
- checkpoint index；
- SP/OP label；
- training run ID；

但保持轨迹内容不变时，模型输出必须逐元素相同。

### 内部 score必须预测实际效应

按预测 (S(a)) 分箱后，实际：

[
A2\text{-use}-A1
]

应随 score单调增加。最高分箱必须具有正的 realised net effect。

### SP能力必须保留

不应再接受“XP接近零但SP减半”作为正确性能。适应后的目标必须是：

- XP显著高于参考 SP backbone；
- SP保持在参考策略可接受范围；
- response-use effect显著大于零；
- decision-focused明显优于 no-response-use；
- 改善不能主要由平均化或不行动产生。

------

# 最终裁定

当前 Path C v3不应继续通过调节：

- threshold；
- budget；
- candidate window；
- family weights；
- prefit步数；
- safety rule；
- actor learning rate；

来试图获得更好的结果。

这些参数只会改变表面症状，因为当前核心仍然是：

[
\text{平均 actor}
+
\text{不进入 actor 的 belief}
+
\text{外部 probe override}.
]

正确的统一修复是：

[
\boxed{
\begin{aligned}
&\text{用 TD-only recurrent action-value representation 替代 scalar critic；}\
&\text{用无标签 value slots 替代人工 partner families；}\
&\text{用动态 action-advantage quotient 定义 }W_C；\
&\text{用一个 response-conditioned outcome model 替代双 response heads；}\
&\text{用直接 next-Q prediction 替代 next-feature prediction；}\
&\text{让 posterior进入每一步正常动作；}\
&\text{用一个 KL-regularized Bellman policy同时处理任务与探查；}\
&\text{删除 base-action override、prefit/calibration和固定threshold。}
\end{aligned}
}
]

这是一套单一模型、单一训练循环和单一执行策略。它直接修复当前因果链断裂的根因，而不是在现有错误架构上继续叠加保护机制。

它不能在未运行前保证某个具体 benchmark数值，但它使“观察伙伴回应后改变后续最优动作”第一次成为代码中真实执行的机制；当前 v3并不具备这一性质。