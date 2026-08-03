# 总体结论

我按六条相互独立的审计线交叉检查了该分支：科学问题、理论与可识别性、模型与训练动力学、伙伴分布、实验统计、创新性与文献边界。

结论是：**当前 DELTA-ZSC V6 的工程契约相当完整，但科研方案尚未在底层形成闭环。** 主要风险不是“某些超参数可能不合适”，而是：

1. 方案同时混用了“伙伴身份/能力”“动态协作协议”“历史信息价值”“动作价值控制”四种不同对象，却没有明确它们之间的生成关系。
2. 训练损失所学习的量、理论所分析的量、生态实验所估计的量和正式部署可读取的量并不相同。
3. 即使最终 XP 分数提高，也无法据此证明提升来自所声称的“Bayesian partner belief”；即使 M1 preflight 通过，也不能证明该机制在科学上成立。
4. 当前正式 SOTA 判决式在统计上不能执行，因为它要求对一个只有已发表均值和标准差、没有原始训练运行节点的基线做 run-node bootstrap 显著性比较。

因此，当前方案不适合直接冻结为正式方法。它更接近一个**高度工程化但科学语义未闭合的复合系统**。

| 审计维度       | 当前判定       | 核心原因                                                  |
| -------------- | -------------- | --------------------------------------------------------- |
| 科学对象       | 不闭合         | “伙伴信念”没有区分稳定伙伴属性与动态共同协议              |
| 因果可识别性   | 不成立         | task GRU 和 belief GRU 都能独立编码伙伴历史               |
| Bayes 语义     | 不成立         | 没有生成模型、似然、规范后验或校准                        |
| 价值学习       | 高风险         | Q 在后验均值训练，却在未监督 Gaussian 样本上计算 regret   |
| 伙伴生成器     | 底层几何不成立 | 四个源锚点只覆盖八维 code 空间的二维子空间                |
| 理论生态外推   | 不成立         | 使用部署时不可见的完整伙伴动作，并误用 Bernoulli 特例参数 |
| 创新性         | 部分可保留     | 多数模块已有直接近邻，真正潜在创新尚未成为方法中心        |
| SOTA 判决      | 统计上无效     | 不能从已发表 mean±SD 构造配对 run-node 差值区间           |
| 工程与研究诚信 | 较强           | run-disjoint、固定最终 checkpoint、资源账本等值得保留     |

------

# 一、最根本的问题：方案没有明确自己究竟在推断什么

OvercookedV2 的 Test-Time Protocol Formation 测的是：两个陌生智能体如何在交互过程中形成新的、动态演化的共同协议，而不仅仅是识别一个事先存在的固定伙伴类型。基准论文明确把这类任务定位为需要在线适应和测试时协议形成的问题。([arXiv](https://arxiv.org/abs/2503.17821))

当前绑定设计却以

[
q_\phi(z_t\mid H_t)
]

作为“伙伴信念”，生成器每个 episode 采样一个固定 code，并用这个 code 生成伙伴行为。尽管 (z_t) 的网络输出随时间变化，但监督体系实际上仍然把差异组织为“不同伙伴 code 产生不同 continuation signature”。这没有区分：

- 伙伴相对稳定的能力或行为倾向 (u)；
- 由双方行为共同产生、可随时改变的协议状态 (c_t)；
- 环境和任务的部分可观测状态 (x_t)。

绑定设计、README 和生成器代码都把核心对象描述为 partner belief/code-conditioned partner，而没有给出动态共同协议的转移定义。

这会产生一个概念性错误：**伙伴策略并不是外生不变参数。** 在协议形成任务中，伙伴的下一步行为会响应 ego 之前的动作；ego 的动作既在完成任务，也在塑造共同协议。现有 T2/T3 理论把隐藏惯例 (z) 设为固定外生变量，把历史仅作为识别 (z) 的证据，因此理论研究的是“识别后路由”，而不是“共同形成协议”。

合理的形式化至少应为：

[
b_t = p(x_t,u,c_t\mid H_t),
]

其中

[
c_{t+1}\sim P(c_{t+1}\mid c_t,H_t,a_t^{ego},a_t^{partner}).
]

当前方案没有这个对象，所以“Bayes coordination”究竟是伙伴分类、能力估计、惯例识别还是共同协议跟踪，并不明确。

------

# 二、belief 机制在当前架构中不可识别

这是代码层面最严重的问题之一。

`task_encoder.py` 中的 task path 是 CNN 加 GRU，它读取当前观测、之前的 ego 动作和 episode boundary。它本身已经能够从完整历史中编码伙伴行为。与此同时，`belief_encoder.py` 读取前后观测、观测差、ego 动作和自身 GRU，也能编码任务状态、地图变化和自我运动。两个通路接收了高度重叠的信息。

actor 随后同时读取：

[
\pi(a_t\mid \text{task_features}_t,\text{belief_summary}_t).
]

context dropout 只把 belief summary 替换为 prior summary，却保留完整的 recurrent task features。因此所谓 prior/generalist 路径仍然能够通过 task GRU 读取伙伴历史，并不是一个真正“不适应伙伴的 generalist”。

问题进一步被梯度路由放大：

- PPO 可以训练 task encoder；
- raw-Q 可以训练 task encoder；
- counterfactual anchor 也可以训练 task encoder；
- belief encoder 只是被单独路由更新。

也就是说，模型完全可以把所有伙伴适应能力藏在 task GRU 中，而让显式 belief 只负责满足辅助损失。

因此，以下结果都不能证明 belief 有效：

- belief latent 对伙伴可分；
- belief swap 改变动作；
- posterior 方差下降；
- response decoder 可预测伙伴信息；
- full-context 优于 prior-summary。

因为 task GRU 仍可能携带相同信息。

这也解释了为什么“加 belief 的消融增量”不是充分证据：不同架构容量、优化路径和正则强度都会变化，但没有隔离 belief 所承载的因果信息。

必须增加三类控制：

1. **task-history leakage control**：task path 只能承载任务状态，不能自由吸收伙伴历史。
2. **history shuffle control**：保持任务状态和当前观测不变，打乱伙伴历史；真正的适应模型应明显退化。
3. **context swap control**：在匹配任务状态之间交换协议表征，动作变化方向必须与真实 continuation-value 排序一致。

否则，“belief-conditioned coordination”在科学上不可识别。

------

# 三、当前所谓 Bayesian posterior 实际上只是一个带均值和方差接口的 RNN embedding

绑定设计把

[
q_\phi(z_t\mid H_t)=\mathcal N(\mu_t,\operatorname{diag}\sigma_t^2)
]

称为 posterior，但当前系统没有定义：

- (p(z)) 对应什么真实随机变量；
- (p(H_t\mid z)) 或 (p(y_{t+1}\mid z,H_t,a_t)) 的生成模型；
- Bayes 更新或近似 Bayes 更新；
- posterior calibration；
- 覆盖率或 proper scoring rule；
- 与真实伙伴不确定性的对应关系。

response loss、PPO、raw-Q、decision equivalence、IB、Q-policy 等目标被任意加权后共同更新 belief，这不是一个 ELBO，也不构成规范的 posterior inference。

更严重的是，当前 uncertainty 是由 log-standard-deviation 在区间 `[-5,2]` 中的位置直接归一化得到的。robust generalist loss 又用这个网络自己输出的 uncertainty 去加权 full-context 与 prior-context 的 KL。

这提供了一条简单的“置信度作弊”路径：

[
\log\sigma\rightarrow -5
\quad\Longrightarrow\quad
\text{normalized uncertainty}\rightarrow0
\quad\Longrightarrow\quad
L_{\text{robust}}\rightarrow0.
]

IB 的注册权重只有 `0.001`，robust loss 权重是 `0.1`，在 belief gradient cap 中两者最大贡献相差 100 倍。

因此 log-variance 不一定表示真实 epistemic uncertainty，它也可能只是：

- actor 的额外连续特征；
- 规避 robust regularization 的控制变量；
- 多目标梯度妥协后的任意数值。

当前方案要么应把它改称为**stochastic context embedding**，不再声称 Bayes；要么必须增加一个明确的生成语义、proper likelihood 和 held-out calibration 协议。

------

# 四、decision regret 在未受监督的 latent 区域上计算

这是一个直接影响训练正确性的底层问题。

raw-Q 的 dense Retrace 和 counterfactual anchor loss，都在由真实合法历史产生的 posterior mean 上训练：

[
Q(x_t,\mu_t,a).
]

但 decision regret 会从

[
z^{(k)}\sim\mathcal N(\mu_t,\sigma_t^2)
]

采样 16 个 latent particle，再调用

[
Q(x_t,z^{(k)},a).
]

`model.py` 甚至为 particle path 人工构造了 `[latent, zeros, zero-uncertainty]` 的 summary。训练代码并没有为这些采样 latent 提供与之对应的真实 continuation target。

所以

[
R_t=
\mathbb E_z\max_aQ(x_t,z,a)
-\max_a\mathbb E_zQ(x_t,z,a)
]

很可能主要测量神经网络在训练分布外 latent 输入上的曲率，而不是伙伴不确定性的任务价值。

之后这个量又被用于：

- decision-regret shaping；
- action-range normalization；
- 训练解释；
- 生成器的 decision-signature 构造。

这形成一条错误放大链：

[
\text{off-manifold Q error}
\rightarrow
\text{fake regret}
\rightarrow
\text{shaped reward / generator diversity}
\rightarrow
\text{new training distribution}
\rightarrow
\text{更严重的 Q 偏差}.
]

M1 如果只检查 posterior mean 处的动作排序 Spearman，并不能验证 particle path。

解决方案只有两个：

1. 对每一个用于 regret 的 latent hypothesis，都提供一致的真实 continuation 标签；
2. 不再在任意 Gaussian particle 上计算 regret，改用由真实合法历史或 bootstrap context encoder 产生的、处于训练支持集内的 hypotheses。

------

# 五、matched-code counterfactual 的因果语义不成立

当前 matched pair 构造方式是：

1. 从真实 rollout 中抽取一个中间环境状态；
2. 将 generator recurrent carry 清零；
3. 给两个分支分别设置随机 code；
4. 把 `episode_start` 设为 true；
5. 从这个中间状态运行 16 步 probe；
6. 用之后的 all-action continuation signature 监督两个 belief mean 的距离。

这相当于把一个“刚开始新 episode 的伙伴 RNN”拼接到一个 episode 中段的环境状态。这个伙伴状态并不是训练或部署中自然出现的状态分布。

更关键的是，设计没有保证 16 步合法历史包含足够证据来区分两个 code。可能出现：

[
H^{(a)}*{t:t+16}\approx H^{(b)}*{t:t+16},
]

但由于两个隐藏 code 未来将产生不同伙伴行为，continuation signature 不同。此时 decision-equivalence loss 要求 belief 将两个几乎相同的合法历史分开。这在信息论上是不可能的，只会迫使网络利用：

- 微小环境噪声；
- task-state 差异；
- RNG fingerprint；
- 数值伪特征。

FOUNDATIONAL 文档本身已经承认：非零 decision regret 不代表信息可以从合法历史中获得。可是 anchor 生成流程没有落实这一限制。

正确做法应是：

- matched histories 必须来自真实完整 episode；
- 保留真实 partner recurrent state；
- 不做 mid-episode code reset；
- 相同合法历史必须作为 invariance pair；
- 只有在 held-out classifier 或可见历史检验确认两段历史可区分后，才允许施加 separation target；
- 不可区分但未来最优动作不同的样本应被记录为 irreducible ambiguity，而不是强迫表征分离。

这不是一个额外“门控机制”，而是监督标签是否可实现的基本合法性条件。

------

# 六、连续伙伴生成器的 code 几何在数学上没有被初始化

这是当前方案中非常具体、但文档没有识别出的根本缺陷。

配置使用八维 partner code。生成器初始化只使用四个 SP/OP/SA/FCP source anchor。`source_code_anchors()` 实际生成：

[
v_2=-v_0,\qquad v_3=-v_1.
]

所以四个八维锚点的线性秩最多为 2。之后训练却从整个

[
[-1,1]^8
]

均匀采样 code。

这意味着：

- source distillation 只约束了一个二维子空间；
- 剩余至少六个方向没有任何行为语义；
- 均匀随机 code 几乎全部位于 source support 之外；
- 所谓连续伙伴分布主要是高维外推，而不是四种有效行为之间的插值。

smoothness loss 也只沿一个固定方向做 `code + 0.05 * direction`，不能约束整个八维空间。

所以当前生成器在训练初期产生的“多样伙伴”很可能只是未约束网络在 OOD code 上的随机行为。

更严重的是，生成器 BR-diversity 使用当前 ego 的 learned Q action signature，并对每个 code 自己访问到的不同状态进行 episode 平均。

因此 BR-diversity 同时混合了：

- 伙伴行为差异；
- 不同 code 导致的状态访问分布差异；
- 当前 Q 的估计误差。

生成器可以通过把 ego 带到不同状态或放大 Q 错误来获得 diversity bonus，而不必产生真正不同的协作惯例。与此同时，imitation weight 在训练进度达到 30% 后降到零，generator 可以完全脱离 source 行为。

这与已有生成式/对抗伙伴工作直接重叠。GOAT 特意冻结生成模型参数、只优化 embedding，以避免 adversarial partner 演变成破坏者；ROTATE 和 UPD 也分别从 regret-driven open-ended training 与 learnability curriculum 处理动态伙伴生成。([arXiv](https://arxiv.org/abs/2504.15457))

合理修订应是：

- 四个 source 时 code 维度最多取 3，并使用仿射独立 simplex anchor；或者至少准备 (d+1) 个独立 source 来支持 (d) 维 code；
- 从拟合得到的 latent support 采样，而不是从整个立方体均匀采样；
- diversity 必须在一组共同环境状态上，用真实 all-action continuation 计算；
- generator behavior model 应冻结或受严格 KL/support constraint；
- adversarial 更新应主要作用于 code 分布，而不是让整个伙伴网络持续漂移。

------

# 七、主论点 Θ2 的生态理论验证目前不成立

## 7.1 使用了部署时不可见的信息

生态轨迹规格明确记录完整伙伴动作序列，不记录 ego 原始观测，并把这种口径称为“完整行为上限”。Bayes 路由器也直接用伙伴动作前缀进行模板匹配。

正式部署政策却只能读取 ego 的局部观测、自己之前的动作、done 和 recurrent state。

因此 E3–E5 测得的不是：

[
I(\text{legal ego history};\text{decision-relevant protocol}),
]

而是：

[
I(\text{full partner action oracle};\text{partner run/type}).
]

即使 P4 通过，也只能证明“知道完整伙伴动作时可以路由”，不能支持 DELTA-ZSC 在合法部署信息下可以适应。

“这是一个上限估计”也不能解决问题，因为当前论文把它用于生态外推、历史需求计算和理论主张，而不是仅作为 unattainable oracle。

## 7.2 生态 (\kappa) 不是 T3 中的 (\kappa)

T3 的 (\kappa) 来自明确的 i.i.d. Bernoulli 证据模型：

[
e_i\mid z\sim
\operatorname{Bernoulli}\left(\frac{1\pm\kappa}{2}\right).
]

生态协议则把 (\kappa) 定义成两组伙伴单步边际动作分布的 TV。伙伴动作在 OvercookedV2 中明显依赖：

- 当前环境状态；
- ego 之前的动作；
- 时间；
- 伙伴 RNN carry；
- 当前协议。

这些证据既不独立，也不同分布。单步边际 TV 不能直接代入

[
\Theta\left(\frac{\log(\Delta/\epsilon)}{\kappa^2}\right).
]

“在 Bernoulli 合成数据上无偏”只证明估计器能恢复其构造时使用的 Bernoulli 参数，不证明它在受控 POMDP 历史中估计同一个信息量。

## 7.3 classifier error 只能给 TV 下界

等先验二分类中：

[
TV=1-2p_e^*
]

只对最优 Bayes classifier 的错误率 (p_e^*) 精确成立。任何有限样本经验 classifier 都满足

[
\hat p_e\ge p_e^*,
]

因此

[
1-2\hat p_e\le TV.
]

当前 lookup-style empirical classifier 得到的最多是一个依赖模型容量和样本量的 TV 下界，不能直接当作 TV 点估计。

## 7.4 (\hat\Delta\cdot\widehat{TV}/2) 不适用于生态任务

T2 精确式要求：

- 两个等先验惯例；
- 两个一一对应的模式；
- 每个历史上的正确/错误收益差都是同一个常数 (\Delta)。

生态协议中的 (\hat\Delta) 却来自全回合面板上“最大模式均值减次大模式均值”的跨伙伴平均。历史相关 payoff gap 与 TV 分别平均后再相乘，一般不等于真实可恢复价值。

一般情形需要的是 value-weighted distinguishability，例如二元特例下：

# [ D_V(t)

\frac12
\int \Delta(h)
\left|p_1^t(h)-p_0^t(h)\right|dh,
]

而不是

[
\mathbb E[\Delta(H)]\cdot TV(P_0^t,P_1^t)/2.
]

## 7.5 T3 证明中存在明确数学错误

FOUNDATIONAL §10.3 使用：

[
\log\frac{1+\kappa}{1-\kappa}\le 2\kappa,
\qquad 0\le\kappa\le\frac12.
]

取 (\kappa=1/2)：

[
\log 3\approx1.099>1=2\kappa.
]

所以该不等式是错误的。之后采用更保守的 (4\kappa^2) 结果可能可以通过另一条界重新证明，但当前证明链本身不成立。

## 7.6 S2 的 180/180 不是理论验证

S2 任务是直接按 T2/T3 假设构造的 Bernoulli 特例，并用同样的 closed-form 公式生成和检查结果。180/180 说明代码正确实现了注册恒等式，但这更接近 theorem unit test，而不是独立数值验证。

所以当前“主论点 Θ2 已在受控任务验证，接下来生态外推”表述过强。

------

# 八、正式 SOTA 判决式在统计上无法执行

`SOTA_BASELINE.md` 明确规定：

- 只使用 OvercookedV2 论文 Table 2 的 FCP 数值；
- 不要求自训复现；
- Simple 使用 `6±29`；
- Wide 使用 `23±40`；
- DELTA 用 run-node bootstrap 判断差值区间是否跨零。

这里存在三个问题。

第一，`6±29` 和 `23±40` 是已发表的均值和标准差，不是 10 个原始 run node。没有原始 FCP 训练运行、矩阵节点和 episode-key 数据，就不能和 DELTA 的运行节点做 paired run-node bootstrap。

第二，不能把 DELTA 的 bootstrap 区间与 FCP 的点估计相减，再称为“两方法差值置信区间”。这忽略了基线训练运行的不确定性。

第三，FCP Table 2 可以作为 **2025 年 OvercookedV2 原论文参考线**，但不能自动等同于 2026 年整个 ZSC/AHT 领域的 SOTA。当前已有 CooT、TALENTS、GOAT、ROTATE、UPD、Theory-of-Mind adaptive ensemble、ICRL4AHT 等直接相关工作；其中并非全部使用完全相同的 Test-Time Simple/Wide 协议，但正因如此，文档必须明确区分“同 benchmark reference”与“领域 SOTA”，而不是只注册一个旧标量。([arXiv](https://arxiv.org/abs/2506.23549))

正式判断应改为：

1. 在固定 Official commit 上重新训练或获取作者原始 FCP/OP/SA checkpoint；
2. 保存全部 run-level 和 cell-level 数据；
3. 使用相同 episode keys；
4. 对训练运行节点做成对或双样本层级推断；
5. 已发表 Table 2 数值只作为外部 sanity check。

否则可以说“超过已发表点估计”，不能说“显著超过 SOTA”。

------

# 九、当前创新性边界被明显高估

## 已经被现有工作占据的部分

“使用历史适应陌生伙伴”不是新颖点。CooT 已经用 recent interaction history 做 in-context coordination，并与 population、fine-tuning 和 Meta-RL 路线比较。([arXiv](https://arxiv.org/abs/2506.23549))

“显式学习 latent partner strategy 并在线调整”也不是新颖点。TALENTS 使用 VAE latent strategy、聚类和 fixed-share regret minimization。([arXiv](https://arxiv.org/abs/2507.05244))

“简单 recurrent agent 能涌现 partner representation”已经被 NeurIPS 2025 工作直接验证，而且该工作强调社会压力和任务分工条件，而非必须依赖专门的 belief architecture。这个结果使 full-history recurrent PPO 成为 DELTA 必须正面对比的核心基线，而不是普通消融。([nips.cc](https://nips.cc/virtual/2025/poster/117556))

“主动识别伙伴并进行 context-aware exploration”已有 PACE。([Proceedings of Machine Learning Research](https://proceedings.mlr.press/v235/ma24n.html))

“在线生成暴露 ego 弱点的伙伴”已有 GOAT、ROTATE 和 UPD。([arXiv](https://arxiv.org/abs/2504.15457))

“部分可观测下的 Bayesian teammate identification”已有 RecBayes；2026 年 7 月的 CE-CM 又进一步把 approximate Bayesian inference 用于 task-invariant hidden capability estimation。([arXiv](https://arxiv.org/abs/2506.15756))

当前 `LITERATURE_MATRIX.md` 对 ReCollab 的描述也不准确或至少严重不完整。该工作实际中心是 retrieval-augmented LLM teammate behavioral modeling，而不是文档中概括的普通 early-window router。 ([arXiv](https://arxiv.org/abs/2512.22129))

## 仍可能形成真正创新的部分

当前项目最有潜力的创新不是“Gaussian belief + actor + generator”的组合，而是：

> **仅基于合法历史，用真实共同前缀、all-action continuation 回报直接监督一个 decision-equivalent protocol state，并以 run-disjoint、algorithm-family-disjoint 的因果控制测量它实际回收了多少协调价值。**

这里有三个可能站得住的创新点：

1. **value-weighted protocol representation**：latent 语义不是伙伴 ID，而是 centered continuation action-value signature。
2. **legal-history causal supervision**：所有监督来自真实可部署历史和有效 counterfactual continuation，不使用伙伴动作 oracle 或隐藏 code。
3. **recoverable-value evaluation**：比较 legal-history policy、history-shuffled policy、state-only policy 和 oracle continuation，测量实际回收比例。

这比“第一个把 Bayes belief、counterfactual、generator、PCGrad 放在一起”更明确，也更容易形成 ICLR/NeurIPS 级主张。

------

# 十、其他重要的训练可行性问题

## 10.1 当前项目状态不满足自己的四源输入假设

生成器代码要求恰好四个 SP/OP/SA/FCP source；`source_code_anchors()` 对 `count != 4` 直接报错。

但证据台账记录服务器当前只有 SP 和 OP 各 10 个完整 seed，SA/FCP 在当前树上不可用。

因此当前 M1 或 formal run 的输入条件尚未成立。它不是“配置以后补”，而是训练分布的核心组成部分尚不存在。

## 10.2 response objective 主要预测运动学，不是协议语义

structured response 预测的是：

- partner visibility；
- relative position；
- direction；
- inventory；
- inventory change。

所谓 interaction change 实际是可见 inventory change，不是伙伴意图、角色承诺、任务分配或协议状态。

decoder 同时读取 task features 和 belief summary。虽然 task features 被 stop-gradient，但 decoder 仍可直接依赖 task features 完成位置、方向和 inventory 预测，从而绕过 belief。

因此 response loss 较低并不能证明 belief 学到了伙伴模型。

## 10.3 当前方法没有定义行动条件的信息价值

decision regret 测的是“如果已经知道正确 latent，动作选择可能改变多少”，但没有估计：

[
p(y_{t+1}\mid H_t,a_t),
]

也没有计算某个 ego action 能获得多少伙伴信息、付出多少任务代价。

所以当前方案主要是被动读取历史，而 Test-Time Protocol Formation 很可能需要主动试探或主动建立惯例。PACE 等工作已经表明 context-aware probing 是独立的重要机制。([Proceedings of Machine Learning Research](https://proceedings.mlr.press/v235/ma24n.html))

## 10.4 belief gradient 是在旧参数上算、在新 heads 上应用

训练顺序是：

1. 在 `snapshot_params` 上计算八个 belief objective gradient；
2. 对 actor/value 做四轮 PPO 和 64 minibatch；
3. 更新 raw-Q；
4. 更新 counterfactual heads；
5. 更新 response head；
6. 最后把旧 snapshot 上计算的 belief gradient 应用到已经变化的 heads 上。

因此最终 belief update 并不是当前模型任何已定义总目标的梯度。

同时，belief update 会改变 actor logits，却不在 PPO scan 报告的最终 approximate KL 中。即使所有 PPO minibatch 各自满足 clipping，随后 belief update 仍可能把完整策略推离 behavior policy。

固定顺序 PCGrad 又把 PPO 放在第一位、robust 放在最后，后面的梯度只能相对前面的梯度投影，因此目标优先级由列表顺序隐式决定，而不是由科学定义决定。

## 10.5 counterfactual 信号量偏小且噪声较高

正式每个 run 只有：

- 29 次 anchor trigger；
- 每次 64 个 anchor world；
- 总计 1,856 个 anchor states；
- 其中 matched pairs 只有 464 对；
- 每动作 4 个 continuation replica；
- continuation horizon 128。

与此同时要训练八维 latent、两个 Q head、decision-equivalence metric、response model、actor coupling 和 generator。

当前方案没有给出：

- all-action return 方差；
- 4 replica 的排序错误概率；
- 1,856 状态对历史空间覆盖率；
- latent metric 的统计功效；
- replay effective sample size。

这意味着“anchor 预算固定且完整”并不等于它在统计上足以学习目标。

## 10.6 总成本高于文档呈现出的直觉

单 run 是 41,607,168 attempted transitions。10 seeds、两个布局，仅 V6 训练就是：

# [ 41{,}607{,}168\times20

832{,}143{,}360
]

次 transition。

再加双布局 10×10×500×400 的正式矩阵，V6 自身已经约 8.72 亿 transition，还不包括：

- upstream SP/OP/SA/FCP；
- common-partner scoreboard；
- baseline 复现；
- mechanism audits；
- 开发矩阵；
- 失败实验。

正式预算按 ego-PPO steps 主对齐、总步数仅辅助报告，会系统性弱化 DELTA 额外 38.9% 训练交互成本。

SOTA 比较必须同时报告：

- total simulator transitions；
- upstream partner cost；
- wall-clock GPU hours；
- peak memory；
- deployable parameters；
- performance–compute frontier。

------

# 十一、研究治理文档内部存在相互矛盾的主张

`RESEARCH_PROGRAM.md` 一方面把“超越 SOTA”定义为唯一核心目标，另一方面把 Θ2 信息时序理论定为主论点，把 Θ3 方法称为“载体论点”。

`PAPER_STANDARD.md` 又把论文定义成 measurement-first method paper，并强调即使方法不成功，L1–L3 也能作为保底论文。

这实际上是两个不同项目：

- 项目 A：以 SOTA performance 为中心的方法论文；
- 项目 B：以信息价值测量和理论外推为中心的 measurement paper。

把二者同时设为中心，会导致：

- 方法设计被理论旁路牵引；
- 理论使用与方法不同的观测口径；
- 方法失败后随时切换论文主张；
- 无法明确主要贡献和主要 baseline。

此外还有三处治理冲突：

1. `TRACKS_AND_GOVERNANCE` 的晋升条件只有注册基线和冻结 commit，而 `EXPERIMENT_LADDER` 又要求 S4 开发矩阵通过。
2. `PAPER_STANDARD` 说 S5 必须比较 SP、OP、FCP、MEP、PLASTIC、PECAN 等，`SOTA_BASELINE` 却规定只需已发表 FCP 标量。
3. S1–S4 全部标记 `scientific_readout_allowed:false`，但 L1–L3 论文主张又直接依赖 S1/S2 结果。

“人类裁决”可以决定资源投入方向，但不应被写成科学论点复活的证据。它应进入 decision log，而不是 thesis justification。

------

# 十二、建议的统一重构方向

不建议再在当前 V6 上逐项增加补丁。更合理的是把项目统一重定义为：

## DELTA-ZSC：Decision-Equivalent Protocol Inference

核心科学主张改为：

> 在只能读取 Official 合法局部历史的条件下，学习一个随交互动态演化、以 ego 动作价值排序为语义的协议状态，可以在训练运行和算法族均留出的伙伴上产生可归因的 XP 提升。

## 12.1 三个明确对象

[
x_t=f_s(H_t)
]

表示任务状态或任务 belief；

[
u=f_u(H_t)
]

表示可选的稳定伙伴能力/倾向；

[
c_t=f_c(H_t)
]

表示动态共同协议状态。

一个共享 actor：

[
\pi_\theta(a_t\mid x_t,u,c_t).
]

仍然只需要一套 actor 和一套共享 critic，不需要为每类伙伴训练独立网络。

## 12.2 直接以动作价值签名定义协议语义

对真实合法历史定义：

# [ A_t(a)

## G(H_t,a)

\frac1{|\mathcal A|}
\sum_{a'}G(H_t,a').
]

使用共同随机数 all-action continuation 估计 (A_t)，训练：

[
\hat A_\psi(x_t,c_t,\cdot)\approx A_t(\cdot).
]

这样 latent 的科学语义不是“伙伴身份”，而是：

> 历史中会改变 ego 最优动作排序的部分。

这与项目原始动机一致，同时避免任意八维 Gaussian 的语义漂移。

## 12.3 重新构造合法 counterfactual

所有 anchor 必须来自真实 episode，并保存真实：

- environment snapshot；
- ego recurrent state；
- partner recurrent state；
- legal ego history；
- partner source lineage。

强制不同 ego 首动作后继续运行，但不重置 partner carry、不替换隐藏 code。

matched pair 分成两类：

- **observable-equivalent pair**：合法历史相同，强制 latent 一致；
- **decision-distinct pair**：合法历史可区分且 action signature 不同，监督 latent 分离。

不可识别 pair 不进入 separation loss，而作为 irreducible regret 数据。

## 12.4 不再在任意 Gaussian particle 上计算 Q

更合理的 uncertainty 来源是：

- bootstrap history encoder；
- distributional action-signature head；
- small set/particle posterior，由真实训练 history 产生。

每个 hypothesis 都必须出现在监督分布中。可以用一个共享 critic 对这些 hypotheses 条件化，仍然不是“一类伙伴一套 critic”。

## 12.5 把 response model 改成行动条件的协议响应模型

训练：

[
p_\omega(y_{t+1}^{partner}\mid H_t,a_t^{ego}),
]

其中 (y) 应包含：

- partner next visible action outcome；
- task allocation event；
- pickup/drop/delivery responsibility；
  -让位、争抢、等待等协议事件。

然后定义行动条件的信息价值：

# [ \operatorname{VOI}(a)

## \mathbb E_y \left[ \max_{a'}\mathbb E[Q(a')\mid H_t,a,y] \right]

## \max_{a'}\mathbb E[Q(a')\mid H_t]

C_{\text{task}}(a).
]

这才能把当前理论中的“等待成本”与 Test-Time Protocol Formation 的主动协商联系起来。

## 12.6 重建伙伴分布

先建立静态但足够广的真实伙伴集：

- 多 seed SP/OP/SA/FCP；
- 不同 checkpoint 阶段；
- 不同超参数；
- heuristic partners；
- 与训练算法族不同的测试伙伴。

若保留生成器：

- 学习 source-policy behavior embedding；
- 对 embedding support 拟合分布；
- 只从 support 内采样；
- diversity 在共同状态上用真实 continuation signature 计算；
- generator 主网络冻结或强 KL 约束；
- adversarial 更新主要调整 code distribution；
- 保存 archive 防止遗忘。

## 12.7 简化优化目标

当前八目标 PCGrad 不利于定位机制。统一版本应收缩到：

# [ L

L_{\mathrm{PPO}}
+\lambda_A L_{\mathrm{signature}}
+\lambda_R L_{\mathrm{response}}
+\lambda_S L_{\mathrm{state/protocol\ separation}}.
]

先让每个目标对应一个清晰的估计量。decision-regret shaping、IB、robust KL、Q-policy coupling 只能在证明带来增量后重新加入，而不能作为从第一步就同时存在的复杂耦合项。

这不是“分阶段阻断”，而是避免让八个不可识别机制互相掩盖。

------

# 十三、建议采用一个统一开发矩阵，而不是继续增加门槛

为了符合“performance 是唯一核心目标”且避免过多 gate，可以直接运行一个统一、端到端开发矩阵：

| 方法                                       | 含义                                              |
| ------------------------------------------ | ------------------------------------------------- |
| B0：Full-history recurrent PPO             | 强制核心基线；检验普通 RNN 是否已经能涌现伙伴适应 |
| B1：B0 + 显式 protocol encoder             | 检验架构本身是否有增益                            |
| B2：B1 + 合法 all-action value supervision | 检验 decision-equivalent supervision              |
| B3：B2 + action-conditioned VOI            | 检验主动协议形成                                  |

要求所有方法：

- 相同训练伙伴分布；
- 相同总 simulator transitions；
- 相同 actor/critic 容量等级；
- 相同 run-disjoint 测试伙伴；
- 3–5 个开发 seed；
- 报 XP、history-shuffle drop、protocol-swap causal effect 和 total compute。

这四个模型共同构成一个完整 end-to-end 方案，不需要反复设置“失败即停机”的制度性门槛。每个组件是否值得保留，由它对最终 XP 的配对增量决定。

ICRL4AHT 的最新结果表明，AD、DPT 等通用 history-conditioned 方法在 OvercookedV2 的未见伙伴和未见布局上经常不能产生清晰的 in-context adaptation，甚至低于随机基线。这个结果说明问题本身有研究价值，但也意味着仅仅增加历史编码器远远不够，必须证明历史信息被转化为正确的价值排序。([arXiv](https://arxiv.org/abs/2605.24423))

------

# 十四、文档应如何重构

| 当前文档                                            | 必须修订的内容                                               |
| --------------------------------------------------- | ------------------------------------------------------------ |
| `RESEARCH_PROGRAM.md`                               | 将唯一中心论点改成“legal-history protocol inference improves XP”；Θ2 降为支持性理论 |
| `RESEARCH_THESIS.md`                                | 删除 measurement paper 与 SOTA method paper 双中心结构       |
| `DELTA_ZSC_COMPLETE_THEORY_AND_DESIGN.md`           | 将 partner belief 改为 task/protocol/capability 三对象；删除无校准 Gaussian regret |
| `FOUNDATIONAL_THEORY_AND_PROOFS.md`                 | 修复 T3；把 T2 明确降为二元常数 gap 特例                     |
| `THEORY_PREDICTIONS.md`                             | 删除生态 (\kappa) 直接代入；改用 legal-history value-weighted distinguishability |
| `TRAJECTORY_AND_ESTIMATION_SPEC.md`                 | 禁止完整伙伴动作作为主估计口径；必须记录并使用 ego 合法历史  |
| `LITERATURE_MATRIX.md`                              | 补入 CooT、TALENTS、GOAT、ROTATE、UPD、RecBayes、ICRL4AHT、CE-CM、ToM-guided adaptation，并重新判断 novelty |
| `SOTA_BASELINE.md`                                  | 要求同 commit 原始 baseline run nodes；已发表标量仅作外部参考 |
| `EXPERIMENT_LADDER.md` / `TRACKS_AND_GOVERNANCE.md` | 合并为一个开发矩阵加一个正式矩阵，消除相互矛盾的晋升条件     |
| `EVIDENCE_LEDGER.md`                                | 将“人类裁决”移到独立 decision log，不作为科学证据            |
| 分支与方法身份                                      | 当前分支名是 `delta-zsc-v5`，绑定方法却是 V6，应统一命名     |

更合理的文档体系只需要四份权威文件：

1. `SCIENTIFIC_SPEC.md`：问题、合法信息、中心主张、estimand、可证伪条件；
2. `METHOD_SPEC.md`：模型、数据流、损失和 partner curriculum；
3. `EVALUATION_SPEC.md`：伙伴划分、baseline、统计和资源；
4. `THEORY.md`：一般结果、特例及适用边界。

其他内容作为 appendices 和 evidence ledger，不再让十余份文档共同定义方法。

------

# 最终判决

当前 DELTA-ZSC V6 最值得保留的不是现有复杂机制，而是以下研究资产：

- Official 合法部署接口；
- run-disjoint lineage 和 identity contamination 审计；
- raw return 与 shaped return 分离；
- 固定最终 checkpoint；
- 不替换失败 seed；
- common-random-number all-action continuation；
- 完整资源账本。

这些是很好的科研基础。

但以下内容需要从根本上重构：

- partner belief 的科学对象；
- task/belief 信息隔离；
- Gaussian posterior 语义；
- particle-Q regret；
- matched-code anchor；
- generator code support；
- 生态 (\kappa/TV/\Delta) 外推；
- SOTA 显著性比较；
- 当前创新性陈述。

**最合理的研究主线不是继续证明“V6 的所有模块都能运行”，而是把项目收缩并重构为：用合法历史预测动态协议所诱导的动作价值排序，并证明这一预测在算法族留出的陌生伙伴上因果性地提高 XP。**

这是当前项目最可能同时满足可行性、创新性和顶会说服力的方向。