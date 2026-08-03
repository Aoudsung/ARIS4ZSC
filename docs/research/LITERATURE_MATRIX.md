# 文献矩阵（探索轨）

> **修订记录（2026-08-03）**。修订原因：外部评审（`Review&Suggestion.md` §9《当前创新性边界被明显高估》）判定本矩阵创新性边界被高估、多个 2025–26 直接相关工作缺失、个别引用无法验证。本次修订：逐条联网核实并补录 CooT、GOAT/ROTATE/UPD、PACE、RecBayes/CE-CM、NeurIPS 2025 涌现伙伴表征、ICRL4AHT；纠正 ReCollab 定位与 TALENTS 编号及出处；新增"五、创新性重评估"小节。评审原文指针：`Review&Suggestion.md` §9。所有补录条目已于 2026-08-03 通过 arXiv/PMLR/NeurIPS 官方页面联网核对编号与标题，无法核实的标注 `unverified`，不作为定位依据。另经全仓库检索，ARNI/DyNet 在本文档及项目中均无条目，无需删除或标注。

状态：探索轨产物，`scientific_readout_allowed: false`。
职责：登记 ZSC 相关工作，回答"谁占了哪块地，哪里还空着"。
上级入口：[RESEARCH_PROGRAM](../RESEARCH_PROGRAM.md)。

## 检索记录

| 日期 | 检索式 | 用途 |
|---|---|---|
| 2026-08-03 | `GOAT zero-shot coordination Overcooked ad hoc teamwork 2025` | 核实 2025 竞争者 |
| 2026-08-03 | `zero-shot coordination Overcooked 2025 new method CEC SBC N-XPlay` | 核实 2025 竞争者 |
| 2026-08-03 | `SBC zero-shot coordination best-response compatibility Overcooked 2025` | 核实 SBC |
| 2026-08-03 | arXiv 摘要页逐一核对：2506.23549、2504.15457、2506.15756、2605.24423、2507.05244、2512.22129、2505.17323、2505.23686 | 补录条目编号与标题核实 |
| 2026-08-03 | PMLR v235 ma24n 页面；nips.cc/virtual/2025/poster/117556 | 核实 PACE 与 NeurIPS 2025 涌现伙伴表征 |
| 2026-08-03 | `UPD` 与 `CE-CM` 独立文献检索（ad hoc teamwork / capability estimation） | 二者均未检索到独立可验证出处，标 unverified |

核实结论：GOAT（arXiv:2504.15457）、CEC（arXiv:2504.12714）、N-XPlay（arXiv:2506.17560）存在且定位明确。SBC 未检索到对应 ZSC 方法，剔除，防止引用幽灵文献。另在检索中补入 GAMMA（Liang et al. 2024）。

2026-08-03 补充核实：CooT（arXiv:2506.23549，ICML 2026）、GOAT（arXiv:2504.15457）、ROTATE（arXiv:2505.23686）、PACE（PMLR v235 ma24n，ICML 2024）、RecBayes（arXiv:2506.15756）、TALENTS（arXiv:2507.05244）、ReCollab（arXiv:2512.22129）、ICRL4AHT（arXiv:2605.24423）、Partner Modelling Emerges in Recurrent Agents（arXiv:2505.17323，NeurIPS 2025 poster 117556）均核实存在。**UPD 与 CE-CM 未检索到独立可验证出处，标注 unverified**；ICRL4AHT 的"ICML 2026 录用"未在 arXiv 页面证实（论文仅使用 ICML 模板），录用状态标 unverified。

种子文献沿用 [PATH_C_PROPOSAL §6](../legacy/v44/design/PATH_C_PROPOSAL.md)，逐篇扩写如下。每张卡片六个字段：主张什么，训练多样性怎么做，伙伴推断怎么做，评估协议，与本项目主张的重叠区，还没人占的空间。

## 一、训练多样性路线

### FCP，Fictitious Co-Play（Strouse et al., NeurIPS 2022）

- 主张：用种群训练和"虚拟自我"作伙伴，能同时提升与真人和陌生智能体的协调。
- 训练多样性：PBT 种群，每个体与自身冻结副本配对，种系谱系保留历史版本。
- 伙伴推断：无，部署时不做在线推断。
- 评估：Overcooked 人机实验加智能体交叉配对。
- 重叠区：本项目 V6 的 generator 初始化也用 SP/OP/SA/FCP 四源，FCP 是 S5 的参照基线。
- 空位：不解释"为什么泛化"，只报分数。

### MEP，Maximum Entropy Population（Zhao et al., NeurIPS 2022）

- 主张：把种群联合策略的熵最大化，能诱导多样惯例，提升跨惯例泛化。
- 训练多样性：熵正则的种群训练。
- 伙伴推断：无。
- 评估：Overcooked 交叉配对与真人评估。
- 重叠区：与本项目"多样性加推断"的分工互补，MEP 只管前半截。
- 空位：同样没有伙伴侧机制，熵最大化的代价（同伴内性能）很少被量化。

### OP，Other-Play（Hu et al., ICML 2020）

- 主张：与环境的随机对称变换同步训练，排除任意惯例，保留对称不变解。
- 训练多样性：对称性随机化，不依赖种群。
- 伙伴推断：无。
- 评估：Overcooked 交叉配对。
- 重叠区：OP 是本项目上游伙伴与初始化来源之一。
- 空位：对称性假设在复杂布局上不成立时没有兜底。

### GAMMA（Liang et al., 2024）

- 主张：用生成式伙伴模型（人类数据加模拟数据）扩展伙伴空间，再做适应。
- 训练多样性：生成模型采样出多样伙伴。
- 伙伴推断：依赖生成的伙伴假设，具体推断机制弱。
- 评估：Overcooked 真人评估，曾报真人场景 SOTA。
- 重叠区：生成式伙伴与本项目 V6 的连续 generator 思路同源。
- 空位：生成伙伴的真实性约束靠 KL 正则，没有任务回报层面的准入检验。

### GOAT（arXiv:2504.15457，已核实）

- 主张：对抗训练，对手在生成式伙伴空间内搜索能暴露合作者弱点的伙伴。论文标题：Improving Human-AI Coordination through Online Adversarial Training and Generative Models（Chaudhary, Liang, Chen, Du, Jaques）。
- 训练多样性：生成模型保证对手真实，对抗过程保证对手难。关键细节：生成模型（VAE decoder）参数冻结，对抗方只在 latent 空间优化 embedding，以避免对抗伙伴退化为破坏者。
- 伙伴推断：无部署时推断，靠训练时被对抗打磨。
- 评估：CMG 加 Overcooked 布局，真人实验，对比 BC+RL、FCP、MEP、CoMeDi、MEP+GAMMA。
- 重叠区：把"伙伴空间难覆盖"当作训练问题，与本项目把它当推断问题形成正面对照。占据"在线生成暴露 ego 弱点的伙伴"的主张。
- 空位：部署时仍然不做在线伙伴推断，遇到训练分布外的伙伴类型没有应对机制。

### ROTATE（arXiv:2505.23686，已核实）

- 主张：Regret-driven Open-ended Training for Ad Hoc Teamwork（Wang, Rahman, Cui, Sung, Stone），以 regret 驱动的开放式训练持续生成新伙伴。
- 训练多样性：开放式 regret 驱动课程。
- 重叠区：与 GOAT 同属对抗/开放式伙伴生成路线，共同占据"动态伙伴生成"位置。
- 空位：训练时方案，不做部署时伙伴推断。

### UPD（unverified）

- 主张：评审 §9 将其与 GOAT、ROTATE 并列为动态伙伴生成路线（learnability curriculum），评审中的引用链接指向 GOAT 论文（arXiv:2504.15457）。
- 核实状态：2026-08-03 两次独立联网检索未找到名为 UPD 的可验证独立文献，标 unverified，不作为定位依据，待获得正式出处后再补。

### CEC，Cross-Environment Cooperation（arXiv:2504.12714）

- 主张：跟同一个伙伴跨多个环境学泛化，比跟多个伙伴在单一环境学更能泛化到新伙伴。
- 训练多样性：程序化生成 Overcooked 布局，用任务多样性代替伙伴多样性。
- 伙伴推断：无。
- 评估：程序化布局加真人评估。
- 重叠区：与本项目共享"泛化到底从哪来"的问题意识，CEC 押注任务侧，本项目押注伙伴推断侧。
- 空位：没回答伙伴惯例本身的结构（类型、可分性、可恢复性）问题。

### N-XPlay（arXiv:2506.17560）

- 主张：把 Other-Play 扩展到 N 人、多团队场景，平衡组内与组间协调。
- 训练多样性：OP 式对称性随机化加团队划分。
- 伙伴推断：无。
- 评估：N-player Overcooked，二、三、五人场景对比 SP。
- 重叠区：评估构造的严谨性可借鉴。
- 空位：仍是训练时方案，团队规模增大后推断需求反而更强，但没做。

### ConventionPlay（OpenReview, X1EnygvhoZ）

- 主张：惯例是策略空间中组内兼容、组间不兼容的簇，可用能力受限种群诱导 repertoire probing 与 convention steering。
- 训练多样性：能力受限种群诱导惯例分化。
- 伙伴推断：steering 含隐式的惯例识别。
- 评估：惯例簇结构分析。
- 重叠区：与本项目"惯例是否可重复、可分离"（Θ1）直接相关。
- 空位：先诱导再识别，本项目测的是被动合法历史里已有的可恢复性。

### PACE（Ma et al., ICML 2024；PMLR v235 ma24n，已核实）

- 主张：训练目标直接优化"伙伴识别"导向的探索（peer identification reward，context-aware probing）。论文标题：Fast Peer Adaptation with Context-aware Exploration。
- 训练多样性：识别导向的探索奖励。
- 伙伴推断：显式伙伴模型，训练时就为识别服务。
- 评估：ZSC 基准。
- 重叠区：主动探查方向，本项目把它放在被动历史不足之后的下一阶段。
- 空位：探查的任务代价（等待、接管）没进回报账。

## 二、伙伴推断与策略选择路线

### PLASTIC / PLASTIC-Policy（Mirsky et al., JAIR 2023）

- 主张：维护对伙伴类型的信念分布，在线更新后按信念选动作。
- 训练多样性：与多种类型伙伴的数据训练。
- 伙伴推断：贝叶斯信念更新，类型集预先定义。
- 评估：多域 ad hoc teamwork。
- 重叠区：信念加信念条件动作的骨架与本项目 V6 相同。
- 空位：类型集来自训练分布，测试伙伴超出类型集时信念无定义；价值排序质量不被检验。

### PECAN（arXiv:2301.06387）

- 主张：用预测与反事实推理在线检验伙伴假设，再选协作行为。
- 伙伴推断：反事实检验伙伴假设。
- 重叠区：反事实思想与本项目 V6 的反事实延续监督同源，但 PECAN 用在推断，本项目用在价值标签。
- 空位：不测量"推断出来之后值多少"，缺价值回收的量化。

### Strategy Matching（arXiv:2210.15099）

- 主张：把观测到的伙伴行为与预学的策略表示匹配，快速定位伙伴类型。
- 伙伴推断：行为到策略库的匹配。
- 重叠区：与本项目"模式库加路由"结构相似。
- 空位：匹配正确率与任务回报之间没有桥。

### GPAT（arXiv:2510.16187）

- 主张：通过策略库组合适应伙伴。
- 伙伴推断：组合权重随交互调整。
- 重叠区：策略库路线的近亲。
- 空位：组合空间随库增长，选择何时收敛没有理论上界。

### TALENTS（arXiv:2507.05244，已核实；原标"NeurIPS 2025"不成立）

- 主张：利用历史与伙伴模型选择协作行为。论文标题：Modeling Latent Partner Strategies for Adaptive Zero-Shot Human-Agent Collaboration（Li et al.）。核心机制：VAE 学习 latent strategy 空间、聚类划分策略类型、fixed-share regret minimization 在线推断并动态调整伙伴策略估计。
- 出处更正：原条目误标 NeurIPS 2025；arXiv 页面显示为 RSS 2025 GenAI-HRI Workshop 最佳论文，无 NeurIPS 2025 录用记录。
- 重叠区：与本项目 H2（历史可恢复价值）同题，且已占据"显式学习 latent 伙伴策略并在线调整"的主张。
- 空位：不区分历史信息里"任务状态"与"伙伴惯例"各贡献多少。

### RecBayes / CE-CM（arXiv:2506.15756；RecBayes 已核实，CE-CM unverified）

- 主张：RecBayes（Ribeiro, Oren, Sardinha, Spaan, Melo，RecBayes: Recurrent Bayesian Ad Hoc Teamwork in Large Partially Observable Domains）用 recurrent Bayesian classifier 仅凭自身观测识别已知队伍与任务，全程不需要环境状态或伙伴动作，可扩展到百万状态规模。占据"部分可观测下 Bayesian teammate identification"的位置。
- CE-CM：评审 §9 描述其为 2026 年 7 月把 approximate Bayesian inference 用于 task-invariant hidden capability estimation 的工作，评审中的引用链接指向 RecBayes 条目；2026-08-03 独立检索未找到可验证出处，标 unverified。
- 重叠区：与本项目信念更新骨架同族，但类型集/队伍集预先定义。
- 空位：测试对象超出预定义类型集时信念无定义；不测量识别的价值回收。

### ReCollab（arXiv:2512.22129，已核实；2026-08-03 纠正定位）

- 主张（纠正后）：中心是 retrieval-augmented LLM teammate behavioral modeling——用 LLM 把短行为轨迹映射为高层伙伴假设、作为伙伴行为世界模型，并用 RAG 检索示例轨迹稳定推断（Overcooked 上分类准确率与回合回报的 Pareto 权衡）。论文标题：ReCollab: Retrieval-Augmented LLMs for Cooperative Ad-hoc Teammate Modeling（Wallace, Siddique, Cao）。原条目将其概括为"普通 early-window router"不准确，予以纠正。
- 伙伴推断：LLM 行为 rubric 分类加 retrieval grounding，非窗口内类型识别加一次路由。
- 重叠区：与本项目"历史驱动的伙伴适应"同域，但走 LLM/RAG 路线；本项目的控制协议（同前缀配对延续、time-indexed 的 V_fix 到 V_full 全链）与其互补而非重合。
- 空位：不测量历史信息的价值回收，分类准确率与回报的权衡是事后报告而非监督信号。

### CooT（arXiv:2506.23549，已核实）

- 主张：用 in-context learning 做实时伙伴适应——以 recent interaction history 为上下文直接生成协调动作（Coordination Transformers），并与 population、fine-tuning、Meta-RL 三类路线正面对比（Overcooked 与 Google Research Football，ICML 2026）。论文标题：CooT: Learning to Coordinate In-Context with Coordination Transformers（Wang et al.）。已占据"使用历史适应陌生伙伴"的主张。
- 伙伴推断：隐式，靠上下文编码。
- 重叠区：端到端生成互补动作，与本项目信念条件 actor 同属端到端路线。
- 空位：生成动作的互补性没有价值层面的可证伪检验。

### Partner Modelling Emerges in Recurrent Agents (But Only When It Matters)（arXiv:2505.17323；NeurIPS 2025 poster 117556，已核实）

- 主张：简单 model-free recurrent agent 在无辅助目标、无专门 belief 架构的条件下可涌现结构化的伙伴能力表征；涌现条件是社会压力——agent 能通过控制任务分工影响伙伴行为（Mon-Williams et al.）。
- 对本项目的含义：该结果使 **full-history recurrent PPO 成为必须正面对比的强制核心基线**（统一开发矩阵 B0），而不是普通消融；"伙伴表征需要专门架构"不再是默认假设。
- 重叠区：与本项目 Θ4 直接对话，本项目用 decision-regret 信号给出"改善任务表现"的可操作版本。
- 空位：该工作考察表征涌现与单局适应，不测跨训练运行的惯例可重复性，也不量化历史信息的价值回收。

## 三、评估与机会测量路线

### ZSC-Eval（Wang et al., NeurIPS 2024 D&B, arXiv:2310.05208）

- 主张：ZSC 需要统一评估工具：BR-Div 筛伙伴，BR-Prox 量 ego 相对近似 best response 的表现。
- 评估：Overcooked 与 Google Research Football 双基准。
- 重叠区：本项目共同伙伴记分板的伙伴筛选思想与之同向。
- 空位：BR-Prox 量的是 ego 能力，不量"伙伴差异里有多少可恢复价值"。

### BRDiv（OpenReview, l5BzfQhROl）

- 主张：以 best-response compatibility 生成训练伙伴。
- 重叠区：本项目 generator 多样性机制的近亲。
- 空位：生成侧工作，不涉及部署时推断。

## 四、基准与环境

### OvercookedV2（Gessler et al., arXiv:2503.17821, ICLR 2025）

- 主张：重做 Overcooked 的 ZSC 评估协议：view size 2、负奖励、随机初始位置、食谱重采样、400 步、10×10 矩阵。
- 与本项目关系：这是 S5 的固定评估协议，`overcooked_v2_iclr2025_5ce1707_v1`，一字不改。

### ICRL4AHT（arXiv:2605.24423，已核实；"ICML 2026 录用"标 unverified）

- 主张：Benchmarking the Limits of In-Context Reinforcement Learning for Ad-Hoc Teamwork（Jing et al.）。基于 Overcooked-V2 的 JAX 高通量实现构建大规模 AHT 基准：RL 与 heuristic 伙伴套件、可控 train-test shift、端到端可复现管线。
- 关键结果：AD（Algorithm Distillation）与 DPT（Decision-Pretrained Transformer）等历史条件方法在未见伙伴与未见布局轨道上全败，经常低于随机基线且无 in-context 提升；正文诊断为 **fail to perform the necessary Bayesian inference**——无法从交互历史识别伙伴策略。
- 对本项目的含义：仅加历史编码器远远不够，必须证明历史信息被转化为正确的价值排序；问题本身的研究价值由该基准背书。
- 核实状态：arXiv 编号与标题已核实（2026-05-23 提交，正文诊断已核对）；"ICML 2026"仅见论文使用 ICML 模板，正式录用未证实，标 unverified。

## 生态位图谱

回答三个问题。

**信念加信念条件控制这条路线被谁占了？** PLASTIC 占了"类型信念加信念选动作"的骨架，PECAN 占了反事实检验假设，CooT 占了端到端生成互补动作与历史 in-context 适应，TALENTS 占了 latent 策略在线调整，NeurIPS 2025 涌现伙伴表征工作占了"普通 RNN 即可涌现表征"。但没有人同时做三件事：信念由真实分支回报监督（不是自蒸馏），信念质量用价值排序相关检验（不是分类准确率），部署价值用同前缀配对延续测量（不是事后匹配）——这三个组合点是否仍成立，以"五、创新性重评估"为准。

**评估与机会测量这条路线被谁占了？** ZSC-Eval 占了伙伴筛选与 ego 能力测量；ReCollab 的定位已纠正为 retrieval-augmented LLM 伙伴行为建模（不再占据"短窗口类型识别"位置）。但没有人把 V_fix 到 V_full 的完整信息价值分解链操作化为 ZSC 诊断，也没有人报告回收率（实测收益除以可认证机会上限）。

**理论定量预测加判决性测量这个位置还空不空？** 空。现有工作要么纯经验（FCP、MEP、GOAT、CEC），要么理论停在受控特例（本项目的 TV 界与历史需求界目前也只到这一步）。把 `Θ(log(Δ/ε)/κ²)` 与 TV 界从受控任务提升为对 OvercookedV2 的定量预测，并预写失配条款，这个位置还没有人站。这是候选论点 Θ2 的生态位依据。

## 五、创新性重评估（2026-08-03，依据评审 §9）

评审 §9 判定原创新性边界被高估。经上述补录与联网核实，以下主张已被现有工作占据，**不得再作为本项目创新点陈述**：

| 已被占据的主张 | 占据者 |
|---|---|
| 使用历史适应陌生伙伴（recent interaction history，in-context coordination） | CooT（arXiv:2506.23549） |
| 显式学习 latent 伙伴策略并在线调整 | TALENTS（arXiv:2507.05244） |
| 简单 recurrent agent 涌现伙伴表征（社会压力与任务分工条件） | NeurIPS 2025 poster 117556（arXiv:2505.17323） |
| 主动识别伙伴与 context-aware probing | PACE（PMLR v235 ma24n） |
| 在线生成暴露 ego 弱点的对抗伙伴 | GOAT（arXiv:2504.15457）、ROTATE（arXiv:2505.23686）；UPD 标 unverified |
| 部分可观测下 Bayesian teammate identification | RecBayes（arXiv:2506.15756）；CE-CM 标 unverified |

NeurIPS 2025 涌现伙伴表征的直接后果：**full-history recurrent PPO 升级为必须正面对比的强制核心基线**（统一开发矩阵 B0），而非普通消融。

项目保留的三个潜在创新点（重构主线的方法中心）：

1. **value-weighted protocol representation**：latent 语义不是伙伴 ID，而是 centered continuation 价值签名（历史中会改变 ego 最优动作排序的部分），用共同随机数 all-action continuation 直接监督。
2. **legal-history causal supervision**：全部监督来自真实可部署历史与有效 counterfactual continuation，不使用伙伴动作 oracle 或隐藏 code。
3. **recoverable-value evaluation**：以 legal-history / history-shuffled / state-only / oracle continuation 四格设计测量实际回收比例（实测收益除以可认证机会上限）。

ARNI/DyNet 说明：2026-08-03 全仓库检索确认本文档与项目中均无 ARNI/DyNet 条目，不存在需要标注 unverified 或删除的引用。
