# EMERGENCE_MECHANISMS_SURVEY — 跨领域"如何刻意制造涌现",服务 Path C

**日期:** 2026-07-09 · **来源:** `/research-lit`(5 个领域并行调研 agent,论文均经网页核验 ✅/⚠️)。
**服务对象:** Path C —— 让"价值充分的交互因子结构"不靠人工设计、而在**单一时序差分(TD, temporal-difference)
价值学习 + agent 自身价值驱动的干预(intervention)**下**自己涌现(emerge)且可辨识(identifiable)**。
**组织方式:** 按"涌现机制",不按领域。

> 术语:TD=时序差分价值学习;ICA=独立成分分析;CRL=因果表征学习;do()=因果干预;可辨识=从数据能唯一
> 还原真因子(至多置换/缩放);序参量(order parameter)=能概括系统宏观状态的少数低维量;BC=行为刻画
> (behavior characterization,演化里用来给行为分格的坐标)。

---

## 1. 一条贯穿五个领域的教训(= 你上一轮的批评,被跨领域证实)

所有领域对"如何制造涌现"的回答里,反复冒出**同一个陷阱**:**你用来度量 / 选择 / 驱动涌现的那个"坐标",
就是你把答案偷偷塞进去的地方。**

| 领域 | 坐标偷答案的方式 |
|---|---|
| 深度学习 | 选了不连续的**度量** → 假的"陡峭涌现"(Schaeffer "mirage");无归纳偏置 → 因子是旋转歧义的表征假象(Locatello 不可能性定理) |
| 开放式 / 质量多样性 | 你手选的**行为刻画(BC)网格轴**就是偷偷写入的因子/目标——novelty search 只把设计从适应度挪到 BC |
| 因果表征学习 | content/style 的划分、"增强只动 style"、"动作只影响一个已知因子"——全是变相手设计 |
| 复杂系统 | 把结构写进**边界/初值** |
| 形态发生 / 元胞自动机 | 把目标结构直接编码进**损失/规则** |

**结论(把你的直觉升级成可操作判据):涌现要合法,它的"结构来源"必须是世界的性质(world property),
不能是我们挑的坐标。** 这条判据下面反复用。

---

## 2. 六条"制造涌现"的机制(跨领域;每条答三问)

**M1 — 压力 + 冗余自由度 + 一个压缩/闭合力。**
① 压力=价值/选择/自由能/重建;自由度=过量容量;外加把解推向结构的力(权重衰减压缩〔grokking〕、机制稀疏
〔CRL〕、自催化闭合〔autocatalytic closure〕)。② 真伪:结构须是"未被要求"的副产物(对未见扰动仍成立),
非损失里直接写死的。③ Path C:单 TD 价值压力 + 过量容量 + **压缩/稀疏力**,让因子解成为最低损失吸引子而非记忆。

**M2 — 辨识来源必须是世界性质;"干预"是最干净的合法来源(核心)。**
① 合法来源(可检验的世界性质):**自身动作作为 do() 干预**、伙伴**时序非平稳**、**机制稀疏**、闭合/自维持。
非法来源(变相手设计):手选 BC、content/style 先验、"动作→一个已知因子"、外部"有趣度"teacher。
② 真伪:做**稀疏度/干预消融**——仅在真实世界结构处才辨识,随机图/伪坐标下崩。③ Path C:**把 agent 自身
动作当作对伙伴的真实 do 干预,直接消费 TD 里天然的 (s,a,s') 前后转移对 + 叠加机制稀疏**——Agent-5(因果表征
簇)明确**证实了这条假设**。红线:**绝不把"动作只影响一个已知伙伴因子"写进假设/辅助损失**(那就滑入非法)。
关键限制:单靠 TD/bisimulation 只到"价值等价类",**必须由动作干预把等价类拆成逐因子可辨识**。

**M3 — 涌现 = 相变 + 可检测阈值,不是单调曲线(给客观真伪判据)。**
① 控制参数(耦合/信噪比/干预强度)越过临界点,序参量从 ≈0 突变为正(Kuramoto/Turing/Haken)。② 随机块模型
(SBM)的 Kesten-Stigum 阈值:**阈下"结构在、但任何算法都测不出"**——正好把"可辨识"量化成一个硬门槛;
配合"有限尺度标度(finite-size scaling)"与"扰动初值仍收敛"两条证伪。③ Path C:把"因子涌现且可辨识"定义为
一个序参量在临界阈上的**相变**,而非单调改进(这也把上一轮被降级的 detectability-threshold 想法,从"过度声张
的主张"救回成"客观真伪测试")。

**M4 — 自修复 / 扰动下的吸引子 = 涌现证书(且是方法内生,不是外挂仪器)。**
① 形态发生 NCA 用"持久+损伤池"训练,使目标结构成为**自修复吸引子**;Levin:损伤后能重建=目标是可与硬件分离
的分布式吸引子。② 真伪最干净的一条:**消融/扰动结构后,它能否自行长回同一结构(attractor),还是只是一次性
读出(lookup)**;对训练中未见的损伤仍能再生=真。③ Path C:**消融/打乱涌现出的因子表征 → 价值学习能否再生
出同一个可辨识分解**。注意:这与你讨厌的"外挂 certification"不同——它是**方法自身动力学的性质**(会不会自愈),
不是事后对着基底跑的一把尺子。

**M5 — 序参量 / 役使(slaving):涌现结构是低维把手,这正是它可辨识的原因。**
① Haken 役使原理:失稳点附近少数慢变序参量"役使"海量微观自由度,系统自发降维。② 序参量是低维可观测量=天然
可辨识句柄。③ Path C:**让待涌现的因子=序参量**——辨识 = 读这一个低维量,而非整张网络;这给"可辨识"一个
优雅的操作定义。

**M6 — 发散压力来自协同演化的对手(最小判据),不是手设目标。**
① 最小判据协同演化(minimal criterion coevolution):两方各自只需"解出/被解出"即可繁殖,发散压力来自**对方
生成的"刚好可解"的问题流**,无 BC、无新颖度度量(最干净的"不手设坐标")。② 真伪:生成器语法的表达力上界
划定了可能因子空间;Enhanced-POET 用迁移+ANNECS 证"没在回收旧结构"。③ Path C:**发散压力来自伙伴交互本身**
——一个因子只有当它"**被干预生成 ∧ 被 TD 价值所需 ∧ 能自维持地喂养自身价值**"时才被承认(minimal criterion),
而不是因为我们列了它;OMNI 式外部"有趣度"是必须避开的 teacher 陷阱(违反"单 TD + 自身价值")。

---

## 3. Path C 配方(五领域收敛出的那一条)

1. **压力:** 单 TD 价值最大化(既定)+ 压缩/稀疏力(M1)。
2. **合法辨识来源(命门,已证实):** agent 自身价值驱动的动作 = 对伙伴的真实 do() 干预,消费 TD 天然的
   (s,a,s') 前后对(M2)+ 机制稀疏;第二合法来源=伙伴时序非平稳。**红线:不得假设"动作→一个已知因子"。**
3. **发散发现(谁找出因子):** 与伙伴的最小判据耦合(M6)——因子须"被干预生成 ∧ 被价值所需 ∧ 自维持",非手列。
4. **因子 = 序参量(M5):** 低维、役使策略 → 辨识=读序参量。
5. **真伪测试(三条,全部方法内生,非外挂证书):**
   - **相变/阈值(M3):** 因子作为序参量在可检测阈上相变式出现,配有限尺度标度——不是单调曲线;
   - **自修复(M4):** 消融因子表征后价值学习**再生**同一分解(吸引子),非重读;
   - **干预因果(M2/DL):** 对某因子做干预,按因子特异地改变价值/行为(Othello-GPT + CRL 范式)。

**一句话:** 别设因子、别设坐标——让**agent 自身干预**提供合法辨识信息,让**与伙伴的最小判据耦合**提供发现压力,
把**因子设成在可检测阈上相变涌现的序参量**,再用**自修复 + 干预因果**这两条方法内生性质当真伪判据。**辨识来源
全部落在"世界性质"一侧,一个都不落在"我们挑的坐标"一侧。**

---

## 4. 合法 vs 非法 辨识来源(随手对照)

| 合法(世界性质,可检验) | 非法(变相手设计,须避开) |
|---|---|
| 自身动作作为 do() 干预(TD 原生 (s,a,s')) | "动作只影响一个已知伙伴因子"写进假设/损失 |
| 伙伴时序非平稳(真回合/模式切换) | 手选行为刻画 BC / content-style 划分 |
| 机制稀疏(世界机制本就稀疏) | 稀疏图人为给定 / 罚过强强拆正交 |
| 闭合 / 自维持(自催化式) | 外部"有趣度"teacher(OMNI 式) |
| 相变阈值(SBM/KS)作客观门槛 | 选不连续度量制造"陡峭涌现"(mirage) |

---

## 5. 已核验文献(按领域;✅ 已抓页核验 / ⚠️ 部分未取到号 / established=经典定律)

**深度学习涌现:** Grokking(Power 2201.02177 ✅);Progress measures(Nanda ICLR23 2301.05217 ✅);Omnigrok
(2210.01117 ✅);Emergent Abilities(Wei TMLR 2206.07682 ✅);**Mirage**(Schaeffer NeurIPS23 2304.15004 ✅);
Quantization Model of Scaling(Michaud 2303.13506 ✅);Induction Heads(Olsson 2209.11895 ✅);Data-Distributional
ICL(Chan NeurIPS22 2205.05055 ✅);Transient ICL(Singh 2311.08360 ✅);Emergent World Reps/Othello-GPT
(Li ICLR23 2210.13382 ✅);Locatello 不可能性(ICML19 1811.12359 ✅)。

**开放式/QD/演化:** Novelty Search(Lehman-Stanley 2011 ✅);MAP-Elites(1504.04909 ✅);QD 综述(Pugh 2016 ✅);
POET(1901.01753 ✅);Enhanced-POET(2003.08536 ✅);AI-GAs(1905.10985 ✅);OMNI(2306.01711 ✅);**Minimal
Criterion Coevolution**(Brant-Stanley GECCO17 ✅);OEE 必要条件(Soros-Stanley ALIFE14 ✅);RAF 自催化集
(2303.01809 ✅);Open-Endedness for ASI(Hughes 2406.04268 ✅)。

**复杂系统/统计物理:** Turing RD(1952 established);Kuramoto(1975 established);Haken 役使(1977 established);
Anderson "More is Different"(1972 established);Prigogine 耗散结构(1977 established);BTW 自组织临界(1987
established);order-from-noise(von Foerster/Atlan established);**SBM 可检测相变**(Decelle 1109.3041 ✅);
KS 阈值证明(Mossel-Neeman-Sly 1311.4115 ✅)。

**形态发生/CA:** Turing(1952 ✅);Wolfram 分类(1984 ✅);Lenia(1812.05433 ✅);**Growing NCA**(Mordvintsev
Distill 2020 ✅);Self-classifying MNIST(Randazzo Distill 2020 ✅);Self-Organising Textures(Distill 2021 ⚠️);
3D NCA(2103.08737 ✅);NCA cart-pole 控制(2106.15240 ✅);Graph CA(2110.14237 ✅);Sensorimotor Agency via
Diversity(Hamon 2402.10236 ✅);Levin 生物电认知胶(2023 ✅)。

**因果表征/可辨识 + 涌现秩序:** TCL Nonlinear ICA(1605.06336 ✅);Aux-variable ICA(1805.08651 ✅);iVAE
(1907.04809 ✅);Locatello(1811.12359 ✅);SSL content/style(2106.04619 ✅);Mechanism Sparsity(2107.10098 ✅);
**Sparse Actions as Interventions**(2401.04890 ✅);CITRIS(2202.03169 ✅);Weakly-Supervised CRL(2203.16437 ✅);
**Interventional CRL**(Ahuja ICML23 2209.11924 ✅);Deep Bisimulation(2006.10742 ✅);Approximate Information
State(2010.08843 ✅);Young 约定演化(Econometrica 1993 ✅);Hart-Mas-Colell 相关均衡(Econometrica 2000 ✅)。

> 未核验号须先过 `/citation-audit` 再进论文;⚠️ 项(Self-Organising Textures 数字 DOI)需补验。

---

## 6. Path C 下一步:最小形式主张的目标

把上面收敛成一条可证/可证伪的命题:

> **在[agent 自身干预有足够覆盖 + 机制稀疏 + 伙伴时序非平稳]下,单 TD 的价值充分表征可证明收敛到伙伴倾向的
> 可辨识因子分解;且这些因子作为序参量,在一个可检测阈值(Kesten-Stigum 式)处相变式涌现。**

三个真伪测试(相变/自修复/干预因果)**都是方法自身动力学的性质**,不是你讨厌的外挂 certification。逐条把假设
标成"世界性质(合法)/变相手设计(非法)",非法的一律不许进。这条命题 = 下一步纸上理论要证到的靶。
