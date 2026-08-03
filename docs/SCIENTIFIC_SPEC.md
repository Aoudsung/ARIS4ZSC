# SCIENTIFIC_SPEC：Decision-Equivalent Protocol Inference（DEPI）科学规格

修订头：本文件于统一重构（`Rigor_与_Generality_统一优化_7f793915.md` A/D/I 节；外部评审归档 [`research/REVIEW_AND_SUGGESTION_2026.md`](research/REVIEW_AND_SUGGESTION_2026.md) §1/§7/§12）中创建，是四份权威文件之首：**问题、合法信息、中心主张、estimand、可证伪条件**的唯一权威口径。方法实现细节见 [`METHOD_SPEC.md`](METHOD_SPEC.md)；伙伴划分、baseline、统计与资源见 [`EVALUATION_SPEC.md`](EVALUATION_SPEC.md)；理论见 [`THEORY.md`](THEORY.md)。（修订头补充，2026-08-04，文档重构任务 #16：原"待建"指针改为已建文件链接，其余正文不动。）

用户裁定（已生效，直接约束本规格）：
1. **SOTA 判决走路径 1**——在固定 Official commit 上重训/获取 FCP/OP/SA run-level 节点，同 episode keys 配对/双样本层级推断；已发表 Table 2 标量仅作外部 sanity check（承接 [`research/STATISTICAL_PREREGISTRATION.md`](research/STATISTICAL_PREREGISTRATION.md) 第二节路径 1）。
2. **latent 信念补全 Bayes 语义**——必须定义生成模型、proper likelihood 与 held-out 校准协议（proper scoring rule + 覆盖率），不得退化为纯 embedding 叙事；具体选型见 [`METHOD_SPEC.md`](METHOD_SPEC.md) 第二节。

状态：`authoritative: true`。本规格与既有文档冲突时以本规格为准；旧"Bayesian partner belief"中心叙事废止。

---

## 一、问题定义：Test-Time Protocol Formation 下的零样本协调

### 1.1 任务设定

基准为 OvercookedV2（ICLR 2025，Official commit `5ce1707cf31c1c115e6f6ba96db7bc9cc80a850e`，协议版本 `overcooked_v2_iclr2025_5ce1707_v1`）的 Test-Time Protocol Formation 测项：

- **两个陌生智能体**（ego 与 partner）在布局 `test_time_simple` / `test_time_wide` 中协作完成取-做-送订单；
- **无预先共享协议**：双方训练算法、种子、超参互不相交，测试时才首次相遇；
- **在线形成协议**：协作惯例（谁取料、谁掌勺、谁交付、让位/争抢约定）**不是事先存在的固定类型**，而是由双方行为在交互中共同产生、随时可变的动态对象；
- **主读数 XP**：与陌生伙伴的零样本协调得分（每配对 500 回合、400 步/回合、Official 评估协议）。

### 1.2 问题的科学内核

伙伴策略不是外生不变参数：伙伴的下一步行为响应 ego 之前的动作，ego 的动作既在完成任务，也在塑造共同协议。因此正确的问题不是"识别一个固定的伙伴类型"，而是：

> **在只能读取合法局部历史的条件下，在线推断一个随交互演化、以 ego 动作价值排序为语义的共同协议状态，并据此行动。**

任何把隐藏惯例设为固定外生变量、把历史仅当作识别证据的形式化（含旧 T2/T3 的"识别后路由"框架），研究的都不是本基准要求的协议形成问题（评审 §1）。

---

## 二、合法信息边界（Official 部署政策逐条清单）

ego 在训练监督、部署与评估中**只允许读取**以下信息（即 Official 递归部署接口的可读面）：

| 编号 | 合法信息项 | 载体 |
|---|---|---|
| L1 | ego 局部观测 o_t（view size 2 的 ego-centric 网格张量，含可见范围内的物体/伙伴可见部分） | `observations` |
| L2 | ego 自身前一步动作 a_{t−1}^ego（含 episode 起始哨兵） | `previous_action` + `episode_start` |
| L3 | done / episode_start 标记 | `episode_starts` / `dones` |
| L4 | 自身可递归状态（recurrent carry，仅由 L1–L3 派生） | `PolicyState.task_carry` 等 |
| L5 | 自身历史观测与动作序列（由 L1–L3 按时间累积，合法历史的定义域） | H_t := (o_{1:t}, a^ego_{0:t−1}, done_{1:t}) |

**禁止项（违反即读数无效，逐条列禁）**：

| 编号 | 禁止信息 | 禁令理由 |
|---|---|---|
| F1 | **partner-action oracle**：完整伙伴动作序列、伙伴动作前缀、伙伴动作直方图 | 部署不可见；评审 §7.1——测得的是 I(完整伙伴动作; run/type) 而非 I(合法历史; 协议) |
| F2 | 伙伴隐藏 code / 生成器内部状态 / partner lineage 标签进入策略或后验 | 部署不可见；仅作监督采集端的台账元数据 |
| F3 | joint / 全局观测、伙伴视角观测 | 部署不可见 |
| F4 | 布局全局状态、食谱真值中超出 ego 可见范围的部分 | 部署不可见 |
| F5 | 任何以 F1–F4 为基底的特征（模板匹配、窗口直方图等）进入主估计 | 同 F1 |

"完整行为上限"（full partner action oracle）只允许作为 **unattainable oracle 参照**读数显式标注，不得用于生态外推、历史需求计算或主张（与 [`research/TRAJECTORY_AND_ESTIMATION_SPEC.md`](research/TRAJECTORY_AND_ESTIMATION_SPEC.md) 2026-08-03 修订口径一致）。

---

## 三、中心主张（单句可证伪陈述）

> **legal-history protocol inference improves XP**：在只能读取 Official 合法局部历史（第二节 L1–L5）的条件下，学习一个随交互动态演化、以 ego 动作价值排序为语义的协议状态，可以在训练运行与算法族均留出（run-disjoint 且 algorithm-family-disjoint）的陌生伙伴上，产生可归因于该协议推断机制的 XP 提升。

可证伪性要点：

- "可归因"由三类强制可识别性控制兑现（task-history leakage / history shuffle / context swap，见 [`METHOD_SPEC.md`](METHOD_SPEC.md) 第四节）；控制不通过则提升不可归因，主张判负；
- "以动作价值排序为语义"由 centered all-action continuation advantage 签名直接监督兑现（第五节 estimand 与 [`METHOD_SPEC.md`](METHOD_SPEC.md) 第三节 L_signature）；
- Θ2（信息时序理论）**降为支持性理论**，不再并列为中心论点（消除评审 §11 双中心冲突）；S2 180/180 表述降级为 theorem unit test，"Θ2 已验证"表述收回。

---

## 四、科学对象：三对象形式化与协议状态转移

### 4.1 三对象定义（评审 §1 / §12.1）

给定合法历史 H_t，定义三个互不重叠的科学对象：

- **x_t = f_x(H_t)**：任务状态/任务信念——环境与任务侧的量（可见物体、订单进度、自身位置与持有物）；
- **u = f_u(H_t)**：伙伴稳定能力/倾向——在单个 episode 内近似不变的伙伴侧属性（操作速度倾向、角色偏好等）；
- **c_t = f_c(H_t)**：**动态共同协议状态**——由双方行为共同产生、可随时改变、且会改变 ego 最优动作排序的部分。这是本方法的中心对象。

联合信念对象（评审 §1 要求的形式化）：

\[
b_t = p(x_t,\ u,\ c_t \mid H_t),
\]

其中协议状态满足转移定义（协议不是固定参数，而是受双方动作驱动的动态过程）：

\[
c_{t+1} \sim P\bigl(c_{t+1} \mid c_t,\ H_t,\ a_t^{ego},\ a_t^{partner}\bigr).
\]

共享单一 actor 与单一 critic（不为伙伴类型训练独立网络）：

\[
\pi_\theta(a_t \mid x_t, u, c_t), \qquad Q_\psi(\cdot \mid x_t, u, c_t).
\]

### 4.2 协议状态的科学语义锚定（评审 §12.2）

协议状态 c_t 的科学语义**不是伙伴身份**，而是：

> **历史中会改变 ego 最优动作排序的部分。**

形式化：对真实合法历史定义共同随机数 all-action continuation 的 centered advantage 签名

\[
A_t(a) = G(H_t, a) - \frac{1}{|\mathcal{A}|}\sum_{a'} G(H_t, a'), \qquad |\mathcal{A}| = 6,
\]

其中 G(H_t, a) 为在 H_t 之后强制 ego 首动作 a、双方按目标策略以共同随机数延续（128 步、γ=0.99、replica 均值、必要时以保守 twin-Q 端点 bootstrap）所得的 raw 回报。方法学习

\[
\hat A_\psi(x_t, c_t, \cdot) \approx A_t(\cdot),
\]

废弃任意 8 维 Gaussian 的身份语义。

---

## 五、Estimand 定义

### 5.1 D_V(t)：value-weighted distinguishability（评审 §7.4；废止 E[Δ]·TV/2）

二元协议特例下，t 时刻的合法历史可区分价值量为

\[
D_V(t) = \frac{1}{2}\int \Delta(h)\, \bigl|p_1^{\,t}(h) - p_0^{\,t}(h)\bigr|\, dh,
\]

- h 为截至 t 的**合法 ego 历史**（L1–L5 口径，禁 F1–F5）；
- p_i^t 为协议 i 诱导的合法历史前缀分布；
- Δ(h) 为历史 h 上正确/错误路由的 payoff gap。

**废止** "Δ̂·TV̂/2 = 分别平均再相乘" 的旧口径；TV̂ 按 [`research/TRAJECTORY_AND_ESTIMATION_SPEC.md`](research/TRAJECTORY_AND_ESTIMATION_SPEC.md) 修订为 TV 下界（1−2p̂_e ≤ TV）、κ̂ 按代理量标注，均不得代入 Θ(log(Δ/ε)/κ²) 做生态外推。

### 5.2 Recoverable value 四格（评审 §9 / §12 创新点③）

对同一组测试伙伴与同一组 episode keys，测四格 XP（或 CRN 延续回报），定义**实际回收比例**：

| 格 | 条件 | 含义 |
|---|---|---|
| G1 legal-history | 完整合法历史驱动的协议策略（本方法） | 回收量分子主体 |
| G2 shuffled-history | 保持任务状态与当前观测不变、打乱伙伴侧历史证据 | 无适应下界 |
| G3 state-only | 仅当前观测，无历史 | 纯任务基线下界 |
| G4 oracle-continuation | 给定真实 continuation-value 排序直接按最优动作行动 | 不可达上限 |

回收比例定义：ρ = (G1 − G2) / (G4 − G2)（G4 − G2 ≤ δ_min 的格子记为无信号并剔除分母）。G1−G2 为 history-shuffle drop 读数，G3 与 G2 之差衡量任务侧与历史侧贡献分解。

### 5.3 XP 主读数与配对增量

- XP 与 S5 正式矩阵同布局同口径（每配对 500 回合，Official 评估协议，双角色评估）；
- 组件取舍的统计单位为**配对增量**：同 seed、同 episode keys、同 run-disjoint 测试伙伴下 B_k − B_{k−1}（B0–B3 定义见 [`research/DEVELOPMENT_MATRIX.md`](research/DEVELOPMENT_MATRIX.md) 与本规格第七节映射表）；
- δ_min = 20.0（一次正确交付的原始回报，`OFFICIAL_CORRECT_DELIVERY_REWARD`），实质显著性预注册见 [`research/STATISTICAL_PREREGISTRATION.md`](research/STATISTICAL_PREREGISTRATION.md)。

---

## 六、可证伪条件（B0→B3 配对增量判据与识别阶梯读数阈值）

### 6.1 中心主张判负条件（任一成立即主张证伪）

| 编号 | 条件 | 判负读数 |
|---|---|---|
| Φ1 | 协议机制无 XP 增益 | B1−B0、B2−B1、B3−B2 的配对增量全部 ≤ 0（同 seed/keys 配对，δ_min=20 尺度下无正信号，3–5 个开发 seed 一致） |
| Φ2 | 历史信息无因果作用 | history-shuffle drop 不显著：G1 − G2 ≤ 0 或配对区间跨零 |
| Φ3 | 协议表征与价值排序脱钩 | context swap 因果一致率 ≤ 0.5（随机水平，见 [`METHOD_SPEC.md`](METHOD_SPEC.md) 4.3） |
| Φ4 | 适应藏在 task 通路 | leakage 审计读数：打乱伙伴历史后 task 表征变化量显著（说明 task GRU 吸收了伙伴历史，结构性隔离失效） |
| Φ5 | Bayes 语义不成立 | 校准协议不通过：held-out log score 劣于先验基线，或校准覆盖率偏离名义 90% 区间超出容忍带（见 [`METHOD_SPEC.md`](METHOD_SPEC.md) 2.5） |
| Φ6 | 正式判决失败 | 按路径 1（用户裁定）配对层级推断下，对重训 FCP/OP/SA run-level 节点的差值区间跨零且 Welch 保守边际不支持 |

### 6.2 识别阶梯读数阈值（现象 Φ 的因果识别，与 G 节设计衔接）

- **L0 零成本定位**：NN/k-分划判别定位零成本维度；读数为判别准确率与维度归因表（无阈值门控，纯定位）；
- **L1 正对照**：lineage/热启动种群上，已知协调结构的任务必须给出方向正确的读数（符号一致率 ≥ 0.9），否则测量管线判废；
- **L2 机制定位**：三控制读数（shuffle drop、swap 一致率、leakage 审计）与 L1 正对照方向一致；
- **L3 稳健性收口**：换前缀评估器、换模式基底后 L2 结论保持（符号一致率 ≥ 0.8）；
- 竞争解释 H1–H4 逐级排除记录入证据台账；阶梯读数本身 `scientific_readout_allowed: false`，正式主张只由 6.1 判据产生。

### 6.3 SOTA 判决口径（用户裁定：路径 1）

- 在固定 Official commit 上重训/获取 FCP、OP、SA 的 run-level 节点（FCP 种群约 24 亿步/布局，为最大成本项），保存全部 run/cell 级数据；
- 己方与基线同 episode keys 评估，统计单位为训练运行（run 级），做配对或双样本层级推断；
- 已发表 Table 2 数值（Simple 6±29、Wide 23±40）降为外部 sanity check，不参与判决；
- 禁止"己方 bootstrap 区间减基线点估计 = 差值置信区间"的任何变体；
- 全口径算力报告（total transitions、upstream partner cost、GPU-hours、peak memory、参数量、performance–compute frontier）见 [`research/DEVELOPMENT_MATRIX.md`](research/DEVELOPMENT_MATRIX.md) 第四节。

---

## 七、B0–B3 与本规格的映射（组件级索引）

| 层级 | 科学含义 | 承载组件（详见 [`METHOD_SPEC.md`](METHOD_SPEC.md) 第八节映射表） |
|---|---|---|
| B0 | Full-history recurrent PPO 强制核心基线 | Official CNN+GRU trunk 单通路（task 通路含伙伴历史），无显式协议对象 |
| B1 | + 显式 protocol encoder（架构增益） | 三对象架构 x_t/u/c_t、结构性信息隔离、actor/critic 接口（METHOD_SPEC §1） |
| B2 | + 合法 all-action value supervision（decision-equivalent 监督增益） | L_signature、anchor 合法化重建、L_separation、三类可识别性控制，及作为 Bayes 似然的行动条件协议响应模型 L_response（METHOD_SPEC §2/§3/§4/§5） |
| B3 | + action-conditioned VOI（主动协议形成增益） | 在 B2 已建立的行动条件协议响应模型之上构建 VOI(a)（信息增益 − 任务代价），并入行动选择（METHOD_SPEC §8） |

矩阵语义与统一条件（相同伙伴分布/总 transitions/容量等级/run-disjoint 测试/3–5 seed）以 [`research/DEVELOPMENT_MATRIX.md`](research/DEVELOPMENT_MATRIX.md) 为唯一权威口径；不设制度性停机门槛，取舍由配对增量决定。

---

## 八、交叉引用与一致性声明

- 本规格中心主张与 [`research/REVIEW_AND_SUGGESTION_2026.md`](research/REVIEW_AND_SUGGESTION_2026.md) §12 核心主张逐字对齐；estimand D_V(t) 与 §7.4、[`research/TRAJECTORY_AND_ESTIMATION_SPEC.md`](research/TRAJECTORY_AND_ESTIMATION_SPEC.md) §五 P4 口径一致；
- 判决统计与 [`research/STATISTICAL_PREREGISTRATION.md`](research/STATISTICAL_PREREGISTRATION.md) 路径 1 一致（用户裁定生效后该文件应追加裁定记录；本规格先行按路径 1 撰写）；
- 开发矩阵与 [`research/DEVELOPMENT_MATRIX.md`](research/DEVELOPMENT_MATRIX.md) 完全一致；本规格不新增门控；
- 旧口径废止清单：partner belief 单一对象（改三对象）、E[Δ]·TV/2（改 D_V(t)）、partner-action oracle 主估计口径（禁）、"显著超过 SOTA"措辞（判决前禁用）、Θ2 中心论点地位（降为支持性）。

**实现落点文件**：本规格为纯科学口径，无代码落点；其全部实现约束由 [`METHOD_SPEC.md`](METHOD_SPEC.md) 逐节给出落点文件。
