# METHOD_SPEC：DEPI 方法实现规格（decision-complete）

修订头：本文件于统一重构（`Rigor_与_Generality_统一优化_7f793915.md` A/B/C 节；外部评审归档 [`research/REVIEW_AND_SUGGESTION_2026.md`](research/REVIEW_AND_SUGGESTION_2026.md) §1/§2/§3/§4/§5/§6/§10/§12）中创建，是 `src/path_c/` 实现批的**唯一权威执行口径**：所有设计决策已在本文件内闭合，实现代理不得保留开放决策；与既有文档冲突时以本文件与 [`SCIENTIFIC_SPEC.md`](SCIENTIFIC_SPEC.md) 为准。

用户裁定（已生效）：(a) SOTA 判决走路径 1（见 [`research/STATISTICAL_PREREGISTRATION.md`](research/STATISTICAL_PREREGISTRATION.md)）；(b) latent 信念补全完整 Bayes 生成语义（本文件第二节）。

状态：`authoritative: true`，供地基批（任务 #10）直接执行。

---

## 一、三对象架构（x_t / u / c_t）

### 1.1 对象与维度（全部定死）

| 对象 | 语义 | encoder | 输入（仅合法信息，SCIENTIFIC_SPEC §2 L1–L5） | 输出维度 |
|---|---|---|---|---|
| x_t | 任务状态 | TaskEncoderCell（GRU-128） | **仅当前观测 o_t**（Official CNN trunk→Dense-128→LayerNorm→GRU） | 128 |
| u | 稳定伙伴能力/倾向 | CapabilityEncoderCell（GRU-64，新增） | [o_t − o_{t−1} 观测差, ego 前一步动作 embedding(16), episode_start 标量] | 16 |
| c_t | 动态协议状态 | ProtocolEncoderCell（GRU-128，新增） | 与 u 相同的证据向量 e_t | 分类后验 π_t∈Δ^K（K=4），c_t=Σ_k π_{t,k} m_k，m_k∈R^16 |

决策记录：

- **task 通路输入收缩为"仅当前观测"**：删除 `task_encoder.py` 的 `previous_action_embedding` 与 `action_projection`（其 kernel 现为零初始化，删除不扰动 Official trunk 权重）；ego 动作历史与全部跨步伙伴证据移入 capability/protocol 通路。这是结构性隔离的第一半（task GRU 在输入层不再接触任何伙伴历史载体）。
- **观测差 o_t − o_{t−1} 是伙伴行为在 ego 视野内的唯一合法证据载体**（伙伴位移/取放经可见格变化体现），归 capability/protocol 通路独占；task 通路不读差分。
- K=4 分类协议状态取代 8 维对角 Gaussian：z_t 为离散协议制度变量（第二节），c_t 为后验均值嵌入；`latent_dim: 8`、`posterior_particles: 16`、`log_standard_deviation_*` 配置项废止，替换为 `protocol_components: 4`、`component_embedding_dim: 16`、`capability_dim: 16`、`capability_hidden_dim: 64`。

### 1.2 信息通路（结构性隔离，非正则）

| 通路 | 读什么 | 不读什么（结构保证） |
|---|---|---|
| task | o_t（含 GRU 自身时序） | 动作 embedding、观测差、任何 partner-side 特征——**输入层不存在这些接线** |
| capability | e_t=(Δo, a^ego_{t−1}, start)；GRU 跨步 | 当前原始观测全量、任务 trunk 特征 |
| protocol | 同 capability 的 e_t；GRU 跨步；输出 π_t 与 c_t | 同上；另外协议后验不得以 partner lineage/code 为输入（F2） |
| actor | [x_t; u; c_t] | partner code、joint 观测 |
| critic | V、Q1、Q2 均以 [x_t; u; c_t] 为条件 | 同上 |
| response decoder（似然头） | 见 2.3：事件项只读 (c_t, u, a^ego_t)；运动学项另读 stop-gradient 帧特征 | 事件项**不得**读 task 特征（封堵评审 §10.2 的绕过路径） |

context dropout 重新定义：以概率 p（沿用 0.30→0.10 退火）把 (u, c_t) 替换为**先验上下文**（π=prior=均匀分布对应的 c_0=均值嵌入，u=零向量）；task 通路不再参与 dropout（其输入已结构性不含伙伴历史，prior 路径即真 generalist，评审 §2 的 context dropout 漏洞随之闭合）。

### 1.3 共享 actor / critic 接口

- actor：`UniversalCoordinationActor(task_features, context)`，其中 `context = concat(u, c_t)`（32 维）；低秩调制保留：`task_basis(x)` ⊙ `context_gain(concat(u,c))`。仍为单 actor。
- critic：`UniversalDuelingCritic(task_features, context)` 输出 (V, Q1, Q2)；粒子/假设路径接口 `*_from_features_and_latent` 改为 `*_from_features_and_context`，**删除** `[latent, zeros, zero-uncertainty]` 人工 summary 构造（评审 §4）。
- 单 actor 单 critic 原则不变：不为伙伴类型/来源训练独立网络。

### 1.4 与现有文件的映射（改哪个文件、替换什么）

| 文件 | 变更 |
|---|---|
| `src/path_c/task_encoder.py` | 删除动作 embedding/`action_projection`；输入收缩为 o_t |
| `src/path_c/belief_encoder.py` | **废止重建**为 `src/path_c/protocol_encoder.py`：`CapabilityEncoderCell` + `ProtocolEncoderCell`（输出 π_t logits）；删除 log σ 头与 normalized uncertainty |
| `src/path_c/belief_set_encoder.py` | `gaussian_summary`/`prior_gaussian_summary` 替换为 `mixture_summary(π, m_k)`/`prior_mixture_summary` |
| `src/path_c/model.py` | `PolicyState` 字段改为 (task_carry, capability_carry, protocol_carry, context_summary, previous_observation, previous_action, episode_start)；`ContextOutput` 携带 (task_features, u, π, c)；删除 belief_mean/log_std/uncertainty 字段与 particle 人工 summary |
| `src/path_c/universal_actor.py` | 接口改 `(task_features, context)`，context=concat(u,c) |
| `src/path_c/universal_critic.py` | 同上；删除 shaped_belief_embedding 特例 |
| `src/path_c/response_decoder.py` | 删除 task_features 参数；拆为事件头（读 c,u,a）与运动学头（读 stop-grad 帧特征 + c,u,a），见 2.3 |
| `src/path_c/types.py` | 同步 dataclass/NamedTuple 字段 |
| `src/path_c/gradient_routing.py` | 所有权表更新（见 3.4）；PCGrad 函数废止 |

**实现落点文件**：`src/path_c/{task_encoder.py, belief_encoder.py→protocol_encoder.py(新), belief_set_encoder.py, model.py, universal_actor.py, universal_critic.py, response_decoder.py, types.py, gradient_routing.py}`。

---

## 二、Bayes 生成语义（用户裁定 b：生成模型 + proper likelihood + held-out 校准）

### 2.1 先验 p(z) 的随机变量定义

- **z_t ∈ {1,…,K}，K=4**：t 时刻交互所实例化的**协议制度**（protocol regime）——定义为"诱导不同 ego centered-advantage 签名 A_t(·) 的协作惯例类别"，类别数 4 与 simplex 锚点一一对应（7.2），注册映射 SP→1、OP→2、SA→3、FCP→4（类别是语义锚，不是身份标签：同一伙伴在不同交互中可跨类别）。
- z_t **不是**固定外生伙伴类型：其转移由双方动作驱动（SCIENTIFIC_SPEC §4.1），后验以递归滤波跟踪。
- **先验** p(z_0) = Uniform(1/K)；episode 起始无任何协议证据。转移先验 P(z_{t+1}|z_t) 取 sticky 对称形式：P(保持)=0.9，P(跳变到任一其他)=0.1/3（常数，注册值；不学习，避免增加不可识别自由度）。

### 2.2 似然 p(y_{t+1} | z_t, H_t, a_t^ego) 的具体形式

**载体：行动条件协议响应模型**（评审 §12.5；计划 C 节）：

\[
p_\omega(y_{t+1} \mid z_t, H_t, a_t^{ego}) = \sum_{k} \pi_{t,k}\, p_\omega(y_{t+1} \mid z_t=k, a_t^{ego}, \mathrm{sg}[\text{frame}_t]),
\]

其中 y_{t+1} 为下一步**协议事件向量**（全部可由合法观测派生）：

1. partner_visibility（Bernoulli）；
2. partner 相对位置类（categorical，沿用 `PARTNER_POSITION_CLASSES`）；
3. partner 朝向（categorical，沿用 `PARTNER_DIRECTION_CLASSES`）；
4. partner inventory 各因子（categorical，沿用现有因子分解）；
5. **协议事件指标** interaction_change（Bernoulli）：交付/取放责任变化、让位/争抢/等待——操作性定义为"partner 可见 inventory 变化 ∨ 相对位置类变化"的联合模式标签。

决策记录（封堵运动学绕过，评审 §10.2）：分量 1–4（运动学）允许读 stop-gradient 帧特征 `frame_t`（当前观测扁平化）——预测运动学必须有坐标系；分量 5（协议事件）**只读 (c_t, u, a_t^ego)**。L_response 对 belief 参数的梯度主通道是分量 5 与分量 1–4 经 π 混合权重的路径。

### 2.3 近似后验与 Bayes 更新规则

- 近似后验 q_φ(z_t | H_t) = Categorical(softmax(ℓ_t))，ℓ_t 由 ProtocolEncoderCell 递归产生：
  \[
  h_t = \mathrm{GRU}_\phi(h_{t-1}, e_t),\quad h_0 \text{ 使 } q_0 = \text{prior};\qquad \ell_t = W_\ell h_t.
  \]
- 这是**摊销递归 Bayes 滤波**：因 z 离散且 K=4，后验预测可精确边缘化（无需重参数化采样），似然训练目标
  \[
  \mathcal{L}_{\text{response}} = -\frac{1}{T}\sum_t \log \sum_{k} \pi_{t,k}\, p_\omega(y_{t+1}\mid z_t=k, a_t^{ego}, \mathrm{sg}[\text{frame}_t])
  \]
  即 proper negative log-likelihood，使 q_φ 成为该生成模型下的规范近似后验（变分滤波的特例：预测似然最大化）。
- 在线"Bayes 更新"读法：每步证据 e_t 进入 GRU 即完成 q_{t-1}→q_t 更新；sticky 转移先验以固定权重混入递归输入（ℓ_t 加 log-sticky 修正项，实现为常数偏置表，不引入参数）。

### 2.4 校准协议（held-out）

- **校准集构造**：run-disjoint 且算法族留出伙伴各 2 个 run（注册名单跑前冻结），episode keys 与训练/评估 key 流分离（根种子偏移 2000）；每 run 64 回合 ×400 步，记录 (H_t, a_t^ego, y_{t+1}) 三元组；该集**不进入任何训练梯度**。
- **proper scoring rule 选型**：主指标 = 后验预测 **log score**（每步 NLL，越低越好）；次指标 = interaction_change 的 **Brier score**。二者均 proper。
- **覆盖率指标**：对 categorical 目标构造后验预测 90% 最高概率集，经验覆盖率 ∈ [0.85, 0.95]；并报告 π_t 对 held-out 协议类别分类器的 reliability 曲线。
- **通过阈值**（跑前注册，失败触发 SCIENTIFIC_SPEC Φ5，Bayes 主张收回为 "stochastic context embedding"）：
  1. log score 优于两个基线（均匀先验混合；无历史帧特征基线）各 ≥ 0.02 nats/step；
  2. 覆盖率落在 [0.85, 0.95]；
  3. 事件 Brier ≤ 0.9 × 先验基线 Brier。
- 频率：开发期每 anchor 触发评估一次；正式 run 只在冻结 checkpoint 评估一次。

### 2.5 封堵置信度作弊（评审 §3）

- **废止**：log σ ∈ [−5, 2] 归一化 uncertainty、`robust_generalist_loss`（uncertainty 加权 full/prior KL）、IB 项。理由：存在 log σ→−5 ⇒ L_robust→0 的作弊路径，且 IB 0.001 vs robust 0.1 相差 100 倍。
- **替换的不确定性来源**（全部不可通过压缩自身参数归零）：
  1. **后验熵** H(q_t)——有界量，由 proper log score 严格惩罚过度自信（校准协议兜底）；
  2. **bootstrap history encoder 集成**：B=3 个 ProtocolEncoderCell（不同初始化种子 + dropout 0.1），认知不确定性 = 集成 π 的总变差距离均值；每个成员参数独立，压缩需同时欺骗三个独立模型与校准集。
- uncertainty 读数只作报告量与 VOI 输入（B3），不进入任何损失权重。

**实现落点文件**：`src/path_c/{protocol_encoder.py(新), response_decoder.py, response_targets.py, calibration.py, model.py, training.py(删除 robust/IB)}`；校准评估应用入口 `experiments/overcooked_v2/calibration_app.py`。

---

## 三、损失收缩（废止八目标 PCGrad）

### 3.1 总目标

\[
L = L_{\mathrm{PPO}} + \lambda_A L_{\mathrm{signature}} + \lambda_R L_{\mathrm{response}} + \lambda_S L_{\mathrm{separation}}.
\]

### 3.2 各项定义、数据来源、初始 λ 与调参规则

| 损失 | 数学定义 | 数据来源 | 初始 λ | 调参规则 |
|---|---|---|---|---|
| L_PPO | 现行 clipped actor+value−entropy（shaped reward、GAE γ=0.99 λ=0.95 不变） | rollout batch | 1.0（含 value_weight 0.5、entropy 0.01） | 不动 |
| L_signature | Huber(Q_c(x,u,c,a), A_t(a)) + 排序铰链 Σ_{(i,j):|A_i−A_j|>2.0} max(0, 0.1 − (Q_i−Q_j)·sign(A_i−A_j))；Q_c=Q−mean_a Q（twin 取 min） | 合法化 anchor 批（第五节）与 replay | λ_A=1.0 | 梯度范数比规则：保持 ‖∇L_i‖/‖∇L_PPO,actor‖ ∈ [0.3, 3]，越界按 ×/÷1.5 调整，一次只动一个 λ，变更先入台账 |
| L_response | 2.3 式 proper NLL（后验预测对 y_{t+1}） | rollout 转移（y 由合法下一观测派生）+ anchor replay | λ_R=1.0 | 同上范数比规则；校准失败按 2.4 处置，不得以加大 λ_R 规避 |
| L_separation | matched pair：observable-equivalent 对 ‖c_i−c_j‖²；decision-distinct 对 max(0, m−‖c_i−c_j‖)，m=0.25·\bar d_obs（distinct 对签名距离均值，跑前注册），权重=签名距离 | 合法化 matched pair（第五节） | λ_S=0.1 | 同上范数比规则 |

### 3.3 PCGrad 废止与 belief gradient 时序修复（评审 §10.4）

- **废止** `combine_belief_gradients`、`project_conflicting_gradient`、`route_response_with_pcgrad` 及八目标 `BELIEF_OBJECTIVE_ORDER`：固定顺序 PCGrad 的优先级由列表顺序隐式决定，且不可识别机制互相掩盖。
- **时序修复**：废止"snapshot_params 上算梯度→PPO/raw-Q/heads 更新后→旧梯度应用到新 heads"的顺序。新规则：**四项损失在同一当前参数上的一次 `value_and_grad` 中计算，一次梯度步更新全部可训练参数**；不再有滞后 snapshot 梯度。raw-Q 的 Retrace 密集更新若保留为独立子步，其梯度**只更新 critic 头**，不得触及 encoder/actor。

### 3.4 belief update 纳入 PPO KL 核算（评审 §10.4）

- 每次组合更新后，在 rollout buffer 上计算 **combined_policy_kl**：行为策略与更新后 π_θ(·|x,u,c) 的近似 KL（含 context 通路参数变化的全部效应）；
- 与既有 approx_kl 一并报告；若 combined_policy_kl > 0.04（注册阈值），当前 PPO 周期的剩余 minibatch 提前终止（early-stop），下一周期 update_epochs 减 1（下限 2）；
- 参数所有权表（`keep_owned_gradients` 重写）：task_encoder ← {ppo, signature}；capability/protocol encoder ← {ppo, response, signature, separation}；actor ← {ppo}；critic ← {ppo(V), signature(Q)}；response_decoder ← {response}。

### 3.5 待证增量后再加入的项（现全部移除）

decision-regret shaping、IB、robust KL、Q-policy coupling：从训练图中移除（代码保留为探索轨开关，默认关闭）；重新加入的前提是在开发矩阵中证明配对增量为正，且先入台账。

**实现落点文件**：`src/path_c/{training.py, gradient_routing.py}`；配置 `experiments/overcooked_v2/configs/delta_zsc_{simple,wide}_formal.yaml` 的 `loss:` 段重写（新增 `loss_v2` 段，旧段标记 deprecated 保留一轮）。

---

## 四、三类可识别性控制（实现方式与报告读数）

### 4.1 task-history leakage control（结构性 + 审计）

- **结构**：1.2 的输入层隔离（task 通路无动作 embedding、无观测差）。
- **审计程序**（每正式 checkpoint 跑一次）：held-out 回合上，(a) 打乱伙伴侧证据重放，测 task 表征相对漂移 ‖Δx‖/‖x‖；(b) 线性探针用 x_t 预测 partner run 身份（留出式，balanced）。
- **读数**：`leakage_probe_accuracy`（目标 ≤0.55，机会 0.5）、`task_repr_drift_under_shuffle`。超阈即 SCIENTIFIC_SPEC Φ4 判负。

### 4.2 history shuffle control

- **程序**：固定当前观测与 task carry；伙伴侧证据历史从另一 partner run（同布局、run-disjoint）抽取并替换 capability/protocol 通路重放；episode keys 与正常评估配对。
- **读数**：`history_shuffle_drop = XP_{G1} − XP_{G2}`（配对 bootstrap CI，9999 次），四格口径见 SCIENTIFIC_SPEC §5.2；drop ≤0 或区间跨零 → Φ2 判负。

### 4.3 context swap control

- **程序**：held-out 回合中检索 task 状态匹配对（x 空间最近邻，‖x_i−x_j‖ ≤ ε，ε 取 held-out 分布 5% 分位）；交换两状态的协议上下文 c（u 保留原位），记录 Δlogits；与该状态对的真实 CRN centered 签名差 ΔA 对比。
- **读数**：`protocol_swap_causal_consistency` = P(sign(Δlogit) 与 ΔA 排序一致)，目标 > 0.65（机会 0.5，注册阈值）；≤0.5 → Φ3 判负。

**实现落点文件**：新审计模块 `src/path_c/identifiability_controls.py`（新文件）；评估入口 `experiments/overcooked_v2/{signal_audit_app.py, official_evaluation_app.py}` 扩展读数。

---

## 五、anchor 合法化重建（评审 §5 / §12.3）

### 5.1 真实 episode 快照内容清单（每 anchor 必存）

1. **环境快照**：完整 simulator state（`environment_state`）；
2. **ego recurrent state**：完整 PolicyState（新三通路 carries + previous_observation/action/episode_start），来自真实 rollout 的 `target_ego_policy_state`；
3. **partner recurrent state**：伙伴在真实 episode 该时刻的**真实 carry**——禁止清零、禁止 carry 替换；
4. **合法历史**：ego 合法观测/动作/done 前缀的索引指针（可由 rollout records 重建，审计用）；
5. **lineage**：partner_source、partner_run_id、checkpoint 阶段、种子。

**禁令**：mid-episode code reset（`_generator_intervention_state` 整个函数删除）、partner carry 清零、`partner_episode_start=True` 强制拼接。

### 5.2 matched pair 构造（不再依赖生成器 code）

- pair 来源：同一 rollout 批次内、task 状态近邻（x 空间）但来自**不同 partner run** 的两个真实快照；CRN 根键配对，各自以真实状态前行 16 步 probe（沿用 `advance_anchor_world` 的 CRN 机制，但世界不再被干预）；然后采集 all-action continuation（6 动作 ×4 replicas ×128 步，预算不变）。

### 5.3 判定程序（decision-complete）

对每个 pair，冻结的 held-out 比较器（基于 16 步合法历史特征：逐步观测差扁平化 + ego 动作序列的固定特征映射 + 逻辑回归，训练集与 anchor 批 disjoint）给出 2AFC 判别准确率 acc：

| 条件 | 分类 | 处置 |
|---|---|---|
| acc ≤ 0.55 且签名距离 ≤ τ_sig | **observable-equivalent** | L_separation 一致性项 ‖c_i−c_j‖² |
| acc > 0.70 且签名距离 > τ_sig(=1.0 raw 回报单位) | **decision-distinct** | L_separation 分离项（margin 按 3.2） |
| 其余 | **irreducible ambiguity** | 不进任何分离/一致性损失；记入 replay 台账 `irreducible_ambiguity` 字段 |

- **读数**：每触发报告三类占比（`pair_class_fractions`）与 `irreducible_ambiguity_fraction`；τ_sig、acc 阈值跑前注册，事后变更须入台账新条目。

**实现落点文件**：`src/path_c/{anchor_sampling.py(重写 matched 分支、删除 _generator_intervention_state), counterfactual_anchor.py(保留 CRN 采集核心), decision_geometry.py}`；快照字段扩展 `src/path_c/storage.py`。

---

## 六、particle / regret 修复（评审 §4 / §12.4）

- **废止**：任意 Gaussian 粒子上的 Q/regret 计算（16 particle path 删除）。
- **hypothesis 来源限定**（封闭集合，每个 hypothesis 都在监督分布内）：
  1. 真实合法历史：anchor 状态处的后验均值上下文 (u, c)；
  2. bootstrap history encoder：2.5 的 B=3 个集成成员对**同一真实历史**的输出 (u^{(k)}, c^{(k)})。
- **一致 continuation 标签**：假设 h^{(k)} 所在 anchor 状态 s 的 CRN all-action 签名 A_s(·) 就是全部假设共享的标签（标签是历史-状态的属性，与编码器成员无关）；不存在无标签假设。
- **regret 定义保留但只在假设集合上计算**：R_t = E_h max_a Q(x,h,a) − max_a E_h Q(x,h,a)，h ∈ {后验均值 ∪ 3 个集成成员}；仅作报告读数与 B3 VOI 输入，不做 reward shaping（待增量证明）。
- **M1 闸门扩展**：动作排序 Spearman 检验同时在 (i) 后验均值上下文 与 (ii) 每个集成成员上下文处执行；通过条件：两路径均 Spearman ≥ 0.8 于 ≥90% anchor 状态；particle path 不达标即阻断正式 run。

**实现落点文件**：`src/path_c/{model.py(删除 particle 接口), raw_q.py, regret_potential.py}`；M1 闸门 `experiments/overcooked_v2/mechanical_e2e_app.py` 与测试 `experiments/overcooked_v2/tests/test_delta_zsc_model_smoke.py` 扩展。

---

## 七、生成器几何修复与伙伴分布（评审 §6 / §10.1 / §12.6）

### 7.1 code 维度与 simplex 锚点

- **code_dim = 3**（4 source ⇒ d ≤ K−1=3；`code_dim: 8` 废止）。
- **仿射独立 simplex 锚点**（正四面体顶点，替换现行 v_2=−v_0、v_3=−v_1 的秩-2 构造）：
  \[
  v_0=\tfrac{(1,1,1)}{\sqrt3},\ v_1=\tfrac{(1,-1,-1)}{\sqrt3},\ v_2=\tfrac{(-1,1,-1)}{\sqrt3},\ v_3=\tfrac{(-1,-1,1)}{\sqrt3};
  \]
  注册映射 SP→v_0、OP→v_1、SA→v_2、FCP→v_3。`source_code_anchors()` 重写；`count != 4` 报错保留。
- **support 内采样**：`sample_partner_codes` 重写为从**拟合的 Dirichlet(α)** 采样（α 由 source distillation/imitation 集上 code 使用分布的矩估计拟合），采样域为 simplex；`[-1,1]^d` 均匀立方体采样废止；smoothness 扰动改为 simplex 内 Dirichlet 邻域扰动。

### 7.2 共同状态 diversity、冻结/KL 约束

- **diversity**：固定 M=64 环境快照 bank（从合法 anchor 快照抽取，注册种子）；成对 code 在共同 bank 上跑 CRN all-action continuation，diversity = 平均成对签名距离。**废止**现行 BR-diversity（混合状态访问差异与 Q 误差）。
- **主网络冻结**：imitation 阶段（progress < 0.3，imitation weight 1.0→0）结束后**冻结生成器主网络参数**；对抗/课程更新只作用于 code 分布参数 α。备用方案（注册）：若冻结使开发 XP 配对增量显著为负，切换为强 KL 约束（对冻结 source 混合策略逐步 KL ≤ 0.1 nats，Lagrangian，沿用现有机制），切换先入台账。
- **archive**：保留历史优良 code 的 archive（容量 256，按 diversity 贡献保留），防遗忘。

### 7.3 SA/FCP 缺失时的静态广域伙伴集替代（默认方案，评审 §10.1）

当前树只有 SP/OP 各 10 seed，SA/FCP 不可用 ⇒ **B0–B2 矩阵默认使用静态广域伙伴集，生成器默认关闭**：

| 组成 | 明细 | 数量 |
|---|---|---|
| SP/OP 多 seed | 现有 SP×10、OP×10 | 20 |
| 多 checkpoint 阶段 | 每 run 取 progress {0.0, 0.5, 1.0} 三个 ckpt | 60 |
| 多超参变体 | OP 宽度变体（Official adapter 既有）×2 seed | 2 |
| heuristic 伙伴 | 2 个脚本化伙伴（greedy courier、stationary helper），经 `PartnerFunctions` 接口实现 | 2 |

- 训练采样：族间均匀、族内按 seed 均匀，采样表跑前注册；**heuristic 族整体留出为测试专用算法族**（family-disjoint 声明的载体），不出现在训练中；
- checkpoint 阶段语义：{0.0}=初始化/弱、{0.5}=中期、{1.0}=终训，构成能力梯度；
- 生成器修复（7.1/7.2）落地后作为探索轨变体重新开启（需 4 source 齐备：SA/FCP 上游按 `upstream` 配置 30M 步补训，成本按 DEVELOPMENT_MATRIX §4 记账）。

**实现落点文件**：`src/path_c/{partner_generator.py, partner_sources.py, external_partner.py}`；heuristic 伙伴新增 `src/path_c/heuristic_partners.py`（新文件）；配置 `configs/delta_zsc_{simple,wide}_formal.yaml` 的 `partner_generator:` 段重写为 `partner_pool:` 段。

---

## 八、B0–B3 与本规格的映射表

| 层级 | 包含的本规格组件（节号） | 主要落点文件 |
|---|---|---|
| B0 | 无本规格新增组件：Official CNN+GRU 单通路 recurrent PPO（task 通路保留动作输入的基线形态，即现行 trunk 原样） | `task_encoder.py`（基线分支）、`training.py`（纯 PPO 路径） |
| B1 | §1 三对象架构全部（含结构性隔离、actor/critic 接口、context dropout 新定义）；§4.1 leakage 控制 | `protocol_encoder.py(新)`、`model.py`、`universal_actor.py`、`universal_critic.py`、`identifiability_controls.py(新)` |
| B2 | §3 损失收缩全部（L_signature/L_response/L_separation、PCGrad 废止、KL 核算）；§2 Bayes 生成语义与校准（L_response 即似然）；§5 anchor 合法化；§6 particle/regret 修复；§4.2/§4.3 shuffle/swap 控制；§7.3 静态伙伴集 | `training.py`、`anchor_sampling.py`、`counterfactual_anchor.py`、`response_decoder.py`、`calibration.py`、`partner_sources.py` |
| B3 | 在 §2 行动条件似然之上构建 VOI(a) = E_y[max_{a'} E[Q(a')|H_t,a,y]] − max_{a'} E[Q(a')|H_t] − C_task(a)，并入 actor 的行动选择（信息增益 − 任务代价） | `regret_potential.py` 重写为 `voi.py`（新）、`training.py` |

与 [`research/DEVELOPMENT_MATRIX.md`](research/DEVELOPMENT_MATRIX.md) 完全一致：B_k 严格包含 B_{k−1}；统一条件（同伙伴分布/总 transitions/容量等级/run-disjoint 测试/3–5 seed）与全口径算力报告以该文件为准；取舍由配对增量决定，不设停机门槛。

---

## 九、一致性声明与冲突处置

1. 本规格与 [`SCIENTIFIC_SPEC.md`](SCIENTIFIC_SPEC.md) 逐项对应：三对象/转移定义→其 §4；estimand→其 §5；可证伪条件 Φ1–Φ6→其 §6；本文件各节产出相应读数。
2. 与 [`research/STATISTICAL_PREREGISTRATION.md`](research/STATISTICAL_PREREGISTRATION.md) 一致：δ_min=20.0；判决路径 1（用户裁定）；本文件不定义判决统计。
3. 与 [`research/DEVELOPMENT_MATRIX.md`](research/DEVELOPMENT_MATRIX.md) 一致：shuffle drop、protocol-swap 因果效应、total compute 读数由本文件 §4/§7 供给实现。
4. 与 [`research/TRAJECTORY_AND_ESTIMATION_SPEC.md`](research/TRAJECTORY_AND_ESTIMATION_SPEC.md) 一致：全部监督与测量只使用合法历史口径（SCIENTIFIC_SPEC §2），partner-action oracle 禁止进入策略、后验与监督标签。
5. 内部矛盾处置：本文件各节决策均附"决策记录"；若实现中发现两节冲突，以节号较小者为准并立即向 leader 报告，不得自行折中。
