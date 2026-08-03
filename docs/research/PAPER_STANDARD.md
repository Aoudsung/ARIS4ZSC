# 顶会标准研究方案（ICLR / NeurIPS 级）

职责：把研究方案提升到 ICLR/NeurIPS 录用标准。本文档定义论文形态、主张层级、定理清单、基线注册、统计协议、表图设计与审稿防御。候选论点裁决（[THESIS_CANDIDATES](THESIS_CANDIDATES.md)）完成后，本文档随之收敛为正式 paper outline。
状态：探索轨产物，`scientific_readout_allowed: false`。

## 一、顶会录用线在哪

ICLR/NeurIPS 对这类工作的录用线可以拆成四问，每问都要有硬答案：

1. **新在哪？** 与 FCP、MEP、GOAT、CEC、PECAN、ReCoLLAB 的差异化必须一句话能说清。本方案的差异点：第一个把信息价值分解链 `V_fix ≤ V_state ≤ V_hist ≤ V_HZ ≤ V_full` 操作化为可估计的 ZSC 诊断，第一个在同前缀配对延续上定义并测量回收率。见 [LITERATURE_MATRIX](LITERATURE_MATRIX.md) 生态位图谱。
2. **证据硬不硬？** 主张必须有判决性实验支撑，且统计推断在训练运行级（不是回合级）成立。
3. **理论有没有牙？** 至少一个定理给出可检验的定量预测，而不只是事后解释。
4. **结果可复现吗？** 协议、预算、seed、代码全部披露，官方评估协议一字不改。

## 二、论文形态与标题候选

形态：**测量驱动的方法论文**（measurement-first method paper）。主干是诊断框架与理论预测，DELTA-ZSC-E2E 作为框架的第一个完整实例与 SOTA 竞争者出场。这个形态的好处：即使方法不是第一，诊断框架与机会测量本身仍构成独立贡献，论文不会整体崩掉。

标题候选（裁决后三选一）：

- "How Much Is an Unseen Partner Worth? Measuring and Recovering Coordination Value in Zero-Shot Coordination"
- "The Recoverable Gap: An Information-Value Decomposition for Zero-Shot Coordination"
- "Before You Adapt: When Is Partner History Worth Reading in Zero-Shot Coordination?"

## 三、主张层级

论文的主张按强度分层，逐层可独立成立。投稿时按实际证据裁剪到最高成立的层级。

| 层级 | 主张 | 依赖 | 对应论点 |
|---|---|---|---|
| L1 工具 | 分解链六控制可在 OvercookedV2 上完整估计，能定位机会/信息/控制三类瓶颈 | S1、S2 | Θ5 |
| L2 发现 | 训练身份重合会系统性夸大适应机会；run-disjoint 分离后机会的真实量级首次被测量 | S1 | Θ1 |
| L3 理论 | TV 匹配界与 `Θ(log(Δ/ε)/κ²)` 历史需求在受控任务精确成立，并对生态任务给出定量预测 | S2、P1-P6 | Θ2 |
| L4 方法 | DELTA-ZSC-E2E 回收了可认证机会的显著部分，XP 优于注册基线 | S4、S5 | Θ3 |

L1 到 L3 不依赖 V6 方法成功。这是论文的保底线。

## 四、贡献清单（四条）

1. **身份重合审计**：证明事后选择上限可被 checkpoint 自匹配完全污染（119.65 → 0），给出 run-disjoint 修正协议。负结果包装为正贡献。
2. **信息价值分解诊断**：六控制设计加回收率，首次把"机会、信息、控制"三类瓶颈在同一面板上分开测量。
3. **匹配界与历史需求定理**：二惯例 TV 精确式，`Θ(log(Δ/ε)/κ²)` 必要与充分历史长度，含受控相图验证与生态外推预测。
4. **端到端信念协调方法（条件性）**：反事实延续监督的信念条件 actor，消融证明每个机制的必要贡献，XP 对比注册基线。

## 五、定理清单

| 编号 | 陈述（缩写） | 证明状态 |
|---|---|---|
| T1 | 分解链单调性：信息包含关系给出 `V_fix ≤ V_state ≤ V_hist ≤ V_HZ ≤ V_full` 与 `V_fix ≤ V_Z ≤ V_HZ` | 平凡，正文一段 |
| T2 | 二惯例静态特例的 TV 精确式：`V_HZ − V_hist = Δ(1−TV)/2`，`V_hist − V_fix = Δ·TV/2`，Bayes 路由器达紧 | 已形式化，[FOUNDATIONAL §10.2](../theory/DELTA_ZSC_FOUNDATIONAL_THEORY_AND_PROOFS.md) |
| T3 | 历史样本复杂度：必要 `n ≥ log(Δ/(4ε))/(4κ²)`，充分 `n ≥ 2log(Δ/ε)/κ²`，合为 `Θ(log(Δ/ε)/κ²)`，常数已复核 | 已形式化，FOUNDATIONAL §10.3 |
| T4 | 路由器紧上界与端到端等式：`J_route − J_sfix = Γ_hist − R_pred`，`J_route − J_fix(0) = Γ_hist − R_pred − C_wait` | 已形式化，FOUNDATIONAL §10.4 |

顶会标准要求 T2、T3 至少各有一个非平凡的生态推论（P4、P5、P6，见 [THEORY_PREDICTIONS](THEORY_PREDICTIONS.md)），并在实验中对照。定理只停在受控特例是常见拒稿理由，推论加对照是解法。

## 六、基线注册

外部基线（S5 必须全部对比，预算对齐按 ego-PPO 步为主、总步数辅报）：

| 基线 | 来源 | 角色 |
|---|---|---|
| SP、OP | Official 配方 | 协议内参照，10 seed 现成 |
| FCP | Strouse et al. 2022 | 训练多样性路线代表，主要对手 |
| MEP | Zhao et al. 2022 | 熵正则路线代表 |
| PLASTIC | Mirsky et al. 2023 | 信念推断路线代表 |
| PECAN | arXiv:2301.06387 | 反事实推断路线代表 |
| GOAT | arXiv:2504.15457 | 2025 真人评估强基线，复现成本评估后决定 |

内部对照（分解链自带，免费且审稿人必问）：state-only 路由，hist-only 路由，统一后接最佳模式，回合开始最佳固定模式。

## 七、统计协议

- 统计单位是训练运行，不是回合。10 seed 双布局矩阵，跨 seed 报告均值与区间。
- 9,999 次 run-node bootstrap，保留行列节点依赖与配对 episode-key 重采样（[FORMAL_EXPERIMENT_PROTOCOL](../FORMAL_EXPERIMENT_PROTOCOL.md) 已定义）。
- 主读数报原始回报差与区间，回收率 ρ̂ 只作辅助读数。
- `δ_min` 以一次正确交付的原始回报为尺度，S5 生成前注册。
- 消融增量用配对 seed 差值检验，不做跨 seed 裸比。

## 八、算力透明度

资源台账披露五项：ego 训练步数、辅助步数（反事实延续、generator、探针）、上游伙伴训练成本、GPU 小时与峰值显存、部署参数量与训练专用参数量分开报。预算数字的权威来源是 [FORMAL_EXPERIMENT_PROTOCOL](../FORMAL_EXPERIMENT_PROTOCOL.md)，本文不复制。

## 九、论文表图设计

| 编号 | 内容 | 数据来源 |
|---|---|---|
| Figure 1 | 受控相图：Δ × κ 平面上的可恢复价值热图，叠理论等值线 | S2 |
| Figure 2 | 生态时间曲线：TV(t)、κ(t)、有效切换窗口、ρ̂(t) 四线同图 | S1、S4 |
| Figure 3 | 信念潜空间的伙伴可分性，按决策相关性着色 | S3 |
| Table 1 | 主矩阵：XP 均值、Gap、区间，全基线对比 | S5 |
| Table 2 | 分解链六控制读数，按布局与伙伴类型分层 | S1 |
| Table 3 | 消融阶梯：每机制的 XP 增量与区间 | S4 |
| Table 4 | 理论预测对实测：P1 到 P6 逐行对照 | S2、S4 |

## 十、审稿防御清单

预写审稿人最可能的五条攻击与防御：

1. "机会可能根本不存在。" 防御：S1 的停机判据已把这种情况写成合法结论（L2 收缩为负结果论文），论文结构不依赖机会为正。
2. "和 ReCoLLAB/PLASTIC 的区别是什么？" 防御：生态位图谱加六控制对照，本文测全链价值而非识别加路由。
3. "理论只证明了两惯例静态特例。" 防御：T2/T3 的生态推论已注册，失配条款预写两种解释，受控加生态双层验证。
4. "辅助预算算不算作弊？" 防御：双口径预算报告，主对比按 ego-PPO 对齐。
5. "单高斯信念表达力不足。" 防御：audit-signals 的信念质量读数加信念消融；若确为瓶颈，作为 limitations 报告并给出升级路径，不藏。

## 十一、执行映射

| 论文章节 | 对应阶梯级 |
|---|---|
| §诊断框架与审计 | S1 |
| §受控理论与相图 | S2 |
| §机制分析 | S3 |
| §方法、消融与开发结果 | S4 |
| §主结果与统计 | S5 |

推进顺序即 [EXPERIMENT_LADDER](EXPERIMENT_LADDER.md) 的 S1 到 S5。每级完成时按本文档第九节回填对应表图，避免最后赶工。
