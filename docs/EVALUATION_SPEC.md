# EVALUATION_SPEC：DEPI 评估规格（伙伴划分、矩阵、判决统计、资源）

修订头：本文件于统一重构（`Rigor_与_Generality_统一优化_7f793915.md` I 节；外部评审归档 [`research/REVIEW_AND_SUGGESTION_2026.md`](research/REVIEW_AND_SUGGESTION_2026.md) §8/§10.6/§14）中创建，是四份权威文件之三：**伙伴划分、baseline、统计与资源**的唯一权威口径。问题与可证伪条件见 [`SCIENTIFIC_SPEC.md`](SCIENTIFIC_SPEC.md)；方法实现见 [`METHOD_SPEC.md`](METHOD_SPEC.md)；理论见 [`THEORY.md`](THEORY.md)。本文件只做汇聚与裁定，不复制被引文件的全文定义。

用户裁定（已生效，直接约束本规格）：SOTA 判决走**路径 1**——固定 Official commit 重训/获取基线 run-level 节点、同 episode keys 配对/双样本层级推断（裁定记录见 [`research/STATISTICAL_PREREGISTRATION.md`](research/STATISTICAL_PREREGISTRATION.md) 第五节，决策登记见 [`status/DECISION_LOG.md`](status/DECISION_LOG.md) D2）。

状态：`authoritative: true`。本规格与既有文档冲突时以本规格与 SCIENTIFIC_SPEC 为准。

---

## 一、伙伴划分与测试伙伴协议

评估的划分纪律共三层，全部为强制项，缺一则该读数不得作为正式主张：

| 划分层级 | 定义 | 权威出处 |
|---|---|---|
| **run-disjoint** | 测试伙伴与所有训练运行 disjoint（同一伙伴来源的不同训练运行之间也不得在训练与测试两侧同时出现）；跨方法共用同一测试集与同一组 episode keys | [`research/DEVELOPMENT_MATRIX.md`](research/DEVELOPMENT_MATRIX.md) 第二节统一条件第 4 条 |
| **algorithm-family-disjoint** | 测试伙伴所属算法族与训练伙伴分布的算法族不相交；中心主张的"可归因"要求在训练运行与算法族**均留出**的陌生伙伴上成立 | [`SCIENTIFIC_SPEC.md`](SCIENTIFIC_SPEC.md) §3 中心主张 |
| **heuristic 族 family-disjoint 测试族** | heuristic 族（脚本化伙伴：greedy courier、stationary helper）**整体留出为测试专用算法族**，不出现在训练中，是 family-disjoint 声明的载体 | [`METHOD_SPEC.md`](METHOD_SPEC.md) §7.3 静态广域伙伴集 |

配套规则：

- 测试伙伴名单与 episode keys **跑前注册冻结**；校准集伙伴同为 run-disjoint 且算法族留出，episode key 流与训练/评估分离（根种子偏移 2000），见 [`METHOD_SPEC.md`](METHOD_SPEC.md) §2.4；
- 身份重合污染对照（identity 格）必须逐格剔除或以污染对照读数显式呈现，历史教训（119.65 表观差距完全来自同 checkpoint 对角格）登记于 [`status/EVIDENCE_LEDGER.md`](status/EVIDENCE_LEDGER.md)；
- 主读数 XP 按 Official 评估协议：每配对 500 回合、400 步/回合、双角色评估、布局 `test_time_simple` / `test_time_wide`（常量出处 `src/path_c/experiment.py`，定义以 [`FORMAL_EXPERIMENT_PROTOCOL.md`](FORMAL_EXPERIMENT_PROTOCOL.md) 为准）。

## 二、B0–B3 统一开发矩阵要点

矩阵语义、统一条件与取舍规则以 [`research/DEVELOPMENT_MATRIX.md`](research/DEVELOPMENT_MATRIX.md) 为**唯一权威口径**，本节只列要点：

1. **矩阵定义**：B0（full-history recurrent PPO 强制核心基线）→ B1（+ 显式 protocol encoder）→ B2（+ 合法 all-action value supervision）→ B3（+ action-conditioned VOI）；B_k 严格包含 B_{k−1}，不得跳级比较。组件映射见 [`SCIENTIFIC_SPEC.md`](SCIENTIFIC_SPEC.md) §7 与 [`METHOD_SPEC.md`](METHOD_SPEC.md) §8。
2. **统一条件**（四方法共用，不得单独放宽）：相同训练伙伴分布、相同总 simulator transitions、相同 actor/critic 容量等级、相同 run-disjoint 测试伙伴、3–5 个开发 seed。
3. **报告读数**：XP、history-shuffle drop、protocol-swap 因果效应、total compute（第四节全口径）。
4. **取舍规则**：组件取舍由配对增量（同 seed、同 episode keys、同测试伙伴的 B_k − B_{k−1}）决定；**不设制度性停机门槛**；取舍记录只追加进 [`status/EVIDENCE_LEDGER.md`](status/EVIDENCE_LEDGER.md)。
5. **治理冻结**：与本矩阵冲突的旧晋升/停机条款（含 EXPERIMENT_LADDER 相关条款）冻结，见 DEVELOPMENT_MATRIX 修订头与 [`research/TRACKS_AND_GOVERNANCE.md`](research/TRACKS_AND_GOVERNANCE.md) 修订段。

## 三、Baseline 集合与 SOTA 判决程序（路径 1 已裁定生效）

### 3.1 Baseline 集合要求

与路径 1 判决程序对齐（并同步登记于 [`research/PAPER_STANDARD.md`](research/PAPER_STANDARD.md) 修订段）：

| 基线 | 要求 | 依据 |
|---|---|---|
| **FCP、OP、SA** | **必训项**：在固定 Official commit 上重训/获取 run-level 节点，构成判决统计的基线节点（FCP 种群约 24 亿步/布局，为判决批前置最大成本项） | [`research/STATISTICAL_PREREGISTRATION.md`](research/STATISTICAL_PREREGISTRATION.md) §5 路径 1 执行要点 1–2 |
| **SP** | **必训项**：Official 配方协议内参照（现成 10 seed 可直接引用，按引用口径记账） | [`research/PAPER_STANDARD.md`](research/PAPER_STANDARD.md) §6 |
| **MEP、PLASTIC、PECAN、GOAT** | **视可得性降级为外部参考**：不参与判决式；复现成本与协议可比性逐一核对后另行注册 | [`research/STATISTICAL_PREREGISTRATION.md`](research/STATISTICAL_PREREGISTRATION.md) §2 路径 1 第 4 条（领域 SOTA 可比性核对） |

### 3.2 判决程序（唯一生效口径）

执行口径以 [`research/STATISTICAL_PREREGISTRATION.md`](research/STATISTICAL_PREREGISTRATION.md) **第五节（裁定记录与路径 1 执行要点）**为唯一权威，要点：

1. **Official commit 冻结**：基线节点与己方评估一律在 `OFFICIAL_SOURCE_COMMIT = 5ce1707cf31c1c115e6f6ba96db7bc9cc80a850e`（`OFFICIAL_PROTOCOL_VERSION = overcooked_v2_iclr2025_5ce1707_v1`）上执行，冻结后不得更换；
2. **run-level 节点**：统计单位为训练运行（run 级），checkpoint + 训练元数据按 run 粒度保存；
3. **同 episode keys**：己方与基线在同一组 episode keys 上评估（500 episodes/格），做配对或双样本层级推断；
4. **Table 2 仅 sanity check**：已发表数值（Simple 6±29、Wide 23±40）**降为外部 sanity check，不参与判决**；
5. **共同禁令**：禁止"己方 bootstrap 区间减基线点估计 = 差值置信区间"的任何变体；"显著超过 SOTA"措辞仅在配对层级推断支持下使用。

### 3.3 Recoverable value 四格评估

G1 legal-history / G2 shuffled-history / G3 state-only / G4 oracle-continuation 四格（同组测试伙伴与同组 episode keys）的定义与回收比例 ρ 以 [`SCIENTIFIC_SPEC.md`](SCIENTIFIC_SPEC.md) §5.2 为准；实现读数（shuffle 程序、swap 程序、leakage 审计）由 [`METHOD_SPEC.md`](METHOD_SPEC.md) §4 供给。

## 四、可证伪判据汇总（指向 SCIENTIFIC_SPEC §6）

正式主张只由 [`SCIENTIFIC_SPEC.md`](SCIENTIFIC_SPEC.md) §6.1 的判据产生；本节仅汇总索引，不重述判负读数细节：

| 编号 | 判据主题 | 评估承载（本规格对应节） |
|---|---|---|
| Φ1 | 协议机制无 XP 增益（B1−B0、B2−B1、B3−B2 配对增量全部 ≤ 0） | 第二节（B0–B3 矩阵）、第六节（δ_min） |
| Φ2 | 历史信息无因果作用（history-shuffle drop 不显著） | 第三节 3.3（四格 G1−G2） |
| Φ3 | 协议表征与价值排序脱钩（swap 因果一致率 ≤ 0.5） | 第三节 3.3 配套读数（METHOD_SPEC §4.3） |
| Φ4 | 适应藏在 task 通路（leakage 审计超标） | METHOD_SPEC §4.1 审计程序 |
| Φ5 | Bayes 语义不成立（校准协议不通过） | METHOD_SPEC §2.4 校准协议 |
| Φ6 | 正式判决失败（路径 1 配对层级推断差值区间跨零） | 第三节 3.2 |

## 五、资源全口径报告（六项）

每次矩阵跑动与判决批必须报告以下全量项目（定义与记账细则以 [`research/DEVELOPMENT_MATRIX.md`](research/DEVELOPMENT_MATRIX.md) 第四节为唯一权威）：

1. **total transitions**（按来源分项：主训练 / 监督标签采集 / 生成器 / 评估 rollout；含约 38.9% 额外训练交互，不得只报主训练步数）；
2. **upstream partner cost**（伙伴 checkpoint 上游训练成本，FCP 种群约 24 亿步/布局量级，按引用/自训分别记账）；
3. **GPU-hours**（按设备型号与跑动分段）；
4. **peak memory**（训练与评估峰值显存）；
5. **参数量**（actor、critic、encoder/heads 分项）；
6. **performance–compute frontier**（XP 对 total transitions 曲线，四方法同图）。

## 六、δ_min 与实质显著性

- **δ_min = 20.0**：一次正确交付的原始回报（`OFFICIAL_CORRECT_DELIVERY_REWARD`）；预注册定义与判决时"点估计 + 区间 + 是否超过 δ_min"三项并报的要求，以 [`research/STATISTICAL_PREREGISTRATION.md`](research/STATISTICAL_PREREGISTRATION.md) 第一节为唯一权威；
- δ_min 与统计程序一律**跑前**注册；事后更换须追加新条目入 [`status/EVIDENCE_LEDGER.md`](status/EVIDENCE_LEDGER.md)，不覆盖。

## 七、探索轨读数调和规则（登记处）

S1–S4 等探索轨产物一律标注 `scientific_readout_allowed: false`，不得直接用作论文有效性主张；与论文依赖的调和规则（治理侧见 [`research/TRACKS_AND_GOVERNANCE.md`](research/TRACKS_AND_GOVERNANCE.md) 修订段）：

1. 探索轨读数须经**正式重跑**（确认轨口径：同划分纪律、同 Official 评估协议、跑前注册）方可进入论文表图；
2. 无法重跑的既有读数，只能按**预注册程序复核**（复核程序跑前注册入台账）后方可有限引用，并显式标注复核口径与残余局限；
3. 两途皆未通过的读数只保留为台账条目与动因记录，不入论文。

## 八、一致性声明

- 伙伴划分与 [`SCIENTIFIC_SPEC.md`](SCIENTIFIC_SPEC.md) §3、[`METHOD_SPEC.md`](METHOD_SPEC.md) §7 逐项一致；矩阵与 [`research/DEVELOPMENT_MATRIX.md`](research/DEVELOPMENT_MATRIX.md) 完全一致，本文件不新增门控；
- 判决统计与 [`research/STATISTICAL_PREREGISTRATION.md`](research/STATISTICAL_PREREGISTRATION.md) 路径 1（用户裁定生效）一致；理论预测与对撞 estimand 的口径边界见 [`THEORY.md`](THEORY.md)；
- 本文件与既有文档冲突时以本文件与 SCIENTIFIC_SPEC 为准；冲突处置只追加登记入台账，不改写旧文。
