# 统一开发矩阵 B0–B3（权威开发矩阵规格）

修订头：本文件于统一重构（Rigor × Generality 统一优化方案 E 节）中创建，以统一开发矩阵取代旧门槛堆叠。本规格是 B0–B3 开发的唯一权威口径；与其冲突的旧晋升/停机条件以本文件为准（见 [EXPERIMENT_LADDER](EXPERIMENT_LADDER.md) 末尾修订段）。

状态：`scientific_readout_allowed: false`（探索轨产物；矩阵读数本身不构成正式主张）。

## 一、矩阵定义

| 方法 | 含义 | 对应增益问题 |
|---|---|---|
| **B0** | Full-history recurrent PPO——强制核心基线 | NeurIPS 2025 已证明普通 RNN 可涌现伙伴表征，必须正面对比：任何显式机制不得劣于纯 recurrence |
| **B1** | B0 + 显式 protocol encoder | 架构增益：显式协议通路相对纯 recurrence 的配对增量 |
| **B2** | B1 + 合法 all-action value supervision | decision-equivalent 监督增益：全部监督来自真实合法历史与共同随机数 all-action continuation 的 centered advantage 签名 |
| **B3** | B2 + action-conditioned VOI(a)（信息增益 − 任务代价） | 主动协议形成增益：从被动读取历史升级为以 ego 动作选择主动获取伙伴信息 |

矩阵语义：B_k 严格包含 B_{k−1} 的全部组件；任何方法不得跳级比较（B2 vs B0 之类的跨级差值不作为取舍依据）。

## 二、统一条件（四方法共用，不得单独放宽）

1. **相同训练伙伴分布**：同一伙伴池与采样分布；伙伴生成/筛选流程在矩阵跑动前冻结。
2. **相同总 simulator transitions**：以环境步总量计公平预算，任何方法的额外训练交互（含监督标签采集）计入 total transitions 并如实披露。
3. **相同 actor/critic 容量等级**：参数量级与结构等级对齐；容量差异作为报告项而非隐藏变量。
4. **相同 run-disjoint 测试伙伴**：测试伙伴与所有训练运行 disjoint，跨方法共用同一测试集与 episode keys。
5. **3–5 个开发 seed**：矩阵批至少 3 个 seed，资源允许取 5；seed 集合跑前注册。

## 三、报告读数（每方法每 seed 全量记录）

- **XP**（zero-shot coordination 主读数，与 S5 同布局同口径）；
- **history-shuffle drop**：保持任务状态与当前观测不变、打乱伙伴历史后的 XP 退化量（核心报告读数，退化显著才支持"适应"主张）；
- **protocol-swap 因果效应**：匹配任务状态间交换协议表征，动作变化方向与真实 continuation-value 排序的一致率；
- **total compute**：见第四节全口径算力报告。

## 四、全口径算力报告（评审 §10.6 整改：当前口径系统性弱化 DELTA 约 38.9% 额外训练成本）

每次矩阵跑动必须报告以下全量项目，不得只报主训练步数：

1. **total transitions**：全部 simulator 交互如实计入——包括 DELTA 监督/校准相关的约 38.9% 额外训练交互，按来源分项列示（主训练 / 监督标签采集 / 生成器 / 评估 rollout）；
2. **upstream partner cost**：伙伴 checkpoint 的上游训练成本（FCP 种群约 24 亿步/布局量级）按引用或自训口径分别记账；
3. **GPU-hours**：按设备型号与跑动分段记录；
4. **peak memory**：训练与评估峰值显存；
5. **参数量**：actor、critic、encoder/heads 分项；
6. **performance–compute frontier**：XP 对 total transitions 的曲线，四方法同图呈现。

## 五、取舍规则与治理

- **组件取舍由配对增量决定**：B_k − B_{k−1} 的配对增量（同 seed、同 episode keys、同测试伙伴）为正且稳健，组件保留；为零或负，组件在探索轨标记为拖累嫌疑。
- **不设制度性停机门槛**：任何一级读数为负不触发计划停机；与"超越 SOTA 唯一目标、最少门控"的裁决一致。取舍记录只追加进[证据台账](../status/EVIDENCE_LEDGER.md)。
- **预算锚定**：B0 开发预算以 `src/path_c/experiment.py` 既有常量为准——SP 正式预算 `OFFICIAL_SP_TOTAL_TIMESTEPS = 30_000_000`，OP 正式预算 `OFFICIAL_OP_TOTAL_TIMESTEPS = 50_000_000`；正式跑动单跑步数 `RUN_BUDGETS["formal"].environment_steps = 29_949_952`（256 envs、64 minibatches/epoch，与 Official commit 对齐）。开发矩阵按该锚等比缩放，缩放比例跑前注册。
- **δ_min 关联**：XP 增量的实质显著性阈值 δ_min = 20.0（`OFFICIAL_CORRECT_DELIVERY_REWARD`，一次正确交付的原始回报），预注册细则见 [STATISTICAL_PREREGISTRATION](STATISTICAL_PREREGISTRATION.md)。

## 六、与既有文档的关系

- 本文件取代 [EXPERIMENT_LADDER](EXPERIMENT_LADDER.md) S4"探索性开发矩阵"的消融设计部分（S4 其余内容保留为历史条目，只追加不改写）。
- SOTA 判决式与统计程序不在本文件范围，见 [SOTA_BASELINE](SOTA_BASELINE.md) 与 [STATISTICAL_PREREGISTRATION](STATISTICAL_PREREGISTRATION.md)。
- 矩阵批完成后回填 [PAPER_STANDARD](PAPER_STANDARD.md) 对应表图。
