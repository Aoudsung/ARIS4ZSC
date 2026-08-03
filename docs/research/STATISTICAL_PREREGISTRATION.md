# 统计预注册（SOTA 判决式）

修订头：本文件于统一重构（Rigor × Generality 统一优化方案 F 节）中创建，承接第一轮审查的 P0 缺口（δ_min 与统计程序跑前预注册）。

修订头（追加，2026-08-03）：用户裁定生效——路径 1 标记为**已裁定生效**，路径 2 标记为**备用（仅在路径 1 资源不可行并经新裁定后启用）**；裁定记录见本文件第五节，决策登记见 [DECISION_LOG](../status/DECISION_LOG.md) D2。原"pending 用户裁定"状态行按只追加纪律保留不动，以本节与第五节为准。

**状态：pending 用户裁定。** 下文两套判决程序并行注册，待用户裁定后二选一生效；未生效前任何判决式读数不得以"已预注册程序"名义引用。

## 一、δ_min 预注册

- **δ_min = 20.0**：一次正确交付的原始回报，取自 `src/path_c/experiment.py` 常量 `OFFICIAL_CORRECT_DELIVERY_REWARD = 20.0`（另见 `OFFICIAL_EPISODES_PER_PAIRING = 500`、`OFFICIAL_EPISODE_STEPS = 400`、`LAYOUTS = ("test_time_simple", "test_time_wide")`）。
- 该阈值在 S5 正式矩阵生成前注册，定义以 [FORMAL_EXPERIMENT_PROTOCOL](../FORMAL_EXPERIMENT_PROTOCOL.md) 为准；判决时须同时报告点估计、区间与"是否超过 δ_min"三项，不得只报其一。

## 二、判决程序（两套并行注册，待裁定二选一生效）

### 路径 1：run-level 节点配对/层级推断（评审建议路径）

1. 在**固定 Official commit**（`OFFICIAL_SOURCE_COMMIT = 5ce1707cf31c1c115e6f6ba96db7bc9cc80a850e`，`OFFICIAL_PROTOCOL_VERSION = overcooked_v2_iclr2025_5ce1707_v1`）上重训/获取 FCP、OP、SA 的 run-level 节点（FCP 种群约 24 亿步/布局，为最大成本项，按 [DEVELOPMENT_MATRIX](DEVELOPMENT_MATRIX.md) 第四节全口径记账）；
2. 己方与基线在**同 episode keys** 上评估，做配对或双样本层级推断（统计单位为训练运行，run 级）；
3. 已发表 Table 2 数值（Simple 6±29、Wide 23±40，见 [SOTA_BASELINE](SOTA_BASELINE.md)）**降为外部 sanity check**，不参与判决；
4. 同时区分"同 benchmark reference"与"领域 SOTA"：CooT / TALENTS / GOAT / ROTATE / UPD / ICRL4AHT 等 2025–26 工作逐一核对协议可比性后另行注册。

### 路径 2：维持发表标量对标（承接 2026-08-03 裁决）

1. 继续只对标已发表标量（Simple 6±29、Wide 23±40）；
2. **判决措辞强制降级**为："超过已发表点估计（基线训练不确定性未计入，口径如实披露）"，不得使用"显著超过 SOTA"；
3. 补报 **Welch 口径保守边际**（以发表均值±标准差为己方 run 分布参照的保守检验，预期边际量级 Simple +22 / Wide +31），作为敏感性披露；
4. 论文与台账按此口径如实披露基线训练不确定性缺失。

## 三、两路径共同禁令

- **禁止**"己方 bootstrap 区间减基线点估计 = 差值置信区间"的任何变体：6±29 / 23±40 是已发表均值±标准差、无原始 run node，不能做配对 run-node bootstrap 差值区间，也不能把单边差值包装成置信区间。
- δ_min 与本节统计程序一律**跑前**注册；事后更换程序须追加新条目入[证据台账](../status/EVIDENCE_LEDGER.md)，不覆盖。

## 四、生效机制

用户裁定后，本文件追加"裁定记录"条目（路径选择、日期、依据），未选路径标注为 `rejected` 并保留全文；被选路径成为 [SOTA_BASELINE](SOTA_BASELINE.md) 第三节判决式的执行口径。

## 五、裁定记录（2026-08-03，用户裁定生效）

- **裁定日期**：2026-08-03；登记于 [DECISION_LOG](../status/DECISION_LOG.md) D2（用户对重构计划 F/C 节的科学裁定）。
- **路径 1：已裁定生效（`effective`）。**
- **路径 2：备用（`standby`）——仅在路径 1 资源不可行并经新裁定后启用**；全文按第四节生效机制保留不动，不按 `rejected` 注销。

### 路径 1 执行要点（生效口径）

1. **Official commit 冻结**：FCP/OP/SA 基线节点与己方评估一律在固定 Official commit（`OFFICIAL_SOURCE_COMMIT = 5ce1707cf31c1c115e6f6ba96db7bc9cc80a850e`，`OFFICIAL_PROTOCOL_VERSION = overcooked_v2_iclr2025_5ce1707_v1`）上执行；冻结后不得换 commit，如需更换须追加新裁定入台账，不覆盖。
2. **run-level 节点保存**：FCP 种群、OP、SA 的训练产物按 run 粒度保存节点（checkpoint + 训练元数据，统计单位为训练运行），FCP 种群重训（约 24 亿步/布局）为判决批前置成本项，按 [DEVELOPMENT_MATRIX](DEVELOPMENT_MATRIX.md) 第四节全口径记账、跑前登记预算。
3. **同 episode keys**：己方与基线节点在同一组 episode keys 上评估（对齐 500 episodes/格口径），保证配对可比性。
4. **配对/双样本层级推断**：在 run 级节点上做配对或双样本层级推断；"显著超过 SOTA"措辞仅在该配对推断支持下使用。
5. **Table 2 仅作 sanity check**：已发表数值（Simple 6±29、Wide 23±40）降为外部 sanity check，不参与判决；第三节共同禁令（禁止用发表均值±标准差构造差值区间）继续适用。

本裁定生效后，[SOTA_BASELINE](SOTA_BASELINE.md) 第三节判决式按路径 1 口径执行，待基线 run 节点就绪后按本预注册程序判决。
