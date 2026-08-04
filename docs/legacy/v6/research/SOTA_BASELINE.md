# SOTA 基线注册专页（历史归档）

状态：探索轨产物，`scientific_readout_allowed: false`（注册数值本身为外部已发表事实）。
职责：注册判决 Θ3（方法载体论点）所用的 SOTA 基线数值、布局映射与判决式。本页数值一经注册不得事后更换；如需更正，追加新条目入[证据台账](../status/EVIDENCE_LEDGER.md)，不覆盖。

## 一、布局映射

| 本仓库代码布局 | 论文对应布局 |
|---|---|
| `test_time_simple` | Test-Time Protocol Formation Simple |
| `test_time_wide` | Test-Time Protocol Formation Wide |

代码中的布局常量见 `src/path_c/experiment.py` 的 `LAYOUTS = ("test_time_simple", "test_time_wide")`。S5 正式矩阵（[FORMAL_EXPERIMENT_PROTOCOL](../FORMAL_EXPERIMENT_PROTOCOL.md)）使用同一对布局，10 seed、每格 500 回合，与基准论文的评估口径对齐。

## 二、注册 SOTA 数值与出处

来源：基准论文 Gessler et al., "OvercookedV2: Rethinking Overcooked for Zero-Shot Coordination", arXiv:2503.17821，Table 2（10 seeds、每格 500 episodes）。

| 布局 | 已发表 SOTA 最优方法 | 注册数值（XP） |
|---|---|---|
| Test Time Simple（`test_time_simple`） | FCP | 6±29 |
| Test Time Wide（`test_time_wide`） | FCP | 23±40 |

按 2026-08-03 人类裁决：SOTA 对比仅以上述已发表数值为准，不要求先自训复现基线 checkpoint。

## 三、判决式

Θ3 成立（在 [EXPERIMENT_LADDER](EXPERIMENT_LADDER.md) S5 正式冻结矩阵上判决）当且仅当：

1. DELTA-ZSC-E2E 的 `J_XP` 显著高于上表注册的 SOTA 基线数值：run-node bootstrap（9,999 次，保留行列节点依赖与配对 episode-key 重采样）的差值区间不跨零；
2. `δ_min` 已预注册（以一次正确交付的原始回报为尺度，S5 生成前注册，定义以 [FORMAL_EXPERIMENT_PROTOCOL](../FORMAL_EXPERIMENT_PROTOCOL.md) 为准）；
3. 共同伙伴记分板（run-disjoint 外部伙伴）排除内部惯例互利的解释；
4. 统计单位为训练运行（run 级），不以回合级推断替代。

以上四项缺一，Θ3 不判成立；但判不成立不等于停止目标——方法轨可换载体继续推进超越 SOTA 的核心目标。

## 四、口径偏移风险披露

上述注册数值是基准论文作者自训 checkpoint 的已发表数值，与本仓库自训基线可能存在口径偏移（训练实现细节、评估环境版本、随机数口径等），风险如实披露。按 2026-08-03 人类裁决，该风险不作为立项门槛，不阻塞方法轨推进；但在论文与台账中必须披露此口径差异，审稿防御按 [PAPER_STANDARD](PAPER_STANDARD.md) 执行。若后续预算允许，可补做自训 FCP 复现作为敏感性检查，结果只追加不覆盖本注册。

## 五、与其他文档的关系

- 晋升门要求"SOTA 基线数值已注册"即指本页（[TRACKS_AND_GOVERNANCE](TRACKS_AND_GOVERNANCE.md)）。
- S5 进入条件与判决见 [EXPERIMENT_LADDER](EXPERIMENT_LADDER.md) S5。
- 基线注册表的简表见 [PAPER_STANDARD](PAPER_STANDARD.md) 第六节。

## 六、判决式统计问题与预注册指针（追加段，待裁定生效）

统一重构评审（F 节）判定第三节现行判决式在统计上不可执行：注册数值 6±29 / 23±40 是已发表均值±标准差、无原始 run node，不能做配对 run-node bootstrap，也不能把己方 bootstrap 区间减基线点估计称为“差值置信区间”。两套合规判决程序已并行预注册于 [STATISTICAL_PREREGISTRATION](STATISTICAL_PREREGISTRATION.md)：路径 1（固定 Official commit 重训/获取 FCP/OP/SA run-level 节点，同 episode keys 配对/双样本层级推断，Table 2 降为外部 sanity check）与路径 2（维持发表标量对标，判决措辞降级为“超过已发表点估计（基线训练不确定性未计入，口径如实披露）”，补 Welch 口径保守边际报告）。δ_min = 20.0（`src/path_c/experiment.py` 常量 `OFFICIAL_CORRECT_DELIVERY_REWARD = 20.0`）同步预注册。状态：pending 用户裁定，裁定后由预注册文件登记生效路径，本段与第三节判决式按生效路径执行；在此之前本段只追加，上文不覆盖。

## 七、裁定生效说明（2026-08-03 追加）

用户裁定（登记于 [DECISION_LOG](../status/DECISION_LOG.md) D2）：SOTA 判决采用**路径 1**——在固定 Official commit 上重训/获取 FCP/OP/SA 的 run-level 节点，同 episode keys 配对/双样本层级推断；已发表 Table 2（Simple 6±29、Wide 23±40）降为外部 sanity check；“显著超过 SOTA”措辞仅在配对推断支持下使用。路径 2 降为备用（仅在路径 1 资源不可行并经新裁定后启用）。生效口径与执行要点见 [STATISTICAL_PREREGISTRATION](STATISTICAL_PREREGISTRATION.md) 第五节。本判决文件待基线 run 节点就绪后按预注册程序执行；在此之前本段只追加，上文不覆盖。
