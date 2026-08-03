# SOTA 基线注册专页

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
