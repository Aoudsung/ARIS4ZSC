# Experiment Audit Report

**Date**: 2026-07-30

**Auditor**: GPT-5.6-sol xhigh（独立只读 reviewer；首选 GPT-5.5 路由在当前环境不可用）

**Project**: Path C V4.4 artifact-only real-return readout

## Overall Verdict: PASS

## Integrity Status: pass

独立 reviewer 未发现 fake ground truth、自归一化、泄漏、破坏配对、伙伴不等权、phantom
result 或登记后改写裁决。正式运行、独立复跑与本地只读副本中的 9 个确定性产物逐字节一致；
`INCONCLUSIVE` 是对预先登记裁决顺序的唯一正确应用。

## Checks

### A. Ground Truth Provenance: PASS

主拟合与评价目标来自冻结 simulator continuation 的原始剩余回报。每个触发状态以 64 个
fit replica 构造六动作价值，并以另外 64 个 evaluation replica 评价已选择动作；模型 Q
不作为 ground truth。证据：`experiments/overcooked_v2/artifact_readout_app.py:610`。

### B. Score Normalization: PASS

只有外层训练伙伴内的特征标准化和目标中心化。最终 lift 保留 simulator raw-return 单位，
未除以模型自身最大值、均值、方差或预测尺度。证据：
`experiments/overcooked_v2/artifact_readout_app.py:747`。

### C. Result File Existence and Reproducibility: PASS

`configuration.json`、`feature_schema.json`、`run_metadata.json`、`split_manifest.parquet`、
`state_readout.parquet`、`bootstrap.parquet`、`summary.json`、`report.md` 与
`output_sha256.json` 均存在。正式远端、独立复跑远端和本地副本的这 9 个文件逐字节一致；
R016 已登记为完成且裁决为 `INCONCLUSIVE`。证据：
`idea-stage/refine-logs/EXPERIMENT_TRACKER.md:23`、
`artifacts/path_c_v44_artifact_readout_v2_20260729/output_sha256.json`。

### D. Dead Code and Pipeline Reachability: PASS

`audit-artifact-readout` 已接入唯一 CLI；端到端 fixture 覆盖命令执行、产物、输入不变性和
确定性复跑。远端 CPU-only 全量测试 78 项全部通过，其中新增读出测试 9 项。证据：
`experiments/overcooked_v2/path_c.py:97`、
`experiments/overcooked_v2/tests/test_path_c_artifact_readout.py:407`。

### E. Scope Assessment: PASS

机器摘要和正式报告都把结论限定为当前四个固定伙伴、当前 178 个固定伙伴触发状态、当前
response code 和六动作候选集合；没有写成伙伴总体、跨训练运行总体或现实世界结论。证据：
`experiments/overcooked_v2/artifact_readout_app.py:1576`、
`docs/status/PATH_C_V4_4_ARTIFACT_VALUE_READOUT_REPORT.md:44`。

### F. Evaluation Type: simulation_only

“真实回报”指环境延续实际产生的 simulator raw return，不是模型 Q、数据集提供的现实
ground truth 或现实世界回报。证据：
`docs/status/PATH_C_V4_4_ARTIFACT_VALUE_READOUT_REPORT.md:9`。

## Additional Integrity Checks

- `split_manifest.parquet` 有 28,800 行；225 个 trigger 各为 64 fit + 64 evaluation，
  `(trigger_id, replica_index)` 无重复。
- 输入 lattice 为 `225 × 128 × (2 + 6×2) = 403,200` 行。
- `state_readout.parquet` 有 806 行；`bootstrap.parquet` 有 20,000 行，state/history 各
  10,000 个唯一 bootstrap 编号。
- 五个冻结输入与四个实现文件的运行前后 SHA-256 完全相同。
- 独立复算的主历史结果为 `L_probe = −0.359685 [−0.932576,+0.085187]`、
  `L_oracle = −0.013508 [−0.164903,+0.139880]`、
  `tau_response = +0.049892 [−0.179547,+0.271533]`，与机器摘要一致。
- Oracle UCB 大于 0，Oracle 与 probe 的 LCB 均不大于 0，且无登记的伙伴方向冲突；
  因此 GO 与两类 NO-GO 均不成立，剩余分支只能是 `INCONCLUSIVE`。

## Action Items

没有阻断项。对外引用机器产物中的简版 `report.md` 时，应同时链接完整状态报告，以保留
“区间条件于固定四伙伴面板和固定 continuation 样本”的限定。若未来改变这两个抽样层，
应另立审计，而不是扩大本次区间的解释范围。

## Claim Impact

- 当前注册回应通道与候选分布的裁决为 `INCONCLUSIVE`：**supported**。
- 当前不启动 V4.5 或新 RL：**supported as a project gate decision**。
- 结论只适用于当前四伙伴面板和固定 continuation 样本：**required and correctly stated**。
- `119.65` 完全来自同 checkpoint 对角格：**unchanged external prior finding; not re-estimated here**。

本报告记录 reviewer 的判断。由于首选 GPT-5.5 backend 未在本会话提供，实际使用了独立的
GPT-5.6-sol xhigh 只读 reviewer；这一替代不应被描述为 GPT-5.5 cross-model 审计。
