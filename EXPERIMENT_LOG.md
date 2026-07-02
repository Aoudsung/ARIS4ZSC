# EXPERIMENT_LOG.md — 正式实验结果记录（ARIS 惯例）

**Created:** 2026-07-02（Phase 0 · G0.5）· **Status:** 骨架，无正式结果
**规则**：只记录**执行门控下的正式运行**（preflight 接受 + 完整性硬门全 true + git commit 哈希 +
远程产物路径）。诊断/冒烟跑不入此文件。每条结果的解读**必须引用** METHOD_LOCK sec18 预注册分支表，
禁止事后重解释。NEW-4 隔离材料（CODEX_IMPL_SPEC v1–v4 数字）永不入此文件。

## 记录模板

```
### <日期> <实验ID per EXPERIMENT_CHAIN_PLAN> — <一句话结果>
- commit: <hash> · config: <path> · 远程产物: <path>
- 完整性: reward_scale_verified=<> evidence_policy=<> oracle_source_count=<> gate=I1-I17 <>
- 规模: seeds=<> episodes/partner=<> partners=<train/heldout 列表>
- 主指标: <ego_correct_completion_rate / throughput ± CI>
- 预注册分支命中: <sec18.x 表格行>
- 伪影自检: <5项清单逐项 / ARTIFACT-SUSPECT 标记>
- Type-B: codex verdict=<> · 人类裁决=<> · 日期=<>
```

## Phase 1 — 基础设施验证（R1.1–R1.3）

### 2026-07-03 Phase 1 — 全部通过（Type-A，基础设施验证，非主张实验）
- commit: `4d680ab` · 远程: `Selfs/ARIS4ZSC-phase1-4d680ab`（git archive 全新同步 + SYNC_PROVENANCE）· GPU 5（避开 ECC 故障的 0/3）· JAX_PLATFORMS=cpu
- **R1.1 ✅**：fidelity gate 远程 exit 0（I1–I17）；pytest 全套件 exit 0（10 文件，~93 项）
- **R1.2 ✅**（cramped_room debug 冒烟，CE 20ep/伙伴 → preflight accepted → train 400 updates → eval 8ep×2 伙伴）：
  - 完整性: `evidence_policy=behavior_inferred_v1` · `oracle_source_count=0` · `observed_dist=0` · `missing=0` · **`reward_scale_verified=true`（曾永假的死旗首次真验证通过，S18 闭环）** · `headline_success_metric=ego_correct_completion_rate` · `allow_diag_skip=false` · `role_match_status=removed_from_formal_main_path_p5`
  - P3 sidecar 四掩码齐全（estimable/skipped/measured_zero/weight_sum + 计数 + 参数）
  - **覆盖门两次正确 fail-closed**（8ep 与 30ep 随机策略下终端选项支持不足 → 拒绝建图，A1 机制真实工作）；冒烟经 `--no_require_task_stage_coverage` + `graph.require_task_stage_coverage=false`（ce_path 分支）放行 —— **正式跑禁止此路径**（见 CHAIN_PLAN §10 执行卡）
  - completion=0.0 为预期（400-update 冒烟 + 覆盖不全图；task_progress_events=84）；非本次验收目标
  - 远程 config 补丁（仅冒烟、未回写本地）: debug.yaml 加 `graph.replay_path`（preflight 复用 replay）、`diagnostics.proxy_episodes=8`、`ce_path=ce_refined.npy`、去 `graph_path`、覆盖门关
- **R1.3 ✅**（NEW-2 历史产物审计）：扫描全部远程目录 155 个 metrics.json → **3 个命中**（guard=fail 且 checkpoint.pt 在盘）：`CPR_REPO/results_rcfix/{asymm_egocredit,armA,armB}/.../seed0`（6/29 RC 修复时代，当时即记录为失败、从未作正面引用 → **无结论污染**）。已就地写 `QUARANTINE_NEW2_guard_fail.txt` 标记（不删数据）
- 实测单位耗时（可行性分析依据，CHAIN_PLAN §9）: CE 采集 ≈7.4s/ep/伙伴（cramped 顺序）· train 启动 ≈2.5min + ≈0.32s/update · eval ≈1304s/(8ep×2伙伴含参照基线) · preflight 复用 replay 后秒级
- 伪影自检: 不适用（无主张结论）· Type-B: 不适用（Type-A 基础设施验证，按 OPERATING_CONSTRAINTS §4 自判可过，留档）

## Phase 2 — 基底证书（R2.1 伙伴可区分性 · R2.2 asymm CE 支持度）

*(待运行)*

## Phase 3 — 决定性实验（E1 四臂去 oracle 重跑 · E2 通道消融 · E3 持久化消融）

*(待运行)*

## Phase 4 — 主张级实验（E4–E7）

*(待运行)*

## Phase 5 — 锁定 → 盲测 → 终表

*(待运行)*
