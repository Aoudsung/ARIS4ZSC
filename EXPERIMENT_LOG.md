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

### 2026-07-03 R2.1 — 差异化 PASS，但可容许判别条件仅限 bottleneck-导航（非 serving-推断）
- 脚本: `scripts/rc2b_partner_differentiation.py`（新，checkpoint-free FSM-ego 探针）+ `rc2b_substrate_certificate.py`（修 partner_set bug）· config: `ocv2_step4_asymm_role_v1.yaml`（partner_set=role_conditioned_v2）· GPU-free（JAX-CPU）
- 产物: `review_bundles/phase2_R2.1_certificate_20260703/{partner_differentiation_v2.json, substrate_cert_v2.json, *.log}`
- **差异化探针（8ep, max-opt 40）：VERDICT=PASS（2/2 因子）**——serving 轴 spread=1.0、bottleneck 轴 spread=1.0、可区分对 27/28。**伙伴库有真实行为多样性**（不同于 sec16 标准伙伴的塌缩）。
- **可容许性证书（5ep, max-opt 80，fsm/random/partner-only）揭示关键分层**：
  ```
  ingredient-near/far-yield : fsm1.0 rand1.0 ponly0.0  → random 也解 = 太易，技能无关
  server-left/right-claim   : fsm1.0 rand1.0 ponly1.0  → partner 独解 = ego 无关
  bottleneck-yield-term-yield: fsm1.0 rand0.0 ponly0.0 → 可容许判别 ✓（train）
  bottleneck-push-term-claim : fsm1.0 rand0.4 ponly0.0 → 可容许判别 ✓（train）
  heldout-handoff-alt-yield  : fsm1.0 rand0.0 ponly0.0 → 唯一被ADMIT的held-out ✓
  heldout-resource-server-claim: fsm1.0 rand1.0 ponly1.0 → partner独解 = 退化 ✗
  ```
- **深层发现（与 METHOD_LOCK sec11 一致）**：asymm 上"谁 serve"因子**行为可区分**，但两模式都不给可容许判别条件（ego-serve 太易 / partner-serve ego 无关）。**唯一可容许判别条件 = bottleneck 导航伙伴**，而 bottleneck 是 throughput-navigable 空间因子（导航技能，非因子推断）。当前 6 训练+2 held-out split 里**只有 1/2 held-out 可容许**（heldout-handoff-admit；heldout-resource 退化）。
- **伪影自检**：max-opt=40 探针曾误报 bottleneck/held-out compl=0（伪影，80 opt 下 fsm=1.0 全部完成）→ 已修正解读。ingredient-far-yield fsm=0.0 但 rand=1.0 = FSM-ego 确定性盲点（探针质量注记，非基底问题）。
- **sec18.4 分支命中**："Distinguishable on a factor subset only → Narrow: re-cut train/held-out along the distinguishable subset"（差异化真、但可容许判别子集需重划 split）。**非干净 PASS→E1；需用户裁决 split/layout**（fork 决策，Type-B）。R2.2（serving CE 是否真非零）将决定 asymm 能否测推断 vs 仅导航。

### 2026-07-03 R2.1b — 吞吐口径 4 布局横扫 + sanity（sec18.9）
- 脚本: `scripts/rc2b_throughput_certificate.py`（新）· config: `ocv2_step4_asymm_role_v1.yaml`（正扫）+ `_tmp_sanity_v2.yaml`（sanity, 已清）
- 产物: `review_bundles/phase2_R2.1_certificate_20260703/{throughput_certificate.json, throughput_sanity_cramped_debug.json, *.log}`
- **正扫结果（4 布局 × 8 伙伴 × 3 seeds × 5 ep × 3 policies = 1440 ep）：0 admit**——但数据反常：
  - `rand_tp > fsm_tp` 广泛出现（asymm/server-*: rand=9.07 vs fsm=1.0）——物理上 FSM 必赢或平手，倒挂说明 FSM 在**争抢终端**
  - forced_coord/coord_ring 几乎全 0 → sec18.9.4 判读表兜底命中："probe budget too small → escalate, do NOT flip verdict"（先怀疑参数）
- **Sanity（cramped_room × v2 × debug executor: `force_path_planning=false`, `max_option_steps=6`, 2 seeds × 3 ep）：** 8 伙伴 **fsm_tp 全 0**，rand_tp 0.17–1.00，ponly_tp 全 0。**这排除了 executor 参数假说**。
- **核心诊断（重要发现）**：R2.1 探针的"FSM 是称职 ego"假设**不适配 v2**。FSM 是刚性 inventory-based pipeline，对 partner role/位置无感；v2 partners 是 role-aware（yield/claim/handoff），需要 role-aware ego 才能协调。FSM 与 partner 争抢终端 → 拖累吞吐 → 探针拒绝任何 substrate。**这是探针失败，非 substrate 失败**。
- **同时收获的正面信号**：cramped × v2 上 **ponly_tp 全 0 across all partners**（v2 partners 不能 solo，与 asymm 完全不同），但 rand ego 存在时能促成送餐——说明 cramped × v2 上 **ego 是必要的**，只是 FSM 太笨。这个信号 asymm 上没有（asymm/server-*: ponly=1.0 solo）。
- **R2.1 探针框架结论**：sec18.4/18.9 的"FSM-oracle admissibility"路径**无法通过 v2**——不因 substrate，因探针方法论。差异化 PASS 是稳固信号（v2 有真实行为多样性），但 admissibility 需换探针（role-aware oracle）或换判据（训练本身作为 admissibility test，即 E1 本身）。

## Phase 3 — 决定性实验（E1 四臂去 oracle 重跑 · E2 通道消融 · E3 持久化消融）

*(待运行)*

## Phase 4 — 主张级实验（E4–E7）

*(待运行)*

## Phase 5 — 锁定 → 盲测 → 终表

*(待运行)*
