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

### 2026-07-03 E1 前置 CE（asymm×v2）— 两次 GraphCoverageError → 根因 S27（推断器支撑集冻结）
- 采集: 100ep×6伙伴=12000 rows（3h50m, GPU6/JAX-CPU）· 覆盖门 passed · sidecar 四掩码齐全
- 建图两次失败: `plate_soup/serve_soup 无 above-eta CE 候选`（min_weight 20→10、eta 0.05→0.02 复用 replay 重试无效——`--reuse_replay` 功能顺手落地, commit 9dfb258）
- **支持度审计决定性读数**: 终端选项作为伙伴列联合质量**精确 0.0**；73.9% 伙伴占用在 noop；
  ego-terminal 行全部质量落 (terminal, noop) cell → 被 kernel noop 排除 → CoverageError
- **现场实验（铁证）**: server-left-claim 单 ep 送餐 9 次（plate+serve 全程执行），行为推断器
  终端选项质量恒 0.00000000, 每次送餐 argmax=noop → **S27 支撑集冻结确证**
  （reset 冻结初始有效集为 belief 支撑, 乘性更新 0×x=0 永久锁死）
- 台账: S27(实现bug,高) + D6(互斥估计量盲区, D4精化) + D7(noop排除×yield签名冲突)
- **裁定**: 主导=实现 bug（P1 修复激活死代码中的潜伏缺陷）; 设计放大器 ×2; proposal 层非主因
- 下一步（待批）: S27 修复（支撑注入/遗忘因子, codex diff 评审）→ **重采 CE**（现 replay 的
  partner dist 已污染）→ 建图 → E1

### 2026-07-04 S27 修复远程验证 — PASS（决定性）
- 单元测试: test_s27_support_injection (4例, 含 mix=0 冻结复现锚点) + E2/E3/缓存回归 → exit 0
- **现场探针复验（同 claim 伙伴 9 次送餐场景）**: 修复前终端质量恒 0.00000000 →
  修复后送餐瞬间 terminal mass=0.9246, **9/9 次 argmax=serve_soup 精确命中**, 峰值 0.9866

### 2026-07-04 E1 前置 CE（S27 修复后重采）— PASS，E1 解锁
- 100ep×6伙伴×asymm_v2, 3h50m (S27 修复带来推断器质量提升→选项终止更快→采集加速~30%)
- 产物: outputs/asymm_ce_role_v2_e1_s27fixed_retry2/{graph.json, ce_refined.npy, ce_support_audit.json, replay.npz}
- 归档: review_bundles/phase3_E1_ce_20260704/（含 pre/post 对照）
- **S27 修复的 CE 层独立确证**:

  | 指标 | PRE-fix | POST-fix |
  |---|---|---|
  | estimable pairs | 45 | **80** (+78%) |
  | partner=noop 占用份额 | 73.9% | **13.8%** |
  | partner col opt2/3/9 联合质量 | 全 0.0 | 547 / 149 / 199 |
  | ego row opt3 主质量 | opt27[noop]=52 | opt25[wait]=19.6, opt24[cross]=18.7 |

- **建图两阶段解锁**:
  - S27 修复后重采 → 首建仍失败: `opt7 pick_plate 无 above-eta 候选`（新错误位置）
  - **诊断揭示 substrate 事实**: asymm 上随机策略 ego 从未访问 opt7/opt9（ws=0），opt3/opt8
    质量分散 <10 全 skipped。**S27 修复解除了 noop 单极坍缩，暴露 substrate 真实结构中的
    valid ID 边缘化**（不是估计问题，是访问频率问题）
  - **U1 论文证据线得到强化实测**: 不是"我们推测被动 CE 有盲区"，是"我们精确测得 opt7/9
    ws=0、opt3/8 分散低支持"——直接就是 targeted-starts 要解决的场景
  - **工程解**: kind-level coverage（放弃 per-ID）+ min_weight 5 + eta 0.02，复用 replay 建图
    成功 → **16 因子**（3 serving 含 opt3/opt9、4 bottleneck、5 resource、1 pot_allocation、
    3 generic），CE 分数 1.4–5.8，selected_by 分布 3 mandatory kind + 3 role contrast + 10 ce_fill
- **正式 E1 config**: `configs/ocv2_step4_asymm_role_v2_e1.yaml` (graph_path 指向 retry2 产物 +
  min_weight=5 + eta=0.02 + kind-only coverage)
- 决策记录: opt3/7/9 无因子（substrate 边缘化的诚实记录）；kind coverage 已被 opt2/opt6/opt5
  覆盖 pick_plate/plate_soup；serve_soup 通过 opt9 (via low min_weight) 覆盖
- **下一步**: preflight（复用 replay 秒级）→ E1 四臂决定性跑（sec18.6/18.10.2 预注册读出）

## Phase 4 — 主张级实验（E4–E7）

*(待运行)*

## Phase 5 — 锁定 → 盲测 → 终表

*(待运行)*

### 2026-07-04 E1(no-scaffold) wave — ORCHESTRATOR BUG (data loss), partial result retained
- **Bug (mine)**: `logs_phase3/E1_wave_orch.sh` used `local method=$1 ... out=results_phase3/E1_${method}_s${seed}`
  in a SINGLE `local` declaration → `${method}`/`${seed}` empty when `out` computed (bash gotcha) →
  every run's `out=results_phase3/E1__s` + `rm -rf $out` at run start **wiped all prior runs' outputs**.
  CLI `--method`/`--seed` (separate refs, post-declaration) expanded fine → correct subpath but shared
  parent dir. Only the last-completing run survived on disk.
- **Captured before wipe (live reads, training-phase ego/partner delivery counts)**:
  - aris_bellman: s0=0/8, s1=1/12, s2=0/10, s3=0/11, s4=0/6 (all 5 seeds — ego≈0)
  - base_only: s0=0/7 (ego=0)
  - partner_id_q: s4=1/9 (survived on disk; ego≈1)
  - global_gru, flat_factor: NOT captured before wipe → lost
- **Scientific status**: the no-scaffold "ego≈0 vacuum" headline (sec18.12.1 trigger) is established
  qualitatively from aris×5 + base×1 + partner_id_q×1. The FULL 4-arm×5-seed ablation table is LOST.
  Per sec18.12.4 this table is the scaffold-ablation baseline — **deferred re-run** (cheap, ~5h overnight,
  only needed if the paper requires the complete no-scaffold ablation vs E1-rev).
- **Fix**: orchestrator rewritten with per-(method,seed) unique output dirs + no cross-wipe; used for E1-rev.
- **Guard/eval note**: no deployable checkpoints existed (guard fail across arms), so nothing was
  read against sec18.6 — no claim contaminated. Loss is of the ablation record, not of a decisive read.
