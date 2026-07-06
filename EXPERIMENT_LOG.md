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

### 2026-07-04 E1-rev pilot (scaffolds ON) — DECISIVE: scaffolds unlock terminal competence
- config: `ocv2_step4_asymm_role_v2_e1rev.yaml` (contrib_team + ego-kind terminal_progress_shaping
  + terminal_exploration, both P5-audited ego-local) · graph: `outputs/asymm_ce_role_v2_e1rev/`
  (sparse-CE rebuild, reused replay) · commit b9dea6f→(config final)
- **objective gate resolution (2 iterations, gate working as designed — S11/T1)**:
  (1) shaping-on config vs shaping-off graph metadata rejected → rebuilt graph with
  `--sparse_ce_support` (excludes terminal bonus from CE support selection, G3/S11 path);
  (2) graph metadata `sparse_ce_support:true` vs config-unset rejected → set
  `graph.sparse_ce_support: true` in config. Both are the anti-stale-graph gate correctly
  enforcing CE-objective ↔ training-objective consistency. NOT failures.
- **RESULT (aris_bellman s0, 5000 updates, 1280s)**:

  | metric | E1(no-scaffold) s0 | **E1-rev(scaffold) s0** |
  |---|---|---|
  | ego_correct_delivery | 0 | **18** |
  | ego_sole_correct_delivery | 0 | **2** |
  | served_soup_count | 0 | **29** |
  | free_rider_guard | fail | **pass** |
  | deployable_checkpoint | None | **checkpoint.pt** |
  | best_greedy_return | None | **46.15** |

- **sec18.12.1 trigger CONFIRMED**: ego≈0 was the P5-clean-scaffold vacuum, not a method failure.
  This is the FIRST genuinely-usable training result in the project — under honest (de-oracled) +
  fair (oracle-free scaffolds) conditions the ego LEARNS to serve (18 correct, 2 ego-sole, guard pass).
- **Full E1-rev wave launched** (fixed orchestrator, 4 arms × 5 seeds, unique dirs). Read per
  sec18.6 + sec18.10.2 + sec18.12.3 once all 20 runs + partner_id_q reference land.

### 2026-07-04 E1-rev 全 wave 完成（25/25）· 潜伏缺陷扫荡 · 窗口一修复验证 PASS
- **wave**: 5 臂 × 5 seeds 全部 exit=0；产物审计全干净（`final=True/run_status=ok/updates=5000`
  ×25，含 LDS-C1 直接核验——skip-on-existence bug 存在但本 wave 无 resume 未触发）。
- **codex 潜伏缺陷扫荡**（3-pass xhigh，`b11af06`）：12 发现（3A/7B/2C），**无一作废已完成
  wave**；窗口一修复 F1–F5 落库（`98fe149` eval 层 + `6ece080` 汇总脚本），codex diff 复评
  BLOCK→4 阻塞项修复→APPROVE-WITH-NITS；远程验证：定向回归 **33 passed**、gate I1–I17
  exit 0（archive 缺 `.aris/`（untracked）致首次 exit=2 误报，copied tool 复跑 = 0）。
- **train 期表**（aggregate_e1rev v1，**非决定性**——决定性 = held-out eval）：

  | arm | seeds | guard pass | competence rate | selCCR (pass) | ego_sole (pass) | best_ret (pass) |
  |---|---|---|---|---|---|---|
  | aris_bellman | 5 | 4 | 0.80 | 0.65 | 5.0 | 42.4 |
  | base_only | 5 | 5 | 1.00 | 0.80 | 7.2 | 64.2 |
  | flat_factor | 5 | 5 | 1.00 | 0.88 | 7.2 | 53.8 |
  | global_gru | 5 | 5 | 1.00 | 0.84 | 6.0 | 54.0 |
  | partner_id_q | 5 | 5 | 1.00 | 0.88 | 7.6 | 56.3 |

- **sec18.12.3 第一行触发（远超阈值）**：全部 5 臂 ≥4/5 seeds 达成终端能力 ⇒
  **E1-rev 即决定性 run**，判读走 sec18.6 五分支 + sec18.10.2 admissibility 四行。
- 观察记录（不解读）：aris_bellman 是唯一有 guard fail 的臂（s3）且 train selCCR 最低；
  train-partner 指标与 held-out ZSC 泛化是两回事——一切结论等两段式 held-out eval。
- **下一步**：stage-1 held-out eval（25ep × 2 held-out × ~24 deployable ckpts，基线缓存
  schema v3 + 固定 eval seed），按 §14.2 执行卡（待授权）。

### 2026-07-05 stage-1 held-out eval 完成（24/24）· sec18.13 诊断链闭合 · 机制定式
- **stage-1 决定性表**（25ep×2 held-out，bootstrap 95% CI，产物 results_phase3_eval/stage1_decisive_table.*）：

  | arm | egoCCR [95% CI] | team tp/ep | ego tp/ep | serve share |
  |---|---|---|---|---|
  | aris_bellman | **0.125** [0.000, 0.375] | 1.000 | 0.125 | 0.250 |
  | base_only | 0.900 [0.700, 1.000] | 1.200 | 0.900 | 0.817 |
  | flat_factor | 0.800 [0.600, 1.000] | 0.900 | 0.800 | 0.950 |
  | global_gru | 1.000 [1.000, 1.000] | 1.100 | 1.000 | 0.950 |
  | partner_id_q | 0.800 [0.600, 1.000] | 0.800 | 0.800 | 1.000 |

- **三表读出**：sec18.10.2 行 1 触发（base 0.900 ≥ 0.9 ⇒ **基底非 ZSC-判别**，任何相对排序
  不得读作 ARIS 优势/劣势的主张证据）；sec18.6 落「仅 ARIS 崩」行（按预注册=可疑回归待查）。
  次级并读：resource-server-claim 上 aris 角色互补（partner 吞吐 2-3/ep、团队回报 59>base 38.7）；
  handoff-alternate-yield 上全臂唯 aris 死锁（partner_id_q 亦 2/5 死锁——该伙伴对 ID-oracle 也难）。
- **sec18.13.1 判决：POLICY-DEFAULT-WAIT**——zeroed 重评 4/4 seeds 行为与 inferred 逐 seed
  一致（塌缩不变）⇒ 信念通道非瓶颈。E2 表（sec18.7）zeroed 行同时完成。
- **trace（非 fast, 3ep）**：belief_zero_delta≈20.5（信念显著平移 Q）但不翻转 wait→act 排序。
- **训练伙伴探针矩阵（关键补充）**：
  | seed × 训练 yield 伙伴 | ingredient-near-yield | bottleneck-yield-terminal-yield |
  |---|---|---|
  | s0 | 0.32 | 0.16 |
  | s2 | **1.00**（52 送餐） | **0.00** |
  接管行为**分布内即呈斑驳状**（同 seed 对某 yield 伙伴满分、对另一 yield 伙伴零分）；
  bottleneck-中介的 yield 变体是系统性薄弱轴；held-out alternate-yield 位于最难角。
- **机制定式（三证据闭合）**：ARIS 策略**默认让位**（zeroed=wait），信念仅对**识别出的
  分布内伙伴**解锁接管（s2×near-yield=1.0 证明解锁存在且可完美）；OOD 伙伴 → 回落默认 →
  对 yielder 死锁。方法的伙伴条件化能力把训练分布的让位偏置学到最彻底——base/gru 学不会
  条件化反而获得 ZSC-鲁棒的无差别单干。诚实结论：**条件化解锁 vs 鲁棒默认的张力**是本轮
  最重要的科学发现，独立于后续修复成败均可入文。
- 附：诊断链工程记录——LDS-A1/B3 修复（`9e4d2f1`，codex APPROVE）解锁受控 zeroed 跑；
  探针首launch 因嵌套引号本地展开失败（编排器事故同族），改脚本文件后成功（教训入档）。
- **下一步**：群体压力干预新预注册（草案见
  review_bundles/latent_defect_sweep_20260704/E1REV2_POPULATION_PREREG_DRAFT.md）待用户 Type-B。

### 2026-07-05/06 override 门 + seed 行为取证（e1rev 重建基底；用户授权远程执行）

- **背景**：原 e1rev 产物（ckpt/CE/图/preflight）已从远程与本地全部丢失（编排器误删旧账）。
  用户批准同配方重建：代码 tar 同步（hash 核验一致）→ CE（role_conditioned_v2，6 伙伴×100ep）
  → preflight → 训 aris seeds 0/1/2/4（5000 updates，全部 guard-pass 发布）→ override 门 +
  行为探针。产物：`CPR_REPO/results_override/`、`outputs/asymm_ce_role_v2_e1rev/`、
  `logs_override/`。**非 bit-identical 原模型**（CE 重采样，图 16 因子、覆盖门全满足）。
- **override 门**（`scripts/override_gate.py`，骑 scripted_priority 钩子；两臂同 seed 逐局配对，
  终端链=serve/plate/pick_plate 强制、否则回落 argmax；codex 复评修正 pairing）：
  对 heldout-handoff-alternate-yield（25ep×4seeds）：
  | seed | argmax | override | 读法 |
  |---|---|---|---|
  | s0 | 1.00 | **142.14** | 汤已煮好、守着不端；强制拿盘即通 |
  | s1 | 82.88 | 82.88（全同） | 自发全链，override 无事可做 |
  | s2 | −8.00 | −8.00（全同） | 从不碰锅 → 终端前置从未成立 → override 惰性 |
  | s4 | −1.42 | −1.42（全同） | 同上 |
  pooled lift CI[24.0,48.0] **由 s0 单独驱动**（per-seed 1/4）——pooled 读数作废，按 seed 读。
  对 heldout-resource-server-claim（跑至 s0）：argmax 187.5 > override 150.6 —— 对 claimer
  让位正确、强制接管有代价（角色不对称成立）。完整 JSON 待该臂跑完补档。
- **行为指纹**（`scripts/probe_behavior.py`，3ep/组，探索性诊断）：
  - s1 argmax：fetch27/deliver27/pick9/plate6/serve6，完成率 1.0——完整迁移；
  - s0 argmax：fetch12/deliver9（会做菜）→ **cross_bottleneck 45 次全败 + wait 42 次**，
    守着煮好的锅不拿盘（pick_plate 明明 valid）——**中链价值错误**；
  - s2 argmax：wait_at_bottleneck×75 从开局；s4 argmax：**noop×177**（fetch valid 但不选）；
  - **s2/s4 全链脚本强制：完成率 1.0、每局 3 汤、回报 119.1（两 ckpt 逐位相同）**——
    环境/伙伴完全允许单干通关，**死锁 100% 在学到的 Q 里**。
- **训练侧取证（根因候选，L1）**：4/4 seed 最优 checkpoint 全在 **update 500–1000/5000**；
  全部记录 `last_ineligible_checkpoint_reason=no_ego_sole_correct_delivery`（后期 checkpoint
  反复零独立上菜）；守卫语义="曾出现过会上菜的快照即发布"。结合脚手架 anneal_updates=2500：
  **终端能力是脚手架窗口期的瞬态，TD 未将其巩固**——退火后 contrib_team 备菜局部最优重新
  接管。sec18.12 "脚手架解锁终端能力" 读数需修正为"暂时解锁、随退火蒸发"。含义：一切
  stage-1 读数（0.125、belief 平移翻不动、四粒度行）均测于 10–20% 训练进度的早期快照。
- **排除项**：守卫未失职（4/4 训练伙伴上真实 ego-sole 2/3/6/9）；重采样图结构健全；
  门接线正确（s0 生效、s1 两臂同因 argmax 本在终端链上）；环境无 bug（全链 100% 通关）。
- **原现象复现判定**：s0/s2/s4 三相与 stage-1 描述"wait_at_bottleneck/cross_bottleneck
  loops or noop"逐字吻合 + 原探针矩阵本就记录 seed 斑驳（s2 对两 yield 伙伴 1.00/0.00）——
  **重建基底忠实复现了现象类，包括其斑驳性**。
- **纪律注记**：sec18.12.4 禁止在见到 E1-rev 结果后再调 reward/exploration；任何 L1 修复
  （退火策略/credit/终端探索持续性）须新预注册条目 + Type-B 裁决后方可执行。
- **下一步（单一）**：待用户裁决——1–2 seed 的"脚手架不退火"诊断训练（anneal→∞，其余全同），
  检验"退火是否是能力蒸发的因"：后期 checkpoint 出现持续 ego-sole ⇒ 退火时机是旋钮；
  仍蒸发 ⇒ credit 结构问题更深。一次测量，直接命中 L1 机制。

### 2026-07-06 L1 巩固预注册执行：0/2 + 数据规模发现（用户授权）

- **L1_CONSOLIDATION_PREREG_DRAFT.md 按写定执行**：l1fix config（仅 bias_end 0.9、
  epsilon_end 0.5 两键，按基线起始值持平）× seeds{0,2} × 5000 updates ×
  --save_all_checkpoints；接线回读通过。codex 部署前复评抓到 yaml 重复键 BLOCKER
  （training 块后段真实 ε 键 0.5→0.1 会静默覆盖前段插入值）——修正后基线机制数字
  更正为：有效终端探索率 = ε(0.5→0.1) × bias(0.9→0.35) ≈ 36%→3.5%。
- **判定 0/2**：探索通道全程常驻（pick_count 75/93）仍不巩固——u500 后独立上菜全零，
  与基线逐点一致。**探索支持塌缩假设被证伪为主因。**
- **上游发现（本轮最重要）**：`updates_per_transition: 8` ⇒ 全训练仅 **625 transitions
  = 32 局经验**（episode_returns len=32）；每条经验被梯度复用 8 次。u500 能力≈前 3 局
  上的塑形先验；后续"蒸发"、seed 斑驳、OOD 四相，在 32 局尺度上都是小样本现象。
  **一切既往本基底结论（G2 时代含 5/5 分离实验、stage-1、override 门）都测于 32 局
  训练量的模型之上**——此事实此前从未被任何文档记录。
- **下一步（单一，待 Type-B）**：数据规模探针预注册——l1fix config + 数据量 ×8
  （updates_per_transition: 1，5000 transitions ≈ 250 局，其余不动），2 seeds，
  判据沿用晚期窗口巩固；巩固 ⇒ 根因=样本饥饿；仍不巩固 ⇒ 信用结构预注册顺位执行。

### 2026-07-06 基建提速：环境执行层 4.2×/单局 ~18×，golden 逐字节一致（用户授权）

- **动机**：数据规模探针前先修执行效率。cProfile 实测（1 局 61.4s，4970 万次函数调用）：
  **~2/3 时间 = 25 万次对 JAX 数组的逐元素索引**（state 读取/事件抽取/featurizer/伙伴脚本
  每步 ~4200 次 getitem，每次走完整 JAX 原语分发）；次因 = torch/OpenMP 144 核线程池对
  小张量的空转（user 2m47s vs real 45s）。
- **改动 1（代码，env_adapter.py 单文件）**：step/reset 后 `jax.device_get` 一次性把
  state/rewards/dones/info 物化为 numpy pytree 再暴露——下游全部标量读取变纳秒级；
  jit step 接受 numpy 叶子（同形状不重编译）。附带删掉 featurizer 路径下每步转换后即
  丢弃的 raw-obs 浪费。**下游零改动。**
- **改动 2（零代码，launch 配方）**：`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1`（配 JAX_PLATFORMS=cpu）。**今后所有远程 train/eval 启动
  必须带此三变量**——不带则单进程占满 144 核互踩，并行舰队吞吐塌方。
- **验证**：golden 1 局逐字节 diff 一致；3 局数值与改前老代码逐项一致（82.88/1.0/6）；
  `experiments/overcooked_v2/tests/` 47/47 通过（p0_p1 回归 + 静态不变量；
  test_deadlock_recovery 收集错误为远程同名目录残留 + 路径问题，先于本次改动存在）。
- **数字**：1 局含启动 45.1s→10.8s（4.2×）；启动 ~8.8s、单局本体 ~37-40s→**~2s（≈18×）**；
  user/real 从 3.6min/35s → 11s/10.8s（线程空转清零 ⇒ 144 核可干净并行 ~百个进程）。
- **外推**：数据规模探针（5000 transitions）从 ~3-4h/seed 降至 **~15-25min/seed**；
  stage-1 级评估（25ep×24 ckpt）从 ~6-8h 降至 **~1h 内**。改动未提交 git（待用户指示）。

### 2026-07-06 DATA-8x 双臂判定：样本饥饿实锤为根因；探索常驻假设二次证伪（用户授权）

- 双臂 × 双 seed（预注册 L1 草案 §6，先写后跑；接线回读全过；episodes=251 双臂确认）：
  **A（l1fix-8x 探索常驻）0/2 巩固；B（e1rev-8x 原退火）2/2 巩固**（B-s2 晚期
  ego_sole 1/2/4/4、回报升至 ~40 且 u5000 仍上行）。
- 结论：(1) **32 局样本饥饿是巩固失败的根因**——原退火调度在 250 局下自然巩固，
  此前一切"瞬态/蒸发"现象是数据量问题的表象；(2) 探索常驻（l1fix）不但无效且有害
  （ε 恒 0.5 ⇒ 数据永远高噪、贪心不收敛），**退火设计无罪，l1fix 调度废弃**；
  (3) 250 局未饱和，正式重建取更大预算。
- 影响面：既往全部基底结论的重测路径明确——不需要动方法/奖励/探索任何一处，
  只需要正常的数据量。U2/信念粒度问题的检验条件（一个巩固的基底）现在有了实现路径。
- 下一步（单一，待 Type-B）：SUBSTRATE_REBUILD 预注册——e1rev 原调度 ×
  total_updates 40000（~2000 局）× 5 seeds + stage-1 held-out 重测。

### 2026-07-06 stage-1 held-out 对比：32 局 vs 250 局基底（用户授权；25ep×2 held-out×seed0 协议）

| 基底 | vs yield 伙伴（egoCCR / ego上菜 / 回报） | vs claim 伙伴（ego / 伙伴正确上菜 / 回报） |
|---|---|---|
| 32 局 s0 | 0.000 / 0 / 5.0（死锁） | 0 / **50** / **59.0**（让位配合） |
| 32 局 s2 | 0.000 / 0 / −6.0（死锁） | 0 / **75** / 16.0（让位） |
| 250 局 s0 | **1.000 / 25 / 41.5** | **25 / 0** / 36.1（全包） |
| 250 局 s2 | **1.000 / 25 / 38.7** | **25 / 0** / 35.8（全包） |

- **死锁消失**：对让位伙伴 egoCCR 0→1.0（两 seed 25/25 局全部 ego 上菜）。
- **让位同时消失**：对抢活伙伴从"让对方上 50–75 次、团队 59"变为"ego 全包 25 次、
  对方 0 次、团队 36"——base_only 式全包表型，s0 上比配合少 ~23 分/局面。
- **u500/sel/final 三份 checkpoint 行为逐分相同**（异常解释）：8x 数据下 u500 已含
  25 局经验（≈旧全量），held-out 贪心行为在 u500 即饱和为"永远上菜"，其后不变；
  选择器照常选 u500，无 bug。中期验证的回落-回升是对训练伙伴的行为，与 held-out 无冲突。
- **判读（sec18.13.2 双判据首次真实咬合）**：第 1 条（yield 接管）2/2 PASS；
  第 2 条（claim 不全包）2/2 **FAIL**。**伙伴条件化行为在两个基底上都不存在**：
  32 局=永远让位，250 局=永远上菜。项目核心问题（信念能否让行为随伙伴切换）第一次
  站在一个有能力的基底上，且有量化奖金（对 claimer：条件化值 ~+23 回报 + 伙伴吞吐 2-3/局）。
- 注意：aris_bellman 的信念机制在 250 局训练中在场，未产生条件化——与"信念平移 Q
  不翻转 argmax"的旧诊断连续。s2 训练验证 u4500 对训练 claimer 出现 partner=8 上菜，
  提示条件化可能随更多数据萌芽——SUBSTRATE_REBUILD（2000 局）的读出应加入
  claim-deference 指标以裁决"数据独自能否长出条件化"这一零假设。

### 2026-07-06 SUBSTRATE_REBUILD（2000 局 × 5 seeds）：伙伴条件化由数据涌现；双判据 4/5 达标（用户授权）

- e1rev 原调度 @ 40000 transitions（eps=2001 确认）× seeds 0–4，guard 5/5，~1h。
- **held-out 决定性表（25ep×2 伙伴×seed0，sel=final 行为一致）**：
  | seed | yield 伙伴 egoCCR / ego上菜 | claim 伙伴 ego / 伙伴正确上菜 / 回报 |
  |---|---|---|
  | s0 | **1.000** / 25 | 0 / **50** / 50.0 |
  | s1 | **1.000** / 25 | 0 / **75** / 24.6 |
  | s2 | 0.000 / 0 | 0 / 25 / 26.8 |
  | s3 | **1.000** / 25 | 0 / 25 / 27.3 |
  | s4 | **1.000** / 25 | 0 / 25 / 27.1 |
- **判定**：R1 巩固 5/5；R2 让位涌现分支触发（5/5 对 claimer 让位 + 4/5 对 yielder 接管
  ——32 局"永远让"、250 局"永远抢"、2000 局"看人下菜碟"）；R3 sec18.13.2 双判据
  **4/5 双条通过**（s2 第 1 条不过，seed 方差如实报）。
- **含义**：(1) 角色自适应在现有 aris_bellman（含其因子信念机制）+ 足量数据下涌现，
  无需 U2/换粒度/改奖励——**U2"必要性"口径死亡**；(2) 但条件化是否由信念通道承载
  未裁决——方法核心主张的正面证据窗口打开：**下一步（单一，待 Type-B）= 本基底 E2
  zeroed**（置零证据通道，看 claim-让位/yield-接管是否塌回无条件行为），一次运行裁决
  信念是否承重；(3) 既往全部否定性诊断（0.125、四粒度表、belief 翻不动）确认为
  32 局样本饥饿的伪影，需按新基底重建证据链。范围：dev-heldout、单布局。

### 2026-07-06 E2-zeroed @ 2000 局基底：信念通道对 claim-配合承重 3/5（用户授权）

- 协议：stage-1 同款 + `--zeroed_partner_option_ablation`（LDS-B3），5 rebuild ckpt，
  对照=已在盘 inferred 评估。声明与完整性门全过。
- **yield 维度 5/5 不变**（接管=状态承载）；**claim 维度 4/5 改变**：s0/s2/s3 配合塌为
  互相干扰（伙伴上菜 50/25/25→0、回报→7.9；同 ego 对 yielder 仍 25/25 ⇒ 能力完好，
  塌的是伙伴识别/配合）；s4 反向（zeroed 双方各 25、回报 58.5>27——证据过度让位案例）；
  s1 不变。
- **结论**：角色条件化中"对抢活者的配合"在 3/5 seed 由证据→信念通道承载——
  ARIS-Bellman 核心主张的首份正面证据（范围：dev-heldout 单布局）。与 32 局 E2
  （zeroed==inferred，通道无作用）对照：通道承重性随数据量修复而出现。
- 下一步候选（待用户）：行为指纹 zeroed-claim 塌态（ego 在干什么：堵路/抢盘/误上菜），
  把"通道承重"从数字变成机制陈述；或直接进入证据链正式重建（加局数/CI/盲测 split）。

### 2026-07-06 E1 多臂对照 @ 2000 局基底（5 方法 × 5 seeds，--fast 统一协议；用户授权）

- 20 基线训练全绿（eps=2001 全臂回读确认，guard 20/20 pass，ITT 无剔除）。首轮评估
  15 臂被完整性门拒（无信念方法产不出 Δ_info/MI；stage-1 原协议为全方法 --fast，
  本轮漏带旗标），按原协议统一 --fast 重评 25 臂。aris fast vs full 数字逐项一致
  ⇒ 诊断收集不扰动 rollout（附带接线验证）。
- **两行判据汇总（row1 对 yielder 接管 / row2 对 claimer 不全包 / 双过）**：
  aris 4/5 | **5/5** | **4/5**；base_only 5/5|2/5|2/5；global_gru 5/5|2/5|2/5；
  flat_factor 5/5|1/5|1/5；partner_id_q 5/5|2/5|2/5。
- **三个关键读数**：
  1. row1 在 2000 局下**全方法普遍通过**（数据即能力，不再有判别力——符合预期）；
  2. **真正的判别轴 = 伙伴对比度**：aris 是唯一"对 yielder 全上（25）、对 claimer
     全让（0）"的方法（4/5 seed，Δ=+25）；全部 baseline 对两类伙伴**无差别上菜**
     （claim 侧 ego 25–50，Δ≤0）。baseline 的 row2"通过"（各 1–2/5）全部经由
     **共同上菜**（25/25）而非让位——row2-screen 的"吞吐不降"可被共serve冒充
     （ChatGPT R1-[09] 预警成真），正式轮主指标应改为**伙伴对比度**
     （Δ ego-deliveries across partner types）+ E2 因果。
  3. **诚实披露（不利面）**：对 claimer 的团队回报，共serve/全包 baseline（57–70）
     **高于** aris 的让位（24.6–50）——该脚本 claimer 在共serve下照常出菜，此布局
     上"角色互补"不等于"回报最优"。主张措辞必须是"因子信念产生伙伴条件化的
     互补行为（对照+E2 因果双证据）"，不得写"对 claimer 回报更高"。
- **预注册 §9 分支判定**：落于两分支之间（baseline both=1–2/5，既非 ≤1/5 也非 ≥3/5）；
  aris（4/5）与全部 baseline（≤2/5）分离成立，但 row2-screen 判据需按上条收紧后
  进入正式轮。base_only 2/5 过 row2 = 判据松弛证据（共serve路径），非伪影。
- **下一步（单一，待 Type-B）**：正式轮 = 盲 held-out 伙伴（全新脚本）+ 50–100 局 +
  多评估 seed + 预注册主指标改为伙伴对比度（Δ ego-serve）+ E2-zeroed 因果对照
  （aris 与任一"对比度非零"的 baseline 同跑）。
