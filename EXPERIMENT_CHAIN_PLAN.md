# EXPERIMENT_CHAIN_PLAN.md — 从当前状态到论文级结果的完整实验链路

**Status:** EXECUTING — Phase 0–2 完成；Phase 3 E1-rev wave 运行中（9/25 final，决定性判读待
held-out eval）· **Updated:** 2026-07-04，见 **补充 C**（§11–§16，执行对账 + 剩余链路调整版）
**前提**：[FINDINGS_LEDGER.md](FINDINGS_LEDGER.md)「静态验收 PASS」段；METHOD_LOCK sec17 修复锁定条目。
**阅读指引**：§0–§10 为原计划（保留存档）；与 补充 C 冲突处以 补充 C 为准（每处变更均引预注册段）。
**边界**：所有实验按 [CUSTOMER.md](CUSTOMER.md) 远程执行（`zsc-customer`，8 GPU），逐次授权；
本文档本身不触发任何执行。结果解读一律走预注册规则 + 伪影自检清单 + Type-B 双签。

---

## 0. 链路总览

```
Phase 0 决策与预注册(静态,√前提)
   └→ Phase 1 基础设施远程验证(Type-A, 小时级)
        └→ Phase 2 基底证书(伙伴可区分性 + CE支持度探针, 天级)  ←— 分叉点
             ├─ 证书PASS → Phase 3 决定性实验 E1-E3(核心科学问题第一次被干净地问出)
             │               └→ Phase 4 主张级实验 E4-E7(五主张逐条)
             │                    └→ Phase 5 方法锁定→盲测→终表→论文
             └─ 证书FAIL → STOP: 换基底(FCP/MEP训练伙伴群体路线)，不得硬跑
```

每个 Phase 有明确的进入门与退出门；**任何 null/负结果先过伪影自检清单**
（[ROOTCAUSE_REVIEW_PLAN.md](ROOTCAUSE_REVIEW_PLAN.md) §3.2）再允许解读。

---

## 1. Phase 0 — 决策与预注册（静态，无 GPU）

| # | 事项 | 类型 |
|---|---|---|
| G0.1 | **§5.1 治理裁决（现已成熟）**：`role_conditioned_v2` candidate 是否允许作为 benchmark-v2 基底进入 Phase 2 认证？允许→走 R2.1 certificate；否决→Phase 2 直接改走"标准伙伴 asymm + FCP/MEP 群体"备选线 | 用户 Type-B |
| G0.2 | **D-A 决策**：role_v1/novb config 的 `require_ego_delivery_selection: false`——翻成 true，或书面论证 contrib_team 下 free-riding 无利可图故不需要（写入 config 注释 + 台账） | 用户 |
| G0.3 | 跟进项微提交（可选但便宜）：F8 文档化 refine 偏置、F9 改名 train 侧 completion、F10 gap-proxy 补 credit_params、F3 清死代码；F12 把 I10–I17 并入 fidelity gate **工具** | Claude 静态 |
| G0.4 | **预注册写入 METHOD_LOCK 新段**（启动前必须落盘）：E1/E2/E3 与 R2 的全部"结果→结论"分支表（见 §4/§3 的表格，逐字入档），声明主指标、样本量、判据阈值 | Claude 起草 + 用户签收 |
| G0.5 | 创建 `EXPERIMENT_LOG.md` 骨架（首个结果落盘时填写） | Claude 静态 |

**退出门**：G0.1/G0.2 有记录的裁决 + G0.4 预注册段已入 METHOD_LOCK。

## 2. Phase 1 — 基础设施远程验证（Type-A，~2–4 GPU 时）

远程注意事项（项目记忆）：JAX 走 CPU（`JAX_PLATFORMS=cpu`）+ torch 走 GPU；检查 GPU 0/3 ECC；
tar-over-ssh 同步；每次代码更新出显式 git diff；正式训练必须 `--preflight_path` 接受的 preflight。

| # | 内容 | 通过判据（机器可查） |
|---|---|---|
| R1.1 | 全测试套件 + fidelity gate 远程重跑 | pytest 全绿；gate I1–I17 exit 0 |
| R1.2 | 1-seed micro-train 冒烟（诊断 config，500–1000 updates）+ 10 ep 评估，端到端产物检查 | `evidence_policy=behavior_inferred_v1`；`oracle_source_count=0`；`reward_scale_verified=true`；CE sidecar 四掩码齐全；checkpoint_selection 记录 eligibility；`headline_success_metric=ego_correct_completion_rate` |
| R1.3 | NEW-2 历史产物审计（一次性脚本）：扫历史 results 目录，列出 guard=fail 但 checkpoint.pt 存在的运行 | 清单落盘 review_bundles；受污染产物标记 quarantine |

**退出门**：R1.1–R1.2 全绿（Type-A 自判可过，记录留档）。

## 3. Phase 2 — 基底证书（分叉点，~30–60 GPU 时）

**R2.1 伙伴可区分性证书**（G0.1 允许的基底上跑；codex probe #4）
- 每候选布局 × train/held-out 伙伴：固定种子 + 随机化起点，60–100 原始步开环 + **带称职脚本 ego 的完整 episode**（修正旧探针 ego=noop 的盲区），测成对轨迹/动作/回报分歧（价值关键阶段）。
- 同时验证：脚本 oracle 可完成任务；partner-only 不封顶（留出可辨识空间）。
- **预注册分支**：≥2 个价值关键因子上各有 ≥2 个可区分行为模式 → PASS，基底可用；
  部分可区分 → 收窄到可区分的因子子集重划 train/held-out；全塌缩 → **FAIL，STOP**，
  走 FCP/MEP 训练伙伴群体备选线（回到 EXPERIMENT_PLAN 原始设计，另行预算）。

**R2.2 asymm CE 支持度探针**（codex probe #3；与 R2.1 并行）
- 每训练伙伴 500–1000 episodes 重测 CE，产物必须含 per-pair `weight_sum` + 四掩码 + 测量值 CI。
- **预注册分支**：serve CE 在支持度充足下**测得非零** → METHOD_LOCK §11 布局结论被推翻，
  asymm 恢复为判别布局候选；**测得零且支持充足** → 几何解释获得支持，asymm 只做 Table-1 sanity。
- 引用规则（已入 sec17 禁令）：无 `estimable_mask=true` 的 0 不得解读。

**R2.3 训后 reference-gap 复检设计**（D2 修复的操作化）：在 Phase 3 每个训练完成的 base_only 上
复测 scripted-oracle vs base_only gap；gap→0 的 split 自动标记"课程已解题"，不得用于主张比较。

**退出门**：R2.1 PASS 且确定了正式 train/held-out split；R2.2 分支已裁决并更新布局选择。

## 4. Phase 3 — 决定性实验（核心主张第一次被干净测试，~100–200 GPU 时）

**E1 去 oracle 四臂重跑（THE decisive run；codex probe #1）**
- 臂：`aris_bellman / flat_factor / global_gru / base_only`（+`partner_id_q` 作 oracle 上界参照，不参与主张）。
- 规模：**5 seeds × 50–100 ep/伙伴**，train split 训练、dev-heldout 评估；全部完整性硬门必须 true。
- 主指标：`ego_correct_completion_rate` + throughput；辅助：time-to-complete、ego/partner 拆分、
  wrong-delivery、首个诊断动作时机。
- **预注册分支（G0.4 入档，不许事后重释）**：

| 结果 | 结论 |
|---|---|
| ARIS > flat > base（CI 分离） | 因子相关性路由获得支持（Claim 2/4 判别证据） |
| ARIS ≈ flat > base | 信念有用、路由无增益 → 主张收窄至 Claim 2 弱式 |
| 四臂 ≈ | 先过伪影自检；全排除后记录"该基底上无 ARIS 特异优势"（诚实负结果） |
| 仅 ARIS 崩 | 视为去 oracle 修复回归，回修不下结论 |
| partner_id_q >> ARIS | 推断信息瓶颈量化 = 上界差距，写入分析 |

**E2 oracle 通道消融**（codex probe #2）：同 checkpoint 两种 eval（行为推断通道 vs 置零通道），
量化推断信息的真实贡献；ARIS 在置零下仍分离 → 证据来自其他通道，须如实报告。

**E3 信念持久化消融**（codex probe #5 + F13）：window-4 无持久 / window-8 无持久 / 持久隐态（主方法）
× 3 seeds；仅持久臂改善"须重复失败尝试才能识别"的 held-out 案例 → P4 机制主张获得支持。

**退出门**：E1 结果经伪影自检 + codex 复核 + 用户 Type-B 签收，写入 EXPERIMENT_LOG。

## 5. Phase 4 — 主张级实验（五主张逐条，~150–300 GPU 时）

以 E1 主结果为条件展开（若 E1 落在"四臂≈"分支，Phase 4 缩减为诊断性质并触发基底复议）：

| # | 实验 | 对应主张 |
|---|---|---|
| E4 | belief-swap / factor-deletion / shuffled-route 因果消融（diagnose_traces 已修复可用） | Claim 3（价值充分性因果） |
| E5 | 支持图鲁棒性：raw top-K vs coverage-constrained vs minus-critical-factor（A1/A2/A7 式，在去 oracle 路径上重做） | Claim 4（图承重） |
| E6 | Δ_info vs MI 的事后诊断价值回归（诊断层，不进选择器） | Claim 1/5（Bellman 诊断价值） |
| E7 | held-out **重组**测试（D3 收窄后的主张口径：已见模式的新组合）+ 适应性伙伴压力测试 | Claim 2/5 |

每个实验launch 前单独预注册分支表（同 G0.4 格式，METHOD_LOCK 追加段）。

## 6. Phase 5 — 锁定 → 盲测 → 终表 → 论文

1. **方法锁定 v2**：以 E1–E7 存活的最简配置冻结（沿 G2-lite 先例）。
2. **盲 held-out split**（METHOD_LOCK §7 纪律）：锁定后才创建，绝不调参；
   盲测失败 → 如实标注"dev 成功、盲测未过"，不许静默修补。
3. **终表**：5 seeds × ≥2 CE seeds × 50–100 ep/伙伴，CI 全报；
   dev-heldout 与 blind-heldout 分列（cautious-claim-scope 纪律）。
4. **论文物料**：EXPERIMENT_LOG → NARRATIVE_REPORT → `/paper-writing` 门
   （PROJECT_DASHBOARD §4 就绪门依然有效：主主张须获支持 + 双签）。
5. **诚实报告义务**：未获支持的主张按"收窄/未证"如实写；NEW-4 材料永不入正文证据链。

## 7. 预算与调度概估

| Phase | GPU 时 | 墙钟（8 GPU） |
|---|---|---|
| 1 | 2–4 | ~半天 |
| 2 | 30–60 | 1–2 天 |
| 3 | 100–200 | 2–4 天 |
| 4 | 150–300 | 3–5 天 |
| 5 | 150–250 | 3–4 天 |
| **合计** | **~450–800** | **~2 周** |

在提案 pilot 预算（500–1000 GPU 时）内；正式预算 2200–4500 富余。若 Phase 2 走 FCP/MEP
备选线，另加 400–800 GPU 时（群体训练），需重新报批。

## 8. 全程不可绕过的纪律（汇总）

- 预注册先于 launch；结论按预注册读出，禁止事后重解释（§3.1）。
- 负结果先过伪影自检清单，命中即 `ARTIFACT-SUSPECT`（§3.2）。
- 完整性硬门（I10–I17）任一为 false 的运行不进任何表。
- 每步远程执行前：显式 git diff + 接受的 preflight + 用户授权。
- Type-B（主张支持/否定、基底更换、方法锁定）一律 codex + 用户双签。
- METHOD_LOCK 只追加；台账 ID 贯穿 handoff→commit→结果→论文表格。
- 盲 split 创建后零调参；NEW-4（CODEX_IMPL_SPEC v1–v4 数字）永久隔离。

---

# 补充 A（2026-07-03）：可行性分析 —— 基于 Phase 1 实测数据

> Phase 1 冒烟给出了真实单位成本，原 §7 的"GPU 时"估算按此校准。**关键路径不是 GPU 算力，
> 而是 CPU-JAX 的环境墙钟 + eval 吞吐**（远程 JAX 只能跑 CPU，torch 在 GPU；见项目记忆）。

## 9.1 实测单位成本（zsc-customer，CPU-JAX，单槽）

| 环节 | 实测 | 外推 |
|---|---|---|
| CE 采集（顺序，cramped 200 步 ep，随机策略） | ≈7.4 s/ep/伙伴 | 20ep×7伙伴 ≈18min；估计+refine+建图 +2–3min |
| CE 采集（asymm 400 步 ep，估） | ≈15–25 s/ep | R2.2 的定价基准（见 9.3） |
| train 启动（env+graph+preflight 门） | ≈2.5 min | 每 run 固定开销 |
| train 更新 | ≈0.32 s/update（cramped debug） | 5000 updates ≈27min ⇒ ≈30min/run **保守值**；历史 asymm G2 全训练 4–6min/run 为乐观下界（两数据点差 6×，Phase 2 首个 asymm train 实测后定稿） |
| eval | 1304 s / (8ep×2伙伴，**含**随机+外部参照基线摊销) | ≈163 s/ep-等效；50ep×2伙伴 ≈2.3h/checkpoint |
| preflight | 复用 replay 后秒级–2min | **必须**设 `graph.replay_path`，否则 gap-proxy 默认自采 100ep ≈12min+ |
| 并行度 | 可用 GPU 槽 ≈4（5,6 + 视占用 2,7；0/3 ECC 故障禁用） | JAX-CPU 共享 ⇒ 并行槽间拖慢 ×1.3–1.5（G2 时代 wave 实测） |

## 9.2 E1 可行性核算（决定性四臂重跑）

- **训练**：4 臂 × 5 seeds = 20 runs。保守 30min/run ⇒ 10 GPU·h；4 槽 wave ⇒ 墙钟 ≈3–4h。
  若 asymm 实测靠近历史 5min/run ⇒ 墙钟 <1h。**非瓶颈。**
- **评估（瓶颈）**：20 checkpoints × 2 held-out × 50ep，按实测 ≈2.3h/ckpt ⇒ 46h 顺序。
  缓解（按序启用）：① 两段式——首轮 25ep 全体 → 预注册分支初判 → 仅决胜臂补到 50–100ep；
  ② 4 槽并行 eval（既有基建，git 9424de3）⇒ ÷3；③ 参照基线跨 checkpoint 共享（同伙伴同
  env 的 random/reference rollout 只算一次——小改，Phase 2 顺手做）。三项齐用 ⇒ **墙钟 ≈4–8h**。
- **统计功效（诚实声明）**：5 seeds 只够检大效应（d≳1.8 @ 80% power）。预注册补充一行：
  **"CI 重叠但方向一致 → 追加 5 seeds 复跑一次再裁"**，防止把功效不足误读成"无差异"
  （此行并入 METHOD_LOCK sec18.6，作为其"all four ≈"分支的前置检查）。
  CI 操作化：seed 级 bootstrap 95%（10k 重采样）/ 或 Mann-Whitney U（5v5 最小 p≈0.008）。
- **合计**：E1 全程（训练+两段评估+分析）≈1–1.5 墙钟天。原 §7 的 Phase 3 估算维持成立。

## 9.3 R2.2 可行性核算（asymm CE 支持度探针）

- **支持度算术**：CR 时代 asymm ≈200ep 只有 17 次 serve 尝试（≈0.085 次/ep，随机策略）。
  目标 weight_sum ≥ min_weight=20 ⇒ 名义上需 ≥235ep/伙伴，但联合权重按伙伴列稀释，
  **500ep/伙伴属边缘、1000ep/伙伴较稳**——预注册的"support 仍不足 ⇒ inconclusive+升级"
  分支（sec18.5）就是为此准备的；升级路径 = targeted terminal-stage starts（独立估计器，
  须与主 CE 分开标注）。
- **墙钟**：1000ep × 6 伙伴 × ≈20s ≈33h 顺序 ⇒ **按伙伴分 6 进程并行 ≈5.5h**（CE 采集
  互不依赖；注意 run_ce_pipeline 是顺序采集——按伙伴并行需拆 6 个单伙伴子任务后合并 replay，
  或接受顺序跑过夜）。**R2.2 是 Phase 2 墙钟大头，安排过夜跑。**
- 先导（便宜）：先跑 `rc2_reachability.py` 式终端可达性探针（分钟级），若随机策略下
  serve 完全不可达则直接走 targeted-starts 估计器，省 33h。

## 9.4 R2.1 可行性（伙伴可区分性证书）

- 开环 trace（60–100 步/伙伴/布局）秒级；带称职 ego 的完整 episode 分钟级。**成本可忽略。**
- 基建复用：`scripts/rc2b_substrate_certificate.py`（Phase A 证书，commit ade0051）扩展
  v2 注册表 + 成对分歧度量。判据操作化（写进脚本，机械可判）：
  ① 成对原始动作序列不一致率 ≥20%（60 步窗）；② 终端阶段行为分布差异（who-serves 频次）
  ③ 回报向量分。**W2/W3 的修复验证就由本证书承担**：v2 的两个新 held-out
  （`heldout-handoff-alternate-yield`、`heldout-resource-server-claim`）与全部训练伙伴的
  成对分歧必须过阈，yield 伙伴的蹲守率（wait 占比）直接可读——证书 FAIL 哪条修哪条，
  不盲修。

## 9.5 风险登记与 go/no-go

| # | 风险 | 概率 | 缓解/出口 |
|---|---|---|---|
| K1 | R2.1 证书 FAIL（v2 仍塌缩） | 低–中（v2 专为差异化设计，但从未行为验证） | 预注册 STOP→FCP/MEP；先修证书点名的项再重跑一次；两次 FAIL 即切线 |
| K2 | FCP/MEP 线基建缺口（CPR_REPO 是否有现成 IPPO 自博弈群体训练未确认） | 中 | E1 出信号后再投；先做 1–2 天预研 spike（JaxMARL 自带 IPPO 基线可用性）；此为 E1 后的第一件事 |
| K3 | eval 墙钟爆炸 | 已量化 | 9.2 的三项缓解；两段式为默认 |
| K4 | R2.2 support 仍不足 | 中 | 预注册 inconclusive 分支 + targeted-starts 升级；先跑可达性先导 |
| K5 | 有人在大样本 CE 上误用 batched collector（S7 交错 bug 未修） | 低 | **明令：正式 CE 只许顺序路径**（已在执行卡）；F1/F2 保持 defer |
| K6 | 正式跑误走 ce_path+覆盖关的冒烟路径 | 低 | 执行卡红线 + graph_path 分支的无条件覆盖门本身就是保险 |
| K7 | JAX-GPU 若被修复（ptxas），单位成本全面下降 | 机会 | Phase 2 开始时花 10min 验一次 `JAX_PLATFORMS=gpu` 冒烟，能用则全线提速 3–10× |

## 9.6 校准后的时间线

```
Phase 2: R2.1 证书(≤0.5天,含修复迭代) + R2.2 过夜(1天) + asymm CE 正式重测(并入 R2.2)
         ⇒ 1.5–2 天
Phase 3: E2/E3 前置小实现+评审(0.5天) + E1 训练(0.5天) + 两段 eval(0.5–1天) + E2/E3(1天)
         ⇒ 2–3 天
Phase 4: 视 E1 分支，2–4 天（E4–E6 依赖诊断脚本，已修复可用）
Phase 5: 终表 5 seeds×2 CE seeds + 盲测 ⇒ 2–3 天
合计 ≈8–12 墙钟天（原估 2 周成立）；若 K7 兑现则显著缩短
```

---

# 补充 B（2026-07-03）：执行卡 —— 每实验的可复制细节

**红线（适用全部执行卡）**：正式跑一律 `graph_path` 分支（pipeline 产出的覆盖完整
graph.json，train:884 无条件覆盖门是保险不是障碍）；`ce_path`+覆盖关仅限冒烟；CE 只许
顺序采集；GPU 用 {5,6}+空闲，禁 {0,3}；每次同步 = git archive 全新目录 + SYNC_PROVENANCE；
所有 run 产物 tar 回本地 `review_bundles/`；结果只按 METHOD_LOCK sec18 预注册表读出。

## 10.1 R2.1 执行卡（伙伴可区分性证书）

```
前置: 无（v2 注册表已在树上；证书本身就是 W2/W3 的验收器）
脚本: 扩展 scripts/rc2b_substrate_certificate.py →
      certificate(layout=asymm_advantages, registry=role_conditioned_v2,
                  probes={open_loop_60, full_episode_with_competent_ego},
                  metrics={pairwise_action_disagreement, terminal_role_distribution,
                           return_vector, yield_camping_rate})
判据: sec18.4 表 + 9.4 的三条操作化阈值; 产物 certificate.json 入 preflight 元数据(I13 关联)
输出: review_bundles/phase2_certificate_<date>/
失败处理: 修证书点名项(一次一 commit, 引台账 W-ID) → 重跑; 两次 FAIL → sec18.4 STOP 分支
```

## 10.2 R2.2 执行卡（asymm CE 支持度探针）

```
前置: R2.1 PASS(定 train 伙伴集); 先导可达性探针(rc2_reachability, 分钟级)
命令: run_ce_pipeline --config <v2配置> --output_dir outputs/asymm_ce_v2_probe
      --seed 42 --episodes-per-partner 1000 (先导若差则直接 targeted-starts)
调度: 过夜; 或拆 6 单伙伴任务并行(合并 replay 后统一 estimate)
读出: ce_support_audit.json 的 serve/plate 相关 cell {weight_sum, estimable, CE, CI}
      → sec18.5 三分支
连带产出: 若 estimable 且图过覆盖门 ⇒ 该产物直接作为 E1 的正式 CE/graph(带哈希)
```

## 10.3 E1 执行卡（决定性四臂重跑）

```
前置: R2.1 PASS + R2.2 产出正式 graph.json + 新建 configs/ocv2_step4_asymm_role_v2.yaml
      (由 role_v1.yaml 派生: partner_set=role_conditioned_v2;
       train=v1前6伙伴, heldout={heldout-handoff-alternate-yield, heldout-resource-server-claim};
       graph.graph_path=<R2.2产物>; require_ego_delivery_selection: true 继承;
       oracle_role_conditioned_ablation: false 继承)
训练: for m in {aris_bellman, base_only, global_gru, flat_factor, partner_id_q(参照)}:
        for s in 0..4: train_aris --config v2.yaml --preflight_path <accepted>
          --graph_variant full_support --method $m --seed $s
      wave 调度 4 槽; 每 run 检查 free_rider_guard + deployable_checkpoint
评估: 两段式 —— 第一段 25ep×2 heldout×全部 ckpt(并行); 按 sec18.6 初判;
      决胜臂补 50–100ep。参照基线共享缓存(Phase 2 顺手实现)。
      eval 硬门全绿为入表条件(evidence_policy/oracle_source/missing/reward_scale)
读出: headline=ego_correct_completion_rate + throughput(次级指标全записать)
      → sec18.6 五分支 + 9.2 功效补充行("重叠且同向→加5 seeds")
产物: EXPERIMENT_LOG 模板逐字段; tar 回 review_bundles/phase3_E1_<date>/
Type-B: codex 复核 + 用户签收后才写任何主张
```

## 10.4 E2/E3 执行卡（消融）

```
E2 前置实现(小, 先 codex diff 评审): evidence.partner_option_inference.mode: {inferred(默认), zeroed}
   — zeroed 时推断器输出置 None/零(eval-only 用途), evidence_policy 记
   'behavior_inferred_v1_zeroed_ablation' 以免与硬门冲突(硬门需放行该显式模式)
E2 运行: E1 决胜 checkpoint × 2 eval 模式 × 50ep → sec18.7 上表
E3 前置实现(小): training.belief_persistence: {on(默认), off} — off 即回退每次选项决策
   零隐态重编码(旧行为), 供消融
E3 运行: {window4+off, window8+off, persistent(主)} × 3 seeds, 同 E1 基底
   → sec18.7 下表; "须重复失败尝试才能识别"的案例集 = held-out 伙伴中 serve 受阻情形
   (diagnose_traces.py 已修复可用于案例抽取)
```

## 10.5 Phase 2 顺手工程项（各一 commit，先静态评审）

```
1. 参照基线缓存: evaluate_aris 对 (layout, partner, env_config) 相同的 random/external
   参照 rollout 结果落盘复用 — E1 评估提速的主项
2. (可选, K7) JAX_PLATFORMS=gpu 10 分钟冒烟验证; 能用则全线改
3. (可选, R1-C) run_ce_pipeline --replay_path 复用开关
4. E2/E3 前置开关(10.4) — 必需项
```

---

# 补充 C（2026-07-04）：执行对账 + E1-rev 运行态下的剩余链路调整

> 本补充在 E1-rev wave 运行中写成。**训练期数字只用于调度决策，不用于主张判读**——
> 判读一律等 held-out eval，按 METHOD_LOCK sec18.6 / 18.10.2 / 18.12.3 三表逐条读出。

## 11. 执行对账（原计划 → 实际，截至 2026-07-04）

| 计划项 | 状态 | 实际结果与证据 |
|---|---|---|
| Phase 0（G0.1–G0.5） | ✅ | `4d680ab`；G0.1 允许 v2 基底、G0.2 翻 true；sec18.1–18.7 预注册入档 |
| Phase 1（R1.1–R1.3） | ✅ GREEN | `a2f813f`；pytest 全绿 + fidelity gate I1–I17 exit 0 + 冒烟产物全字段合规 |
| R2.1 证书 | ✅ PASS（方法学修订） | 差异化证书 PASS；sec18.9 throughput 三布局扫描完成；sec18.10 裁决：FSM-oracle 探针**废弃**，E1 本身承担 admissibility 判读（`2f5e356`） |
| R2.2 支持度探针 | ⤵ 塌缩为归档哨兵 | sec18.10.3：≤300ep×1seed，E1 后执行；其分叉功能由 E1 承担 |
| §10.5 顺手工程 | ✅ 全部落地 | E2/E3 开关 + 参照基线缓存 `0e42573`；`--reuse_replay` `9dfb258`；仅 K7（JAX-GPU 冒烟）未做 |
| E1 前置 CE | ✅（经 S27 修复重采） | 图 `0d0beba`；S27 修复 `930c37b`/`c1e112b`；audit：estimable 45→80、noop 占用 73.9%→13.8%；16 因子含全部终端 kind |
| E1 四臂决定性跑 | ⤵ 被 **E1-rev** 替换 | 见 §12.3；sec18.12 已签收（`b9dea6f`），pilot 决定性（`2ba8a1c`），全 wave 运行中 |

## 12. 链路结构变更（四项；均已预注册/留痕，无事后重释）

### 12.1 Phase 2 分叉点重构（sec18.10）
原「R2.1-oracle-admissibility + R2.2-CE 分叉」被 **E1-as-admissibility** 取代：
base_only 在 held-out 上 CCR ≥0.9 ⇒ 基底非判别；partner_id_q 与其余臂的分离度 ⇒ 推断
信息缺口是否存在。R2.1 差异化证书（已 PASS）仍是唯一硬前置。原 §3 的 R2.2 分叉描述作废。

### 12.2 S27 插入（计划外根因修复，台账 S27/D6/D7）
E1 前置 CE 两次 GraphCoverageError → 根因 = 行为推断器**支撑集冻结**（乘性 Bayes 更新
0×x=0 永久锁死 t=0 无效选项；实现 bug）。修复（支撑注入 `support_mix`）+ 现场复验
（claim 伙伴 9/9 次送餐 argmax=serve、终端质量 0.0→0.92）+ **CE 全量重采**。连带实测：
opt7/opt9 被动访问 weight_sum=0 —— 这是 U1（干预式 CE）的直接实证动机，非推测。

### 12.3 E1 → E1-rev（sec18.12，用户已签收）
- **E1(no-scaffold) 实测**：全部完成 run ego_correct ∈ {0,1}（partner 6–12），guard 判
  free-riding，无可部署 checkpoint。根因：P5 清理在 role_v1 config 里连带关掉了两个
  **ego-kind-only（P5-clean）脚手架**，E1 测的是这个真空，不是方法本身。
- **编排器事故（诚实记录 `19b653d`，我方 bug）**：单 `local` 语句吞位置参数 → 全部 run
  共享同一输出目录并互相 `rm -rf`；no-scaffold wave 仅存 aris 5 seeds + base 1 seed 的
  训练指标。编排器已重写（唯一目录 + 禁 cross-wipe）并用于 E1-rev。完整 no-scaffold
  消融补跑列为可选项（§16 K10）。
- **E1-rev pilot 决定性**：同 seed 同图，仅加回 P5-clean 脚手架 → ego 0→18、guard
  fail→pass（`2ba8a1c`）⇒ 全 wave 授权启动。
- E1(no-scaffold) 存活数据 = **脚手架消融基线**（sec18.12.4）：量化终端能力中脚手架
  贡献 vs 方法贡献，入论文诚实报告节。

### 12.4 创新轨道插入（sec18.11 / ICLR_UPGRADE_PLAN，`d17fedc`→`bb90b4e`）
用户指令：正确性为前提、创新性为目标、ICLR 为标尺。U1（targeted-start + 干预式 CE +
支持度证书）与 U2（FactorModeFilter，情节持久因子模式信念）设计规格已过 codex 评审
（rev1 全采纳）。终态 E1 加臂为**六臂**（+aris(mode-filter)、+干预图 vs 被动图对比）；
FCP/MEP spike 解除 E1 门控提前启动。**定位声明：当前 E1-rev 是 tier-1 诊断跑；8 月的
多臂跑才是论文主表**——这缓和了 E1-rev 单次结果的叙事压力，但不豁免其预注册判读。

## 13. 运行态快照（2026-07-04，wave 进行中；PID 1815648，修复版编排器）

**进度**：9/25 final（aris_bellman 5/5、base_only 4/5）；3 in flight（base s4、gru s0/s1）；
13 排队（gru s2–4、flat_factor 0–4、partner_id_q 0–4）。3 GPU 槽（{4,5,6}），~50–90min/run
⇒ 预计再 **~5–7h** 完成。

训练期指标（**仅调度用**；selCCR = 选中 checkpoint 在 train-partner greedy 验证上的
ego_correct_completion_rate）：

| arm | guard | selCCR（seed 序） | ego 送餐（训练期区间） |
|---|---|---|---|
| aris_bellman | 4/5 pass（s3 fail） | 0.4 / 0.6 / 0.8 / – / 0.8 | 6–18 |
| base_only（s0–3） | 4/4 pass | 1.0 / 1.0 / 0.4 / 1.0 | 23–33 |

两条纪律读数（现在就能定的只有这两条）：
1. **sec18.12.3 第一行已触发**：≥2 臂各 ≥3/5 seeds 出现终端能力（guard 过、可部署
   checkpoint 存在）⇒ **E1-rev 即决定性 run**，判读走 sec18.6 + 18.10.2 全表。
2. base_only 训练期强势**在预期分支内**（对训练伙伴过拟合恰是 ZSC 论点的对照面）。若
   held-out 上 base CCR ≥0.9 → 触发 18.10.2 第一行（基底非判别）→ 走 §14.3 分支映射；
   在那之前不做任何方向性解读。aris s3（guard fail）按 I 门纪律不入表，但计入
   sec18.12.3 co-primary「terminal competence rate」分母。

## 14. 调整后的剩余链路（本补充的核心输出）

### 14.0 家务（wave 完成前）
- ✅ e1.yaml `relevance_semantics: legacy_id_pair` 修正补提交（`ee24ccd`；远端启动时已生效，
  本地补齐 provenance）。
- **方法层冻结**：E1-rev 判读完成前不动任何方法代码（sec18.11 禁令：不得以决定性 run
  的结果调 U1/U2 设计）。
- （可选，10 分钟）K7 JAX-GPU 冒烟——若 ptxas 已修则 eval 全线提速，排在 eval 启动前验。

### 14.1 wave 完成 → 归档（Type-A，自判可过）
- 全 25 run 的 metrics.json / free_rider_guard / checkpoint_selection / 训练日志
  tar 回 `review_bundles/phase3_E1rev_<date>/`；EXPERIMENT_LOG 填 wave 汇总行。
- 汇总表按臂给出：guard 通过率（= competence rate co-primary）、可部署 checkpoint 数。

### 14.2 两段式 held-out eval（决定性判读；本节取代 §10.3 评估段）
```
对象:   全部 deployable checkpoints（当前预计 ~20）
第一段: 25ep × 2 held-out（heldout-handoff-alternate-yield, heldout-resource-server-claim）
        × 全部 ckpt；--baseline_cache_dir 共享缓存；全部 ckpt 固定同一 eval --seed
        （缓存 key 含 seed，同臂跨 ckpt 复用参照基线）；3–4 槽并行
        墙钟估计: 20 ckpt × ~68min(25ep×2伙伴, 按 §9.1 实测 163s/ep 折算) ÷ 3.5 槽
        − 基线缓存摊销 ⇒ ~6–8h
硬门:   evidence_policy=behavior_inferred_v1（require_inferred 强制）; oracle_source_count=0;
        missing 率阈内; reward_scale_verified=true; I10–I17 全绿 —— 任一 false 不入表
注:     脚手架是训练期 reward shaping，eval 的 headline（held-out ego_correct_completion_rate）
        不含 shaping 项 —— E1-rev 与 E1 的 eval 口径天然同一
初判:   sec18.6 五分支 + sec18.10.2 admissibility 四行 + sec18.12.3 → 定决胜臂
第二段: 仅决胜臂补 50–100ep（终表口径）
签收:   三表逐条 + 伪影自检清单（ROOTCAUSE_REVIEW_PLAN §3.2）→ codex 复核 → 用户 Type-B
```

### 14.3 判读后分支映射（预注册表 → 链路走向；不新增判读规则，只映射后续动作）
| 判读结果（按 18.10.2 / 18.6） | 链路走向 |
|---|---|
| 基底判别 且 ARIS > flat > base（CI 分离） | 主线确认：E2/E3 → U1/U2 实现 → 8 月多臂正式跑 → Phase 4/5 |
| 基底判别 且 ARIS ≈ flat（或 ≈ base） | 按预注册收窄主张、如实入档；U1/U2 成为方法改进主战场（其动机独立成立，见 §12.2/§12.4）；FCP spike 提前 |
| base held-out CCR ≥ 0.9（基底非判别） | asymm×v2 降为机制分析布局；headline 转 FCP/MEP 群体线（T3.2）；E1-rev 全部结果作 diagnostic 入档 |
| partner_id_q ≈ 各臂（无信息缺口） | 推断主张在该基底不可读；routing/简化主张仍按 sec18.6 读；用 sec18.9 已有扫描数据选替代布局 |
| CI 重叠但方向一致 | +5 seeds 复跑一次再裁（§9.2 预注册行；先于任何"无差异"结论） |

### 14.4 E1-rev 判读后的执行序列（顺序，各自门控）
1. **E2 + E3 消融**：开关已落地（`0e42573`）；决胜 checkpoint × zeroed/inferred 两模式、
   {win4-off, win8-off, persistent} × 3 seeds → sec18.7 两表。
2. **R2.2 归档哨兵**（≤300ep × 1 seed，过夜档；关闭 serve-CE"哨兵零 vs 测量零"问题）。
3. **U1 实现**（T2.1，~2–4 天）+ **U2 实现**（T2.2，~4–6 天）+ C1 命题形式化 +
   /novelty-check ×2（并行，~2 天）。
4. **FCP/MEP spike**（1–2 天，与 3 并行；K2 退役；全量群体训练仍待 spike 判读 + 用户批准）。
5. **8 月多臂 E1 正式跑**（六臂，ICLR_UPGRADE_PLAN §5 读出）→ Phase 4（E4–E7）→
   Phase 5（终表 + 盲测）。原 §4–§6 的实验定义与纪律不变，锚点从"E1"移到"多臂 E1"。

## 15. 时间线 v2（对 §9.6 的校准；ICLR ~9 月下旬）

```
07-04        wave 完成 + 归档（14.1）
07-05        stage-1 held-out eval（6–8h）+ 三表初判 + codex/用户 Type-B 双签
07-06        stage-2 决胜臂补 ep + E2/E3 消融启动 + R2.2 哨兵过夜
07-07..07-18 U1/U2 实现与远程验证 + FCP spike + C1 形式化/查新（ICLR_UPGRADE_PLAN tier-2/3）
08 月        多臂 E1 正式跑（scripted v2 机制线 + FCP headline 线）+ E4–E7
09 月上旬    终表（5 seeds × 2 CE seeds × 50–100ep, CI 全报）+ 盲测 split + 写作
09 月下旬    提交（先过 /paper-claim-audit + /citation-audit + kill-argument）
```

## 16. 风险登记更新（对 §9.5）

| # | 状态 | 说明 |
|---|---|---|
| K1 | 关闭（重构） | R2.1-FAIL 分叉被 sec18.10 E1-as-admissibility 取代 |
| K2 | 缓解中 | FCP spike 解除门控（sec18.11(4)），排入 14.4(4)；全量投入仍待批 |
| K3 | 已缓解 | 两段式 + 并行 eval + 基线缓存全部落地（`0e42573`） |
| K4 | 关闭（降级） | R2.2 → 归档哨兵；支持度问题的真正解法 = U1 targeted-starts |
| K5/K6 | 维持 | 红线不变：正式 CE 只许顺序采集；正式跑只许 graph_path 分支 |
| K7 | 未验 | JAX-GPU 10 分钟冒烟排在 eval 前（14.0） |
| **K8 新** | open | guard-fail 种子折损（aris s3，1/5）→ 功效下降。缓解：§9.2 预注册 +5 seeds 行；guard-fail 计入 competence-rate co-primary 而非静默丢弃 |
| **K9 新** | open | base_only 训练期强势 ⇒ held-out 可能落「无 ARIS 特异优势」或「基底非判别」分支。**处置已预注册**（14.3 全表）；ICLR 主线（U1/U2 + FCP）的动机独立于该结果，不依赖其为正 |
| **K10 新** | open（已量化） | 脚手架依赖批评（G2 旧账）：E1 vs E1-rev = 实测脚手架消融，但 no-scaffold 侧因编排器事故仅存 aris 5 seeds + base 1 seed；若审稿需完整对照，补跑 no-scaffold wave（~10 GPU·h，可选，用修复版编排器） |
