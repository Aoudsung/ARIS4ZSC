# EXPERIMENT_CHAIN_PLAN.md — 从当前状态到论文级结果的完整实验链路

**Status:** READY（静态验收全绿，待 Phase 0 两项决策后启动）· **Date:** 2026-07-02
**前提**：[FINDINGS_LEDGER.md](FINDINGS_LEDGER.md)「静态验收 PASS」段；METHOD_LOCK sec17 修复锁定条目。
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
