# FINDINGS_LEDGER.md — 根因裁定发现台账（唯一真相源）

> 治理精简 2026-07-08：本文件的门已按 GOVERNANCE_CUTLIST.md 处置；加/减门须过 OPERATING_CONSTRAINTS.md §7 退休阀门。

**Status:** ROUND-1 已收敛，7处分歧全部裁决(采纳codex) · **执行暂缓**（用户裁定"都先不做"）
**Date:** 2026-07-02 · **Branch:** `rc-rootcause-fix`

> 规则（[ROOTCAUSE_REVIEW_PLAN.md](../../archive/2026-07_asymm-role-v2-line/ROOTCAUSE_REVIEW_PLAN.md) §2.1）：后续一切讨论、提交、重跑结论
> **只能引用本表 ID**；不在表里的问题视为不存在；不想修的问题必须记 `wontfix+理由`，不能
> 静默丢弃。历史记录**追加不覆盖**——分歧的原始双方论证保留在文末，裁决作为追加记录，不删除
> 争议过程本身。

**状态图例**：`待codex` · `已裁定` · `分歧`(不再使用，本轮已清零) · `待人类裁决(新发现)` ·
`修复中` · `已提交` · `已验证` · `已关闭` · `wontfix` · `阻塞于治理`

**快照统计**：51 条。codex 本轮覆盖 50/51（仅W8待下一轮）。**分歧：0**（7处本轮全部裁决，
均采纳codex修正）。**治理未决：0**（§5.1 已于 2026-07-08 归档——该治理决策已在 Phase 0 G0.1/G0.2 闭合，见下）。

**当前动作状态：暂停**。用户明确"都先不做"——不打包clean zip、不进
`ROOTCAUSE_FIX_EXECUTOR_PROMPT.txt`、不再调用codex。本文件此次更新仅做**决策固化**（把已经
做出的判决写入台账+更新计划文档的修复顺序），不触发任何代码/远程动作。下一次推进由用户发起。

---

## §5.1 治理决策——已闭合（2026-07-08 归档，依据 GOVERNANCE_CUTLIST.md）

原状态（2026-07-02）："等我修订完代码再决定"——即 role_conditioned_v1 工作区伙伴库重设计是否
可用（是否违反 7/1 合成数据禁令）曾被"明确延后"，并把 W 系列整体标记 `阻塞于治理`、把修复顺序
步骤6 前置于本决策。

残留事实（留档）：该决策已在 Phase 0 裁决闭合——role_conditioned_v2 有条件允许作受控机制诊断基底
（G0.1，headline ZSC 主张走 FCP/MEP 群体），`require_ego_delivery_selection` 置 true（G0.2）；详见
下方 Phase 0 执行记录与 METHOD_LOCK sec18.1–18.2。因此本节不再作为"阻塞于治理"的启动门，W 系列
早先的"阻塞于治理"标记同被此裁决取代。全节叙事已归档（见文末"已归档 / 已合并"段）。

---

## codex 全量优先级排序（1–37，跨支柱+次级统一排序，供未来排期参考）

```
1 P1  2 P4  3 P5  4 P3  5 P2  6 S20  7 S17  8 W2  9 S18  10 S1
11 S8  12 S12  13 D2  14 S16  15 S23  16 S10  17 S11  18 S3  19 S2  20 W3
21 S9  22 S22  23 S24  24 S21  25 S6  26 S7  27 W5  28 S19  29 W4  30 S4
31 S26  32 W1  33 W6  34 D1  35 D3  36 D4  37 D5
（W7=REFUTED无排名；E1-E9=豁免无排名；NEW-1/2/3=新发现未排名）
```

**修复顺序已按分歧7裁决更新**——详见 [ROOTCAUSE_REVIEW_PLAN.md](../../archive/2026-07_asymm-role-v2-line/ROOTCAUSE_REVIEW_PLAN.md) §2.3
（本次同步更新，P5提前、评估完整性提前、新增治理门步骤）。

---

## 支柱（P）

| ID | 断言（我方原判） | 严重度(我方) | codex verdict | 人类裁决(2026-07-02) | 处置 | 目标提交/验证 | 状态 |
|---|---|---|---|---|---|---|---|
| P1 | 脚本伙伴真实执行选项被直接注入证据流(train/eval/CE)；行为推断器死代码 | 高 | CONFIRMED·HIGH·#1 | **采纳codex**：Q-B命题措辞软化为"P1使parity更可能、使'习得推断'主张不可采信，但不构成parity的机械证明——效应量只能由去oracle重跑测定"（不是"P1⇒parity"的机械蕴含） | fix | 步1(去oracle)，不变 | 已裁定 |
| P3 | asymm的`serve_soup CE=0.000`是支持度不足哨兵值非测量 | 高 | CONFIRMED·HIGH·#4 | 无分歧，维持原判 | fix | 步3(CE支持度sidecar+重测) | 已裁定 |
| P4 | 信念=4步滑窗每次从零隐状态重编码，无跨决策记忆 | 高 | CONFIRMED·HIGH·#2 | 无分歧，维持原判 | fix | 步2(信念持久化)；含 LDS-B6：每个正式 config 须显式钉住 belief_persistence | 已裁定 |
| P5 | 奖励/探索/回放播种以伙伴`terminal_policy`真值为条件 | 高 | CONFIRMED·HIGH·#3 | 无分歧(措辞已一致)；**排序采纳codex**：从原步5提前到步4 | fix | **步4**(原步5，提前) | 已裁定 |

> **P2 已归档（2026-07-08）**：P2 说的"恢复项 +100 吞没角色奖励⇒伙伴同一"只存在于历史 HEAD 版本；
> 当前工作区树已重缩放、无该吞没算术，属 no-op，无可执行修复。残留事实：历史 HEAD 的相关数字为
> artifact-suspect，已隔离/作废。详见文末"已归档 / 已合并"段。

---

## 次级实现问题（S）— 本轮全部codex确认/降级微调，无用户分歧，维持原判

| ID | 断言 | 严重度(我方) | codex verdict | 状态 |
|---|---|---|---|---|
| S1 | 失败通道漏`"max_steps"` | 中 | CONFIRMED·MED-HIGH·#10 | 已裁定 |
| S2 | GRU吃零填充为证据 | 低 | CONFIRMED·MED·#19 | 已裁定 |
| S3 | 失败步重复路由 | 低 | CONFIRMED·MED·#18 | 已裁定 |
| S4 | pot_became_cooked≡ready重复通道 | 低 | CONFIRMED·LOW·#30 | 已裁定 |
| S6 | 裸ce_sampler不过滤held-out | 中 | REFINED·LOW-MED·#25(降级,非正式路径) | 已裁定 |
| S7 | 批处理CE local-return塌缩 | 中(潜伏) | CONFIRMED·LOW-MED·#26(降级,同上) | 已裁定 |
| S8 | 覆盖率门统计团队级事件 | 中 | CONFIRMED·MED-HIGH·#11 | 已裁定 |
| S9 | 跳过零与测得零不可区分 | 中高 | CONFIRMED·MED·#21 | 已裁定 |
| S10 | γ/horizon硬编码 | 中 | CONFIRMED·MED·#16 | 已裁定 |
| S11 | `--sparse_ce_support`对目标门不可见 | 中 | CONFIRMED·MED·#17 | 已裁定 |
| S12 | preflight回退用team credit | 中低 | REFINED·MED-HIGH·#12(升级) | 已裁定 |
| S16 | global_gru诊断形状崩溃 | 中 | CONFIRMED·MED-HIGH·#14 +实锤log | 已裁定 |
| S17 | `--allow_diag_skip`短路全部完整性门 | 中高 | CONFIRMED·HIGH·#7(升级) +实锤 | 已裁定 |
| S18 | `reward_scale_verified`永假 | 中 | CONFIRMED·MED-HIGH·#9 +实锤；含 NEW-3（已归档 role-v1 eval 报告 `reward_scale_verified=false`，与 S18 同因）——排除口径保留：归档 role-v1 数字视为诊断性非正式结果，不进任何主张证据链 | 已裁定 |
| S19 | diagnose_traces.py三缺陷 | 中 | CONFIRMED·LOW-MED·#28 | 已裁定 |
| S20 | completion计入伙伴/错误送餐 | 中 | CONFIRMED·HIGH·#6(大幅升级) | 已裁定 |
| S21 | eval种子惰性+跨伙伴复用序列 | 中 | CONFIRMED·MED·#24 | 已裁定 |
| S22 | asymm.yaml的Q臂间bound不匹配，role_v1已修 | 低(半闭) | REFINED·MED·#22——**与我方"半闭"判断完全一致** | 已裁定(双方一致) |
| S23 | train_partners未设置静默回退 | 中 | CONFIRMED·MED·#15 | 已裁定 |
| S24 | provenance缺失仅告警 | 低中 | CONFIRMED·MED·#23 | 已裁定 |
| S26 | event_summary字段重复声明 | 极低 | CONFIRMED·LOW·#31 | 已裁定 |

---

## 工作区伙伴库重设计（W）— §5.1 治理已闭合（2026-07-02），下列 `阻塞于治理` 标记已作废；活的可用性把关归 R2.1 伙伴差异化证书

| ID | 断言（我方原判） | 严重度(我方) | codex verdict | 人类裁决(2026-07-02) | 处置 | 状态 |
|---|---|---|---|---|---|---|
| W1 | 恢复项(+100)未随角色奖励(±4000)重缩放⇒可能活锁 | 中高 | REFUTED该活锁论证·LOW·#32："+4000恒>−1000+100=−900，swamping论证在当前树FALSE" | **采纳codex**：降级LOW，不再是伙伴库修复的阻塞项，仅剩tier内平局微小偏置(可选打磨) | wontfix(活锁风险)/optional(tie-break打磨) | 已裁定(阻塞于治理，非阻塞于本条) |
| W2 | held-out claim伙伴与训练伙伴近重复 | 高 | CONFIRMED·MED-HIGH·#8 | 无分歧，维持原判 | fix(若治理放行) | 已裁定(阻塞于治理) |
| W3 | yield层内排序⇒蹲守瓶颈非备菜 | 中 | REFINED·MED·#20："代码支持该风险，但需执行探针证明持续蹲守" | **采纳codex**：确定性分级从"算术确定"移至"需执行探针"（记录供未来轮次/预注册使用；REVIEW_BRIEF.md本身作为round-1历史记录不回改） | fix+探针验证(若治理放行) | 已裁定(阻塞于治理) |
| W4 | alternate瓶颈策略不真正交替 | 低 | CONFIRMED·LOW-MED·#29 | 无分歧 | fix(若治理放行) | 已裁定(阻塞于治理) |
| W5 | _congestion_penalty死连词 | 低 | CONFIRMED·LOW-MED·#27(细节小修正:"忽略伙伴到目标距离") | 无分歧 | fix(若治理放行) | 已裁定(阻塞于治理) |
| W6 | 空有效集回退指向option0非noop | 极低 | CONFIRMED·LOW·#33 | 无分歧 | fix(若治理放行) | 已裁定(阻塞于治理) |
| W7 | press_recipe_button对所有伙伴恒被tier-banned | 低 | **REFUTED**："按钮选项合法性键于BUTTON_RECIPE_INDICATOR实体，普通布局本就无按钮选项" | **采纳codex证伪** | **wontfix**：非真实缺陷 | **已关闭(wontfix)** |
| W8 | _path_length_penalty死分支(cosmetic) | 极低 | 未覆盖 | — | — | 待codex(下一轮) |

---

## 设计层局限（D）

| ID | 断言 | 严重度(我方) | codex verdict | 人类裁决(2026-07-02) | 状态 |
|---|---|---|---|---|---|
| D1 | ~~伙伴差异性前提从未写为可检验前提，accept_layout()从未操作化该指标~~ | 高 | REFINED·LOW-MED·#34："此断言已过时(stale)：当前`layout_diagnostics.py:89-112`确已包含`partner_return_variance_proxy`" | **采纳codex，更正断言**：`accept_layout()` **已经**包含`partner_return_variance_proxy`代理指标；原"从未操作化"表述不成立。**存活的更精确批评已转移到D2**（该代理只是训练前粗粒度门，无训后复检——即真正的盲点） | 已裁定(断言已更正，见下方[修正记录]) |
| D2 | reference_base_gap门只训练前检查，无训后复检 | 中 | CONFIRMED·MED-HIGH·#13 | 无分歧；D1软化后，**D2成为本类设计局限中存活的核心批评** | 已裁定 |
| D3 | 闭合模式集⇒仅重组式泛化，Exp2预期与非声明矛盾 | 中 | REFINED·MED claim-scope·#35，措辞一致 | 无分歧 | 已裁定 |
| D4 | CE估计量把可测性与占用率混同 | 高 | CONFIRMED·HIGH estimator·#36 | 无分歧 | 已裁定 · 降级为读 claim 前审计：读该 claim 时限定"支持相对"口径（S27 已由 P3 sidecar + 修复处理），不作启动门 |
| D5 | 价值充分性相对训练伙伴分布定义(鸡生蛋) | 中高 | CONFIRMED·MED claim-scope·#37 | 无分歧 | 已裁定 |

**[D1修正记录 2026-07-02]**：round-1我方综合曾表述"提案从未把伙伴差异性前提写成可检验前提，
计划最接近的指标(partner-induced return variance)未进入验收代码"。codex复核`layout_diagnostics.py:
89-112`后指出该表述对**当前代码**已过时——`accept_layout()`确已包含`partner_return_variance_proxy`。
用户已裁定采纳此更正。**准确表述应为**："`accept_layout()`已含伙伴差异性代理指标，但该代理仅在
训练前(pretraining)粗粒度检查一次，无法阻止训后课程把base_only抬到天花板（这是D2，非D1）；
换言之，机制存在但时机/粒度不足，而非'完全缺失'。" 项目memory已同步更正
（`project_rootcause_adjudication_20260702.md`）。

---

## 豁免（E）— 全部存活，无分歧

| ID | codex verdict |
|---|---|
| E1-E9 | 全部 **CONFIRMED**("Exoneration stands")，含E5备注"见NEW-2关于选择门控方式"(关联新发现) |

---

## 新发现（NEW，来自 codex）— 待人类裁决（本轮未处理，非分歧，是全新待办）

| ID | 断言 | severity | 说明 | 状态 |
|---|---|---|---|---|
| **NEW-2** | checkpoint选择在free-rider guard判定**之前**就已保存`checkpoint.pt`；guard事后标`fail`不撤回文件(除非`select_final=true`) | **高**(与P1/P4/P3/P5同级优先) | 意味着历史上任何guard=fail的跑，其`checkpoint.pt`可能仍是被拒绝的free-riding checkpoint。**建议在信任任何历史"guard: fail"跑的结论前，先做一次产物审计**（哪些checkpoint.pt对应fail verdict、是否被下游引用过） | 待人类裁决(新发现，优先级高，未纳入本轮6项分歧) |

> NEW-1、NEW-3 已于 2026-07-08 处置：**NEW-1 归档**（preflight 回退用错伙伴池/信用目标——已修复并 ACCEPTED，代码护栏独立存在，见下方"静态验收 PASS"段）；**NEW-3 合并入 S18**（同 `reward_scale_verified` 根因；"归档 role-v1 数字视为诊断性非正式结果"的排除口径已并入 S18）。NEW-2 保留。详见文末"已归档 / 已合并"段。

---

## 修复顺序——步骤叙事已归档（2026-07-08，依据 GOVERNANCE_CUTLIST.md）

原步1–步7 的顺序叙事已归档：步骤叙事本身从未抓到任何失败，且其中"任何决定性重跑前必过""任何
新主张声明前必过"等"拦启动"措辞与 OPERATING_CONSTRAINTS §7（默认拦 claim、不拦启动）相抵。

残留事实（留档）：步1–5 已完成实施与静态验收（见下方"静态验收 PASS"段，P1/P4/P5/P3/S16/S17/S18/
S19/S20 均已 `已验证(静态)`）；两个仍活的残留——W 伙伴库重设计、preflight/selection 治理门——各自
另立门跟踪（前者见 W 表 + r2.1 差异化证书，后者见 S12/S23/NEW-2 各自归属），不再以统一"修复顺序"
的启动门形式存在。原七步顺序的一行摘要见文末"已归档 / 已合并"段。

---

## 下一步（用户主导，本会话不再推进）

- **执行路径A（进`ROOTCAUSE_FIX_EXECUTOR_PROMPT.txt`）**：暂缓。
- **执行路径B（clean-zip二轮独立评审）**：暂缓。
- **§5.1 治理决策**：已闭合（2026-07-02 Phase-0 G0.1/G0.2 裁定），见上方专节与文末归档记录。活的可用性把关现由 R2.1 伙伴差异化证书承担。
- 待用户发起时，直接引用本文件的"修复顺序"章节即可，无需重新裁决已解决的7项分歧。

---

## 验收记录（2026-07-02 晚，用户宣布"已完成审计与代码修订"后的静态验收）

**结论：修复顺序步1–5 全部未实施；当前树不可用于正式训练/测试。**

时间线证据：全部源码 mtime ≤ 15:58（`option_executor.py` 6/28、`factor_belief.py`/`replay.py`
6/24、最晚 `train_aris.py` 15:58），而 round-1 评审（五路审计 19:16 + codex 19:27）**就是对
这棵树做的**——用户所指"代码修订"= CODEX_IMPL_SPEC v1–v4（mtime 7/1 20:36 – 7/2 15:52），
先于评审完成，评审 verdict 已经覆盖它。评审之后无任何源码改动。

| 必修项 | 验收 grep 结果 | 状态 |
|---|---|---|
| P1 去oracle | `option_executor.py:63-64` 仍传 `partner_action.option_id/option_dist`；推断器仍无非测试实例化 | **未修** |
| P4 信念持久化 | `evidence_window: 4` 未变；`factor_belief.py` 仍零隐态重编码（文件自6/24未动） | **未修** |
| P3 CE支持度 | `run_ce_pipeline.py:210,224` 仍硬编码 `min_weight=20.0`；`weight_sum` 仍算完即弃 | **未修** |
| P5 课程去oracle | `role_contrib_team` 仍激活；**面反而扩大**：v3/v4 新增 `ego_terminal_penalty_under_claim=0.3`（`sparse_credit.py:123`，键真实policy）+ 真实 `terminal_policy` 新传入选项选择调用（`train_aris.py:452`） | **未修且扩大** |
| S17 完整性门 | `allow_diag_skip` 接线未变 | **未修** |
| NEW-2 checkpoint时序 | 选择仍 `_val_mean > best_greedy_return` 先行，guard 事后 | **未修** |

**NEW-4（新登记，治理层）**：CODEX_IMPL_SPEC v1–v4 期间发生了 METHOD_LOCK 冻结之外的**方法层
干预**（v3/v4 relevance 语义改为 `ego_complement_projection`、新奖励惩罚项、按远程 eval 数字
迭代调参），且：(a) 无预注册（违反计划§3.1）；(b) 运行在 role_v1 伙伴上（§5.1 治理未决）；
(c) 证据/奖励通路带 P1/P5 污染。**v1–v4 规格内引用的全部远程结果（ego/prt 翻转、rmr 对比）
只能标记 diagnostic-only，不得作为任何主张证据。** 状态：待人类裁决。

处置指向：修订要求见本次会话交付（步1–5+NEW-2 逐项验收判据）；先落基线提交再修
（2478 行未提交 diff 违反增量可归因纪律）。

**追记（2026-07-05 文档清理）**：CODEX_IMPL_SPEC v1–v4 四个文件已按用户裁决从工作树
删除（清理 commit：`chore(docs): archive completed review bundles + remove
quarantined/stale docs`，git 历史可完整取回）。隔离裁决本身**不变**——其中的数字仍然
永不进入任何主张证据链。role_v1 运行的原始产物已压缩存于
`archive/raw/role_conditioned_v1_20260702_102332.tar.gz`（见 MANIFEST）。

---

## 修复执行记录（2026-07-02，当前分支 baseline 之后）

Source commit boundary: `wip: pre-fix baseline` 之后的独立修复 diff。状态 `STATIC-PATCHED`
表示源码/文档已修订并可进入静态/远程非实验验证；不表示训练、eval、CE 生成、远程实验或科学
Type-B 裁决已经完成。

| ID | 当前处置 | 静态修复状态 | 验收边界 |
|---|---|---|---|
| P1 | must-fix | STATIC-PATCHED | `option_executor.py` 在 `extract_event` 前剥离 scripted `option_id/option_dist/confidence`；train/eval/CE 使用 behavior-inferred partner-option 语义。 |
| P4/S2 | must-fix | STATIC-PATCHED | `EvidenceBuffer` 持久化 factor-belief hidden state；replay 存 masks/lengths；belief encoder 使用 mask，零填充不再当证据。 |
| P3/S9/S10/S11 | must-fix | STATIC-PATCHED | CE 估计输出 `weight_sum/estimable_mask/skipped_mask/measured_zero_mask` sidecar；`min_weight/gamma/horizon` 来自 config/metadata；unsupported zero 不再可解释为无外部性。 |
| P5 | must-fix | STATIC-PATCHED | 主方法 reward/exploration/replay/eval/checkpoint path 不消费 true `terminal_policy`；`role_contrib_team`、role exploration、role replay、terminal replay seed、`ego_terminal_penalty_under_claim` 只允许显式 oracle ablation/curriculum baseline。 |
| S17 | must-fix | STATIC-PATCHED | `allow_diag_skip` 不再跳过 forced-noop、evidence policy、oracle-source、missing-evidence 等 hard integrity checks。 |
| S20 | must-fix | STATIC-PATCHED | completion 拆分 ego/team/partner/wrong，headline 使用 ego-owned correct completion。 |
| NEW-2 | must-fix | STATIC-PATCHED | `checkpoint.pt/checkpoint_best.pt` 发布前先检查 ego-owned delivery eligibility；guard 不再只是事后标 fail。 |
| NEW-4 | governance | QUARANTINED | CODEX_IMPL_SPEC v1-v4 与其远程数字仅 diagnostic-only；不得进入支持/反驳主张证据链。 |

剩余边界：P2/W/D 伙伴基底治理仍待 Type-B 人类裁决；`role_conditioned_v2_candidate`
最多是 benchmark-v2 candidate，须先有 partner-differentiation certificate 后才可用于正式主张。

Reviewer follow-up: subagent `Sartre` flagged four repair blockers after the first
repair commit: unmasked CE `.npy` use in training, CE support params not enforced,
detached P4 belief filter, and stale `checkpoint.pt` survivability. The follow-up
diff fixes all four and records the rerun in
`provenance/CURRENT_TREE_ROOTCAUSE_REPAIR_VALIDATION_20260702.md`.

---

## 静态验收 PASS（2026-07-02 深夜，3 路深度复核 d1633ae+033fd1c）

三个独立复核 agent 对全部必修项做了对抗性验证（含反模式搜查：oracle 搬家/直通推断器/
未来泄漏/只加门不修机制/config 静默重开）。**结论：全部 ACCEPTED，零拒绝级 blocker。**

| 复核域 | verdict | 关键确认 |
|---|---|---|
| P1 去oracle | **ACCEPTED** | `act()`不发射真值；executor 剥离+行为推断重标注；confidence 语义修正；eval 硬门(oracle_source_count>0→fail，不可被 allow_diag_skip 绕过)；无搬家路径；partner_id 只进声明的 partner_id_q 基线 |
| P4/S1/S2/S3 | **ACCEPTED** | 隐状态仅 episode 边界重置；无未来泄漏（transition 字段选项开始前快照）；梯度流正确（**033fd1c 承重**：d1633ae 单独存在 GRU-对-TD 不可训练 + eval 窗口双计，由它修复）；train/eval 共用实现；CE 无信念路径；`"max_steps"`入失败集、`option_invalid`已删、S3 boundary-row 替代重复路由、S2 掩码生效 |
| P5 | **ACCEPTED** | 守卫覆盖全部4个config可达oracle机制(无flag即raise)；`_select_option`签名已无`partner_terminal_policy`；reward 三层剥离(train:3000/sparse_credit:155/ce_sampler:415,984)；唯一残留 protocol 读取是 curriculum_group 采样分组(数据分布设计，非证据/奖励) |
| P3 生产者+消费者 | **ACCEPTED** | min_weight/γ/horizon 全部 config 化并入元数据；`weight_sum/estimable_mask/skipped_mask/measured_zero_mask` sidecar；masked npy 正式消费 + unmasked 审计；train 侧再掩码+目标门 fail-closed，**无 legacy 逃逸路径** |
| S16/S17/S18/S20 | **ACCEPTED** | gru 诊断 unsupported_method 优雅记录；硬门清单明确(forced_noop/evidence_policy/observed_dist/missing/oracle_source)；provenance 哈希口径端到端对齐(train子集)；completion headline=ego_sole_correct |
| NEW-1/NEW-2/S23 | **ACCEPTED** | preflight 伙伴子集+credit 修复；checkpoint 资格前置到保存之前+陈旧产物清理+启动接线检查；train_partners 强制显式(无split声明豁免需显式flag) |

**跟进项（非阻塞，登记为 F 系列）**：
F1 批处理CE推断器按选项重置vs顺序按episode(语义不一致) · F2 CE不读`evidence.partner_option_inference`config ·
F3 死代码`build_/attach_behavior_option_inferencer`+未用import · F4 `partner_option_known`通道退化恒1.0 ·
F5 router对oracle-like只计数不阻断(训练侧无运行时门，eval有硬门——纵深防御备注) ·
F6 failure-boundary行presence位与snapshot-mask语义小失配 · F7 bootstrap用在线信念模型(无target网络，标准做法备注) ·
F8 refine正值-only幸存偏置仍未文档化(ce_sampler.py:812) · F9 train侧greedy`completion_rate`仍team语义(建议改名或ego-sole重导出) ·
F10 gap-proxy回退缺credit_params(layout_diagnostics.py:132-144) · F12 fidelity gate**工具**未实现I10-I17机械检查(仅文档声明) ·
F13 evidence_window 4vs8消融值得列入实验链 · F14 W3±1500伙伴行为重设计随de-oracle捆绑(已被benchmark-v2 candidate框架覆盖，仍需certificate)

**待用户决策（进入正式实验前）**：
D-A `require_ego_delivery_selection`（role_v1/role_v1_novb 曾为 **false**，asymm/armA/armB 为 true）
——**已合并入 S20/NEW-2 自我交付族**并于 Phase 0 G0.2 置 **true**（书面豁免路线被否决）；本项已闭合，
2026-07-08 治理精简折入该族，见文末"已归档 / 已合并"段。
D-B **§5.1 治理决策现在成熟**（用户原话"等我修订完代码再决定"——代码修订已完成并验收）：
role_conditioned_v2 candidate 能否作为 benchmark-v2 使用（须先过 partner-differentiation certificate）。

**状态推进**：P1/P3/P4/P5/S1/S2/S3/S16/S17/S18/S20/S23/NEW-1/NEW-2 → `已验证(静态)`；
远程 I10-I18 验证 = 实验链 Phase 1（见 [EXPERIMENT_CHAIN_PLAN.md](../../archive/2026-07_asymm-role-v2-line/EXPERIMENT_CHAIN_PLAN.md)）。

---

## Phase 0 执行记录（2026-07-02，静态验收 PASS 之后）

用户裁决 + 执行完成（全部静态，无实验运行）：

| 项 | 裁决/动作 | 状态 |
|---|---|---|
| **D-B (G0.1)** | `role_conditioned_v2` **有条件允许**：仅作受控机制诊断基底（benchmark-v2-diagnostic），须过 R2.1 certificate；headline ZSC 主张走 FCP/MEP 群体，其投资以 E1 信号为门。全文 METHOD_LOCK sec18.1 | **已裁决** |
| **D-A (G0.2)** | `require_ego_delivery_selection` → **true**（role_v1 + novb，带语义注释）；书面豁免路线被否决（contrib_team 对 prep-only 在 claim 伙伴送餐上仍付费，mean-return 无法证明能过滤 always-prep 塌缩）。METHOD_LOCK sec18.2 | **已裁决+已改** |
| F8 | refine 正值-only 幸存偏置：代码注释 + metadata 标志 `bootstrap_positive_only_survivorship_bias: true` | 已修 |
| F9 | greedy validation 指标拆分：`team_delivery_episode_rate` + `ego_correct_completion_rate` 取代裸 completion_rate；checkpoint_selection/metrics 两处消费者同步；eval 侧读取方（rc2*_reachability 读 eval 聚合）不受影响 | 已修 |
| F10 | `estimate_reference_base_gap_proxy` 补 `credit_params/terminal_progress/cost_per_step/exclude_terminal_progress` 线程，caller 从 config 取值（与 CE fallback 同模式） | 已修 |
| F3 | 删除死代码 `build_/attach_behavior_option_inferencer` + evaluate_aris 未用 import（eval 实际用 `make_behavior_option_inferencer`，已验证） | 已修 |
| F12 | fidelity gate 工具实现 **I10–I17 机械检查**（同 I1–I9 风格：静态 token/AST 绊线，歧义倾向 WARN）；当前树 **17/17 PASS，exit 0**；`FIDELITY_GATE.{md,json}` 自此由工具生成；I18 为过程级，由 EXPERIMENT_CHAIN_PLAN Phase 3–5 + METHOD_LOCK 追踪（sec18.3） | 已修+已验证 |
| G0.4 | 预注册入档：R2.1/R2.2/E1/E2/E3 全部"结果→结论"分支表 → METHOD_LOCK sec18.4–18.7 | 已入档 |
| G0.5 | `EXPERIMENT_LOG.md` 骨架创建（记录模板 + 引用预注册义务 + NEW-4 排除） | 已建 |

未在本批处理（保持原状态）：F1/F2（批处理 CE 一致性——正式路径为顺序，非阻塞）、
F4（vacuous 通道）、F5（router 纵深防御备注）、F6（presence-bit 失配备注）、F7（无 target 信念网络备注）、
F13（窗口消融→已列入 E3）、F14（→由 R2.1 certificate 覆盖）、W8（下轮 codex 复核）。

**Phase 0 完成。下一步 = Phase 1（远程基础设施验证 R1.1–R1.3），需用户授权远程执行。**

---

## Phase 1 执行记录（2026-07-03 凌晨，远程已授权）

**R1.1 ✅ R1.2 ✅ R1.3 ✅ — Phase 1 GREEN**（细节见 EXPERIMENT_LOG.md Phase 1 段）。

要点：六项完整性旗全绿（`reward_scale_verified=true` 首次真实验证通过——S18 死旗复活）；
P3 sidecar 与覆盖门在真实运行中按设计工作（覆盖门两次正确 fail-closed）；NEW-2 历史审计
155 个 metrics 扫描 → 3 个 guard-fail 幸存 checkpoint（全部 6/29 results_rcfix 时代，无结论
污染，已就地 QUARANTINE 标记）。

**Phase 1 新发现（登记）**：
- **R1-A（正式链路口径，重要）**：train 的 `graph_path` 分支在 train_aris:884 **无条件**跑覆盖
  门（对正式跑是正确行为）；`ce_path` 分支才受 `graph.require_task_stage_coverage` 控制，且
  走 sidecar 合并元数据（无 5 个 provenance 哈希 → S24 legacy 警告路径）。**规定：正式跑一律
  用 pipeline 产出的 graph.json（graph_path 分支）+ 覆盖完整图；ce_path+覆盖关 仅限冒烟。**
  已写入 CHAIN_PLAN §10 执行卡。
- **R1-B（可行性）**：CPU-JAX 下 eval 是墙钟大头（≈163s/ep 含参照基线摊销），E1 评估需两段式
  + 并行化（CHAIN_PLAN §9）。
- **R1-C（管线增强候选，非必需）**：run_ce_pipeline 无 replay 复用开关，重估计必重采集；
  R2.2 大样本探针按此计入预算，或加 `--replay_path` 复用（小改，待定 wontfix/defer）。
- **E2/E3 前置小实现（已在 §10 执行卡登记）**：E2 需 eval 侧"推断通道置零"模式开关；
  E3 需 `training.belief_persistence` 开关（window 臂只改 config 即可）。

---

## E1-CE 阻塞根因裁定（2026-07-03 深夜）——新实现 bug S27 + 两个设计放大器

**触发**：E1 前置 CE 管线（asymm×v2, 100ep×6伙伴, 12000 rows）两次 GraphCoverageError：
`plate_soup/serve_soup 无 above-eta CE 候选`。支持度审计显示终端选项作为**伙伴列**的
联合质量**精确为 0.0**（非低于阈值），73.9% 伙伴占用质量堆在 noop 列。

| ID | 断言 | 锚点/证据 | 分类 | 严重度 |
|---|---|---|---|---|
| **S27** | **行为推断器支撑集冻结**：`reset()` 把 belief 初始化为 `normalize(初始状态的有效选项)`；初始时终端选项无效 → belief=0；`update()` 是乘性 Bayes（`belief=normalize(belief×adjusted)`），0×任何=0 **永久锁死**。episode 中途变有效的选项永远无法进入支撑集 | `option_inferencer.py:40-42`(reset), `:103`(乘性更新)。**现场证据**：server-left-claim 一个 ep 送餐 9 次，推断终端质量恒 0.00000000，每次送餐 argmax=noop（远程实测 2026-07-03） | **实现 BUG**（P1 修复激活了死代码中的潜伏缺陷；P1 验收审"不耗 oracle/只用行为"，支撑集动力学不在验收单上——验收盲区） | **高**（阻塞 E1；污染 12000-row replay 的 partner dist——修复后须重采集） |
| **D6** | 互斥型协调外部性（who-serves）的反相关结构自我压制 (ego-terminal × partner-terminal) 联合占用质量——占用加权二阶 CE 对互斥类因子存在结构性盲区（D4 的机制精化） | 随机 ego 终端行 52/12000；对角 cell 结构性低质量 | 估计量设计局限（D4 精化） | 中（S27 修复后 (ego-prep × partner-serve) 列应恢复——claim 伙伴 9 次/ep 送餐占用充足；对角 cell 仍薄，interventional CE / targeted-starts 为既有逃生通道） |
| **D7** | `EXCLUDED_FACTOR_OPTION_KINDS={noop}` 把"伙伴闲置"排除出因子空间，但 partner-idle 恰是 yield 终端模式的行为签名；与 S27 叠加：推断器把一切压到 noop → (terminal,noop) 成唯一有支持 cell → 又被排除 → GraphCoverageError | `graph_builder.py:58`；retry1 audit: row opt3 全部 52.0 质量在 noop 列 | kernel 设计选择（本身有理，与现象冲突） | 中（S27 修复后可能自然缓解；若仍需 partner-idle 模式，需设计决策） |

**自然实验闭环**：G2 时代（oracle 伙伴标签、P1 修复前）同布局建图成功、serve CE=0.73；
现在（行为推断权重）终端列精确 0.0——差异被 S27 完全解释。

**归属裁定**：主导=**实现 bug（S27，新引入）**；放大器=两个设计选择（D6/D7）；proposal 层
仅 A8 假设（"CE 可由群体 rollout 估计"）未分析行为推断权重的支撑集/样本效率——非本次主因。

**S27 修复方案（待实施，走 codex diff 评审）**：标准 Bayes 滤波支撑注入——update 时
`belief = normalize((belief + ε·valid_now_mask) × adjusted)` 或与 uniform(valid_now) 做
λ-混合（遗忘因子）；只使用公共信息 valid_options(state)，不触碰 oracle 边界。修复后
**必须重采集 CE replay**（现有 12000 rows 的 partner dist 已被烙入冻结 bug）。

**仪器化投资兑现备注**：本次能把故障从"CE=0"精确定位到"支撑集冻结的乘性更新"，依赖的
正是 P3 修复的 per-pair support sidecar（精确区分 skipped/measured/exact-zero）+ P1 修复的
行为推断路径可实测性。裁定→修复→验证循环按设计工作。

---

## 创新升级登记（2026-07-04，用户指令：正确性前提下提升创新性至 ICLR 水准）

| ID | 内容 | 源头 | 状态 |
|---|---|---|---|
| U1 | 干预式因子发现：targeted-start 采集 + interventional CE 接线 + 因子支持度证书；配套 C1 命题（互斥因子被动估计不可能性）形式化 | D6（互斥盲区，12000 行精确零支持实测） | 设计中（ICLR_UPGRADE_PLAN §2-C1；预注册 sec18.11） |
| U2 | 持久协议模式信念 FactorModeFilter：推断潜变量从瞬时选项改为 episode-持久因子模式；瞬时推断器降级为诊断/消融臂 | S27（支撑集冻结暴露"瞬时选项是错误潜变量"） | 设计中（ICLR_UPGRADE_PLAN §2-C2；预注册 sec18.11） |
| S27-fix | 支撑注入修复（belief ← (1-λ)·belief + λ·uniform(valid_now)，λ=0.05 config 化；mix=0 保留冻结行为供消融）+ 回归测试 ×4（含冻结复现锚点） | S27 | **代码落地、静态绿（compile+gate I1-I17）；远程验证排队（SSH 待用户预热）**；验证后重采 CE |
| T3.1 | FCP/MEP spike 提前（改判 sec18.1 的 E1-信号门控：spike 不再等待，全量投资仍待 spike 结论+用户批准） | ICLR 评审必问 scripted partners（K2 提前退役） | 待启动 |

**纪律注记**：以上全部在 E1 出数之前预注册（sec18.11），读出规则先于运行写死；
mode-filter 不敌 option-infer 时如实降级 C2（预注册的诚实降级路径）。

## 潜伏缺陷扫荡（LDS，2026-07-04，codex 三 pass xhigh；bundle: review_bundles/latent_defect_sweep_20260704/）

裁决基线：**无一发现作废已完成的 E1-rev wave**（LDS-C1 逐产物核验、LDS-B1 不触 headline、
LDS-B2 属门严格性）。修复窗口纪律见 bundle/SWEEP_SUMMARY.md §6。

| ID | 级 | 类 | 一句话 | 状态 |
|---|---|---|---|---|
| LDS-B1 | A | C6 | terminal shaping 经 `_training_reward` 漏入 eval 回报口径 | **FIXED** `98fe149`（include_terminal_shaping；train 逐位不变） |
| LDS-B2 | A | C11 | eval reward-scale 门只记录不硬拦 | **FIXED** `98fe149`（`_enforce_reward_scale` fail-closed + smoke 逃生口） |
| LDS-C1 | A | C9 | 编排器 metrics.json 存在即视为完成（非终态可被 skip） | **PARTIAL**：消费端防御 FIXED `6ece080`（aggregate 硬校验 final/run_status/updates + 去重）；生产端 success-marker 待写入下一个编排器模板；本 wave 已核验未触发 |
| LDS-C2 | B | C9 | 基线缓存 key 漏 sparse-credit/terminal-shaping/allow_shared_shaping | **FIXED** `98fe149`（schema v3，旧条目全失效） |
| LDS-C3 | B | C5 | eval 无 canonical throughput 字段（分母口径可漂） | **FIXED** `98fe149`（`_throughput_fields`，sec18.9.2 口径，None≠0） |
| PM-7 | — | — | 25-run 表无合规汇总工具 | **FIXED** `6ece080`（aggregate_e1rev.py，已对真实 25 run 出表） |
| LDS-A1 | B | C7 | E2 zeroed 门漏 confidence/半置零 | 降级为事后审计：E2 跑完后核验推断通道是否真置零（含 confidence/半置零），不作 E2 启动门 |
| LDS-B3 | B | C6 | E2 无 CLI 消融开关（需篡改 ckpt config） | 降级为事后审计：E2 跑完后核验消融条件确已置零，不作 E2 启动门 |
| LDS-B4 | B | C3 | `graph.sparse_ce_support` config 被 CE 脚本忽略（认 CLI flag） | 降级为事后审计：从构建对象回读有效 config，核验 sparse_ce_support 是否被 CLI 静默覆盖，不拦启动 |
| LDS-B5 | B | C11 | `--reuse_replay` 门盲于 support_mix/reward/shaping 等 | OPEN，窗口二（CE 复用前修） |
| LDS-B7 | C | C11 | eval 完整性门可被零证据 vacuous 通过 | 降级为事后审计：并入 P3/S9 支持度审计，读数前核验非零证据（现有门已两次正确 fail-closed），不拦启动 |
| LDS-C4 | C | C4 | summary 均值把缺失字段静默当 0.0 | 降级为事后审计：旧字段一次性审 summary 均值是否把缺失当 0（新字段已 fail-closed），不拦启动 |

修复验证记录：codex diff 复评 BLOCK（4 阻塞：cache key 漏 allow_shared_shaping、competence
判据字段、eval 漏 serve_share、eval 缺去重）→ 全部修复 → **APPROVE-WITH-NITS**（nit 已修，
thread 019f2d46）。远程（全新 archive 目录 ARIS4ZSC-w1fix-6ece080）：定向回归 33 passed、
gate I1–I17 exit 0、aggregate_e1rev 对真实 25 run 出表成功（表见 EXPERIMENT_LOG 2026-07-04）。

### S28（2026-07-04，LDS-B2 新硬门首战捕获；实现 bug，中）
- **现象**：stage-1 held-out eval 全部 24 job 被 reward-scale 硬门拒绝（graph_json_sha256
  mismatch：checkpoint 内嵌记录 1e3af3 vs 文件侧 4ea75d）。
- **取证**：graph.json mtime（16:47）早于全部 checkpoint（16:55 起）——文件未漂移；全部语义
  字段匹配；剥离运行时记账键（provenance/formal_experiment/graph_source/preflight_gate）后
  文件与全部抽样 checkpoint 内容哈希一致（d945aa）。
- **根因**：`_stamp_runtime_provenance` 在 train 添加运行时 metadata 键**之后**盖 spec 哈希章
  → checkpoint 内嵌记录与盘上文件哈希**结构性永不相等**；train 门在盖章前比较故历来通过；
  旧 eval 只记录不拦（LDS-B2 本体）故从未暴露——**新门首战即挖出长期潜伏的 provenance bug**。
- **修复**：`graph_content_hash`（剥离运行时键的内容哈希）双侧重算 GRAPH_HASH_FIELD
  （`108bb89`，codex APPROVE-WITH-NITS）；门牙保留（factor/CE/语义 metadata 漂移仍 FAIL，
  双向回归测试）；真 checkpoint 直验 mismatches:{} → eval 重启放行。
- **连带教训（记录）**：`python -c` 验证时 cwd 遮蔽 PYTHONPATH（sys.path[0]=''）——首次验证
  误报 STILL_FAILING；脚本路径执行不受此影响（编排器安全）。

---

## 已归档 / 已合并（2026-07-08，依据 GOVERNANCE_CUTLIST.md）

本段按 GOVERNANCE_CUTLIST.md 第二节 FINDINGS_LEDGER.md 处置表落库。归档＝移出在架门、留档不删；
合并＝删去重复、折入唯一归属。加/减门须过 OPERATING_CONSTRAINTS.md §7 退休阀门。

**归档（移出在架门，仅留档）**

- **P2**（恢复项 +100 吞没角色奖励⇒伙伴同一）：当前工作区树已重缩放为 no-op，无可执行修复；历史
  HEAD 数字为 artifact-suspect、已隔离/作废。归档理由＝闭合事故，当前树无内容可执行。残留一行留在 P 表下。
- **NEW-1**（preflight 回退可能用错伙伴池/信用目标）：已修复并 ACCEPTED（见"静态验收 PASS"段——
  preflight 伙伴子集 + credit 修复），代码护栏独立存在。归档理由＝闭合事故，已离开待裁决面。
- **修复顺序 步1–步7**（步1 去 oracle 证据 P1；步2 信念持久化 P4/S2；步3 CE 支持度 sidecar+重测
  P3/S9/S10/S11；步4 课程去 oracle P5；步5 评估完整性 S16/S17/S18/S19/S20；步6 伙伴库修复 W2/W3/W5/W6；
  步7 preflight/selection 治理门 S12/S23/NEW-2 等）：步骤叙事从未抓到失败，且含"任何决定性重跑/新主张
  前必过"的拦启动措辞（与 §7 相抵）。归档理由＝无记录失败的礼仪叙事。残留：步1–5 已静态验收，两个活的
  残留（W 伙伴库重设计、preflight/selection 治理门）各自另立门。
- **§5.1 治理决策——明确延后**（role_conditioned_v1 伙伴库可用性是否违反 7/1 合成数据禁令）：已在
  Phase 0 裁决闭合——role_conditioned_v2 有条件允许作诊断基底（G0.1）、`require_ego_delivery_selection`
  置 true（G0.2）。归档理由＝闭合事故。残留一行留在原节与顶部快照。

**合并（删重复、折入唯一归属）**

- **NEW-3**（归档 role-v1 config 与产物 provenance 口径不一致）→ 折入 **S18**：同 `reward_scale_verified`
  根因；唯一内容"归档 role-v1 数字视为诊断性非正式结果、不进主张证据链"已并入 S18 的 codex verdict 列。
- **LDS-B6**（`belief_persistence` 未在任何正式 config 显式钉）→ 折入 **P4**：属 P4 的 config 形态；
  "每个正式 config 须显式钉住 belief_persistence"已并入 P4 的目标提交/验证列。
- **D-A**（`require_ego_delivery_selection` 正式 config 须为 true）→ 折入 **S20/NEW-2** 自我交付族：
  同族且已于 Phase 0 G0.2 置 true（见上"待用户决策"条已标闭合）。

**降级（仍在架，就地从"拦启动"改为事后/读数前审计）**

以下五项检查实质保留，只是不再作"开跑前必绿"的启动门；改动已就地写入各自表行：

- **LDS-A1 / LDS-B3**：E2 跑完后核验推断通道是否真置零（不作 E2 启动门）。
- **LDS-B4**：从构建对象回读有效 config，核验 `graph.sparse_ce_support` 是否被 CLI 静默覆盖。
- **LDS-B7**：并入 P3/S9 支持度审计，读数前核验非零证据。
- **LDS-C4**：旧字段一次性审 summary 均值是否把缺失当 0（新字段已 fail-closed）。
- **D4**：读该 claim 前限定"支持相对"口径。
