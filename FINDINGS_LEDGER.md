# FINDINGS_LEDGER.md — 根因裁定发现台账（唯一真相源）

**Status:** ROUND-1 已收敛，7处分歧全部裁决(采纳codex) · **执行暂缓**（用户裁定"都先不做"）
**Date:** 2026-07-02 · **Branch:** `rc-rootcause-fix`

> 规则（[ROOTCAUSE_REVIEW_PLAN.md](ROOTCAUSE_REVIEW_PLAN.md) §2.1）：后续一切讨论、提交、重跑结论
> **只能引用本表 ID**；不在表里的问题视为不存在；不想修的问题必须记 `wontfix+理由`，不能
> 静默丢弃。历史记录**追加不覆盖**——分歧的原始双方论证保留在文末，裁决作为追加记录，不删除
> 争议过程本身。

**状态图例**：`待codex` · `已裁定` · `分歧`(不再使用，本轮已清零) · `待人类裁决(新发现)` ·
`修复中` · `已提交` · `已验证` · `已关闭` · `wontfix` · `阻塞于治理`

**快照统计**：51 条。codex 本轮覆盖 50/51（仅W8待下一轮）。**分歧：0**（7处本轮全部裁决，
均采纳codex修正）。**治理未决：1**（§5.1，见下，明确延后，非阻塞当前状态）。

**当前动作状态：暂停**。用户明确"都先不做"——不打包clean zip、不进
`ROOTCAUSE_FIX_EXECUTOR_PROMPT.txt`、不再调用codex。本文件此次更新仅做**决策固化**（把已经
做出的判决写入台账+更新计划文档的修复顺序），不触发任何代码/远程动作。下一次推进由用户发起。

---

## §5.1 治理决策——明确延后（不是待定，是用户主动选择晚定）

**用户裁定（2026-07-02）**："等我修订完代码再决定。"

含义：role_conditioned_v1 工作区伙伴库重设计是否可用（是否违反7/1合成数据禁令）这一问题，
**不在本次会话裁定**，触发条件是"相关代码修订工作完成之后"。在此之前：

- W系列（W1-W8）整体标记 `阻塞于治理`，即使个别条目（如W1、W7）本轮已经通过分歧裁决更正了
  严重度/是否成立，**这不解除治理阻塞**——分歧裁决解决的是"这条发现本身对不对"，治理决策解决
  的是更上位的"这个重设计的伙伴库能不能作为正式benchmark使用"，两者独立，不能互相替代。
- 修复顺序步骤6（伙伴库修复，见下）保持"前置：§5.1治理裁决"不变。
- 提醒：`ROOTCAUSE_FIX_EXECUTOR_PROMPT.txt` 步骤7本身也要求这个治理决策（"mark it as benchmark
  v2... requires human governance decision before becoming a benchmark"）——两处来源独立要求
  同一个决策，说明这确实是真实阻塞点，不是本计划过度设卡。

---

## codex 全量优先级排序（1–37，跨支柱+次级统一排序，供未来排期参考）

```
1 P1  2 P4  3 P5  4 P3  5 P2  6 S20  7 S17  8 W2  9 S18  10 S1
11 S8  12 S12  13 D2  14 S16  15 S23  16 S10  17 S11  18 S3  19 S2  20 W3
21 S9  22 S22  23 S24  24 S21  25 S6  26 S7  27 W5  28 S19  29 W4  30 S4
31 S26  32 W1  33 W6  34 D1  35 D3  36 D4  37 D5
（W7=REFUTED无排名；E1-E9=豁免无排名；NEW-1/2/3=新发现未排名）
```

**修复顺序已按分歧7裁决更新**——详见 [ROOTCAUSE_REVIEW_PLAN.md](ROOTCAUSE_REVIEW_PLAN.md) §2.3
（本次同步更新，P5提前、评估完整性提前、新增治理门步骤）。

---

## 支柱（P）

| ID | 断言（我方原判） | 严重度(我方) | codex verdict | 人类裁决(2026-07-02) | 处置 | 目标提交/验证 | 状态 |
|---|---|---|---|---|---|---|---|
| P1 | 脚本伙伴真实执行选项被直接注入证据流(train/eval/CE)；行为推断器死代码 | 高 | CONFIRMED·HIGH·#1 | **采纳codex**：Q-B命题措辞软化为"P1使parity更可能、使'习得推断'主张不可采信，但不构成parity的机械证明——效应量只能由去oracle重跑测定"（不是"P1⇒parity"的机械蕴含） | fix | 步1(去oracle)，不变 | 已裁定 |
| P2 | HEAD:+100吞没±4/−1⇒伙伴同一；工作区已重缩放但恢复项未同步 | 高 | REFINED·HIST-HIGH/现状MED·#5 | **采纳codex**：历史结论(HEAD)归档为artifact-suspect，**当前工作区树不需要专项修复**——工作区已无该swamping算术，原判"高"仅适用于历史版本 | archive(历史)/no-op(当前树) | 无需独立修复步骤；残余风险随W系列跟踪(阻塞于治理) | 已裁定 |
| P3 | asymm的`serve_soup CE=0.000`是支持度不足哨兵值非测量 | 高 | CONFIRMED·HIGH·#4 | 无分歧，维持原判 | fix | 步3(CE支持度sidecar+重测) | 已裁定 |
| P4 | 信念=4步滑窗每次从零隐状态重编码，无跨决策记忆 | 高 | CONFIRMED·HIGH·#2 | 无分歧，维持原判 | fix | 步2(信念持久化) | 已裁定 |
| P5 | 奖励/探索/回放播种以伙伴`terminal_policy`真值为条件 | 高 | CONFIRMED·HIGH·#3 | 无分歧(措辞已一致)；**排序采纳codex**：从原步5提前到步4 | fix | **步4**(原步5，提前) | 已裁定 |

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
| S18 | `reward_scale_verified`永假 | 中 | CONFIRMED·MED-HIGH·#9 +实锤，关联NEW-3 | 已裁定 |
| S19 | diagnose_traces.py三缺陷 | 中 | CONFIRMED·LOW-MED·#28 | 已裁定 |
| S20 | completion计入伙伴/错误送餐 | 中 | CONFIRMED·HIGH·#6(大幅升级) | 已裁定 |
| S21 | eval种子惰性+跨伙伴复用序列 | 中 | CONFIRMED·MED·#24 | 已裁定 |
| S22 | asymm.yaml的Q臂间bound不匹配，role_v1已修 | 低(半闭) | REFINED·MED·#22——**与我方"半闭"判断完全一致** | 已裁定(双方一致) |
| S23 | train_partners未设置静默回退 | 中 | CONFIRMED·MED·#15 | 已裁定 |
| S24 | provenance缺失仅告警 | 低中 | CONFIRMED·MED·#23 | 已裁定 |
| S26 | event_summary字段重复声明 | 极低 | CONFIRMED·LOW·#31 | 已裁定 |

---

## 工作区伙伴库重设计（W）— 全部 `阻塞于治理`（§5.1，见上）

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
| D4 | CE估计量把可测性与占用率混同 | 高 | CONFIRMED·HIGH estimator·#36 | 无分歧 | 已裁定 |
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
| NEW-1 | preflight回退路径可能用错伙伴池/信用目标 | MED-HIGH | `layout_diagnostics.py`无replay可复用时调用`make_training_partners`不带`partner_set`，收集replay不带`credit_params` | 待人类裁决(新发现) |
| NEW-3 | 当前role-v1 config与已归档role-v1产物目标/provenance口径不一致 | MED | 归档role-v1 eval报告`reward_scale_verified=false`，与S18同因；**归档role-v1数字应视为诊断性非正式结果** | 待人类裁决(新发现，关联S18) |

---

## 修复顺序（已按分歧7裁决更新——权威版本见 [ROOTCAUSE_REVIEW_PLAN.md](ROOTCAUSE_REVIEW_PLAN.md) §2.3）

```
步1 去oracle证据(P1)                                          [无前置]
步2 信念持久化(P4,S2)                                          [前置:步1]
步3 CE支持度sidecar+重测asymm，仅估计量不碰policy(P3,S9,S10,S11) [可与步1-2并行]
步4 课程去oracle(P5)                          [原步5，2026-07-02采纳codex提前]
步5 评估完整性(S16,S17,S18,S19,S20)            [原步6，2026-07-02采纳codex提前；任何决定性重跑前必过]
步6 伙伴库修复(W2,W3,W5,W6；W1已证伪不再需要)   [前置:§5.1治理裁决——当前延后，见上]
步7 preflight/selection治理门(D1[已更正],D2,S12,S23,NEW-1,NEW-2)
                                    [2026-07-02 codex新增建议已采纳；任何新主张声明前必过]
```

**当前状态：以上顺序已裁定，但整体执行暂缓**（用户："都先不做"）。下一次推进由用户发起，
不需要重新走分歧裁决流程——直接从步1开始，或先处理NEW-2的历史产物审计。

---

## 下一步（用户主导，本会话不再推进）

- **执行路径A（进`ROOTCAUSE_FIX_EXECUTOR_PROMPT.txt`）**：暂缓。
- **执行路径B（clean-zip二轮独立评审）**：暂缓。
- **§5.1治理决策**：延后至"代码修订完成后"，见上方专节。
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
D-A `require_ego_delivery_selection` 在 role_v1/role_v1_novb 两个正式候选 config 中为 **false**
（asymm/armA/armB 为 true）——要么论证 contrib_team 已使 free-rider 无利可图故不需要，要么翻开。
D-B **§5.1 治理决策现在成熟**（用户原话"等我修订完代码再决定"——代码修订已完成并验收）：
role_conditioned_v2 candidate 能否作为 benchmark-v2 使用（须先过 partner-differentiation certificate）。

**状态推进**：P1/P3/P4/P5/S1/S2/S3/S16/S17/S18/S20/S23/NEW-1/NEW-2 → `已验证(静态)`；
远程 I10-I18 验证 = 实验链 Phase 1（见 [EXPERIMENT_CHAIN_PLAN.md](EXPERIMENT_CHAIN_PLAN.md)）。

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
