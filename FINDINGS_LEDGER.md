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
