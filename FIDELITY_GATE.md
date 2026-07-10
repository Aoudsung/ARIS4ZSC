# ARIS-Bellman Fidelity Gate

> 治理精简 2026-07-08：本文件的门已按 docs/status/GOVERNANCE_CUTLIST.md 处置；加/减门须过 OPERATING_CONSTRAINTS.md §7 退休阀门。

**Overall: ✅ PASS** (GREEN)

Register: method-fidelity (Type-A). Does **not** judge claim support or scientific merit — that is the cross-model jury + human.

**门的高度（2026-07-08 治理精简，依据 docs/status/GOVERNANCE_CUTLIST.md）。** 这套门是一次静态检查——一条命令、几秒钟跑完。它整体要求在**读结论之前 / 部署改动过的方法代码之前**为绿，**不是每次开跑前必须先过的启动关卡**。这与 OPERATING_CONSTRAINTS.md §7「默认高度＝拦 claim，不拦启动」一致：本门不设每次启动或实验中途的人工签字，也不作开跑前的强制关卡。

- **I1–I9 是事后 / 部署前的「方法身份」审计。** 它们确认部署与训练代码仍是那套单目标 Bellman 方法——没有偷偷长出「选择器」（按信息增益、互信息、信息价值来挑动作的分支）或辅助损失。这些检查「出生即绿」、对不上任何有记录的失败，只在**改动方法代码后 / 读结论前**复核，不拦启动。
- **I10–I17 是载重的真缺陷护栏。** 每一条都对得上一次真实、有记录、昂贵的失败（P1、P4、P3、P5、S17、S20、NEW-2、S23），是本门的核心。它们对应的运行时保护（去 oracle 的证据路径、盲测划分不污染、preflight 布局合法性）在别处以运行时报错作为真正的启动关卡；本静态门只核对「代码里那些保护仍在」，这一核对属部署前审计，未被本次精简削弱。

（提示：把「代码里保护仍在」核对成绿，是部署前审计；真正拦启动的运行时 preflight、数据预算、去 oracle、盲测划分门，分别归属于 OPERATING_CONSTRAINTS.md §2/§6 与 PROJECT_DASHBOARD.md §4，此处不重复设门。）

下表是工具的 17 项静态检查快照（pass/fail 语义不变，工具仍照常运行 I1–I9）。每项的高度与处置见上文分组与文末「已归档 / 已合并」：I1–I9 为事后/部署前审计，I10–I17 为载重护栏。

| Check | Invariant | Verdict | Evidence |
|---|---|---|---|
| I1 | no G-TVOI/MI/probe selector in deploy path | ✅ PASS | deploy/select/train path free of selector symbols (Δ_info/MI stay post-hoc in diagnostics.py) |
| I2 | single TD loss, no auxiliary losses | ✅ PASS | src/aris_bellman/td.py returns one F.mse_loss; experiments/overcooked_v2/train_aris.py backprops aris_td_loss with no aux terms |
| I3 | action selection is pure Bellman argmax | ✅ PASS | experiments/overcooked_v2/train_aris.py:_select_option selects via argmax with no info-gain/selector branch |
| I4 | CE is preprocessing, not in training loop | ✅ PASS | experiments/overcooked_v2/train_aris.py does not call CE estimation; CE lives in ce_sampler.py (preprocessing) |
| I5 | reward-scale single-source (structural) | ✅ PASS | cost/shaped coefs flow from training.* into CE, preflight, eval, and td target (numeric equivalence is semantic → out of scope, jury/human) |
| I6 | articulation-point bottlenecks, not degree<=2 | ✅ PASS | experiments/overcooked_v2/layout_parser.py uses Tarjan articulation points + region-size filter; no degree<=2 heuristic |
| I7 | preflight hard gate + acceptance, no smoke bypass | ✅ PASS | experiments/overcooked_v2/train_aris.py:_enforce_preflight_gate requires an accepted report; no smoke bypass |
| I8 | no factor-accuracy as a primary metric | ✅ PASS | evaluation is return / reference-gap grounded; no factor-label-accuracy metric |
| I9 | factor deletion removes latent+route+relevance | ✅ PASS | experiments/overcooked_v2/graph_builder.py:make_graph_spec derives relevance+route_map+mode_mask from `factors`; deleting a factor structurally removes all three |
| I10 | P1: main evidence path oracle-free (behavior-inferred partner option) | ✅ PASS | partner act() emits no true option label; executor strips the raw partner action before extract_event and re-annotates from the behavior inferencer |
| I11 | P4: belief hidden state persists across option decisions | ✅ PASS | EvidenceBuffer carries a persistent hidden; transitions store window-base hidden snapshots; belief decodes from hidden (episode-scoped memory) |
| I12 | P3: CE artifacts expose per-pair support (skipped != measured zero) | ✅ PASS | estimator emits weight_sum + estimable/skipped/measured-zero masks; pipeline reads graph.ce_min_weight from config |
| I13 | P5: no true terminal-policy conditioning in the main method path | ✅ PASS | role-conditioned reward/exploration/replay raise without the explicit ablation flag; option selection carries no terminal-policy argument |
| I14 | S17: allow_diag_skip cannot bypass hard eval integrity checks | ✅ PASS | forced-noop / evidence-policy / observed-dist / missing-evidence / oracle-source checks run regardless of --allow_diag_skip (flag scopes diagnostics only) |
| I15 | S20: headline completion is ego-owned, actor-split metrics recorded | ✅ PASS | eval headline = ego_correct_completion_rate keyed on ego-sole deliveries; greedy validation reports team vs ego-sole rates under distinct names |
| I16 | NEW-2: checkpoint eligibility gates before publication | ✅ PASS | deploy eligibility is checked before best-selection/save; stale deployable checkpoints are unlinked when no eligible checkpoint is selected |
| I17 | S23: explicit train-partner split required for split claims | ✅ PASS | unset train_partners hard-errors when a held-out split is declared; CE pipeline filters to train partners; all-partner runs require the explicit no-split flag |

## 已归档 / 已合并（2026-07-08，依据 docs/status/GOVERNANCE_CUTLIST.md）

这些处置只改「门的高度与归属」，不改工具行为：工具仍机械运行 I1–I9，pass/fail 语义不变，没有任何检查被删除。实质检查全部保留。

- **I8（不得以 factor-accuracy 为主指标）— 归档。** 它是 WARN-only（只告警、不阻断），内容只是把 OPERATING_CONSTRAINTS.md §5 的一段散文重述了一遍，对不上任何有记录的失败。claim 尺度（能不能把某数字当主指标）的把关，由 cross-model（不同模型族）陪审 + 人工最终读数裁决负责。因此 I8 退出在架的载重门面；残留事实：工具仍会跑并报告它，但它不再作为独立的载重门。
- **I3（动作选择纯 Bellman argmax）— 合并进 I1。** 与 I1「部署路径无选择器」禁的是同一件事（部署期不得出现按信息增益/互信息/信息价值挑动作的分支）；这份实质由 I1 唯一承载。
- **I4（CE＝协调外部性估计，只在预处理阶段算、不进训练环）— 合并进 I2（训练环单目标 / training-loop purity）。** 与 I2「单一 TD 损失、无辅助损失」同为「训练环里只有一个目标」；折成一条，由 I2 承载。
- **I1、I2、I5、I6、I7、I9 — 降级为事后 / 部署前审计（非删除，实质全部保留）。** 它们出生即绿、对不上有记录的失败，改为在「改动方法代码后 / 读结论前」复核，不再拦每次启动。
  - I1、I2：方法身份审计（改方法代码后 / 读数前）。
  - I5：降为结构备注——数值等价属语义，归 jury / human；S18/S28/NEW-3 由运行时门抓，不靠此 WARN 检查。
  - I6：改 layout parser 时审（对重命名脆弱、易误报）。
  - I7：核对代码里 preflight 的 `raise` 仍在，属改代码时的审计。**残留事实：真正拦启动的运行时 preflight 硬门未被削弱**，对应 S27（12000 行 replay 被无效布局污染、事后无法补救），归属 OPERATING_CONSTRAINTS.md §2 与 PROJECT_DASHBOARD.md §4。
  - I9：跑因子删除消融时审（该消融尚未跑）。
