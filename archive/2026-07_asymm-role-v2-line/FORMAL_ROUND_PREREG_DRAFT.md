# FORMAL_ROUND_PREREG — 盲测正式轮预注册（草案，供 Type-B 签收；签收后逐字入 METHOD_LOCK sec18.14）

> 治理精简 2026-07-08：本文件的门已按 GOVERNANCE_CUTLIST.md 处置；加/减门须过 OPERATING_CONSTRAINTS.md §7 退休阀门。

**Date:** 2026-07-06 · **依据**：EXPERIMENT_CHAIN_PLAN 补充 E（流程校准）；用户授权 "OK，执行"（Phase A 静态起草）
**取代关系**：L1_CONSOLIDATION_PREREG_DRAFT.md（§1–§9）作为基底修复期台账**关闭**；本文件是正式轮唯一预注册。
**dev-heldout 地位**：heldout-handoff-alternate-yield / heldout-resource-server-claim 已降级为判据校准器
（补充 E §18.3），其上全部结果 = 候选证据，不进正式表。

---

## 1. 冻结的实验对象（零重训）

25 个已训 checkpoint，全部 @ 2000 局（40k transitions，eps=2001 逐臂已回读）：
- aris_bellman × seeds 0–4（`results_rebuild/s*/`）
- base_only / global_gru / flat_factor / partner_id_q × seeds 0–4（`results_e1arms/*/`）
**ITT**：25 臂全部评估入表，无剔除（aris s2 的 yield 侧失败照记）。partner_id_q 为 id 参照，
不入四粒度主对照行（§9 先例）。

## 2. 盲伙伴集 spec（blind_v1，6 个；参数空间原则构造，零模型参照）

构造约束：只用注册表已消费的合法取值（role ∈ {ingredient_person, dish_person, server,
prep_partner, flexible}；terminal_policy ∈ {yield, claim, None}；bottleneck ∈ {yield, push,
alternate}；pot ∈ {near, far}；delivery ∈ {nearest, left, right}；counter ∈ {handoff, clear}），
组合均为训练集（role_conditioned_v2）与 dev-heldout **未用过**的新组合；dish_person 角色族
对 v2 训练的模型完全新颖（仅旧 standard7 用过）。

| 名 | ProtocolSpec | 族 | 设计理由（纯参数空间） |
|---|---|---|---|
| blind-dish-yield | role=dish_person, terminal_policy=yield, curriculum_group=blind_yield | yield | 新角色族：偏好盘/装盘链但让出终端——"在终端区活动却不上菜"的新让位风格 |
| blind-prep-alternate-yield | role=prep_partner, pot_preference=far, bottleneck_policy=alternate, terminal_policy=yield, curriculum_group=blind_yield | yield | 支援型让位 + 交替过桥：v2 集未用过的 prep_partner × alternate 组合 |
| blind-dish-claim | role=dish_person, terminal_policy=claim, delivery_preference=nearest, curriculum_group=blind_claim | claim | **干扰敏感候选**：整条终端链（取盘/装盘/上菜）全部加成——与 ego 的终端争用面最大化（server-claim 只争 serve 一点） |
| blind-server-nearest-claim-push | role=server, delivery_preference=nearest, bottleneck_policy=push, terminal_policy=claim, counter_preference=clear, curriculum_group=blind_claim | claim | claim 族新参数化：nearest 配送 + push 过桥（两取值均未在 claim 族用过） |
| blind-bottleneck-alternate-neutral | role=flexible, bottleneck_policy=alternate, counter_preference=clear, curriculum_group=blind_offaxis | 轴外 | **terminal_policy=None**：对上菜轴零参数化，行为差异只在通道/台面轴——回应循环论证（R1-[01]） |
| blind-ingredient-near-neutral | role=ingredient_person, pot_preference=near, counter_preference=handoff, curriculum_group=blind_offaxis | 轴外 | 备菜焦点 + 无终端立场：既不让也不抢，上菜仅由任务进度局部驱动 |

**封存程序**：签收后将上表逐字实现为 `partner_pool.py` 新注册表 `blind_v1`（独立注册表，
永不并入训练集）。落地 commit 的哈希 + 本节文本哈希作为**台账留档**记入 METHOD_LOCK sec18.14
（记录用途、非拦启动门：盲测完整性由本节的参数空间构造 + 独立注册表 + 单看规则保证，哈希
只是可追溯台账；依据 GOVERNANCE_CUTLIST.md，无篡改记录）。
**体检（checkpoint-free，不触碰任何训练模型；盲集健康检查，保留）**：①FSM 脚本 ego 共玩
可完成性；②成对可区分性探针（rc2b 系，FSM-ego）。
**干扰敏感性 A/B（可选侧读数，非启动门）**：仅当要报 §5 的 C3' 侧读数时才跑——脚本 ego
全链抢干版 vs 备菜让位版各 25 局对 blind-dish-claim / blind-server-nearest-claim-push，判据 =
让位版团队回报 ≥ 1.2× 抢干版（达标 ⇒ 该伙伴记 "interference-sensitive"，供 §5-C3' 读出；
不达标 ⇒ 记 "tolerant"，C3' 读出范围相应受限，照实注明）。不报 C3' 时无需跑，不拦正式轮
启动（依据 GOVERNANCE_CUTLIST.md，无对应真实失败）。
**单看规则（盲性审计）**：正式轮前任何 `evaluate_aris` 调用出现 `blind-` 伙伴名即盲性作废
（远程 logs_* 可 grep 稽查）；体检探针不加载任何 checkpoint。

## 3. 评估协议（正式轮，评估-only）

25 臂 × 6 盲伙伴 × **50 局 × 评估 seed {0, 1}**（= 每臂每伙伴 100 局）× `--fast`（stage-1
一致协议；已验证 fast=full 不扰动 rollout）× 线程 caps。完整性硬门全开；每臂入表带
数据量行（OPERATING_CONSTRAINTS §6）。E2-zeroed 因果臂见 §5。

## 4. 主指标（先于任何盲评估冻结）

**伙伴对比度**（逐训练 seed）：
Δ = mean(ego 正确上菜/局 | yield 族 B1,B2) − mean(ego 正确上菜/局 | claim 族 B3,B4)。
- **S1（条件化存在）**：aris 的 5-seed 中位 Δ ≥ **0.5/局**，且 pooled Δ 的
  episode-bootstrap（seed 分层，10k 次）95% CI 下界 > 0；
- **S2（方法特异）**：aris 中位 Δ − 最强 baseline 中位 Δ ≥ **0.25/局**，且逐 seed 配对
  （同号 seed）aris > 该 baseline 于 ≥4/5 对。
（阈值来源：dev-heldout 校准值 Δ_aris≈1.0/局、Δ_baseline≤0——取其一半/四分之一为先验
门；在盲伙伴上留余量。功效诚实声明：5 seeds 的 seed 级检验为符号型/描述型，推断主体在
episode 级 bootstrap；措辞阶梯已按此校准。）
**轴外伙伴（B5,B6）**：描述性 co-report（回报/完成率/阻塞 vs baselines），不设门——
用途是"方法在非角色轴伙伴上不退化"的循环论证防御，不进主判据。
**C3' 专项读出**：在体检认证为 interference-sensitive 的 claim 伙伴上，比较 aris 团队回报
vs 最佳 baseline 团队回报（两个方向都照实报；aris ≥ 则互补-回报张力解除）。
**co-report（全臂统一）**：claim 族伙伴吞吐保持、yield 族 egoCCR、团队回报（含不利对比）、
blocking_rate。

## 5. E2-zeroed 因果臂（随行）

aris × 5 seeds × 盲伙伴集 × zeroed（LDS-B3 声明式）；另对任何中位 Δ ≥ 0.25 的 baseline
同跑。**成功判据**：zeroed 使 aris 中位 Δ 塌缩 ≥ **50%**。（dev 校准：置零曾使 claim 侧
配合崩塌 3/5。）

## 6. 措辞阶梯（读数前 / 落 claim 前审计，非启动门）

下表把 §4–§5 已冻结的数值分支映射到论文口径。**判据本身（S1/S2/E2 的数值条件与分支结构）
仍是先写后看的冻结决策规则，不动**（对应 INC-4：挡事后叙事套噪声）；只有"分支 → 论文措辞"
这一层降级为**读数时、落 claim 前**审计即可，不作为拦正式轮启动的门（依据 GOVERNANCE_CUTLIST.md：
无 overclaim 记录，数值分支锁定后措辞读数时审）。**不主张清单**并入本读数产物（与措辞阶梯
同一读数时消费，不再单列）。

| 条件 | 论文口径 |
|---|---|
| S1 ∧ S2 ∧ E2 塌缩 | "因子局部信念唯一产生伙伴条件化的角色互补行为，且该条件化由信念通道因果承载"（范围：单布局、脚本伙伴族、盲 held-out） |
| S1 ∧ S2，E2 混合 | 同上去掉"因果承载"，信念通道证据按 seed 纹理如实呈现 |
| S1 ∧ ¬S2（baseline 也条件化） | "伙伴条件化随充分数据涌现，因子信念非必需"——主张转数据/基准发现 + E2 可解剖性 |
| ¬S1 | dev-heldout 全部结果降档为过程过拟合案例；回 Phase A 重校准（预注册允许的唯一回路） |
**不主张清单**（属本措辞阶梯的一部分，同一读数产物）：跨布局、人类伙伴、population 对比
（后者列 Phase C 债务）、回报优势（除非 C3' 专项读出为正）。

## 7. Type-B 签点与执行序

①本文件签收（运行授权；OPERATING_CONSTRAINTS §7 保留的两个人工判断点之一）→ 逐字入
METHOD_LOCK sec18.14；③正式轮一次执行 → 终读判读（最终读数裁决，§7 保留的第二个人工判断点）。
执行序：签① → 实现注册表与体检（checkpoint-free，codex diff 评审）→ blind_v1 落地、体检报告、
封存哈希作为**台账留档**（机器可检，不设中途人工签字）→ 正式评估（单次，单看规则 + 完整性
硬门全开）→ 签③。预计成本：体检 ~20 min，正式轮 25 臂 ×6 伙伴 ×100 局 ≈ 15k 局 + zeroed 臂
≈ 数小时墙钟（评估-only）。

（原"签 ② 盲测落地"中途人工签字已归档——见文末归档节；其捆绑的落地 / 体检 / 封存均作为
codex 评审 + 机器可检台账独立保留。终读裁决仍标为签 ③，保持既有引用。）

---

## 已归档 / 已合并（2026-07-08，依据 GOVERNANCE_CUTLIST.md）

- **签 ②（盲测落地 · 中途人工签字）——已归档**：对不上任何真实记录失败；它捆绑的检查
  （blind_v1 落地、体检报告、封存哈希）都作为 codex diff 评审 + 机器可检台账独立保留，盲测
  完整性由参数空间构造 + 独立注册表 + 单看规则保证。中途签字是"启动很重"的最大单一来源；
  保留的人工判断是签 ①（运行授权）与签 ③（最终读数裁决）。
- **§6 不主张清单——已合并**：折入同节的措辞阶梯（与措辞阶梯同一读数时消费的读数产物），
  不再单列。
- 事后审计化（降级，非删除，详见正文，不再拦正式轮启动）：封存哈希入 METHOD_LOCK（§2，台账
  留档非门）、干扰敏感性 A/B 认证（§2，仅报 C3' 时才跑）、措辞阶梯（§6，读数前 / 落 claim 前
  审计）。§4–§5 的数值冻结决策判据不动。
