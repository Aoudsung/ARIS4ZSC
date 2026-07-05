# ROOTCAUSE_REVIEW_PLAN.md — codex 深评 → 结果回流 → 防走偏 闭环计划

**Status:** PLAN — 待用户批准后执行 · **Date:** 2026-07-02 · **Branch:** `rc-rootcause-fix`
**角色分工:** Claude（静态起草/回流整理）· codex（异模型族评审）· 用户（Type-B 门 + 一切执行授权）

> **边界声明**：本计划中 Claude 的全部动作均为静态维护（读文件、写 Markdown、准备
> prompt 包），符合 [OPERATING_CONSTRAINTS.md](OPERATING_CONSTRAINTS.md) §1。codex 评审由
> **用户在终端运行 `codex exec`**（项目记忆：MCP 工具调用会被自动拒绝）。所有远程
> 重跑/探针均执行门控，需逐次授权。

---

## 0. 裁决对象与总目标

**被评审物** = 2026-07-02 五路静态审计形成的根因裁定：

> 核心主张（ZSC = 对价值充分交互因子信念的 Bellman 控制）**从未被测试**；迄今负结果由
> 【P1 oracle 证据泄漏 + P2 伙伴基底退化 + P3 CE 哨兵零 + P4 信念无跨决策记忆】复合导致；
> 方法本身既未证实、也未证伪。（发现登记册见附录 A。）

**总目标**：让异模型族评审者（codex，GPT-5.5，xhigh）**独立地证伪**上述裁定，而非为其
背书；把 codex 的裁决**机械地**灌回「台账 → 提交 → 再验证 → 活文档」；并预装护栏，使
后续任何修订都无法再次把伪影（哨兵值、泄漏、饱和指标、退化基底）读成科学结论。

| 用户需求 | 交付物 | 本文档章节 |
|---|---|---|
| 协调文档要点，让 codex 有效深评 | `review_bundles/rootcause_review_20260702/REVIEW_BRIEF.md` + `PROMPT.txt` | §1 |
| 评审结果有效回流到方案与代码 | `FINDINGS_LEDGER.md`（唯一真相源）+ 回流协议 | §2 |
| 避免后续修订再次走偏 | 预注册 + 伪影自检门 + FIDELITY_GATE I10–I14 + 冻结/隔离规则 | §3 |

---

## 1. 阶段一：评审简报（REVIEW_BRIEF）要点规格

**设计原则：不给结论求背书，给证据求证伪。** 五路审计全部出自 Claude 子代理——codex 的
价值恰在异模型族打掉同族盲点。简报必须让 codex 能逐条复核 file:line，并被强制要求对每
个顶层命题**同时构造正反两面最强论证**。

### 1.1 反锚定纪律（写进简报头部，作为对 codex 的指令）

1. 默认对每条发现持怀疑；只有**自己打开 file:line 复现了该行为**才允许标 `CONFIRMED`。
2. 主表只呈现「ID · 断言 · 代码锚点 · 代码事实」；我方的分类/严重度/裁定放在**附录**，
   要求 codex **先写下自己的 verdict 再读附录**（不可强制，但明示流程）。
3. 对每个顶层命题，必须先为**反假设**写出最强论证，再下判。

### 1.2 简报七板块（必含内容）

**板块 1 — 裁决问题（对抗式表述）**，五问，每问附反假设：

| # | 命题（我方裁定） | 反假设（codex 必须先论证） |
|---|---|---|
| Q-A | 核心主张未被测试 | A′：现有某结果子集**已足以**证实或证伪方法 |
| Q-B | parity 是 oracle 泄漏(P1)的机械结果 | B′：即便各臂共享 oracle，ARIS 因子结构仍应产生可测差异。**关键测例**：已作废的 cramped 合成伙伴结果中，flat_factor 拿同样 oracle 证据仍被 ARIS 以 2× 击败（10.0 vs 5.0）——如何与 B 相容？ |
| Q-C | asymm 的 serve CE=0.000 是哨兵伪影，§11 布局结论不安全 | C′：17 次随机可达尝试本身就是布局属性的证据，加样本后 CE 仍将≈0（涉及 D4 估计量之争：随机选项策略下的占用率 ≠ 有能力博弈下的外部性） |
| Q-D | 4 步滑窗 + 无跨决策隐状态 ⇒ 声明的「累积失败痕迹」机制结构性缺席，且是 held-out 持续犯错的主因 | D′：窗口只是 config；in-dist 成功证明窗口已够；held-out 失败另有主因（训练分布狭窄） |
| Q-E | 附录 A 的分类/严重度/修复排序正确 | E′：独立重排；指出被高估、漏判、误判的条目 |

外加一个**advisory 问题**（标注为接近 Type-B、最终归人类）：Q-F 去 oracle + 信念持久化
之后，TD-only + 纯行为证据在**原理上**是否足以学出因子推断？给出论证与判据设计。

**板块 2 — 证据锚点表**（可逐条复核，schema 固定）：

```
ID | 断言(一句话) | 锚点(file:line,…) | 代码事实(它做了什么) | 证伪方法(静态可查 / 需执行探针)
```

内容 = 附录 A 全部条目（P1–P5、S1–S26、W1–W7、D1–D5、E1–E9 豁免项）。豁免项必须在
表内——codex 应同样尝试推翻「已确认正确」的部分。

**板块 3 — 确定性分层**：明确区分
- **算术/结构确定**（P2 打分吞没、P3 min_weight 哨兵机制、P4 滑窗结构）→ codex 靠阅读
  即可确认或推翻；
- **需执行探针**（asymm 加样后 serve CE 是否转正；去 oracle 后各臂是否分离）→ codex 只
  负责**设计探针方案**（样本量、判据、预期分支），不负责运行（执行门控）。

**板块 4 — 范围边界**：本次评审裁决的是**代码保真性与既有结论的伪影安全性**（Type-A
偏向）；方法本身的科学 go/no-go 属 Type-B，归 codex+人类双签（OPERATING_CONSTRAINTS §4）。
明确告知 codex：不要为方法「盖章」，Q-F 仅供参考。

**板块 5 — codex 必答清单**：① 逐条 verdict（含豁免项）；② Q-A…Q-D 反假设分析；
③ 我方综合中被高估/错误的条目；④ 它自己的 must-fix 排序 + 与 §2.3 排序的分歧点及理由；
⑤ Q-F advisory；⑥ 新发现（NEW-x 编号，同 schema）。

**板块 6 — 附件清单（manifest）**：五路审计的蒸馏发现表；[METHOD_LOCK.md](METHOD_LOCK.md)
§8/§11/§15–16；提案 §5.9/§11（主张+非声明）；[FIDELITY_GATE.md](FIDELITY_GATE.md)；两个在用
config；**精确代码 manifest**（文件+行段清单，工作区状态，注明哪些是未提交修改——特别是
partner_pool.py 的 role_conditioned_v1 重设计属未提交）。

**板块 7 — 返回格式（固定，保证可机械回流）**：

```
## Verdicts
ID | verdict(CONFIRMED/REFUTED/REFINED) | 依据(file:line) | 修正内容 | codex severity | codex rank
## Counter-hypothesis analyses   (Q-A′..Q-D′ 逐个：最强论证 → 判决 → 置信度)
## New findings                  (NEW-1.. 同锚点表 schema)
## Revised fix ordering          (与 §2.3 的 diff + 理由)
## Probe designs                 (每个需执行判定的问题：探针、判据、预期分支)
## Q-F advisory
```

### 1.3 运行方式

- 用户终端：`codex exec`（xhigh reasoning），输入 = `PROMPT.txt`（内联 REVIEW_BRIEF +
  manifest 指向的文件路径；codex 在本仓库工作目录内自行读码）。
- 输出落盘：`review_bundles/rootcause_review_20260702/CODEX_OUTPUT.md`。
- 若输出偏离板块 7 格式 → 一次格式重询（不改实质内容），仍不合则人工摘录进台账并注明。

---

## 2. 阶段二：评审结果回流协议

**原则：一次评审只有在「每条发现都有归属、裁决、处置、提交、复核」时才算被吸收。**

### 2.1 发现台账 `FINDINGS_LEDGER.md`（唯一真相源，repo 根）

每条发现一行，字段：

```
ID | 来源(audit-N/codex/NEW) | 断言 | 锚点 | 分类 | codex verdict | 人类裁决 |
处置(fix/defer/wontfix+理由) | 目标提交 | 验证方法 | 复核状态(open/fixed/verified/closed)
```

分类固定五类：**实现错误 / 基底设计缺陷 / 估计量局限 / 方法设计局限 / 治理**。
规则：任何后续讨论、提交、重跑结论**只能引用台账 ID**；不在台账里的问题不存在，
不想修的问题必须记 wontfix+理由——评审蒸发不掉。

### 2.2 分歧裁决（Type-B，双签）

- codex verdict 与我方裁定冲突时：显式对质（各自最强论证并列），**用户判决**，获胜论证
  记入台账；不做折中平均。
- codex `REFUTED` 一条我方 pillar → 该 pillar 关联的全部下游修复挂起，先解决对质。
- 任何一方都不得自我开释（OPERATING_CONSTRAINTS §4）。

### 2.3 修复顺序（依赖链）与提交纪律

> **REVISED 2026-07-02**（round-1 codex 评审 + 用户裁决「分歧7」，已采纳）：原表格是本计划
> 初稿的猜测排序；codex 独立复核后给出跨全量 findings 的排序（[FINDINGS_LEDGER.md](FINDINGS_LEDGER.md)
> 「codex 全量优先级排序」），核心改动——**P5 提前**（原步5→步4）、**评估完整性提前**（原步6→步5，
> 理由：任何决定性重跑前必须先过，不能等到伙伴库修完才修）、**W1 移出阻塞项**（codex 证伪其
> 活锁论证）、**新增步7 治理门**。用户已裁决「全部采纳」。下表为当前权威版本；原表格逻辑保留在
> git 历史中，不在此处重复（追加式修订，不做静默覆盖）。

依赖顺序（前者不修，后者实验必被污染）；**每步一个静态 handoff 文档**（模式同
[RC_REWARD_CREDIT_FIX.md](archive/docs/RC_REWARD_CREDIT_FIX.md)），经 codex diff 评审（CODE_REVIEW=true）后才进远程验证：

| 步 | 修复 | 台账主 ID | 前置 |
|---|---|---|---|
| 1 | **去 oracle 证据**：接入 `option_inferencer`（或 config 开关置零 `partner_option_*` 通道），train/eval/CE 三处一致 | P1 | 无——**一切比较实验的前提** |
| 2 | **信念持久化**：跨选项决策保留隐状态，或窗口≥数个选项跨度 + 修零填充掩码(S2) | P4, S2 | 步1 |
| 3 | **CE 支持度 sidecar**（per-pair weight_sum + skipped mask）+ min_weight 可配 + **重测 asymm** | P3, S9, S10, S11 | 可与步1–2并行（不碰 policy） |
| 4 | **课程去 oracle**：`role_contrib_team`、角色探索、回放播种、`ego_terminal_penalty_under_claim`、选项选择调用里的 `partner_terminal_policy` 真值分支全部移出主方法臂；保留时只能是独立 oracle/curriculum 消融，或主张明确降级为「课程条件化」 | P5, NEW-4 | 步1（**原步5，2026-07-02提前**） |
| 5 | **评估完整性**：`--allow_diag_skip` 只跳诊断不跳完整性门(S17)、provenance 口径(S18)、gru 诊断形状(S16)、completion 指标拆分(S20)、diagnose_traces.py(S19) | S16, S17, S18, S19, S20 | 任何决定性 eval/重跑之前（**原步6，2026-07-02提前**） |
| 6 | **伙伴库修复**：换掉近重复 held-out claim 伙伴(W2)、yield 层内排序(W3)、其余 W4-W6 小修；**W1 已被 codex 证伪不再是阻塞项**（活锁论证在当前树不成立，仅剩 tier 内平局微小偏置，可选打磨） | W2–W6 | **§5.1 治理裁决**（2026-07-02 用户裁定「等修订完代码再决定」，当前延后） |
| 7 | **preflight/selection 治理门**（新增）：把伙伴差异性代理指标的粒度不足(D1已更正/D2)、preflight 回退口径(S12)、split 静默回退(S23)、preflight 伙伴池/信用目标(NEW-1)、**checkpoint 可在 guard 判定前被保存**(NEW-2) 打包为「新主张提出前必须先过」的门 | D1(已更正), D2, S12, S23, NEW-1, NEW-2 | 步1–6完成后，任何「支持/否定主张」声明前 |

提交纪律（项目既有教训）：**需保行为处先 no-op 提取 + 黄金轨迹**；**一次一机制**；提交
message 引用台账 ID；多机制修复必须带组件消融。

### 2.4 每步的保存性检查（substrate-not-solve-away，写进各 handoff 的「必须不做」）

- 步1：去 oracle **不得**用另一路真值标签顶替（推断器输入只允许可观测行为）。
- 步2：持久化**不得**引入未来信息泄漏（隐状态只沿时间正向传播）。
- 步3：估计量修复**不碰** policy/method 任何一行。
- 步4：课程去 oracle **不得**只改文档措辞掩盖问题——若保留角色条件化课程，必须明确降级为
  独立消融/课程专用基线，不得继续作为主方法证据；不得用 `ego_terminal_penalty_under_claim`
  或 option-selection 的 `partner_terminal_policy` 传参换通道保留 oracle。
- 步5：评估完整性修复**不得**新增门却不修数据流本身（如：光收窄 `--allow_diag_skip` 范围但
  不修 provenance 口径不一致的根因）；`ROOTCAUSE_FIX_EXECUTOR_PROMPT.txt` 的反模式清单同样适用。
- 步6（若治理放行）：伙伴库修复**不得**顺手把「谁 serve」变成几何可解（保留协商性）。
- 步7：治理门**只加检查、不改变已修复的机制行为**——它是「新主张前必须先过」的验收关卡，
  不是又一层可以掩盖上游未修问题的防御性包装。
- 每步 handoff 附一条「该修复是否替方法完成了它应演示的事？」的显式回答。

### 2.5 再验证（远程、执行门控、Type-B 接受）

- 全部按 [CUSTOMER.md](CUSTOMER.md) 远程执行；每次代码更新出显式 git diff。
- NEW-4 隔离：CODEX_IMPL_SPEC v1-v4 及其远程数字只能作为 diagnostic-only 记录；不得进入
  `EXPERIMENT_LOG.md` 的支持/反驳结论，也不得作为 METHOD_LOCK 新 claim 的依据。
- **决定性重跑**（去 oracle 后：ARIS 是否仍适应？四臂是否分离？asymm serve CE 是否转正？）
  = Type-B → 结果须 codex + 用户接受后才可写任何主张；结论按 §3.1 预注册规则读出，
  **不允许事后重解释**。
- 首个真实结果落盘时创建 `EXPERIMENT_LOG.md`（补齐 ARIS 惯例缺口）。

### 2.6 活文档更新（追加，不静默覆盖）

- METHOD_LOCK：只允许**带日期新段**（沿 §8/§11/§15 的既有先例）；禁止改写旧段。
- FIDELITY_GATE：新不变量 I10–I14（§3.3）实现后并入门。
- PROJECT_DASHBOARD §3：主张表加一列「**当前可测？被哪条台账 ID 阻塞**」。
- 台账 ID 全链路贯穿：handoff → commit → 结果 → METHOD_LOCK 段落。

---

## 3. 阶段三：防走偏护栏

两次真实翻车（CE-0 → 「布局无外部性」；parity → 「方法无优势」）的共同根因是**缺少
「这是不是伪影」的前置门**。六道护栏，每道对应一次已发生的走偏。

### 3.1 决策规则预注册

每个待跑实验，**启动前**在 METHOD_LOCK 新段写死「结果 → 结论」映射（沿 §5
branch-by-decision-rule 的格式）。首个必须预注册的是去-oracle 决定性重跑，示例分支：

```
ARIS > flat > base      → 因子相关性路由获得支持（核心主张的判别证据）
ARIS ≈ flat > base      → 信念有用、路由无增益（主张收窄）
四臂 ≈                  → 先过 §3.2 伪影自检；全部排除后才允许记「方法无优势」
ARIS 崩、基线不崩       → 去 oracle 实现引入回归，修复而非下结论
```

### 3.2 伪影自检清单（负结果的强制前置门）

任何 null/负结果在被允许驱动 pivot 或写入 METHOD_LOCK **之前**，逐项排除：

- [ ] 低于支持度/哨兵值？（CE=0 教训——查 per-pair support）
- [ ] 仪器根本看不见它？（估计量占用率偏置 D4）
- [ ] oracle/泄漏使所有臂拉平？（parity 教训）
- [ ] 指标饱和或计数口径宽松？（completion 单次送餐即 1.0，含伙伴送餐）
- [ ] 基底退化？（伙伴行为同一性探针）

任一命中 → 该结果标记 `ARTIFACT-SUSPECT`，不得作为科学结论记录。

### 3.3 FIDELITY_GATE 新增不变量（I10–I14 规格；静态检查，边界内可跑）

| # | 不变量 | 静态检查要点 |
|---|---|---|
| I10 | 部署证据路径无 oracle：`partner_option_*` 通道只允许来自行为推断器 | grep option_executor/evidence_router 对 `partner_action.option_id` 的路由 + config 开关状态 |
| I11 | 信念可累积：跨决策隐状态存在，或窗口 ≥ 阈值 | factor_belief/replay 结构 + `evidence_window` |
| I12 | CE 产物必含 per-pair support；无 `support ≥ min_weight` 不得引用「CE=0」 | sidecar schema + graph 元数据 |
| I13 | 伙伴可区分性 preflight：layout×pool 成对轨迹分歧探针未过 → 禁止 ARIS-vs-基线主张 | 探针产物存在且记入 preflight 元数据 |
| I14 | 完整性旗必须活着：`reward_scale_verified` 在匹配运行上可为 true；`--allow_diag_skip` 只跳诊断 | 门作用域分离 + provenance 口径一致 |

（先写规格进 handoff，实现属步 6 / 门工具更新，仍在静态边界内。）

### 3.4 主张范围冻结

主张措辞、何为「支持」、dev-heldout vs blind-heldout 的区分、终表最低 seed/episode 数
（≥5 seeds × 50–100 ep，沿 METHOD_LOCK §6）一次性冻结；任何变更须**带日期的决策记录**，
禁止静默编辑（cautious-claim-scope 纪律）。

### 3.5 关注点隔离

基底/估计量/评估修复**不得**触碰 policy/method 代码，反之亦然；每个 handoff/PR 声明所属
层；**方法改动与基底改动永不共用一次提交**。（free-riding 修复干净 vs 伙伴库重设计危险，
差别就在这条。）

### 3.6 人类 Type-B 检查点（自治可驱动、绝不自我开释）

固定四处：① 接受台账中任一 pillar 的最终裁决；② 接受任何一次重跑为「支持/否定主张」；
③ 变更主张范围；④ **启用重设计的伙伴库**（§5.1）。

---

## 4. 执行序列

| # | 动作 | 执行者 | 边界 | 门 | 产物 |
|---|---|---|---|---|---|
| 1 | 写 REVIEW_BRIEF + PROMPT 包（§1 规格实例化） | Claude | 静态 | — | `review_bundles/rootcause_review_20260702/` |
| 2 | 建 FINDINGS_LEDGER（附录 A 预填，状态=待 codex） | Claude | 静态 | — | `FINDINGS_LEDGER.md` |
| 3 | 治理裁决：role_conditioned_v1 伙伴库 | **用户** | — | Type-B | METHOD_LOCK 带日期记录 |
| 4 | 运行 codex 评审 | **用户**（`codex exec`，xhigh） | 用户执行 | — | `CODEX_OUTPUT.md` |
| 5 | verdict 灌入台账；分歧对质与判决 | Claude + 用户 | 静态 + Type-B | 双签 | 台账更新 |
| 6 | 按 §2.3 逐步写静态修复 handoff | Claude | 静态 | codex diff 评审 | handoff 文档 + 提交 |
| 7 | 远程验证（探针 + 决定性重跑） | 用户授权远程 | 执行门控 | preflight + I 门全绿 | 结果 + EXPERIMENT_LOG |
| 8 | 决定性结果接受 | codex + 用户 | Type-B | §3.1 预注册规则 | METHOD_LOCK 新段 |

步 1–2 可立即执行且互不阻塞；步 3 需在步 6 触及伙伴库（§2.3 步4）之前完成即可，
不阻塞评审启动。

## 5. 阻塞的用户决策

**5.1 治理（阻塞 §2.3 步4）**：工作区未提交的 role_conditioned_v1 伙伴库重设计 =
METHOD_LOCK 末尾的「选项(b)：改脚本伙伴打分，可能超出『标准』」，与 7/1 合成数据禁令
存在边界问题。两个选项：
- **允许**：登记为「标准化伙伴库 v2」，按 W1–W3 修完再用；
- **不允许**：判别性 ZSC 测试换伙伴来源（如训练出的 FCP/MEP 群体——EXPERIMENT_PLAN
  的原始设计），伙伴库重设计从工作区剔除。

**5.2 评审授权**：批准本计划后，步 4 由用户在终端运行（Claude 备好 PROMPT 包，一条命令）。

---

## 附录 A. 发现登记册（台账种子 + 简报锚点表来源）

> 完整锚点（file:line + 代码事实 + 证伪方法）在 REVIEW_BRIEF 中展开；此处为 ID 索引。
> 分类：IMPL=实现错误 SUB=基底设计缺陷 EST=估计量局限 MTH=方法设计局限 GOV=治理。

**支柱（P）**
- P1 IMPL/高：伙伴真值选项注入证据流（train/eval/CE；≥11 通道；推断器死代码；违反 option_executor 自身契约）
- P2 SUB/高：HEAD `_protocol_score` +100 几何奖励吞 +4/−1 角色奖励 → 紧凑布局伙伴行为同一（7f0d4fa 基建修复引入）
- P3 EST+IMPL/高：serve_soup CE=0.000(asymm) 为 min_weight=20 哨兵（17 vs cramped 91 样本）；§11 布局结论不安全
- P4 IMPL-vs-声明/高：信念 = 4 原始步滑窗、零隐态重编码，「累积失败痕迹」机制结构缺席
- P5 声明层/高：奖励/探索/播种以伙伴 terminal_policy 真值为条件；eval 回报同源

**次级（S，节选主项）**
- S1 IMPL/中：`ego_option_terminated_failed` 漏 `"max_steps"`；`"option_invalid"` 无生产者
- S2 IMPL/低：GRU 吃零填充为证据 · S3 失败步证据重复 · S4 pot cooked≡ready 通道重复
- S6 IMPL/中：裸 ce_sampler CLI 不排除 held-out 伙伴 · S7 批处理 CE local-return 塌缩（潜伏）
- S8 IMPL/中：覆盖门证团队事件非 ego 可估性 · S9 skipped-vs-measured 零不区分 + refine 幸存偏置
- S10 γ/horizon 硬编码 · S11 `--sparse_ce_support` 对目标门不可见 · S12 preflight 回退 team credit
- S16 IMPL：gru 诊断形状崩溃 · S17 IMPL/中：`--allow_diag_skip` 短路全部完整性门（记录运行全臂使用）
- S18 IMPL/中：`reward_scale_verified` 永假（provenance 口径） · S19 IMPL：diagnose_traces.py 三缺陷不可运行
- S20 completion 计伙伴/错误送餐且饱和 · S21 eval 种子惰性 + 跨伙伴同序列 · S22 value_bound 臂间不匹配(asymm cfg)
- S23 `train_partners` 未设→全池回退 · S24 provenance 缺失仅告警 · S26 specs.py 重复字段

**工作区伙伴库重设计（W，未提交）**
- W1 SUB-new/中高：恢复项未随 ×1000 重缩放 → 角色伙伴活锁风险
- W2 SUB-new/高：`heldout-yield-terminal-claim` ≈ 训练伙伴近重复，且是 v4 分析的 claim 伙伴
- W3 SUB-new/中：yield 层内排序 → 蹲守瓶颈而非备菜 · W4 alternate 不交替 · W5 拥堵死连词 · W6 空集回退指 option0 · W7 button 全禁

**设计层（D）**
- D1 MTH：伙伴差异性前提从未写为可检前提；`partner-induced return variance` 未入 accept_layout()
- D2 MTH：reference_base_gap 门只在训前查，课程天花板可击穿
- D3 MTH：闭合模式集，仅重组式泛化（非声明 1/6 诚实，Exp2 预期矛盾）
- D4 EST：CE 估计量把可测性与占用率混同 · D5 MTH：价值充分性相对训练分布定义（鸡生蛋）

**豁免（E，已验证正确——codex 应同样尝试推翻）**
- E1 TD 损失/目标/double-Q/动态 next-option 掩码 · E2 相关性路由 + 因子删除三要素
- E3 变体接线真实（parity 非接线伪造） · E4 无伙伴 ID 入 ARIS · E5 checkpoint 选择仅训练伙伴
- E6 当前 config 下 split 强制有效 · E7 sparse_credit 归因正确 · E8 合成伙伴回退干净
- E9 A1/A2/A7 消融「图对 ARIS 控制器因果」成立
