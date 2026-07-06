# CLAUDE_ADJUDICATION_ROUND2 — ChatGPT Pro 第二轮深审（对 rev2）的逐条裁决

**Date:** 2026-07-05 · **输入**：`CHATGPT_ROUND2_OUTPUT.md`（23 条，对象 = rev2 修订后
的三份文档）· **前轮**：`CLAUDE_ADJUDICATION.md`（round-1，22 条，A/B/C 三类修订已执行）
**核对材料**：U2_ROLE_BELIEF_PROPOSAL.md（rev2）/ ARIS_DESIGN_MASTER.md / U2_MODE_FILTER_SPEC.md /
U1_INTERVENTIONAL_CE_SPEC.md / ICLR_UPGRADE_PLAN §5 / METHOD_LOCK sec18.11–18.13 /
EXPERIMENT_CHAIN_PLAN 补充 D
**性质**：静态裁决，未执行任何实验。处置建议待用户裁决后落盘。

---

## 0. 总裁决

第一轮打的是"结论怎么读"（读出越权、证据缺项）——rev2 已修。**第二轮打的是 rev2 新增
内容本身的可执行性与可判定性**：§4a/§4b 把"该做什么"写全了，但"怎么算过/不过"大面积
留空；且有两处 rev2 自己引入/漏掉的**实锤内部不一致**（[07][08]，见下）。23 条中
（[03] 与 [23] 为同一问题）：**17 条成立、4 条部分成立（已有部分覆盖或最强表述需降级）、
0 条误读**。没有一条引用失实。

**诚实声明**：[07]（spec §6 读出未随 rev2 降级）与 [08]（§4a 把依赖训练后 posterior 的
诊断放进了"训练前"门）是 rev2 修订执行时引入/漏掉的缺陷，属于上一轮静态落盘的质量问题，
本轮裁决予以确认。

---

## 1. 逐条裁决表

| # | 原severity | 裁决 | 核对结果（文本锚点） | 处置类型 |
|---|---|---|---|---|
| 01 E1-rev3 事实上门控命题级检验（条件性可证伪） | 致命 | **成立**（表述需校正） | proposal §4 读出表第 3 行：E1-rev3 失败 ⇒ population 后备启动、U2 降为工程改进——E1-ext 便不再执行。校正：E1-rev3 失败合法地终结 **U2 机制主张**；但 ext-A 伙伴组 / OASIS-N / 容量对照这些资产对 population 后备线**同样承重**（评估生态与判据判别力与方法无关），无条件执行的理由是复用，不是"命题不许死" | A：最小反证包（ext-A ≥4 伙伴、ext-B OASIS、ext-C ≥1 对照）改为**无条件执行**，E1-rev3 只决定训练预算扩不扩 |
| 02 "非角色伙伴"多为角色轴连续化 | 致命 | **成立**（部分已覆盖） | §4b-1 四族中三族（p 插值、噪声路径、混合角色）仍以"谁上菜"为生成主轴；仅 RL self-play 可能脱轴，且未要求其无终端偏好参数化 | C：ext-A 增设"完全非上菜轴"子组（纯 team-reward self-play、拥塞规避/资源预取/路径短视/等待阈值/探索温度参数化）+ 分析次序规则：先预测性指标、后语义命名 |
| 03+23 "可解释、可预测互补行为的低维结构"无操作定义（reviewer 自认全文最薄一句） | 致命 | **成立** | §4b-1 读出原文即该句；无阈值、无主指标、无失败条件、无盲评协议 | B：替换为四硬指标——posterior 对未来伙伴行为/贡献的 held-out AUC 或 log-lik ≥τ；posterior-conditioned 策略相对 matched 基线的 role-adaptive regret 降低 ≥δ；模式数由复杂度-性能曲线定，不肉眼判；解释性 = MI 阈值（含 nuisance 对照）或盲评协议；mode permutation 与跨种子对齐后稳定 |
| 04 "开局不确定共享"未形式化；uniform prior 是数学初值不是习得认知状态 | 致命 | **成立** | §3 核心论证原文未变；§4a-2 OOD 审计只测特征漂移，不测"不确定状态下的行为是否被学到" | B：ambiguous-prefix 诊断（多伙伴前 N 步同行为、N 后分化，检验前缀期校准不确定+信息采集、分化后快速切换）+ 报告训练 replay 中"posterior 高熵且动作选择影响回报"状态占比（≈0 则该论证只是口号） |
| 05 TD-only + 语义诊断无硬失败阈值 = 事后解释 | 致命 | **成立** | SPEC §1 TD-only 原文；§4b-3 诊断清单无阈值无后果 | B：预注册语义失败条件（MI(nuisance)>MI(role)、belief-swap 不稳定、跨种子对齐<阈值 任一触发 ⇒ 论文措辞只许 "persistent discrete latent state"，禁用"角色信念"）+ supervised diagnostic-only probe（同证据预测机会条件化角色，probe 能而 modes 不对齐 ⇒ TD 没学到角色） |
| 06 E1-rev3"单一变量"不成立（≥6 个耦合变化） | 致命 | **成立**（部分已覆盖） | §4 仍写"**单一变量**：信念模块 gru/option-infer → BayesModeFilter"；实际同时改递归形式/隐态维度/λ 先验/mask 语义/mode embedding/数值域。E1-ext-C 已承担分解归因，故 E1-rev3 处是措辞问题 | A：改为"单一**模块**替换（模块内多变量耦合；分解归因由 E1-ext-C 承担）"；E1-ext-C 网格并入 [12] 的细化 |
| 07 spec §6 读出仍写"C2 headline"，与 rev2 降级口径冲突 | 重大 | **成立（实锤，rev2 漏改）** | U2_MODE_FILTER_SPEC §6 第 1 行 "aris(bayes_mode) > aris(gru) > flat ⇒ 结构化模式递归是真实贡献（C2 headline）"；ICLR_UPGRADE_PLAN §5 同款 "→ 模式层信念是真实贡献(C2 支持)"。METHOD_LOCK sec18.11(3) 指向该 §5 | A：spec §6 与 ICLR_UPGRADE_PLAN §5 同步降级（"⇒ 在当前脚本伙伴+现图上为候选机制贡献；C2 headline 需 E1-ext-C 容量匹配 + E1-ext-A 非角色伙伴 + E4 belief-swap 同时支持"）；sec18.14 定稿时声明读出口径以其为准 |
| 08 §4a 混合"训练前可做"与"训练后才存在"的诊断 | 重大 | **成立（实锤，rev2 起草错误）** | §4a 标题写"执行于训练/判读之前"，但 §4a-2 的"后验熵、角色切换延迟"与 §4a-3 的"后验只允许在第一类偏向让位"都依赖**训练后的 U2 posterior** | A：§4a 拆两层——训练前门（特征 OOD、support audit + override、命名盲化）与训练后机制门（posterior 熵/切换延迟分层、matched-scenario posterior、belief-swap）；训练后门不过 ⇒ E1-rev3 只算性能 smoke，禁入机制证据 |
| 09 override 门无统计判据，stop/go 主观 | 重大 | **成立** | §4a-1 只有定性分支，无样本量/效应阈值/CI | B：预注册 override 门统计（状态抽样规则、最少状态数、每状态 rollout 数、主指标=强制接手相对等待的 20-step return lift、bootstrap CI 下界>0、最小效应 δ、完成率提升比例） |
| 10 OASIS-N 信息预算未定义（可能特权 or 稻草人） | 重大 | **成立** | §4b-2 的 yield 计分用"上菜对对方公共可行"手工谓词 = 高于 U2 的符号状态抽象 | B：三档定义——OASIS-raw（只读与 U2 相同的 x_t^f）/ OASIS-public（公共状态+valid options）/ OASIS-oracle-feasibility（上界，不进主表）+ 一个廉价学习基线（logistic/HMM role filter + 有限状态控制器）。U2 须胜 raw 与学习版、接近 public |
| 11 "ego 贡献>0"仍是非零事件门（25 局投一次料即过） | 重大 | **成立** | §4 加严判据第三项原文 ">0"——与 round-1 批评 egoCCR>0 同类缺陷在加严项里复现 | B：改 normalized assistance——相对 idle-ego / nonblocking-ego / base_only 对照的"每完成汤的被使用备菜贡献率"≥δ、partner 等待时间下降；阈值入 sec18.14 |
| 12 容量匹配对照不真匹配（参数量/先验/支撑/优化性） | 重大 | **成立**（E1-ext-C 的细化） | §4b-3 三对照未匹配参数量；BayesMode 另有 embedding/共享 g_θ/log-domain/mask/λ | B：加 param-matched GRU、λ-matched GRU、frozen-random-g_θ Bayes、GRU+mask+centered-belief 接口；报告参数量表；"角色粒度贡献"分解为低维持久/Bayes 归一/词表/mask/λ 五个可归因成分（与 [06] 网格合并） |
| 13 U2×图证据纠缠：E1-rev3 用现图，论文另有 U1 图与 E5 | 重大 | **部分成立** | ICLR_UPGRADE_PLAN §5 已有"干预式CE图 vs 被动CE图（同臂对比）"——图对照存在但非全交叉；主表用哪张图确实未声明 | A：预注册主表图口径——论文主表 U2 结论须在**同一张最终图**上读出；若 E1-rev3 旧图≠最终图，旧图结果标注为开发记录。全 2×2（GRU×干预图）仅在归因需要时加一臂 |
| 14 "interventional CE" 术语过强（伙伴未被干预，FSM 塑形分布） | 重大 | **成立** | U1 spec 承认 CE_int 是"FSM 分布下的估计量"，pair-level runner 为可选二期；术语仍叫 interventional CE | A：改名"interventional support top-up"（干预式支持补采），CE_int → CE_topup；"interventional causal effect"术语保留给 pair-level runner 或 paired counterfactual starts 实现之后（C1 headline 的升级条件） |
| 15 source quota / tie-breaking 未冻结 = 图级调参通道 | 重大 | **成立** | U1 §2.3 "加 source quotas 或 tie-breaking"未定值 | B：实施前冻结 quota 数值、tie-break 顺序、max_factors、类别优先级、CE 阈值，写入 sec18.14；加敏感性（quota±1、max_factors±k、随机 tie seeds），结论仅单一 quota 下成立则图发现主张降级 |
| 16 I20-blind 只防字面量，防不住 registry/顺序/元数据泄漏 | 重大 | **成立**（部分已覆盖） | graph 侧存在 `mandatory_role_contrast` 类别、mode_mask 构造顺序、partner_set 排列等非字符串通道；mode permutation 检验已在 §4b-3，但 id/顺序/目录级未覆盖 | B：I20-blind 升级为 semantic blind audit——partner/option/mode/factor id 随机重排、分析脚本只读 hashed id、解盲前出主表；mode-order permutation invariance 与 §4b-3 合并执行 |
| 17 指标族扩大但无 primary endpoint / 多重比较规则 | 重大 | **部分成立** | E1-rev3 有明确门（双判据+两项轨迹指标进读出）；但全局无检验层级/校正，E1-ext 各读出无主指标（与 [03] 同源） | B：sec18.14 定义 primary composite（role-adaptive regret：yield 接管 regret + claim 干扰 regret + 团队分非劣罚项）+ 层级检验（先 primary 后诊断族，Holm 或 hierarchical），其余全部标注 diagnostics |
| 18 时间线低估 E1-ext 矩阵 → 选择性完成风险 | 重大 | **部分成立** | E1-ext-C/D 是训练臂乘数（约 7–8 个新训练臂）；但 E1-rev 实测吞吐 25 run/天（总方案 §7），物理量可行——风险在管理不在算力 | C：现在冻结 8 月实验矩阵与**最低可投稿子集**（建议：ext-A 4 伙伴、OASIS-raw/public、2 个 matched 对照、λ∈{0,0.02}、1 个 population 基线），每项标 seeds/episodes/停止规则/失败写作口径；其余标 extended |
| 19 "FCP 风格群体线"规格不足以当正面对比 | 重大 | **成立** | §4b-5 与总方案 §7-5 只有 spike 描述，无算法/池规模/预算/选择协议 | C：population baseline 规格预注册（池生成法、伙伴数、训练预算、多样性目标、checkpoint 选择、评估协议）；不能忠实复现公开 FCP/MEP 设定则命名"diverse-population baseline"，不得引用为同等对比 |
| 20 人类/human-BC 伙伴不在命题级证据里 | 中等 | **成立**（处置=先限定口径） | §4b-1 无人类数据；OvercookedV2 asymm 布局公开 human-human 轨迹可得性未知 | C：写作口径先限定为 scripted/RL-partner ZSC（呼应 cautious-claim-scope 纪律）；human-BC proxies（3–5 个）列为 extended，视数据可得性裁决 |
| 21 zeroed 自检"变好=实现 bug"解释过硬 | 中等 | **成立** | §4 附带读数原文"如果 U2 置零后反而变好，说明实现有问题"——排除了"学习似然有害/失准"这一合法机制读出 | A：三分支读出——zeroed 变差=证据有用；持平=证据被忽略；**变好且全部工程门通过=学习似然有害/失准（机制发现）**；仅当 mask/provenance/梯度异常时判实现 bug |
| 22 "不加伙伴、不动奖励"与证据包生态边界不清 | 中等 | **成立** | rev2 一句话仍以"不加伙伴"开头，而 §4b 含伙伴构造与 population 叠加 | A：表述拆分——E1-rev3="无新训练伙伴、不动奖励的纯模块干预"；E1-ext="主张验证所需的评估/对照生态"；U2+population="另一个组合方法，不属纯 U2" |

---

## 2. 合并后的五类问题（round-2 的问题所在）

**Q1 可判定性真空（03+23、05、09、11、17）**——rev2 新增的门与读出大多没有"怎么算过"
的硬定义：E1-ext-A 读出是定性句子、语义诊断无失败后果、override 门无统计判据、
"贡献>0"是非零事件门、指标族无 primary/校正。全部可静态修（预注册定义），这是本轮最大
也最便宜的一类。

**Q2 证据包构造有效性（02、10、12、20）**——非角色伙伴仍绕着上菜轴转、OASIS 信息预算
未定、容量对照不真匹配、人类外部有效性缺位。需要少量新增对照臂与口径限定。

**Q3 rev2 自引入的内部不一致（07、08、06、21、22、13 部分）**——spec §6/升级计划 §5
读出没同步降级（漏改）、§4a 训练前/后诊断混排（起草错误）、"单一变量"措辞、zeroed
读出过硬、"不加伙伴"叙事边界、主表图口径未声明。全部零成本文字修正。

**Q4 U1 的术语与自由度（14、15）**——"interventional CE"名不副实（应叫干预式支持补采，
因果术语留给 pair-level runner）；quota/tie-breaking 未冻结是图级调参通道。

**Q5 执行结构（01、04、18、19）**——最小反证包应无条件执行（资产对后备线同样承重）；
"不确定状态"的训练支持要量测（ambiguous-prefix + 高熵状态占比）；8 月矩阵要现在冻结
最低子集；population 基线要规格化。

## 3. 对 reviewer 的降级/校正（供追问轮使用）

- [01]：E1-rev3 失败合法终结 U2 机制主张，"条件性可证伪"的指控对**机制**不成立；成立的
  部分是评估资产复用——无条件执行的理由是它们对 population 后备线同样承重。
- [06]：分解归因已由 E1-ext-C 承担；E1-rev3 处是措辞修正，不是新实验需求。
- [13]：ICLR_UPGRADE_PLAN §5 已含同臂"干预图 vs 被动图"对照，reviewer 低估；真缺口是
  主表图口径声明。
- [18]：E1-rev 实测吞吐（25 run/天闭环）使矩阵物理可行；成立的部分是"最低子集未冻结"
  的管理风险，不是算力不可行。

## 4. 处置分类（执行方式变更记录，2026-07-05）

**用户反馈：rev2 的逐条补丁式修订不可接受（补丁自产不一致，[07][08] 即证据）。**
处置方式改为**重构**：本轮 A/B/C 各项不再逐条落盘，而是整体并入重写的
U2_ROLE_BELIEF_PROPOSAL **rev3**（主张→子命题→证据→判读→门的结构，文末带两轮溯源表）；
读出口径收敛至 rev3 §5 唯一权威，U2_MODE_FILTER_SPEC §6 与 ICLR_UPGRADE_PLAN §5 已
指针化。round-1 遗留的 [16]（因子间后验相关）与 [20]（ITT 主表）也已在 rev3 §5.2-G3 与
§5 报告规则中获得结构位置。U1 相关项（R2-14/15）与 design master 行刷新列入 rev3 §10
"签收后一次性同步清单"。原分类表保留如下作裁决记录。

**A 类（零成本文字/预注册修正，签收前必改）**：[07] spec §6 + ICLR_UPGRADE_PLAN §5 读出
同步；[08] §4a 拆训练前/后两层；[06] "单一变量"→"单一模块替换"；[21] zeroed 三分支读出；
[22] 叙事边界拆分；[14] U1 术语改名；[13] 主表图口径；[01] 最小反证包无条件化。

**B 类（判定性硬化——预注册阈值与少量新增诊断）**：[03+23] E1-ext-A 四硬指标；[05] 语义
失败条件 + diagnostic-only probe；[09] override 统计门；[11] normalized assistance；
[17] primary composite + 层级检验；[04] ambiguous-prefix + 高熵状态占比；[16] semantic
blind audit；[10] OASIS 三档 + 学习基线；[12] 容量对照矩阵细化。

**C 类（scope/资源增项，需用户裁决）**：[02] 完全非上菜轴伙伴子组 + 分析次序规则；
[18] 冻结 8 月矩阵与最低可投稿子集；[19] population baseline 规格预注册；[20] 写作口径
限定 + human-BC proxies 列 extended。
