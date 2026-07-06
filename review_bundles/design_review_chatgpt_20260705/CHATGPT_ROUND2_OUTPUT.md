## 1. 问题清单

[01][致命] 修订版把核心命题的裁决推迟到 E1-ext，但 E1-rev3 仍在事实上控制是否进入命题级证据阶段。

被攻击的具体主张：文档说“它是否支撑‘角色粒度’这一论文核心命题，由 E1-rev3（smoke）与 E1-ext 证据包共同裁决，不由单个实验裁决”；同时 E1-rev3 被定位为“是否进入 8 月正式实验的继续执行门”，而 E1-ext 被排到后续 8 月窗口。

隐含假设是什么、为什么可能不成立：隐含假设是 smoke gate 只是工程筛查，不影响命题可证伪性。但如果 E1-rev3 失败后直接启动 population 后备、U2 降级为工程改进，那么非角色伙伴、OASIS-N、容量匹配、λ sweep 等真正能 falsify 核心命题的实验可能根本不会被完整执行。这样“核心命题由 E1-ext 裁决”变成了条件性承诺：只有脚本世界 smoke 成功，才进入更困难的非循环检验。

具体失败情景或审稿人指控：“你的核心命题不是无条件可证伪的。若 U2 在脚本伙伴上失败，你就不会跑非角色伙伴；若它成功，你才收集命题级证据。这是 development gating，不是 scientific falsification。”

能提前解决/检验它的最小实验或修改：把 E1-ext 的最小反证包改成无条件执行：至少 E1-ext-A 的 4 个非角色伙伴、E1-ext-B 的 OASIS-N、E1-ext-C 的一个容量匹配对照必须不依赖 E1-rev3 成败执行。E1-rev3 可以决定是否扩大训练预算，但不能决定是否检验核心命题。

[02][致命] “非角色生成伙伴组”仍大量围绕同一上菜角色轴构造，可能只是把 claim/yield 从离散标签改成连续旋钮。

被攻击的具体主张：E1-ext-A 计划构造“不带 claim/yield 标签”的伙伴，包括“终端偏好概率 p∈[0,1] 连续插值、噪声路径/延迟变体、混合角色（不同因子上不同倾向）”。

隐含假设是什么、为什么可能不成立：隐含假设是只要不显式给 claim/yield 标签，就不再是角色生成世界。但“终端偏好概率”“混合角色”“噪声路径/延迟变体”仍然以“谁上菜/谁让位”为主轴生成。它们可能只是 claim/yield 的连续化、噪声化和组合化，而不是独立于角色词表的伙伴分布。

具体失败情景或审稿人指控：“你们声称加入了 non-role partners，但这些伙伴仍由 terminal-claim propensity 生成。你们没有脱离原 ontology，只是把角色生成器从 categorical 改成 continuous。”

能提前解决/检验它的最小实验或修改：E1-ext-A 至少加入一组真正非角色轴生成的伙伴：独立 RL self-play seeds 只以 team reward 训练、无 terminal preference 参数；人类或人类 BC 轨迹；以拥塞规避、资源预取、路径短视、等待阈值、随机探索温度等非上菜轴控制的伙伴。分析时先不使用 claim/yield 解释，先问 U2 posterior 是否提高 best-response prediction 和 team regret，再事后解释模式。

[03][致命] E1-ext 的核心读出“形成可解释、可预测互补行为的低维结构”没有操作性定义。

被攻击的具体主张：E1-ext-A 的读出是“U2 后验在无标签伙伴上是否仍形成可解释、可预测互补行为的低维结构”。

隐含假设是什么、为什么可能不成立：隐含假设是“可解释”“可预测”“低维结构”这些词在实验后不会被主观解释。但当前没有阈值、统计模型、主要指标、失败条件、解释者盲评协议或多重比较控制。这个读出太容易被事后叙事占用。

具体失败情景或审稿人指控：“任何三维 posterior 都能被画成某种低维结构。你们没有预注册什么算 interpretable，也没有说明预测的是 partner action、ego best response、team score 还是 counterfactual role switch。”

能提前解决/检验它的最小实验或修改：把读出改成四个硬指标：第一，posterior 对未来 partner 行为或 partner contribution 的 held-out log-likelihood / AUC；第二，posterior-conditioned ego action 相对 best-response oracle 的 regret；第三，mode 数量用 MDL/AIC 或 performance-complexity curve 定义，而不是肉眼看低维；第四，解释性用盲评协议或预定义互信息阈值，但 MI 不能只对 generator 标签，还要排除位置、时间、是否持物等 nuisance。

[04][致命] “开局不确定状态共享”的理论论证仍没有形式化；当前 OOD 审计只能发现漂移，不能证明训练中学到的是对 epistemic uncertainty 的正确动作。

被攻击的具体主张：核心论证仍写道，“角色后验开局是均匀的……‘开局不确定’这个状态在训练和部署时一模一样”，“训练时 ego 对着 6 个伙伴反复经历‘不确定 → 观察 → 确认角色 → 相应行动’”。

隐含假设是什么、为什么可能不成立：隐含假设是训练时的“均匀 posterior”对应真实 epistemic uncertainty，且网络学会了在不确定状态下试探。但如果训练伙伴很快被早期低级特征识别，均匀 posterior 只存在极短时间，TD 目标可能把它学成默认让位或默认备菜，而不是探索。OOD 审计能发现 x_t^f 分布外，但不能证明 uniform belief 的控制语义被学对。

具体失败情景或审稿人指控：“你们说 unknown partner maps to a known uncertainty state, but the training distribution may contain almost no consequential uncertainty. Uniform prior is a mathematical initial condition, not a learned epistemic state.”

能提前解决/检验它的最小实验或修改：构造 ambiguous-prefix 训练/评估诊断：多个伙伴在前 N 步行为完全相同，N 步后分化为 claim/yield/flexible。检查 U2 是否在前缀期间保持校准不确定、执行信息采集动作，并在分化后快速切换。再报告训练 replay 中 posterior entropy 高且动作选择影响回报的状态占比；如果占比接近零，“不确定状态共享”只是口号。

[05][致命] g_θ 仍只由 TD 梯度训练，E1-ext-C 的语义诊断若没有硬失败阈值，会变成事后解释。

被攻击的具体主张：U2 spec 明确 g_θ 的梯度“只经 belief→Q→TD loss 传导；无监督模式标签、无辅助 likelihood 损失、无 selector 损失、无 oracle 模式标签”。 E1-ext-C 计划用 MI、mode permutation、belief-swap 等语义诊断。

隐含假设是什么、为什么可能不成立：隐含假设是 TD-only 会自然给 modes 赋予角色语义，并且事后诊断足以证明这一点。但 TD 目标只关心控制回报；mode 可以编码时间、位置、汤是否 ready、partner 离上菜台距离、ego 是否持盘等 reward-predictive nuisance。没有硬阈值时，任何部分相关性都能被解释成“角色”。

具体失败情景或审稿人指控：“This is not Bayesian role inference. It is a differentiable recurrent bottleneck trained end-to-end. You call its dimensions roles after seeing correlations.”

能提前解决/检验它的最小实验或修改：预注册失败条件：若 posterior 与低级 nuisance 的 MI 高于与行为角色/互补动作需求的 MI，或者 belief-swap 不稳定改变对应 Q-margin，或者跨种子 mode alignment 低于阈值，则不得使用“角色信念”表述，只能称为“persistent discrete latent state”。此外加一个 supervised-diagnostic-only probe，不参与训练，用同一证据预测机会条件化角色；若 probe 能预测而 U2 modes 不对齐，说明 TD objective 没学到角色。

[06][致命] E1-rev3 的“单一变量”说法不成立；BayesModeFilter 同时改变了递归形式、隐态维度、持久性先验、mask 语义、mode embedding 和数值域。

被攻击的具体主张：E1-rev3 写“单一变量：信念模块 gru/option-infer → BayesModeFilter，训练伙伴、奖励、CE 图、种子、预算、评估协议全部不动”。 但 spec 中 BayesModeFilter 引入固定模式集合、λ 混合、log posterior、mode embedding、g_θ 似然网络、mode_mask 流经等结构。

隐含假设是什么、为什么可能不成立：隐含假设是“换信念模块”就是一个干净变量。但实际至少改变了六个变量：记忆瓶颈维度、递归 inductive bias、先验遗忘、有效支撑、数值稳定机制、下游 centered-belief 输入分布。若 U2 成功，不能知道收益来自角色粒度、低维瓶颈、持久先验、mask 防泄漏，还是 log-domain 稳定化。

具体失败情景或审稿人指控：“The ablation is not isolating role granularity. You replaced a GRU with a different state dimension, different recurrence, different support constraints, and different priors.”

能提前解决/检验它的最小实验或修改：最小 ablation grid：GRU hidden=K_max；persistent 3-logit filter without Bayes normalization；Bayes recurrence with frozen random g_θ；Bayes recurrence with λ=0 and λ matched to GRU decay; BayesMode without mode semantics but same parameter count; GRU with same mode_mask and centered-belief interface。E1-rev3 只能叫 “module replacement smoke test”，不能叫单变量机制证据。

[07][重大] U2 spec 的预注册读出仍旧过度：它说 Bayes > GRU > flat 就是“结构化模式递归是真实贡献（C2 headline）”，与 proposal rev2 的降级口径冲突。

被攻击的具体主张：proposal rev2 明确“E1-rev3 不裁决论文核心命题”，只作为 smoke 与候选证据。 但 U2_MODE_FILTER_SPEC 的读出仍写：“aris(bayes_mode) > aris(gru) > flat | 结构化模式递归是真实贡献（C2 headline）”。

隐含假设是什么、为什么可能不成立：隐含假设是“核心命题”与“C2 headline”可以并行存在，不会在写作时被混用。但 reviewer 只会看到同一方法章节中 spec 和 proposal 的证据口径不一致：一个说不裁决，一个说 headline 贡献成立。

具体失败情景或审稿人指控：“The authors downgraded their main experiment in one document, but another preregistered readout still promotes the same comparison to headline evidence. The evidentiary standard is unstable.”

能提前解决/检验它的最小实验或修改：把 spec §6 同步改为：“Bayes > GRU > flat ⇒ 在当前脚本伙伴与当前图上，BayesMode module 是候选机制贡献；headline C2 需 E1-ext-C 容量匹配、E1-ext-A 非角色伙伴、E4 belief-swap 同时支持。” 所有 METHOD_LOCK 条款使用同一读出口径。

[08][重大] §4a 机制诊断门在时序上混合了“训练前可做”和“训练后才有意义”的诊断。

被攻击的具体主张：§4a 被定义为“执行于训练/判读之前”的前置门；其中包括 OOD 审计要报告“后验熵、角色切换延迟与 ego 行为”，matched-scenario 要求“后验只允许在第一类显著偏向让位”。

隐含假设是什么、为什么可能不成立：隐含假设是训练前就能得到 U2 posterior 的语义行为。但 BayesModeFilter 的 g_θ 是通过 TD 训练得到的；训练前没有已学习的 posterior。若用当前 ARIS 的 GRU belief 代替，又不能检验 U2 的机制。若训练后再跑，则不再是训练前 gate。

具体失败情景或审稿人指控：“Your preregistered pre-training gate refers to quantities that do not exist until after training. This creates discretion: failed diagnostics can be reclassified as post-hoc analysis.”

能提前解决/检验它的最小实验或修改：把 §4a 拆成两类。训练前 gate 只做 feature OOD、control support、override、命名盲化。训练后机制 gate 才做 posterior entropy、role-switch latency、matched-scenario posterior、belief-swap。并明确：训练后机制 gate 不通过，则 E1-rev3 结果最多作为性能 smoke，不得进入机制证据。

[09][重大] override gate 没有阈值，“有效但 argmax 不选”与“override 也无效”的判定可能变成主观开关。

被攻击的具体主张：§4a-1 写“override 有效但 argmax 不选 ⇒ 问题在控制学习；override 也无效 ⇒ 问题在任务状态或图路由，U2 训练暂停”。

隐含假设是什么、为什么可能不成立：隐含假设是“override 有效”可以清楚判定。但后续回报差在 Overcooked 中高方差、强依赖状态选择与 partner 轨迹；没有预设样本数、效应阈值、置信区间、状态抽样方法，gate 可能被人为解释。

具体失败情景或审稿人指控：“The authors decided after inspecting trajectories whether override was effective enough to proceed. This is a subjective stop/go criterion.”

能提前解决/检验它的最小实验或修改：预注册 override gate：状态抽样规则、最少状态数、每状态 rollout 数、primary metric、bootstrap CI、最小效应阈值。例如“强制接手一步相对 argmax 等待的 20-step return lift > δ 且 95% CI 下界 >0，且至少 X% 状态后续完成率提升”。否则暂停条件不具备审计性。

[10][重大] OASIS-N 已被加入，但当前定义可能同时太弱和太强；没有信息预算，无法成为公允廉价基线。

被攻击的具体主张：E1-ext-B 定义 OASIS-N 为“前 N 步只做前置任务并观察对方是否走向盘子/上菜台，维护 claim/yield 计分……阈值只在训练伙伴上定一次”。命题级结论要求 U2 不劣于且至少一项显著优于 OASIS-N。 总方案也把 OASIS-N 作为必须战胜的对照。

隐含假设是什么、为什么可能不成立：隐含假设是这个 baseline 足够强且公平。但如果 OASIS-N 使用“上菜对对方公共可行”等手工 predicate，它可能拥有比 U2 更高级的 symbolic state abstraction；如果限制过严，它又可能是 strawman。两种情况都会削弱比较。

具体失败情景或审稿人指控：“Your rule baseline is either privileged by hand-coded feasibility predicates or deliberately underpowered. The comparison does not tell us whether learned role belief is necessary.”

能提前解决/检验它的最小实验或修改：定义三档规则 baseline：OASIS-raw 只看与 U2 相同的 x_t^f；OASIS-public 可读公共状态和 valid options；OASIS-oracle-feasibility 作为上界但不进主表。再加一个低成本学习 baseline：用训练伙伴轨迹训练 logistic/HMM role filter + finite-state controller。U2 至少应战胜 OASIS-raw 和学习版，接近 OASIS-public。

[11][重大] 成功标准第 2 条已经加严，但“ego 有正向备菜贡献 > 0”仍太弱，可能一帧贡献冒充协作。

被攻击的具体主张：E1-rev3 第二条要求 claim 伙伴上“对方上菜吞吐不低于当前 ARIS；团队得分不低于当前 ARIS；ego 有正向备菜贡献……ego 备菜产出被对方使用 > 0”。

隐含假设是什么、为什么可能不成立：隐含假设是 “>0” 可以排除低干扰旁观。但一次偶然投料、一个被 partner 使用的中间产物就能满足 >0；这仍可能不是稳定配合。团队分不低于当前 ARIS 也可能由 partner 单干保证。

具体失败情景或审稿人指控：“The agent passed the ‘assistance’ criterion by contributing one ingredient in 25 games. This is not cooperation; it is a nonzero-event threshold.”

能提前解决/检验它的最小实验或修改：改成 normalized assistance lift：相对 idle-ego、nonblocking-ego、base_only-ego 三个对照，ego 备菜贡献率、partner waiting reduction、team score lift 必须超过阈值。主指标应是 “partner-used ego contribution per completed soup” 或 “counterfactual team lift”，不是 >0。

[12][重大] 容量匹配对照仍不够精确；hidden=3 的 GRU 与 BayesMode 的参数量、结构先验和有效支撑都不匹配。

被攻击的具体主张：E1-ext-C 计划比较 “persistent GRU(hidden=3)、指数滑动平均滤波、learned 3-logit filter”，并做匿名/过完备 K 扫描。

隐含假设是什么、为什么可能不成立：隐含假设是 hidden=3 就等价于 K=3 的 BayesMode 容量。但 BayesMode 还有 mode embeddings、共享 g_θ、log-domain normalization、mask、λ prior；GRU hidden=3 的可表达性与优化难度完全不同。若 BayesMode 胜出，可能是优化更稳定、支撑更完整、先验更强，不是角色粒度。

具体失败情景或审稿人指控：“The matched baselines are not matched. You compare an engineered Bayesian recurrence with a tiny underparameterized GRU and call the difference role granularity.”

能提前解决/检验它的最小实验或修改：给出参数量表和 recurrence capability 表；加入 param-matched GRU、state-matched GRU、prior-matched GRU、mask-matched GRU、same-gθ learned filter。把“角色粒度贡献”拆成 “low-dimensional persistent state”“Bayesian normalization”“role vocabulary”“mode mask”“λ prior”五个 ablation。

[13][重大] U2 与支持图的证据链仍纠缠：E1-rev3 沿用现图，而论文主张还包含 U1 干预图与 E5 图鲁棒性。

被攻击的具体主张：E1-rev3 写 CE 图“沿用现图”；总方案把“角色粒度命题”与“支持图承重”“干预式发现必要性”分列，但正式 8 月实验又包括 U1 干预图 vs 被动图、E5 图鲁棒性。

隐含假设是什么、为什么可能不成立：隐含假设是 U2 的角色信念贡献可以在旧图上独立读出，并在后续 U1 图上保持。但如果旧图缺终端互斥因子，U2 可能失败不是角色粒度错，而是图缺因子；如果 U1 图改变后 U2 成功，贡献可能来自图而不是 belief filter。

具体失败情景或审稿人指控：“The belief module and graph construction are not separable. You evaluate U2 on one graph, then tell the paper story with another graph.”

能提前解决/检验它的最小实验或修改：预注册 2×2：GRU vs BayesMode × passive/current graph vs interventional graph。至少在最终论文主表中，U2 的结论必须在同一最终图上给出；若只在 U1 图上成立，E1-rev3 旧图结果只能作为工程开发记录。

[14][重大] U1 的 “interventional CE” 仍可能只是 FSM 分布下的支持补采，不是严格的因果交互效应估计。

被攻击的具体主张：U1 top-up 对支持不足选项跑 “ego 策略 = FSM + do(ω when valid)”，伙伴仍是黑箱脚本；CE_passive 与 CE_int 分开估计。 文档承认 FSM top-up 下 CE_int 是“FSM 分布下的估计量”，并把 pair-level runner 放为“可选二期”。

隐含假设是什么、为什么可能不成立：隐含假设是强制 ego option 足以估计“我方选项 × 对方选项”的交互因子。但 partner 选项没有被干预，状态分布由 FSM 强烈塑形，CE_int 可能只是“FSM 把游戏推进到终端后某选项相关”，不是 pair-level causal interaction。

具体失败情景或审稿人指控：“This is not interventional causal effect estimation. It is targeted data collection under an FSM policy, followed by conditional association.”

能提前解决/检验它的最小实验或修改：如果不实现 pair-level intervention runner，就把 U1 改名为 “interventional support top-up for factor discovery”，不要叫 interventional CE。若要保留 C1 headline，必须至少对关键终端因子实现 pair-level runner 或 paired counterfactual starts：同一公共状态下强制 ego option ω 与替代 option，比较 partner response 和 local value。

[15][重大] source-aware graph selection 的 quotas / tie-breaking 仍未具体化，给图构建留下自由度。

被攻击的具体主张：U1 graph_builder 说候选构造成 source-aware pair list，并在每个 mandatory 类别里“加 source quotas 或 tie-breaking”，以避免 interventional pool 挤占 passive 候选。

隐含假设是什么、为什么可能不成立：隐含假设是“source quotas 或 tie-breaking”是工程细节。但在 max_factors 有限时，quota/tie-breaking 直接决定选哪些因子，进而决定 U2 能看见哪些证据。未预注册的 quota 是图结构调参通道。

具体失败情景或审稿人指控：“After seeing results, the authors could tune source quotas and tie-breaking to include the factors that help. This is graph-level hyperparameter leakage.”

能提前解决/检验它的最小实验或修改：实施前冻结：quota 数值、tie-breaking 顺序、max_factors、mandatory 类别优先级、CE threshold。再做 sensitivity：quota±1、max_factors±k、random tie seeds。若结论只在一个 quota 下成立，图发现主张降级。

[16][重大] I20-blind 的 grep 门只能防字面量泄漏，不能防 ontology 通过 option id、mode order、registry、目录结构和 graph metadata 泄漏。

被攻击的具体主张：I20-blind 候选门要求代码不得按 “claim/yield/flexible、handoff/resource-server”等字面量分支，使用 grep tripwire。 Proposal 也把伙伴命名盲化检查作为 §4a 门。

隐含假设是什么、为什么可能不成立：隐含假设是没有字符串分支就没有 ontology leakage。但泄漏可以通过 option ids 的固定顺序、mode_mask 的构造顺序、config 文件名、partner_set 排列、graph 中 mandatory_role_contrast、日志聚合脚本的映射表、目录名等进入。

具体失败情景或审稿人指控：“You removed string matches, but the role ontology is still encoded in the registry and mode ordering. The model and analysis know which mode corresponds to claim/yield through the substrate.”

能提前解决/检验它的最小实验或修改：做 semantic blind audit：随机重排 partner ids、option ids、mode order、factor ids、目录名与 config labels；训练和分析脚本只读 hashed identifiers；外部 blind evaluator 在解盲前生成所有主表。再做 mode order permutation invariance：不同 mode 初始化和 ordering 下，性能与语义诊断应稳定。

[17][重大] 评估指标族扩大后，没有定义主指标与多重比较规则，仍可能发生“挑好看的行为指标”问题。

被攻击的具体主张：E1-rev3 增加团队分、吞吐、ego contribution、首次接管延迟、idle、serve latency、角色切换、碰撞、ego-induced waiting 等 co-report 指标。 总方案也说用“方法无关的行为指标族”统一 co-report。

隐含假设是什么、为什么可能不成立：隐含假设是多指标共同报告就能防止后验挑选。但没有指定 primary endpoint、优先级、失败条件、非劣标准、显著性规则和冲突处理。多指标越多，越容易找到某一项显著改善。

具体失败情景或审稿人指控：“The authors report a dashboard of behavioral metrics and emphasize whichever supports the story. There is no primary criterion.”

能提前解决/检验它的最小实验或修改：预注册一个 primary composite：例如 role-adaptive regret = yield takeover regret + claim interference regret + team score noninferiority penalty。其余作为 diagnostics。对多指标使用 Holm-Bonferroni 或 hierarchical testing：先过团队分/角色 regret，再看解释性指标。

[18][重大] 时间线低估了 E1-ext 的实验矩阵，容易把命题级证据包压成 underpowered appendix。

被攻击的具体主张：时间线把 E1-rev3 小计设为 4–5 个工作日，E1-ext-A/B/C/D/E 放到 8 月正式实验窗口，与 E2–E5 合排。 总方案还把 8 月多组正式实验压进约 2.5 周。

隐含假设是什么、为什么可能不成立：隐含假设是 E1-ext 可以在有限窗口内完整、同等质量地执行。但 E1-ext 至少包含：8–12 非角色伙伴、OASIS-N、多种容量对照、K sweep、λ sweep、population、U2+population。乘上 5 seeds、50–100 局、可能 2 CE seeds，实际矩阵远超 2.5 周的干净执行能力。

具体失败情景或审稿人指控：“The decisive evidence package is too large for the schedule, so the paper likely reports a selectively completed subset.”

能提前解决/检验它的最小实验或修改：现在就列实验矩阵和最低可投稿子集。最低子集建议：E1-ext-A 4 个非角色伙伴、E1-ext-B OASIS-public/OASIS-raw、E1-ext-C 两个 matched controls、λ={0,0.02}、一个 FCP baseline。其余列为 extended。每项标注 seeds、episodes、停止规则、失败时写作口径。

[19][重大] population 线已经被纳入，但“FCP 风格群体线”仍不够具体，不能防 reviewer 的正面对比要求。

被攻击的具体主张：E1-ext-E 写 “FCP 风格群体线数据可用后，跑 U2+population 组实测‘正交且可叠加’声明；若 population 单独满足双判据，U2 主张转为样本/结构效率与可解释性”。

隐含假设是什么、为什么可能不成立：隐含假设是“FCP 风格”足以定义强 baseline。但 reviewer 会要求具体算法、partner population size、training budget、partner selection、diversity objective、checkpoint selection、evaluation protocol。没有这些，population baseline 可能被认为弱化或不标准。

具体失败情景或审稿人指控：“The paper says it compares with a population method, but the implementation is an unspecified FCP-style line, not a recognized strong baseline.”

能提前解决/检验它的最小实验或修改：预注册 population baseline spec：partner pool generation method、number of partners、training updates、selection objective、ego architecture、reward, compute budget, validation selection, held-out partner set。至少一个 baseline 应尽量贴近公开 FCP/MEP/TrajeDi 设定；若无法复现，则写成 “diverse-population baseline”，不要引用为同等对比。

[20][中等] 人类伙伴仍不在命题级最小证据里；非角色 RL 伙伴不能替代人类 ZSC 的说服力。

被攻击的具体主张：E1-ext-A 的非角色伙伴包括 RL self-play、p 插值、噪声路径/延迟、混合角色，但没有人类或人类行为克隆。

隐含假设是什么、为什么可能不成立：隐含假设是非角色 synthetic partners 足以支撑 ZSC role belief 论证。ICLR/NeurIPS 的 Overcooked ZSC 评审通常会把 human compatibility 视为外部有效性关键；尤其本文主张“协作角色”，更容易被问人类是否也呈现这些模式。

具体失败情景或审稿人指控：“The paper is about zero-shot coordination but never tests with humans or human-like partners. All partners are generated inside the authors’ simulator ontology.”

能提前解决/检验它的最小实验或修改：最低成本不是 full human study，而是收集少量 human-human 或 human-agent trajectories，训练 3–5 个 behavior-cloned human proxies，作为 blind held-out。若无法做，论文标题和 claim 必须限定为 scripted/RL partner ZSC，不要暗示 human coordination。

[21][中等] “zeroed 通道自检”当前解释过硬；U2 在 zeroed 下变好不一定是实现 bug，也可能暴露证据通道反向干扰。

被攻击的具体主张：proposal 写 zeroed 通道对 U2 重跑，预期“置零后退回默认”；“如果 U2 置零后反而变好，说明实现有问题”。

隐含假设是什么、为什么可能不成立：隐含假设是证据通道一定正向有用，zeroed 变好只能是 bug。但如果 g_θ TD-only 学到错误似然，证据会把 posterior 推向错误模式；zeroed 反而保留均匀先验，可能减少误导。这是机制失败，不一定是工程 bug。

具体失败情景或审稿人指控：“The authors classify an informative negative result as an implementation bug, precluding a legitimate interpretation that learned evidence hurts.”

能提前解决/检验它的最小实验或修改：把 zeroed 读出分三类：zeroed 变差 ⇒ evidence useful；zeroed 持平 ⇒ evidence ignored；zeroed 变好且工程门通过 ⇒ learned likelihood harmful / miscalibrated，不自动归因 bug。只有同时出现 mask/provenance/gradient异常，才叫实现问题。

[22][中等] 当前文档仍有表述层不一致：“不加伙伴、不动奖励”的 U2 叙事与 E1-ext/population/非角色伙伴证据包之间边界不够清楚。

被攻击的具体主张：U2 proposal 一句话称“不加伙伴、不动奖励，把信念模块从逐步猜对方动作换成持续判断对方角色”；但命题级证据包包括非角色伙伴组、OASIS-N、capacity controls、population 叠加组。

隐含假设是什么、为什么可能不成立：隐含假设是“方法改动不加伙伴”与“论文证据需要加伙伴”不会混淆。但审稿人可能读成：训练不加伙伴是方法主张，评估/证明又依赖新增 partner ecology；如果没有写清，这是 presentation inconsistency。

具体失败情景或审稿人指控：“The method is advertised as not requiring more partners, but the paper’s decisive evidence depends on constructing additional partner populations. The scope of the claim is unclear.”

能提前解决/检验它的最小实验或修改：把表述拆开：E1-rev3 是 “no new training partners, no reward change, module-only intervention”；E1-ext 是 “claim validation across additional evaluation/training baselines”。若 population 用于训练，必须明确这是 separate combined method，不属于 pure U2。

[23][中等] 当前最自信但支撑最薄的一句话是：“U2 后验在无标签伙伴上是否仍形成可解释、可预测互补行为的低维结构。”

被攻击的具体主张：该句是 E1-ext-A 的核心读出。

隐含假设是什么、为什么可能不成立：它把四个难题压成一句话：无标签伙伴是否真无标签、posterior 是否低维、低维结构是否可解释、可解释结构是否能预测互补行为。当前每一项都缺硬定义。

具体失败情景或审稿人指控：“The central evidence package is defined in qualitative language. The authors can declare success after plotting any separable posterior trajectory.”

能提前解决/检验它的最小实验或修改：将这句话替换为可判定版本：“在 blind non-role partners 上，posterior-conditioned policy 相对 non-posterior matched baseline 的 role-adaptive regret 降低 ≥δ；posterior 对未来 partner terminal/prep contribution 的 held-out AUC ≥τ；posterior 与三类 nuisance 的 MI 不超过与互补行为需求 MI 的 α 倍；结果在 mode permutation 与 seed alignment 后稳定。”

## 2. 最强拒稿意见书

修订版把旧过强结论后置，但核心证据仍未形式化：non-role 伙伴多是 claim/yield 轴变体，“低维可解释结构”无判据，BayesMode 对比 GRU 非单变量。E1-ext 像补救清单而非可执行主实验，U1 的 interventional CE 也仍是 FSM 分布下的支持补采。

## 3. 十个我们必须能当场回答的问题

1. E1-rev3 失败时，你们是否仍无条件执行 E1-ext-A/B/C 的最小反证包？
2. 哪些 E1-ext-A 伙伴的生成过程完全不含 terminal preference、claim/yield、handoff、resource-server 或其连续变体？
3. “低维、可解释、可预测互补行为”对应的 primary metric、阈值和失败条件分别是什么？
4. 如果 BayesMode 胜过 GRU，如何排除收益来自 λ、log-domain 稳定性、mode_mask、低维瓶颈或参数量，而不是角色粒度？
5. §4a 的 posterior entropy 和 matched-scenario posterior 是训练前 gate 还是训练后机制 gate？
6. OASIS-N 能读取哪些信息？它与 U2 的 x_t^f、公共状态、valid options 是否信息预算一致？
7. 如果 zeroed evidence 让 U2 变好，你们会把它判为 bug、证据 miscalibration，还是机制失败？
8. U2 的最终主表使用旧 CE 图、passive 图、interventional 图，还是三者都报？
9. U1 的 CE_int 如果只在 FSM+target 下成立、不在 uniform+target 下成立，论文中是否还使用“interventional causal effect”这个术语？
10. 如果 K=8 和 K=3 同样好，且 modes 与 claim/yield/flexible 不稳定对齐，你们的核心命题如何改写？

## 4. 未言明的承重假设清单

1. E1-rev3 作为继续执行门不会引入条件性可证伪问题。
2. “不带 claim/yield 标签”足以等价于“非角色生成”。
3. 终端偏好连续插值伙伴能检验非循环角色命题，而不是检验 claim/yield 的连续版。
4. 6 个训练伙伴足以让 ego 学到对 epistemic uncertainty 的控制策略。
5. 训练中存在足够多“posterior 高熵且动作选择影响回报”的状态。
6. TD-only 的 g_θ 会学到机会条件化角色证据，而不是 reward-predictive nuisance。
7. BayesMode 与 GRU 的性能差可以归因于“角色粒度”，而非 λ、mask、低维、log-domain 或优化稳定性。
8. §4a 机制诊断能在训练前完整执行，尽管部分指标依赖训练后的 U2 posterior。
9. OASIS-N 的手写 predicate 不会比 U2 获得更多任务结构信息。
10. “ego contribution >0”足以排除低干扰旁观。
11. 多指标 co-report 不会导致事后挑选，尽管没有 primary composite。
12. U2 在旧 CE 图上的结论能迁移到 U1 interventional 图。
13. FSM+do(ω) top-up 能支撑 “interventional CE” 术语，而不仅是支持补采。
14. source quota / tie-breaking 不会成为图结构调参通道。
15. grep-based blind gate 足以消除 ontology leakage。
16. E1-ext 的大矩阵能在 8 月窗口内以同等质量完成。
17. FCP-style baseline 足以代表 population 方法，尽管算法细节尚未冻结。
18. 没有人类或 human-BC 伙伴也能支撑“协作角色”这个 ZSC 命题。
19. zeroed evidence 变好只能说明实现 bug，而不会说明 learned evidence harmful。
20. spec、proposal、master 的读出口径会在写作时自然一致，不需要现在强制同步。