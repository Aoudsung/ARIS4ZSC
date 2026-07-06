## 1. 问题清单

[01][致命] 当前实验几乎把“角色是正确粒度”变成了在角色生成器世界里恢复生成器标签。

被攻击的具体主张：“测试台”被定义为“asymm_advantages 布局 + 角色条件化伙伴库”，协调轴明确是“谁去上菜”，伙伴被描述为 claim / yield / 抢行 / 让路；同时核心创新声称“ZSC 信念潜变量的正确粒度是少数离散的协作角色”。

隐含假设是：如果在一个由 claim/yield/flexible 协议生成的伙伴库上，三值角色后验表现好，就能说明“角色”是 ZSC 的自然潜变量。这个假设很可能不成立；它最多说明你们选择了与数据生成机制同构的归纳偏置。

具体失败情景或审稿人指控：“你们没有发现 ZSC 的正确信念粒度；你们把测试伙伴按角色脚本生成，又把模型潜变量设计成同一套角色词表，然后证明模型能恢复脚本标签。这是 ontology leakage，不是表征发现。”

最小实验或修改：必须加入至少一类非角色生成伙伴。最低配置是三组：第一，RL self-play / population training 产生、没有 claim/yield 标签的伙伴；第二，claim-yield 连续插值伙伴，例如以概率 p 接管终端、p 从 0 到 1 连续变化；第三，人类或人类行为克隆轨迹。评估时隐藏伙伴生成器标签与名称，只给轨迹；若 U2 后验仍形成可解释、可预测互补行为的低维结构，角色命题才开始非循环。

[02][致命] claim/yield/flexible 作为 K_f 初值与伙伴库词表重合，使“无标签训练”无法消除先验标签泄漏。

被攻击的具体主张：U2 proposal 写明 K_f 初值取“claim/yield/flexible 三值”，并将其作为记录在案而非调参对象；总方案中伙伴库与测试轴本身也以 claim/yield 描述。

隐含假设是：只要训练时没有显式模式标签监督，就不存在角色标签泄漏。这个假设不成立。标签可通过架构先验、模式数量、模式命名、mode_mask、评估伙伴命名、实验者选择 held-out 对象间接进入。

具体失败情景或审稿人指控：“你们说没有 oracle role label，但你们把 latent cardinality 和 semantic bins 直接从 partner protocol registry 里拿来。无标签 TD 训练只是在已知标签空间里拟合行为，不是发现正确粒度。”

最小实验或修改：做三种盲化对照。第一，把三种模式匿名化、随机打乱、不同种子重采样 mode embedding，看语义是否跨种子对齐。第二，用 K=2/3/5/8 的过完备和欠完备模式，不告诉模型 claim/yield/flexible，比较角色可恢复性与控制性能。第三，让外部脚本随机改名、隐藏 handoff/resource-server 等伙伴名称，保证训练、评估、日志分析代码不能根据命名分支。

[03][致命] “开局不确定状态训练与部署共享”只是把 OOD 问题从伙伴身份推迟到证据特征 x_t^f。

被攻击的具体主张：proposal 的中心论证是“角色后验开局是均匀的”，“开局不确定这个状态在训练和部署时一模一样”，陌生伙伴“照样落在里面”。 但实现中后验更新依赖 x_t^f 与 g_θ(x_t^f,e_f,z)，且 g_θ 只通过 TD 梯度学习。

隐含假设是：陌生伙伴即使行为风格分布外，仍会产生训练中见过的证据特征分布，g_θ 的似然比仍有意义。这个假设是整个 U2 泛化论证中最薄的一环。

具体失败情景或审稿人指控：“你们把 unknown partner 映射成 uniform prior，但第一步证据进入 learned likelihood 后就可能 OOD。一个陌生 yield 伙伴如果绕路、等待、反复试探、卡位或以不同速度接近终端，x_t^f 出分布后，Bayes posterior 只是 overconfident garbage。”

最小实验或修改：在执行前加入证据特征 OOD 审计。对每个 held-out partner，记录 x_t^f 在训练伙伴特征分布下的 kNN distance / Mahalanobis distance / density rank；按 OOD 分位数分层报告 posterior entropy、role-switch latency、ego serve 行为。再构造“同角色不同风格”伙伴：保持 yield/claim 语义不变，但随机化路径、延迟、微动作、停顿、避让策略；如果 posterior 校准崩溃，U2 只能说是 in-distribution role filter。

[04][致命] TD-only 的 BayesModeFilter 不保证学到“角色语义”，可能只是一个三维可微记忆瓶颈。

被攻击的具体主张：实现规格明确写 g_θ 梯度“只经 belief→Q→TD loss 传导”，无监督模式标签、无辅助 likelihood、无 selector loss、无 oracle 模式标签。 Proposal 也强调“无新损失、无模式标签、无 oracle”。

隐含假设是：TD 控制目标会自然选择语义为 claim/yield/flexible 的 latent，而不是选择任何有助于 Q 值的历史特征，例如时间步、汤是否快好、对方是否靠近狭窄通道、ego 当前是否持盘、partner 最近速度等。

具体失败情景或审稿人指控：“你们把一个小 GRU 换成了一个三维 differentiable recurrent bottleneck，然后把它叫 posterior。没有观测 likelihood 目标、没有校准、没有可识别性约束，后验语义不可验证。所谓 role belief 是 post-hoc naming。”

最小实验或修改：必须在训练后做语义识别诊断，但不能把诊断标签用于训练。最低要求：跨种子 mode alignment；posterior 与隐藏 generator role 的 mutual information；posterior 与低级 nuisance 变量的 mutual information；mode permutation 后 Q 行为是否相同；强制 belief-swap，看把 b_yield 注入同一状态是否稳定增加 ego serve Q-margin。还要加入容量匹配对照：K=3 的 persistent GRU / exponential moving average / learned three-logit filter，证明收益来自角色结构而不是低维持久记忆。

[05][致命] 即使后验正确识别“对方在让”，argmax-Q 也未必接手；训练支持可能根本没有“yield posterior → ego serve”的经验。

被攻击的具体主张：proposal 说每一步“没上菜”会让 yield 概率上升，“积累到阈值，路由随之切换，ego 接手”，且“不确定时先试探”由网络在训练中自己学会。 但同一 proposal 的诊断已经显示，当前模型中“等待”始终排在最前，信念 Q 平移翻不动排序。

隐含假设是：训练数据和 TD bootstrap 中存在足够多的状态-动作支持，让模型学会在高 yield belief 下 ego serve 的优势。如果默认策略很少探索接手，Bayes 后验再准也只是给 Q 网络输入一个从未被正向 credit 绑定过的变量。

具体失败情景或审稿人指控：“你们修的是 inference，但失败实际在 control support。Q-learning 不会从没有执行过、没有回报传播的接管动作中发明正确策略。U2 成功如果发生，可能来自额外随机性或训练偶然探索，不是 posterior 机制。”

最小实验或修改：在 U2 训练前后做 support audit：统计 b_yield 高时，ego serve / plate / delivery 相关动作的访问频率、TD target、Q-margin、成功回报。再做 action override：在高 b_yield 且 serve valid 的状态强制 ego serve 一步，看后续回报是否高于等待。如果 override 有效但 argmax 不选，问题是控制学习；如果 override 也无效，问题是任务状态或图路由。这个实验比继续训练 5 个种子更先验。

[06][致命] “两条都达标 ⇒ 论文核心命题成立”是全文最危险的过度读出。

被攻击的具体主张：预注册读出表写明“两条都达标”则“角色层信念实现局内角色适应 → 论文核心命题成立，四种粒度的对照表齐全”。

隐含假设是：E1-rev3 的两条件成功足以从“U2 这个结构在两个 held-out 伙伴上工作”推出“角色粒度是 ZSC 信念潜变量的正确粒度”。这是从工程成功到科学命题的非法跃迁。

具体失败情景或审稿人指控：“你的主结论不由实验推出。实验只比较了一个新 filter 与旧 filter，在两个角色脚本伙伴上是否触发接手/让位。它不能排除持久性、低维正则化、遗忘参数、训练随机性、评估指标选择、廉价规则等替代解释。”

最小实验或修改：把读出改成：“在当前角色脚本伙伴库与该布局上，持久离散模式 filter 改善 role-adaptive behavior。”核心命题成立需要额外证据：非角色生成伙伴、容量匹配 baselines、匿名/过完备 latent、posterior 语义检验、跨布局复现。E1 成功只能进入候选证据，不能作为命题成立门。

[07][重大] “负证据积累”把“没去上菜”直接解释成 yield，但在 Overcooked 中 absence-of-action 有多个公共状态原因。

被攻击的具体主张：proposal 写道，“对方一直没去上菜”是关键证据；每一步“没上菜”都会让“他是让位型”的概率升一点。

隐含假设是：只要对方长时间没有执行终端动作，就说明其角色倾向于 yield。这个假设只在“对方有机会去上菜但选择不去”的反事实成立时才合理。

具体失败情景或审稿人指控：“伙伴没上菜可能是因为汤没好、盘子不可达、ego 挡路、路径拥堵、伙伴被困、伙伴在等 ego 完成前置步骤，或者 partner policy 正在执行另一条贡献链。你们的 filter 把 public-state feasibility 和 partner preference 混在一起。”

最小实验或修改：把负证据改成 opportunity-conditioned negative evidence：只在“partner 可达终端、可持盘或可取盘、路径未被 ego 阻塞、上菜动作公共可行”的时间步累计。构造 matched scenario：同样 100 步没上菜，但原因分别是 yield、被 ego 挡住、无盘、汤未好、partner route bug。posterior 只能在第一类显著偏 yield，否则解释性主张失败。

[08][重大] “角色自适应”判据是在 base_only 完成率失效后提出，容易被指控为为本方法量身定制。

被攻击的具体主张：总方案承认 base_only held-out 0.90，裸完成率无判别力，于是主张“角色自适应”两条并举，并说 base_only 结构上不可能满足第二条。

隐含假设是：新判据虽然是在看到 base_only 高分后确立，但仍是公允、任务独立、非量身定制的评价。这个假设需要额外证据；仅有“预注册”不足以消除后验指标选择嫌疑。

具体失败情景或审稿人指控：“你们先发现常规指标不能赢，然后定义了一个只有你们结构希望满足的 behavioral criterion。这个 criterion 不是社区标准，也没有证明与 ZSC 泛化相关。”

最小实验或修改：把角色自适应指标拆成更一般的、与方法无关的指标：role-conditioned regret against best-response oracle、total team score、partner contribution preservation、idle/deadlock time、handoff latency、blocking/collision rate、partner-normalized throughput。把这些指标对所有 baselines 统一报告，而不是只作为 U2 成功门。预注册只能作为防御材料，不能替代公允基线。

[09][重大] 第二条“对方吞吐不低于当前 ARIS”门槛太弱，可能把“弱不干扰”冒充“配合”。

被攻击的具体主张：成功标准第二条是对 resource-server-claim “保持让位配合，对方的上菜吞吐不低于当前 ARIS 的水平”。 当前 ARIS 对 claim 伙伴的正面发现是让出终端、专心备菜，团队得分比 base_only 高。

隐含假设是：partner throughput 不下降就意味着 ego 在配合。这个假设不成立；ego 可以什么都不做、只是不挡路，也可能让 partner 独立完成。

具体失败情景或审稿人指控：“你的方法通过第二条不是因为它协作，而是因为它退化成低干扰旁观者。partner 吞吐维持住了，但 ego 的备菜贡献、路径让行、资源递交、总效率并没有提高。”

最小实验或修改：加入 partner-normalized assistance score：partner 与 idle ego / blocking ego / scripted helper ego 对照下的 throughput；ego prep 被 partner 使用的比例；partner 端等待时间；total score non-inferiority；collision/blocking counts。第二条应要求“partner throughput 不降 + team score 不降 + ego 有正向辅助贡献”，否则只说明不破坏。

[10][重大] 存在一个足够强的廉价手写基线，可能同时满足两条成功标准，必须成为必须战胜的对照。

被攻击的具体主张：总方案把“若角色自适应判据也被某个简单基线满足”列为风险，但说“目前无迹象”。

隐含假设是：没有简单规则能在该布局上同时“对 yield 接手、对 claim 不全包”。在单一上菜轴布局中，这个假设很危险。

具体失败情景或审稿人指控：“一个 50 行规则策略就能解决你们的两条指标：前 N 步看伙伴是否走向盘子/上菜台，若去就让，若不去就接管。你们的深度 belief 模型没有战胜最明显的 finite-state observer。”

最小实验或修改：实现并公开这个强廉价基线，命名可为 OASIS-N（Observe-Assign-Switch）。规则如下：前 N=40 或 60 步只做前置任务并观察 partner；维护 claim score，包括 partner 接近 delivery/table 的速度、取盘/端汤/进入终端区、尝试上菜、在终端区等待；维护 yield score，只在“上菜/取盘对 partner 公共可行但 partner 连续 M 次未尝试”时增加。若 claim score>τ_c，ego 避开终端、专注取菜/投料/装盘、让路；若 yield score>τ_y 或汤 ready 后 timeout，ego 取盘并上菜；使用 hysteresis 防止频繁切换。阈值只在训练伙伴上调一次，或用固定手工阈值。正式结论必须要求 U2 在两条角色自适应指标、总分、idle time、partner assistance 上显著优于 OASIS-N。

[11][重大] 四粒度对照表目前不是公平的“粒度比较”，更像不同成熟度、容量和调参预算的历史产物拼表。

被攻击的具体主张：总方案把 partner_id_q / global_gru / 当前 ARIS / U2 放在同一“四种认知伙伴方式”表中，并说前三行已测、U2 待测。 Proposal 还说现有 stage-1 结果可直接构成前三行，本实验补第四行。

隐含假设是：这些结果只差“信念粒度”，训练协议、容量、证据输入、调参预算、实现质量、诊断修复程度都可比。这个假设几乎肯定不成立。

具体失败情景或审稿人指控：“global_gru 不是学不出条件化，而是你们没有给它同等容量、持久性、metric、调参和诊断预算。option-infer 的 0.125 可能是实现路径缺陷，不是逐步动作粒度的理论失败。U2 是 fresh implementation with all lessons baked in。”

最小实验或修改：建立 matched baseline suite：相同证据输入、相同 hidden dimensionality 或参数预算、相同 TD objective、相同训练更新、相同完整性门、同等 λ/持久窗口/探索设置。至少包括 persistent GRU(K=3)、global GRU retuned under role-adaptive metric、option-infer with support bug fixes、BayesMode with random/anonymous modes。否则“四粒度命题”只能作为诊断历史，不能作为因果实验。

[12][重大] 支持图承重证据来自历史分离实验，但当前基底、U2 filter、奖励与去 oracle 后的机制都已变化，旧结论不可直接承重。

被攻击的具体主张：总方案称“每个设计选择都有自己的实测正证与反证”，其中支持图正证是 A1/A2/A7 vs G2，5/5 成功、删上菜因子退化。 主张映射中 E5 图鲁棒性仍排在后续。

隐含假设是：旧基底、旧图、旧 reward/inference 条件下的图消融，能支撑当前论文里的图机制主张。由于你们已经修过 oracle 泄漏、支撑集冻结、奖励脚手架、推断路径，这个假设需要重证。

具体失败情景或审稿人指控：“你们把历史 debugging experiment 当作 current evidence。关键图消融不是在最终系统上跑的，不能证明最终 ARIS-Bellman 的图是承重件。”

最小实验或修改：在 U2 实现后、无 oracle、当前奖励、当前 evidence policy、当前 graph gate 下重跑：full graph、top-K CE graph、no-serve-factor graph、random matched graph、dense all-factor graph。用相同 seeds 和 role-adaptive 指标读出。重跑前，论文中不要写“图不是装饰，是因果承重件”；只能写“历史诊断提示图可能承重”。

[13][重大] 被动支持盲区存在不等于干预式发现对下游承重，U1 的必要性还没有被证明。

被攻击的具体主张：总方案从“互斥型协调点联合支持度恰好为零”直接得出“交互因子的发现需要干预”。 U1 规格也把被动 CE 结构性失明作为 C1 动机。

隐含假设是：盲区 cell 正是下游策略需要的承重因子。盲区可能是真实存在但任务无关，或者可由覆盖约束/公共状态/单边因子间接解决。

具体失败情景或审稿人指控：“你们证明了 passive estimator 在某些 mutually exclusive cells 上 positivity 失败，但没有证明这些 cells 的 CE 对 policy learning 或 evaluation performance 有贡献。Measurement blind spot ≠ decision-relevant blind spot。”

最小实验或修改：做三段式实验：第一，U1 top-up 后确实改变哪些 selected factors；第二，控制 max_factors 与 coverage，比较 passive graph、interventional graph、passive+manual-terminal graph；第三，对新增 interventional factors 做 leave-one-factor-out 和 belief-swap。只有“图差异 + 下游行为差异 + 新因子消融导致退化”同时成立，才能说干预必要。

[14][重大] U1 的 interventional CE 是新 estimand；FSM top-up 可能发现的是 FSM 分布下的因子，不是 ego 学到策略会用的因子。

被攻击的具体主张：U1 明确说 CE_int 与 CE_passive 不混合，因为干预行改变采样分布；top-up ego 策略是 FSM + do(ω when valid)。 采集侧还写 FSM 用来把状态推进到终端阶段。

隐含假设是：FSM top-up 下估计出的交互因子，适合用于训练最终 learned ego 的 Q 路由。这个假设未被保证；estimand shift 可能使图对训练策略不匹配。

具体失败情景或审稿人指控：“你们用一个 hand-coded FSM 生成 intervention support，然后把这些 off-policy interactions 写进 graph。模型表现提升可能来自把 FSM 的任务结构注入了 graph，而不是从交互数据中发现因子。”

最小实验或修改：同一 required factor 用多种 top-up policy 估计：uniform+target、FSM+target、learned weak policy+target、pair-level intervention runner。报告 CE ranking stability 和下游 graph stability。若只有 FSM top-up 图有效，论文必须承认这是 FSM-guided graph construction，而非一般干预式发现。

[15][重大] “整局持续角色信念”与 λ=0.02 遗忘、窗口基 replay 截断之间存在语义张力。

被攻击的具体主张：核心论证依赖“整局内持续更新”的角色判断和开局不确定状态训练/部署共享。 但规格默认 λ=0.02，并承认 P4 截断 replay 不是从 episode 起始全量重算。

隐含假设是：λ 混合和窗口截断不会破坏“整局持久”的机制语义。λ=0.02 的每步遗忘半衰期约 34 步；在 200 步量级的局内任务中，这不是轻微数值项，而是强先验。

具体失败情景或审稿人指控：“你们卖点是 episode-persistent belief，但实现每步向 uniform 混合，而且训练 replay 从 stale window base 近似重放。所谓整局后验在训练和部署中并不一致。”

最小实验或修改：执行前固定 λ sweep：0、0.002、0.02、0.1，并报告 posterior half-life、entropy、role switch latency、return。再比较训练时 window replay posterior 与 evaluation full online posterior 的 KL divergence。若 λ 敏感，不能把默认 0.02 当工程常数；必须作为关键机制变量进入预注册。

[16][重大] 因子局部独立信念与加性 Q 修正可能表达不了跨因子一致的伙伴角色。

被攻击的具体主张：设计承诺二是“对每个因子维护”角色判断；承诺三是角色判断只通过“因子局部的价值修正”进入 Q，Q 是基础价值加因子修正之和。

隐含假设是：伙伴在各个协调因子上的行为模式可以独立建模，并且对动作价值的影响可加性组合。现实中，一个 partner 的终端 claim、通道抢行、盘子偏好、等待策略可能高度相关，也可能在局内随状态耦合变化。

具体失败情景或审稿人指控：“你们的 factor-local posterior 可能在 serving factor 上认为 partner yield，在 passage factor 上认为 partner claim，导致 Q 修正互相抵消或产生不一致行为。真实协调需要 joint latent role 或 hierarchical latent。”

最小实验或修改：记录每局各因子 posterior 的相关矩阵和互信息；构造 partner 在不同因子上组合角色的 held-out set，例如 claim-serving/yield-passage、yield-serving/claim-passage。加入 global episode role + factor residual 的对照。如果 global+local 明显优于 pure factor-local，当前“因子局部角色”主张需收窄。

[17][重大] 评价只检查“接手/不全包”，可能漏掉角色震荡、延迟接管、低效让行和互相试探。

被攻击的具体主张：成功标准是 yield 伙伴 egoCCR>0，以及 claim 伙伴 partner throughput 不低于当前 ARIS。

隐含假设是：只要最终有 ego 上菜，或 partner 吞吐不降，就说明实现了局内角色适应。这个判据太粗，尤其在 base completion 已经饱和的环境中。

具体失败情景或审稿人指控：“模型可能 180 步都在互等，最后 timeout 前 ego 上一次菜，egoCCR>0；也可能对 claim 伙伴前期抢活、后期让位，partner 吞吐未降但 idle/collision 大量上升。你们把结果变量当机制变量。”

最小实验或修改：加入 trajectory-level metrics：首次有效接管时间、有效 serve opportunity 到动作的 latency、双人 idle time、角色切换次数、碰撞/堵路、partner waiting induced by ego、每阶段贡献分解。成功标准应至少要求接管延迟与 idle time 相对当前 ARIS/base 降低，而不只是 egoCCR>0。

[18][重大] “population 系正交”不能替代正面对比；ICLR 审稿人会要求你们战胜或叠加到强 population baseline。

被攻击的具体主张：proposal 把 FCP/MEP/TrajeDi 归为“改数据分布，我们改表征，正交且可叠加”，并把加训练伙伴路线降级为后备方案。

隐含假设是：方法学正交性足以避免与 population 方法正面对比。顶会审稿不会接受这个理由；ZSC 论文首先会被问“相同预算下，比 FCP/MEP/TrajeDi 或其简化版强在哪里”。

具体失败情景或审稿人指控：“你们的方法只在一个手工伙伴库上击败弱 baselines；没有与 standard ZSC population training 比较。所谓正交没有实验证明，可能只是被更广训练分布支配。”

最小实验或修改：至少跑一个强 population baseline：FCP-style diverse partner population 或 MEP-style partner ensemble，使用相同训练预算、相同 reward、相同评估伙伴、相同 role-adaptive metrics。再跑 U2 + population，验证是否叠加。若 population 单独满足两条成功标准，U2 的主张必须转为“样本/结构效率”或“解释性”，不能是 headline 性能。

[19][重大] 5 种子 × 25 局 × “≥3/5 egoCCR>0”的二值门对主张太脆，真实后果不只是功效低，而是结论会被阈值工程支配。

被攻击的具体主张：E1-rev3 使用 5 个种子、25 局 × 2 held-out 伙伴，并以 ≥3/5 种子 egoCCR>0 作为 yield 成功标准。

隐含假设是：这个二值标准足以区分机制成功/失败。实际上，egoCCR>0 是极低门槛，一个偶发上菜即可让种子算成功；而 3/5 与 2/5 的差异无法支撑论文级主张。

具体失败情景或审稿人指控：“结果可能是 seed 1,2,3 各偶然上一次菜，seed 4,5 完全死锁。你们把 sparse binary event 转成 mechanism success，没有置信区间、没有 effect size、没有 sequential correction。”

最小实验或修改：E1 可以作为 smoke test，但正式读出必须用 50–100 局全种子、seed-level continuous metrics、binomial confidence interval、hierarchical model。主指标应是每局接管率、接管 latency、team score、idle time，而非“任意 egoCCR>0”。预注册“≥3/5”只能用于是否继续执行，不应作为论文结论门。

[20][中等] hard gate 与“guard 不过不进性能均值”的处理可能让工程不稳定被低估。

被攻击的具体主张：总方案写 guard 不过的种子计入“能力率”分母但不进性能均值。

隐含假设是：能力率与性能均值分开报告不会误导。若 U2 增加了递归、checkpoint、mask、replay 复杂度，硬门失败本身就是方法可靠性的一部分。

具体失败情景或审稿人指控：“U2 的坏种子被列为 guard failure，从 mean return 中剔除；论文主表看起来更好，但部署者关心的是 training run 成功概率乘以成功时性能。”

最小实验或修改：primary metric 用 intention-to-treat：所有启动训练的 seeds 都进入主表，guard fail 记为 0 或单独 failure outcome；secondary table 才报告 conditional-on-pass performance。否则复杂方法对简单 baseline 有隐藏劣势。

[21][中等] 工程测试覆盖了 mode_mask、checkpoint、log-domain，但这些测试不能支持“角色后验可解释”主张。

被攻击的具体主张：规格列出 I19-mode gate、归一性、masked-mode leakage、梯度流、checkpoint provenance 等测试。

隐含假设是：这些工程 fidelity tests 通过后，BayesModeFilter 的“belief”解释就稳固。实际上这些测试只证明张量、掩码、梯度和保存加载没错，不证明 posterior 是 calibrated belief，更不证明 mode 有角色语义。

具体失败情景或审稿人指控：“你的 unit tests 证明了 softmax 行和 checkpoint 一致，但没有任何 test 证明 log-likelihood 是 likelihood、posterior 是 posterior、mode 是 role。”

最小实验或修改：把 semantic fidelity gate 加进测试计划：posterior calibration by role opportunity、counterfactual posterior response、cross-seed mode alignment、posterior entropy under ambiguous partners、OOD evidence detection。工程 gate 与机制 gate 分开命名，避免把 implementation correctness 当科学证据。

[22][中等] “诚实链/预注册”在 ML 评审中不会自动加分，甚至可能暴露这是连续修补后的窄结论。

被攻击的具体主张：总方案把“可复核的诚实链”列为方法论创新，称预注册、硬门、负结果留档可沿台账追溯。 结论还强调“任何好看但没预注册的数字不进论文”。

隐含假设是：审稿人会因过程透明而提高对主张的评价。多数 ICLR/NeurIPS 审稿会把它当 reproducibility hygiene，而非科学贡献；若主实验窄，诚实链反而让人看到指标是如何在失败后迁移的。

具体失败情景或审稿人指控：“The paper is unusually transparent, but transparency does not compensate for circular benchmarks, weak baselines, and limited scope.”

最小实验或修改：把诚实链放入 reproducibility / artifact appendix；正文只用它防御“post-hoc metric”问题。不要把它列为核心创新，除非开源完整 ledger、raw artifacts、preregistered scripts，并让第三方能一键复核主表。

## 2. 最强拒稿意见书

本文声称“少数离散角色”是 ZSC 信念的正确粒度，但测试台、伙伴生成器、潜变量词表和成功标准都围绕同一 claim/yield 上菜轴构造；成功只说明模型在角色脚本世界中恢复了脚本标签。Bayes 后验由 TD 端到端训练，无标签、无似然校准、无可识别性证据。评价在 base 完成率失效后改为角色自适应，且未战胜显然的手写观察-接管基线。证据不足以支撑 ICLR 级方法主张。

## 3. 十个我们必须能当场回答的问题

1. 如果把所有伙伴名称、claim/yield/flexible 词表和 generator metadata 从实验者与代码中隐藏，U2 还能定义同一个 latent space 吗？
2. U2 成功后，如何排除它学到的是“三维持久记忆瓶颈”而不是角色后验？
3. 在同一角色但不同路径风格、不同延迟、不同噪声的伙伴上，x_t^f 的 OOD 距离是多少，posterior 是否校准？
4. 训练数据中有多少状态满足“posterior 高 yield、ego 有机会 serve、ego 实际 serve 并获得正回报”？
5. 一个 hand-coded observe-then-switch baseline 在同一两条成功标准上表现如何？
6. 如果 K=2、K=5、K=8 或匿名随机 modes 与 K=3 claim/yield/flexible 表现相同，你们还会说“角色粒度正确”吗？
7. 为什么“两条都达标”能推出核心命题成立，而不是只推出 BayesModeFilter 在两个脚本伙伴上有效？
8. global_gru、partner_id_q、option-infer 是否在同等调参预算、同等 role-adaptive metric、同等容量与同等修复后重新比较过？
9. U1 干预图相对 passive 图新增的具体因子，删掉后是否导致 U2 下游行为退化？
10. 如果 FCP/MEP/TrajeDi 风格 population baseline 单独满足角色自适应标准，U2 的主张将如何改写？

## 4. 未言明的承重假设清单

1. 角色脚本生成的伙伴库足以检验“角色是正确粒度”这个非脚本世界命题。
2. claim/yield/flexible 三值不是从生成器泄漏来的 ontology，而是合理的任务先验。
3. 陌生伙伴的行为证据特征 x_t^f 与训练伙伴同分布或至少在 g_θ 可外推范围内。
4. TD-only reward 会把 latent modes 对齐到人类可解释角色，而不是对齐到任意 reward-predictive nuisance。
5. 负证据“没上菜”只在 partner 有机会但选择不做时累计，而不会混入不可达、未准备好、被挡路等公共状态原因。
6. 训练中存在足够的接管探索，使 Q 能学习“对方让位时我该接手”。
7. 因子局部 posterior 的加性 Q 修正足以表达跨因子一致的伙伴策略。
8. λ=0.02 的遗忘不破坏“整局持续角色信念”。
9. P4 window-base replay 的近似不会造成训练 posterior 与部署 posterior 语义不一致。
10. 当前 ARIS 的 partner throughput 是合理的“配合”基线，而不是过低门槛。
11. egoCCR>0 是有意义的接管成功，而不是偶发动作。
12. 25 局评估足以估计 role-adaptive behavior，而不是只做 smoke test。
13. 旧基底上的支持图消融仍适用于去 oracle、U2、当前 reward 和当前 graph gate 后的系统。
14. FSM top-up 估计出的 CE_int 对 learned ego 的训练分布有效。
15. 没有简单规则 baseline 可以利用单一上菜轴达成同样行为。
16. population 方法与表征方法“正交”可以作为暂不正面对比的理由。
17. hard gate failure 与方法性能可分离，而不是方法稳定性的组成部分。
18. 预注册叙事能弥补指标迁移和实验窄域，而不是只作为可复核性附录。

## 5. 修复优先级建议

第一，执行前必须加入非角色生成或连续角色谱的伙伴测试，并做盲化。理由：这是唯一能直接拆掉循环论证的修复。最低成本不是人类数据，而是生成 8–12 个没有 claim/yield 标签的 policy partners：RL/self-play seeds、probabilistic terminal preference interpolation、noisy route variants、mixed-role partners。E1-rev3 可以保留为工程 smoke test，但不能作为核心命题读出。

第二，必须实现 OASIS-N 这种廉价 observe-then-switch baseline，并把 role-adaptive 指标改成方法无关的行为指标组。理由：当前最大审稿风险不是 U2 不提升，而是一个手写规则也能通过两条成功标准。若 U2 不能稳定战胜该 baseline，论文主张应改为“可学习结构化信念接近强规则 baseline，并具备可扩展性”，而不是“正确信念粒度”。

第三，必须在训练正式跑完前加 U2 机制诊断门：posterior 语义、证据 OOD、action support、λ/截断敏感性。理由：这四项能区分“inference 修好了但 control 没学会”“posterior 是无语义瓶颈”“OOD 特征导致后验垃圾”“整局信念被遗忘/截断破坏”。没有这些诊断，即使 E1 成功，机制论文也会被认为只是在单一布局上调出一个有效模块。