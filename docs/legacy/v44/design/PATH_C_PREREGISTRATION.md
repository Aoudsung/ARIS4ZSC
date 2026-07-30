# Path C 预注册清单 — 公开基准对齐版

**日期：** 2026-07-24
**机器可读模板：** `experiments/overcooked_v2/configs/path_c_preregistration.yaml`
**当前状态：** 修订模板，不可直接用于运行。2026-07-23 的旧伙伴池 Test Time Simple
结果用于修改本设计，因此新版 Test Time Simple 不是原始预注册的确认性重复；它可以按同一
self-play/cross-play 口径与已发表表格并列，但必须标注为结果后设计修订。新版方法在 Simple
启动前冻结，Test Time Wide 使用完全相同的代码和规则作布局复现。所有正式配置与产物继续
记录 `scientific_readout_allowed: false`，等待最终人工裁决。

本文件中 self-play（SP）指同一次独立训练内的策略彼此协作；cross-play（XP）指不同独立
训练运行得到的策略在测试时配对。

本文件解释机器可读模板中需要修订的正式协议。模板中的
`template_not_valid_for_runs` 仍表示不可用于正式运行。原“从五个自建系统中选择最强基线”
方案已撤销；机器可读模板尚未完成对应代码迁移。

## 1. 语义绑定

标准 XP 主结果只需要固定实际影响可比性的对象：

- code commit；
- resolved runtime config；
- R015 专用主体可见历史合同及其 `ego_evidence_contract` 制品；
- 官方布局与环境参数；
- 训练 seed、训练步数和评估日程；
- 随机数键调度版本。

partner registry、伙伴 option policy、`ResponseSummarySpecV1`、audit battery 与 four-role split
manifest 只在机制 instrument 或合成分析中使用。它们的缺失不得阻止标准 XP 性能评估，
但会阻止对应机制主张。

## 2. 共享 evidence contract

冻结 R015 专用主体可见历史合同的字段尺寸、编码顺序和最大窗口，并用
`ego_evidence_contract` 制品绑定。字段集合只采用
[R015 规范 §1.1](PATH_C_OPPORTUNITY_AUDIT_SPEC.md) 的权威合同，主方法与所有 full-history 或
belief-based 消融不得另增字段。可唯一恢复的伙伴动作必须绑定恢复函数与语义哈希；直接读取
评估器伙伴动作必须单列为 action-augmented 信息增强诊断。身份、机制、风格、seed 和布局
风格标签禁止进入主证据路径。

旧代码类 `EgoEvidenceSpecV1` 包含评估器记录的伙伴原始动作，只属于历史机制仪器，不能作为
R015 或官方观测主方法的输入合同。

理论中的 history 和表示恢复主张只相对于该冻结信息集成立。审计 history 若更丰富，必须降级为 instrument-only 结论，不得据此声称 learned representation 能恢复相同对象。

## 3. 主回应摘要

冻结 `ResponseSummarySpecV1` 的：

- 固定 response classes；
- latency bin upper bounds；
- terminal、censored、invalid-script 与 support-violation special tokens；
- 唯一 token 编码规则和 vocabulary hash；
- `q` 等于冻结词表长度；
- design split 上的 summary family 选择结果。

主 theorem kernel 只使用一个 finite categorical token。structured multi-label readout
只作次要结果，不得与主 total-variation distance 混用。

## 4. 模型与探针

冻结循环网络结构、输出头数量、回合级 bootstrap 概率、独立固定先验、候选动作支持、
选择概率记录和每次探查成本。旧的归一化动作相对价值分歧只保留为历史消融；通用回应信息
控制器只作基线。正式 `decision_focused` 条件必须运行同一任务转移和随机流上的
`B_use`、`B_mask` 与共享延续控制器成对分支，并按 `J_use-V_mask` 选择候选。当前有限伙伴
原型两步代理量没有这些分支，只能登记为非科学诊断，不能进入正式读数。锁定审计使用独立
冻结的开放循环候选注册表，不使用训练中的自适应选择规则。

决策导向控制器的阈值生成规则固定为校准空分布第 95 百分位与零的较大者，并使用严格大于
比较；通用回应信息控制器继续使用自身校准空分布第 80 百分位。随机安全探查的概率等于校准
决策分数严格超过决策导向最终阈值的实际比例。适应前的训练校准与适应后的部署校准各运行
500 个禁止实际探查的回合；前者只供适应训练，后者按条件独立生成且是正式评估唯一可读取的
阈值来源。不能为获得固定探查比例而降低阈值。

## 5. 合成工具与公开基准数据

伙伴对象分成三种互不混淆的角色：

- **训练原型支持：** 可以用于回应模型、价值模型和在线过滤器；
- **开发伙伴族：** 任何成员曾用于校准、阈值选择、R015 支持构造或方法选择的伙伴族；
- **held-out 伙伴族：** 按训练算法、训练目标和约定生成机制整体与前两者不相交，只在方法和阈值冻结后评估。

正式 OvercookedV2 模型的信念坐标是四个伙伴家族原型，不是具体 checkpoint 身份。四个坐标
固定为：轨迹起点冻结的当前适应策略、本单元骨干历史、两个外来自博弈运行的历史、两个
Other-Play 运行的历史。每个官方运行按训练进度机械保存三个 checkpoint；家族内成员只用于
抽样、损失路由和审计，不进入模型输入或信念。相同训练算法、目标和约定生成机制产生的不同
seed 仍不构成 held-out 伙伴族；进入任何训练、校准或方法选择的伙伴家族不得计入 held-out
评估。

- 至少三种 value mechanism；每种至少四个、优选五个独立 identity/style realization。
- fingerprint 与 mechanism 近完全交叉；包含 posterior support 重叠的机制对。
- synthetic 使用 registry 的 true value class；ecological 主分析使用 frozen cross-fitted value bins。
- train、design、calibration、locked_audit 四角色的 identity、style 与 seed group deterministic 且互不重叠；该四路切分只服务合成工具、机制分析与内部消融。
- 论文主性能不再使用自建 identity split 选择 baseline，而采用 ICLR 2025 OvercookedV2 的独立训练 cross-play（不同训练运行在测试时配对，以下简称 XP）协议。
- 主布局是 `test_time_simple`，预先指定的第二布局是 `test_time_wide`。`asymm_advantages` 只作内部消融。
- 任一 mechanism 的独立 group 数不足四个时，只影响对应机制分析，不阻止标准 XP 性能评估。

## 6. 已发表对照与内部消融

主表的对照来自同协议已发表研究，而不是本项目自建网络。主公开参照固定为 ICLR 2025
OvercookedV2 Table 2 中的 Self-Play、State-Augmented、Other-Play 与 Fictitious Co-Play。
Test Time Simple 当前找到的同协议最佳已发表 XP 为 `6±29`，Test Time Wide 为 `23±40`，
两者均来自 Fictitious Co-Play。这里的“最佳”只适用于完全相同的布局和评估协议，不表示
脱离协议的 OvercookedV2 全局最佳结果。

full-history recurrent、residualized recurrent、exact belief filter、learned hidden Markov model
filter、particle belief filter、random probe、no probe 与 direct-information 均降为内部消融或
机制诊断。它们不再参加“最强 baseline”选择，也不再是主性能评估的前置制品。

若后续得到原作者 checkpoint 或逐 seed 数据，优先在同一 evaluator 中重算。没有公开
checkpoint 时，直接并列论文数字并明确标为外部参照；不为填满主表重新训练已发表方法。

## 7. 主 endpoint

主 endpoint 是：官方 Test Time 协议下的平均 XP episode return。

必须冻结：

- `test_time_simple` 与 `test_time_wide` 的官方环境参数；
- 每个 episode 固定 400 个环境步；
- 10 个完全独立外层训练单元；每个单元独立拥有一个自博弈骨干、两个外来自博弈运行、
  两个 Other-Play 运行、五个运行各自的三个固定进度 checkpoint、预拟合、训练校准和三路
  随机数；四条件只允许在单元内部共享这些对象；
- 四个伙伴家族各占适应回合的 25%，先平衡家族再平衡家族内成员；四条件共享条件名称无关的
  家族、成员、席位和环境随机数日程；
- 正式种子根为 `2026072302`，按布局、训练单元、角色、成员、checkpoint、阶段和回合经
  SHA-256 分域派生；禁止跨布局、跨训练单元及与历史开发 seed 碰撞；
- 每单元预拟合 1,000,000 环境步和训练校准 500 回合；每条件适应 10,000,000 环境步并在
  最终 checkpoint 上独立部署校准 500 回合；
- 90 个有向跨 seed XP 配对，覆盖双方玩家位置；
- 每个配对 500 个 episode；
- 原始 episode return 的聚合公式；
- 均值、标准差和 95% 置信区间的报告方式。

直接对齐论文的均值和标准差按官方配对单元重算。95% 置信区间使用训练单元节点重采样：每次
对 10 个训练单元索引有放回抽样一次，所得计数同时作为该单元在主体和伙伴位置的权重，只用
`i!=j` 的原有有向配对重算均值；少于两个不同训练单元时按下一预生成随机键重抽。固定 9,999
次有效重采样和运行前登记的随机种子；不得把行、列
seed 分开抽样，也不得把共享同一训练策略的 45,000 个 XP episode 当作独立重复。

SP episode return 与 SP−XP gap 必须同时报告，但不能替代 XP。只看 gap 会奖励 SP 与 XP
同时很低的退化方法。

适应前骨干和四个适应条件分别形成十策略群体。每个群体只包含十个最终策略，禁止混入训练
伙伴或 checkpoint 历史成员。跨策略配对两侧分别加载自己的参数、循环状态、四家族信念和
部署校准阈值。

决策导向群体另执行回应屏蔽部署对照：90 个有向跨策略配对各使用 500 个匹配回合块，每块
只干预主体一侧并产生 `A1`、`A2-mask`、`A2-use` 三个分支。汇总单独报告
`Delta_response=A2-use-A2-mask`、`Delta_cost=A1-A2-mask` 和
`Delta_net=A2-use-A1`，并机械验证 `Delta_net=Delta_response-Delta_cost`。该对照不进入
官方 self-play/cross-play 摘要。

扣除探针成本后的 return-versus-budget curve area under the curve（回报—预算曲线面积）
降为支持结果。value-directed、random probe 与 no probe 的差异属于消融实验。

若只有已发表均值和标准差，论文只能写“高于已发表参照值”；只有取得原始逐 seed 数据
或公开 checkpoint 并在同一 evaluator 中重算后，才能声称统计显著优于现有最好结果。

### 7.1 外部有效性假设的三个固定端点

外部有效性假设 H3 使用三个彼此不能替代的端点，均为“决策导向探查减匹配无探查”的原始
回合回报配对差：

- `Delta_Wide`：Test Time Wide 的平均 XP 配对差，沿用 10 个独立训练单元、90 个有向配对
  和训练单元节点重采样；
- `Delta_heldout`：先对每个冻结伙伴求配对差，再在伙伴族内等权、最后在预注册 held-out
  伙伴族之间等权；主体训练单元和伙伴策略是重采样单位，伙伴族保持固定分层；
- `Delta_Hanabi`：Hanabi 未见伙伴集合上的平均配对差；至少包含 10 个独立主体训练单元、
  两个未进入任何训练或开发环节的伙伴生成机制，且每个机制至少四个冻结策略；伙伴生成机制
  保持固定分层，主体训练单元和伙伴策略是重采样单位。

H3 的名义错误率预算为 0.05，其中 `gamma_cal=0.005` 分给运行前校准，剩余 0.045 在三个
科学端点之间等分。每个端点使用名义单侧错误率 `alpha_H3=0.015`，报告 98.5% 单侧下界。
在生成任何 H3 确认性数据前，冻结零效应生成器
套件；它至少覆盖方差相同的近似对称回报、偏斜且含大量零值的回报、训练单元或伙伴族之间
方差不同的回报，以及同一训练单元参与多个配对形成的强节点相关。每种生成器都保持正式协议
的回报界、训练单元数、伙伴分层、完整配对结构和零平均处理效应。

对每个生成器和每个端点独立生成 10,000 份零效应数据，并在每份数据上原样运行正式的
9,999 次重采样和下界判定。记预先登记的“生成器 × 端点”组合总数为 `M_cal`，单个组合的
误报次数为 `F`。对每个组合计算置信度 `1−gamma_cal/M_cal` 的单侧 Clopper–Pearson 精确上界；
该上界是根据有限次二项误报计数得到的保守概率上限。由并集上界，全部组合的误报率上界
同时有效的置信度至少为 `1−gamma_cal=99.5%`。只有所有组合的上界均不超过
`alpha_H3=0.015`，百分位区间才可用于预注册确认性判定。对登记的零效应套件，错误授予
这项资格的概率至多为 0.005；只有在真实端点另外满足各注册区间误报率不超过 0.015 的覆盖
假设时，三个科学端点的错误率之和才至多为 0.045。有限模拟不能证明真实 Wide、held-out
或 Hanabi 数据律满足该假设，因此不得把两部分相加写成无条件的真实数据总体错误率保证。
任一组合未通过时，H3 记为不可裁决；不得查看 H3 结果后调生成器、改置信水平或用同一批
数据更换区间方法。这项有限套件校准只检验登记的依赖结构，不是对所有回报分布的有限样本
覆盖定理；公开结果必须把三个下界标为名义区间。H3 只在校准通过且三个下界全部高于零时成立；不得跨端点平均或用一项收益
抵消另一项失败。held-out 端点还要求每个预注册伙伴族
的点估计均为正并报告族别区间。Hanabi 每个有向配对的回合数必须在看不到结果时由回报界或
精度分析冻结，不得按观察到的效应追加。

Wide 沿用上述 9,999 次节点重采样。held-out 和 Hanabi 各使用 9,999 次分层重采样：每次有
放回抽取主体训练单元，并在每个固定伙伴族或生成机制内有放回抽取伙伴策略，再按端点的等权
规则重算。三项使用运行前登记的独立随机种子和单侧百分位下界。无效回合按预先生成的替补
随机键成块替换；缺少 checkpoint、整项策略配对或预注册伙伴族时不插补，对应端点记为不可裁决。

机器可读模板在加入上述三个端点、Hanabi 领域字段、伙伴族清单、重采样种子和区间生成规则
前仍不可用于 H3 运行。

H3 冻结时必须一次填完以下对象：

- Wide 的 9,999 次训练单元节点重采样种子；
- 零效应生成器的参数与代码哈希、四类依赖结构的固定清单、组合总数 `M_cal`、
  `gamma_cal=0.005`、每类 10,000
  份数据的随机种子、9,999 次重采样的按数据集索引派生规则、正式区间判定脚本哈希，以及
  每个组合的误报次数和置信度 `1−gamma_cal/M_cal` 的单侧 Clopper–Pearson 上界；
- held-out 伙伴族名称、生成配置或代码哈希、每个策略 checkpoint 哈希、与主体训练单元的完整
  配对表、无效块替补随机键和分层重采样种子；
- Hanabi 环境与协议版本、玩家数、官方可见历史字段、合法候选动作、注册回应摘要、领域安全
  事件与风险上限、原始回报界、10 个主体训练 seed、至少两个生成机制各四个伙伴 checkpoint、
  完整配对表、由看不到结果的精度分析得到的每配对回合数、无效块替补随机键和分层重采样种子。

任一字段为空时只允许继续静态设计，不允许读取 H3 科学结论。

## 8. belief-kernel instrument

冻结 `M`、同时比较的 cell 数、family-wise `confidence_delta`、`L_inner`、`T_probe`、audit information states、partner prior、完整 probe battery、sampling table 与成本单位。

Partner prior 使用版本化的 `path_c_theta_prior_v1` mapping。它绑定 partner registry 的
SHA-256 内容哈希，列出非空且不重复的 `theta_id` 支持；每个概率严格为正，全部概率之和
必须为一。

每个 outer replicate 必须独立抽取完整隐藏状态
`U=(theta, execution_state) ~ P(U|history)`。同一 outer state 上可对所有 frozen probes
配对，并可用 `L_inner` 个未来随机性重复；统计 cluster 仍是 outer replicate。解析权重的
Rao–Blackwell 混合只作次要估计，不得借用为独立 categorical draws 推导的 sampling bound。

精确模式要求：

- positive-mass pruning 为零；
- sparse state merging；
- generation 与 inference 共用 `option_distribution`；
- tiny-horizon brute-force differential fixture；
- posterior 和 reset bias bounds 均为零。

近似模式必须报告 `rho_prune`、posterior bias 与 reset bias。Tier 2 hash-and-match 只用于注册的高复现、满足 positivity 的 audit units，hash 后还要逐字节复核；每 episode/key 最多一个 outer draw。

## 9. 版本化统计字段与决策

仪器测量必须包含：

- `alpha_upper`：所有同一 value class pair 的 cell-specific upper confidence bound 最大值；
- `beta_lower`：所有不同 value class pair 的 cell-specific lower confidence bound 最小值；
- `sampling_radius`；
- `posterior_bias_bound`；
- `reset_bias_bound`；
- exact 或 approximate 状态；
- support 与 outer effective sample size。

`instrument_valid` 的必要数值条件是 `beta_lower > alpha_upper`，并同时要求所有软件、
重构、支持度和 cluster inference 检查通过。旧式 epsilon margin 字段没有第三版含义，必须显式迁移。

最终 `instrument_measurement` 必须由 `path_c_instrument_evidence_v2` 重算；每个 cell
内嵌完整、严格反序列化的 outer-replica kernel records，并绑定 posterior support、battery、snapshot、audit unit、probe、sampler seed 与 fresh-fork random-key coordinate。canonical response token 只能从选定 probe 的首个 inner response 提取；旧版手填 token 数组、alpha/beta 或 validity booleans 均被拒绝。
软件一致性必须另行绑定 module-registry test id、source commit、逐项 outcome 与归档 report
SHA-256；单独一组布尔字段不足以通过软件门。

标准 XP 性能与机制 instrument 分开。软件能运行官方协议后即可进行标准性能训练与评估；
instrument validity 只约束“恢复了价值相关伙伴表示”这一机制措辞，不阻止读取 XP 回报。
次要回应、表示、泄漏、power/null 或机制分析不能替代主 endpoint。

R015 是当前资源排序中的下一项测量，不是标准跨策略配对（cross-play，简称 XP）的通用启动
条件。用户已授权在读数前构造必要的静态代码，但正式训练和科学读数仍由 R015 决定是否继续。
R015 普通负结果只适用于其固定支持和注册控制器；没有经证明的完整信息上界时，不得扩大为
整个基底的否定。

held-out 伙伴族评估另行冻结伙伴族清单、生成代码或配置哈希、每族伙伴数、回合数和汇总规则。
它与官方 90 个跨 seed 配对分表报告，不能塞入同一独立样本计数。

## 10. 次要分析与功效

H2 把实际探查次数除以回合最大探查预算，得到 `[0,1]` 内的归一化探查率，并冻结
`epsilon_probe=0.05`。三个共同条件是：价值无关且回应可区分时，决策导向探查率的单侧置信
上界低于 `epsilon_probe`；同一条件下，通用回应信息方法减决策导向方法的探查率差的单侧置信
下界高于 `epsilon_probe`；价值相关且回应可区分时，决策导向方法减无探查的原始回报差的
单侧置信下界高于零。三项各使用错误率 `0.05/3`，全部成立才支持 H2。

独立生成的 identity group 是主要重采样单位，同组 episode 不作为独立重复。identity group 数、
每组 episode 数和重采样种子由看不到结果的联合功效模拟冻结，并报告各条件功效、联合功效和
Monte Carlo 标准误。Ecological value classes 只能来自
冻结的 cross-fitted value bins；mechanism labels 是 secondary proxy。Permutation 必须在预注册
strata 内按独立 identity group 进行。次要结果采用 hierarchical gatekeeping，不得替代 instrument 或主 endpoint。

## 11. 数据预算与产物

每个 run 必须从产物回读有效 episode 和 transition 数。claim readout 的当前基底下限为至少
2000 episodes 与 40000 transitions，并且训练曲线需满足 consolidation 检查。smoke run 只用于
检查接线，禁止进入表格、诊断或 claim。

机制分析的数据合同只接纳可实际解析的 content-addressed、append-only Parquet shards；每行的
split group、角色、机制、身份、风格、seed group 和布局必须与冻结 manifest 一致，episode 与
transition 总数由物理行重算。训练数据默认不保存完整 snapshot，audit harvest 单独运行。所有
机制关键 artifact 绑定第 1 节中与该分析实际有关的语义对象。标准 XP 主结果直接保存每个
训练 seed、配对、玩家位置与 episode 的原始回报，不要求先生成 instrument Parquet。

## 12. 运行前最小核对

标准 XP 训练与评估只需要下列项目：

- [ ] 实际代码提交和 resolved runtime config 已保存。
- [ ] 官方 Test Time 环境参数、10 个完整独立外层训练单元、每单元 5 个互不复用的官方
  上游策略、正式训练预算、90 个有向 XP 配对和每配对 500 个 episode 已绑定清单与哈希。
- [ ] 标准 SP/XP evaluator 能保存逐 episode 原始回报，并按 seed 配对和玩家位置重算结果。
- [ ] 旧 design baseline selection、instrument 和 locked ledger 不在标准 XP 依赖链上。
- [ ] 用户已明确授权相应远程执行。

机制 instrument 只有在实际运行该分析时，才另外固定 evidence schema、response vocabulary、
partner registry、option policy、split manifest、probe battery、抽样规模、随机数语义和对应内容
哈希。静态 cost estimator 也只检查 instrument rollout 的预算。未运行机制分析时，这些项目
不得阻止标准 XP 训练或评估。

新版正式清单使用 `path_c_outer_units_v2`，每个布局包含十个单元和五十个互不复用的官方
上游运行。原规则要求最终 checkpoint 通过冻结能力门槛；50 个 Simple 上游运行完成后，
项目于 2026-07-25 在看到 3 个固定运行低于门槛的事实后修订该规则。修订版只记录门槛结果，
不筛选、不替换 seed，也不重复训练直到通过；全部固定最终 checkpoint 及其早期和中期
checkpoint 均保留在原定历史群体。由于修订发生在上游回报可见之后，修订版 Simple 和 Wide
只能作为结果知情的探索性实验，不能按原始确认性预登记裁决。旧 `path_c_model_v1/v2`
配置、开发 seed 101/102/201/202 与旧正式产物只供审计，不能被新版恢复复用。

项目负责人已于 2026-07-25 明确授权上述只报告、不筛选的修订，并要求继续完成评估。

用户已在 2026-07-24 授权：六个目标测试和不读取性能的机械接线检查通过后，直接执行完整
Test Time Simple；完成预算、哈希和原始行重算后，不按性能修改代码或配置，继续执行完全
冻结的 Test Time Wide。只允许 GPU 4、6、7，每张卡同时最多一个训练作业，GPU 5 在任何
情况下都不得使用。
