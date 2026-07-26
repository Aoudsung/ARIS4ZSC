# Path C 第三版模块设计

self-play（SP）指同一次独立训练内的策略彼此协作；cross-play（XP）指不同独立训练运行
得到的策略在测试时配对。

**日期：** 2026-07-24
**状态：** 正式主链已修订为家族级伙伴原型与受虚构协作启发的 checkpoint 历史群体。
第三版配置、第二版外层训练单元清单、训练校准与部署校准分离、标准评估矩阵和回应屏蔽部署
对照保留为历史实现。V4.1 已完成 50 项远端软件测试和首开发单元复跑，登记为 `tested`；
回应屏蔽效应仍为 0，因此该状态不表示机制验收或正式冻结。旧伙伴池的 Test Time Simple
结果只保留为设计诊断，旧 Test Time Wide 作业已封存，不进入新版读数。
**规范来源：** `PATH_C_PROPOSAL.md`、`PATH_C_THEORY.md`、
`experiments/overcooked_v2/configs/path_c_preregistration.yaml` 与
`experiments/overcooked_v2/configs/module_registry.yaml`。

本文只说明第三版代码对象及依赖关系。执行权限仍由根目录
`OPERATING_CONSTRAINTS.md` 管理；本地未执行项目代码。远端测试由用户明确授权，且只形成软件验证记录，不形成科学读数。

## 1. 不可变的实现口径

1. 主方法的表征只接收时序差分动作价值损失的梯度。回应预测、信息增益、身份分类、重构和对比学习均不得反向传播到主表征。
2. 所有基于历史的可部署方法共享 R015 专用主体可见历史合同，并用 `ego_evidence_contract` 制品绑定；字段集合只采用 [R015 规范 §1.1](PATH_C_OPPORTUNITY_AUDIT_SPEC.md)，不在本文件另增字段。旧代码类 `EgoEvidenceSpecV1` 含评估器记录的伙伴原始动作，只属于历史机制仪器。可唯一恢复的伙伴动作必须绑定恢复函数与语义哈希；直接读取评估器伙伴动作属于 action-augmented 信息增强诊断。
3. 正式主分数是相同任务转移和随机流上的注册回应使用分支价值减最佳回应屏蔽参照。有限伙伴
   原型两步代理量、合法动作上的归一化动作相对价值（advantage）分歧和公共状态基线残差都只
   作诊断或消融。
4. 主回应对象是 `ResponseSummarySpecV1` 的有限单一 token。时延边界和词表在 design split 上冻结；terminal、censored、invalid-script 与 support-violation 均有独立 token。结构化多标签回应只作次要读数。
5. 精确后验模式不得剪除任何正后验质量。近似模式必须记录丢弃后验质量 `rho_prune`，并把它计入后验偏差上界。
6. 生成与推断共用纯函数 `option_distribution(theta, runtime_state, public_state)`，避免两份伙伴动作概率实现发生漂移。
7. 快照使用不可变 `SnapshotV1`。轨迹重放只使用原始随机数键；审计分支使用从命名单元和索引导出的新随机数键，并分别命名 JAX、NumPy、Python 与 Torch 随机流。
8. 机制 instrument 的审计数据不能选择网络、回应摘要、探针电池、阈值或切分。标准 XP 主结果不再经过自建 baseline 选择。

## 2. 修订后的实验顺序

```text
公开协议静态对齐与标准 SP/XP 接线检查
  -> 五个官方上游运行的固定 checkpoint 历史与最终能力观察报告
  -> 四个家族级伙伴原型的预拟合、训练校准、适应和部署校准
  -> Test Time Simple 十个独立训练单元的完整标准矩阵与回应屏蔽对照
  -> 完全冻结代码下的 Test Time Wide 复现
  -> held-out 伙伴族和 Hanabi 的后续独立验证
  -> random/no-probe 等消融
  -> belief-kernel 与指纹机制分析
```

审计仪器有效性的版本化字段为 `alpha_upper`、`beta_lower`、
`sampling_radius`、`posterior_bias_bound` 与 `reset_bias_bound`。仪器有效当且仅当
`beta_lower > alpha_upper`，且预注册的软件、支持度和重构检查均成立。

主性能是 ICLR 2025 Test Time 协议下的平均 XP episode return，同时报告 SP 与 SP−XP gap。
扣除探针成本后的回报—预算曲线降为支持结果。审计仪器有效性只约束机制解释，不阻止
标准 XP 评估。当前代码中的 design baseline selection 与 locked AUC evaluator 因此不再是
权威主结果接口。

## 3. 数据流与依赖关系

```text
官方 Test Time 环境配置
  -> 共享 R015 专用主体可见历史合同
  -> 完整 episode 时序展开与 episode 级 bootstrap
  -> 计划中的顺序分支 J_use−V_mask 决策导向探查策略
  -> 十个完全独立训练 seed
  -> 90 个有向跨 seed 配对，覆盖双方玩家位置
  -> SP、XP、SP−XP gap 与不确定性

独立审计采集
  -> 不可变 SnapshotV1 与随机数键调度
  -> 每个 outer replica 独立抽取完整 U=(theta, execution_state)
  -> 同一 outer replica 上配对冻结探针，inner fork 只估计未来噪声
  -> ResponseSummarySpecV1 token
  -> 每个 outer replica 一行、该行内保留全部配对探针与 inner fork 的 kernel 表
  -> cell-specific 同时置信界
  -> alpha_upper、beta_lower 与仪器有效性
```

`M` 只表示独立的完整隐藏状态 outer draws；`L_inner` 只表示每个 outer draw
上的未来噪声重复；`T_probe` 表示探针脚本时长。不得用 `M * L_inner` 冒充独立的
belief-kernel 样本量。按解析权重计算的 Rao–Blackwell 混合估计只作次要结果，除非另有与其数据结构完全一致的集中界证明。

## 4. 模块接口

### A1 配置与语义绑定

- 严格拒绝未知字段和第三版已删除字段。
- 启用任一 Path C 功能时，读取冻结预注册并绑定 code commit、resolved config、模块注册表、伙伴注册表、伙伴 option policy、evidence spec、回应词表、审计电池、切分 manifest 与随机数键调度版本的哈希。
- 预注册模板中的数值空槽和 `template_not_valid_for_runs` 状态使其不能用于运行。

### A2 共享证据与 recurrent sequence 模型

- `ego_evidence_contract` 给所有官方观测历史方法绑定同一信息集；旧
  `EgoEvidenceSpecV1` 只供历史机制仪器使用。
- `forward_sequence(EvidenceBatch)` 返回 `Q[B,T,K,A]` 与表征 `Z[B,T,D]`。
- 每个 learned head 有独立 recurrent state；固定随机 prior 与 learned network 独立。
- dueling advantage 只在合法动作集合上居中，避免非法动作数量改变合法动作价值。
- 训练和数据制品记录合法动作上的 ensemble effective rank、head correlation、固定 prior 贡献与 return-floor 触发率；这些只用于诊断，不能改变主判定。

### A3 时序差分训练

- `SequenceTDBatch` 明确包含长度、动作、奖励、半马尔可夫折扣、完成标记和合法动作掩码。
- online 与 target 网络均对完整序列展开；target shift 与 episode 边界显式处理。
- bootstrap mask 为 `[B,K]`，在一个 episode 内固定；每个 head 单独归一化，并对零支持 head 失败关闭。
- `K=1`、prior 为零且 bootstrap 全开时，应与完整序列的单头 recurrent 基线轨迹一致。该性质已有测试定义，但尚未执行。

### A4 探针选择

- 旧的逐头归一化 advantage 分歧已经降为历史消融；广义 Jensen–Shannon 散度是通用回应
  信息基线。
- 当前新增的 `finite_prototype_two_action_surrogate` 使用伙伴原型、回应记号和下一原子动作
  的有限模型代理。它没有任务转移 `x`、完整隐藏执行状态或实际 `B_use/B_mask/C` 分支，且
  其回应使用代理值按数学构造不低于屏蔽代理值。因此它只能用于诊断，配置固定
  `scientific_readout_allowed=false`。
- 正式训练和评估把 `decision_focused` 保留给尚未实现的
  `registered_response_sequential_branch_v1`；任何试图用两步代理量进入正式读数的配置都会
  失败关闭。
- 代理量保留时，最佳屏蔽参照只从同一当前历史、同一折扣原始回报头和完整六动作候选集求
  最大值，不再混入使用塑形训练回报的共享 critic。
- 无探查正式参照和通用回应信息基线只有在实际触发候选时才改 actor 动作；被动伙伴价值预测
  不能替换其基础动作。正式校准从内容寻址的逐决策账本重算 500 局完整性、零探查、角色与
  伙伴平衡及冻结分位数，不能只相信摘要中的阈值。
- 每个机会记录候选动作、完整分数、propensity、预算和成本。
- random-probe、no-probe 与 direct-information 对照使用相同动作支持和成本定义。
- locked audit 只使用冻结的 open-loop probe battery，不使用训练时的自适应规则。

### A5 历史机制仪器的公共状态基线残差

- 旧机制仪器的保留决策码是合法动作上的归一化动作相对价值；它不是当前 proposal 的正式
  `J_use-V_mask` 控制器。
- 公共状态基线残差只用于次要可视化和增量诊断，不能替代主决策码，也不能反向决定训练时探针。

### A6 R015 配对审计合同

- `path_c_r015.py` 只负责加载冻结登记、核验两伙伴族支持报告、验证 `A1`、`A2-mask`、
  `A2-use` 的配对字段，并从已有完整回合原始回报重算效应区间和判定。
- 它要求当前探查回应只在 `A2-use` 进入伙伴信念接口，原始官方局部观测和后续被动回应对
  两个 A2 实验组保持一致；无探查时三个正式实验组必须绑定同一完整轨迹。
- 冻结轨迹验证器必须从主体和伙伴 checkpoint 逐步重算双方动作及循环状态，并核对 `A1`
  选择、两组探查前动作、注册脚本和回应使用/屏蔽后的延续动作。安全验证器必须从官方历史、
  实际选中脚本和每个支持原型 checkpoint 重放独立分支；数据集自报值不能替代重算。
- 它不是环境运行器，不生成官方观测过滤器、相容隐藏状态、候选分支或实验数据。模块状态
  `implemented` 只表示裁决接线存在。

### B1 合成伙伴与真值

- 至少三种 value mechanism，每种至少四个、优选五个独立 identity/style realization。
- fingerprint 与 mechanism 近完全交叉；另有 posterior support 重叠的机制对。
- synthetic 主分析使用 registry 提供的 true value class；mechanism label 不是生态数据中的 true value class。
- 生态数据使用冻结的 cross-fitted value bins，并报告分类不确定性；机制标签只作次要代理变量。

### B2 回应与存储

- `ResponseSummarySpecV1` 把窗口映射为冻结词表中的唯一 token id，并从词表长度读取 `q`。
- 原始事件和结构化多标签列可以保留，但不得替代 theorem-level categorical kernel。
- 训练日志与 audit snapshot 分离；当前主数据合同只接纳可实际解析和逐行核验的 content-addressed、append-only Parquet shard。Arrow 或 Zarr 只有在加入同等严格的内容解析器后才可登记。每行绑定 episode、decision、split group 和语义哈希。
- Parquet validator 逐行核对冻结的 probe cost；选中 probe 恰好计费一次，未选中时 realized probe cost 必须为零。

### B3 生态价值类

- Synthetic 使用 registry 的真值类别，但该标签只能用于合成诊断。
- Ecological 数据从不含训练期 terminal shaping、并明确扣除原子步成本与探针成本的 held-out episode return 生成角色内部、out-of-fold 的冻结 value bins；机制标签不能代替生态真值。
- 阈值、episode-to-fold 分配、类别分配、角色和 split manifest 哈希共同写入内容哈希制品。

当前实现按角色和冻结 fold 交叉拟合生态回报估计。每个目标 episode 的估计只使用其他 fold 的 donor episode；分类阈值也只由其他 fold 拟合。制品同时记录估计标准误、donor 数量、阈值距离、实际 held-out return 和 split manifest 哈希，加载时逐行核对回报漂移。B3 因此记为 `implemented`；这只表示静态实现存在，不表示估计质量已经由运行验证。

### B4 两伙伴族支持

- 第一族是循环 Independent Proximal Policy Optimization；第二族是循环集成 Double
  Q-learning 的 1000 万步纯 self-play。两族共同使用官方局部 5×5 观测和六原子动作，但训练
  算法、优化目标、奖励退火和动作规则分别登记。
- 每族固定两个独立 seed。准入器对 checkpoint、训练配置、环境配置、训练实现、训练清单和
  独立运行标识作对称核验，要求四个 checkpoint 内容哈希和四个运行标识均不同。
- “训练实现”不再只表示一个入口文件，而是明确列出的仓库源码依赖集合；训练器和准入器用
  同一套“相对路径加文件内容哈希”算法重算该集合的总哈希。模型配置、训练清单中的架构和
  checkpoint 元数据中的架构必须三者完全一致，加载后的模型还要再次报告同一架构。
- 支持报告同时绑定支持登记文件的内容哈希、准入器源码依赖集合、评估 seed，以及逐回合
  证据文件的路径、内容哈希和行数。静态验证器从全部 `4×4×100` 行重新计算候选准入和配对
  矩阵；报告中的汇总值不能自行充当证据。
- 行为分布的 Jensen–Shannon 散度只作描述，不设人为距离门槛。四个候选中任一未通过同一
  能力下限时，正式支持保持为空，不能用其余三个形成部分支持。

### C1 回应预测次要读数

- 回应预测使用冻结的单 token 词表、固定 readout family 和 episode-cluster 置信区间。
- 该读数只进入 secondary profile，不能替代标准 XP episode return。

当前严格生成器从内容寻址的来源测量重建 `secondary_profile`，逐项绑定估计量、角色、episode 聚类单位、估计值、置信区间、多重比较校正和来源制品哈希。评估器重新计算并核对内容哈希，同时固定 `decision_eligible=false`，所以该读数不能替代或否决主结果。C1 因此记为 `implemented`；尚无测试执行证据。

### C2 旧内部曲线统计

- 对每个预算点计算 normalized net return，其中探针成本在归一化前扣除。
- 在 design split 上按冻结规则选择曲线下面积最高的可部署基线，并写入内容哈希。
- design 与 locked audit 使用两个独立的原始 episode ledger；locked audit 只比较主方法与已冻结基线，并以 identity group 为 bootstrap cluster。
- locked ledger 覆盖该角色的全部冻结 group；主统计只从 `primary_layout_strata` 派生的 group view 取曲线，secondary layout group 保留在 ledger 中但不得混入主估计。
- 主要输出包含估计值、置信区间和 `primary_effective`，不压缩成单一布尔结论。

曲线、曲线下面积、design 选择和 locked 统计的重算器已经存在，但它们现在只服务内部消融与机制分析。该对象不能单独支撑公开 benchmark 主张，也不再要求先生成五个自建候选的训练制品。标准主统计需要另行增加：`test_time_simple`/`test_time_wide` 的 SP、XP、90 个有向跨 seed 配对、双方玩家位置、每配对 500 个 400 步 episode、均值、标准差和 95% 置信区间。

### C3 历史机制仪器的保留决策码与泄漏读数

- 该历史仪器的保留决策码是合法动作上的归一化动作相对价值；它只服务次要机制分析，不是
  当前主方法的探查分数。
- recurrent hidden state、公共状态残差和身份泄漏均为次要读数；泄漏分析只可条件化于独立获得的 value class 与公共上下文。

### C4 功效与次要分析

- Positive 与 null 场景均按 identity group 聚合 effect，并分别报告 effect、普通置信区间、两单侧等价检验区间、各自功效和联合功效。
- 两单侧等价检验在每侧使用 `alpha`，对应双侧 `1 - 2 * alpha` 区间；不得误用普通 `1 - alpha` 区间。
- 机制标签、structured multi-label response、layout shift 与 representation leakage 均明确标为 secondary，不能改变主 endpoint 状态。

### C5 内部 acting-policy 消融

所有可部署方法实现统一接口：

- `reset()`：开始新 episode，并重置其内部历史；
- `act(observation, probe_budget_state)`：在相同动作支持和预算下选择行动；
- `observe(transition)`：只接收统一 transition；
- `representation()`：暴露只读读数；
- `metrics()`：报告环境步数、探针成本、更新数、参数、训练计算、wall-clock 与推断延迟。

主方法、full-history recurrent network、同样 residualized recurrent network、exact belief filter、learned hidden Markov model filter、particle belief filter、random probe 与 no probe 若用于内部消融，仍使用相同环境交互、梯度更新预算和评估日程。Identity embedding 是 oracle diagnostic。这些对象不再作为主表的 SOTA baseline。

正式运行时必须显式冻结 `probe_action_ids`。该集合必须是完整动作支持的真子集，而且每个实际状态至少保留一个合法的非探针动作；否则环境在收费前失败关闭，不能把全部合法行动都解释成探针。

当前已定义九个 design 策略和八个 locked-audit 策略的 factory registry、统一 runner、独立原始 ledger 与 OCV2 environment adapter。这是旧计划的实现。公开主结果只需要 Path C 自身 10 个独立 seed 的训练制品；其他 recurrent 或 belief checkpoint 缺失只应跳过对应消融，不能阻止标准 XP 评估。代码尚未完成这项解耦，因此 C5 保持 `planned`。

### C6 两阶段决策对象

`PathCDecision` 当前仍按旧顺序报告软件一致性、仪器有效性、内部曲线主效果和次要测量。修订后应把标准 XP 性能与 instrument 分开：缺 instrument 只禁止机制 wording，不能把已经产生的 SP/XP 标成不可读。

标准 XP evaluator 只要求可运行的软件、官方环境配置、Path C checkpoint 和逐 episode 回报。instrument、design ledger、baseline selection 或 locked ledger 的缺失不得阻止它。旧 artifact manifest 仍把这些对象串成依赖链，需要在 R005 后修订。

### I1 快照与随机数语义

- `SnapshotV1` 保存环境、伙伴、ego hidden state、原始 replay keys 与语义版本。
- `replay_from_snapshot` 只能复用 original keys；`fork_from_snapshot` 使用由
  `(manifest_seed, unit_id, outer_id, probe_id, inner_id, stream_name)` 决定的新 keys。
- runner 使用 snapshot copy 和显式 keys，不共享可变 `set_state` 对象。
- 对相同 immutable snapshot 与 fork coordinate 连续调用必须返回相同输出，且 source snapshot 的内容哈希保持不变。

OCV2 bridge 已定义无 pickle codec、内容寻址 bundle 与 fresh-runtime hook。`OCV2Adapter` 现在提供分离的状态捕获、恢复和无副作用单步接口；原始观察、环境状态和随机数 key 均复制后保存。生产 rollout hook 从不可变快照恢复新 adapter，并注入新建的伙伴和 ego 控制器，不复用调用间状态。I1 因此记为 `implemented`；尚无执行测试或确定性重复运行证据。

### I2 完整隐藏状态 posterior sampler

- 每个 outer replicate 独立抽取完整 `U=(theta, execution_state)`。
- 精确 forward recursion 使用 sparse state merging，不用阈值剪枝控制状态数。
- 近似模式报告最终丢弃的 posterior mass，并把它加入 posterior bias bound。
- tiny-horizon brute-force enumerator 与 forward recursion 进行 differential test。

当前 bridge 的旧仪器路径除核对 `EgoEvidenceSpecV1` history 外，还要求每个多原子步窗口携带完整公共状态路径和评估器记录的伙伴原子动作。生产绑定器使用与生成阶段相同的 scripted 或 latent controller；控制器在每个 option 边界调用同一个 `option_distribution`，枚举与该动作记录相容的所有隐藏 option 选择。转移 hook 返回相容后继状态的条件分布，likelihood hook 返回归一化前的总证据概率，两者乘积恢复精确联合质量；状态以包含嵌入 OCV2 public-state pytree 的专用 codec 合并。由于这条路径使用主体不可直接获得的评估器字段，它只属于 action-augmented 机制仪器，不能支持官方观测主方法或 R015 的可实现性结论。I2 因此只记为“旧仪器静态实现存在”；尚无执行测试或规模可行性证据。

### I3 完整隐藏状态 outer sampling

- 每个 outer id 由 `(base_seed, audit_unit_id, outer_id)` 独立派生随机数种子，分块调用与整段调用得到同一序列。
- 主估计每个 outer replica 抽取完整 `U=(theta, execution_state)`；单一 realized checkpoint fork 和 Rao–Blackwell 混合只能作次要诊断。
- kernel table 每个 outer replica 只保留一行，并在该行内保存全部配对探针与 inner fork。

### I4 精确历史匹配

- Tier 2 只用于预注册的高复现 audit units，先比较 SHA-256，再逐字节比较完整可观察历史。
- 同一 episode 与 history key 最多进入一次；抽样使用无放回确定性排列，并强制 positivity 与足够独立 episode。

### I5 冻结探针电池与成本

- 核心 battery 与被测方法无关；design-only extension 必须在 locked audit 前冻结。
- 无效 option fallback 是 intervention 定义的一部分；`T_probe` 与 `L_inner` 分别表示脚本时长和条件重复。
- 静态成本同时计入 core probes 与 design-only extension，只计算启动上界，不执行 rollout。

### I6 kernel 统计界

- 主 kernel 表每个 outer replica 只保留一行，并在该行内保留全部配对探针和 inner fork；不能只保存聚合 count。
- 同一 outer state 上的探针与 inner forks 是 cluster 相关，推断单位仍是 outer replica。
- 先构造每个 cell 的置信界，再计算 `alpha_upper=max(within upper confidence bound)` 与
  `beta_lower=min(between lower confidence bound)`；不得先取经验最大最小后只加一个公共半径。
- exact/approximate 状态、`rho_prune`、support 与每个 cell 的有效样本量必须入表。

`path_c_instrument_evidence_v2` 只接受独立 full-posterior 有放回抽样。它从完整 kernel record 重放 sampler support choice，核对 battery、snapshot、audit unit、probe、posterior probability、outer cluster 与 inner-fork random-key coordinate，并机械生成有效样本量和 support coverage；旧版自报 token 数组和 validity booleans 不兼容且直接失败。

每个 instrument cell 的 value class、audit unit、snapshot、probe、sampler seed、outer 起点和 outer 数量都进入预采集 cell registry 的内容哈希；证据中的 registry 必须与预注册哈希一致。`support_violation` 只能对应冻结词表中的专用 token，不能作为普通响应进入 alpha 或 beta；任一探针或 inner fork 出现该标记都会使主 instrument validity 失败。

### I7 有效性对照与软件一致性

- 核心 battery 与被测方法无关；design-only extension 必须在 locked audit 前冻结。
- 无效 option 的 fallback 是 intervention 定义的一部分，不得在运行中自适应修改。
- point-mass 对照检查 naive fork 与正确估计器等价；overlapping-posterior 对照检查预期 discrepancy 与正确恢复，而不是强求一次随机样本必然失败。
- 最终 `instrument_measurement` 必须由归档的 outer-cluster token cells 重算；汇总字段不能自行成为证据。
- 软件和 instrument validity test 必须绑定模块注册表 test id、source commit 与归档 report SHA-256。

### I8 切分与角色内 cross-fitting

- `SplitManifestV1` 产生 train、design、calibration、locked_audit 四个 deterministic、group-disjoint 角色。
- 数据收集只接受冻结的 `split_group_id`，角色由 manifest 派生；评估器实际读取 Parquet 行并重算 episode、transition 与 group coverage。
- 主 endpoint 只改变 identity：四个角色共享同一个已登记具体布局，identity、style 与 seed group 互不重叠；layout shift 使用另一布局单列为次要 endpoint。角色内部 value labels 使用 manifest 绑定的 cross-fitting。
- split manifest 现在从 `split_core_sha256`、manifest seed、group id 和组内 seed 序号确定性生成全局唯一的 64 位数值 seed，并把完整映射及其哈希写入冻结制品。数据收集、Parquet 校验与 acting return ledger 都拒绝不属于对应 group 的数值 seed。I8 因此记为 `implemented`；这只表示静态实现完成，不表示测试已经运行或预注册已经冻结。

### D1 产物合同

每个 claim-critical artifact 必须绑定预注册中列出的全部语义哈希、schema version、有效 episode 和 transition 数、四角色 split、random key schedule 与原始输入 shard。任何同名但语义哈希不同的产物均不得混用。

当前严格合同尚未贯通正式 acting runtime、策略 factory、环境伙伴映射、kernel outer-replica 原始记录和次要测量重算路径。预注册模板也未填充或冻结。因此 D1 保持 `planned`；已有局部内容哈希校验不等于完整 claim-critical 合同。

第三版 artifact manifest 现在必须声明单一 `active_stage`。每个阶段只允许读取该阶段对应的数据角色和测量集合：software 只读 train，instrument 只读 calibration，design freeze 只读 design，主结果与次要机制阶段才可读 locked audit。未知、提前出现或跨阶段的数据角色会在打开 Parquet 内容前失败。

### D2 软件一致性测试定义

测试定义覆盖 property、differential、metamorphic、end-to-end schema 与静态成本边界。它们绑定到模块注册表中的 test identifiers。本次修订没有运行任何测试，所以这些定义只证明检查对象已经写明，不证明实现通过。

## 5. 模块注册表一致性

下面的表由 `experiments/overcooked_v2/configs/module_registry.yaml` 管理。`implemented`
只表示静态实现存在；它不表示测试通过、数值已冻结或实验可运行。

<!-- PATH_C_MODULE_TRACEABILITY:BEGIN -->
| Module ID | Status |
|---|---|
| A1_CONFIG_BINDING | tested |
| A2_PRIME_RECURRENT_SEQUENCE | tested |
| A3_SEQUENCE_TEMPORAL_DIFFERENCE | tested |
| A4_NORMALIZED_ADVANTAGE_PROBE | tested |
| A5_BASE_RESIDUAL_SECONDARY | tested |
| B1_SYNTHETIC_FACTORIAL | tested |
| B2_RESPONSE_AND_STORAGE | tested |
| B3_ECOLOGICAL_VALUE_CLASSES | tested |
| C1_RESPONSE_READOUT_SECONDARY | tested |
| C2_PRIMARY_RETURN_BUDGET_AUC | planned |
| C3_RETAINED_DECISION_CODE | tested |
| C4_POWER_AND_SECONDARY_REPORT | tested |
| C5_ACTING_BASELINE_BENCHMARK | planned |
| C6_TWO_STAGE_DECISION | tested |
| I1_IMMUTABLE_OCV2_SNAPSHOT | tested |
| I2_EXACT_SHARED_POSTERIOR | tested |
| I3_FULL_STATE_OUTER_SAMPLING | tested |
| I4_EXACT_HISTORY_MATCHING | tested |
| I5_FROZEN_AUDIT_BATTERY | tested |
| I6_SIMULTANEOUS_KERNEL_BOUNDS | tested |
| I7_VALIDITY_CONTROLS | tested |
| I8_SPLIT_AND_CROSS_FITTING | tested |
| V3_BACKBONE_ADMISSION_ADAPTATION_PROBE | implemented |
| PATH_C_FAMILY_POOL_FORMAL_CHAIN | implemented |
| PATH_C_VQBC_V4_1_CHAIN | tested |
| R015_TWO_FAMILY_SUPPORT | implemented |
| R015_PAIRED_AUDIT_ADJUDICATION | implemented |
| D1_ARTIFACT_CONTRACT | planned |
| D2_CONFORMANCE_TEST_DEFINITIONS | tested |
<!-- PATH_C_MODULE_TRACEABILITY:END -->

第三版骨干适配模块和家族级伙伴池正式链保持 `implemented`。VQBC V4.1 已在注册远端环境完成
Flax、Optax、JaxMARL 和 GPU 覆盖的四个目标测试文件，结果为 50 项通过、0 失败、0 跳过；
同一 seed-100 开发单元、四模式评估和回应屏蔽对照也已完成。因此活跃项
`PATH_C_VQBC_V4_1_CHAIN` 标为 `tested`。该状态只表示登记测试已有通过证据；回应屏蔽效应仍为
0，不能据此写成机制验收通过或 `frozen`。表中其余 `tested` 状态仍是历史模块在各自记录
commit 上的已有证据，不传递到本轮变更。

## 6. 当前实施动作

2026-07-19 用户裁决后，当前动作改为按 §7 实现完整模型。远端接线检查、R015 第二版
运行等旧动作降为可选，随模型开发需要再排期。

## 7. 完整模型优先的实现设计（2026-07-19）

### 7.1 目标与裁决背景

用户于 2026-07-19 裁决：不再用逐个门控推进方法设计，先实现一个完整的模型，再通过
实验迭代优化。此前的静态审计发现：提案主方法控制器
`registered_response_sequential_branch_v1`（决策导向的顺序探查分支）与其匹配对照
`registered_random_safe_probe_v1` 在训练器和评估器中都是"未实现"异常；手写 PyTorch
适应层锁定在 2026-07-14 已裁决弃用的自写骨干上；全库没有一条能把"训练出带探查机制的
策略"和"官方配对评估"连起来的模型链。本节定义补齐这条链的实现。

目标状态：一条连通的模型训练链，四个提案条件（决策导向探查、通用回应信息探查、随机安全
探查、无探查）全部可训练；训练入口按“伙伴池核验 → 预拟合 → 阈值校准 → 适应训练”执行
并可恢复。标准评估使用独立入口读取同一条件的 10 个最终策略，执行完整 self-play 与
cross-play 矩阵。早期开发阶段的固定骨干—熟悉训练伙伴配对只保留为历史诊断，不再属于
生产阶段，也不能称为零样本协作评估。

### 7.2 基座决定

- 新适应层建在官方 Flax 训练栈上：骨干网络与参数格式沿用
  `official/overcooked_v2_experiments_adapter.py` 封装的官方循环 actor-critic 网络
  （编码器 + GRU + actor 头 + critic 头，Orbax 参数树，动作次序
  `right,down,left,up,stay,interact`）。官方包只在远端环境可导入，所有 jax/flax
  导入必须像现有适配器一样延迟到调用时。
- 历史开发伙伴池直接复用五个已核验官方 checkpoint：主体 seed 100（自博弈）、伙伴
  seed 101/102（自博弈族）、seed 201/202（Other-Play 族）。伙伴策略按
  `path_c_flax_policy.OfficialFlaxPolicy` 冻结加载，不参与梯度。
- 手写 PyTorch 线（`path_c_backbone_ppo.py`、`path_c_adaptation.py`、
  `path_c_standard_training.py`）冻结为历史，不再扩展；其中仍然正确的合同（伙伴池
  准入报告、回应词表、评估协议）由新链复用定义而不是复制代码。
- OPERATING_CONSTRAINTS §6 第 6 条继续适用：新适应训练循环属于自写训练代码，承担
  正式计算前须复现参照行为；开发迭代不受此限。

### 7.3 模型结构（提案 §4.2 的 Flax 实现）

新模块 `PathCFlaxAdaptationModel` 包含：

- **循环控制骨干**：结构与官方网络的编码器 + GRU 完全一致，从官方主体 checkpoint
  的对应参数初始化；输入为官方局部观测、主体自身上一动作、原始回报和回合边界。
- **共享 critic**：时序差分价值头。这是唯一向骨干反传梯度的头。
- **共享 actor**：在 `stop_gradient` 后的骨干特征上做策略梯度；从官方 checkpoint 的
  actor 头初始化。
- **每家族价值头**：四个伙伴家族原型各一个，时序差分目标，输入特征截断梯度。
- **每家族回应头**：四个伙伴家族原型各一个，在截断梯度特征上预测注册回应摘要的分布；回应摘要
  词表沿用 `path_c_response_summary.py` 的注册本地回应定义。
- **每家族一步转移—回应模型**：给定当前特征与候选动作，预测（回应摘要，下一步特征
  摘要）联合分布，用于评估候选动作的两分支价值；输入特征同样截断梯度。

梯度路由是方法身份约束：骨干只接收共享 critic 的时序差分梯度；伙伴家族编号只路由
训练目标到对应头，checkpoint 成员编号只用于平衡抽样与审计，两者都不进入骨干、actor
或任何测试时输入。

### 7.4 信念与探查控制器（提案 §3.4 的可训练实现）

- **在线信念**：对四个伙伴家族维护软概率 `b(θ|h)`，用各家族回应头对实际观察到的注册
  回应摘要的似然做贝叶斯更新；`B_use` 与 `B_mask` 的唯一差别是当前候选窗口产生的
  回应摘要坐标是否进入更新。
- **候选集**：安全门（风险上限）、早期窗口、回合预算内的普通任务动作或短脚本，
  且候选不同于 actor 原动作。
- **分数**：对每个候选 `p`，用一步转移—回应模型与每原型价值头按信念加权计算
  `Ĵ_use(p)`（回应摘要进入信念支路后的延续价值）与 `Ĵ_mask(p)`（屏蔽该摘要的延续
  价值）；`V̂_mask = max{V̂_base, max_{p'} Ĵ_mask(p')}`；主分数
  `S_seq(p) = Ĵ_use(p) − V̂_mask`。只有严格超过 `max(0, q0.95)` 才执行探查，
  其中 `q0.95` 来自禁止实际探查的校准分数空分布。该实现登记为
  `registered_response_sequential_branch_v1`，替换现有"未实现"异常。
- **匹配对照**同一控制器接口下实现：`registered_random_safe_probe_v1`（相同候选
  支持、安全门、窗口，触发统计在校准数据上冻结）、`generic_response_information`
  （概率加权广义 Jensen–Shannon 散度评分，沿用现口径）、`off`（无探查）。

### 7.5 训练与评估流程

`path_c_model_v1` 和 `path_c_model_v2` 只解释既有开发及旧正式产物。新版正式训练使用
`path_c_model_v3` 与 `path_c_outer_units_v2` 清单；正式种子根为 `2026072302`。清单固定
10 个外层训练单元，每个单元独立绑定一个自博弈骨干、两个外来自博弈运行、两个
Other-Play 运行和三路根随机数。每个官方运行按既有训练更新日程机械保存三个 checkpoint，
不按回报选取。2026-07-25 结果知情修订后，最终 checkpoint 的能力阈值只形成审计报告，
不筛选、替换或重复训练固定 seed；全部五十个 Simple 上游运行均进入原定单元。修订后的
Simple 和 Wide 均标记为结果知情的探索性实验。

伙伴抽样先等权选择四个家族，再在家族内均匀轮换成员。四个家族依次是：轨迹起点冻结的
当前适应策略、本单元骨干的三个历史 checkpoint、两个外来自博弈运行的六个历史 checkpoint、
两个 Other-Play 运行的六个历史 checkpoint。当前适应策略伙伴独立维护控制器、循环状态、
信念、预算和随机数，整个 400 步轨迹内参数不变，伙伴分支对训练损失的梯度为零。

四条件只在同一单元内共享上游对象、预拟合与训练校准；跨单元不得复用 checkpoint、轨迹、
训练标识或随机流。新版训练入口按以下阶段执行并可恢复：

1. **伙伴池核验**：核验五个官方运行、十五个固定历史 checkpoint、最终能力观察报告、
   全部固定 seed 均被保留、哈希和三项真实环境启动守卫。
2. **预拟合**：冻结骨干与 actor，只拟合回应头、每家族价值头和一步转移—回应模型
   （正式预算 1M 环境步）。
3. **训练校准**：在 500 个禁止实际探查的回合上冻结适应阈值。
4. **适应训练**：四个条件各自独立训练 10M 环境步。
5. **部署校准**：每个条件在最终 checkpoint 上独立运行 500 个禁止实际探查的回合；
   正式评估只读取这份部署校准。

适应前骨干和四个适应条件分别形成一个包含 10 个策略的群体。标准评估入口对每个群体执行
10 个对角 self-play 配对和 90 个非对角有向 cross-play 配对，每配对 500 个回合。配对两侧
分别加载自己的参数、循环状态、四家族信念与部署阈值。有向矩阵已经覆盖双方玩家位置，不再
额外复制角色交换。训练伙伴和历史 checkpoint 不得进入正式矩阵。

决策导向群体另执行 90 个有向 cross-play 配对、每配对 500 个匹配回合块的回应屏蔽部署
对照。每块产生 `A1`、`A2-mask` 和 `A2-use` 三条分支回报，并单独报告
`Delta_response=A2-use-A2-mask`、`Delta_cost=A1-A2-mask` 与
`Delta_net=A2-use-A1`；该摘要不得混入标准 self-play 或 cross-play 表。

### 7.6 文件计划

2026-07-19 当日更新：应用户要求，实现改为交付 Codex 执行，且不再采用平铺单文件，改为
分层包结构——方法核心包 `src/path_c/`（contracts / model / belief / probe / training /
evaluation / pipeline 子包）+ OvercookedV2 对接层
`experiments/overcooked_v2/model_dock/` + 单一入口脚本与开发/正式配置。完整文件树、
接口合同、配置模式、训练与评估语义、测试要求和里程碑见
[PATH_C_MODEL_IMPLEMENTATION_SPEC.md](PATH_C_MODEL_IMPLEMENTATION_SPEC.md)（唯一实现
规格；本节早先的平铺文件计划作废）。`path_c_adaptation.py` 与
`path_c_standard_evaluation.py` 中的"未实现"异常在新控制器落地并通过远端测试后另行
处理；在此之前保持现状，避免半成品冒充主方法。

2026-07-24 新增的正式对象位于 `src/path_c/contracts/outer_units.py`、
`src/path_c/training/rollout.py`、`src/path_c/evaluation/response_contrast.py`、
`experiments/overcooked_v2/model_dock/`、`experiments/overcooked_v2/official/` 和三个
正式入口脚本中。历史 PyTorch、R015 和熟悉训练伙伴评估函数保持只读，不进入新版生产阶段。

### 7.7 保留的正确性不变量（不是流程门）

1. 数据充分性申报与开发/确认标注（OPERATING_CONSTRAINTS §6）。
2. 不把 oracle 信息（伙伴身份、机制、风格标签）放进可部署输入；原型编号只做训练
   路由。
3. 梯度路由约束（§7.3）。
4. 接线回读：每次运行从产物回读实际生效条件，不信配置意图。

### 7.8 第四版替代决定（2026-07-25）

§7.1–§7.7 记录第三版形成过程，只用于历史解释。当前实现改为价值商信念条件 Bellman
控制器（Value-Quotient Belief-Conditioned Bellman Controller，简称 VQBC）第四版：
每个外层训练单元只训练一个模型，本单元自己的官方 checkpoint 作为冻结参考策略，8 个
无标签动作价值槽通过动态 complete-link 价值商形成后验，唯一回应概率核同时服务
`J_use`、`J_mask` 和通用回应信息部署模式。正常动作始终由后验条件策略产生，不再使用
外部探查动作覆盖、候选窗口、探查预算、阈值校准、可训练 actor、预拟合阶段或伙伴家族
训练路由。

第四版训练顺序固定为 `pool_check → training`，正式预算合并为 11M 环境步；同一 checkpoint
以 `posterior_use`、`prior_only`、`reference_only` 和
`generic_response_information` 四种模式部署。标准评估继续使用 10 个 self-play 配对和
90 个有向 cross-play 配对，每配对 500 回合；左右两侧分别加载自己的参考参数、循环状态、
信念和 KL 散度温度。回应屏蔽对照现在属于第四版实现范围。

训练指标从第一次完整 rollout 起记录责任度熵、每槽有效质量、当前价值商数量、回应码
使用和困惑度、KL 散度第 95 百分位数、参考动作偏离率、后验熵以及
`J_use-J_mask` 分布；这些指标用于判断无标签槽是否实际分化，不替代开发运行授权或科学
证据。

完整接口、公式、文件、训练和验收规定见
[PATH_C_MODEL_IMPLEMENTATION_SPEC.md](PATH_C_MODEL_IMPLEMENTATION_SPEC.md) §14。第四版当前
已完成远端 41 项目标测试和首个 Test Time Simple 开发单元训练、四模式评估及回应屏蔽
对照，因此实现状态为 `tested`。开发结果显示八个无标签价值槽仍保持对称、所有活动价值商数
均为 1，后验没有改变正常动作；该结果不具备科学结论权限，也不授权十单元正式训练。


### 7.9 V4.1 潜在 Bellman 混合根因修复（2026-07-26）

第四版执行闭环已经证明后验能够进入正常动作，但首开发单元的八槽责任度、价值商和后验全程
塌缩。V4.1 不调整探查阈值、训练步数或伙伴标签，而是统一替换潜在结构学习内核：

1. 每个 slot、每个 twin 使用独立 dueling Q 参数；共享仅止于 TD-only recurrent backbone；
2. episode responsibility 使用完整 400 步上的 TD、response、reward 与 next-Q 联合证据和，
   并由 target network 与 episode bootstrap mask交叉拟合；
3. 每个 epoch重新执行 E-step，完整 lane只进入一个 minibatch，target每个 minibatch执行
   Polyak update；
4. slot posterior永久保留并在slot层Bayes更新；value quotient只作为当前控制视图；
5. 只有 `distance + radius_i + radius_j <= epsilon_eq` 时才认证等价，uncertainty不再促进合并；
6. response code target不再读取当前 responsibility，解除潜在assignment与回应码本的循环依赖；
7. 回应屏蔽触发使用尺度相关float32容差，不再将机器舍入误差视为正信息价值。

V4.1 schema 为 `path_c_model_v4_1`，checkpoint manifest与metadata分别升级为
`path_c_flax_checkpoint_v3` 和 `path_c_model_checkpoint_metadata_v3`。旧V4 checkpoint、共享slot
decoder字段和class-to-uniform-slot posterior投影均失败关闭。详细代码—根因映射和远端复跑
裁决见 `docs/status/PATH_C_VQBC_V4_1_ROOT_REPAIR.md`。本节记录实施时的本地静态边界；随后完成的
远端测试和同一开发单元复跑见 §7.10。十单元正式训练继续关闭。

### 7.10 V4.1 首开发单元远端复跑（2026-07-26）

V4.1 在独立远端目录完成四个目标测试文件，结果为 50 项通过、0 失败、0 跳过。seed-100
开发训练从产物回读 1,228,800 个环境步、3,072 个完整回合和 96 次更新；四模式评估完成
800,000 个环境步，回应屏蔽对照完成 600,000 个有效分支环境步。全部产物保持
`scientific_readout_allowed: false`。

潜在结构出现了实质变化：最近 rollout 的 12,800 个状态全部形成 8 个活动价值商，最终后验
熵均值降至 0.080809；`posterior_use` 与 `prior_only` 在 500 个匹配回合中的 452 个回合产生
不同的参考动作偏离数。因此 V4.1 已打破第四版的均匀槽对称性，后验也会改变正常动作。不过
最终 32 个回合的责任度有效质量为 `[0,4,23,1,0,3,0,1]`，使用仍集中在少数专家。

决定性缺口仍然存在：500 个回应屏蔽回合的预测信息净值均明显为正，范围为 0.139714 至
0.278620，但 `A2-use-A2-mask` 逐行、十个等频分箱和最高分箱效应全部为 0。这说明当前回应
使用通道尚未承载可观察的奖励效应。十单元正式训练继续关闭；随后完成的第 0 步控制路径追踪
见 §7.11。

### 7.11 回应屏蔽控制路径追踪（2026-07-26）

追加的远端开发诊断没有训练参数，只沿相同 checkpoint 追踪 32 个第 0 步样本和 32 个完整
双分支回合，共 25,664 个有效环境步。实际回应更新使槽信念与屏蔽分支的 L1 距离均值达到
0.367，因此回应编码、似然选择和屏蔽开关均已接入真实后验路径。

失效发生在后验与执行动作之间。回应后下一步 `J_use` 的最大绝对变化均值为 2.364，真实动作
分布的总变差距离均值却只有 0.000741，下一步采样动作和最大概率动作均不改变。完整回合中只有
3/32 个左侧策略最终改变动作，且没有任何逐步奖励或回报差异。

静态代码与追踪读数共同确定原因：信息价值公式使用无约束的下一动作最大值，执行策略却由冻结
参考策略和 KL 散度温度共同约束。前者计算的控制切换通常不是后者会执行的动作，所以高
`S(a)` 不能解释真实分支的可用价值。正式训练继续关闭；下一次代码修订应使回应价值的延续
算子直接对应真实执行分布。
