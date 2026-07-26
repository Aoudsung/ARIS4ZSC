# PROJECT_DASHBOARD.md

**ARIS-Bellman for Zero-Shot Coordination — pipeline status & decisive-results tracker.**
Last updated: 2026-07-26 · Read this first for "where am I" (30 seconds).

> Required entrypoint per [CLAUDE.md](CLAUDE.md). Execution rules live in
> [OPERATING_CONSTRAINTS.md](OPERATING_CONSTRAINTS.md); this file is *status*, not *permission*.
>
> 治理精简 2026-07-08：本文件的门已按 docs/status/GOVERNANCE_CUTLIST.md 处置；加/减门须过 OPERATING_CONSTRAINTS.md §7 退休阀门。

---

## Terminology (authoritative)

- **ARIS** = the harness (Auto-claude-code Research In Sleep). The tooling under `.claude/skills/`.
- **ARIS-Bellman** = the research method. Factor-local Bellman control for ZSC.
  Code in `src/aris_bellman/` + `experiments/overcooked_v2/`.
- **SP / XP** = self-play within one independent training run / cross-play between
  different independent training runs.

This project (`ARIS4ZSC`) = *using ARIS the harness to develop the ARIS-Bellman method.*

---

## 1. Project identity

| | |
|---|---|
| **Title** | Probe What Changes the Decision: Decision-Focused Active Partner Inference for Zero-Shot Coordination |
| **Current line** | Path C = 面向协作决策的信息价值方法：只在伙伴回应可能改变后续控制且收益超过任务成本时探查。价值无关身份不变性和梯度路由只作机制约束与证伪检查，不再承担论文公开身份。 |
| **Current proposal** | [PATH_C_PROPOSAL.md](idea-stage/refine-logs/PATH_C_PROPOSAL.md), [PATH_C_THEORY.md](idea-stage/refine-logs/PATH_C_THEORY.md), [PATH_C_MODULE_DESIGN.md](idea-stage/refine-logs/PATH_C_MODULE_DESIGN.md), [PATH_C_EXECUTION_PLAN.md](idea-stage/refine-logs/PATH_C_EXECUTION_PLAN.md), [PATH_C_PREREGISTRATION.md](idea-stage/refine-logs/PATH_C_PREREGISTRATION.md), [PATH_C_OPPORTUNITY_AUDIT_SPEC.md](idea-stage/refine-logs/PATH_C_OPPORTUNITY_AUDIT_SPEC.md)（当前实现是 `path_c_model_v4` 的价值商信念条件 Bellman 控制器。首个 Test Time Simple 开发单元已完成训练、四模式评估和回应屏蔽对照；八个无标签槽没有分化，因此十单元正式训练未启动。第三版 Test Time Wide 继续在隔离目录运行，只保留其已授权历史链，不覆盖第四版）；实现状态由 `experiments/overcooked_v2/configs/module_registry.yaml` 管理。 |
| **Thesis** | 在官方局部历史下，智能体应按注册回应通道对受限控制器的增量价值选择任务内探查，并相对最佳通道屏蔽动作扣除机会成本；主要证据来自匹配 cross-play 回报，而不是身份恢复。 |
| **Benchmark** | JaxMARL **OvercookedV2** Test Time Simple（主基准）/Wide（布局复现）+ **Hanabi**（必需第二领域，尚未启动）+ toy_factor_game（理论回归） |
| **Target tier** | ICLR / NeurIPS / ICML class |
| **Compute budget** | 第四版首开发单元从产物回读 1,228,800 个训练环境步和 3,072 个完整训练回合；四模式开发评估为 800,000 个环境步，回应屏蔽对照由原始触发步回读 595,520 个有效分支环境步。旧伙伴池 Test Time Simple 正式链从产物回读 2,010,480,640 个环境步；家族级伙伴池第三版 Test Time Simple 为 2,072,480,640 个环境步。第三版 Test Time Wide 仍在执行，实际预算待最终审计后回读。冻结前另有一次两布局共享的官方 seed 999 生产验收，本次实际执行 29,949,952 步，不属于十训练单元或标准矩阵。 |

---

## 2. Pipeline status — CURRENT STAGE

**Stage: 2026-07-26 第四版价值商信念条件 Bellman 控制器的首个 Test Time Simple 开发单元已完成。远端四个目标测试文件为 41 项通过、0 失败；训练产物回读 1,228,800 个环境步、3,072 个完整回合和 96 次更新。全部 12 个定期训练观测点的活动价值商数均为 1，责任度和后验熵始终为 `log(8)`。四模式各 500 个匹配回合中，`posterior_use` 与 `prior_only` 逐 seed 完全相同；回应屏蔽对照的 500 个回合三分支回报也逐行相同。当前实现没有让八个无标签价值槽分化，后验不承载正常动作差异，因此不启动十单元正式训练。下一项唯一优先工作是修复槽对称性和价值商形成机制，再重复同一个首开发单元。完整证据和 SHA-256 见 [EXPERIMENT_LOG.md](docs/status/EXPERIMENT_LOG.md) 顶部条目；全部读数保持 `scientific_readout_allowed: false`。**

**Parallel historical run: 2026-07-25 家族级伙伴池第三版 Test Time Simple 已完成 10 个训练单元、五个标准配对矩阵、决策导向回应屏蔽对照和最终审计。六个目标测试文件为 79 项通过、0 失败、0 错误、0 跳过；完整链从产物回读 2,072,480,640 个环境步。决策导向的自我配对/跨策略配对均值为 79.90/−0.28，关闭探查为 94.01/−1.98；两者匹配跨策略配对平均差为 +1.7071，训练单元节点重采样 95% 区间为 [−2.8061, 7.7083]，没有达到 20 分实际意义门槛。注册回应使用效应为 +0.0156，净效应为 +0.0431。因为上游能力阈值是在观察三个未达标任务后改为只报告，本次运行属于结果相关协议修订后的探索性结果，继续保持 `scientific_readout_allowed: false`。最终审计 SHA-256 为 `4e5d16c752491023103732f5d90a9bbcd03d6f241a19b03ded4380cf4d296815`，完整报告见 [PATH_C_TEST_TIME_SIMPLE_RESULTS.md](docs/status/PATH_C_TEST_TIME_SIMPLE_RESULTS.md)。统一驱动进程 `4182410` 已完成 Test Time Wide 的准备、目标测试、机械接线和冻结阶段，正在继续 Wide；尚无 Wide 最终审计读数。**

**Historical stage: 2026-07-20 Path C 已完成固定骨干与伙伴池上的第二个独立适应训练 seed 四条件熟悉训练伙伴诊断。六个目标测试文件为 48 项通过、0 失败、0 错误、0 跳过，三项真实启动守卫和 81 项原始产物审计检查全部通过。seed-2 完整链从产物回读为 5,504,000 个环境步和 13,760 个完整回合；适应前骨干锚点为 38.6806，决策导向、关闭探查、随机安全探查和通用回应信息的诊断平均原始回报依次为 -28.0903、-21.1111、18.1250 和 11.7361。匹配回合中，决策导向减随机安全探查为 -46.2153，决策导向减关闭探查为 -6.9792，通用回应信息减关闭探查为 +32.8472；seed-1 的三个相应方向均未复现。两训练 draw 等权后，决策导向减随机探查为 -14.8785，决策导向减关闭探查为 -2.6563。该复现固定同一官方骨干和训练伙伴池，不是独立外层训练单元；全部读数只属于 `development` 诊断，不支持或反驳预登记的第一或第二项科学假设。完整预算、五行结果表、探查率和证据哈希见 `docs/status/EXPERIMENT_LOG.md`。**

下段为 2026-07-20 第一个零锚阈值训练 draw 的旧阶段口径，保留作历史：

**Stage: 2026-07-20 Path C 零锚探查阈值四条件开发迭代已经完成。决策阈值现为无探查决策分数 95 分位数与 0 的较大值，本轮阈值为 0.0097556；随机安全探查概率为严格超阈值比例 0.05，通用回应信息仍使用自身 80 分位数。六个目标测试文件为 48 项通过、0 失败、0 错误、0 跳过，三项真实启动守卫全部通过。预拟合权重和校准原始行与首次运行逐字相同；新增的适应前官方骨干评估完成 9 个配对乘每配对 64 个回合且零探查。四个条件各完成 1,024,000 个适应环境步、2,560 个适应回合和 576 个评估回合；完整链从产物回读为 5,504,000 个环境步和 13,760 个完整回合。决策导向的评估探查数从首次运行的 10,537 降至 3,338，本轮平均原始回报为 10.6597；关闭探查为 8.9931，两者使用相同回合 seed 的配对平均差为 +1.6667，配对差标准误为 2.5324。随机安全探查为 -5.7986，通用回应信息为 -20.9028。原始行重算、比率、预算、条件间 seed、收据和哈希检查全部通过。远端还证明相同参数、数据和 seed 的 GPU 第一次梯度更新不具备逐字确定性，因此跨运行的安静条件 checkpoint 和回报差不能作阈值因果解释。全部读数仍只属于 `development` 诊断，不支持或反驳预登记的第一项科学假设；十策略正式清单未生成，正式配置没有运行。下一项唯一优先工作是在不再改变机制的前提下，用第二个独立训练 seed 重复同一套四条件匹配开发运行，直接检验决策导向减关闭探查的差是否能跨训练运行复现。完整证据见 `docs/status/EXPERIMENT_LOG.md` 顶部条目。**

下段为 2026-07-20 首次四条件开发运行的旧阶段口径，保留作历史：

**Stage: 2026-07-20 Path C 完整模型的首次四条件开发运行已经完成。六个目标测试文件最终为 44 项通过、0 失败、0 错误、0 跳过；观测布局、交付计数和官方网络热启动一致性三项真实启动守卫全部通过，其中新模型与官方网络的循环状态、动作头（actor）对数概率和价值头（critic）最大绝对误差均为 0。共享预拟合完成 204,800 个环境步和 512 个回合，共享校准完成 51,200 个环境步和 128 个回合；四个条件各完成 1,024,000 个适应环境步、2,560 个适应回合，以及 9 个配对乘每配对 64 个回合的开发评估。原始行重算与保存摘要逐字段一致，全部哈希、非负交付计数、关闭探查条件零探查和有限参数检查均通过。完整证据见 `docs/status/EXPERIMENT_LOG.md` 的 2026-07-20 条目。全部运行仍标为 `development` 且禁止科学读数；十策略正式清单未生成，正式配置没有运行，本项目仍没有 Path C 确认性结果。下一步只分析九个配对中的席位差异与探查覆盖，先解释决策导向和随机安全探查的描述性回报为什么低于关闭探查，再决定下一次模型修改。**

下段为 2026-07-19 及以前的旧阶段口径，保留作历史：

**Stage: 2026-07-19 用户裁决——完整模型优先。项目不再用逐个门控推进方法设计：先按提案实现一个完整的模型（提案 §4.2 的适应网络、§3.4 的顺序探查分数、四个匹配条件、配对评估），再通过实验迭代优化。实现基座沿用 2026-07-14 已签署的官方训练器决定：新适应层建在官方 Flax 循环网络之上，从现有五个官方 checkpoint（主体 seed 100，伙伴 101/102 自博弈、201/202 Other-Play）出发形成第一个开发伙伴池；手写 PyTorch 适应线冻结为历史，不再扩展。R015 保留全部历史证据，地位回到提案 §2.3 的原始定位——可选的支持性测量，不再排在模型实现之前。执行边界不变（远端运行仍须逐次授权），数据充分性纪律不变；开发阶段的训练与读数按开发诊断记录，确认性冻结推迟到正式十单元实验之前。实现设计见 PATH_C_MODULE_DESIGN.md §7。**

**Stage: R015 官方伙伴支持已经成立，延续动作规则仍为 `map_prototype_committed_cook_v1`。2026-07-17 第一版过滤器在六个候选上全部得到 100% 零支持关闭率；该失败、原始证据和机械报告继续作为历史记录，不被第二版代码改写。2026-07-18 已静态实现第二版端到端路径：过滤器对六种伙伴动作和与官方观测相容的环境结果作边缘化，再抽取相容后继；离线设计和在线执行共享同一设备内一步更新；规划按剩余长度分批，在分支头只更新一次信念并用编译循环推进后缀；三实验组的配对执行、独立重放、15 份机器制品、四原型乘 2,500 轮的正式抽样日程，以及“设计数据 → 两遍冻结装配 → 80 块试点 → 正式轮次”的可恢复统一入口均已接通。正式抽样日程使用 `path_c_r015_formal_sampling_schedule_v4`：每个轮—原型坐标在数据前绑定不变的审计单元编号和 32 个逐槽直接派生的 seed；32 位值允许偶然碰撞且不得重抽，只有整块机械失效才按编号推进，任何结果都不得参与选择。以上只是未运行的代码状态：第二版尚未产生设计数据，没有选择粒子数、重采样时机或规划分支数，没有冻结清单或试点结果，也没有 R015 科学读数。正式轮次仍须在试点完成后取得独立授权。预登记保持未冻结，`scientific_readout_allowed: false`。**

2026-07-18 的实现把计算效率改为端到端执行路径，而不是只做单个候选的试验代码。过滤候选
使用一次编译的设备循环、按原型分组的伙伴网络调用、固定父粒子槽和可恢复的分批处理；首个
候选通过后不会计算更高候选。规划使用嵌套样本复用、按剩余回合长度分组和设备内循环；真实
配对回合也在设备段内连续推进，只在登记的因果边界返回宿主。统一入口会把每阶段的完成产物
和 SHA-256 写入状态，恢复时只跳过摘要仍一致的完整阶段。机器制品生成器在冻结前回读五个
checkpoint、Flax 权重摘要、训练清单、环境源码闭包、回应定义和三类重放入口；试点协议必须
先填入本次设计选择、冻结输出和报告位置，仍含 `pending` 的模板不能进入冻结。上述代码未在
本轮执行，不能把静态实现写成性能、正确性或科学结果。

正式执行不再只依赖单个块文件：成功块账本、机械失效账本、每个块摘要和外部运行摘要共同
绑定 `path_c_r015_formal_dataset_v3`。孤立块恢复会按同一冻结请求重跑并比较科学与随机数证据；
经核验的机械计时、速率和 JIT 缓存计数不参与确定性相等，
单次效应查看使用“裁决前已消费 → 终局落盘 → 完成收据”的单向摘要链。规划车道按伙伴原型
稳定分组以复用编译图，五个不可变 Flax 参数树摘要按对象身份复用，在线循环只返回可重建前态
的后态树，同一咨询点的完整信念只摘要一次。这些改动覆盖统一正式路径，不改变抽样、效应量或
统计规则，也尚未形成运行证据。

正式整块替换规则也已从运行时临时重试改为数据前固定的抽样合同
`iid_round_vectors_ego_cook_seat_with_prefrozen_replacements_v4`。每个轮—原型坐标的
`audit_unit_id` 在 32 次尝试间保持不变，编号 0 至 31 的 seed 各按登记坐标直接派生并在正式数据前写入清单；32 位值允许碰撞且不得重抽。只有
整块机械失效才推进，32 个全部失效就终止整个正式审计。该静态修订没有运行正式块，也没有
改变任何统计常数、效应定义或查看规则。

2026-07-14 根因裁决在相同布局和 3000 万步预算下复现官方 JaxMARL Independent
Proximal Policy Optimization 约 130 分，而本项目手写 PyTorch 训练器只达到约 19 分并发生
能力消退。用户随后签署五项决定：正式主体与伙伴生产改用官方训练器；主体钉在厨师席；主体
使用不与伙伴支持共享 seed 的官方 seed 100；每局 0.5 次正确交付的准入下限不变；自写训练器
承担正式计算前必须复现其参照曲线。静态实现已经加入官方运行登记、Flax 参数树摘要、官方
策略接口、保持既有 1,600 行证据结构的生成接口和固定席位抽样合同。远端核查已确定 Hydra
覆盖、Orbax 参数路径、循环网络调用、动作顺序、官方评估随机键与事件重算，以及
Other-Play 的独立配料置换。初始化 checkpoint 保存—恢复和完整 400 步回合均通过。用户随后
裁决 experiments-package 网络就是正式参照，旧 130 曲线降为诊断历史；名义 3000 万步保持
官方意图，实际有效步数按整数日程回读。seed 999 最终四分之一平均原始回报为 168.0493，
相对最高区间比例为 1.0，机械验收通过。随后五个正式运行和唯一一次联合准入均完成；四个
候选的每局正确交付率分别为 5.80、8.50、8.31、5.63，全部超过 0.5 下限。2026-07-15 静态
工单进一步固定粒子数与重采样时机网格、零支持关闭率等三项过滤器门槛、`R` 与 `2R` 的
200 点一致规则、随机数角色
隔离、待签清单读回和九项试点检查。2026-07-15 用户确认 Fable 已完成复核，并授权串行执行
设计数据、机械冻结和试点；补充授权把 6 GPU 小时从硬停止改为成本预估。远端静态测试 11 项
通过，但实际生产接口核查在数据前发现 `C(q,x)` 只有抽象名。用户签署的数据前补全规定：
唯一最大后验原型自己的 checkpoint 在厨师席执行；最大值精确并列时回退官方 seed 100；五个
成员的循环状态从回合开始并行推进，切换不清零。静态代码和预登记已按该规则补全，且规划与
执行绑定同一实现。用户说明无需再次进行 Fable 复核。远端目标测试随后 48 项全部通过；真实
checkpoint 核查在策略加载前发现生产清单绑定旧适配器摘要，而当前适配器已加入延续控制器接线，
摘要不同。第一次第 0 段据此失败，设计数据没有产生。随后已完成适配器恢复与运行逻辑迁出；
四项既有闭包不一致也已修复。同一范围扩展为 192 项后全部通过，五个真实 checkpoint 的官方
编译调用与 R015 包装器在 logits、动作和循环状态上逐项相同。运行侧只缓存并编译官方网络，
未改训练适配器；400 步诊断从约 14 分钟降至 45.08 秒。设计启动前又发现规划器在每个未来
步骤对每条真实分支重新传播全部粒子，最小允许候选即产生至少 358,888,243,200 次粒子环境
转移；当前源码若预先计算全部三组规划候选则为 2,512,217,702,400 次。绕过该嵌套需要修改
已登记规划含义，因此设计数据仍为零，正式审计仍锁定。

随后用户在数据前批准按当前粒子云抽取每候选固定数量隐藏状态，并把规划分支内信念冻结在
分支头；真实执行仍逐步更新信念。过滤候选评估改为设备端固定批宽扫描后，在两条历史和
256 粒子候选上与逐步参照完成四项逐位等价证明。80 条固定历史上的六个冻结候选已经全部
运行；机械选择报告显示六者零支持关闭率均为 1.0，超过 0.02 上限，故状态为
`filter_design_precision_infeasible`。该结果在任何规划选择、冻结清单、试点或正式效应读取
之前关闭了执行链。

2026-07-13 用户把 R015 设为当前最高优先级测量：在继续扩展主动探查组件前，先审计官方
局部观测下是否存在可购买的决策信息。静态复核已把对象修正为注册回应通道的增量使用效应、
相对最佳回应屏蔽参照的净收益，以及可选的完整信息保守上界。该通道效应不冒充全部伙伴信息
价值；旧版“直告伙伴身份的固定短视控制器是全类上界”表述作废。R015 是本轮资源排序，
不是新增的通用启动条件。

历史批的 seed 101/102 属于同一循环 Independent Proximal Policy Optimization 伙伴族；第二个
历史开发支持伙伴族是循环集成 Double Q-learning 的 seed 201/202。四个旧批训练已经完成，合计
80,000,000 环境步、200,000 个 400 步回合、800 个指标窗口和约 6.1872 GPU 小时。四份
checkpoint 文件 SHA-256、四份模型权重 SHA-256 和四个训练运行标识分别互异。一次冻结的
联合准入生成 16 个有向配对各 100 局、共 1,600 条完整证据；只有 seed 101 自身配对达到
每局 1.02 次正确交付，其余三个候选均为 0.00，故正式支持保持为空。软件完成性核查
（Type-A）只判断运行完成、预算和制品真实性，不把这些回报当作 R015 科学结果。证据已拉回
[本地审阅包](review_bundles/r015_partner_production_20260713/)。这些候选已经退出新的正式支持链，
但历史证据不改写。R015 不承担 held-out 伙伴族证据。Hanabi 必需
第二领域、独立 held-out 伙伴族评估和面向协作的信息价值定位记录于提案 §1、§6 与 §10。

2026-07-13 的远端运行使用 GPU 4、CUDA 12.9 和 JAX/JAXLIB 0.4.38；两个不可纠正
ECC 计数均为 0。骨干完成 30,000,000 步、75,000 局和 19,200 次更新，累计 30,801
次正确交付。准入按官方 PPO 的随机动作评估路径执行：22.5M 与 30M 快照分别达到
0.56/0.59 次正确交付每局，7.5M 与 15M 未入池。适应层完成 10,000,000 步、25,000
局和 100 次更新，累计 9,916 次正确交付，但探针总数为 0，末窗口平均原始回报为
−0.96。本次运行证明训练链路可执行，不是 10 seed 配对结果，也不支持主动探针主张。

2026-07-13 的早期修订把响应头改为每个独立伙伴一个条件模型，在线维护伙伴概率并用
概率加权的广义 Jensen-Shannon 散度选候选动作；活动池现在要求至少两个独立训练 seed。
正式适应训练前还必须完成 1M 步响应头专属拟合与 500 局无探针阈值校准。seed 102 骨干
已完成；早期适应链路的活动池联合准入、响应拟合、校准和新 10M 适应训练均未运行，旧
seed 101 产物没有被覆盖。本段活动池不同于本轮已经执行且未形成正式支持的 R015 联合准入。
远端 GPU 6 使用 CUDA 12.9 与 JAX/JAXLIB 0.4.38 完成的 177 项针对性回归只绑定早期
零探针修订。R015 本轮另在 GPU 7 完成独立接线核查：不可纠正 ECC 错误为零，CUDA 12.9、
JAX/JAXLIB 0.4.38、GPU 后端和实际编译计算通过；同一范围最终 136 项测试通过、0 失败、
0 错误、0 跳过。该范围覆盖 checkpoint 保存与加载、模型权重哈希、两伙伴族来源核验、R015
控制器与过滤器、配对运行、三类重放、OvercookedV2 状态快照和正式路径拒绝有限原型两步
代理量。它仍只是软件证据，不是训练、能力准入或科学读数。
这 136 项是历史范围。2026-07-14 新增的官方 Flax 产物与策略接口先在 GPU 4 通过 155 项
测试；参照身份、seed 999 验收规则和有效步数合同修订后，六个目标文件在 GPU 4 重新执行
159 项测试，0 失败、0 错误、0 跳过。Other-Play 启动器的数据前接线修复另执行 21 项针对性
测试及 1 项递归修复测试，均通过。

早期 seed 102 骨干在 GPU 6 完成 30,000,000 步、75,000 局和 19,200 次更新；300 个指标窗口
与模型参数全部有限，累计 33,784 次正确交付。该运行当时只增加了同一伙伴族内的独立原型。
2026-07-13 的旧伙伴生产已经另行生成第二伙伴族，但四候选联合准入没有形成正式支持；
2026-07-14 的官方伙伴生产随后形成新的四候选正式支持。两批都没有构成探查效果证据。

本轮代码另加入伙伴条件折扣原始回报头和
`finite_prototype_two_action_surrogate`。后者只在有限伙伴原型、回应记号和下一原子动作上计算
两步代理量，没有任务转移 `x`、完整隐藏执行状态或实际回应使用/屏蔽延续分支。它已明确降为
非科学诊断；训练器和评估器把 `decision_focused` 保留给尚未实现的正式顺序分支，并拒绝用
该代理量产生科学读数。独立的 R015 审计控制器现已静态实现官方历史白名单、四原型分层粒子
过滤、当前回应投影、完整剩余回合成对规划、首次正分与确定同分规则；配对运行模块实现未触发
时一次执行并复制规范字节、触发时从 `t_p` 分叉、状态快照推进和三个独立重放入口。
`path_c_r015.py` 继续从冻结规划分支和 checkpoint 重算价值、动作、循环状态与安全结果，不信任
数据中的自报内容。旧四伙伴 checkpoint 已经生成，但因为联合准入失败而没有形成正式支持，
也不进入修订后的官方训练器链；新的官方主体与四个伙伴 checkpoint 已产生并通过来源、文件
摘要、Flax 权重摘要和训练运行标识核验，伙伴支持报告为 `admitted`。四原型等权先验已经
静态登记；粒子数、重采样时机、规划分支数以及 R015 正式清单和哈希仍为 `pending`。
静态实现和伙伴准入通过不等于正式控制器制品已经冻结，当前也不能启动正式裁决。1000 万步
学习版正式控制器的未实现保护没有解除。

同轮静态复核还关闭了两个正式对照旁路：无探查条件和通用回应信息基线在未触发探查时保留
actor 原动作；正式校准必须从内容寻址的 500 局逐决策账本重算完整性与阈值。现有正式通用
回应信息配置的摘要哈希仍为 `pending`，因此不能启动科学训练。

The earlier asymmetric-layout + role-conditioned v2 partner line reached a
negative blind-test readout, and the Link-A substrate certificates reached a
second negative result on 2026-07-09:
the terminal handoff substrate still made partner tendency readable from public
state, so waiting and reacting captured nearly all oracle value. Those results
are evidence for redirecting the project, not positive support for the original
ARIS-Bellman claims.

Path C is now the active line. The third-version interfaces and the P1–P5 code-review
findings have been revised and bound to code commit
`fc647fb00255c5bfc58253a38fa145cb8864afd7`. On GPU 5, the remote CUDA JAX suite
passes 290 tests with no skips; the log SHA-256 is
`6b30b0cd623ebe650f3ea084de2dafb8cea64aef7bb8413587ab70c6b17fbc85` and the
JUnit report SHA-256 is
`d40b43bc7084c8ce916bbc537fd7a6919da28b8b3a760ef552b197b3199f97ab`.
The formal 101-row module-registry software report SHA-256 is
`61f669b0c980f454a138eadd22ed3a846621bb94e86e8bec568023b546af8bf4` and it
binds module-registry SHA-256
`2c22792a0a16318f63ee2efa6dc9adcd542c9a78377c65f650e8d17577e4adb7`.
The preregistration remains an inadmissible template with unfilled numeric and
semantic-hash slots. One Path C formal training seed now exists, but no multi-seed
pairing evaluation, primary result, or scientific readout is recorded.

R003's minimal software semantics are complete after a fail-closed CUDA JAX
compatibility scan on GPU 5.
All four proposal candidate layouts expose 96-dimensional public observations, but
their option counts are 28, 28, 26 and 27 and their ordered option identifiers differ.
Across all 23 registered JaxMARL layouts, 22 satisfy the two-agent evidence contract;
the largest group sharing one exact observation-and-option contract has size 1, below
the four role-isolated layouts required by the proposal. The bound Type-A artifact is
`.codex_remote_validation/path_c_r003_all_layout_semantics_bound_fc647fb_20260711.json`
with SHA-256
`6268916dc37a5571e024224e919e596e3d80e1346cd3c86d74266ee8db4798a1`.
The former single `EgoEvidenceSpecV1` vocabulary and raw numeric probe scripts would
therefore change action meaning across layouts. The project did not add a cross-layout
action layer. R003 consequently fixed `asymm_advantages` as an internal single-layout
software target; that choice remains a valid Type-A compatibility result but is no longer
the paper's main benchmark layout.

The resulting code object is commit
`b6f32578837dd3b5146c355500b911400cb42f78`. GPU 5 then passed 108 targeted tests
and the broader 339-test suite with zero failures and zero skips. The broader JUnit
report SHA-256 is
`89116841aad9d63a2c1e7f4a6641b0f17f7b44e27979fc967ce2ebfa9a3e3390`.
The old R010 input fixes `asymm_advantages`, a 96-dimensional public observation,
28 options, the 30-theta `path_c_synthetic` partner registry and the existing random-key
schedule. Its remote artifact SHA-256 is
`1a548d0800d4052b45920cc746184af3de280004fff686c96af52b07ce9c2406`.
It has not been run, and no locked data was accessed. The planned R010 instrument smoke
is now paused because it does not test compatibility with published benchmark results.

The experiment direction was corrected on 2026-07-11. The main paper comparison now
uses the published ICLR 2025 OvercookedV2 Test Time protocol: `test_time_simple` is the
primary layout and `test_time_wide` is the prespecified replication. The standard metric
is mean cross-play (XP) episode return, where independently trained policies are paired
at test time. The official scale is 10 independent seeds, 90 directed cross-seed pairings
covering both player positions, and 500 episodes per pairing. Self-play return and the
self-play-minus-XP gap are co-reported. A literature check through 2026-07-11 found no
later published result using the exact same Test Time Simple/Wide protocol. Fictitious
Co-Play is therefore the best directly comparable published reference found, with XP
`6±29` on Test Time Simple and `23±40` on Test Time Wide; it is not labelled a
protocol-independent global best result.
The official repository publishes code and configuration but no release or downloadable
baseline checkpoint, so published numbers are the current external reference.

The R005 static gap check completed on 2026-07-11 without running any code. Loading the
two Test Time layouts is not a blocker (both are registered in jaxmarl v0.1.0, and R003
already reset all 22 two-agent layouts). Among environment kwargs, only
`indicate_successful_delivery` requires an `OCV2Adapter` code change; view size, random
agent positions, 400 steps and path-planning flags are config-only. The real blockers are
six protocol-semantic gaps: privileged global observation (the 96-dim featurizer reads the
full grid and `state.recipe`), an option action space whose primitive expansion reads full
simulator state, fixed-ego-versus-scripted-partner training with no self-play, reshaped
return accounting instead of raw episode returns, no pairing-matrix/role-swap evaluation,
and a gradient-update budget with no 30M-env-step path. The fixed revision: build a
parallel standard path (official partial observation → recurrent TD ensemble Q → primitive
actions), train each seed self-contained with a within-run self-play partner pool (the same
partner-formation class as Fictitious Co-Play, still a single TD loss), and evaluate with a
pairing-matrix driver producing official SP/XP records; the legacy option/CE/featurizer/
scripted-partner stack is retained only as the mechanism-instrument and ablation line.
Details: `idea-stage/refine-logs/EXPERIMENT_PLAN.md` §9.

The parallel standard path was then implemented statically on 2026-07-11. It adds the
delivery-indicator adapter argument and slot-neutral stepping; exact Simple/Wide
environment files; a convolutional local-observation encoder feeding the existing
recurrent ensemble value network over six primitive actions; seed-contained self-play
pool formation followed by Path C training; and a separate checkpoint-pairing evaluator
that persists raw 400-step episode returns and summarizes 10 self-play plus 90 directed
cross-play pairings. The smoke configuration is explicitly marked as ineligible for
scientific readout and requires the CUDA JAX backend. No local or remote code was run,
and the working tree has not been committed, so this implementation is not yet a
software compatibility result.

R010 passed remotely on 2026-07-11 using GPU 0 and the CUDA 12 JAX wrapper. The
targeted regression contains 56 passed tests with no failures or skips. The smoke
artifact reports `jax_backend=gpu`, device `cuda:0`, two independent training seeds,
2 self-play pairings, 2 directed cross-play pairings, 2 episodes per pairing and all
8 raw-return rows. Each seed completed 1,600 environment steps, split evenly between
self-play pool formation and Path C training. The run also corrected a static shape
assumption: with the delivery indicator enabled, Simple observations are 5×5×39 and
Wide observations are 5×5×43. The final smoke remains ineligible for scientific
readout; its zero returns are not evidence about performance.

After the same-day static review fixed the probe gate and batch hot paths, the evidence
was re-bound remotely on 2026-07-12 to commit `c70bf03...`. The expanded targeted suite
passes 63 tests, including step-unroll versus sequence-forward equivalence and partner
sub-batching versus the full-batch reference. A fresh R010 smoke again completed two
training seeds and all 8 evaluation rows under CUDA JAX. This closes the software
re-binding item.

The R020 formal configuration was frozen statically on 2026-07-12 (no runs): per seed
30,000,000 environment steps split as a 10M self-play partner-pool phase plus a 20M
Path C phase (both counted, so the per-seed total stays within the official per-run
budget); four equally spaced pool snapshots; the deliverable policy trains from scratch
against the pool (warm-starting from the self-play endpoint would lock in a single
convention — the failure mode this benchmark measures); probe enabled at disagreement
threshold 0.02; 250 parallel environments (divides every phase, snapshot and evaluation
boundary); value-method defaults for optimizer, replay and target-network cadence.
Carriers: `configs/path_c_standard_formal_{simple,wide}.yaml` (5×5×39 and 5×5×43
observations); rationale in `EXPERIMENT_PLAN.md` §10. Launch is two-batch: the first
authorized run is `test_time_simple` seed 101 alone, whose read-back fills the
GPU-hour budget row and checks late-curve stability (Type-A only) before the remaining
9 simple and 10 wide seeds are authorized.

The user authorized only the first R020 run on 2026-07-12. Test Time Simple seed 101
completed on remote GPU 0 with config SHA-256 `8557c064...750b47`, code commit
`c70bf03...`, former PID 2866846 and log
`.codex_remote_validation/path_c_r020_simple_seed101_c70bf03_20260712.log`.
The final manifest confirms 30M joint environment steps and 75,000 joint 400-step
episodes: 10M/25,000 for self-play-pool formation and 20M/50,000 for Path C. All four
partner snapshots and the final checkpoint exist; no partial trajectories were discarded.
Elapsed time was about 4.14 GPU hours at about 2,012 joint environment steps per second.
The aggregate losses are finite and the log contains no runtime error, but the trainer did
not persist a time-resolved loss curve or raw episode return. Therefore this run establishes
completion and throughput, not late-training stability, learned task competence, or a
scientific result. No other seed is authorized.

The existing seed-101 checkpoint was retained and evaluated directly on 2026-07-12; it was
not retrained or overwritten. In 500 self-play episodes of 400 steps each, every raw episode
return was exactly zero and neither player position triggered a probe. This is a complete
single-checkpoint calibration readout, not the published ten-seed result. Checkpoint health
inspection found no non-finite parameters, and all four partner snapshots plus the final
model share the same fixed-prior state. The zero returns therefore cannot be dismissed as a
corrupt checkpoint or numerical failure, but self-play alone cannot distinguish a convention
mismatch from failure to coordinate with the partners used during training.

The standard trainer now records one observation-only metric row every 100,000 environment
steps for future formal seeds: temporal-difference losses, raw step rewards, completed-episode
returns, gradient norms, parameter-group health, exploration and probes. The remote CUDA JAX
regression passes 66 tests, and a fresh R010 smoke produced two metric rows per seed plus
plots. Its eight evaluation rows retained the exact pre-change SHA-256, confirming that the
recording path did not change that fixed smoke behavior. Seed 101's historical curves remain
unrecoverable and are not fabricated.

Three legacy modules remain deliberately `planned`. The static revision now includes a
cross-fitted ecological return estimator that excludes the target episode outcome,
a content-addressed secondary-profile recomputation path, a restorable OCV2 adapter,
restorable partner controllers, and a split-manifest-bound numeric seed chain from
collection through evaluation. Exact posterior inference now binds the registered
generation controllers, enumerates hidden option choices over the complete primitive
action and public-state path, and uses the same option distribution as generation. The
current unresolved work is different from the old registry wording: evaluate the existing
final policy against its four within-run partner snapshots in both player positions, using
raw returns, before spending compute on additional seeds. Only Path C itself must be trained
for the main table; the internal recurrent and belief systems are ablations. This project is
therefore not yet benchmark-ready.

The instrument correction is binding: a single saved simulator state estimates only
the response law conditional on that realized hidden state. Every outer replicate must
instead draw the complete hidden state `U=(theta, execution_state)` independently from
the posterior given the registered history. `M` counts those complete-state draws;
`L_inner` only estimates future continuation noise. Exact enumeration supplies the
posterior from which complete states are sampled; an analytically weighted mixture is
secondary unless it receives a matching concentration proof. Exact mode cannot prune
positive posterior mass.

The legacy response kernel uses the finite vocabulary from `ResponseSummarySpecV1`; its
history-based agents share the instrument-only `EgoEvidenceSpecV1`; and train, design, calibration and
locked-audit roles are group-disjoint. Replay uses original random keys, while audit
forks use fresh named random streams. An identical immutable snapshot and fork
coordinate must reproduce the same output without mutating the source snapshot.

---

## 3. Decisive results — claims ↔ experiments ↔ status

The old five-claim table is superseded by Path C. It remains historically useful,
but none of its claims has recorded support suitable for paper writing.

Path C's key object is `W_C`: the public-context residual value quotient, meaning
the smallest partner abstraction that changes control after public state is
already known. The primary retained code is the normalized advantage decision code;
subtracting a learned public-state baseline is only a secondary diagnostic. `F_C`
means a value-irrelevant identity or style fingerprint that may predict raw behavior
but must not explain the retained value representation.
All claim-level readouts are Type-B (cross-model + human acquittal required; see
[OPERATING_CONSTRAINTS.md](OPERATING_CONSTRAINTS.md) §4).

| Required result | Question | Status |
|---|---|---|
| Software conformance | Do sequence, evidence, response, posterior, random-key and artifact semantics match their frozen versions? | 🟡 TEST TIME SIMPLE FROZEN PATH PASSED 62 TARGET TESTS AND FULL ARTIFACT AUDIT; R015 V2 REMAINS AN UNRUN OPTIONAL SUPPORT LINE |
| Published-protocol compatibility | Can Path C produce 10 independent policies and the official `test_time_simple`/`test_time_wide` SP/XP, all-pairing and role-swapped records? | 🟡 TEST TIME SIMPLE COMPLETE: 10 INDEPENDENT UNITS AND FIVE 100-PAIRING MATRICES; TEST TIME WIDE NOT RUN |
| Standard benchmark performance | What are Path C's mean XP episode returns on both Test Time layouts relative to the published FCP references `6±29` and `23±40`? | 🟡 TEST TIME SIMPLE RAW SP/XP SUMMARY COMPLETE; SCIENTIFIC READOUT DISABLED AND HUMAN ADJUDICATION REQUIRED; TEST TIME WIDE NOT RUN |
| Probe ablation | Under the same Path C checkpoint and interaction cost, does value-directed probing outperform random and no probe? | 🟡 TEST TIME SIMPLE FORMAL MATRICES COMPLETE; PROJECT COMPARISON RECORDED SEPARATELY; NO SCIENTIFIC CLAIM ADJUDICATED |
| Instrument and mechanism validity | Do the belief-kernel, positive/null and fingerprint checks support the scoped representation interpretation? | ⬜ NOT RUN; DOES NOT BLOCK STANDARD XP PERFORMANCE |

Source of truth for run status: [EXPERIMENT_TRACKER.md](idea-stage/refine-logs/EXPERIMENT_TRACKER.md)
(execution checklist) + `docs/status/EXPERIMENT_LOG.md` (results record — active;
ARIS convention, see §6).

---

## 4. Readiness gates (what unlocks each downstream phase)

| Phase / skill | Locked until… |
|---------------|---------------|
| Path C Phase A — static build and audit | Unlocked for static maintenance only: file edits, code inspection, configuration drafting, and preregistration drafting. No local tests or runs. |
| Path C standard benchmark smoke | ✅ R010 passed remotely and post-review evidence was re-bound to `c70bf03...` on 2026-07-12; smoke artifacts are Type-A only. |
| R015 独立设计数据 | ⏸ 2026-07-19 起降级为可选支持性测量，不再排在模型实现之前。第一版 80 条历史和六候选的 `filter_design_precision_infeasible` 结果作为历史证据保留。第二版端到端代码已静态实现但尚未远端核查或运行；没有第二版过滤器或规划分支数选择，没有冻结清单或试点结果。 |
| 开发诊断迭代（`run_kind: development`，开发期训练与读数） | ✅ Unlocked：按 [PATH_C_MODULE_DESIGN.md](idea-stage/refine-logs/PATH_C_MODULE_DESIGN.md) §7 的完整模型实现推进；远端执行，逐次或成批用户授权；从产物回读有效数据预算；读数只作开发诊断，永不进入主张表（[OPERATING_CONSTRAINTS.md](OPERATING_CONSTRAINTS.md) §6.7）。 |
| Path C main training and XP evaluation | 第四版首开发单元已完成，但槽没有分化，十单元正式训练保持未启动。第三版家族级伙伴池 Test Time Simple 已完成 79 项目标测试、10 个训练单元、五个标准矩阵、回应屏蔽对照和最终审计；其统一驱动仍在隔离运行 Test Time Wide，Wide 最终审计尚未完成。 |
| Path C mechanism evaluation | The relevant posterior/reset and instrument checks have recorded evidence. These checks gate mechanism wording only. |
| `/auto-review-loop` (W2) | ≥1 decisive result supported by cross-model verdict |
| `/paper-writing` (W3) | `NARRATIVE_REPORT.md` exists + main claims supported |

Do not start a locked phase. Crossing a gate requires recorded evidence, not inference.

---

## 5. Method invariants (the load-bearing constraints)

These are *correctness* constraints of ARIS-Bellman. Violating them invalidates the
science, not just the run. Full lists: [OvercookedV2_plan.md](artifacts/OvercookedV2_plan.md)
§21 (22 "what not to do"), §15 (preflight); proposal §11 (non-claims), §12 (impl alignment).

Core set — checked mechanically by [`.aris/tools/aris_bellman_fidelity_gate.py`](.aris/tools/aris_bellman_fidelity_gate.py). Earlier bound revisions have archived passing evidence; the current uncommitted R015 and second-partner-family revision has not run this check and must not inherit the old result. Per [OPERATING_CONSTRAINTS.md](OPERATING_CONSTRAINTS.md) §7 this is a **pre-claim / pre-deploy audit**: require a current passing record before reading a claim or deploying changed method code, not before every run start.

- **决策分数不是通用互信息。** 当前提案主分数是 `J_use−V_mask`；广义
  Jensen–Shannon 散度只作通用回应信息基线，旧归一化 advantage 分歧只作历史消融。
- **有限原型代理不是正式分数。** `finite_prototype_two_action_surrogate` 没有任务转移 `x`、
  完整隐藏执行状态或回应使用/屏蔽延续分支，只能作非科学诊断；正式路径必须拒绝它。
- **共享骨干只接收 critic 的时序差分价值梯度。** 系统整体不是单一损失：actor、伙伴回应
  模型和伙伴条件价值头可以各自训练，但都必须在进入共享骨干前截断梯度；伙伴编号只用于
  训练路由，不能进入测试时信息。
- **共享历史合同。** 主方法和官方观测历史消融只使用
  [R015 规范 §1.1](idea-stage/refine-logs/PATH_C_OPPORTUNITY_AUDIT_SPEC.md) 定义并由
  `ego_evidence_contract` 制品绑定的 R015 专用主体可见历史；旧 `EgoEvidenceSpecV1`
  含评估器记录的伙伴原始动作，只属于历史机制仪器。身份、机制和风格不能进入正式信息集。
- **主要终点优先。** 唯一主要终点是 `test_time_simple` 上决策导向探查相对无探查的平均
  XP 原始回合回报。`test_time_wide`、held-out 伙伴族和 Hanabi 共同承担 H3；SP、SP−XP
  gap 与探查预算曲线都是伴随或支持读数。
- **Complete-state outer draws.** Each belief-kernel outer replicate independently
  samples the complete hidden state. Inner continuations do not increase the outer
  sample count.
- **Finite canonical response.** The theorem-level kernel uses one token from the
  frozen `ResponseSummarySpecV1`; structured multi-label outputs are secondary.
- **Instrument cost is scoped.** The static audit-cost artifact applies to belief-kernel
  instrument rollouts. It does not gate standard Test Time training or XP evaluation.
- **Decision order（2026-07-19 用户裁决，取代旧 "Decision order is fixed" 不变量）。** 完整模型
  优先：先按提案实现完整模型链（[PATH_C_MODULE_DESIGN.md](idea-stage/refine-logs/PATH_C_MODULE_DESIGN.md)
  §7），再通过开发诊断运行迭代（[OPERATING_CONSTRAINTS.md](OPERATING_CONSTRAINTS.md) §6.7）；
  确认性冻结与预注册推迟到正式十单元实验之前。R015 回到提案 §2.3 的原始定位——可选的
  支持性测量；它不决定主方法是否值得训练，不排在模型实现之前。方法冻结后仍先跑 Simple
  性能比较，再做机制分析，机制仪器只用于约束机制措辞。
- **Summaries are recomputed.** Standard SP/XP statistics are rebuilt from raw episode,
  seed-pair and player-position records. Instrument alpha/beta is separately rebuilt
  from outer-cluster token cells.
- **Dataset roles are derived.** Version-3 collection accepts a frozen split group,
  derives its role from `SplitManifestV1`, and the evaluator parses Parquet rows to
  recompute episode, transition and group coverage rather than trusting manifest totals.
- **CE is preprocessing.** Never estimate CE inside the training loop.
- **Reward-scale consistency** across preflight, CE local returns, and training target.
- **Articulation-point bottlenecks**, not `degree ≤ 2`. This is a pre-claim audit for
  the Exp-4 support-graph claim — check it before reading that claim, not as a start gate.
- **Factor deletion removes 3 things**: latent state + evidence route + action
  relevance. This is a pre-ablation audit — run it before the Exp-3 value-sufficiency
  ablation and before reading its claim, not as a start gate.
- **No oracle labels in deployable agents.** Identity, mechanism, style and true value
  classes never enter deployable evidence or training. Synthetic registry truth is
  allowed only for instrument controls, value-class labels and explicitly separated
  oracle diagnostics; it is not a deployable benchmark input.

Preflight is this dashboard recorded layout-validity start condition. Other allowed start conditions are
defined only in [OPERATING_CONSTRAINTS.md](OPERATING_CONSTRAINTS.md) §7.2; this dashboard does not
repeat or extend that rule.

---

## 6. Artifact map & known divergences

| Artifact | Location | Notes |
|----------|----------|-------|
| Final proposal | `idea-stage/refine-logs/FINAL_PROPOSAL.md` | ✅ |
| Experiment plan | `idea-stage/refine-logs/EXPERIMENT_PLAN.md` | ✅ |
| Experiment tracker | `idea-stage/refine-logs/EXPERIMENT_TRACKER.md` | ✅ (status checklist) |
| Experiment log (results) | `docs/status/EXPERIMENT_LOG.md` | ✅ active |
| Path C proposal | `idea-stage/refine-logs/PATH_C_PROPOSAL.md` | ✅ active current line |
| Path C execution plan | `idea-stage/refine-logs/PATH_C_EXECUTION_PLAN.md` | ✅ active current plan |
| Path C preregistration | `idea-stage/refine-logs/PATH_C_PREREGISTRATION.md` | 🟡 third-version template; numeric and semantic hashes not frozen |
| Path C module design | `idea-stage/refine-logs/PATH_C_MODULE_DESIGN.md` | 🟡 §7.8 记录第四版价值商信念条件 Bellman 控制器；首开发单元已有远端证据，但槽分化失败，正式十单元未启动 |
| Path C model implementation spec | `idea-stage/refine-logs/PATH_C_MODEL_IMPLEMENTATION_SPEC.md` | ✅ 第四版完整接口、公式、训练、四模式评估和回应屏蔽规定见 §14；实现位于 `src/path_c/vqbc/` 与 OvercookedV2 对接层 |
| Path C module registry | `experiments/overcooked_v2/configs/module_registry.yaml` | 🟡 `PATH_C_VQBC_V4_CHAIN` 已由远端 41 项目标测试支持为 `tested`；仍没有新 commit SHA，且没有模块进入 `frozen` |
| Migration plan | `artifacts/OvercookedV2_plan.md` | ✅ |

**Divergence to reconcile (low priority):** ARIS convention is `refine-logs/` at
project root (sibling of `idea-stage/`); this project nests it as
`idea-stage/refine-logs/`. Downstream skills that hardcode `refine-logs/…` may not
find these. Decide: move, or symlink, or pin the path in `.aris/config.json`.

### Path C module traceability

The machine-readable source is `experiments/overcooked_v2/configs/module_registry.yaml`.
The P1–P5 code object is bound to commit
`fc647fb00255c5bfc58253a38fa145cb8864afd7`; its remote CUDA JAX run passed 290
tests with no skips and archived a JUnit report hash. Modules with complete registered
test coverage are therefore `tested`. C2, C5 and D1 retain their historical `planned`
labels because the registry has not yet been migrated to the public-benchmark protocol.
Nothing is `frozen`.

The registry identifiers `C2_PRIMARY_RETURN_BUDGET_AUC` and
`C5_ACTING_BASELINE_BENCHMARK` describe the superseded internal-baseline plan.
Their `planned` status no longer means that five self-built baselines must be trained
before the public benchmark. A later code revision must either rename them or scope
them explicitly to ablations, and add the standard Test Time SP/XP evaluator.

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
| PATH_C_VQBC_V4_CHAIN | tested |
| D1_ARTIFACT_CONTRACT | planned |
| D2_CONFORMANCE_TEST_DEFINITIONS | tested |
<!-- PATH_C_MODULE_TRACEABILITY:END -->

---

## 7. Entry points

- Execution rules → [OPERATING_CONSTRAINTS.md](OPERATING_CONSTRAINTS.md)
- Agent coordination / Codex scope → [AGENTS.md](AGENTS.md)
- Remote contract → [CUSTOMER.md](CUSTOMER.md)
- Project posture (machine-readable) → [`.aris/config.json`](.aris/config.json)

---

## 8. Historical Root-Cause Repair Status — 2026-07-02

This section records the closed repair line that preceded Path C. It is not the
current stage. Historical OvercookedV2 results remain diagnostic only under
`FINDINGS_LEDGER` P1/P2/P3/P4/P5/S17/S20/NEW-2/NEW-3/NEW-4. Later blind-test and
Link-A readouts closed the asymmetric-layout + role-conditioned v2 partner line
as a positive-paper path;
their negative results now motivate Path C.

| Claim / artifact | current status | blocking / dependency IDs | allowed interpretation |
|---|---|---|---|
| Oracle-free factor evidence | static repair complete, remote pending | P1, NEW-G3, I10 | clean-test candidate after diff review |
| Accumulated belief / failure traces | static repair complete, ablation pending | P4, S1-S3, I11 | mechanism claim pending ablation |
| CE support graph claims | static repair complete, CE probe pending | P3, S8-S11, D4-D5, I12 | support-relative only until CE probe |
| Black-box main-method objective | static repair complete, objective gate pending | P5, I13 | no role/terminal-policy main-claim evidence |
| Formal eval / checkpoint path | static repair complete, gated rerun pending | S17, S20, NEW-2, I14-I18 | may be used for preregistered rerun after diff review |
| `role_conditioned_v2_candidate` | BENCHMARK-CANDIDATE | P2, W1-W7, D1-D5 | inspect/certify only; not approved benchmark |

Dashboard rule (master quarantine-release): no result may move a claim from pending to supported/refuted — and no historical OvercookedV2 headline result may be read as validation or refutation — until the relevant ledger IDs above are either closed by remote verification or explicitly waived by Type-B human decision with reason. The former standalone "Core ZSC ARIS-vs-baseline claim" and "Historical headline OvercookedV2 results" rows fold into this rule plus the section intro (which already records "the core claim is not validated or refuted" and "historical results remain diagnostic only under FINDINGS_LEDGER P1/P2/P3/P4/P5/S17/S20/NEW-2/NEW-3/NEW-4"); the core claim's dependency IDs (P1, P3, P4, P5, S17, S20, NEW-2) are distributed across the mechanism rows above.

---

## 已归档 / 已合并（2026-07-08，依据 docs/status/GOVERNANCE_CUTLIST.md）

治理精简处置留档。以下门已从在架门面移除，仅在此记录来龙去脉；载重残留（若有）已注明去向。

- **归档 · §4 `/result-to-claim` 解锁门**：原为「某个实验已产出真实 `EXPERIMENT_LOG.md` 结果文件才解锁 `/result-to-claim`」。归档理由：重言——没有结果文件本就无从做 claim，此门未挡过任何有记录的失败。载重残留「读数须从产物回读、不得凭 config 意图」已由数据充分性纪律（[OPERATING_CONSTRAINTS.md](OPERATING_CONSTRAINTS.md) §6）覆盖。
- **归档 · §4 rebuttal / resubmit / camera-ready 解锁门**：原为「收到外部评审 / 录用通知才解锁」。归档理由：逻辑前提，不护任何失败模式——没有外部评审本就无从反驳。相关执行边界仍由 [OPERATING_CONSTRAINTS.md](OPERATING_CONSTRAINTS.md) §1 保。
- **合并 · §5 「Preflight is a hard gate」静态不变量**：折进 §4 的 Formal Exp 1–5 启动条件（同源同义，对应 S27——无效布局污染 12000 行 replay，事后无法补救）。其他允许的启动条件只以 [OPERATING_CONSTRAINTS.md](OPERATING_CONSTRAINTS.md) §7.2 为准；`--preflight_path` 细节见其 §2。
- **合并 · §8 表「Historical headline OvercookedV2 results = QUARANTINED」行**：折进 §8 主隔离规则与段首说明。历史结果仅诊断、既不证实也不证伪的口径保留在段首句与主隔离规则中（依赖 ID P1/P2/P3/P4/P5/S17/S20/NEW-2/NEW-3/NEW-4）。
- **合并 · §8 表「Core ZSC ARIS-vs-baseline claim = testable only after remote gates」行**：折进 §8 主隔离规则（claim 表↔不变量↔fidelity 门三处重述塌成一处）。「核心 claim 尚无科学结论」的口径保留在段首句；其依赖 ID（P1、P3、P4、P5、S17、S20、NEW-2）已分布在各机制行。
