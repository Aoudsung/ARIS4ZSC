# Experiment Tracker — Path C 公开基准对齐版

self-play（SP）指同一次独立训练内的策略彼此协作；cross-play（XP）指不同独立训练运行
得到的策略在测试时配对。

**当前阶段：** 2026-07-29 裁决——V4.4 已冻结。既有 403,200 条 continuation 的
64/64 split-replica、四折 LOPO 真实回报读出为 INCONCLUSIVE，不启动 V4.5、新 seed 或
超参数实验。项目回到现象存在性测量：先用模式侧与伙伴侧训练运行完全分离的面板检验
跨运行兼容性机会，再决定是否研究历史可恢复性。下表保留旧里程碑作为执行历史；当前证据
以 [PROJECT_DASHBOARD.md](../../PROJECT_DASHBOARD.md) 和
[PATH_C_PROPOSAL.md](PATH_C_PROPOSAL.md) 为准。
**执行边界：** 本表不授权运行。所有项目执行只能在用户明确授权后于 `zsc-customer` 远端进行。

| Run ID | 里程碑 | 目的 | 系统或制品 | 决定性读数 | 优先级 | 状态 | 备注 |
|---|---|---|---|---|---|---|---|
| R001 | 历史软件工作 | 固定 P1–P5 修订代码 | git commit `fc647fb…` | 提交存在 | 必须 | 完成 | 历史软件证据 |
| R002 | 历史软件工作 | 验证第三版软件 | GPU 5 CUDA JAX 完整测试 | 290 passed、0 skipped | 必须 | 完成 | CUDA 12.9、JAX 0.4.38 |
| R003 | 历史软件工作 | 检查布局 option 兼容性 | 全布局语义扫描与单布局简化 | 339 passed；语义产物 `1a548d08…` | 必须 | 完成但不再决定主布局 | `asymm_advantages` 只证明旧接口可运行；公开基准对齐后不再是主布局 |
| R004 | M0 | 固定公开比较对象 | ICLR 2025 OvercookedV2 Test Time 表格、官方配置与仓库状态 | Test Time Simple FCP XP `6±29`；Wide `23±40`；官方仓库无 release/checkpoint | 必须 | 完成 | 数字只用于同协议外部参照 |
| R005 | M0 | 检查当前代码与标准 XP 协议的差距 | `OCV2Adapter`、策略训练接口、角色交换、XP 聚合器 | 差距清单与最小代码修改范围 | 必须 | 完成（2026-07-11） | 六项差距与修订方案见 EXPERIMENT_PLAN.md §9 与 EXPERIMENT_LOG 同日 R005 条目；纯静态，未运行代码 |
| R010 | M1 | 检查标准公开协议接线 | `test_time_simple` 极小远端 smoke | Type-A pass/fail、SP/XP 行结构、双方角色均覆盖 | 必须 | 完成并再次复核（2026-07-12） | CUDA JAX GPU 0；修订二后 70 项回归通过；每 seed 两行指标和曲线、共 8 行评估；数值因架构和训练回报修订而重新生成；仅软件证据 |
| R015 | M1.5 | 审计官方观测下的单次安全探查机会 | 回应屏蔽参照、探查后屏蔽注册回应通道、探查后使用该通道；两个伙伴族各两个原型 | `Delta_response`、`Delta_net` 的分布无关同时区间；可选完整信息保守上界 | 可选支持测量（2026-07-19 起） | 两族各两候选已登记，产物与正式控制器待实现 | `Delta_response` 只表示显式注册通道的增量使用效应；四个候选必须全部通过新来源核验与同一能力下限后才会一次性准入，普通负结果只否定注册控制器 |
| R016 | M1.5 | 读取冻结 V4.4 continuation 中的真实回应与后续动作价值 | 225 状态 × 128 replica；64 fit / 64 evaluation；四伙伴 LOPO；10,000 bootstrap | `L_oracle`、`L_probe`、`tau_response` 的登记单侧区间 | 必须 | 完成（2026-07-29），INCONCLUSIVE | oracle `−0.0135 [−0.1649,+0.1399]`；history probe `−0.3597 [−0.9326,+0.0852]`；两次正式读出的 9 个确定性产物逐字节一致，完整性复核 PASS；V4.4 继续冻结 |
| R020 | M2 | 训练 Path C 主方法群体 | Path C × 10 独立 seed；骨干形成与适应预算分开报告 | 每 seed 有效环境步、episode、晚期 return | 必须 | simple seed 101 完成；seed 102 骨干完成（2026-07-13） | seed 102 完成 30M 步、75,000 局和 19,200 次更新，累计 33,784 次正确交付；联合准入和后续适应尚未启动 |
| R030 | M3 | 计算 Test Time Simple 主结果 | 10 seed、90 个有向 XP 配对、每配对 500 个 400 步 episode | 平均 XP return、标准差、训练单元节点重采样 95% CI、SP、SP−XP gap | 必须 | 阻塞 | 与同布局已发表 SP/State-Augmented/Other-Play/FCP 数字并列 |
| R031 | M3 | 在 Test Time Wide 复现 | 同 R030 的冻结方法和配对规模 | `Delta_Wide` 的 98.5% 单侧下界 | H3 必须 | 阻塞 | H3 三项之一；预先指定的第二布局，不是独立领域 |
| R032 | M3 | 评估 held-out 伙伴族 | 至少两个未参与训练支持、校准、R015 或方法选择的伙伴生成机制 | `Delta_heldout` 的 98.5% 单侧下界和族别区间 | H3 必须 | 阻塞 | 与官方同方法跨 seed 矩阵分表报告；支持内伙伴不能计入 held-out |
| R033 | M3 | 在 Hanabi 验证同一信息价值对象 | 合法任务动作、官方可见回应、决策导向/通用信息/随机安全/无探查及通道屏蔽诊断 | `Delta_Hanabi` 的 98.5% 单侧下界；注册回应通道效应单列 | H3 必须 | 阻塞 | Wide 不能替代第二领域；运行前另立 Hanabi 领域规范并冻结 H3 区间 |
| R040 | M4 | 检验主动探针组件 | value-directed、random probe、no probe | 同 checkpoint 下的 return 与成本差异 | 主结果后必须 | 阻塞 | 消融实验，不称为 SOTA baseline |
| R041 | M4 | 检验记忆与表示组件 | 删除记忆、删除 residualization、必要内部上界 | XP return 与探针预算曲线 | 主结果后必须 | 阻塞 | full-history/HMM/particle 只作内部比较 |
| R042 | M4 | 检验机制解释 | 合成正例、null、信念平均干预分布、指纹泄漏 | 各自预先定义的统计区间 | 支持主张 | 阻塞 | 只约束机制措辞，不阻止 R030/R031 |
| R050 | M5 | 多 episode 伙伴适应扩展 | ICRL4AHT Track 1 | 平均 return、适应曲线、末 20−前 20 episode | 可选 | 阻塞 | 只有完整采用公开 manifest 与协议时才运行；不与 ICLR XP 混排 |

## 已撤销的旧运行

- 旧 R031/R032：训练 full-history、residualized、exact/HMM/particle 等自建主基线。
- 旧 R033/R034：生成九策略 design ledger，并从五个自建系统中选择“最强基线”。
- 旧 R040/R041：以八策略 locked ledger 的内部曲线差作为唯一主结果。

这些运行不再是论文主结果的前置条件。相关代码可保留用于消融，但不得继续驱动主实验预算。

## 当前单一步骤

当前单一步骤是设计 run-disjoint 兼容性机会审计：模式库、开发伙伴和确认性伙伴不共享
checkpoint、训练运行或直接共同训练关系，并让每个预定义类型包含多个独立实例。旧 20×500
固定角色矩阵只保留为 checkpoint 重合诊断，不再训练基于其标签的在线路由器。R016 若要
继续，只能增加独立触发状态与伙伴支持并重复真实回报读出；在 oracle 区间收窄前不训练控制器。
