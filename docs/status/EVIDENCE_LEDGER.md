# 证据台账（只追加）

每条：日期、结论、证据指针。结论被推翻时追加新条目，不改旧条目。
首批条目摘录自 legacy 源出台账 [legacy/v44/status/FINDINGS_LEDGER.md](../legacy/v44/status/FINDINGS_LEDGER.md)，原始证据细节以源出台账为准，摘录不改原文。

---

## 自 legacy 移植条目

- **2026-07-09** Link-A 隐藏性证书两轮结构性判负，旧方法线在 OvercookedV2 asymm 基底上死亡，项目转向 Path C。证据：源出台账第 1 条。

- **2026-07-14** 自写 torch 训练器同预算只得 19 分（官方 IPPO 复现约 130），统一根因为训练栈缺陷；此后正式产物一律用官方训练器。证据：源出台账第 2 条。

- **2026-07-28** V4.4 开发训练保持自我配对能力（四模式均约 168），但保守预测收益零转化：47 触发 → 12 动作差异 → 0 奖励差异；固定伙伴方向不可重复。证据：源出台账第 3 条，`legacy/v44/status/PATH_C_V4_4_SEED100_DEVELOPMENT_RESULTS.md`。

- **2026-07-29** 冻结 checkpoint 反事实审计（1.01 亿环境步、225 触发态 × 128 replica）：预测下界覆盖率 29.33%（登记 95%），预测与经验动作排序 Spearman +0.013，高分组经验收益低于低分组。V4.4 预测器不携带真实动作价值排序。证据：源出台账第 4 条。

- **2026-07-29** 官方策略库 5×4 交叉矩阵：最佳固定策略跨伙伴均值 +19.82（SP-100），逐伙伴最优选择均值约 139；差距是策略模式选择差距。证据：源出台账第 5 条，`legacy/v44/status/PATH_C_V4_4_COUNTERFACTUAL_AND_LIBRARY_MATRIX_REPORT.md`。

- **2026-07-29** 代码级根因裁决：V4.4 八个槽是同一共享执行策略的八个 critic，outcome 模型以 target 网络自身 Q 为标签（自蒸馏），学到的是"同一策略的多个自洽价值估计"而非"伙伴条件控制价值"。证据：源出台账第 6 条。

- **2026-07-29** 研究问题重述：本基底是隐藏离散语境问题，语境为伙伴惯例，单步不可读，回合级价值约 120 分；原提案在动作粒度上定价信息，而价值全部位于惯例粒度。证据：源出台账第 7 条。

- **2026-07-29** V4.4 artifact-only 真实回报读出：225 触发状态 64/64 拆分、四折 LOPO、10,000 次伙伴分层 bootstrap，oracle lift −0.0135 [−0.1649,+0.1399]，历史 probe lift −0.3597 [−0.9326,+0.0852]，区间均跨 0，登记裁决 INCONCLUSIVE。119.65 表观差距完全来自同 checkpoint 对角格。run-disjoint 兼容性机会仍未测量。证据：源出台账第 8 条，`legacy/v44/status/PATH_C_V4_4_ARTIFACT_VALUE_READOUT_REPORT.md`。

- **2026-07-30** V4.4 读出完整性复核：9 个确定性产物逐字节一致，SHA-256 清单不变，独立重算与机器摘要一致，独立 reviewer 裁决 PASS。结论不扩大证据范围。证据：源出台账第 9 条，`legacy/v44/EXPERIMENT_AUDIT.md`。

## 本台账新增条目

- **2026-08-03** 文档重构前置：工作区 72 个未提交改动提交为快照 commit `6b9f598`（agent/delta-zsc-v5 分支），作为研究方案文档重构的回滚基线。证据：`git log --oneline -2`。

- **2026-08-03** 根级孤儿文档归档：仓库根级 2148 行旧版理论文档（V5 r3，不在任何 git 仓库内）复制至 `docs/legacy/v5_r3/`，附非权威声明，原文件保留。证据：`docs/legacy/v5_r3/README.md`。

- **2026-08-03** 文档重构完成：研究层文档（RESEARCH_PROGRAM、research/ 六份、status/ 两份、PROTOCOL_INDEX）全部新建；README 导航追加。论点裁决留空，待 S1 与文献复核。证据：本次提交 diff。

- **2026-08-03** 字节级哈希锁废除：按用户决策，SHA 逐字节校验属不必要措施。已移除 `test_delta_zsc_repository.py` 中的 `test_registered_design_document_is_byte_exact` 与注册哈希常量，理论文档恢复为可正常修订（确认轨 commit 冻结后仍不得改方法）。仓库其余 8 项测试全绿。证据：本次提交 diff。

- **2026-08-03** T2/T3/T4 定理形式化完成：TV 精确式、历史样本复杂度、路由紧上界等式写入 FOUNDATIONAL §10，必要性下界修正为标准 Le Cam 二点形式，常数复核与 legacy 草案一致。证据：`docs/theory/DELTA_ZSC_FOUNDATIONAL_THEORY_AND_PROOFS.md` §10，commit `eeca646`。

- **2026-08-03** S2 受控相图验证全过：180 个参数点（服务器执行，commit `4e146bb`）上 T1 链 180/180、T2 精确式 180/180（最大残差 8.9e-16）、T3 边界 180/180，窗口外无净收益成立。理论工具验证通过，生态估计协议解锁。探索轨读数，`scientific_readout_allowed: false`。证据：服务器 `runs/exploration/s2_phase/summary.json`。

- **2026-08-03** S1 面板盘点与 pilot：服务器现存上游只有 SP 与 OP 各 10 个完整 seed（ckpt_final），SA/FCP 在 V6 树上不可用；第一版面板限定 sp/op 两类型，入台账说明。Pilot（3 seed/类型，36 配对 × 5 回合）行数与标记完整，探索性读数 Γ_compat≈69.3，污染对照 identity 均值 146.7 vs run-disjoint 14.0，复现历史污染形态。回合数太少，不构成结论。证据：服务器 `runs/exploration/s1_panel/pilot/summary.json`。

- **2026-08-03** S1 放量预算登记：全量面板取 sp/op 各 10 seed，20×20 配对共 400 格 × 500 回合 × 400 步 = 80,000,000 环境步。证据：本条目。

- **2026-08-03** S1 执行方式改多卡分片：单 GPU 串行方案实测 GPU 利用率近零且只用 1/8 卡，已停止（commit `032f93f` 给面板工具加了分片与合并支持），改为 GPU 2/4/5/6 四分片并行，产物入 `runs/exploration/s1_panel/full/shard_{0..3}`，完成后跑 `--merge-shards` 出最终汇总。证据：服务器 shard 日志与 nvidia-smi 读数。

- **2026-08-03** S1 迁移值分析完成：伙伴运行按 seed 0-4 开发、5-9 留出，全程剔除 identity 格。开发半冻结映射 sp→rnn-sp-seed-0、op→rnn-op-seed-7（开发分 78.2/88.3）；留出半上 V_transfer=−75.85，V_fix,test=+7.75（最佳单一模式 rnn-op-seed-9），迁移差 −83.60，伙伴运行级 bootstrap 区间 [−166.9,+27.9]（单元数 8）；去 identity 的留出 oracle 仍有 84.22。探索轨读数，`scientific_readout_allowed: false`。证据：服务器 `runs/exploration/s1_panel/full/transfer_summary.json`。

- **2026-08-03** 论点裁决：Θ1 判死。迁移差点估计大幅为负且区间不给出正证书，命中 THESIS_CANDIDATES 注册的 falsifier（机会为正但冻结映射在留出伙伴上不迁移），也命中 EXPERIMENT_LADDER S1 停机判据；结论收缩：机会存在于个体/运行层面（去 identity oracle 84.22），类型级惯例映射不可跨运行复用。Θ3 失去前置（S1 停机则不投入方法训练），同步出局；Θ4 依赖 V6 训练与表征分析，本轮不具备裁决条件，挂起。主论点 Θ2（信息时序定理生态验证，S2 已 180/180 通过，生态推论 P1-P6 已注册），次论点 Θ5（分解链诊断工具，本面板就是它的首个完整实例）。落选与挂起理由以本条为准。

- **2026-08-03** 论文形态收缩到 L1-L3：按 PAPER_STANDARD 主张层级，L4（方法 SOTA）随 Θ3 出局摘除；L1（分解链诊断，本次面板加 S2 相图）、L2（身份重合污染的第二次独立复现：污染对照 172.4，迁移失败）、L3（TV 匹配界与历史需求定理及生态预测）为投稿主体。标题候选与表图设计随 RESEARCH_THESIS 第二版收敛。

- **2026-08-03** E0 类型内排序一致性分析：对每个伙伴 run 按 19 个非 identity 模式 run 的回报排序，同类型伙伴对（各 45 对）的 Kendall tau 均值 sp=0.040、op=0.084，合并 0.062（std 0.34，范围 −0.54 到 +0.80）；跨类型基线 100 对均值 0.065（std 0.29）。类型内一致性与跨类型基线持平，支持"无可迁移惯例结构"而非"结构存在但太弱"，补强 L2 第三层机制证据。探索轨读数，`scientific_readout_allowed: false`。证据：服务器 `runs/exploration/s1_panel/full/structure_summary.json`。

- **2026-08-03** P4 预测重定义：按用户决策，生态预测测量工具改为插入式 Bayes 路由器（零 RL，模板匹配后验加面板均值表选模式）。P4 由"学习路由器学习曲线拐点步数"改为"插入式 Bayes 路由器实测回收率对理论值 Δ·TV/2"；完整注册见 `docs/research/TRAJECTORY_AND_ESTIMATION_SPEC.md`。本轮不训 Wide 上游，P6 布局级结论后置，边界入台账。

- **2026-08-03** E2 轨迹采集完成：ego = rnn-sp-seed-0，20 伙伴 × 100 回合 × 400 步，伙伴动作序列全记录，episode key 流与面板隔离（根种子偏移 1000）。证据：服务器 `runs/exploration/s2_ecology/full/trajectories.json`。

- **2026-08-03** E3 估计器首版合成验证失败：1 近邻留出分类器在已知 κ 的 Bernoulli 合成轨迹上 TV̂ 相对偏差最高 1.0（κ̂ 无偏），触发 fail-closed。按失配条款修估计器：换留出式经验 Bayes 分类器（离散特征键频数多数类），规格已同步，重验通过：κ̂ 绝对偏差 ≤0.0015，TV̂ 相对偏差 ≤8%。

- **2026-08-03** E4 生态估计完成（Simple 布局，探索轨）：类型级 κ̂=0.137，个体级 κ̂ 均值 0.099（范围 0.029 到 0.175）；类型级 TV̂(t) 曲线：t=20 约 0.55，t=200 约 0.84，t=400 约 0.90（中段有窗口噪声振荡）；Δ̂（面板代理）=27.3。经 T3 代入：类型级证据所需历史约 2ln(8)/0.137²≈220 步，在 400 步回合内；个体级均值需约 425 步，超出回合长度，弱个体（κ̂=0.029）需约 5000 步。初步形态：类型级证据在窗口内可恢复，个体级平均在窗口外；但此处的可分性是行为层面的，不等同于价值层面的可恢复性（后者由 E5 对撞检验）。另注：κ̂ 首版误用窗口 TV 口径（超限）已修正为单步边际 TV。证据：服务器 `runs/exploration/s2_ecology/ecology_estimates.json`。

- **2026-08-03** 人类裁决（后于本日论点裁决）：一、核心原则：在 OvercookedV2 benchmark 中超越 SOTA 的 performance 是唯一核心目标，所有中间过程服务于推进该目标，不设拖延性门槛；方法轨是唯一关键路径，理论/诊断轨为并行旁路、无阻塞权。二、复活 Θ3/V6：推翻本日裁决中 Θ3 的传递性门控连带出局，Θ3（V6 端到端 Bayes 协调为载体）复活为方法载体论点；S1 停机判据的效力限定为仅对 Θ1，"不再投入方法训练"不外溢到 Θ3/方法轨。复议依据：传递性门控越界＋S1 个体级读数（去 identity oracle 84.22）支持个体推断路线。三、SOTA 基线：仅以基准论文 Gessler et al.（arXiv:2503.17821）Table 2 已发表数值为准：Test Time Simple 最优 FCP 6±29，Test Time Wide 最优 FCP 23±40（10 seeds、每格 500 episodes）；此为论文作者自训 checkpoint 数值，口径偏移风险如实披露，但按本裁决不作为立项门槛，判决式仍要求 J_XP 显著高于注册基线数值，注册专页见 `docs/research/SOTA_BASELINE.md`。四、配套治理修订：晋升门精简（保留 commit 冻结与预测/基线登记，删除主论点裁决与 S1/S2 前置），S3 整级降级为 M1 式单 seed preflight（门槛值跑前入台账，失败换载体不停目标），S5 进入条件改为开发矩阵通过＋commit 冻结＋SOTA 基线已注册，论文形态恢复含 L4；本条同时作为 FORMAL_EXPERIMENT_PROTOCOL 判决处补充注册基线的修订动因登记。证据：本条与本次提交 diff。

- **2026-08-03** 接受外部六线评审结论，项目统一重构为 DEPI（Decision-Equivalent Protocol Inference）：按评审（科学问题、理论与可识别性、模型与训练动力学、伙伴分布、实验统计、创新性与文献边界六条独立审计线）判决，接受"工程契约完整但科学语义未闭合"结论，不在 V6 上做增量补丁，中心主张改为 legal-history protocol inference improves XP——在只能读取 Official 合法局部历史的条件下，学习随交互动态演化、以 ego 动作价值排序为语义的协议状态，在训练运行与算法族均留出的伙伴上产生可归因的 XP 提升；Θ2 降为支持性理论。评审全文归档于 `docs/research/REVIEW_AND_SUGGESTION_2026.md`（与源文件逐字节一致，cmp 校验通过）。S2 180/180 表述修正：该结果是按 T2/T3 假设构造的 Bernoulli 特例 closed-form 自检，属 theorem unit test 而非独立数值验证，本台账此前"主论点 Θ2 已验证"（见本日"论点裁决"与"S2 受控相图验证全过"条目）相关表述予以收回；按只追加纪律不改动既有原文，修正以本条为准。标注：本台账既有"2026-08-03 人类裁决（后于本日论点裁决）"条目已移入 `docs/status/DECISION_LOG.md`（本条为迁移指针），该条目自此仅作为资源与方向决策记录，不再作为科学论点的证据；原行保留于本台账，不删不改。证据：`docs/research/REVIEW_AND_SUGGESTION_2026.md`、`docs/status/DECISION_LOG.md`、本次提交 diff。

- **2026-08-03** 用户裁定登记（重构计划 F/C 节两项悬而未决问题）：一、SOTA 判决路径采用路径 1（固定 Official commit 重训/获取 FCP/OP/SA run-level 节点，同 episode keys 配对/双样本层级推断；已发表 Table 2 Simple 6±29、Wide 23±40 降为外部 sanity check；"显著超过 SOTA"措辞仅在配对推断支持下使用；FCP 种群重训成为判决批前置成本项）；二、latent 信念语义采用补全 Bayes 语义（为后验补生成模型、似然与 held-out 校准协议/proper scoring rule，保留 Bayes 叙事，METHOD_SPEC 必须定义生成语义与校准），不改称 context embedding。决策登记见 `docs/status/DECISION_LOG.md` D2；生效口径与执行要点见 `docs/research/STATISTICAL_PREREGISTRATION.md` 第五节（路径 1 已裁定生效、路径 2 备用）；`docs/research/SOTA_BASELINE.md` 第七节已追加生效说明。本条为裁定登记（科学程序口径），不作为任何科学论点成立的证据。证据：`docs/status/DECISION_LOG.md` D2、`docs/research/STATISTICAL_PREREGISTRATION.md`、本次提交 diff。

- **2026-08-04** 文档重构登记（重构计划 I 节落地，任务 #16，纯文档编辑）：新建权威文件 `docs/EVALUATION_SPEC.md`（伙伴划分、B0–B3 要点、路径 1 判决、Φ1–Φ6 汇总、资源六项、δ_min=20.0）与 `docs/THEORY.md`（T3 修复后样本复杂度、T2 降级与 D_V(t)、生态 κ̂ 代理口径与禁用条款、S2 theorem unit test 定性、定理适用边界表），四份权威文件至此齐备。治理冲突修复（只追加修订段）：TRACKS_AND_GOVERNANCE 晋升条件以 DEVELOPMENT_MATRIX 为准、冲突旧条款冻结，并登记探索轨读数须经正式重跑/预注册复核方可入论文的调和规则；PAPER_STANDARD 登记 S5 基线集合与路径 1 对齐（SP/OP/FCP/SA 必训，MEP/PLASTIC/PECAN/GOAT 视可得性降为外部参考）与论文中心统一为 DEPI 主张（双中心表述废止）；PROTOCOL_INDEX 登记四份权威文件层级（其余 docs/research、docs/theory 降为支撑附录）与命名统一（分支 delta-zsc-v5、方法名 DEPI 取代 V6）。本条为文档治理登记，不作为任何科学论点成立的证据。证据：本次提交 diff。

- **2026-08-04** DEPI 审查修订落地：依据《当前仍存在的问题》的 22 类缺口，完成 exact categorical filter、shared-latent joint response likelihood、task/partner 结构隔离、低频 capability、run-disjoint frozen comparator、actor decision supervision、post-update KL 控制、原子 checkpoint state、静态分层伙伴池、posterior calibration、B0–B2/G1–G4 执行链、正式统计门与 DEPI 身份迁移；B3 仍按权威规格 fail closed 为 `not_implemented`。最终本地工程复核：active source compile 通过，108 项 `test_depi_*` 全部通过，CLI/static/repository 一致性门通过。本地 `.venv` 为 Python 3.13，因此本条只是工程验证，不是注册 Python 3.10/单 CUDA 运行，也不作为 XP、校准或机制主张的科学证据。修订原因与证据：用户审查文档、本轮 working-tree diff 和本地验证输出。

- **2026-08-04** DEPI 修订后缺口复核与更正：上一条的 108 项为阶段性计数，不是最终验收数。进一步审计发现并补齐五类证据边界：M1 masked metrics；真实 bootstrap 训练、fit/evaluation replicas 分离、完整历史与 resume/deployment roundtrip 测试；CUDA preflight 的 checkpoint/deployment 往返与加载后前向；common/calibration/Official/capacity/formal-claim 产物的原始节点、身份和 SHA-256 回溯；以及 manifest-disjoint `comparator_fit`/`comparator_validation` 分区和与最终 deployment 参数指纹一致的 final-checkpoint M1 必要机制门。当前工程验收为 118 项全量测试通过，active source compile、CLI help、diff whitespace 和 active-source legacy identifier 门通过；仅有 Orbax 在本地单设备恢复时的 sharding metadata 警告。本地仍为 Python 3.13/CPU，未执行注册 Python 3.10、单 CUDA preflight、mechanical E2E、B0–B2 开发矩阵或 formal 评估；因此本条只证明代码与合同接线通过本地工程验收，不证明 XP 提升、校准通过或机制归因成立。证据：本轮 working-tree diff、`pytest -q experiments/overcooked_v2/tests`、compileall 与静态门输出。

- **2026-08-04** 《当前仍存在的问题》对应的非正式运行代码修订完成并再次验收：新增 memoryless instant-partner 控制通路与严格 R0/B0/B1/B2 嵌套；event/likelihood、exact categorical KL、固定 auxiliary transaction、奖励尺度与置信/漂移加权、分层伙伴采样、fresh-final M1、区间 calibration、可识别性与 recoverable-value 口径、raw-backed modular formal claim 均按当前权威规格接线。开发矩阵扩展为 core 加 R0/B0/B1 total-budget controls，B1-extra 直接检验同架构下 decision supervision 相对更多 PPO 的成本效率。冻结 comparator 新增 `collect-pair-comparator-source`：由固定 Official reference ego 与 lineage-disjoint fit/validation partners 在真实 simulator 中生成带 checkpoint/manifest/config/key/resource 绑定的合法历史及独立 replica continuation source，禁止再以自由 JSON 代替采集链；development evaluator 与 formal validator 逐项回溯 deployment、training run identity、config、partner sampler、resource ledger 和完整 ordered-pair raw episodes。README 身份漂移、正式 manifest、response-event 信息边界、comparator source、真实 Official reset/step/history collector、精确预算与 raw episode 完整性均有回归测试。最终本地工程验收：129 项 `test_depi_*` 全部通过，active source compile、全部新增/关键 CLI help、diff whitespace、legacy/retired identifier 门均通过；仅保留 Orbax 单设备 sharding metadata 警告。本地是 Python 3.13/CPU；未执行注册 Python 3.10、CUDA preflight、mechanical E2E、comparator 正式采集/拟合、开发矩阵或任何 formal 评估，因此本条不构成 XP、校准、机制或 SOTA 证据。证据：本轮 working-tree diff、全量 pytest/compile/CLI/static gate 输出。

- **2026-08-04** 遗留缺口代码闭合登记：方法身份升级到 v7/config 17/checkpoint 7/deployment 8；exchangeable component 新增共享 critic 的 one-hot `S_k` 路径、posterior-mixture empirical signature 直接约束、全动作 pairwise response divergence、one-hot actor 干预以及 K∈{2,4,8}、seed 0--9 的 permutation-aligned stability artifact；capability 新增四项合法窗口统计 prediction 与按 partner-run group mean 计算的跨伙伴 variance floor，保留 consistency 并新增 no-capability 消融；开发矩阵加入 deterministic-context、decision-only、Q-only、actor-only、no-separation、no-capability 六项同容量/同预算消融，并把所有主结论路径强制为十 seeds；删除旧 K-prototype/regime-label 死代码。三层执行链新增真实 Official history/continuation 到 comparator fit 集成测试、自托管单 GPU workflow_dispatch preflight，以及两个 ego、少量 fresh partners、50 episodes 的 `scientific-dry-run` 全 artifact/claim-report 排练，所有 dry-run claims 固定为 false 且 `scientific_readout_allowed=false`。本地工程验收为 139 项 `test_depi_*` 全通过，compile、CLI、diff whitespace 与 active legacy-token gates 通过；Orbax 仅有单设备恢复 sharding metadata 警告。未执行远程 CUDA、完整 scientific dry run、development matrix 或 formal runs，因此本条只证明代码/合同/测试接线，不构成任何 XP、校准、机制或 SOTA 科学证据。
