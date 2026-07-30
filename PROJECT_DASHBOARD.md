# Path C 项目状态

最后更新：2026-07-29。

工作流程依据：[`docs/RESEARCH_PIPELINE.md`](docs/RESEARCH_PIPELINE.md)。当前处于其
阶段 1（现象存在性测量）：同 checkpoint 污染已定位，既有注册回应通道的真实回报读出
为 INCONCLUSIVE；跨独立训练运行的兼容性机会尚未测量。

## 当前实现

- 方法版本：`path_c_v4_4_retrace_calibrated_control_r1`。
- 状态：seed-100 开发训练和评估、冻结 checkpoint 的反事实任务价值审计、403,200 条 continuation 的 artifact-only 真实回报读出以及官方策略库固定伙伴交叉矩阵均已完成；远端 78 项全量测试（含新增读出模块的 9 项）通过。
- 活跃源码：`src/path_c/`。
- OvercookedV2 接入：`experiments/overcooked_v2/official_adapter.py`。
- 唯一入口：`python -m experiments.overcooked_v2.path_c`。
- 布局配置：`path_c_simple.yaml` 与 `path_c_wide.yaml`，schema version 4。
- 十单元正式训练：关闭。

## 官方策略库固定伙伴交叉矩阵

五个官方最终 checkpoint 已在 Test Time Simple 中分别作为固定角色主体，与原四个冻结伙伴
完成五乘四只读矩阵。每格 500 个回合；完整矩阵为 10,000 个回合和 4,000,000 个环境步，
其中新执行 8,000 个回合和 3,200,000 个环境步。没有训练或 checkpoint 更新。

五个主体跨四伙伴的平均原始回报依次为：Self-Play seed 100 `+19.82`、seed 101 `−7.34`、
seed 102 `−2.47`、Other-Play seed 201 `−8.11`、seed 202 `−4.73`。矩阵内部同时存在超过
+100 的强配对和低于 −90 的弱配对，说明固定伙伴读数强烈依赖具体主体—伙伴组合。

逐伙伴事后最优均值为 `139.47`，但四个最优格恰好都是主体与伙伴共享同一 checkpoint 的
对角格。逐列删除对应对角格后，最优值变为 `102.56、167.88、−96.84、−94.32`，均值
回到 `19.82`，与最佳固定主体完全相同。因此 `119.65` 不是已测得的可泛化兼容性机会，
而是当前面板中的 checkpoint 身份重合效应。

完整原始行已独立重算，20 个格各有 500 个唯一回合，同一伙伴的五个主体使用相同 episode
seed，交付计数非负，旧训练和面板产物未改写。完整结果见
[`PATH_C_V4_4_COUNTERFACTUAL_AND_LIBRARY_MATRIX_REPORT.md`](docs/status/PATH_C_V4_4_COUNTERFACTUAL_AND_LIBRARY_MATRIX_REPORT.md)；
逐项原始结果见
[`PATH_C_V4_4_SEED100_DEVELOPMENT_RESULTS.md`](docs/status/PATH_C_V4_4_SEED100_DEVELOPMENT_RESULTS.md#10-官方策略库与固定伙伴的交叉矩阵)。
全部新产物清单 SHA-256 为
`fe11b52601b55926d1e25d9c74b351211d1ce141ccfd93afde0899031e080164`。

该矩阵是官方策略库对熟悉伙伴池的固定角色开发诊断，不是十训练单元标准自我配对与跨策略
配对矩阵，不能作为官方 Table 2 指标。`scientific_readout_allowed: false`，十单元正式训练仍关闭。

## 冻结 checkpoint 反事实任务价值审计裁决

V4.4 第 1,228,800 环境步 checkpoint 已完成只读审计。47 个自我配对触发点和 178 个固定
伙伴触发点均使用 128 组共同随机数重放，完整保存 403,200 条 continuation。审计实际执行
101,149,120 个环境步；原训练、评估、固定伙伴产物和 checkpoint 均未改写。

主要结果是：预测下界分数覆盖率只有 29.33%，低于登记的 95%；预测与经验动作排序的平均
Spearman 等级相关系数为 +0.01287；高分一半的经验回应前收益为 −0.01687，低分一半为
+0.00527。自我配对能力仍为 168.48，但当前预测器没有把更有价值的触发状态排到更高位置。

八项正式训练解除条件中，动作价值重复估计、严格正的排序相关、多个固定伙伴出现正的重复
回应前收益以及能力保持四项满足；其中排序相关接近 0，不能解释为强相关。预测下界覆盖率和
高分组收益两项未满足；Other-Play seed 202 的负效应根因及槽语义两项证据不足。正式十单元
训练继续关闭。

完整结果、口径和证据哈希见
[`PATH_C_V4_4_SEED100_DEVELOPMENT_RESULTS.md`](docs/status/PATH_C_V4_4_SEED100_DEVELOPMENT_RESULTS.md#9-冻结-checkpoint-的反事实任务价值审计)。
最终远端清单摘要为 `bf4416fa2a323769882f4ad19a6f9937e627c1e44ba15db97707e756d18d25ac`。

## V4.4 artifact-only 真实回报读出裁决

现有 403,200 条 continuation 已按 `(trigger_id, replica_index)` 的版本化 SHA-256 顺序固定
拆为 64 个 fit 与 64 个 evaluation replica。225 个触发状态中，178 个来自四个固定伙伴
（37、60、17、64），47 个来自自配对。主读出器只使用 agent-0 官方 post-observation、
触发时刻与动作、冻结合法历史 carry、上一动作与团队回报，以及 use 分支可见的 response
code；伙伴身份、slot、模型 Q/LCB、隐藏环境状态与未来字段不进入模型。

四折 LOPO、伙伴等权、10,000 次触发状态级分层 bootstrap 的原始剩余回报结果为：

- 主历史读出 `L_probe = −0.3597 [−0.9326, +0.0852]`；
- split-replica oracle `L_oracle = −0.0135 [−0.1649, +0.1399]`；
- 当前触发动作回应效应 `tau_response = +0.0499 [−0.1795, +0.2715]`。

oracle UCB 大于 0，未达到“当前候选无机会”的 NO-GO；oracle 与 probe 的 LCB 都不大于
0，也未达到第二类 NO-GO 或 GO。显著伙伴方向冲突为 false，登记裁决因此是
**INCONCLUSIVE**。同伙伴乐观点估计 `+0.0837` 而 LOPO 点估计为负，提示跨伙伴支持可能
不足；但 oracle 自身也围绕 0 未决，不能宣称机会已经存在。

这些边界条件于当前四伙伴面板和固定的 64 个 evaluation continuation replica；bootstrap
重采样触发状态并重拟合整条管线，但没有再重采样 continuation replica，因此不是伙伴总体
或完整 Monte Carlo 不确定性区间。

完整口径、伙伴分解、输入哈希与证据边界见
[`PATH_C_V4_4_ARTIFACT_VALUE_READOUT_REPORT.md`](docs/status/PATH_C_V4_4_ARTIFACT_VALUE_READOUT_REPORT.md)。
正式运行与独立复跑的 9 个确定性产物逐字节一致，输入与实现哈希在两次运行前后均未改变；
独立实验完整性复核为 PASS。V4.4 继续冻结，十单元正式训练和 V4.5 均不启动。

## V4.4 seed-100 开发裁决

V4.4 在 Test Time Simple 完成 1,228,800 个训练环境步、3,072 个训练回合和 96 次更新。四种部署模式各完成 500 个匹配自我配对回合；回应屏蔽完成 500 个匹配回合；四个固定伙伴与四种部署模式共完成 8,000 个回合。全部产物仍为开发诊断，`scientific_readout_allowed: false`。

完整的原始行重算、训练过程、固定伙伴结果和证据哈希见
[`PATH_C_V4_4_SEED100_DEVELOPMENT_RESULTS.md`](docs/status/PATH_C_V4_4_SEED100_DEVELOPMENT_RESULTS.md)。

训练后的自我配对能力保持在参考策略量级：`posterior_use`、`prior_only`、`reference_only` 和 `generic_response_information` 的平均原始回报分别为 168.48、168.76、168.20 和 168.20。

保守下界（lower-confidence bound，LCB）尚未转化为任务收益：

- 500 个回应屏蔽回合中触发 47 个，触发率为 9.4%；
- 47 个触发回合中只有 12 个出现动作轨迹差异，没有回合出现奖励差异；
- 回应收益、任务成本和净收益在全部回合、触发回合及按实际执行动作 LCB 重算的十个等频分箱中均为 0；
- 触发时实际执行动作 LCB 全部严格高于数值容差，范围为 `4.33e-5` 至 `1.05e-2`，因此结果不是由触发条件未执行造成的。

固定伙伴面板中的 `posterior_use - prior_only` 匹配回报差依次为：

- Self-Play seed 101：+2.12，回合级标准误 0.98；
- Self-Play seed 102：+0.12，回合级标准误 0.24；
- Other-Play seed 201：+2.64，回合级标准误 1.73；
- Other-Play seed 202：−2.32，回合级标准误 2.01。

该方向没有在四个伙伴间重复；只有第一个伙伴的描述性常态近似区间不跨 0。一个训练单元和回合级区间不能作为独立训练重复。正式十单元门因此继续关闭。

当前自报回应屏蔽摘要按策略平均 LCB 分箱，而实际触发使用执行动作 LCB。原始行完整保留，本次裁决使用后者独立重算；后续应修正摘要字段名称与分箱输入，但不得追溯改写本次原始产物。

## V4.3 开发裁决

V4.3 完成 49 项远端测试、1,228,800 环境步训练、2,000 个四模式评估回合和 500 组回应屏蔽对照。它修复了“回应只能产生动作无关数值平移”的问题：230/500 个对照回合产生动作轨迹差异，触发点 next-policy TV 均值为 0.0545。

科学目标仍未成立：

- `posterior_use`、`prior_only`、`reference_only`、`generic_response_information` 平均回报分别为 168.24、167.96、168.20、168.44；
- 回应屏蔽效应为 −0.08；
- 498/500 回合最终回报完全相同；
- 2 个回合因回应使用而下降，0 个回合改善；
- 训练伙伴回报未随训练改善。

该结果是开发诊断，`scientific_readout_allowed: false`。

## V4.4 修订对象

```text
full-episode Retrace
    + episode-coherent expert exploration
    + outcome uncertainty calibration
    + LCB policy-mediated trigger
    + fixed-partner development panel
```

具体变化：

- 一步 Bellman target 替换为完整回合 off-policy Retrace；
- 训练时每个回合固定一个 latent expert，用其动作价值构造一致的支持 policy；
- 保留少量均匀 floor，保证六动作均有非零支持；
- use/mask next-Q 均值与方差共同构造保守控制值；
- 回应屏蔽触发与分箱使用实际执行动作的 policy-gain LCB，而非未校准均值；
- 新增单 checkpoint × 固定 SP/OP 伙伴 panel；
- TD-only responsibility、持续 posterior、KL 执行器和完整 E-step 记录保持不变。

## 语义主干

```text
layout config + run kind
    → registered budget
    → one training unit
    → one TrainState
    → one run_identity.json
    → one population or partner panel
    → standard / response-contrast / panel evaluation
```

## 下一步

V4.4 修复循环已按流水线阶段 3 停机判据（五轮修订任务效应 0→0→0→−0.08→0）停机；
其败因由反事实审计（预测覆盖率 29.33%、排序相关 +0.013）和官方策略库交叉矩阵
（表观 `119.65` 完全由同 checkpoint 对角格承载）共同承载；新增真实回报读出又表明当前
触发/候选分布的 oracle 和跨伙伴 readout 都未决。继续修改当前估计器或启动新 RL 没有依据。

主线下一测量是**run-disjoint 兼容性机会审计**：模式侧和伙伴侧使用互不共享 checkpoint、
训练运行或直接共同训练关系的独立样本；每种预定义伙伴类型包含多个独立实例；在平衡任务
seed 上以配对完整回报比较伙伴无关最佳模式、开发伙伴冻结的类型—模式映射及确认性伙伴上
的前瞻迁移值。只有这一层先证明跨运行机会，才测官方历史能否在有效切换窗口内恢复它。

已有 20 配对 × 500 回合不能承担这一主检验：它的标签由含同 checkpoint 对角格的矩阵产生，
即使按 episode 切分，探针仍可能只学习 checkpoint 指纹。它仅保留为身份敏感性描述，不再
作为进入策略组合的门。

若未来仍把当前注册回应通道作为旁支问题，只增加独立触发状态与伙伴支持、继续做真实配对
回报读出，直到 oracle 区间收窄；在此之前不训练控制器。十单元正式训练继续关闭。
