# Path C VQBC V4.1：潜在 Bellman 混合与持久价值商根因修复

**日期：** 2026-07-26
**状态：** 远端 50 项目标测试、首开发单元复跑和控制路径追踪完成；预测延续最大化与真实 KL 约束执行不一致，十单元正式训练继续关闭
**运行权限：** `scientific_readout_allowed: false`
**模型 schema：** `path_c_model_v4_1`

## 1. 修复对象

V4 首个开发单元稳定完成 1,228,800 个环境步，但八个无标签价值槽始终保持均匀责任度，所有状态只有一个活动价值商，槽后验始终为均匀分布，四种部署模式成对逐 seed 相同，回应屏蔽三分支效应为零。该结果表明执行策略已正确读取后验，但产生后验的潜在价值混合没有形成。

代码审查确认失败来自同一闭锁结构：

1. `episode_responsibilities` 对 400 个时间步和两个 Q 估计器取平均，显著压缩了 episode 级槽间证据；
2. 八个槽只通过 slot embedding 区分，却共享同一 Q 解码器，bootstrap mask 无法建立真正独立的 Bellman 假设；
3. 责任度只读取各槽的自洽 TD 误差，未读取实际 response、reward 和 next-control consequence 的解释能力；
4. response code target 又依赖已经塌缩的责任度，形成无启动源的循环依赖；
5. 价值商将不确定区间重叠解释为等价，即 `distance <= radius_i + radius_j + tolerance`，使 uncertainty 越大越容易合并；
6. class-level Bayes 更新后将 posterior 均匀写回类内槽，永久删除了未来上下文仍可能需要的类内证据；
7. 一个 rollout 的责任度被 4×50 个单 lane 更新重复使用，且 target 只在整个 rollout 后 Polyak 一次；
8. response contrast 使用严格 `S>0`，将 float32 舍入误差当作真实机制触发。

这些问题必须作为一个统一潜在混合内核修复；仅调整 temperature、prior scale、训练步数或 quotient tolerance 不能解决机制闭锁。

## 2. V4.1 统一控制与训练对象

V4.1 保留 V4 已经正确的两项设计：

- official recurrent policy 作为冻结 reference policy；
- posterior 通过 KL-regularized Bellman control 直接进入每一步正常动作。

潜在学习内核改为：

```text
官方 ego 历史
  -> TD-only recurrent representation
  -> 独立 twin dueling Bellman experts
  -> target/OOB episode joint evidence E-step
  -> persistent slot posterior
  -> conservative value-quotient control view
  -> J_use / J_mask / unified execution policy
  -> observed response
  -> slot-level Bayesian update
```

价值商只作为当前控制等价视图。它不再替代 slot posterior，也不再把类内 posterior 重置为均匀分布。

## 3. 代码修订

### 3.1 独立 Bellman experts

`src/path_c/vqbc/model.py`

- 删除 slot embedding + shared `slot_hidden`/`advantages` 解码器；
- 每个 slot、每个 twin 拥有独立 hidden、value 和 advantage 参数；
- fixed randomized prior 同样按 slot 和 twin 独立，且保持 stop-gradient；
- official critic hidden/value 参数机械复制到每个 learned expert，保留任务能力初始化；
- outcome model 为每个 slot 提供独立 response、reward 和 response-conditioned next-Q 输出层；
- backbone 仍只接收 Bellman/action-value 梯度。

### 3.2 联合 episode 责任度

`src/path_c/vqbc/objectives.py`

责任度改为完整 episode 上的控制证据和：

\[
E_{e,m}=\sum_t \left[\ell_{TD}(t,m)+\ell_y(t,m)+\ell_r(t,m)+\ell_{Q'}(t,m)\right].
\]

随后：

\[
q_{e,m}=\operatorname{stopgrad}\,\operatorname{softmax}(-E_{e,m}/\tau).
\]

具体修订包括：

- 时间维从 `mean` 改为 `sum`；
- response NLL、reward NLL、next-Q NLL 进入 E-step；
- responsibility 和 energy 均停止梯度；
- episode-level bootstrap mask 同时约束 E-step、Bellman loss 和 outcome loss；
- 不可用槽被置为 `-inf`，空行保留一个 keyed fallback slot；
- differentiable batch 仍不包含 partner identity、seed、family 或 checkpoint metadata。

### 3.3 解除 response-code 循环依赖

`src/path_c/vqbc/codebook.py`

- response signature target 不再使用当前 episode responsibility；
- target twin 先做 robust mean，再对 latent experts 作 permutation-invariant 平均；
- codebook target 因而独立于当前 E-step assignment，避免“槽分化依赖 code，而 code 又依赖槽分化”的闭环。

### 3.4 持久 slot posterior

`src/path_c/vqbc/quotient.py`、`rollout.py`、OvercookedV2 runtime/evaluator

- Bayes 更新直接在 slot 层执行：

\[
b_{t+1}(m)\propto b_t(m)p_m(y_t\mid z_t,a_t).
\]

- class view中的 response/reward/next-Q 使用当前 slot posterior 的条件加权；
- class 聚合不修改内部 slot belief；
- 旧 `quotient_bayes_update` 失败关闭，防止任何调用重新启用 class-to-uniform-slot projection；
- `posterior_use` 和 `prior_only` 现在只有在模型确实没有学到 slot-conditional evidence 时才会相同。

### 3.5 保守价值等价认证

两个槽只有在控制签名差异的上置信界已经足够小时才合并：

\[
\|\hat A_i-\hat A_j\|_\infty+r_i+r_j\le\epsilon_{eq}.
\]

V4 使用的区间重叠规则已经删除。uncertainty 保留不同假设，不再促进合并。当前 hard quotient 主要用于控制诊断；执行价值和 Bayesian filtering 保持 slot-level，以避免尚未证明 Bellman closure 时误删未来证据。

### 3.6 交替 E/M 更新与 target 语义

`src/path_c/vqbc/training.py`、`experiments/overcooked_v2/model_dock/vqbc_runtime.py`

- 每个 rollout 只采样一次 episode bootstrap mask；
- 每个 epoch 使用当前 target model 重新计算 E-step；
- 每个 epoch只遍历一次不重复的完整-lane minibatch partition；
- 开发配置由 32 lanes / 50 minibatches 改为 32 lanes / 8 minibatches；
- formal 保持 250 lanes / 50 minibatches；
- target parameters 在每个 optimizer minibatch 后执行 Polyak update；
- `slot_count=8` 的内部硬编码已移除；
- diagnostics 新增 responsibility energy margin、energy spread 和 bootstrap active fraction。

### 3.7 数值有效触发

回应屏蔽触发改为 scale-aware float32 tolerance。只有信息净值超过对应 J 值量级上的 256 ULP 才被视为正触发。V4 中 `1e-6` 到 `1e-5` 的数值残差不再启动三分支实验。

## 4. 不兼容性与恢复边界

V4.1 有意拒绝 V4 checkpoint 和 config：

- config schema：`path_c_model_v4_1`；
- checkpoint manifest：`path_c_flax_checkpoint_v3`；
- checkpoint metadata：`path_c_model_checkpoint_metadata_v3`；
- train-state hash namespace和model-weight hash namespace均已更新；
- 旧 `slot_embedding_dim`、`response_embedding_dim` 配置字段被拒绝；
- V4 checkpoint不能用于 V4.1 resume或deployment。

这避免了将共享槽解码器的旧参数静默加载进独立专家架构。

## 5. 本地验证

本地环境包含 JAX、PyYAML 和 pytest，但缺少 Flax、Optax 与 JaxMARL。因此本次验证分为：

- 纯 JAX 数学、配置和评估合同测试；
- 全部修改文件的 Python bytecode 编译；
- 静态参数路径、schema和调用点检查；
- 未执行的远端依赖测试明确保留为后续开发门。

当前本地结果：

- 四个 VQBC 目标测试文件：35 项通过、2 项跳过；`test_path_c_vqbc_training.py` 因缺少 Flax 整体跳过，另 1 项因缺少 JaxMARL 跳过；
- `python -m compileall`：通过；
- `git diff --check`：通过；
- 本地 pytest 日志 SHA-256：`5cf06cf29429e557cbcfbee3ce4eda708399487adb3f520f8eb28ea708942e9b`；
- 本地 JUnit XML SHA-256：`b4d59e75acb9fd9ae795d90486f8a2eb6245bb69b366d61a9d2eb2abd0a08688`；
- 完整 Flax/Optax model、optimizer、checkpoint和真实 JaxMARL rollout测试：本地未运行。

## 6. 远端复跑结果与裁决

同一 seed-100 开发单元已在注册远端环境完成。四个目标测试文件为 50 项通过、0 失败、0
跳过；训练、四模式评估和回应屏蔽对照分别回读 1,228,800、800,000 和 600,000 个有效环境
步。复跑前登记的八项检查结果如下：

1. **通过。** 最终责任度能量间隔均值为 1103.261，第 5 和第 95 百分位数为 529.204 和
   1656.389，显著大于数值噪声。
2. **通过。** 最终后验熵均值为 0.080809，不再是八槽均匀分布的 `log(8)`。
3. **通过。** 最近 rollout 的 12,800 个状态全部形成 8 个经当前等价规则认证的控制类。
4. **部分通过。** 代码测试证明槽后验跨当前视图持久保存；开发评估又证明后验使用会改变
   正常动作。不过本轮状态始终为 8 个控制类，没有提供“先合并、后拆分”的运行轨迹证据。
5. **通过。** `posterior_use` 与 `prior_only` 在 500 个匹配 seed 中有 452 个回合的参考动作
   偏离数不同，376 个回合的回应码计数不同。
6. **不通过。** 500 个回合的预测信息净值范围为 0.139714 至 0.278620，但
   `A2-use-A2-mask` 逐行、十个等频分箱及最高分箱效应全部为 0。
7. **描述性通过。** 后验使用、固定先验和参考策略的平均回报分别为 168.92、169.04 和
   168.84；后验使用的训练期 Kullback–Leibler 散度（KL 散度）均值为 0.004618，低于登记的
   每步 0.02 目标。这里不作科学结论。
8. **未执行。** 当前没有已实现并登记的仅供审计的公共上下文价值签名路线。第 6 项已经足以
   关闭正式训练，本轮没有为取得额外描述性证据而扩大运行范围。

本轮证明第四版的槽对称性根因已经修复，且后验确实参与正常动作；但它没有证明实际回应被
控制器因果使用。`PATH_C_VQBC_V4_1_CHAIN: tested` 只表示远端软件测试已有通过证据，不表示
机制验收。十单元正式训练继续关闭。随后完成的第 0 步实际控制路径追踪见 §7。

## 7. 第 0 步回应屏蔽控制路径追踪

后续远端追踪使用相同最终 checkpoint，追加 25,664 个有效环境步且不更新参数。结果逐层排除
了两类接线故障：实际回应码并非恒定，`mask_response` 也确实保留回应前信念。32 个第 0 步
样本中，使用回应后的槽信念与屏蔽分支的 L1 距离均值为 0.367；实际回应在八槽上的对数似然
极差均值为 1.522。

差异在执行策略处被大幅压缩。回应后下一步 `J_use` 的最大绝对变化均值为 2.364，但六个动作
共同变化占了绝大部分；扣除共同变化量后，动作相关变化的最大绝对值均值为 0.059，动作间极差
均值为 0.088。真实执行分布的总变差距离均值仅为 0.000741，32 个样本的下一步采样动作和
最大概率动作均没有变化。完整 400 步双分支追踪中，只有 3/32 个回合最终出现左侧动作差异，
右侧动作、逐步奖励和最终回报仍全部相同。

代码路径给出了原因：`bellman_control_values` 在 `J_use/J_mask` 内使用无约束的 `max_u`，假定
回应后可以直接选取预测最优动作；实际 `regularized_policy` 则把价值残差加在冻结参考策略的
对数概率上，并受 KL 散度温度约束。第 0 步执行分布赋予所采样动作的概率均值为 0.939，且
先前评估记录的 KL 散度很小，因此内部最大化认为可利用的控制切换通常不会被真实执行器采用。
触发还记录 `max_a S(a)`，但 A2 分支
执行从 `J_use` 分布采样的动作；32 个样本中只有 6 个执行了最大 `S(a)` 的动作。由于所有被
执行动作的 `S(a)` 仍为正，该记录错位只放大分箱解释问题，不是主要失效原因。

第 0 步、完整分支和触发评分追踪的 SHA-256 依次为
`495d3698bcf1c18f770203c9beda51a7c233ca50a583c355218d8a4d75d83e18`、
`fbeba85f20d102b4a17167e6f7b3c66ac62e3d27b834dd5c8d52b97cfaec04d2` 和
`c148993f0c7ca67bbfaa6162196f377b2b8c0a35c724133ca385651a20e63612`。根因已收窄为回应价值
延续算子与真实 KL 约束执行分布不一致；下一步应先统一这两个对象，不再追加训练预算。
