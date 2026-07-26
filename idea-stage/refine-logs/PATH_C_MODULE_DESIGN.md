# Path C V4.2 活跃模块设计

最后更新：2026-07-26。状态：`implemented`，尚未运行重构后的测试或实验。

## 1. 范围

本文件只描述当前 V4.2。旧 PyTorch、R015、Path C V1–V4.1、旧标准训练链、`src/aris_bellman` 和 toy factor game 已从工作树删除；它们仍可由 Git 提交 `f0ba51c` 和历史实验记录追溯。

V4.2 保留以下研究对象：

- 官方策略作为动作先验，官方循环特征允许通过 Bellman 损失更新；
- 无标签伙伴槽与整回合持续后验；
- 两个独立的 dueling Q 估计器和停止梯度的随机先验；
- 回应编码器和学习式回应码本；
- 由动作相对价值定义的动态动作价值等价类别；
- 回应、即时原始回报、回应使用延续值和回应屏蔽延续值模型；
- 相对官方策略的 Kullback–Leibler 散度正则执行策略和两个温度；
- Bellman、结果模型、回应编码器、码本和责任分配训练；
- `posterior_use`、`prior_only`、`reference_only`、`generic_response_information` 四种部署模式；
- 参照动作、回应屏蔽、回应使用三个匹配分支。

## 2. 六个模块

| 模块 | 唯一职责 |
|---|---|
| `src/path_c/model.py` | 接收官方循环特征，计算双重 Q、随机先验、回应分布、回报和两种延续值，并编码回应 |
| `src/path_c/method.py` | 更新槽后验，形成动作价值等价类别，计算回应使用与屏蔽值、执行策略、温度和回应码本 |
| `src/path_c/training.py` | 定义训练批次、损失、责任分配、两个优化器和目标网络更新 |
| `src/path_c/runner.py` | 推进环境、主体、伙伴和随机数，生成完整训练轨迹 |
| `src/path_c/evaluation.py` | 定义标准配对矩阵、四种部署模式、三分支效应和统计 |
| `src/path_c/storage.py` | 读取两份布局配置，使用 Orbax 保存与恢复 checkpoint，并无损保存完整记录 |

OvercookedV2 的外部接口集中在 `experiments/overcooked_v2/official_adapter.py`。三个公开操作集中在 `experiments/overcooked_v2/path_c.py`，不再保留重复入口。

## 3. 数学对象及运行消费者

### 3.1 槽后验

令伙伴槽为 \(m\in\{1,\ldots,M\}\)，历史为 \(h_t\)，主体动作是 \(a_t\)，登记回应码是 \(y_t\)。后验按

\[
\log b_{t+1}(m)
=\log b_t(m)+\log p(y_t\mid m,h_t,a_t)-\log Z_t
\]

更新。`method.slot_bayes_update` 实现该式；`runner.py` 在每次真实转移后调用；`evaluation` 的回应使用分支使用同一更新。槽成员身份不进入模型输入。

### 3.2 双重动作价值与随机先验

两个估计器分别计算

\[
Q_{e,m}(z_t,b_t,a)=V_{e,m}(z_t,b_t)+A_{e,m}(z_t,b_t,a)
-\frac{1}{|\mathcal A|}\sum_{a'}A_{e,m}(z_t,b_t,a'),
\]

并加上停止梯度的随机先验：

\[
\widetilde Q_{e,m}=Q_{e,m}^{\text{learned}}+\beta Q_{e,m}^{\text{prior}}.
\]

`model.py` 计算该值；`training.py` 的 Bellman 损失是官方循环特征的唯一训练信号，结果模型和回应编码器读取停止梯度特征。

### 3.3 动作价值等价类别

每个槽的动作相对价值签名和两个估计器的不确定半径为

\[
s_m(a)=\tfrac12\sum_e\left(\widetilde Q_{e,m}(a)-\max_{a'}\widetilde Q_{e,m}(a')\right),
\]

\[
r_m=\tfrac12\max_a\left|s_{0,m}(a)-s_{1,m}(a)\right|.
\]

只有当一组槽内所有成对距离加半径不超过登记容差时，`method.complete_link_value_class_ids` 才把它们放入同一动作价值等价类别。该类别只用于当前控制统计，不覆盖持续槽后验。

### 3.4 回应码

回应编码器把 \((o_t,a_t,o_{t+1},d_t)\) 映射到签名。最近码本中心给出非终止回应码，终止转移使用单独终止码。目标签名来自目标 Q 的平均动作相对价值，并在生成目标时使用均匀槽信念：

\[
g_t=\frac{1}{2M}\sum_{e,m}
\left(\widetilde Q_{e,m}(a)-\max_{a'}\widetilde Q_{e,m}(a')\right).
\]

`model.encode_response_codes` 编码真实回应；`method.update_codebook` 更新码本；`training.prepare_frozen_assignments` 在一次更新开始前冻结目标码和责任分配。

### 3.5 共同物理权重下的回应使用与屏蔽

结果模型输出 \(p_m(y\mid h,a)\)、即时原始回报 \(\hat r_m(h,a)\)，以及两个估计器的回应使用和回应屏蔽延续值。两分支必须使用同一个物理联合权重

\[
w(m,y\mid h,a)=b(m)p_m(y\mid h,a).
\]

因此

\[
J^e_{\text{use}}(a)=
\sum_m b(m)\hat r_m(a)
+\gamma\sum_{m,y}w(m,y\mid a)C^e_{\text{use}}(m,a,y),
\]

\[
J^e_{\text{mask}}(a)=
\sum_m b(m)\hat r_m(a)
+\gamma\sum_{m,y}w(m,y\mid a)C^e_{\text{mask}}(m,a,y).
\]

控制值取两个估计器的较小值。回应价值、最佳屏蔽值和动作净值分别为

\[
R(a)=J_{\text{use}}(a)-J_{\text{mask}}(a),\qquad
V_{\text{mask}}=\max_a J_{\text{mask}}(a),
\]

\[
N(a)=J_{\text{use}}(a)-V_{\text{mask}}.
\]

`method.bellman_control_values` 是这组公式的唯一实现；`model.model_forward` 消费它，训练目标和评估不另写一份公式。

### 3.6 相对官方策略的执行分布

给定官方动作对数概率 \(\ell_0(a)\) 和控制分数 \(S(a)\)，执行分布为

\[
\pi_S(a)=\operatorname{softmax}\left(\ell_0(a)+S(a)/\alpha\right).
\]

`posterior_use` 使用 \(J_{\text{use}}\)；`prior_only` 先把槽后验重置为均匀分布；`reference_only` 直接使用官方分布；`generic_response_information` 使用回应分布的 Jensen–Shannon 信息量。`method.regularized_policy` 计算执行分布和相对官方策略的 Kullback–Leibler 散度。

两个温度分别服务于回应价值策略和通用回应信息策略，并按

\[
\log\alpha\leftarrow
\operatorname{clip}\bigl(\log\alpha+\eta(\overline{D}_{KL}-\delta)\bigr)
\]

更新。`method.update_log_temperature` 是唯一标量公式实现，`training.update_policy_temperatures` 在每个完整轨迹更新后更新 `PolicyState` 中的两个温度。

### 3.7 训练目标

Bellman 目标使用目标网络和实际目标执行分布：

\[
y_t=r_t+\gamma(1-d_t)
\sum_a\pi_{t+1}(a)\min_e \widetilde Q^{\text{target}}_{e,m}(a).
\]

整回合责任按每槽的 Bellman、回应、回报和两种延续值证据之和形成，随后停止梯度。训练分为两个优化器：

- Bellman 优化器更新官方循环网络和学习式 Q，随机先验不更新；
- 结果优化器更新回应、回报、延续值和回应编码器，不更新官方循环网络和 Q。

目标网络使用 Polyak 更新。伙伴参数在轨迹开始时冻结，伙伴分支不接收梯度。

### 3.8 三分支效应

回应屏蔽评估在首次正的策略净效应处建立：

- `A1`：从回应屏蔽策略采样动作并屏蔽该回应；
- `A2-mask`：执行回应使用策略选中的同一动作，但屏蔽该回应；
- `A2-use`：执行同一动作并正常使用回应。

定义

\[
\Delta_{response}=A2_{use}-A2_{mask},\quad
\Delta_{cost}=A1-A2_{mask},\quad
\Delta_{net}=A2_{use}-A1.
\]

必须满足 \(\Delta_{net}=\Delta_{response}-\Delta_{cost}\)。`evaluation.effect_components` 从完整分支回报重算该恒等式。

## 4. 集中状态

| 状态 | 字段用途 |
|---|---|
| `PolicyState` | 官方参考循环状态、可训练循环状态、槽后验、上一动作、上一团队回报、回合开始标记和两个温度；上一动作与上一回报只用于形成 V4.2 heads 的历史特征 |
| `RunnerState` | 环境、主体状态、伙伴状态、伙伴成员、席位、随机数和完整回合累计量 |
| `TrainState` | 在线参数、目标参数、两个优化器状态、码本、运行状态、随机数和实际训练计数 |
| `TransitionBatch` | 一次完整轨迹及所有训练损失实际读取的张量 |

效应分解、策略差异和动作价值类别是同一次前向的结果，不作为第二份可变状态保存。

## 5. 官方接口

锁定依赖为 JaxMARL 0.1.0 和官方实验仓库提交
`5ce1707cf31c1c115e6f6ba96db7bc9cc80a850e`。两者均在项目依赖文件中直接登记。重构前已阅读以下官方实现及其调用关系：

- `ppo/models/rnn.py`：官方循环 actor–critic 网络；
- `ppo/models/model.py`：网络构造和循环状态初始化；
- `ppo/ippo.py`：Independent Proximal Policy Optimization 训练；
- `ppo/policy.py`：正式评估策略接口；
- `ppo/utils/store.py`：官方 Orbax checkpoint 保存；
- `eval/policy.py` 与 `eval/rollout.py`：策略配对和标准回合；
- JaxMARL `overcooked_v2/overcooked.py` 与 `common.py`：动作、观测、`reset`、`step_env`、终止和事件语义。

同时阅读了同一锁定提交中围绕网络、训练、checkpoint、评估和
OvercookedV2 环境的测试目录与配置调用示例。它们只用于确定公开接口的
真实输入输出和边界行为；项目代码不复制这些测试的内部实现，也不依赖
源码字符串或私有调用顺序。

当前适配器直接使用这些公开入口。官方包内部的训练脚本使用脚本目录相对导入，`official_adapter._official_symbol` 只在导入一个公开符号期间临时提供该目录；它不修改官方模块或替换运行行为。官方一步网络调用返回下一循环状态、动作分布和价值；一步调用的下一循环状态同时是该时刻提供给 V4.2 heads 的循环特征。上一动作嵌入和上一团队回报经零初始化线性层加到该特征上，因此新增 heads 初始化时不改变官方动作分布；这两个输入随后只由 Bellman 优化器学习。序列特征通过对同一公开一步调用作设备端扫描获得。

外部包没有公开两类本项目需要的数据：JaxMARL 自动重置前的终止观测和逐次正确、错误交付记录，以及带上一团队回报和每步方法诊断的策略接口。`VectorEnvironment.step_with_keys` 直接包裹公开 `step_env`，统一完成一次自动重置；交付记录使用锁定版本 `process_interact` 的公开状态和动作语义逐主体计算，不从观察频道反推。标准评估的设备端循环保留完整方法状态和决策行。另有一个不参与生产评估的记录适配函数，官方集成测试使用相同策略、环境和随机数，直接比较它与官方 `get_rollout` 的总回报。

Simple 与 Wide 使用同一适配器；两份配置只改变布局。不存在复制的官方卷积网络、参数叶路径映射、源码字符串检查或 Wide 运行时补丁。

## 6. 运行、恢复和输出

公开入口只有：

```text
python -m experiments.overcooked_v2.path_c upstream ...
python -m experiments.overcooked_v2.path_c train ...
python -m experiments.overcooked_v2.path_c evaluate ...
```

训练恢复读取指定输出目录的最新完整 Orbax step。运行元数据只记录完整解析配置、Git 提交、依赖版本、seed、实际环境步、更新数和完整回合数。

训练按每次更新分别保存完整决策、完整回合和指标 JSONL 文件。标准评估保存每个回合的 Parquet 行以及每一步两侧的完整决策，包括指示器触发、回应码、信念、策略偏离和效应分解。回应屏蔽评估为每个匹配回合保存一行，其中同时包含三个分支的回报、正确与错误交付、指示器触发和分支分歧位置；没有触发时仍保留三个相等分支。大数组以分块无损压缩 NumPy 文件保存。控制台只显示摘要，但标准输出和标准错误完整写入日志并打印数据路径。

公共函数不通过首尾切片或字符串上限改变返回语义。训练公式中的时间移位切片只对齐 \(t\) 与 \(t+1\)，不删除已登记记录。

未采集的训练临时量只有每个小批次的完整梯度、前向激活、责任能量矩阵、bootstrap mask 和目标网络中间张量。这些量只服务当次参数更新，不生成缩短版或伪装成完整数据的替代文件；需要研究它们时，必须先把采集范围加入明确的实验设计。checkpoint 中仍完整保存恢复训练所需的参数、优化器、码本、运行状态和实际计数。

## 7. 标准评估

每个群体严格包含 10 个独立训练策略。每种部署模式执行 10 个对角自我配对和 90 个有向跨策略配对，每配对 500 个 400 步回合。对应配对和回合在四种部署模式中使用相同环境 seed。

官方摘要分别从自我配对和跨策略配对的配对均值计算总体均值与跨配对标准差，并报告二者差。回应屏蔽对照独立保存和汇总，不进入标准矩阵。

## 8. 测试原则与当前状态

六个测试文件分别验证数学、模型、官方集成、训练、评估和数据完整往返。期望值来自字面量手算、官方公开接口或人工构造事实；测试不读取生产源码，不断言内部变量名，也不使用生产摘要生成期望值。

截至 2026-07-26，本次重构只完成静态实现和检索，没有运行 Python、测试、训练、评估或远端命令，因此不能标为 `tested`。旧远端实验不受本次工作树修改影响；完成其审计后，新实现仍须在新目录取得单独运行授权。
