# DELTA-ZSC V5 r3 Signal Contract：完整理论、设计与正式实验规约

> 本文档是 `agent/delta-zsc-v5` 活动代码的唯一规范性设计文档。历史
> V4.4、V5 r1/r2 设计与失败运行仅是归档材料，不能覆盖本文档。本文档、
> 配置、代码、测试和部署包必须以同一提交发布。

完整的形式化背景、定理与逐步证明保留在
[`DELTA_ZSC_FOUNDATIONAL_THEORY_AND_PROOFS.md`](./DELTA_ZSC_FOUNDATIONAL_THEORY_AND_PROOFS.md)。
该附卷只提供理论基础；其中与本 r3 文档冲突的旧工程条款均被本文件替代。

## 1. 固定身份与证据边界

```text
METHOD_VERSION = delta_zsc_v5_decision_equivalent_bayes_r3_signal_contract
CONFIG_VERSION = 7
MANIFEST_VERSION = 2
OFFICIAL_PROTOCOL_VERSION = overcooked_v2_iclr2025_5ce1707_v1
OFFICIAL_SOURCE_COMMIT = 5ce1707cf31c1c115e6f6ba96db7bc9cc80a850e
```

正式环境、测试接口、随机键协议、完整 400 步原始回报以及 10×10 矩阵
均由固定 Official 提交定义。r3 可以有自己的训练机制，但测试时只能读取
当前官方局部观测、`done`、自身 recurrent carry 和 action key。伙伴身份、
生成器 code、未来轨迹、环境真值状态、伙伴参数、训练期教师和反事实分支
均不得进入部署策略。

r2 运行必须正常终止并原样归档为 `stable_signal_collapse`，显式标记：

```text
scientific_readout=false
resume_allowed=false
```

r3 不得恢复任何 r2 参数、优化器、generator、anchor 或 calibration artifact。
只有 lineage 合法的 Official 策略可作为冻结 source/reference。机械测试只证明
执行路径可达，资格状态机只证明各项预注册信号契约是否成立；二者都不构成
DELTA 优于任何基线的证据。

## 2. 科学对象

令伙伴机制为未知变量 \(\xi\)，合法交互历史为 \(H_t\)，当前任务特征为
\(X_t\)。DELTA 学习连续决策后验

\[
q_\phi(z_t\mid H_t)
=\sum_{k=1}^{K}\alpha_{t,k}
\mathcal N(\mu_{t,k},\operatorname{diag}\sigma_{t,k}^2),
\]

并用一套共享策略与两套数值独立 raw-Q heads：

\[
\pi_\theta(a\mid X_t,q_\phi),\qquad
Q^{(1)}_{\rm raw}(X_t,z,a),\quad Q^{(2)}_{\rm raw}(X_t,z,a).
\]

连续分量是同一后验的多个假设，不是伙伴类别；网络容量不随伙伴数量增长。
两个 raw-Q heads 只用于保守数值估计，不表示两类伙伴。PPO 的 shaped value
与 raw-Q 严格分离：前者估计训练用 shaped return，后者只估计 raw return。

两个伙伴仅在它们对 ego 最优响应的影响相同的意义下决策等价。真实反事实
centered action signature 定义为

\[
A_{\rm CF}(h,a;\xi)
=G^{\bar\pi_e}(h,a;\xi)
-\frac1{|\mathcal A|}\sum_{a'}G^{\bar\pi_e}(h,a';\xi),
\]

其中第一动作被强制为 \(a\)，之后始终执行同一冻结 target-policy epoch
控制器 \(\bar\pi_e\)。任何用于同一个 raw-Q 的监督必须共享这个 continuation
policy，禁止混合不同策略版本的 return 标签。

## 3. 六项信号契约与状态机

```text
BOOTSTRAP_BASE
  -> C0 -> ESTABLISH_DECISION_SUPPORT
  -> C1 -> LEARN_RAW_Q
  -> C2 -> PASSIVE_ADAPTATION
  -> C3+C4 -> GENERATOR_EXPANSION
  -> generator admission -> ACTIVE_INFORMATION
  -> C5 -> FULL_DELTA
```

契约失败不会删除或替换 seed，而是退回最高已验证安全层：

| 最高资格 | 最终部署 |
|---|---|
| C0 失败 | 精确 owner-SP source fallback |
| 仅 C0 | qualified robust base |
| C1–C2，C3/C4 失败 | qualified robust base |
| C3、C4，C5 失败 | calibrated passive conditional DELTA |
| C5 | calibrated full active DELTA |

正式 seed 0–9 均必须导出 artifact 并进入 Official 10×10 矩阵。资格失败、低分、
训练提前退回 fallback 均不得触发 seed 重跑或替换。

### 3.1 C0：Base 能力契约

每个 DELTA seed 独占一个 owner-SP source checkpoint。初始化采用合法观察、carry、
done 和 source logits 的行为蒸馏，不做参数树硬映射。Conditional residual 最后一层
严格零初始化；gate=0 时必须通过逐状态 logit、carry 和轨迹一致性检查。

使用相同伙伴和相同 episode keys 形成配对差

\[
D_{\rm comp}=R_{\rm candidate}-R_{\rm owner-SP}.
\]

均值与 CVaR20 的 one-sided 95% LCB 均须不低于
\(-\delta_{\rm delivery}\)。\(\delta_{\rm delivery}\) 从 Official 环境读取，当前为
一次正确交付的 20 分。未通过时只与 seed 独占的 qualified external partners
继续 base fine-tuning；15M ego steps 仍失败则冻结 owner-SP fallback。

### 3.2 C1：决策支持存在性契约

每个 seed 独占且 run-disjoint：generator 初始化 SP/OP/SA/FCP 各 1 run，
development/support 四机制各 4 run，calibration 四机制各 5 run。三组之间不得有
checkpoint、parent run、population 或 co-training lineage 重合。

冻结 qualified base 的 reference audit 必须同时证明：

1. between-partner signature variance 的 LCB 高于 within-partner Monte Carlo noise；
2. empirical oracle-context 相对 state-only action selection 的 evaluation lift LCB > 0；
3. 至少两个由独立 partner runs 支持的稳定最优动作区域；
4. held-out partner ordering transfer 高于随机。

C1 失败时 conditional residual、generator diversity 和 regret 均保持关闭。

### 3.3 C2：Raw-Q 可用性契约

每 16 outer updates 建立一个冻结 `TargetPolicyEpoch`：

```python
TargetPolicyEpoch(
    epoch_id,
    actor_fingerprint,
    belief_fingerprint,
    raw_q_fingerprint,
    target_params,
    started_update,
)
```

同一 epoch 内，Retrace、training anchor continuation、endpoint bootstrap 与
decision regret 都读取同一 target actor、belief 和 `min(Q1,Q2)`。epoch 切换时
同步新的 target snapshot，旧 epoch replay 退出 current raw-Q 训练池。

Recurrent Retrace 固定为：raw reward、\(\lambda=0.95\)、记录的 behavior action
probability、terminal 严格截断、chunk 末端 target expected-Q bootstrap；两个 live
heads 拟合同一个 stop-gradient conservative target。

C2 同时要求：

1. Q-selected action 相对 base action 的 evaluation lift LCB > 0；
2. 显著动作对上的 pairwise concordance LCB > 0.5；
3. raw-Q calibration error 不超过预注册 Monte Carlo error bound；
4. fit/evaluation 排序在相同 target epoch 内稳定。

C2 前 raw-Q 不得改变 actor，也不得产生非零 regret shaping。

### 3.4 C3：合法历史可恢复契约

在相同伙伴与 episode keys 上比较 oracle-context、online-context 和 state-only。
Oracle context 只由独立真实 action signatures 构成，用于诊断，永不进入正式 loss、
行为或 checkpoint 选择。Online context 只能由 Official 合法历史获得。

要求 oracle-context 与 online-context 相对 state-only 的完整回报增益 LCB 均 > 0。
只有 oracle gain 为正才报告 recovery ratio。Generator code、partner ID、future
trajectory 和 reward 均禁止进入 online path。C3 失败时退回 base。

### 3.5 C4：Conditional 控制契约

C3 前 conditional residual 只保持零耦合，模型可训练表征和 raw-Q，但不得用
conditional residual 执行动作。C3 后允许

\[
w(h)D_{\rm KL}\left(
\operatorname{softmax}(\operatorname{sg}Q_{\rm raw}/\tau)
\,\|\,\pi_c\right),
\]

其中 \(w(h)\) 仅在 raw-Q margin、双 head 一致性和 calibration 同时合格时非零。

Development partners 上必须满足：conditional-base return lift LCB > 0，
negative-transfer rate <= 5%，conditional-to-base KL <= 0.03，residual RMS <= 1.0。

### 3.6 C5：Decision regret 任务价值契约

使用同 epoch 保守 raw-Q 定义

\[
\mathcal R_{\rm dec}(X,b)
=\mathbb E_{z\sim b}\max_aQ_{\rm raw}(X,z,a)
-\max_a\mathbb E_{z\sim b}Q_{\rm raw}(X,z,a).
\]

只有以下五个子门全部通过才启用权重 0.1：

1. fit-oracle action 相对 base action 的 evaluation lift LCB > 0；
2. Q-selected action 相对 base action的 evaluation lift LCB > 0；
3. online-context 相对 state-only 完整回报 lift LCB > 0；
4. top-regret anchors 的 empirical oracle lift 相对 bottom-regret 的差值 LCB > 0；
5. full-episode paired audit 中 active shaping 相对 passive conditional 非劣。

势函数为 \(\Phi=-\mathcal R_{\rm dec}\)，训练 shaping 为

\[
F_t=0.1[\mathcal R_t-\gamma\mathcal R_{t+1}],
\]

terminal next-regret 强制为零。C5 失败部署 passive conditional。

## 4. 为什么状态机足以阻断退化闭环

下游机制仅在其全部上游前提已经由独立 raw-return audit 验证后才可进入控制：

\[
\text{能力}\Rightarrow\text{决策异质性}\Rightarrow
\text{同语义 raw-Q}\Rightarrow\text{在线可恢复}\Rightarrow
\text{条件控制收益}\Rightarrow\text{信息价值}.
\]

若任一蕴含前件不成立，状态机禁止对应后件影响部署。因此“所有模块形式上运行、
但科学信号为零”的 r2 稳定假成功状态不再可达。这个保证是控制流保证，不是
性能保证：它保证不把未经验证的模块作为有效 DELTA 机制，却不保证任何契约必然
通过。

## 5. Generator 能力流形与资格

Generator 不随机启动。本 seed 独占的 qualified SP/OP/SA/FCP 策略分别获得一个
训练期 code anchor，其 logits 被蒸馏进一个共享 code-conditioned recurrent
generator。Code 只用于训练 generator，不是部署伙伴类别，也不进入 ego online path。

初始化门要求 held-out behavior KL <= 0.05、paired mean/CVaR20 noninferiority LCB
不低于 -20、随机 code 插值不退化为 uniform policy。保存
`last_qualified_generator_params` 与匹配 optimizer state/counter。每次 generator
更新相对最近合格快照满足 KL <= 0.03；失败时参数、optimizer state 和 step counter
精确回滚，且失败参数不进入 mixture 或 snapshot。

每 outer update 采集 32 个完整 400 步 episodes。一个 episode 固定一个 code、
一版参数、carry 和 behavior log-probability；done 后严格结束，数据不得跨 episode
污染。Generator recurrent PPO 使用 Official shaped reward，4 epochs、8 environment
minibatches 及 Official PPO LR/GAE/clip/entropy。Raw return 只用于资格和报告。

BR-diversity 仅在 C1/C2 后启用。Admission 同时要求能力非劣、between-code signature
variance 高于噪声、oracle-code 相对 state-only lift LCB > 0，以及至少两个稳定不同
最优动作的 code regions。

连续 admission 次数决定伙伴混合：

| 资格历史 | current | snapshot | external |
|---|---:|---:|---:|
| 未通过 | 0 | 0 | 1.0 |
| 首次 | 0.125 | 0.125 | 0.75 |
| 连续两次 | 0.25 | 0.25 | 0.50 |
| 连续四次 | 0.50 | 0.25 | 0.25 |

资格下降时 current/snapshot 质量立即转给 external。

## 6. Training anchors、audit anchors 与 replay

### 6.1 Training anchors

每 16 updates 与 target epoch 对齐。Pilot 为 128 ordinary 与 64 matched-code
candidates，每动作 2 replicas、horizon 128；选择 24 ordinary 与 24 matched-code，
每动作 8 fit replicas、horizon 128，endpoint 用同 epoch target conservative raw-Q
bootstrap。不采 evaluation replicas。

Matched-code 必须先从同一 snapshot 分支两个 code，以冻结 probe/base policy 运行
16 步形成真实不同的合法 Official histories，再采 post-evidence signatures。相同
pre-evidence history 的 online posterior 必须完全相同，禁止 hidden-code separation。

单次最大 attempted transition 预算为

\[
128\cdot6\cdot2\cdot128
+2\cdot64\cdot6\cdot2\cdot128
+(24+2\cdot24)\cdot6\cdot8\cdot128
+2\cdot64\cdot16
=837,632.
\]

正式 457 updates 的 epoch-start 最大触发次数为 29，因此最大 training-anchor 预算为
24,291,328。实际 ledger 必须记录因 C1 资格门而真正执行的次数，不能用最大预算
替代实测值。

### 6.2 Audit anchors

只在 C0 后、15M、22.5M 和 final 四个里程碑采集。每次 32 ordinary + 32
post-evidence matched-code pairs，fit=32、evaluation=64、horizon=400，25% 状态均匀
抽样，使用独立随机域且永不进入任何 loss。

单次 attempted transitions 为

\[
(32+2\cdot32)\cdot6\cdot(32+64)\cdot400
+2\cdot32\cdot16=22,119,424,
\]

四次合计最大 88,477,696。

### 6.3 Replay 隔离

每条 replay item 绑定 target epoch ID/fingerprint、partner source/run、抽样类型、
fit return mean/standard error 与 collection step。每 epoch 容量 512；每 outer
update 用独立 raw-Q optimizer 执行 4 个 anchor minibatches。旧 epoch 数据只作审计，
不得训练 current raw-Q。

## 7. Response 信号与梯度路由

Response-to-belief 只预测可见伙伴响应：visibility BCE，含 `not-visible` 的相对位置
categorical CE，visible-mask direction CE，inventory categorical CE，以及
class-balanced interaction-change BCE。Reward/done 只允许作为 belief stop-gradient 的
诊断 heads。

每 outer update 的顺序固定为：PPO；raw-Q/anchor ×4；response ×2；generator PPO。
在 belief 参数上，若 response gradient 与 decision gradient 内积为负，执行 PCGrad
投影；投影后 response norm 不超过 decision norm。Response 不得更新 task encoder、
actor、shaped value 或 raw-Q。所有参数组分别裁剪，不使用全局 clip 混合信号。

## 8. Actor 稳定契约

Base 在 C0 前冻结 trunk，C0 后用主 PPO learning rate 的 0.1 倍更新；conditional
residual 零初始化。Lane role 只有互斥 base/conditional，不存在 teacher behavior lane。
Privileged teacher 只作 oracle 诊断。

保留 Official entropy coefficient。Conditional 只受相对 base 的非塌缩惩罚

\[
[H_{\rm base}-0.1-H_{\rm conditional}]_+,
\]

而不是固定高熵目标。同时约束 KL <= 0.03 与 residual RMS <= 1.0。PPO scan 分别
读取 base-role 与 conditional-role KL；任一超过 0.03 时余下 minibatches 为 no-op，
optimizer 与 LR counter 均不增加。

## 9. Calibration 与部署

每 seed 独占 20 个 run-disjoint calibration partners。Partner-run-block conformal
使用 current raw-Q 的真实 continuation residual 与 latent support。有效 run blocks
少于 19 时强制 abstain 到 base。

正式 Full artifact 必须调用冻结 calibrated gate：support 不足或 conditional gain
LCB 未超过 base 时执行 base，通过才执行 conditional/active。`gate=1` 只能由
`always_on_ablation=true` 显式生成，名称固定为 `DELTA-r3 always-on`，不得作为 Full。

部署包只包含 task/belief encoder、belief-set encoder、shared base/conditional actor、
shaped value、raw-Q heads、response decoder 和冻结 calibration/support artifact。
Generator、source policies、oracle context、optimizer、replay 和训练 runner 不得进入。

## 10. Official 正式评估

Simple 与 Wide 各训练 10 个固定 Official outer keys：

```python
train_keys = jax.random.split(jax.random.PRNGKey(42), 10)
```

每个布局以十个最终 artifacts 构造 10 diagonal SP 与 90 ordered off-diagonal XP
cells，每格 500 episodes，所有 cells 共享同一 500-key 向量，策略 stochastic、carry
逐 episode reset，只累加完整 400 步 `agent_0` raw return。最终 checkpoint 机械选取，
禁止用回报选择中间 checkpoint。

报告

\[
J_{SP}=\frac1{10}\sum_i\bar R_{ii},\quad
J_{XP}=\frac1{90}\sum_{i\ne j}\bar R_{ij},\quad
Gap=J_{SP}-J_{XP}.
\]

论文未注册 Official Gap 标准差公式，因此不得杜撰。补充的 9,999 次 run-node
bootstrap 必须明确标为 DELTA supplemental statistics。正式 seed 的学习曲线不得用于
修改代码或配置。

## 11. 关键理论保证

### 定理 1：合法历史不可区分下界

若两个伙伴在关键决策前产生相同合法历史，却分别要求互斥动作，选错损失均为
\(\Delta\)，则任意合法历史策略在二者均匀先验下平均遗憾至少为 \(\Delta/2\)。

**证明。** 相同历史迫使策略使用同一动作分布。设选择伙伴一最优动作的概率为
\(p\)，则两个伙伴的期望损失分别至少为 \((1-p)\Delta\) 与 \(p\Delta\)，均值恰为
\(\Delta/2\)。因此 pre-evidence hidden-code separation 不可能由合法 online
encoder 实现。∎

### 定理 2：Target-policy epoch 的目标一致性

在一个 epoch 内，若 Retrace、anchor continuation 与 endpoint bootstrap 都使用
\(\bar\pi_e\)，则它们分别是同一价值函数 \(Q^{\bar\pi_e}_{raw}\) 的 on/off-policy
估计与第一动作干预估计，不存在 continuation-policy 语义冲突。

**证明。** 三者条件化于相同历史和第一动作；第一步后所有条件转移的行为评价目标
均为 \(\bar\pi_e\)。Tower property 给出每种 estimator 的条件期望都是
\(Q^{\bar\pi_e}_{raw}(h,a)\)。若混入 \(\bar\pi_{e'}\)，一般存在 MDP 使两策略后续动作
回报不同，两个标签期望不同，故 epoch fingerprint 隔离是必要条件。∎

### 定理 3：Potential shaping 的策略不变性

一个 target epoch 内冻结 \(\Phi=-\mathcal R_{dec}\)，terminal \(\Phi=0\)，则
\(F_t=\gamma\Phi_{t+1}-\Phi_t\) 的折扣和望远镜化为初始状态常数
\(-\Phi_0\)，不改变该冻结 MDP 的最优策略集合。C5 前权重为零，因此未经校准的
势函数不会改变策略。

### 定理 4：Split-replica 选择评价无同样本选择偏差

若 fit replicas 只用于选择动作，evaluation replicas 使用独立随机域，条件于 anchor
状态，evaluation mean 是所选动作真实 continuation value 的无偏估计。共同随机数只
降低动作差的方差，不破坏两组独立性。∎

### 定理 5：信号契约安全退化

对任一正式 seed，状态机总能导出一个 artifact；其部署机制不超过最高通过契约。

**证明。** C0 失败有 owner source；C0 通过有 frozen base；C3/C4 通过才注册
conditional；C5 通过才注册 active；calibration 不足则 gate 回退 base。每个分支均
有终止 artifact，且 transition 只向更高层发生。因此资格失败不会造成 seed 缺失，
也不会把未资格模块加入部署。∎

这些定理不推出 C0–C5 必然通过，也不推出 benchmark 回报提升。

## 12. Checkpoint、CUDA 与资源账本

Checkpoint identity 必须绑定 target epoch、qualified base、last-qualified generator、
generator optimizer state/counter、replay、qualification、phase、所有 optimizer
counters、runner、random domains 与 resource ledger fingerprint。r2 schema 恢复硬失败。

正式 worker 要求 `JAX backend=gpu`、`JAX_PLATFORMS=cuda` 且只可见一张 GPU。
Training/audit anchors 使用各自固定形状 kernel 和独立 microbatch preflight；分批只
改变内存与时间，所有科学 keys 在选择批大小前生成。峰值显存必须小于 40,000 MiB，
禁止 CPU fallback、OOM、NaN 与 retrace 风暴。

资源账本分别报告：owner/base distillation observation collection、Official source
训练、generator 完整 episode 训练、training anchors、audit anchors、calibration、
ego PPO、evaluation、GPU-hours、峰值显存、部署参数、训练期参数与推理延迟。不得将
方法特有辅助模拟隐藏在 30M ego PPO 预算中。

## 13. 冻结程序

1. 归档 r2；
2. 文档、代码、配置和测试形成同一原子变更；
3. 运行全部单元测试、固定 Official runtime 检查与 CUDA formal-shape preflight；
4. 使用不属于 Official 0–9 的 engineering seed 覆盖 C0–C5、fallback、calibration、
   deployment 与 resume；
5. 单一提交并推送；
6. 为正式 seed 生成独占 owner/init/support/calibration manifests；
7. 冻结后启动 Official seed 0–9；
8. 不以任何正式 seed 学习曲线修改方法；
9. 十个 artifacts 全部进入 Official 矩阵。

最终可证伪链是：伙伴有能力；伙伴确实要求不同动作；raw-Q 估计同一 continuation
policy；合法历史可恢复动作差异；conditional actor 将差异变成回报；decision
regret 能预测信息任务价值。任一环节不成立，代码必须自动回到上一个已验证层。
