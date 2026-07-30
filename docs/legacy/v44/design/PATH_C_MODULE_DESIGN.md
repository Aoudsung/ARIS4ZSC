# Path C V4.4 retrace-calibrated-control r1 模块设计

状态：`implemented`。方法标识：`path_c_v4_4_retrace_calibrated_control_r1`。

## 1. 设计目标

模型只保留一条可执行因果链：

```text
ego history
  → Bellman-trained control state
  → latent value hypotheses and slot posterior
  → response-conditioned next-action distributions
  → uncertainty-calibrated runtime policy
  → environment response and sparse reward
  → full-episode Retrace credit
```

伙伴身份、seed、SP/OP 标签和 checkpoint 编号不进入训练 batch的模型输入、Q heads、response model、posterior、value class或触发规则。它们只作为审计字段写入完整记录。

V4.3 已经证明回应能够改变真实动作，但未形成正任务价值。V4.4 因而不再处理“是否接线”，而是修复三个仍未闭合的对象：稀疏奖励的长时信用分配、训练期动作支持的整回合一致性、以及预测收益相对模型误差的校准。

## 2. 信息状态

官方 recurrent network 接收公开局部观测和回合边界，输出官方循环特征 `f_t`。独立 control memory在进入GRU前融合上一ego动作与上一原始团队回报：

\[
u_t=\operatorname{LayerNorm}(f_t+E(a_{t-1})+P(r_{t-1})),
\qquad
c_t=\operatorname{GRU}(c_{t-1},u_t).
\]

Q和outcome heads读取`c_t`。官方可训练循环、`E`、`P`和control GRU只接收Bellman loss；outcome和response encoder读取停止梯度的`c_t`。

## 3. 独立 Bellman experts

每个slot \(m\) 和 twin estimator \(e\) 具有独立dueling head：

\[
Q_{e,m}(c_t,b_t,a)=V_{e,m}(c_t,b_t)+A_{e,m}(c_t,b_t,a)
-\frac1{|\mathcal A|}\sum_uA_{e,m}(c_t,b_t,u).
\]

belief输入停止梯度。固定随机prior按estimator与slot独立且不更新。

## 4. 持久后验与动态 value class

真实response code为\(y_t\)时：

\[
\log b_{t+1}(m)=\log b_t(m)+\log p_m(y_t\mid c_t,a_t)-\log Z_t.
\]

posterior始终保存在slot层。value class只用于当前控制统计；两个slots仅在动作相对价值距离和twin不确定性上界共同低于容差时合并，不会抹平类内posterior。

## 5. TD-only latent responsibility

每条环境lane对应一个完整400步episode。slot responsibility只使用target network的Bellman evidence：

\[
E_{i,m}^{TD}=\sum_t\frac12\sum_e
\rho\!\left(Q_{e,m}(c_t,b_t,a_t)-Y_{t,m}^{Ret}\right),
\]

\[
q_{i,m}=\operatorname{sg}\operatorname{softmax}_m
\left(-E_{i,m}^{TD}/\tau\right).
\]

response NLL、reward NLL、next-Q NLL和next-reference MSE不进入责任分配。它们只训练outcome model并进入审计行。

## 6. 完整回合 Retrace

训练行为\(\mu_t\)与目标部署策略\(\pi_t\)不同，因此Bellman target使用记录的behavior probability计算Retrace。令：

\[
\rho_t=\frac{\pi_t(a_t\mid h_t)}{\mu_t(a_t\mid h_t)},
\qquad
c_t=\lambda\min(\bar\rho,\rho_t),
\]

其中登记值为\(\lambda=0.9\)、\(\bar\rho=1\)。一步误差为：

\[
\delta_t=r_t+\gamma(1-d_t)
\sum_u\pi_{t+1}(u)\min_eQ^-_{e,m}(h_{t+1},u)
-Q^-_{m}(h_t,a_t).
\]

完整回合target按反向递推：

\[
Y_t^{Ret}=Q^-_m(h_t,a_t)+\delta_t
+\gamma(1-d_t)c_{t+1}
\left(Y_{t+1}^{Ret}-Q^-_m(h_{t+1},a_{t+1})\right).
\]

回合边界阻断trace。`retrace_lambda=0`时退化为一步target。目标部署策略始终是实际评估使用的policy，behavior mixture只提供数据支持。

## 7. 整回合一致的训练行为

每个episode开始时，为每条lane固定采样一个 `(estimator, slot)` 假设\((e_i,m_i)\)。该假设在整个episode内保持不变，构造：

\[
\pi^{exp}_{i,t}(a)
\propto
\pi_{ref,t}(a)
\exp\left(Q_{e_i,m_i}(c_t,b_t,a)/\alpha_{exp}\right).
\]

训练behavior为：

\[
\mu_t(a)=
(1-\eta-\epsilon)\pi_t(a)
+\eta\pi^{exp}_{i,t}(a)
+\epsilon/|\mathcal A|.
\]

登记值为\(\eta=0.25\)、\(\alpha_{exp}=0.5\)、\(\epsilon=0.02\)。该设计使一次episode中的探索围绕同一潜在控制假设展开，避免每步独立均匀扰动破坏任务序列。部署和评估固定\(\eta=\epsilon=0\)。

## 8. 回应编码

response encoder从\((o_t,a_t,o_{t+1},d_t)\)生成有限code。训练target使用均匀belief下current-to-next centered-advantage delta：

\[
\Delta A_{e,m,t}(u)=A_{e,m}(c_{t+1},b_0,u)-A_{e,m}(c_t,b_0,u).
\]

使用slot/estimator置换不变统计：

\[
\mu_t(u)=\mathbb E_{e,m}[\Delta A_{e,m,t}(u)],
\qquad
\sigma_t(u)=\operatorname{Std}_{e,m}[\Delta A_{e,m,t}(u)].
\]

最终signature为\([\mu_t,\sigma_t]\in\mathbb R^{2|\mathcal A|}\)。terminal transition使用独立terminal code。

## 9. Action-vector outcome model

对当前候选动作\(a\)和可能回应\(y\)，outcome model预测：

- \(p_m(y\mid c,a)\)；
- 即时原始团队回报\(\hat r_m(c,a)\)及其标准差；
- use belief下的下一动作Q向量均值和标准差\((\mu^U_{e,m},\sigma^U_{e,m})\)；
- mask belief下的下一动作Q向量均值和标准差\((\mu^M_{e,m},\sigma^M_{e,m})\)；
- 回应条件的下一reference logits \(\ell^0(c,a,y,u)\)。

next-Q targets来自target network在真实下一control feature上的完整动作向量。next-reference target来自真实下一官方reference distribution，并固定additive-logit gauge。

## 10. 保守控制值与下一策略

当前belief为\(b\)时：

\[
p(y\mid a)=\sum_m b(m)p_m(y\mid a),
\qquad
\rho^{a,y}(m)=\frac{b(m)p_m(y\mid a)}{p(y\mid a)}.
\]

以登记系数\(\beta=1\)构造下一动作lower-confidence value：

\[
G^{U,-}_{e,m}=\mu^U_{e,m}-\beta\sigma^U_{e,m},
\qquad
G^{M,-}_{e,m}=\mu^M_{e,m}-\beta\sigma^M_{e,m}.
\]

use与mask controller分别得到：

\[
S_U^{a,y}(u)=\min_e\sum_m\rho^{a,y}(m)G^{U,-}_{e,m}(a,y,u),
\]

\[
S_M^{a,y}(u)=\min_e\sum_mb(m)G^{M,-}_{e,m}(a,y,u).
\]

两者通过与runtime完全相同的KL算子形成下一策略：

\[
\pi_U^{a,y}(u)\propto\exp(\ell^0_{a,y}(u)+S_U^{a,y}(u)/\alpha),
\]

\[
\pi_M^{a,y}(u)\propto\exp(\ell^0_{a,y}(u)+S_M^{a,y}(u)/\alpha).
\]

物理评价对两分支都使用真实条件posterior\(\rho^{a,y}\)。terminal response的continuation均值与不确定性都固定为零。

## 11. J_use、J_mask 与不确定性校准

保守控制值为：

\[
J_U^e(a)=\sum_mb(m)\hat r_m(a)
+\gamma\sum_yp(y\mid a)
\sum_m\rho^{a,y}(m)\sum_u\pi_U^{a,y}(u)G^{U,-}_{e,m}(a,y,u),
\]

\[
J_M^e(a)=\sum_mb(m)\hat r_m(a)
+\gamma\sum_yp(y\mid a)
\sum_m\rho^{a,y}(m)\sum_u\pi_M^{a,y}(u)G^{M,-}_{e,m}(a,y,u).
\]

当前执行policy使用保守\(J_U\)：

\[
\pi(a)\propto\pi_{ref}(a)\exp(J_U(a)/\alpha).
\]

原始回报诊断仍使用mean Q在该保守policy下的期望，避免把KL或不确定性惩罚误当成环境回报。

## 12. 平移不变 policy gain LCB

对每个候选动作和回应，令\(\Delta\pi=\pi_U-\pi_M\)，\(\bar G=(\mu^U+\mu^M)/2\)。mean policy-mediated gain为：

\[
\Delta_e(a)=\gamma\sum_y p(y\mid a)
\sum_m\rho^{a,y}(m)\sum_u\Delta\pi^{a,y}(u)\bar G_{e,m}(a,y,u).
\]

由于\(\sum_u\Delta\pi(u)=0\)，该量对动作共同平移不敏感。模型标准差通过相同权重传播为\(s_e(a)\)，登记下置信界：

\[
LCB(a)=\min_e\left(\Delta_e(a)-\beta s_e(a)\right).
\]

回应屏蔽只在实际执行动作同时满足：

```text
LCB(action) > numerical tolerance
expected_next_policy_TV(action) >= 0.001
```

时触发。分箱按LCB而非mean gain排序；mean、uncertainty和LCB均保存。

## 13. KL温度

每个rollout更新后，使用更新后模型在完整batch上的score，通过固定24次log-space bisection求解：

\[
\frac1N\sum_iD_{KL}(\pi_{\alpha,i}\Vert\pi_{ref,i})=0.02.
\]

若目标位于登记温度区间外，返回对应边界。posterior-use和generic-information分别求解温度。

## 14. 固定伙伴开发 panel

标准单checkpoint自我配对可能在高参考回报处饱和，不能单独判断伙伴条件控制。`evaluate-panel`将一个训练ego分别与训练单元登记的冻结官方伙伴配对：默认从每个start/midpoint/final三元组中取final checkpoint，并在四种部署模式中复用相同episode seeds。

panel只用于development，不替代10×10正式矩阵。partner index只用于分层输出，不进入模型。

## 15. 完整记录

每个update和epoch保存全部E-step行：

```text
update, epoch, environment, episode, audit_partner_member, slot,
responsibility, td_energy, response_nll_energy, reward_nll_energy,
next_q_use_nll_energy, next_q_mask_nll_energy,
next_reference_mse_energy, bootstrap_available,
mean_importance_ratio, mean_trace_coefficient
```

每步决策另保存deployment/behavior/exploration logits、behavior action probability、选定expert、mean gain、uncertainty、LCB和next-policy TV。`audit_partner_member`不进入训练tensor或推断路径。

## 16. 参数消费者

- Bellman optimizer：官方可训练循环、control memory、learned Q experts；
- outcome optimizer：response、reward、next-Q、next-reference和response encoder；
- random priors：冻结。

两组optimizer独立裁剪；outcome梯度不能进入control feature。

## 17. 代码所有权

| 文件 | 唯一职责 |
|---|---|
| `src/path_c/experiment.py` | 配置、固定预算、训练单元、population |
| `src/path_c/method.py` | Bayes、value class、保守J值、KL策略、温度、码本 |
| `src/path_c/model.py` | control memory、Q experts、action-vector outcome、response encoder |
| `src/path_c/training.py` | Retrace、TD-only responsibility、loss、optimizer、target update |
| `src/path_c/runner.py` | coherent-support rollout与伙伴状态 |
| `src/path_c/evaluation.py` | 配对、结果行、验证和统计 |
| `src/path_c/storage.py` | 单一run identity、Orbax和无损记录 |
| `official_adapter.py` | 锁定官方软件接口 |
| `training_app.py` | 单一训练用例与完整E-step记录 |
| `deployment.py` | checkpoint到policy的唯一装配 |
| `standard_evaluation_app.py` | 标准矩阵 |
| `response_contrast_app.py` | 三分支对照 |
| `partner_panel_app.py` | 固定官方伙伴开发panel |

## 18. 版本边界

V4.4改变了config schema、Bellman target、training behavior、TrainState、decision rows、response-contrast rows和评估入口。V4.3 checkpoint和输出目录不能恢复到V4.4。
