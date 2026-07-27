# Path C V4.3 executable-response-value r1 模块设计

状态：`implemented`。方法标识：`path_c_v4_3_executable_response_value_r1`。

## 1. 设计目标

模型只保留一条可执行因果链：

```text
ego history
  → Bellman-trained control state
  → latent value hypotheses and slot posterior
  → response-conditioned next-action values
  → runtime-identical KL policies
  → environment response
  → slot-level Bayes update
```

伙伴身份、seed、SP/OP 标签和 checkpoint 编号不进入训练 batch、Q heads、response model、posterior、value class 或触发规则。它们只允许作为审计字段写入完整记录。

## 2. 信息状态

官方 recurrent network接收公开局部观测和回合边界，输出官方循环特征 `f_t`。独立 control memory 在进入 GRU 前融合上一 ego 动作与上一原始团队回报：

\[
u_t=\operatorname{LayerNorm}(f_t+E(a_{t-1})+P(r_{t-1})),
\qquad
c_t=\operatorname{GRU}(c_{t-1},u_t).
\]

Q 和 outcome heads 读取 `c_t`。官方可训练循环、`E`、`P` 和 control GRU 只接收 Bellman loss；outcome 和 response encoder读取停止梯度的 `c_t`。

## 3. 独立 Bellman experts

每个 slot \(m\) 和 twin estimator \(e\) 具有独立 dueling head：

\[
Q_{e,m}(c_t,b_t,a)=V_{e,m}(c_t,b_t)+A_{e,m}(c_t,b_t,a)
-\frac1{|\mathcal A|}\sum_uA_{e,m}(c_t,b_t,u).
\]

belief输入停止梯度。固定随机prior按estimator与slot独立且不更新。

## 4. TD-only latent responsibility

每条完整环境lane对应一个完整400步episode。slot responsibility只使用target network的Bellman evidence：

\[
E_{i,m}^{TD}=\sum_t\frac12\sum_e
\rho\!\left(Q_{e,m}(c_t,b_t,a_t)-y_{t,m}\right),
\]

\[
q_{i,m}=\operatorname{sg}\operatorname{softmax}_m
\left(-E_{i,m}^{TD}/\tau\right).
\]

response NLL、reward NLL、next-Q NLL和next-reference MSE不进入上式。它们只训练outcome model，并逐epoch写入审计记录。这样latent slot由动作价值一致性定义，而不是由可识别但控制无关的回应指纹定义。

## 5. 持久后验与动态value class

真实response code为\(y_t\)时：

\[
\log b_{t+1}(m)=\log b_t(m)+\log p_m(y_t\mid c_t,a_t)-\log Z_t.
\]

posterior始终保存在slot层。value class只用于当前控制统计；两个slots仅在动作相对价值距离和twin不确定性上界共同低于容差时合并，不会抹平类内posterior。

## 6. 回应编码

response encoder从\((o_t,a_t,o_{t+1},d_t)\)生成有限code。训练target不再是下一状态的静态value signature，而是均匀belief下current-to-next centered-advantage delta：

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

## 7. Action-vector outcome model

对当前候选动作\(a\)和可能回应\(y\)，outcome model预测：

- \(p_m(y\mid c,a)\)；
- 即时原始团队回报\(\hat r_m(c,a)\)；
- use belief下的下一动作Q向量\(G^U_{e,m}(c,a,y,b^y,u)\)；
- mask belief下的下一动作Q向量\(G^M_{e,m}(c,a,y,b,u)\)；
- 回应条件的下一reference logits \(\ell^0(c,a,y,u)\)。

next-Q targets直接来自target network在真实下一control feature上的完整动作向量。next-reference target来自真实下一官方reference distribution，并固定additive-logit gauge。

## 8. 共同物理posterior与可执行next policy

当前belief为\(b\)时：

\[
p(y\mid a)=\sum_m b(m)p_m(y\mid a),
\qquad
\rho^{a,y}(m)=\frac{b(m)p_m(y\mid a)}{p(y\mid a)}.
\]

use controller的pessimistic next-action score：

\[
S_U^{a,y}(u)=\min_e\sum_m\rho^{a,y}(m)G^U_{e,m}(a,y,u).
\]

mask controller仍用旧belief选动作：

\[
S_M^{a,y}(u)=\min_e\sum_m b(m)G^M_{e,m}(a,y,u).
\]

两者通过与runtime完全相同的KL算子形成下一策略：

\[
\pi_U^{a,y}(u)\propto
\exp\left(\ell^0_{a,y}(u)+S_U^{a,y}(u)/\alpha\right),
\]

\[
\pi_M^{a,y}(u)\propto
\exp\left(\ell^0_{a,y}(u)+S_M^{a,y}(u)/\alpha\right).
\]

物理评价对两分支都使用真实条件posterior \(\rho^{a,y}\)。因此mask只阻断controller使用回应，不会改变世界属于哪个slot的评价者posterior。

## 9. J_use、J_mask与平移不变收益

\[
J_U^e(a)=\sum_mb(m)\hat r_m(a)
+\gamma\sum_yp(y\mid a)
\sum_m\rho^{a,y}(m)\sum_u\pi_U^{a,y}(u)G^U_{e,m}(a,y,u),
\]

\[
J_M^e(a)=\sum_mb(m)\hat r_m(a)
+\gamma\sum_yp(y\mid a)
\sum_m\rho^{a,y}(m)\sum_u\pi_M^{a,y}(u)G^M_{e,m}(a,y,u).
\]

控制值对twins取保守最小值。当前执行policy仍为：

\[
\pi(a)\propto\pi_{ref}(a)\exp(J_U(a)/\alpha).
\]

主要可执行回应读数是policy-mediated gain：

\[
\Delta_{policy}(a)=\gamma\min_e\sum_yp(y\mid a)
\sum_m\rho^{a,y}(m)\sum_u
(\pi_U^{a,y}(u)-\pi_M^{a,y}(u))\bar G_{e,m}(a,y,u),
\]

其中\(\bar G=(G^U+G^M)/2\)。由于两个策略概率差之和为零，该量对所有动作共同价值平移严格不敏感。同时记录：

\[
\mathbb E_y[TV(\pi_U^{a,y},\pi_M^{a,y})].
\]

回应屏蔽只在实际执行动作的policy-mediated gain为正，且expected next-policy TV达到登记floor时触发。

## 10. 训练behavior support

训练时执行：

\[
\mu(a\mid h)=(1-\epsilon)\pi(a\mid h)+\epsilon/|\mathcal A|,
\qquad \epsilon=0.1.
\]

Bellman target仍使用目标部署策略\(\pi\)，不使用support mixture进行bootstrap。评估中\(\epsilon=0\)。完整记录保存behavior logits和被执行动作的behavior probability。

## 11. KL温度

每个rollout更新后，使用更新后模型在完整batch上的score，通过固定24次log-space bisection求解：

\[
\frac1N\sum_iD_{KL}(\pi_{\alpha,i}\Vert\pi_{ref,i})=0.02.
\]

若目标位于登记区间外，返回对应边界。posterior-use和generic-information分别求解温度。

## 12. 完整记录

每个update和epoch保存全部E-step行：

```text
update, epoch, environment, episode, audit_partner_member, slot,
responsibility, td_energy, response_nll_energy, reward_nll_energy,
next_q_use_nll_energy, next_q_mask_nll_energy,
next_reference_mse_energy, bootstrap_available
```

`audit_partner_member`不进入训练tensor或任何推断路径。

## 13. 参数消费者

- Bellman optimizer：官方可训练循环、control memory、learned Q experts；
- outcome optimizer：response、reward、next-Q、next-reference和response encoder；
- random priors：冻结。

两组optimizer独立裁剪；outcome梯度不能进入control feature。

## 14. 代码所有权

| 文件 | 唯一职责 |
|---|---|
| `src/path_c/experiment.py` | 配置、固定预算、训练单元、population |
| `src/path_c/method.py` | Bayes、value class、可执行J值、KL策略、温度、码本 |
| `src/path_c/model.py` | control memory、Q experts、action-vector outcome、response encoder |
| `src/path_c/training.py` | targets、TD-only responsibility、loss、optimizer、target update |
| `src/path_c/runner.py` | 训练support rollout与伙伴状态 |
| `src/path_c/evaluation.py` | 配对、结果行、验证和统计 |
| `src/path_c/storage.py` | 单一run identity、Orbax和无损记录 |
| `official_adapter.py` | 锁定官方软件接口 |
| `training_app.py` | 单一训练用例与完整E-step记录 |
| `deployment.py` | checkpoint到policy的唯一装配 |
| `standard_evaluation_app.py` | 标准矩阵 |
| `response_contrast_app.py` | 三分支对照 |

## 15. 版本边界

V4.3改变了model tree、response signature宽度、TransitionBatch、checkpoint state、evaluation rows和config schema。V4.2/control-memory r1 checkpoint不能恢复到V4.3。
