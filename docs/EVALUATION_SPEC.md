# EVALUATION_SPEC：Unified DELTA-ZSC 三假设评估

`authoritative: true`

本规格只允许三项论文假设。所有其他读数均为诊断，不得在观察结果后升级为新的贡献、gate或主张。

## 1. 数据分割

伙伴parent按lineage分为：

- `development_support`：训练base与latent model；
- `calibration`：held-out posterior-predictive诊断；
- `confirmatory`：H1与H3正式评估；
- external baseline自身训练数据必须与confirmatory panel隔离。

以下任一重叠均fail closed：

- checkpoint SHA；
- parent training run；
- co-training group；
- owner source；
- calibration与confirmatory；
- training support与confirmatory。

伙伴算法名称不是统计单位；独立parent run才是伙伴侧推断单位。

## 2. 自然嵌套模型

正式开发只训练三个模型：

1. `base`：task-only recurrent PPO + memoryless instant partner；
2. `response_only`：与base完全相同的base policy，外加只用response likelihood训练的latent model；
3. `joint`：与前两者完全相同的base policy，latent model同时使用response与decision emission。

`full`不重新训练。它在同一个`joint` deployment上开启解析VOI，因此只有额外推理计算，没有额外参数、训练数据或optimizer。

同一layout与seed下，三个训练run必须验证：

- base parameter fingerprint完全相同；
- training partner sampler hash相同；
- base rollout/action key domains相同；
- PPO environment steps相同；
-base optimizer配置相同。

若base fingerprint不同，H2不可解释，必须修复执行路径后重跑，不能用统计调整补偿。

## 3. H1：性能

### 3.1 命题

\[
H1:
J_{full}-J_{best\ external}>0
\]

并要求点估计至少达到一次正确交付：

\[
\widehat{J_{full}-J_{best\ external}}\ge20.
\]

### 3.2 外部基线

至少包含：

- Official SP；
- Official State-Augmented；
- Official OP；
- Official FCP；
- strong full-history recurrent policy；
- 至少一个与当前文献对应、合法信息和预算匹配的history/context adaptation baseline。

仓内proxy可作为内部结构control，但必须标记 `not_a_published_method_reproduction`，不能用于“超过已发表方法”或领域SOTA措辞。

若没有忠实同期baseline，H1最多写为“超过Official benchmark references与内部强controls”。

### 3.3 Official evaluation

每个layout：

- ego seed indexes 0–9；
- confirmatory partner parents至少覆盖SP、SA、OP、FCP及deterministic heuristic family；
- 每个ego/partner/role pairing 500个400-step episodes；
- 两种ego role均评估；
- raw unshaped simulator return；
- 所有方法共享episode keys；
- 保存每个episode raw row。

H1分别在 `test_time_simple` 与 `test_time_wide` 判定。完整论文H1只有在两布局均满足注册条件时通过；任一布局失败必须原样报告。

### 3.4 推断

训练run是ego侧一级单位，partner parent是伙伴侧一级单位，episode只在节点内部估计均值。使用paired seed/node bootstrap；若baseline无法合法配对，则使用预注册的双样本hierarchical bootstrap并明确标注。

每布局通过条件：

- `one_sided_LCB_95(full-best_external)>0`；
- point estimate `>=20`。

## 4. H2：decision emission贡献

### 4.1 命题

\[
H2:
J_{joint}-J_{response-only}>0.
\]

joint和response-only：

- 共享完全相同的base policy parameters；
- 使用相同response architecture、K、transition、optimizer、rollouts和伙伴；
- 唯一差别是joint latent MLE在anchor时刻包含 `p_psi(A|z,c)`。

因此H2直接回答：

> privileged counterfactual decision emission是否使合法response posterior获得可执行的决策语义？

H2以相同confirmatory panel、roles和episode keys进行paired comparison。每布局通过条件：

\[
one\text{-}sided\ LCB_{95}(J_{joint}-J_{response-only})>0.
\]

不得用response NLL、component divergence、M1或训练loss替代H2。

## 5. H3：belief因果决策价值

### 5.1 干预

在同一个source anchor保持不变：

- environment snapshot；
- source partner policy与carry；
- task features `x_s`；
- instant geometry `r_s`；
- analytic behavior statistics `u_s`；
- base logits；
- decision-emission component values；
- CRN all-action source returns。

只把正确belief `b_s` 替换为另一个partner run中task-state最近邻的donor belief `b_d`。不调用context encoder，不拼接donor recurrent hidden，不改变source partner。

计算两种mirror policy：

\[
\pi_c=Mirror(\pi_0,E_{b_s}[Q]),
\]

\[
\pi_d=Mirror(\pi_0,E_{b_d}[Q]).
\]

单anchor因果读数：

\[
\Delta V_s
=(\pi_c-\pi_d)^TG_{source}.
\]

### 5.2 命题

\[
H3:\mathbb E[\Delta V_s]>0.
\]

以partner parent为block bootstrap单位；正式扩展可同时报告ego/partner/donor crossed bootstrap。通过条件：

\[
one\text{-}sided\ LCB_{95}(\Delta V)>0.
\]

belief TV、action TV和task matching distance只作描述性读数。

## 6. Confirmatory hierarchy

为了避免多项独立claim中的选择性叙事，使用固定顺序：

```text
H1 performance
  -> H2 decision-emission mechanism
       -> H3 causal belief value
```

- 所有数值始终报告；
- 只有H1通过，H2才获得confirmatory解释；
- 只有H1和H2通过，H3才获得confirmatory解释；
- 被锁定的结果仍作为exploratory evidence报告；
- 不通过时不得新增替代claim。

## 7. VOI读数

`full-joint`是预注册的次级paired contrast：

\[
J_{full}-J_{joint}.
\]

它衡量解析myopic VOI的增量，但不单独构成第四项论文假设。无论正负都完整报告。

同时报告：

- VOI均值和分位数；
- VOI为零的状态比例；
- full相对joint的action TV；
- adaptation KL分布；
- 触及KL边界的比例。

## 8. Posterior-predictive诊断

calibration panel在训练和confirmatory之外。报告run-block bootstrap：

- full component response NLL；
- nested base-response NLL；
- uniform-mixture NLL；
- component residual predictive gain；
- filter gain relative to uniform mixture；
- posterior entropy与component utilization；
- position/direction/event reliability。

这些指标判断模型设定和response支持是否健康，但不是独立论文claim，不阻断H1 raw performance报告。

若component residual对base response无held-out增量，必须在论文中说明latent response modes没有得到证据支持；不得增加separation loss制造component差异。

## 9. Decision-emission诊断

在fit replicas训练、独立evaluation replicas测量：

- per-state action-value RMSE；
- tie-aware Kendall `tau_b`；
- top-action agreement；
- selected-action empirical regret；
- predictive interval coverage；
- decision NLL；
- component-conditioned decision diversity。

诊断只使用fresh final deployment和fresh anchors，不复用训练anchor。

## 10. K敏感性

主设置固定 `K=4`。开发期只对joint模型运行：

- `K=2`；
- `K=4`；
- `K=8`。

K sensitivity使用5个预注册development seeds和一个layout进行资源受控诊断；不乘以全部模型、消融和正式矩阵。K的选择必须在confirmatory结果生成前冻结。

## 11. 开发资源设计

每个layout的主开发矩阵：

```text
base          5 seeds
response_only 5 seeds
joint         5 seeds
full          reuse joint deployments, no training
```

若pilot run-level variance表明5 seeds不足以检测20分物质效果，则在不查看confirmatory结果前将全部三个训练模型统一扩展到10 seeds。

正式矩阵固定10 seeds。

不再运行：

- 14个K=4变体；
- comparator/no-separation消融；
- actor-only/Q-only/posterior-only补丁消融；
- 60-run fixed-p_stay sensitivity；
- 让baseline采集并丢弃无用anchors的伪预算匹配。

## 12. 公平性

报告两种成本视图：

1. **base interaction matched**：三个训练模型拥有相同PPO steps；joint额外anchor成本单列；
2. **fully loaded**：包含upstream伙伴、anchor continuation、训练、calibration、causal evaluation、formal episodes、GPU hours和wall time。

若构造total-budget control，应让baseline将相同额外simulator budget用于更多真实PPO rollout，而不是采集后丢弃；该control只需在主K和5个development seeds运行。

## 13. Raw artifact合同

每个evaluation目录必须包含：

- immutable run identity；
- deployment SHA与parameter fingerprints；
- config fingerprint；
- partner manifest SHA；
- episode key domain；
- per-episode parquet；
-完整resource ledger；
- method/variant/layout/role/seed；
- episode count与完整性检查。

summary必须从raw rows重算，不接受手写mean、自由格式score JSON或训练曲线。

## 14. 结果措辞

允许的最高措辞由证据决定：

- 只超过Official refs：`outperforms the registered OvercookedV2 reference methods`；
- 超过内部history controls：`outperforms capacity/budget-matched history-conditioned controls`；
- 超过忠实同期方法：才可讨论领域SOTA；
- H2通过：`decision emission contributes incremental coordination performance`；
- H3通过且hierarchy unlocked：`belief has positive causal decision value under the registered source-world intervention`。

不得把posterior calibration、latent clustering或测试通过写成协调性能贡献。
