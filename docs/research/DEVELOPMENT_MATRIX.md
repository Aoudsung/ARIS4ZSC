# Unified DELTA-ZSC 开发矩阵

`authoritative: false`

本文件是 `EVALUATION_SPEC.md` 的执行说明，不新增方法、阈值或论文claim。

## 1. 目标

开发实验只回答四个必要问题：

1. base policy能否形成可靠任务能力；
2. response latent model能否从合法历史预测伙伴响应；
3. privileged decision emission是否带来joint相对response-only的XP增量；
4. joint参数开启解析VOI后是否进一步改善或损害performance。

不再逐条验证旧审稿补丁。

## 2. 主矩阵

每个layout使用paired seeds 0–4：

| Training | Evaluation policies | 科学作用 |
|---|---|---|
| `base` | base | 无历史适应基座 |
| `response_only` | response-only | response inference但无decision adaptation |
| `joint` | joint、full | decision-emitting latent；full只开启解析VOI |

每layout只有15个训练run，而不是旧版的大规模variant笛卡尔积。

## 3. 强制一致性

同layout/seed的base、response-only、joint必须共享：

- config中除variant外的全部字段；
- partner manifest与sampler；
- rollout/environment/action keys；
- base initialization；
- PPO minibatch permutations；
- PPO steps；
- base optimizer。

由于latent model不会影响training behavior policy，同seed三条run的final base parameter fingerprint必须完全一致。任何不一致表示代码路径泄漏，不能进入统计summary。

## 4. Development evaluation

每个deployment在同一development-only partner panel上：

- 双角色；
- 每pairing 100 episodes；
- shared episode keys；
- raw unshaped return；
-保存per-episode parquet。

主要paired contrasts：

```text
joint - response_only
full - joint
full - base
```

其中只有 `joint-response_only` 对应H2机制pilot；其他用于性能和VOI诊断。

## 5. K诊断

在预注册的一个layout和seeds 0–4，仅训练joint：

```text
K=2
K=4
K=8
```

比较：

- held-out response NLL；
- decision NLL与evaluation-replica ordering；
- joint XP；
- belief utilization；
- inference latency；
- fully loaded cost。

K选择不能只依据最高XP；若多个K统计接近，选择更小模型。K在confirmatory前冻结。

## 6. Pilot功效

以paired seed difference估计：

\[
\hat\sigma_\Delta
=SD(J_{joint,s}-J_{response,s}).
\]

使用20分最小效果、预注册alpha和目标power估计正式seed数。正式协议已固定10 seeds；pilot只用于判断development是否需要从5扩展到10，而不改变正式规模。

## 7. Anchor预算

joint训练每隔固定outer updates从当前base rollout采集anchors。anchors：

- 只用当前rollout；
-一次性进入当前latent update；
- 不matched、不pair、不replay；
- fit/evaluation replicas分离；
- 所有action branches共享CRN。

Development记录effective anchor count、standard errors、decision NLL和额外simulator steps。若anchor噪声过高，应修改采样数的统计设计，而不是增加confidence loss。

## 8. 失败解释

- response-only response NLL无增量：response model或伙伴分布不支持latent modes；
- joint decision NLL好但XP无增量：decision geometry未转化为有效KL adaptation；
- joint优于response-only但causal belief value≤0：增量可能来自其他训练效应，H3不成立；
- full低于joint：myopic VOI近似不适用，保留joint为passive方法；
- full与joint相同且VOI≈0：当前response不提供可利用主动信息；
- base很弱：先修复基础任务训练，不通过加大adaptation弥补。

任何失败都不得触发旧式“新增一个loss和一个消融”的默认响应。

## 9. 资源上限

开发主矩阵：

```text
3 training variants × 5 seeds × 2 layouts = 30 runs
```

K诊断：

```text
3 K values × 5 seeds × 1 layout = 15 joint runs
```

其中K=4主joint可复用，新增约10 runs。总开发训练不超过40个独立run；full不训练。

这使development规模足以估计关键增量，同时避免旧版数百run矩阵。

## 10. 输出

每个run生成：

```text
run_identity.json
partner_pool.json
metrics.jsonl
checkpoints/
final_deployment/
resource_ledger.json
training_summary.json
```

每个evaluation生成raw parquet、identity、summary和resource ledger。开发summary必须验证base fingerprints和paired keys，不接受手工score文件。
