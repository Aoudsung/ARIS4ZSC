# FORMAL_EXPERIMENT_PROTOCOL：Unified DELTA-ZSC 正式执行合同

`authoritative: true`

本协议只适用于 `delta_joint_response_decision_bayes_v1`。旧DEPI checkpoint、comparator、development matrix、claim report和CUDA artifact均不可复用。

## 1. 冻结身份

正式运行前必须固定：

- clean committed Git SHA；
- `src/delta_zsc` active implementation；
- unified config schema 1；
- checkpoint/deployment schema 1；
- Official source commit `5ce1707cf31c1c115e6f6ba96db7bc9cc80a850e`；
- Python 3.10、JAX 0.4.38、Flax 0.10.3、Optax 0.2.5；
-两个layout配置；
-训练、calibration、confirmatory partner manifests；
-三项假设与统计规则；
-所有RNG domain和seed indexes。

任何方法公式、信息边界、K/H/delta、伙伴划分或统计单位变化均要求新commit、新method或schema版本，并重新开始development evidence。

## 2. 运行环境

### 2.1 CPU工程链

每个冻结commit必须通过：

```bash
python -m compileall -q src/delta_zsc experiments/overcooked_v2
pytest -q experiments/overcooked_v2/tests/test_unified_delta_*.py
python -m experiments.overcooked_v2.delta_zsc --help
```

CI还必须拒绝active path中的retired token：

```text
pair comparator
separation margin
context dropout
learned capability loss
decision/pseudo-posterior weight
gradient routing
partner generator
decision regret shaping
```

### 2.2 CUDA acceptance

正式训练只接受：

- 单张显式注册物理GPU；
- `JAX_PLATFORMS=cuda`；
- `JAX_DEFAULT_MATMUL_PRECISION=highest`；
- 启动时显存≤1 GiB；
- 启动利用率≤10%；
- volatile uncorrectable ECC=0；
- bundled CUDA-12 `ptxas`；
- 完整parameter/optimizer/runner checkpoint roundtrip；
- deployment export/load/action forward；
- real Official environment与real frozen partner checkpoint。

CUDA preflight必须使用mechanical config执行至少一个完整outer update，包括：

1. base-policy rollout；
2. joint response-decision latent likelihood；
3. CRN decision anchor；
4. base PPO；
5. checkpoint save/restore；
6. deployment bundle；
7. joint与full action forward；
8.有限性、KL上界与资源读数。

通过CPU测试但未通过CUDA acceptance，不得启动development训练。

## 3. Partner lineage

manifest沿用固定Official checkpoint provenance，但active method只消费：

- `development_support`；
- `calibration`；
- `confirmatory`。

旧 `comparator_fit` 与 `comparator_validation` role不进入unified训练。

### 3.1 Training support

采样概率：

```text
mechanism uniform
 -> hyperparameter family uniform
 -> checkpoint stage uniform
 -> parent/run uniform
```

每个checkpoint必须保存：

- SHA-256；
- parent training run ID；
- generation mechanism；
- seed/index与真实JAX key；
- checkpoint stage；
- hyperparameter family；
- upstream resource ledger。

### 3.2 Lineage隔离

training、calibration与confirmatory在以下任一层面不得重叠：

- checkpoint hash；
- parent training run；
- co-training group；
- seed/checkpoint复制链；
- ego owner source。

同一算法族允许出现在不同split，但必须由独立parent runs生成。

## 4. RNG合同

正式ego seed indexes固定0–9。每个seed的root key必须由repository registration产生，不能手工替换。

每个outer update从checkpointed root key确定性派生：

- rollout；
- partner action；
- environment transition；
- anchor state selection；
- anchor CRN replica；
- latent update；
- PPO lane permutation。

anchor各action branch在同一个anchor/replica下共享partner/environment randomness。fit与evaluation replicas使用不重叠domain。

resume后后续所有keys必须与uninterrupted run一致。

## 5. Mechanical run

机械运行使用：

```text
experiments/overcooked_v2/configs/delta_unified_simple_mechanical.yaml
seed_index = -1
variant = joint
```

它只验证：

-环境/partner适配；
-参数初始化；
- response/decision likelihood；
- anchor collector；
- PPO和latent optimizer；
- checkpoint/deployment；
- evaluation policy interface。

机械结果不得进入科学统计。

## 6. Development设计

### 6.1 主矩阵

每个layout先执行paired seeds 0–4：

```text
base
response_only
joint
```

full不训练，直接用joint deployment执行variant override。

要求同seed三条run的base-policy trajectory与base parameter fingerprint一致。不同只允许出现在latent parameters与anchor resource cost。

### 6.2 K诊断

只在一个预注册layout、joint variant、seeds 0–4执行：

```text
K = 2, 4, 8
```

K选择在读取confirmatory结果前冻结。若K=4没有明显问题且性能相当，保持K=4作为主设置，避免post-hoc选择最优K。

### 6.3 Development扩展

根据base/joint paired difference的run-level pilot variance，用一次正确交付20分作为最小关注效果计算seed需求。若5 seeds不足，统一扩展base、response-only、joint至10 seeds；不得只扩展表现较好的variant。

### 6.4 Development判定

Development用于：

- 检查joint likelihood是否有限；
- 检查base fingerprints是否匹配；
- 检查decision anchor independent evaluation；
- 估计方差与算力；
- 冻结K与最终config。

Development不要求所有诊断通过才能“允许继续”。是否进入正式实验由负责人根据性能、机制和资源整体裁决，但所有development结果必须保留。

## 7. 正式训练

每个layout正式训练：

```text
base          seeds 0..9
response_only seeds 0..9
joint         seeds 0..9
```

总计每layout30个ego runs，两个layout60个。full由20个joint deployments解析评估，不产生新训练run。

训练数据始终由base policy生成。正式run不可：

- 恢复旧DEPI checkpoint；
- 加载comparator；
- 使用confirmatory伙伴；
- 替换失败seed；
- 跳过nonfinite update；
- 修改anchor interval或replica数；
- 因中间训练曲线提前终止；
- 在训练后选择非final checkpoint。

正式deployment固定final checkpoint。

## 8. Checkpoint与resume

每个checkpoint必须原子保存：

- base/latent parameters；
- base/latent optimizer states；
-完整environment state；
- partner state与当前member；
- ego task carry、Beta statistics、belief、previous observation/action；
- role allocation；
- process RNG；
- environment steps与update count；
- resource ledger。

恢复时验证schema、method、config、partner manifest、parameter tree与run identity。缺少任何字段或hash不一致均fail closed。

测试必须验证：

```text
N uninterrupted updates
==
K updates + checkpoint + restore + (N-K) updates
```

比较parameters、optimizers、runner、RNG、actions、metrics与resource counters。

## 9. 正式评估顺序

对每个layout：

1. 冻结全部deployment manifests；
2. 评估base；
3. 评估response-only；
4. 评估joint；
5. 对同一joint bundle以`variant_override=full`评估full；
6. 评估Official与外部baselines；
7. 运行held-out calibration；
8. 运行belief causal evaluation；
9. 从raw artifacts生成layout summary；
10. 两layout均完成后生成论文级conjunction report。

所有方法共享confirmatory partner panel、roles、episode keys与raw return定义。

不得在读取某个方法结果后更换伙伴panel或episodes。

## 10. Raw evaluation完整性

每个evaluation目录必须验证：

- 10个ego seeds；
- 所有confirmatory partner parents；
- 两种roles；
- 每pairing 500 episodes；
- 每episode 400 steps上限；
- 不重复/不缺失episode key；
- deployment/config/manifest SHA；
- raw return finite；
- resource ledger。

summary只能读取这些目录并重算，不接受手写score。

## 11. Calibration与因果评估

### 11.1 Calibration

每个layout使用同一共享、lineage-disjoint calibration panel。报告nested response base、full component response和uniform mixture的run-block intervals。它是诊断，不决定raw XP是否可报告。

### 11.2 Causal belief

使用fresh source anchors与独立evaluation replicas。donor来自不同partner parent并按task features最近邻匹配。干预只替换belief，不拼接recurrent hidden。

H3 artifact保存per-anchor：

- source/donor partner parent；
- task distance；
- belief TV；
- policy TV；
- source-world value difference；
- continuation contract和keys。

## 12. 三假设summary

每layoutsummary只生成：

- H1 full vs strongest external baseline；
- H2 joint vs response-only；
- H3 correct vs shuffled belief；
- 描述性full-joint VOI contrast；
- 所有diagnostics与resources。

使用 `EVALUATION_SPEC.md` 的ordered hierarchy。不得追加第四项confirmatory claim。

## 13. 资源账本

每run分列：

- base PPO steps；
- anchor continuation steps；
- upstream partner steps；
- calibration steps；
- causal/mechanism steps；
- formal evaluation steps；
- GPU hours；
- wall-clock；
- peak memory；
- deployable parameters；
- inference latency。

报告：

- marginal ego cost；
- shared upstream cost；
- amortized cost；
- fully loaded reproduction cost。

full相对joint只增加evaluation inference，不增加training cost或parameters。

## 14. 失败处理

- nonfinite：保存失败artifact并终止该run；
- OOM：保存编译/显存证据，不能静默减小科学batch；
- checkpoint损坏：终止，不从其他seed复制；
-伙伴checkpoint缺失：manifest失败；
- evaluation缺节点：summary失败；
- H1/H2/H3失败：原样报告，不追加方法补丁后继续使用同一正式数据。

若正式结果揭示方法缺陷，下一版本必须新建method identity、重新development并使用新的confirmatory experiment，而不是在当前正式矩阵上迭代。

## 15. 产物目录

推荐结构：

```text
artifacts/
  unified-delta/
    <commit>/
      manifests/
      cuda-acceptance/
      development/
      formal/
        test_time_simple/
          training/{base,response_only,joint}/seed-*/
          evaluation/{base,response_only,joint,full,baselines}/
          calibration/
          causal/
          summary/
        test_time_wide/
          ...
      paper-summary/
```

每层保存run identity和source hashes，使任何summary都能追溯到原始checkpoint、config、伙伴和episode rows。
