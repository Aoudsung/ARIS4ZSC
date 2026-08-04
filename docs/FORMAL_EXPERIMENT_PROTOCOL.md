# FORMAL_EXPERIMENT_PROTOCOL：冻结正式实验合同

`authoritative: true`

本文件固定 confirmatory execution。与
[`SCIENTIFIC_SPEC.md`](SCIENTIFIC_SPEC.md)、[`METHOD_SPEC.md`](METHOD_SPEC.md) 或
[`EVALUATION_SPEC.md`](EVALUATION_SPEC.md) 冲突时不得运行；必须先在未观察正式结果的状态下
同步修订合同、代码、配置和测试，并在证据账本记录原因。

## 1. 固定软件与硬件边界

- Python 3.10；JAX 0.4.38；
- JaxMARL 与 `overcooked_v2_experiments` 必须从 Official commit
  `5ce1707cf31c1c115e6f6ba96db7bc9cc80a850e` 以 `--no-deps -e` 安装；
- 正式入口只接受 clean、已提交、可解析为完整 40 位 Git SHA 的 checkout；
- `JAX_PLATFORMS=cuda`、`JAX_DEFAULT_MATMUL_PRECISION=highest`；
- 每个 worker 的 `CUDA_VISIBLE_DEVICES` 必须只指向一个已登记 physical GPU；JAX 必须实测
  只暴露一个 GPU device；
- worker 起始显存占用不超过 1,024 MiB、起始利用率不超过 10%、无 volatile uncorrectable
  ECC error；
- CUDA preflight 的峰值显存必须严格小于 40,000 MiB。

CPU 单元测试、伪造设备元数据、可见 GPU 但实际 CPU backend，均不能代替正式 acceptance。

## 2. 固定随机化与失败政策

正式训练 seed 是 index 0–9。每个 key 精确等于 `split(PRNGKey(42), 10)[index]`，并由代码
分出 rollout、partner、anchor-fit、anchor-evaluation、comparator、M1、calibration 等互不
别名的 named domains。index `-1` 只属于 engineering，正式 manifest、run identity、summary
和替代运行中均禁止出现。

Official evaluation root seed 固定为 0。相同 paired contrast 必须共享 episode-key schedule，
fit/evaluation continuation 必须使用不同 key 域。

正式数值失败按原样保留并报告。不得以改变参数、缩小预算、重新抽 seed、从 checkpoint
重试、换用 SP checkpoint 或挑选较好 restart 的方式替换失败节点。

## 3. 固定环境、模型与预算

两个正式配置为：

- `experiments/overcooked_v2/configs/depi_simple_formal.yaml`；
- `experiments/overcooked_v2/configs/depi_wide_formal.yaml`。

环境固定 400 steps、local view size 2、successful-delivery indicator、negative rewards、随机
agent positions 和 delivery 后重新采 recipe。主方法 B2 固定 K=4。

每个 DEPI seed 使用 256 environments、29,949,952 simulator steps、256-step recurrent
rollout、4 epochs、64 minibatches。PPO 固定 learning rate 2.5e-4、gradient clip 0.25、
gamma 0.99、GAE lambda 0.95、policy/value clip 0.2、entropy 0.01、value weight 0.5、5%
warmup、线性 annealing 和 Adam epsilon 1e-5。完整损失、anchor 和 calibration 参数以正式
配置及其加载时 fail-closed 校验为准。

Official upstream SP 每 run 30,000,000 steps、OP 每 run 50,000,000 steps；训练池使用注册
checkpoint stages 0.0、0.5、1.0。所有 DEPI、upstream、baseline、continuation 与 evaluation
simulator transitions 都进入资源账本。

## 4. partner lineage 与隔离

正式 manifest schema 为 3。每个资源必须包含 layout、role、mechanism、hyperparameter
family、stage、seed/run lineage、checkpoint path 与 SHA-256。每个 DEPI owner seed 恰有一个
独立 owner-SP source；support、comparator fit、comparator validation、posterior calibration
和 confirmatory runs 按 manifest 规则互斥。

schema 3 的角色名固定为 `owner_source`、`development_support`、`comparator_fit`、
`comparator_validation`、`calibration`、`confirmatory`。正式 comparator fit 与 validation
各至少使用两个 fresh、独立、final-stage SP/OP parent runs；不得复用 Official seed index。
首次冻结前的 `6/8 : 1/8 : 1/8` support/fit/validation lane 分配以及冻结后只在 episode
boundary 回收 comparator lanes 的规则由 [`METHOD_SPEC.md`](METHOD_SPEC.md) §6 固定。

正式训练池由 SP/OP 各 10 个独立 parent runs 的三个阶段，以及两个独立 final-stage OP
width variants 构成。heuristic family 只进入 Common-Partner 测试。任何 hash 缺失、文件变化、
role 重叠、另一 owner seed 的资源混入或 fresh run 回用均 fail closed；正式运行禁止
`--skip-manifest-hash-check`。

## 5. 冻结前执行门

在创建正式结果前必须依次完成：

1. Python 3.10 环境安装与依赖一致性检查；
2. active source compile、全部 `test_depi_*.py`、CLI smoke 和 legacy-token CI gate；
3. mechanical end-to-end；
4. 单 CUDA `cuda-preflight`，覆盖一次真实 compiled rollout/update、checkpoint round-trip、
   deployment round-trip 和峰值显存门；
5. B0–B2 × K={2,4,8} 开发矩阵及 paired summary；
6. 只在未观察正式结果时完成方法/合同 freeze，并提交 clean commit。

任一门失败则不得开始正式 confirmatory runs。B3 保持 `not_implemented`，不构成冻结条件。

## 6. 正式执行顺序

对两个布局和所有注册 seeds/panels，产物链为：

```text
upstream
  -> build-partner-manifest / validate-manifest
  -> cuda-preflight
  -> train
  -> build-depi-policy-manifest
  -> calibrate-posterior
  -> evaluate-official / summarize-official
  -> evaluate-common-br-prox / evaluate-common
  -> evaluate-identifiability
  -> evaluate-recoverable-value
  -> summarize-capacity-control
  -> summarize-resources
  -> build-formal-claim-report
```

`audit-signals` 是审计读数，不得改写训练结果。每个 stage 写独立目录、完整 stdout/stderr、
run identity、输入 hash 和 resource ledger；后续 stage 只读取已冻结 artifact，不重新训练或
静默补齐字段。

## 7. 评估矩阵与统计

Official 每个 ego/partner/layout/role pairing 固定 500 episodes。正式主比较、Common-Partner
panel、capacity control、BR-Prox、posterior calibration、identifiability、recoverable-value
和 B0–B2 开发增量必须满足 [`EVALUATION_SPEC.md`](EVALUATION_SPEC.md)。正式 scoreboard
固定 9,999 次 node bootstrap、单侧 95% LCB、独立 run 推断，并同时检查 20-point point-estimate
物质效应。

每个 DEPI formal ego run 还必须在最终 policy update 后执行 M1；其 model fingerprint
必须与 deployment bundle 一致。DEPI policy manifest 必须绑定两个布局各 10 个
M1 artifact 的路径和 SHA-256，formal claim report 必须从 Official raw identity 回溯并
重新校验。任一 seed 未通过只锁住机制归因，不得中止或替换训练结果。

不得只发布均值。原始 per-episode returns、run-level nodes、置信界、失败节点、资源和 lineage
均须保留。两个布局中任一机制门失败，formal claim report 必须锁住机制归因，但仍发布所有
benchmark 结果。

## 8. 合同变更与结果有效性

冻结后，本文、正式配置、权威 method/science/evaluation spec、源代码和依赖 commit 构成一个
不可拆分合同。任何改变都创建新方法版本和新实验系列，不得与旧节点拼接。contract-doc 修改
必须与描述的代码/配置处于同一 commit，并把原因追加到
[`status/EVIDENCE_LEDGER.md`](status/EVIDENCE_LEDGER.md)。
