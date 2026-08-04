# ARIS4ZSC — DEPI

DEPI（Decision-Equivalent Protocol Inference）是面向 OvercookedV2 Test-Time
Protocol Formation 赛道的零样本协作方法。当前代码身份为：

- `METHOD_VERSION = depi_decision_consistent_evidence_gated_filter_v8`
- `CONFIG_VERSION = 18`
- `CHECKPOINT_SCHEMA_VERSION = 8`
- `MANIFEST_VERSION = 4`
- deployment bundle schema：`9`
- deployment artifact：`DEPI`

旧方法、checkpoint、优化器和训练产物不兼容；历史设计仅保存在
[`docs/legacy/`](docs/legacy/README.md)。

## 权威入口

实现和实验只能以以下四份文档为准：

1. [`docs/SCIENTIFIC_SPEC.md`](docs/SCIENTIFIC_SPEC.md)：问题、合法信息边界与可证伪主张；
2. [`docs/METHOD_SPEC.md`](docs/METHOD_SPEC.md)：模型、损失、anchor、状态与 artifact 契约；
3. [`docs/EVALUATION_SPEC.md`](docs/EVALUATION_SPEC.md)：开发矩阵、正式评估与机制归因；
4. [`docs/THEORY.md`](docs/THEORY.md)：理论保证和不能声称的内容。

固定正式协议见
[`docs/FORMAL_EXPERIMENT_PROTOCOL.md`](docs/FORMAL_EXPERIMENT_PROTOCOL.md)。

## 当前方法

```text
合法 ego 历史
  ├─ 当前 task-only 观测 ──> task GRU x_t
  ├─ 当前伙伴 semantic planes ──> 无记忆 instant encoder r_t
  ├─ 即时伙伴 planes 差 + ego 前一动作 ──> 4 维慢时标 capability u_t
  └─ 共享 response likelihood + evidence-gated sticky transition ──> 精确离散滤波 pi_t
                                                      └─> c_t = sum pi_t,k m_k

(x_t, r_t, u_t, c_t) ──> 单一 actor / dueling critic
```

B0–B2 在进入 task GRU 前清零显式 other-agent semantic planes，但伙伴造成的合法任务世界
后果仍可进入 task carry；代码不声称 task representation 与伙伴历史独立。即时伙伴几何只经
无 carry 的 `r_t` 进入执行。协议 carry 就是类别后验 `pi_t`，不存在额外 recognition GRU。
component 只称可交换 response regimes，不对应 SP、OP、SA、FCP 或协议真值。
每个 component 的 one-hot `z=k` 通过共享 critic 形成 `S_k`，其 posterior mixture 直接拟合
empirical action signature；final anchors 另外导出 response/action divergence、utilization、
collapse、actor ordering intervention 和跨 seed permutation-aligned stability。

`u_t` 每 16 步发布；四维全部直接预测同一合法 16 步窗口的可见率、可见移动率、可见持物率
和可见 inventory-change 率。anti-collapse 使用不读取 run ID 的 batch variance floor 与
covariance penalty，零向量不再是完整辅助目标的最优解。主 filter 使用 `p_stay=0.97`，没有
新可见伙伴证据时保持 posterior；开发敏感性注册 `0.90/0.97/0.99`。

训练目标为 PPO、联合 response NLL、capability semantic/anti-collapse、真实 continuation
decision supervision 和 matched-pair separation。反事实标签只使用 128 步 CRN raw-return continuation；不使用
学习到的端点值。actor 直接拟合由真实 all-action continuation 形成的动作分布。
训练、comparator、M1、BR-Prox 和机制评估共用同一折扣 continuation contract；decision
responsibility `q^A` 通过 KL 直接约束 posterior，另有 response-only-posterior 消融。辅助 actor
事务有独立精确 KL 门，超阈整次回滚；anchor 最大年龄固定为 4 个 outer updates。

## 开发矩阵

- R0：容量匹配的 full-observation recurrent PPO 强参照；
- B0：task-only recurrence + memoryless instant-partner `r_t`；
- B1：B0 加 capability、精确 response filter 和 likelihood；
- B2：加入合法 all-action decision supervision 与 matched-pair separation；
- B3：尚未实现；任何配置或报告都必须 fail closed，不得把它计作完成层级。

开发主矩阵在 `K=4` 执行 core 与 `R0-extra/B0-extra/B1-extra` 总预算对照；七项机制消融也只
在 `K=4` 执行，新增 `response_only_posterior`。`K=2/8` 只运行 B1/B2 sensitivity。所有格固定
seed index `0..9`，正式配置固定 `K=4`。跨 seed component diagnostics 在同一只读 comparator
validation history panel 上对齐。

## 环境与测试

正式运行要求 Python 3.10、JAX 0.4.38，以及来自 Official commit
`5ce1707cf31c1c115e6f6ba96db7bc9cc80a850e` 的 JaxMARL 和
`overcooked_v2_experiments`。仓库自带 `.venv` 时可执行：

```bash
source .venv/bin/activate
python -m compileall -q src/path_c experiments/overcooked_v2
pytest -q experiments/overcooked_v2/tests/test_depi_*.py
python -m experiments.overcooked_v2.path_c --help
```

最小机械链路：

```bash
python -m experiments.overcooked_v2.path_c mechanical-e2e \
  --config experiments/overcooked_v2/configs/depi_simple_mechanical_e2e.yaml \
  --partner-manifest /ABSOLUTE/PATH/partner_manifest.json \
  --require-cuda \
  --output runs/mechanical/depi
```

主要命令依次为：`upstream`、`build-partner-manifest`、`validate-manifest`、
`collect-pair-comparator-source`、`fit-pair-comparator`、
`train`、`build-depi-policy-manifest`、`calibrate-posterior`、
`evaluate-official`、`summarize-official`、`evaluate-identifiability`、
`evaluate-recoverable-value`、`evaluate-common`、`summarize-resources`、
`build-formal-claim-report`。开发矩阵使用 `run-development-matrix`、
`evaluate-development-matrix` 与 `summarize-development-matrix`。
伙伴覆盖使用 `build-decision-coverage-source`/`summarize-decision-coverage`，filter 敏感性使用
`run-protocol-sensitivity-matrix`。三个同期适应对照以诚实仓内 proxy 注册和评估，不冒充
CooT/RecBayes 等已发表方法复现，并机械绑定 DEPI 的训练 pool、interaction budget 与容量。
在正式耗时运行前，可用 `scientific-dry-run` 以两个 ego、少量 fresh partners 和真实 simulator
排练 calibration、development、identifiability、recoverable-value 与 claim-report artifact 链；
该链固定为非科学证据。

正式训练只接受 seed index `0..9`；`-1` 仅供工程运行。正式与 CUDA preflight
均要求单张 CUDA GPU，并在超过注册显存上限时 fail closed；preflight 必须传入真实冻结
comparator 与 reference ego checkpoint，覆盖 anchor、separation、aux、save/resume、deployment
和 fresh-final M1。当前仓库不把测试通过、
机械运行或机制读数当作 SOTA 结果；只有冻结协议下生成的原始正式矩阵才可进入主张。
