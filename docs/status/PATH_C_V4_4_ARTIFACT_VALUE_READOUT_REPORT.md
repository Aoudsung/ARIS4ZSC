# Path C V4.4 真实回报读出审计

日期：2026-07-29
读出版本：`path_c_v44_artifact_readout_v2`
裁决：**INCONCLUSIVE**

## 1. 结论先行

现有 403,200 条冻结 continuation 尚不能证明当前注册回应通道存在稳定的增量任务价值，也不能证明该机会不存在。这里的“真实回报”特指环境延续实际产生的 simulator raw return，而不是模型 Q 标签或现实世界回报。以状态、合法冻结历史 carry 和可见 response code 为输入的主读出器，在四折 leave-one-partner-out（LOPO）上的净 lift 为 `−0.3597`，单侧 `LCB95/UCB95` 为 `−0.9326/+0.0852`；split-replica oracle 为 `−0.0135`，单侧 `LCB95/UCB95` 为 `−0.1649/+0.1399`。上下界跨越 0，因此既不满足 GO，也不满足两个 NO-GO 条件。

这项裁决只条件于当前四个固定伙伴、当前 178 个固定伙伴触发状态、当前 response code 和六动作候选集合。它不支持启动 V4.5 或任何新 RL 训练；它也不能外推为 Test Time Simple 上所有主动探查都无价值。

本审计与“跨独立训练运行是否存在可恢复的策略兼容性”是两个不同层次的科学对象。旧矩阵的 `119.65` 完全来自四个同 checkpoint 对角格；这是由独立策略矩阵报告建立的先前结论，不由本次 readout 重新估计。本次结果不改变、也不重新包装这一结论。

## 2. 冻结数据与完整性

- 触发状态：225 个，其中固定伙伴 178 个、自配对 47 个。
- 固定伙伴触发数：37、60、17、64；聚合时先求伙伴内均值，再对四个伙伴等权。
- 每状态 128 个共同随机数 replica。每个 replica 含一对 pre-response use/mask 延续，以及六个 post-response 强制动作各自的一对 use/mask 延续，共 14 行；总计 `225 × 128 × 14 = 403,200` 行。
- SHA-256 排序把每个 `(trigger_id, replica_index)` 固定拆为 64 个 fit replica 和 64 个 evaluation replica；同一 replica 的所有动作与 use/mask 分支始终同侧。
- 正式运行前后五个输入文件的 SHA-256 完全一致，`frozen_inputs_unchanged=true`。

输入哈希如下：

| 文件 | SHA-256 |
|---|---|
| `continuations.parquet` | `9b7fbe66e1c25064a2dded4b85d7927fc3686c5707f78891230bbaace12309f1` |
| `reconstruction.parquet` | `6e968e1040ff8aaad25363009894ef1a6dcdf07b52b4c1899d4d797526e945cf` |
| `trigger_values.parquet` | `d46dc9d1d25231659956ad710467cf1e12167542b63cc844312ab1981b5bce9e` |
| `summary.json` | `2e0ec1a6abce7fbb3ffbac4573d253f08c452bce66ffcad0facc985d9e019958` |
| `run_metadata.json` | `75f8c8897c885d6485410c4ae0dd5d26841cbf017bfa953121af8f468f9ee3af` |

## 3. 估计设计

主尺度是 evaluation-half 的原始剩余回报。读出器先用 `q_mask(x,a)` 估计屏蔽回应时的六动作价值，再用 `delta(x,r,a)` 估计配对的 use-minus-mask 差值，令 `q_use=q_mask+delta`。外层每次留出一个完整伙伴；标准化、零方差删除、response code 词表和 ridge 正则选择只读取另外三个伙伴。正则通过训练伙伴内部 LOPO、以伙伴等权的六动作 MSE 选择。

两个预声明特征层同时运行：

- 状态层：agent-0 官方 post-observation、归一化触发时刻、触发动作与候选动作。
- 主历史层：状态层加冻结 checkpoint 的 `reference_carry`、`trainable_carry`、上一动作和上一团队回报；use 分支额外读取可见 response code。

伙伴身份、seed、SP/OP 标签、`control_carry`、slot、模型 Q/LCB、完整环境状态、特权任务阶段、未来 continuation 字段与测试折统计量都不进入模型。主历史层共有 1,240 个展开特征，状态层为 982 个；候选动作以六个 action-specific 输出索引表示。

统计边界使用 seed `20260729` 的 10,000 次伙伴分层、触发状态级 bootstrap。每次重采样重新执行标准化、内层选参和四折 LOPO。第 5 与第 95 percentile 分别报告为单侧 95% LCB 与 UCB；它们条件于当前四伙伴面板和已经固定的 64 个 evaluation continuation replica，不包含对 continuation Monte Carlo 样本的第二层重采样，不是伙伴总体置信边界，也不应合称为双侧 95% 区间。因而本次 INCONCLUSIVE 也只在这一条件范围内成立；尚未执行两层 bootstrap 敏感性分析。

## 4. 主结果

| 量 | 点估计 | LCB95 | UCB95 |
|---|---:|---:|---:|
| 主历史读出 `L_probe` | −0.3597 | −0.9326 | +0.0852 |
| split-replica `L_oracle` | −0.0135 | −0.1649 | +0.1399 |
| 当前触发动作 `tau_response` | +0.0499 | −0.1795 | +0.2715 |
| 状态层读出 `L_probe` | −0.1601 | −0.3694 | +0.2420 |

四伙伴主历史读出如下：

| 伙伴 | 状态数 | `L_probe` | LCB95 | UCB95 | `L_oracle` | `tau_response` |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 37 | −0.1858 | −0.4983 | +0.2200 | −0.0760 | +0.1182 |
| 1 | 60 | −0.0469 | −0.1302 | +0.0573 | +0.0208 | +0.0052 |
| 2 | 17 | −0.9375 | −1.8015 | +0.4963 | −0.0184 | +0.1103 |
| 3 | 64 | −0.2686 | −2.4612 | +0.3613 | +0.0195 | −0.0342 |

没有伙伴的区间给出稳定正或稳定非正方向，也不存在“一个伙伴 LCB95 大于 0、另一个伙伴 UCB95 不大于 0”的登记冲突。伙伴 2 只有 17 个状态，伙伴 0–3 的区间宽度差异也表明当前面板对伙伴异质性的分辨力有限。

## 5. 辅助读数

- 同伙伴乐观读出：`L_probe=+0.0837`。四个伙伴点估计均为正（`+0.1436, +0.0104, +0.0735, +0.1074`），但这不是跨伙伴主裁决。
- 47 个自配对状态的独立描述性读出：`L_probe=+0.0066`、`L_oracle=+0.0532`、`tau_response=+0.0532`；不进入主裁决。
- 主历史 LOPO 的 discounted-return probe lift 为 `−0.0560`；正确交付 lift 为 `−0.0064`，错误交付 lift 为 `+0.0115`。
- use 与 mask 动作分别和 split-replica oracle 一致的比例为 18.73% 与 17.15%，两侧同时一致为 13.70%。这些只用于解释，不替代原始回报裁决。

同伙伴点估计略正而 LOPO 点估计为负，说明跨伙伴支持不足可能是当前读出失败的一部分；但 oracle 自身也围绕 0 未决，因此不能把结果简化为“机会明确存在，只是模型不会外推”。

## 6. 登记裁决的逐项应用

1. `UCB95(L_oracle)=+0.1399>0`，不满足 oracle 无机会的 NO-GO。
2. `LCB95(L_oracle)=−0.1649≤0`，不满足“oracle 明确为正而 probe 明确不正”的 NO-GO。
3. `LCB95(L_probe)=−0.9326≤0`，不满足 GO。
4. 因此按预先固定顺序裁决为 **INCONCLUSIVE**；显著伙伴冲突为 false。

未决来自三个可见限制：178 个固定伙伴触发状态在伙伴间高度不均衡、partner-2 只有 17 个状态、而主问题要求从三个伙伴外推到第四个。下一步不应通过训练新控制器来掩盖这个不确定性。

## 7. 对项目决策的影响

- V4.4 保持冻结，不启动 V4.5、新 seed 或超参数实验。
- 当前注册回应通道与当前触发/候选分布没有得到正价值证书；也未达到限定范围内的 NO-GO 证书。
- 若仍把注册回应通道作为旁支问题，下一次测量应增加独立触发状态和伙伴支持，并继续使用真实配对回报；在区间收窄前不训练控制器。
- Path C 的主科学入口仍是 run-disjoint 的策略兼容性机会测量。它必须使用彼此独立的模式训练运行和伙伴训练运行，不能回到由同 checkpoint 对角格产生的 `119.65`。

## 8. 产物与复跑

正式机器产物位于远端：

`/apps/users/cxw/Document/CodeSpace/Selfs/CPR_REPO/.codex_remote_validation/path_c_v44_artifact_readout_v2_20260729`

独立复跑产物位于：

`/apps/users/cxw/Document/CodeSpace/Selfs/CPR_REPO/.codex_remote_validation/path_c_v44_artifact_readout_v2_20260729_reproduction`

工作区只读副本位于：

`artifacts/path_c_v44_artifact_readout_v2_20260729`

其中包含配置、输入前后哈希、实现源码与测试文件哈希、运行环境版本、输出哈希清单、split manifest、逐状态 fit/evaluation 均值与预测/选择、逐折与逐伙伴结果、10,000 次 bootstrap 分布、最终裁决和复跑命令。正式运行使用独立输出目录，未改写原始反事实 artifact。

正式运行与独立复跑的 9 个确定性产物逐字节一致；两次运行均记录
`frozen_inputs_unchanged=true` 与 `implementation_unchanged=true`。远端 CPU-only 全量测试共
78 项（含新增读出模块的 9 项），全部通过；CPU-only 用于避开服务器 CUDA/PTX 工具链的
收集期兼容性错误，不改变审计数据、实现或正式产物。

## 9. 独立完整性复核

独立只读 reviewer 对 ground-truth 来源、尺度归一化、结果存在性、CLI 可达性、证据范围、
输入与实现不变性、逐字节复现、数值复算和登记裁决顺序逐项复核，最终结论为 **PASS**，
evaluation type 归类为 `simulation_only`。它从逐状态产物独立复算了伙伴等权的
`L_probe`、`L_oracle`、`tau_response` 以及 10,000 次 bootstrap 分位数，均与机器摘要完全
一致。复核没有发现实质性完整性问题；完整记录见仓库根目录的 `EXPERIMENT_AUDIT.md` 与
`EXPERIMENT_AUDIT.json`。

审计边界不因 PASS 而扩大：这些结果仍只条件于当前四伙伴面板和固定 continuation 样本；
独立 reviewer 使用的是可用的 GPT-5.6-sol xhigh 只读代理，因为技能首选的 GPT-5.5 路由
在本环境不可用。
