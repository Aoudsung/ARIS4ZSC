# DEPI 研究计划

本计划只记录研究阶段与决策逻辑。所有注册数字、seed、预算、版本和门限都只链接权威合同，
避免产生第二个容易漂移的来源。

## 当前目标

在 OvercookedV2 Test-Time Protocol Formation 固定任务上，先证明当前实现忠实满足
[`SCIENTIFIC_SPEC.md`](SCIENTIFIC_SPEC.md) 和 [`METHOD_SPEC.md`](METHOD_SPEC.md)，再用
[`EVALUATION_SPEC.md`](EVALUATION_SPEC.md) 检验 B1 的结构增量、B2 的决策监督增量以及
held-out partner performance。机制归因必须闭合“合法历史—posterior/context—empirical
action ordering—真实 continuation—XP”链条。

## 阶段与决策

### R0：实现一致性

代码、配置、checkpoint/deployment schema、CLI、测试和 active 文档使用同一方法身份；删除
被取代的生成式训练状态、连续 latent、重复 optimizer 和不可达模块；通过静态禁词、compile、
全量 CPU 单测与 checkpoint/resume/deployment round-trip。

### R1：机械与 CUDA acceptance

按 [`FORMAL_EXPERIMENT_PROTOCOL.md`](FORMAL_EXPERIMENT_PROTOCOL.md) 运行 mechanical E2E 和
单 CUDA preflight。若 exact filter、joint likelihood、combined update、anchor continuation、
resume 或显存门任一失败，返回实现修复，不进入科学比较。

### R2：开发可证伪矩阵

执行 [`research/DEVELOPMENT_MATRIX.md`](research/DEVELOPMENT_MATRIX.md)。先看 B1−B0 是否支持
结构隔离，再看 B2−B1 是否支持 decision supervision；同时检查 K sensitivity、posterior
calibration、M1 和真实 continuation 机制读数。失败即记录相应 falsifier，不通过调整正式 seed
或报告口径挽救。

### R3：冻结与正式执行

只有 R0–R2 的预注册门完成、合同无歧义且仓库 clean committed 后才 freeze。正式运行之后不再
改变方法、依赖、配置、伙伴 panel 或统计。所有失败节点保留，所有 benchmark 结果发布。

### R4：claim 边界

formal claim report 自动消费 Official、Common-Partner、容量、资源、开发矩阵、calibration、
identifiability 和 recoverable-value artifacts。性能门与机制门分开；机制门失败时只撤回机制
措辞，不删除性能结果。B3 在真实 action-conditioned value-of-information 实现、测试和新
预注册完成前始终不进入结论。

## 停止规则

- 出现 F1+ 信息泄漏、partner lineage 重叠、标签使用学习价值、正式 seed 替换或 artifact
  伪造：该实验系列无效，停止汇总。
- 开发增量、calibration 或机制控制失败：接受反证，定位具体链路；不得直接扩大正式算力。
- 正式运行数值失败：按合同报告，禁止改变方法后续跑同一注册节点。
- 只有新假设、独立版本和新的事前合同才能启动下一实验系列。
