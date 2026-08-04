# DEPI 状态面板

更新时间：2026-08-04。此页只报告状态，不复制权威合同中的版本、seed、预算或阈值，也不把
工程验证写成科学证据。

## 当前状态

| 轨道 | 状态 | 含义 |
|---|---|---|
| 方法实现 | 审查优化已接线 | R0/B0–B2 分层矩阵、七项同容量机制消融、total-budget controls、decision-consistent posterior、evidence-gated filter、无 run-ID 的 capability anti-collapse、共享 panel component diagnostics、task-matched comparator、fresh-final M1 和模块化 claim 均有 active code path |
| 本地验证 | 通过 | active source compile 和全量 DEPI tests 通过；CLI 及静态一致性门通过。本地 Python 3.13 结果只是工程验证，不替代注册 Python 3.10/CUDA 接受运行 |
| 三层执行链 | 已接线、未远程执行 | CPU Official/anchor/comparator 集成测试已接线；workflow_dispatch 自托管单 GPU preflight 与两 ego `scientific-dry-run` 编排已注册。当前状态不能替代远程 CUDA/完整 dry-run 的实际产物 |
| 开发证据 | 未生成 | 尚无可用于嵌套增量和总预算对照裁决的完整 raw-backed paired matrix artifact |
| 正式证据 | 未生成 | 尚无冻结协议下的 Official/Common-Partner/formal claim artifact |
| B3 | 未实现 | 明确 fail closed，不计作完成方法层级 |

## 当前 claim 边界

可以声称当前工作区已把审查问题转化为可执行结构与验证接口；不能声称 DEPI 已提升 XP、已超过
baseline、posterior 已通过校准或机制归因成立。历史探索结果只保留在 append-only ledger 与
legacy，不进入当前 claim。

## 下一步

1. 在注册运行时执行 [`../FORMAL_EXPERIMENT_PROTOCOL.md`](../FORMAL_EXPERIMENT_PROTOCOL.md)
   要求的 mechanical E2E 与 CUDA preflight；
2. 先采集并冻结 comparator，再执行 [`../research/DEVELOPMENT_MATRIX.md`](../research/DEVELOPMENT_MATRIX.md)，裁决嵌套增量、七项消融、K sensitivity 与总预算对照；
3. 仅在开发门完成、合同复核且仓库 clean committed 后冻结正式执行；
4. 无论机制门结果如何，都按 [`../EVALUATION_SPEC.md`](../EVALUATION_SPEC.md) 报告普通 XP。

权威入口见 [`../PROTOCOL_INDEX.md`](../PROTOCOL_INDEX.md)。
