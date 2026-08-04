# DEPI 统计预注册摘要

本页描述推断结构，不复制注册的版本、seed、episode 数、bootstrap 次数、阈值或预算。所有
精确值以 [`EVALUATION_SPEC.md`](../EVALUATION_SPEC.md)、
[`FORMAL_EXPERIMENT_PROTOCOL.md`](../FORMAL_EXPERIMENT_PROTOCOL.md) 和两个 formal config 为唯一
来源；代码在加载时逐字段拒绝漂移。

## 1. 主结局与节点

主结局是固定 Official episode horizon 的 raw simulator return。正式主分析使用
`independent_run`：episode 先在 ego-run/partner-run/layout/role 节点内聚合，再对独立 run
节点推断。episode 不是独立正式样本；pooled episode 标准误和逐帧样本量不得用于 superiority。

Official、Common-Partner、capacity、BR-Prox、calibration 和 mechanism artifacts 都必须保留
原始 run lineage、checkpoint/manifest hashes 与 episode-key schedule。缺失节点、重复 lineage
或 hash 不一致时不做插补。

## 2. 主比较与多层结论

性能结论按布局分别报告，并使用权威规格中的单侧下置信界与物质效应双门。开发矩阵使用同
seed、同 episode keys 的 paired increment；正式不同方法若没有合法一一配对关系，则保持
independent-run inference，不通过任意排序制造配对。

结论分三层：

1. 描述性：均值、run-level dispersion 和完整 interval；
2. benchmark superiority：注册统计门通过；
3. protocol-mechanism attribution：benchmark 之外，开发、校准、identifiability、
   recoverable-value、容量和资源门在两个布局全部通过。

较高层失败不删除较低层结果。

## 3. Common-Partner 与 baseline 公平性

正式 comparator 由固定 Official commit 训练或从具有完整 lineage 的等价 Official artifact
取得；published table 只能作为外部 sanity check，不能代替 run-level nodes。所有方法使用同一
环境、role、episode keys、评估 horizon 和 raw-return 定义。FCP population 的训练成本、伙伴
资源与选模成本必须进入账本。

partner panel 按 mechanism family 和 parent run 留出。训练、comparator、calibration 与
confirmatory 的 run-disjointness 在 manifest 层校验，不在分析后删除困难伙伴。

## 4. calibration 与机制统计

calibration 使用 partner run 为 primary block、episode 为 secondary block 的层级重采样；
pooled score 只作描述。NLL、两个运动学 coverage 和 event Brier 同时进入 gate，不对成功子集
作选择性报告。

history shuffle、context swap 和 G1–G4 使用固定 checkpoint、matched current state 与共同
随机数 continuation。报告 unit 是独立 partner/ego run block；单状态、单 anchor 或 replica
不能被当作新的独立 run。M1 的多个 bootstrap members 是诊断路径，不是额外正式 seeds。

## 5. 缺失、失败与分析冻结

- numerical failure 保留为失败节点，不 restart、不替换；
- 不按观察到的分数删除 partner、seed、role、layout 或 episode；
- 不从多个 checkpoint 选择最佳者进入正式表；
- 不在结果后更换 bootstrap unit、单双侧检验、最小效应规则或 claim gate；
- 任何合同变更创建新方法身份和全新实验系列。

正式 summary 和 claim builder 只消费 schema-valid 原始 artifacts，并写出所有输入的 SHA-256。
手写布尔 gate、只含汇总均值的 CSV 或没有 lineage 的外部分数均不是正式证据。
