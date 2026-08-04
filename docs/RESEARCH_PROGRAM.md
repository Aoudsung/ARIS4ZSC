# Unified DELTA-ZSC 研究计划

本文件记录研究推进顺序，不重复权威文档中的公式、阈值或预算。

## 当前目标

在固定OvercookedV2 Test-Time Protocol Formation上，检验以下完整因果链：

```text
合法伙伴response
 -> joint latent belief
 -> posterior action value
 -> KL-bounded analytic policy change
 -> source-world continuation gain
 -> held-out XP
```

唯一核心结果仍是held-out benchmark performance。理论、calibration和机制测量用于解释performance，而不是替代performance或延迟端到端模型运行。

## 研究阶段

### A. 工程闭合

- active source compile；
- unified unit/integration tests；
- mechanical real-environment run；
- single-GPU CUDA acceptance；
- checkpoint/resume与deployment roundtrip。

这些结果只证明代码可执行。

### B. Development

- base、response-only、joint paired seeds；
- joint deployment同时评估joint与full；
- 验证同seed base parameter fingerprints一致；
- 完成decision-emission independent-replica诊断；
- 完成K={2,4,8}受控诊断；
- 估计run-level variance和资源。

Development保持完整端到端系统，不采用逐组件实现或“失败即永久阻断”的工程门。发现问题时优先修正joint probability model、数据合法性或analytic decision operator，禁止重新堆叠独立loss。

### C. 方法冻结

在查看confirmatory结果前冻结：

- commit；
- K/H/delta；
- configs；
- partner manifests；
- evaluation keys；
- H1/H2/H3统计；
- baseline集合；
-资源报告口径。

### D. Formal

- 双布局、10 seeds训练base/response-only/joint；
- raw Official/Common-Partner evaluation；
- joint bundle解析评估full；
- held-out calibration；
- causal belief evaluation；
- 三假设hierarchical summary；
- 完整资源与失败记录。

## 决策原则

1. **Performance优先。** 中间诊断只服务于理解和提高held-out XP。
2. **单一原理优先。** 新机制必须从joint latent model或KL-constrained decision problem推导。
3. **数据合法性优先。** 先修estimand、lineage、CRN和统计单位，再讨论网络结构。
4. **简单解释优先。** 同等性能下选择更少component、更小模型和更低成本。
5. **失败可发表性不是方法设计目标。** 不以“即使方法失败仍可写measurement paper”为中心组织当前项目。
6. **正式结果不可用于方法调参。** 失败后进入新method版本和新confirmatory实验。

## 论文目标

只有以下证据同时成立，才形成完整方法论文：

- H1：Full DELTA在强外部基线上取得物质性能增量；
- H2：joint response-decision training优于response-only；
- H3：正确belief在source-world干预下具有正因果决策价值。

论文贡献不以模块数量表达，而以统一模型、合法部署和实证闭环表达。
