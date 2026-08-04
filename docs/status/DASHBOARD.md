# Unified DELTA-ZSC 状态面板

更新时间：2026-08-05。

本页只记录执行状态。方法、阈值、预算和结论边界以 `docs/PROTOCOL_INDEX.md` 指向的权威文档为准。

## 当前状态

| 轨道 | 状态 | 含义 |
|---|---|---|
| 科学方案 | 已统一重构 | 当前方案由joint response-decision latent model、analytic behavior statistics、learned transition、KL mirror policy与myopic VOI组成；旧多loss/comparator/separation链已退出active方法 |
| Active代码 | 已建立新路径 | `src/delta_zsc` 与 `experiments/overcooked_v2/unified_*` 已实现统一模型、训练、deployment、evaluation、calibration、causal intervention与三假设summary |
| 配置 | 已收缩 | 方法段只包含K、continuation horizon与adaptation KL budget；simple/wide formal、development和mechanical配置已注册 |
| 文档 | 已切换权威身份 | SCIENTIFIC/METHOD/THEORY/EVALUATION/FORMAL/MIGRATION与本索引已改为unified DELTA |
| CPU验证 | 待当前分支CI完成 | 已添加compile、core/model/training tests、CLI smoke与retired-token gate；尚不能把代码提交等同于测试通过 |
| CUDA acceptance | 未执行 | 必须在self-hosted单GPU上完成real environment/partner one-update run、checkpoint和deployment roundtrip |
| Development证据 | 未生成 | 尚无base/response-only/joint paired raw evaluation、K诊断或pilot variance结果 |
| Formal证据 | 未生成 | 尚无10-seed双布局、calibration、causal或三假设正式artifact |
| 论文主张 | 全部关闭 | 不能声称性能提升、decision emission有效、belief有因果价值或达到SOTA |

## 已消除的旧主路径

Active方法不再使用：

- learned capability GRU及四项正则；
- response/signature/rank/component/posterior/actor/separation多重loss；
- matched-pair comparator；
- context dropout；
- auxiliary actor transaction；
- fixed `p_stay` sensitivity；
- partner generator；
- 14变体development matrix；
- 10类独立claim gates。

旧代码仍可作为Official benchmark adapter或历史审计材料存在，但不能进入unified training/deployment graph。

## 当前可声称内容

仅可声称：

- 已将根因分析转化为新的统一概率模型与独立代码路径；
- 方法参数从多loss权重收缩为K、H、delta；
- 训练与部署边界在代码和文档中已重新定义；
- 已建立CPU/CUDA/原始评估接口。

不能声称：

- CI或CUDA已经通过；
- joint likelihood已经稳定训练；
- full优于baseline；
- H1/H2/H3成立；
- posterior已经校准；
- 当前结果达到ICLR/NeurIPS标准。

## 下一执行顺序

1. 完成当前分支CPU CI并修复全部真实错误；
2. 使用mechanical config和真实frozen partner执行CUDA acceptance；
3. 运行5-seed base/response-only/joint development；
4. 验证同seed base parameter fingerprints一致；
5. 运行joint K={2,4,8}受控诊断并冻结K；
6. 依据pilot variance决定是否将development扩展到10 seeds；
7. 冻结正式commit/config/manifests；
8. 执行双布局正式训练、raw evaluation、calibration与causal belief evaluation；
9. 只由raw artifacts生成H1/H2/H3 summary。

任何失败节点均原样保留，不得通过重新引入旧补丁链绕过。
