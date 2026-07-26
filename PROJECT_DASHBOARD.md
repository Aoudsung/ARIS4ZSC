# PROJECT_DASHBOARD.md

最后更新：2026-07-26。本文件只记录当前状态；执行权限见 [OPERATING_CONSTRAINTS.md](OPERATING_CONSTRAINTS.md)。

## 项目目标

Path C 研究在标准 OvercookedV2 零样本协作评估中，伙伴回应是否能通过改变后续动作选择产生可测量价值。正式评估分别报告同一训练策略自我配对（self-play）和不同训练策略跨策略配对（cross-play）。

当前方法版本是 V4.2。它保留无标签伙伴槽、持续后验、双重动作价值估计、学习式回应码本、动作价值等价类别、回应使用与屏蔽模型、相对官方策略的 Kullback–Leibler 散度正则以及回应屏蔽三分支对照。

## 当前代码状态

| 项目 | 状态 |
|---|---|
| 重构前基线 | Git 提交 `f0ba51c`，提交说明为“记录 VQBC V4.2 重构前基线” |
| 活跃分支 | `codex/path-c-simplification` |
| 活跃源码 | `src/path_c/model.py`、`method.py`、`training.py`、`runner.py`、`evaluation.py`、`storage.py` |
| 运行入口 | `python -m experiments.overcooked_v2.path_c`，包含 `upstream`、`train`、`evaluate` 三个子命令 |
| 布局配置 | `path_c_simple.yaml` 与 `path_c_wide.yaml` |
| 实现状态 | `implemented`：静态重构已写入工作树，尚未运行本地或远端测试 |
| 旧执行代码 | PyTorch、R015、Path C V1–V4.1、旧标准链、`src/aris_bellman` 和 toy factor game 已从工作树删除，可由基线提交追溯 |
| 部署状态 | 未部署；不得覆盖当前远端旧版本实验，也不得恢复旧 checkpoint |

本次重构只改变工程结构、官方集成、状态组织、存储和测试方式。V4.2 的数学对象及四种部署模式保持在活跃实现中。详细映射见 [PATH_C_MODULE_DESIGN.md](idea-stage/refine-logs/PATH_C_MODULE_DESIGN.md)。

## 远端实验状态

- 家族级伙伴池旧版本的 Test Time Simple 已完成审计。完整技术报告见 [PATH_C_TEST_TIME_SIMPLE_RESULTS.md](docs/status/PATH_C_TEST_TIME_SIMPLE_RESULTS.md)。该实验发生过结果相关协议修订，只属于探索性结果，`scientific_readout_allowed: false`。
- 状态文档最后一次记录时，旧版本 Test Time Wide 仍在隔离远端目录运行，尚无最终审计。当前静态重构没有连接服务器、终止进程、修改产物或部署文件。
- 历史运行的预算、数值、路径和摘要保留在 [EXPERIMENT_LOG.md](docs/status/EXPERIMENT_LOG.md)，不在当前状态页重复展开。

## 标准评估口径

每个策略群体包含 10 个独立训练策略：

- 10 个对角自我配对；
- 90 个有向跨策略配对；
- 每个配对 500 个 400 步回合；
- 自我配对和跨策略配对分别以配对均值汇总，并报告跨配对标准差；
- 四种部署模式使用相同的配对和环境随机数；
- 回应屏蔽对照单独报告，不混入标准矩阵。

小规模运行只能验证接线。开发读数只能指导设计。正式读数仍需从完整原始行重算，并由用户作最终科学裁决。

## 下一步

唯一下一步是等待当前远端旧版本 Test Time Wide 完成或明确停止，并完成其最终审计。之后，在新的远端目录中对简化实现依次运行六类行为测试、官方接口比较、一回合环境交互、一次训练更新、Orbax 恢复和微型配对。新的科研训练或正式评估需要另行授权。
