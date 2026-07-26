# ARIS4ZSC

本仓库研究伙伴回应能否在 OvercookedV2 零样本协作中通过改变后续动作选择产生价值。当前活动方法是 Path C V4.2。

## 先读

- `OPERATING_CONSTRAINTS.md`：执行权限、开发与正式结果的边界、实际数据预算和错误处理。
- `PROJECT_DASHBOARD.md`：当前实现、远端实验和下一步状态。
- `AGENTS.md` 与 `CLAUDE.md`：科研代码研发规则。
- `idea-stage/refine-logs/PATH_C_MODULE_DESIGN.md`：V4.2 数学对象到活动代码的唯一映射。
- `docs/status/EXPERIMENT_LOG.md`：历史实验和产物记录。

## 活动代码

- `src/path_c/`：模型、方法数学、训练、运行、评估和存储六个模块。
- `experiments/overcooked_v2/official_adapter.py`：锁定版本官方实现和 JaxMARL OvercookedV2 的局部适配。
- `experiments/overcooked_v2/path_c.py`：唯一命令入口，提供 `upstream`、`train` 和 `evaluate`。
- `experiments/overcooked_v2/configs/`：Test Time Simple 与 Test Time Wide 各一份配置。
- `experiments/overcooked_v2/tests/`：数学、模型、官方集成、训练、评估和数据六类行为测试。

旧 PyTorch、R015、Path C V1–V4.1、旧标准执行链、`src/aris_bellman` 和 toy factor game 已从活动工作树删除。它们仍可由 Git 提交 `f0ba51c`、历史文档和既有结果追溯。

## 命令入口

```text
python -m experiments.overcooked_v2.path_c upstream ...
python -m experiments.overcooked_v2.path_c train ...
python -m experiments.overcooked_v2.path_c evaluate ...
```

当前简化实现只完成静态代码，尚未运行新测试或实验。未经用户明确授权，不运行本地项目、远端测试、训练、评估或部署。
