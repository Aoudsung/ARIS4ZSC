# Path C control-memory r1 执行计划

当前状态：锁定远端环境的机械验证和 seed-100 开发复跑已完成；正式训练仍关闭。

## 入口

```bash
python -m experiments.overcooked_v2.path_c upstream ...
python -m experiments.overcooked_v2.path_c build-units ...
python -m experiments.overcooked_v2.path_c train ...
python -m experiments.overcooked_v2.path_c build-population ...
python -m experiments.overcooked_v2.path_c evaluate ...
```

## 首次机械验证

1. 安装锁定依赖并运行六个测试文件，要求零失败、零错误、零跳过。
2. 比较官方适配器与官方公开网络在固定非零carry输入上的下一carry、logits和value。
3. 运行一个400步环境回合，核对终止观测、自动重置和交付事件。
4. 执行一次完整训练更新，核对参数有限、伙伴无梯度、记录行数完整。
5. 保存并恢复真实 `TrainState`，核对环境步、更新数、runner state、optimizer和codebook。
6. 执行一个微型有向标准配对和一个三分支回应屏蔽配对。

任何失败直接保留异常并修复根因；不增加fallback、静默裁剪或替代实现。机械验证不授权科研训练。

## 开发复跑

机械验证通过后，使用新输出目录重复seed-100、1,228,800步开发训练。回读：

- responsibility质量和有效slot数；
- posterior熵与posterior-to-action变化；
- use/mask policy TV；
- policy-level predicted raw effect与真实A2-use/A2-mask回报差；
- `posterior_use`相对`prior_only`、`reference_only`的匹配结果。

机制未成立时，不启动十单元正式训练。

## 2026-07-27 执行结果

- 六个目标测试文件共 42 项通过，0 失败、0 错误、0 跳过。
- 真实 `TrainState` 已完成 Orbax 保存、类型保持恢复和完整训练恢复。
- 开发训练从产物回读 1,228,800 个环境步、3,072 个完整回合和 96 次更新。
- 四种部署模式各完成 500 个匹配 seed 的单策略自配对诊断，共 800,000 个环境步；该入口
  明确标记为开发诊断，不生成标准自我配对或跨策略配对统计。
- 500 个回应屏蔽对照回合完成 600,000 个分支环境步。全部回合触发，但实际
  `A2-use − A2-mask`、`A1 − A2-mask` 和 `A2-use − A1` 平均值均为 0。
- 最终 rollout 的平均有效后验槽数为 1.23，平均活动动作价值等价类别数为 1.60，
  use/mask 策略总变差均值为 `1.85e-5`。回应屏蔽后验的平均 L1 距离为 0.338，但只有
  4/500 个回合出现左侧动作差异，且没有回报差异。
- 当前记录没有保存责任分配本身，因此责任分配质量登记为“未采集”，不生成看似完整的替代值。

由此，control memory 的软件接线通过，但回应信息仍未对实际任务回报形成可测量增量价值。
按本计划的既定条件，不启动十单元正式训练。
