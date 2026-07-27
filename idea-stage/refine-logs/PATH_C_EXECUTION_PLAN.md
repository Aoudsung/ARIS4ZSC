# Path C control-memory r1 执行计划

当前状态：`implemented`，尚未在锁定远端环境运行。

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
