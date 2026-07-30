# Path C V4.4 retrace-calibrated-control r1 执行计划

当前状态：代码修订完成；V4.4尚未在锁定远端环境训练或评估。十单元正式训练关闭。

## 入口

```bash
python -m experiments.overcooked_v2.path_c upstream ...
python -m experiments.overcooked_v2.path_c build-units ...
python -m experiments.overcooked_v2.path_c train ...
python -m experiments.overcooked_v2.path_c build-population ...
python -m experiments.overcooked_v2.path_c evaluate ...
python -m experiments.overcooked_v2.path_c evaluate-panel ...
```

## 机械验证

1. 在锁定依赖环境运行全部测试，要求零失败、零错误、零跳过。
2. 核对官方适配器与官方公开网络在非零carry输入上的下一carry、logits和value。
3. 运行一个400步环境回合，核对终止观测、自动重置和交付事件。
4. 执行一次完整训练更新，核对：
   - Retrace target有限且shape正确；
   - `lambda=0`与一步target一致；
   - behavior probability、importance ratio和trace coefficient可回算；
   - TD-only responsibility与最小TD能量一致；
   - outcome uncertainty、LCB与terminal zero-continuation成立；
   -伙伴参数无梯度；
   -完整responsibility行数。
5. 保存并恢复真实`TrainState`，核对环境步、更新数、runner state、optimizer、codebook和episode-persistent behavior expert。
6. 执行一个微型固定伙伴 panel；该检查只核对真实伙伴加载、四种部署模式、状态重置、完整记录和匹配随机数，不读取科研表现。
7. 验证动作共同价值平移不能产生policy-gain LCB。
8. 验证温度二分得到的batch mean KL接近登记目标。

任何失败直接保留异常并修复根因；不增加fallback、静默裁剪、`nan_to_num`或替代实现。机械验证不授权科研训练。

## Seed-100 开发复跑

使用全新目录和V4.4 schema，保持：

- Test Time Simple；
- reference SP seed 100；
- model seed 10101；
- 1,228,800环境步；
- 500个匹配四模式评估seed；
- 500组回应屏蔽回合。

同时增加固定伙伴panel：ego checkpoint分别与训练单元登记的SP/OP final partners运行四种模式，所有模式使用匹配seed。

完整回读：

1. 每epoch TD responsibility、有效slot质量与bootstrap support；
2. mean importance ratio、trace coefficient及其时间分布；
3. 后段奖励通过Retrace传播到早期状态的target变化；
4. 六动作behavior coverage及每个episode的固定expert；
5. batch KL、求解温度与reference能力；
6. outcome mean/std、policy gain uncertainty与LCB覆盖；
7. expected next-policy TV；
8. executed-action mean gain、LCB与realized response effect的分箱校准；
9. A2-use/A2-mask首次动作、观测、回应码和奖励差异；
10. self-pair和固定伙伴panel中的`posterior_use-prior_only`；
11. 训练伙伴回报是否随更新改善。

## 正式训练前判据

以下条件必须同时成立：

- responsibility继续由TD evidence形成，且没有重新退化为单槽或随机槽；
- Retrace target相对一步target产生可解释的长时信用传播；
- coherent exploration提高替代动作覆盖，但reference/self-play能力没有显著下降；
- policy-gain LCB与realized response effect具有正向校准；
- 回应屏蔽至少出现可重复的正任务效应，而非只出现动作差异；
- 固定伙伴panel中至少部分伙伴的`posterior_use`稳定优于`prior_only`；
- 通用回应信息不系统性优于decision-focused posterior control；
- 训练伙伴回报不再长期停留在显著负值。

任一条件不成立时，不启动十单元正式训练。
