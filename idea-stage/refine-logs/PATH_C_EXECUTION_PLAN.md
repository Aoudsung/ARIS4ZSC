# Path C V4.3 executable-response-value r1 执行计划

当前状态：代码修订完成；V4.3尚未在锁定远端环境训练或评估。十单元正式训练关闭。

## 入口

```bash
python -m experiments.overcooked_v2.path_c upstream ...
python -m experiments.overcooked_v2.path_c build-units ...
python -m experiments.overcooked_v2.path_c train ...
python -m experiments.overcooked_v2.path_c build-population ...
python -m experiments.overcooked_v2.path_c evaluate ...
```

## 机械验证

1. 安装锁定依赖并运行六个测试文件，要求零失败、零错误、零跳过。
2. 比较官方适配器与官方公开网络在固定非零carry输入上的下一carry、logits和value。
3. 运行一个400步环境回合，核对终止观测、自动重置和交付事件。
4. 执行一次完整训练更新，核对：
   - Q/outcome shapes；
   - TD-only responsibility；
   - behavior support概率；
   - 参数有限；
   - 伙伴参数无梯度；
   - 完整responsibility行数。
5. 保存并恢复真实`TrainState`，核对环境步、更新数、runner state、optimizer和codebook。
6. 执行一个微型标准配对和一个三分支回应屏蔽配对。
7. 验证action-independent Q平移不能触发policy-mediated gain。
8. 验证温度二分得到的batch mean KL接近登记目标。

任何失败直接保留异常并修复根因；不增加fallback、静默裁剪、`nan_to_num`或替代实现。机械验证不授权科研训练。

## 开发复跑

使用新目录重复：

- Test Time Simple；
- reference SP seed 100；
-模型seed 10101；
- 1,228,800环境步；
- 500个匹配评估seed；
- 500个回应屏蔽回合。

完整回读：

1. 每epoch TD responsibility、有效slot质量与bootstrap support；
2. response/reward/next-Q证据与TD responsibility的关系；
3. 六动作behavior coverage；
4. 实际batch KL和求解温度；
5. posterior entropy与value-class count；
6. expected next-policy TV；
7. executed-action policy-mediated gain；
8. A2-use/A2-mask首次动作、观测、回应码和奖励差异；
9. `posterior_use`相对`prior_only`、`reference_only`的匹配回报；
10. reference/self-play能力保持。

## 正式训练前判据

以下条件必须同时成立：

- responsibility由TD证据形成且跨回合可解释；
- next-policy TV显著超过数值噪声；
- 高policy-mediated gain样本中出现真实动作差异；
- 至少部分匹配回合出现奖励或最终回报差异；
- predicted gain与realized response effect具有正向校准；
-任务能力没有因support training或KL求解显著下降。

任一条件不成立时，不启动十单元正式训练。
