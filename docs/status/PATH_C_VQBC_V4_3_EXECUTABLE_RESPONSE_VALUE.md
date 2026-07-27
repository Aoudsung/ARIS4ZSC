# Path C VQBC V4.3 — Executable Response Value Repair

日期：2026-07-27。状态：代码实现完成，尚未运行远端开发训练。结果权限：`scientific_readout_allowed: false`。

## 1. 修订依据

V4.2 control-memory r1 已证明：

- response code 与 Bayes posterior 接线有效；
- posterior 与 mask posterior 存在明显差异；
- control memory 能读取多步 ego 历史；
- use/mask 策略仍几乎相同，平均总变差仅 `1.85e-5`；
- 500 个回应屏蔽回合没有任务效应。

因此问题位于 latent slot语义、action support和回应条件的可执行动作价值之间。

## 2. 统一根因修订

V4.3 同时修改：

1. responsibility只由完整回合TD evidence形成；
2. outcome head从标量continuation改为下一动作Q向量；
3. use/mask内部next policy使用runtime同一KL算子；
4. 触发使用policy-mediated gain和next-policy TV；
5. 训练行为加入0.1均匀动作支持；
6. 温度由24次batchwise bisection求解；
7. response code编码centered-advantage delta；
8. 全量保存E-step responsibility与证据分量。

## 3. 不兼容边界

方法标识：

```text
path_c_v4_3_executable_response_value_r1
```

配置schema升级为3。旧模型tree、response codebook、TransitionBatch、checkpoint和evaluation rows不能恢复或拼接到V4.3。

## 4. 下一次运行

下一次只允许：

- 六类远端测试；
- 真实TrainState Orbax round-trip；
- 一个完整训练更新；
- 微型标准配对与回应屏蔽；
- 相同预算的seed-100 Test Time Simple开发复跑。

开发复跑未通过policy-mediated机制判据时，十单元正式训练继续关闭。
