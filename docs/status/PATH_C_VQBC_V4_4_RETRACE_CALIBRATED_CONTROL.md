# Path C V4.4 Retrace-Calibrated Control Repair

**日期：** 2026-07-28  
**状态：** seed-100 开发训练、评估和冻结 checkpoint 反事实任务价值审计已完成；正式十单元训练关闭
**方法标识：** `path_c_v4_4_retrace_calibrated_control_r1`  
**科学读数：** `scientific_readout_allowed: false`

完整运行结果见
[`PATH_C_V4_4_SEED100_DEVELOPMENT_RESULTS.md`](PATH_C_V4_4_SEED100_DEVELOPMENT_RESULTS.md)。

冻结 checkpoint 审计恢复第 1,228,800 环境步参数，对 225 个真实触发状态分别执行 128 次
回应前配对延续，以及六动作、两信念分支、128 次回应后延续。最终 69 项远端测试全部通过。
预测下界分数的经验覆盖率为 29.33%，预测与经验动作排序的平均 Spearman 等级相关系数为
+0.01287；高分一半的经验回应前收益为 −0.01687，低分一半为 +0.00527。该结果说明当前
动作价值排序与不确定性分数尚不足以支持正式扩展，十单元训练继续关闭。

## 1. 触发证据

V4.3 完成了计划中的 Test Time Simple seed-100 开发运行。它证明回应通道已经能够改变实际动作：500 个回应屏蔽回合中 230 个出现动作轨迹差异，触发点下一策略总变差均值为 0.05450。任务收益仍未形成：回应使用效应为 −0.08，498/500 回合回报完全相同，2 个回合下降，0 个回合改善。四模式中 `posterior_use` 只比 `prior_only` 高 0.28，低于 `generic_response_information` 0.20。

此外，训练伙伴回报没有随 96 次更新改善，对部分伙伴末段回报仍显著为负。这说明 V4.3 已修复执行接线，但没有建立足以改善稀疏任务回报的长时控制价值。

## 2. 根因判断

V4.3 剩余失败由三个相互作用的对象构成：

1. **一步信用分配不足。** 400 步任务中的交付奖励无法稳定传回早期的伙伴识别和角色选择状态。
2. **训练支持缺少整回合一致性。** 每步均匀支持增加动作覆盖，却不形成可完成任务的连续替代控制序列。
3. **预测收益没有风险校准。** Mean policy-mediated gain能够改变动作，但高预测分箱与真实收益无单调关系，说明模型误差足以覆盖预测增益。

单一自我配对评估还存在能力饱和：参考策略平均回报约168，无法单独说明模型是否能适应训练支持中的外部伙伴。

## 3. 统一修订

V4.4 同时引入：

- full-episode off-policy Retrace；
- episode-persistent latent-expert exploration；
- small uniform action floor；
- outcome standard deviation驱动的lower-confidence control；
- executed-action policy-gain LCB触发；
- fixed official partner development panel。

TD-only responsibility、持续slot posterior、共同物理posterior、runtime-identical KL policy、完整记录和简化语义主干保持不变。

## 4. 版本隔离

V4.4 改变config version、TrainState、TransitionBatch、behavior policy、Bellman target、decision rows和response-contrast rows。V4.3 checkpoint与输出目录不能恢复或复用。下一次运行必须使用全新目录。

## 5. 远端验收顺序

1. 全部测试零失败、零错误、零跳过；
2. 一次训练更新及真实TrainState恢复；
3. 微型标准配对、回应屏蔽和固定伙伴panel；
4. 同预算seed-100开发训练；
5. 四模式自配对、固定伙伴panel和回应屏蔽校准。

十单元正式训练继续关闭，直至回应屏蔽出现正任务效应且固定伙伴panel显示可重复的后验使用收益。
