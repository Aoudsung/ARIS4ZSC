# 官方训练器接入说明

软件接线核查（Type-A）只验证源码接口、checkpoint、策略调用、环境运行和证据生成是否真实
可用，不训练正式策略，也不产生科学读数。2026-07-14 的远端核查已经确定六项接口；随后用户
在任何正式数据产生前修订参照身份和有效步数合同。独立 seed 999 的机械曲线验收尚未运行，
正式伙伴生产也没有启动。

## 已登记运行与状态

- 官方 `rnn-sp`：主体基线 seed 100，伙伴候选 seed 101、102；
- 官方 `rnn-op`：伙伴候选 seed 201、202；
- 全部登记 `test_time_simple`、名义 3000 万环境步、
  `indicate_successful_delivery: true` 和 `WANDB_MODE: disabled`；
- 主体基线 seed 100 不进入四个伙伴候选，也不与其共享训练运行标识；
- 五份配置的接口字段已经核对为 `verified`，`type_a_acceptance.status` 为
  `amended_pending_registered_reference_run`；
- `static_registered_not_frozen`、`scientific_readout_allowed: false` 和全部正式制品的
  `pending` 状态不变。

修订后的唯一网络参照就是 `overcooked_v2_experiments` 0.0.1 的 `ActorCriticRNN`：循环网络
之后的动作网络（actor）与价值网络（critic）隐层均使用 `orthogonal(sqrt(2))`，对应已发表
`145±22`。旧约 130
曲线只保留为诊断历史。正式花销前须以独立 `rnn-sp` seed 999 运行一次完整官方日程；最终
四分之一的原始回报均值须不低于 100，且不低于四个区间最高均值的 0.9。
正式生产前的读取器会从 457 个逐更新值重新计算四段均值和两个条件，并核对网络源码摘要；
自报的通过字段不能单独解锁生产。

`TOTAL_TIMESTEPS` 保持官方名义意图 30,000,000。有效步数回读合同按整数更新日程固定为：
`rnn-sp` 457×256×256 = 29,949,952，`rnn-op` 1831×64×256 = 29,999,104。配置用等值整数
`30000000` 表示名义意图，避免 PyYAML 把 `3.0e7` 读成字符串。

## 已确定的六项接口

1. **Hydra 配置。** 可编辑安装位于
   `/apps/users/cxw/Document/CodeSpace/Selfs/TG-SSA/external/overcookedv2_experiments/experiments/overcooked_v2_experiments`。
   官方配置使用 `+experiment=rnn-sp` 或 `+experiment=rnn-op`，以及
   `+env=test_time_simple`。`rnn-sp` 解析为 256 个环境、每次展开 256 步、4 个 epoch、
   64 个 minibatch、塑形在 1500 万步归零；`rnn-op` 保留官方 64 个环境和 0.02 熵系数，
   只把名义总步数和塑形期限覆盖为已登记的 3000 万与 1500 万。
2. **checkpoint。** 官方 `store_checkpoint` 原生使用 Orbax，目录为
   `run_0/ckpt_final`，根对象恰为 `config` 与 `params`，参数树路径为 `[params]`。因此没有
   创建 `ppo_main_with_checkpoint.py` 副本，也没有修改官方源码。
3. **循环策略。** 网络为 `ActorCriticRNN`，循环状态由 `initialize_carry` 初始化；
   `apply` 返回下一循环状态、分类分布和值。六个动作依次为
   `right, down, left, up, stay, interact`。动作规则
   `official_flax_categorical_actor_v1` 使用温度 1 分类采样。
4. **官方评估。** 根随机键先分为 rollout 与 reset；每步再分动作采样键和环境键，两个智能体
   各得独立采样键。原始团队回报取共享的 `reward["agent_0"]`；正确交付由
   `new_correct_delivery` 重算，错误交付和配方提示按钮事件由动作前状态与交互动作重算。
5. **源码集合。** 训练核、配置、网络、保存与恢复、官方评估、本仓适配器和 JaxMARL 环境的
   实际路径与 SHA-256 保存在
   `review_bundles/r015_official_type_a_20260714/type_a_round_trip_final3.json`。该记录还绑定
   正式参照网络与候选网络的同一源码摘要。
6. **Other-Play。** `rnn-op` 的环境配置含 `op_ingredient_permutations: [0, 1]`。64 次固定
   reset 实际观察到每个智能体的两种配料 0/1 置换，并观察到两个智能体独立抽样；配料 2
   保持不变。

官方 `ppo/main.py` 还会导入未使用于本批的行为克隆路径，而远端缺少其
`overcooked_ai_py` 依赖；`ppo/ippo.py` 同时使用脚本相对的 `models.rnn` 导入。本仓适配器不
修改官方树，而是直接组合同一 Hydra 配置、加入官方 `ppo` 目录以满足该相对导入，并调用
官方 `make_train`、网络、保存和评估函数。后续 seed 999 验收与正式生产的所有命令仍必须经
以下 CUDA 12 外壳执行：

```text
JAX_PLATFORMS=cuda,cpu CUDA_VISIBLE_DEVICES=<gpu> \
  bash experiments/overcooked_v2/scripts/with_jax_cuda12.sh \
  .venv/bin/python <已核对的本仓适配器入口>
```

## 远端核查结果

GPU 4 为 NVIDIA L40，当前与累计不可纠正 ECC 均为 0；JAX/JAXLIB 均为 0.4.38，GPU
后端和 CUDA 12.9 实际编译通过。初始化参数经 Orbax 保存和恢复后，
`path_c_flax_weights_sha256_v1` 摘要保持
`039279cd9e865cb8798816f7320f35797c7ad0c3567ce71bc850b8fa2ab11b66`。固定观察下，
官方策略与本仓包装器的动作、logits 和下一循环状态逐项相同；恢复后的 checkpoint 运行完整
400 步回合后，双方动作序列、原始回报和事件重算相同。

六个目标测试文件最终通过 155 项，0 失败、0 错误、0 跳过。两个只供接线核查的吞吐测量
分别实际执行两次：`rnn-sp` 每次 131,072 步，共 262,144 步；`rnn-op` 每次 32,768 步，
共 65,536 步。保守生产成本预测为 4.532100866750365 GPU 小时，低于 5.4 小时启动上限；
修订后的 seed 999 验收上限为 0.6 GPU 小时；与上述生产预测合计约
5.132100866750365 GPU 小时。没有运行 seed 999、五个正式训练或 `4×4×100` 联合准入。
