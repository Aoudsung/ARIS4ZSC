# U2_MODE_FILTER_SPEC — 持久协议模式信念 BayesModeFilter（设计规格，送 codex 评审）

**Status:** DESIGN — 未实现；与 S27 修复 diff、U1 规格同批送 codex 评审后实施
**Ledger:** U2（预注册 METHOD_LOCK sec18.11；计划 ICLR_UPGRADE_PLAN §2-C2）
**Date:** 2026-07-04

## 0. 动机（S27 的深层教训）

S27 暴露的不只是 bug：**瞬时选项是错误的推断潜变量**。提案理论（§5.2）的潜变量是
episode-持久的因子模式 z_f，实现却经由"推断伙伴瞬时选项 → 路由成证据 → GRU 隐态"
的间接链。瞬时选项支撑集随状态变化（S27 冻结类缺陷的温床）、生命周期数步、语义靠
下游自由 GRU 重新学习——不可解释、样本饥饿、与理论错位。

**U2 = 用结构化 Bayes 递归替换自由 GRU 递归**：信念原生活在模式层，持久性/归一性/
支撑完整性**按构造成立**（模式集固定有限，永不"无效"——S27 类缺陷结构性不存在）。

## 1. 数学设计

每因子 f 维护模式后验 b_t(z_f) ∈ Δ^{K_f}，递归：

```
b̃_t(z)   = (1-λ)·b_{t-1}(z) + λ/K_f                    # 遗忘/混合底（原理化处理
                                                        #  proposal §5.9 ρ-有界自适应伙伴）
b_t(z)   ∝ b̃_t(z) · exp( g_θ(x_t^f, e_f, z) )          # 学习的逐模式对数似然
```

- `x_t^f`：该因子的路由证据行（现有 64 维，行为-only，post-P1）；
- `e_f`：因子嵌入（现有 factor_features）；
- `g_θ`：小共享 MLP（输入 [x; e_f; mode-embedding]，输出标量 log-lik）——**跨因子共享**
  参数（同现有 belief 模型的共享模式）；
- λ（`mode_forgetting`，默认 0.02，config 化）：与 S27 的 support_mix 同构——在模式层
  它同时就是自适应伙伴的漂移先验，一个机制两个语义（论文卖点之一）。

**训练：仍是单一 TD 损失**（I2 不破）。b_t 是 g_θ 的可微函数（softmax-normalized 递归），
TD 梯度经 belief→Q 路径回传穿过递归——结构等同 P4 现状（截断于窗口基隐态），只是
递归单元从 GRUCell 换成 Bayes 更新。无辅助损失、无监督模式标签（I1/I8 不破）。

## 2. 接口契约（与现有 P4 机制完全对齐）

`BayesModeFilter(nn.Module)` 作为 `FactorLocalBeliefModel` 的同接口姊妹类
（`src/aris_bellman/factor_belief.py`）：

| 接口 | 语义映射 |
|---|---|
| `initial_hidden(B, F, device)` | 返回均匀 **log-posterior** `[B,F,K_max]`（"hidden"=对数后验——直接复用 P4 全部持久化/replay/快照机制，`EvidenceBuffer._belief_hidden`、`belief_window_base_snapshot`、`OptionTransition.belief_hidden_t/next` 形状协议不变，仅第三维从 hidden_dim 变 K_max —— 见 §3 形状适配） |
| `step_history(evidence, hidden, ...)` | 一步 Bayes 递归（含 λ-混合 + active_mask 直通语义同现状） |
| `encode_history(seq, initial_hidden, masks...)` | 窗口重放递归（TD 侧，梯度可通，零填充掩码语义同 P4/S2） |
| `forward` / `belief_from_hidden(hidden)` | `softmax(log_posterior)`（+ 现有 mode_mask 掩码归一） |

下游 `factor_q.py`（centered-belief advantage、relevance 路由）、`_state_repr`、
`_advance_persistent_belief`、eval 共享路径**零改动**——只在 `_build_belief_model`
按 config 分派。

## 3. 精确改动点

1. `src/aris_bellman/factor_belief.py`：新增 `BayesModeFilter` 类（~120 行）。
2. `train_aris.py::_build_belief_model`：读 `training.belief_filter: gru(默认)|bayes_mode`
   分派构造；hidden 第三维按类别取 hidden_dim 或 K_max（`EvidenceBuffer` 的
   `set_belief_hidden` 形状校验读模型的 `hidden_dim` 属性——BayesModeFilter 暴露
   `hidden_dim=K_max` 即可兼容，其余机制不感知差异）。
3. config：`belief_filter: gru` 显式入三个正式 config + `mode_forgetting: 0.02`。
4. checkpoint/provenance：`belief_filter` 入 runtime_provenance 与 objective 元数据
   （防 gru/bayes checkpoint 混装载——eval 侧读 checkpoint 记录的类别重建）。

## 4. 不变量影响（I1–I17）
- I1/I2/I8：无新损失、无选择器、无模式标签监督——不变。
- I11（持久化）：**加强**（后验持久是构造性质，非训练性质）。
- I13/I10：证据输入不变（行为-only），无新 oracle 面。
- 新增候选 **I19-mode**（实施后入 gate）：`belief_filter` 必须记录于 checkpoint 元数据
  且 train/eval 一致；bayes_mode 的后验行和恒=1（数值容差）。

## 5. 测试计划（write-only）
1. 递归性质：归一性（∑_z b=1）、持久性（无证据时 b 仅按 λ 向均匀漂移）、
   支撑完整（任意步 b(z)≥λ/K_f·(1-λ)^0 下界 >0——S27 类冻结不可能）。
2. 梯度流：TD loss 对 g_θ 参数梯度非零（对齐 P4 golden 测试模式）。
3. 兼容：`belief_filter: gru` 与现状 **bit-一致**（golden）；checkpoint 互斥装载报错。
4. E2 正交性：zeroed 通道模式下 BayesModeFilter 正常运行（证据少≠崩溃）。

## 6. 预注册读出（已入 sec18.11 / ICLR_UPGRADE_PLAN §5）
| E1 臂对比 | 结论 |
|---|---|
| aris(bayes_mode) > aris(gru) > flat | 结构化模式递归是真实贡献（C2 headline） |
| aris(bayes_mode) ≈ aris(gru) | C2 降级为"等效+可解释+免疫 S27 类缺陷"的工程论证（诚实路径，仍可作为分析节） |
| aris(bayes_mode) < aris(gru) | 如实报告；C2 退出 headline，检查 λ/容量假设后仅作负结果附录 |

诊断增强（C3 联动）：显式后验使 belief-swap / Δ_info / 探测行为分析直接可读——E4/E6 的
机制证据质量提升是 C2 的次级卖点，无论主对比落哪个分支都成立。

## 7. 回滚与提交
默认 `gru` ⇒ 现状 bit-不变；一机制一提交：① BayesModeFilter 类+单元测试
② 构造分派+provenance ③ config+gate 候选。回滚 = config 一行。

## 8. 请 codex 重点审
1. "hidden=log-posterior" 复用 P4 持久化机制的形状/语义映射有无隐性破绽
   （尤其 replay 存储的 float32 精度对 log 域递归的影响、window-base 重放的等价性）？
2. TD-through-Bayes-recursion 的梯度路径与截断（窗口基 detach）是否与 P4 现状严格同构？
3. λ-混合与 mode_mask 掩码归一的交互（被 mask 的模式是否会经 λ 泄质量）？
4. checkpoint 混装载的所有入口是否都被 provenance 校验覆盖（含 partner_id_q 等分支）？
