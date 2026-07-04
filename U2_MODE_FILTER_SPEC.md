# U2_MODE_FILTER_SPEC — 持久协议模式信念 BayesModeFilter（设计规格，送 codex 评审）

**Status:** DESIGN — 未实现；**已通过 codex 评审为 GO-WITH-CHANGES**（本轮修订版含所有必需修改）
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

**训练：单一 TD objective**（codex Q2 措辞修订）。b_t 是 g_θ 的可微函数（softmax-normalized
递归），TD 梯度经 belief→Q 路径回传穿过递归。**表述规范**（避免夸大）：本设计**不声明**
"与 P4 数学上严格同构"（strictly isomorphic），而声明"**与 P4 相同的截断边界（窗口基
detach）与优化器路径，仅递归单元由 GRUCell 换为解析 Bayes 滤波器（含 λ-遗忘与可学习
log-likelihood g_θ）**"。g_θ 梯度**只**经 belief→Q→TD loss 传导；无监督模式标签、无辅助
likelihood 损失、无 selector 损失、无 oracle 模式标签（I1/I8 不破）。论文/文档措辞：
"single TD objective with an analytic differentiable Bayes-filter recurrence and learned
likelihood"（避免暗示存在独立 Bayesian evidence 目标）。

**数值规则（codex Q1 修订，log-domain 强制）**：
- b_t 内部**始终以 log-posterior 形式存储**（"hidden = log_b"）；`belief_from_hidden`
  输出 `softmax(log_b)`（with mode_mask）。
- 归一化用 `logsumexp`（而非 log(sum(exp(·)))）；float32 存储需 clip `log_b` 到 `[-30, 30]`
  防溢出/下溢；被 mode_mask 掩掉的模式设 `log_b = -inf`（**不参与** logsumexp）。
- λ-混合以 log-domain 表达：`log_b̃ = logaddexp(log(1-λ) + log_b_prev, log(λ/K_valid) - inf_for_masked)`。
- **窗口基 replay 说明（codex Q1 强调）**：P4 的截断 replay 是"从窗口基重放 W 步"，不等同
  于"从 episode 起始 Bayes 全量重算"。文档明确写：**BayesModeFilter 复用 P4 截断，不作
  额外全 episode 一致性主张**。

## 2. 接口契约（与现有 P4 机制完全对齐）

`BayesModeFilter(nn.Module)` 作为 `FactorLocalBeliefModel` 的同接口姊妹类
（`src/aris_bellman/factor_belief.py`）：

| 接口 | 语义映射 |
|---|---|
| `initial_hidden(B, F, device)` | 返回均匀 **log-posterior** `[B,F,K_max]`（"hidden"=对数后验——直接复用 P4 全部持久化/replay/快照机制，`EvidenceBuffer._belief_hidden`、`belief_window_base_snapshot`、`OptionTransition.belief_hidden_t/next` 形状协议不变，仅第三维从 hidden_dim 变 K_max —— 见 §3 形状适配） |
| `step_history(evidence, hidden, mode_mask, ...)` | 一步 Bayes 递归。**codex Q3 强制**：`mode_mask` 必须作为参数流经 `step_history` / `encode_history`（现签名 factor_belief.py:55 无此参数，属**破坏性接口修改**——两选一实施路径见 §3）。掩码模式的 log_b 恒为 -inf；未掩码模式行归一为 1（logsumexp）。 |
| `encode_history(seq, initial_hidden, masks...)` | 窗口重放递归（TD 侧，梯度可通，零填充掩码语义同 P4/S2） |
| `forward` / `belief_from_hidden(hidden)` | `softmax(log_posterior)`（+ 现有 mode_mask 掩码归一） |

下游 `factor_q.py`（centered-belief advantage、relevance 路由）、`_state_repr`、
`_advance_persistent_belief`、eval 共享路径**零改动**——只在 `_build_belief_model`
按 config 分派。

## 3. 精确改动点

1. `src/aris_bellman/factor_belief.py`：新增 `BayesModeFilter` 类（~150 行含 log-domain 数值
   规则与 mode_mask 处理）。
2. **mode_mask 流经方案（codex Q3，二选一）**：
   - **方案 A（推荐）**：修改 `FactorLocalBeliefModel.step_history` / `encode_history` 签名
     添加可选 `mode_mask: torch.Tensor | None`（默认 None 时对 GRU 无副作用，保 gru bit-兼容）；
     `_advance_persistent_belief(train_aris.py:2939)` 加对应 pass。工程量小、语义清晰。
   - **方案 B**：BayesModeFilter 在类内**自持** `mode_mask`（构造时从 graph 计算固定 mask），
     `step_history` 签名不变。接口更兼容，但 mask 变化需重构类实例（少见但要文档化）。
   - **决策**：方案 A（预计实施选项）——mask 是每因子每模式的 substrate 属性，作为参数
     更透明，且 GRU 分支不改行为。
3. `train_aris.py::_build_belief_model`：读 `training.belief_filter: gru(默认)|bayes_mode`
   分派构造；hidden 第三维按类别取 hidden_dim 或 K_max（`EvidenceBuffer` 的
   `set_belief_hidden` 形状校验读模型的 `hidden_dim` 属性——BayesModeFilter 暴露
   `hidden_dim=K_max` 即可兼容，其余机制不感知差异）。
4. config：`belief_filter: gru` 显式入三个正式 config + `mode_forgetting: 0.02`。
5. **checkpoint/provenance（codex Q4 强制扩展）**：checkpoint payload 增字段：
   `belief_filter` (class name), `belief_filter_hidden_dim`, `belief_filter_max_modes`,
   `mode_forgetting`, `graph_hash`；save 时（train_aris.py:3244）写入，load 时
   (evaluate_aris.py:266-280 附近, train_aris.py:3292 `_checkpoint_loads`) 严格校验一致，
   任一失配即拒。**`random_policy` 与 `partner_id_q` 分支**：写 `belief_filter: "unused"`
   + 空 belief state（不参与校验），或严格拒绝加载（二选一，见测试 §5.5）。

## 4. 不变量影响（I1–I17 + 新 I19-mode）
- I1/I2/I8：无新损失、无选择器、无模式标签监督——不变（措辞按 §1 修订）。
- I11（持久化）：**加强**（后验持久是构造性质，非训练性质）。
- I13/I10：证据输入不变（行为-only），无新 oracle 面。
- **新 I19-mode（fidelity gate 工具入门后加入，codex Q4 扩展）**：
  - checkpoint 必须携带 `belief_filter` / `belief_filter_hidden_dim` / `belief_filter_max_modes`
    / `mode_forgetting` / `graph_hash`；train↔eval 一致（否则拒绝加载）；
  - bayes_mode 模式下，每因子 log_b 的有效模式行 `softmax(log_b[valid]) ≈ 1`（数值容差 1e-4）；
  - **掩码模式零质量守恒**：`softmax(log_b)[masked] == 0.0`（数值容差）；
  - random_policy/partner_id_q 分支 belief_filter 记录为 `"unused"` 或加载被明确拒绝；
  - GRU/Bayes 混装载 = FAIL。

## 5. 测试计划（write-only，codex 要求扩展）
1. 递归性质：归一性（∑_{z 有效} softmax(log_b) = 1，log-domain via logsumexp）、
   持久性（无证据时 b 仅按 λ 向 uniform-over-valid 漂移）、
   支撑完整（S27 类冻结不可能——任意步任意有效模式 softmax > 0）。
2. **Masked-mode leakage test（codex Q3 强制）**：构造某因子有 K_max=5 但 mode_mask 只
   允许 2 个模式；跑 100 步；验证 `softmax(log_b)[masked_indices]` 恒为 0（数值容差）；
   验证 λ-混合过程不会给被 mask 模式赋 log_b > -inf。
3. **梯度流**：TD loss 对 g_θ 参数梯度非零（对齐 P4 golden 测试模式）；梯度对**掩码模式**
   参数为 0（验证 mask 上游截断）。
4. **Stale window-base truncation equivalence（codex Q1 强制）**：合成同一 episode 两次
   TD replay：（a）从窗口基重放（当前 P4 语义），（b）从 episode 起点重放。验证 (a) 与 (b)
   **不必**相等（这是设计接受的近似）——测试断言两者关系符合 §1 "same truncation boundary"
   声明，避免文档里出现"strictly isomorphic"类夸张。
5. **Checkpoint provenance（codex Q4 强制）**：
   5.1 GRU checkpoint 尝试加载到 bayes_mode config → 明确 raise（不 silent）；
   5.2 belief_filter_hidden_dim 或 max_modes 或 mode_forgetting 或 graph_hash 任一不匹配 → raise；
   5.3 random_policy / partner_id_q checkpoint 加载：`belief_filter="unused"` 分支不校验，
       其他分支拒绝。
6. 兼容：`belief_filter: gru` 与现状 **bit-一致**（golden，全套现有 P4 测试通过）。
7. E2 正交性：zeroed 通道模式下 BayesModeFilter 正常运行（证据少≠崩溃，只影响 posterior
   收敛速度）。

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
