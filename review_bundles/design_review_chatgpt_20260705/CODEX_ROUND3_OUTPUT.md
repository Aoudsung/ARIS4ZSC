# CODEX 跨模型审计输出（对 rev3）— 2026-07-05

**审稿人**：codex / GPT-5.5，reasoning=xhigh，sandbox=read-only（本项目指定跨模型审稿人，
OPERATING_CONSTRAINTS §3）· **thread**：019f303d-709e-7922-8dd3-d1c4c4249259
**对象**：U2_ROLE_BELIEF_PROPOSAL.md (rev3) + U2_MODE_FILTER_SPEC + ARIS_DESIGN_MASTER +
U1_INTERVENTIONAL_CE_SPEC；含两轮 ChatGPT 输出与作者裁决作为交叉材料
**特点**：codex 有仓库读权限，读了 `src/aris_bellman/factor_q.py` 等源码——够到 ChatGPT
（无代码访问）结构性挖不到的实现层。原始输出逐字如下。

---

（以下为 codex 原始返回，未编辑）

**1. 问题清单**

[01][致命] `M∧A∧G∧D ⇔ 核心命题` 是伪等价。四子命题联合不等价于"角色是正确粒度"：真角色世界里手写 OASIS 也可能通过（D 失败但命题仍真）；四项都过也可能只是 TD 训练出的低维状态进度变量有效。最小修复：`⇔` 改成 outcome-to-claim matrix；D 归为方法必要性/新颖性，不归为粒度命题本体。

[02][致命] 阈值推迟到 sec18.14 冻结仍留阈值工程窗口。δ_a/e_a/τ_G/ρ/κ 均占位；作者已知 stage-1、P1 行为、当前 ARIS 正面发现与失败形态，可按可过性选阈值/horizon/MI nuisance 集/CI 口径。最小修复：任何 T0/T1 运行前冻结数值、非劣 margin、bootstrap 单位、Holm family、prediction horizon，用 P0-only 或外部 null 定阈值。

[03][致命] T0"无条件"但无硬后果，可能变无条件摆件。§4.3 说 T0 无条件执行，§5.1 无 T0 分支。若 OASIS-public 已满足两判据、GRU(h=3) 追平、P3 生态无效，文本仍可继续 T1 并后置解释。最小修复：加 T0 决策表——OASIS 过线→D 预失败重写主张；P3 无有效 ambiguity→G 不可启用；GRU(h=3) 追平→A 预降级。

[04][致命] P3"非上菜轴"与 G 的预测目标互相抵消。§4.1-P3 禁终端偏好参数，§5.2-G1 预测"未来终端/备菜贡献"。P3 只测拥塞/路径风格则与服务角色无关；G 又用 terminal/prep 目标把上菜本体论塞回来。最小修复：P3 入选条件必须是"同一公共状态下诱发不同最优 ego response"；G 主指标改 state-stratified counterfactual best-response regret。

[05][致命] T1 失败不能直接叫"M 证伪"。T1 用"现图"。U2 可能在低 min_weight/kind coverage 现图失败、在 U1 final graph 成功。最小修复：T1 失败只写"current graph+budget route failed"；M 正式反证须等 final graph 或至少一个 GRU/U2 × current/final graph 对照。

[06][重大] G 预测效度可能只是公共状态预测，不是角色后验。posterior 可经 x_t^f 间接编码时间/汤状态/位置/持物，预测未来贡献并不需要角色。最小修复：加 state-only probe、posterior 在 state/opportunity bins 内 shuffle 的负控、残差化 public state 后的 conditional AUC/log-lik。

[07][重大] Bayes posterior 尖锐度是隐藏自由度。SPEC §1 用 exp(g_θ) 更新，g_θ 只经 TD 训练；TD 可把 logits 放大到饱和，MI、跨种子对齐、belief-swap 都被温度支配。最小修复：冻结/审计 posterior temperature、logit norm、entropy；temperature sweep + calibration/ECE；语义指标须在固定温度下稳定。

[08][重大] window-base replay 可能改变 TD 目标。SPEC §1 接受从窗口基重放；stored base hidden 来自旧 g_θ，当前网络只重放 W 步，TD bootstrap 看到混合时代 posterior。最小修复：采样 transition 比较 full-online vs window-replay posterior 的 KL；W sweep；周期性 full recompute ablation；敏感则不能称 posterior 语义稳定。

[09][重大] 当前 Q 接口对信念的使用太弱，撑不起"不确定状态共享"。承诺三因子局部加性修正；实现中 belief-uniform 线性进 Q（factor_q.py:264）。uniform/high-entropy belief 对修正项接近零，无法主动选择 probe；serve 若需 terminal-yield AND bottleneck-clear，纯加性修正表达不了。最小修复：ambiguous-prefix 中测 uniform belief 下 probe Q-margin；构造 AND/XOR mixed-role toy；失败则加 entropy feature、pairwise factor term 或 global role latent。

[10][重大] A 归因仍受模型选择和调参预算污染。U2 有 λ/K/匿名模式 sweep，baseline 没同等预算；最佳 U2 config 可能在同评估集上被挑出。最小修复：预先指定 primary U2 config；λ/K 只在 P0/P2 validation 选；每族等 hyperparameter budget；主表报全 sweep 而非只报 winner。

[11][重大] "主表同一张图"不能切断 U1/U2 归因纠缠。final U1 graph 给新终端证据路线，BayesMode 成功、GRU 旧图失败，则提升来自 graph×belief 交互。最小修复：至少跑 GRU-final-graph、U2-final-graph、U2-current-graph；最好 2×2 报 interaction。

[12][重大] U1 术语修复还没真正落盘到主叙事。总方案 §5 仍写"干预式交互因子发现"，U1 文件标题仍是 INTERVENTIONAL_CE；审稿材料当前仍在卖 causal discovery。最小修复：现在就全局改成 interventional support top-up / support repair with certificates；pair-level runner 之前禁用 interventional causal effect headline。

[13][重大] 5 seeds 下 Holm/CI 分离基本没有论文级功效。episode bootstrap 低估训练随机性；多指标 Holm 后无功效只能讲趋势。最小修复：seed-paired hierarchical bootstrap 或 mixed model；先给 MDE/power table；headline 增 seeds，或把 T2 声明为 pilot evidence。

[14][中等] "ZSC 角色"缺 human-like 外部效度。P4 human-BC 列 extended；评审可能要求 human-like partner。最小修复：3-5 个 human-BC proxy；不可得则标题/摘要明确限定 scripted/RL-partner ZSC。

[15][中等] U2 仍依赖 transient option inference，上游粒度没真消失。总方案流程"对方动作→行为推断→因子证据路由→角色滤波"；P3/OOD 失败可能来自上游 option evidence 而非 role filter。最小修复：报告 upstream evidence/option-inference OOD 与错误率；加 raw-observation-to-role evidence ablation 或至少把失败归因分开。

**2. rev3 修复真伪评级表**

| 关键旧问题 | 评级 | 理由 |
|---|---|---|
| E1 读出越权 / 条件性可证伪 | 部分 | T1 降 smoke 是实质进步，但 T0 无硬后果，T1 失败仍过度写成 M 证伪。 |
| 循环论证 / 词表泄漏 | 部分 | P3、盲化、K sweep 有结构位置；但 G 目标仍带 terminal/prep 本体论，P3 可能变无关分布。 |
| "低维可解释结构"无操作定义 | 部分 | G1/G2/G3 形式化指标族，但关键阈值、prediction horizon、state 控制仍缺。 |
| TD-only 后验语义 | 部分 | probe、MI、belief-swap 有后果；但 posterior 温度、公共状态混杂、线性 Q 接口未处理。 |
| "单一变量"与容量匹配 | 部分 | 改"单一模块"并扩 L4；仍缺等调参预算、nested selection、参数/优化预算公平性。 |
| spec §6 读出口径分叉 | 真解决 | SPEC §6 已指向 rev3 §5，headline 读出降为候选机制贡献。 |
| 训练前/训练后 gate 混排 | 真解决 | §6.1/§6.2 三层拆分清楚，时序问题基本消失。 |
| zeroed 变好=bug 过硬 | 真解决 | §5.3 三分支承认 learned likelihood harmful，结构性修复。 |
| OASIS 信息预算 | 部分 | 三档 + learning baseline 对；但具体规则、阈值、D 逻辑地位未冻。 |
| U1 "interventional CE" 与 quota 自由度 | 换皮 | rev3 承诺之后同步；当前 U1/总方案仍保留强术语，quota/tie-breaking 仍未定。 |

**3. 最强拒稿意见书**

论文把"角色是正确粒度"形式化为一组性能与诊断门，但这些门既不等价于该命题，也不足以隔离替代解释。P3/G 读出仍用上菜/备菜本体，BayesMode 后验可由 TD 学成状态进度或温度化记忆，Q 接口又只线性使用 centered belief。阈值和模型选择尚未冻结，U1 图与 U2 贡献仍纠缠。即便实验全过，最强结论也只是一个结构化 recurrent state 在单布局伙伴生态中有效，不是 ZSC 角色粒度的发现。

**4. 十个必须能当场回答的问题**

1. 哪些 T0 结果会让你在 T1 前停止、降级或重写主张？
2. 为什么 D 是"角色粒度正确"的组成部分，而不是方法必要性/新颖性检验？
3. δ_a/e_a/τ_G/ρ/κ、非劣 margin、prediction horizon、bootstrap 单位分别是多少？
4. G1 如何排除 public-state-only probe 已能预测未来 terminal/prep contribution？
5. posterior 的 logit scale/temperature 如何校准，过尖或过平算不算失败？
6. window replay posterior 与 full-online posterior 的 KL 多大时禁用语义解释？
7. centered linear belief correction 如何在 uniform belief 下选择信息采集动作？
8. 最终主表是哪张图？GRU-final-graph 与 U2-final-graph 是否同跑？
9. λ/K sweep 与 L4 baseline 是否有相同调参预算和独立 validation set？
10. 若 OASIS 过线、K=8 最好、或 U2 只在 final U1 graph 上成功，核心 claim 怎么改？

**5. 未言明的承重假设清单**

1. M/A/G/D 四项不仅相关，而且联合等价于核心命题。
2. 廉价规则基线能否过关可以判定"角色粒度"真伪。
3. "先于 T2 冻结阈值"足以防止阈值工程。
4. T0 资产即使没有 stop/go 后果，也能解决条件性证伪问题。
5. P3 非上菜轴伙伴仍会产生与角色信念相关的决策 ambiguity。
6. posterior 的概率尖锐度可解释，而不是 TD logit scale 产物。
7. window-base replay 不会把训练中的 posterior 语义变成 stale hidden artifact。
8. belief-uniform 的线性加性接口足以表达 uncertainty control 与跨因子角色。
9. 5 个训练 seeds 足以支持多 baseline、多指标、Holm 校正后的命题级判读。
10. "同一最终图读主表"足以分离 U1 图构造贡献和 U2 信念贡献。

**6. 修复优先级建议**

1. 先改命题逻辑和决策表。去掉 ⇔，把 M/A/G/D 改成 claims matrix，给 T0/T1/T2 每个失败分支写硬后果；否则后续所有实验都能被重解释。
2. 立刻冻结 §5 的可执行统计规格。所有阈值、margin、horizon、bootstrap 层级、Holm family、K/λ 选择规则和 power/MDE；否则预注册仍会被认为是事后阈值工程。
3. 烧 U2 种子前加三个技术 sanity gate：posterior temperature/calibration、window-vs-full posterior KL、centered-belief 表达力 toy。三项不过则 T1 成功也不能解释为"角色信念"。
