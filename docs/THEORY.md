# THEORY：DEPI 理论索引与适用边界

修订头：本文件于统一重构（`Rigor_与_Generality_统一优化_7f793915.md` D/I 节；外部评审归档 [`research/REVIEW_AND_SUGGESTION_2026.md`](research/REVIEW_AND_SUGGESTION_2026.md) §7/§14）中创建，是四份权威文件之四：**理论地图与适用边界**的唯一权威索引。本文件**不复制证明全文**，只给出定理定位、修订状态与禁用条款；证明全文以 [`theory/DELTA_ZSC_FOUNDATIONAL_THEORY_AND_PROOFS.md`](theory/DELTA_ZSC_FOUNDATIONAL_THEORY_AND_PROOFS.md)（2026-08-03 修订版）为准。

状态：`authoritative: true`。理论陈述的适用边界以本文件为准；旧文档中与本文件冲突的生态外推用法一律废止（逐条见第四节）。

---

## 一、理论地图总览

| 定理 | 主题 | 修订状态 | 证明全文位置 |
|---|---|---|---|
| T1 | 分解链单调性 V_fix ≤ V_state ≤ V_hist ≤ V_HZ ≤ V_full | 有效（平凡，信息包含关系） | 历史登记于 [`research/PAPER_STANDARD.md`](research/PAPER_STANDARD.md) §5（支撑层） |
| T2 | 二惯例 TV 精确式与匹配界 | **已降级**：仅二元等先验、一一对应模式、常数 gap Δ 特例有效 | [`theory/DELTA_ZSC_FOUNDATIONAL_THEORY_AND_PROOFS.md`](theory/DELTA_ZSC_FOUNDATIONAL_THEORY_AND_PROOFS.md) §10.2 |
| T3 | 历史样本复杂度 Θ(log(Δ/ε)/κ²) | **证明链已重建**（原不等式证伪），注册陈述不变 | 同上 §10.3 |
| T4 | 路由紧上界与端到端等式 | 有效 | 同上 §10.4 |
| D_V(t) | 一般情形 estimand：value-weighted distinguishability | 2026-08-03 注册，取代 E[Δ]·TV/2 | 同上 §10.2 适用范围声明 |

定位说明：Θ2（信息时序理论）**降为支持性理论**，不再并列为中心论点；中心主张以 [`SCIENTIFIC_SPEC.md`](SCIENTIFIC_SPEC.md) §3 为准。

## 二、T3 修复后的样本复杂度（2026-08-03 修订版）

以 [`theory/DELTA_ZSC_FOUNDATIONAL_THEORY_AND_PROOFS.md`](theory/DELTA_ZSC_FOUNDATIONAL_THEORY_AND_PROOFS.md) §10.3 修订版为唯一权威，要点：

1. **原证明错误（已证伪）**：原必要性证明使用不等式 log((1+κ)/(1−κ))≤2κ，取 κ=1/2 时 log 3≈1.099>1，数值验证证伪；"紧口径 kl≤2κ²、分母 2"的表述一并废止。
2. **重建链条**：由修正界 atanh(κ)≤κ/(1−κ²)（0≤κ<1）得 log((1+κ)/(1−κ))≤2κ/(1−κ²)≤(8/3)κ（0≤κ≤1/2），证明链经此界重建。
3. **单步 KL 夹逼**（数值验证）：2κ²≤kl≤(8/3)κ²，kl/κ²∈[2.0000, 2.1972]（κ∈(0,1/2]）；必要性阶与 Hoeffding 充分性阶一致。
4. **注册陈述（不变）**：必要性下界采用保守单步 KL≤4κ² 口径，
   \[ n\ge\frac{\log(\Delta/(4\varepsilon))}{4\kappa^2}; \]
   充分性 n≥2log(Δ/ε)/κ²；合为 Θ(log(Δ/ε)/κ²)，0<κ≤1/2、0<ε≤Δ/8。紧形式 n≥3log(Δ/(4ε))/(8κ²) 一并登记。
5. **前提**：T3 的 κ 来自明确的 i.i.d. Bernoulli 证据模型；生态口径不满足该前提（见第四节禁用条款）。

## 三、T2 降级与一般 estimand D_V(t)

1. **T2 降级为二元常数-gap 特例**：TV 精确式（V_HZ−V_hist=Δ(1−TV)/2、V_hist−V_fix=Δ·TV/2、p_e*=(1−TV)/2）只在二元等先验、模式与惯例一一对应、所有历史上收益差为同一常数 Δ 的静态模型内成立（[`theory/DELTA_ZSC_FOUNDATIONAL_THEORY_AND_PROOFS.md`](theory/DELTA_ZSC_FOUNDATIONAL_THEORY_AND_PROOFS.md) §10.2 适用范围声明）。
2. **一般 estimand**（评审 §7.4；同步登记于 [`SCIENTIFIC_SPEC.md`](SCIENTIFIC_SPEC.md) §5.1）：
   \[ D_V(t)=\frac12\int\Delta(h)\,\bigl|p_1^{\,t}(h)-p_0^{\,t}(h)\bigr|\,dh, \]
   其中 h 为合法 ego 历史（SCIENTIFIC_SPEC §2 L1–L5 口径），p_i^t 为协议 i 诱导的合法历史前缀分布，Δ(h) 为历史 h 上的 payoff gap。仅当 Δ(h)≡Δ 常数时退化为 Δ·TV/2。
3. **废止**："E[Δ(H)]·TV/2 = 分别平均再相乘"的用法在一般情形不成立，禁止用于生态任务（旧口径废止清单见 [`SCIENTIFIC_SPEC.md`](SCIENTIFIC_SPEC.md) §8）。

## 四、生态 κ̂ 代理口径声明与禁用条款

口径全文以 [`research/THEORY_PREDICTIONS.md`](research/THEORY_PREDICTIONS.md) 与 [`research/TRAJECTORY_AND_ESTIMATION_SPEC.md`](research/TRAJECTORY_AND_ESTIMATION_SPEC.md) 的 2026-08-03 修订版为准（两文件为支撑层，探索轨 `scientific_readout_allowed: false`），本文件登记其边界：

1. **κ̂ 降为代理量**：生态口径的单步边际 TV 不满足 T3 的 i.i.d. Bernoulli 前提——证据依赖当前环境状态、ego 之前的动作、时间、伙伴 RNN carry 与当前协议，既不独立也不同分布。
2. **禁用条款**：**禁止**将生态 κ̂ 代入 Θ(log(Δ/ε)/κ²) 计算历史需求或做生态外推；据此已入台账的窗口读数（类型级约 220 步、个体级均值约 425 步、弱个体约 5000 步）全部建立在代理口径上，须按修正口径复核；复核完成前"425 步个体级窗口外"结论不得作为结论引用。
3. **分类器误差只给 TV 下界**：TV=1−2p_e^* 只对最优 Bayes 分类器精确成立；有限样本经验分类器读数一律按 1−2p̂_e≤TV 的下界报告与使用，不得当作 TV 点估计。
4. **partner-action oracle 禁令**：完整伙伴动作序列禁止作为主估计口径，只允许作为 unattainable oracle 参照并显式标注，不得用于生态外推、历史需求计算或主张（与 [`SCIENTIFIC_SPEC.md`](SCIENTIFIC_SPEC.md) §2 禁止项 F1 一致）。

## 五、S2 定性：theorem unit test

S2 受控相图（180/180）是按 T2/T3 假设构造的 Bernoulli 特例，用同一 closed-form 公式生成并检查结果，只说明代码正确实现了注册恒等式，性质为 **theorem unit test**，**不是**对理论的独立数值验证；"主论点 Θ2 已验证"的表述收回（台账修正条目见 [`status/EVIDENCE_LEDGER.md`](status/EVIDENCE_LEDGER.md) 2026-08-03 DEPI 重构条目；同步声明见 [`research/THEORY_PREDICTIONS.md`](research/THEORY_PREDICTIONS.md) S2 降级段）。

## 六、各定理适用边界表

| 定理 | 成立前提 | 有效范围 | 禁用场景 |
|---|---|---|---|
| T1 | 信息包含关系（无模型假设） | 分解链任意受控/生态任务 | 不得反向推出各层价值差的具体量级 |
| T2 | 二元等先验、一一对应模式、常数 gap Δ、静态收益 | 受控特例；S2 自检 | 生态任务可恢复价值计算（改用 D_V(t)）；"分别平均再相乘"全面禁用 |
| T3 | T2 模型 + n 个条件独立 Bernoulli 证据（参数 (1±κ)/2）、0<κ≤1/2、0<ε≤Δ/8 | 受控 i.i.d. 证据模型；注册上下界常数 | 生态 κ̂ 代入 Θ(log(Δ/ε)/κ²)；生态历史需求定量推断 |
| T4 | 固定共同前缀 q 与切换时刻、模式库给定、只读官方历史 | 一般设定（学习路由器给出 V_hist 构造性下界） | 学习路由器失败不得解读为"无机会证书"（无机会须由受控上界或 run-disjoint 面板给出） |
| D_V(t) | 合法历史前缀分布可估计、Δ(h) 可测 | 一般情形的预测与对撞 estimand | 禁止以完整伙伴动作 oracle 基底估计后冒充合法口径 |

## 七、一致性声明

- 本文件与 [`theory/DELTA_ZSC_FOUNDATIONAL_THEORY_AND_PROOFS.md`](theory/DELTA_ZSC_FOUNDATIONAL_THEORY_AND_PROOFS.md) 2026-08-03 修订版逐项一致（T3 重建、T2 降级、D_V 注册）；与 [`SCIENTIFIC_SPEC.md`](SCIENTIFIC_SPEC.md) §5.1（D_V）、§3（Θ2 降级）一致；
- 生态预测 P1–P6 的口径修订以 [`research/THEORY_PREDICTIONS.md`](research/THEORY_PREDICTIONS.md) 修订版为准，本文件只登记边界不复述预测行；
- 理论引用进入论文时须按 [`EVALUATION_SPEC.md`](EVALUATION_SPEC.md) 第七节探索轨读数调和规则处理；本文件与支撑层文档冲突时以本文件为准。
