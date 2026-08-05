# L0 维度定位：预期读数判读标准（H1–H4）

状态：探索轨，`scientific_readout_allowed: false`。
配套脚本：`l0_nn_partition_discriminant.py`（输出 `l0_summary.json`，`verdict` 字段已内置本节规则的自动初判；人工裁决以本节为准，脚本初判仅为草稿）。

## 0. 服务器运行方式

```bash
cd /apps/users/cxw/Document/CodeSpace/Selfs/ARIS4ZSC_delta_zsc_v5
# 先自检（无需数据，验证脚本在目标环境行为正确）
python analysis/l0_dimension_localization/l0_nn_partition_discriminant.py --smoke
# 再跑真实数据（npz 规格见 DATA_SPEC.md）
python analysis/l0_dimension_localization/l0_nn_partition_discriminant.py \
  --input runs/exploration/s1_panel/full/l0_dimension_localization/l0_dataset.npz \
  --output-dir runs/exploration/s1_panel/full/l0_dimension_localization/
```

## 1. 核心读数定义（均为 run-disjoint 口径 + 特征行置换零线）

| 读数 | 含义 |
|---|---|
| `sig_lift(t)` = `sig_spearman_disjoint − sig_spearman_disjoint_null` | 前缀 t 对 continuation 价值签名的可预测性抬升（主读数） |
| `type_lift(t)` = `acc_type_disjoint − null_type_disjoint` | 前缀对 sp/op 类型标签的可分性抬升 |
| `run_lift(t)` = `acc_run_disjoint − null_run_disjoint` | 前缀对个体 run 标签的可分性抬升 |
| `route_agreement_disjoint − route_chance` | 由预测签名选模式与真签名最优模式的一致率抬升（机会回收代理） |
| 曲线形状：各 lift 随 `prefix_t` 的轨迹 | 信息累积形状（H2 判别用） |

所有判读以最终切点（t=400）为主，曲线形状为辅；样本级 bootstrap 区间跨零的读数一律视为不可读。

## 2. 各竞争解释的判读规则

### H1 低信噪比（面板/轨迹中的价值结构本身淹没在噪声里）
- **排除**：`sig_lift(400) > 0.05`，或 `type_lift`/`run_lift`/路由一致率抬升任一 > 0.05 且区间不跨零——至少一类结构可恢复，"全面低信噪比"不成立。
- **保留**：最终切点全部 lift ≤ 0.02（区间跨零）——Φ 的负读数可能只是测量噪声，现象本身需重新审视（此为最高警报：动摇 S1 判读的数据质量前提，须先复核面板方差与回合数）。

### H2 前缀不足（窗口内前缀历史信息不够，并非无结构）
- **保留**：`sig_lift(t)` 在 [50, 400] 内单调上升、末段斜率未趋平（`sig_lift(400) − sig_lift(最短切点) > 0.10`）且 `sig_lift(400) < 0.3`——与 E4"个体级证据需约 425 步（超出 400 步窗口）"同向，属预期内保留。
- **排除/弱化**：`sig_lift(t)` 在中段已趋平，或 `sig_lift(400)` 已达可观水平（≥0.3）——窗口内前缀已承载结构，H2 不成立，剩余负读数须由 H1/H3 解释。
- 注：若曲线趋平于低水平，H2 排除但 H1/H3 竞争，以 §2 其余项裁决。

### H3 模式基底伪结构（面板"机会"是模式库巧合匹配，而非可路由的真实结构）
- **保留为个体级结构（Φ 的正面确认）**：`sig_lift` 显著为正、`run_lift ≤ 0.02`（身份不可跨 run 预测）、路由一致率抬升 > 0.05——前缀能恢复价值签名并支撑模式选择，但结构不可压缩为 run 标签/类型，与 E0 的 τ 读数和迁移差 −83.60 自洽，同时构成对"机会真实存在（oracle 84.22 非伪结构）"的零成本支持。
- **判 INCONCLUSIVE**：`sig_lift` 本身不显著——本分析无法单独裁决 H3，需 L1 正对照（lineage/热启动种群 assay）。
- **排除/弱化**：`sig_lift` 显著但路由一致率抬升 ≤ 0（预测签名选不出正确模式）——签名几何与面板机会脱钩，伪结构嫌疑转向签名估计口径，先复核签名聚合（§DATA_SPEC 四）。

### H4 布局混淆（读数只是 Simple 布局特异产物）
- 本轮数据仅含 Simple 布局，**恒定输出 BOUNDARY_ONLY**：任何结论只声明到 Simple，Wide 后置（对齐 P6 口径与台账 P4 条目）。多布局数据到位前，H4 既不排除也不保留，属边界声明。

## 3. 组合判读速查

| 读数形态 | 结论 |
|---|---|
| sig_lift 高、run_lift ≈ 0、type_lift ≈ 0、路由抬升 > 0 | Φ 确认：机会在个体级签名维度，类型/身份维度无结构；H1 排除、H3 弱化、H2 视曲线形状 |
| sig_lift 高且曲线趋平、run_lift 也高 | 前缀同时携带身份与价值信息——注意检查 signature 是否按 run 聚合（降级口径）导致的 identity 泄漏，回 DATA_SPEC §四 复核 |
| 全部 lift ≈ 0 | H1 保留（最高警报），暂停 L1 及以后投入，先做数据质量复核 |
| sig_lift 低、曲线持续上升 | H2 保留，与 E4 425 步判读合流：升级方向是加长回合或前缀延续测量，而非加模式 |

## 4. 与既有台账读数的互检钩子

- 若本分析 `sig_lift` 显著为正，应与 E0（类型内 τ 0.062 ≈ 跨类型 0.065）并读：签名可预测但类型分划无排序一致性 ⇒ 结构维度在"个体签名"而非"类型惯例"。
- 路由一致率抬升若接近 `Δ̂·TV̂(400)/2` 量级（Δ̂=27.3、TV̂(400)≈0.90 的 E4 读数），为 P4 对撞提供前哨一致性；偏差过大则标记待 E5 复核。
- 本分析属零成本旁路：任何读数不改写 S1 停机判据与 Θ 裁决，仅按只追加纪律建议入台账条目（由任务负责人登记）。

## 5. 降级口径声明（全回合面板签名广播）

若 `signature` 采用降级口径（伙伴级全回合均值广播到各前缀样本，不随前缀变化）：
- `sig_lift` 度量的是"前缀识别伙伴个体价值画像"的能力，语义弱于前缀延续级口径；
- 此时 run_lift 与 sig_lift 高度同构，§3 第二行的 identity 泄漏检查强制生效；
- 结论强度降一级：只能给出 H1/H2 的排除/保留，H3 一律 INCONCLUSIVE。
