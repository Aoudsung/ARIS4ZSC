# L0 零成本维度定位：数据规格说明

状态：探索轨，`scientific_readout_allowed: false`。
上游：重构计划 G 节"现象 Φ 的因果识别阶梯"L0（NN/k-分划映射判别）；`docs/status/EVIDENCE_LEDGER.md` S1/E0/E4 条目。
用途：在远程服务器上用已有 S1 面板 + E2 轨迹数据对现象 Φ（"类型级结构不迁移但个体级机会存在"：迁移差 −83.60、类型内 Kendall τ 0.062 ≈ 跨类型 0.065、去 identity oracle 84.22）做维度定位，判别竞争解释 H1（低信噪比）/ H2（前缀不足）/ H3（模式基底伪结构）/ H4（布局混淆）。

## 一、分析所需的最小数据（判别分析的三要素）

1. 每个伙伴 run 的**前缀历史表征或动作序列**（预测变量 x）；
2. 每个前缀样本的 **continuation 价值标签**（预测目标 y_sig：该前缀在各模式 run 下的延续价值签名）；
3. **类型/run 标签**（预测目标 y_id：伙伴 run 身份与 sp/op 类型，用于对比"预测签名"与"预测身份"的可分性）。

## 二、交付文件：一个 npz，字段如下

建议交付路径：服务器侧
`runs/exploration/s1_panel/full/l0_dimension_localization/l0_dataset.npz`

| 字段 | 形状 | 类型 | 含义 |
|---|---|---|---|
| `prefix_features` | (N, T, D) | float32 | N 个前缀样本 × T 个前缀切点（升序，如 [50,100,200,400]）× D 维特征。特征口径与 E3/E4 对齐：截至切点 t 的伙伴动作窗口直方图（6 维，窗宽 w=20，t<20 零填充）拼接累计动作直方图（6 维），D=12 即可；若服务器另有表征（如模板匹配 embedding）可直接替换，D 不限 |
| `signature` | (N, M) | float32 | 每个前缀样本的延续价值签名：M 个非 identity 模式 run 在该前缀后的延续均值回报，**行内 centered**（减去该行均值）。这是预测目标 y_sig |
| `run_label` | (N,) | int | 前缀样本所属伙伴 run 编号（0..19，预测目标 y_id 细粒度） |
| `type_label` | (N,) | int | 伙伴类型 0=sp / 1=op（预测目标 y_id 粗粒度） |
| `layout` | (N,) | int | 布局编号（本轮 Simple=0；Wide 后置，H4 结论边界据此声明） |
| `prefix_t` | (T,) | int | 前缀切点步数，与 `prefix_features` 第二维对应 |
| `mode_run_ids` | (M,) | int | 签名维对应的模式 run 编号（须与面板非 identity 模式集合一致） |

规模要求：
- 伙伴 run 数：sp/op 各 ≥10（现有 20 满足）；
- 模式 run 数 M ≥15（现有 19 个非 identity 模式满足）；
- 前缀样本 N：每伙伴 run ≥100 个前缀样本（来自 E2 的 100 回合 × 每回合多个切点），总量 N ≈ 2000–8000；
- 签名若来自面板全回合口径，每 (伙伴, 模式) 格 ≥50 回合（现有 500 满足）后按前缀条件聚合。

## 三、服务器现有产物 → 交付字段的映射

1. `signature`：
   - 首选口径（前缀延续级）：沿用 S1 第二版面板设计（共同前缀状态快照 + 多模式配对延续 + 共同随机数），对 E2 前缀切点做延续测量；
   - 降级口径（全回合面板代理，登记为代理量）：由 `runs/exploration/s1_panel/full/` 逐回合 parquet（列：模式类型、模式 run、伙伴类型、伙伴 run、标记、episode 索引、原始回报）按 (伙伴 run, 模式 run) 聚合均值，排除 `identity_match` 行，得到 (20, M) 伙伴级签名，再广播到该伙伴的各前缀样本（此时签名不随前缀变化，属粗口径，判读时降级处理，见 READOUT_CRITERIA §5）。
2. `prefix_features`：来自 `runs/exploration/s2_ecology/full/trajectories.json`（每回合伙伴动作序列，0-5 索引，长 400），按 §二 特征口径计算。注意 episode key 流与面板隔离（根种子偏移 1000），只影响特征不影响签名标签的有效性。
3. 标签：面板 manifest（sp/op 各 10 seed）直接给出 `run_label`/`type_label`。

## 四、禁止事项与完整性要求

- 不得混入 `identity_match` 格（历史污染形态，台账 2026-07-29/08-03 条目）；
- 签名的模式集合与 E0 排序一致性分析使用的 19 个非 identity 模式保持一致，保证读数可互检；
- npz 所有数组行序一致（第 i 个前缀样本对应同一行的 signature/run/type/layout）；
- 若某前缀样本签名存在缺失（NaN），整行剔除并在日志记录剔除数，禁止插补。
