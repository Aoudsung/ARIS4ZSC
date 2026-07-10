# L1_CONSOLIDATION_PREREG — 终端能力巩固修复（预注册草案 → 签收后入 METHOD_LOCK sec18.15）

> **已归档 / 已被取代（2026-07-08）**：本文件是一份**已关闭的台账**——§1–§9 的预注册与执行记录
> 均已完成，基底修复期结论（样本饥饿是根因、探索常驻假设被证伪、2000 局基底上伙伴条件化由
> 数据自然涌现）已由 **FORMAL_ROUND_PREREG_DRAFT.md** 接手为正式轮唯一预注册。此处仅作历史
> 留存，不再作为在架的门面；下方各节标题中的"先于运行写定"是历史台账用语，见文末归档节。
>
> 治理精简 2026-07-08：本文件的门已按 GOVERNANCE_CUTLIST.md 处置；加/减门须过 OPERATING_CONSTRAINTS.md §7 退休阀门。

**Date:** 2026-07-06 · **授权**：用户 2026-07-06 "执行代码修订并运行实验"（Type-B 放行；
本条覆盖 sec18.12.4 "不得再调 exploration" 的禁令——以新预注册条目的形式，先于运行写定）
**触发证据**：EXPERIMENT_LOG 2026-07-05/06（override 门 + seed 取证 + 基线逐 checkpoint 验证史）

## 1. 机制陈述（全部实测，修正版）

- 终端链高价值：全链强制 119.1/局、终端强制 142 vs 死锁 1.0（override 门）。
- **奖励脚手架不退火**（terminal_progress_shaping 恒定：pick 0.5 / plate 1.5 / serve 0.0
  ——早先"shaping 2500 退火"的说法有误，那两个 anneal 键属于 disabled 的 role_exploration 块）。
- 真正塌缩的是**有效终端探索率 = ε × bias**：ε 0.5→0.1（config 显式设定，5000 updates；
  初稿误写为"走代码默认 0.2→0.05"，codex 部署前复评纠正——yaml 后段存在真实 ε 键）
  × 终端 bias 0.9→0.35（2500 updates）⇒ 开局 ≈36% → 末期 ≈3.5%（10 倍跌幅）。
- 基线逐 checkpoint 验证史（4 seeds，metrics.greedy_validation，已在盘）：ego 独立上菜
  只出现在 u500–1000（2/3/6/9 次，回报 40–52）；此后近乎清一色归零（seed2/4 连续 9 个零），
  晚期窗口 u3500–5000 非零率 1/4、1/4、0/4、0/4；晚期回报（5–27）**低于**早期（40–52）。
- 推断：探索支持塌缩 → 终端动作从行为分布消失（replay 5000 transitions，旧经验快速老化）
  → Q 对不再执行的动作无法维持 → 备菜/等待接管 argmax。晚期回报更低这一点不利于
  "备菜真值更优"的替代解释。

## 2. 处置（单一机制：探索调度不退火；config-only，不改代码）

新配置 `ocv2_step4_asymm_role_v2_l1fix.yaml` = e1rev 逐字拷贝，仅改：
1. `terminal_exploration.bias_end: 0.35 → 0.9`（= bias_start，退火失效）；
2. `epsilon_end: 0.1 → 0.5`（= epsilon_start 原值，按基线起始值持平、不发明新数；bias 只在
   ε 步内起作用，ε 退火会把终端通道整体乘没，故两因子同持平——一个机制）。
   实施注记：codex 部署前复评抓到重复键 BLOCKER（我在 training 块前段误插了一对
   epsilon 键，而 yaml.safe_load 取后段真键 0.5/0.1，会静默覆盖为错误值）——已删除插入、
   改真键。教训与 [[verify-experiment-condition-wiring]] 同族：运行后必须从
   resolved_config.json 读回生效值。
其余（奖励、图、CE、preflight、伙伴、预算 5000、种子协议）逐字不动。图/CE/preflight 复用
`outputs/asymm_ce_role_v2_e1rev/` 与 `results_phase2/E1_preflight.json`（与基线同一基底，单变量）。

## 3. 运行与读出（先于运行写定）

**运行**：aris_bellman × seeds {0, 2} × 5000 updates，`--save_all_checkpoints`（留全部周期
checkpoint 供后续 OOD 复测）；输出 `results_l1fix/persist_s{0,2}`。基线对照 = 已在盘的
`results_override/e1rev_aris_bellman_s*`（同 seed 同基底，唯一差异 = 本条 §2）。
接线自检：运行后从 `resolved_config.json` 读回 epsilon_start=0.5、epsilon_end=0.5 与
terminal_exploration.bias_end=0.9（不信 intent，防 yaml 重复键类事故）。

**主判据（巩固）**：晚期窗口 u3500–5000 的 4 次验证中 **≥3 次 ego_sole > 0**（逐 seed 读）。
**读出分支**：
| 结果 | 读法 |
|---|---|
| 2/2 seed 巩固 | L1 假设成立（支持塌缩是因）；在巩固基底上重测 stage-1（0.125/信念诊断/四粒度全部作废重测）；U2 顺位在其后 |
| 1/2 巩固 | 部分支持；补 seeds 1/4 再判（预注册的唯一加跑） |
| 0/2 巩固（探索常驻仍蒸发） | 支持塌缩非唯一因 ⇒ 信用结构问题（contrib_team 备菜流）成为首要嫌疑，开新预注册（credit 再平衡），不得在本条内调参 |
| 训练不稳定/守卫全失败 | 伪影流程处理，不解读 |
**附带读数（不作门）**：晚期验证回报 vs 基线晚期（预期回升向 40–52 区间）；
`terminal_exploration_pick_count` 全程曲线（确认通道确实常驻）。

## 4. 诚实边界

- 本条只裁决"探索支持塌缩是否是巩固失败的因"，**不**裁决任何 ZSC/粒度主张；
- 若巩固成立，held-out 泛化（OOD 塌缩四相）是**另一个问题**——巩固只是让 stage-1 有一个
  非瞬态的基底可测；
- ε 恒 0.2 会改变全局数据分布（不只终端通道）——若巩固成立，归因细化（ε-only vs bias-only）
  留给后续消融，本条不展开；
- codex 复评：网络持续中断时按"公式已直读核验 + config-only"记录豁免，diff 附档。

---

## 5. 执行记录与判定（2026-07-06，运行后追加）

- 接线回读通过：两 seed resolved_config 生效值 eps 0.5→0.5、bias 0.9→0.9；
  terminal_exploration_pick_count = 75 / 93（通道全程真实触发）。
- **判定：0/2 巩固**。两 seed 与基线曲线一致：u500 独立上菜 3/4 次，u1000–u5000 全零；
  best_update=500。→ 触发预注册分支三：**探索支持塌缩不是（唯一）因**。
- **执行中发现的上游事实（改变分支三的优先级）**：`updates_per_transition: 8` =
  每 transition 做 8 次梯度 ⇒ 5000 updates 仅收集 **625 条 option 转移 = 32 局训练经验**
  （episode_returns len=32 自洽）。u500 "能力" ≈ 前 3 局数据上的塑形先验；"蒸发" 是
  32 局尺度上小样本反复覆写的自然结果。**样本饥饿在信用结构假设的上游**——在 32 局
  训练量上检验 credit 再平衡不可判读。分支三的"信用结构预注册"顺延；下一个单一步骤
  应为数据规模探针（另行预注册，待 Type-B）。

---

## 6. 数据规模探针（DATA-8x，2026-07-06 用户授权"运行约 250 局的测试"；先于运行写定）

**假设**：样本饥饿（32 局）是巩固失败的上游主因。
**双臂设计**（同墙钟一波跑完，回答两个问题）：
- **A 臂 l1fix-8x**：l1fix（探索常驻）+ `updates_per_transition: 8→1` ⇒ 5000 transitions
  ≈ 250 局，更新预算不变（5000）。回答"数据+支持下能否巩固"。
- **B 臂 e1rev-8x**：e1rev 原退火调度 + 同样 8× 数据。回答"数据够了之后，探索常驻还
  需不需要"。
seeds {0,2} × 双臂 = 4 runs（GPU 1/2/4/5）；`--save_all_checkpoints`；launch 配方含
线程 caps（OMP/MKL/OPENBLAS=1）。图/CE/preflight 仍复用同一基底（单变量组）。
**主判据**（沿用 §3）：晚期窗口 u3500–5000 的 4 次验证 ≥3 次 ego_sole > 0（逐 seed）。
**读出分支**：
| A 臂 | B 臂 | 读法 |
|---|---|---|
| 巩固 | 巩固 | 根因=样本饥饿本身；探索常驻非必需（可回退 e1rev 调度），进入"巩固基底重测 stage-1" |
| 巩固 | 不巩固 | 根因=样本饥饿×支持塌缩联合；保留 l1fix 调度进入重测 |
| 不巩固 | 不巩固 | 250 局仍不足或信用结构问题；下一单一步骤=数据再 ×8（2000 局，现在只要 ~2h）后再判 credit |
| B 巩固 A 不巩固 | —— | 反直觉结果，按伪影自检流程复查接线后如实记录 |
**附带读数**：episode_returns 长度（预期 ~250）；晚期验证回报 vs 32 局基线；
guard/最优 checkpoint update 分布。

### §6 执行记录与判定（2026-07-06，运行后追加）

- 接线回读通过：A 臂 upt=1/eps 0.5→0.5/bias 0.9；B 臂 upt=1/eps 0.5→0.1/bias 0.35；
  两臂 episodes=251（数据量 ×8 确认生效）。guard 4/4 pass。
- **判定：A（l1fix-8x，探索常驻）0/2；B（e1rev-8x，原退火）2/2 巩固。**
  B-s0 晚期 ego_sole：u4000/4500/5000 = 1/2/4（回报回升至 ~33）；
  B-s2：u3500–5000 = 1/2/4/4（回报升至 ~40，u5000 仍上行）。
  A 臂与 32 局时形态一致（u500 尖峰后归零；s2 仅 u5000=2）。
- **读法（分支表第 4 行，接线复查后如实记录）**：
  1. **样本饥饿是根因**——数据 32→250 局后，*原退火调度*自然巩固，无需任何机制改动；
  2. **探索常驻假设二次证伪且有害**：ε 恒 0.5 使数据永远高噪、贪心策略无法收敛——
     退火（先探索后收敛）本来就是正确设计；l1fix 调度就此废弃，回退 e1rev 原调度；
  3. 曲线在 u5000 仍上行 ⇒ 250 局尚未饱和，正式重建应取更大预算。
- **下一步（单一，待 Type-B）**：基底重建预注册——e1rev 原调度 + 数据再 ×8
  （total_updates 40000 ⇒ ~2000 局）× 5 seeds + stage-1 held-out 重测；
  预计 ~1h 训练（并行）+ ~1h 评估（含本日提速）。

---

## 7. SUBSTRATE_REBUILD（DATA-64x，2000 局；2026-07-06 用户授权"运行 2000 局的实验"；先于运行写定）

**设计**：e1rev 原调度逐字（退火语义按原 config 键：ε 随 total_updates 相对退火、终端 bias
绝对 2500 updates——不发明新数），`updates_per_transition: 1` + `total_updates: 5000→40000`
（≈2000 局）× **seeds 0–4** × `--save_all_checkpoints`（80 个周期 checkpoint）× 线程 caps。
图/CE/preflight 仍复用同一基底。产物 `results_rebuild/`；训练完自动接 stage-1 协议评估
（25ep × 2 held-out × seed0，checkpoint.pt 与 checkpoint_final.pt 各评）→ `results_rebuild_eval/`。

**读出（先写后看）**：
- **R1 巩固**：末 4 次验证（u38500–40000）ego_sole>0 ≥3/4（逐 seed）；
- **R2 零假设（本轮主问）——条件化是否随数据自然涌现**：held-out claim 伙伴上的
  伙伴正确上菜数与 ego 全包率。分支：
  | 结果 | 读法 |
  |---|---|
  | 让位涌现（伙伴吞吐实质恢复、非全包，≥3/5 seed） | 条件化可由数据独自习得 ⇒ 表征主张重写（U2"必要性"死亡，只剩效率/可解释性口径） |
  | 仍全包（≥4/5 seed 伙伴吞吐≈0） | 条件化缺口在 2000 局仍存 ⇒ 成为 U2/信念机制在坚实基底上的诚实靶子（量化奖金 ~+23 回报/局） |
  | seed 间分裂 | 逐 seed 报告；加 seed/加局数再裁，不得挑好看的读 |
- **R3**：sec18.13.2 双判据逐 seed 报告（预期第 1 条通过；第 2 条为 R2 的门形式）；
  yield 伙伴 egoCCR 不得回退（≥0.8）。
**预算**：~1h 训练（5 GPU 并行）+ ~10 min 评估。GPU 按 ECC 现查现选（避开故障卡）。

### §7 执行记录与判定（2026-07-06，运行后追加）

- 接线：total_updates=40000/upt=1 回读确认；eps=2001 逐 seed 确认；guard 5/5 pass；
  训练 ~1h、评估 ~7min（提速后）。sel 与 final 行为逐分相同（选择器选中段 checkpoint，
  best_update 分布 3500–31500，不再挤在 u500——巩固的另一证据）。
- **R1：5/5 巩固**（late4 全 ≥3/4；末端验证 ego_sole 1–5 持续非零）。
- **R2（零假设主问）：让位涌现分支触发——5/5 seed 对 claim 伙伴完全让位**
  （ego 上菜 0/125 局，伙伴正确上菜 25–75/25 局），**且** 4/5 seed 对 yield 伙伴完全接管
  （egoCCR 1.000，25/25；s2 例外 0.000）。**伙伴条件化行为在 2000 局下由数据独自涌现**，
  未动任何方法层。
- **R3：sec18.13.2 双判据 4/5 seed 双条通过**（s2 第 1 条不过）——U2 proposal 当初设计
  要达成的预注册成功判据，被"现有方法 + 足量数据"直接满足。
- **读法（按分支表逐字）**：条件化可由数据独自习得 ⇒ **U2"必要性"主张死亡**；
  但注意：现有方法**本就含因子信念机制**——本结果不证明"信念无用"，只证明"换粒度
  (U2) 非必需"。条件化是否由信念通道承载（方法核心主张！）未裁决——下一单一步骤
  = 在本基底上重跑 E2 zeroed（置零证据通道看让位是否塌），一次便宜运行直接裁决
  信念通道是否承重。
- 范围注记：dev-heldout（2 脚本伙伴、单布局、eval seed 0、25ep）；claims 不出此范围；
  DASHBOARD/主张状态不动，待 Type-B。

---

## 8. E2-zeroed @ 2000 局基底（2026-07-06 用户授权"跑 E2 置零诊断"；先于运行写定）

**问题**：涌现的伙伴条件化（对 claimer 让位/对 yielder 接管）是否由**推断证据→因子信念
通道**承载（方法核心主张），还是由公共状态特征直接承载。
**协议**：同 stage-1（25ep × 2 held-out × seed0），5 个 rebuild checkpoint.pt，
`--zeroed_partner_option_ablation`（LDS-B3 声明式置零；完整性门核对
behavior_inferred_v1_zeroed_ablation）。对照 = 已在盘的 inferred 评估。
**读出分支**：
| zeroed 相对 inferred | 读法 |
|---|---|
| 条件化塌回无条件（对 claimer 开抢 或 对 yielder 回落死锁/不接管，≥3/5 seed 方向一致） | **信念通道承重**——方法核心主张首份干净正面证据（因子信念是角色条件化的载体） |
| 行为不变（逐 seed 与 inferred 一致） | 条件化由公共状态直接承载，信念通道非必需——主张需收窄 |
| 混合/部分塌 | 逐 seed 报告，按塌的维度（claim-让位 vs yield-接管）分开归因 |

### §8 执行记录与判定（2026-07-06，运行后追加）

- 5/5 运行完成，ablation 声明 + 完整性门全过（zeroed_partner_option: true）。
- **判定：部分塌分支触发，按维度归因清晰**——
  - **yield 维度（接管）：5/5 zeroed 与 inferred 一致**（s0/1/3/4 照常 25/25 上菜，s2 照常失败）
    ⇒ "对让位者接管"由公共状态承载，不依赖证据通道；
  - **claim 维度（对抢活者的配合）：4/5 zeroed 改变行为**——s0/s2/s3 塌为互相干扰
    （伙伴正确上菜 50/25/25 → **0**，回报 → ~7.9；同一 zeroed ego 对 yielder 仍满分
    ⇒ 能力未损，塌的是"识别并配合抢活者"本身）；s4 反向翻转（ego 也开始上菜，
    双方各 25、回报 27→58.5——证据使 s4 过度让位，zeroed 反而更优，属 rev3 §5.3
    "学习证据有害"分支的实例）；s1 完全不变（条件化由状态承载的例外 seed）。
- **读法**：**对抢活伙伴的角色配合在 3/5 seed 上由推断证据→信念通道承重**——方法核心
  主张（因子信念承载角色条件化）获得首份干净的正面证据，且有诚实的 per-seed 纹理
  （1 例状态承载、1 例证据有害）。结合 32 局时代 E2（zeroed==inferred）：通道从
  "无作用"变为"承重"，发生在数据量修复之后——"机制没错，数据饥饿"叙事再添一环。
- 范围：dev-heldout、单布局、25ep、eval seed 0；主张按此范围表述。

---

## 9. E1 多臂对照 @ 2000 局基底（2026-07-06 用户授权"启动"；先于运行写定）

**设计**：base_only / global_gru / flat_factor / partner_id_q × seeds 0–4（20 训练），
config = `ocv2_step4_asymm_role_v2_e1rev_data64x.yaml` 逐字（同图/CE/preflight/协议/数据量；
方法由 CLI --method 指定）；aris_bellman 臂复用 §7 已在盘结果。评估 = stage-1 同款
（25ep × 2 held-out × seed0，checkpoint.pt）。**定位：对照筛读**（论文终表另走盲测+50–100ep）。
**纪律**：每臂入档必报有效局数（应 2001，产物回读）；ITT（guard-fail 照记不剔除）。
**筛读判据**（沿用 sec18.13.2 形式）：行1 = yield 伙伴 egoCCR>0；行2（screen 版）=
claim 伙伴 partner_correct ≥ 25/25ep 且非全包（ego_dlv=25 且 partner=0 记全包）。
**读出分支**：
| 结果 | 读法 |
|---|---|
| 仅 aris 双条通过（每 baseline ≤1/5 seed 双过） | 四粒度对照成立（公平版：同基底同数据同协议全新训练）——方法主张=对照+因果（E2）双证据链 |
| 某 baseline ≥3/5 双过 | 角色自适应非因子信念特有 ⇒ 主张收窄；对通过的 baseline 同跑 E2-zeroed 比较通道承重性（gru 也消费证据路由，zeroed 对其同样定义），差异化转因果/可解释层 |
| base_only 行2 通过 | 判据判别力异常 ⇒ 伪影自检流程，不解读 |
| partner_id_q（准 id 参照）通过 | 照实报为参照上界，不入四粒度主对照行 |

---

## 已归档 / 已合并（2026-07-08，依据 GOVERNANCE_CUTLIST.md）

- **整份 L1 台账——已关闭并被取代**：§1–§9 的预注册与执行记录已完成，基底修复期结论已移交
  FORMAL_ROUND_PREREG_DRAFT.md（正式轮唯一预注册）。本文件作历史留存，不再是在架门面。
- **「先于运行写定」（write-before-run）——已合并**：该"决策判据先写后看"的原则与 prereg
  冻结决策规则同源，单一归属在冻结决策表 / 分支表（现由 FORMAL_ROUND_PREREG_DRAFT.md 承载）；
  L1 各节标题中重复出现的"先于运行写定"是历史台账用语，不再作为独立重述的规则。冻结决策规则
  本身（对应 INC-4，挡事后叙事套噪声）未削弱，只是收归单一归属。
