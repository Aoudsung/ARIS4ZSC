# EXPERIMENT_LOG.md — 正式实验结果记录（ARIS 惯例）

**Created:** 2026-07-02（Phase 0 · G0.5）· **Status:** active historical
record; latest recorded scientific state = Link-A round 2 failed on 2026-07-09,
and the active project line is now Path C.
**规则**：本文件记录执行门控下的正式运行、已经成为方向决策依据的诊断终读、以及基底证书终读。
每条 claim 读数必须引用对应预注册或冻结判读规则，禁止事后重解释。NEW-4 隔离材料
（CODEX_IMPL_SPEC v1–v4 数字）永不入此文件。

## 当前状态快照 — 2026-07-09

- **旧不对称布局 + role-conditioned v2 伙伴路线已关闭为正向论文路径。** 32 局训练量上的负面诊断已被
  2000 局基底推翻，随后正式盲测又显示 dev-heldout 的伙伴条件化没有迁移到新盲测伙伴；
  这些记录可作为受限负结论和方法转向依据，不能作为 ARIS-Bellman 正向主张证据。
- **Link-A 基底证书第 2 轮失败。** 终端交接机会仍能从瞬时公共状态读出；等待观察后再反应
  捕获了接近 oracle 的价值，预判式信念没有被迫产生额外价值。下一条基底线必须让等待或
  反应付出真实代价。
- **当前活跃线是 Path C。** Path C 指 active value probing for value-sufficient residual
  partner abstractions，即用价值驱动探针恢复公共状态之外仍改变控制的最小伙伴抽象。Path C
  目前只有静态方案、默认关闭代码脚手架和预注册草案；本文件尚无 Path C 运行结果。
- **下一条可记录事件。** Path C 预注册冻结或 Phase A 静态核验完成；不得把任何小批量接线
  检查读成科学结论。

### 2026-07-11 Path C 远端完整软件测试 — PASS（Type-A，非科学结果）
- 范围：只验证 `experiments/overcooked_v2/tests`；本地未运行项目代码、训练或实验。
- 远端：`zsc-customer` 的 `/apps/users/cxw/Document/CodeSpace/Selfs/CPR_REPO`，GPU5，项目虚拟环境与 CUDA 12 包装脚本。
- 依赖修订：完整运行首次发现 `pyarrow` 未声明，导致九项 Parquet 测试跳过；现已将 `pyarrow>=14.0` 加入项目依赖，远端安装版本为 25.0.0。启用后修复了无符号 64 位 seed 写入溢出，并把重复的分组 seed 校验缓存为每个唯一分组与 seed 组合只校验一次。
- 最终结果：退出码 0；269 项全部通过、0 项跳过、0 项失败。
- 日志：远端 `.codex_remote_validation/path_c_full_20260711_final_all.log`；SHA-256 为 `637f3c6c9e21926578558bcc6a87cca6b7b9a65d39f6f2139fd80cd8dd64dba6`。
- 状态限制：测试对象是基于 `07fa7a63f3a61a95616f4748fd11eb634afd3c5b` 的未提交修订。模块注册表因此只把全套件执行状态改为 `passed`，不把任何模块提升为 `tested` 或 `frozen`；提升仍需最终提交并绑定逐测试通过记录。

### 2026-07-11 Path C P1–P5 代码评审修订验证 — PASS（Type-A，非科学结果）
- 评审来源：`review_bundles/path_c_v3_code_review_20260711/FINDINGS.md`。修订覆盖阶段化判定、64 位 seed 身份链、完整 group×seed 排程、探针数值域和被弱化的生产路径测试。
- 可见性证明：先只恢复测试，远端出现 13 个预期失败；日志为 `.codex_remote_validation/path_c_findings_p1_p5_red_20260711.log`。
- 目标回归：23 项新增或恢复测试全部通过；日志为 `.codex_remote_validation/path_c_findings_p1_p5_target_green_20260711.log`。
- 完整回归：退出码 0；290 项全部通过、0 项跳过、0 项失败；日志为 `.codex_remote_validation/path_c_findings_p1_p5_full_20260711.log`。
- 代码绑定：P1–P5 修订已提交为 `fc647fb00255c5bfc58253a38fa145cb8864afd7`。
- 提交绑定的 CPU 诊断：290 项通过、0 项跳过；只用于确认工具链故障不是代码断言失败，不能替代 CUDA 验收。
- CUDA 工具链事故：首次直接调用系统 CUDA 11.8 `ptxas`，因不支持 JAX 生成的 PTX 8.3 在收集阶段失败；切换到虚拟环境 CUDA 12.9 后，GPU 3 又报告不可纠正 ECC 错误，因此该卡被排除。
- 提交绑定的正式 CUDA 验收：GPU 5 的不可纠正 ECC 计数为 0；JAX 0.4.38 / jaxlib 0.4.38 在 GPU backend 完成实际编译计算，随后完整套件 290 项通过、0 项跳过、0 项失败，耗时 423.81 秒。日志为 `.codex_remote_validation/path_c_r002_bound_fc647fb_cuda_gpu5_20260711.log`，SHA-256 为 `6b30b0cd623ebe650f3ea084de2dafb8cea64aef7bb8413587ab70c6b17fbc85`；JUnit 逐测试报告为 `.codex_remote_validation/path_c_r002_bound_fc647fb_cuda_gpu5_20260711.xml`，SHA-256 为 `d40b43bc7084c8ce916bbc537fd7a6919da28b8b3a760ef552b197b3199f97ab`。
- 正式软件测试报告：从上述 JUnit 和最终模块注册表生成 101 条注册测试记录；文件为 `.codex_remote_validation/path_c_software_test_report_fc647fb_20260711.json`，SHA-256 为 `61f669b0c980f454a138eadd22ed3a846621bb94e86e8bec568023b546af8bf4`，绑定模块注册表 SHA-256 `2c22792a0a16318f63ee2efa6dc9adcd542c9a78377c65f650e8d17577e4adb7`。
- 状态：具备完整注册测试覆盖的 21 个模块提升为 `tested`；C2、C5、D1 仍为 `planned`。预注册未冻结，E0、训练、design 选择、locked audit 与任何科学读数均未运行。

### 2026-07-11 R003 布局与 option 语义兼容性扫描 — 失败关闭（Type-A，非科学结果）

- 授权与范围：用户明确要求继续执行 R003；本轮只检查非数值语义接线，不读取
  `locked_audit`，不训练，也不产生可用于研究主张的读数。
- 执行设备：正式扫描通过
  `experiments/overcooked_v2/scripts/with_jax_cuda12.sh` 运行，固定
  `CUDA_VISIBLE_DEVICES=5`。GPU 5 是 NVIDIA L40，运行前显存占用 77 MiB，易失性
  SRAM/DRAM 可纠正与不可纠正 ECC 计数均为 0；产物回读
  `jax_default_backend=gpu`、`selected_device_kind=NVIDIA L40`。一次先行的模块路径查询
  未固定设备并触发已知 GPU 3 ECC 错误，没有生成结果；随后全部 JAX 检查均改用上述
  GPU 5 包装脚本。
- 提交绑定：远端仓库的基础提交不是同步源码的提交，因此不能把远端 `git HEAD` 当作
  代码身份。本轮逐文件核对七个实际读取的项目源码，远端 SHA-256 与本地完全一致，且
  这些文件从 `fc647fb00255c5bfc58253a38fa145cb8864afd7` 到当前证据提交无差异；绑定产物
  明确记录该源码提交、远端基础提交以及两个外部 JaxMARL 源文件的 SHA-256。
- proposal 候选布局
  `asymm_advantages`、`asymm_advantages_recipes_center`、
  `asymm_advantages_recipes_left`、`asymm_advantages_recipes_right` 的公共观测维度均为
  96，但 option 数分别为 28、28、26、27，option 名称与数值编号签名全部不同。
- 全库复核：JaxMARL 登记 23 个布局；22 个双智能体布局可完成环境重置、96 维公共观测
  构造与 option 库构造，`long_room` 因只有一个智能体而被固定双智能体证据合同拒绝。
  22 个可用布局中，按“观测维度 + 有序 option 名称与编号”分组的最大组大小为 1，低于
  四角色隔离所需的 4；不存在可直接复用的四布局组合。
- 判定：不能让四个不同布局共用当前单一 `EgoEvidenceSpecV1` 和裸 option 编号电池。用户随后
  明确要求删除非科研必需的绑定与门控；因此不实现跨布局动作映射层。主 identity 实验改为
  四角色固定同一 `asymm_advantages` 布局，只隔离 identity、style 与 seed group；layout
  shift 使用另一布局单列为 secondary。这也避免把 layout 与 identity 同时移出分布。
- R003 保持进行中而非失败：split manifest 第三版已删除跨角色 layout 隔离，同时要求主
  identity stratum 使用一个具体布局。最终 battery 数量、身份组数与数值 seed 仍由后续
  calibration 决定，不提前制造哈希闭包。
- 绑定产物：远端
  `.codex_remote_validation/path_c_r003_all_layout_semantics_bound_fc647fb_20260711.json`，
  SHA-256 为 `6268916dc37a5571e024224e919e596e3d80e1346cd3c86d74266ee8db4798a1`。

### 2026-07-09 CUDA/JAX 远程环境体检 — PASS（Type-A，非科学结果）
- 范围: 只修复 `zsc-customer` 上项目运行环境对 JAX/CUDA 的选择；不改系统
  `/usr/local/cuda`，不重装 JAX，不运行训练，不生成 Path C 科学读数。
- 落地: 新增 `experiments/overcooked_v2/scripts/with_jax_cuda12.sh` 和
  `experiments/overcooked_v2/scripts/check_jax_cuda.py`。包装脚本把虚拟环境里的 CUDA 12
  `ptxas` 放到 `PATH` 最前面，设置
  `XLA_FLAGS=--xla_gpu_cuda_data_dir=<venv>/site-packages/nvidia/cuda_nvcc`，并默认使用
  `CUDA_VISIBLE_DEVICES=5`、`JAX_PLATFORMS=cuda`；GPU3 因 ECC 错误默认避开。
- 体检命令:
  `CUDA_VISIBLE_DEVICES=5 bash experiments/overcooked_v2/scripts/with_jax_cuda12.sh .venv/bin/python experiments/overcooked_v2/scripts/check_jax_cuda.py`
- 体检结果: `jax=0.4.38`、`jaxlib=0.4.38`；
  `ptxas=.venv_tgssa/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin/ptxas`；
  `Cuda compilation tools, release 12.9, V12.9.86`；JAX backend=`gpu`；
  devices=`['cuda:0']`；最小编译 `jnp.arange(4).sum()` 返回 `6.0`。
- 负向保护: `CUDA_VISIBLE_DEVICES=3 bash experiments/overcooked_v2/scripts/with_jax_cuda12.sh true`
  在 JAX 初始化前退出，提示 GPU3 有 ECC 错误，应选择其他 GPU。
- Path C 接线检查:
  `CUDA_VISIBLE_DEVICES=5 bash experiments/overcooked_v2/scripts/with_jax_cuda12.sh .venv/bin/python -m pytest experiments/overcooked_v2/tests/test_path_c_scaffold.py -q`
  → `11 passed, 1 warning`。该结果只证明远程环境和脚手架可运行；不得读成 Path C 的方法证据。

## 记录模板

```
### <日期> <实验ID per EXPERIMENT_CHAIN_PLAN> — <一句话结果>
- commit: <hash> · config: <path> · 远程产物: <path>
- 完整性: reward_scale_verified=<> evidence_policy=<> oracle_source_count=<> gate=I1-I17 <>
- 规模: seeds=<> episodes/partner=<> partners=<train/heldout 列表>
- 主指标: <ego_correct_completion_rate / throughput ± CI>
- 预注册分支命中: <sec18.x 表格行>
- 伪影自检: <5项清单逐项 / ARTIFACT-SUSPECT 标记>
- Type-B: codex verdict=<> · 人类裁决=<> · 日期=<>
```

## Phase 1 — 基础设施验证（R1.1–R1.3）

### 2026-07-03 Phase 1 — 全部通过（Type-A，基础设施验证，非主张实验）
- commit: `4d680ab` · 远程: `Selfs/ARIS4ZSC-phase1-4d680ab`（git archive 全新同步 + SYNC_PROVENANCE）· GPU 5（避开 ECC 故障的 0/3）· JAX_PLATFORMS=cpu
- **R1.1 ✅**：fidelity gate 远程 exit 0（I1–I17）；pytest 全套件 exit 0（10 文件，~93 项）
- **R1.2 ✅**（cramped_room debug 冒烟，CE 20ep/伙伴 → preflight accepted → train 400 updates → eval 8ep×2 伙伴）：
  - 完整性: `evidence_policy=behavior_inferred_v1` · `oracle_source_count=0` · `observed_dist=0` · `missing=0` · **`reward_scale_verified=true`（曾永假的死旗首次真验证通过，S18 闭环）** · `headline_success_metric=ego_correct_completion_rate` · `allow_diag_skip=false` · `role_match_status=removed_from_formal_main_path_p5`
  - P3 sidecar 四掩码齐全（estimable/skipped/measured_zero/weight_sum + 计数 + 参数）
  - **覆盖门两次正确 fail-closed**（8ep 与 30ep 随机策略下终端选项支持不足 → 拒绝建图，A1 机制真实工作）；冒烟经 `--no_require_task_stage_coverage` + `graph.require_task_stage_coverage=false`（ce_path 分支）放行 —— **正式跑禁止此路径**（见 CHAIN_PLAN §10 执行卡）
  - completion=0.0 为预期（400-update 冒烟 + 覆盖不全图；task_progress_events=84）；非本次验收目标
  - 远程 config 补丁（仅冒烟、未回写本地）: debug.yaml 加 `graph.replay_path`（preflight 复用 replay）、`diagnostics.proxy_episodes=8`、`ce_path=ce_refined.npy`、去 `graph_path`、覆盖门关
- **R1.3 ✅**（NEW-2 历史产物审计）：扫描全部远程目录 155 个 metrics.json → **3 个命中**（guard=fail 且 checkpoint.pt 在盘）：`CPR_REPO/results_rcfix/{asymm_egocredit,armA,armB}/.../seed0`（6/29 RC 修复时代，当时即记录为失败、从未作正面引用 → **无结论污染**）。已就地写 `QUARANTINE_NEW2_guard_fail.txt` 标记（不删数据）
- 实测单位耗时（可行性分析依据，CHAIN_PLAN §9）: CE 采集 ≈7.4s/ep/伙伴（cramped 顺序）· train 启动 ≈2.5min + ≈0.32s/update · eval ≈1304s/(8ep×2伙伴含参照基线) · preflight 复用 replay 后秒级
- 伪影自检: 不适用（无主张结论）· Type-B: 不适用（Type-A 基础设施验证，按 OPERATING_CONSTRAINTS §4 自判可过，留档）

## Phase 2 — 基底证书（R2.1 伙伴可区分性 · R2.2 asymm CE 支持度）

### 2026-07-03 R2.1 — 差异化 PASS，但可容许判别条件仅限 bottleneck-导航（非 serving-推断）
- 脚本: `scripts/rc2b_partner_differentiation.py`（新，checkpoint-free FSM-ego 探针）+ `rc2b_substrate_certificate.py`（修 partner_set bug）· config: `ocv2_step4_asymm_role_v1.yaml`（partner_set=role_conditioned_v2）· GPU-free（JAX-CPU）
- 产物: `review_bundles/phase2_R2.1_certificate_20260703/{partner_differentiation_v2.json, substrate_cert_v2.json, *.log}`
- **差异化探针（8ep, max-opt 40）：VERDICT=PASS（2/2 因子）**——serving 轴 spread=1.0、bottleneck 轴 spread=1.0、可区分对 27/28。**伙伴库有真实行为多样性**（不同于 sec16 标准伙伴的塌缩）。
- **可容许性证书（5ep, max-opt 80，fsm/random/partner-only）揭示关键分层**：
  ```
  ingredient-near/far-yield : fsm1.0 rand1.0 ponly0.0  → random 也解 = 太易，技能无关
  server-left/right-claim   : fsm1.0 rand1.0 ponly1.0  → partner 独解 = ego 无关
  bottleneck-yield-term-yield: fsm1.0 rand0.0 ponly0.0 → 可容许判别 ✓（train）
  bottleneck-push-term-claim : fsm1.0 rand0.4 ponly0.0 → 可容许判别 ✓（train）
  heldout-handoff-alt-yield  : fsm1.0 rand0.0 ponly0.0 → 唯一被ADMIT的held-out ✓
  heldout-resource-server-claim: fsm1.0 rand1.0 ponly1.0 → partner独解 = 退化 ✗
  ```
- **深层发现（与 METHOD_LOCK sec11 一致）**：asymm 上"谁 serve"因子**行为可区分**，但两模式都不给可容许判别条件（ego-serve 太易 / partner-serve ego 无关）。**唯一可容许判别条件 = bottleneck 导航伙伴**，而 bottleneck 是 throughput-navigable 空间因子（导航技能，非因子推断）。当前 6 训练+2 held-out split 里**只有 1/2 held-out 可容许**（heldout-handoff-admit；heldout-resource 退化）。
- **伪影自检**：max-opt=40 探针曾误报 bottleneck/held-out compl=0（伪影，80 opt 下 fsm=1.0 全部完成）→ 已修正解读。ingredient-far-yield fsm=0.0 但 rand=1.0 = FSM-ego 确定性盲点（探针质量注记，非基底问题）。
- **sec18.4 分支命中**："Distinguishable on a factor subset only → Narrow: re-cut train/held-out along the distinguishable subset"（差异化真、但可容许判别子集需重划 split）。**非干净 PASS→E1；需用户裁决 split/layout**（fork 决策，Type-B）。R2.2（serving CE 是否真非零）将决定 asymm 能否测推断 vs 仅导航。

### 2026-07-03 R2.1b — 吞吐口径 4 布局横扫 + sanity（sec18.9）
- 脚本: `scripts/rc2b_throughput_certificate.py`（新）· config: `ocv2_step4_asymm_role_v1.yaml`（正扫）+ `_tmp_sanity_v2.yaml`（sanity, 已清）
- 产物: `review_bundles/phase2_R2.1_certificate_20260703/{throughput_certificate.json, throughput_sanity_cramped_debug.json, *.log}`
- **正扫结果（4 布局 × 8 伙伴 × 3 seeds × 5 ep × 3 policies = 1440 ep）：0 admit**——但数据反常：
  - `rand_tp > fsm_tp` 广泛出现（asymm/server-*: rand=9.07 vs fsm=1.0）——物理上 FSM 必赢或平手，倒挂说明 FSM 在**争抢终端**
  - forced_coord/coord_ring 几乎全 0 → sec18.9.4 判读表兜底命中："probe budget too small → escalate, do NOT flip verdict"（先怀疑参数）
- **Sanity（cramped_room × v2 × debug executor: `force_path_planning=false`, `max_option_steps=6`, 2 seeds × 3 ep）：** 8 伙伴 **fsm_tp 全 0**，rand_tp 0.17–1.00，ponly_tp 全 0。**这排除了 executor 参数假说**。
- **核心诊断（重要发现）**：R2.1 探针的"FSM 是称职 ego"假设**不适配 v2**。FSM 是刚性 inventory-based pipeline，对 partner role/位置无感；v2 partners 是 role-aware（yield/claim/handoff），需要 role-aware ego 才能协调。FSM 与 partner 争抢终端 → 拖累吞吐 → 探针拒绝任何 substrate。**这是探针失败，非 substrate 失败**。
- **同时收获的正面信号**：cramped × v2 上 **ponly_tp 全 0 across all partners**（v2 partners 不能 solo，与 asymm 完全不同），但 rand ego 存在时能促成送餐——说明 cramped × v2 上 **ego 是必要的**，只是 FSM 太笨。这个信号 asymm 上没有（asymm/server-*: ponly=1.0 solo）。
- **R2.1 探针框架结论**：sec18.4/18.9 的"FSM-oracle admissibility"路径**无法通过 v2**——不因 substrate，因探针方法论。差异化 PASS 是稳固信号（v2 有真实行为多样性），但 admissibility 需换探针（role-aware oracle）或换判据（训练本身作为 admissibility test，即 E1 本身）。

## Phase 3 — 决定性实验（E1 四臂去 oracle 重跑 · E2 通道消融 · E3 持久化消融）

### 2026-07-03 E1 前置 CE（asymm×v2）— 两次 GraphCoverageError → 根因 S27（推断器支撑集冻结）
- 采集: 100ep×6伙伴=12000 rows（3h50m, GPU6/JAX-CPU）· 覆盖门 passed · sidecar 四掩码齐全
- 建图两次失败: `plate_soup/serve_soup 无 above-eta CE 候选`（min_weight 20→10、eta 0.05→0.02 复用 replay 重试无效——`--reuse_replay` 功能顺手落地, commit 9dfb258）
- **支持度审计决定性读数**: 终端选项作为伙伴列联合质量**精确 0.0**；73.9% 伙伴占用在 noop；
  ego-terminal 行全部质量落 (terminal, noop) cell → 被 kernel noop 排除 → CoverageError
- **现场实验（铁证）**: server-left-claim 单 ep 送餐 9 次（plate+serve 全程执行），行为推断器
  终端选项质量恒 0.00000000, 每次送餐 argmax=noop → **S27 支撑集冻结确证**
  （reset 冻结初始有效集为 belief 支撑, 乘性更新 0×x=0 永久锁死）
- 台账: S27(实现bug,高) + D6(互斥估计量盲区, D4精化) + D7(noop排除×yield签名冲突)
- **裁定**: 主导=实现 bug（P1 修复激活死代码中的潜伏缺陷）; 设计放大器 ×2; proposal 层非主因
- 下一步（待批）: S27 修复（支撑注入/遗忘因子, codex diff 评审）→ **重采 CE**（现 replay 的
  partner dist 已污染）→ 建图 → E1

### 2026-07-04 S27 修复远程验证 — PASS（决定性）
- 单元测试: test_s27_support_injection (4例, 含 mix=0 冻结复现锚点) + E2/E3/缓存回归 → exit 0
- **现场探针复验（同 claim 伙伴 9 次送餐场景）**: 修复前终端质量恒 0.00000000 →
  修复后送餐瞬间 terminal mass=0.9246, **9/9 次 argmax=serve_soup 精确命中**, 峰值 0.9866

### 2026-07-04 E1 前置 CE（S27 修复后重采）— PASS，E1 解锁
- 100ep×6伙伴×asymm_v2, 3h50m (S27 修复带来推断器质量提升→选项终止更快→采集加速~30%)
- 产物: outputs/asymm_ce_role_v2_e1_s27fixed_retry2/{graph.json, ce_refined.npy, ce_support_audit.json, replay.npz}
- 归档: review_bundles/phase3_E1_ce_20260704/（含 pre/post 对照）
- **S27 修复的 CE 层独立确证**:

  | 指标 | PRE-fix | POST-fix |
  |---|---|---|
  | estimable pairs | 45 | **80** (+78%) |
  | partner=noop 占用份额 | 73.9% | **13.8%** |
  | partner col opt2/3/9 联合质量 | 全 0.0 | 547 / 149 / 199 |
  | ego row opt3 主质量 | opt27[noop]=52 | opt25[wait]=19.6, opt24[cross]=18.7 |

- **建图两阶段解锁**:
  - S27 修复后重采 → 首建仍失败: `opt7 pick_plate 无 above-eta 候选`（新错误位置）
  - **诊断揭示 substrate 事实**: asymm 上随机策略 ego 从未访问 opt7/opt9（ws=0），opt3/opt8
    质量分散 <10 全 skipped。**S27 修复解除了 noop 单极坍缩，暴露 substrate 真实结构中的
    valid ID 边缘化**（不是估计问题，是访问频率问题）
  - **U1 论文证据线得到强化实测**: 不是"我们推测被动 CE 有盲区"，是"我们精确测得 opt7/9
    ws=0、opt3/8 分散低支持"——直接就是 targeted-starts 要解决的场景
  - **工程解**: kind-level coverage（放弃 per-ID）+ min_weight 5 + eta 0.02，复用 replay 建图
    成功 → **16 因子**（3 serving 含 opt3/opt9、4 bottleneck、5 resource、1 pot_allocation、
    3 generic），CE 分数 1.4–5.8，selected_by 分布 3 mandatory kind + 3 role contrast + 10 ce_fill
- **正式 E1 config**: `configs/ocv2_step4_asymm_role_v2_e1.yaml` (graph_path 指向 retry2 产物 +
  min_weight=5 + eta=0.02 + kind-only coverage)
- 决策记录: opt3/7/9 无因子（substrate 边缘化的诚实记录）；kind coverage 已被 opt2/opt6/opt5
  覆盖 pick_plate/plate_soup；serve_soup 通过 opt9 (via low min_weight) 覆盖
- **下一步**: preflight（复用 replay 秒级）→ E1 四臂决定性跑（sec18.6/18.10.2 预注册读出）

## Phase 4 — 主张级实验（E4–E7）

*(待运行)*

## Phase 5 — 锁定 → 盲测 → 终表

*(待运行)*

### 2026-07-04 E1(no-scaffold) wave — ORCHESTRATOR BUG (data loss), partial result retained
- **Bug (mine)**: `logs_phase3/E1_wave_orch.sh` used `local method=$1 ... out=results_phase3/E1_${method}_s${seed}`
  in a SINGLE `local` declaration → `${method}`/`${seed}` empty when `out` computed (bash gotcha) →
  every run's `out=results_phase3/E1__s` + `rm -rf $out` at run start **wiped all prior runs' outputs**.
  CLI `--method`/`--seed` (separate refs, post-declaration) expanded fine → correct subpath but shared
  parent dir. Only the last-completing run survived on disk.
- **Captured before wipe (live reads, training-phase ego/partner delivery counts)**:
  - aris_bellman: s0=0/8, s1=1/12, s2=0/10, s3=0/11, s4=0/6 (all 5 seeds — ego≈0)
  - base_only: s0=0/7 (ego=0)
  - partner_id_q: s4=1/9 (survived on disk; ego≈1)
  - global_gru, flat_factor: NOT captured before wipe → lost
- **Scientific status**: the no-scaffold "ego≈0 vacuum" headline (sec18.12.1 trigger) is established
  qualitatively from aris×5 + base×1 + partner_id_q×1. The FULL 4-arm×5-seed ablation table is LOST.
  Per sec18.12.4 this table is the scaffold-ablation baseline — **deferred re-run** (cheap, ~5h overnight,
  only needed if the paper requires the complete no-scaffold ablation vs E1-rev).
- **Fix**: orchestrator rewritten with per-(method,seed) unique output dirs + no cross-wipe; used for E1-rev.
- **Guard/eval note**: no deployable checkpoints existed (guard fail across arms), so nothing was
  read against sec18.6 — no claim contaminated. Loss is of the ablation record, not of a decisive read.

### 2026-07-04 E1-rev pilot (scaffolds ON) — DECISIVE: scaffolds unlock terminal competence
- config: `ocv2_step4_asymm_role_v2_e1rev.yaml` (contrib_team + ego-kind terminal_progress_shaping
  + terminal_exploration, both P5-audited ego-local) · graph: `outputs/asymm_ce_role_v2_e1rev/`
  (sparse-CE rebuild, reused replay) · commit b9dea6f→(config final)
- **objective gate resolution (2 iterations, gate working as designed — S11/T1)**:
  (1) shaping-on config vs shaping-off graph metadata rejected → rebuilt graph with
  `--sparse_ce_support` (excludes terminal bonus from CE support selection, G3/S11 path);
  (2) graph metadata `sparse_ce_support:true` vs config-unset rejected → set
  `graph.sparse_ce_support: true` in config. Both are the anti-stale-graph gate correctly
  enforcing CE-objective ↔ training-objective consistency. NOT failures.
- **RESULT (aris_bellman s0, 5000 updates, 1280s)**:

  | metric | E1(no-scaffold) s0 | **E1-rev(scaffold) s0** |
  |---|---|---|
  | ego_correct_delivery | 0 | **18** |
  | ego_sole_correct_delivery | 0 | **2** |
  | served_soup_count | 0 | **29** |
  | free_rider_guard | fail | **pass** |
  | deployable_checkpoint | None | **checkpoint.pt** |
  | best_greedy_return | None | **46.15** |

- **sec18.12.1 trigger CONFIRMED**: ego≈0 was the P5-clean-scaffold vacuum, not a method failure.
  This is the FIRST genuinely-usable training result in the project — under honest (de-oracled) +
  fair (oracle-free scaffolds) conditions the ego LEARNS to serve (18 correct, 2 ego-sole, guard pass).
- **Full E1-rev wave launched** (fixed orchestrator, 4 arms × 5 seeds, unique dirs). Read per
  sec18.6 + sec18.10.2 + sec18.12.3 once all 20 runs + partner_id_q reference land.

### 2026-07-04 E1-rev 全 wave 完成（25/25）· 潜伏缺陷扫荡 · 窗口一修复验证 PASS
- **wave**: 5 臂 × 5 seeds 全部 exit=0；产物审计全干净（`final=True/run_status=ok/updates=5000`
  ×25，含 LDS-C1 直接核验——skip-on-existence bug 存在但本 wave 无 resume 未触发）。
- **codex 潜伏缺陷扫荡**（3-pass xhigh，`b11af06`）：12 发现（3A/7B/2C），**无一作废已完成
  wave**；窗口一修复 F1–F5 落库（`98fe149` eval 层 + `6ece080` 汇总脚本），codex diff 复评
  BLOCK→4 阻塞项修复→APPROVE-WITH-NITS；远程验证：定向回归 **33 passed**、gate I1–I17
  exit 0（archive 缺 `.aris/`（untracked）致首次 exit=2 误报，copied tool 复跑 = 0）。
- **train 期表**（aggregate_e1rev v1，**非决定性**——决定性 = held-out eval）：

  | arm | seeds | guard pass | competence rate | selCCR (pass) | ego_sole (pass) | best_ret (pass) |
  |---|---|---|---|---|---|---|
  | aris_bellman | 5 | 4 | 0.80 | 0.65 | 5.0 | 42.4 |
  | base_only | 5 | 5 | 1.00 | 0.80 | 7.2 | 64.2 |
  | flat_factor | 5 | 5 | 1.00 | 0.88 | 7.2 | 53.8 |
  | global_gru | 5 | 5 | 1.00 | 0.84 | 6.0 | 54.0 |
  | partner_id_q | 5 | 5 | 1.00 | 0.88 | 7.6 | 56.3 |

- **sec18.12.3 第一行触发（远超阈值）**：全部 5 臂 ≥4/5 seeds 达成终端能力 ⇒
  **E1-rev 即决定性 run**，判读走 sec18.6 五分支 + sec18.10.2 admissibility 四行。
- 观察记录（不解读）：aris_bellman 是唯一有 guard fail 的臂（s3）且 train selCCR 最低；
  train-partner 指标与 held-out ZSC 泛化是两回事——一切结论等两段式 held-out eval。
- **下一步**：stage-1 held-out eval（25ep × 2 held-out × ~24 deployable ckpts，基线缓存
  schema v3 + 固定 eval seed），按 §14.2 执行卡（待授权）。

### 2026-07-05 stage-1 held-out eval 完成（24/24）· sec18.13 诊断链闭合 · 机制定式
- **stage-1 决定性表**（25ep×2 held-out，bootstrap 95% CI，产物 results_phase3_eval/stage1_decisive_table.*）：

  | arm | egoCCR [95% CI] | team tp/ep | ego tp/ep | serve share |
  |---|---|---|---|---|
  | aris_bellman | **0.125** [0.000, 0.375] | 1.000 | 0.125 | 0.250 |
  | base_only | 0.900 [0.700, 1.000] | 1.200 | 0.900 | 0.817 |
  | flat_factor | 0.800 [0.600, 1.000] | 0.900 | 0.800 | 0.950 |
  | global_gru | 1.000 [1.000, 1.000] | 1.100 | 1.000 | 0.950 |
  | partner_id_q | 0.800 [0.600, 1.000] | 0.800 | 0.800 | 1.000 |

- **三表读出**：sec18.10.2 行 1 触发（base 0.900 ≥ 0.9 ⇒ **基底非 ZSC-判别**，任何相对排序
  不得读作 ARIS 优势/劣势的主张证据）；sec18.6 落「仅 ARIS 崩」行（按预注册=可疑回归待查）。
  次级并读：resource-server-claim 上 aris 角色互补（partner 吞吐 2-3/ep、团队回报 59>base 38.7）；
  handoff-alternate-yield 上全臂唯 aris 死锁（partner_id_q 亦 2/5 死锁——该伙伴对 ID-oracle 也难）。
- **sec18.13.1 判决：POLICY-DEFAULT-WAIT**——zeroed 重评 4/4 seeds 行为与 inferred 逐 seed
  一致（塌缩不变）⇒ 信念通道非瓶颈。E2 表（sec18.7）zeroed 行同时完成。
- **trace（非 fast, 3ep）**：belief_zero_delta≈20.5（信念显著平移 Q）但不翻转 wait→act 排序。
- **训练伙伴探针矩阵（关键补充）**：
  | seed × 训练 yield 伙伴 | ingredient-near-yield | bottleneck-yield-terminal-yield |
  |---|---|---|
  | s0 | 0.32 | 0.16 |
  | s2 | **1.00**（52 送餐） | **0.00** |
  接管行为**分布内即呈斑驳状**（同 seed 对某 yield 伙伴满分、对另一 yield 伙伴零分）；
  bottleneck-中介的 yield 变体是系统性薄弱轴；held-out alternate-yield 位于最难角。
- **机制定式（三证据闭合）**：ARIS 策略**默认让位**（zeroed=wait），信念仅对**识别出的
  分布内伙伴**解锁接管（s2×near-yield=1.0 证明解锁存在且可完美）；OOD 伙伴 → 回落默认 →
  对 yielder 死锁。方法的伙伴条件化能力把训练分布的让位偏置学到最彻底——base/gru 学不会
  条件化反而获得 ZSC-鲁棒的无差别单干。诚实结论：**条件化解锁 vs 鲁棒默认的张力**是本轮
  最重要的科学发现，独立于后续修复成败均可入文。
- 附：诊断链工程记录——LDS-A1/B3 修复（`9e4d2f1`，codex APPROVE）解锁受控 zeroed 跑；
  探针首launch 因嵌套引号本地展开失败（编排器事故同族），改脚本文件后成功（教训入档）。
- **下一步**：群体压力干预新预注册（草案见
  review_bundles/latent_defect_sweep_20260704/E1REV2_POPULATION_PREREG_DRAFT.md）待用户 Type-B。

### 2026-07-05/06 override 门 + seed 行为取证（e1rev 重建基底；用户授权远程执行）

- **背景**：原 e1rev 产物（ckpt/CE/图/preflight）已从远程与本地全部丢失（编排器误删旧账）。
  用户批准同配方重建：代码 tar 同步（hash 核验一致）→ CE（role_conditioned_v2，6 伙伴×100ep）
  → preflight → 训 aris seeds 0/1/2/4（5000 updates，全部 guard-pass 发布）→ override 门 +
  行为探针。产物：`CPR_REPO/results_override/`、`outputs/asymm_ce_role_v2_e1rev/`、
  `logs_override/`。**非 bit-identical 原模型**（CE 重采样，图 16 因子、覆盖门全满足）。
- **override 门**（`scripts/override_gate.py`，骑 scripted_priority 钩子；两臂同 seed 逐局配对，
  终端链=serve/plate/pick_plate 强制、否则回落 argmax；codex 复评修正 pairing）：
  对 heldout-handoff-alternate-yield（25ep×4seeds）：
  | seed | argmax | override | 读法 |
  |---|---|---|---|
  | s0 | 1.00 | **142.14** | 汤已煮好、守着不端；强制拿盘即通 |
  | s1 | 82.88 | 82.88（全同） | 自发全链，override 无事可做 |
  | s2 | −8.00 | −8.00（全同） | 从不碰锅 → 终端前置从未成立 → override 惰性 |
  | s4 | −1.42 | −1.42（全同） | 同上 |
  pooled lift CI[24.0,48.0] **由 s0 单独驱动**（per-seed 1/4）——pooled 读数作废，按 seed 读。
  对 heldout-resource-server-claim（跑至 s0）：argmax 187.5 > override 150.6 —— 对 claimer
  让位正确、强制接管有代价（角色不对称成立）。完整 JSON 待该臂跑完补档。
- **行为指纹**（`scripts/probe_behavior.py`，3ep/组，探索性诊断）：
  - s1 argmax：fetch27/deliver27/pick9/plate6/serve6，完成率 1.0——完整迁移；
  - s0 argmax：fetch12/deliver9（会做菜）→ **cross_bottleneck 45 次全败 + wait 42 次**，
    守着煮好的锅不拿盘（pick_plate 明明 valid）——**中链价值错误**；
  - s2 argmax：wait_at_bottleneck×75 从开局；s4 argmax：**noop×177**（fetch valid 但不选）；
  - **s2/s4 全链脚本强制：完成率 1.0、每局 3 汤、回报 119.1（两 ckpt 逐位相同）**——
    环境/伙伴完全允许单干通关，**死锁 100% 在学到的 Q 里**。
- **训练侧取证（根因候选，L1）**：4/4 seed 最优 checkpoint 全在 **update 500–1000/5000**；
  全部记录 `last_ineligible_checkpoint_reason=no_ego_sole_correct_delivery`（后期 checkpoint
  反复零独立上菜）；守卫语义="曾出现过会上菜的快照即发布"。结合脚手架 anneal_updates=2500：
  **终端能力是脚手架窗口期的瞬态，TD 未将其巩固**——退火后 contrib_team 备菜局部最优重新
  接管。sec18.12 "脚手架解锁终端能力" 读数需修正为"暂时解锁、随退火蒸发"。含义：一切
  stage-1 读数（0.125、belief 平移翻不动、四粒度行）均测于 10–20% 训练进度的早期快照。
- **排除项**：守卫未失职（4/4 训练伙伴上真实 ego-sole 2/3/6/9）；重采样图结构健全；
  门接线正确（s0 生效、s1 两臂同因 argmax 本在终端链上）；环境无 bug（全链 100% 通关）。
- **原现象复现判定**：s0/s2/s4 三相与 stage-1 描述"wait_at_bottleneck/cross_bottleneck
  loops or noop"逐字吻合 + 原探针矩阵本就记录 seed 斑驳（s2 对两 yield 伙伴 1.00/0.00）——
  **重建基底忠实复现了现象类，包括其斑驳性**。
- **纪律注记**：sec18.12.4 禁止在见到 E1-rev 结果后再调 reward/exploration；任何 L1 修复
  （退火策略/credit/终端探索持续性）须新预注册条目 + Type-B 裁决后方可执行。
- **下一步（单一）**：待用户裁决——1–2 seed 的"脚手架不退火"诊断训练（anneal→∞，其余全同），
  检验"退火是否是能力蒸发的因"：后期 checkpoint 出现持续 ego-sole ⇒ 退火时机是旋钮；
  仍蒸发 ⇒ credit 结构问题更深。一次测量，直接命中 L1 机制。

### 2026-07-06 L1 巩固预注册执行：0/2 + 数据规模发现（用户授权）

- **L1_CONSOLIDATION_PREREG_DRAFT.md 按写定执行**：l1fix config（仅 bias_end 0.9、
  epsilon_end 0.5 两键，按基线起始值持平）× seeds{0,2} × 5000 updates ×
  --save_all_checkpoints；接线回读通过。codex 部署前复评抓到 yaml 重复键 BLOCKER
  （training 块后段真实 ε 键 0.5→0.1 会静默覆盖前段插入值）——修正后基线机制数字
  更正为：有效终端探索率 = ε(0.5→0.1) × bias(0.9→0.35) ≈ 36%→3.5%。
- **判定 0/2**：探索通道全程常驻（pick_count 75/93）仍不巩固——u500 后独立上菜全零，
  与基线逐点一致。**探索支持塌缩假设被证伪为主因。**
- **上游发现（本轮最重要）**：`updates_per_transition: 8` ⇒ 全训练仅 **625 transitions
  = 32 局经验**（episode_returns len=32）；每条经验被梯度复用 8 次。u500 能力≈前 3 局
  上的塑形先验；后续"蒸发"、seed 斑驳、OOD 四相，在 32 局尺度上都是小样本现象。
  **一切既往本基底结论（G2 时代含 5/5 分离实验、stage-1、override 门）都测于 32 局
  训练量的模型之上**——此事实此前从未被任何文档记录。
- **下一步（单一，待 Type-B）**：数据规模探针预注册——l1fix config + 数据量 ×8
  （updates_per_transition: 1，5000 transitions ≈ 250 局，其余不动），2 seeds，
  判据沿用晚期窗口巩固；巩固 ⇒ 根因=样本饥饿；仍不巩固 ⇒ 信用结构预注册顺位执行。

### 2026-07-06 基建提速：环境执行层 4.2×/单局 ~18×，golden 逐字节一致（用户授权）

- **动机**：数据规模探针前先修执行效率。cProfile 实测（1 局 61.4s，4970 万次函数调用）：
  **~2/3 时间 = 25 万次对 JAX 数组的逐元素索引**（state 读取/事件抽取/featurizer/伙伴脚本
  每步 ~4200 次 getitem，每次走完整 JAX 原语分发）；次因 = torch/OpenMP 144 核线程池对
  小张量的空转（user 2m47s vs real 45s）。
- **改动 1（代码，env_adapter.py 单文件）**：step/reset 后 `jax.device_get` 一次性把
  state/rewards/dones/info 物化为 numpy pytree 再暴露——下游全部标量读取变纳秒级；
  jit step 接受 numpy 叶子（同形状不重编译）。附带删掉 featurizer 路径下每步转换后即
  丢弃的 raw-obs 浪费。**下游零改动。**
- **改动 2（零代码，launch 配方）**：`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1`（配 JAX_PLATFORMS=cpu）。**今后所有远程 train/eval 启动
  必须带此三变量**——不带则单进程占满 144 核互踩，并行舰队吞吐塌方。
- **验证**：golden 1 局逐字节 diff 一致；3 局数值与改前老代码逐项一致（82.88/1.0/6）；
  `experiments/overcooked_v2/tests/` 47/47 通过（p0_p1 回归 + 静态不变量；
  test_deadlock_recovery 收集错误为远程同名目录残留 + 路径问题，先于本次改动存在）。
- **数字**：1 局含启动 45.1s→10.8s（4.2×）；启动 ~8.8s、单局本体 ~37-40s→**~2s（≈18×）**；
  user/real 从 3.6min/35s → 11s/10.8s（线程空转清零 ⇒ 144 核可干净并行 ~百个进程）。
- **外推**：数据规模探针（5000 transitions）从 ~3-4h/seed 降至 **~15-25min/seed**；
  stage-1 级评估（25ep×24 ckpt）从 ~6-8h 降至 **~1h 内**。改动未提交 git（待用户指示）。

### 2026-07-06 DATA-8x 双臂判定：样本饥饿实锤为根因；探索常驻假设二次证伪（用户授权）

- 双臂 × 双 seed（预注册 L1 草案 §6，先写后跑；接线回读全过；episodes=251 双臂确认）：
  **A（l1fix-8x 探索常驻）0/2 巩固；B（e1rev-8x 原退火）2/2 巩固**（B-s2 晚期
  ego_sole 1/2/4/4、回报升至 ~40 且 u5000 仍上行）。
- 结论：(1) **32 局样本饥饿是巩固失败的根因**——原退火调度在 250 局下自然巩固，
  此前一切"瞬态/蒸发"现象是数据量问题的表象；(2) 探索常驻（l1fix）不但无效且有害
  （ε 恒 0.5 ⇒ 数据永远高噪、贪心不收敛），**退火设计无罪，l1fix 调度废弃**；
  (3) 250 局未饱和，正式重建取更大预算。
- 影响面：既往全部基底结论的重测路径明确——不需要动方法/奖励/探索任何一处，
  只需要正常的数据量。U2/信念粒度问题的检验条件（一个巩固的基底）现在有了实现路径。
- 下一步（单一，待 Type-B）：SUBSTRATE_REBUILD 预注册——e1rev 原调度 ×
  total_updates 40000（~2000 局）× 5 seeds + stage-1 held-out 重测。

### 2026-07-06 stage-1 held-out 对比：32 局 vs 250 局基底（用户授权；25ep×2 held-out×seed0 协议）

| 基底 | vs yield 伙伴（egoCCR / ego上菜 / 回报） | vs claim 伙伴（ego / 伙伴正确上菜 / 回报） |
|---|---|---|
| 32 局 s0 | 0.000 / 0 / 5.0（死锁） | 0 / **50** / **59.0**（让位配合） |
| 32 局 s2 | 0.000 / 0 / −6.0（死锁） | 0 / **75** / 16.0（让位） |
| 250 局 s0 | **1.000 / 25 / 41.5** | **25 / 0** / 36.1（全包） |
| 250 局 s2 | **1.000 / 25 / 38.7** | **25 / 0** / 35.8（全包） |

- **死锁消失**：对让位伙伴 egoCCR 0→1.0（两 seed 25/25 局全部 ego 上菜）。
- **让位同时消失**：对抢活伙伴从"让对方上 50–75 次、团队 59"变为"ego 全包 25 次、
  对方 0 次、团队 36"——base_only 式全包表型，s0 上比配合少 ~23 分/局面。
- **u500/sel/final 三份 checkpoint 行为逐分相同**（异常解释）：8x 数据下 u500 已含
  25 局经验（≈旧全量），held-out 贪心行为在 u500 即饱和为"永远上菜"，其后不变；
  选择器照常选 u500，无 bug。中期验证的回落-回升是对训练伙伴的行为，与 held-out 无冲突。
- **判读（sec18.13.2 双判据首次真实咬合）**：第 1 条（yield 接管）2/2 PASS；
  第 2 条（claim 不全包）2/2 **FAIL**。**伙伴条件化行为在两个基底上都不存在**：
  32 局=永远让位，250 局=永远上菜。项目核心问题（信念能否让行为随伙伴切换）第一次
  站在一个有能力的基底上，且有量化奖金（对 claimer：条件化值 ~+23 回报 + 伙伴吞吐 2-3/局）。
- 注意：aris_bellman 的信念机制在 250 局训练中在场，未产生条件化——与"信念平移 Q
  不翻转 argmax"的旧诊断连续。s2 训练验证 u4500 对训练 claimer 出现 partner=8 上菜，
  提示条件化可能随更多数据萌芽——SUBSTRATE_REBUILD（2000 局）的读出应加入
  claim-deference 指标以裁决"数据独自能否长出条件化"这一零假设。

### 2026-07-06 SUBSTRATE_REBUILD（2000 局 × 5 seeds）：伙伴条件化由数据涌现；双判据 4/5 达标（用户授权）

- e1rev 原调度 @ 40000 transitions（eps=2001 确认）× seeds 0–4，guard 5/5，~1h。
- **held-out 决定性表（25ep×2 伙伴×seed0，sel=final 行为一致）**：
  | seed | yield 伙伴 egoCCR / ego上菜 | claim 伙伴 ego / 伙伴正确上菜 / 回报 |
  |---|---|---|
  | s0 | **1.000** / 25 | 0 / **50** / 50.0 |
  | s1 | **1.000** / 25 | 0 / **75** / 24.6 |
  | s2 | 0.000 / 0 | 0 / 25 / 26.8 |
  | s3 | **1.000** / 25 | 0 / 25 / 27.3 |
  | s4 | **1.000** / 25 | 0 / 25 / 27.1 |
- **判定**：R1 巩固 5/5；R2 让位涌现分支触发（5/5 对 claimer 让位 + 4/5 对 yielder 接管
  ——32 局"永远让"、250 局"永远抢"、2000 局"看人下菜碟"）；R3 sec18.13.2 双判据
  **4/5 双条通过**（s2 第 1 条不过，seed 方差如实报）。
- **含义**：(1) 角色自适应在现有 aris_bellman（含其因子信念机制）+ 足量数据下涌现，
  无需 U2/换粒度/改奖励——**U2"必要性"口径死亡**；(2) 但条件化是否由信念通道承载
  未裁决——方法核心主张的正面证据窗口打开：**下一步（单一，待 Type-B）= 本基底 E2
  zeroed**（置零证据通道，看 claim-让位/yield-接管是否塌回无条件行为），一次运行裁决
  信念是否承重；(3) 既往全部否定性诊断（0.125、四粒度表、belief 翻不动）确认为
  32 局样本饥饿的伪影，需按新基底重建证据链。范围：dev-heldout、单布局。

### 2026-07-06 E2-zeroed @ 2000 局基底：信念通道对 claim-配合承重 3/5（用户授权）

- 协议：stage-1 同款 + `--zeroed_partner_option_ablation`（LDS-B3），5 rebuild ckpt，
  对照=已在盘 inferred 评估。声明与完整性门全过。
- **yield 维度 5/5 不变**（接管=状态承载）；**claim 维度 4/5 改变**：s0/s2/s3 配合塌为
  互相干扰（伙伴上菜 50/25/25→0、回报→7.9；同 ego 对 yielder 仍 25/25 ⇒ 能力完好，
  塌的是伙伴识别/配合）；s4 反向（zeroed 双方各 25、回报 58.5>27——证据过度让位案例）；
  s1 不变。
- **结论**：角色条件化中"对抢活者的配合"在 3/5 seed 由证据→信念通道承载——
  ARIS-Bellman 核心主张的首份正面证据（范围：dev-heldout 单布局）。与 32 局 E2
  （zeroed==inferred，通道无作用）对照：通道承重性随数据量修复而出现。
- 下一步候选（待用户）：行为指纹 zeroed-claim 塌态（ego 在干什么：堵路/抢盘/误上菜），
  把"通道承重"从数字变成机制陈述；或直接进入证据链正式重建（加局数/CI/盲测 split）。

### 2026-07-06 E1 多臂对照 @ 2000 局基底（5 方法 × 5 seeds，--fast 统一协议；用户授权）

- 20 基线训练全绿（eps=2001 全臂回读确认，guard 20/20 pass，ITT 无剔除）。首轮评估
  15 臂被完整性门拒（无信念方法产不出 Δ_info/MI；stage-1 原协议为全方法 --fast，
  本轮漏带旗标），按原协议统一 --fast 重评 25 臂。aris fast vs full 数字逐项一致
  ⇒ 诊断收集不扰动 rollout（附带接线验证）。
- **两行判据汇总（row1 对 yielder 接管 / row2 对 claimer 不全包 / 双过）**：
  aris 4/5 | **5/5** | **4/5**；base_only 5/5|2/5|2/5；global_gru 5/5|2/5|2/5；
  flat_factor 5/5|1/5|1/5；partner_id_q 5/5|2/5|2/5。
- **三个关键读数**：
  1. row1 在 2000 局下**全方法普遍通过**（数据即能力，不再有判别力——符合预期）；
  2. **真正的判别轴 = 伙伴对比度**：aris 是唯一"对 yielder 全上（25）、对 claimer
     全让（0）"的方法（4/5 seed，Δ=+25）；全部 baseline 对两类伙伴**无差别上菜**
     （claim 侧 ego 25–50，Δ≤0）。baseline 的 row2"通过"（各 1–2/5）全部经由
     **共同上菜**（25/25）而非让位——row2-screen 的"吞吐不降"可被共serve冒充
     （ChatGPT R1-[09] 预警成真），正式轮主指标应改为**伙伴对比度**
     （Δ ego-deliveries across partner types）+ E2 因果。
  3. **诚实披露（不利面）**：对 claimer 的团队回报，共serve/全包 baseline（57–70）
     **高于** aris 的让位（24.6–50）——该脚本 claimer 在共serve下照常出菜，此布局
     上"角色互补"不等于"回报最优"。主张措辞必须是"因子信念产生伙伴条件化的
     互补行为（对照+E2 因果双证据）"，不得写"对 claimer 回报更高"。
- **预注册 §9 分支判定**：落于两分支之间（baseline both=1–2/5，既非 ≤1/5 也非 ≥3/5）；
  aris（4/5）与全部 baseline（≤2/5）分离成立，但 row2-screen 判据需按上条收紧后
  进入正式轮。base_only 2/5 过 row2 = 判据松弛证据（共serve路径），非伪影。
- **下一步（单一，待 Type-B）**：正式轮 = 盲 held-out 伙伴（全新脚本）+ 50–100 局 +
  多评估 seed + 预注册主指标改为伙伴对比度（Δ ego-serve）+ E2-zeroed 因果对照
  （aris 与任一"对比度非零"的 baseline 同跑）。

### 2026-07-06 盲测正式轮终读（sec18.14.2；单看已消耗；待用户 Type-B 签③）

- 60 评估全绿（25 臂 ×2 eval seed + aris 置零 10）。判读按预注册脚本逐字执行。
- **阶梯落 L3（按字面）**：S1 边界真（aris 中位 Δ 恰 0.500，CI>0）；S2 假（比 base_only
  低 1.0，逐 seed 1/5）；E2 不塌（置零后 Δ 反升至 1.0）。
- **超出阶梯的关键发现——对比度指标被"机会挤压"混淆**：无伙伴感知的 base_only 也有
  Δ=+1.5，因为盲 claim 伙伴（尤其 blind-dish-claim 争夺整条终端链）**机械地**压缩了
  ego 的上菜机会；dev-heldout 的 claim 伙伴宽容共 serve，所以 Δ 在 dev 上能分方法、
  在盲上不能。后继指标必须做机会归一化；**blind_v1 已烧掉**（首看已消耗），指标重设计
  须在非盲伙伴上校准后另立新盲集。
- 迁移失败照录：aris 对盲 yield 伙伴接管未迁移（0.6–0.7/局 vs dev 1.0；s2/s4 Δ≈0）；
  C3' 两个干扰敏感伙伴上 aris 团队分均低于最佳 baseline（54 vs 69、38 vs 57），且**全部
  方法远低于脚本让位最优（221–238）**——巨大开放空间，基准判别力良好。
- **结论**：候选主张 C1'/C2' 现形式不成立。盲测协议完成了它的设计使命：dev-heldout 上
  的"条件化"没有作为可泛化能力迁移，且指标混淆只有盲集能暴露。存活资产：L1 数据充分性
  发现（机制无关、坚实）、重建的基底与提速基建、判据校准史、以及一个"无人能解"的
  干扰敏感 claimer 判别场。

## 诊断线 — D1 回应可预测性（第一因裁决；方向决策输入）

### 2026-07-07 D1 终读：标签被瞬时状态饱和、探针不可识别 → 预注册分支 1（修仪器）；数据-vs-目标裁决本轮未达成（用户授权"都执行"+"C也直接执行"）

- commit: `6b69813`（预注册）`2c9fd01`（实现）· prereg: D1_RESPONSE_PREDICTABILITY_PREREG_DRAFT.md（修订记录在文档内，均先于数据）· 远程产物: `results_diag_d1/{chunks,run1}/` · 本地归档: `review_bundles/diag_d1_20260707/`
- 完整性: oracle_source_count=0（42/42 chunks，loader 强制）· evidence_policy=behavior_inferred_v1 · golden 人工核验 PASS（窗口边界/归属/截尾逐事件核对）· 单看纪律（readout 前置 gate-PASS 校验，readout_frozen.json 拒二次写入）· 部署前 codex 三轮评审 BLOCK→BLOCK→APPROVE（thread 019f386c）
- 规模（产物回读）: 6000 局评估-only rollout（EPS_SCALE=2 补救：第 1 轮 indist_val 4385<5000 触发预注册 ×2 路径）→ 240k 决策点；判读划分门内：indist_val **8720**（最小类 1432）、blind_terminal **13558**（最小类 1922）；签 B 达成。接线判据操作化修正照录：kind 级孪生对 server-left/right-claim TV=0.0002（位置型孪生按构造 kind 不可分，显式记录），每伙伴对其余最大 TV ≥ 0.499
- **主读数（K=5，主门，FULL/NOHIST 各 3-seed 集成）**：
  | 划分 | AUC_full | AUC_nohist | 增益 G | G 95%CI |
  |---|---|---|---|---|
  | indist_val | 0.9999 | 0.9999 | +2.5e-06 | [-1.8e-5, +2.1e-5] |
  | blind_terminal | 1.0000 | 1.0000 | +1.3e-05 | [-6.9e-6, +4.4e-5] |
  ID-oracle 分布内上界参照 0.99996——**连真实伙伴身份都无增益**。
- **判据**: R1 FAIL（G≈0 ≪ 0.10；Δlogloss CI 跨 0）· R2 FAIL · K 方向噪声级（|G|≤4.5e-5）· **分支命中 = 预注册 §6 第 1 行**（修可观测性/标签定义；predictive 与 population 两条转向均暂缓）
- **机制解剖（codex 复核口径：探针不可识别，非"公共流无信号"）**: 主门内"接下来谁上菜"几乎被瞬时公共状态（谁持汤/位置/ego 当前选项）完全决定——NOHIST 单靠状态即 AUC≈1.0，标签没有给任何伙伴倾向信息留残差，G 判据被天花板压死。泄漏替代解释被排除：轴外族 AUC 仅 0.85、dev 仅 0.97，饱和非全域伪影。共报佐证（均非主证据）：dev 上 FULL 增益为负（-0.0069，logloss 0.72 vs 0.52）——监督探针复现方法层 E2"置零反升"病理（历史条件化对 held-out 伙伴是负资产；dev 软划分未过下限，仅作警示）；唯一未饱和的轴外族出现 +0.0065/+0.0154 增益萌芽（无 CI，共报）；盲 yield 族 partner_serves 正例=0（纯让位者从不上菜→该标签同样暴露不了 yield 侧倾向）。R3 次级：NMI 身份 0.46 > 族 0.43（指纹倾向；弱仪器警告——历史分支在零梯度压力下训练，不能外推）
- 伪影自检（prereg §8 逐项）: 类率前置（盲 yield 单类照录）✓ 双门同向（饱和处处一致）✓ K 噪声级照录 ✓ 按局分块 CI ✓ 逐 ego 分解（残差集中于 random ego；argmax/fullchain 恰为 1.0）✓ 样本量产物回读 ✓
- **Type-B（签 C；用户预授权直接执行，分支表为绑定判读工具）**: codex 交叉评审 verdict = 判据应用 CONFIRM；分支落点、饱和解剖、D1-rev 方向 CONFIRM-WITH-CAVEATS（caveats 全部采纳入档）· **裁决 = 分支 1：第一因（数据 vs 目标函数）本轮不可判，仪器先行**；D1 基建（数据管线/充分性门/单看流程/歧义剔除）完好可复用
- 下一步（单一，待用户）: **D1-rev 预注册修订**——采样点改为机会起点（汤就绪/将就绪且双方均未持汤、ego 未在终端链内），标签改为"K 窗口内谁首先发起终端链"，且标签必须从原始动作/状态转移导出（**不得**用推断 kind——防标签-推断器循环，codex caveat）；chunk 增记步级事件流后重生成（~25 min）+ 充分性下限重定标

### 2026-07-07 D1-rev 终读：仪器有效（阳性对照 +0.125），发现升级为基底级——终端轴回应是瞬时公共状态的近确定函数且迁移至盲伙伴；字面落分支 1，第一因实质回答 = 基底未实例化"隐模式推断"现象（用户授权"ok执行"，签 C 直通）

- commit: `da32140`（实现 + prereg §9，先于数据冻结）· 远程产物: `results_diag_d1rev/{chunks,run1}/` · 归档: `review_bundles/diag_d1rev_20260707/`
- 完整性: oracle_source_count=0（42/42）· golden 人工核验 PASS（D1 饱和状态被新门正确排除、原始库存位发起编码逐事件核对、持盘不关门符合 §9）· 单看纪律 + gate-PASS 前置 · codex 部署前 BLOCK→APPROVE-WITH-NITS
- 规模（产物回读）: 6000 局 → 判读划分门内：indist_val **16922**（最小类 3129）、blind_terminal **27947**（最小类 4458）；门一轮 PASS 零软失败（dev 11185 亦过下限）
- **主读数（K=5 主门，3-seed 集成）**：
  | 划分 | AUC_full | AUC_nohist | G | G 95%CI |
  |---|---|---|---|---|
  | indist_val | 0.9988 | 0.9986 | +2.5e-4 | [+5.0e-5, +4.9e-4] |
  | blind_terminal | 0.9993 | 0.9979 | +1.3e-3 | [+1.1e-3, +1.6e-3] |
  | 严格门（就绪+空手，分布内与盲） | 1.0000 | 1.0000 | ≈0 | — |
  ID-oracle 分布内 0.9988 与 NOHIST 平齐——**真实伙伴身份在状态之外零增量**（仅分布内参照；欠训替代解释被 tune_report 排除）。
- **判据**: R1 FAIL（2.5e-4 ≪ 0.10；CI 为正但微观）· R2 技术性 TRUE 但**空洞**（门槛=0.5×2.5e-4，如实标注、不作证据）· K 方向一致 · R3 翻正（NMI 族 0.528 > 身份 0.483）· **分支命中 = §6 第 1 行（字面）**
- **跨仪器解剖（codex 签 C 九项全 CONFIRM/CONFIRM-WITH-CAVEATS，caveats 全采纳）**：
  1. **仪器有效**：阳性对照 blind-ingredient-near-neutral（唯一终端策略非脚本化、上菜由任务进度驱动的伙伴；n=9339、正例 2375）历史增益 **+0.125**、logloss 大幅改善——"历史分支坏了/标签测不出历史"两个替代解释被击毙；
  2. **基底级发现（按 codex 校正措辞）**：在公共状态+有效选项+ego 选项条件化之后，本基底终端轴的伙伴回应**没有可辨的残差倾向**，且该状态充分性**迁移至盲伙伴**（0.998–1.0；严格门赛点时刻双侧恰 1.0）；
  3. dev 负增益复现且首次过下限（-0.0296，advisory）——历史条件化对 held-out 伙伴是负资产，与方法层 E2"置零反升"同构，第二次独立复现；
  4. **追溯解释**：TD-aris 学成伙伴指纹（无可迁移信号可学）、状态反应式 baseline 的 ZSC 稳健、partner_id_q 无增益、盲轮全方法不适配——同一基底事实的四个投影。
- 伪影自检: 类率前置（盲 yield 零正例照录）✓ 双门同向（严格门双侧 1.0）✓ K 一致 ✓ 按局分块 CI ✓ 逐 ego 分解（残差集中 random ego）✓ 计数回读 ✓
- **Type-B（签 C，用户预授权直通，分支表绑定）**: codex verdict = **CONFIRM-WITH-CAVEATS** · **裁决：D1 链两轮字面均落分支 1；第一因的实质回答 = 基底在终端轴上未实例化"隐藏倾向推断"现象——predictive 预训练与 population 多样化两条转向在本基底上均不被支持（都只会学到/产出状态可读行为）；承载下一步的决策 = 基底/伙伴重设计（时间延展、位置歧义、潜模式型倾向；阳性对照伙伴族为存在性证明与设计模板）——超出 D1 分支表，属新的 Type-B 用户决策**
- **存活资产**: D1/D1-rev 仪器与全套流程沉淀为**现象存在性证书**——任何新基底/新伙伴族在进入方法实验前，必须先通过"NOHIST 不饱和 + 盲伙伴历史增益 ≥ 阈值"的证书检查（呼应"基底不得消解现象"纪律的正向形式：基底必须先实例化现象）

## Link-A — 基底 v3 潜模式伙伴（THREE_LINKS_IMPLEMENTATION_PLAN §1）

### 2026-07-07 Link-A 证书第 1 轮：FAIL（6/6 条带）——意图电报化 + 规则不可识别（真缺陷）、C-5 聚合口径错误（翻案级）、golden 族-场景失配；串行门控生效，B/C 零投入（用户经远端 codex 执行）

- as-run 快照: 本地 `700cc9f` · 远端 `results_linka_latent_v3_20260707`（deploy diff + 备份齐全）· chunks 54/54 · 远端 pytest 5 passed · D1 充分性门 PASS · oracle_source_count=0
- **判定表**: C-1 0.9664(>0.80) · C-2 +0.0115/CI lo +0.0090(<0.10) · C-3 +0.0158(<0.05) · C-4 history 0.896✓/state-only 0.849✗ · C-5 min lift 0 · C-6 golden 5/6（escalate 触发不可达——fullchain golden ego 从不让位）
- **解剖（三类）**:
  1. 真设计缺陷: 潜模式驱动终端 claim/yield 开关 → 立刻经"终端接近轨迹"位置表达 → 瞬时状态泄漏当前策略（state-only BA 0.849）；切换**规则**不可识别（族 BA 0.327≈随机 0.25、参数 0.086）——每局仅见 1–2 次切换；
  2. **证书统计缺陷（翻案）**: C-5 实现取 min-per-partner vs 该伙伴事后最优静态 ego——对二策略模式**按构造≈0**。VOI 原始数据实际显示跨伙伴最佳回应剧烈冲突（titfortat3: 让 209.6/抢 124.4；escalate2: 抢 127.1/让 5.3），单一盲策略池化口径下 mode_oracle ≈+23% 超最佳单一静态——**价值相关多样性已被造出，是统计量没测对**；
  3. harness 缺口: golden 场景未按族配 ego。
- **正向信号照录**: C-2 较 v2 提升 9 倍且 CI 首次干净为正；C-3 首次为正；C-1 从 0.998 移动到 0.966——方向全对，量级不足。
- 第 2 轮 delta（详 review_bundles/linka_cert_r1_20260707/PROTOCOL_NOTES.md）: ①表达通道移到"共同前置、最后一刻分化"的离散行为（handoff 接受/拒绝、bottleneck 让行/抢行、末步转向）；②放弃规则识别野心（族/参数降为永久共报）；③golden 按族配 ego；④**C-5 聚合修正为"对最优单一盲策略（含 reactive 臂）的池化提升"——属 [F@A3] 条带定义变更，待用户签**；⑤C-1/C-3 条带不预调。

### 2026-07-09 Link-A 证书第 2 轮：FAIL（6/6）——干净的基底级负结果；去电报化后倾向仍状态可读，反应式够用、预判无价值（结构性，非参数性）；codex 签 C CONFIRM-WITH-CAVEATS（用户"继续推进"授权，签 C 直通）

- 代码: `5f9ce3a`（去电报化选项选择恒抢活形 + 让位仅末刻中止 + reactive 池化 VOI + prepchain ego）+ `025b1ee`（GPU 续跑脚本）· 远端 `results_linka_v3r2/` · 证书 `artifacts/latent_v3_r2_certificate.{json,md}` · 归档 `review_bundles/linka_cert_r2_20260709/`
- 执行事故（不影响科学）: SSH 隔夜两断（编排器被 init 收养仍跑，chunks 72/72 + 门 PASS 无损）；**CPU tune 11h 才 8/21**（数据 8 万点、大 4-5×）→ 切 GPU4（0 ECC、setsid 脱离控制连接）**21 模型 ~18min**；3-seed 集成吸收设备浮点差异
- 完整性: oracle_source_count=0（72/72）· 门一轮 PASS（indist_val 79547 / blind_terminal 99294）· 机制 golden 核验（非控制器 bug）
- **证书判定表**:
  | 带 | 值 | 门 | r1 |
  |---|---|---|---|
  | C-1 不饱和 | NOHIST 盲 AUC **0.9475** | ≤0.80 | 0.966 |
  | C-2 历史信号 | +0.0199（CI lo +0.018） | ≥0.10 | 0.0115 |
  | C-3 身份超状态 | +0.0141 | ≥0.05 | 0.0158 |
  | C-4 可识别性 | history 0.905✓ / **state-only 0.857✗** | ≥0.75/≤0.60 | 0.849 |
  | C-5 信息价值 | 池化 oracle 175.5 vs **reactive 169.8 = +3.3%** | ≥0.15 | (口径错) |
  | C-6 接线 | oracle=0✓ / golden 5/6 | 全过 | 5/6 |
  C-5 各臂: oracle 175.5 / full 160.2 / prep 72.4 / **reactive 169.8**；最强盲 = reactive
- **机制 golden 核验**: static-yield×抢活ego partner_first=0/60（yield_abort 触发、ego 接手）；static-claim×让位ego partner_first=20（抢活透传）——机制正常。**escalate_after_defer 真 bug**：双方都让→汤摆着没人拿→机会门永不关→升级计数（只在关闭时累加）永不触发。**此 bug 恰点破核心：机会持久不关 = 反应式够用的根因。**
- **核心裁决（基底级、由正常工作族驱动）**: 即便选项**选择**去电报化、让位只以末刻中止表达，潜倾向**仍可从瞬时公共状态读出**（state-only 0.857，几乎未动）——因让位**表达**（接近-绕开）本身即位置签名。故反应式"等着看伙伴转不转"的 ego 捕获 oracle 约 97% 价值（+3.3%），**预判信念在本基底无价值**。
- **结构性非参数性**: 非调 ε/dwell/wait 可修。预判仅在"反应太迟或太贵"时胜出；当前终端交接机会持久、观察无代价 → 反应恒够用。**追溯统一整条弧**（D1 饱和、盲轮全不适配、latent_v3 两轮）：基底从未对等待-反应施加代价。
- **Type-B（签 C，用户预授权直通）· codex CONFIRM-WITH-CAVEATS**: (A) 去掉 reactive 臂 oracle vs full 仍仅 +9.6%<15%，fail 稳健；(B) C-4 state-only 参数性（换非位置表达可修）但修不了基底——藏到末刻+机会不关只帮反应臂，C-5 失败结构性；(C) escalate 修了救不回（需该族 oracle 172→224+，超量纲）；(E) "一修就过"DISPUTE（五带独立差太远）。措辞收窄为"本终端交接基底缺等待的不可逆代价"，非全 Overcooked 定理
- **方向建议（Type-B，用户裁量；超出分支表）**: 第 3 轮加**预判强制结构=等待的真实代价**——(a) 易腐汤/机会 N 步关闭；(b) 承诺代价（备餐须先于揭示、被抢则废）；(c) **同时多锅分工+可达/时间约束**使反应式无法两头兼顾（最干净，让反应不足可证）。北极星判据：reactive 臂须证明性够不着 oracle
- 存活资产: D1/Link-A 仪器（含 reactive 池化 C-5）经两轮验证 = 现象存在性证书；去电报化控制器 + GPU 基建可复用；escalate bug 随第 3 轮基底重设计一并修
