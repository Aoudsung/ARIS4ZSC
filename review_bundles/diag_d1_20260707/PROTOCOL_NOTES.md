# D1 执行过程记录(diag_d1_20260707)

**预注册**:`D1_RESPONSE_PREDICTABILITY_PREREG_DRAFT.md`(签 A 记录:用户 2026-07-07
"都执行";签 C 执行方式:用户同日"C也直接执行"——预注册 §6 分支表为绑定判读工具,
codex 交叉评审照走,人类终审权保留但不阻塞)。

## 提交链(可归因)

- `f278263` docs(sec18.14): 盲测终读 + 用户签③裁决(证据链落库)
- `6b69813` prereg(D1): 预注册草案
- `2c9fd01` feat(d1): 生成器 + 训练器 + 扇出驱动(含预注册修订:宽松门操作化、
  严格归属、逐 K 独立模型组)

## codex 部署前评审(OPERATING_CONSTRAINTS §3,thread 019f386c)

- 第 1 轮:BLOCK——2 BLOCKER(同步双送归属歧义;读出未验门)+ 4 MAJOR + 1 MINOR + 1 NIT
- 第 2 轮:BLOCK——2 MAJOR(归一化统计未过歧义滤;ls|wc 在 pipefail 下可提前中止)
- 第 3 轮:**APPROVE**(静态评审,未运行)

## 部署与核验

- 远程:`CPR_REPO/experiments/overcooked_v2/scripts/`(scp;远程 overcooked_v2 目录
  untracked,source of truth = 本地 git)。py_compile + bash -n 全过。
- 提速补丁前置依赖:确认已在本地 git(`36f9c9e`,env_adapter numpy 物化),
  本地/远程全量 .py md5 一致 —— 预注册 §7 前置依赖已解除。
- 锚 checkpoint(argmax ego + env/option_lib/graph 构造):
  `results_e1arms/base_only_s0/asymm_advantages/base_only/full_support/seed0/checkpoint.pt`
  (sha256 记录于各 chunk meta)。

## golden 人工核验(签 B 硬门之一,PASS)

fullchain ego × server-left-claim × 2 局(seed 999):
- 窗口边界:dp011 的 ego 送餐(step 32)使 dp007–dp011 label=1,dp006 窗口
  [20, 31) 恰不含 step 32 → label=0,无差一错误;
- 归属:dp021/dp025 的 partner 送餐(step 74/89)→ dp017–dp025 label=2,
  首击语义正确(dp021 窗口首击为自身步 74 的 partner 送餐);
- 截尾:dp035–dp039(剩余决策 <K)censored=1;
- 门:gate_main 与锅 ready/持汤时序一致;gate_relaxed 覆盖烹饪期;
- 接线:oracle_source_count=0;ambiguous_delivery_steps=0;hist_len 0→32 正常饱和。

## 充分性门第 1 轮(EPS_SCALE=1,42/42 chunks,判 FAIL)

| 划分 | 门内决策点 | 最小类 | 判 |
|---|---|---|---|
| indist_val(硬) | 4385 | 721 | **行数 FAIL**(<5000) |
| blind_terminal(硬) | 6777 | 949 | PASS |
| dev(软/共报) | 2232 | 205 | 软失败(判据校准器,不阻塞) |
| indist_train(软) | 17467 | 2775 | PASS |
| blind_offaxis(软) | 4156 | 1152 | 软失败记录 |

**补救(预注册 §5 既定路径)**:EPS_SCALE=2 全量重生成(同基种子,前 100 局逐位
不变的确定性超集;train 200 局/伙伴/ego,dev、blind 100)。预期 indist_val ≈ 8.8k。

## 充分性门第 2/3 轮(EPS_SCALE=2,6000 局,判 PASS)

第 2 轮:行数/类数全过(indist_val 8720/最小类 1432;blind_terminal 13558/1922;
dev 4514 仍软失败照录)——但接线检查 FAIL:min 成对 TV=0.00022。
定位:`server-left-claim ↔ server-right-claim` 孪生对,仅差配送位置偏好,
**kind 级事件流按构造不可区分**;其余全部伙伴对 TV ≥ 0.499(mean 0.445)。
判据操作化修正(Type-A 检查层,非判读判据;先于任何迁移读数):
"min 成对 TV > 0.01" → "每伙伴对其余最大 TV > 0.05 + 塌缩孪生对显式记录"。
第 3 轮:**PASS**。伴随事实照录:ambig 剔除率 k5=0.6%、截尾率 10.2%、
kind 级历史无法分辨位置型孪生伙伴(解读 G 时须记住此表征粒度上限)。

**签 B(Type-A,自判可过,OPERATING_CONSTRAINTS §4)**:达成于 2026-07-07。

## 单看纪律

- 调参只看分布内验证;dev/blind 迁移读数由 readout 阶段冻结脚本单次计算;
  readout 前置校验 gate PASS;readout_frozen.json 拒绝二次写入。

## 终读与签 C(2026-07-07;单看已消耗)

判据:R1 FAIL(G_indist=+2.5e-06,CI 跨 0)· R2 FAIL · K 方向噪声级 → 分支 1。
解剖:探针不可识别(标签被瞬时状态饱和;NOHIST 状态基线 AUC≈1.0,ID-oracle 亦无增益);
泄漏替代解释被轴外 0.85 / dev 0.97 排除。共报:dev FULL 负增益(E2 置零反升病理的
监督层回声,警示级)、轴外未饱和处 +0.0065/+0.0154 增益萌芽、盲 yield 正例=0。
codex 签 C 交叉评审(同 thread):判据应用 CONFIRM;分支落点/解剖/D1-rev 方向
CONFIRM-WITH-CAVEATS——caveats:记录口径用"探针不可识别";dev 负增益不作主证据;
轴外萌芽仅提示;D1-rev 标签禁用推断 kind(防标签-推断器循环)、防机会起点几何捷径、
预期行数下降需重定下限。
裁决:**第一因(数据 vs 目标函数)本轮不可判,仪器先行(D1-rev)**;两条转向路线
(predictive / population)均维持暂缓。完整数字见 readout_frozen.json(本目录)。
