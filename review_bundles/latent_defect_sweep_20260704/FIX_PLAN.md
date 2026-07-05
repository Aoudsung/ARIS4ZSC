# FIX_PLAN — 窗口一修复方案（eval 前 must-fix，5 项）

**Date:** 2026-07-04 · **前提**：E1-rev wave 已全部完成（25/25 exit=0，`[E1REV ALL DONE]`），
21+4 个 metrics.json 均 `final=True/run_status=ok/updates=5000` 已核验。
**依据**：SWEEP_SUMMARY.md 裁决 + PASS_{A,B,C}_OUTPUT.txt 原始发现。
**纪律**：一机制一 commit + 每 commit 带回归测试；改前 codex diff 复评；train 行为 bit-不变
（保存性），仅动 eval/门/汇总层；方法层不触（sec18.11 冻结令）。
**执行顺序**：F1→F5 实现 → 本地静态检查（py_compile + fidelity gate）→ codex diff 复评 →
逐 commit 落库 → 远程 pytest + gate 验证 → 用 F5 脚本对真实 25 run 出 §14.1 汇总（验收即产出）。

---

## F1 — LDS-B2 [A]：eval reward-scale 门从"只报"改"硬拦"

**现状**（evaluate_aris.py:127-135, 236-239）：`reward_scale_status` 逐 variant 算出后仅写进
输出 JSON（`reward_scale_verified: all(...)`），eval 正常返回 → 不合格产物照样落盘。
**改法**：
- `evaluate()` 内、`reward_scale_status` 计算后立即检查：任一 variant
  `reward_scale_verified=false` → `raise RuntimeError`（消息含逐 variant mismatch 详情）。
- 新 CLI 逃生口 `--allow_unverified_reward_scale`（默认 False，help 注明 smoke-only），
  置位时降级为 stderr 警告 + JSON 照记（保留诊断用途）。
**保存性**：正式路径上该状态本应为 true（E1/E1-rev 图均带正确元数据）；只有本就不该入表的
跑法会被拦。train 不动。
**测试**：`tests/test_eval_reward_scale_gate.py` — 构造 status=false → evaluate 主路径 raise；
带 flag → 不 raise 且 JSON 记 false；status=true → 不 raise。（用最小 stub，不跑 env。）
**commit**：`fix(eval-B2): hard-fail reward-scale verification in formal eval`

## F2 — LDS-B1 [A]：eval 回报口径排除 terminal_progress_shaping

**现状**：`_training_reward`（train_aris.py:3021）= actor-sparse + **terminal_bonus** +
shaped_coef×shaped；eval（evaluate_aris.py:602 及同文件其他调用点，实现时 grep 全找）用它做
回报记账。E1 与 E1-rev 的 reward 配置差异**恰好只有** terminal_progress_shaping
（两者 shaped_reward_coef=1.0、sparse_credit=contrib_team 相同）→ E1-rev 的 eval 回报
与 E1 不同尺度。
**改法**：
- `_training_reward` 加 kwarg `include_terminal_shaping: bool = True`；False 时 terminal_bonus
  记 0。train 调用点不动（默认 True → train bit-不变）。
- evaluate_aris **全部**回报记账调用点传 `include_terminal_shaping=False`（含
  random/external 参照基线共用的 episode 路径——同一口径才可比）。
- 输出 header 增记 `eval_return_excludes: ["terminal_progress_shaping"]`（口径自描述）。
**保存性/golden**：E1 系 config（shaping disabled）terminal_bonus 恒 0 → eval 数值逐位不变；
E1-rev eval 回到 E1 尺度（这正是修复目的）。headline `ego_correct_completion_rate` 与
回报无关，两侧均不受影响。
**测试**：`tests/test_eval_return_excludes_terminal_shaping.py` — 同一合成 event 序列：
shaping-on config 下 train 口径含 bonus、eval 口径不含；shaping-off 两口径逐位相等。
**commit**：`fix(eval-B1): eval return accounting excludes terminal_progress_shaping`

## F3 — LDS-C2 [B→窗口一]：基线缓存 key 纳入 sparse-credit + terminal-shaping 签名

**现状**（evaluate_aris.py:1116-1129 + provenance.py:62-69）：`reward_config_payload` 只有
layout/cost_coef/cost_per_step/shaped_reward_coef —— 漏 `sparse_credit_params` 与
`terminal_progress_params`，跨 reward 变体可能串缓存。
**改法**：`_baseline_env_payload` 增两键：
`"sparse_credit": sparse_credit_params(training_cfg)`、
`"terminal_progress": terminal_progress_params(training_cfg)`（复用 train 同源 helper，
不自造签名）；`_BASELINE_CACHE_SCHEMA` 2→3（旧条目整体失效，宁可重算不可串值）。
F2 之后 terminal shaping 理论上不再影响 eval 回报，但仍入 key（过度 key 只损失命中率、
永不损失正确性）。
**测试**：扩展 `tests/test_e2_e3_cache_switches.py` 缓存段——翻转 sparse_credit 模式 /
terminal_progress.enabled → key 变；schema bump → 旧 schema 条目不命中。
**commit**：`fix(eval-C2): baseline cache key includes sparse-credit + terminal-progress signatures (schema v3)`

## F4 — LDS-C3 [B→窗口一]：canonical throughput 字段（sec18.9.2 口径）

**现状**：aggregate（evaluate_aris.py:1434-1470）只有计数原料，无 throughput 字段；
sec18.6 把 throughput 列为 headline 之一，分母若留给汇总脚本则口径可漂。
**改法**（口径逐字取自 sec18.9.2，不新造语义）：aggregate 增三字段——
- `team_correct_delivery_throughput_per_episode` = correct_delivery_count / episodes
- `ego_correct_delivery_throughput_per_episode` = ego_correct_delivery_count / episodes
- `ego_serve_share` = ego_correct / (ego_correct + partner_correct)，分母 0 时记 None
  （不记 0.0——C4 教训，缺测≠零）
summary 增对应均值，**直接索引**新字段（fail-closed，不用 .get 默认值——不复制 LDS-C4 模式）。
**测试**：`tests/test_eval_throughput_fields.py` — 合成 episodes：数值正确性 + 分母=episodes
+ serve_share 分母 0 → None。
**commit**：`feat(eval-C3): canonical throughput fields per sec18.9.2 lens`

## F5 — PM-7：E1-rev 专用汇总脚本（含 LDS-C1 消费端防御）

**现状**：`parse_role_v1_v4.py` 只均值 eval option-kind stats，不读 train metrics /
guard / final —— 25-run 表无现成合规工具。
**新文件**：`experiments/overcooked_v2/scripts/aggregate_e1rev.py`
- **train 模式**（--results_dir + --tag e1rev）：逐 run 读 metrics.json，
  **硬校验** `final==True && run_status=="ok" && updates_done==--expected_updates(默认5000)`，
  不合格即列名报错退出（= LDS-C1 消费端防御：非终态产物永远进不了表）。
  产出每臂：seeds 数、guard 通过数、**terminal competence rate**（deployable_checkpoint
  非空 / 全部 seeds，sec18.12.3 co-primary——guard-fail 计入分母）、性能均值
  （selCCR/ego_sole/best_greedy_return，**仅 guard-pass runs**）。
- **eval 模式**（--eval_glob）：读 eval JSON，**直接索引** headline 字段（缺字段 = 硬错，
  禁 missing→0）；每臂 held-out `ego_correct_completion_rate` + F4 throughput 字段的
  mean + seed 级 bootstrap 95% CI（10k 重采样，§9.2 口径）。
- 输出 json + markdown 双格式；元数据记录公式字符串与排除规则（口径自描述）。
**测试**：`tests/test_aggregate_e1rev.py` — 合成 fixtures：非终态 run → 硬错；guard-fail
计入 competence 分母且排除出性能均值；eval 缺 headline 字段 → 硬错。
**验收即产出**：远程对真实 25 run 跑 train 模式 → 输出即 §14.1 归档汇总表。
**commit**：`feat(scripts-PM7): e1rev aggregation with hard final-status validation`

---

## 验证链（全部完成才算"妥善修订"）

1. 本地静态：`python -m py_compile` 全改动文件；`.aris/tools/aris_bellman_fidelity_gate.py
   --root .` I1-I17 仍全 PASS。
2. codex diff 复评（MCP read-only xhigh）：重点审 F2 保存性（train 逐位不变）、F1 逃生口
   不弱化正式路径、F3 key 无遗漏、F5 规则与 sec18.12.3 一致。BLOCK 项修完再 commit。
3. 逐 commit 落库（5 commits，引 LDS-ID + 台账）。
4. 远程验证：同步 → pytest（新增 4 个测试文件 + 既有回归）+ fidelity gate → 绿。
5. F5 对真实 25 run 出表（train 模式）→ 结果落 EXPERIMENT_LOG + review_bundles 归档。

## 窗口二遗留（不在本方案内，各实验前修）
LDS-A1/B3（E2 前）、LDS-B4/B5（CE 重建/复用前）、LDS-B6（E3 前）、LDS-B7/C4（门加固）、
LDS-C1 生产端 success-marker（写进下一个编排器模板）。
