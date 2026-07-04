# SWEEP_SUMMARY — codex 潜伏缺陷扫荡汇总裁决

**Date:** 2026-07-04 · **Reviewer:** codex (gpt-5.x, xhigh) via MCP，三 pass 独立
**Threads:** A=`019f2cd4` · B=`019f2cda` · C=`019f2ced`
**共 12 条发现**（3 条 [A] / 7 条 [B] / 2 条 [C]）。原始报告见 PASS_{A,B,C}_OUTPUT.txt。

## 顶层结论（先说最重要的）

**已完成的 E1-rev 训练 wave 结果不受任何一条发现影响，无需重跑。** 三条 [A] 逐一核验：
- **LDS-C1**（编排器 skip-on-existence）：直接审计 21 个已完成 run → 全部 `final=True /
  run_status=ok / updates=5000`。wave 是全新跑（无 resume），bug 未触发。仅在未来
  resume/超时场景才咬人。
- **LDS-B1**（shaping 漏入 eval return）：影响的是 `mean_return` / reference-gap 等**回报类**
  读数，**不影响决定性 headline `ego_correct_completion_rate`**（它按 episode 完成事件算，
  不含 shaping 项）。
- **LDS-B2**（reward_scale 门在 eval 只报不拦）：门的严格性问题，不是既有产物污染。

即：全部 12 条都指向**即将执行的 held-out eval + 后续 E2/E3/R2.2/U1/U2**，没有一条作废已跑数据。

## 分级 + 修复窗口映射（按 brief §6 纪律）

| ID | 级 | 类 | 一句话 | 窗口 |
|---|---|---|---|---|
| **LDS-B1** | A | C6 | terminal shaping 经 `_training_reward` 漏入 eval 回报口径 | **窗口一（eval 前必修）**——但仅影响回报类次级指标；headline 安全 |
| **LDS-B2** | A | C11 | `reward_scale_verified` 在 eval 只记录不 hard-fail | **窗口一**——eval 前把门改为强制 |
| **LDS-C1** | A | C9 | 编排器把 metrics.json 存在=完成（非终态也跳过） | **窗口一（but 仅 resume 前）**——当前 wave 已核验干净；resume/重跑前必修 |
| LDS-A1 | B | C7 | E2 zeroed 门只查 id/dist 为 None，漏 confidence + 半置零 | 窗口二（E2 前） |
| LDS-B3 | B | C6 | E2 无 CLI 消融开关，zeroed 需篡改 ckpt config 才能跑 | 窗口二（E2 前） |
| LDS-B4 | B | C3 | `graph.sparse_ce_support` config 被 CE 脚本忽略（认 CLI flag） | 窗口二（R2.2/U1 重建图前） |
| LDS-B5 | B | C11 | `--reuse_replay` 只查 partner/credit，漏 support_mix/reward/shaping 等 | 窗口二（CE 复用前） |
| LDS-B6 | B | C2 | `belief_persistence` 全配置未显式钉、依赖代码默认 True | 窗口二（E3 前，钉进 config） |
| LDS-C2 | B | C9 | 基线缓存 key 漏 sparse-credit/terminal-shaping → 跨条件串值 | 窗口二（若 eval 用共享缓存则升窗口一，见下） |
| LDS-C3 | B | C5 | eval 无 canonical `throughput` 字段，分母留给汇总脚本 | 窗口一（eval 判读前冻结分母）——sec18.6 headline 之一 |
| LDS-B7 | C | C11 | eval 完整性门可被零证据 vacuous 通过 | 窗口二 |
| LDS-C4 | C | C4 | summary 均值把缺失字段静默当 0.0 | 窗口二 |

## Eval 前 must-fix 清单（窗口一，held-out eval 启动前）

1. **LDS-B1** — eval 回报口径与训练 shaping 分离（eval reward path 强制
   `terminal_progress_shaping.enabled=false`，或走显式 sparse/env-completion 回报）。
   *否则回报类次级指标与 E1 不同尺度，不可比。headline 不受影响但次级读数会失真。*
2. **LDS-B2** — eval 把 `reward_scale_verified=false` 改为 hard-fail（复用 train 侧强制逻辑）。
3. **LDS-C3** — 冻结 throughput 分母（加 canonical 字段或在汇总脚本里定一种公式并记进元数据）。
   sec18.6 把 throughput 列为 headline，判读前必须口径唯一。
4. **LDS-C2** — **若 held-out eval 启用共享 `--baseline_cache_dir`**（§14.2 执行卡默认启用），
   则升为窗口一：修 cache key（纳入 sparse-credit + terminal-shaping 签名）或按 reward/scaffold
   变体分目录。否则 E1-rev(scaffold) 与 E1/base 的参照基线可能串缓存。
5. **PM-7 汇总脚本**（新写）：`parse_role_v1_v4.py` 不读 train metrics/guard/final；需新脚本，
   规则 = guard-fail 排除出性能均值但计入 competence-rate 分母（sec18.12.3）、禁 missing→0、
   throughput 单一公式。

## Resume/重跑前 must-fix

- **LDS-C1** — 编排器改用独立 success marker（仅在 `final=true && run_status=ok &&
  updates 达标 && guard/ckpt 字段齐` 后写）；非零退出不留可跳过标记。当前 wave 无需动，
  但任何 resume 前必修。

## 回流动作（Claude 执行）

- [x] 三 pass 输出落盘 + 本汇总。
- [ ] 逐条进 FINDINGS_LEDGER（分配正式 ID，标 CONFIRMED/severity/窗口）。
- [ ] 窗口一 5 项：wave 判读前，走 codex diff 复评 + 一机制一 commit + 回归测试（brief §6）。
- [ ] 窗口二 7 项：各自在对应实验（E2/E3/R2.2/U1）启动前修。
- 注：全部为 eval/编排/门层，**不触方法层**（sec18.11 冻结令未破）。
