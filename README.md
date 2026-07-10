# ARIS4ZSC — 仓库地图

一句话：用 ARIS 这套自动科研工具，开发 ARIS-Bellman 方法（面向零样本协调的因子局部
Bellman 控制）。状态和规则看下面的入口文件；这份 README 只回答"什么在哪"。

## 先读这几个（根目录入口）
- **CLAUDE.md** / **AGENTS.md** — 给 Claude Code / Codex 的项目指令
- **OPERATING_CONSTRAINTS.md** — 能执行什么；门的生命周期（§7 退休阀门）
- **PROJECT_DASHBOARD.md** — 现在在哪一步、五个决定性主张的状态
- **CUSTOMER.md** — 远程执行合同（zsc-customer）
- **FIDELITY_GATE.md** / **.json** — 方法保真门读数。工具 `.aris/tools/` 直接读写这两个
  文件，所以留在根目录

## 活文档 `docs/`
- **docs/status/** — 常查的活台账
  - `METHOD_LOCK.md` — 方法冻结 + 预注册决策表
  - `EXPERIMENT_LOG.md` — 正式实验结果记录（追加写；最新是 Link-A）
  - `FINDINGS_LEDGER.md` — 根因裁定台账（asymm 线的支柱 P/S；仍是当前保留门的出处）
  - `GOVERNANCE_CUTLIST.md` — 2026-07-08 治理精简的逐条处置记录
- **docs/plans/** — 当前活跃计划
  - `THREE_LINKS_IMPLEMENTATION_PLAN.md` — 当前主线（Link-A→B→C 涌现测试）

## 归档 `archive/`
- **archive/2026-07_asymm-role-v2-line/** — 已关闭的 asymm×role_v2 方法线整套（盲测判负、
  被 three-links 取代）：实验链计划、ICLR 升级计划、设计总方案、根因评审计划、D1/正式轮/L1
  预注册、U1/U2 规格与提案。历史留存，不再是活门。见该目录下的 README。
- **archive/docs/**、**archive/raw/** — 更早的归档

## 代码与产物
- **src/** — ARIS-Bellman 方法实现
- **experiments/overcooked_v2/** — OvercookedV2 适配 + 配置
- **tests/** — 测试
- **artifacts/** — 迁移计划等（`OvercookedV2_plan.md`）
- **idea-stage/refine-logs/** — 提案、实验计划、追踪表
- **results/**、**review_bundles/**、**provenance/** — 结果、评审包、溯源
- **.aris/** — 项目姿态配置 + 保真门工具

_生成于 2026-07-08 根目录整理：根目录文件从 22 个降到 8 个。_
