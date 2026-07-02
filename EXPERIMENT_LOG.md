# EXPERIMENT_LOG.md — 正式实验结果记录（ARIS 惯例）

**Created:** 2026-07-02（Phase 0 · G0.5）· **Status:** 骨架，无正式结果
**规则**：只记录**执行门控下的正式运行**（preflight 接受 + 完整性硬门全 true + git commit 哈希 +
远程产物路径）。诊断/冒烟跑不入此文件。每条结果的解读**必须引用** METHOD_LOCK sec18 预注册分支表，
禁止事后重解释。NEW-4 隔离材料（CODEX_IMPL_SPEC v1–v4 数字）永不入此文件。

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

*(待运行)*

## Phase 2 — 基底证书（R2.1 伙伴可区分性 · R2.2 asymm CE 支持度）

*(待运行)*

## Phase 3 — 决定性实验（E1 四臂去 oracle 重跑 · E2 通道消融 · E3 持久化消融）

*(待运行)*

## Phase 4 — 主张级实验（E4–E7）

*(待运行)*

## Phase 5 — 锁定 → 盲测 → 终表

*(待运行)*
