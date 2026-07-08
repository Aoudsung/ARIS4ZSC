# Link-A 证书第 1 轮记录(linka_cert_r1_20260707)

**计划**:THREE_LINKS_IMPLEMENTATION_PLAN.md §1(签 A1 后由用户经远端 codex 执行)。
**as-run 快照**:本地 commit `700cc9f`;远端 run dir `results_linka_latent_v3_20260707`,
deploy diff `logs_linka/linka_deploy_diff.patch`,备份 `.codex_remote_backup/linka_20260707_192551`。
chunks 54/54;远端 pytest 5 passed;D1 充分性门 PASS;oracle_source_count=0。

## 证书判定:FAIL(6/6 条带)

| 带 | 值 | 门 | 判 |
|---|---|---|---|
| C-1 不饱和 | NOHIST 盲 AUC 0.9664 | ≤0.80 | FAIL(v2 时代 0.998 → 有移动,远未达标) |
| C-2 历史信号 | +0.0115,CI lo +0.0090 | ≥0.10 | FAIL(v2 的 9 倍、CI 首次干净为正,量级差 10 倍) |
| C-3 身份超状态 | +0.0158 | ≥0.05 | FAIL(首次为正,不足) |
| C-4 可识别性 | history 0.896 ✓ / state-only 0.849 ✗ | ≥0.75 / ≤0.60 | FAIL(半过) |
| C-5 信息价值 | min relative lift = 0 | ≥0.15 | FAIL(见"聚合口径"节) |
| C-6 接线/golden | 5/6 族 | 全过 | FAIL(escalate 触发不可达) |

## 三类失败解剖

**类 1|基底设计缺陷(真)**
- 意图电报化:潜模式驱动的是终端 claim/yield 开关,开关立刻通过"终端接近轨迹"表达
  → 瞬时位置泄漏当前策略(state-only BA 0.849)。表达通道选错了。
- 切换规则不可识别:族分类 BA 0.327(4 类随机=0.25)、参数 BA 0.086——每局最多见
  1–2 次切换,规则(为什么切)在行为流里几乎不可见。规则识别不是现象存在的必要条件,
  判读目标(当前策略推断)不受此项否定,但 Link C 的 switch-tracking 只能测反应不能测预判。

**类 2|证书统计缺陷(翻案级)**
- C-5 的 0 是聚合口径产物:实现取 min-per-partner(mode_oracle vs 该伙伴的事后最优
  静态 ego)——对二策略模式按构造≈0(每个模式总有一个静态最佳回应)。而 VOI 数据
  本身显示**跨伙伴最佳回应剧烈冲突**:
  | 伙伴 | mode_oracle | 纯抢(fullchain) | 纯让(prepchain) |
  |---|---|---|---|
  | titfortat3 | 209.6 | 124.4 | 209.6 |
  | escalate2 | 127.1 | 127.1 | **5.3** |
  | patience5 | 118.7 | 118.7 | (低) |
  单一盲策略池化口径:mode_oracle ≈ 455 vs 最佳单一静态(纯抢)≈ 370 → **≈+23%**。
  价值相关的多样性已经被造出来了,是探针统计量没测对。正确口径 = mode_oracle 对
  "证书集上最优的单一盲策略(含 reactive wait-and-see 臂)"的池化提升。
- 修正属 [F@A3] 条带定义变更 → 需用户签字后方可作为第 2 轮判据。

**类 3|harness 缺口**
- escalate_after_defer 的 golden 场景用 fullchain ego(从不让位)→ "连续让位 k 次后
  升级"触发按构造不可达。族-场景匹配问题,非伙伴 bug:golden 需按族配 ego
  (escalate 配 prepchain、titfortat 配混合)。

## 第 2 轮 delta(伙伴侧属预授权迭代;C-5 口径属条带修订待签)

1. 表达通道改造(打 C-1/C-4-state):倾向表达从终端接近轨迹移到**共同前置、最后一刻
   分化**的离散行为——候选:handoff 接受/拒绝(柜台站位同,交接瞬间分化)、bottleneck
   让行/抢行(相遇瞬间分化)、就绪汤"最后一步转向"(接近路径共享,末 1–2 格分化)。
2. 保留已证明产生最佳回应冲突的族结构;**放弃规则识别野心**——C-4 判读维持"当前
   策略",族/参数识别永久降为共报。
3. golden 按族配 ego 场景(C-6)。
4. C-5 聚合修正(如上,待签)+ 盲基线臂加入 reactive ego;band 维持 ≥15%。
5. C-1/C-3 条带不预调,表达通道改造后按实测再议。

**门控纪律执行情况**:串行门控生效——Link B/C 零投入,失败发生在最便宜的一环。
