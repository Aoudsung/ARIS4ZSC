# Link-A 证书第 2 轮记录(linka_cert_r2_20260709)

**计划**:THREE_LINKS_IMPLEMENTATION_PLAN §1;第 2 轮 delta 见第 1 轮记录
review_bundles/linka_cert_r1_20260707/PROTOCOL_NOTES.md。
**代码**:commit `5f9ce3a`(去电报化 + reactive 池化 VOI + prepchain ego)+ GPU 续跑脚本。
**远端**:run dir `results_linka_v3r2/`,证书 `artifacts/latent_v3_r2_certificate.{json,md}`。
**授权**:用户"ok执行"+"继续推进"(第 2 轮全链,签 C 直通)。

## 执行事故与恢复(不影响科学)

- 阶段 1 首启(round2.sh CPU 全链):SSH 主连接两次隔夜断连,原后台任务报 exit 255;
  但编排器被 init 收养仍在跑,chunks 72/72 + 门 PASS 无损失。
- 诊断真问题:**CPU tune 跑 11 小时才 8/21 模型**(数据量 8 万点、比 D1-rev 大 4-5 倍)。
- 处置(纯执行决策,科学等价):kill CPU 链 → GPU4(空闲、0 ECC,避开故障 0/3)+
  setsid 脱离控制连接 → tune/readout/mode_cert 走 GPU、voi 留 CPU(JAX)。
  **21 模型 ~18 分钟完成**(对比 CPU 预计 ~24h)。3-seed 集成带吸收设备浮点差异。

## 证书判定:FAIL(6/6)——但这次是干净的基底级负结果

| 带 | 值 | 门 | r1 | 判 |
|---|---|---|---|---|
| C-1 不饱和 | NOHIST 盲 AUC 0.9475 | ≤0.80 | 0.966 | FAIL |
| C-2 历史信号 | +0.0199, CI lo +0.018 | ≥0.10 | 0.0115 | FAIL(CI 干净正、量级仍 5× 不足) |
| C-3 身份超状态 | +0.0141 | ≥0.05 | 0.0158 | FAIL |
| C-4 可识别性 | history 0.905 ✓ / **state-only 0.857** ✗ | ≥0.75 / ≤0.60 | 0.849 | FAIL(去电报化后几乎没动) |
| C-5 信息价值 | 池化 mode_oracle 175.5 vs **reactive 169.8 = +3.3%** | ≥0.15 | (口径错) | FAIL |
| C-6 接线 | oracle=0 ✓ / golden 5/6 | 全过 | 5/6 | FAIL(escalate 真 bug) |

C-5 各臂池化:mode_oracle 175.5 / fullchain 160.2 / prepchain 72.4 / **reactive 169.8**;
最强盲基线 = reactive。static-only legacy lift −0.0075。

## 机制核验(golden,确认非控制器 bug)

- static-yield × fullchain ego:partner_first=**0/60**,trigger=8(yield_abort)触发,
  ego 接手(label1=35)——让位中止机制正常工作;
- static-claim × prepchain ego:partner_first(label2)=20——抢活透传正常;
- **escalate_after_defer 真 bug**:ego 与伙伴都让位 → 汤摆着没人拿 → 机会门永不关闭 →
  升级计数(只在机会关闭时累加)永不触发 → 全 label 0、无 ego_defer 触发。
  **这个 bug 恰好点破核心机制:机会持久不关 = 反应式 ego 够用的根因。**

## 核心裁决(codex 签 C 交叉评审:CONFIRM-WITH-CAVEATS)

**发现(基底级、由正常工作的族驱动)**:即便把选项**选择**去电报化(始终抢活形),
让位只以"最后一刻中止终端交互"表达,潜倾向**仍可从瞬时公共状态读出**(C-4 state-only
0.857)——因为让位的**表达**(接近-绕开/盘旋)本身就是位置签名。于是反应式"等着看
伙伴转不转"的 ego 捕获了 mode_oracle 约 97% 的价值(C-5 +3.3%),**预判信念在本基底
上基本无价值**。

**结构性(非参数性)**:不是调 ε/dwell/wait 能修的。escalate bug 是明证——双方都让位时
机会持久不关。反应式 ego 总能等、看非转化、再接手,因为机会不关且迟接手无代价。
**预判只在"反应太迟或太贵"时才胜过反应。** 当前 Overcooked 终端交接机会持久、观察无代价
→ 反应恒够用。

**codex 关键量化/校正**:(A) 即便去掉 reactive 臂,mode_oracle vs fullchain 也只 +9.6%,
仍 <15% —— fail 对 reactive 调参质疑稳健;(B) C-4 state-only 是参数性(换非位置表达可修),
但**修不了基底**:藏到最后一刻+机会不关只会帮反应臂,C-5/北极星失败是结构性;
(C) escalate 修了救不回(需该族 oracle 172→224+,超量纲);(E) "一修就过"被 DISPUTE
(五条独立带都差太远)。措辞须收窄为"本终端交接基底缺少等待的不可逆代价",非全 Overcooked 定理。

**追溯统一**:同一原理解释整条弧——D1 饱和(谁上菜被状态决定)、盲轮全方法不适配、
latent_v3 两轮——基底从未对"等待-反应"施加代价,故预判信念从不必要。

## 方向建议(Type-B,用户裁量;超出 D1/Link-A 分支表)

第 3 轮基底须加**预判强制结构 = 等待的真实代价**:
(a) 易腐汤/机会 N 步后关闭;(b) 承诺代价——ego 备餐(取盘/占位)须在伙伴揭示前开始、
被伙伴抢走则浪费;(c) **同时多锅分工**+可达/时间约束使反应式 ego 无法两头兼顾(最干净,
让"反应不足"可证)。北极星判据:reactive 臂必须**证明性地**够不着 mode_oracle。

存活资产:D1/Link-A 仪器(含 reactive 池化 C-5)已是经两轮验证的现象存在性证书;
去电报化控制器与 GPU 基建可复用;escalate bug 待第 3 轮随基底重设计一并修。
