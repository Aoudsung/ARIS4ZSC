# rootcause_review_20260702 — codex 评审包

异模型族（codex GPT-5.5, xhigh）对 2026-07-02 根因裁定的对抗性静态评审。
计划见 [ROOTCAUSE_REVIEW_PLAN.md](../../ROOTCAUSE_REVIEW_PLAN.md)。

## 文件
- `REVIEW_BRIEF.md` — 权威简报：5 个裁决问题（含反假设）、证据锚点表（file:line + grep 串 + 证伪法）、范围边界、必答清单、返回格式。
- `PROMPT.txt` — 投喂 codex 的驱动 prompt（自含角色/反锚定/输出要求，指向简报）。
- `CODEX_OUTPUT.md` — **codex 写入此处**（评审产物）。
- `CLEAN_ZIP_DEEP_REVIEW_PROMPT.txt` — 面向另一个静态审查者的 clean zip 深评 prompt。
- `ROOTCAUSE_FIX_EXECUTOR_PROMPT.txt` — 面向根因修复执行者的实施 prompt（以 clean zip + 审查结果为输入）。

## 运行（用户终端，repo 根目录）
```
cd /Users/aoudsung/Documents/ARIS4ZSC
codex exec --model gpt-5.5 -c model_reasoning_effort=xhigh "$(cat review_bundles/rootcause_review_20260702/PROMPT.txt)"
```
（模型名/参数按本机 codex 配置调整；关键是 xhigh reasoning + 只读静态 + 能读本仓库文件。）

## 评审后
verdict 按计划 §2 灌入 `FINDINGS_LEDGER.md`；分歧走 Type-B 双签；再按 §2.3 依赖链
（去 oracle 优先）逐步写静态修复 handoff。**先解阻塞决策**：role_conditioned_v1 伙伴库
是否违反合成数据禁令（计划 §5.1）。
