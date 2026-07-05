# archive/docs — 归档的历史文档（2026-07-05 文档清理）

这些文档已被后续工作取代，但仍被治理/记录文档引用，故移入此处保留而非删除。
原始评审数据的压缩归档见 `../raw/MANIFEST.txt`。

## 旧 → 新位置映射

| 原路径（根目录） | 现路径 | 取代它的文档/记录 |
|---|---|---|
| README_FIXES_20260624.md | archive/docs/README_FIXES_20260624.md | FINDINGS_LEDGER「修复执行记录」+ METHOD_LOCK sec17 |
| RC_REWARD_CREDIT_FIX.md | archive/docs/RC_REWARD_CREDIT_FIX.md | FINDINGS_LEDGER（P 系修复）+ METHOD_LOCK sec17/18 |

## 同期删除（git 历史可取回，工作树不再保留）

| 文件 | 原因 |
|---|---|
| CODEX_IMPL_SPEC.md / _v2 / _v3 / _v4 | NEW-4 隔离材料，用户裁决删除；隔离令与追记见 FINDINGS_LEDGER NEW-4 段 |
| EXPERIMENT_AUDIT.md / .json | 06-27 审计的 FAIL 结论已被 07-02 根因修复记录取代，且无任何文档引用；原始 trace 保留在 .aris/traces/experiment-audit/ |

注：METHOD_LOCK 与 FINDINGS_LEDGER 为追加式文档，其中对上述旧路径的历史性提及**不作
修改**——以本清单为准查找现位置。
