# RAOB-ZSC 本地-服务器 Guidance

## 当前映射关系

本地项目路径：

```text
/Users/aoudsung/Documents/RAOB
```

当前唯一正确的服务器项目路径：

```text
/apps/users/cxw/Document/CodeSpace/Selfs/RAOB_ZSC
```

服务器依赖复用来源：

```text
/apps/users/cxw/Document/CodeSpace/Selfs/ZSC_coordinator
```

当前 RAOB-ZSC 服务器项目内应保留以下链接：

```text
/apps/users/cxw/Document/CodeSpace/Selfs/RAOB_ZSC/.venv -> /apps/users/cxw/Document/CodeSpace/Selfs/ZSC_coordinator/.venv
/apps/users/cxw/Document/CodeSpace/Selfs/RAOB_ZSC/external -> /apps/users/cxw/Document/CodeSpace/Selfs/ZSC_coordinator/external
```

不要把当前项目同步到旧 ZSC 项目路径或历史临时路径：

```text
/apps/users/cxw/Document/CodeSpace/Selfs/ZSC_coordinator
/apps/users/cxw/Document/CodeSpace/Selfs/ZSC_coordinator_bct_aligned_20260525
```

本地当前项目资料：

```text
RAOB_ZSC_Method_Design.md
guidance.md
```

## 连接方式

SSH 连接命令只使用 `zsc` 别名；不要再使用旧连接别名：

```bash
ssh zsc
```

当前 `zsc` 对应：

```text
HostName 10.8.128.25
User cxw-qbQsYTpa
Port 6988
```

服务器项目 Python：

```text
/apps/users/cxw/Document/CodeSpace/Selfs/RAOB_ZSC/.venv/bin/python
```

该 Python 当前通过 `.venv` symlink 复用：

```text
/apps/users/cxw/Document/CodeSpace/Selfs/ZSC_coordinator/.venv/bin/python
```

进入服务器项目目录：

```bash
cd /apps/users/cxw/Document/CodeSpace/Selfs/RAOB_ZSC
```

如需交互式 SSH/SCP 登录，使用已授权的服务器登录信息；不要在无关文档、日志或命令历史中重复扩散。

## 项目创建与依赖复用

如果服务器项目目录不存在，先新建 RAOB-ZSC 项目目录，并在项目内复用 ZSC 项目的 `.venv` 和 `external`：

```bash
project=/apps/users/cxw/Document/CodeSpace/Selfs/RAOB_ZSC
source=/apps/users/cxw/Document/CodeSpace/Selfs/ZSC_coordinator

mkdir -p "$project"
ln -s "$source/.venv" "$project/.venv"
ln -s "$source/external" "$project/external"
```

如果链接已存在，不要重复创建；先检查：

```bash
ls -la /apps/users/cxw/Document/CodeSpace/Selfs/RAOB_ZSC
```

## 已验证可行的同步流程

本地只做文档、静态阅读、打包和上传；所有运行、测试、训练和 audit 都在服务器执行。

推荐流程是本地打包、上传到服务器 `/tmp`、服务器项目目录内解包。当前 RAOB-ZSC 是新项目目录，不覆盖 ZSC_coordinator；如后续覆盖已有 RAOB-ZSC 代码，先备份目标文件。

本地打包：

```bash
cd /Users/aoudsung/Documents/RAOB
COPYFILE_DISABLE=1 tar -czf /private/tmp/raob_zsc_sync.tar.gz \
  .gitignore pyproject.toml guidance.md AGENTS.md Method_design.md README.md \
  raob tests configs Audit
```

上传到服务器：

```bash
scp /private/tmp/raob_zsc_sync.tar.gz zsc:/tmp/raob_zsc_sync.tar.gz
```

服务器备份并解包：

```bash
cd /apps/users/cxw/Document/CodeSpace/Selfs/RAOB_ZSC
backup=/apps/users/cxw/Document/CodeSpace/Selfs/RAOB_ZSC_server_backup_$(date +%Y%m%d_%H%M%S)
mkdir -p "$backup"
tar -czf "$backup/code_snapshot.tar.gz" \
  .gitignore pyproject.toml guidance.md AGENTS.md Method_design.md README.md \
  raob tests configs Audit \
  2>/dev/null || true
tar -xzf /tmp/raob_zsc_sync.tar.gz
```

如果 `tar -xzf` 输出 macOS extended header warning，例如 `LIBARCHIVE.xattr...`，通常可以忽略；这不是代码解包失败。

## 服务器验证命令

所有验证默认在服务器执行：

```bash
cd /apps/users/cxw/Document/CodeSpace/Selfs/RAOB_ZSC

PYTHONPATH=. .venv/bin/python -m ruff check raob tests
PYTHONPATH=. .venv/bin/python -m compileall -q raob tests
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
```

## 服务器资源使用

正式诊断默认榨干当前 Docker/cgroup 配额，而不是按宿主机可见核心数盲目扩展。当前服务器可能显示 `nproc=144`，但可用配额按 18 cores 处理；此时 Linux `ps` 中约 `1800%` 或容器面板中 `100%` 都表示已打满配额。

默认资源设置：

```bash
export CPU_TARGET_CORES=18
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

当前工程阶段是 SRVF-MAPPO classic Overcooked batch plumbing smoke。不要运行
OGC、OvercookedV2 或历史脚本路径。

验证后清理缓存，保持服务器项目干净：

```bash
cd /apps/users/cxw/Document/CodeSpace/Selfs/RAOB_ZSC
rm -rf .pytest_cache .ruff_cache
find raob tests -type d -name __pycache__ -prune -exec rm -rf {} +
```

## 关键代码树一致性检查

如需确认本地暂存代码与服务器代码一致，可在对应代码目录分别运行：

```bash
python3 - <<'PY'
import hashlib
from pathlib import Path

roots = ["raob", "tests", "configs", "Audit"]
files = []
for root in roots:
    p = Path(root)
    if p.exists():
        files.extend(x for x in p.rglob("*") if x.is_file() and "__pycache__" not in x.parts)
for name in ["pyproject.toml", "guidance.md", "AGENTS.md", "Method_design.md", "README.md"]:
    p = Path(name)
    if p.exists():
        files.append(p)

h = hashlib.sha256()
for p in sorted(files, key=lambda x: str(x)):
    h.update(str(p).encode())
    h.update(b"\0")
    h.update(p.read_bytes())
    h.update(b"\0")
print(len(files), h.hexdigest())
PY
```

只有文件数和 SHA256 相同，才可声明关键代码树一致。

## 注意事项

- 当前项目只映射到 `/apps/users/cxw/Document/CodeSpace/Selfs/RAOB_ZSC`。
- 服务器连接只使用 `zsc` SSH 别名；不要使用其他旧别名。
- 服务器 `.venv` 与 `external` 通过 symlink 复用 ZSC_coordinator，默认不要复制整份环境。
- 本地不执行任何代码运行、测试、训练或 audit。
- 覆盖服务器 RAOB-ZSC 代码前必须备份当前 RAOB-ZSC 目标文件。
- 不要覆盖、移动或清理 `/apps/users/cxw/Document/CodeSpace/Selfs/ZSC_coordinator`。
- 不要生成过多垃圾文件；验证后清理 Python、pytest、ruff 缓存。
- 不要复用旧 run dir 作为新正式训练输出目录。
- 如果训练进程是在代码同步前启动的，它不会自动加载新代码；需要停止旧进程并用新 run dir 重启。
- 停训练优先使用 `kill -INT <pid>`，让 Python 尽量走 `finally` 并 flush cache。
- 当前 benchmark 接入层位于 `raob/benchmarks`；它只负责 classic Overcooked-AI wrapper、GOAT classic partner policy 和必要 tensor helper，不承载训练 loss。
- 当前核心实现位于 `raob/srvf_mappo.py`，配置样例位于 `configs/srvf_mappo_classic.yaml`。正式实验前必须明确 partner policy registry 和 evaluation protocol，不得把 random/random smoke 表述为正式 benchmark 结果。
