# Repository Guidelines

## Project Structure & Module Organization

- `raob/`: core package code for belief updates, successor utility, data loading, checkpointing, and benchmark adapters.
- `raob/benchmarks/`: benchmark-specific chart and environment bridge code.
- `scripts/`: command-line audits, training, evaluation, diagnostics, and result summaries.
- `tests/`: pytest suite for RAOB behavior, benchmark scaffolding, diagnostics, and return evaluation.
- `configs/`: experiment configs such as `configs/raob_experiment.yaml`.
- `Method_design.md`: research design source of truth for implementation tasks and fixed constraints.
- `Audit/`: required location for post-implementation audit notes.
- `runs/`: generated experiment inputs, checkpoints, diagnostics, and JSON outputs.

## Execution Location Constraint

Do not execute any tests, scripts, training, evaluation, or runtime commands locally. If operation, validation, training, evaluation, or diagnostics are needed, execute them on the server according to `guidance.md`.

Training and evaluation examples are in `README.md`; keep formal runs on `--strict-group-ids` unless doing diagnostics, and run them only on the server.

## Server Resource Utilization

For server execution, default to high-resource settings unless the user explicitly requests a small smoke run. Target saturation of the container/cgroup CPU quota, not the host's full visible CPU count. If the quota is 18 cores, this may appear as about `1800%` in Linux `ps` or about `100%` in container-normalized dashboards; both mean the allocated CPU is saturated. Use cgroup detection when possible, otherwise default to `CPU_TARGET_CORES=18`, `--workers 18`, and per-worker `--torch-num-threads 1`. Use GPU for tensor/model phases whenever supported: set `CUDA_VISIBLE_DEVICES=0` and pass `--device cuda`. Raw OGC/JAX replay phases may remain CPU-bound, but should parallelize independent probes/layouts/sequences across workers. Do not satisfy resource targets by changing planner behavior, planner horizon, TD(lambda), grouping, chart features, reward model, reward scale, covariance, update gates, low-rank rules, source-prior selection, actor heads, skill layers, history latents, partner labels, or beta-to-policy paths.

## Implementation Workflow

Before writing implementation code, read `Method_design.md` and identify the exact task, fixed research requirements, allowed assumptions, and constraints. Do not treat an implementation idea as valid until checked against this design.

After implementation, audit changed code against `Method_design.md` and record results under `Audit/`, for example `Audit/audit_<topic>_<date>.md`. If any mismatch appears, finish the audit, revise the mismatch, then repeat implementation, audit, and revision until the audit confirms full correspondence.

## Coding Style & Naming Conventions

Use Python 3.10+ and the existing style: 4-space indentation, useful type hints, snake_case functions and variables, PascalCase classes, and uppercase constants. Prefer explicit validation errors. Script flags use kebab-case mapped to snake_case `argparse` attributes.

## Testing Guidelines

Tests use `pytest` under `tests/test_*.py`. Name tests by behavior, for example `test_identifiable_update_rejects_genuinely_nonidentifiable_evidence`. Add focused tests for gates, posterior updates, utility scoring, grouping, and benchmark contracts. `tests/conftest.py` already limits PyTorch threads.

## Commit & Pull Request Guidelines

No local `.git` history is available, so conventions cannot be inferred. Use concise imperative commits such as `Add horizon reachability audit`. PRs should include motivation, changed RAOB behavior, commands run, and relevant `runs/` or `Audit/` paths.

## RAOB Contract Notes

Preserve the constraints stated in `README.md`: beta enters only through `M(g,a) beta`, response-law groups are not partner-ID groups in formal runs, control uses primitive-action posterior-aware MPC, and successor utility must not become a scalar return patch.
