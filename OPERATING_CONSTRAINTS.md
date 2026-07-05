# OPERATING_CONSTRAINTS.md

**Authoritative execution boundary for the ARIS4ZSC project.**
Last updated: 2026-06-26 · Owner: project lead

> Precedence: this file is the source of truth for *what an agent may execute* in
> this project. If a generic ARIS skill's default behavior conflicts with the
> boundary below, **this file wins** — stop at a static handoff instead.
> Referenced as a required entrypoint by [CLAUDE.md](CLAUDE.md) and
> [AGENTS.md](AGENTS.md).

---

## 0. Terminology (read once, avoid the collision)

Two different things in this repo are both called "ARIS":

| Term | Means | Lives in |
|------|-------|----------|
| **ARIS** (the harness) | Auto-claude-code **R**esearch **I**n **S**leep — the agent framework (80 skills, cross-model review). The *tooling*. | `.claude/skills/` → `~/aris_repo` |
| **ARIS-Bellman** (the method) | The research contribution: factor-local Bellman control for ZSC. The *science under study*. | `src/aris_bellman/`, `experiments/overcooked_v2/` |

When this doc says "skill", "harness", "review loop" → ARIS the harness.
When it says "method", "the model", "training", "factor" → ARIS-Bellman.
Full project identity: [PROJECT_DASHBOARD.md](PROJECT_DASHBOARD.md).

---

## 1. Execution boundary

**Current constraint: no local code execution or local project runs unless the
user explicitly relaxes this rule, per session.**

### Allowed (static maintenance — no approval needed)
- Read / search files, inspect `git status` / `git log` / diffs.
- Create or edit Markdown coordination documents.
- Update configuration needed for agent coordination (`.aris/`, `.claude/hooks/`,
  this file, the dashboard).

### Forbidden without explicit per-session authorization
- Local tests, local project runners, experiment execution, local result
  generation.
- LaTeX compilation, paper submission, rebuttal, camera-ready work.
- Remote SSH execution used as a substitute for the contract in §2 without
  authorization.
- Starting `/auto-review-loop`, `/paper-writing`, `/research-pipeline`,
  `/rebuttal`, or any acceptance / camera-ready workflow before its readiness
  gate is satisfied by recorded evidence (gates: [PROJECT_DASHBOARD.md](PROJECT_DASHBOARD.md) §4).

### ARIS skill scope
ARIS skills may be used only within this boundary. When a skill would run tests,
experiments, LaTeX, paper-writing, review loops, or remote commands, **stop at a
static handoff** unless the user has explicitly authorized that action.

---

## 2. Remote execution contract

All experiments run **remote-only**, never local. Contract pinned in
[CUSTOMER.md](CUSTOMER.md):

- Host: `ssh zsc-customer` (persistent control connection, 8h reuse; password in CUSTOMER.md).
- Project path (has `.venv`): `/apps/users/cxw/Document/CodeSpace/Selfs/CPR_REPO`.
- 8 GPUs available.
- **Discipline: after every code update, produce an explicit `git` diff.**
- Formal run contract (preflight → CE → graph → train → eval) is defined in
  [README_FIXES_20260624.md](archive/docs/README_FIXES_20260624.md). Formal training **requires an
  accepted preflight** via `--preflight_path`; rejected-layout smoke runs must not
  go through the formal trainer.

Even when remote execution is authorized, the no-local-exec rule above still
holds for the host machine.

---

## 3. ARIS invocation defaults for this project

When invoking any experiment- or review-class ARIS skill in this project, apply
these defaults (override inline only with explicit user authorization). Mirrored
machine-readably in [`.aris/config.json`](.aris/config.json); will be **enforced**
by the Phase-1 PreToolUse hook (`.claude/hooks/`, not yet installed).

| Parameter | Project default | Why |
|-----------|----------------|-----|
| `GPU` | `remote` | No local execution; §2 contract |
| `AUTO_PROCEED` | `false` | Human gates every irreversible / outward step |
| `human checkpoint` | `true` | Approval before experiment launch + claim acceptance |
| `CODE_REVIEW` | `true` | Cross-model review of experiment code before deploy |
| `reviewer` | `codex` (GPT-5.5, xhigh) | Must be a **different model family** than the Claude executor |
| `assurance` | `submission` for any final artifact | Full 5-layer audit chain gates the result |

---

## 4. Acceptance-gate rule (autonomy boundary)

ARIS is built to run autonomously ("research in sleep"). In this project, autonomy
is allowed to **drive** but never to **acquit** — per
`shared-references/acceptance-gate.md`:

- **Type-A gates** (did it run / compile / finish — machine-checkable): an agent
  may self-judge.
- **Type-B gates** (is the claim supported / is the result good / is the proof
  valid): **never** self-judged. Route to the cross-model reviewer (codex) and,
  in this project, also to a human checkpoint.

The five decisive scientific claims ([PROJECT_DASHBOARD.md](PROJECT_DASHBOARD.md) §3)
are Type-B. No loop may declare them supported on its own verdict.

---

## 5. Enforcement status

| Layer | Mechanism | Status |
|-------|-----------|--------|
| Declared intent | this file + [`.aris/config.json`](.aris/config.json) | ✅ in place |
| Always-loaded restatement | [CLAUDE.md](CLAUDE.md) entrypoints | ✅ in place |
| Mechanical enforcement | `.claude/hooks/no_local_exec_guard.py` PreToolUse guard (stop-and-**ask** on local exec; remote `zsc-customer` never gated; fail-open) | 🟡 drafted — **pending user approval** via `/hooks` or session restart |
| Scientific-invariant gate | `.aris/tools/aris_bellman_fidelity_gate.py` — static, 9 checks (I1–I9), 6-state verdict, exit 1 on RED; runs in-boundary | ✅ built — green on current tree (`FIDELITY_GATE.{json,md}`) |

The guard is registered in `.claude/settings.json` but Claude Code requires the
user to approve a new hook before it runs. Until approved, the boundary remains
**advisory** and depends on the agent honoring this doc — treat a local-execution
request as a stop-and-confirm. Once approved, matching local-exec commands raise an
approval prompt (the user may authorize a one-off, or "always allow" to relax for
the session); the guard can only add a prompt, never hard-block.
