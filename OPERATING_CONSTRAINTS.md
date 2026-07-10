# OPERATING_CONSTRAINTS.md

**Authoritative execution boundary for the ARIS4ZSC project.**
Last updated: 2026-07-08 · Owner: project lead

> 治理精简 2026-07-08：本文件的门已按 docs/status/GOVERNANCE_CUTLIST.md 处置；加/减门须过 OPERATING_CONSTRAINTS.md §7 退休阀门。

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
- **Drift check (after-the-fact audit, not a start gate): after a code update,
  produce an explicit `git` diff and review it when checking what changed.** This
  is a cheap post-hoc drift check — it does not gate whether a run may start (§7.2).
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
| `AUTO_PROCEED` | `false` | Stops the loop for human judgment only at the two points that have caught real failures — run authorization and final read-out adjudication (§7.5). Not a sign-off before every launch or mid-experiment. |
| `human checkpoint` | `run authorization + final read-out only` | Per §7.5, human sign-off is reserved for run authorization (the execution-boundary consent) and final read-out adjudication. No per-launch or mid-experiment sign-off — that maps to no recorded catch; the claim-acceptance judgment is carried by the Type-B acceptance gate (§4). |
| `CODE_REVIEW` | `true` | Cross-model review of experiment code before deploy |
| `reviewer` | `codex` (GPT-5.5, xhigh) | Must be a **different model family** than the Claude executor |

Final-artifact assurance (the full audit chain, `assurance = submission`) is
folded into the Type-B acceptance gate — see §4.

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

**Final / outward artifacts** (submission-scale deliverables) are the product-scale
form of the same rule: they require the full assurance audit chain
(`assurance = submission`) and may never be self-signed. This is the merged home
of the former §3 `assurance` default.

---

## 5. Enforcement status

| Layer | Mechanism | Status |
|-------|-----------|--------|
| Declared intent | this file + [`.aris/config.json`](.aris/config.json) | ✅ in place |
| Always-loaded restatement | [CLAUDE.md](CLAUDE.md) entrypoints | ✅ in place |
| Mechanical enforcement | `.claude/hooks/no_local_exec_guard.py` PreToolUse guard (stop-and-**ask** on local exec; remote `zsc-customer` never gated; fail-open) | 🟡 drafted — **pending user approval** via `/hooks` or session restart |
| Scientific-invariant gate (pre-claim / pre-deploy audit) | `.aris/tools/aris_bellman_fidelity_gate.py` — static, 9 checks (I1–I9), 6-state verdict, exit 1 on RED; runs in-boundary | ✅ built — green on current tree (`FIDELITY_GATE.{json,md}`) |

The static fidelity gate (I1–I9) is an **after-the-fact audit, not a start gate**:
require it green *before reading a claim or before deploying changed method code*,
not before every run (§7.2). It checks method identity; it never decides whether a
run may launch.

The guard is registered in `.claude/settings.json` but Claude Code requires the
user to approve a new hook before it runs. Until approved, the boundary remains
**advisory** and depends on the agent honoring this doc — treat a local-execution
request as a stop-and-confirm. Once approved, matching local-exec commands raise an
approval prompt (the user may authorize a one-off, or "always allow" to relax for
the session); the guard can only add a prompt, never hard-block.

---

## 6. Data-sufficiency discipline (2026-07-06 — user directive after the 32-episode misdiagnosis)

**Origin (measured, one month of damage):** the formal e1rev configs carried
`total_updates: 5000` × `updates_per_transition: 8` ⇒ **625 transitions ≈ 32
episodes of total training experience**, and no document ever stated this
number. Every negative diagnosis produced on that substrate — held-out egoCCR
0.125, the four-granularity table, "belief shifts Q but cannot flip wait",
"terminal competence evaporates" — was later shown to be a sample-starvation
artifact: at 2000 episodes the SAME method with NO mechanism change passed the
preregistered role-adaptation criterion (4/5 seeds) and the E2 zeroed test
showed the belief channel load-bearing. Cost: ~1 month, three external review
rounds, multiple mechanism hypotheses chased.

**Rules (binding for every future run):**

1. **Effective data budget must be computed and recorded** for every training
   run, at launch and in the log entry: transitions
   (= total_updates / updates_per_transition under the 1-collect loop) and
   episodes (≈ transitions / max_episode_options). Read it back from produced
   artifacts (`metrics.episode_returns` length, `resolved_config.json`), never
   from config intent.
2. **Smoke-scale runs may satisfy Type-A gates only** (does it run / compile /
   produce artifacts). They must be labeled SMOKE in EXPERIMENT_LOG and their
   numbers may not enter any table, comparison, diagnosis, or claim readout.
   Cheap smoke tests remain encouraged — for wiring, never for science.
3. **Type-B / claim-level readouts require a data-sufficient substrate.** The
   floor is set by the latest recorded sufficiency evidence for that substrate
   (currently asymm×role_conditioned_v2: **≥ 2000 episodes / 40k transitions**,
   EXPERIMENT_LOG 2026-07-06), plus a passed consolidation check (late-window
   validation competence persists; if the curve is still rising, scale further
   before adjudicating).
4. **Any change to a data-quantity parameter** (`updates_per_transition`,
   `total_updates`, `max_episode_options`, `replay_size`) requires restating
   the effective episode count in the change record.
5. When a result looks like a method failure, **check the data budget before
   hypothesizing mechanisms** — "how many episodes did this model actually
   experience?" is the first diagnostic question, not the last.

---

## 7. Gate lifecycle — every gate must be able to die (2026-07-08)

**Origin:** the governance layer grew monotonically — every recorded failure
added a permanent gate, discipline, ledger ID, or pre-registration, and none
were ever retired. The accumulated mass made starting an experiment feel like
"build all the gates first." A full audit (`docs/status/GOVERNANCE_CUTLIST.md`) found ~124
gate-entries across 7 docs collapsing to ~20–25 unique load-bearing rules; the
rest were duplicate restatements, closed incidents, or ceremony mapping to no
recorded failure. This section is the ratchet's retirement valve.

**Rules (binding for every future gate):**

1. **A new gate must record two things or it is not added:** the *specific real
   recorded failure* it prevents, and its *retirement condition*. A rule that
   maps to no recorded incident is at most a post-hoc audit, never a
   start-blocking gate.
2. **Default altitude = gate the claim, not the run start.** A check may block
   *starting* a run only when its failure cannot be detected or repaired after
   the fact — currently: preflight layout validity, effective data budget,
   oracle-free evidence path, and blind-split non-contamination. Everything else
   (static fidelity invariants, wiring read-backs, wording ladders, method-
   identity checks) runs as a **pre-claim / pre-deploy audit**, not a per-launch
   hoop. The fidelity gate is one fast static command; require it green *before
   reading a claim or changing method code*, not before every run.
3. **One home per rule.** A rule lives in exactly one file; everywhere else links
   to it. Do not restate the same gate in CLAUDE.md + this file + AGENTS.md +
   the dashboard + the ledger.
4. **Periodic sweep.** When a gate's incident is closed or its retirement
   condition is met, archive it. Closed/tautological gates do not stay listed as
   active.
5. **Human sign-off is scarce.** Retain human Type-B judgment at the two points
   that have caught real failures — run authorization (the execution-boundary
   consent) and final read-out adjudication. Do not add per-launch or
   mid-experiment human sign-offs; they map to no recorded catch and are pure
   start-latency.

---

## 已归档 / 已合并（2026-07-08，依据 docs/status/GOVERNANCE_CUTLIST.md）

- **`assurance = submission` 五层审计链（原 §3 表的一行）** → 合并进 §4 Type-B
  接受门。理由：它没有独立对应的真实失败，是"不得自我签收对外产物"原则在产物尺度上
  的重复表述。唯一内容（最终/对外产物须过完整审计链、不得自判）现由 §4 承载。

注：以下三项为**降级**（保留检查、改为事后/读数前审计），仍在各自章节的在架面上，
不移入本节——每次代码更新后的 `git` diff（§2，事后漂移检查）、`AUTO_PROCEED` /
`human checkpoint` 的启动前签字（§3，只保留 §7.5 的两个人工判断点）、静态 fidelity
门 I1–I9（§5，读数前 / 改方法后审计）。
