# THREE_LINKS_IMPLEMENTATION_PLAN — Instantiating, Pressuring, and Fairly Testing Emergent Zero-Shot Coordination

**Date:** 2026-07-07 · **Status:** design plan, execution protocol clarified, awaiting user sign (Sign-A1)
**Upstream evidence:** D1/D1-rev cross-instrument adjudication (commits `619a224`, `da32140`,
`87b3d0b`; METHOD_LOCK sec18.14.3 blind-round sign-③). The current substrate's terminal-axis
partner response is a near-deterministic function of instantaneous public state and this
state-sufficiency transfers to blind partners; identity carries zero information beyond state
(ID-oracle ≈ state-only baseline); the only partner with a genuinely latent disposition
(task-progress-driven neutral) shows +0.125 history gain — the phenomenon-existence proof and
design template.

**Scope:** three serially-gated links. Link A makes the phenomenon (hidden-disposition
inference) exist and certifies it. Link B makes the training distribution exercise it.
Link C runs the first FAIR test of the original emergence proposition. **METHOD_LOCK is not
touched anywhere in this plan** — the method under test remains factor-local belief +
factor-local Q under a single TD loss (invariants I1–I18 stay green). Any method-layer change
remains gated behind Link C's preregistered outcome and a separate METHOD_LOCK amendment.

**Serial-gating principle (binding):** no Link-B investment before the Link-A certificate
passes; no Link-C run before Link-B coverage and seals are green. This is the structured form
of the 32-episode lesson: a failed upstream link renders all downstream spend uninformative.

**Notation:** numbers tagged **[F@A3]**, **[F@B1]**, **[F@C2]** are provisional defaults that
freeze at the indicated sign point; pre-data amendments allowed with recorded rationale
(same convention as the D1 prereg revisions).

---

## 0. Objective and non-goals

**Objective.** Produce a substrate + training distribution on which "zero-shot coordination
via belief over latent partner dispositions" is (a) instantiated, (b) certified measurable,
and (c) fairly testable for *emergence from plain Bellman control* — then run that test.

**Non-goals.** This plan does not claim emergence will occur; it does not modify the method;
it does not pre-commit to any post-test pivot. Success of the plan = the emergence question
receives its first evidence-bearing answer, in either direction.

---

## 1. Link A — Substrate v3: latent-mode partners + Phenomenon-Existence Certificate

### 1.1 Design requirements (each anchored to a measured failure)

| # | Requirement | Evidence anchor |
|---|---|---|
| A-R1 | **Geometry–disposition independence**: partners with identical spatial behavior profiles must differ in disposition | D1-rev: ID-oracle ≈ state-only ⇒ current roles are positionally readable |
| A-R2 | **Temporal extension**: disposition = latent mode with dwell time ≫ identification time | The +0.125 positive-control partner is progress-driven (temporal), not positional |
| A-R3 | **Contingency**: some dispositions are reaction functions of ego's behavior, existing only in interaction history | Gives information-gathering a real return gradient (Link C credit-path shortening) |
| A-R4 | **Calibrated residual stochasticity**: within-mode noise present but mode differences dominate | D1-rev per-ego decomposition: residual concentrated in random-ego rollouts |
| A-R5 | **No oracle leakage**: mode state never enters ego-side evidence | P1 boundary + oracle_source_count=0 gates (already mechanical) |

### 1.2 Mode-manifold parameterization

A v3 partner = `(geometry_profile, disposition, dynamics, noise)` with **geometry sampled
independently of disposition** (A-R1 by construction).

**Geometry profiles** (reuse existing role machinery): `{ingredient_near, ingredient_far,
server_zone, bottleneck_zone, prep_zone, flexible}` — patrol/home/pot/delivery preferences
only; **no terminal-policy content**.

**Disposition families** (training manifold; each has a scalar/discrete parameter vector):

| Family | Semantics | Params (training ranges) |
|---|---|---|
| `static_claim` / `static_yield` | anchors (existing behavior) | — |
| `patience(N)` | yields for the first N terminal opportunities, then claims | N ∈ {2..6} **[F@A3]** |
| `block_switch(d)` | semi-Markov switching claim↔yield | dwell d ∈ [12, 25] option-decisions **[F@A3]** |
| `tit_for_tat(m)` | yields next m opportunities after ego claims one; claims after ego defers twice | m ∈ {1..3} **[F@A3]** |
| `escalate_after_defer(k)` | starts yield; escalates to claim after k consecutive ego deferrals | k ∈ {2, 3} **[F@A3]** |

**Dynamics:** episode-level latent `z ~ prior over (family, params)`; within-episode
transitions only via the family's own rule (block_switch dwell, reaction triggers). Dwell
floor 12 option-decisions vs identification budget ≤ 8 opportunity observations (A-R2 ratio
≥ 1.5) **[F@A3]**.

**Noise:** per-mode ε-random action rate ε ∈ [0.05, 0.15] **[F@A3]**.

**Episode horizon:** `max_episode_options: 60` for v3 runs (≥ 1 expected mode switch in
≥ 50% of block_switch episodes; enables the switch-tracking metric in Link C) **[F@A3]**.

### 1.3 Code plan

| Item | Path / mechanism |
|---|---|
| Mode controller | NEW `experiments/overcooked_v2/partner_modes.py` — `LatentModeController` **wrapping** `ScriptedProtocolPartner` by composition (dwell timers, opportunity counters, reaction triggers on ego's public terminal acts). Existing registries untouched (no-op-extraction discipline). |
| Registry | `partner_pool.py`: new registry `latent_v3_dev` (FIXED named specs for certification/reproducibility) + `sample_mode_spec(rng, family_quotas)` (parametric sampler for Link B). Determinism contract: partner behavior is a pure function of (spec, seed, interaction history). |
| Tests | unit tests per family (trigger tables, dwell arithmetic, reaction latency); golden trace per family (1 episode, hand-verified like D1 goldens). |
| Leak surface | none new: controller changes partner-internal policy only; executor path (`option_primitive_step`) already strips partner internals (`behavior_observed:oracle_stripped`); certificate re-asserts `oracle_source_count=0`. |
| Review | codex diff review (project default CODE_REVIEW=true) before any generation. |

### 1.4 Phenomenon-Existence Certificate (Sign-A3 gate; D1 pipeline reused verbatim)

Run `diag_d1_dataset.py` + `diag_d1_train.py` on candidate v3 partners (same ego mixture,
same gates/labels as D1-rev, EPS_SCALE=2). **Certificate bands (all must hold; [F@A3]):**

| Check | Band | Kills which failure |
|---|---|---|
| C-1 unsaturation | NOHIST blind AUC_pf ≤ **0.80** | D1's state-sufficiency |
| C-2 signal exists & needs history | G_blind = FULL − NOHIST ≥ **0.10**, CI lower > 0 | "nothing to infer" |
| C-3 identity beyond state | IDORACLE − NOHIST (in-dist) ≥ **0.05** | positional readability (A-R1 empirical) |
| C-4 identifiability | history-based mode classifier balanced-acc ≥ **0.75** within ≤ 8 opportunity observations; state-only mode classifier ≤ **0.60** | "too hidden" (tightrope left edge) |
| C-5 value of information | paired scripted probes (override-gate machinery): mode-informed best-response ego vs mode-blind ego team return ≥ **+15%** on mode-sensitive partners | "predictable but value-irrelevant" |
| C-6 wiring | oracle_source_count=0; evidence_policy=behavior_inferred_v1; golden PASS | leakage / label bugs |

C-4/C-5 are small additions to `diag_d1_train.py` (a mode-label head evaluated at increasing
observation counts) and a new paired probe script riding `probe_behavior.py` machinery.

**Iteration loop:** each certificate round costs ~1–1.5 h (18× speedup infra). If a band
fails, adjust the flagged parameter axis (ε, dwell, geometry decorrelation, reaction
sharpness) and re-run; every round's numbers are logged (no silent tuning). Certificate data
never trains the method (evaluation-only, like all D1 data).

---

## 2. Link B — Training distribution: mode-manifold sampling under audit

### 2.1 Sampler (kills memorization pressure-free training)

- Each training episode draws `spec ~ SamplerTrain`: geometry ⊥ disposition ⊥ dynamics ⊥
  noise, with family quotas (each disposition family ≥ **15%** episode share **[F@B1]**).
- **No persistent individuals**: per-episode resample, no cross-episode identity. The
  "memorize who's who" shortcut (blind-round pathology: even partner_id_q learned nothing
  transferable) is removed by construction — identity does not exist; only modes do.
- Anchors: `static_claim`/`static_yield` retained in the manifold so the old behaviors are a
  measure-zero-ish subset, preserving comparability with the v2 era.

### 2.2 Held-out architecture and seals

| Level | Construction | Use |
|---|---|---|
| in-dist val | fresh draws from SamplerTrain | training-time validation |
| dev-heldout | held-out **regions** of the same parameter space (e.g., N=5, dwell∈[20,25], m=3 excluded from training ranges) — declared interpolation vs extrapolation cuts | calibration, iterated freely |
| **blind_v3** | disposition **mechanism classes disjoint from the training manifold** (e.g., resource-budget-conditional, pot-state-conditional, long-memory delayed-mirror — exact specs authored only at seal time), zero model contact, sealed with code+spec hashes per the sec18.14 procedure, single-look | Link C primary readout |

**Independence declaration (binding, pre-training):** SamplerTrain family list and blind_v3
mechanism-class list are disjoint by construction and hash-sealed together — the D1-era rule
that keeps zero-shot from degrading into interpolation.

### 2.3 Coverage and sufficiency audits (Type-A, artifact-readback)

- Per (disposition family × factor) opportunity-event counts read back from generated
  training data (CE-support-audit style); minimum counts per cell **[F@B1]**.
- Effective data budget lines on every run (OPERATING_CONSTRAINTS §6): episodes/transitions
  read from artifacts, never config intent.

### 2.4 Pipeline changes

- `ce_sampler.py`: partner acquisition via SamplerTrain (seeded); CE support audit reported
  per family instead of per individual.
- `evaluate_aris.py`: `--partner_set latent_v3_dev / blind_v3` via the existing eval-only
  override (f870d77); zeroed ablation (LDS-B3) unchanged.
- NEW `scripts/aggregate_emerge.py`: opportunity-normalized metrics (below) — the
  blind-round metric-confound lesson implemented once, reused everywhere.

---

## 3. Link C — Fair emergence test (method untouched)

### 3.1 Data-sufficiency recalibration (the old 2000-episode floor is v2-specific)

Scaling probe: aris_bellman only, episodes ∈ {1000, 2000, 4000, 8000} × 2 seeds, original
e1rev schedule, consolidation criterion per §6 discipline (late-window validation competence
persists; still-rising ⇒ scale further). Floor := smallest plateau point; full runs at the
floor **[F@C2]**. Cost: ≈ (episodes × ~2 s)/seed generation-equivalent; the 8000-ep probe
≈ 3–4 h/seed at current infra.

### 3.2 E-EMERGE (preregistered; skeleton frozen here, numerics at Sign-C2)

- **Arms:** aris_bellman, base_only, global_gru, flat_factor (5 seeds each) + `mode_oracle`
  reference arm (receives the true current mode as input — upper reference only, never a
  claim arm; replaces partner_id_q, which is meaningless without persistent individuals).
- **Training:** SamplerTrain episodes at the recalibrated floor; single TD loss; fidelity
  gate green; ITT (no seed exclusion).
- **Evaluation:** blind_v3 (single look) + dev-heldout (calibration); ≥ 100 episodes/partner
  × 2 eval seeds; thread-caps execution card.
- **Primary metric (opportunity-normalized):** mode-conditioned behavioral contrast
  Δ_norm = P(ego takes terminal opportunity | accumulated evidence indicates yield-mode)
  − P(ego takes | evidence indicates claim-mode), computed **only over opportunities where
  taking was feasible** (feasibility from the D1 gate machinery) — this is the
  opportunity-normalization the blind round showed is mandatory.
- **Secondary (the emergent-belief signature):** within-episode **switch tracking** — median
  option-decision latency from a partner mode switch to the ego crossing to the
  new-mode-appropriate response, reported against the mode_oracle arm's latency.
- **Causal:** E2-zeroed must collapse Δ_norm (≥ 50% **[F@C2]**); belief-swap (E3 machinery)
  co-reported.
- **Provisional criteria [F@C2]:** S1 (emergence): aris Δ_norm median ≥ **0.30** with
  episode-bootstrap CI > 0 on blind_v3; S2 (method-specificity): aris − best baseline ≥
  **0.15** with per-seed pairing ≥ 4/5; S3 (causal carriage): zeroed collapse ≥ 50%.
  Wording ladder pre-written at Sign-C2 (S1∧S2∧S3 → full claim; partial rows fixed in
  advance; ¬S1 → honest negative with the substrate certificate making it evidence-bearing).

### 3.3 Evidence-gated escalation (recorded once, not planned further)

If E-EMERGE fails S1 on a certified substrate at a sufficiency-verified budget, the
predictive-representation method change acquires the standing it currently lacks (the
certificate guarantees a supervised-learnable, transferable signal exists that TD did not
capture — the D1 branch-3 configuration). It would proceed only via a formal METHOD_LOCK
amendment prereg. Nothing beyond this sentence is pre-planned.

---

## 4. Execution protocol — what each link actually does

This section is the agent-facing execution contract. Treat Sections 1–3 as the design
specification and this section as the operational order. Each change set belongs to one
link only; do not mix Link-A substrate work, Link-B distribution/seal work, and Link-C
experiment/readout work in the same diff.

**Common rules.**

- The active next step after Sign-A1 is **Link A/A2 only**.
- Local-machine work is static maintenance unless the user explicitly relaxes the project
  execution constraint. Runnable tests, dataset generation, training, evaluation, and probes
  execute only under the remote contract in `CUSTOMER.md` after explicit authorization.
- `METHOD_LOCK` and method-layer code stay unchanged. Any diff touching selection logic,
  losses, factor-belief semantics, or factor-local Q is out of scope for this plan.
- Every executable round records commit hash, command, seed set, artifact paths, and readback
  counts. Scientific interpretation uses artifact readback, never config intent.
- A failed upstream gate stops the chain. Link B does not start without the Link-A
  certificate artifact; Link C does not start without the Link-B coverage artifact and
  blind-set seal.

### 4.1 Link A execution — instantiate and certify the phenomenon

**Entry condition:** Sign-A1 recorded by the user. No experiment-status gate is assumed passed
unless its artifact is inspected.

**Execution order:**

1. Implement the v3 partner substrate by composition: add `partner_modes.py`, add
   `latent_v3_dev` and fixed certification specs in `partner_pool.py`, and add the parametric
   `sample_mode_spec(...)` API needed later by Link B.
2. Add static review material and runnable test artifacts: family trigger tables, dwell-timer
   checks, reaction-latency checks, and one golden trace spec per disposition family. These
   files are created before execution; they are not run locally under the current project
   constraint.
3. Extend the D1 certificate machinery only at the diagnostic layer: mode-label readout by
   observation count, state-only mode classifier, and paired mode-informed vs mode-blind
   scripted probes. The extension must not create an ego-side oracle input.
4. Obtain code review on the Link-A diff before any certificate generation.
5. When remote execution is explicitly authorized, run certificate rounds on `latent_v3_dev`.
   Each round adjusts only the single parameter axis implicated by the failing band
   (`epsilon`, dwell, geometry decorrelation, or reaction sharpness), then logs the full
   before/after numbers.

**Required output artifact:** `artifacts/latent_v3_phenomenon_certificate.{md,json}` with
C-1..C-6 values, confidence intervals where applicable, oracle-source count, golden verdicts,
commit hash, commands, seeds, and the final frozen [F@A3] parameter values.

**Exit condition:** all C-1..C-6 certificate bands pass on the inspected artifact. Only then
may Link B begin.

### 4.2 Link B execution — make training exercise the phenomenon

**Entry condition:** the Link-A certificate artifact exists and passes C-1..C-6.

**Execution order:**

1. Implement `SamplerTrain` so every episode samples geometry, disposition, dynamics, and
   noise independently, with no persistent individual identity.
2. Freeze Link-B family quotas and minimum opportunity-count cells at Sign-B1 before any
   training data is used for model fitting.
3. Update CE/data plumbing to consume `SamplerTrain` and report coverage by disposition
   family and factor, not by named individual.
4. Define `dev-heldout` as declared held-out parameter regions and use it only for calibration.
5. Author `blind_v3` mechanism classes only after the training manifold is frozen. Seal the
   blind specs and relevant code hashes together before any model contact.
6. Implement `aggregate_emerge.py` once for opportunity-normalized metrics and reuse it for
   both dev-heldout and blind readouts.
7. When remote execution is explicitly authorized, generate the audited training/CE artifacts
   and read back actual episodes, transitions, opportunity cells, and family shares.

**Required output artifacts:** `artifacts/latent_v3_training_coverage.{md,json}` and
`artifacts/blind_v3_seal.md`. The coverage artifact records effective data budget from
artifacts, per-family/per-factor opportunity counts, quota compliance, CE-support audit
status, and the commit hash. The seal records blind mechanism classes, spec hashes, code
hashes, creation time, and a zero-contact declaration.

**Exit condition:** coverage green, effective budget recorded from artifacts, and `blind_v3`
sealed with zero model contact. Only then may Link C begin.

### 4.3 Link C execution — run the fair emergence test

**Entry condition:** Link-B coverage is green and the `blind_v3` seal is recorded.

**Execution order:**

1. Run the ARIS-only sufficiency recalibration probe on the certified substrate after remote
   execution is explicitly authorized. Read back actual episode counts, transitions, and
   late-window validation behavior from artifacts.
2. Freeze the full-run floor at the smallest recorded plateau point and write
   `artifacts/latent_v3_sufficiency_floor.md`. If no plateau is recorded by 8000 episodes,
   stop at substrate-difficulty adjudication instead of scaling silently.
3. At Sign-C2, freeze the exact E-EMERGE numerics, seed list, primary/secondary/causal
   metrics, and wording ladder before the blind single look.
4. Run the preregistered arms unchanged: `aris_bellman`, `base_only`, `global_gru`,
   `flat_factor`, plus `mode_oracle` as an upper-reference arm only. Use ITT accounting:
   launched seeds remain in the readout.
5. Evaluate dev-heldout for calibration and `blind_v3` once for the primary readout. Do not
   tune from the blind result.
6. Aggregate opportunity-normalized `Delta_norm`, switch-tracking latency, E2-zeroed collapse,
   and belief-swap results; then route the readout through cross-model review and human
   Type-B adjudication before any claim status changes.

**Required output artifacts:** `artifacts/latent_v3_sufficiency_floor.md`,
`artifacts/e_emerge_prereg.md`, `artifacts/e_emerge_readout.{md,json}`, and an
`EXPERIMENT_LOG.md` entry if a real result is produced. The readout must include per-seed
rows, bootstrap intervals, blind single-look declaration, zeroed/swap causal checks, and the
review/adjudication status.

**Exit condition:** E-EMERGE has a recorded Type-B adjudication: supported, partially
supported under the preregistered wording ladder, or negative. A negative result on a
certified substrate is evidence-bearing and is the only point at which a predictive-
representation method amendment becomes eligible for a separate prereg.

## 5. Phases, sign points, cost

| Phase | Content | Gate | Est. wall time |
|---|---|---|---|
| A1 | this plan signed | user | — |
| A2 | partner_modes.py + registry + tests + goldens | codex APPROVE | 0.5–1 d |
| A3 | certificate rounds until C-1..C-6 pass | **Phenomenon-Existence Certificate** | 2–4 rounds × ~1.5 h |
| B1 | SamplerTrain + quotas + audits + CE update + blind_v3 sealed | coverage green + seals | 0.5–1 d |
| C1 | sufficiency recalibration | plateau recorded | 0.5–1 d |
| C2 | E-EMERGE prereg (numerics frozen) | user Type-B | — |
| C3 | E-EMERGE run + single-look readout + codex cross-review + adjudication | branch table | 1–2 d |

Total ≈ 1 week wall clock including review cycles; compute well inside the project's
2,200–4,500 GPU-hour envelope (the heavy lever remains the 18× eval speedup + thread caps).

---

## 6. Risks and kill-criteria

| Risk | Mitigation / kill |
|---|---|
| Dispositions too hidden (inference impossible) | C-4 identifiability band; iterate ε/dwell; kill a family that cannot pass C-4 within 4 rounds |
| Dispositions value-irrelevant (predictable but useless) | C-5 VOI band via paired scripted probes |
| Geometry re-leaks disposition | C-3 band + geometry⊥disposition sampling; state-only mode classifier ≤ 0.60 |
| Reaction functions destabilize scripted probes | certificate ego-mixture already varies ego behavior; probe scripts assert termination |
| Switch tracking unmeasurable (too few switches) | horizon 60 + dwell ≤ 25 guarantees switch exposure; readback count gate |
| blind_v3 accidentally overlaps manifold | mechanism-class disjointness declaration + hash seal before training |
| Sufficiency floor explodes (>8000 eps) | plateau probe caps spend; if no plateau by 8000, stop and adjudicate substrate difficulty before scaling (data-discipline §6.3) |
| Metric confound recurrence | opportunity normalization implemented once in aggregate_emerge.py; certificate machinery reused as feasibility oracle |
| Scope creep into the method | METHOD_LOCK untouched; fidelity gate mechanical; any method edit = separate prereg |

---

## 7. Relation to existing locks and disciplines

- **METHOD_LOCK / G2:** unchanged. This plan is substrate/evaluation-layer only.
- **G0.1 governance:** v3 inherits the "scripted = mechanism-diagnostic substrate" framing;
  FCP/MEP population remains the later benchmark-evidence track (unchanged decision).
- **Substrate-must-not-solve-away discipline:** inverted and operationalized — the substrate
  must now *instantiate* the phenomenon, and the D1 instrument is the standing preservation
  check (certificate re-run required after any v3 partner change).
- **Data-sufficiency discipline (§6):** floors recalibrated per substrate; all counts from
  artifacts.
- **Blind-set discipline (sec18.14):** seal procedure, single look, contact audit — reused
  verbatim for blind_v3.
- **D1 prereg lineage:** `D1_RESPONSE_PREDICTABILITY_PREREG_DRAFT.md` §9 instrument is the
  certificate engine; its judged thresholds become Link-A acceptance bands (direction
  inverted: what D1 required to *pass into* branch 3, Link A requires as *entry condition*).
