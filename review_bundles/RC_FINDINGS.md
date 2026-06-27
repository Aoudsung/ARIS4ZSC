# RC Findings — Root-Cause Confirmation Results

Companion to `review_bundles/ROOT_CAUSE_CONFIRMATION_PLAN.md`. Records what RC-1 and RC-2
established. **No fix plan here** (next step, separate doc).

**Provenance.** Produced by diagnostic instrumentation on top of commit `c27911e`. The
instrumentation is *diagnostic-only and currently uncommitted*: `evaluate_aris.py` (+58,
two `getattr`-gated hooks in `_select_option`), `train_aris.py` (+18, `--save_all_checkpoints`),
and two driver scripts (`scripts/rc1_qaudit.py`, `scripts/rc2_reachability.py`). It does **not**
change the deploy/train selection path; **fidelity gate I1–I9 = GREEN** after the edits.

Artifacts (cxw2 → pulled local):
- `results/ocv2_rc1/rc1_qaudit.json` — per-checkpoint Q-decomposition (10 checkpoints, one run).
- `results/ocv2_rc1/rc1_train_metrics.json` — the run's training metrics (greedy validation + td).
- `results/ocv2_rc1/rc2_reachability.json` — scripted-pipeline reachability, 6 partners × 5 episodes.

Run: `aris_bellman / full_support / seed 0 / 5000 updates`, config `ocv2_step4.yaml`
(Huber + Double-Q + greedy checkpoint-selection, `advantage_norm=none`), CE `outputs/p1_verify`.

---

## RC-1 — Policy collapse to noop  ·  **CONFIRMED (decisive)**

**Root cause: deadly-triad Q-VALUE divergence in the base head — not a loss-scale problem
and not an advantage-sign problem.**

Per-checkpoint Q-decomposition at the **fixed deterministic initial state** (same state every
checkpoint; `Δ = Q(noop) − max_{valid opt≠noop} Q(opt)`; adv = Q − q_base):

| update | greedy val_ret | init Δ | argmax@init | q_base(noop) | q_base(best opt) | adv(best opt) | noop-win frac |
|--:|--:|--:|:--|--:|--:|--:|--:|
| 500 | 3.85 | −2.3 | fetch_ingredient | 15.7 | 11.2 | 6.8 | 0.00 |
| 1000 | 3.40 | −44 | fetch_ingredient | 209 | 182 | 70 | 0.80 |
| 1500 | 2.31 | −1.6k | fetch_ingredient | 8.0k | 6.8k | 2.8k | 0.00 |
| 2000 | 5.33 | −12.0k | fetch_ingredient | 182.9k | 137.8k | 57.1k | 0.70 |
| 2500 | 3.31 | −38.1k | fetch_ingredient | 1.16M | 870k | 323k | 0.75 |
| 3000 | 5.27 | −45.2k | fetch_ingredient | 4.44M | 3.28M | 1.21M | 0.65 |
| **3500** | **−0.40** | **+179.6k** | **noop** | **13.3M** | **9.66M** | **3.46M** | **1.00** |
| 4000 | −0.40 | +315k | noop | 33.2M | 24.0M | 8.85M | 1.00 |
| 4500 | −0.40 | +2.57M | noop | 71.4M | 49.9M | 18.9M | 1.00 |
| 5000 | −0.40 | +8.43M | noop | 135M | 91.9M | 35.0M | 1.00 |

### Mechanism (each step evidenced by the table)
1. **q_base diverges geometrically** — `q_base` grows ≈ ×10 per 500 updates, from 15.7 to
   **1.35×10⁸**. Returns in this task are O(10); 10⁸ is ~6 orders of magnitude too large.
   This is classic deadly-triad value divergence (off-policy bootstrapping + function
   approximation), and it runs from the very first checkpoints.
2. **noop's base diverges fastest.** `adv(noop) ≡ 0` (noop is relevant to no factor, so its
   value is purely the base head). `q_base(noop)` exceeds `q_base(best opt)` and the gap
   `q_base(noop) − q_base(opt)` grows with the divergence.
3. **The factor-advantage compensates early.** Through update 3000 the option's advantage
   `adv(opt)` (1.21M at 3000) exceeds the base gap (4.44M−3.28M = 1.16M), so productive
   options stay on top → argmax = fetch, policy still ~5.3.
4. **At 3500 the base gap overtakes the advantage.** 13.3M−9.66M = 3.64M > adv = 3.46M →
   `Q(noop) > Q(opt)` → argmax flips to noop → greedy return −0.40. From there the gap only
   widens (all-noop, noop-win = 1.00).

### What this resolves / refutes
- **Resolves the reviewer's puzzle** ("u1000 collapsed with TD loss only 0.97"). TD loss is a
  *residual*; when prediction and target blow up together the residual stays small. So
  `td_loss_last_window` looked fine (0.0006→…) until the very end (`5.6×10⁷`) while the **values**
  had already diverged by 6 orders of magnitude. **TD loss is the wrong diagnostic; Q-value
  magnitude is the right one.**
- **Explains why Huber + Double-Q did not help.** Both act on the *loss/target estimate*, not on
  value magnitude; the loss residual was already small, so they had nothing to bite on.
- **Refutes the advantage-sign hypothesis** (plan H1a-ii). The factor-advantage is *positive and
  helps options*; `advantage_norm` would not address a diverging base head.
- **Checkpoint-selection (P0-step2) is confirmed to only mask this** — it keeps the pre-divergence
  update-2000 checkpoint; the training itself diverges regardless of length.

### Status
Root cause **confirmed**. The disease is unbounded value growth under bootstrapping; the fix must
bound value/target magnitude (reward/return scaling, target/value clipping, lower lr / larger
target interval, base-head regularization) — **not** loss shape or advantage normalization. *(Fix
design deferred to the separate plan.)*

---

## RC-2 — completion = 0  ·  **PARTIALLY CONFIRMED + a confound I must disclose**

### What is confirmed
Across **all 6 partners**, under both the learned greedy policy and a scripted kind-priority
pipeline, **`serve_soup` and `plate_soup` never become valid (0 attempts)** → no soup is ever
plated/served → `completion = 0`, `ego_delivery = 0`. The pot never reaches cooked-soup-ready.
Early stages work (`fetch_ingredient` 10/10). So `completion = 0` is **accurate** (not a counting
bug) and the return metric only ever reflects the fetch/deliver substages.

Scripted reachability (`results/ocv2_rc1/rc2_reachability.json`):

| partner | deliver att/ok/timeout | serve att | plate att | pick_plate att/ok/timeout | completion |
|:--|:--|--:|--:|:--|--:|
| ingredient-near | 165 / 5 / 160 | 0 | 0 | — | 0 |
| ingredient-far | 165 / 5 / 160 | 0 | 0 | — | 0 |
| dish-server | 10 / 10 / 0 | 0 | 0 | 155 / **0** / 155 | 0 |
| server-left | 10 / 10 / 0 | 0 | 0 | (similar) | 0 |
| bottleneck-yield | 10 / 10 / 0 | 0 | 0 | (similar) | 0 |
| flexible-balanced | 10 / 10 / 0 | 0 | 0 | (similar) | 0 |

### Observed (independent of the confound)
**Poor intra-option execution reliability** at `max_option_steps = 6`:
- `deliver_ingredient_to_pot` as low as **5/165 (3%)**, the rest `max_steps` timeouts (2 partners).
- `pick_plate` **0/155 (0%)** success, all timeouts (4 partners).
Both consistent with weak option navigation — `force_path_planning: false` and a 6-step option
budget — i.e. the macro-options often cannot reach their target tile within the budget.

### The confound (honest)
My RC-2 scripted policy is a **fixed kind-priority** list that ranks `pick_plate` *above*
pot-filling (`deliver`/`fetch`). For the 4 partners where deliver succeeds 100%, the agent grabs
a plate prematurely and then **loops on `pick_plate` (155 timeouts)** instead of continuing to fill
the pot — so the pot never reaches 3 ingredients for a reason caused by *my probe*, not
necessarily the environment. A fixed priority **cannot** drive this state-dependent pipeline
(fill pot → cook → pick plate → plate → serve).

### Status
**"Completion is structurally unreachable" is NOT established** — I am walking back the stronger
phrasing from my earlier message. What is established:
1. completion is not achieved by **any** learned or (this) scripted policy, across all partners;
2. several macro-options have very low execution reliability at the current option budget /
   navigation setting.
Whether the full pipeline is reachable **at all** needs a **state-aware** reachability probe
(fill-pot-before-plate, and/or an option-navigation fix) — call it **RC-2b**, an open item.

---

## Cross-cutting

- **Method fidelity preserved.** Both `_select_option` hooks are `getattr`-gated (default off); the
  deploy/train selection path is byte-identical; fidelity gate I1–I9 GREEN.
- **RC-3 (graph load-bearing) and RC-4 (Δ_info vs MI) remain gated** — they require a setup that
  is both non-divergent (RC-1 fixed) and completion-reachable (RC-2b settled). Running them now
  would measure a diverged / non-completing policy.
- **Bottom line for claims/gates:** the return-matrix currently compares ingredient-farming under
  a policy whose training diverges; it does not yet test any completion-based claim.

## Reproduce
```
# one instrumented run, all-checkpoint save
python experiments/overcooked_v2/train_aris.py --config experiments/overcooked_v2/configs/ocv2_step4.yaml \
  --graph_variant full_support --method aris_bellman --seed 0 --updates 5000 \
  --save_all_checkpoints --output_dir results/ocv2_rc1
# RC-1 Q-decomposition trajectory
python experiments/overcooked_v2/scripts/rc1_qaudit.py \
  --run-dir results/ocv2_rc1/cramped_room/aris_bellman/full_support/seed0 --episodes 3 \
  --out results/ocv2_rc1/rc1_qaudit.json
# RC-2 scripted reachability
python experiments/overcooked_v2/scripts/rc2_reachability.py \
  --checkpoint results/ocv2_rc1/cramped_room/aris_bellman/full_support/seed0 --episodes 5 \
  --out results/ocv2_rc1/rc2_reachability.json
```
