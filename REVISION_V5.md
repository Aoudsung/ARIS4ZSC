# DELTA-ZSC v5 — what changed and why

`METHOD_VERSION = delta_belief_conditioned_raw_return_pairwise_crn_v5`,
`CHECKPOINT_SCHEMA_VERSION = 5`. Every v4 checkpoint, optimizer state and
deployment bundle is deliberately incompatible.

This document records the revision that followed the four diagnostic
experiments on `runs/engineering/postfix_matched`. It is a description of the
code, not a claim about results: **no scientific evidence exists for v5 yet.**

## 1. What the diagnostics established

Measured with frozen parameters on `postfix_matched/seed0`:

| Measurement | Value |
|---|---|
| anchor pairs separated by ≥ 2 standard errors | **0** |
| median best-second margin vs median replica standard error | 0.0004 vs 0.0031 |
| anchor rows containing an exact tie | **100%** |
| fit↔evaluation agreement on the best action | 0.3750 |
| training NLL when both component residual branches are held at zero | 0.082 of 7.55 (**1.1%**) |
| outer updates on which the decision channel received gradient | **28 of 3656** (0.766%) |

Read together: the decision target was not estimable, the `K` component means
were not identified by the data, and the head that consumed them was trained on
0.77% of updates. Top-action agreement over six actions cannot distinguish
"the head is wrong" from "the measurement could not tell", which is why every
ordering metric sat at chance (0.1688 → 0.1042 against 0.1667).

## 2. The estimand changed

**Removed.** `decision_model.py` in full — the component-wise Gaussian return
mixture, its shared/residual decomposition, the trainable log-determinant, the
five-dimensional Helmert contrast covariance, and the
`decision_component_log_probability` objective.

**Added.** `belief_value.py` — a single belief-conditioned raw-return value in
dueling form. The advantage is centred under the acting policy, so the value
head carries the state's level and the advantage carries only the contrast the
mirror step consumes. An ensemble of advantage heads reports disagreement;
there is no trainable variance, because a scale the objective can shrink for
free measures nothing.

Component-conditional values are now a *read-out* — the same critic at each
one-hot posterior — not a separately parameterised head.

## 3. The target changed

`contrast.py` — the anchor target is the same-replica action **difference**.
Replica `r` of action `a` and replica `r` of action `b` share one CRN draw, so
the partner, the environment noise and the continuation draw cancel before
averaging, and the standard error of that difference is an honest measure of
whether the pair is resolvable at all.

Weighting is `1 / (σ²_ab + σ̄²)` with `σ̄²` the batch mean contrast variance.
Plain inverse-variance weighting is wrong here: under CRN two actions whose
continuations re-merge give bit-identical returns and therefore *zero* sample
variance, which would hand "these two actions are exactly equal" a weight
1000× that of a genuinely resolvable pair. The pooled term bounds the weight
above and introduces no tunable constant.

Nothing is thresholded away. Unresolvable pairs still train "these two actions
are close" — the honest content of the measurement.

## 4. The measurement changed

Two-tier anchor sampling (`anchors.py`). Every candidate world gets a cheap
low-replica pilot pass; only the states with the highest pairwise SNR are
measured at the registered budget. Selection uses the pilot's own replicas and
the kept states are re-measured under fresh keys, so the contrasts that reach
the loss are not the ones that won the selection.

`anchor_buffer.py` retains recent anchor **worlds** — never returns. Every
continuation is re-run under current parameters, so the CRN estimand the
alternating update protects is unchanged; only the *search* for separable
states is reused.

`task_phase` is now 18 strata (`holding × partner_visible × pot_active`, plus
the two delivery outcomes) read from the ego's own legal frame, replacing a
four-way label derived from reward alone that could not tell an empty-handed
agent from one holding a finished dish.

## 5. The update ordering changed

```
rollout → response → successor → raw-Q TD → anchor contrast → PPO actor
```

The first three channels are parameter-disjoint, so one reverse pass yields
each of their gradients exactly; they differ only in which Adam state commits
them. The contrast is taken afterwards in its own pass, because it writes the
critic the TD step just moved.

Each channel has **its own Adam state**. An anchor batch reaches roughly one
update in 130; a shared state would let 129 zero gradients decay the
second-moment estimate before the one gradient carrying the ordering
information arrived, and that gradient would then be applied with an enormous
effective step.

The critic bootstraps from a Polyak target copy (`τ = 0.01`, a time constant of
about one anchor interval), so the critic the mirror reads does not chase its
own estimation noise.

## 6. Deployment changed

Deployment improves against `Q_ψ(x, b, ·)`, not `Σ_k b_k μ_k`. The mirror step
is now `robust_mirror_policy_logits`: it improves against a conservative lower
bound `sign(A)·max(|A| − β·dispersion, 0)` at one ensemble standard deviation.
The unmodified solver is invariant to positive rescaling of the advantage, so
it spent the full 0.04 KL budget on a contrast of any magnitude — including
contrasts no measurement supported.

Active VOI evaluates the critic at a *predicted landing state*
`χ(x_t, b_t, a_t, y)` (`successor_feature.py`) rather than at the current
state. Pricing a probe at the current state charges it two steps of delay and
credits it with none of the position those steps buy.

The model is trained on the observed `y`, but at VOI time the successor is
evaluated once per probe at the **expected** outcome, `E_y[Q(χ(y))] ≈ Q(χ(E[y]))`.
A landing state per enumerated outcome is a `[time, lane, probe, 66, component]`
tensor — 52 million critic rows and tens of gigabytes at the registered formal
shape, materialised by the final audit at every timestep. The posterior VOI
prices is still integrated over all sixty-six outcomes exactly; what is
approximated is only the landing state's dependence on the response.

## 7. The base network changed

DELTA's task encoder was a flat MLP over the 5×5×39 frame. The pinned Official
baselines use a six-layer CNN with a LayerNorm and a Flax `GRUCell`. So DELTA
and its own baseline differed in feature extractor as well as in method — and
no Official checkpoint could initialise it.

`base_policy.py` now implements the Official encoder layer for layer, and
`nn.py`'s GRU matches the Official cell exactly (separate input and hidden
projections; the reset gate applies to the whole hidden projection including
its bias, which the previous concatenated form could not express).

`official_initializer.py` transplants a trained Official SP checkpoint into
`base_params`. Every Official parameter maps onto exactly one DELTA parameter;
the only inputs with no counterpart — the instantaneous partner encoding and
the posterior — enter the actor and value trunks as additional rows
initialised to **zero**.

The weights are an exact copy; the *inputs* are not. Every variant except
`history_rnn` feeds the task encoder a partner-masked frame, because the
teammate may reach the policy only through the legal response channel, whereas
the Official checkpoint was trained on the full frame. The transplanted encoder
therefore runs slightly off its fitted distribution. It still transfers most of
its competence — shaped return 24.8 at the first update against the partner
mixture, against -0.7 from random initialisation — but calling it "numerically
identical to the Official policy", as an earlier draft of this document did, was
wrong.

## 8. Audit metrics

`final_decision_audit.json` now reports, all on evaluation replicas the loss
never sees:

- `anchor_oracle_repeatability` — fit↔evaluation agreement on the best action.
  This bounds every other ordering number in the file: when the two halves of
  the same anchor disagree, nothing measured against the fit half is skill.
- `contrast_resolvable_fraction`, `anchor_evaluation_exact_tie_fraction`
- `anchor_evaluation_sign_agreement` restricted to resolvable pairs
- `anchor_evaluation_kendall_tau_b` — replaces Spearman over
  `argsort(argsort(...))`, which was ranking exact ties by action index
- `anchor_best_action_regret` with its standard error
- `mirror_same_world_improvement`, `_win_rate`, `mirror_realised_kl` — the
  mirror policy and the base policy scored against the *same* measured worlds
- `successor_feature_identity_baseline` — the "nothing moves in two steps" null
  a successor model must beat to contribute anything VOI can use

## 9. Two defects found by measuring, not reasoning

- `_probe_action_index` read `output.active_voi`, which is identically zero in
  the training path (`compute_decision` is off there). The successor model
  would have trained entirely on probe action 0.
- `@lru_cache` on the 66-outcome encoding table cached a **traced** array
  belonging to whichever jit trace built it first, which then escaped into
  every later trace. The failure is call-order dependent, so a test suite that
  touches the function outside jit first hides it. The table is now built with
  NumPy, and a test builds it inside one jit and uses it inside another.

## 10. A pre-existing acceptance gate was measuring the wrong thing

`peak_device_memory_bytes` took the maximum over `peak_bytes_in_use`,
`peak_pool_bytes` and `bytes_in_use`. `peak_pool_bytes` is the arena XLA
preallocates — `XLA_PYTHON_CLIENT_MEM_FRACTION` times the device — and is
unrelated to what a run used. Measured on an L40, a formal-shaped deployment
step reported **34,116 MiB** by the pool and **1,038 MiB** by actual use.
Since 0.75 × 46,068 MiB sits just under the registered 40,000 MiB ceiling, the
CUDA acceptance gate passed on this hardware regardless of what the run did.
It now reads actual allocation only. The registered threshold is unchanged.

## 11. Status

72 tests pass on CPU locally and on the server. Implementation and CPU-side
acceptance are complete; **scientific evidence is zero.** No development
matrix, no formal runs, no H1/H2/H3, and therefore no SOTA claim.

One design decision is left open deliberately: the formal script initialises
from Official `sp_seed201`, which is disjoint from all three partner panels but
gives DELTA an SP checkpoint's training on top of its own budget. The H1
comparison against the `sp` baseline therefore carries a step-count confound.
It is recorded in `run_identity.json` and the resource ledger rather than
hidden, and it is one flag to drop.
