# ARCHITECTURE — CETR-ZSC equation-to-code map

`authoritative: true`

The active source boundary is the CETR package and its Overcooked integration.
The method identity and registered values live in
[`src/cetr_zsc/config.py`](../src/cetr_zsc/config.py). The paths below are the
only active equation-to-code targets in this contract; V6 DELTA paths are
retired and are not wrappers or fallback implementations.

## 1. Equation and object map

| Equation or object | Active implementation |
|---|---|
| method identity, config schema, method constants and run identity | `src/cetr_zsc/config.py` |
| Official-isomorphic CNN-to-GRU actor and scalar value baseline | `src/cetr_zsc/model.py` |
| parent nominal distribution `p0`, lower-half set `Q`, and closed-form `q*` | `src/cetr_zsc/risk.py` |
| complete raw return-to-go, fixed advantages, shared normalization, PPO and dual terms | `src/cetr_zsc/losses.py`, `src/cetr_zsc/training.py` |
| deterministic fold×role lane coverage and whole-episode collection | `src/cetr_zsc/runner.py` |
| parent manifests, mechanism strata, stage grouping, and lineage-disjoint panels | `src/cetr_zsc/partners.py`, `src/cetr_zsc/manifest.py` |
| cross-fitting, `prepare_episode_batch`, primal-dual transaction ordering, and train-state updates | `src/cetr_zsc/training.py` |
| active CLI and command dispatch | `experiments/overcooked_v2/cetr_zsc.py` |
| training orchestration, τ artifact binding, preflight seed mapping, and checkpoint scheduling | `experiments/overcooked_v2/training_app.py` |
| raw evaluation rows, external mean/tail and self-play measurements | `experiments/overcooked_v2/evaluation_app.py` |
| descriptive evaluation summaries and the unique GO/NO-GO/INCONCLUSIVE claim entry | `experiments/overcooked_v2/claim_app.py` |
| seed-matched Official-SP measurement and versioned `tau_SP` artifact | `experiments/overcooked_v2/reference_sp_app.py` |
| actor-only deployment bundle and provenance records for `o_t,h_t -> pi_theta -> a_t` runtime | `experiments/overcooked_v2/deployment.py` |

## 2. Objective wiring

The risk module owns the finite parent-level calculation

\[
\widehat\rho_{\mathrm{ext}}
 =\min_{q\in\mathcal Q_{1/2}(p_0)}
   \sum_g q_g\widehat J_g,
\qquad
\mathcal Q_{1/2}(p_0)=
 \{q\in\Delta:0\le q_g\le2p_{0,g}\}.
\]

`src/cetr_zsc/partners.py` supplies parent identity for training and evaluation
bookkeeping only. `src/cetr_zsc/risk.py` groups checkpoint stages under their
parent and returns weights; it never adds parent metadata to the actor input.

The losses module combines the risk-weighted external policy-gradient estimator
with the bilateral self-play contribution and the scalar-baseline variance
reduction term:

\[
\mathcal L(\theta,\lambda)=
\rho_{\mathrm{ext}}(\theta)+
\lambda(J_{\mathrm{SP}}(\theta)-\tau_{\mathrm{SP}}).
\]

`src/cetr_zsc/training.py` enforces the order: collect completed episodes,
compute closed-form cross-fitted weights from raw episodic returns, prepare the
complete `EpisodeBatch`, perform the single shared advantage normalization before
minibatch slicing, perform one primal PPO update, then perform one projected dual
update. It does not merge the dual state into deployment parameters.

## 3. Model and deployment state

`src/cetr_zsc/model.py` contains one partner-agnostic recurrent actor with the
Official CNN-to-GRU structure and one scalar value baseline. Both self-play
sides use the same actor parameters but independent recurrent carries. The value
baseline is used only during training and is not an action correction.

`experiments/overcooked_v2/deployment.py` exports only the actor and its legal
recurrent carry. Its runtime graph is

\[
(o_t,h_t)\longrightarrow\pi_\theta\longrightarrow a_t.
\]

It does not consume parent groups, `p0`, `q`, cross-fitting folds, `lambda`,
reference metadata, partner IDs, hidden state, future responses, or
counterfactual returns. The deployment bundle is version 7 and contains only
the actor parameter subtree required by this graph. The reference artifact and
training manifest are moved into same-directory provenance records, including
`provenance.json`; they are audit bindings, not runtime state.

## 4. Training data flow

`src/cetr_zsc/runner.py` collects complete episodes. It records raw rewards,
terminal boundaries, actor log probabilities, scalar-baseline outputs, and the
independent self-play stream needed for the bilateral self-composition gradient.
There is no cross-episode continuation target.

`src/cetr_zsc/runner.py` assigns every registered external parent on every update
without random parent sampling. Each parent receives the fold×role lanes `A0`,
`A1`, `B0`, and `B1`; its checkpoint stage rotates by the update index and lane
slot. `src/cetr_zsc/training.py` assigns the two folds and computes weights from
completed parent returns: fold A weights only fold B and fold B weights only fold
A. There is no observed-subset re-normalization.

`prepare_episode_batch` then applies one shared advantage normalization to the
complete batch before the minibatch schedule is sliced. The training transaction
is intentionally narrow:

```text
collect complete episodes with deterministic full parent coverage
    -> estimate parent returns from raw episodic returns
    -> cross-fit lower-half q weights
    -> prepare complete batch and normalize once
    -> one primal PPO update
    -> one dual update
```

No posterior, latent response branch, active probe path, continuation anchor,
mirror policy, group-wise maximization loss, or snapshot-only self-play path is
part of the active architecture. Those V6 mechanisms were retired rather than
hidden behind compatibility switches.

## 5. Partner and panel flow

`src/cetr_zsc/partners.py` defines the training-support mechanism mixture and
parent-level lineage records. The development-support panel and confirmatory
panel are disjoint in both parent and co-training lineage. A checkpoint stage
belongs to its parent and cannot be counted as an independent parent.

`experiments/overcooked_v2/reference_sp_app.py` measures the seed-matched
Official-SP reference and writes the version-2 artifact from which `tau_SP` is
derived. The artifact records a resolved absolute `source_checkpoint`; training
requires exact textual equality with the resolved `--sp-initializer` path. The
engineering preflight uses seed `-1`, mapped to the seed-0 initializer and
reference artifact. The reference artifact is an input to training identity and
audit, never to the runtime action path.

## 6. Evaluation and claim flow

`experiments/overcooked_v2/evaluation_app.py` consumes the common policy and
partner manifests and reports raw complete-episode returns, external mean,
parent-level lower-half CVaR, and self-play minus the reference-derived target.
It preserves ego/partner role and lineage units. `summarize-evaluations` is
descriptive only.

`experiments/overcooked_v2/claim_app.py` is the unique GO/NO-GO/INCONCLUSIVE
entry point. External contrasts use crossed bootstrap resampling of the ego-run
and parent-lineage axes, independently within each replicate; methods are
aligned by run/seed index. SP uses paired per-seed differences
`d_s = J_SP^(s) - tau_s` and bootstrap resamples seeds. The baseline method set
must be exactly `{\"fcp\"}`. It does not select a favorable checkpoint or replace
a missing seed. The `10×10` population matrix is supplementary and does not
replace the lineage-disjoint confirmatory estimand.

## 7. Artifact and historical boundary

`src/cetr_zsc/config.py` is the sole identity authority for the active
configuration. `training_app.py`, `evaluation_app.py`, `reference_sp_app.py`,
`deployment.py`, and `claim_app.py` must bind resolved config and lineage fields
by name, not by copied constants.

The former V6/DEPI implementation is removed from the active tree and retained
only in git history. No active import, fallback, or deployment bundle may load
its checkpoints or reinterpret its schemas as CETR artifacts.
