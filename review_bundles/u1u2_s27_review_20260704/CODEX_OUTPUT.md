## [A] verdict: BLOCK

Findings:

1. BLOCK - `support_mix` is not actually config-wired on the formal CE collection path. `make_behavior_option_inferencer()` only reads `support_mix` when a config is passed (`experiments/overcooked_v2/option_inferencer.py:239`, `experiments/overcooked_v2/option_inferencer.py:252`), but the CE collectors instantiate it with no config (`experiments/overcooked_v2/ce_sampler.py:115`, `experiments/overcooked_v2/ce_sampler.py:294`, `experiments/overcooked_v2/ce_sampler.py:315`, `experiments/overcooked_v2/ce_sampler.py:335`, `experiments/overcooked_v2/ce_sampler.py:504`), and `run_ce_pipeline.py` passes no evidence config into `collect_option_replay` (`experiments/overcooked_v2/scripts/run_ce_pipeline.py:224`). Impact: the default post-fix behavior is active, but the claimed `mix=0` ablation and any non-default `support_mix`/`temperature` CE provenance are false for CE replay generation. This can silently mismatch CE partner weights against train/eval evidence. Required change: add a config argument to sequential and batched CE collection, pass it to `make_behavior_option_inferencer(..., require_inferred=True)`, and record `mode`, `support_mix`, `temperature`, and classifier/heuristic source in `replay_metadata` and `ce_refined.meta.json`.

2. NIT - the update uses two validity sources in one step. The support floor uses `valid_options(prev_state)` (`experiments/overcooked_v2/option_inferencer.py:101`), while likelihood validity prefers `is_valid_for_state` (`experiments/overcooked_v2/option_inferencer.py:118`). Today this is semantically identical because `is_valid_for_state` just indexes `valid_options` (`experiments/overcooked_v2/options.py:77`), so the extra call is acceptable for current option counts. To avoid future divergence, make one validity mask the source of truth or assert equality when a custom library supplies both.

Question answers:

- Mix bias into wrong options: with `support_mix=0.05`, `temperature=1.0`, and the current likelihood floor, the prior injection is small enough that repeated matching primitive evidence should dominate. The observed post-fix terminal mass supports that. The risk is not the default; it is high-temperature or action-aliased states where many valid options predict the same primitive/progress, causing soft partner distributions to smear CE mass. Log partner-support histograms and run the `support_mix=0` ablation after config wiring is fixed.
- Near-zero normalize: at default temperature, valid OCV2 options always include `noop` (`experiments/overcooked_v2/options.py:379`), and valid likelihoods get `1e-4` (`experiments/overcooked_v2/option_inferencer.py:146`), so the final normalize should not see an all-zero vector. Edge case: very small or non-positive `temperature` is clamped to `1e-6` (`experiments/overcooked_v2/option_inferencer.py:149`), which can underflow all likelihood powers and trigger uniform fallback in `_normalize` (`experiments/overcooked_v2/option_inferencer.py:213`). If temperature remains sweepable, validate a sane lower bound or move the update to log space.
- Temperature interaction: support injection happens before the temperature exponent (`experiments/overcooked_v2/option_inferencer.py:111`, `experiments/overcooked_v2/option_inferencer.py:149`). Low temperature sharpens evidence and can underflow; high temperature flattens likelihood and lets the uniform floor matter more. Default `1.0` is fine; non-defaults need metadata and sensitivity runs.
- Per-step `valid_options()` cost/semantics: acceptable today because the option set is small and `is_valid_for_state` is a wrapper. It is still worth caching a single validity mask once per update to avoid O(N^2) calls if expected-cost checks grow.

## [B] verdict per-question + overall GO-WITH-CHANGES

Q1. Leak path mixing top-up rows into `CE_passive`: CHANGE REQUIRED. Current estimator entry points accept an undifferentiated row list (`experiments/overcooked_v2/ce_sampler.py:668`) and `run_ce_pipeline.py` estimates/refines directly from `rows` (`experiments/overcooked_v2/scripts/run_ce_pipeline.py:293`, `experiments/overcooked_v2/scripts/run_ce_pipeline.py:315`). `load_replay_npz` also returns all rows without policy partitioning (`experiments/overcooked_v2/ce_sampler.py:922`). Required spec edits: in U1 section 2.1/2.2, define `partition_replay_rows(rows)` and require `CE_passive` to use only `collection_policy=="uniform_random"` while `CE_int` uses only `interventional_topup:*`; in section 4 add a synthetic mixed-replay test that fails if any row crosses partitions. Old rows must default to `uniform_random` in `_row_from_json_dict` (`experiments/overcooked_v2/ce_sampler.py:1199`).

Q2. FSM base-policy occupancy bias: GO-WITH-CHANGES. The bias does not invalidate `CE_int` if it is declared as an interventional/top-up estimand, but it does invalidate comparing its magnitudes as if they came from the passive training distribution. Required spec edits: in U1 section 2.1/2.2 metadata, record `base_policy`, `target_option`, `bias_prob`, valid-hit count, fallback count, partner set, support objective, and start-state policy. In section 5, report passive and interventional readouts in separate columns.

Q3. Union semantics and A1: CHANGE REQUIRED. "Union only widens" is false under `max_factors` and diversity caps. The current selector reserves required coverage, then fills a fixed budget (`experiments/overcooked_v2/graph_builder.py:143`, `experiments/overcooked_v2/graph_builder.py:230`, `experiments/overcooked_v2/graph_builder.py:273`); a large interventional candidate pool can consume fill slots or caps and crowd out passive candidates. Required spec edits: in U1 section 2.3, make candidate union source-aware before selection, preserve the existing mandatory coverage ordering, add source quotas or tie-breaking, and store both `passive_ws` and `interventional_ws` even when one source wins the score. Do not flatten `CE_passive` and `CE_int` into one score matrix.

Q4. P1/P5 boundary for ego-side forcing: GO-WITH-CHANGES. Ego forcing is inside the controller's own action space, so it is not a partner-oracle leak by itself. The holes are the FSM and targeted-start implementation: they must not read partner name/protocol/role/terminal policy, and state surgery must not encode hidden partner variables. Required spec edits: in U1 section 2.1, state that the FSM may read only public state, layout, and `option_lib.valid_options`; partner behavior stays black-box and partner weights still come from the S27 behavior inferencer. In section 3/I10, add a gate check for no partner-protocol access in top-up code.

Two-estimator call: KEEP SEPARATE. Importance-weighting top-up rows into one passive estimand is the wrong scientific move because the motivating cells have zero or near-zero passive support, so positivity fails and variance is unbounded. Use `CE_passive` for passive-distribution claims and `CE_int` for interventional support/candidate discovery; optional overlap-only importance diagnostics can be added later but should not replace the two-estimator design.

Overall: GO-WITH-CHANGES. The scientific design is sound, but implementation must be row-partitioned and source-aware before it is safe.

Required U1 spec edits:

- U1 section 2.1: add `collection_policy` to `OptionReplayRow` and serialization compatibility; old rows default to `uniform_random`.
- U1 section 2.2: add explicit row partitioning before every estimator/refine call; save separate passive/int replay counts and audits.
- U1 section 2.3: define source-aware selection under `max_factors`/diversity caps; every factor gets `estimator_source` plus `support_certificate`.
- U1 section 4: add mixed-replay leak tests, selector budget/crowding tests, and metadata parity tests.

## [C] verdict per-question + overall GO-WITH-CHANGES

Q1. `hidden = log-posterior` reuse: GO-WITH-CHANGES. Shape reuse is plausible if `BayesModeFilter.hidden_dim=K_max`, because `EvidenceBuffer` only requires `[num_factors, hidden_dim]` (`src/aris_bellman/replay.py:92`). The semantic crack is that replay stores detached float32 hidden arrays, not model provenance (`src/aris_bellman/replay.py:97`, `src/aris_bellman/specs.py:165`). Float32 log-domain is acceptable only with `logsumexp`, finite floors, and explicit masked-mode handling. Window-base replay is equivalent to P4 truncation, not to full-episode Bayes recomputation under current parameters; state that explicitly in U2 section 2/5.

Q2. TD-through-recursion gradient/truncation: GO-WITH-CHANGES. Operationally it can match P4: stored hidden bases are detached (`experiments/overcooked_v2/train_aris.py:2946`), TD replays the visible window through `encode_history` with gradients (`experiments/overcooked_v2/train_aris.py:1798`, `experiments/overcooked_v2/train_aris.py:2135`), and the bootstrap side is under `torch.no_grad()` (`src/aris_bellman/td.py:40`). Do not claim "strictly isomorphic" mathematically; claim the same truncation boundary and optimizer path as P4, with the recurrence cell changed.

Q3. `lambda` mixing x `mode_mask`: CHANGE REQUIRED. The current `step_history` interface has no `mode_mask` argument (`src/aris_bellman/factor_belief.py:55`), and `_advance_persistent_belief` cannot pass one (`experiments/overcooked_v2/train_aris.py:2939`). If Bayes mixing is over `K_max`, padded modes receive mass and can persist in hidden even if `belief_from_hidden` masks them later. Required spec edits: in U2 section 2/3, either pass `mode_mask` through `step_history`/`encode_history` or store per-factor valid-mode masks inside `BayesModeFilter`; mix uniform over valid modes only, set invalid modes to `-inf` or a non-updating sentinel, and gate that masked-mode probability remains zero while valid-mode rows sum to one.

Q4. Checkpoint cross-loading: CHANGE REQUIRED. Eval reconstructs from checkpoint config and strict-loads the belief model (`experiments/overcooked_v2/evaluate_aris.py:266`, `experiments/overcooked_v2/evaluate_aris.py:280`), which will catch many mismatches, but current checkpoint payload has no top-level `belief_filter` provenance (`experiments/overcooked_v2/train_aris.py:3244`) and `_checkpoint_loads` only calls `torch.load` (`experiments/overcooked_v2/train_aris.py:3292`). Required spec edits: in U2 section 3/4, store and validate `belief_filter`, class name, `hidden_dim`, `max_modes`, `mode_forgetting`, and graph hash at checkpoint save/load. Define behavior for `random_policy` and `partner_id_q`: either `belief_filter="unused"` with no required belief state, or a strict saved-state contract.

Single TD loss honesty: YES, with wording. It is honest to claim a single TD objective if `g_theta` receives gradients only through `belief -> Q -> TD loss`, with no supervised mode labels, auxiliary likelihood loss, selector loss, or oracle mode labels. Phrase it as "single TD objective with an analytic differentiable Bayes-filter recurrence and learned likelihood"; do not imply a separate Bayesian evidence objective was optimized.

Overall: GO-WITH-CHANGES. The architecture is a credible C2 upgrade, but the spec must fix mask-aware recurrence and checkpoint provenance before implementation.

Required U2 spec edits:

- U2 section 2: add mask-aware Bayes recurrence and log-domain numerical rules.
- U2 section 3: update the interface contract if `mode_mask` must flow through `step_history`/`encode_history`; otherwise state how the module owns valid-mode masks.
- U2 section 4/I19: expand the gate to checkpoint provenance, masked-mode zero mass, valid-mode row sums, and train/eval filter consistency.
- U2 section 5: add tests for masked-mode leakage, stale window-base truncation equivalence, and GRU/Bayes checkpoint rejection.

## Cross-cutting risks (interactions among A/B/C, fidelity-gate impact I1-I17)

- S27 config/provenance must be fixed before U1. Otherwise `CE_passive`/`CE_int` support certificates can be built with an inferencer config different from train/eval, weakening I10/I12 and the proposed I18-CE.
- U1 interventional rows must stay preprocessing-only. Mixing them into train-time replay or online CE updates would violate I4 even if the estimator metadata is correct.
- U2 does not remove the need for S27-correct CE collection. The graph and U1 support certificates are still built from behavior-inferred partner option distributions unless the CE estimator itself is redesigned around mode evidence.
- `CE_int` and BayesModeFilter can both make rare terminal behavior more visible. That is good, but it raises a paper-risk: improvements must be attributed by arm (`passive` vs `interventional`, `gru` vs `bayes_mode`) or I1/I2/I8 claims become muddled.
- Existing I1-I17 gates are not enough for these changes. Add I18-CE for estimator-source separation and I19-mode for belief-filter provenance/mask normalization; do not treat current gate PASS as covering U1/U2.

---

## Post-review integration (2026-07-04)

Resolutions committed by Claude Opus 4.7 after codex review:

- [A] S27 provenance BLOCK — FIXED in commit c1e112b (evidence_config threaded
  through collect_option_replay / collect_option_replay_batched; inferencer
  provenance recorded in replay_metadata; 19/19 tests pass remote incl. new
  test_ce_inferencer_provenance).
- [A] Nit on dual validity sources — acknowledged, current wraparound is
  equivalent (options.py:77); no change.
- [B] U1 spec — all 4 required-change items integrated (row partitioning +
  interventional metadata block + source-aware selection + P1/P5 gate check);
  KEEP-SEPARATE two-estimator design retained per codex.
- [C] U2 spec — all 4 required-change items integrated (log-domain rules with
  logsumexp/clip, mode_mask through step_history/encode_history via option A,
  checkpoint provenance fields, single-TD-loss wording tightened; strictly-
  isomorphic claim removed and replaced with "same truncation boundary and
  optimizer path as P4, recurrence cell changed").
- Cross-cutting — I18-CE and I19-mode defined in U1/U2 specs §3-§4; will be
  landed as fidelity-gate tool checks (F12-style) when U1/U2 code lands.
