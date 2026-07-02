# ARIS-Bellman Fidelity Gate

**Overall: STATIC-REPAIR PASS / REMOTE PENDING**. This gate is Type-A implementation fidelity only. It does **not** validate or refute the scientific ZSC claim; Type-B human adjudication and remote reruns are still required.

| Check | Invariant | Verdict | Evidence |
|---|---|---|---|
| I1 | no G-TVOI/MI/probe selector in deploy path | PASS | Deploy/select/train path remains Bellman argmax; diagnostics remain post-hoc. |
| I2 | single TD loss, no auxiliary losses | PASS | `src/aris_bellman/td.py` remains the TD loss source; no new auxiliary factor-label loss was added. |
| I3 | action selection is pure Bellman argmax | PASS | `_select_option` still selects from Q values; behavior inferencer is evidence, not an action selector. |
| I4 | CE is preprocessing, not in training loop | PASS | CE estimation remains in `ce_sampler.py` / `run_ce_pipeline.py`; train loop consumes graph artifacts. |
| I5 | reward-scale single-source | PASS-STATIC | CE pipeline now records config-derived gamma/horizon/min_weight/objective metadata; remote artifact equality still pending. |
| I6 | articulation-point bottlenecks, not degree<=2 | PASS | Existing Tarjan articulation-point layout parser unchanged. |
| I7 | preflight hard gate + acceptance, no smoke bypass | PASS-STATIC | Preflight fallback resolves configured partner subset and sparse-credit/terminal-progress semantics; all-partner fallback requires an explicit no-split smoke flag. |
| I8 | no factor-accuracy as a primary metric | PASS | Headline metrics are return/ego-owned completion/reference-gap, not factor-label accuracy. |
| I9 | factor deletion removes latent+route+relevance | PASS | Graph builder structural deletion semantics preserved. |
| I10 | P1 oracle-free main evidence | PASS-STATIC | `option_executor.py` strips true partner labels; `event_extractor.py` records behavior source; `evidence_router.py` reports behavior_inferred_v1 and rejects oracle-like sources in formal eval. |
| I11 | P4 persistent belief or explicit finite-window claim | PASS-STATIC | `EvidenceBuffer` persists belief hidden state; `FactorLocalBeliefModel` accepts masks and hidden state; transitions carry hidden snapshots. |
| I12 | P3 CE support sidecar | PASS-STATIC | `ce_sampler.py` emits `weight_sum`, `estimable_mask`, `skipped_mask`, `measured_zero_mask`, `min_weight`, gamma/horizon/objective fields. |
| I13 | P5 no true terminal-policy conditioning in main path | PASS-STATIC | Main configs set `oracle_role_conditioned_ablation:false`; training rejects role-conditioned reward/exploration/replay unless explicitly ablation-only; train/eval/CE strip `partner_terminal_policy` from reward params. |
| I14 | S17 no global diagnostic skip | PASS-STATIC | `_validate_eval_integrity` hard-fails forced noop, wrong evidence policy, observed distributions, missing evidence, and oracle source regardless of `allow_diag_skip`; the flag only applies to optional diagnostics. |
| I15 | S20 actor-specific metrics | PASS-STATIC | Eval records ego-owned completion, team completion, partner delivery, wrong delivery, and actor-local delivery counts separately. |
| I16 | NEW-2 checkpoint eligibility before publication | PASS-STATIC | Training saves/publishes `checkpoint.pt` only when greedy validation satisfies ego-owned delivery eligibility when required. |
| I17 | S6/S23 explicit split provenance | PASS-STATIC | Train and CE require explicit `training.train_partners` for split claims and record selected partner subset. |
| I18 | decisive rerun archive and human Type-B checkpoint | PENDING | Remote experiments and human decisions are deliberately not run in this static patch. |
