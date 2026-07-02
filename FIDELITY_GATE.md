# ARIS-Bellman Fidelity Gate

**Overall: ✅ PASS** (GREEN)

Register: method-fidelity (Type-A). Does **not** judge claim support or scientific merit — that is the cross-model jury + human.

| Check | Invariant | Verdict | Evidence |
|---|---|---|---|
| I1 | no G-TVOI/MI/probe selector in deploy path | ✅ PASS | deploy/select/train path free of selector symbols (Δ_info/MI stay post-hoc in diagnostics.py) |
| I2 | single TD loss, no auxiliary losses | ✅ PASS | src/aris_bellman/td.py returns one F.mse_loss; experiments/overcooked_v2/train_aris.py backprops aris_td_loss with no aux terms |
| I3 | action selection is pure Bellman argmax | ✅ PASS | experiments/overcooked_v2/train_aris.py:_select_option selects via argmax with no info-gain/selector branch |
| I4 | CE is preprocessing, not in training loop | ✅ PASS | experiments/overcooked_v2/train_aris.py does not call CE estimation; CE lives in ce_sampler.py (preprocessing) |
| I5 | reward-scale single-source (structural) | ✅ PASS | cost/shaped coefs flow from training.* into CE, preflight, eval, and td target (numeric equivalence is semantic → out of scope, jury/human) |
| I6 | articulation-point bottlenecks, not degree<=2 | ✅ PASS | experiments/overcooked_v2/layout_parser.py uses Tarjan articulation points + region-size filter; no degree<=2 heuristic |
| I7 | preflight hard gate + acceptance, no smoke bypass | ✅ PASS | experiments/overcooked_v2/train_aris.py:_enforce_preflight_gate requires an accepted report; no smoke bypass |
| I8 | no factor-accuracy as a primary metric | ✅ PASS | evaluation is return / reference-gap grounded; no factor-label-accuracy metric |
| I9 | factor deletion removes latent+route+relevance | ✅ PASS | experiments/overcooked_v2/graph_builder.py:make_graph_spec derives relevance+route_map+mode_mask from `factors`; deleting a factor structurally removes all three |
| I10 | P1: main evidence path oracle-free (behavior-inferred partner option) | ✅ PASS | partner act() emits no true option label; executor strips the raw partner action before extract_event and re-annotates from the behavior inferencer |
| I11 | P4: belief hidden state persists across option decisions | ✅ PASS | EvidenceBuffer carries a persistent hidden; transitions store window-base hidden snapshots; belief decodes from hidden (episode-scoped memory) |
| I12 | P3: CE artifacts expose per-pair support (skipped != measured zero) | ✅ PASS | estimator emits weight_sum + estimable/skipped/measured-zero masks; pipeline reads graph.ce_min_weight from config |
| I13 | P5: no true terminal-policy conditioning in the main method path | ✅ PASS | role-conditioned reward/exploration/replay raise without the explicit ablation flag; option selection carries no terminal-policy argument |
| I14 | S17: allow_diag_skip cannot bypass hard eval integrity checks | ✅ PASS | forced-noop / evidence-policy / observed-dist / missing-evidence / oracle-source checks run regardless of --allow_diag_skip (flag scopes diagnostics only) |
| I15 | S20: headline completion is ego-owned, actor-split metrics recorded | ✅ PASS | eval headline = ego_correct_completion_rate keyed on ego-sole deliveries; greedy validation reports team vs ego-sole rates under distinct names |
| I16 | NEW-2: checkpoint eligibility gates before publication | ✅ PASS | deploy eligibility is checked before best-selection/save; stale deployable checkpoints are unlinked when no eligible checkpoint is selected |
| I17 | S23: explicit train-partner split required for split claims | ✅ PASS | unset train_partners hard-errors when a held-out split is declared; CE pipeline filters to train partners; all-partner runs require the explicit no-split flag |
