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
