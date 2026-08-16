# CETR-ZSC status dashboard

Status: implementation reset after the V6 DELTA retirement.

| Item | Status |
|---|---|
| Active method identity | implemented in [`src/cetr_zsc/config.py`](../../src/cetr_zsc/config.py): `constrained_episodic_tail_robust_zsc_v1` |
| Active configuration contract | implemented; registered values are owned by [`src/cetr_zsc/config.py`](../../src/cetr_zsc/config.py) (`version: 5`) |
| CETR package boundary | in progress |
| CETR model, risk, losses, runner, partners, and training modules | in progress |
| Overcooked CETR CLI and application integration | in progress |
| Reference-SP measurement and derived `tau_SP` artifact | in progress |
| Development-support and confirmatory lineage-disjoint panels | in progress |
| Formal training | not started |
| Formal held-out external evaluation | not started |
| CETR performance claim | none |
| CETR evidence | none |
| CETR training results | none — **CETR 尚无训练结果** |
| V6 DELTA / DEPI implementation | retired and removed from the active tree; prior revisions remain only in git history |

The active deployment contract is a single partner-agnostic recurrent actor whose
runtime reads local observation/history only. Parent groups, tail weights,
cross-fitting folds, and the SP dual are training-only. The primary confirmatory
object is the lineage-disjoint held-out external-partner panel; the population
matrix is supplementary.

No compilation, mechanical check, development artifact, or historical V6 result
is a CETR performance claim. Formal runs have not begun, and there is currently
no CETR return evidence to summarize.
