# CETR-ZSC status dashboard

Status: implementation reset after the V6 DELTA retirement.

| Item | Status |
|---|---|
| Active method identity | implemented in [`src/cetr_zsc/config.py`](../../src/cetr_zsc/config.py): `constrained_episodic_tail_robust_zsc_v1` |
| Active configuration contract | implemented; registered values are owned by [`src/cetr_zsc/config.py`](../../src/cetr_zsc/config.py) (`version: 5`) |
| CETR package boundary | implemented |
| CETR model, risk, losses, runner, partners, and training modules | implemented; cetr-ci green on 2026-08-16 (37/37 isolated CPU tests, compile gate, config validation, CLI smoke, namespace gate) |
| Overcooked CETR CLI and application integration | implemented; 14-subcommand smoke passes in CI |
| Reference-SP measurement and derived `tau_SP` artifact | `measure-reference-sp` command implemented; no artifact produced yet (server-side) |
| Development-support and confirmatory lineage-disjoint panels | audited 2026-08-16: **not satisfied on the server** — FCP support parents absent, no unified four-mechanism manifest, seed-matched Official-SP final checkpoint paths unconfirmed (see the parent-directory experiment report) |
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
