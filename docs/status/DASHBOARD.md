# CETR-ZSC status dashboard

Status: first scientific-audit repair group and application-layer repair landed; formal launch remains blocked by server-panel prerequisites.

| Item | Status |
|---|---|
| Active method identity | implemented in [`src/cetr_zsc/config.py`](../../src/cetr_zsc/config.py): `constrained_episodic_tail_robust_zsc_v1` |
| Active configuration contract | implemented; registered values are owned by [`src/cetr_zsc/config.py`](../../src/cetr_zsc/config.py) (`version: 6`, checkpoint schema `8`) |
| CETR package boundary | first audit repair group landed: whole-episode estimand, complete deterministic external coverage, fold/role cross-fitting, and one pre-minibatch shared normalization are contracted |
| CETR model, risk, losses, runner, partners, and training modules | package repair landed; no performance evidence is implied |
| Overcooked CETR CLI and application integration | application-layer repair landed: τ artifact v2 binding, seed-`-1` preflight mapping, crossed claim statistics, exact FCP baseline gate, and actor-only deployment bundle |
| cetr-ci | active workflow is the authority for CI conclusions; no new result is inferred by this dashboard; CUDA preflight remains dispatch-gated |
| Reference-SP measurement and derived `tau_SP` artifact | version-2 path landed; `source_checkpoint` is resolved and must equal the resolved `--sp-initializer`; no formal artifact has been measured yet |
| Formal launch prerequisites | server-panel gap: FCP support parents, one unified manifest, normalized seed-checkpoint paths, and measured `tau` artifacts are still required |
| Development-support and confirmatory lineage-disjoint panels | audited 2026-08-16: **not satisfied on the server** — FCP support parents absent, no unified four-mechanism manifest, seed-matched Official-SP final checkpoint paths unconfirmed (see the parent-directory experiment report) |
| Deployment contract | target bundle version `7`: actor parameter subtree only; reference artifact and training manifest move to same-directory `provenance.json` |
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
matrix is supplementary. Evaluation summaries are descriptive; only `claim` may
produce GO/NO-GO/INCONCLUSIVE, using crossed external bootstrap and paired
seed-level SP differences against exactly the FCP method set.

No compilation, mechanical check, development artifact, or historical V6 result
is a CETR performance claim. Formal runs have not begun, and there is currently
no CETR return evidence to summarize.
