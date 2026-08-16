# CETR-ZSC implementation matrix

Active identity: `constrained_episodic_tail_robust_zsc_v1`.

The registered identity and numerical contract are owned by
[`src/cetr_zsc/config.py`](src/cetr_zsc/config.py). The matrix below traces the scientific requirements to
the active CETR package and Overcooked integration; V6 DELTA paths are not
fallback implementations.

| Requirement | Implementation | Enforced contract |
|---|---|---|
| Method identity and registered configuration | [`src/cetr_zsc/config.py`](src/cetr_zsc/config.py) | one method version and resolved config authority; no copied registration tables |
| Single partner-agnostic recurrent actor | `src/cetr_zsc/model.py` | Official-isomorphic CNN-to-GRU actor; entire actor tree trainable |
| Training-only scalar value baseline | `src/cetr_zsc/model.py`, `src/cetr_zsc/losses.py` | baseline reduces policy-gradient variance and never changes deployment logits |
| Parent groups and mechanism-balanced support | `src/cetr_zsc/partners.py` | SP/OP/SA/FCP support; stages remain inside parent groups; labels never enter actor input |
| Lineage-disjoint confirmatory panel | `src/cetr_zsc/partners.py`, `experiments/overcooked_v2/evaluation_app.py` | parent and co-training lineage do not overlap development support and confirmation |
| Complete episodic collection | `src/cetr_zsc/runner.py` | one complete episode per lane; raw return is the training estimand |
| Parent-level lower-half risk | `src/cetr_zsc/risk.py` | `q` is capped by nominal parent mass and computed from completed raw returns |
| Double cross-fitting | `src/cetr_zsc/risk.py`, `src/cetr_zsc/training.py` | one fold determines weights used only on the other fold |
| Fixed raw-return advantages and shared normalization | `src/cetr_zsc/losses.py` | return-to-go and advantages are fixed at collection; no group-wise normalization |
| Bilateral self-play gradient | `src/cetr_zsc/runner.py`, `src/cetr_zsc/losses.py` | both current-policy sides contribute log-probability terms |
| Reference-derived SP constraint | `experiments/overcooked_v2/reference_sp_app.py`, `src/cetr_zsc/losses.py` | measured `tau_SP`, adaptive nonnegative dual, no manual tolerance or cap |
| Primal-dual transaction | `src/cetr_zsc/training.py` | collect → risk weights → fixed advantages → primal PPO → dual update |
| Training orchestration and identity binding | `experiments/overcooked_v2/training_app.py` | resolved config, parent manifests, and lineage remain bound to each run |
| Common evaluation metrics | `experiments/overcooked_v2/evaluation_app.py` | external mean, lower-half CVaR, and self-play minus `tau_SP` from raw episodes |
| Decisive statistical decision | `experiments/overcooked_v2/claim_app.py` | GO/NO-GO/INCONCLUSIVE rules; no best-checkpoint or failed-seed substitution |
| Official-SP reference measurement | `experiments/overcooked_v2/reference_sp_app.py` | reference initializes CETR and derives the target; it is not a deployment branch |
| Legal deployment path | `experiments/overcooked_v2/deployment.py` | only local observation/history → one actor → action |
| Active CLI | `experiments/overcooked_v2/cetr_zsc.py` | one CETR entry point for registered workflows |

The current dashboard records which implementation rows have actually been
completed. Formal training and evaluation have not started, so the matrix is a
requirements trace, not evidence of performance.

V6 DELTA and DEPI are retired, removed from the active tree, and retained only
in git history. Their posterior, continuation, mirror/VOI, and snapshot-update
machinery is not part of CETR.
