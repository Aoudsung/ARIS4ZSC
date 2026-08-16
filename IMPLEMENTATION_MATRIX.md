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
| Parent groups and mechanism-balanced support | `src/cetr_zsc/partners.py`, `src/cetr_zsc/runner.py` | SP/OP/SA/FCP support; stages remain inside parent groups; deterministic full parent coverage uses fold×role `A0/A1/B0/B1`; labels never enter actor input |
| Lineage-disjoint confirmatory panel | `src/cetr_zsc/partners.py`, `experiments/overcooked_v2/evaluation_app.py` | parent and co-training lineage do not overlap development support and confirmation |
| Complete episodic collection | `src/cetr_zsc/runner.py` | one complete episode per lane; raw return is the training estimand; no random parent sampling |
| Parent-level lower-half risk | `src/cetr_zsc/risk.py` | `q` is capped by nominal parent mass and computed from completed raw returns; no observed-subset re-normalization |
| Double cross-fitting | `src/cetr_zsc/risk.py`, `src/cetr_zsc/training.py` | `q^A` weights only fold B and `q^B` weights only fold A; monitored tail is `0.5 * (q^A · J_hat^B + q^B · J_hat^A)` |
| Fixed raw-return advantages and shared normalization | `src/cetr_zsc/training.py`, `src/cetr_zsc/losses.py` | `prepare_episode_batch` normalizes the complete batch exactly once before minibatch slicing; no group-wise or minibatch-wise normalization |
| Bilateral self-play gradient | `src/cetr_zsc/runner.py`, `src/cetr_zsc/losses.py` | both current-policy sides contribute log-probability terms |
| Reference-derived SP constraint | `experiments/overcooked_v2/reference_sp_app.py`, `src/cetr_zsc/losses.py` | version-2 measured `tau_SP`; resolved absolute source checkpoint must equal resolved initializer; adaptive nonnegative dual, no manual tolerance or cap |
| Primal-dual transaction | `src/cetr_zsc/training.py` | collect → raw-return risk weights → complete-batch normalization → primal PPO → dual update |
| Training orchestration and identity binding | `experiments/overcooked_v2/training_app.py` | resolved config, parent manifests, lineage, τ artifact, and seed mapping remain bound to each run |
| Common evaluation metrics | `experiments/overcooked_v2/evaluation_app.py` | external mean, lower-half CVaR, and self-play minus `tau_SP` from raw episodes; summaries are descriptive |
| Decisive statistical decision | `experiments/overcooked_v2/claim_app.py` | unique `claim` entry; crossed ego-run/parent-lineage bootstrap, paired seed-index SP bootstrap, exact `{"fcp"}` baseline; no best-checkpoint or failed-seed substitution |
| Official-SP reference measurement | `experiments/overcooked_v2/reference_sp_app.py` | version-2 reference artifact with resolved absolute checkpoint; reference initializes CETR and derives target, not a deployment branch |
| Legal deployment path | `experiments/overcooked_v2/deployment.py` | bundle version 7 contains actor parameters only; provenance records are same-directory; runtime is local observation/history → one actor → action |
| Active CLI | `experiments/overcooked_v2/cetr_zsc.py` | one CETR entry point for registered workflows |

The current dashboard records which implementation rows have actually been
completed. The first package audit repair group and application-layer binding
are landed. Formal training and evaluation have not started, so the matrix is a
requirements trace, not evidence of performance.

V6 DELTA and DEPI are retired, removed from the active tree, and retained only
in git history. Their posterior, continuation, mirror/VOI, and snapshot-update
machinery is not part of CETR.
