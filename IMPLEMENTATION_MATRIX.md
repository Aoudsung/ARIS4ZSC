# DELTA-ZSC v6 implementation matrix

Active identity: `delta_self_consistent_decision_grounded_residual_v6`.

| Requirement | Implementation | Enforced contract |
|---|---|---|
| Immutable Official competence | `base_policy.py`, `official_initializer.py` | required seed-matched full-observation reference, absent from Adam/gradients |
| Zero trainable residual | `base_policy.py` | base equals reference exactly at initialization |
| Unified KL envelope | `mirror_policy.py`, `model.py` | base and final execution projected relative to immutable reference |
| Self/cross-play collection | `runner.py` | fixed half lanes and independent self-partner recurrent state |
| Paired minimax PPO | `training.py`, `losses.py` | equal SP/XP membership, per-group normalization, worst-group actor/value |
| XP-only latent channels | `training.py`, `training_app.py` | response, raw value, successor and contrast consume sliced XP batch |
| XP-only anchors | `runner.py`, `anchors.py` | sparse candidates restricted to second-half lanes and remapped locally |
| Structural grounding | `belief_value.py`, `latent_model.py` | detached posterior-weighted full component embedding; no auxiliary loss |
| Exact response posterior | `belief_filter.py`, `response_model.py` | episode-static semantic-only Bayes correction |
| Dense raw-return critic | `belief_value.py`, `losses.py` | TD(lambda) plus pairwise CRN contrasts on one critic |
| Delayed active value | `bayes_voi.py`, `successor_feature.py` | exact 66-outcome posterior marginal and two-step successor value |
| Four execution modes | `deployment.py`, `evaluation_app.py` | reference/residual/passive/active recorded in raw rows and summaries |
| Artifact incompatibility | `config.py`, `storage.py`, `deployment.py` | checkpoint 6, deployment 5, evaluation 3 |
| Formal execution shape | `config.py` | 128 lanes, 64 paired minibatches, peak memory below 40,000 MiB |
| Scientific boundary | authoritative docs and claim builder | no CUDA/development/formal evidence means no v6 performance claim |

The local CPU suite validates implementation contracts. Real CUDA execution
and return evidence remain pending.
