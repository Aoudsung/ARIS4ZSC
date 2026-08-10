# DELTA-ZSC v5 implementation matrix

This matrix maps the active method and experiment contract to source,
artifacts and executable regression evidence. The active identity is
`delta_belief_conditioned_raw_return_pairwise_crn_v5`.

| Requirement | Active implementation | Contract / artifact | Regression evidence |
|---|---|---|---|
| One shared task actor-critic | `base_policy.py`, `model.py` | disjoint `base_params` and `latent_params` | shared-interface and optimizer-ownership tests |
| Episode-static response latent | `belief_filter.py`, `latent_model.py` | uniform reset only at true episode start | persistence/reset and online/sequence equivalence tests |
| Legal deployment history only | `types.PolicyState`, `observation.py`, `behavior_statistics.py` | no partner ID, lineage, hidden state or counterfactual return | repository, response and belief-isolation tests |
| Shared occurrence cannot alter belief | `response_model.py` | occurrence heads have no component axis | shared-occurrence posterior-isolation test |
| Centered semantic response residuals | `response_model.py` | pooled response plus zero-mean K residual | response-centering and model-execution tests |
| Layout-derived task channels | `observation.task_channel_blocks` | Simple `39`; Wide `43` channels | ingredient-count and four-ingredient extraction tests |
| Unlabeled K=2/4/8 initializer | `semantic_initializer.py`, `calibration_app.py` | schema-1 paired NPZ/JSON artifacts | centering, round-trip and provenance tests |
| Belief-conditioned raw-return value | `belief_value.py` | dueling value plus diagnostic ensemble advantage | belief-value and action-centering tests |
| Dense raw-reward supervision | `belief_value.td_lambda_targets`, `losses.raw_task_value_loss` | every rollout step; Polyak target copy | direct-return and terminal-boundary tests |
| Pairwise CRN action calibration | `contrast.py`, `losses.pairwise_crn_contrast_loss` | same-replica differences and measured standard error | offset, shared-noise and precision-weight tests |
| Delayed legal probe response | `observation.extract_probe_response_target` | `o[t+1] -> o[t+2]`, second-action exclusion | delayed-window and terminal-mask tests |
| Two-step successor prediction | `successor_feature.py` | predicted task/instant features at `t+2` | successor-ordering and loss tests |
| Exact 66-outcome delayed VOI | `bayes_voi.py` | finite enumeration, information gain report-only | exact-mass and decision-irrelevant-information tests |
| KL-constrained mirror policy | `mirror_policy.py` | registered adaptation budget | analytic KL test |
| Collection-time estimator ordering | `training.py`, `runner.py` | latent update commits before PPO; fixed PPO targets | transaction-order and fixed-advantage tests |
| Current/probe-successor real continuations | `anchors.py` | matched CRN, fit/evaluation replicas, base continuation | rollout, bridge, reward-exclusion and schema tests |
| Bounded formal anchor selection | `anchors.select_anchor_indexes` | Floyd exact sampling without full-grid sort | sparse/full equivalence and sampler tests |
| Uniform registered partner distribution | `partners.py`, `training_app.py` | manifest probabilities unchanged over training | static sampler and training-wiring tests |
| Partner panel disjointness | `manifest.py`, `delta_manifest_app.py` | support, calibration, development coverage, confirmatory | manifest construction and lineage tests |
| Seed-index SP initialization | `official_initializer.py`, `development_matrix_app.py` | seed `s` maps to support `run-s/ckpt_final` | SP transplant and 55-command matrix tests |
| Exact development budget | `development_matrix_app._anchor_budget` | pilot + current + successor transitions | passive/active/extra-budget tests |
| Initializer collector layout configs | `delta_unified_{wide,grounded_coord_ring}_initializer_collector.yaml` | development protocol, base variant, anchors disabled | nine-config contract test |
| Formal seed discipline | `training_app.run_training`, `run_cuda_preflight` | formal `0..9`; engineering `-1` only in preflight | CLI/training entry tests |
| Formal execution shape | `config.FORMAL_NUM_ENVS`, `training_app.py` | 256 lanes and peak memory below 40,000 MiB | config and formal-entry tests; real CUDA execution pending |
| Official upstream assets | `upstream_app.py`, `upstream_pipeline_app.py` | config-selected layout, preassigned roots, support/panels/baselines/FCP lineage | CLI and repository registration; server execution pending |
| Policy manifest v2 | `baseline_app.py`, `evaluation_app.build_policy_manifest` | ordered run paths plus parent/co-training lineage | manifest/evaluation CLI tests |
| Paper population evaluator | `evaluation_app._run_population_matrix` | directed `(10,10,500)`, root 42 | population statistics and CLI tests |
| Common-partner evaluator | `evaluation_app._run_common_partner` | `(10,16,2,500)`, root 0 | common summary and key-schedule tests |
| Non-permuted OP evaluation | `official_adapter.VectorEnvironment.create` | `op_ingredient_permutations=False` | evaluator configuration test |
| Formal evaluation CUDA accounting | `evaluation_app.run_evaluation` | one visible GPU and `<40,000 MiB` | formal-entry tests; server measurement pending |
| Layout-specific claim boundary | population/common summaries and experiment plan | one layout alone does not establish full H1/H2/H3 | summary schema and repository tests |
| Artifact compatibility | `storage.py`, `deployment.py` | checkpoint schema 5, deployment schema 4, manifest schema 2 | round-trip tests |
| One active namespace | packaging, repository test, active workflow | no active legacy import | seven repository tests |

## Local validation summary

- Active compilation and source-format checks: passed.
- Registered configurations: 7.
- CLI subcommands: 19.
- Isolated CPU tests collected: 82; final result is recorded in
  `validation/ISOLATED_TEST_RESULTS.json`.
- Exact delayed outcomes: 66.
- CUDA, development return, formal Wide return and all scientific claims remain
  empirical server work until their artifacts exist.
