# DELTA-ZSC v4 implementation matrix

This matrix is the release traceability record for
`delta_episode_static_centered_residual_delayed_exact_voi_v4`. It maps every required method contract to its active implementation, configuration/artifact boundary, and executable regression evidence.

| Requirement | Active implementation | Contract / artifact | Regression evidence |
|---|---|---|---|
| One shared actor-critic, not one policy per component | `src/delta_zsc/base_policy.py`; `DeltaModel` owns exactly `base_params` and `latent_params` | `docs/METHOD_SPEC.md` §§2, 14 | `test_model_executes_all_variants_with_one_shared_interface`; `test_base_and_latent_updates_are_separate_finite_transactions` |
| Episode-static latent | `belief_filter.episode_static_prior`, `belief_filter.filter_update`; no active `transition.py` | method v4; checkpoint schema 4 | `test_episode_static_filter_persists_and_resets_only_at_episode_start`; repository namespace tests |
| Uniform reset only at real episode boundary | `latent_model.observe_response`; `model.observe_after_transition` | `PolicyState.episode_start` | `test_episode_static_filter_persists_and_resets_only_at_episode_start`; `test_batched_filter_sequence_matches_online_steps` |
| Shared occurrence factors cannot create component evidence | `response_model.response_shared_factor_log_probabilities`; shared logits have no K axis | `ResponsePrediction` shared fields | `test_shared_occurrence_heads_cannot_change_online_belief`; `test_decision_parameters_cannot_change_online_belief` |
| Component-semantic conditional geometry | `response_semantic_factor_log_probabilities`: position, direction, inventory | `DirectResponsePrediction` K-axis fields | `test_v4_response_and_decision_residuals_are_component_centered`; core model execution test |
| Aligned 31-class interface semantics | `observation.align_egocentric_frames`, `_with_interact_exclusion`, `extract_interface_target` | 30 structured events + `OTHER/MULTI`; independent recipe mask | `test_interface_alignment_and_interact_exclusion_contract`; `test_delayed_probe_target_uses_second_action_and_masks_terminal_windows` |
| Shared baseline + centered response residual | `response_model._semantic_logits`, `_center_components`, `_predict_event` | context residual + direct embedding residual; standard fan-in residual initialization | `test_v4_response_and_decision_residuals_are_component_centered` |
| Shared pooled event structure | `response_model._event_shared_logits` | facility + direction + object + `OTHER/MULTI`, one normalized 31-class vector | centered residual tests; exact outcome tests |
| Direct component-to-output path | `response_model._init_semantic_head`, `_init_event_head`, `_semantic_logits` | separate `*_residual_embedding` parameters | centered residual test and initializer round-trip tests |
| Unlabeled spectral-simplex initialization | `semantic_initializer.fit_spectral_simplex_initializer`, `spectral_simplex_bias`; `calibration_app.build_semantic_initializer` | paired `semantic_component_initializer.npz/.json`, schema 1 | `test_spectral_simplex_initializer_is_centered_and_round_trips`; `test_deterministic_and_spectral_initializers_are_centered_and_round_trip` |
| Pooled conditional residual extraction | `calibration_app._project_event_frames`, `_cross_fitted_pooled_event_probabilities`, `_fit_pooled_linear_event_model` | fixed Rademacher projection and cross-fitted pooled predictor recorded in initializer source | initializer provenance tests; contract CLI smoke |
| No privileged labels in initializer | initializer metadata requires `uses_partner_labels=false`; conditional label oracle is report-only | exact calibration run IDs and parent lineages | `test_fitted_initializer_provenance_is_layout_and_lineage_bound` |
| Calibration/training lineage disjointness | `validate_semantic_initializer_provenance`; manifest validation | development/formal semantic runs require fitted artifact | `test_fitted_initializer_provenance_is_layout_and_lineage_bound`; manifest tests |
| K-specific initializer resolution | `development_matrix_app._initializer_for_component_count` | root or `k-K/` artifact directory | `test_development_matrix_resolves_k_specific_initializer_root` |
| Shared decision baseline + centered component residual | `decision_model._prediction_from_hidden`, `decision_predict`, `successor_decision_predict` | current and successor `DecisionPrediction` | `test_v4_response_and_decision_residuals_are_component_centered` |
| Component-independent decision variance | `decision_model._prediction_from_hidden` broadcasts shared variance across K | Helmert contrast Gaussian | centered decision residual test; action-contrast basis test |
| Full CRN covariance in action-contrast space | `decision_model.action_contrast_matrix`, `decision_component_log_probability`; `anchors._measurement_covariance` | A-1 dimensional covariance with fit/evaluation replica separation | `test_action_contrast_basis_is_orthonormal_and_offset_invariant`; anchor tests |
| Separately normalized proper-score channels | `losses.latent_composite_loss` | `L_shared + L_semantic + L_decision`, each channel own count | `test_three_channel_objective_is_exact_sum_of_normalized_channels`; `test_response_only_variant_never_uses_decision_anchor_channel` |
| One composite latent backward transaction | `training.update_latent_model` | one `value_and_grad` over the composite objective | finite transaction and outer ordering tests |
| Latent-before-PPO ordering | `training.training_update` | labels and features use collection-time base tree | `test_outer_transaction_commits_latent_before_ppo` |
| PPO independent of latent/adaptation graph | `losses.ppo_loss` calls `model.sequence(... compute_latent=False, execute_adaptation=False)` | disjoint optimizer trees | finite transaction test; batched base replay test |
| Delayed causal probe target | `observation.extract_probe_response_target`; delayed section of `losses.latent_composite_loss` | `o[t+1] -> o[t+2]`, second action, two-terminal mask | both delayed-target tests in core/v4 semantics |
| Delayed probe predictor | `response_model.init_probe_response_params`, `probe_response_predict` | current frame/history + candidate probe -> delayed compact response | exact outcome/VOI tests; delayed target tests |
| One base bridge after active probe | `model.step`; `PolicyState.probe_continuation_pending`; `observe_after_transition` | active eligibility is false on bridge lane | `test_active_probe_commits_exactly_one_base_continuation` |
| Current-state all-action CRN target | `anchors.collect_all_action_continuations` | force first action, continue collection-time base policy | `test_mock_end_to_end_rollout_and_anchor_use_base_policy` |
| Probe-conditioned successor CRN target | `anchors.collect_probe_successor_continuations` | force probe, base bridge, force decision at t+2, nested `lax.map` | `test_probe_anchor_forces_decision_only_after_base_bridge`; `test_successor_anchor_waits_for_delayed_response_and_excludes_bridge_rewards` |
| Probe and bridge reward exclusion | `collect_probe_successor_continuations` discards both rewards before successor return loop | successor target starts at forced t+2 decision | successor-anchor timing test |
| Passive avoids privileged successor simulation | `_collect_from_world(... collect_successor=False)` returns false-masked placeholders | fixed AnchorBatch signature, zero valid successor observations | `test_passive_anchor_batch_has_no_successor_observation` |
| Active resource cost is explicit | development matrix anchor budget and resource ledger | active successor branches counted separately | `test_development_budget_distinguishes_passive_and_active_anchor_cost`; resource tests |
| Exact 66-outcome delayed marginal | `bayes_voi.compact_active_outcome_log_probabilities` | 2 unavailable + 2 no-change + 62 changed/event outcomes | `test_compact_delayed_distribution_has_exactly_66_normalized_outcomes`; generated diagnostics |
| Probe-conditioned successor utility required | `bayes_voi._validate_inputs` rejects shared current matrix | shape `[..., P, K, A]` only | `test_probe_conditioned_successor_utility_is_required_and_jittable` |
| Decision-relevant VOI, not information reward | `myopic_value_of_information_details` evaluates posterior-optimal successor action | information gain report-only | uninformative and decision-irrelevant VOI tests; synthetic diagnostics |
| Correct temporal discount | `model.step` uses `gamma ** 2 * raw_voi` | response and decision become available at t+2 | active bridge/successor timing tests; method docs |
| No numerical integration or clamp | `bayes_voi.py` is a finite exact sum | no Halton, sample count, quadrature error, or value clamp | repository doc test; exact outcome tests |
| KL-constrained passive/active execution | `mirror_policy.mirror_policy_logits` | registered `adaptation_kl_budget` | `test_mirror_policy_satisfies_kl_constraint` |
| Sparse anchor snapshots preserve full recording | `runner.make_anchor_snapshot_rollout_kernel`, `anchors.gather_time_lane` | uniform sparse state selection | `test_sparse_anchor_rollout_matches_full_recording` |
| Checkpoint/deployment state compatibility | `storage.py`; `deployment.py`; checkpoint schema 4 | semantic initializer metadata and pending bridge state preserved | `test_checkpoint_and_deployment_round_trip` |
| Six registered configs share three method fields | `config.py`; six YAML files | `K`, `H`, `delta` only | `test_all_registered_configs_load_and_method_has_three_fields`; contract suite |
| Unified CLI and development matrix | `experiments/overcooked_v2/delta_zsc.py`; `development_matrix_app.py` | fresh process per cell, explicit initializer | three CLI tests; CLI contract smoke |
| One active namespace | `pyproject.toml`, active experiment imports, workflow | retired DEPI v8 only under legacy paths | seven repository tests; contract namespace audit |
| Final audit shape correctness | `training_app._final_decision_audit` reduces probe event JS over probe before active-anchor mean | anchor-shaped report values | complete 50-test regression plus focused final-audit tests |
| Installation entrypoints resolve one runtime stack | `pyproject.toml`, `requirements.txt` | exact dependency-set equality, including pinned Orbax | `test_requirements_match_pyproject_runtime_dependencies_exactly`; contract dependency audit |
| Reproducible isolated validation | `validation/run_all_isolated_tests.py`, `run_contract_validation.sh` | JSON and text evidence; Python 3.10 grammar audit | `validation/ISOLATED_TEST_RESULTS.json`: 50/50 passed |

## Release acceptance summary

- Active compilation: passed.
- Registered configurations: 6/6 loaded.
- CLI subcommands: 16/16 help paths parsed.
- Isolated tests: 50/50 passed, 0 failed, 0 timed out.
- Exact delayed outcomes: 66.
- Active transition model: none; episode-static identity only.
- v4 CUDA/formal performance evidence: not generated in this package.
