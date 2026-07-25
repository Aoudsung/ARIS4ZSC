"""Static-contract tests for Path C v3 work packages H1, H2, and H3."""

from __future__ import annotations

import json
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from experiments.overcooked_v2.path_c_adaptation import (
    AdaptationTrainingSpec,
    PathCAdaptationPolicy,
    PathCAdaptationTrainer,
    require_admitted_pool,
    validate_probe_calibration_artifacts,
)
from experiments.overcooked_v2.path_c_pool_admission import (
    AdmissionSpec,
    _action_distribution_jsd,
)
from experiments.overcooked_v2.path_c_backbone_ppo import (
    RecurrentIPPOBackbone,
    RecurrentIPPOBackboneTrainer,
    sample_actor_actions,
)
from experiments.overcooked_v2.path_c_standard_evaluation import (
    StandardPairingEvaluator,
    validate_evaluation_probe_config,
)
from experiments.overcooked_v2.path_c_response_probe import (
    PartnerConditionedValueEnsemble,
    REGISTERED_LOCAL_RESPONSE_SPEC,
    ResponseModelEnsemble,
    calibrate_response_disagreement_threshold,
    finite_prototype_two_action_values,
    episode_bootstrap_mask,
    initial_partner_belief,
    mean_pairwise_jsd,
    local_non_agent_change_from_default_observation,
    partner_visible_from_default_observation,
    response_token_from_visibility,
    response_tokens_from_observations,
    select_finite_prototype_two_action_surrogate,
    select_response_probe,
    update_partner_belief,
    validate_response_probe_config,
    weighted_response_information,
)
from experiments.overcooked_v2.path_c_standard import (
    PrimitiveObservationBatch,
    PrimitiveRecurrentEnsembleQ,
    choose_primitive_actions,
)
from experiments.overcooked_v2.path_c_standard_diagnostics import (
    decompose_raw_reward_events,
)


def _legacy_probe(q_values: torch.Tensor, *, allowed: bool = True):
    return choose_primitive_actions(
        q_values,
        torch.ones(q_values.shape[0], q_values.shape[-1], dtype=torch.bool),
        rng=np.random.default_rng(7),
        probe_enabled=True,
        disagreement_threshold=1e-4,
        max_probe_regret=2.0,
        probe_allowed=np.full(q_values.shape[0], allowed),
    )


def test_relative_probe_gate_is_invariant_to_q_translation():
    q_values = torch.tensor([[[2.0, 0.0, 0.0], [2.0, 1.5, 0.0]]])
    reference = _legacy_probe(q_values)
    shifted = _legacy_probe(q_values + 173.25)
    assert shifted.actions.tolist() == reference.actions.tolist()
    assert shifted.is_probe.tolist() == reference.is_probe.tolist()
    assert shifted.probe_regret_pass.tolist() == reference.probe_regret_pass.tolist()
    assert shifted.probe_candidate_regret == pytest.approx(
        reference.probe_candidate_regret
    )


def test_probe_budget_window_mask_and_exploration_override():
    q_values = torch.tensor([[[2.0, 0.0], [2.0, 1.5]]])
    assert _legacy_probe(q_values, allowed=False).is_probe.tolist() == [False]
    exploratory = choose_primitive_actions(
        q_values,
        torch.ones(1, 2, dtype=torch.bool),
        rng=np.random.default_rng(0),
        epsilon=1.0,
        probe_enabled=True,
        disagreement_threshold=1e-4,
        max_probe_regret=2.0,
        probe_allowed=np.asarray([True]),
    )
    assert exploratory.is_exploration.tolist() == [True]
    assert exploratory.is_probe.tolist() == [False]


def test_greedy_candidate_is_not_counted_as_a_probe():
    decision = choose_primitive_actions(
        torch.tensor([[[2.0, 0.0], [2.0, 0.0]]]),
        torch.ones(1, 2, dtype=torch.bool),
        rng=np.random.default_rng(0),
        probe_enabled=True,
        disagreement_threshold=1.0e-4,
        max_probe_regret=1.0,
        probe_allowed=np.asarray([True]),
    )
    assert decision.probe_candidate_equals_greedy.tolist() == [True]
    assert decision.is_probe.tolist() == [False]


def test_raw_reward_event_decomposition_uses_explicit_official_events():
    events = decompose_raw_reward_events(
        np.asarray([20.0, -20.0, -5.0, 40.0, 15.0, 0.0])
    )
    assert events.correct_delivery_count == 3
    assert events.wrong_delivery_count == 1
    assert events.indicator_activation_count == 1
    assert events.ambiguous_step_count == 1
    assert events.ambiguous_raw_rewards == (15.0,)


def test_identical_response_models_have_zero_jsd_and_no_probe():
    probabilities = torch.full((2, 6, 5, 7), 1.0 / 7.0)
    scores = mean_pairwise_jsd(probabilities)
    assert torch.allclose(scores, torch.zeros_like(scores))
    decision = select_response_probe(
        actor_logits=torch.zeros(2, 6),
        critic_q_values=torch.zeros(2, 5, 6),
        candidate_probabilities=probabilities,
        valid_actions=torch.ones(2, 6, dtype=torch.bool),
        response_disagreement_threshold=0.02,
        max_probe_regret=1.0,
        probe_allowed=np.ones(2, dtype=bool),
    )
    assert decision.is_probe.tolist() == [False, False]


def test_pairwise_jsd_is_symmetric_and_nonnegative():
    distributions = torch.tensor([[[[0.9, 0.1], [0.2, 0.8]]]])
    reversed_models = distributions.flip(-2)
    assert mean_pairwise_jsd(distributions).item() >= 0.0
    assert mean_pairwise_jsd(distributions).item() == pytest.approx(
        mean_pairwise_jsd(reversed_models).item()
    )


def test_partner_response_information_is_zero_only_for_matching_hypotheses():
    probabilities = torch.tensor([[
        [[0.5, 0.5], [0.5, 0.5]],
        [[0.9, 0.1], [0.1, 0.9]],
    ]])
    weights = initial_partner_belief(1, 2, device="cpu")
    information = weighted_response_information(probabilities, weights)
    assert information[0, 0].item() == pytest.approx(0.0, abs=1.0e-7)
    assert information[0, 1].item() > 0.0
    assert information[0, 1].item() <= float(np.log(2.0)) + 1.0e-7


@pytest.mark.parametrize(
    ("probabilities", "weights"),
    [
        (torch.tensor([[[[float("nan"), 1.0], [0.5, 0.5]]]]), torch.tensor([[0.5, 0.5]])),
        (torch.tensor([[[[-0.1, 1.1], [0.5, 0.5]]]]), torch.tensor([[0.5, 0.5]])),
        (torch.tensor([[[[0.5, 0.5], [0.5, 0.5]]]]), torch.tensor([[1.0, -0.1]])),
    ],
)
def test_partner_response_information_rejects_invalid_probability_inputs(
    probabilities: torch.Tensor,
    weights: torch.Tensor,
):
    with pytest.raises(ValueError):
        weighted_response_information(probabilities, weights)


def test_partner_conditioned_probe_selects_an_informative_safe_action():
    probabilities = torch.tensor([[
        [[0.5, 0.5], [0.5, 0.5]],
        [[0.9, 0.1], [0.1, 0.9]],
    ]])
    decision = select_response_probe(
        actor_logits=torch.tensor([[2.0, 0.0]]),
        critic_q_values=torch.tensor([[[2.0, 1.5], [2.0, 1.5]]]),
        candidate_probabilities=probabilities,
        valid_actions=torch.ones(1, 2, dtype=torch.bool),
        response_disagreement_threshold=0.02,
        max_probe_regret=1.0,
        probe_allowed=np.asarray([True]),
        partner_weights=initial_partner_belief(1, 2, device="cpu"),
    )
    assert decision.actions.tolist() == [1]
    assert decision.disagreement_pass.tolist() == [True]
    assert decision.regret_pass.tolist() == [True]
    assert decision.is_probe.tolist() == [True]


def test_two_action_surrogate_separates_response_proxy_and_task_cost():
    response_probabilities = torch.tensor([[
        [[0.5, 0.5], [0.5, 0.5]],
        [[1.0, 0.0], [0.0, 1.0]],
    ]])
    continuation_values = torch.tensor([[
        [[3.0, 0.0], [3.0, 0.0]],
        [[4.0, 0.0], [0.0, 4.0]],
    ]])
    values = finite_prototype_two_action_values(
        response_probabilities=response_probabilities,
        continuation_values=continuation_values,
        partner_belief=torch.tensor([[0.5, 0.5]]),
    )
    assert values.j_mask.detach().cpu().numpy() == pytest.approx(
        np.asarray([[3.0, 2.0]])
    )
    assert values.j_use.detach().cpu().numpy() == pytest.approx(
        np.asarray([[3.0, 4.0]])
    )
    assert values.v_mask.tolist() == pytest.approx([3.0])
    assert values.i_response.detach().cpu().numpy() == pytest.approx(
        np.asarray([[0.0, 2.0]])
    )
    assert values.c_task.detach().cpu().numpy() == pytest.approx(
        np.asarray([[0.0, 1.0]])
    )
    assert values.s_seq.detach().cpu().numpy() == pytest.approx(
        np.asarray([[0.0, 1.0]])
    )


def test_two_action_surrogate_selector_uses_net_proxy_not_identity_information():
    response_probabilities = torch.tensor([[
        [[0.5, 0.5], [0.5, 0.5]],
        [[1.0, 0.0], [0.0, 1.0]],
    ]])
    continuation_values = torch.tensor([[
        [[3.0, 0.0], [3.0, 0.0]],
        [[4.0, 0.0], [0.0, 4.0]],
    ]])
    decision = select_finite_prototype_two_action_surrogate(
        actor_logits=torch.tensor([[2.0, 0.0]]),
        candidate_probabilities=response_probabilities,
        continuation_values=continuation_values,
        partner_belief=torch.tensor([[0.5, 0.5]]),
        valid_actions=torch.ones(1, 2, dtype=torch.bool),
        surrogate_score_threshold=0.5,
        max_probe_task_cost=1.0,
        probe_allowed=np.asarray([True]),
    )
    assert decision.actions.tolist() == [1]
    assert decision.candidate_score.tolist() == pytest.approx([1.0])
    assert decision.candidate_regret.tolist() == pytest.approx([1.0])
    assert decision.is_probe.tolist() == [True]


def test_partner_value_heads_cannot_backpropagate_into_shared_features():
    ensemble = PartnerConditionedValueEnsemble(
        4, n_actions=2, hidden_dim=3, ensemble_size=2
    )
    features = torch.randn(2, 2, 4, requires_grad=True)
    loss, _ = ensemble.partner_conditioned_mse(
        features,
        torch.zeros(2, 2, dtype=torch.long),
        torch.ones(2, 2, dtype=torch.long),
        torch.zeros(2, 2),
        torch.tensor([0, 1]),
    )
    loss.backward()
    assert features.grad is None


def test_response_probe_does_not_charge_an_action_already_sampled_by_actor():
    probabilities = torch.tensor([[
        [[0.5, 0.5], [0.5, 0.5]],
        [[0.9, 0.1], [0.1, 0.9]],
    ]])
    decision = select_response_probe(
        actor_logits=torch.tensor([[2.0, 0.0]]),
        critic_q_values=torch.tensor([[[2.0, 1.5], [2.0, 1.5]]]),
        candidate_probabilities=probabilities,
        valid_actions=torch.ones(1, 2, dtype=torch.bool),
        response_disagreement_threshold=0.02,
        max_probe_regret=1.0,
        probe_allowed=np.asarray([True]),
        partner_weights=initial_partner_belief(1, 2, device="cpu"),
        baseline_actions=torch.tensor([1]),
    )
    assert decision.candidate_equals_baseline.tolist() == [True]
    assert decision.actions.tolist() == [1]
    assert decision.is_probe.tolist() == [False]


def test_partner_belief_update_moves_toward_the_predictive_hypothesis():
    prior = initial_partner_belief(2, 2, device="cpu")
    predictions = torch.tensor([
        [[0.9, 0.1], [0.1, 0.9]],
        [[0.9, 0.1], [0.1, 0.9]],
    ])
    posterior = update_partner_belief(
        prior,
        predictions,
        torch.tensor([0, 1]),
        probability_floor=1.0e-6,
    )
    assert posterior[0].tolist() == pytest.approx([0.9, 0.1])
    assert posterior[1].tolist() == pytest.approx([0.1, 0.9])
    unchanged = update_partner_belief(
        prior,
        predictions,
        torch.tensor([0, 1]),
        probability_floor=1.0e-6,
        observed=torch.tensor([False, False]),
    )
    assert torch.equal(unchanged, prior)


def test_partner_belief_update_rejects_invalid_probability_mass():
    token = torch.tensor([0])
    with pytest.raises(ValueError, match="belief"):
        update_partner_belief(
            torch.tensor([[1.1, -0.1]]),
            torch.tensor([[[0.8, 0.2], [0.2, 0.8]]]),
            token,
            probability_floor=1.0e-6,
        )
    with pytest.raises(ValueError, match="sum to one"):
        update_partner_belief(
            torch.tensor([[0.5, 0.5]]),
            torch.tensor([[[0.8, 0.8], [0.2, 0.8]]]),
            token,
            probability_floor=1.0e-6,
        )


def test_partner_conditioned_response_loss_routes_complete_episodes():
    ensemble = ResponseModelEnsemble(
        4, n_actions=2, vocabulary_size=2, hidden_dim=3, ensemble_size=2
    )
    features = torch.randn(2, 3, 4)
    actions = torch.tensor([[0, 1, 0], [1, 0, 1]])
    tokens = torch.tensor([[0, 0, 0], [1, 1, 1]])
    loss, member_losses = ensemble.partner_conditioned_cross_entropy(
        features,
        actions,
        tokens,
        torch.tensor([0, 1]),
    )
    assert torch.isfinite(loss)
    assert len(member_losses) == 2
    loss.backward()
    assert all(
        any(parameter.grad is not None for parameter in model.parameters())
        for model in ensemble.models
    )


def test_response_config_separates_legacy_bootstrap_and_partner_modes():
    common = {
        "controller": "response_voi", "enabled": True,
        "response_disagreement_threshold": 0.02,
        "advantage_disagreement_threshold": 0.02,
        "max_probe_regret": 1.0, "probe_budget_per_episode": 20,
        "probe_window_environment_steps": 100,
        "response_ensemble_size": 2, "response_hidden_dim": 4,
        "bootstrap_seed": 1, "critic_bootstrap_p": 0.5,
    }
    partner = validate_response_probe_config({
        **common,
        "response_model_mode": "partner_conditioned",
        "belief_probability_floor": 1.0e-6,
    })
    assert partner["response_model_mode"] == "partner_conditioned"
    legacy = validate_response_probe_config({
        **common,
        "response_bootstrap_p": 0.5,
    })
    assert legacy["response_model_mode"] == "bootstrap_legacy"


def test_probe_threshold_calibration_uses_only_safe_non_greedy_candidates():
    threshold, count = calibrate_response_disagreement_threshold(
        [0.001, 0.010, 0.030, 0.900, 0.500],
        [0.1, 0.2, 2.0, 0.1, 0.1],
        [False, False, False, True, False],
        max_probe_regret=1.0,
        quantile=0.80,
        candidate_equals_baseline=[False, False, False, False, True],
    )
    assert count == 2
    assert threshold == pytest.approx(np.quantile([0.001, 0.010], 0.80))


def test_response_vocabulary_marks_an_unseen_partner_without_privileged_state():
    assert response_token_from_visibility(
        partner_action=0,
        partner_visible=False,
        local_non_agent_change=False,
        partner_action_channel=True,
    ) == 6
    assert response_token_from_visibility(
        partner_action=0,
        partner_visible=False,
        local_non_agent_change=False,
        partner_action_channel=False,
    ) == REGISTERED_LOCAL_RESPONSE_SPEC.token_ids[
        "response_unseen__latency_le_1"
    ]


def test_partner_visibility_comes_from_the_official_local_observation_layer():
    observation = np.zeros((2, 5, 5, 39), dtype=np.float32)
    observation[0, 2, 3, 10] = 1.0  # three ingredients => other position channel 10
    visible = partner_visible_from_default_observation(
        observation, indicate_successful_delivery=True
    )
    assert visible.tolist() == [True, False]
    wide_observation = np.zeros((1, 5, 5, 43), dtype=np.float32)
    wide_observation[0, 1, 1, 11] = 1.0  # four ingredients => channel 11
    assert partner_visible_from_default_observation(
        wide_observation, indicate_successful_delivery=True
    ).tolist() == [True]
    tokens = response_tokens_from_observations(
        partner_actions=np.asarray([[5, 2]]),
        partner_visible=np.asarray([[True, False]]),
        local_non_agent_change=np.asarray([[False, False]]),
        partner_action_channel=True,
    )
    assert tokens.tolist() == [[5, 6]]
    previous = np.zeros((1, 1, 5, 5, 39), dtype=np.float32)
    following = previous.copy()
    following[0, 0, 2, 3, 20] = 1.0
    assert local_non_agent_change_from_default_observation(
        previous, following, indicate_successful_delivery=True
    ).tolist() == [[True]]


def test_response_probe_truth_table_requires_non_greedy_candidate():
    probabilities = torch.full((1, 2, 2, 2), 0.5)
    probabilities[:, 0, 0] = torch.tensor([0.99, 0.01])
    probabilities[:, 0, 1] = torch.tensor([0.01, 0.99])
    decision = select_response_probe(
        actor_logits=torch.tensor([[2.0, 0.0]]),
        critic_q_values=torch.tensor([[[2.0, 1.5], [2.0, 1.5]]]),
        candidate_probabilities=probabilities,
        valid_actions=torch.ones(1, 2, dtype=torch.bool),
        response_disagreement_threshold=0.02,
        max_probe_regret=1.0,
        probe_allowed=np.asarray([True]),
    )
    assert decision.candidate_equals_greedy.tolist() == [True]
    assert decision.is_probe.tolist() == [False]


def test_episode_bootstrap_masks_are_deterministic_and_head_distinct():
    first = episode_bootstrap_mask(
        range(32), ensemble_size=5, bootstrap_seed=41, bootstrap_p=0.5
    )
    second = episode_bootstrap_mask(
        range(32), ensemble_size=5, bootstrap_seed=41, bootstrap_p=0.5
    )
    assert torch.equal(first, second)
    assert torch.unique(first.T, dim=0).shape[0] > 1
    assert bool(first.any(dim=1).all())


def test_backbone_accepts_its_declared_observation_shape_for_step_and_sequence():
    model = RecurrentIPPOBackbone(
        (5, 5, 3),
        visual_embedding_dim=8,
        recurrent_dim=8,
        conv_channels=(4, 4, 4),
    )
    batch = PrimitiveObservationBatch(
        observations=torch.zeros(2, 3, 5, 5, 3),
        previous_actions=torch.full((2, 3), 6, dtype=torch.long),
        previous_rewards=torch.zeros(2, 3),
        episode_starts=torch.tensor([[True, False, False], [True, False, False]]),
        valid_actions=torch.ones(2, 3, 6, dtype=torch.bool),
        time_mask=torch.ones(2, 3, dtype=torch.bool),
        lengths=torch.full((2,), 3, dtype=torch.long),
    )
    logits, values, features = model.forward_sequence(batch)
    assert logits.shape == (2, 3, 6)
    assert values.shape == (2, 3)
    assert features.shape == (2, 3, 8)
    step_batch = PrimitiveObservationBatch(
        observations=batch.observations[:, :1],
        previous_actions=batch.previous_actions[:, :1],
        previous_rewards=batch.previous_rewards[:, :1],
        episode_starts=batch.episode_starts[:, :1],
        valid_actions=batch.valid_actions[:, :1],
        time_mask=batch.time_mask[:, :1],
        lengths=torch.ones(2, dtype=torch.long),
    )
    step_logits, step_values, step_features, _ = model.forward_step(
        step_batch,
        model.initial_state(2, device="cpu"),
    )
    assert step_logits.shape == (2, 6)
    assert step_values.shape == (2,)
    assert step_features.shape == (2, 8)


def test_stochastic_actor_sampling_uses_caller_owned_uniforms():
    logits = torch.tensor([[0.0, 0.0], [0.0, 0.0], [8.0, -8.0]])
    actions = sample_actor_actions(logits, np.asarray([0.25, 0.75, 0.999]))
    assert actions.tolist() == [0, 1, 0]


def test_actor_and_response_losses_cannot_reach_trunk():
    model = PathCAdaptationPolicy(
        (5, 5, 3),
        recurrent_dim=8,
        visual_embedding_dim=8,
        conv_channels=(4, 4, 4),
        n_critic_heads=2,
        response_ensemble_size=2,
        response_hidden_dim=4,
    )
    features = torch.randn(2, 3, 8, requires_grad=True)
    model.actor_logits(features).sum().backward()
    assert features.grad is None
    assert all(parameter.grad is None for parameter in model.trunk_parameters)


def test_adaptation_manifest_preserves_new_mode_and_reads_legacy_checkpoints():
    model = PathCAdaptationPolicy(
        (5, 5, 3), recurrent_dim=8, visual_embedding_dim=8,
        conv_channels=(4, 4, 4), n_critic_heads=2,
        response_ensemble_size=2, response_hidden_dim=4,
        response_model_mode="partner_conditioned",
    )
    restored = PathCAdaptationPolicy.from_manifest(model.architecture_manifest())
    assert restored.response_model_mode == "partner_conditioned"
    legacy_manifest = model.architecture_manifest()
    legacy_manifest.pop("response_model_mode")
    legacy = PathCAdaptationPolicy.from_manifest(legacy_manifest)
    assert legacy.response_model_mode == "bootstrap_legacy"
    features = torch.randn(2, 3, 8)
    loss, _ = model.response_ensemble.bootstrap_cross_entropy(
        features.detach(),
        torch.zeros(2, 3, dtype=torch.long),
        torch.zeros(2, 3, dtype=torch.long),
        torch.ones(2, 2, dtype=torch.bool),
    )
    loss.backward()
    assert all(parameter.grad is None for parameter in model.trunk_parameters)


def test_critic_loss_reaches_trunk():
    model = PathCAdaptationPolicy(
        (5, 5, 3), recurrent_dim=8, visual_embedding_dim=8,
        conv_channels=(4, 4, 4), n_critic_heads=2,
        response_ensemble_size=2, response_hidden_dim=4,
    )
    feature = torch.randn(2, 8, requires_grad=True)
    model.critic_q(feature).sum().backward()
    assert feature.grad is not None and float(feature.grad.abs().sum()) > 0.0


def test_adaptation_critic_uses_per_episode_head_bootstrap_support():
    source = inspect.getsource(PathCAdaptationTrainer.optimize_losses)
    assert "critic_bootstrap_mask" in source
    assert "critic_weights" in source


def test_admission_report_is_fail_closed(tmp_path):
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        require_admitted_pool(missing)
    report = tmp_path / "report.json"
    report.write_text(json.dumps({
        "schema_version": "path_c_pool_admission_report_v1",
        "scientific_readout_allowed": False,
        "admitted": False,
        "members": {"a": {"admitted": False}},
    }))
    with pytest.raises(ValueError, match="did not pass"):
        require_admitted_pool(report)


def test_partner_pool_requires_independent_training_seeds(tmp_path):
    report = tmp_path / "same_seed.json"
    report.write_text(json.dumps({
        "schema_version": "path_c_pool_admission_report_v1",
        "scientific_readout_allowed": False,
        "admitted": True,
        "policy_action_selection": "stochastic",
        "members": {
            "a": {"admitted": True, "training_seed": 101},
            "b": {"admitted": True, "training_seed": 101},
        },
    }))
    with pytest.raises(ValueError, match="independently trained"):
        require_admitted_pool(report)
    report.write_text(json.dumps({
        "schema_version": "path_c_pool_admission_report_v2",
        "scientific_readout_allowed": False,
        "admitted": True,
        "policy_action_selection": "stochastic",
        "members": {
            "seed_101_step_1": {"admitted": True, "training_seed": 101},
            "seed_102_step_1": {"admitted": True, "training_seed": 102},
        },
    }))
    assert len(require_admitted_pool(report)["members"]) == 2


def test_admission_records_behavior_distance_without_a_new_distance_gate():
    assert _action_distribution_jsd(
        [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]
    ) == pytest.approx(0.0)
    assert _action_distribution_jsd([10, 0, 0, 0, 0, 0], [0, 10, 0, 0, 0, 0]) > 0.0
    spec = AdmissionSpec.from_mapping({
        "evaluation_seed": 1,
        "checkpoint_paths": ["a.pt", "b.pt"],
        "output_dir": "out",
        "evaluation": {
            "episodes_per_pairing": 100,
            "evaluation_batch_size": 100,
            "admission_floor": 0.5,
            "minimum_admitted_members": 2,
            "minimum_distinct_training_seeds": 2,
            "members_per_training_seed": 1,
            "policy_action_selection": "stochastic",
        },
    })
    assert spec.minimum_distinct_training_seeds == 2
    assert not hasattr(spec, "minimum_behavior_distance")


def test_response_prefit_is_non_scientific_and_response_only():
    spec = AdaptationTrainingSpec.from_mapping({
        "run_kind": "diagnostic",
        "scientific_readout_allowed": False,
        "training_mode": "response_only",
        "seed": 101,
        "output_dir": "out",
        "training": {
            "total_environment_steps": 100000,
            "batch_size_envs": 250,
            "rollout_steps": 400,
            "metrics_interval_environment_steps": 100000,
            "target_update_environment_steps": 100000,
            "actor_update_epochs": 1,
            "shaping_horizon_environment_steps": 50000,
            "stop_if_no_delivery_by_environment_steps": 50000,
            "trunk_learning_rate": 0.0001,
            "actor_learning_rate": 0.00025,
            "response_learning_rate": 0.00025,
            "entropy_coefficient": 0.01,
            "gamma": 0.99,
            "gradient_clip_norm": 0.25,
            "ppo_clip": 0.2,
        },
    })
    assert spec.training_mode == "response_only"
    assert spec.scientific_readout_allowed is False


def test_adaptation_has_a_callable_rollout_engine_and_entrypoint():
    assert callable(PathCAdaptationTrainer.run)
    source = inspect.getsource(PathCAdaptationTrainer.run)
    assert "step_joint" in source
    assert "optimize_losses" in source
    entrypoint = Path(__file__).parents[1] / "scripts" / "train_path_c_adaptation.py"
    assert "run_adaptation_training" in entrypoint.read_text(encoding="utf-8")


def test_adaptation_metrics_expose_every_probe_condition():
    source = inspect.getsource(PathCAdaptationTrainer.run)
    for field in (
        "candidate_evaluated_count",
        "disagreement_threshold_pass_count",
        "regret_pass_count",
        "candidate_equals_greedy_count",
        "candidate_equals_baseline_count",
        "budget_and_window_pass_count",
        "other_conditions_without_disagreement_pass_count",
        "disagreement_only_block_count",
        "regret_block_count",
        "greedy_candidate_block_count",
        "budget_block_count",
        "joint_probe_count",
        "response_token_counts",
        "normalized_probe_budget_use",
    ):
        assert field in source


def test_probe_calibration_uses_partner_pool_without_updates():
    spec = AdaptationTrainingSpec.from_mapping({
        "run_kind": "diagnostic",
        "scientific_readout_allowed": False,
        "training_mode": "probe_calibration",
        "seed": 101,
        "output_dir": "out",
        "training": {
            "total_environment_steps": 200000,
            "batch_size_envs": 250,
            "rollout_steps": 400,
            "metrics_interval_environment_steps": 100000,
            "target_update_environment_steps": 100000,
            "actor_update_epochs": 1,
            "shaping_horizon_environment_steps": 100000,
            "stop_if_no_delivery_by_environment_steps": 200000,
            "trunk_learning_rate": 0.0001,
            "actor_learning_rate": 0.00025,
            "response_learning_rate": 0.00025,
            "entropy_coefficient": 0.01,
            "gamma": 0.99,
            "gradient_clip_norm": 0.25,
            "ppo_clip": 0.2,
        },
    })
    assert spec.training_mode == "probe_calibration"
    source = inspect.getsource(PathCAdaptationTrainer.run)
    assert 'self.spec.training_mode == "joint"' in source
    assert "calibrate_response_disagreement_threshold" in source


def test_formal_calibration_is_recomputed_from_content_addressed_rows():
    source = inspect.getsource(validate_probe_calibration_artifacts)
    for required_fragment in (
        "calibration_rows_sha256",
        "calibration_rows_line_count",
        "calibrate_response_disagreement_threshold",
        "response_initialization_checkpoint_sha256",
        "partner_pool_admission_report_sha256",
        "environment_config_sha256",
        "calibration_contract_sha256",
    ):
        assert required_fragment in source
    config_path = (
        Path(__file__).parents[1]
        / "configs"
        / "path_c_adaptation_simple_generic_information.yaml"
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert payload["run_kind"] == "formal"
    assert payload["probe_calibration_summary_sha256"] == "pending"
    assert payload["probe_calibration_quantile"] == pytest.approx(0.80)


def test_backbone_uses_the_official_clipped_half_value_loss():
    source = inspect.getsource(RecurrentIPPOBackboneTrainer._ppo_update)
    assert "clipped_values" in source
    assert "critic_loss = 0.5 * torch.maximum" in source


def test_evaluation_separates_two_action_surrogate_from_generic_information_baseline():
    source = inspect.getsource(StandardPairingEvaluator._evaluate_chunk)
    initialization_source = inspect.getsource(StandardPairingEvaluator.__init__)
    decision_branch = source.index(
        'controller == "finite_prototype_two_action_surrogate"'
    )
    decision_call = source.index(
        "select_finite_prototype_two_action_surrogate", decision_branch
    )
    generic_branch = source.index('controller == "generic_response_information"')
    generic_call = source.index("select_response_probe", generic_branch)
    assert decision_branch < decision_call < generic_branch < generic_call
    assert "self.probe_configs[pairing.policy_0_seed]" in source
    assert "self.probe_configs[pairing.policy_1_seed]" in source
    assert "probe_calibration_binding" in initialization_source
    assert (
        "content-addressed, recomputed calibration ledger"
        in initialization_source
    )


def test_no_probe_and_generic_information_keep_the_actor_base_action():
    training_source = inspect.getsource(PathCAdaptationTrainer._choose_ego_actions)
    evaluation_source = inspect.getsource(StandardPairingEvaluator._evaluate_chunk)
    training_end = training_source.index("pending_continuation_values is not None")
    evaluation_end = evaluation_source.index("pending_values is not None")
    training_override = training_source[training_end - 140 : training_end + 40]
    evaluation_override = evaluation_source[evaluation_end - 140 : evaluation_end + 40]
    assert 'controller == "finite_prototype_two_action_surrogate"' in training_override
    assert 'controller == "finite_prototype_two_action_surrogate"' in evaluation_override
    assert 'controller == "off"' not in training_override
    assert 'controller == "off"' not in evaluation_override
    assert '"generic_response_information"' not in training_override
    assert '"generic_response_information"' not in evaluation_override


def test_evaluation_rejects_the_wrong_probe_schema_before_rollout():
    adaptation = PathCAdaptationPolicy(
        (5, 5, 3), recurrent_dim=8, visual_embedding_dim=8,
        conv_channels=(4, 4, 4), n_critic_heads=2,
        response_ensemble_size=2, response_hidden_dim=4,
    )
    legacy = PrimitiveRecurrentEnsembleQ(
        (5, 5, 3), n_heads=2, visual_embedding_dim=8, encoder_dim=8,
        recurrent_dim=8, conv_channels=(4, 4, 4), prior_scale=0.0,
    )
    legacy_probe = {
        "enabled": False,
        "disagreement_threshold": None,
        "max_probe_regret": None,
        "probe_budget_per_episode": None,
        "probe_window_environment_steps": None,
    }
    with pytest.raises(ValueError, match="response-controller"):
        validate_evaluation_probe_config(adaptation, legacy_probe)
    response_probe = {
        "controller": "off", "enabled": False,
        "response_disagreement_threshold": 0.02,
        "advantage_disagreement_threshold": 0.02,
        "max_probe_regret": 1.0, "probe_budget_per_episode": 20,
        "probe_window_environment_steps": 100,
        "response_ensemble_size": 2, "response_hidden_dim": 4,
        "bootstrap_seed": 1, "response_bootstrap_p": 0.5,
        "critic_bootstrap_p": 0.5,
    }
    with pytest.raises(ValueError, match="legacy checkpoints"):
        validate_evaluation_probe_config(legacy, response_probe)


def test_probe_off_is_a_string_in_both_isolation_configs():
    config_dir = Path(__file__).parents[1] / "configs"
    for filename in (
        "path_c_isolation_backbone_probe_off.yaml",
        "path_c_isolation_random_probe_off.yaml",
    ):
        payload = yaml.safe_load((config_dir / filename).read_text(encoding="utf-8"))
        assert payload["probe"]["controller"] == "off"


def test_two_action_surrogate_is_diagnostic_and_formal_method_fails_closed():
    config_dir = Path(__file__).parents[1] / "configs"
    payload = yaml.safe_load(
        (config_dir / "path_c_adaptation_simple.yaml").read_text(encoding="utf-8")
    )
    assert payload["run_kind"] == "diagnostic"
    assert payload["scientific_readout_allowed"] is False
    assert payload["condition_id"] == "finite_prototype_two_action_surrogate"
    assert payload["evidence"]["partner_action_channel"] is False
    assert payload["probe"]["controller"] == "finite_prototype_two_action_surrogate"
    assert payload["probe"]["response_route"] == "use_registered_response"
    source = inspect.getsource(PathCAdaptationTrainer.__init__)
    assert "Formal proposal runs must use official local observations only" in source
    assert "Formal proposal condition_id and controller disagree" in source
    assert "prototype two-action surrogate cannot replace them." in source
