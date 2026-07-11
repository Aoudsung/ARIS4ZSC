from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from src.aris_bellman.replay import EpisodeSequenceReplayBuffer
from src.aris_bellman.specs import GraphSpec, OptionSpec
from experiments.overcooked_v2.path_c_sequence import (
    DecisionEvidenceBuffer,
    EgoEvidenceSpecV1,
    EpisodeBootstrapRecord,
    EpisodeEvidenceBuffer,
    EvidenceBatch,
    RecurrentEnsembleQ,
    SequenceTDBatch,
    collate_episode_records,
    ensemble_diversity_telemetry,
    per_head_masked_mean,
    sequence_td_loss,
    valid_action_dueling,
)
from experiments.overcooked_v2.residual_signature import (
    normalized_advantage_disagreement,
    normalized_advantage_signature,
    residual_control_signature,
)


def _spec() -> EgoEvidenceSpecV1:
    return EgoEvidenceSpecV1(
        observation_dim=4,
        num_primitive_actions=6,
        num_options=3,
        max_primitive_steps_per_decision=4,
        progress_event_dim=2,
        observation_schema="raw_public_obs_v1",
        primitive_action_names=("right", "down", "left", "up", "stay", "interact"),
        option_names=("probe_left", "probe_right", "noop"),
        progress_event_names=("task_progress", "delivery_progress"),
    )


def _encoded_state(
    spec: EgoEvidenceSpecV1,
    *,
    observation_value: float,
    valid_actions=(True, True, False),
    ego_actions=(),
    partner_actions=(),
    option_id=None,
    reward=0.0,
    terminated=False,
    truncated=False,
) -> torch.Tensor:
    return spec.encode_decision(
        observation=torch.full((spec.observation_dim,), float(observation_value)),
        ego_primitive_actions=ego_actions,
        partner_primitive_actions=partner_actions,
        ego_option_id=option_id,
        duration=len(ego_actions),
        reward=reward,
        progress_events=torch.zeros(spec.progress_event_dim),
        valid_actions=torch.tensor(valid_actions, dtype=torch.bool),
        terminated=terminated,
        truncated=truncated,
    )


def _episode_record(
    spec: EgoEvidenceSpecV1,
    episode_id: str,
    *,
    length: int,
    mask: tuple[bool, ...],
):
    bootstrap = EpisodeBootstrapRecord(
        episode_id=episode_id,
        mask=mask,
        bootstrap_p=0.5,
        draw_seed=7,
    )
    episode = EpisodeEvidenceBuffer(spec, bootstrap)
    initial_valid = torch.tensor([True, True, False])
    episode.start(
        _encoded_state(spec, observation_value=0.0),
        initial_valid,
    )
    for decision in range(length):
        is_last = decision == length - 1
        next_valid = torch.tensor([True, True, False])
        next_evidence = _encoded_state(
            spec,
            observation_value=float(decision + 1),
            ego_actions=(decision % spec.num_primitive_actions,),
            partner_actions=((decision + 1) % spec.num_primitive_actions,),
            option_id=decision % 2,
            reward=0.25,
            truncated=is_last,
        )
        episode.append_transition(
            action=decision % 2,
            reward=0.25,
            discount=0.9,
            done=False,
            truncated=is_last,
            next_evidence=next_evidence,
            next_valid_actions=next_valid,
        )
    return episode.to_record()


def test_ego_evidence_spec_is_strict_versioned_and_hash_stable():
    spec = _spec()
    restored = EgoEvidenceSpecV1.from_dict(spec.to_dict())
    assert restored == spec
    assert restored.sha256() == spec.sha256()
    assert len(spec.sha256()) == 64
    assert spec.field_slices()["valid_actions"].stop <= spec.evidence_dim

    nuisance_payload = dict(spec.to_dict())
    nuisance_payload["identity"] = "partner_7"
    with pytest.raises(ValueError, match="forbidden nuisance"):
        EgoEvidenceSpecV1.from_dict(nuisance_payload)

    unknown_payload = dict(spec.to_dict())
    unknown_payload["unregistered_feature"] = 1
    with pytest.raises(ValueError, match="unknown field"):
        EgoEvidenceSpecV1.from_dict(unknown_payload)


def test_evidence_mapping_rejects_nuisance_and_retains_exact_joint_action_order():
    spec = _spec()
    payload = {
        "observation": torch.zeros(4),
        "ego_primitive_actions": [0, 1],
        "partner_primitive_actions": [2, 3],
        "ego_option_id": 1,
        "duration": 2,
        "reward": 0.5,
        "progress_events": [1.0, 0.0],
        "valid_actions": [True, True, False],
        "terminated": False,
        "truncated": False,
    }
    encoded = spec.encode_mapping(payload)
    changed_partner_action = spec.encode_mapping(
        {**payload, "partner_primitive_actions": [3, 2]}
    )
    assert encoded.shape == (spec.evidence_dim,)
    assert not torch.equal(encoded, changed_partner_action)
    assert torch.equal(
        spec.extract_valid_actions(encoded),
        torch.tensor([True, True, False]),
    )

    with pytest.raises(ValueError, match="forbidden nuisance"):
        spec.encode_mapping({**payload, "mechanism_key": "hidden_mode"})
    with pytest.raises(ValueError, match="duration"):
        spec.encode_mapping({**payload, "duration": 1})


def test_decision_buffer_refuses_to_truncate_primitive_history():
    spec = EgoEvidenceSpecV1(
        observation_dim=2,
        num_primitive_actions=3,
        num_options=2,
        max_primitive_steps_per_decision=1,
        progress_event_dim=1,
        observation_schema="raw_public_obs_v1",
        primitive_action_names=("move", "stay", "interact"),
        option_names=("probe", "noop"),
        progress_event_names=("task_progress",),
    )
    buffer = DecisionEvidenceBuffer(spec)
    buffer.append_primitive(0, 1, reward=0.5, progress_event=[1.0])
    encoded = buffer.encode_boundary(
        observation=[0.0, 1.0],
        ego_option_id=0,
        valid_actions=[True, False],
    )
    assert encoded.shape == (spec.evidence_dim,)
    with pytest.raises(ValueError, match="refusing to truncate"):
        buffer.append_primitive(1, 2)


def test_episode_bootstrap_is_deterministic_and_nonempty_without_environment_rng():
    first = EpisodeBootstrapRecord.sample(
        episode_id="episode-17",
        n_heads=5,
        bootstrap_p=0.2,
        manifest_seed=123,
    )
    second = EpisodeBootstrapRecord.sample(
        episode_id="episode-17",
        n_heads=5,
        bootstrap_p=0.2,
        manifest_seed=123,
    )
    assert first == second
    assert first.sha256() == second.sha256()
    assert len(first.mask) == 5
    assert any(first.mask)


def test_episode_collation_preserves_lengths_and_episode_level_masks():
    spec = _spec()
    first = _episode_record(spec, "e0", length=1, mask=(True, False, True))
    second = _episode_record(spec, "e1", length=2, mask=(False, True, True))
    batch = collate_episode_records([first, second])

    assert batch.evidence.evidence.shape == (2, 3, spec.evidence_dim)
    assert batch.actions.shape == (2, 2)
    assert torch.equal(batch.lengths, torch.tensor([1, 2]))
    assert torch.equal(
        batch.bootstrap_mask,
        torch.tensor([[True, False, True], [False, True, True]]),
    )
    assert torch.equal(batch.actions[0], torch.tensor([0, -1]))
    assert not batch.evidence.valid_actions[0, 2].any()


def test_episode_prefix_and_replay_never_resample_individual_transitions():
    spec = _spec()
    first = _episode_record(spec, "e0", length=2, mask=(True, False))
    second = _episode_record(spec, "e1", length=1, mask=(False, True))
    replay = EpisodeSequenceReplayBuffer(capacity=3, seed=4)
    replay.add(first)
    replay.add(second)

    sampled = replay.sample(2)
    assert len(replay) == 3
    assert replay.episode_count == 2
    assert all(record.episode_id in {"e0", "e1"} for record in sampled)
    assert all(record.transition_length in {1, 2} for record in sampled)

    bootstrap = EpisodeBootstrapRecord(
        episode_id="prefix",
        mask=(True,),
        bootstrap_p=1.0,
        draw_seed=2,
    )
    episode = EpisodeEvidenceBuffer(spec, bootstrap)
    episode.start(_encoded_state(spec, observation_value=0.0), [True, True, False])
    prefix = episode.evidence_batch()
    assert prefix.evidence.shape == (1, 1, spec.evidence_dim)
    assert torch.equal(prefix.lengths, torch.tensor([1]))


def test_active_path_c_and_global_gru_share_the_recurrent_sequence_core():
    from experiments.overcooked_v2.train_aris import _build_q_network

    options = [
        OptionSpec(
            id=index,
            name=name,
            kind="noop",
            target_id=None,
            target_pos=None,
            entity_ids=(),
            region_ids=(),
            max_steps=2,
        )
        for index, name in enumerate(("probe", "noop"))
    ]
    graph = GraphSpec(
        layout_name="unit_layout",
        options=options,
        factors=[],
        relevance=np.zeros((2, 0), dtype=bool),
        option_mask=np.ones(2, dtype=bool),
        factor_mask=np.zeros(0, dtype=bool),
        mode_mask=np.zeros((0, 0), dtype=bool),
        route_map={},
    )
    config = {
        "training": {
            "hidden_dim": 8,
            "obs_encoder": "mlp",
            "value_bound": {},
            "max_episode_options": 4,
        },
        "path_c": {
            "evidence_spec": {
                "enable": True,
                "schema_version": "ego_evidence_spec_v1",
                "max_primitive_actions_per_decision": 4,
                "max_episode_decisions": 4,
            },
            "ensemble": {
                "architecture": "recurrent_sequence_v1",
                "n_heads": 2,
                "prior_scale": 0.1,
            },
        },
    }
    path_c_model = _build_q_network("aris_bellman", 4, graph, config)
    global_gru = _build_q_network("global_gru", 4, graph, config)

    assert isinstance(path_c_model, RecurrentEnsembleQ)
    assert isinstance(global_gru, RecurrentEnsembleQ)
    assert path_c_model.n_heads == 2
    assert global_gru.n_heads == 1
    assert global_gru.prior_scale == 0.0
    assert path_c_model.evidence_spec_sha256 == global_gru.evidence_spec_sha256

    golden_config = copy.deepcopy(config)
    golden_config["path_c"]["ensemble"]["n_heads"] = 1
    golden_config["path_c"]["ensemble"]["prior_scale"] = 0.0
    method_k1 = _build_q_network("aris_bellman", 4, graph, golden_config)
    baseline_k1 = _build_q_network("global_gru", 4, graph, golden_config)
    baseline_k1.load_state_dict(method_k1.state_dict())
    valid_actions = torch.ones(2, 3, method_k1.n_actions, dtype=torch.bool)
    valid_actions[1, 2] = False
    evidence = EvidenceBatch(
        torch.randn(2, 3, method_k1.evidence_dim),
        valid_actions,
        torch.tensor([3, 2]),
        spec_sha256=method_k1.evidence_spec_sha256,
    )
    method_trace, method_state = method_k1.forward_sequence(evidence)
    baseline_trace, baseline_state = baseline_k1.forward_sequence(evidence)
    assert torch.equal(method_trace, baseline_trace)
    assert torch.equal(method_state, baseline_state)


def test_episode_buffer_rejects_evidence_transition_mismatch():
    spec = _spec()
    bootstrap = EpisodeBootstrapRecord(
        episode_id="mismatch",
        mask=(True,),
        bootstrap_p=1.0,
        draw_seed=1,
    )
    episode = EpisodeEvidenceBuffer(spec, bootstrap)
    episode.start(
        _encoded_state(spec, observation_value=0.0),
        [True, True, False],
    )
    next_evidence = _encoded_state(
        spec,
        observation_value=1.0,
        ego_actions=(0,),
        partner_actions=(1,),
        option_id=1,
        reward=0.25,
        truncated=True,
    )
    with pytest.raises(ValueError, match="executed action"):
        episode.append_transition(
            action=0,
            reward=0.25,
            discount=0.9,
            done=False,
            truncated=True,
            next_evidence=next_evidence,
            next_valid_actions=[True, True, False],
        )


def test_valid_action_dueling_ignores_invalid_advantage_logits():
    value = torch.tensor([[0.0]])
    advantage = torch.tensor([[1.0, 3.0, 100.0]])
    valid = torch.tensor([[True, True, False]])
    changed_invalid = advantage.clone()
    changed_invalid[0, 2] = -10_000.0

    first = valid_action_dueling(value, advantage, valid)
    second = valid_action_dueling(value, changed_invalid, valid)
    assert torch.allclose(first[:, :2], torch.tensor([[-1.0, 1.0]]))
    assert torch.equal(first[:, :2], second[:, :2])
    assert first[0, 2] < -1.0e8


def test_recurrent_ensemble_shapes_padding_and_k1_support():
    evidence = torch.randn(2, 3, 5)
    valid = torch.tensor(
        [
            [[True, True, False], [True, False, True], [True, True, True]],
            [[True, True, False], [True, True, False], [False, False, False]],
        ]
    )
    batch = EvidenceBatch(evidence, valid, torch.tensor([3, 2]))
    model = RecurrentEnsembleQ(
        evidence_dim=5,
        n_actions=3,
        n_heads=1,
        encoder_dim=7,
        recurrent_dim=11,
        prior_scale=0.0,
    )
    q_values, representation = model.forward_sequence(batch)
    assert q_values.shape == (2, 3, 1, 3)
    assert representation.shape == (2, 3, 11)
    assert torch.all(q_values[1, 2] == 0)
    assert torch.all(representation[1, 2] == 0)


def test_k1_sequence_trace_matches_stepwise_recurrent_golden():
    evidence = torch.randn(1, 4, 5)
    valid = torch.tensor(
        [[[True, True, False], [True, False, True], [True, True, True], [False, True, True]]]
    )
    batch = EvidenceBatch(evidence, valid, torch.tensor([4]))
    model = RecurrentEnsembleQ(
        evidence_dim=5,
        n_actions=3,
        n_heads=1,
        encoder_dim=7,
        recurrent_dim=11,
        prior_scale=0.0,
    )
    sequence_q, _ = model.forward_sequence(batch)

    encoded = model.shared_encoder(evidence)
    gru = model.head_grus[0]
    hidden = torch.zeros(1, model.recurrent_dim)
    golden_steps = []
    for step in range(evidence.shape[1]):
        input_gates = torch.nn.functional.linear(
            encoded[:, step],
            gru.weight_ih_l0,
            gru.bias_ih_l0,
        )
        hidden_gates = torch.nn.functional.linear(
            hidden,
            gru.weight_hh_l0,
            gru.bias_hh_l0,
        )
        input_reset, input_update, input_new = input_gates.chunk(3, dim=-1)
        hidden_reset, hidden_update, hidden_new = hidden_gates.chunk(3, dim=-1)
        reset_gate = torch.sigmoid(input_reset + hidden_reset)
        update_gate = torch.sigmoid(input_update + hidden_update)
        new_gate = torch.tanh(input_new + reset_gate * hidden_new)
        hidden = new_gate + update_gate * (hidden - new_gate)
        state = hidden[:, None, :]
        golden_steps.append(
            valid_action_dueling(
                model.value_heads[0](state),
                model.advantage_heads[0](state),
                valid[:, step : step + 1],
            )
        )
    golden_q = torch.cat(golden_steps, dim=1).unsqueeze(2)
    assert torch.allclose(sequence_q, golden_q)


def test_recurrent_ensemble_rejects_an_unbound_evidence_schema():
    spec = _spec()
    batch = EvidenceBatch(
        torch.zeros(1, 1, spec.evidence_dim),
        torch.tensor([[[True, True, False]]]),
        torch.tensor([1]),
        spec_sha256="0" * 64,
    )
    model = RecurrentEnsembleQ(
        evidence_dim=spec.evidence_dim,
        n_actions=spec.num_options,
        n_heads=1,
        evidence_spec_sha256=spec.sha256(),
    )
    with pytest.raises(ValueError, match="frozen evidence spec"):
        model.forward_sequence(batch)


def test_each_learned_head_has_independent_recurrent_state():
    batch = EvidenceBatch(
        torch.randn(1, 3, 4),
        torch.ones(1, 3, 2, dtype=torch.bool),
        torch.tensor([3]),
    )
    model = RecurrentEnsembleQ(
        evidence_dim=4,
        n_actions=2,
        n_heads=2,
        encoder_dim=5,
        recurrent_dim=6,
    )
    before, _ = model.forward_sequence(batch)
    with torch.no_grad():
        for parameter in model.head_grus[1].parameters():
            parameter.add_(0.5)
    after, _ = model.forward_sequence(batch)
    assert torch.allclose(before[:, :, 0], after[:, :, 0])
    assert not torch.allclose(before[:, :, 1], after[:, :, 1])
    assert model.head_grus[0] is not model.head_grus[1]


def test_recurrent_hidden_state_resets_independently_for_each_episode_row():
    model = RecurrentEnsembleQ(
        evidence_dim=4,
        n_actions=2,
        n_heads=1,
        encoder_dim=5,
        recurrent_dim=6,
    )
    target_episode = torch.randn(1, 3, 4)
    single = EvidenceBatch(
        target_episode,
        torch.ones(1, 3, 2, dtype=torch.bool),
        torch.tensor([3]),
    )
    batched = EvidenceBatch(
        torch.cat((torch.randn(1, 3, 4), target_episode), dim=0),
        torch.ones(2, 3, 2, dtype=torch.bool),
        torch.tensor([3, 3]),
    )
    q_single, _ = model.forward_sequence(single)
    q_batched, _ = model.forward_sequence(batched)
    assert torch.allclose(q_single[0], q_batched[1])


def test_fixed_prior_is_functionally_independent_of_learned_modules():
    batch = EvidenceBatch(
        torch.randn(1, 2, 4),
        torch.ones(1, 2, 2, dtype=torch.bool),
        torch.tensor([2]),
    )
    model = RecurrentEnsembleQ(
        evidence_dim=4,
        n_actions=2,
        n_heads=2,
        encoder_dim=6,
        recurrent_dim=8,
        prior_scale=0.25,
        prior_seed=91,
    )
    prior_hash = model.prior_state_sha256()
    prior_before = model.forward_prior_sequence(batch)
    with torch.no_grad():
        for parameter in model.shared_encoder.parameters():
            parameter.add_(1.0)
        for parameter in model.head_grus[0].parameters():
            parameter.mul_(0.0)
    prior_after = model.forward_prior_sequence(batch)
    model.train()

    assert prior_hash == model.prior_state_sha256()
    assert torch.equal(prior_before, prior_after)
    assert model.prior_encoder is not None and model.prior_encoder.training is False
    assert all(not parameter.requires_grad for parameter in model.prior_encoder.parameters())
    assert all(not parameter.requires_grad for parameter in model.prior_grus.parameters())


def test_online_and_target_must_use_identical_fixed_prior():
    online = RecurrentEnsembleQ(4, 2, n_heads=2, prior_scale=0.1, prior_seed=3)
    target = copy.deepcopy(online)
    online.assert_same_fixed_prior(target)
    with torch.no_grad():
        next(target.prior_grus.parameters()).add_(1.0)
    with pytest.raises(ValueError, match="same fixed prior"):
        online.assert_same_fixed_prior(target)


def test_per_head_loss_normalization_exposes_zero_support_heads():
    losses = torch.tensor(
        [
            [[1.0, 10.0, 2.0], [3.0, 20.0, 4.0]],
            [[5.0, 30.0, 6.0], [7.0, 40.0, 8.0]],
        ]
    )
    transition_mask = torch.tensor([[True, True], [True, False]])
    episode_bootstrap = torch.tensor(
        [[True, False, True], [False, False, True]]
    )
    per_head, support, supported = per_head_masked_mean(
        losses,
        transition_mask,
        episode_bootstrap,
    )
    assert torch.allclose(per_head, torch.tensor([2.0, 0.0, 4.0]))
    assert torch.equal(support, torch.tensor([2, 0, 3]))
    assert torch.equal(supported, torch.tensor([True, False, True]))


class _CountingSequenceQ(torch.nn.Module):
    def __init__(self, q_values: torch.Tensor):
        super().__init__()
        self.register_buffer("q_values", q_values)
        self.n_heads = int(q_values.shape[2])
        self.n_actions = int(q_values.shape[3])
        self.calls = 0

    def forward_sequence(self, batch: EvidenceBatch):
        self.calls += 1
        return self.q_values.to(batch.evidence.device), torch.zeros(
            batch.evidence.shape[0],
            batch.evidence.shape[1],
            1,
            device=batch.evidence.device,
        )

    def assert_same_fixed_prior(self, other):
        del other


def _counting_td_batch(bootstrap=True) -> SequenceTDBatch:
    evidence = EvidenceBatch(
        torch.zeros(1, 3, 2),
        torch.tensor([[[True, True], [True, True], [False, False]]]),
        torch.tensor([3]),
    )
    return SequenceTDBatch(
        evidence=evidence,
        actions=torch.tensor([[0, 0]]),
        rewards=torch.zeros(1, 2),
        discounts=torch.tensor([[0.5, 1.0]]),
        dones=torch.tensor([[False, True]]),
        lengths=torch.tensor([2]),
        bootstrap_mask=torch.tensor([[bootstrap]]),
    )


def test_sequence_td_loss_unrolls_each_network_once_and_uses_aligned_double_q_shift():
    online_q = torch.tensor(
        [[[[0.0, 0.0]], [[1.0, 2.0]], [[-torch.inf, -torch.inf]]]]
    )
    target_q = torch.tensor(
        [[[[0.0, 0.0]], [[30.0, 20.0]], [[-torch.inf, -torch.inf]]]]
    )
    online = _CountingSequenceQ(online_q)
    target = _CountingSequenceQ(target_q)
    output = sequence_td_loss(
        online,
        target,
        _counting_td_batch(),
        td_loss="mse",
        double_q=True,
        require_matching_prior=False,
        return_details=True,
    )
    assert online.calls == 1
    assert target.calls == 1
    assert output.targets.shape == (1, 2, 1)
    assert torch.equal(output.targets[0, :, 0], torch.tensor([10.0, 0.0]))


def test_sequence_td_loss_fails_closed_when_all_heads_have_zero_support():
    q_values = torch.zeros(1, 3, 1, 2)
    with pytest.raises(ValueError, match="No ensemble head"):
        sequence_td_loss(
            _CountingSequenceQ(q_values),
            _CountingSequenceQ(q_values),
            _counting_td_batch(bootstrap=False),
            require_matching_prior=False,
        )


def test_normalized_advantage_is_primary_offset_invariant_control_code():
    q_values = torch.tensor([[5.0, 3.0, 100.0]])
    valid = torch.tensor([[True, True, False]])
    signature = normalized_advantage_signature(q_values, option_mask=valid)
    shifted = normalized_advantage_signature(q_values + 17.0, option_mask=valid)
    assert torch.equal(
        signature["normalized_advantage"],
        torch.tensor([[0.0, -2.0, 0.0]]),
    )
    assert torch.equal(
        signature["normalized_advantage"],
        shifted["normalized_advantage"],
    )
    assert torch.equal(signature["best_option"], torch.tensor([0]))


def test_primary_normalized_advantage_does_not_depend_on_base_only_residual():
    q_values = torch.tensor([[5.0, 3.0, 1.0]])
    valid = torch.tensor([[True, True, True]])
    primary = normalized_advantage_signature(q_values, option_mask=valid)
    residual_a = residual_control_signature(q_values, torch.zeros_like(q_values))
    residual_b = residual_control_signature(
        q_values,
        torch.tensor([[0.0, -100.0, 0.0]]),
    )
    assert not torch.equal(residual_a["advantage"], residual_b["advantage"])
    assert torch.equal(
        primary["normalized_advantage"],
        normalized_advantage_signature(q_values, option_mask=valid)[
            "normalized_advantage"
        ],
    )


def test_normalized_advantage_disagreement_selects_only_valid_actions():
    heads = torch.tensor(
        [[[3.0, 1.0, 1000.0], [1.0, 2.0, -1000.0]]]
    )
    valid = torch.tensor([[True, True, False]])
    disagreement = normalized_advantage_disagreement(
        heads,
        option_mask=valid,
        stat="variance",
    )
    assert disagreement["per_option"].shape == (1, 3)
    assert disagreement["per_option"][0, 2] == 0.0
    assert int(disagreement["per_option"].argmax(dim=-1).item()) == 1


def test_ensemble_diversity_telemetry_uses_only_valid_normalized_advantages():
    q_values = torch.tensor(
        [[
            [3.0, 1.0, 1000.0],
            [1.0, 3.0, -1000.0],
            [2.0, 2.0, 500.0],
        ]]
    )
    valid = torch.tensor([[True, True, False]])
    first = ensemble_diversity_telemetry(q_values, valid)
    changed_invalid = q_values.clone()
    changed_invalid[..., 2] = torch.tensor([[-7.0, 8.0, 9.0]])
    second = ensemble_diversity_telemetry(changed_invalid, valid)

    assert first == second
    assert first["n_heads"] == 3
    assert first["support_size"] == 2
    assert 0.99 <= first["effective_rank"] <= 3.0
    assert len(first["head_correlation_matrix"]) == 3


def test_ensemble_diversity_telemetry_exposes_head_collapse_and_prior_share():
    collapsed = torch.tensor(
        [[[4.0, 2.0], [4.0, 2.0], [4.0, 2.0]]]
    )
    prior = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]]
    )
    telemetry = ensemble_diversity_telemetry(
        collapsed,
        torch.tensor([[True, True]]),
        prior_q_values=prior,
        prior_scale=0.5,
    )

    assert telemetry["effective_rank"] == pytest.approx(1.0)
    assert telemetry["head_correlation_absolute_mean"] == pytest.approx(1.0)
    assert telemetry["prior_scaled_rms"] > 0.0
    assert telemetry["prior_contribution_ratio"] > 0.0


def test_ensemble_diversity_telemetry_reports_zero_rank_for_constant_code():
    telemetry = ensemble_diversity_telemetry(
        torch.zeros(1, 2, 3),
        torch.tensor([[True, True, False]]),
    )
    assert telemetry["effective_rank"] == 0.0
    assert telemetry["head_correlation_absolute_mean"] == 0.0
    with pytest.raises(ValueError, match="prior_q_values"):
        ensemble_diversity_telemetry(
            torch.zeros(1, 2, 3),
            torch.tensor([[True, True, False]]),
            prior_scale=0.1,
        )


def test_training_metrics_archive_ensemble_and_floor_binding_telemetry():
    from experiments.overcooked_v2.train_aris import (
        _metrics_summary,
        _record_path_c_ensemble_telemetry,
    )

    metrics = {
        "path_c_probe_skipped_threshold_count": 2,
        "path_c_probe_skipped_return_floor_count": 1,
        "path_c_probe_selected_count": 1,
    }
    observation = ensemble_diversity_telemetry(
        torch.tensor([[[3.0, 1.0], [1.0, 3.0]]]),
        torch.tensor([[True, True]]),
    )
    _record_path_c_ensemble_telemetry(metrics, observation)
    _record_path_c_ensemble_telemetry(metrics, observation)
    summary = _metrics_summary(metrics)

    assert summary["path_c_probe_floor_binding"]["rate"] == pytest.approx(0.5)
    archived = summary["path_c_ensemble_telemetry_summary"]
    assert archived["observation_count"] == 2
    assert archived["effective_rank_mean"] == pytest.approx(
        observation["effective_rank"]
    )
    assert archived["head_correlation_absolute_mean"] == pytest.approx(
        observation["head_correlation_absolute_mean"]
    )


def test_sequence_replay_contract_rejects_unreachable_warmup_and_long_episodes():
    from experiments.overcooked_v2.train_aris import (
        _validate_sequence_replay_contract,
    )

    valid = {
        "training": {
            "replay_size": 8,
            "warmup_transitions": 4,
            "max_episode_options": 6,
        }
    }
    assert _validate_sequence_replay_contract(valid) == {
        "replay_capacity_transitions": 8,
        "warmup_transitions": 4,
        "maximum_episode_transitions": 6,
    }
    unreachable = copy.deepcopy(valid)
    unreachable["training"]["warmup_transitions"] = 9
    with pytest.raises(ValueError, match="can never reach"):
        _validate_sequence_replay_contract(unreachable)
    oversized_episode = copy.deepcopy(valid)
    oversized_episode["training"]["max_episode_options"] = 9
    with pytest.raises(ValueError, match="complete sequence episode"):
        _validate_sequence_replay_contract(oversized_episode)
