from __future__ import annotations

import inspect
import itertools
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from experiments.overcooked_v2 import path_c_standard
from experiments.overcooked_v2 import path_c_standard_evaluation
from experiments.overcooked_v2 import path_c_standard_training
from experiments.overcooked_v2.env_adapter import OCV2Adapter
from experiments.overcooked_v2.batched_rollout import _validated_uint32_seeds
from experiments.overcooked_v2.path_c_standard import (
    PrimitiveObservationBatch,
    PrimitiveRecurrentEnsembleQ,
    StandardEnvConfig,
    choose_primitive_actions,
    load_standard_checkpoint,
    save_standard_checkpoint,
    shared_team_reward,
    single_step_batch,
    validate_probe_config,
)
from experiments.overcooked_v2.path_c_standard_evaluation import (
    StandardEpisodeReturn,
    StandardEvaluationSpec,
    build_standard_pairings,
    summarize_standard_calibration_rows,
    summarize_standard_rows,
)
from experiments.overcooked_v2.path_c_standard_diagnostics import (
    TRAINING_METRICS_SCHEMA_VERSION,
    TrainingWindowAccumulator,
    model_parameter_health,
    snapshot_trainable_parameters,
)
from experiments.overcooked_v2.path_c_standard_training import (
    PrimitiveEpisodeBuilder,
    StandardTrainingSpec,
    collate_primitive_episodes,
    primitive_sequence_td_loss,
    select_partner_actions,
)


def _environment(layout: str = "test_time_simple") -> dict:
    return {
        "schema_version": "path_c_standard_env_v1",
        "env": {
            "layout": layout,
            "max_steps": 400,
            "observation_type": "default",
            "agent_view_size": 2,
            "negative_rewards": True,
            "random_agent_positions": True,
            "sample_recipe_on_delivery": True,
            "indicate_successful_delivery": True,
            "force_path_planning": False,
            "random_reset": False,
        },
    }


def _observation_batch(batch_size: int = 2, steps: int = 3) -> PrimitiveObservationBatch:
    time_mask = torch.ones(batch_size, steps, dtype=torch.bool)
    return PrimitiveObservationBatch(
        observations=torch.zeros(batch_size, steps, 5, 5, 39),
        previous_actions=torch.full((batch_size, steps), 6, dtype=torch.long),
        previous_rewards=torch.zeros(batch_size, steps),
        episode_starts=torch.zeros(batch_size, steps, dtype=torch.bool),
        valid_actions=torch.ones(batch_size, steps, 6, dtype=torch.bool),
        time_mask=time_mask,
        lengths=torch.full((batch_size,), steps, dtype=torch.long),
    )


def test_standard_environment_matches_published_test_time_semantics():
    for layout in ("test_time_simple", "test_time_wide"):
        config = StandardEnvConfig.from_mapping(_environment(layout))
        assert config.layout == layout
        assert config.max_steps == 400
        assert config.agent_view_size == 2
        assert config.observation_type == "default"
        assert config.indicate_successful_delivery is True
        assert config.force_path_planning is False

    invalid = _environment()
    invalid["env"]["indicate_successful_delivery"] = False
    with pytest.raises(ValueError, match="indicate_successful_delivery"):
        StandardEnvConfig.from_mapping(invalid)


def test_adapter_exposes_delivery_indicator_and_slot_neutral_step():
    parameters = inspect.signature(OCV2Adapter.__init__).parameters
    assert parameters["indicate_successful_delivery"].default is False
    assert callable(OCV2Adapter.step_joint)
    assert callable(OCV2Adapter.step_joint_from_state)


def test_standard_environment_files_are_exact_and_do_not_add_privileged_inputs():
    config_dir = Path(path_c_standard.__file__).parent / "configs"
    for filename, layout in (
        ("ocv2_test_time_simple_standard.yaml", "test_time_simple"),
        ("ocv2_test_time_wide_standard.yaml", "test_time_wide"),
    ):
        payload = yaml.safe_load((config_dir / filename).read_text(encoding="utf-8"))
        parsed = StandardEnvConfig.from_mapping(payload)
        assert parsed.layout == layout
        assert set(payload["env"]) == {
            "layout",
            "max_steps",
            "observation_type",
            "agent_view_size",
            "negative_rewards",
            "random_agent_positions",
            "sample_recipe_on_delivery",
            "indicate_successful_delivery",
            "force_path_planning",
            "random_reset",
        }


def test_standard_layouts_expose_the_published_local_observation_shapes():
    assert StandardEnvConfig(layout="test_time_simple").make_adapter().env.obs_shape == (
        5,
        5,
        39,
    )
    assert StandardEnvConfig(layout="test_time_wide").make_adapter().env.obs_shape == (
        5,
        5,
        43,
    )


def test_standard_policy_contract_contains_only_agent_local_evidence():
    field_names = {field.name for field in fields(PrimitiveObservationBatch)}
    assert field_names == {
        "observations",
        "previous_actions",
        "previous_rewards",
        "episode_starts",
        "valid_actions",
        "time_mask",
        "lengths",
    }
    source = "\n".join(
        (
            inspect.getsource(path_c_standard),
            inspect.getsource(path_c_standard_training),
            inspect.getsource(path_c_standard_evaluation),
        )
    )
    for forbidden_import in (
        "obs_featurizer",
        "option_executor",
        "partner_pool",
        "sparse_credit",
        "evaluate_aris",
        "ce_sampler",
    ):
        assert f"import {forbidden_import}" not in source
        assert f"from experiments.overcooked_v2.{forbidden_import}" not in source


def test_primitive_recurrent_model_unrolls_and_carries_hidden_state():
    model = PrimitiveRecurrentEnsembleQ(
        [5, 5, 39],
        n_heads=2,
        visual_embedding_dim=16,
        encoder_dim=12,
        recurrent_dim=10,
        conv_channels=[8, 8, 4],
        prior_scale=0.1,
        prior_seed=7,
    )
    batch = _observation_batch()
    q_values, representation = model.forward_sequence(batch)
    assert q_values.shape == (2, 3, 2, 6)
    assert representation.shape == (2, 3, 20)

    step_batch = PrimitiveObservationBatch(
        observations=batch.observations[:, :1],
        previous_actions=batch.previous_actions[:, :1],
        previous_rewards=batch.previous_rewards[:, :1],
        episode_starts=torch.ones(2, 1, dtype=torch.bool),
        valid_actions=batch.valid_actions[:, :1],
        time_mask=batch.time_mask[:, :1],
        lengths=torch.ones(2, dtype=torch.long),
    )
    state = model.initial_state(2, device="cpu")
    step_q, step_representation, next_state = model.forward_step(step_batch, state)
    assert step_q.shape == (2, 2, 6)
    assert step_representation.shape == (2, 20)
    assert next_state.learned.hidden[0].shape == (1, 2, 10)


def test_probe_selection_uses_ensemble_disagreement_over_primitive_actions():
    q_values = torch.tensor(
        [
                [
                    [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [2.0, 1.5, 0.0, 0.0, 0.0, 0.0],
                ]
        ]
    )
    decision = choose_primitive_actions(
        q_values,
        torch.ones(1, 6, dtype=torch.bool),
        rng=np.random.default_rng(3),
        epsilon=0.0,
        probe_enabled=True,
        disagreement_threshold=1e-4,
    )
    assert decision.actions.tolist() == [1]
    assert decision.is_probe.tolist() == [True]
    assert decision.is_exploration.tolist() == [False]


def test_probing_rejects_thresholds_that_would_probe_on_every_step():
    q_values = torch.zeros(1, 2, 6)
    for degenerate_threshold in (None, 0.0, -0.5):
        with pytest.raises(ValueError, match="positive disagreement_threshold"):
            choose_primitive_actions(
                q_values,
                torch.ones(1, 6, dtype=torch.bool),
                rng=np.random.default_rng(0),
                epsilon=0.0,
                probe_enabled=True,
                disagreement_threshold=degenerate_threshold,
            )


def test_probe_config_validation_is_fail_closed():
    with pytest.raises(ValueError, match="explicit probe block"):
        validate_probe_config(None)
    with pytest.raises(ValueError, match="explicit boolean"):
        validate_probe_config({})
    with pytest.raises(ValueError, match="disagreement_threshold"):
        validate_probe_config({"enabled": True})
    with pytest.raises(ValueError, match="strictly positive"):
        validate_probe_config({"enabled": True, "disagreement_threshold": 0.0})
    with pytest.raises(ValueError, match="Unknown probe field"):
        validate_probe_config({"enabled": False, "threshold": 1.0})
    disabled = validate_probe_config({"enabled": False})
    assert disabled["enabled"] is False
    enabled = validate_probe_config(
        {"enabled": True, "disagreement_threshold": 0.02, "return_floor": 0.0}
    )
    assert enabled == {
        "enabled": True,
        "disagreement_threshold": 0.02,
        "return_floor": 0.0,
        "disagreement_stat": "variance",
    }


def test_shared_team_reward_asserts_the_common_reward_invariant():
    common = {"agent_0": np.asarray([1.0, -0.5]), "agent_1": np.asarray([1.0, -0.5])}
    assert shared_team_reward(common).tolist() == [1.0, -0.5]
    split = {"agent_0": np.asarray([1.0, -0.5]), "agent_1": np.asarray([1.0, 0.5])}
    with pytest.raises(ValueError, match="shared team reward"):
        shared_team_reward(split)


def test_step_unroll_matches_sequence_forward_on_one_episode_rows():
    torch.manual_seed(5)
    model = PrimitiveRecurrentEnsembleQ(
        [5, 5, 7],
        n_heads=2,
        visual_embedding_dim=12,
        encoder_dim=10,
        recurrent_dim=8,
        conv_channels=[6, 6, 4],
        prior_scale=0.1,
        prior_seed=13,
    )
    model.eval()
    batch_size, steps = 2, 4
    torch.manual_seed(11)
    observations = torch.rand(batch_size, steps, 5, 5, 7)
    previous_actions = torch.randint(0, 7, (batch_size, steps))
    previous_actions[:, 0] = 6
    previous_rewards = torch.randn(batch_size, steps)
    episode_starts = torch.zeros(batch_size, steps, dtype=torch.bool)
    episode_starts[:, 0] = True
    batch = PrimitiveObservationBatch(
        observations=observations,
        previous_actions=previous_actions,
        previous_rewards=previous_rewards,
        episode_starts=episode_starts,
        valid_actions=torch.ones(batch_size, steps, 6, dtype=torch.bool),
        time_mask=torch.ones(batch_size, steps, dtype=torch.bool),
        lengths=torch.full((batch_size,), steps, dtype=torch.long),
    )
    with torch.no_grad():
        sequence_q, sequence_representation = model.forward_sequence(batch)
    state = model.initial_state(batch_size, device="cpu")
    for step in range(steps):
        step_batch = PrimitiveObservationBatch(
            observations=observations[:, step : step + 1],
            previous_actions=previous_actions[:, step : step + 1],
            previous_rewards=previous_rewards[:, step : step + 1],
            episode_starts=episode_starts[:, step : step + 1],
            valid_actions=batch.valid_actions[:, step : step + 1],
            time_mask=batch.time_mask[:, step : step + 1],
            lengths=torch.ones(batch_size, dtype=torch.long),
        )
        with torch.no_grad():
            step_q, step_representation, state = model.forward_step(step_batch, state)
        assert torch.allclose(step_q, sequence_q[:, step], atol=1e-5)
        assert torch.allclose(
            step_representation,
            sequence_representation[:, step],
            atol=1e-5,
        )


def test_episode_td_batch_uses_raw_rewards_and_episode_level_head_masks():
    builders = []
    for episode_index in range(2):
        builder = PrimitiveEpisodeBuilder(
            episode_id=f"episode-{episode_index}",
            initial_observation=np.zeros((5, 5, 39), dtype=np.uint8),
            n_actions=6,
            n_heads=2,
            bootstrap_p=0.5,
            bootstrap_seed=11,
        )
        builder.append(
            action=episode_index,
            reward=-1.25 + episode_index,
            done=True,
            next_observation=np.ones((5, 5, 39), dtype=np.uint8),
        )
        builders.append(builder.finish(observation_shape=(5, 5, 39)))
    batch = collate_primitive_episodes(
        builders,
        observation_shape=(5, 5, 39),
        n_actions=6,
        n_heads=2,
        gamma=0.99,
        device="cpu",
    )
    assert batch.rewards[:, 0].tolist() == pytest.approx([-1.25, -0.25])
    assert batch.bootstrap_mask.shape == (2, 2)

    online = PrimitiveRecurrentEnsembleQ(
        [5, 5, 39],
        n_heads=2,
        visual_embedding_dim=8,
        encoder_dim=8,
        recurrent_dim=8,
        conv_channels=[4, 4, 4],
    )
    target = PrimitiveRecurrentEnsembleQ.from_manifest(online.architecture_manifest())
    target.load_state_dict(online.state_dict())
    output = primitive_sequence_td_loss(online, target, batch)
    assert output.loss.ndim == 0
    assert output.head_support.shape == (2,)


def test_standard_checkpoint_binds_architecture_and_effective_data_budget(tmp_path):
    model = PrimitiveRecurrentEnsembleQ(
        [5, 5, 39],
        n_heads=2,
        visual_embedding_dim=8,
        encoder_dim=8,
        recurrent_dim=8,
        conv_channels=[4, 4, 4],
    )
    path = tmp_path / "final.pt"
    save_standard_checkpoint(
        path,
        model,
        seed=17,
        environment_steps=30_000_000,
        episodes=75_000,
        phase="path_c_final",
        extra_metadata={"layout": "test_time_simple"},
    )
    restored, metadata = load_standard_checkpoint(path, device="cpu")
    assert restored.architecture_manifest() == model.architecture_manifest()
    assert metadata["environment_steps"] == 30_000_000
    assert metadata["episodes"] == 75_000


def test_training_spec_separates_smoke_from_formal_data_budget(tmp_path):
    base = {
        "run_kind": "smoke",
        "scientific_readout_allowed": False,
        "seed": 1,
        "output_dir": str(tmp_path),
        "training": {
            "total_environment_steps": 1600,
            "self_play_environment_steps": 800,
            "path_c_environment_steps": 800,
            "batch_size_envs": 2,
            "partner_pool_snapshot_steps": [800],
            "ego_initialization": "self_play_final",
            "replay_capacity_episodes": 8,
            "minibatch_episodes": 2,
            "learning_starts_episodes": 2,
            "updates_per_vector_step": 1,
            "target_update_environment_steps": 400,
            "epsilon_decay_environment_steps": 800,
        },
    }
    assert StandardTrainingSpec.from_mapping(base).run_kind == "smoke"
    invalid = dict(base)
    invalid["scientific_readout_allowed"] = True
    with pytest.raises(ValueError, match="Smoke"):
        StandardTrainingSpec.from_mapping(invalid)

    formal = {**base, "run_kind": "formal", "scientific_readout_allowed": True}
    with pytest.raises(ValueError, match="30,000,000"):
        StandardTrainingSpec.from_mapping(formal)


def test_training_window_records_loss_reward_return_and_parameter_health():
    torch.manual_seed(29)
    model = PrimitiveRecurrentEnsembleQ(
        [5, 5, 7],
        n_heads=2,
        visual_embedding_dim=10,
        encoder_dim=8,
        recurrent_dim=6,
        conv_channels=[4, 4, 4],
        prior_scale=0.1,
        prior_seed=3,
    )
    reference = snapshot_trainable_parameters(model)
    with torch.no_grad():
        next(parameter for parameter in model.parameters() if parameter.requires_grad).add_(0.5)
    torch_rng_before = torch.random.get_rng_state().clone()
    numpy_rng_before = np.random.get_state()
    accumulator = TrainingWindowAccumulator(
        batch_size=2,
        n_heads=2,
        gradient_clip_norm=1.0,
        reference=reference,
    )
    accumulator.record_environment_batch(
        np.asarray([1.0, -1.0]),
        np.asarray([False, False]),
        probe_count=1,
    )
    accumulator.record_environment_batch(
        np.asarray([0.0, 2.0]),
        np.asarray([True, True]),
        probe_count=2,
    )
    accumulator.record_update(
        loss=0.25,
        per_head_loss=(0.2, 0.3),
        per_head_support=(4, 5),
        gradient_norm=2.0,
    )
    row = accumulator.build_row(
        seed=101,
        layout="test_time_simple",
        phase="path_c",
        phase_environment_steps=4,
        total_environment_steps=10_000_004,
        cumulative_episodes=2,
        gradient_updates=1,
        epsilon=0.05,
        cumulative_probe_count=3,
        model=model,
    )
    assert row["schema_version"] == TRAINING_METRICS_SCHEMA_VERSION
    assert row["window_environment_steps"] == 4
    assert row["td_loss"]["mean"] == pytest.approx(0.25)
    assert row["raw_step_reward"] == {
        "sum": 2.0,
        "mean_per_environment_step": 0.5,
        "positive_event_count": 2,
        "negative_event_count": 1,
        "zero_event_count": 1,
    }
    assert row["raw_episode_return"]["count"] == 2
    assert row["raw_episode_return"]["mean"] == pytest.approx(1.0)
    assert row["gradient_l2_norm_before_clipping"]["clipped_fraction"] == 1.0
    assert (
        row["model_parameter_health"]["learned"]["delta_l2_from_reference"]
        > 0.0
    )
    assert row["model_parameter_health"]["fixed_prior"]["nonfinite_count"] == 0
    assert torch.equal(torch.random.get_rng_state(), torch_rng_before)
    numpy_rng_after = np.random.get_state()
    assert numpy_rng_after[0] == numpy_rng_before[0]
    assert np.array_equal(numpy_rng_after[1], numpy_rng_before[1])
    assert numpy_rng_after[2:] == numpy_rng_before[2:]


def test_parameter_health_covers_every_standard_model_group():
    model = PrimitiveRecurrentEnsembleQ(
        [5, 5, 7],
        n_heads=2,
        visual_embedding_dim=10,
        encoder_dim=8,
        recurrent_dim=6,
        conv_channels=[4, 4, 4],
        prior_scale=0.1,
        prior_seed=3,
    )
    health = model_parameter_health(model)
    assert set(health["groups"]) == {
        "observation_encoder",
        "shared_encoder",
        "recurrent_heads",
        "value_heads",
        "advantage_heads",
    }
    assert all(item["parameter_count"] > 0 for item in health["groups"].values())
    assert health["learned"]["nonfinite_count"] == 0
    assert health["fixed_prior_state_sha256"] == model.prior_state_sha256()


def test_calibration_spec_accepts_one_formal_checkpoint_and_five_diagnostics(tmp_path):
    payload = {
        "run_kind": "calibration",
        "scientific_readout_allowed": False,
        "evaluation_seed": 9101,
        "output_dir": str(tmp_path / "calibration"),
        "checkpoint_paths": [str(tmp_path / "path_c_final.pt")],
        "diagnostic_checkpoint_paths": [
            str(tmp_path / f"checkpoint_{index}.pt") for index in range(5)
        ],
        "evaluation": {
            "episodes_per_pairing": 500,
            "evaluation_batch_size": 250,
            "expected_policy_count": 1,
            "standard_deviation_ddof": 0,
            "bootstrap_replicates": 100,
            "bootstrap_confidence": 0.95,
        },
    }
    spec = StandardEvaluationSpec.from_mapping(payload)
    assert spec.run_kind == "calibration"
    assert spec.episodes_per_pairing == 500
    assert len(spec.checkpoint_paths) == 1
    assert len(spec.diagnostic_checkpoint_paths) == 5


def test_calibration_summary_uses_raw_self_play_returns_only():
    rows = [
        StandardEpisodeReturn(
            schema_version="path_c_standard_episode_return_v1",
            layout="test_time_simple",
            split="sp",
            policy_0_seed=101,
            policy_1_seed=101,
            episode_index=index,
            canonical_episode_seed=1000 + index,
            raw_episode_return=value,
            environment_steps=400,
            policy_0_probe_count=index,
            policy_1_probe_count=index + 1,
        )
        for index, value in enumerate((0.0, 20.0))
    ]
    summary = summarize_standard_calibration_rows(
        rows,
        policy_seed=101,
        episodes_per_pairing=2,
    )
    assert summary["raw_return_mean"] == pytest.approx(10.0)
    assert summary["zero_return_fraction"] == pytest.approx(0.5)
    assert summary["positive_return_fraction"] == pytest.approx(0.5)
    assert summary["return_definition"] == (
        "sum_of_raw_rewards_agent_0_over_400_steps"
    )


def test_pairing_matrix_has_ten_sp_and_ninety_directed_xp_pairings():
    pairings = build_standard_pairings(range(10))
    assert len(pairings) == 100
    assert sum(item.split == "sp" for item in pairings) == 10
    assert sum(item.split == "xp" for item in pairings) == 90
    assert _pairing_set(pairings) == _pairing_set(
        [type(pairings[0])(seed, seed) for seed in range(10)]
        + [
            type(pairings[0])(first, second)
            for first, second in itertools.permutations(range(10), 2)
        ]
    )


def _pairing_set(items):
    return {(item.policy_0_seed, item.policy_1_seed) for item in items}


def test_partner_sub_batching_matches_full_batch_forwarding():
    torch.manual_seed(23)
    partner_pool = [
        PrimitiveRecurrentEnsembleQ(
            [5, 5, 7],
            n_heads=2,
            visual_embedding_dim=10,
            encoder_dim=8,
            recurrent_dim=6,
            conv_channels=[4, 4, 4],
            prior_scale=0.1,
            prior_seed=pool_index,
        ).eval()
        for pool_index in range(2)
    ]
    batch_size = 4
    partner_indices = np.asarray([0, 1, 0, 1], dtype=np.int64)
    sub_states = [
        partner.initial_state(batch_size, device="cpu") for partner in partner_pool
    ]
    full_states = [
        partner.initial_state(batch_size, device="cpu") for partner in partner_pool
    ]
    rng_state = np.random.default_rng(0)
    for step in range(3):
        step_rng = np.random.default_rng(step)
        observations = step_rng.random((batch_size, 5, 5, 7), dtype=np.float32)
        previous_actions = step_rng.integers(0, 6, size=batch_size, dtype=np.int64)
        previous_rewards = step_rng.random(batch_size, dtype=np.float32)
        episode_starts = np.asarray([step == 0] * batch_size, dtype=bool)

        sub_actions, sub_states = select_partner_actions(
            partner_pool,
            sub_states,
            partner_indices,
            observations,
            previous_actions,
            previous_rewards,
            episode_starts,
            n_actions=6,
            device="cpu",
            rng=rng_state,
        )

        # Reference semantics: every partner forwards the full batch and only
        # its assigned rows are kept (the pre-optimization implementation).
        full_actions = np.zeros(batch_size, dtype=np.int64)
        step_batch = single_step_batch(
            observations,
            previous_actions,
            previous_rewards,
            episode_starts,
            n_actions=6,
            device="cpu",
        )
        for pool_index, partner in enumerate(partner_pool):
            with torch.no_grad():
                q_values, _, full_states[pool_index] = partner.forward_step(
                    step_batch,
                    full_states[pool_index],
                )
            greedy = (
                q_values.mean(dim=1).argmax(dim=-1).numpy().astype(np.int64)
            )
            mask = partner_indices == pool_index
            full_actions[mask] = greedy[mask]

        assert sub_actions.tolist() == full_actions.tolist()


def test_uint32_seed_casts_refuse_silent_wraparound():
    assert _validated_uint32_seeds([0, 1, 2**32 - 1]).dtype == np.uint32
    with pytest.raises(ValueError, match="2\\*\\*32-1"):
        _validated_uint32_seeds([-1])
    with pytest.raises(ValueError, match="2\\*\\*32-1"):
        _validated_uint32_seeds([2**32])
    with pytest.raises(TypeError, match="integers"):
        _validated_uint32_seeds([1.5])


def test_evaluation_batch_size_must_divide_the_episode_schedule(tmp_path):
    checkpoints = [str(tmp_path / f"seed_{index}.pt") for index in range(2)]
    payload = {
        "run_kind": "smoke",
        "scientific_readout_allowed": False,
        "evaluation_seed": 7,
        "output_dir": str(tmp_path),
        "checkpoint_paths": checkpoints,
        "evaluation": {
            "episodes_per_pairing": 3,
            "evaluation_batch_size": 2,
            "expected_policy_count": 2,
        },
    }
    with pytest.raises(ValueError, match="divide episodes_per_pairing"):
        StandardEvaluationSpec.from_mapping(payload)


def test_summary_aggregates_raw_returns_over_pairings_not_scripted_partners():
    seeds = (0, 1, 2)
    rows = []
    for pairing in build_standard_pairings(seeds):
        for episode_index in range(2):
            rows.append(
                StandardEpisodeReturn(
                    schema_version="path_c_standard_episode_return_v1",
                    layout="test_time_simple",
                    split=pairing.split,
                    policy_0_seed=pairing.policy_0_seed,
                    policy_1_seed=pairing.policy_1_seed,
                    episode_index=episode_index,
                    canonical_episode_seed=episode_index,
                    raw_episode_return=float(
                        10 * pairing.policy_0_seed
                        + pairing.policy_1_seed
                        + episode_index
                    ),
                    environment_steps=400,
                    policy_0_probe_count=0,
                    policy_1_probe_count=0,
                )
            )
    summary = summarize_standard_rows(
        rows,
        policy_seeds=seeds,
        episodes_per_pairing=2,
        standard_deviation_ddof=0,
        bootstrap_replicates=100,
        bootstrap_confidence=0.95,
        bootstrap_seed=9,
    )
    assert summary["sp_pairing_count"] == 3
    assert summary["xp_directed_pairing_count"] == 6
    assert summary["raw_episode_row_count"] == 18
    assert summary["return_definition"] == (
        "sum_of_raw_rewards_agent_0_over_400_steps"
    )
