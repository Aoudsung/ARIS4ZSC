from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("flax")
pytest.importorskip("optax")
from flax.core import freeze, unfreeze

from src.path_c.vqbc.codebook import (
    empty_codebook,
    target_response_signatures,
    update_codebook,
)
from src.path_c.vqbc.checkpoint import (
    load_vqbc_checkpoint,
    save_vqbc_checkpoint,
)
from src.path_c.vqbc.config import VQBCConfig, VQBCModelConfig
from src.path_c.vqbc.integrity import get_parameter_leaf
from src.path_c.vqbc.model import (
    build_vqbc_model,
    explicit_official_parameter_mapping,
    initialize_vqbc_from_official,
)
from src.path_c.vqbc.objectives import (
    FrozenAssignments,
    bellman_targets,
    episode_responsibilities,
    outcome_loss,
    sample_bootstrap_mask,
)
from src.path_c.vqbc.types import (
    VQBCKLState,
    VQBCPolicyState,
    VQBCRolloutBatch,
    VQBCRolloutState,
    VQBCTrainState,
)
from src.path_c.vqbc.training import (
    rollout_health_diagnostics,
    update_kl_state,
)


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"


def _model_config() -> VQBCModelConfig:
    return VQBCModelConfig(
        hidden_dim=128,
        head_hidden_dim=128,
        action_embedding_dim=16,
        slot_embedding_dim=16,
        response_embedding_dim=16,
        slot_count=8,
        response_count=16,
        action_count=6,
        prior_scale=0.01,
        log_standard_deviation_minimum=-5.0,
        log_standard_deviation_maximum=2.0,
    )


def _model_and_inputs() -> tuple:
    model = build_vqbc_model(
        model_config=_model_config(),
        official_dimensions={
            "gru_hidden_dim": 128,
            "activation": "relu",
        },
    )
    observations = jnp.ones((2, 2, 5, 5, 39), dtype=jnp.float32)
    actions = jnp.full((2, 2), 6, dtype=jnp.int32)
    rewards = jnp.zeros((2, 2), dtype=jnp.float32)
    starts = jnp.asarray([[True, True], [False, False]])
    params = model.init(
        jax.random.PRNGKey(0),
        model.initial_carry(2),
        observations,
        actions,
        rewards,
        starts,
    )["params"]
    return model, params, observations, actions, rewards, starts


def _tree_l1(tree: object) -> float:
    return float(
        sum(
            np.abs(np.asarray(value)).sum()
            for value in jax.tree_util.tree_leaves(tree)
        )
    )


def test_model_output_shapes_and_zero_initialized_outcome() -> None:
    model, params, observations, actions, rewards, starts = _model_and_inputs()
    unused_carry, output = model.apply(
        {"params": params},
        model.initial_carry(2),
        observations,
        actions,
        rewards,
        starts,
    )
    del unused_carry
    assert output["q_values"].shape == (2, 2, 2, 8, 6)
    assert output["response_logits"].shape == (2, 2, 8, 6, 16)
    assert output["next_q_mean"].shape == (2, 2, 2, 8, 6, 16, 6)
    np.testing.assert_array_equal(
        output["reward_mean"], np.zeros(output["reward_mean"].shape)
    )
    np.testing.assert_array_equal(
        output["next_q_mean"], np.zeros(output["next_q_mean"].shape)
    )
    np.testing.assert_allclose(output["response_probabilities"], 1.0 / 16.0)
    terminal = output["next_q_mean"][..., 15, :]
    np.testing.assert_array_equal(terminal, np.zeros(terminal.shape))


def test_only_bellman_path_reaches_backbone_and_random_prior_is_frozen() -> None:
    model, params, observations, actions, rewards, starts = _model_and_inputs()

    def q_loss(candidate: object) -> object:
        unused_carry, output = model.apply(
            {"params": candidate},
            model.initial_carry(2),
            observations,
            actions,
            rewards,
            starts,
        )
        del unused_carry
        return jnp.sum(jnp.square(output["q_values"]))

    q_grad = jax.grad(q_loss)(params)
    assert _tree_l1(q_grad["backbone"]) > 0.0
    assert _tree_l1(q_grad["q_heads"]["PriorEstimator_0"]) == 0.0
    assert _tree_l1(q_grad["q_heads"]["PriorEstimator_1"]) == 0.0

    outcome_probe = unfreeze(params)
    outcome_probe["outcome"]["hidden"]["kernel"] = jnp.ones_like(
        outcome_probe["outcome"]["hidden"]["kernel"]
    )
    outcome_probe["outcome"]["response_logits"]["kernel"] = jnp.ones_like(
        outcome_probe["outcome"]["response_logits"]["kernel"]
    )
    outcome_probe = freeze(outcome_probe)

    def detached_outcome_loss(candidate: object) -> object:
        unused_carry, output = model.apply(
            {"params": candidate},
            model.initial_carry(2),
            observations,
            actions,
            rewards,
            starts,
        )
        del unused_carry
        return (
            jnp.sum(output["response_logits"])
            + jnp.sum(output["reward_mean"])
            + jnp.sum(output["next_q_mean"])
        )

    outcome_grad = jax.grad(detached_outcome_loss)(outcome_probe)
    assert _tree_l1(outcome_grad["backbone"]) == 0.0


def _set_nested(tree: dict, path: tuple[str, ...], value: object) -> None:
    target = tree
    for component in path[:-1]:
        target = target.setdefault(component, {})
    target[path[-1]] = value


def test_official_initialization_adds_response_encoder_without_actor() -> None:
    model, params, observations, actions, rewards, starts = _model_and_inputs()
    official: dict = {}
    mapping = explicit_official_parameter_mapping()
    for target_path, source_path in mapping.items():
        _set_nested(
            official, source_path, get_parameter_leaf(params, target_path)
        )
    initialized = initialize_vqbc_from_official(
        model,
        random_key=jax.random.PRNGKey(7),
        official_params=official,
        example_observations=observations,
        example_previous_actions=actions,
        example_previous_team_rewards=rewards,
        example_episode_start=starts,
    )
    assert "response_encoder" in initialized
    assert all("actor" not in "/".join(path).lower() for path in mapping)


def test_double_q_target_and_episode_responsibility_match_hand_values() -> None:
    next_q = jnp.asarray(
        [
            [
                [[3.0, 1.0], [2.0, 4.0]],
                [[1.0, 5.0], [4.0, 0.0]],
            ]
        ]
    )
    probabilities = jnp.asarray([[0.25, 0.75]])
    targets = bellman_targets(
        rewards=jnp.asarray([1.0]),
        dones=jnp.asarray([False]),
        next_execution_probabilities=probabilities,
        target_next_q_values=next_q,
        gamma=0.5,
    )
    minimum = np.minimum(np.asarray(next_q[0, 0]), np.asarray(next_q[0, 1]))
    expected = 1.0 + 0.5 * (minimum * np.asarray([0.25, 0.75])).sum(-1)
    np.testing.assert_allclose(targets[0], expected)
    q_values = jnp.zeros((1, 1, 2, 2, 2))
    responsibilities = episode_responsibilities(
        q_values=q_values,
        actions=jnp.asarray([[0]]),
        targets=targets[:, None, :],
        temperature=1.0,
    )
    np.testing.assert_allclose(responsibilities.sum(-1), 1.0)


def test_bootstrap_fallback_keeps_one_slot_and_is_keyed() -> None:
    mask = sample_bootstrap_mask(
        jax.random.PRNGKey(9),
        environment_count=32,
        slot_count=8,
        probability=1.0e-8,
    )
    assert bool(jnp.all(jnp.any(mask, axis=-1)))


def test_codebook_uses_nonterminal_signatures_and_replaces_stale_codes() -> None:
    state = empty_codebook(code_count=3, signature_dim=2)
    signatures = jnp.asarray(
        [[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [3.0, 3.0]]]
    )
    update = update_codebook(
        state,
        signatures=signatures,
        valid_mask=jnp.asarray([[True, True, True, False]]),
        key=jax.random.PRNGKey(4),
        decay=0.99,
        replacement_after_rollouts=10,
    )
    assert bool(update.state.initialized)
    assert update.state.embeddings.shape == (3, 2)


def test_response_signature_stops_responsibility_and_averages_twins() -> None:
    advantages = jnp.asarray(
        [
            [
                [
                    [[1.0, 0.0], [0.0, 1.0]],
                    [[3.0, 0.0], [0.0, 3.0]],
                ]
            ]
        ]
    )
    responsibilities = jnp.asarray([[0.25, 0.75]])
    signature = target_response_signatures(
        target_centered_advantages=advantages,
        stopped_responsibilities=responsibilities,
    )
    np.testing.assert_allclose(signature, [[[0.5, 1.5]]])


def test_slot_outcome_loss_uses_only_stopped_responsibility_weights() -> None:
    response_logits = jnp.zeros((1, 1, 2, 1, 16))
    reward_mean = jnp.asarray([[[[0.0], [10.0]]]])
    reward_log_std = jnp.zeros_like(reward_mean)
    next_q_mean = jnp.zeros((1, 1, 2, 2, 1, 16, 1))
    target_next_q = jnp.zeros((1, 1, 2, 2, 1))
    common = {
        "response_logits": response_logits,
        "reward_mean": reward_mean,
        "reward_log_standard_deviation": reward_log_std,
        "next_q_mean": next_q_mean,
        "next_q_log_standard_deviation": jnp.zeros_like(next_q_mean),
        "response_codes": jnp.zeros((1, 1), dtype=jnp.int32),
        "rewards": jnp.zeros((1, 1)),
        "target_next_q_values": target_next_q,
        "dones": jnp.zeros((1, 1), dtype=jnp.bool_),
        "actions": jnp.zeros((1, 1), dtype=jnp.int32),
    }
    first = outcome_loss(
        **common,
        stopped_responsibilities=jnp.asarray([[1.0, 0.0]]),
    )
    second = outcome_loss(
        **common,
        stopped_responsibilities=jnp.asarray([[0.0, 1.0]]),
    )
    assert float(second.total) > float(first.total)
    weight_gradient = jax.grad(
        lambda weights: outcome_loss(
            **common, stopped_responsibilities=weights
        ).total
    )(jnp.asarray([[0.5, 0.5]]))
    np.testing.assert_array_equal(
        np.asarray(weight_gradient), np.zeros((1, 2))
    )


def test_rollout_training_batch_has_no_partner_identity_fields() -> None:
    forbidden = {
        "family",
        "family_id",
        "seed",
        "training_seed",
        "checkpoint_index",
        "training_run_id",
        "partner_index",
        "partner_member_index",
    }
    assert forbidden.isdisjoint(VQBCRolloutBatch._fields)


def test_training_diagnostics_expose_slot_code_policy_and_value_health() -> None:
    batch = VQBCRolloutBatch(
        observations=jnp.zeros((3, 2, 1)),
        response_next_observations=jnp.zeros((2, 2, 1)),
        episode_start=jnp.zeros((3, 2), dtype=jnp.bool_),
        previous_actions=jnp.zeros((3, 2), dtype=jnp.int32),
        previous_team_rewards=jnp.zeros((3, 2)),
        initial_value_carry=jnp.zeros((2, 1)),
        reference_logits=jnp.asarray(
            [
                [[2.0, 0.0], [2.0, 0.0]],
                [[2.0, 0.0], [0.0, 2.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        ),
        execution_logits=jnp.asarray(
            [
                [[2.0, 0.0], [0.0, 2.0]],
                [[0.0, 2.0], [0.0, 2.0]],
            ]
        ),
        generic_execution_logits=jnp.asarray(
            [
                [[2.0, 0.0], [2.0, 0.0]],
                [[2.0, 0.0], [0.0, 2.0]],
            ]
        ),
        slot_log_beliefs=jnp.full((3, 2, 2), -jnp.log(2.0)),
        actions=jnp.asarray([[0, 1], [1, 1]], dtype=jnp.int32),
        rewards=jnp.zeros((2, 2)),
        dones=jnp.asarray([[False, False], [False, True]]),
        response_codes=jnp.asarray([[0, 1], [1, 3]], dtype=jnp.int32),
        episode_ids=jnp.zeros((2, 2), dtype=jnp.int32),
        episode_steps=jnp.zeros((2, 2), dtype=jnp.int32),
        completed_episode_returns=jnp.full((2, 2), jnp.nan),
        quotient_counts=jnp.asarray([[1, 2], [2, 2]], dtype=jnp.int32),
        j_use=jnp.asarray(
            [
                [[1.0, 0.0], [0.0, 2.0]],
                [[3.0, 1.0], [2.0, 4.0]],
            ]
        ),
        j_mask=jnp.zeros((2, 2, 2)),
    )
    assignments = FrozenAssignments(
        responsibilities=jnp.asarray([[1.0, 0.0], [0.25, 0.75]]),
        bootstrap_mask=jnp.ones((2, 2), dtype=jnp.bool_),
        bellman_targets=jnp.zeros((2, 2, 2)),
        response_signature_targets=jnp.zeros((2, 2, 2)),
        response_code_targets=jnp.asarray(
            [[0, 1], [1, 3]], dtype=jnp.int32
        ),
    )
    diagnostics = rollout_health_diagnostics(
        batch=batch, assignments=assignments, response_count=4
    )

    np.testing.assert_allclose(
        diagnostics["responsibility"]["effective_mass_by_slot"],
        [1.25, 0.75],
    )
    assert float(diagnostics["responsibility"]["entropy"]["mean"]) > 0.0
    np.testing.assert_array_equal(
        diagnostics["quotient"]["count_histogram"], [1, 3]
    )
    assert float(diagnostics["quotient"]["active_count"]["mean"]) == 1.75
    np.testing.assert_array_equal(
        diagnostics["response_code"]["observed"]["all_codes"]["counts"],
        [1, 2, 0, 1],
    )
    assert (
        float(
            diagnostics["response_code"]["observed"]["all_codes"][
                "perplexity"
            ]
        )
        > 1.0
    )
    assert (
        float(
            diagnostics["policy"]["posterior_argmax_deviation_rate"]
        )
        == 0.5
    )
    assert (
        float(diagnostics["policy"]["generic_argmax_deviation_rate"])
        == 0.0
    )
    assert (
        float(diagnostics["policy"]["normalized_posterior_entropy_mean"])
        == pytest.approx(1.0)
    )
    assert diagnostics["policy"]["posterior_kl"]["p95"].shape == ()
    assert (
        float(
            diagnostics["value_information"][
                "executed_action_j_use_minus_j_mask"
            ]["positive_fraction"]
        )
        == 1.0
    )


def test_main_and_generic_kl_temperatures_update_independently() -> None:
    updated = update_kl_state(
        VQBCKLState(
            log_temperature=jnp.asarray(0.0),
            generic_log_temperature=jnp.asarray(0.0),
        ),
        posterior_mean_kl=jnp.asarray(0.12),
        generic_mean_kl=jnp.asarray(0.0),
        target_kl=0.02,
        learning_rate=1.0e-3,
        minimum_temperature=0.05,
        maximum_temperature=20.0,
    )
    assert float(updated.log_temperature) > 0.0
    assert float(updated.generic_log_temperature) < 0.0


def test_checkpoint_round_trip_keeps_resume_random_and_rollout_state(
    tmp_path: Path,
) -> None:
    model, params, observations, unused_actions, unused_rewards, unused_starts = (
        _model_and_inputs()
    )
    del model, unused_actions, unused_rewards, unused_starts
    config_path = CONFIG_ROOT / "path_c_vqbc_v4_development_simple.yaml"
    config = VQBCConfig.from_mapping(
        yaml.safe_load(config_path.read_text(encoding="utf-8")),
        base_dir=CONFIG_ROOT,
    )
    policy = VQBCPolicyState(
        reference_carry=jnp.zeros((2, 128)),
        value_carry=jnp.ones((2, 128)),
        slot_log_belief=jnp.full((2, 8), -jnp.log(8.0)),
        previous_action=jnp.asarray([1, 2], dtype=jnp.int32),
        previous_team_reward=jnp.asarray([0.5, -0.5]),
        episode_start=jnp.asarray([False, True]),
        log_temperature=jnp.asarray([0.1, 0.2]),
        generic_log_temperature=jnp.asarray([0.3, 0.4]),
    )
    rollout = VQBCRolloutState(
        environment_state={"state": jnp.asarray([7, 8])},
        observations=observations[0],
        policy_state=policy,
        partner_carry={"carry": jnp.asarray([9, 10])},
        partner_member_index=jnp.asarray([0, 3]),
        ego_seat=jnp.asarray([0, 1]),
        episode_step=jnp.asarray([12, 13]),
        episode_id=jnp.asarray([20, 21]),
        episode_return=jnp.asarray([2.0, 3.0]),
        completed_episodes=jnp.asarray(4),
        effective_environment_steps=jnp.asarray(800),
        random_key=jax.random.PRNGKey(31),
    )
    codebook = empty_codebook(code_count=15, signature_dim=6)._replace(
        initialized=jnp.asarray(True),
        last_replaced_codes=jnp.asarray(
            [True] + [False] * 14, dtype=jnp.bool_
        ),
    )
    state = VQBCTrainState(
        online_params=params,
        target_params=params,
        bellman_optimizer_state={"count": jnp.asarray(5)},
        outcome_optimizer_state={"count": jnp.asarray(6)},
        codebook=codebook,
        kl_state=VQBCKLState(
            log_temperature=jnp.asarray(0.2),
            generic_log_temperature=jnp.asarray(0.4),
        ),
        rollout_state=rollout,
        random_key=jax.random.PRNGKey(32),
        effective_environment_steps=jnp.asarray(800),
        completed_episodes=jnp.asarray(4),
        update_count=jnp.asarray(5),
    )
    save_vqbc_checkpoint(tmp_path, config=config, train_state=state)
    restored, metadata, unused_manifest = load_vqbc_checkpoint(
        tmp_path, target_state=state, expected_config=config
    )
    del unused_manifest
    np.testing.assert_array_equal(restored.random_key, state.random_key)
    np.testing.assert_array_equal(
        restored.rollout_state.random_key, state.rollout_state.random_key
    )
    np.testing.assert_array_equal(
        restored.codebook.last_replaced_codes,
        state.codebook.last_replaced_codes,
    )
    assert metadata.reference_training_run_id == (
        config.backbone_init.training_run_id
    )


def test_v4_source_never_imports_two_legacy_heads() -> None:
    source_root = Path(__file__).resolve().parents[3] / "src" / "path_c" / "vqbc"
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in source_root.glob("*.py")
    )
    assert "PrototypeResponseHeads" not in text
    assert "PrototypeTransitionHeads" not in text
    assert "next_feature" not in text
    assert "src.path_c.model" not in text
    assert "src.path_c.probe" not in text
