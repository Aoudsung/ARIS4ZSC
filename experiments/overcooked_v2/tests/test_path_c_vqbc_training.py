from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
yaml = pytest.importorskip("yaml")
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from src.path_c.vqbc.checkpoint import load_vqbc_checkpoint, save_vqbc_checkpoint
from src.path_c.vqbc.codebook import empty_codebook, target_response_signatures
from src.path_c.vqbc.config import VQBCConfig, VQBCModelConfig
from src.path_c.vqbc.integrity import get_parameter_leaf
from src.path_c.vqbc.model import (
    build_vqbc_model,
    explicit_official_parameter_mapping,
    initialize_vqbc_from_official,
)
from src.path_c.vqbc.objectives import outcome_loss
from src.path_c.vqbc.policy import uniform_slot_log_belief
from src.path_c.vqbc.training import environment_minibatch_schedule
from src.path_c.vqbc.types import (
    VQBCCodebookState,
    VQBCKLState,
    VQBCPolicyState,
    VQBCRolloutState,
    VQBCTrainState,
)


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"


def _model_config() -> VQBCModelConfig:
    return VQBCModelConfig(
        hidden_dim=128,
        head_hidden_dim=128,
        action_embedding_dim=16,
        slot_count=8,
        response_count=16,
        action_count=6,
        prior_scale=0.01,
        log_standard_deviation_minimum=-5.0,
        log_standard_deviation_maximum=2.0,
    )


def _model_and_inputs() -> tuple:
    pytest.importorskip("flax")
    model = build_vqbc_model(
        model_config=_model_config(),
        official_dimensions={"gru_hidden_dim": 128, "activation": "relu"},
    )
    observations = jnp.ones((2, 2, 5, 5, 39), dtype=jnp.float32)
    actions = jnp.full((2, 2), 6, dtype=jnp.int32)
    rewards = jnp.zeros((2, 2), dtype=jnp.float32)
    starts = jnp.asarray([[True, True], [False, False]])
    beliefs = uniform_slot_log_belief((2, 2), 8)
    params = model.init(
        jax.random.PRNGKey(0),
        model.initial_carry(2),
        observations,
        actions,
        rewards,
        starts,
        beliefs,
    )["params"]
    return model, params, observations, actions, rewards, starts, beliefs


def _tree_l1(tree: object) -> float:
    return float(
        sum(
            np.abs(np.asarray(value)).sum()
            for value in jax.tree_util.tree_leaves(tree)
        )
    )


def test_model_output_shapes_and_zero_initialized_behavior_continuation() -> None:
    model, params, observations, actions, rewards, starts, beliefs = _model_and_inputs()
    unused_carry, output = model.apply(
        {"params": params},
        model.initial_carry(2),
        observations,
        actions,
        rewards,
        starts,
        beliefs,
    )
    del unused_carry
    assert output["q_values"].shape == (2, 2, 2, 8, 6)
    assert output["response_logits"].shape == (2, 2, 8, 6, 16)
    assert output["continuation_use_mean"].shape == (2, 2, 2, 8, 6, 16)
    assert output["continuation_mask_mean"].shape == (2, 2, 2, 8, 6, 16)
    np.testing.assert_array_equal(output["reward_mean"], 0.0)
    np.testing.assert_array_equal(output["continuation_use_mean"], 0.0)
    np.testing.assert_array_equal(output["continuation_mask_mean"], 0.0)
    np.testing.assert_allclose(output["response_probabilities"], 1.0 / 16.0)
    np.testing.assert_array_equal(output["continuation_use_mean"][..., 15], 0.0)


def test_only_bellman_path_reaches_backbone_and_random_prior_is_frozen() -> None:
    flax_core = pytest.importorskip("flax.core")
    freeze, unfreeze = flax_core.freeze, flax_core.unfreeze

    model, params, observations, actions, rewards, starts, beliefs = _model_and_inputs()

    def q_loss(candidate: object) -> object:
        unused, output = model.apply(
            {"params": candidate},
            model.initial_carry(2),
            observations,
            actions,
            rewards,
            starts,
            beliefs,
        )
        del unused
        return jnp.sum(jnp.square(output["q_values"]))

    q_grad = jax.grad(q_loss)(params)
    assert _tree_l1(q_grad["backbone"]) > 0.0
    assert _tree_l1(q_grad["q_heads"]["PriorEstimator_0"]) == 0.0
    assert _tree_l1(q_grad["q_heads"]["PriorEstimator_1"]) == 0.0

    probe = unfreeze(params)

    def set_kernels(tree: dict) -> None:
        for key, value in tree.items():
            if isinstance(value, dict):
                set_kernels(value)
            elif key == "kernel":
                tree[key] = jnp.ones_like(value)

    set_kernels(probe["outcome"])
    probe = freeze(probe)

    def detached_loss(candidate: object) -> object:
        unused, output = model.apply(
            {"params": candidate},
            model.initial_carry(2),
            observations,
            actions,
            rewards,
            starts,
            beliefs,
        )
        del unused
        return (
            jnp.sum(output["response_logits"])
            + jnp.sum(output["reward_mean"])
            + jnp.sum(output["continuation_use_mean"])
            + jnp.sum(output["continuation_mask_mean"])
        )

    outcome_grad = jax.grad(detached_loss)(probe)
    assert _tree_l1(outcome_grad["backbone"]) == 0.0


def test_each_slot_has_independent_q_parameters_and_belief_projection() -> None:
    unused_model, params, *unused = _model_and_inputs()
    del unused_model, unused
    estimator = params["q_heads"]["LearnedEstimator_0"]
    for slot in range(8):
        assert f"learned_estimator_0_slot_{slot}_hidden" in estimator
        assert f"learned_estimator_0_slot_{slot}_belief_projection" in estimator
        assert f"learned_estimator_0_slot_{slot}_advantages" in estimator
    assert "slot_hidden" not in estimator
    assert "slot_embedding" not in estimator


def test_belief_conditioning_can_change_q_at_fixed_recurrent_features() -> None:
    flax_core = pytest.importorskip("flax.core")
    freeze, unfreeze = flax_core.freeze, flax_core.unfreeze

    model, params, observations, actions, rewards, starts, beliefs = _model_and_inputs()
    modified = unfreeze(params)
    projection = modified["q_heads"]["LearnedEstimator_0"][
        "learned_estimator_0_slot_0_belief_projection"
    ]
    kernel = jnp.zeros_like(projection["kernel"])
    kernel = kernel.at[0].set(0.5)
    kernel = kernel.at[1].set(-0.5)
    projection["kernel"] = kernel
    modified = freeze(modified)
    left = beliefs.at[..., 0].set(jnp.log(0.9)).at[..., 1].set(jnp.log(0.1 / 7.0))
    left = left.at[..., 2:].set(jnp.log(0.1 / 7.0))
    right = beliefs.at[..., 1].set(jnp.log(0.9)).at[..., 0].set(jnp.log(0.1 / 7.0))
    right = right.at[..., 2:].set(jnp.log(0.1 / 7.0))

    def apply(current_belief):
        unused, output = model.apply(
            {"params": modified},
            model.initial_carry(2),
            observations,
            actions,
            rewards,
            starts,
            current_belief,
        )
        del unused
        return output["learned_q_values"][..., 0, 0, :]

    assert not np.allclose(np.asarray(apply(left)), np.asarray(apply(right)))


def _set_nested(tree: dict, path: tuple[str, ...], value: object) -> None:
    target = tree
    for component in path[:-1]:
        target = target.setdefault(component, {})
    target[path[-1]] = value


def test_official_initialization_preserves_backbone_and_adds_belief_heads() -> None:
    model, params, observations, actions, rewards, starts, unused_beliefs = _model_and_inputs()
    del unused_beliefs
    official: dict = {}
    mapping = explicit_official_parameter_mapping()
    for target_path, source_path in mapping.items():
        _set_nested(official, source_path, get_parameter_leaf(params, target_path))
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
    assert any("belief_projection" in "/".join(path) for path, _ in _all_paths(initialized))


def _all_paths(tree: object, prefix: tuple[str, ...] = ()):
    if isinstance(tree, dict) or hasattr(tree, "items"):
        for key, value in tree.items():
            yield from _all_paths(value, (*prefix, str(key)))
    else:
        yield prefix, tree


def test_outcome_loss_uses_stopped_slot_responsibility() -> None:
    response_logits = jnp.zeros((1, 1, 2, 1, 16))
    reward_mean = jnp.asarray([[[[0.0], [10.0]]]])
    reward_log_std = jnp.zeros_like(reward_mean)
    continuation = jnp.zeros((1, 1, 2, 2, 1, 16))
    common = {
        "response_logits": response_logits,
        "reward_mean": reward_mean,
        "reward_log_standard_deviation": reward_log_std,
        "continuation_use_mean": continuation,
        "continuation_use_log_standard_deviation": jnp.zeros_like(continuation),
        "continuation_mask_mean": continuation,
        "continuation_mask_log_standard_deviation": jnp.zeros_like(continuation),
        "response_codes": jnp.zeros((1, 1), dtype=jnp.int32),
        "rewards": jnp.zeros((1, 1)),
        "continuation_use_targets": jnp.zeros((1, 1, 2, 2)),
        "continuation_mask_targets": jnp.zeros((1, 1, 2, 2)),
        "actions": jnp.zeros((1, 1), dtype=jnp.int32),
    }
    first = outcome_loss(
        **common, stopped_responsibilities=jnp.asarray([[1.0, 0.0]])
    )
    second = outcome_loss(
        **common, stopped_responsibilities=jnp.asarray([[0.0, 1.0]])
    )
    assert float(second.total) > float(first.total)
    weight_gradient = jax.grad(
        lambda weights: outcome_loss(
            **common, stopped_responsibilities=weights
        ).total
    )(jnp.asarray([[0.5, 0.5]]))
    np.testing.assert_array_equal(weight_gradient, 0.0)


def test_response_signature_is_independent_of_responsibility() -> None:
    advantages = jnp.asarray(
        [[[[1.0, 0.0], [0.0, 1.0]], [[3.0, 0.0], [0.0, 3.0]]]]
    )
    signature = target_response_signatures(
        target_centered_advantages=advantages
    )
    np.testing.assert_allclose(signature, [[1.0, 1.0]])


def test_environment_minibatch_schedule_partitions_each_epoch_without_repeats() -> None:
    schedule = np.asarray(
        environment_minibatch_schedule(
            jax.random.PRNGKey(5),
            environment_count=32,
            minibatches_per_epoch=8,
            update_epochs=4,
        )
    )
    assert schedule.shape == (4, 8, 4)
    for epoch in schedule:
        np.testing.assert_array_equal(np.sort(epoch.reshape(-1)), np.arange(32))


def test_checkpoint_round_trip_rejects_old_state_namespace(tmp_path: Path) -> None:
    unused_model, params, observations, *unused = _model_and_inputs()
    del unused_model, unused
    config_path = CONFIG_ROOT / "path_c_vqbc_v4_2_development_simple.yaml"
    config = VQBCConfig.from_mapping(
        yaml.safe_load(config_path.read_text(encoding="utf-8")),
        base_dir=CONFIG_ROOT,
    )
    policy = VQBCPolicyState(
        reference_carry=jnp.zeros((2, 128)),
        value_carry=jnp.ones((2, 128)),
        slot_log_belief=uniform_slot_log_belief((2,), 8),
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
        initialized=jnp.asarray(True)
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
    assert metadata.schema_version == "path_c_model_checkpoint_metadata_v4"
    manifest_path = tmp_path / "manifest.json"
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    payload["schema_version"] = "path_c_flax_checkpoint_v3"
    manifest_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        load_vqbc_checkpoint(tmp_path, target_state=state, expected_config=config)



def test_executed_response_codes_drive_outcome_and_stale_belief_targets() -> None:
    source = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "path_c"
        / "vqbc"
        / "training.py"
    ).read_text(encoding="utf-8")
    assert "response_codes=current.response_codes" in source
    assert "response_codes=batch.response_codes" in source
    assert "target_codes=assignments.response_code_targets" in source
    assert "response_codes=assignments.response_code_targets" not in source


def test_v4_2_source_removes_unconstrained_continuation_max_and_next_q_head() -> None:
    source_root = Path(__file__).resolve().parents[3] / "src" / "path_c" / "vqbc"
    text = "\n".join(path.read_text(encoding="utf-8") for path in source_root.glob("*.py"))
    assert "next_q_mean" not in text
    assert "PrototypeResponseHeads" not in text
    assert "PrototypeTransitionHeads" not in text
    assert "max(use_continuation" not in text
