from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("flax")
from flax.core import freeze, unfreeze

from src.path_c.method import uniform_slot_log_belief
from src.path_c.model import (
    build_model,
    initial_control_carry,
    initialize_heads,
    model_forward,
)


def _model() -> object:
    return build_model(
        hidden_dim=8,
        slot_count=3,
        action_count=6,
        response_count=5,
        prior_scale=0.01,
        action_embedding_dim=4,
        log_standard_deviation_minimum=-5.0,
        log_standard_deviation_maximum=2.0,
    )


def _inputs(time: int = 3, batch: int = 2):
    features = jnp.ones((time, batch, 8), dtype=jnp.float32)
    actions = jnp.full((time, batch), 6, dtype=jnp.int32)
    rewards = jnp.zeros((time, batch), dtype=jnp.float32)
    starts = jnp.zeros((time, batch), dtype=jnp.bool_).at[0].set(True)
    belief = uniform_slot_log_belief((time, batch), 3)
    return features, actions, rewards, starts, belief


def test_all_v4_4_heads_have_registered_shapes() -> None:
    model = _model()
    features, actions, rewards, starts, belief = _inputs()
    carry = initial_control_carry(2, 8)
    params = model.init(
        jax.random.PRNGKey(1),
        carry,
        features,
        actions,
        rewards,
        starts,
        belief,
        method=model.sequence,
    )["params"]
    response_params = model.init(
        jax.random.PRNGKey(11),
        jnp.zeros((3, 2, 5, 5, 39), dtype=jnp.float32),
        jnp.zeros((3, 2), dtype=jnp.int32),
        jnp.zeros((3, 2, 5, 5, 39), dtype=jnp.float32),
        jnp.zeros((3, 2), dtype=jnp.bool_),
        method=model.encode_response,
    )["params"]["response_encoder"]
    mutable_params = unfreeze(params)
    mutable_params["response_encoder"] = response_params
    params = freeze(mutable_params)
    next_carry, output = model.apply(
        {"params": params},
        carry,
        features,
        actions,
        rewards,
        starts,
        belief,
        method=model.sequence,
    )
    assert next_carry.shape == (2, 8)
    assert output["features"].shape == (3, 2, 8)
    assert output["q_values"].shape == (3, 2, 2, 3, 6)
    assert output["response_logits"].shape == (3, 2, 3, 6, 5)
    assert output["next_q_use_mean"].shape == (3, 2, 2, 3, 6, 5, 6)
    assert output["next_q_mask_mean"].shape == (3, 2, 2, 3, 6, 5, 6)
    assert output["next_reference_logits_mean"].shape == (3, 2, 6, 5, 6)

    signatures = model.apply(
        {"params": params},
        jnp.zeros((3, 2, 5, 5, 39), dtype=jnp.float32),
        jnp.zeros((3, 2), dtype=jnp.int32),
        jnp.zeros((3, 2, 5, 5, 39), dtype=jnp.float32),
        jnp.zeros((3, 2), dtype=jnp.bool_),
        method=model.encode_response,
    )
    assert signatures.shape == (3, 2, 12)


def _set_projection_kernels_nonzero(tree: object) -> object:
    mutable = unfreeze(tree)

    def visit(node: object, path: tuple[str, ...] = ()) -> None:
        if not isinstance(node, dict):
            return
        for key, value in node.items():
            next_path = (*path, str(key))
            if (
                key == "kernel"
                and any(
                    name in "/".join(next_path)
                    for name in (
                        "PreviousActionProjection",
                        "PreviousRewardProjection",
                    )
                )
            ):
                node[key] = (
                    jnp.arange(value.size, dtype=value.dtype).reshape(value.shape)
                    + 1
                ) * 0.01
            else:
                visit(value, next_path)

    visit(mutable)
    return freeze(mutable)


def test_early_action_and_reward_enter_later_control_memory() -> None:
    model = _model()
    features, actions, rewards, starts, belief = _inputs(time=4, batch=1)
    carry = initial_control_carry(1, 8)
    params = model.init(
        jax.random.PRNGKey(2),
        carry,
        features,
        actions,
        rewards,
        starts,
        belief,
        method=model.sequence,
    )["params"]
    params = _set_projection_kernels_nonzero(params)

    changed_actions = actions.at[1, 0].set(2)
    changed_rewards = rewards.at[1, 0].set(3.0)
    _, baseline = model.apply(
        {"params": params},
        carry,
        features,
        actions,
        rewards,
        starts,
        belief,
        method=model.sequence,
    )
    _, changed = model.apply(
        {"params": params},
        carry,
        features,
        changed_actions,
        changed_rewards,
        starts,
        belief,
        method=model.sequence,
    )
    # Evidence changed at t=1 must remain in the recurrent state at t>=2.
    assert not np.array_equal(
        np.asarray(baseline["features"][2:]),
        np.asarray(changed["features"][2:]),
    )


def test_outcome_loss_cannot_backpropagate_into_control_feature() -> None:
    model = _model()
    features = jnp.ones((1, 8), dtype=jnp.float32)
    belief = uniform_slot_log_belief((1,), 3)
    params = model.init(
        jax.random.PRNGKey(3),
        features,
        belief,
        method=model.from_features,
    )["params"]

    def outcome_sum(candidate: jax.Array) -> jax.Array:
        raw = model.apply(
            {"params": params}, candidate, belief, method=model.from_features
        )
        return (
            jnp.sum(raw["reward_mean"])
            + jnp.sum(raw["response_logits"])
            + jnp.sum(raw["next_q_use_mean"])
            + jnp.sum(raw["next_reference_logits_mean"])
        )

    np.testing.assert_array_equal(
        np.asarray(jax.grad(outcome_sum)(features)),
        np.zeros((1, 8), dtype=np.float32),
    )


def test_zero_initialized_outcome_preserves_reference_policy() -> None:
    model = _model()
    features = jnp.ones((2, 8), dtype=jnp.float32)
    belief = uniform_slot_log_belief((2,), 3)
    params = initialize_heads(
        model,
        random_key=jax.random.PRNGKey(4),
        example_official_features=features,
        example_previous_actions=jnp.full((2,), 6, dtype=jnp.int32),
        example_previous_team_rewards=jnp.zeros((2,), dtype=jnp.float32),
        example_episode_start=jnp.ones((2,), dtype=jnp.bool_),
        example_slot_log_belief=belief,
        example_observations=jnp.zeros((1, 2, 5, 5, 39), dtype=jnp.float32),
        hidden_dim=8,
    )
    _, raw = model.apply(
        {"params": params},
        initial_control_carry(2, 8),
        features,
        jnp.full((2,), 6, dtype=jnp.int32),
        jnp.zeros((2,), dtype=jnp.float32),
        jnp.ones((2,), dtype=jnp.bool_),
        belief,
        method=model.step,
    )
    reference = jnp.asarray(
        [[1.0, 0.5, -0.5, 0.0, 0.2, -1.0]] * 2,
        dtype=jnp.float32,
    )
    output = model_forward(
        raw_output=raw,
        reference_logits=reference,
        slot_log_belief=belief,
        temperature=jnp.ones((2,)),
        generic_temperature=jnp.ones((2,)),
        deployment_mode="posterior_use",
        gamma=0.99,
        uncertainty_penalty=1.0,
    )
    np.testing.assert_allclose(
        np.asarray(jax.nn.softmax(output.execution_logits, axis=-1)),
        np.asarray(jax.nn.softmax(reference, axis=-1)),
        atol=1e-7,
    )
    np.testing.assert_allclose(
        np.asarray(output.predicted_policy_mediated_effect), 0.0, atol=1e-7
    )
    np.testing.assert_allclose(
        np.asarray(output.predicted_next_policy_total_variation), 0.0, atol=1e-7
    )
