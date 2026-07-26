from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from src.path_c.method import uniform_slot_log_belief
from src.path_c.model import build_model, initialize_heads, model_forward


def _model() -> object:
    return build_model(
        hidden_dim=12,
        slot_count=3,
        action_count=6,
        response_count=5,
        prior_scale=0.01,
        action_embedding_dim=4,
        log_standard_deviation_minimum=-5.0,
        log_standard_deviation_maximum=2.0,
    )


def _history_inputs(features: jax.Array) -> tuple[jax.Array, jax.Array]:
    prefix = features.shape[:-1]
    return (
        jnp.full(prefix, 6, dtype=jnp.int32),
        jnp.zeros(prefix, dtype=jnp.float32),
    )


def test_all_v4_2_heads_have_expected_shapes() -> None:
    model = _model()
    features = jnp.ones((2, 12), dtype=jnp.float32)
    belief = uniform_slot_log_belief((2,), 3)
    previous_actions, previous_rewards = _history_inputs(features)
    params = model.init(
        jax.random.PRNGKey(2),
        features,
        previous_actions,
        previous_rewards,
        belief,
    )["params"]
    output = model.apply(
        {"params": params},
        features,
        previous_actions,
        previous_rewards,
        belief,
    )
    assert output["q_values"].shape == (2, 2, 3, 6)
    assert output["learned_q_values"].shape == (2, 2, 3, 6)
    assert output["prior_q_values"].shape == (2, 2, 3, 6)
    assert output["centered_advantages"].shape == (2, 2, 3, 6)
    assert output["features"].shape == (2, 12)
    assert output["response_logits"].shape == (2, 3, 6, 5)
    assert output["response_probabilities"].shape == (2, 3, 6, 5)
    assert output["reward_mean"].shape == (2, 3, 6)
    assert output["reward_log_standard_deviation"].shape == (2, 3, 6)
    assert output["continuation_use_mean"].shape == (2, 2, 3, 6, 5)
    assert output["continuation_use_log_standard_deviation"].shape == (
        2,
        2,
        3,
        6,
        5,
    )
    assert output["continuation_mask_mean"].shape == (2, 2, 3, 6, 5)
    assert output["continuation_mask_log_standard_deviation"].shape == (
        2,
        2,
        3,
        6,
        5,
    )

    forward = model_forward(
        raw_output=output,
        reference_logits=jnp.zeros((2, 6), dtype=jnp.float32),
        slot_log_belief=belief,
        temperature=jnp.ones((2,), dtype=jnp.float32),
        generic_temperature=jnp.ones((2,), dtype=jnp.float32),
        deployment_mode="posterior_use",
        gamma=0.99,
    )
    assert forward.value_class_ids.shape == (2, 3)
    assert forward.supported_value_class_count.shape == (2,)
    for value in (
        forward.j_use,
        forward.j_mask,
        forward.per_action_response_value,
        forward.per_action_net_value,
        forward.information_gain,
        forward.execution_logits,
        forward.mask_execution_logits,
    ):
        assert value.shape == (2, 6)
    for value in (
        forward.predicted_response_effect,
        forward.predicted_policy_cost,
        forward.predicted_net_effect,
        forward.predicted_regularized_net_effect,
        forward.predicted_policy_total_variation,
    ):
        assert value.shape == (2,)


def test_twin_estimators_and_random_prior_are_distinct() -> None:
    model = _model()
    features = jnp.arange(24, dtype=jnp.float32).reshape(2, 12)
    belief = uniform_slot_log_belief((2,), 3)
    previous_actions, previous_rewards = _history_inputs(features)
    params = model.init(
        jax.random.PRNGKey(3),
        features,
        previous_actions,
        previous_rewards,
        belief,
    )["params"]
    output = model.apply(
        {"params": params},
        features,
        previous_actions,
        previous_rewards,
        belief,
    )
    assert not np.array_equal(
        np.asarray(output["prior_q_values"][:, 0]),
        np.asarray(output["prior_q_values"][:, 1]),
    )
    np.testing.assert_allclose(
        np.asarray(output["q_values"]),
        np.asarray(
            output["learned_q_values"]
            + 0.01 * output["prior_q_values"]
        ),
        atol=1e-7,
    )


def test_belief_condition_can_be_learned_from_behavioral_loss() -> None:
    model = _model()
    features = jnp.ones((2, 12), dtype=jnp.float32)
    uniform = uniform_slot_log_belief((2,), 3)
    previous_actions, previous_rewards = _history_inputs(features)
    params = model.init(
        jax.random.PRNGKey(4),
        features,
        previous_actions,
        previous_rewards,
        uniform,
    )["params"]
    concentrated = jnp.log(
        jnp.asarray([[0.98, 0.01, 0.01], [0.01, 0.98, 0.01]])
    )

    def separation(candidate: object) -> jax.Array:
        uniform_q = model.apply(
            {"params": candidate},
            features,
            previous_actions,
            previous_rewards,
            uniform,
        )["learned_q_values"]
        concentrated_q = model.apply(
            {"params": candidate},
            features,
            previous_actions,
            previous_rewards,
            concentrated,
        )["learned_q_values"]
        return -jnp.sum(jnp.square(concentrated_q - uniform_q + 1.0))

    gradient = jax.grad(separation)(params)
    learned_params = jax.tree_util.tree_map(
        lambda value, change: value + 0.01 * change, params, gradient
    )
    uniform_output = model.apply(
        {"params": learned_params},
        features,
        previous_actions,
        previous_rewards,
        uniform,
    )
    concentrated_output = model.apply(
        {"params": learned_params},
        features,
        previous_actions,
        previous_rewards,
        concentrated,
    )
    assert not np.array_equal(
        np.asarray(uniform_output["learned_q_values"]),
        np.asarray(concentrated_output["learned_q_values"]),
    )


def test_bellman_head_has_feature_gradient_and_outcome_head_does_not() -> None:
    model = _model()
    features = jnp.arange(24, dtype=jnp.float32).reshape(2, 12) / 10.0
    belief = uniform_slot_log_belief((2,), 3)
    previous_actions, previous_rewards = _history_inputs(features)
    params = model.init(
        jax.random.PRNGKey(5),
        features,
        previous_actions,
        previous_rewards,
        belief,
    )["params"]

    def q_loss(candidate: jax.Array) -> jax.Array:
        return jnp.sum(
            model.apply(
                {"params": params},
                candidate,
                previous_actions,
                previous_rewards,
                belief,
            )["q_values"]
        )

    def outcome_loss(candidate: jax.Array) -> jax.Array:
        output = model.apply(
            {"params": params},
            candidate,
            previous_actions,
            previous_rewards,
            belief,
        )
        return jnp.sum(output["response_logits"]) + jnp.sum(
            output["reward_mean"]
        )

    assert np.any(np.asarray(jax.grad(q_loss)(features)) != 0.0)
    np.testing.assert_array_equal(
        np.asarray(jax.grad(outcome_loss)(features)),
        np.zeros_like(np.asarray(features)),
    )


def test_initial_heads_preserve_official_action_distribution() -> None:
    model = _model()
    features = jnp.ones((2, 12), dtype=jnp.float32)
    belief = uniform_slot_log_belief((2,), 3)
    previous_actions, previous_rewards = _history_inputs(features)
    observations = jnp.zeros((1, 2, 5, 5, 39), dtype=jnp.float32)
    params = initialize_heads(
        model,
        random_key=jax.random.PRNGKey(6),
        example_features=features,
        example_previous_actions=previous_actions,
        example_previous_team_rewards=previous_rewards,
        example_slot_log_belief=belief,
        example_observations=observations,
    )
    raw = model.apply(
        {"params": params},
        features,
        previous_actions,
        previous_rewards,
        belief,
    )
    reference_logits = jnp.asarray(
        [[1.0, 0.5, -0.5, 0.0, 0.2, -1.0]] * 2,
        dtype=jnp.float32,
    )
    output = model_forward(
        raw_output=raw,
        reference_logits=reference_logits,
        slot_log_belief=belief,
        temperature=jnp.ones((2,)),
        generic_temperature=jnp.ones((2,)),
        deployment_mode="posterior_use",
        gamma=0.99,
    )
    np.testing.assert_array_equal(
        np.asarray(output.execution_logits), np.asarray(reference_logits)
    )
