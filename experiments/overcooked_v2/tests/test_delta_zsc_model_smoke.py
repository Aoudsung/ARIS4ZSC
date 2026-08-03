from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("flax")
optax = pytest.importorskip("optax")

from experiments.overcooked_v2.deployment import (  # noqa: E402
    DEPLOYABLE_PARAM_NAMES,
    deployable_parameters,
)
from src.path_c.belief_set_encoder import (  # noqa: E402
    gaussian_samples,
    gaussian_summary,
    prior_gaussian_summary,
)
from src.path_c.experiment import load_config  # noqa: E402
from src.path_c.model import (  # noqa: E402
    build_model,
    initial_policy_state,
    initialize_model_parameters,
)
from src.path_c.runner import observe_policy_after_transition  # noqa: E402
from src.path_c.storage import pytree_fingerprint  # noqa: E402
from src.path_c.training import make_optimizer  # noqa: E402
from src.path_c.types import PolicyState  # noqa: E402


ROOT = Path(__file__).resolve().parents[3]


def _fixture(batch_size: int = 2):
    config = load_config(
        ROOT / "experiments/overcooked_v2/configs/delta_zsc_simple_development.yaml",
        run_kind="mechanical",
    )
    observation_shape = (5, 5, 39)
    model = build_model(
        observation_shape=observation_shape,
        action_count=6,
        task_hidden_dim=config.model.task_hidden_dim,
        belief_hidden_dim=config.model.belief_hidden_dim,
        latent_dim=config.model.latent_dim,
        actor_hidden_dim=config.model.actor_hidden_dim,
        critic_hidden_dim=config.model.critic_hidden_dim,
        response_hidden_dim=config.model.response_hidden_dim,
        modulation_rank=config.model.modulation_rank,
        action_embedding_dim=config.model.action_embedding_dim,
        log_standard_deviation_minimum=config.model.log_standard_deviation_minimum,
        log_standard_deviation_maximum=config.model.log_standard_deviation_maximum,
    )
    state = initial_policy_state(
        batch_size=batch_size,
        observation_shape=observation_shape,
        action_count=6,
        task_hidden_dim=config.model.task_hidden_dim,
        belief_hidden_dim=config.model.belief_hidden_dim,
        latent_dim=config.model.latent_dim,
    )
    observation = jnp.arange(
        batch_size * 5 * 5 * 39, dtype=jnp.float32
    ).reshape((batch_size,) + observation_shape) / 255.0
    params = initialize_model_parameters(
        model,
        key=jax.random.PRNGKey(0),
        example_state=state,
        example_observation=observation,
        partner_code_dim=config.partner_generator.code_dim,
    )
    return config, model, params, state, observation


def test_single_gaussian_model_forward_and_minimal_gradient_update() -> None:
    config, model, params, state, observation = _fixture()
    task = params["task_encoder"]
    assert task["official_conv_0"]["kernel"].shape == (1, 1, 39, 128)
    assert task["official_conv_5"]["kernel"].shape == (3, 3, 32, 32)
    assert task["task_gru"]["ir"]["kernel"].shape == (128, 128)

    next_state, output = model.apply(
        {"params": params}, state, observation, jnp.zeros((2,), dtype=jnp.bool_),
        method=model.step,
    )
    assert output.policy_logits.shape == (2, 6)
    assert output.state_value.shape == (2,)
    assert output.raw_q1.shape == output.raw_q2.shape == (2, 6)
    assert output.belief_mean.shape == (2, 8)
    assert output.belief_log_standard_deviation.shape == (2, 8)
    assert output.belief_summary.shape == (2, 17)
    assert np.all(np.isfinite(np.asarray(output.policy_logits)))
    assert np.all(np.asarray(output.belief_log_standard_deviation) >= -5.0)
    assert np.all(np.asarray(output.belief_log_standard_deviation) <= 2.0)

    response = model.apply(
        {"params": params},
        output.task_features,
        output.belief_summary,
        jnp.asarray([0, 5], dtype=jnp.int32),
        method=model.response_from_context_and_action,
    )
    assert response.visibility_logit.shape == (2,)
    assert response.relative_position_logits.shape == (2, 26)
    assert response.direction_logits.shape == (2, 4)
    assert response.inventory_logits.shape == (2, 5, 4)

    observations = jnp.broadcast_to(observation[None], (3,) + observation.shape)
    previous_actions = jnp.zeros((3, 2), dtype=jnp.int32)
    starts = jnp.zeros((3, 2), dtype=jnp.bool_)
    dropout = jnp.asarray([[False, True], [True, False], [False, False]])
    final_state, sequence = model.apply(
        {"params": params}, state, observations, previous_actions, starts, dropout,
        method=model.sequence,
    )
    context_state, context = model.apply(
        {"params": params}, state, observations, previous_actions, starts,
        method=model.context_sequence,
    )
    assert sequence.policy_logits.shape == (3, 2, 6)
    np.testing.assert_allclose(context.task_features, sequence.task_features, atol=1e-6)
    np.testing.assert_allclose(context.belief_mean, sequence.belief_mean, atol=1e-6)
    np.testing.assert_allclose(context_state.task_carry, final_state.task_carry, atol=1e-6)
    assert final_state.previous_observation.dtype == jnp.float32

    assert set(deployable_parameters(params)) == set(DEPLOYABLE_PARAM_NAMES)
    assert "partner_generator" not in params
    assert "base_actor" not in params and "context_residual" not in params

    def objective(candidate):
        _, current = model.apply(
            {"params": candidate}, state, observation,
            jnp.zeros((2,), dtype=jnp.bool_), method=model.step,
        )
        return jnp.mean(jnp.square(current.raw_q1)) + jnp.mean(
            jnp.square(current.policy_logits)
        )

    gradients = jax.grad(objective)(params)
    optimizer, optimizer_state = make_optimizer(
        params, learning_rate=1.0e-4, gradient_clip_norm=0.5
    )
    updates, _ = optimizer.update(gradients, optimizer_state, params)
    updated = optax.apply_updates(params, updates)
    assert pytree_fingerprint(updated) != pytree_fingerprint(params)


def test_gaussian_sampling_summary_and_prior_are_registered() -> None:
    mean = jnp.zeros((2, 8), dtype=jnp.float32)
    log_std = jnp.zeros_like(mean)
    uncertainty = jnp.full((2,), 5.0 / 7.0)
    summary = gaussian_summary(mean, log_std, uncertainty)
    prior = prior_gaussian_summary(summary, 8)
    samples, weights = gaussian_samples(
        jax.random.PRNGKey(9), mean=mean, log_standard_deviation=log_std,
        sample_count=16,
    )
    assert summary.shape == prior.shape == (2, 17)
    assert samples.shape == (2, 16, 8)
    assert weights.shape == (2, 16)
    np.testing.assert_allclose(jnp.sum(weights, axis=-1), 1.0)
    np.testing.assert_allclose(prior[..., :16], 0.0)
    np.testing.assert_allclose(prior[..., -1], 5.0 / 7.0)


def test_policy_state_and_deployable_inputs_cannot_contain_reward() -> None:
    assert "previous_reward" not in PolicyState._fields
    assert "reward" not in inspect.signature(initial_policy_state).parameters
    config, unused_model, unused_params, state, observation = _fixture()
    del unused_model, unused_params
    first = observe_policy_after_transition(
        stepped_state=state,
        action=jnp.asarray([1, 2], dtype=jnp.int32),
        reward=jnp.asarray([20.0, -10.0]),
        done=jnp.asarray([False, False]),
        next_observation=observation,
        model_config=config.model,
    )
    second = observe_policy_after_transition(
        stepped_state=state,
        action=jnp.asarray([1, 2], dtype=jnp.int32),
        reward=jnp.asarray([-400.0, 400.0]),
        done=jnp.asarray([False, False]),
        next_observation=observation,
        model_config=config.model,
    )
    for left, right in zip(
        jax.tree_util.tree_leaves(first), jax.tree_util.tree_leaves(second), strict=True
    ):
        np.testing.assert_array_equal(left, right)
