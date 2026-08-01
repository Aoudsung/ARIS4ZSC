from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("flax")

from src.path_c.belief_set_encoder import stratified_mixture_samples  # noqa: E402
from src.path_c.experiment import load_config  # noqa: E402
from src.path_c.model import (  # noqa: E402
    build_model,
    initial_policy_state,
    initialize_model_parameters,
)
from src.path_c.regret_potential import (  # noqa: E402
    decision_regret_from_action_values,
    potential_shaping,
)
from src.path_c.resources import peak_device_memory_bytes  # noqa: E402
from src.path_c.runner import (  # noqa: E402
    DECISION_REGRET_STATE_CHUNK_SIZE,
    chunked_decision_regret,
)
from src.path_c.training import (  # noqa: E402
    gather_actions,
    response_prediction_loss,
)
from src.path_c.types import ContextOutput, ResponsePrediction  # noqa: E402


ROOT = Path(__file__).resolve().parents[3]


def _fixture() -> tuple[object, object, object, object, object]:
    config = load_config(
        ROOT
        / "experiments"
        / "overcooked_v2"
        / "configs"
        / "delta_zsc_simple_development.yaml",
        run_kind="mechanical",
    )
    observation_shape = (5, 5, 8)
    model = build_model(
        observation_shape=observation_shape,
        action_count=6,
        **{
            name: getattr(config.model, name)
            for name in (
                "task_hidden_dim",
                "belief_hidden_dim",
                "latent_dim",
                "mixture_components",
                "belief_embedding_dim",
                "actor_hidden_dim",
                "critic_hidden_dim",
                "response_hidden_dim",
                "modulation_rank",
                "action_embedding_dim",
                "log_variance_minimum",
                "log_variance_maximum",
                "response_log_std_minimum",
                "response_log_std_maximum",
            )
        },
    )
    state = initial_policy_state(
        batch_size=2,
        observation_shape=observation_shape,
        action_count=6,
        task_hidden_dim=config.model.task_hidden_dim,
        belief_hidden_dim=config.model.belief_hidden_dim,
        latent_dim=config.model.latent_dim,
        mixture_components=config.model.mixture_components,
    )
    observation = jnp.arange(2 * 5 * 5 * 8, dtype=jnp.float32).reshape(
        (2,) + observation_shape
    ) / 255.0
    params = initialize_model_parameters(
        model,
        key=jax.random.PRNGKey(11),
        example_state=state,
        example_observation=observation,
        partner_code_dim=config.partner_generator.code_dim,
    )
    unused_state, output = model.apply(
        {"params": params},
        state,
        observation,
        jnp.ones((2,), dtype=jnp.float32),
        method=model.step,
    )
    del unused_state
    return config, model, params, state, output


def _all_action_prediction(module, task_features, belief_embedding):
    prefix = task_features.shape[:-1]
    actions = jnp.broadcast_to(
        jnp.arange(module.action_count, dtype=jnp.int32),
        prefix + (module.action_count,),
    )
    task = jnp.broadcast_to(
        task_features[..., None, :],
        prefix + (module.action_count, task_features.shape[-1]),
    )
    belief = jnp.broadcast_to(
        belief_embedding[..., None, :],
        prefix + (module.action_count, belief_embedding.shape[-1]),
    )
    return module.decoder(jax.lax.stop_gradient(task), belief, actions)


def _selected_from_all(all_prediction, actions, observation_shape):
    mean, log_std, reward, reward_log_std, done = all_prediction
    action_axis = mean.ndim - 2
    return ResponsePrediction(
        observation_delta_mean=gather_actions(
            mean.reshape(mean.shape[:action_axis + 1] + observation_shape),
            actions,
            action_axis=action_axis,
        ),
        observation_delta_log_std=gather_actions(
            log_std.reshape(log_std.shape[:action_axis + 1] + observation_shape),
            actions,
            action_axis=action_axis,
        ),
        reward_mean=gather_actions(reward, actions),
        reward_log_std=gather_actions(reward_log_std, actions),
        done_logit=gather_actions(done, actions),
    )


def test_selected_response_matches_all_action_reference_and_gradient() -> None:
    unused_config, model, params, unused_state, output = _fixture()
    del unused_config, unused_state
    actions = jnp.asarray([1, 5], dtype=jnp.int32)
    selected = model.apply(
        {"params": params},
        output.task_features,
        output.belief_embedding,
        actions,
        method=model.response_from_context_and_action,
    )
    all_prediction = model.apply(
        {"params": params},
        output.task_features,
        output.belief_embedding,
        method=_all_action_prediction,
    )
    reference = _selected_from_all(all_prediction, actions, (5, 5, 8))
    for actual, expected in zip(selected, reference):
        np.testing.assert_allclose(
            np.asarray(actual), np.asarray(expected), rtol=1.0e-6, atol=1.0e-6
        )

    observations = jnp.stack(
        (
            jnp.zeros((2, 5, 5, 8), dtype=jnp.float32),
            jnp.ones((2, 5, 5, 8), dtype=jnp.float32),
        ),
        axis=0,
    )
    rewards = jnp.asarray([0.25, -0.5], dtype=jnp.float32)
    dones = jnp.asarray([False, True], dtype=jnp.bool_)

    def response_loss_from_selected(decoder_params):
        prediction = model.apply(
            {"params": {"response_decoder": decoder_params}},
            output.task_features,
            output.belief_embedding,
            actions,
            method=model.response_from_context_and_action,
        )
        return response_prediction_loss(
            prediction=prediction,
            observations=observations,
            response_next_observations=observations[1:],
            rewards=rewards,
            dones=dones,
        )

    def response_loss_from_reference(decoder_params):
        prediction = model.apply(
            {"params": {"response_decoder": decoder_params}},
            output.task_features,
            output.belief_embedding,
            method=_all_action_prediction,
        )
        selected_prediction = _selected_from_all(
            prediction, actions, (5, 5, 8)
        )
        return response_prediction_loss(
            prediction=selected_prediction,
            observations=observations,
            response_next_observations=observations[1:],
            rewards=rewards,
            dones=dones,
        )

    def response_loss_from_selected_belief(belief):
        prediction = model.apply(
            {"params": params},
            output.task_features,
            belief,
            actions,
            method=model.response_from_context_and_action,
        )
        return response_prediction_loss(
            prediction=prediction,
            observations=observations,
            response_next_observations=observations[1:],
            rewards=rewards,
            dones=dones,
        )

    def response_loss_from_reference_belief(belief):
        prediction = model.apply(
            {"params": params},
            output.task_features,
            belief,
            method=_all_action_prediction,
        )
        selected_prediction = _selected_from_all(
            prediction, actions, (5, 5, 8)
        )
        return response_prediction_loss(
            prediction=selected_prediction,
            observations=observations,
            response_next_observations=observations[1:],
            rewards=rewards,
            dones=dones,
        )

    np.testing.assert_allclose(
        np.asarray(response_loss_from_selected(params["response_decoder"])),
        np.asarray(response_loss_from_reference(params["response_decoder"])),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(
            jax.grad(response_loss_from_selected_belief)(
                output.belief_embedding
            )
        ),
        np.asarray(
            jax.grad(response_loss_from_reference_belief)(
                output.belief_embedding
            )
        ),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    selected_loss_gradient = jax.grad(response_loss_from_selected)(
        params["response_decoder"]
    )
    reference_loss_gradient = jax.grad(response_loss_from_reference)(
        params["response_decoder"]
    )
    for actual, expected in zip(
        jax.tree_util.tree_leaves(selected_loss_gradient),
        jax.tree_util.tree_leaves(reference_loss_gradient),
    ):
        np.testing.assert_allclose(
            np.asarray(actual), np.asarray(expected), rtol=1.0e-6, atol=1.0e-6
        )

    def selected_sum(belief):
        prediction = model.apply(
            {"params": params},
            output.task_features,
            belief,
            actions,
            method=model.response_from_context_and_action,
        )
        return sum(jnp.sum(value) for value in prediction)

    def reference_sum(belief):
        prediction = model.apply(
            {"params": params},
            output.task_features,
            belief,
            method=_all_action_prediction,
        )
        selected_prediction = _selected_from_all(
            prediction, actions, (5, 5, 8)
        )
        return sum(jnp.sum(value) for value in selected_prediction)

    np.testing.assert_allclose(
        np.asarray(jax.grad(selected_sum)(output.belief_embedding)),
        np.asarray(jax.grad(reference_sum)(output.belief_embedding)),
        rtol=1.0e-6,
        atol=1.0e-6,
    )

    def selected_parameter_sum(decoder_params):
        prediction = model.apply(
            {"params": {"response_decoder": decoder_params}},
            output.task_features,
            output.belief_embedding,
            actions,
            method=model.response_from_context_and_action,
        )
        return sum(jnp.sum(value) for value in prediction)

    def reference_parameter_sum(decoder_params):
        prediction = model.apply(
            {"params": {"response_decoder": decoder_params}},
            output.task_features,
            output.belief_embedding,
            method=_all_action_prediction,
        )
        selected_prediction = _selected_from_all(
            prediction, actions, (5, 5, 8)
        )
        return sum(jnp.sum(value) for value in selected_prediction)

    selected_gradient = jax.grad(selected_parameter_sum)(
        params["response_decoder"]
    )
    reference_gradient = jax.grad(reference_parameter_sum)(
        params["response_decoder"]
    )
    for actual, expected in zip(
        jax.tree_util.tree_leaves(selected_gradient),
        jax.tree_util.tree_leaves(reference_gradient),
    ):
        np.testing.assert_allclose(
            np.asarray(actual), np.asarray(expected), rtol=1.0e-6, atol=1.0e-6
        )


def test_critic_only_teacher_matches_policy_teacher_without_decoder_params() -> None:
    config, model, params, unused_state, output = _fixture()
    del unused_state
    latent = jnp.linspace(
        -0.5, 0.5, 2 * config.model.latent_dim, dtype=jnp.float32
    ).reshape((2, config.model.latent_dim))
    teacher = model.apply(
        {"params": params},
        output.task_features,
        latent,
        1.0,
        method=model.from_features_and_latent,
    )
    params_without_decoder = {
        name: value for name, value in params.items() if name != "response_decoder"
    }
    action_values = model.apply(
        {"params": params_without_decoder},
        output.task_features,
        latent,
        method=model.action_values_from_features_and_latent,
    )
    np.testing.assert_allclose(
        np.asarray(action_values),
        np.asarray(teacher.action_values),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_chunking_preserves_keys_samples_and_regret() -> None:
    config, model, params, state, unused_output = _fixture()
    del unused_output
    observations = jnp.arange(3 * 2 * 5 * 5 * 8, dtype=jnp.float32).reshape(
        (3, 2, 5, 5, 8)
    ) / 511.0
    unused_final, context = model.apply(
        {"params": params},
        state,
        observations,
        jnp.zeros((3, 2), dtype=jnp.int32),
        jnp.zeros((3, 2), dtype=jnp.float32),
        jnp.zeros((3, 2), dtype=jnp.bool_),
        method=model.context_sequence,
    )
    del unused_final
    key = jax.random.PRNGKey(93)
    chunked_two = chunked_decision_regret(
        model=model,
        target_params=params,
        context=context,
        key=key,
        posterior_particles=config.model.posterior_particles,
        state_chunk_size=2,
    )
    chunked_five = chunked_decision_regret(
        model=model,
        target_params=params,
        context=context,
        key=key,
        posterior_particles=config.model.posterior_particles,
        state_chunk_size=5,
    )

    state_count = 6
    keys = jax.random.split(key, state_count)
    flat_task = context.task_features.reshape((state_count, -1))
    flat_logits = context.mixture_logits.reshape(
        (state_count, config.model.mixture_components)
    )
    flat_means = context.mixture_means.reshape(
        (state_count, config.model.mixture_components, config.model.latent_dim)
    )
    flat_log_variances = context.mixture_log_variances.reshape(flat_means.shape)

    def one(feature, logits, means, log_variances, sample_key):
        samples, weights = stratified_mixture_samples(
            sample_key,
            mixture_logits=logits,
            means=means,
            log_variances=log_variances,
            sample_count=config.model.posterior_particles,
        )
        features = jnp.broadcast_to(
            feature[None, :],
            (config.model.posterior_particles, feature.shape[-1]),
        )
        action_values = model.apply(
            {"params": params},
            features,
            samples,
            method=model.action_values_from_features_and_latent,
        )
        return decision_regret_from_action_values(action_values, weights)

    reference = jax.vmap(one)(
        flat_task,
        flat_logits,
        flat_means,
        flat_log_variances,
        keys,
    ).reshape((3, 2))
    np.testing.assert_allclose(
        np.asarray(chunked_two), np.asarray(reference), rtol=1.0e-6, atol=1.0e-6
    )
    np.testing.assert_allclose(
        np.asarray(chunked_five), np.asarray(reference), rtol=1.0e-6, atol=1.0e-6
    )


def test_terminal_potential_is_zero_and_target_regret_is_detached() -> None:
    current = jnp.asarray([2.0, 3.0], dtype=jnp.float32)
    following = jnp.asarray([100.0, 5.0], dtype=jnp.float32)
    dones = jnp.asarray([True, False], dtype=jnp.bool_)

    shaped = potential_shaping(
        current,
        following,
        dones,
        gamma=0.99,
        weight=0.1,
    )
    np.testing.assert_allclose(
        np.asarray(shaped),
        np.asarray([0.2, 0.1 * (3.0 - 0.99 * 5.0)]),
        rtol=1.0e-6,
        atol=1.0e-6,
    )

    def total(cur, nxt):
        return jnp.sum(
            potential_shaping(
                cur,
                nxt,
                dones,
                gamma=0.99,
                weight=0.1,
            )
        )

    current_gradient, next_gradient = jax.grad(total, argnums=(0, 1))(
        current, following
    )
    np.testing.assert_array_equal(np.asarray(current_gradient), 0.0)
    np.testing.assert_array_equal(np.asarray(next_gradient), 0.0)


def test_formal_regret_tensor_budget_eliminates_failed_decoder_shape() -> None:
    old_decoder_hidden_bytes = 257 * 256 * 16 * 6 * 128 * 4
    assert old_decoder_hidden_bytes == 3_233_808_384
    critic_hidden_bytes_per_chunk = (
        DECISION_REGRET_STATE_CHUNK_SIZE * 16 * 128 * 4
    )
    assert critic_hidden_bytes_per_chunk == 32 * 2**20
    assert old_decoder_hidden_bytes / critic_hidden_bytes_per_chunk == 96.375


@pytest.mark.skipif(
    os.environ.get("DELTA_RUN_FORMAL_GPU_MEMORY_TEST") != "1",
    reason="Explicit 46-GiB GPU acceptance fixture only.",
)
def test_formal_regret_fixture_fits_registered_gpu() -> None:
    config = load_config(
        ROOT
        / "experiments"
        / "overcooked_v2"
        / "configs"
        / "delta_zsc_simple_formal.yaml",
        run_kind="formal",
    )
    observation_shape = (5, 5, 39)
    model = build_model(
        observation_shape=observation_shape,
        action_count=6,
        **{
            name: getattr(config.model, name)
            for name in (
                "task_hidden_dim",
                "belief_hidden_dim",
                "latent_dim",
                "mixture_components",
                "belief_embedding_dim",
                "actor_hidden_dim",
                "critic_hidden_dim",
                "response_hidden_dim",
                "modulation_rank",
                "action_embedding_dim",
                "log_variance_minimum",
                "log_variance_maximum",
                "response_log_std_minimum",
                "response_log_std_maximum",
            )
        },
    )
    state = initial_policy_state(
        batch_size=1,
        observation_shape=observation_shape,
        action_count=6,
        task_hidden_dim=config.model.task_hidden_dim,
        belief_hidden_dim=config.model.belief_hidden_dim,
        latent_dim=config.model.latent_dim,
        mixture_components=config.model.mixture_components,
    )
    params = initialize_model_parameters(
        model,
        key=jax.random.PRNGKey(707),
        example_state=state,
        example_observation=jnp.zeros((1,) + observation_shape, dtype=jnp.float32),
        partner_code_dim=config.partner_generator.code_dim,
    )
    critic_params = {
        name: params[name]
        for name in ("belief_set_encoder", "universal_critic")
    }
    context = ContextOutput(
        task_features=jnp.zeros((257, 256, 128), dtype=jnp.float32),
        mixture_logits=jnp.zeros((257, 256, 4), dtype=jnp.float32),
        mixture_means=jnp.zeros((257, 256, 4, 8), dtype=jnp.float32),
        mixture_log_variances=jnp.zeros((257, 256, 4, 8), dtype=jnp.float32),
        support_score=jnp.ones((257, 256), dtype=jnp.float32),
    )
    evaluate = jax.jit(
        lambda value, sample_key: chunked_decision_regret(
            model=model,
            target_params=critic_params,
            context=value,
            key=sample_key,
            posterior_particles=16,
        )
    )
    regret = evaluate(context, jax.random.PRNGKey(708))
    jax.block_until_ready(regret)
    assert regret.shape == (257, 256)
    assert np.all(np.isfinite(np.asarray(regret)))
    peak = peak_device_memory_bytes()
    assert 0 < peak < 40_000 * 2**20
