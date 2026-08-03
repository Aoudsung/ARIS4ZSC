from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("flax")

from src.path_c.experiment import load_config  # noqa: E402
from src.path_c.model import (  # noqa: E402
    build_model,
    initial_policy_state,
    initialize_model_parameters,
)
from src.path_c.resources import peak_device_memory_bytes  # noqa: E402
from src.path_c.runner import (  # noqa: E402
    DECISION_REGRET_STATE_CHUNK_SIZE,
    chunked_decision_regret,
    decision_regret_chunk,
)
from src.path_c.response_targets import (  # noqa: E402
    PartnerResponseTargets,
    structured_partner_response_loss,
)
from src.path_c.types import ContextOutput, ResponsePrediction  # noqa: E402


ROOT = Path(__file__).resolve().parents[3]


def _fixture():
    config = load_config(
        ROOT / "experiments/overcooked_v2/configs/delta_zsc_simple_development.yaml",
        run_kind="mechanical",
    )
    shape = (5, 5, 39)
    model = build_model(
        observation_shape=shape,
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
        batch_size=2,
        observation_shape=shape,
        action_count=6,
        task_hidden_dim=config.model.task_hidden_dim,
        belief_hidden_dim=config.model.belief_hidden_dim,
        latent_dim=config.model.latent_dim,
    )
    observation = jnp.arange(2 * 5 * 5 * 39, dtype=jnp.float32).reshape(
        (2,) + shape
    ) / 255.0
    params = initialize_model_parameters(
        model,
        key=jax.random.PRNGKey(11),
        example_state=state,
        example_observation=observation,
        partner_code_dim=config.partner_generator.code_dim,
    )
    _, output = model.apply(
        {"params": params}, state, observation,
        jnp.zeros((2,), dtype=jnp.bool_), method=model.step,
    )
    return config, model, params, state, output


def _all_action_prediction(module, task_features, belief_summary):
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
        belief_summary[..., None, :],
        prefix + (module.action_count, belief_summary.shape[-1]),
    )
    return module.decoder(jax.lax.stop_gradient(task), belief, actions)


def _selected(all_prediction, actions):
    lane = jnp.arange(actions.shape[0], dtype=jnp.int32)
    return ResponsePrediction(*(value[lane, actions] for value in all_prediction))


def _targets():
    return PartnerResponseTargets(
        visibility=jnp.asarray([1.0, 0.0]),
        relative_position=jnp.asarray([3, 25]),
        direction=jnp.asarray([1, 0]),
        inventory=jnp.asarray([[0, 1, 2, 3, 0], [0, 0, 0, 0, 0]]),
        interaction_change=jnp.asarray([1.0, 0.0]),
        visible_mask=jnp.asarray([1.0, 0.0]),
    )


def test_selected_response_equals_all_action_gather_and_gradients() -> None:
    _, model, params, _, output = _fixture()
    actions = jnp.asarray([1, 5], dtype=jnp.int32)
    selected = model.apply(
        {"params": params}, output.task_features, output.belief_summary, actions,
        method=model.response_from_context_and_action,
    )
    all_prediction = model.apply(
        {"params": params}, output.task_features, output.belief_summary,
        method=_all_action_prediction,
    )
    reference = _selected(all_prediction, actions)
    for actual, expected in zip(selected, reference, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def selected_loss(belief):
        prediction = model.apply(
            {"params": params}, output.task_features, belief, actions,
            method=model.response_from_context_and_action,
        )
        return structured_partner_response_loss(prediction, _targets()).total

    def gathered_loss(belief):
        prediction = model.apply(
            {"params": params}, output.task_features, belief,
            method=_all_action_prediction,
        )
        return structured_partner_response_loss(
            _selected(prediction, actions), _targets()
        ).total

    np.testing.assert_allclose(
        jax.grad(selected_loss)(output.belief_summary),
        jax.grad(gathered_loss)(output.belief_summary),
        rtol=1e-6,
        atol=1e-6,
    )


def test_chunking_preserves_gaussian_samples_regret_and_action_ranges() -> None:
    config, model, params, state, _ = _fixture()
    observations = jnp.arange(3 * 2 * 5 * 5 * 39, dtype=jnp.float32).reshape(
        (3, 2, 5, 5, 39)
    ) / 511.0
    _, context = model.apply(
        {"params": params}, state, observations,
        jnp.zeros((3, 2), dtype=jnp.int32),
        jnp.zeros((3, 2), dtype=jnp.bool_), method=model.context_sequence,
    )
    key = jax.random.PRNGKey(93)
    chunked_two = chunked_decision_regret(
        model=model, target_params=params, context=context, key=key,
        posterior_particles=config.model.posterior_particles, state_chunk_size=2,
    )
    chunked_five = chunked_decision_regret(
        model=model, target_params=params, context=context, key=key,
        posterior_particles=config.model.posterior_particles, state_chunk_size=5,
    )
    count = 6
    reference = decision_regret_chunk(
        model=model,
        target_params=params,
        task_features=context.task_features.reshape((count, -1)),
        belief_mean=context.belief_mean.reshape((count, config.model.latent_dim)),
        belief_log_standard_deviation=context.belief_log_standard_deviation.reshape(
            (count, config.model.latent_dim)
        ),
        sample_keys=jax.random.split(key, count),
        posterior_particles=config.model.posterior_particles,
    )
    for observed in (chunked_two, chunked_five):
        np.testing.assert_allclose(observed[0], reference[0].reshape((3, 2)), atol=1e-6)
        np.testing.assert_allclose(observed[1], reference[1].reshape((3, 2)), atol=1e-6)


def test_regret_chunk_has_bounded_formal_hidden_tensor() -> None:
    # 4,096 states x 16 particles x 128 critic units x fp32.
    critic_hidden_bytes = DECISION_REGRET_STATE_CHUNK_SIZE * 16 * 128 * 4
    assert critic_hidden_bytes == 32 * 2**20


@pytest.mark.skipif(
    os.environ.get("DELTA_RUN_FORMAL_GPU_MEMORY_TEST") != "1",
    reason="Explicit registered CUDA acceptance fixture only.",
)
def test_formal_regret_fixture_fits_registered_cuda_memory() -> None:
    config, model, params, _, _ = _fixture()
    context = ContextOutput(
        task_features=jnp.zeros((257, 256, 128), dtype=jnp.float32),
        belief_mean=jnp.zeros((257, 256, 8), dtype=jnp.float32),
        belief_log_standard_deviation=jnp.zeros((257, 256, 8), dtype=jnp.float32),
        normalized_uncertainty=jnp.full((257, 256), 5.0 / 7.0),
    )
    critic_params = {"universal_critic": params["universal_critic"]}
    regret, ranges = chunked_decision_regret(
        model=model,
        target_params=critic_params,
        context=context,
        key=jax.random.PRNGKey(708),
        posterior_particles=config.model.posterior_particles,
    )
    jax.block_until_ready((regret, ranges))
    assert regret.shape == ranges.shape == (257, 256)
    assert np.all(np.isfinite(np.asarray(regret)))
    peak = peak_device_memory_bytes()
    assert 0 < peak < 40_000 * 2**20
