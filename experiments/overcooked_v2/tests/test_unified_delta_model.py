from __future__ import annotations

import numpy as np


def _models(batch_size: int = 2):
    import jax
    import jax.numpy as jnp

    from src.delta_zsc.model import (
        FULL_VARIANT,
        UnifiedAgent,
        build_models,
        initial_agent_state,
        initialize_parameters,
    )

    observation_shape = (5, 5, 39)
    base, latent = build_models(
        observation_shape=observation_shape,
        action_count=6,
        task_hidden_dim=32,
        instant_partner_dim=8,
        latent_hidden_dim=16,
        response_hidden_dim=32,
        action_embedding_dim=8,
        component_count=4,
    )
    state = initial_agent_state(
        batch_size=batch_size,
        observation_shape=observation_shape,
        task_hidden_dim=32,
        component_count=4,
    )
    observation = jnp.zeros((batch_size, *observation_shape), dtype=jnp.float32)
    base_params, latent_params = initialize_parameters(
        base_model=base,
        latent_model=latent,
        base_key=jax.random.PRNGKey(0),
        latent_key=jax.random.PRNGKey(1),
        example_state=state,
        example_observation=observation,
    )
    agent = UnifiedAgent(
        base_model=base,
        latent_model=latent,
        action_count=6,
        component_count=4,
        adaptation_kl_budget=0.04,
        gamma=0.99,
    )
    return agent, state, observation, base_params, latent_params, FULL_VARIANT


def test_model_initializes_and_executes_all_paths() -> None:
    agent, state, observation, base_params, latent_params, variant = _models()
    next_state, output = agent.step(
        base_params=base_params,
        latent_params=latent_params,
        state=state,
        observation=observation,
        variant=variant,
    )
    assert next_state.belief.shape == (2, 4)
    assert output.base.base_logits.shape == (2, 6)
    assert output.latent.decision_mean.shape == (2, 4, 6)
    assert output.latent.value_of_information.shape == (2, 6)
    assert np.all(np.asarray(output.latent.adaptation_kl) <= 0.04001)
    assert np.all(np.isfinite(np.asarray(output.latent.adapted_logits)))


def test_response_base_control_is_nested_not_zero_embedding_ood() -> None:
    import jax.numpy as jnp

    agent, state, observation, _, latent_params, _ = _models()
    statistics = jnp.concatenate(
        (
            state.behavior.alpha / (state.behavior.alpha + state.behavior.beta),
            jnp.log1p(state.behavior.alpha + state.behavior.beta),
        ),
        axis=-1,
    )
    full = agent.latent_model.apply(
        {"params": latent_params},
        observation,
        statistics,
        state.previous_action,
        include_component_residual=True,
        method=agent.latent_model.response_logits,
    )
    base = agent.latent_model.apply(
        {"params": latent_params},
        observation,
        statistics,
        state.previous_action,
        include_component_residual=False,
        method=agent.latent_model.response_logits,
    )
    # Residual heads initialize to zero, making the nested control exact before
    # learning and always valid after learning.
    np.testing.assert_allclose(
        np.asarray(full.visibility), np.asarray(base.visibility), atol=1e-6
    )


def test_response_spatial_encoder_distinguishes_partner_locations() -> None:
    import jax.numpy as jnp

    from src.path_c.task_encoder import official_partner_channel_indexes

    agent, state, observation, _, latent_params, _ = _models(batch_size=2)
    partner_position_channel = official_partner_channel_indexes(39)[0]
    frames = observation.at[0, 0, 0, partner_position_channel].set(1.0)
    frames = frames.at[1, 4, 4, partner_position_channel].set(1.0)
    statistics = jnp.concatenate(
        (
            state.behavior.alpha / (state.behavior.alpha + state.behavior.beta),
            jnp.log1p(state.behavior.alpha + state.behavior.beta),
        ),
        axis=-1,
    )
    logits = agent.latent_model.apply(
        {"params": latent_params},
        frames,
        statistics,
        state.previous_action,
        include_component_residual=False,
        method=agent.latent_model.response_logits,
    )
    assert not np.allclose(
        np.asarray(logits.relative_position[0]),
        np.asarray(logits.relative_position[1]),
    )
