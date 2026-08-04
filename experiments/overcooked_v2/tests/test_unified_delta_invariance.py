from __future__ import annotations

from types import SimpleNamespace

import numpy as np


def test_base_update_is_identical_across_trainable_variants() -> None:
    import jax
    import jax.numpy as jnp

    from src.delta_zsc.model import (
        BASE_VARIANT,
        JOINT_VARIANT,
        RESPONSE_ONLY_VARIANT,
        UnifiedAgent,
        build_models,
        initial_agent_state,
        initialize_parameters,
    )
    from src.delta_zsc.training import base_update, make_optimizer
    from src.delta_zsc.types import RolloutBatch

    batch_size = 2
    time_count = 2
    observation_shape = (5, 5, 39)
    base_model, latent_model = build_models(
        observation_shape=observation_shape,
        action_count=6,
        task_hidden_dim=16,
        instant_partner_dim=8,
        latent_hidden_dim=8,
        response_hidden_dim=16,
        action_embedding_dim=4,
        component_count=2,
    )
    state = initial_agent_state(
        batch_size=batch_size,
        observation_shape=observation_shape,
        task_hidden_dim=16,
        component_count=2,
    )
    observations = jnp.zeros(
        (time_count + 1, batch_size, *observation_shape), dtype=jnp.float32
    )
    base_params, latent_params = initialize_parameters(
        base_model=base_model,
        latent_model=latent_model,
        base_key=jax.random.PRNGKey(0),
        latent_key=jax.random.PRNGKey(1),
        example_state=state,
        example_observation=observations[0],
    )
    agent = UnifiedAgent(
        base_model=base_model,
        latent_model=latent_model,
        action_count=6,
        component_count=2,
        adaptation_kl_budget=0.04,
        gamma=0.99,
    )
    probability = jnp.full((time_count, batch_size, 6), 1.0 / 6.0)
    batch = RolloutBatch(
        observations=observations,
        response_next_observations=observations[1:],
        previous_actions=jnp.zeros(
            (time_count + 1, batch_size), dtype=jnp.int32
        ),
        episode_starts=jnp.zeros(
            (time_count + 1, batch_size), dtype=jnp.bool_
        ).at[0].set(True),
        actions=jnp.asarray([[0, 1], [2, 3]], dtype=jnp.int32),
        rewards=jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32),
        shaped_rewards=jnp.asarray(
            [[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32
        ),
        dones=jnp.zeros((time_count, batch_size), dtype=jnp.bool_),
        behavior_probabilities=probability,
        old_log_probabilities=jnp.full(
            (time_count, batch_size), -jnp.log(6.0)
        ),
        old_values=jnp.zeros((time_count + 1, batch_size)),
        ppo_mask=jnp.ones((time_count, batch_size)),
        initial_state=state,
    )
    ppo = SimpleNamespace(
        gamma=0.99,
        gae_lambda=0.95,
        normalize_advantages=True,
        clip_epsilon=0.2,
        value_weight=0.5,
        entropy_weight=0.01,
    )
    optimizer, optimizer_state = make_optimizer(
        base_params,
        learning_rate=2.5e-4,
        gradient_clip_norm=0.25,
        adam_epsilon=1.0e-5,
    )
    updated = {}
    for variant in (BASE_VARIANT, RESPONSE_ONLY_VARIANT, JOINT_VARIANT):
        params, _, _ = base_update(
            agent=agent,
            base_params=base_params,
            latent_params=latent_params,
            optimizer=optimizer,
            optimizer_state=optimizer_state,
            batch=batch,
            ppo=ppo,
            variant=variant,
        )
        updated[variant] = params
    reference = jax.tree_util.tree_leaves(updated[BASE_VARIANT])
    for variant in (RESPONSE_ONLY_VARIANT, JOINT_VARIANT):
        for left, right in zip(
            reference,
            jax.tree_util.tree_leaves(updated[variant]),
            strict=True,
        ):
            np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
