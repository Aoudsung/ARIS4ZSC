from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("flax")
pytest.importorskip("optax")

from experiments.overcooked_v2.deployment import (  # noqa: E402
    DEPLOYABLE_PARAM_NAMES,
    deployable_parameters,
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


ROOT = Path(__file__).resolve().parents[3]


def test_model_initialization_forward_and_minimal_gradient_update() -> None:
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
    observation = jnp.zeros((2,) + observation_shape, dtype=jnp.float32)
    params = initialize_model_parameters(
        model,
        key=jax.random.PRNGKey(0),
        example_state=state,
        example_observation=observation,
        partner_code_dim=config.partner_generator.code_dim,
    )
    task = params["task_encoder"]
    assert task["official_conv_0"]["kernel"].shape == (1, 1, 8, 128)
    assert task["official_conv_1"]["kernel"].shape == (1, 1, 128, 128)
    assert task["official_conv_2"]["kernel"].shape == (1, 1, 128, 8)
    assert task["official_conv_3"]["kernel"].shape == (3, 3, 8, 16)
    assert task["official_conv_4"]["kernel"].shape == (3, 3, 16, 32)
    assert task["official_conv_5"]["kernel"].shape == (3, 3, 32, 32)
    assert task["official_dense"]["kernel"].shape[-1] == 128
    assert task["task_gru"]["ir"]["kernel"].shape == (128, 128)
    _, output = model.apply(
        {"params": params},
        state,
        observation,
        jnp.ones((2,), dtype=jnp.float32),
        method=model.step,
    )
    assert output.execution_logits.shape == (2, 6)
    assert output.action_values.shape == (2, 6)
    assert output.response_observation_delta_mean.shape == (2, 6) + observation_shape
    assert np.all(np.isfinite(np.asarray(output.execution_logits)))

    assert set(DEPLOYABLE_PARAM_NAMES).issubset(params)
    assert "code_teacher" in params
    assert "full_trajectory_teacher" in params
    assert set(deployable_parameters(params)) == set(DEPLOYABLE_PARAM_NAMES)

    def objective(candidate):
        _, current = model.apply(
            {"params": candidate},
            state,
            observation,
            jnp.ones((2,), dtype=jnp.float32),
            method=model.step,
        )
        return jnp.mean(jnp.square(current.action_values)) + jnp.mean(
            jnp.square(current.execution_logits)
        )

    gradients = jax.grad(objective)(params)
    optimizer, optimizer_state = make_optimizer(
        params, learning_rate=1.0e-4, gradient_clip_norm=0.5
    )
    updates, _ = optimizer.update(gradients, optimizer_state, params)
    import optax

    updated = optax.apply_updates(params, updates)
    assert pytree_fingerprint(updated) != pytree_fingerprint(params)


def test_deployable_belief_never_consumes_unavailable_transition_reward() -> None:
    config = load_config(
        ROOT
        / "experiments"
        / "overcooked_v2"
        / "configs"
        / "delta_zsc_simple_development.yaml",
        run_kind="mechanical",
    )
    observation_shape = (5, 5, 8)
    state = initial_policy_state(
        batch_size=2,
        observation_shape=observation_shape,
        action_count=6,
        task_hidden_dim=config.model.task_hidden_dim,
        belief_hidden_dim=config.model.belief_hidden_dim,
        latent_dim=config.model.latent_dim,
        mixture_components=config.model.mixture_components,
    )
    updated = observe_policy_after_transition(
        stepped_state=state,
        action=jnp.asarray([1, 2], dtype=jnp.int32),
        reward=jnp.asarray([20.0, -10.0], dtype=jnp.float32),
        done=jnp.asarray([False, False]),
        next_observation=jnp.zeros((2,) + observation_shape, dtype=jnp.float32),
        model_config=config.model,
    )
    np.testing.assert_array_equal(np.asarray(updated.previous_reward), [0.0, 0.0])
