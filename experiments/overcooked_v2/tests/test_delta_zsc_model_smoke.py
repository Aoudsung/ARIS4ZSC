"""Model smoke tests for the DEPI three-object architecture.

Specification entries covered (docs/METHOD_SPEC.md):
- §1.1 three pathways (task/capability/protocol) with shared component
  embeddings m_k (K=4, D=16) and context concat(u, c_t).
- §1.2 prior context dropout replaces (u, c_t) with the prior context.
- §1.4 PolicyState field list and ContextOutput contents.
- §2.2 response mixture heads each carry a trailing K component axis.
- §3.3 a single combined gradient step over the deployable parameter tree.
"""

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
    mixture_summary,
    prior_mixture_summary,
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
from src.path_c.types import ContextOutput, PolicyState  # noqa: E402


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
        capability_hidden_dim=config.model.capability_hidden_dim,
        protocol_hidden_dim=config.model.protocol_hidden_dim,
        capability_dim=config.model.capability_dim,
        protocol_components=config.model.protocol_components,
        component_embedding_dim=config.model.component_embedding_dim,
        actor_hidden_dim=config.model.actor_hidden_dim,
        critic_hidden_dim=config.model.critic_hidden_dim,
        response_hidden_dim=config.model.response_hidden_dim,
        modulation_rank=config.model.modulation_rank,
        action_embedding_dim=config.model.action_embedding_dim,
    )
    state = initial_policy_state(
        batch_size=batch_size,
        observation_shape=observation_shape,
        action_count=6,
        task_hidden_dim=config.model.task_hidden_dim,
        capability_hidden_dim=config.model.capability_hidden_dim,
        protocol_hidden_dim=config.model.protocol_hidden_dim,
        capability_dim=config.model.capability_dim,
        component_embedding_dim=config.model.component_embedding_dim,
        protocol_components=config.model.protocol_components,
    )
    observation = jnp.arange(
        batch_size * 5 * 5 * 39, dtype=jnp.float32
    ).reshape((batch_size,) + observation_shape) / 255.0
    params = initialize_model_parameters(
        model,
        key=jax.random.PRNGKey(0),
        example_state=state,
        example_observation=observation,
    )
    return config, model, params, state, observation


def test_depi_forward_and_minimal_gradient_update() -> None:
    config, model, params, state, observation = _fixture()
    task = params["task_encoder"]
    assert task["official_conv_0"]["kernel"].shape == (1, 1, 39, 128)
    assert task["official_conv_5"]["kernel"].shape == (3, 3, 32, 32)
    assert task["task_gru"]["ir"]["kernel"].shape == (128, 128)
    assert params["protocol_component_embeddings"]["embedding"].shape == (4, 16)

    next_state, output = model.apply(
        {"params": params}, state, observation, jnp.zeros((2,), dtype=jnp.bool_),
        method=model.step,
    )
    assert output.policy_logits.shape == (2, 6)
    assert output.state_value.shape == (2,)
    assert output.raw_q1.shape == output.raw_q2.shape == (2, 6)
    assert output.action_values.shape == (2, 6)
    # §1.1/§1.4: u is 16-dimensional, pi is a K=4 posterior, c_t is 16-dimensional.
    assert output.capability.shape == (2, 16)
    assert output.protocol_probabilities.shape == (2, 4)
    assert output.protocol_embedding.shape == (2, 16)
    assert output.context_summary.shape == (2, 32)
    assert output.posterior_entropy.shape == (2,)
    np.testing.assert_allclose(
        np.sum(np.asarray(output.protocol_probabilities), axis=-1), 1.0, atol=1e-6
    )
    assert np.all(np.asarray(output.protocol_probabilities) >= 0.0)
    assert np.all(np.isfinite(np.asarray(output.policy_logits)))

    context = ContextOutput(
        output.task_features,
        output.capability,
        output.protocol_probabilities,
        output.protocol_embedding,
    )
    response = model.apply(
        {"params": params},
        context,
        observation,
        jnp.asarray([0, 5], dtype=jnp.int32),
        method=model.response_from_context_and_action,
    )
    # §2.2: every response head carries a trailing K=4 component axis.
    assert response.visibility_logit.shape == (2, 4)
    assert response.relative_position_logits.shape == (2, 4, 26)
    assert response.direction_logits.shape == (2, 4, 4)
    assert response.inventory_logits.shape == (2, 4, 5, 4)
    assert response.interaction_change_logit.shape == (2, 4)
    assert response.posterior_log_probabilities.shape == (2, 4)

    observations = jnp.broadcast_to(observation[None], (3,) + observation.shape)
    previous_actions = jnp.zeros((3, 2), dtype=jnp.int32)
    starts = jnp.zeros((3, 2), dtype=jnp.bool_)
    dropout = jnp.asarray([[False, True], [True, False], [False, False]])
    final_state, sequence = model.apply(
        {"params": params}, state, observations, previous_actions, starts, dropout,
        method=model.sequence,
    )
    context_state, context_sequence = model.apply(
        {"params": params}, state, observations, previous_actions, starts,
        method=model.context_sequence,
    )
    assert sequence.policy_logits.shape == (3, 2, 6)
    np.testing.assert_allclose(
        context_sequence.task_features, sequence.task_features, atol=1e-6
    )
    np.testing.assert_allclose(
        context_sequence.capability, sequence.capability, atol=1e-6
    )
    np.testing.assert_allclose(
        context_sequence.protocol_probabilities,
        sequence.protocol_probabilities,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        context_state.task_carry, final_state.task_carry, atol=1e-6
    )
    assert final_state.previous_observation.dtype == jnp.float32

    assert set(deployable_parameters(params)) == set(DEPLOYABLE_PARAM_NAMES)
    assert set(DEPLOYABLE_PARAM_NAMES) == {
        "task_encoder",
        "capability_encoder",
        "protocol_encoder",
        "protocol_component_embeddings",
        "universal_actor",
    }
    assert "partner_generator" not in params
    assert "universal_critic" in params and "response_decoder" in params

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


def test_mixture_summary_and_uniform_prior_are_registered() -> None:
    probabilities = jnp.asarray(
        [[0.7, 0.1, 0.1, 0.1], [0.25, 0.25, 0.25, 0.25]], dtype=jnp.float32
    )
    embeddings = jnp.arange(4 * 16, dtype=jnp.float32).reshape((4, 16))
    summary = mixture_summary(probabilities, embeddings)
    prior = prior_mixture_summary(probabilities, embeddings)
    assert summary.shape == prior.shape == (2, 16)
    np.testing.assert_allclose(summary[1], prior[1], atol=1e-6)
    np.testing.assert_allclose(prior[0], jnp.mean(embeddings, axis=0), atol=1e-6)
    np.testing.assert_allclose(
        summary[0], jnp.einsum("k,kd->d", probabilities[0], embeddings), atol=1e-6
    )


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
