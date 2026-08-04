from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_formal_config_has_exactly_three_method_choices() -> None:
    from src.delta_zsc.config import load_config

    config = load_config(
        Path("experiments/overcooked_v2/configs/delta_unified_simple_formal.yaml")
    )
    assert set(config.method.__dataclass_fields__) == {
        "latent_components",
        "continuation_horizon",
        "adaptation_kl_budget",
    }
    assert config.method.latent_components == 4
    assert config.method.continuation_horizon == 128
    assert config.method.adaptation_kl_budget == 0.04


def test_beta_behavior_statistics_are_analytic_and_reset() -> None:
    import jax.numpy as jnp

    from src.delta_zsc.behavior_statistics import (
        behavior_features,
        initial_behavior_posterior,
        update_behavior_posterior,
    )

    state = initial_behavior_posterior((2,))
    observation = jnp.zeros((2, 5, 5, 39), dtype=jnp.float32)
    updated = update_behavior_posterior(
        state,
        previous_observation=observation,
        current_observation=observation,
        episode_start=jnp.asarray([False, True]),
    )
    assert updated.alpha.shape == (2, 4)
    assert updated.beta.shape == (2, 4)
    assert behavior_features(updated).shape == (2, 8)
    assert float(updated.beta[0, 0]) == pytest.approx(2.0)
    assert float(updated.beta[1, 0]) == pytest.approx(2.0)


def test_online_filter_uses_uniform_start_and_physical_transition() -> None:
    import jax.numpy as jnp

    from src.delta_zsc.filtering import filter_step

    previous = jnp.asarray([[0.9, 0.1]], dtype=jnp.float32)
    transition = jnp.asarray([[0.8, 0.2], [0.3, 0.7]], dtype=jnp.float32)
    neutral = jnp.zeros_like(previous)
    start = filter_step(
        previous, transition, neutral, episode_start=jnp.asarray([True])
    )
    np.testing.assert_allclose(np.asarray(start.posterior), [[0.5, 0.5]], atol=1e-6)
    continued = filter_step(
        previous, transition, neutral, episode_start=jnp.asarray([False])
    )
    np.testing.assert_allclose(
        np.asarray(continued.posterior), np.asarray(previous @ transition), atol=1e-6
    )


def test_mirror_policy_respects_registered_kl_budget() -> None:
    import jax.numpy as jnp

    from src.delta_zsc.mirror_policy import kl_constrained_policy

    base = jnp.zeros((3, 6), dtype=jnp.float32)
    values = jnp.asarray(
        [
            [10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 5.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    policy = kl_constrained_policy(base, values, kl_budget=0.04)
    assert np.all(np.asarray(policy.kl_to_base) <= 0.04001)
    assert np.allclose(np.sum(np.asarray(policy.probabilities), axis=-1), 1.0)


def test_voi_is_zero_when_response_is_component_independent() -> None:
    import jax.numpy as jnp

    from src.delta_zsc.voi import myopic_value_of_information

    belief = jnp.asarray([[0.5, 0.5]], dtype=jnp.float32)
    transition = jnp.eye(2, dtype=jnp.float32)
    values = jnp.asarray([[[1.0, 0.0], [0.0, 1.0]]], dtype=jnp.float32)
    response = jnp.full((1, 2, 2, 3), 1.0 / 3.0, dtype=jnp.float32)
    result = myopic_value_of_information(
        belief=belief,
        transition=transition,
        component_action_values=values,
        response_outcome_probability=response,
    )
    np.testing.assert_allclose(np.asarray(result.value), 0.0, atol=1e-6)


def test_joint_likelihood_uses_decision_evidence_without_loss_weight() -> None:
    import jax.numpy as jnp

    from src.delta_zsc.filtering import sequence_log_likelihood

    transition = jnp.eye(2, dtype=jnp.float32)
    response = jnp.zeros((2, 1, 2), dtype=jnp.float32)
    decision = jnp.asarray([[[0.0, -10.0]], [[0.0, 0.0]]], dtype=jnp.float32)
    result = sequence_log_likelihood(
        transition=transition,
        response_log_likelihood=response,
        decision_log_likelihood=decision,
        decision_mask=jnp.asarray([[1.0], [0.0]]),
        episode_starts=jnp.asarray([[True], [False]]),
        valid_mask=jnp.ones((2, 1), dtype=jnp.float32),
    )
    assert float(result.decision_count) == 1.0
    assert float(result.final_posterior[0, 0]) > 0.99
