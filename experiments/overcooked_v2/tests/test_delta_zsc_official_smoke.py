from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("jaxmarl")

from experiments.overcooked_v2.official_adapter import (  # noqa: E402
    ACTION_ORDER,
    VectorEnvironment,
    _official_symbol,
    official_pairing_rollouts,
)
from src.path_c.experiment import load_config  # noqa: E402


ROOT = Path(__file__).resolve().parents[3]


def test_official_environment_reset_and_step_smoke() -> None:
    config = load_config(
        ROOT
        / "experiments"
        / "overcooked_v2"
        / "configs"
        / "delta_zsc_simple_development.yaml",
        run_kind="mechanical",
    )
    config = replace(
        config,
        environment=replace(config.environment, num_envs=2, episode_steps=8),
    )
    environment = VectorEnvironment.create(config)
    state, observations = environment.reset(jax.random.PRNGKey(1))
    assert observations.shape[:2] == (2, 2)
    assert environment.observation_shape == tuple(observations.shape[2:])
    actions = jnp.full((2, 2), 4, dtype=jnp.int32)
    next_state, next_observations, rewards, dones, info = environment.step(
        state, actions, jax.random.PRNGKey(2)
    )
    del next_state
    assert next_observations.shape == observations.shape
    assert rewards.shape == dones.shape == (2,)
    assert info["terminal_observations"].shape == observations.shape
    assert np.all(np.isfinite(np.asarray(rewards)))
    assert ACTION_ORDER == ("right", "down", "left", "up", "stay", "interact")


def test_vectorized_pairing_wrapper_is_trajectory_identical_to_official_get_rollout() -> None:
    from jaxmarl.environments.overcooked_v2.overcooked import ObservationType, OvercookedV2

    class RandomPolicy:
        def init_hstate(self, batch_size, key=None):
            del batch_size, key
            return None

        def compute_action(self, obs, done, hstate, key):
            del obs, done
            return jax.random.randint(key, (), 0, 6), hstate

    environment = OvercookedV2(
        layout="test_time_simple",
        observation_type=ObservationType.DEFAULT,
        agent_view_size=2,
        negative_rewards=True,
        random_agent_positions=True,
        sample_recipe_on_delivery=True,
        indicate_successful_delivery=True,
        max_steps=400,
    )
    root = jax.random.PRNGKey(0)
    left = RandomPolicy()
    right = RandomPolicy()
    observed, keys = official_pairing_rollouts(
        left_policy=left,
        right_policy=right,
        environment=environment,
        root_key=root,
        episodes=2,
    )
    PolicyPairing = _official_symbol(
        "overcooked_v2_experiments.eval.policy", "PolicyPairing"
    )
    get_rollout = _official_symbol(
        "overcooked_v2_experiments.eval.rollout", "get_rollout"
    )
    expected = jax.vmap(
        lambda key: get_rollout(PolicyPairing(left, right), environment, key)
    )(jax.random.split(root, 2))
    np.testing.assert_array_equal(np.asarray(keys), np.asarray(jax.random.split(root, 2)))
    np.testing.assert_array_equal(
        np.asarray(observed.actions_seq["agent_0"]),
        np.asarray(expected.actions_seq["agent_0"]),
    )
    np.testing.assert_array_equal(
        np.asarray(observed.actions_seq["agent_1"]),
        np.asarray(expected.actions_seq["agent_1"]),
    )
    np.testing.assert_allclose(
        np.asarray(observed.total_reward), np.asarray(expected.total_reward)
    )
