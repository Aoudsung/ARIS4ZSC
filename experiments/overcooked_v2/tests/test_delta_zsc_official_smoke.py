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
