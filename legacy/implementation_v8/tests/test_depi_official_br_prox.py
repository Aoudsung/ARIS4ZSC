from __future__ import annotations

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from experiments.overcooked_v2.official_br_prox_app import (  # noqa: E402
    empirical_official_br_prox_pairing,
)


class _FixedPolicy:
    def __init__(self, action: int):
        self.action = int(action)

    def init_hstate(self, batch_size: int, key=None):
        del key
        return jnp.zeros((batch_size, 1), dtype=jnp.float32)

    def compute_action(self, obs, done, hstate, key):
        del obs, done, key
        return jnp.asarray(self.action, dtype=jnp.int32), hstate + 1.0


class _TinyEnvironment:
    max_steps = 3

    def reset(self, key):
        del key
        state = jnp.asarray(0, dtype=jnp.int32)
        obs = {
            "agent_0": jnp.asarray([0.0], dtype=jnp.float32),
            "agent_1": jnp.asarray([0.0], dtype=jnp.float32),
        }
        return obs, state

    def step(self, key, state, actions):
        del key
        next_state = state + 1
        done = next_state >= self.max_steps
        observations = {
            "agent_0": jnp.asarray([next_state], dtype=jnp.float32),
            "agent_1": jnp.asarray([next_state], dtype=jnp.float32),
        }
        reward = jnp.asarray(actions["agent_0"], dtype=jnp.float32)
        rewards = {"agent_0": reward, "agent_1": reward}
        dones = {"agent_0": done, "agent_1": done, "__all__": done}
        return observations, next_state, rewards, dones, {}


def test_empirical_br_prox_uses_independent_fit_and_evaluation_returns() -> None:
    rows = empirical_official_br_prox_pairing(
        ego_policy=_FixedPolicy(0),
        partner_policy=_FixedPolicy(0),
        ego_role=0,
        environment=_TinyEnvironment(),
        root_key=jax.random.PRNGKey(0),
        anchors=2,
        fit_replicas=2,
        evaluation_replicas=3,
        continuation_horizon=3,
        gamma=1.0,
        continuation_policy_fingerprint="tiny-policy",
        episodes=4,
    )
    assert len(rows) == 2
    assert all(row["fit_oracle_action"] == 5 for row in rows)
    assert all(row["raw_local_br_regret"] == pytest.approx(5.0) for row in rows)
    assert all(row["scope"].startswith("one_action_deviation") for row in rows)
    assert np.mean([row["br_prox"] for row in rows]) > 0.0
