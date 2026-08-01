from __future__ import annotations

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from src.path_c.counterfactual_anchor import (  # noqa: E402
    AnchorFunctions,
    AnchorWorld,
    centered_fit_signature,
    collect_counterfactual_anchors,
    evaluate_selected_actions,
)


def test_counterfactual_anchor_pairs_randomness_and_splits_replicas() -> None:
    def ego_policy_step(state, observation, gate, keys):
        del observation, gate, keys
        probabilities = jnp.full((state.shape[0], 3), 1.0 / 3.0)
        return state, probabilities

    def ego_observe(state, *unused):
        return state

    def partner_policy_step(state, observation, episode_start, keys):
        del observation, episode_start, keys
        return jnp.zeros((state.shape[0],), dtype=jnp.int32), state, None

    def partner_observe(state, *unused):
        return state

    def environment_step(state, joint_action, keys):
        noise = jax.vmap(lambda key: jax.random.uniform(key, minval=-0.2, maxval=0.2))(keys)
        reward = joint_action[:, 0].astype(jnp.float32) + noise
        observations = jnp.zeros((state.shape[0], 2, 1), dtype=jnp.float32)
        done = jnp.ones((state.shape[0],), dtype=jnp.bool_)
        info = {"terminal_observations": observations}
        return state, observations, reward, done, info

    world = AnchorWorld(
        environment_state=jnp.zeros((2, 1)),
        observations=jnp.zeros((2, 2, 1)),
        ego_state=jnp.zeros((2, 1)),
        partner_state=jnp.zeros((2, 1)),
        partner_episode_start=jnp.ones((2,), dtype=jnp.bool_),
        ego_roles=jnp.zeros((2,), dtype=jnp.int32),
        done=jnp.zeros((2,), dtype=jnp.bool_),
        raw_return=jnp.zeros((2,), dtype=jnp.float32),
    )
    kwargs = dict(
        anchor_ids=jnp.asarray([7, 8]),
        root_keys=jax.random.split(jax.random.PRNGKey(0), 2),
        world=world,
        rollout_flat_indexes=jnp.asarray([0, 1]),
        policy_states=jnp.zeros((2, 1)),
        observations=jnp.zeros((2, 1)),
        partner_codes=jnp.zeros((2, 2)),
        partner_sources=jnp.zeros((2,), dtype=jnp.int32),
        partner_run_ids=jnp.asarray([3, 4]),
        functions=AnchorFunctions(
            ego_policy_step=ego_policy_step,
            ego_observe=ego_observe,
            partner_policy_step=partner_policy_step,
            partner_observe=partner_observe,
            environment_step=environment_step,
        ),
        action_count=3,
        fit_replicas=3,
        evaluation_replicas=2,
        continuation_horizon=1,
    )
    batch = collect_counterfactual_anchors(**kwargs)
    microbatched = collect_counterfactual_anchors(
        **kwargs,
        # Three-world fixed executable for two real worlds exercises inactive
        # padding without introducing a scientific root key or output row.
        microbatch_size=3 * 3 * (3 + 2),
    )
    for full, chunked in zip(
        jax.tree_util.tree_leaves(batch),
        jax.tree_util.tree_leaves(microbatched),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(full), np.asarray(chunked))
    fit = np.asarray(batch.fit_returns_by_action[0])
    evaluation = np.asarray(batch.evaluation_returns_by_action[0])
    np.testing.assert_allclose(np.diff(fit), [1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(np.diff(evaluation), [1.0, 1.0], atol=1e-6)
    assert not np.allclose(fit, evaluation)
    np.testing.assert_allclose(
        np.asarray(centered_fit_signature(batch)).mean(axis=-1), 0.0, atol=1e-6
    )
    selected = evaluate_selected_actions(batch, jnp.asarray([2, 2]))
    np.testing.assert_allclose(
        np.asarray(selected),
        np.asarray(batch.evaluation_returns_by_action)[:, 2],
    )
