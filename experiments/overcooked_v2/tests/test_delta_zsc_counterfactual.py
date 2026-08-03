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
)


def test_v6_anchor_common_randomness_padding_and_sufficient_statistics() -> None:
    def ego_policy_step(state, observation, keys):
        del observation, keys
        return state, jnp.full((state.shape[0], 3), 1.0 / 3.0)

    def ego_observe(state, *unused):
        return state

    def partner_policy_step(state, observation, episode_start, keys):
        del observation, episode_start, keys
        return jnp.zeros((state.shape[0],), dtype=jnp.int32), state, None

    def partner_observe(state, *unused):
        return state

    def environment_step(state, joint_action, keys):
        noise = jax.vmap(
            lambda key: jax.random.uniform(key, minval=-0.2, maxval=0.2)
        )(keys)
        agent0_reward = joint_action[:, 0].astype(jnp.float32) + noise
        agent1_reward = joint_action[:, 1].astype(jnp.float32) + 10.0 + noise
        observations = jnp.zeros((state.shape[0], 2, 1), dtype=jnp.float32)
        done = jnp.ones((state.shape[0],), dtype=jnp.bool_)
        return state, observations, agent0_reward, done, {
            "terminal_observations": observations,
            "raw_rewards_by_agent": jnp.stack(
                (agent0_reward, agent1_reward), axis=-1
            ),
        }

    world = AnchorWorld(
        environment_state=jnp.zeros((2, 1)),
        observations=jnp.zeros((2, 2, 1)),
        ego_state=jnp.zeros((2, 1)),
        partner_state=jnp.zeros((2, 1)),
        partner_episode_start=jnp.ones((2,), dtype=jnp.bool_),
        ego_roles=jnp.asarray([0, 1], dtype=jnp.int32),
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
            ego_endpoint_value=lambda state, observation: jnp.zeros(
                (observation.shape[0],), dtype=jnp.float32
            ),
        ),
        action_count=3,
        fit_replicas=3,
        evaluation_replicas=0,
        continuation_horizon=1,
        collection_policy_logits=jnp.asarray([[1.0, 0.0, -1.0]] * 2),
        collection_update=jnp.asarray(16),
        collection_target_fingerprint=jnp.asarray([11, 12], dtype=jnp.uint32),
        matched_pair_ids=jnp.asarray([-1, 4], dtype=jnp.int32),
    )
    batch = collect_counterfactual_anchors(**kwargs)
    microbatched = collect_counterfactual_anchors(
        **kwargs,
        # Three-world executable for two scientific worlds exercises zero-key
        # inactive padding without adding an output row.
        microbatch_size=3 * 3 * 3,
    )
    for full, chunked in zip(
        jax.tree_util.tree_leaves(batch),
        jax.tree_util.tree_leaves(microbatched),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(full), np.asarray(chunked))

    fit = np.asarray(batch.fit_returns_by_action[0])
    np.testing.assert_allclose(np.diff(fit), [1.0, 1.0], atol=1.0e-6)
    np.testing.assert_allclose(
        np.diff(np.asarray(batch.fit_returns_by_action[1])),
        [1.0, 1.0],
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(batch.return_sum_by_action),
        np.asarray(batch.fit_returns_by_action) * 3.0,
        atol=1.0e-6,
    )
    np.testing.assert_array_equal(np.asarray(batch.replica_count), 3)
    np.testing.assert_allclose(
        np.asarray(centered_fit_signature(batch)).mean(axis=-1), 0.0, atol=1e-6
    )
    np.testing.assert_array_equal(np.asarray(batch.collection_update), [16, 16])
    np.testing.assert_array_equal(np.asarray(batch.matched_pair_ids), [-1, 4])
