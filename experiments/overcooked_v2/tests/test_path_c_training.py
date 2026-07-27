from __future__ import annotations

import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from src.path_c.method import empty_codebook, update_codebook
from src.path_c.training import (
    bellman_loss,
    bellman_targets,
    environment_minibatch_schedule,
    episode_responsibility_evidence,
    polyak_update,
    sample_bootstrap_mask,
)


def test_bellman_target_matches_literal_expected_return() -> None:
    rewards = jnp.asarray([[2.0]], dtype=jnp.float32)
    dones = jnp.asarray([[False]])
    probabilities = jnp.asarray([[[0.25, 0.75]]], dtype=jnp.float32)
    target_q = jnp.asarray(
        [[[[[4.0, 8.0]], [[2.0, 10.0]]]]], dtype=jnp.float32
    )
    observed = bellman_targets(
        rewards=rewards,
        dones=dones,
        next_execution_probabilities=probabilities,
        target_next_q_values=target_q,
        gamma=0.5,
    )
    pessimistic = np.minimum([4.0, 8.0], [2.0, 10.0])
    expected = 2.0 + 0.5 * float(np.dot([0.25, 0.75], pessimistic))
    np.testing.assert_allclose(np.asarray(observed), [[[expected]]], atol=1e-6)


def test_responsibility_uses_complete_episode_sum() -> None:
    q_values = jnp.asarray(
        [
            [[[[1.0, 0.0], [3.0, 0.0]], [[1.0, 0.0], [3.0, 0.0]]]],
            [[[[2.0, 0.0], [0.0, 0.0]], [[2.0, 0.0], [0.0, 0.0]]]],
        ],
        dtype=jnp.float32,
    )
    actions = jnp.asarray([[0], [0]], dtype=jnp.int32)
    targets = jnp.zeros((2, 1, 2), dtype=jnp.float32)
    responsibilities, energies = episode_responsibility_evidence(
        q_values=q_values,
        actions=actions,
        targets=targets,
        temperature=1.0,
    )

    def huber(value: float) -> float:
        absolute = abs(value)
        return 0.5 * absolute * absolute if absolute <= 1.0 else absolute - 0.5

    expected_energy = np.asarray(
        [huber(1.0) + huber(2.0), huber(3.0) + huber(0.0)]
    )
    expected_probability = np.exp(-expected_energy)
    expected_probability /= expected_probability.sum()
    np.testing.assert_allclose(np.asarray(energies[0]), expected_energy, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(responsibilities[0]), expected_probability, atol=1e-6
    )


def test_bootstrap_never_creates_empty_lane() -> None:
    mask = sample_bootstrap_mask(
        jax.random.PRNGKey(12),
        environment_count=64,
        slot_count=8,
        probability=0.0,
    )
    np.testing.assert_array_equal(
        np.asarray(mask).sum(axis=-1), np.ones((64,), dtype=np.int64)
    )


def test_environment_schedule_uses_each_lane_once_per_epoch() -> None:
    schedule = environment_minibatch_schedule(
        jax.random.PRNGKey(9),
        environment_count=32,
        minibatches_per_epoch=8,
        update_epochs=4,
    )
    assert schedule.shape == (4, 8, 4)
    for epoch in np.asarray(schedule):
        np.testing.assert_array_equal(np.sort(epoch.reshape(-1)), np.arange(32))


def test_bellman_weighting_and_polyak_are_literal() -> None:
    q_values = jnp.asarray(
        [[[[[1.0, -1.0], [3.0, -3.0]], [[1.0, -1.0], [3.0, -3.0]]]]],
        dtype=jnp.float32,
    )
    observed = bellman_loss(
        q_values=q_values,
        actions=jnp.asarray([[0]], dtype=jnp.int32),
        targets=jnp.zeros((1, 1, 2), dtype=jnp.float32),
        stopped_responsibilities=jnp.asarray([[0.25, 0.75]]),
        bootstrap_mask=jnp.ones((1, 2), dtype=jnp.bool_),
    )
    expected = (2.0 * 0.25 * 0.5 + 2.0 * 0.75 * 2.5) / 2.0
    np.testing.assert_allclose(float(observed.total), expected, atol=1e-6)

    updated = polyak_update(
        {"weight": jnp.asarray([0.0, 10.0])},
        {"weight": jnp.asarray([4.0, 2.0])},
        0.25,
    )
    np.testing.assert_allclose(np.asarray(updated["weight"]), [1.0, 8.0])


def test_codebook_update_uses_literal_signatures() -> None:
    state = empty_codebook(code_count=2, signature_dim=2)
    update = update_codebook(
        state,
        signatures=jnp.asarray([[0.0, 0.0], [4.0, 0.0]], dtype=jnp.float32),
        key=jax.random.PRNGKey(7),
        decay=0.0,
        replacement_after_rollouts=10,
    )
    assert {tuple(row) for row in np.asarray(update.state.embeddings).tolist()} == {
        (0.0, 0.0),
        (4.0, 0.0),
    }
