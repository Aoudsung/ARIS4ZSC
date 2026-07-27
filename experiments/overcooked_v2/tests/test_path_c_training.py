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
    outcome_loss,
    polyak_update,
    response_encoder_loss,
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


def test_responsibility_uses_only_complete_episode_td_sum() -> None:
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


def test_responsibility_respects_literal_bootstrap_availability() -> None:
    q_values = jnp.zeros((1, 1, 2, 2, 1), dtype=jnp.float32)
    responsibility, unused = episode_responsibility_evidence(
        q_values=q_values,
        actions=jnp.zeros((1, 1), dtype=jnp.int32),
        targets=jnp.zeros((1, 1, 2), dtype=jnp.float32),
        temperature=1.0,
        availability_mask=jnp.asarray([[True, False]]),
    )
    del unused
    np.testing.assert_array_equal(
        np.asarray(responsibility), np.asarray([[1.0, 0.0]])
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


def test_outcome_loss_trains_action_vector_and_reference_targets() -> None:
    # T=1, B=1, E=2, M=2, A=1, Y=2, U=2.
    response_logits = jnp.zeros((1, 1, 2, 1, 2), dtype=jnp.float32)
    reward_mean = jnp.zeros((1, 1, 2, 1), dtype=jnp.float32)
    reward_log_std = jnp.zeros_like(reward_mean)
    next_q = jnp.zeros((1, 1, 2, 2, 1, 2, 2), dtype=jnp.float32)
    next_q_log_std = jnp.zeros_like(next_q)
    next_reference = jnp.zeros((1, 1, 1, 2, 2), dtype=jnp.float32)
    loss = outcome_loss(
        response_logits=response_logits,
        reward_mean=reward_mean,
        reward_log_standard_deviation=reward_log_std,
        next_q_use_mean=next_q,
        next_q_use_log_standard_deviation=next_q_log_std,
        next_q_mask_mean=next_q,
        next_q_mask_log_standard_deviation=next_q_log_std,
        next_reference_logits_mean=next_reference,
        response_codes=jnp.asarray([[0]], dtype=jnp.int32),
        rewards=jnp.asarray([[0.0]], dtype=jnp.float32),
        next_q_use_targets=jnp.zeros((1, 1, 2, 2, 2), dtype=jnp.float32),
        next_q_mask_targets=jnp.zeros((1, 1, 2, 2, 2), dtype=jnp.float32),
        next_reference_logits_targets=jnp.zeros((1, 1, 2), dtype=jnp.float32),
        actions=jnp.asarray([[0]], dtype=jnp.int32),
        stopped_responsibilities=jnp.asarray([[0.25, 0.75]], dtype=jnp.float32),
        bootstrap_mask=jnp.ones((1, 2), dtype=jnp.bool_),
    )
    assert math.isfinite(float(loss.total))
    np.testing.assert_allclose(float(loss.metrics["response_nll"]), math.log(2.0), atol=1e-6)
    np.testing.assert_allclose(float(loss.metrics["next_reference_mse"]), 0.0, atol=1e-7)


def test_response_encoder_uses_full_delta_signature_width() -> None:
    predicted = jnp.asarray([[0.0, 1.0, 2.0, 3.0]], dtype=jnp.float32)
    codebook = jnp.asarray(
        [[0.0, 1.0, 2.0, 3.0], [4.0, 4.0, 4.0, 4.0]],
        dtype=jnp.float32,
    )
    result = response_encoder_loss(
        predicted_signatures=predicted,
        target_signatures=predicted,
        target_codes=jnp.asarray([0], dtype=jnp.int32),
        codebook_embeddings=codebook,
    )
    assert float(result.metrics["response_encoder_commitment"]) == 0.0
    assert math.isfinite(float(result.total))


def test_codebook_update_uses_literal_signatures() -> None:
    state = empty_codebook(code_count=2, signature_dim=4)
    update = update_codebook(
        state,
        signatures=jnp.asarray(
            [[0.0, 0.0, 1.0, 1.0], [4.0, 0.0, 2.0, 1.0]],
            dtype=jnp.float32,
        ),
        key=jax.random.PRNGKey(7),
        decay=0.0,
        replacement_after_rollouts=10,
    )
    assert {
        tuple(row) for row in np.asarray(update.state.embeddings).tolist()
    } == {(0.0, 0.0, 1.0, 1.0), (4.0, 0.0, 2.0, 1.0)}


def test_training_behavior_support_is_a_literal_uniform_mixture() -> None:
    from src.path_c.method import PolicyState, uniform_slot_log_belief
    from src.path_c.runner import RunnerFunctions, policy_action

    batch = 1
    slot_count = 2
    action_count = 3
    response_count = 2
    reference_logits = jnp.asarray([[2.0, 0.0, -1.0]], dtype=jnp.float32)

    def reference_step(carry, observations, episode_start):
        del observations, episode_start
        return carry, reference_logits, jnp.zeros((batch,), dtype=jnp.float32)

    def online_step(params, carry, observations, episode_start):
        del params, observations, episode_start
        return carry, jnp.zeros((batch, 4)), reference_logits, jnp.zeros((batch,))

    def heads_apply(params, control_carry, features, previous_action, previous_reward, episode_start, belief):
        del params, features, previous_action, previous_reward, episode_start, belief
        zeros_q = jnp.zeros((batch, 2, slot_count, action_count), dtype=jnp.float32)
        response = jnp.full(
            (batch, slot_count, action_count, response_count),
            1.0 / response_count,
            dtype=jnp.float32,
        )
        next_q = jnp.zeros(
            (batch, 2, slot_count, action_count, response_count, action_count),
            dtype=jnp.float32,
        )
        return control_carry, {
            "features": jnp.zeros((batch, 4), dtype=jnp.float32),
            "q_values": zeros_q,
            "learned_q_values": zeros_q,
            "prior_q_values": zeros_q,
            "centered_advantages": zeros_q,
            "response_logits": jnp.zeros_like(response),
            "response_probabilities": response,
            "reward_mean": jnp.zeros((batch, slot_count, action_count), dtype=jnp.float32),
            "reward_log_standard_deviation": jnp.zeros((batch, slot_count, action_count), dtype=jnp.float32),
            "next_q_use_mean": next_q,
            "next_q_use_log_standard_deviation": jnp.zeros_like(next_q),
            "next_q_mask_mean": next_q,
            "next_q_mask_log_standard_deviation": jnp.zeros_like(next_q),
            "next_reference_logits_mean": jnp.zeros(
                (batch, action_count, response_count, action_count), dtype=jnp.float32
            ),
        }

    functions = RunnerFunctions(
        online_step=online_step,
        reference_step=reference_step,
        heads_apply=heads_apply,
        encode_response=lambda *unused: None,
        partner_step=lambda *unused: None,
        partner_observe=lambda *unused: None,
    )
    state = PolicyState(
        reference_carry=jnp.zeros((batch, 1)),
        trainable_carry=jnp.zeros((batch, 1)),
        control_carry=jnp.zeros((batch, 4)),
        slot_log_belief=uniform_slot_log_belief((batch,), slot_count),
        previous_action=jnp.full((batch,), action_count, dtype=jnp.int32),
        previous_team_reward=jnp.zeros((batch,), dtype=jnp.float32),
        episode_start=jnp.ones((batch,), dtype=jnp.bool_),
        log_temperature=jnp.zeros((batch,), dtype=jnp.float32),
        generic_log_temperature=jnp.zeros((batch,), dtype=jnp.float32),
    )
    unused_state, unused_action, output, record, unused_generic = policy_action(
        functions=functions,
        params={"official": {}, "heads": {}},
        policy_state=state,
        observations=jnp.zeros((batch, 1)),
        key=jax.random.split(jax.random.PRNGKey(33), batch),
        deployment_mode="posterior_use",
        gamma=0.99,
        behavior_support=0.1,
    )
    del unused_state, unused_action, unused_generic
    target = 0.9 * jax.nn.softmax(output.execution_logits, axis=-1) + 0.1 / action_count
    np.testing.assert_allclose(
        np.asarray(jax.nn.softmax(record.behavior_logits, axis=-1)),
        np.asarray(target),
        atol=1e-7,
    )
