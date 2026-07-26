from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from src.path_c.vqbc.policy import (
    bellman_control_values,
    deployment_belief_after_response,
    generic_response_information,
    regularized_policy,
)
from src.path_c.vqbc.quotient import (
    aggregate_slots,
    complete_link_class_ids,
    quotient_bayes_update,
)


def test_complete_link_prevents_chain_merge_of_incompatible_endpoints() -> None:
    signatures = jnp.asarray([[[0.0], [1.0], [2.0]]])
    radii = jnp.asarray([[0.6, 0.6, 0.6]])
    class_ids = complete_link_class_ids(signatures, radii)
    np.testing.assert_array_equal(np.asarray(class_ids), [[0, 0, 1]])


def test_uniform_class_aggregation_ignores_within_class_slot_belief() -> None:
    class_ids = jnp.asarray([[0, 0, 1]])
    response = jnp.asarray(
        [
            [
                [[0.8, 0.2]],
                [[0.2, 0.8]],
                [[0.5, 0.5]],
            ]
        ]
    )
    reward = jnp.zeros((1, 3, 1))
    continuation = jnp.zeros((1, 2, 3, 1, 2, 1))
    first = aggregate_slots(
        class_ids=class_ids,
        slot_log_belief=jnp.log(jnp.asarray([[0.1, 0.5, 0.4]])),
        response_probabilities=response,
        reward_mean=reward,
        next_q_mean=continuation,
    )
    second = aggregate_slots(
        class_ids=class_ids,
        slot_log_belief=jnp.log(jnp.asarray([[0.5, 0.1, 0.4]])),
        response_probabilities=response,
        reward_mean=reward,
        next_q_mean=continuation,
    )
    np.testing.assert_allclose(first.class_belief, second.class_belief)
    np.testing.assert_allclose(
        first.response_probabilities, second.response_probabilities
    )
    first_values = bellman_control_values(
        class_belief=first.class_belief,
        response_probabilities=first.response_probabilities,
        reward_mean=first.reward_mean,
        next_q_mean=first.next_q_mean,
        gamma=0.99,
    )
    second_values = bellman_control_values(
        class_belief=second.class_belief,
        response_probabilities=second.response_probabilities,
        reward_mean=second.reward_mean,
        next_q_mean=second.next_q_mean,
        gamma=0.99,
    )
    reference = jnp.zeros((1, 1))
    np.testing.assert_array_equal(
        regularized_policy(reference, first_values.j_use, 1.0).logits,
        regularized_policy(reference, second_values.j_use, 1.0).logits,
    )


def test_j_use_updates_belief_inside_max_and_j_mask_keeps_prior() -> None:
    belief = jnp.asarray([0.5, 0.5])
    response = jnp.asarray(
        [
            [[0.9, 0.1]],
            [[0.1, 0.9]],
        ]
    )
    reward = jnp.zeros((2, 1))
    continuation_one = jnp.asarray(
        [
            [[[10.0, 0.0], [10.0, 0.0]]],
            [[[0.0, 10.0], [0.0, 10.0]]],
        ]
    )
    continuation = jnp.stack((continuation_one, continuation_one), axis=0)
    values = bellman_control_values(
        class_belief=belief,
        response_probabilities=response,
        reward_mean=reward,
        next_q_mean=continuation,
        gamma=1.0,
    )
    np.testing.assert_allclose(values.j_use, [9.0], atol=1.0e-6)
    np.testing.assert_allclose(values.j_mask, [5.0], atol=1.0e-6)


def test_quotient_update_writes_class_probability_uniformly_to_members() -> None:
    updated = quotient_bayes_update(
        class_ids=jnp.asarray([0, 0, 1]),
        class_belief=jnp.asarray([0.6, 0.4, 0.0]),
        class_response_probabilities=jnp.asarray(
            [
                [[0.8, 0.2]],
                [[0.1, 0.9]],
                [[0.5, 0.5]],
            ]
        ),
        action=jnp.asarray(0),
        response_code=jnp.asarray(0),
        class_counts=jnp.asarray([2.0, 1.0, 0.0]),
        class_mask=jnp.asarray([True, True, False]),
    )
    probabilities = jnp.exp(updated)
    np.testing.assert_allclose(probabilities[0], probabilities[1])
    np.testing.assert_allclose(probabilities.sum(), 1.0)


def test_zero_outcome_policy_is_elementwise_equal_to_reference() -> None:
    reference = jnp.asarray([[0.2, -0.3, 0.1, 0.0, -0.2, 0.4]])
    score = jnp.zeros_like(reference)
    execution = regularized_policy(reference, score, 1.0)
    np.testing.assert_array_equal(
        np.asarray(execution.logits),
        np.asarray(reference),
    )


def test_generic_information_reads_only_the_shared_response_kernel() -> None:
    belief = jnp.asarray([0.5, 0.5])
    response = jnp.asarray(
        [
            [[0.9, 0.1], [0.5, 0.5]],
            [[0.1, 0.9], [0.5, 0.5]],
        ]
    )
    information = generic_response_information(
        class_belief=belief,
        response_probabilities=response,
    )
    assert float(information[0]) > 0.0
    np.testing.assert_allclose(information[1], 0.0, atol=1.0e-6)


def test_prior_only_never_absorbs_a_response() -> None:
    updated = deployment_belief_after_response(
        mode="prior_only",
        current_log_belief=jnp.log(jnp.asarray([0.7, 0.2, 0.1])),
        updated_log_belief=jnp.log(jnp.asarray([0.05, 0.9, 0.05])),
    )
    np.testing.assert_allclose(jnp.exp(updated), [1.0 / 3.0] * 3)


def test_belief_change_can_reverse_normal_action_order() -> None:
    response = jnp.full((2, 2, 2), 0.5)
    reward = jnp.zeros((2, 2))
    continuation_one = jnp.asarray(
        [
            [
                [[5.0, 0.0], [5.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ],
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 5.0], [0.0, 5.0]],
            ],
        ]
    )
    continuation = jnp.stack((continuation_one, continuation_one), axis=0)
    left = bellman_control_values(
        class_belief=jnp.asarray([0.9, 0.1]),
        response_probabilities=response,
        reward_mean=reward,
        next_q_mean=continuation,
        gamma=1.0,
    )
    right = bellman_control_values(
        class_belief=jnp.asarray([0.1, 0.9]),
        response_probabilities=response,
        reward_mean=reward,
        next_q_mean=continuation,
        gamma=1.0,
    )
    assert int(jnp.argmax(left.j_use)) != int(jnp.argmax(right.j_use))
