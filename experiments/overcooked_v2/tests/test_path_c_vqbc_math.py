from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from src.path_c.vqbc.model import vqbc_forward
from src.path_c.vqbc.policy import (
    bellman_control_values,
    deployment_belief_after_response,
    generic_response_information,
    regularized_policy,
)
from src.path_c.vqbc.training import prepare_frozen_assignments
from src.path_c.vqbc.types import VQBCRolloutBatch
from src.path_c.vqbc.quotient import (
    aggregate_slots,
    complete_link_class_ids,
    quotient_bayes_update,
    slot_bayes_update,
)


def test_value_quotient_requires_certified_equivalence_not_overlap() -> None:
    signatures = jnp.asarray([[[0.0], [1.0], [2.0]]])
    uncertain = complete_link_class_ids(
        signatures, jnp.asarray([[0.6, 0.6, 0.6]])
    )
    np.testing.assert_array_equal(np.asarray(uncertain), [[0, 1, 2]])
    equal = complete_link_class_ids(
        jnp.asarray([[[0.0], [0.0], [2.0]]]),
        jnp.zeros((1, 3)),
    )
    np.testing.assert_array_equal(np.asarray(equal), [[0, 0, 1]])


def test_class_view_conditions_on_persistent_within_class_posterior() -> None:
    class_ids = jnp.asarray([[0, 0, 1]])
    response = jnp.asarray(
        [[[[0.8, 0.2]], [[0.2, 0.8]], [[0.5, 0.5]]]]
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
    np.testing.assert_allclose(first.response_probabilities[0, 0, 0], [0.3, 0.7])
    np.testing.assert_allclose(second.response_probabilities[0, 0, 0], [0.7, 0.3])



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


def test_slot_bayes_preserves_within_class_evidence_and_old_projection_fails() -> None:
    updated = slot_bayes_update(
        slot_log_belief=jnp.log(jnp.asarray([0.1, 0.5, 0.4])),
        slot_response_probabilities=jnp.asarray(
            [
                [[0.8, 0.2]],
                [[0.2, 0.8]],
                [[0.5, 0.5]],
            ]
        ),
        action=jnp.asarray(0),
        response_code=jnp.asarray(0),
    )
    probabilities = jnp.exp(updated)
    assert float(probabilities[0]) != float(probabilities[1])
    np.testing.assert_allclose(probabilities.sum(), 1.0)
    with pytest.raises(RuntimeError, match="retired"):
        quotient_bayes_update()



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


def test_vqbc_forward_uses_persistent_slot_posterior_to_change_normal_actions() -> None:
    batch, estimators, slots, actions, responses = 2, 2, 8, 6, 16
    q_values = jnp.zeros((batch, estimators, slots, actions))
    q_values = q_values.at[:, :, 0, 0].set(4.0)
    q_values = q_values.at[:, :, 1, 1].set(4.0)
    response = jnp.full((batch, slots, actions, responses), 1.0 / responses)
    next_q = jnp.zeros(
        (batch, estimators, slots, actions, responses, actions)
    )
    next_q = next_q.at[:, :, 0, 0, :, 0].set(5.0)
    next_q = next_q.at[:, :, 1, 1, :, 1].set(5.0)
    raw = {
        "features": jnp.zeros((batch, 3)),
        "q_values": q_values,
        "learned_q_values": q_values,
        "prior_q_values": jnp.zeros_like(q_values),
        "centered_advantages": q_values
        - jnp.max(q_values, axis=-1, keepdims=True),
        "response_logits": jnp.log(response),
        "response_probabilities": response,
        "reward_mean": jnp.zeros((batch, slots, actions)),
        "reward_log_standard_deviation": jnp.zeros(
            (batch, slots, actions)
        ),
        "next_q_mean": next_q,
        "next_q_log_standard_deviation": jnp.zeros_like(next_q),
    }
    reference = jnp.zeros((batch, actions))
    left_belief = jnp.log(
        jnp.asarray([[0.8, 0.1] + [0.1 / 6.0] * 6] * batch)
    )
    right_belief = jnp.log(
        jnp.asarray([[0.1, 0.8] + [0.1 / 6.0] * 6] * batch)
    )

    def forward(log_belief):
        return vqbc_forward(
            raw_output=raw,
            reference_logits=reference,
            slot_log_belief=log_belief,
            temperature=1.0,
            generic_temperature=1.0,
            deployment_mode="posterior_use",
            gamma=0.99,
        )

    left = forward(left_belief)
    right = forward(right_belief)
    np.testing.assert_array_equal(
        np.asarray(jnp.argmax(left.execution_logits, axis=-1)), [0, 0]
    )
    np.testing.assert_array_equal(
        np.asarray(jnp.argmax(right.execution_logits, axis=-1)), [1, 1]
    )
    assert np.all(np.asarray(jnp.max(left.quotient_ids, axis=-1) + 1) >= 2)


def test_episode_responsibility_accumulates_control_evidence() -> None:
    from src.path_c.vqbc.objectives import episode_responsibilities

    q_values = jnp.asarray(
        [
            [[[[0.1], [0.5]], [[0.1], [0.5]]]],
            [[[[0.1], [0.5]], [[0.1], [0.5]]]],
        ]
    )
    responsibilities = episode_responsibilities(
        q_values=q_values,
        actions=jnp.zeros((2, 1), dtype=jnp.int32),
        targets=jnp.zeros((2, 1, 2)),
        temperature=1.0,
    )
    assert float(responsibilities[0, 0]) > float(responsibilities[0, 1])


def test_responsibility_availability_excludes_bootstrap_slots() -> None:
    from src.path_c.vqbc.objectives import episode_responsibilities

    responsibilities = episode_responsibilities(
        q_values=jnp.zeros((2, 1, 2, 2, 1)),
        actions=jnp.zeros((2, 1), dtype=jnp.int32),
        targets=jnp.zeros((2, 1, 2)),
        temperature=1.0,
        availability_mask=jnp.asarray([[False, True]]),
    )
    np.testing.assert_array_equal(
        np.asarray(responsibilities), np.asarray([[0.0, 1.0]])
    )


def test_outcome_likelihood_separates_slots_when_td_is_equal() -> None:
    from src.path_c.vqbc.objectives import episode_responsibilities

    time, batch, estimators, slots = 2, 1, 2, 2
    actions, responses, next_actions = 1, 2, 1
    response_logits = jnp.asarray(
        [
            [[[[8.0, -8.0]], [[-8.0, 8.0]]]],
            [[[[8.0, -8.0]], [[-8.0, 8.0]]]],
        ]
    )
    responsibilities = episode_responsibilities(
        q_values=jnp.zeros((time, batch, estimators, slots, actions)),
        actions=jnp.zeros((time, batch), dtype=jnp.int32),
        targets=jnp.zeros((time, batch, slots)),
        temperature=1.0,
        response_logits=response_logits,
        reward_mean=jnp.zeros((time, batch, slots, actions)),
        reward_log_standard_deviation=jnp.zeros((time, batch, slots, actions)),
        next_q_mean=jnp.zeros(
            (time, batch, estimators, slots, actions, responses, next_actions)
        ),
        next_q_log_standard_deviation=jnp.zeros(
            (time, batch, estimators, slots, actions, responses, next_actions)
        ),
        response_codes=jnp.zeros((time, batch), dtype=jnp.int32),
        rewards=jnp.zeros((time, batch)),
        target_next_q_values=jnp.zeros(
            (time, batch, estimators, slots, next_actions)
        ),
        dones=jnp.zeros((time, batch), dtype=jnp.bool_),
    )
    assert float(responsibilities[0, 0]) > 0.999


def test_frozen_assignment_e_step_uses_target_model_and_dynamic_slot_count() -> None:
    time_count, environment_count = 2, 4
    slot_count, action_count, response_count = 2, 2, 4

    class FakeModel:
        control_only = object()

        def apply(
            self,
            unused_variables,
            initial_carry,
            observations,
            previous_actions,
            previous_rewards,
            starts,
            method=None,
        ):
            del unused_variables, previous_actions, previous_rewards, starts, method
            time, batch = observations.shape[:2]
            q = jnp.zeros((time, batch, 2, slot_count, action_count))
            q = q.at[..., 0, 0].set(1.0)
            q = q.at[..., 1, 1].set(1.0)
            response_logits = jnp.zeros(
                (time, batch, slot_count, action_count, response_count)
            )
            reward = jnp.zeros((time, batch, slot_count, action_count))
            next_q = jnp.zeros(
                (
                    time,
                    batch,
                    2,
                    slot_count,
                    action_count,
                    response_count,
                    action_count,
                )
            )
            return initial_carry, {
                "features": jnp.zeros((time, batch, 1)),
                "q_values": q,
                "learned_q_values": q,
                "prior_q_values": jnp.zeros_like(q),
                "centered_advantages": q
                - jnp.max(q, axis=-1, keepdims=True),
                "response_logits": response_logits,
                "response_probabilities": jax.nn.softmax(
                    response_logits, axis=-1
                ),
                "reward_mean": reward,
                "reward_log_standard_deviation": jnp.zeros_like(reward),
                "next_q_mean": next_q,
                "next_q_log_standard_deviation": jnp.zeros_like(next_q),
            }

    batch = VQBCRolloutBatch(
        observations=jnp.zeros((time_count + 1, environment_count, 1)),
        response_next_observations=jnp.zeros(
            (time_count, environment_count, 1)
        ),
        episode_start=jnp.zeros(
            (time_count + 1, environment_count), dtype=jnp.bool_
        ),
        previous_actions=jnp.zeros(
            (time_count + 1, environment_count), dtype=jnp.int32
        ),
        previous_team_rewards=jnp.zeros(
            (time_count + 1, environment_count)
        ),
        initial_value_carry=jnp.zeros((environment_count, 1)),
        reference_logits=jnp.zeros(
            (time_count + 1, environment_count, action_count)
        ),
        execution_logits=jnp.zeros(
            (time_count, environment_count, action_count)
        ),
        generic_execution_logits=jnp.zeros(
            (time_count, environment_count, action_count)
        ),
        slot_log_beliefs=jnp.full(
            (time_count + 1, environment_count, slot_count),
            -jnp.log(float(slot_count)),
        ),
        actions=jnp.zeros(
            (time_count, environment_count), dtype=jnp.int32
        ),
        rewards=jnp.zeros((time_count, environment_count)),
        dones=jnp.asarray(
            [[False] * environment_count, [True] * environment_count]
        ),
        response_codes=jnp.zeros(
            (time_count, environment_count), dtype=jnp.int32
        ),
        episode_ids=jnp.zeros(
            (time_count, environment_count), dtype=jnp.int32
        ),
        episode_steps=jnp.zeros(
            (time_count, environment_count), dtype=jnp.int32
        ),
        completed_episode_returns=jnp.zeros(
            (time_count, environment_count)
        ),
        quotient_counts=jnp.ones(
            (time_count, environment_count), dtype=jnp.int32
        ),
        j_use=jnp.zeros((time_count, environment_count, action_count)),
        j_mask=jnp.zeros((time_count, environment_count, action_count)),
    )
    availability = jnp.asarray(
        [[True, False], [False, True], [True, True], [True, False]]
    )
    assignments = prepare_frozen_assignments(
        model=FakeModel(),
        online_params={"ignored": jnp.asarray(1.0)},
        target_params={"ignored": jnp.asarray(2.0)},
        batch=batch,
        codebook_embeddings=jnp.asarray(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        ),
        key=jax.random.PRNGKey(3),
        temperature=jnp.asarray(1.0),
        generic_temperature=jnp.asarray(1.0),
        gamma=0.99,
        responsibility_temperature=1.0,
        bootstrap_probability=0.8,
        bootstrap_mask=availability,
        terminal_response=3,
        environment_chunk_size=2,
    )
    assert assignments.responsibilities.shape == (environment_count, slot_count)
    assert assignments.responsibility_energies.shape == (
        environment_count,
        slot_count,
    )
    assert assignments.bellman_targets.shape == (
        time_count,
        environment_count,
        slot_count,
    )
    assert assignments.response_signature_targets.shape == (
        time_count,
        environment_count,
        action_count,
    )
    assert assignments.response_code_targets.shape == (
        time_count,
        environment_count,
    )
    np.testing.assert_allclose(
        np.asarray(assignments.responsibilities).sum(axis=-1), 1.0
    )
    np.testing.assert_array_equal(
        np.asarray(assignments.responsibilities)[~np.asarray(availability)],
        0.0,
    )


def test_development_minibatches_are_exact_nonrepeating_lane_partitions() -> None:
    from src.path_c.vqbc.training import environment_minibatch_schedule

    schedule = environment_minibatch_schedule(
        jax.random.PRNGKey(17),
        environment_count=32,
        minibatches_per_epoch=8,
        update_epochs=4,
    )
    assert schedule.shape == (4, 8, 4)
    for epoch in np.asarray(schedule):
        np.testing.assert_array_equal(np.sort(epoch.reshape(-1)), np.arange(32))


def test_response_trigger_rejects_float32_roundoff_scale_values() -> None:
    from src.path_c.vqbc.response_contrast import (
        first_positive_trigger,
        information_trigger_tolerance,
    )

    j_use = jnp.asarray([[100.0, 100.0]])
    j_mask = jnp.asarray([[100.0, 100.0]])
    tolerance = float(information_trigger_tolerance(j_use, j_mask)[0])
    assert tolerance > 1.0e-5
    assert first_positive_trigger(np.full((4, 6), 1.0e-6)) == (None, None)
    step, value = first_positive_trigger(
        np.vstack((np.zeros((1, 6)), np.full((1, 6), 1.0e-2)))
    )
    assert step == 1
    assert value == pytest.approx(1.0e-2)
