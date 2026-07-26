from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from src.path_c.vqbc.model import vqbc_forward
from src.path_c.vqbc.objectives import (
    episode_responsibility_evidence,
    raw_policy_continuation_targets,
)
from src.path_c.vqbc.policy import (
    bellman_control_values,
    deployment_belief_after_response,
    generic_response_information,
    policy_effect_decomposition,
    regularized_objective_value,
    regularized_policy,
)
from src.path_c.vqbc.quotient import (
    aggregate_slots,
    complete_link_class_ids,
    posterior_supported_class_count,
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
    first = aggregate_slots(
        class_ids=class_ids,
        slot_log_belief=jnp.log(jnp.asarray([[0.1, 0.5, 0.4]])),
        response_probabilities=response,
        reward_mean=reward,
    )
    second = aggregate_slots(
        class_ids=class_ids,
        slot_log_belief=jnp.log(jnp.asarray([[0.5, 0.1, 0.4]])),
        response_probabilities=response,
        reward_mean=reward,
    )
    np.testing.assert_allclose(first.class_belief, second.class_belief)
    np.testing.assert_allclose(first.response_probabilities[0, 0, 0], [0.3, 0.7])
    np.testing.assert_allclose(second.response_probabilities[0, 0, 0], [0.7, 0.3])


def test_slot_bayes_preserves_within_class_evidence_and_projection_is_retired() -> None:
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


def test_use_and_mask_share_the_same_physical_world_weighting() -> None:
    belief = jnp.asarray([0.5, 0.5])
    response = jnp.asarray(
        [
            [[0.9, 0.1]],
            [[0.1, 0.9]],
        ]
    )
    reward = jnp.zeros((2, 1))
    use_one = jnp.asarray(
        [
            [[10.0, 0.0]],
            [[0.0, 10.0]],
        ]
    )
    mask_one = jnp.full_like(use_one, 5.0)
    use = jnp.stack((use_one, use_one), axis=0)
    mask = jnp.stack((mask_one, mask_one), axis=0)
    values = bellman_control_values(
        slot_belief=belief,
        response_probabilities=response,
        reward_mean=reward,
        continuation_use_mean=use,
        continuation_mask_mean=mask,
        gamma=1.0,
    )
    np.testing.assert_allclose(values.j_use, [9.0], atol=1.0e-6)
    np.testing.assert_allclose(values.j_mask, [5.0], atol=1.0e-6)
    expected_joint = belief[:, None, None] * response
    np.testing.assert_allclose(values.physical_joint, expected_joint)
    # The posterior is diagnostic only; neither branch substitutes it for the
    # physical joint law used in the expectation.
    np.testing.assert_allclose(values.updated_belief[:, 0, 0], [0.9, 0.1])


def test_policy_effect_decomposition_matches_raw_identity() -> None:
    reference = jnp.asarray([[1.0, -0.5, 0.2]])
    j_use = jnp.asarray([[3.0, 0.5, 1.0]])
    j_mask = jnp.asarray([[2.0, 0.0, 1.5]])
    effects = policy_effect_decomposition(
        reference_logits=reference,
        j_use=j_use,
        j_mask=j_mask,
        temperature=0.7,
    )
    np.testing.assert_allclose(
        effects.raw_net_effect,
        effects.raw_response_effect - effects.raw_policy_cost,
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    direct = (
        np.sum(np.asarray(effects.use_policy.probabilities) * np.asarray(j_use), axis=-1)
        - np.sum(
            np.asarray(effects.mask_policy.probabilities) * np.asarray(j_mask),
            axis=-1,
        )
    )
    np.testing.assert_allclose(effects.raw_net_effect, direct)


def test_action_independent_value_shift_does_not_change_runtime_policy() -> None:
    reference = jnp.asarray([[2.0, 0.0, -1.0]])
    mask = jnp.asarray([[0.1, 0.5, -0.2]])
    use = mask + 7.0
    effects = policy_effect_decomposition(
        reference_logits=reference,
        j_use=use,
        j_mask=mask,
        temperature=1.0,
    )
    np.testing.assert_allclose(
        effects.use_policy.probabilities,
        effects.mask_policy.probabilities,
        atol=1.0e-7,
    )
    np.testing.assert_allclose(effects.total_variation, 0.0, atol=1.0e-7)
    np.testing.assert_allclose(effects.raw_response_effect, 7.0, atol=1.0e-6)


def test_regularized_value_matches_runtime_policy_objective() -> None:
    reference = jnp.asarray([[0.2, -0.3, 0.1]])
    score = jnp.asarray([[1.4, -0.2, 0.5]])
    temperature = 0.8
    policy = regularized_policy(reference, score, temperature)
    reference_log = jax.nn.log_softmax(reference, axis=-1)
    policy_log = jax.nn.log_softmax(policy.logits, axis=-1)
    direct = jnp.sum(policy.probabilities * score, axis=-1) - temperature * jnp.sum(
        policy.probabilities * (policy_log - reference_log), axis=-1
    )
    np.testing.assert_allclose(
        regularized_objective_value(reference, score, temperature),
        direct,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_zero_score_policy_is_elementwise_equal_to_reference() -> None:
    reference = jnp.asarray([[0.2, -0.3, 0.1, 0.0, -0.2, 0.4]])
    execution = regularized_policy(reference, jnp.zeros_like(reference), 1.0)
    np.testing.assert_array_equal(execution.logits, reference)


def test_generic_information_reads_only_slot_response_kernel() -> None:
    belief = jnp.asarray([0.5, 0.5])
    response = jnp.asarray(
        [
            [[0.9, 0.1], [0.5, 0.5]],
            [[0.1, 0.9], [0.5, 0.5]],
        ]
    )
    information = generic_response_information(
        slot_belief=belief,
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


def _forward_raw(slot_count: int = 2, action_count: int = 6, response_count: int = 2):
    q = jnp.zeros((2, slot_count, action_count))
    q = q.at[:, 0, 0].set(2.0)
    q = q.at[:, 1, 1].set(2.0)
    response = jnp.full((slot_count, action_count, response_count), 1.0 / response_count)
    reward = jnp.zeros((slot_count, action_count))
    reward = reward.at[0, 0].set(4.0)
    reward = reward.at[1, 1].set(4.0)
    continuation = jnp.zeros((2, slot_count, action_count, response_count))
    return {
        "features": jnp.zeros((3,)),
        "q_values": q,
        "learned_q_values": q,
        "prior_q_values": jnp.zeros_like(q),
        "centered_advantages": q - jnp.max(q, axis=-1, keepdims=True),
        "response_logits": jnp.log(response),
        "response_probabilities": response,
        "reward_mean": reward,
        "reward_log_standard_deviation": jnp.zeros_like(reward),
        "continuation_use_mean": continuation,
        "continuation_use_log_standard_deviation": jnp.zeros_like(continuation),
        "continuation_mask_mean": continuation,
        "continuation_mask_log_standard_deviation": jnp.zeros_like(continuation),
    }


def test_persistent_belief_changes_normal_action_order() -> None:
    raw = _forward_raw()
    reference = jnp.zeros((6,))
    left = vqbc_forward(
        raw_output=raw,
        reference_logits=reference,
        slot_log_belief=jnp.log(jnp.asarray([0.9, 0.1])),
        temperature=1.0,
        generic_temperature=1.0,
        deployment_mode="posterior_use",
        gamma=0.99,
    )
    right = vqbc_forward(
        raw_output=raw,
        reference_logits=reference,
        slot_log_belief=jnp.log(jnp.asarray([0.1, 0.9])),
        temperature=1.0,
        generic_temperature=1.0,
        deployment_mode="posterior_use",
        gamma=0.99,
    )
    assert int(jnp.argmax(left.execution_logits)) == 0
    assert int(jnp.argmax(right.execution_logits)) == 1


def test_supported_quotient_count_excludes_negligible_slot_mass() -> None:
    class_ids = jnp.arange(8, dtype=jnp.int32)
    belief = jnp.asarray([0.55, 0.4495] + [0.0005 / 6.0] * 6)
    count = posterior_supported_class_count(
        class_ids=class_ids,
        slot_log_belief=jnp.log(belief),
        probability_floor=1.0e-3,
    )
    assert int(count) == 2


def test_full_episode_responsibility_accumulates_control_evidence() -> None:
    q_values = jnp.zeros((3, 1, 2, 2, 1))
    q_values = q_values.at[:, :, :, 0, 0].set(0.1)
    q_values = q_values.at[:, :, :, 1, 0].set(0.5)
    responsibilities, energies = episode_responsibility_evidence(
        q_values=q_values,
        actions=jnp.zeros((3, 1), dtype=jnp.int32),
        targets=jnp.zeros((3, 1, 2)),
        temperature=1.0,
    )
    assert float(responsibilities[0, 0]) > float(responsibilities[0, 1])
    assert float(energies[0, 1] - energies[0, 0]) > 0.0


def test_outcome_evidence_can_separate_td_equivalent_slots() -> None:
    time, batch, estimators, slots, actions, responses = 2, 1, 2, 2, 1, 2
    response_logits = jnp.asarray(
        [
            [[[[8.0, -8.0]], [[-8.0, 8.0]]]],
            [[[[8.0, -8.0]], [[-8.0, 8.0]]]],
        ]
    )
    zeros_cont = jnp.zeros((time, batch, estimators, slots, actions, responses))
    responsibilities, unused = episode_responsibility_evidence(
        q_values=jnp.zeros((time, batch, estimators, slots, actions)),
        actions=jnp.zeros((time, batch), dtype=jnp.int32),
        targets=jnp.zeros((time, batch, slots)),
        temperature=1.0,
        response_logits=response_logits,
        reward_mean=jnp.zeros((time, batch, slots, actions)),
        reward_log_standard_deviation=jnp.zeros((time, batch, slots, actions)),
        continuation_use_mean=zeros_cont,
        continuation_use_log_standard_deviation=zeros_cont,
        continuation_mask_mean=zeros_cont,
        continuation_mask_log_standard_deviation=zeros_cont,
        response_codes=jnp.zeros((time, batch), dtype=jnp.int32),
        rewards=jnp.zeros((time, batch)),
        continuation_use_targets=jnp.zeros((time, batch, estimators, slots)),
        continuation_mask_targets=jnp.zeros((time, batch, estimators, slots)),
    )
    del unused
    assert float(responsibilities[0, 0]) > 0.999


def test_raw_continuation_target_uses_exact_execution_distribution() -> None:
    probabilities = jnp.asarray([[0.25, 0.75]])
    q_values = jnp.asarray(
        [
            [
                [[3.0, 1.0], [2.0, 4.0]],
                [[1.0, 5.0], [4.0, 0.0]],
            ]
        ]
    )
    target = raw_policy_continuation_targets(
        execution_probabilities=probabilities,
        target_q_values=q_values,
        dones=jnp.asarray([False]),
    )
    expected = np.einsum("a,ema->em", np.asarray(probabilities[0]), np.asarray(q_values[0]))
    np.testing.assert_allclose(target[0], expected)
