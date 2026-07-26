from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from src.path_c.method import (
    bellman_control_values,
    complete_link_value_class_ids,
    normalized_log_belief,
    policy_effect_decomposition,
    regularized_policy,
    slot_bayes_update,
    update_log_temperature,
)


def test_slot_posterior_matches_literal_bayes_update() -> None:
    log_belief = jnp.log(jnp.asarray([[0.6, 0.4]], dtype=jnp.float32))
    response = jnp.asarray(
        [[[[0.8, 0.2]], [[0.25, 0.75]]]], dtype=jnp.float32
    )
    observed = slot_bayes_update(
        slot_log_belief=log_belief,
        slot_response_probabilities=response,
        action=jnp.asarray([0]),
        response_code=jnp.asarray([1]),
    )
    unnormalized = np.asarray([0.6 * 0.2, 0.4 * 0.75])
    expected = np.log(unnormalized / unnormalized.sum())
    np.testing.assert_allclose(np.asarray(observed[0]), expected, atol=1e-6)


def test_value_classes_require_complete_link_equivalence() -> None:
    signatures = jnp.asarray(
        [[[0.0, 0.0], [0.0004, 0.0002], [0.003, 0.0]]],
        dtype=jnp.float32,
    )
    radii = jnp.asarray([[0.0001, 0.0001, 0.0001]])
    classes = complete_link_value_class_ids(
        signatures, radii, tolerance=0.001
    )
    np.testing.assert_array_equal(np.asarray(classes), [[0, 0, 1]])


def test_use_and_mask_share_literal_physical_weights() -> None:
    belief = np.asarray([[0.7, 0.3]], dtype=np.float32)
    response = np.asarray(
        [
            [
                [[0.8, 0.2], [0.4, 0.6]],
                [[0.1, 0.9], [0.5, 0.5]],
            ]
        ],
        dtype=np.float32,
    )
    reward = np.asarray([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    use = np.arange(1 * 2 * 2 * 2 * 2, dtype=np.float32).reshape(
        1, 2, 2, 2, 2
    ) / 10.0
    masked = use + 0.25
    observed = bellman_control_values(
        slot_belief=jnp.asarray(belief),
        response_probabilities=jnp.asarray(response),
        reward_mean=jnp.asarray(reward),
        continuation_use_mean=jnp.asarray(use),
        continuation_mask_mean=jnp.asarray(masked),
        gamma=0.9,
    )

    expected_use = np.zeros((1, 2, 2), dtype=np.float32)
    expected_mask = np.zeros((1, 2, 2), dtype=np.float32)
    for estimator in range(2):
        for action in range(2):
            for slot in range(2):
                for response_code in range(2):
                    weight = belief[0, slot] * response[0, slot, action, response_code]
                    expected_use[0, estimator, action] += weight * (
                        reward[0, slot, action]
                        + 0.9 * use[0, estimator, slot, action, response_code]
                    )
                    expected_mask[0, estimator, action] += weight * (
                        reward[0, slot, action]
                        + 0.9 * masked[
                            0, estimator, slot, action, response_code
                        ]
                    )
    np.testing.assert_allclose(
        np.asarray(observed.j_use_by_estimator), expected_use, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(observed.j_mask_by_estimator), expected_mask, atol=1e-6
    )


def test_policy_effect_identity_uses_literal_objectives() -> None:
    reference = jnp.asarray([[1.0, -0.5, 0.25]], dtype=jnp.float32)
    use = jnp.asarray([[2.0, 1.0, -1.0]], dtype=jnp.float32)
    mask = jnp.asarray([[0.5, 0.25, -0.25]], dtype=jnp.float32)
    effects = policy_effect_decomposition(
        reference_logits=reference,
        j_use=use,
        j_mask=mask,
        temperature=jnp.asarray([0.8]),
    )
    np.testing.assert_allclose(
        np.asarray(effects.raw_net_effect),
        np.asarray(effects.raw_response_effect - effects.raw_policy_cost),
        atol=1e-6,
    )


def test_zero_scores_leave_reference_policy_unchanged() -> None:
    reference = jnp.asarray([[0.1, 1.4, -0.7]], dtype=jnp.float32)
    policy = regularized_policy(
        reference, jnp.zeros_like(reference), jnp.asarray([0.5])
    )
    np.testing.assert_array_equal(
        np.asarray(policy.logits), np.asarray(reference)
    )


def test_dual_temperature_update_matches_literal_clipped_step() -> None:
    observed = update_log_temperature(
        log_temperature=jnp.log(jnp.asarray(2.0)),
        mean_kl=jnp.asarray(0.07),
        target_kl=0.02,
        learning_rate=0.1,
        minimum_temperature=0.5,
        maximum_temperature=4.0,
    )
    expected = min(
        max(math.log(2.0) + 0.1 * (0.07 - 0.02), math.log(0.5)),
        math.log(4.0),
    )
    np.testing.assert_allclose(float(observed), expected, atol=1e-7)


def test_non_bellman_response_loss_cannot_move_feature() -> None:
    from src.path_c.model import build_model

    model = build_model(
        hidden_dim=8,
        slot_count=2,
        action_count=3,
        response_count=4,
        prior_scale=0.01,
        action_embedding_dim=4,
        log_standard_deviation_minimum=-5.0,
        log_standard_deviation_maximum=2.0,
    )
    features = jnp.ones((1, 8), dtype=jnp.float32)
    belief = normalized_log_belief(jnp.zeros((1, 2), dtype=jnp.float32))
    previous_actions = jnp.asarray([3], dtype=jnp.int32)
    previous_rewards = jnp.zeros((1,), dtype=jnp.float32)
    params = model.init(
        jax.random.PRNGKey(1),
        features,
        previous_actions,
        previous_rewards,
        belief,
    )["params"]

    def outcome_sum(candidate: jax.Array) -> jax.Array:
        raw = model.apply(
            {"params": params},
            candidate,
            previous_actions,
            previous_rewards,
            belief,
        )
        return jnp.sum(raw["reward_mean"]) + jnp.sum(raw["response_logits"])

    gradient = jax.grad(outcome_sum)(features)
    np.testing.assert_array_equal(np.asarray(gradient), np.zeros((1, 8)))
