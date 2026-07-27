from __future__ import annotations

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
    solve_temperature_for_target_kl,
    target_response_signatures,
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


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    weights = np.exp(shifted)
    return weights / weights.sum()


def test_use_and_mask_share_physical_posterior_but_choose_distinct_next_policies() -> None:
    # [batch=1, slot=2], [slot,current_action=1,response=2]
    belief = np.asarray([[0.7, 0.3]], dtype=np.float32)
    response = np.asarray(
        [[[[0.8, 0.2]], [[0.1, 0.9]]]], dtype=np.float32
    )
    reward = np.zeros((1, 2, 1), dtype=np.float32)
    # [B,E,M,A,Y,U]
    use_q = np.asarray(
        [
            [
                [[[[4.0, 0.0], [1.0, 3.0]]], [[[0.0, 2.0], [4.0, 0.0]]]],
                [[[[3.8, 0.2], [1.1, 2.8]]], [[[0.1, 1.9], [3.8, 0.2]]]],
            ]
        ],
        dtype=np.float32,
    )
    mask_q = use_q.copy()
    next_reference = np.zeros((1, 1, 2, 2), dtype=np.float32)
    observed = bellman_control_values(
        slot_belief=jnp.asarray(belief),
        response_probabilities=jnp.asarray(response),
        reward_mean=jnp.asarray(reward),
        next_q_use_mean=jnp.asarray(use_q),
        next_q_mask_mean=jnp.asarray(mask_q),
        next_reference_logits_mean=jnp.asarray(next_reference),
        temperature=jnp.asarray([1.0]),
        gamma=0.9,
    )

    # Literal evaluator for estimator 0.
    expected_use = 0.0
    expected_mask = 0.0
    expected_tv = 0.0
    for y in range(2):
        marginal = sum(
            belief[0, m] * response[0, m, 0, y] for m in range(2)
        )
        posterior = np.asarray(
            [
                belief[0, m] * response[0, m, 0, y] / marginal
                for m in range(2)
            ]
        )
        # The runtime policy is built from the pessimistic score across twins.
        use_scores = np.stack(
            [
                sum(
                    posterior[m] * use_q[0, estimator, m, 0, y]
                    for m in range(2)
                )
                for estimator in range(2)
            ]
        )
        mask_scores = np.stack(
            [
                sum(
                    belief[0, m] * mask_q[0, estimator, m, 0, y]
                    for m in range(2)
                )
                for estimator in range(2)
            ]
        )
        use_score = np.min(use_scores, axis=0)
        mask_score = np.min(mask_scores, axis=0)
        pi_use = _softmax(use_score)
        pi_mask = _softmax(mask_score)
        use_value = sum(
            posterior[m] * np.dot(pi_use, use_q[0, 0, m, 0, y])
            for m in range(2)
        )
        mask_value = sum(
            posterior[m] * np.dot(pi_mask, mask_q[0, 0, m, 0, y])
            for m in range(2)
        )
        expected_use += 0.9 * marginal * use_value
        expected_mask += 0.9 * marginal * mask_value
        expected_tv += marginal * 0.5 * np.abs(pi_use - pi_mask).sum()

    np.testing.assert_allclose(
        np.asarray(observed.j_use_by_estimator[0, 0, 0]),
        expected_use,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(observed.j_mask_by_estimator[0, 0, 0]),
        expected_mask,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(observed.per_action_expected_next_policy_tv[0, 0]),
        expected_tv,
        atol=1e-6,
    )


def test_action_independent_next_q_shift_cannot_create_policy_mediated_gain() -> None:
    belief = jnp.asarray([[0.5, 0.5]], dtype=jnp.float32)
    response = jnp.full((1, 2, 1, 2), 0.5, dtype=jnp.float32)
    base = jnp.asarray(
        [[[[[[1.0, -1.0], [1.0, -1.0]]], [[[1.0, -1.0], [1.0, -1.0]]]],
           [[[[1.0, -1.0], [1.0, -1.0]]], [[[1.0, -1.0], [1.0, -1.0]]]]]],
        dtype=jnp.float32,
    )
    shifted = base + 7.0
    values = bellman_control_values(
        slot_belief=belief,
        response_probabilities=response,
        reward_mean=jnp.zeros((1, 2, 1)),
        next_q_use_mean=shifted,
        next_q_mask_mean=base,
        next_reference_logits_mean=jnp.zeros((1, 1, 2, 2)),
        temperature=jnp.asarray([1.0]),
        gamma=0.99,
    )
    np.testing.assert_allclose(
        np.asarray(values.per_action_policy_mediated_gain), 0.0, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(values.per_action_expected_next_policy_tv), 0.0, atol=1e-6
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


def test_zero_scores_leave_reference_probabilities_unchanged() -> None:
    reference = jnp.asarray([[0.1, 1.4, -0.7]], dtype=jnp.float32)
    policy = regularized_policy(
        reference, jnp.zeros_like(reference), jnp.asarray([0.5])
    )
    np.testing.assert_allclose(
        np.asarray(policy.probabilities),
        np.asarray(jax.nn.softmax(reference, axis=-1)),
        atol=1e-7,
    )


def test_temperature_bisection_hits_the_registered_kl() -> None:
    reference = jnp.asarray(
        [[1.0, 0.0, -1.0], [0.2, -0.3, 0.7]], dtype=jnp.float32
    )
    score = jnp.asarray(
        [[2.0, -1.0, 0.5], [-0.5, 1.0, 2.5]], dtype=jnp.float32
    )
    temperature, achieved = solve_temperature_for_target_kl(
        reference_logits=reference,
        score=score,
        target_kl=0.02,
        minimum_temperature=0.05,
        maximum_temperature=20.0,
        iterations=24,
    )
    assert 0.05 <= float(temperature) <= 20.0
    np.testing.assert_allclose(float(achieved), 0.02, atol=2e-5)


def test_response_signature_is_advantage_delta_mean_and_dispersion() -> None:
    current = jnp.zeros((1, 2, 3, 2), dtype=jnp.float32)
    following = jnp.asarray(
        [[
            [[1.0, -1.0], [3.0, -3.0], [5.0, -5.0]],
            [[1.0, -1.0], [1.0, -1.0], [1.0, -1.0]],
        ]],
        dtype=jnp.float32,
    )
    observed = target_response_signatures(
        current_centered_advantages=current,
        next_centered_advantages=following,
    )
    delta = np.asarray(following)
    expected = np.concatenate(
        (delta.mean(axis=(1, 2)), delta.std(axis=(1, 2))), axis=-1
    )
    np.testing.assert_allclose(np.asarray(observed), expected, atol=1e-6)


def test_non_bellman_response_loss_cannot_move_feature() -> None:
    try:
        from src.path_c.model import build_model
        import flax  # noqa: F401
    except ImportError:
        return

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
    params = model.init(
        jax.random.PRNGKey(1),
        features,
        belief,
        method=model.from_features,
    )["params"]

    def outcome_sum(candidate: jax.Array) -> jax.Array:
        raw = model.apply(
            {"params": params}, candidate, belief, method=model.from_features
        )
        return jnp.sum(raw["reward_mean"]) + jnp.sum(raw["response_logits"])

    gradient = jax.grad(outcome_sum)(features)
    np.testing.assert_array_equal(np.asarray(gradient), np.zeros((1, 8)))
