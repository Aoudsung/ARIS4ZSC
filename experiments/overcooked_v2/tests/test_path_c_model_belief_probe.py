from __future__ import annotations

import numpy as np
import pytest

from src.path_c.belief.update import log_bayes_update, update_mask, update_use
from src.path_c.probe.candidates import enumerate_candidates
from src.path_c.probe.controllers import (
    information_controller,
    off_controller,
    random_safe_controller,
    sequential_controller,
)
from src.path_c.probe.calibration import calibrate, nearest_rank
from src.path_c.probe.scores import SequentialScores, sequential_scores


def _jax_modules():
    jax = pytest.importorskip("jax")
    return jax, pytest.importorskip("jax.numpy")


def test_two_prototype_log_bayes_update_and_use_mask_difference() -> None:
    unused_jax, jnp = _jax_modules()
    prior = jnp.log(jnp.asarray([[0.25, 0.75]], dtype=jnp.float32))
    likelihood = jnp.asarray([[0.8, 0.2]], dtype=jnp.float32)
    posterior = np.exp(np.asarray(log_bayes_update(prior, likelihood, probability_floor=1e-8)))
    np.testing.assert_allclose(posterior, [[4.0 / 7.0, 3.0 / 7.0]], rtol=1e-6)
    probabilities = jnp.asarray([[[0.8, 0.2], [0.2, 0.8]]])
    token = jnp.asarray([0])
    used = np.exp(np.asarray(update_use(prior, probabilities, token, probability_floor=1e-8)))
    masked = np.exp(np.asarray(update_mask(prior, probabilities, token, probability_floor=1e-8)))
    np.testing.assert_allclose(masked, [[0.25, 0.75]], rtol=1e-6)
    assert not np.allclose(used, masked)


def _manual_scores(belief, probability, reward, continuation, base, safe, gamma):
    batch, prototypes, actions, responses = probability.shape
    j_use = np.zeros((batch, actions))
    j_mask = np.zeros((batch, actions))
    for row in range(batch):
        for action in range(actions):
            for prototype in range(prototypes):
                for response in range(responses):
                    likelihood = probability[row, :, action, response]
                    posterior = belief[row] * likelihood
                    posterior = posterior / posterior.sum()
                    continuation_use = np.dot(
                        posterior, continuation[row, prototype, action, response]
                    )
                    continuation_mask = np.dot(
                        belief[row], continuation[row, prototype, action, response]
                    )
                    mass = belief[row, prototype] * probability[row, prototype, action, response]
                    j_use[row, action] += mass * (
                        reward[row, prototype, action] + gamma * continuation_use
                    )
                    j_mask[row, action] += mass * (
                        reward[row, prototype, action] + gamma * continuation_mask
                    )
    v_base = j_mask[np.arange(batch), base]
    v_mask = np.maximum(v_base, np.max(np.where(safe, j_mask, -np.inf), axis=1))
    return j_use, j_mask, v_base, v_mask, np.where(safe, j_use - v_mask[:, None], -np.inf)


def test_exact_one_step_score_matches_independent_hand_calculation() -> None:
    unused_jax, jnp = _jax_modules()
    belief = np.asarray([[0.4, 0.6], [0.7, 0.3]])
    probability = np.asarray(
        [
            [[[0.8, 0.2], [0.5, 0.5]], [[0.3, 0.7], [0.6, 0.4]]],
            [[[0.4, 0.6], [0.9, 0.1]], [[0.7, 0.3], [0.2, 0.8]]],
        ]
    )
    reward = np.asarray(
        [[[1.0, -0.5], [0.0, 2.0]], [[-1.0, 0.25], [1.5, 0.75]]]
    )
    continuation = (
        np.arange(2 * 2 * 2 * 2 * 2, dtype=np.float64).reshape(2, 2, 2, 2, 2)
        / 10.0
    )
    base = np.asarray([0, 1])
    safe = np.asarray([[False, True], [True, False]])
    expected = _manual_scores(belief, probability, reward, continuation, base, safe, 0.9)
    observed = sequential_scores(
        log_belief=jnp.log(jnp.asarray(belief)),
        response_probabilities=jnp.asarray(probability),
        reward_estimates=jnp.asarray(reward),
        next_values=jnp.asarray(continuation),
        base_action=jnp.asarray(base),
        candidate_mask=jnp.asarray(safe),
        gamma=0.9,
        probability_floor=1e-8,
    )
    for actual, wanted in zip(observed, expected, strict=True):
        np.testing.assert_allclose(np.asarray(actual), wanted, rtol=1e-6, atol=1e-6)


def test_nearest_rank_quantile_is_non_interpolating_and_rejects_bad_inputs() -> None:
    assert nearest_rank([5.0, 1.0, 4.0, 2.0, 3.0], 0.80) == 4.0
    assert nearest_rank([2.0, 2.0, 9.0, 1.0], 0.80) == 9.0
    assert nearest_rank([7.0], 0.80) == 7.0
    for values, quantile in (([], 0.80), ([1.0, np.nan], 0.80), ([1.0], 0.0), ([1.0], 1.0)):
        with pytest.raises(ValueError):
            nearest_rank(values, quantile)


def test_zero_anchored_decision_threshold_and_independent_information_quantile() -> None:
    decision_values = [float(value) for value in range(-10, 10)]
    information_values = [float(value) / 100.0 for value in range(20)]
    rows = [
        {
            "has_safe_candidate": True,
            "max_decision_score": decision,
            "max_information_score": information,
            "probed": False,
        }
        for decision, information in zip(
            decision_values, information_values, strict=True
        )
    ]
    result = calibrate(
        rows,
        decision_null_quantile=0.95,
        information_quantile=0.80,
    )
    assert result.decision_null_quantile_value == 8.0
    assert result.decision_threshold == 8.0
    assert result.decision_null_exceedance_count == 1
    assert result.random_trigger_probability == 0.05
    assert result.information_threshold == 0.15

    negative_rows = [
        {
            "has_safe_candidate": True,
            "max_decision_score": value,
            "max_information_score": 0.25,
            "probed": False,
        }
        for value in (-4.0, -2.0, -2.0, -1.0)
    ]
    anchored = calibrate(
        negative_rows,
        decision_null_quantile=0.95,
        information_quantile=0.80,
    )
    assert anchored.decision_null_quantile_value == -1.0
    assert anchored.decision_threshold == 0.0
    assert anchored.decision_null_exceedance_count == 0
    assert anchored.random_trigger_probability == 0.0
    manual = calibrate(
        rows,
        decision_null_quantile=0.95,
        information_quantile=0.80,
        decision_threshold_override=-3.0,
    )
    assert manual.decision_threshold == 0.0


def test_candidate_window_budget_and_sampled_base_action_exclusion() -> None:
    unused_jax, jnp = _jax_modules()
    candidates = enumerate_candidates(
        safe_action_mask=jnp.asarray([[True, True, False], [True, True, True]]),
        base_action=jnp.asarray([1, 0]),
        episode_step=jnp.asarray([2, 10]),
        budget_remaining=jnp.asarray([1, 2]),
        candidate_window=10,
    )
    np.testing.assert_array_equal(np.asarray(candidates.mask), [[True, False, False], [False, False, False]])
    exhausted = enumerate_candidates(
        safe_action_mask=jnp.ones((1, 3), dtype=jnp.bool_),
        base_action=jnp.asarray([2]),
        episode_step=jnp.asarray([0]),
        budget_remaining=jnp.asarray([0]),
        candidate_window=10,
    )
    assert not bool(np.asarray(exhausted.any_candidate[0]))


def test_strict_threshold_random_safety_and_off_controller() -> None:
    jax, jnp = _jax_modules()
    score = sequential_scores(
        log_belief=jnp.log(jnp.asarray([[0.5, 0.5]])),
        response_probabilities=jnp.full((1, 2, 3, 2), 0.5),
        reward_estimates=jnp.asarray([[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]]),
        next_values=jnp.zeros((1, 2, 3, 2, 2)),
        base_action=jnp.asarray([0]),
        candidate_mask=jnp.asarray([[False, True, False]]),
        gamma=0.99,
        probability_floor=1e-8,
    )
    maximum = float(np.asarray(score.s_seq[0, 1]))
    equal = sequential_controller(
        scores=score,
        candidate_mask=jnp.asarray([[False, True, False]]),
        base_action=jnp.asarray([0]),
        budget_remaining=jnp.asarray([1]),
        threshold=maximum,
    )
    assert not bool(np.asarray(equal.probed[0]))
    assert set(equal.score_components) == {
        "j_use",
        "j_mask",
        "v_base",
        "v_mask",
        "s_seq",
        "response_information",
    }
    assert np.isfinite(np.asarray(equal.s_seq)).all()
    assert np.isnan(np.asarray(equal.response_information)).all()
    nonpositive = sequential_controller(
        scores=SequentialScores(
            j_use=jnp.zeros((2, 2)),
            j_mask=jnp.zeros((2, 2)),
            v_base=jnp.zeros((2,)),
            v_mask=jnp.zeros((2,)),
            s_seq=jnp.asarray([[-jnp.inf, 0.0], [-jnp.inf, -0.1]]),
        ),
        candidate_mask=jnp.asarray([[False, True], [False, True]]),
        base_action=jnp.asarray([0, 0]),
        budget_remaining=jnp.asarray([1, 1]),
        threshold=0.0,
    )
    assert not np.asarray(nonpositive.probed).any()
    information = information_controller(
        information_scores=jnp.asarray([[0.0, 0.5, 0.2]]),
        candidate_mask=jnp.asarray([[False, True, False]]),
        base_action=jnp.asarray([0]),
        budget_remaining=jnp.asarray([1]),
        threshold=0.1,
    )
    assert np.isnan(np.asarray(information.s_seq)).all()
    np.testing.assert_allclose(np.asarray(information.response_information), 0.5)
    safe = jnp.asarray([[False, True, False, True]])
    for seed in range(16):
        decision = random_safe_controller(
            key=jax.random.PRNGKey(seed),
            candidate_mask=safe,
            base_action=jnp.asarray([0]),
            budget_remaining=jnp.asarray([1]),
            trigger_probability=1.0,
        )
        assert int(np.asarray(decision.chosen_action[0])) in {1, 3}
        assert not bool(np.asarray(decision.actor_owned_action[0]))
    off = off_controller(base_action=jnp.asarray([2]), budget_remaining=jnp.asarray([5]))
    assert int(np.asarray(off.chosen_action[0])) == 2
    assert not bool(np.asarray(off.probed[0]))


def test_visible_goal_safety_uses_only_local_observation() -> None:
    unused_jax, jnp = _jax_modules()
    from experiments.overcooked_v2.model_dock.env_dock import visible_goal_safe_action_mask

    observations = np.zeros((2, 5, 5, 39), dtype=np.float32)
    observations[:, 2, 2, 3] = 1.0  # Direction.RIGHT is the third direction layer.
    observations[0, 2, 3, 21] = 1.0  # visible goal one cell ahead
    mask = visible_goal_safe_action_mask(
        jnp.asarray(observations),
        action_order=("right", "down", "left", "up", "stay", "interact"),
    )
    assert not bool(np.asarray(mask[0, 5]))
    assert bool(np.asarray(mask[1, 5]))
    assert np.asarray(mask[:, :5]).all()


def test_remote_environment_startup_guards_bind_observations_and_delivery_flags() -> None:
    pytest.importorskip("jax")
    pytest.importorskip("jaxmarl")
    from experiments.overcooked_v2.model_dock.env_dock import (
        OvercookedV2VectorEnvironment,
        verify_environment_startup_contracts,
    )

    report = verify_environment_startup_contracts(
        OvercookedV2VectorEnvironment.create(num_envs=2),
        action_order=("right", "down", "left", "up", "stay", "interact"),
    )
    assert report["observation_layout"]["passed"] is True
    assert report["observation_layout"]["goal_channel"] == 21
    assert report["observation_layout"]["other_agent_position_channel"] == 10
    assert report["delivery_counters"] == {
        "passed": True,
        "per_step_success_flag": True,
        "terminal_autoreset_preserves_success": True,
        "wrong_delivery_floor": 0,
        "checked_vector_lanes": 2,
    }
