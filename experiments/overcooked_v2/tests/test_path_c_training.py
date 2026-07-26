from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from src.path_c.method import empty_codebook, uniform_slot_log_belief, update_codebook
from src.path_c.model import build_model, initialize_heads
from src.path_c.runner import RunnerFunctions, partner_callbacks
from src.path_c.training import (
    bellman_loss,
    bellman_targets,
    episode_responsibility_evidence,
    make_optimizers,
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


def test_responsibility_uses_complete_fixed_trajectory() -> None:
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


def test_responsibility_respects_a_literal_rollout_bootstrap_mask() -> None:
    q_values = jnp.asarray(
        [[[[[0.0], [0.0]], [[0.0], [0.0]]]]], dtype=jnp.float32
    )
    responsibilities, unused_energies = episode_responsibility_evidence(
        q_values=q_values,
        actions=jnp.asarray([[0]], dtype=jnp.int32),
        targets=jnp.zeros((1, 1, 2), dtype=jnp.float32),
        temperature=1.0,
        availability_mask=jnp.asarray([[True, False]]),
    )
    del unused_energies
    np.testing.assert_array_equal(
        np.asarray(responsibilities), np.asarray([[1.0, 0.0]])
    )


def test_bootstrap_sampling_never_creates_an_empty_environment_lane() -> None:
    mask = sample_bootstrap_mask(
        jax.random.PRNGKey(12),
        environment_count=64,
        slot_count=8,
        probability=0.0,
    )
    np.testing.assert_array_equal(
        np.asarray(mask).sum(axis=-1), np.ones((64,), dtype=np.int64)
    )


def test_bellman_loss_and_response_encoder_have_independent_targets() -> None:
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
    # Huber(1)=0.5 and Huber(3)=2.5 for each of two estimators.
    expected = (2.0 * 0.25 * 0.5 + 2.0 * 0.75 * 2.5) / 2.0
    np.testing.assert_allclose(float(observed.total), expected, atol=1e-6)

    encoder = response_encoder_loss(
        predicted_signatures=jnp.asarray([[0.0, 1.0]], dtype=jnp.float32),
        target_signatures=jnp.asarray([[0.0, 1.0]], dtype=jnp.float32),
        target_codes=jnp.asarray([0], dtype=jnp.int32),
        codebook_embeddings=jnp.asarray(
            [[0.0, 1.0], [2.0, 2.0]], dtype=jnp.float32
        ),
    )
    expected_classification = -math.log(
        math.exp(0.0) / (math.exp(0.0) + math.exp(-5.0))
    )
    np.testing.assert_allclose(
        float(encoder.total), expected_classification, atol=1e-6
    )


def test_target_update_and_codebook_update_change_only_stated_values() -> None:
    target = {"weight": jnp.asarray([0.0, 10.0])}
    online = {"weight": jnp.asarray([4.0, 2.0])}
    updated = polyak_update(target, online, 0.25)
    np.testing.assert_allclose(np.asarray(updated["weight"]), [1.0, 8.0])

    state = empty_codebook(code_count=2, signature_dim=2)
    codebook = update_codebook(
        state,
        signatures=jnp.asarray([[0.0, 0.0], [4.0, 0.0]], dtype=jnp.float32),
        key=jax.random.PRNGKey(7),
        decay=0.0,
        replacement_after_rollouts=10,
    ).state
    observed = np.asarray(codebook.embeddings)
    assert {tuple(row) for row in observed.tolist()} == {(0.0, 0.0), (4.0, 0.0)}


def test_two_optimizers_update_only_their_mathematical_consumers() -> None:
    import optax

    model = build_model(
        hidden_dim=8,
        slot_count=2,
        action_count=3,
        response_count=4,
        prior_scale=0.01,
        action_embedding_dim=3,
        log_standard_deviation_minimum=-5.0,
        log_standard_deviation_maximum=2.0,
    )
    features = jnp.arange(8, dtype=jnp.float32).reshape(1, 8) / 10.0
    previous_actions = jnp.asarray([3], dtype=jnp.int32)
    previous_rewards = jnp.zeros((1,), dtype=jnp.float32)
    belief = uniform_slot_log_belief((1,), 2)
    observations = jnp.zeros((1, 1, 3, 3, 2), dtype=jnp.float32)
    head_params = initialize_heads(
        model,
        random_key=jax.random.PRNGKey(20),
        example_features=features,
        example_previous_actions=previous_actions,
        example_previous_team_rewards=previous_rewards,
        example_slot_log_belief=belief,
        example_observations=observations,
    )
    params = {
        "official": {"weight": jnp.asarray([1.0], dtype=jnp.float32)},
        "heads": head_params,
    }
    optimizers = make_optimizers(
        params,
        bellman_learning_rate=0.01,
        outcome_learning_rate=0.01,
        gradient_clip_norm=100.0,
    )
    def bellman_objective(candidate):
        output = model.apply(
            {"params": candidate["heads"]},
            features,
            previous_actions,
            previous_rewards,
            belief,
        )
        return jnp.sum(output["q_values"]) + jnp.sum(
            candidate["official"]["weight"]
        )

    def outcome_objective(candidate):
        output = model.apply(
            {"params": candidate["heads"]},
            features,
            previous_actions,
            previous_rewards,
            belief,
        )
        return jnp.sum(output["reward_mean"]) + jnp.sum(
            output["response_logits"]
        )

    bellman_gradient = jax.grad(bellman_objective)(params)
    outcome_gradient = jax.grad(outcome_objective)(params)
    bellman_updates, unused_bellman_state = optimizers.bellman_optimizer.update(
        bellman_gradient, optimizers.bellman_state, params
    )
    outcome_updates, unused_outcome_state = optimizers.outcome_optimizer.update(
        outcome_gradient, optimizers.outcome_state, params
    )
    del unused_bellman_state, unused_outcome_state
    bellman_params = optax.apply_updates(params, bellman_updates)
    outcome_params = optax.apply_updates(params, outcome_updates)

    def forward(candidate):
        return model.apply(
            {"params": candidate["heads"]},
            features,
            previous_actions,
            previous_rewards,
            belief,
        )

    before = forward(params)
    after_bellman = forward(bellman_params)
    after_outcome = forward(outcome_params)
    assert not np.array_equal(
        np.asarray(after_bellman["learned_q_values"]),
        np.asarray(before["learned_q_values"]),
    )
    np.testing.assert_array_equal(
        np.asarray(after_bellman["prior_q_values"]),
        np.asarray(before["prior_q_values"]),
    )
    np.testing.assert_array_equal(
        np.asarray(after_bellman["reward_mean"]),
        np.asarray(before["reward_mean"]),
    )
    np.testing.assert_array_equal(
        np.asarray(after_outcome["q_values"]),
        np.asarray(before["q_values"]),
    )
    assert not np.array_equal(
        np.asarray(after_outcome["reward_mean"]),
        np.asarray(before["reward_mean"]),
    )
    assert not np.array_equal(
        np.asarray(bellman_params["official"]["weight"]),
        np.asarray(params["official"]["weight"]),
    )
    np.testing.assert_array_equal(
        np.asarray(outcome_params["official"]["weight"]),
        np.asarray(params["official"]["weight"]),
    )


def test_frozen_current_partner_has_no_parameter_gradient() -> None:
    model = build_model(
        hidden_dim=8,
        slot_count=2,
        action_count=3,
        response_count=4,
        prior_scale=0.01,
        action_embedding_dim=3,
        log_standard_deviation_minimum=-5.0,
        log_standard_deviation_maximum=2.0,
    )
    observations = jnp.asarray([[1.0, -1.0]], dtype=jnp.float32)
    belief = uniform_slot_log_belief((1,), 2)
    head_params = model.init(
        jax.random.PRNGKey(8),
        jnp.ones((1, 8)),
        jnp.asarray([3], dtype=jnp.int32),
        jnp.zeros((1,), dtype=jnp.float32),
        belief,
    )["params"]

    def online_step(params, carry, values, starts):
        del starts
        features = values @ params
        logits = features[:, :3]
        return carry, features, logits, jnp.zeros((values.shape[0],))

    def reference_step(carry, values, starts):
        del values, starts
        return carry, jnp.zeros((1, 3)), jnp.zeros((1,))

    def heads_apply(
        params,
        features,
        previous_actions,
        previous_team_rewards,
        slot_belief,
    ):
        return model.apply(
            {"params": params},
            features,
            previous_actions,
            previous_team_rewards,
            slot_belief,
        )

    base = RunnerFunctions(
        online_step=online_step,
        reference_step=reference_step,
        heads_apply=heads_apply,
        encode_response=lambda *unused: None,
        partner_step=lambda *unused: None,
        partner_observe=lambda *unused: None,
    )
    initial, step, unused_observe = partner_callbacks(
        base_functions=base,
        official_initial_carry=lambda batch: jnp.zeros((batch, 8)),
        static_partner_step=lambda member, values, carry, starts, key: (
            jnp.zeros(member.shape, dtype=jnp.int32),
            carry,
        ),
        slot_count=2,
        action_count=3,
        initial_temperature=1.0,
        gamma=0.99,
        terminal_response=3,
    )
    del unused_observe
    params = {
        "official": jnp.ones((2, 8), dtype=jnp.float32),
        "heads": head_params,
    }

    def partner_value(candidate):
        unused_action, unused_state, context = step(
            candidate,
            jnp.asarray([0], dtype=jnp.int32),
            observations,
            initial(1),
            jnp.asarray([True]),
            jax.random.PRNGKey(9),
        )
        del unused_action, unused_state
        return jnp.sum(context.dynamic_output.q_values)

    gradient = jax.grad(partner_value)(params)
    for leaf in jax.tree_util.tree_leaves(gradient):
        np.testing.assert_array_equal(np.asarray(leaf), np.zeros_like(np.asarray(leaf)))
