from __future__ import annotations

import gzip
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pytest

from src.path_c.training.adaptation import (
    adaptation_loss,
    clipped_actor_loss,
    environment_minibatch,
    environment_minibatches,
)
from src.path_c.training.prefit import make_prefit_optimizer, prefit_update
from src.path_c.training.rollout import (
    RolloutCallbacks,
    balanced_family_member_assignment,
    collect_rollout,
    initialize_rollout,
)


def test_family_first_schedule_is_exactly_balanced_and_uniform_within_family() -> None:
    unused_jax, jnp = _jax_modules()
    episode_ids = jnp.arange(2_500)
    assignment = np.asarray(
        balanced_family_member_assignment(episode_ids, (1, 3, 6, 6))
    )
    prototype_lookup = np.asarray(
        (0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3)
    )
    families = prototype_lookup[assignment]
    assert [int(np.sum(families == index)) for index in range(4)] == [625] * 4
    for members in ((0,), (1, 2, 3), tuple(range(4, 10)), tuple(range(10, 16))):
        counts = [int(np.sum(assignment == member)) for member in members]
        assert max(counts) - min(counts) <= 1


def test_current_policy_partner_is_frozen_for_one_rollout_and_owns_independent_state() -> None:
    source = inspect.getsource(collect_rollout)
    assert "jax.tree_util.tree_map(" in source
    assert "jax.lax.stop_gradient, params" in source
    assert "current.partner_carry" in source
    assert "current.partner_log_belief" in source
    assert "current.partner_budget_remaining" in source
    assert "partner_model_key" in source
    assert "partner_controller_key" in source
    assert "jax.lax.scan(one_step, state, xs=None, length=length)" in source
    assert "partner_indices=recorded[19]" in source
    assert "partner_member_indices=recorded[20]" in source


def _jax_modules():
    jax = pytest.importorskip("jax")
    pytest.importorskip("optax")
    return jax, pytest.importorskip("jax.numpy")


class _FakeEnvironment:
    num_envs = 2
    observation_shape = (3,)
    num_actions = 3
    episode_steps = 2

    def reset(self, key: Any):
        del key
        _, jnp = _jax_modules()
        state = jnp.zeros((2,), dtype=jnp.int32)
        observations = jnp.zeros((2, 2, 3), dtype=jnp.float32)
        return state, observations

    def step(self, state: Any, joint_actions: Any, key: Any):
        del key
        _, jnp = _jax_modules()
        next_state = state + 1
        done = next_state >= 2
        observations = jnp.stack(
            (
                jnp.stack((next_state, joint_actions[:, 0], joint_actions[:, 1]), axis=-1),
                jnp.stack((next_state, joint_actions[:, 1], joint_actions[:, 0]), axis=-1),
            ),
            axis=1,
        ).astype(jnp.float32)
        info = {
            "correct_delivery": jnp.zeros((2,), dtype=jnp.int32),
            "wrong_delivery": jnp.zeros((2,), dtype=jnp.int32),
            "indicator_cost": jnp.zeros((2,), dtype=jnp.float32),
        }
        return next_state, observations, jnp.ones((2,)), done, info


def _callbacks():
    jax, jnp = _jax_modules()

    def model_apply(params, carry, observations, starts):
        del params, starts
        time, batch = observations.shape[:2]
        prototypes, actions, responses, features = 2, 3, 2, 2
        response_probability = jnp.asarray(
            [[[[0.9, 0.1]] * actions, [[0.2, 0.8]] * actions]]
        )
        response_probability = jnp.broadcast_to(
            response_probability, (time, batch, prototypes, actions, responses)
        )
        output = {
            "features": jnp.ones((time, batch, features)),
            "actor_logits": jnp.zeros((time, batch, actions)),
            "shared_value": jnp.zeros((time, batch)),
            "prototype_values": jnp.zeros((time, batch, prototypes)),
            "response_logits": jnp.log(response_probability),
            "response_probabilities": response_probability,
            "transition_response_logits": jnp.log(response_probability),
            "transition_response_probabilities": response_probability,
            "reward_estimates": jnp.zeros((time, batch, prototypes, actions)),
            "next_feature_summaries": jnp.zeros(
                (time, batch, prototypes, actions, responses, features)
            ),
            "next_values": jnp.zeros(
                (time, batch, prototypes, actions, responses, prototypes)
            ),
        }
        return carry + 1, output

    def partner_step(indices, observations, carry, starts, key):
        del indices, observations, starts
        return jax.random.randint(key, (carry.shape[0],), 0, 3), carry + 1

    return RolloutCallbacks(
        model_apply=model_apply,
        partner_step=partner_step,
        response_tokens=lambda previous, following, done, info: jnp.zeros(
            (previous.shape[0],), dtype=jnp.int32
        ),
        safe_action_mask=lambda observation: jnp.ones(
            (observation.shape[0], 3), dtype=jnp.bool_
        ),
        event_values=lambda info: (
            info["correct_delivery"],
            info["wrong_delivery"],
            info["indicator_cost"],
        ),
    )


def test_fake_environment_rollout_threads_carry_belief_and_random_key() -> None:
    jax, jnp = _jax_modules()
    environment = _FakeEnvironment()
    initial = initialize_rollout(
        environment=environment,
        model_initial_carry=lambda batch: jnp.zeros((batch, 2)),
        partner_initial_carry=lambda batch: jnp.zeros((batch, 2)),
        random_key=jax.random.PRNGKey(9),
        num_prototypes=2,
        budget_per_episode=1,
    )
    arguments = dict(
        environment=environment,
        callbacks=_callbacks(),
        params={"unused": jnp.asarray(0.0)},
        controller="off",
        candidate_window=1,
        budget_per_episode=1,
        gamma=0.99,
        gae_lambda=0.95,
        probability_floor=1e-8,
        decision_threshold=0.0,
        information_threshold=0.0,
        random_trigger_probability=0.0,
        num_prototypes=2,
    )
    middle, first_batch = collect_rollout(state=initial, length=1, **arguments)
    assert not np.allclose(
        np.exp(np.asarray(middle.log_belief)), np.full((2, 2), 0.5)
    )
    final, second_batch = collect_rollout(state=middle, length=1, **arguments)
    assert first_batch.actions.shape == (1, 2)
    assert first_batch.response_logits.shape == (1, 2, 2, 3, 2)
    np.testing.assert_allclose(np.asarray(first_batch.advantages), 1.0)
    np.testing.assert_allclose(np.asarray(first_batch.returns), 1.0)
    assert final.model_carry.shape == (2, 2)
    assert final.partner_carry.shape == (2, 2)
    assert int(np.asarray(final.completed_episodes)) == 2
    assert int(np.asarray(final.effective_environment_steps)) == 4
    assert not np.array_equal(np.asarray(final.random_key), np.asarray(initial.random_key))
    assert np.isfinite(np.asarray(final.log_belief)).all()
    assert np.asarray(first_batch.actor_owned_action).all()
    assert np.asarray(second_batch.actor_owned_action).all()
    assert int(np.asarray(first_batch.has_safe_candidate).sum()) == 2
    assert int(np.asarray(second_batch.has_safe_candidate).sum()) == 0


def test_fixed_rollout_targets_survive_current_value_changes_and_minibatching() -> None:
    jax, jnp = _jax_modules()
    environment = _FakeEnvironment()
    initial = initialize_rollout(
        environment=environment,
        model_initial_carry=lambda batch: jnp.zeros((batch, 2)),
        partner_initial_carry=lambda batch: jnp.zeros((batch, 2)),
        random_key=jax.random.PRNGKey(19),
        num_prototypes=2,
        budget_per_episode=1,
    )
    unused_state, batch = collect_rollout(
        state=initial,
        length=2,
        environment=environment,
        callbacks=_callbacks(),
        params={"unused": jnp.asarray(0.0)},
        controller="off",
        candidate_window=1,
        budget_per_episode=1,
        gamma=0.99,
        gae_lambda=0.95,
        probability_floor=1e-8,
        decision_threshold=0.0,
        information_threshold=0.0,
        random_trigger_probability=0.0,
        num_prototypes=2,
    )
    del unused_state
    fixed_advantages = jnp.asarray(((1.0, 2.0), (4.0, 8.0)), dtype=jnp.float32)
    fixed_returns = jnp.asarray(((2.0, 3.0), (5.0, 9.0)), dtype=jnp.float32)
    batch = batch._replace(advantages=fixed_advantages, returns=fixed_returns)
    actor_scales = jnp.asarray(((0.1, 0.4), (0.8, 1.3)), dtype=jnp.float32)
    current_actor_logits = jax.nn.one_hot(
        batch.actions, batch.actor_logits.shape[-1]
    ) * actor_scales[..., None]

    def outputs(shared_value):
        return {
            "actor_logits": current_actor_logits,
            "shared_value": shared_value,
            "prototype_values": batch.prototype_values,
            "response_logits": batch.response_logits,
            "transition_response_logits": batch.transition_response_logits,
            "reward_estimates": batch.reward_estimates,
            "next_feature_summaries": batch.next_feature_summaries,
        }

    weights = {
        "response": 0.0,
        "prototype_value": 0.0,
        "transition_response": 0.0,
        "reward": 0.0,
        "next_feature": 0.0,
    }
    first_value = jnp.zeros_like(batch.shared_values)
    second_value = jnp.asarray(((7.0, -3.0), (11.0, 2.0)), dtype=jnp.float32)
    unused_total, first_metrics = adaptation_loss(
        outputs(first_value),
        batch,
        gamma=0.99,
        clip_epsilon=0.2,
        entropy_coefficient=0.0,
        value_coefficient=0.5,
        auxiliary_loss_weights=weights,
    )
    del unused_total
    assert abs(float(np.asarray(first_metrics["actor"]))) > 1e-6
    for _ in range(4):
        unused_total, repeated_metrics = adaptation_loss(
            outputs(second_value),
            batch,
            gamma=0.99,
            clip_epsilon=0.2,
            entropy_coefficient=0.0,
            value_coefficient=0.5,
            auxiliary_loss_weights=weights,
        )
        del unused_total
        np.testing.assert_allclose(
            np.asarray(repeated_metrics["actor"]),
            np.asarray(first_metrics["actor"]),
        )
    expected_critic = 0.5 * np.mean(
        np.square(np.asarray(second_value) - np.asarray(fixed_returns))
    )
    np.testing.assert_allclose(
        np.asarray(repeated_metrics["critic"]), expected_critic
    )

    minibatches = environment_minibatches(
        batch,
        jnp.asarray((1, 0), dtype=jnp.int32),
        2,
    )
    np.testing.assert_array_equal(
        np.asarray(minibatches[0].advantages[:, 0]),
        np.asarray(fixed_advantages[:, 1]),
    )
    np.testing.assert_array_equal(
        np.asarray(minibatches[1].returns[:, 0]),
        np.asarray(fixed_returns[:, 0]),
    )
    streamed = environment_minibatch(
        batch,
        jnp.asarray((1,), dtype=jnp.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(streamed.advantages[:, 0]),
        np.asarray(fixed_advantages[:, 1]),
    )


def test_prefit_optimizer_cannot_modify_backbone_or_shared_actor() -> None:
    jax, jnp = _jax_modules()
    params = {
        "backbone": {"kernel": jnp.asarray([1.0])},
        "shared_heads": {
            "actor_logits": {"kernel": jnp.asarray([2.0])},
            "critic_value": {"kernel": jnp.asarray([3.0])},
        },
        "prototype_values": {"kernel": jnp.asarray([4.0])},
    }
    optimizer, optimizer_state = make_prefit_optimizer(params, learning_rate=0.1)

    def loss_function(candidate):
        leaves = jax.tree_util.tree_leaves(candidate)
        value = sum(jnp.sum(jnp.square(leaf)) for leaf in leaves)
        return value, {"registered": value}

    updated, unused_state, metrics = prefit_update(
        params=params,
        optimizer_state=optimizer_state,
        optimizer=optimizer,
        loss_function=loss_function,
    )
    del unused_state
    np.testing.assert_array_equal(updated["backbone"]["kernel"], params["backbone"]["kernel"])
    np.testing.assert_array_equal(
        updated["shared_heads"]["actor_logits"]["kernel"],
        params["shared_heads"]["actor_logits"]["kernel"],
    )
    assert not np.array_equal(
        np.asarray(updated["prototype_values"]["kernel"]),
        np.asarray(params["prototype_values"]["kernel"]),
    )
    assert bool(np.asarray(metrics["parameters_finite"]))


def test_controller_owned_steps_have_exactly_zero_actor_gradient() -> None:
    jax, jnp = _jax_modules()
    actions = jnp.asarray([[0, 1]])
    old_log_probability = jnp.log(jnp.asarray([[0.5, 0.5]]))
    advantages = jnp.asarray([[1.0, -1.0]])

    def loss(logits, mask):
        return clipped_actor_loss(
            logits=logits,
            actions=actions,
            old_log_probabilities=old_log_probability,
            advantages=advantages,
            actor_owned_action=mask,
            clip_epsilon=0.2,
            entropy_coefficient=0.01,
        )[0]

    logits = jnp.zeros((1, 2, 2))
    masked_gradient = jax.grad(
        lambda value: loss(value, jnp.zeros((1, 2), dtype=jnp.bool_))
    )(logits)
    owned_gradient = jax.grad(
        lambda value: loss(value, jnp.ones((1, 2), dtype=jnp.bool_))
    )(logits)
    np.testing.assert_array_equal(np.asarray(masked_gradient), 0.0)
    assert np.linalg.norm(np.asarray(owned_gradient)) > 0.0


def test_formal_probe_ledger_is_gzip_and_omits_nonprobe_steps(
    tmp_path: Path,
) -> None:
    from experiments.overcooked_v2.model_dock.stage_runtime import (
        _write_rollout_records,
    )
    from src.path_c.training.metrics import JsonlLedger

    batch = SimpleNamespace(
        episode_ids=np.asarray(((0,), (0,))),
        episode_steps=np.asarray(((0,), (1,))),
        base_actions=np.asarray(((0,), (0,))),
        actions=np.asarray(((0,), (1,))),
        probed=np.asarray(((False,), (True,))),
        actor_owned_action=np.asarray(((True,), (False,))),
        candidate_actions=np.asarray(((1,), (1,))),
        decision_scores=np.asarray(((0.0,), (0.5,))),
        information_scores=np.asarray(((0.0,), (0.25,))),
        has_safe_candidate=np.asarray(((True,), (True,))),
        budget_remaining=np.asarray(((20,), (19,))),
        chosen_j_use=np.asarray(((0.0,), (2.0,))),
        chosen_j_mask=np.asarray(((0.0,), (1.0,))),
        value_base=np.asarray(((0.0,), (0.5,))),
        value_mask=np.asarray(((0.0,), (1.25,))),
        chosen_s_seq=np.asarray(((0.0,), (0.75,))),
        chosen_response_information=np.asarray(((0.0,), (0.25,))),
        partner_indices=np.asarray(((0,), (0,))),
        ego_seats=np.asarray(((0,), (0,))),
        episode_completed=np.asarray(((False,), (True,))),
        completed_episode_returns=np.asarray(((np.nan,), (3.0,))),
        completed_episode_probe_counts=np.asarray(((0,), (1,))),
        correct_deliveries=np.asarray(((0,), (1,))),
        wrong_deliveries=np.asarray(((0,), (0,))),
        indicator_costs=np.asarray(((0.0,), (0.0,))),
    )
    decision_path = tmp_path / "probe_decisions.jsonl.gz"
    episode_path = tmp_path / "episodes.jsonl"
    _write_rollout_records(
        decision_ledger=JsonlLedger(decision_path, truncate=True),
        episode_ledger=JsonlLedger(episode_path, truncate=True),
        batch=batch,
        controller="registered_response_sequential_branch_v1",
        partner_pool=(SimpleNamespace(training_run_id="partner_0"),),
        maximum_probe_budget=20,
        record_all_decisions=False,
    )
    with gzip.open(decision_path, "rt", encoding="utf-8") as handle:
        decisions = [json.loads(line) for line in handle if line.strip()]
    assert len(decisions) == 1
    assert decisions[0]["episode_step"] == 1
    assert decisions[0]["probed"] is True
    episodes = [
        json.loads(line)
        for line in episode_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(episodes) == 1
    assert episodes[0]["safe_candidate_opportunity_count"] == 2
    assert episodes[0]["probe_count"] == 1


def test_training_stages_release_rollout_references_before_next_collection() -> None:
    from experiments.overcooked_v2.model_dock.stage_runtime import PathCStageRuntime

    prefit_source = inspect.getsource(PathCStageRuntime.prefit)
    adaptation_source = inspect.getsource(PathCStageRuntime.adaptation)
    for source in (prefit_source, adaptation_source):
        assert "last_batch" not in source
        assert "last_metric_row" in source
        assert (
            "jax.block_until_ready((params, optimizer_state, last_metrics))" in source
        )
    assert "del minibatch, minibatches, batch" in prefit_source
    assert "jax.lax.scan" in adaptation_source
    assert "environment_minibatches(" not in adaptation_source
    assert "del environment_groups, permutation, batch" in adaptation_source
