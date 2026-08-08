from __future__ import annotations

from dataclasses import replace

import numpy as np


def _setup(variant: str = "delta_active"):
    import jax

    from src.delta_zsc.config import load_config
    from src.delta_zsc.model import DeltaModel

    config = load_config(
        "experiments/overcooked_v2/configs/delta_unified_simple_mechanical.yaml",
        run_kind="mechanical",
    )
    config = replace(
        config,
        method_variant=variant,
        model=replace(
            config.model,
            task_hidden_dim=16,
            task_embedding_dim=16,
            instant_partner_dim=8,
            latent_hidden_dim=16,
            latent_embedding_dim=8,
            action_embedding_dim=4,
        ),
        ppo=replace(config.ppo, update_epochs=1),
    )
    model = DeltaModel(config, (5, 5, 39), 6)
    base, latent = model.init_parameters(jax.random.PRNGKey(0))
    return config, model, base, latent


def _batch(model, base, latent, *, time_count: int = 4, lanes: int = 4):
    import jax
    import jax.numpy as jnp

    from src.delta_zsc.losses import categorical_log_probability
    from src.delta_zsc.types import RolloutBatch

    observations = jax.random.normal(
        jax.random.PRNGKey(2), (time_count + 1, lanes, 5, 5, 39)
    ) * 0.01
    previous_actions = jnp.zeros((time_count + 1, lanes), dtype=jnp.int32)
    starts = jnp.zeros((time_count + 1, lanes), dtype=jnp.bool_).at[0].set(True)
    _, output = model.sequence(
        base,
        latent,
        model.initial_state(lanes),
        observations,
        previous_actions,
        starts,
    )
    actions = jnp.argmax(output.base_policy_logits[:-1], axis=-1)
    return RolloutBatch(
        observations=observations,
        response_next_observations=observations[1:],
        previous_actions=previous_actions,
        episode_starts=starts,
        actions=actions,
        rewards=jnp.full((time_count, lanes), 0.1),
        shaped_rewards=jnp.zeros((time_count, lanes)),
        dones=jnp.zeros((time_count, lanes), dtype=jnp.bool_),
        old_log_probabilities=categorical_log_probability(
            output.base_policy_logits[:-1], actions
        ),
        old_values=output.value[:-1],
        ppo_mask=jnp.ones((time_count, lanes)),
        beliefs=output.belief,
        initial_policy_state=model.initial_state(lanes),
    )


def _anchors():
    import jax.numpy as jnp

    from src.delta_zsc.types import AnchorBatch

    fit = jnp.asarray(
        [[2.0, 1.0, 0.0, -1.0, -2.0, -3.0], [-1, 0, 1, 2, 3, 4]],
        dtype=jnp.float32,
    )
    replicas = jnp.stack((fit - 0.1, fit + 0.1), axis=-1)
    probe = jnp.broadcast_to(fit[:, None, :], (2, 6, 6))
    probe_replicas = jnp.stack((probe - 0.1, probe + 0.1), axis=-1)
    return AnchorBatch(
        time_indexes=jnp.asarray([1, 2], dtype=jnp.int32),
        lane_indexes=jnp.asarray([0, 1], dtype=jnp.int32),
        fit_returns_by_action=fit,
        evaluation_returns_by_action=fit,
        measurement_covariances=jnp.broadcast_to(
            jnp.eye(5, dtype=jnp.float32)[None] * 0.01, (2, 5, 5)
        ),
        action_mask=jnp.ones_like(fit, dtype=jnp.bool_),
        fit_replica_returns_by_action=replicas,
        evaluation_replica_returns_by_action=replicas,
        probe_fit_returns_by_action=probe,
        probe_evaluation_returns_by_action=probe,
        probe_measurement_covariances=jnp.broadcast_to(
            jnp.eye(5, dtype=jnp.float32)[None, None] * 0.01,
            (2, 6, 5, 5),
        ),
        probe_action_mask=jnp.ones_like(probe, dtype=jnp.bool_),
        probe_fit_replica_returns_by_action=probe_replicas,
        probe_evaluation_replica_returns_by_action=probe_replicas,
    )


def _same_tree(left, right) -> bool:
    import jax

    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    if len(left_leaves) != len(right_leaves):
        return False
    return all(
        np.array_equal(np.asarray(jax.device_get(one)), np.asarray(jax.device_get(other)))
        for one, other in zip(left_leaves, right_leaves)
    )


def test_base_and_latent_updates_are_separate_finite_transactions() -> None:
    import jax

    from src.delta_zsc.optimizer import init_adam
    from src.delta_zsc.training import environment_minibatch_schedule, training_update

    _, model, base, latent = _setup()
    batch = _batch(model, base, latent)
    schedule = environment_minibatch_schedule(
        jax.random.PRNGKey(4),
        environment_count=4,
        minibatches_per_epoch=1,
        update_epochs=1,
    )
    updated_base, updated_latent, _, _, metrics = training_update(
        model=model,
        base_params=base,
        latent_params=latent,
        base_optimizer_state=init_adam(base),
        latent_optimizer_state=init_adam(latent),
        batch=batch,
        anchors=_anchors(),
        schedule=schedule,
        total_optimizer_steps=10,
    )
    assert not _same_tree(updated_base, base)
    assert not _same_tree(updated_latent, latent)
    assert float(metrics["ppo"]["base_update_applied"]) == 1.0
    assert float(metrics["latent"]["latent_update_applied"]) == 1.0
    assert np.isfinite(float(metrics["latent"]["latent_composite_nll"]))
    assert float(metrics["latent"]["latent_decision_observations"]) == 2.0
    assert float(
        metrics["latent"]["latent_successor_decision_observations"]
    ) == 12.0
    # Full-tree channel pullbacks are intentionally excluded from the training
    # transaction.  Exact semantic/decision alignment is computed report-only
    # on the shared component embeddings in the final anchor audit.
    assert float(metrics["latent"]["latent_gradient_alignment_available"]) == 0.0



def test_outer_transaction_commits_latent_before_ppo(monkeypatch) -> None:
    """CRN labels must be scored against their collection-time base tree."""

    import jax.numpy as jnp
    import src.delta_zsc.training as training

    events: list[str] = []

    def fake_latent(**kwargs):
        assert float(kwargs["base_params"]) == 0.0
        assert float(kwargs["latent_params"]) == 0.0
        events.append("latent")
        return jnp.asarray(1.0), jnp.asarray(1.0), {
            "latent_update_applied": jnp.asarray(1.0)
        }

    def fake_base(**kwargs):
        assert events == ["latent"]
        events.append("ppo")
        return (
            kwargs["base_params"] + 1.0,
            kwargs["optimizer_state"] + 1.0,
            {"base_update_applied": jnp.asarray(1.0)},
        )

    monkeypatch.setattr(training, "update_latent_model", fake_latent)
    monkeypatch.setattr(training, "update_base_policy", fake_base)
    monkeypatch.setattr(training, "slice_rollout_lanes", lambda batch, indexes: batch)

    result = training.training_update(
        model=object(),
        base_params=jnp.asarray(0.0),
        latent_params=jnp.asarray(0.0),
        base_optimizer_state=jnp.asarray(0.0),
        latent_optimizer_state=jnp.asarray(0.0),
        batch=object(),
        anchors=object(),
        schedule=jnp.zeros((1, 1, 1), dtype=jnp.int32),
        total_optimizer_steps=1,
    )
    assert events == ["latent", "ppo"]
    assert tuple(float(value) for value in result[:4]) == (1.0, 1.0, 1.0, 1.0)

def test_response_only_variant_never_uses_decision_anchor_channel() -> None:
    import jax

    from src.delta_zsc.losses import latent_composite_loss

    _, model, base, latent = _setup("response_only")
    batch = _batch(model, base, latent)
    anchors = _anchors()
    result = jax.jit(
        lambda candidate: latent_composite_loss(
            model, candidate, base, batch, anchors
        )
    )(latent)
    assert float(result.metrics["latent_decision_observations"]) == 0.0
    assert float(result.metrics["latent_successor_decision_observations"]) == 0.0


def test_batched_base_policy_sequence_matches_step_replay() -> None:
    """P1 batching leaves the registered recurrent policy calculation intact."""

    import jax
    import jax.numpy as jnp

    from src.delta_zsc.base_policy import base_policy_sequence, base_policy_step
    _, model, base, _ = _setup("base")
    observations = jax.random.normal(
        jax.random.PRNGKey(31), (7, 3, 5, 5, 39), dtype=jnp.float32
    )
    starts = jnp.asarray(
        [
            [True, True, True],
            [False, False, False],
            [False, True, False],
            [False, False, False],
            [True, False, False],
            [False, False, True],
            [False, False, False],
        ],
        dtype=jnp.bool_,
    )
    initial = model.initial_state(3).task_carry

    beliefs = jnp.full(starts.shape + (4,), 0.25, dtype=jnp.float32)

    def one(carry, values):
        observation, episode_start, belief = values
        next_carry, task, instant, logits, value = base_policy_step(
            base,
            carry,
            observation,
            episode_start,
            belief,
            mask_partner_history=True,
        )
        return next_carry, (task, instant, logits, value)

    reference_carry, reference = jax.lax.scan(
        one, initial, (observations, starts, beliefs)
    )
    actual = base_policy_sequence(
        base,
        initial,
        observations,
        starts,
        beliefs,
        mask_partner_history=True,
    )
    (
        actual_carry,
        unused_task_carries,
        actual_task,
        actual_instant,
        actual_logits,
        actual_value,
    ) = actual
    del unused_task_carries
    for left, right in zip(
        (actual_carry, actual_task, actual_instant, actual_logits, actual_value),
        (reference_carry, *reference),
        strict=True,
    ):
        np.testing.assert_allclose(
            np.asarray(left), np.asarray(right), rtol=2.0e-5, atol=2.0e-5
        )


def test_batched_filter_sequence_matches_online_steps() -> None:
    import jax
    import jax.numpy as jnp

    _, model, base, latent = _setup("delta_active")
    observations = jax.random.normal(
        jax.random.PRNGKey(32), (5, 3, 5, 5, 39), dtype=jnp.float32
    )
    previous_actions = jax.random.randint(
        jax.random.PRNGKey(33), (5, 3), 0, 6
    )
    starts = jnp.zeros((5, 3), dtype=jnp.bool_).at[0].set(True)
    initial = model.initial_state(3)

    def one(state, values):
        observation, action, start = values
        return model.step(
            base,
            latent,
            state._replace(previous_action=action, episode_start=start),
            observation,
            compute_latent=True,
            compute_decision=False,
            execute_adaptation=False,
        )

    reference_state, reference = jax.lax.scan(
        one, initial, (observations, previous_actions, starts)
    )
    actual_state, actual = model.sequence(
        base,
        latent,
        initial,
        observations,
        previous_actions,
        starts,
        compute_latent=True,
        compute_decision=False,
        execute_adaptation=False,
    )
    for reference_leaf, actual_leaf in zip(
        jax.tree_util.tree_leaves(reference_state),
        jax.tree_util.tree_leaves(actual_state),
        strict=True,
    ):
        np.testing.assert_allclose(
            np.asarray(actual_leaf),
            np.asarray(reference_leaf),
            rtol=2.0e-5,
            atol=2.0e-5,
        )
    for name in (
        "task_features",
        "instant_partner",
        "base_policy_logits",
        "value",
        "belief",
        "behavior_features",
        "response_negative_log_likelihood",
    ):
        np.testing.assert_allclose(
            np.asarray(getattr(actual, name)),
            np.asarray(getattr(reference, name)),
            rtol=2.0e-5,
            atol=2.0e-5,
        )


def test_anchor_schema_contains_no_stale_posterior_or_comparator() -> None:
    from src.delta_zsc.types import AnchorBatch

    assert set(AnchorBatch._fields) == {
        "time_indexes",
        "lane_indexes",
        "fit_returns_by_action",
        "evaluation_returns_by_action",
        "measurement_covariances",
        "action_mask",
        "fit_replica_returns_by_action",
        "evaluation_replica_returns_by_action",
        "probe_fit_returns_by_action",
        "probe_evaluation_returns_by_action",
        "probe_measurement_covariances",
        "probe_action_mask",
        "probe_fit_replica_returns_by_action",
        "probe_evaluation_replica_returns_by_action",
    }
