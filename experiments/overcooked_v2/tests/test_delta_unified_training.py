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
        old_values=output.value,
        advantages=jnp.zeros((time_count, lanes)),
        returns=jnp.zeros((time_count, lanes)),
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
    from src.delta_zsc.contrast import pairwise_contrasts_from_replicas

    mask = jnp.ones_like(fit, dtype=jnp.bool_)
    contrast = pairwise_contrasts_from_replicas(replicas, mask)
    probe_contrast = pairwise_contrasts_from_replicas(
        probe_replicas, jnp.ones_like(probe, dtype=jnp.bool_)
    )
    return AnchorBatch(
        time_indexes=jnp.asarray([1, 2], dtype=jnp.int32),
        lane_indexes=jnp.asarray([0, 1], dtype=jnp.int32),
        contrast_mean=contrast.mean,
        contrast_standard_error=contrast.standard_error,
        contrast_valid=contrast.valid,
        probe_contrast_mean=probe_contrast.mean,
        probe_contrast_standard_error=probe_contrast.standard_error,
        probe_contrast_valid=probe_contrast.valid,
        task_phase=jnp.zeros((2,), dtype=jnp.int32),
        policy_version=jnp.asarray(0, dtype=jnp.int32),
        fit_returns_by_action=fit,
        evaluation_returns_by_action=fit,
        action_mask=jnp.ones_like(fit, dtype=jnp.bool_),
        fit_replica_returns_by_action=replicas,
        evaluation_replica_returns_by_action=replicas,
        probe_fit_returns_by_action=probe,
        probe_evaluation_returns_by_action=probe,
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
    from src.delta_zsc.training import (
        environment_minibatch_schedule,
        init_latent_optimizer,
        training_update,
    )

    _, model, base, latent = _setup()
    batch = _batch(model, base, latent)
    schedule = environment_minibatch_schedule(
        jax.random.PRNGKey(4),
        environment_count=4,
        minibatches_per_epoch=1,
        update_epochs=1,
    )
    (
        updated_base,
        updated_latent,
        updated_target,
        _,
        _,
        metrics,
    ) = training_update(
        model=model,
        base_params=base,
        latent_params=latent,
        target_latent_params=latent,
        base_optimizer_state=init_adam(base),
        latent_optimizer_state=init_latent_optimizer(latent),
        batch=batch,
        anchors=_anchors(),
        schedule=schedule,
        total_optimizer_steps=10,
    )
    from src.delta_zsc.losses import DECISION_METRIC_NAMES

    assert not _same_tree(updated_base, base)
    assert not _same_tree(updated_latent, latent)
    # The target moves, but by a fraction of the online step: it is a slow copy,
    # not a second online critic.
    assert not _same_tree(updated_target, latent)
    import jax
    import jax.numpy as jnp

    def _distance(left, right):
        leaves = jax.tree_util.tree_leaves(
            jax.tree_util.tree_map(lambda a, b: jnp.sum(jnp.square(a - b)), left, right)
        )
        return float(jnp.sqrt(sum(leaves)))

    assert _distance(updated_target, latent) < 0.5 * _distance(updated_latent, latent)
    assert float(metrics["ppo"]["base_update_applied"]) == 1.0
    assert float(metrics["latent"]["latent_update_applied"]) == 1.0
    assert np.isfinite(float(metrics["latent"]["latent_composite_nll"]))
    # The raw-return critic trains inside this transaction on every step.
    assert float(metrics["latent"]["raw_value_total"]) > 0.0

    # The anchor contrast is a *separate* executable, applied by the caller
    # after this one commits.  Keeping it out of this module is what makes the
    # anchor path compile: fused, the two 256-step recurrent scans produced an
    # HLO module XLA could not finish.
    from src.delta_zsc.training import update_contrast_channel

    calibrated, contrast_state, contrast_metrics = update_contrast_channel(
        model=model,
        latent_params=updated_latent,
        base_params=updated_base,
        optimizer_state=init_latent_optimizer(latent)["contrast"],
        batch=batch,
        anchors=_anchors(),
    )
    assert float(contrast_metrics["contrast_weight"]) > 0.0
    assert float(contrast_metrics["latent_contrast_update_applied"]) == 1.0
    # Only the critic moved; the response channels are untouched by it.
    assert not _same_tree(calibrated["belief_value"], updated_latent["belief_value"])
    assert _same_tree(calibrated["response"], updated_latent["response"])
    # Every decision key is present whether or not the anchor channel fired, so
    # accumulation across updates cannot silently drop the measured ones.
    for name in DECISION_METRIC_NAMES:
        assert name in metrics["latent"]



def test_outer_transaction_commits_latent_before_ppo(monkeypatch) -> None:
    """CRN labels must be scored against their collection-time base tree."""

    import jax.numpy as jnp
    import src.delta_zsc.training as training

    events: list[str] = []

    def fake_latent(**kwargs):
        assert float(kwargs["base_params"]) == 0.0
        assert float(kwargs["latent_params"]) == 0.0
        events.append("latent")
        return jnp.asarray(1.0), jnp.asarray(1.0), jnp.asarray(1.0), {
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
        target_latent_params=jnp.asarray(0.0),
        base_optimizer_state=jnp.asarray(0.0),
        latent_optimizer_state=jnp.asarray(0.0),
        batch=object(),
        anchors=object(),
        schedule=jnp.zeros((1, 1, 1), dtype=jnp.int32),
        total_optimizer_steps=1,
    )
    assert events == ["latent", "ppo"]
    assert tuple(float(value) for value in result[:5]) == (1.0, 1.0, 1.0, 1.0, 1.0)

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
    assert float(result.metrics["contrast_weight"]) == 0.0
    assert float(result.metrics["raw_value_total"]) == 0.0


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
        "contrast_mean",
        "contrast_standard_error",
        "contrast_valid",
        "probe_contrast_mean",
        "probe_contrast_standard_error",
        "probe_contrast_valid",
        "task_phase",
        "policy_version",
        "fit_returns_by_action",
        "evaluation_returns_by_action",
        "action_mask",
        "fit_replica_returns_by_action",
        "evaluation_replica_returns_by_action",
        "probe_fit_returns_by_action",
        "probe_evaluation_returns_by_action",
        "probe_action_mask",
        "probe_fit_replica_returns_by_action",
        "probe_evaluation_replica_returns_by_action",
    }


def test_response_and_decision_channels_are_parameter_disjoint() -> None:
    """The two latent channels must not be able to fight over a parameter.

    This replaces the reported semantic/decision gradient cosine.  The decision
    side is now the belief-conditioned critic, which reads the task features,
    the instant partner encoding and the posterior stop-gradiented and does not
    touch the component embeddings; the response side owns the embeddings and
    the emissions.  Disjointness is therefore a structural fact, and a test is
    the right place to keep it one -- wiring the critic into the embeddings
    would silently reintroduce the coupling the audit used to have to measure.
    """

    import jax
    import jax.numpy as jnp

    from src.delta_zsc.losses import (
        latent_composite_loss,
        pairwise_crn_contrast_loss,
        raw_task_value_loss,
    )

    _, model, base, latent = _setup()
    batch = _batch(model, base, latent)
    anchors = _anchors()

    def decision_only(candidate):
        _, output = model.sequence(
            base,
            candidate,
            batch.initial_policy_state,
            batch.observations,
            batch.previous_actions,
            batch.episode_starts,
            compute_latent=True,
            compute_decision=False,
            execute_adaptation=False,
        )
        value, _ = raw_task_value_loss(model, candidate, output, batch)
        contrast, _ = pairwise_crn_contrast_loss(candidate, output, anchors)
        return value + contrast

    gradients = jax.jit(jax.grad(decision_only))(latent)
    for name in ("component_embeddings", "response", "probe_response"):
        norm = float(
            jnp.sqrt(
                sum(
                    jnp.sum(jnp.square(leaf))
                    for leaf in jax.tree_util.tree_leaves(gradients[name])
                )
            )
        )
        assert norm == 0.0, f"{name} received decision-channel gradient {norm}"
    critic_norm = float(
        jnp.sqrt(
            sum(
                jnp.sum(jnp.square(leaf))
                for leaf in jax.tree_util.tree_leaves(gradients["belief_value"])
            )
        )
    )
    assert critic_norm > 0.0

    # And the converse: the full composite must still reach the critic, so the
    # channel is genuinely part of the training transaction rather than an
    # unreferenced head.
    full = jax.jit(
        jax.grad(lambda c: latent_composite_loss(model, c, base, batch, anchors).total)
    )(latent)
    full_critic = float(
        jnp.sqrt(
            sum(
                jnp.sum(jnp.square(leaf))
                for leaf in jax.tree_util.tree_leaves(full["belief_value"])
            )
        )
    )
    assert full_critic > 0.0


def test_raw_value_channel_trains_without_any_anchor() -> None:
    """The failure this replaces: 28 non-zero decision updates out of 3656.

    The component head only entered the objective when an anchor batch existed.
    The critic must receive gradient on an ordinary update that carries none.
    """

    import jax
    import jax.numpy as jnp

    from src.delta_zsc.losses import latent_composite_loss

    _, model, base, latent = _setup()
    batch = _batch(model, base, latent)
    result = jax.jit(
        lambda c: latent_composite_loss(model, c, base, batch, None)
    )(latent)
    assert float(result.metrics["raw_value_total"]) > 0.0
    assert float(result.metrics["contrast_weight"]) == 0.0

    gradients = jax.jit(
        jax.grad(lambda c: latent_composite_loss(model, c, base, batch, None).total)
    )(latent)
    norm = float(
        jnp.sqrt(
            sum(
                jnp.sum(jnp.square(leaf))
                for leaf in jax.tree_util.tree_leaves(gradients["belief_value"])
            )
        )
    )
    assert norm > 0.0


def test_ppo_advantage_is_fixed_across_candidate_parameters() -> None:
    """The advantage must not move while it is being optimised.

    PPO takes its surrogate ratio against the collection-time policy, so the
    advantage has to be computed once from the collection-time critic and held
    fixed for every epoch and minibatch -- which is what the Official
    implementation does.  This loss used to recompute GAE from whichever
    candidate critic the optimiser currently held, so each minibatch optimised
    a slightly different objective and the value target was a regression onto
    the network doing the regressing.  Identical hyperparameters do not make
    that the same algorithm.

    Evaluating the loss at two different parameter trees must therefore report
    the same advantage.
    """

    import jax
    import jax.numpy as jnp

    from src.delta_zsc.losses import ppo_loss

    _, model, base, latent = _setup()
    batch = _batch(model, base, latent)
    # A materially different critic: perturb every base parameter.
    other = jax.tree_util.tree_map(lambda leaf: leaf + 0.5, base)

    first = ppo_loss(model, base, latent, batch)
    second = ppo_loss(model, other, latent, batch)

    assert float(first.metrics["gae_advantage_mean"]) == float(
        second.metrics["gae_advantage_mean"]
    )
    assert float(first.metrics["gae_target_mean"]) == float(
        second.metrics["gae_target_mean"]
    )
    # And the value head really did move, so the test is not vacuous.
    assert float(first.metrics["ppo_value"]) != float(second.metrics["ppo_value"])


def test_rollout_batch_carries_the_terminal_bootstrap_value() -> None:
    """A fixed GAE cannot be reconstructed without the T+1-th value."""

    import jax.numpy as jnp

    _, model, base, latent = _setup()
    batch = _batch(model, base, latent)
    steps = int(jnp.asarray(batch.rewards).shape[0])
    assert int(jnp.asarray(batch.old_values).shape[0]) == steps + 1
    assert int(jnp.asarray(batch.advantages).shape[0]) == steps
    assert int(jnp.asarray(batch.returns).shape[0]) == steps


def _official_calculate_gae(rewards, dones, values, last_value, gamma, gae_lambda):
    """_calculate_gae from the pinned Official ippo.py, transcribed.

    Kept as an independent transcription rather than a call into the Official
    package: the point is to catch DELTA's estimator drifting away from the
    reference, and a shared implementation could not.
    """

    import jax
    import jax.numpy as jnp

    def _get_advantages(carry, transition):
        gae, next_value = carry
        done, value, reward = transition
        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * gae_lambda * (1 - done) * gae
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_value), last_value),
        (dones, values, rewards),
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + values


def test_gae_matches_the_official_implementation() -> None:
    """DELTA's advantage must be the Official one, not merely a plausible one.

    Identical hyperparameters do not make two implementations the same
    algorithm; this loss previously recomputed GAE from the candidate critic
    inside every minibatch, which no hyperparameter check would have caught.
    """

    import jax
    import jax.numpy as jnp

    from src.delta_zsc.losses import generalized_advantage_estimation

    steps, lanes = 24, 5
    key = jax.random.PRNGKey(0)
    rewards = jax.random.normal(jax.random.fold_in(key, 1), (steps, lanes))
    values = jax.random.normal(jax.random.fold_in(key, 2), (steps + 1, lanes))
    dones = (
        jax.random.uniform(jax.random.fold_in(key, 3), (steps, lanes)) < 0.15
    )
    gamma, lam = 0.99, 0.95

    ours_adv, ours_ret = generalized_advantage_estimation(
        rewards=rewards, dones=dones, values=values, gamma=gamma, gae_lambda=lam
    )
    theirs_adv, theirs_ret = _official_calculate_gae(
        rewards,
        dones.astype(jnp.float32),
        values[:-1],
        values[-1],
        gamma,
        lam,
    )
    np.testing.assert_allclose(
        np.asarray(ours_adv), np.asarray(theirs_adv), atol=1e-5, rtol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(ours_ret), np.asarray(theirs_ret), atol=1e-5, rtol=1e-5
    )


def test_ppo_loss_terms_match_the_official_formulas() -> None:
    """Actor, value and entropy terms must be the Official ones.

    Transcribed from the pinned _loss_fn: clipped surrogate on a
    std-normalised advantage, value loss as the max of clipped and unclipped
    squared error against the fixed target, entropy subtracted with its
    coefficient.
    """

    import jax
    import jax.numpy as jnp

    from src.delta_zsc.losses import categorical_log_probability, ppo_loss

    _, model, base, latent = _setup()
    batch = _batch(model, base, latent)
    cfg = model.config.ppo

    result = ppo_loss(model, base, latent, batch)

    # Recompute the Official way from the same rollout.
    _, output = model.sequence(
        base, latent, batch.initial_policy_state, batch.observations,
        batch.previous_actions, batch.episode_starts,
        compute_latent=False, execute_adaptation=False, beliefs=batch.beliefs,
    )
    logits, value = output.base_policy_logits[:-1], output.value[:-1]
    gae = jnp.asarray(batch.advantages, jnp.float32)
    targets = jnp.asarray(batch.returns, jnp.float32)
    gae = (gae - gae.mean()) / (gae.std() + 1e-8)

    log_prob = categorical_log_probability(logits, batch.actions)
    ratio = jnp.exp(log_prob - batch.old_log_probabilities)
    actor = -jnp.minimum(
        ratio * gae,
        jnp.clip(ratio, 1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon) * gae,
    ).mean()

    old_value = jnp.asarray(batch.old_values, jnp.float32)[:-1]
    clipped = old_value + jnp.clip(
        value - old_value, -cfg.value_clip_epsilon, cfg.value_clip_epsilon
    )
    value_loss = 0.5 * jnp.maximum(
        jnp.square(value - targets), jnp.square(clipped - targets)
    ).mean()

    logp = jax.nn.log_softmax(logits, axis=-1)
    entropy = (-jnp.sum(jnp.exp(logp) * logp, axis=-1)).mean()

    np.testing.assert_allclose(float(result.metrics["ppo_actor"]), float(actor), atol=1e-5)
    np.testing.assert_allclose(float(result.metrics["ppo_value"]), float(value_loss), atol=1e-5)
    np.testing.assert_allclose(float(result.metrics["ppo_entropy"]), float(entropy), atol=1e-5)
    np.testing.assert_allclose(
        float(result.total),
        float(actor + cfg.value_weight * value_loss - cfg.entropy_weight * entropy),
        atol=1e-5,
    )
