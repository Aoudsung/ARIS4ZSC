"""DEPI combined-loss training (METHOD_SPEC §3).

L = L_PPO + lambda_A * L_signature + lambda_R * L_response + lambda_S * L_separation

All four terms are evaluated at the current parameters inside one
``jax.value_and_grad`` and a single gradient step updates every trainable
parameter (§3.3).  The eight-objective PCGrad / snapshot-gradient machinery is
abolished.  After each combined update the rollout batch reports
``combined_policy_kl`` (k1 estimator of the behaviour-vs-updated policy KL,
including every context-pathway effect); exceeding the registered threshold
terminates the remaining minibatches of the current PPO epoch (§3.4).
"""

from __future__ import annotations

from typing import Any, Mapping

from .types import LossBundle, RolloutBatch, TrainState, TrainingCoreState, TrainingUpdate


def gather_actions(values: Any, actions: Any, *, action_axis: int = -1) -> Any:
    import jax.numpy as jnp

    source = jnp.asarray(values)
    index = jnp.asarray(actions, dtype=jnp.int32)
    axis = action_axis if action_axis >= 0 else source.ndim + action_axis
    shape = list(index.shape)
    while len(shape) < source.ndim:
        shape.insert(axis, 1)
    return jnp.squeeze(jnp.take_along_axis(source, index.reshape(tuple(shape)), axis=axis), axis=axis)


def huber(error: Any, delta: float = 1.0) -> Any:
    import jax.numpy as jnp

    absolute = jnp.abs(jnp.asarray(error, dtype=jnp.float32))
    quadratic = jnp.minimum(absolute, float(delta))
    return 0.5 * jnp.square(quadratic) + float(delta) * (absolute - quadratic)


def categorical_log_probability(logits: Any, actions: Any) -> Any:
    import jax

    return gather_actions(jax.nn.log_softmax(logits, axis=-1), actions)


def categorical_entropy(logits: Any) -> Any:
    import jax
    import jax.numpy as jnp

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    probabilities = jnp.exp(log_probs)
    return -jnp.sum(probabilities * log_probs, axis=-1)


def generalized_advantage_estimation(*, rewards: Any, dones: Any, values: Any, gamma: float, gae_lambda: float):
    import jax
    import jax.numpy as jnp

    reward = jnp.asarray(rewards, dtype=jnp.float32)
    done = jnp.asarray(dones, dtype=jnp.bool_)
    value = jnp.asarray(values, dtype=jnp.float32)
    if value.shape[0] != reward.shape[0] + 1:
        raise ValueError("GAE values must contain T+1 states.")
    delta = reward + float(gamma) * (1.0 - done.astype(jnp.float32)) * value[1:] - value[:-1]

    def backward(carry: Any, items: tuple[Any, Any]):
        delta_t, done_t = items
        advantage = delta_t + float(gamma) * float(gae_lambda) * (1.0 - done_t.astype(jnp.float32)) * carry
        return advantage, advantage

    _, reverse = jax.lax.scan(backward, jnp.zeros_like(delta[-1]), (delta[::-1], done[::-1]))
    advantages = reverse[::-1]
    return jax.lax.stop_gradient(advantages), jax.lax.stop_gradient(advantages + value[:-1])


def ppo_actor_loss(*, logits: Any, actions: Any, old_log_probabilities: Any, advantages: Any, clip_epsilon: float, normalize_advantages: bool, mask: Any | None = None):
    import jax.numpy as jnp

    advantage = jnp.asarray(advantages, dtype=jnp.float32)
    weight = jnp.ones_like(advantage) if mask is None else jnp.asarray(mask, dtype=jnp.float32)
    denominator = jnp.maximum(jnp.sum(weight), 1.0e-8)
    if normalize_advantages:
        mean = jnp.sum(advantage * weight) / denominator
        variance = jnp.sum(jnp.square(advantage - mean) * weight) / denominator
        advantage = (advantage - mean) / (jnp.sqrt(variance) + 1.0e-8)
    log_ratio = categorical_log_probability(logits, actions) - jnp.asarray(old_log_probabilities)
    ratio = jnp.exp(log_ratio)
    clipped = jnp.clip(ratio, 1.0 - float(clip_epsilon), 1.0 + float(clip_epsilon))
    loss = -jnp.sum(jnp.minimum(ratio * advantage, clipped * advantage) * weight) / denominator
    approx_kl = jnp.sum(((ratio - 1.0) - log_ratio) * weight) / denominator
    return loss, {
        "actor_loss": loss,
        "approx_kl": approx_kl,
        "clip_fraction": jnp.sum((jnp.abs(ratio - 1.0) > float(clip_epsilon)).astype(jnp.float32) * weight) / denominator,
    }


def clipped_value_loss(*, predictions: Any, old_predictions: Any, targets: Any, clip_epsilon: float, mask: Any | None = None) -> Any:
    import jax.numpy as jnp

    predicted = jnp.asarray(predictions, dtype=jnp.float32)
    old = jnp.asarray(old_predictions, dtype=jnp.float32)
    target = jnp.asarray(targets, dtype=jnp.float32)
    clipped = old + jnp.clip(predicted - old, -float(clip_epsilon), float(clip_epsilon))
    items = 0.5 * jnp.maximum(
        jnp.square(predicted - target),
        jnp.square(clipped - target),
    )
    weight = jnp.ones_like(items) if mask is None else jnp.asarray(mask, dtype=jnp.float32)
    return jnp.sum(items * weight) / jnp.maximum(jnp.sum(weight), 1.0e-8)


def centered(values: Any) -> Any:
    import jax.numpy as jnp

    array = jnp.asarray(values, dtype=jnp.float32)
    return array - jnp.mean(array, axis=-1, keepdims=True)


def signature_loss(
    *,
    raw_q1: Any,
    raw_q2: Any,
    advantage_targets: Any,
    action_mask: Any,
    hinge_margin: float,
    hinge_advantage_gap: float,
) -> tuple[Any, Any, Any]:
    """L_signature: centered-Q Huber fit plus rank-preserving hinge (§3.2).

    Q_c = min(Q1, Q2) - mean_a min(Q1, Q2); targets are the centered CRN
    continuation returns A(a).  The hinge enforces the action ranking implied
    by advantages whose gap exceeds ``hinge_advantage_gap``.
    """

    import jax.numpy as jnp

    conservative = jnp.minimum(
        jnp.asarray(raw_q1, dtype=jnp.float32), jnp.asarray(raw_q2, dtype=jnp.float32)
    )
    q_centered = conservative - jnp.mean(conservative, axis=-1, keepdims=True)
    target = centered(advantage_targets)
    mask = jnp.asarray(action_mask, dtype=jnp.float32)
    denominator = jnp.maximum(jnp.sum(mask), 1.0)
    fit = jnp.sum(mask * huber(q_centered - target)) / denominator

    gap = jnp.abs(target[..., :, None] - target[..., None, :])
    order = jnp.sign(target[..., :, None] - target[..., None, :])
    q_gap = q_centered[..., :, None] - q_centered[..., None, :]
    pair_mask = (gap > float(hinge_advantage_gap)) & (order > 0.0)
    pair_mask = pair_mask & (mask[..., :, None] > 0.0) & (mask[..., None, :] > 0.0)
    hinge = jnp.maximum(
        0.0, float(hinge_margin) - q_gap * order
    )
    pair_denominator = jnp.maximum(jnp.sum(pair_mask.astype(jnp.float32)), 1.0)
    hinge_loss = jnp.sum(pair_mask.astype(jnp.float32) * hinge) / pair_denominator
    total = fit + hinge_loss
    return total, fit, hinge_loss


def signature_anchor_objective(*, model: Any, params: Any, anchors: Any, loss_v2: Any):
    """Anchor-batch signature loss; contributes 0 while anchors are disabled."""

    import jax.numpy as jnp

    zero = jnp.asarray(0.0, dtype=jnp.float32)
    if anchors is None:
        return zero, {"signature_loss": zero, "signature_huber_loss": zero, "signature_hinge_loss": zero}
    _, output = model.apply(
        {"params": params},
        anchors.policy_states,
        anchors.observations,
        method=model.step,
    )
    total, fit, hinge_loss = signature_loss(
        raw_q1=output.raw_q1,
        raw_q2=output.raw_q2,
        advantage_targets=anchors.fit_returns_by_action,
        action_mask=anchors.action_mask,
        hinge_margin=float(loss_v2.rank_hinge_margin),
        hinge_advantage_gap=float(loss_v2.rank_hinge_advantage_gap),
    )
    return total, {
        "signature_loss": total,
        "signature_huber_loss": fit,
        "signature_hinge_loss": hinge_loss,
    }


def separation_loss(
    *,
    context_a: Any,
    context_b: Any,
    equivalent_mask: Any,
    margin: Any,
    weights: Any,
) -> Any:
    """L_separation on matched pairs (§3.2).

    Observable-equivalent pairs (``equivalent_mask``) contribute
    ``||c_i - c_j||^2``; decision-distinct pairs contribute
    ``max(0, m - ||c_i - c_j||)`` with ``m`` the registered margin.
    """

    import jax.numpy as jnp

    delta = jnp.asarray(context_a, dtype=jnp.float32) - jnp.asarray(context_b, dtype=jnp.float32)
    distance = jnp.sqrt(jnp.sum(jnp.square(delta), axis=-1) + 1.0e-12)
    equivalent = jnp.asarray(equivalent_mask, dtype=jnp.bool_)
    margin_value = jnp.asarray(margin, dtype=jnp.float32)
    items = jnp.where(
        equivalent,
        jnp.square(distance),
        jnp.maximum(0.0, margin_value - distance),
    )
    pair_weights = jnp.asarray(weights, dtype=jnp.float32)
    return jnp.sum(pair_weights * items) / jnp.maximum(jnp.sum(pair_weights), 1.0)


def response_objective(*, model: Any, params: Any, batch: RolloutBatch):
    """Proper mixture NLL L_response over rollout transitions (§2.3)."""

    import jax.numpy as jnp

    from .response_targets import (
        extract_partner_response_targets,
        mixture_response_loss,
        official_partner_observation_planes,
    )

    _, prediction = model.apply(
        {"params": params},
        batch.initial_policy_state,
        batch.observations,
        batch.previous_actions,
        batch.episode_starts,
        batch.actions,
        method=model.response_sequence,
    )
    planes = official_partner_observation_planes(batch.observations.shape[-1])
    targets = extract_partner_response_targets(
        batch.observations[:-1], batch.response_next_observations, planes=planes
    )
    losses = mixture_response_loss(prediction, targets, mask=batch.ppo_mask)
    return losses.total, {
        "response_total_loss": losses.total,
        "response_visibility_loss": losses.visibility,
        "response_position_loss": losses.relative_position,
        "response_direction_loss": losses.direction,
        "response_inventory_loss": losses.inventory,
        "response_event_loss": losses.interaction_change,
    }


def compute_loss(*, model: Any, params: Mapping[str, Any], batch: RolloutBatch, config: Any, anchors: Any = None, separation_terms: Any = None) -> LossBundle:
    """Combined four-loss objective at the current parameters (§3.1/§3.3).

    ``separation_terms`` is the §3.2 ``SeparationTerms`` payload: its
    classification fields are precomputed constants, but the two ego
    forward passes run here on ``params`` so L_separation contributes a
    live gradient (§3.4 ownership: capability/protocol encoders).
    """

    import jax.numpy as jnp

    _, output = model.apply(
        {"params": params},
        batch.initial_policy_state,
        batch.observations,
        batch.previous_actions,
        batch.episode_starts,
        batch.context_dropout_masks,
        method=model.sequence,
    )
    advantages, returns = generalized_advantage_estimation(
        rewards=batch.shaped_rewards,
        dones=batch.dones,
        values=batch.old_values,
        gamma=config.ppo.gamma,
        gae_lambda=config.ppo.gae_lambda,
    )
    actor, actor_metrics = ppo_actor_loss(
        logits=output.policy_logits[:-1],
        actions=batch.actions,
        old_log_probabilities=batch.old_log_probabilities,
        advantages=advantages,
        clip_epsilon=config.ppo.clip_epsilon,
        normalize_advantages=config.ppo.normalize_advantages,
        mask=batch.ppo_mask,
    )
    value = clipped_value_loss(
        predictions=output.state_value[:-1],
        old_predictions=batch.old_values[:-1],
        targets=returns,
        clip_epsilon=config.ppo.value_clip_epsilon,
        mask=batch.ppo_mask,
    )
    entropy = jnp.sum(categorical_entropy(output.policy_logits[:-1]) * batch.ppo_mask) / jnp.maximum(jnp.sum(batch.ppo_mask), 1.0)
    ppo_total = (
        actor
        + float(config.ppo.value_weight) * value
        - float(config.ppo.entropy_weight) * entropy
    )

    response_total, response_metrics = response_objective(
        model=model, params=params, batch=batch
    )
    signature_total, signature_metrics = signature_anchor_objective(
        model=model, params=params, anchors=anchors, loss_v2=config.loss_v2
    )
    # §3.2/§3.4: L_separation is re-evaluated here on the *current* params so
    # it joins this same value_and_grad; only the §5.3 classification
    # (masks/weights/margin) is a precomputed constant carried in the payload.
    if separation_terms is None:
        separation_total = jnp.asarray(0.0, dtype=jnp.float32)
    else:
        pair_count = jnp.asarray(separation_terms.equivalent_mask).shape[0]
        dropped = jnp.zeros((pair_count,), dtype=jnp.bool_)
        _, output_a = model.apply(
            {"params": params},
            separation_terms.ego_state_a,
            separation_terms.probe_observations[0],
            dropped,
            method=model.step,
        )
        _, output_b = model.apply(
            {"params": params},
            separation_terms.ego_state_b,
            separation_terms.probe_observations[1],
            dropped,
            method=model.step,
        )
        separation_total = separation_loss(
            context_a=output_a.protocol_embedding,
            context_b=output_b.protocol_embedding,
            equivalent_mask=separation_terms.equivalent_mask,
            margin=separation_terms.margin,
            weights=separation_terms.weights,
        )

    total = (
        ppo_total
        + float(config.loss_v2.signature_weight) * signature_total
        + float(config.loss_v2.response_weight) * response_total
        + float(config.loss_v2.separation_weight) * separation_total
    )
    return LossBundle(total, {
        **actor_metrics,
        **response_metrics,
        **signature_metrics,
        "separation_loss": separation_total,
        "ppo_total_loss": ppo_total,
        "total_loss": total,
        "value_loss": value,
        "entropy": entropy,
        "mean_raw_reward": jnp.mean(batch.rewards),
        "mean_shaped_reward": jnp.mean(batch.shaped_rewards),
        "mean_posterior_entropy": jnp.mean(output.posterior_entropy[:-1]),
        "context_dropout_fraction": jnp.mean(batch.context_dropout_masks[:-1].astype(jnp.float32)),
        # The policy logits already condition on (x, u, c); the k1 estimate
        # therefore includes every context-pathway parameter effect (§3.4).
        "combined_policy_kl": actor_metrics["approx_kl"],
    })


def official_learning_rate_schedule(*, learning_rate: float, warmup_fraction: float, update_count: int, minibatches_per_epoch: int, update_epochs: int) -> Any:
    import optax

    warmup_updates = int(float(warmup_fraction) * int(update_count))
    steps_per_update = int(minibatches_per_epoch) * int(update_epochs)
    warmup = optax.linear_schedule(0.0, float(learning_rate), warmup_updates * steps_per_update)
    cosine = optax.cosine_decay_schedule(float(learning_rate), max(int(update_count) - warmup_updates, 1) * steps_per_update)
    return optax.join_schedules((warmup, cosine), (warmup_updates * steps_per_update,))


def official_reward_shaping_factor(environment_step: int | Any, *, horizon: int) -> Any:
    import jax.numpy as jnp

    return jnp.clip(1.0 - jnp.asarray(environment_step, dtype=jnp.float32) / float(horizon), 0.0, 1.0)


def make_optimizer(params: Any, *, learning_rate: float, gradient_clip_norm: float, adam_epsilon: float = 1.0e-5, anneal_learning_rate: bool = False, warmup_fraction: float = 0.0, update_count: int = 1, minibatches_per_epoch: int = 1, update_epochs: int = 1):
    import optax

    rate = official_learning_rate_schedule(
        learning_rate=learning_rate, warmup_fraction=warmup_fraction,
        update_count=update_count, minibatches_per_epoch=minibatches_per_epoch,
        update_epochs=update_epochs,
    ) if anneal_learning_rate else float(learning_rate)
    optimizer = optax.chain(optax.clip_by_global_norm(float(gradient_clip_norm)), optax.adam(rate, eps=float(adam_epsilon)))
    return optimizer, optimizer.init(params)


def polyak_update(target: Any, online: Any, coefficient: float) -> Any:
    import jax

    tau = float(coefficient)
    return jax.tree_util.tree_map(lambda old, new: (1.0 - tau) * old + tau * new, target, online)


def training_core_state(state: TrainState) -> TrainingCoreState:
    return TrainingCoreState(state.params, state.target_params, state.ppo_optimizer_state)


def merge_training_core(state: TrainState, core: TrainingCoreState) -> TrainState:
    return state._replace(params=core.params, target_params=core.target_params, ppo_optimizer_state=core.ppo_optimizer_state)


def apply_training_core_update(*, model: Any, core: TrainingCoreState, optimizer: Any, batch: RolloutBatch, config: Any, anchors: Any = None, separation_terms: Any = None):
    """One combined gradient step over all four losses (§3.3)."""

    import jax
    import jax.numpy as jnp
    import optax

    def objective(candidate: Any):
        loss = compute_loss(
            model=model, params=candidate, batch=batch, config=config,
            anchors=anchors, separation_terms=separation_terms,
        )
        return loss.total, loss.metrics

    (_, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(core.params)
    updates, optimizer_state = optimizer.update(gradients, core.ppo_optimizer_state, core.params)
    params = optax.apply_updates(core.params, updates)
    candidate = TrainingCoreState(params, core.target_params, optimizer_state)
    leaves = jax.tree_util.tree_leaves((candidate.params, candidate.ppo_optimizer_state))
    finite = jnp.isfinite(metrics["total_loss"]) & jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in leaves]))
    committed = jax.lax.cond(finite, lambda _: candidate, lambda _: core, operand=None)
    kl_stop = metrics["combined_policy_kl"] > float(
        config.loss_v2.combined_policy_kl_threshold
    )
    return committed, {
        **metrics,
        "optimizer_applied": finite.astype(jnp.float32),
        "nonfinite_update": (~finite).astype(jnp.float32),
        "kl_early_stop": kl_stop.astype(jnp.float32),
    }


def scan_training_updates(*, model: Any, core: TrainingCoreState, optimizer: Any, batch: RolloutBatch, schedule: Any, config: Any, anchors: Any = None, separation_terms: Any = None):
    """Scan minibatches with the §3.4 early-stop on combined_policy_kl.

    ``anchors`` and ``separation_terms`` are anchor-batch-global payloads
    (§5 matched-pair separation data path); they are shared by every
    minibatch step of the scan, never lane-sliced.  The separation payload
    carries states/observations/masks; the loss forward passes execute on
    the scan's current params inside ``compute_loss`` (§3.2).
    """

    import jax
    import jax.numpy as jnp

    indexes = jnp.asarray(schedule, dtype=jnp.int32)
    flat = indexes.reshape((-1, indexes.shape[-1]))

    def one(carry: tuple[Any, Any], lane_indexes: Any):
        current, active = carry

        def apply(value: Any):
            updated, metrics = apply_training_core_update(
                model=model, core=value, optimizer=optimizer,
                batch=slice_rollout_lanes(batch, lane_indexes), config=config,
                anchors=anchors, separation_terms=separation_terms,
            )
            stop = (metrics["nonfinite_update"] > 0.5) | (metrics["kl_early_stop"] > 0.5)
            return updated, {
                **metrics,
                "training_aborted": stop.astype(jnp.float32),
            }

        def skip(value: Any):
            zero = jnp.asarray(0.0, dtype=jnp.float32)
            return value, {
                "actor_loss": zero, "approx_kl": zero, "clip_fraction": zero,
                "response_total_loss": zero, "response_visibility_loss": zero,
                "response_position_loss": zero, "response_direction_loss": zero,
                "response_inventory_loss": zero, "response_event_loss": zero,
                "signature_loss": zero, "signature_huber_loss": zero,
                "signature_hinge_loss": zero, "separation_loss": zero,
                "ppo_total_loss": zero, "total_loss": zero, "value_loss": zero,
                "entropy": zero, "mean_raw_reward": zero, "mean_shaped_reward": zero,
                "mean_posterior_entropy": zero, "context_dropout_fraction": zero,
                "combined_policy_kl": zero, "optimizer_applied": zero,
                "nonfinite_update": zero, "kl_early_stop": zero,
                "training_aborted": jnp.asarray(1.0),
            }

        updated, metrics = jax.lax.cond(active, apply, skip, current)
        return (
            updated,
            active & (metrics["training_aborted"] < 0.5),
        ), metrics

    (final, _), metrics = jax.lax.scan(one, (core, jnp.asarray(True)), flat)
    return final, metrics


def slice_rollout_lanes(batch: RolloutBatch, indexes: Any) -> RolloutBatch:
    import jax

    values = {}
    for name, value in batch._asdict().items():
        values[name] = (
            jax.tree_util.tree_map(lambda leaf: leaf[indexes], value)
            if name in {"initial_policy_state", "initial_target_policy_state"}
            else value[:, indexes]
        )
    return RolloutBatch(**values)


def environment_minibatch_schedule(key: Any, *, environment_count: int, minibatches_per_epoch: int, update_epochs: int) -> Any:
    import jax
    import jax.numpy as jnp

    if environment_count % minibatches_per_epoch:
        raise ValueError("Environment lanes must divide exactly into minibatches.")
    lane_count = environment_count // minibatches_per_epoch
    keys = jax.random.split(key, update_epochs)
    return jnp.stack([jax.random.permutation(k, environment_count).reshape((minibatches_per_epoch, lane_count)) for k in keys])


def apply_training_update(*, model: Any, state: TrainState, optimizer: Any, batch: RolloutBatch, config: Any, anchors: Any = None, separation_terms: Any = None) -> TrainingUpdate:
    core, metrics = apply_training_core_update(
        model=model, core=training_core_state(state), optimizer=optimizer,
        batch=batch, config=config, anchors=anchors, separation_terms=separation_terms,
    )
    return TrainingUpdate(merge_training_core(state, core), metrics)


__all__ = [
    "apply_training_core_update", "apply_training_update",
    "categorical_entropy", "categorical_log_probability", "centered",
    "clipped_value_loss", "compute_loss", "environment_minibatch_schedule",
    "gather_actions", "generalized_advantage_estimation", "huber",
    "make_optimizer", "merge_training_core", "official_learning_rate_schedule",
    "official_reward_shaping_factor", "polyak_update", "ppo_actor_loss",
    "response_objective", "scan_training_updates", "separation_loss",
    "signature_anchor_objective", "signature_loss", "slice_rollout_lanes",
    "training_core_state",
]
