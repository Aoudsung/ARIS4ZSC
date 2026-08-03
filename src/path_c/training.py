"""Official PPO plus V6 end-to-end actor/belief objectives."""

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


def gaussian_kl_loss(mean: Any, log_standard_deviation: Any, *, free_bits_per_dimension: float) -> Any:
    from .belief_set_encoder import gaussian_kl_standard_normal

    return gaussian_kl_standard_normal(
        mean,
        log_standard_deviation,
        free_bits_per_dimension=free_bits_per_dimension,
    )


def robust_generalist_loss(*, model: Any, params: Any, output: Any, mask: Any):
    import jax
    import jax.numpy as jnp

    from .belief_set_encoder import prior_gaussian_summary

    task = jax.lax.stop_gradient(output.task_features[:-1])
    full_logits = model.apply(
        {"params": params}, task, output.belief_summary[:-1], method=model.policy_logits_from_features_and_summary
    )
    prior_summary = prior_gaussian_summary(
        output.belief_summary[:-1], output.belief_mean.shape[-1]
    )
    prior_logits = model.apply(
        {"params": params}, task, prior_summary, method=model.policy_logits_from_features_and_summary
    )
    full_log = jax.nn.log_softmax(full_logits, axis=-1)
    prior_log = jax.nn.log_softmax(prior_logits, axis=-1)
    probability = jnp.exp(full_log)
    items = jnp.asarray(output.normalized_uncertainty[:-1]) * jnp.sum(
        probability * (full_log - prior_log), axis=-1
    )
    valid = jnp.asarray(mask, dtype=jnp.float32)
    loss = jnp.sum(valid * items) / jnp.maximum(jnp.sum(valid), 1.0)
    return loss, {
        "robust_generalist_kl": loss,
        "full_context_entropy": jnp.mean(categorical_entropy(full_logits)),
        "prior_context_entropy": jnp.mean(categorical_entropy(prior_logits)),
    }


def _q_policy_from_output(*, output: Any, model: Any, params: Any, batch: Any, config: Any):
    import jax
    import jax.numpy as jnp

    q1, q2 = output.raw_q1[:-1], output.raw_q2[:-1]
    q = jax.lax.stop_gradient(jnp.minimum(q1, q2))
    ordered = jnp.sort(q, axis=-1)
    gap = ordered[..., -1] - ordered[..., -2]
    disagreement = jnp.mean(jnp.abs(q1 - q2), axis=-1)
    weight = jax.nn.sigmoid(
        (gap - float(config.loss.q_policy_gap_midpoint))
        / float(config.loss.q_policy_gap_temperature)
    ) * jnp.exp(-disagreement / float(config.loss.q_policy_disagreement_temperature))
    full_logits = model.apply(
        {"params": params}, jax.lax.stop_gradient(output.task_features[:-1]), output.belief_summary[:-1], method=model.policy_logits_from_features_and_summary
    )
    target = jax.nn.softmax(q / float(config.loss.q_policy_temperature), axis=-1)
    items = jnp.sum(
        target * (jnp.log(jnp.maximum(target, 1.0e-8)) - jax.nn.log_softmax(full_logits, axis=-1)),
        axis=-1,
    )
    valid = jnp.asarray(batch.ppo_mask, dtype=jnp.float32)
    loss = jnp.sum(valid * weight * items) / jnp.maximum(jnp.sum(valid), 1.0)
    return loss, {
        "q_policy_coupling_loss": loss,
        "q_policy_weight_mean": jnp.sum(valid * weight) / jnp.maximum(jnp.sum(valid), 1.0),
        "q_policy_gap_mean": jnp.mean(gap),
        "q_policy_head_disagreement_mean": jnp.mean(disagreement),
    }


def compute_loss(*, model: Any, params: Mapping[str, Any], batch: RolloutBatch, config: Any, anchors: Any = None, quotient_pairs: Any = None) -> LossBundle:
    import jax.numpy as jnp

    if anchors is not None or quotient_pairs is not None:
        raise ValueError("V6 replay updates are independent of PPO minibatches.")
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
    q_policy, q_metrics = _q_policy_from_output(
        output=output, model=model, params=params, batch=batch, config=config
    )
    robust, robust_metrics = robust_generalist_loss(
        model=model, params=params, output=output, mask=batch.ppo_mask
    )
    total = (
        actor
        + float(config.ppo.value_weight) * value
        - float(config.ppo.entropy_weight) * entropy
        + float(config.loss.q_policy_weight) * q_policy
        + float(config.loss.robust_generalist_weight) * robust
    )
    return LossBundle(total, {
        **actor_metrics,
        **q_metrics,
        **robust_metrics,
        "total_loss": total,
        "value_loss": value,
        "entropy": entropy,
        "mean_raw_reward": jnp.mean(batch.rewards),
        "mean_shaped_reward": jnp.mean(batch.shaped_rewards),
        "mean_belief_uncertainty": jnp.mean(output.normalized_uncertainty[:-1]),
        "context_dropout_fraction": jnp.mean(batch.context_dropout_masks[:-1].astype(jnp.float32)),
    })


def owner_behavior_distillation_loss(*, model: Any, params: Any, batch: RolloutBatch, owner_logits: Any):
    import jax
    import jax.numpy as jnp

    _, output = model.apply(
        {"params": params}, batch.initial_policy_state, batch.observations,
        batch.previous_actions, batch.episode_starts,
        jnp.zeros_like(batch.context_dropout_masks), method=model.sequence,
    )
    source = jax.nn.softmax(jnp.asarray(owner_logits, dtype=jnp.float32), axis=-1)
    source_log = jax.nn.log_softmax(jnp.asarray(owner_logits, dtype=jnp.float32), axis=-1)
    candidate_log = jax.nn.log_softmax(output.policy_logits[:-1], axis=-1)
    items = jnp.sum(source * (source_log - candidate_log), axis=-1)
    mask = jnp.asarray(batch.ppo_mask, dtype=jnp.float32)
    loss = jnp.sum(mask * items) / jnp.maximum(jnp.sum(mask), 1.0)
    return loss, {"owner_distillation_kl": loss}


def belief_objective_gradients(*, model: Any, params: Any, batch: Any, config: Any):
    """Compute PPO/Q-policy/robust/IB belief gradients from one frozen snapshot."""

    import jax
    import jax.numpy as jnp

    def objective(name: str):
        def loss(candidate: Any):
            _, output = model.apply(
                {"params": candidate}, batch.initial_policy_state, batch.observations,
                batch.previous_actions, batch.episode_starts,
                batch.context_dropout_masks, method=model.sequence,
            )
            if name == "ppo":
                advantages, returns = generalized_advantage_estimation(
                    rewards=batch.shaped_rewards, dones=batch.dones,
                    values=batch.old_values,
                    gamma=config.ppo.gamma, gae_lambda=config.ppo.gae_lambda,
                )
                actor, _ = ppo_actor_loss(
                    logits=output.policy_logits[:-1], actions=batch.actions,
                    old_log_probabilities=batch.old_log_probabilities,
                    advantages=advantages, clip_epsilon=config.ppo.clip_epsilon,
                    normalize_advantages=config.ppo.normalize_advantages, mask=batch.ppo_mask,
                )
                value = clipped_value_loss(
                    predictions=output.state_value[:-1],
                    old_predictions=batch.old_values[:-1],
                    targets=returns,
                    clip_epsilon=config.ppo.value_clip_epsilon,
                    mask=batch.ppo_mask,
                )
                entropy = (
                    jnp.sum(
                        categorical_entropy(output.policy_logits[:-1])
                        * batch.ppo_mask
                    )
                    / jnp.maximum(jnp.sum(batch.ppo_mask), 1.0)
                )
                return (
                    actor
                    + float(config.ppo.value_weight) * value
                    - float(config.ppo.entropy_weight) * entropy
                )
            if name == "q_policy":
                value, _ = _q_policy_from_output(
                    output=output, model=model, params=candidate, batch=batch, config=config
                )
                return value
            if name == "robust":
                value, _ = robust_generalist_loss(
                    model=model, params=candidate, output=output, mask=batch.ppo_mask
                )
                return value
            if name == "information_bottleneck":
                return gaussian_kl_loss(
                    output.belief_mean[:-1], output.belief_log_standard_deviation[:-1],
                    free_bits_per_dimension=config.loss.information_bottleneck_free_bits_per_dimension,
                )
            raise AssertionError(name)
        return loss

    values: dict[str, Any] = {}
    gradients: dict[str, Any] = {}
    for name in ("ppo", "q_policy", "robust", "information_bottleneck"):
        values[name], gradients[name] = jax.value_and_grad(objective(name))(params)
    return values, gradients


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


def apply_training_core_update(*, model: Any, core: TrainingCoreState, optimizer: Any, batch: RolloutBatch, config: Any, anchors: Any = None, quotient_pairs: Any = None, task_trunk_scale: Any = 1.0, base_actor_scale: Any = 1.0):
    import jax
    import jax.numpy as jnp
    import optax
    from .gradient_routing import keep_owned_gradients, scale_gradient_prefixes

    del task_trunk_scale, base_actor_scale

    def objective(candidate: Any):
        loss = compute_loss(model=model, params=candidate, batch=batch, config=config, anchors=anchors, quotient_pairs=quotient_pairs)
        return loss.total, loss.metrics

    (_, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(core.params)
    gradients = keep_owned_gradients(gradients, loss_name="ppo")
    updates, optimizer_state = optimizer.update(gradients, core.ppo_optimizer_state, core.params)
    params = optax.apply_updates(core.params, updates)
    candidate = TrainingCoreState(params, core.target_params, optimizer_state)
    leaves = jax.tree_util.tree_leaves((candidate.params, candidate.ppo_optimizer_state))
    finite = jnp.isfinite(metrics["total_loss"]) & jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in leaves]))
    committed = jax.lax.cond(finite, lambda _: candidate, lambda _: core, operand=None)
    return committed, {**metrics, "optimizer_applied": finite.astype(jnp.float32), "nonfinite_update": (~finite).astype(jnp.float32)}


def scan_training_updates(*, model: Any, core: TrainingCoreState, optimizer: Any, batch: RolloutBatch, schedule: Any, config: Any, task_trunk_scale: Any = 1.0, base_actor_scale: Any = 1.0):
    import jax
    import jax.numpy as jnp

    del task_trunk_scale, base_actor_scale
    indexes = jnp.asarray(schedule, dtype=jnp.int32)
    flat = indexes.reshape((-1, indexes.shape[-1]))

    def one(carry: tuple[Any, Any], lane_indexes: Any):
        current, active = carry

        def apply(value: Any):
            updated, metrics = apply_training_core_update(
                model=model, core=value, optimizer=optimizer,
                batch=slice_rollout_lanes(batch, lane_indexes), config=config,
            )
            stop = metrics["nonfinite_update"] > 0.5
            return updated, {
                **metrics,
                "training_aborted_nonfinite": stop.astype(jnp.float32),
            }

        def skip(value: Any):
            zero = jnp.asarray(0.0, dtype=jnp.float32)
            return value, {
                "actor_loss": zero, "approx_kl": zero, "clip_fraction": zero,
                "total_loss": zero, "value_loss": zero, "entropy": zero,
                "q_policy_coupling_loss": zero, "q_policy_weight_mean": zero,
                "q_policy_gap_mean": zero, "q_policy_head_disagreement_mean": zero,
                "robust_generalist_kl": zero, "full_context_entropy": zero,
                "prior_context_entropy": zero, "mean_raw_reward": zero,
                "mean_shaped_reward": zero, "mean_belief_uncertainty": zero,
                "context_dropout_fraction": zero, "optimizer_applied": zero,
                "nonfinite_update": zero,
                "training_aborted_nonfinite": jnp.asarray(1.0),
            }

        updated, metrics = jax.lax.cond(active, apply, skip, current)
        return (
            updated,
            active & (metrics["training_aborted_nonfinite"] < 0.5),
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


def apply_training_update(*, model: Any, state: TrainState, optimizer: Any, batch: RolloutBatch, config: Any, anchors: Any = None, quotient_pairs: Any = None) -> TrainingUpdate:
    core, metrics = apply_training_core_update(
        model=model, core=training_core_state(state), optimizer=optimizer,
        batch=batch, config=config, anchors=anchors, quotient_pairs=quotient_pairs,
    )
    return TrainingUpdate(merge_training_core(state, core), metrics)


__all__ = [
    "apply_training_core_update", "apply_training_update",
    "belief_objective_gradients", "categorical_entropy", "categorical_log_probability",
    "clipped_value_loss", "compute_loss", "environment_minibatch_schedule",
    "gather_actions", "gaussian_kl_loss", "generalized_advantage_estimation", "huber",
    "make_optimizer", "merge_training_core", "official_learning_rate_schedule",
    "official_reward_shaping_factor", "owner_behavior_distillation_loss", "polyak_update",
    "ppo_actor_loss", "robust_generalist_loss", "scan_training_updates",
    "slice_rollout_lanes", "training_core_state",
]
