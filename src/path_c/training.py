"""Recurrent PPO and decision-equivalent auxiliary training for DELTA-ZSC."""

from __future__ import annotations

from typing import Any, Mapping

from .response_targets import (
    PartnerResponseTargets,
    structured_partner_response_loss,
)
from .types import (
    CounterfactualAnchorBatch,
    LossBundle,
    QuotientPairBatch,
    RolloutBatch,
    TrainState,
    TrainingCoreState,
    TrainingUpdate,
)


def gather_actions(values: Any, actions: Any, *, action_axis: int = -1) -> Any:
    import jax
    import jax.numpy as jnp

    source = jnp.asarray(values)
    index = jnp.asarray(actions, dtype=jnp.int32)
    axis = action_axis if action_axis >= 0 else source.ndim + action_axis
    shape = list(index.shape)
    while len(shape) < source.ndim:
        shape.insert(axis, 1)
    return jnp.squeeze(
        jnp.take_along_axis(source, index.reshape(tuple(shape)), axis=axis),
        axis=axis,
    )


def huber(error: Any, delta: float = 1.0) -> Any:
    import jax.numpy as jnp

    absolute = jnp.abs(jnp.asarray(error, dtype=jnp.float32))
    quadratic = jnp.minimum(absolute, float(delta))
    return 0.5 * jnp.square(quadratic) + float(delta) * (absolute - quadratic)


def categorical_log_probability(logits: Any, actions: Any) -> Any:
    import jax

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return gather_actions(log_probs, actions)


def categorical_entropy(logits: Any) -> Any:
    import jax
    import jax.numpy as jnp

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    probabilities = jnp.exp(log_probs)
    return -jnp.sum(probabilities * log_probs, axis=-1)


def stable_root_mean_square(values: Any) -> Any:
    """Return an RMS with a finite derivative at the all-zero origin.

    The conditional residual is deliberately initialized to exactly zero.
    ``sqrt(mean(x**2))`` has an infinite derivative at that point; composing
    it with an inactive hinge still produces ``0 * inf == NaN`` in reverse
    mode.  Adding the smallest positive normal float preserves the RMS at all
    scientifically relevant scales while making the zero-residual gradient
    finite.  This is a numerical definition at a non-differentiable point,
    not a loss, model, or budget change.
    """

    import jax.numpy as jnp

    array = jnp.asarray(values, dtype=jnp.float32)
    mean_square = jnp.mean(jnp.square(array))
    return jnp.sqrt(mean_square + jnp.finfo(array.dtype).tiny)


def conditional_entropy_noncollapse_penalty(
    base_logits: Any,
    conditional_logits: Any,
    mask: Any,
    *,
    tolerance: float = 0.1,
) -> Any:
    """Relative entropy floor without any fixed high-entropy target."""

    import jax.numpy as jnp

    base_entropy = categorical_entropy(base_logits)
    conditional_entropy = categorical_entropy(conditional_logits)
    weight = jnp.asarray(mask, dtype=jnp.float32)
    denominator = jnp.maximum(jnp.sum(weight), 1.0)
    return jnp.sum(
        weight
        * jnp.maximum(
            base_entropy - float(tolerance) - conditional_entropy,
            0.0,
        )
    ) / denominator


def generalized_advantage_estimation(
    *,
    rewards: Any,
    dones: Any,
    values: Any,
    gamma: float,
    gae_lambda: float,
) -> tuple[Any, Any]:
    """GAE for values containing T+1 states."""

    import jax
    import jax.numpy as jnp

    reward = jnp.asarray(rewards, dtype=jnp.float32)
    done = jnp.asarray(dones, dtype=jnp.bool_)
    value = jnp.asarray(values, dtype=jnp.float32)
    if value.shape[0] != reward.shape[0] + 1:
        raise ValueError("GAE values must contain T+1 states.")
    delta = reward + float(gamma) * (1.0 - done.astype(jnp.float32)) * value[1:] - value[:-1]

    def backward(carry: Any, values_at_time: tuple[Any, Any]) -> tuple[Any, Any]:
        delta_t, done_t = values_at_time
        advantage = delta_t + float(gamma) * float(gae_lambda) * (
            1.0 - done_t.astype(jnp.float32)
        ) * carry
        return advantage, advantage

    _, reversed_advantages = jax.lax.scan(
        backward,
        jnp.zeros_like(delta[-1]),
        (delta[::-1], done[::-1]),
    )
    advantages = reversed_advantages[::-1]
    returns = advantages + value[:-1]
    return jax.lax.stop_gradient(advantages), jax.lax.stop_gradient(returns)


def ppo_actor_loss(
    *,
    logits: Any,
    actions: Any,
    old_log_probabilities: Any,
    advantages: Any,
    clip_epsilon: float,
    normalize_advantages: bool,
    mask: Any | None = None,
) -> tuple[Any, Mapping[str, Any]]:
    import jax.numpy as jnp

    advantage = jnp.asarray(advantages, dtype=jnp.float32)
    weight = (
        jnp.ones_like(advantage)
        if mask is None
        else jnp.asarray(mask, dtype=jnp.float32)
    )
    denominator = jnp.maximum(jnp.sum(weight), 1.0e-8)
    if normalize_advantages:
        mean = jnp.sum(advantage * weight) / denominator
        variance = jnp.sum(jnp.square(advantage - mean) * weight) / denominator
        advantage = (advantage - mean) / (jnp.sqrt(variance) + 1.0e-8)
    new_log_prob = categorical_log_probability(logits, actions)
    log_ratio = new_log_prob - jnp.asarray(old_log_probabilities)
    ratio = jnp.exp(log_ratio)
    clipped = jnp.clip(ratio, 1.0 - float(clip_epsilon), 1.0 + float(clip_epsilon))
    objective = jnp.minimum(ratio * advantage, clipped * advantage)
    loss = -jnp.sum(objective * weight) / denominator
    approx_kl = jnp.sum(((ratio - 1.0) - log_ratio) * weight) / denominator
    clip_fraction = jnp.sum((jnp.abs(ratio - 1.0) > float(clip_epsilon)).astype(jnp.float32) * weight) / denominator
    return loss, {
        "actor_loss": loss,
        "approx_kl": approx_kl,
        "clip_fraction": clip_fraction,
        "mean_ratio": jnp.sum(ratio * weight) / denominator,
    }


def clipped_value_loss(
    *,
    predictions: Any,
    old_predictions: Any,
    targets: Any,
    clip_epsilon: float,
    mask: Any | None = None,
) -> Any:
    import jax.numpy as jnp

    predicted = jnp.asarray(predictions, dtype=jnp.float32)
    old = jnp.asarray(old_predictions, dtype=jnp.float32)
    target = jnp.asarray(targets, dtype=jnp.float32)
    clipped = old + jnp.clip(predicted - old, -float(clip_epsilon), float(clip_epsilon))
    items = jnp.maximum(huber(predicted - target), huber(clipped - target))
    weight = jnp.ones_like(items) if mask is None else jnp.asarray(mask, dtype=jnp.float32)
    return jnp.sum(items * weight) / jnp.maximum(jnp.sum(weight), 1.0e-8)


def gaussian_mixture_kl_upper_bound(
    mixture_logits: Any,
    means: Any,
    log_variances: Any,
    *,
    free_bits: float,
) -> Any:
    """Component-weighted upper bound against N(0,I), including weight KL."""

    import jax
    import jax.numpy as jnp

    logits = jnp.asarray(mixture_logits, dtype=jnp.float32)
    weights = jax.nn.softmax(logits, axis=-1)
    log_weights = jax.nn.log_softmax(logits, axis=-1)
    mu = jnp.asarray(means, dtype=jnp.float32)
    log_var = jnp.asarray(log_variances, dtype=jnp.float32)
    component_kl = 0.5 * jnp.sum(
        jnp.exp(log_var) + jnp.square(mu) - 1.0 - log_var, axis=-1
    )
    uniform_log = -jnp.log(jnp.asarray(weights.shape[-1], dtype=jnp.float32))
    weight_kl = jnp.sum(weights * (log_weights - uniform_log), axis=-1)
    total = jnp.sum(weights * component_kl, axis=-1) + weight_kl
    if free_bits > 0.0:
        # Free bits do not penalize posterior information below the allowance.
        total = jnp.maximum(total - float(free_bits), 0.0)
    return jnp.mean(total)


def response_prediction_loss(
    *,
    prediction: Any,
    targets: PartnerResponseTargets,
) -> Any:
    return structured_partner_response_loss(prediction, targets).total


def base_behavior_distillation_loss(
    *,
    model: Any,
    params: Any,
    batch: RolloutBatch,
    owner_logits: Any,
) -> tuple[Any, Mapping[str, Any]]:
    """Distil the independent owner-SP using only its legal policy surface."""

    import jax
    import jax.numpy as jnp

    _, output = model.apply(
        {"params": params},
        batch.initial_policy_state,
        batch.observations,
        batch.previous_actions,
        batch.previous_rewards,
        batch.episode_starts,
        jnp.zeros_like(batch.gate_overrides),
        method=model.sequence,
    )
    source = jax.nn.softmax(jnp.asarray(owner_logits, dtype=jnp.float32), axis=-1)
    candidate_log = jax.nn.log_softmax(output.base_logits[:-1], axis=-1)
    source_log = jax.nn.log_softmax(jnp.asarray(owner_logits, dtype=jnp.float32), axis=-1)
    if source.shape != candidate_log.shape:
        raise ValueError("Owner-SP logits do not align with the legal DELTA history.")
    mask = jnp.asarray(batch.ppo_mask, dtype=jnp.float32)
    pointwise = jnp.sum(source * (source_log - candidate_log), axis=-1)
    loss = jnp.sum(mask * pointwise) / jnp.maximum(jnp.sum(mask), 1.0)
    residual_rms = jnp.sqrt(jnp.mean(jnp.square(output.residual_logits[:-1])))
    return loss, {
        "base_distillation_kl": loss,
        "base_distillation_residual_rms": residual_rms,
    }


def compute_loss(
    *,
    model: Any,
    params: Mapping[str, Any],
    batch: RolloutBatch,
    config: Any,
    anchors: CounterfactualAnchorBatch | None = None,
    quotient_pairs: QuotientPairBatch | None = None,
) -> LossBundle:
    """PPO-only loss for r3.

    Raw-Q/anchors, structured response and generator PPO have independent
    optimizers and are invoked by the lifecycle in that fixed order.  Passing
    anchor data into the PPO executable is therefore a hard error.
    """

    import jax
    import jax.numpy as jnp

    if anchors is not None or quotient_pairs is not None:
        raise ValueError("r3 anchor/replay updates must not run inside a PPO minibatch.")

    unused_final, output = model.apply(
        {"params": params},
        batch.initial_policy_state,
        batch.observations,
        batch.previous_actions,
        batch.previous_rewards,
        batch.episode_starts,
        batch.gate_overrides,
        method=model.sequence,
    )
    del unused_final
    advantages, returns = generalized_advantage_estimation(
        rewards=batch.shaped_rewards,
        dones=batch.dones,
        values=output.state_value,
        gamma=config.ppo.gamma,
        gae_lambda=config.ppo.gae_lambda,
    )
    gate = jnp.asarray(batch.gate_overrides[:-1] > 0.5, dtype=jnp.float32)
    valid = jnp.asarray(batch.ppo_mask, dtype=jnp.float32)
    base_mask = valid * (1.0 - gate)
    conditional_mask = valid * gate
    # Base-role PPO owns the base actor.  Conditional-role PPO owns only the
    # residual; stop-gradient prevents conditional data from moving the base.
    conditional_training_logits = (
        jax.lax.stop_gradient(output.base_logits[:-1])
        + output.residual_logits[:-1]
    )
    base_actor, base_actor_metrics = ppo_actor_loss(
        logits=output.base_logits[:-1],
        actions=batch.actions,
        old_log_probabilities=batch.old_log_probabilities,
        advantages=advantages,
        clip_epsilon=config.ppo.clip_epsilon,
        normalize_advantages=config.ppo.normalize_advantages,
        mask=base_mask,
    )
    conditional_actor, conditional_actor_metrics = ppo_actor_loss(
        logits=conditional_training_logits,
        actions=batch.actions,
        old_log_probabilities=batch.old_log_probabilities,
        advantages=advantages,
        clip_epsilon=config.ppo.clip_epsilon,
        normalize_advantages=config.ppo.normalize_advantages,
        mask=conditional_mask,
    )
    valid_count = jnp.maximum(jnp.sum(valid), 1.0)
    base_fraction = jnp.sum(base_mask) / valid_count
    conditional_fraction = jnp.sum(conditional_mask) / valid_count
    actor = base_fraction * base_actor + conditional_fraction * conditional_actor
    value = clipped_value_loss(
        predictions=output.state_value[:-1],
        old_predictions=batch.old_values[:-1],
        targets=returns,
        clip_epsilon=config.ppo.value_clip_epsilon,
        mask=batch.ppo_mask,
    )
    base_entropy_items = categorical_entropy(output.base_logits[:-1])
    conditional_entropy_items = categorical_entropy(conditional_training_logits)
    entropy = (
        jnp.sum(base_entropy_items * base_mask)
        + jnp.sum(conditional_entropy_items * conditional_mask)
    ) / valid_count
    # Information bottleneck is optimized by the decision/representation
    # auxiliary path, never by PPO.
    ib = jax.lax.stop_gradient(gaussian_mixture_kl_upper_bound(
        output.mixture_logits[:-1],
        output.mixture_means[:-1],
        output.mixture_log_variances[:-1],
        free_bits=config.loss.information_bottleneck_free_bits,
    ))
    base_log = jax.nn.log_softmax(output.base_logits[:-1], axis=-1)
    conditional_log = jax.nn.log_softmax(conditional_training_logits, axis=-1)
    base_prob = jnp.exp(base_log)
    conditional_prob = jnp.exp(conditional_log)
    base_entropy = -jnp.sum(base_prob * base_log, axis=-1)
    conditional_entropy = -jnp.sum(conditional_prob * conditional_log, axis=-1)
    conditional_denominator = jnp.maximum(jnp.sum(conditional_mask), 1.0)
    entropy_noncollapse = conditional_entropy_noncollapse_penalty(
        output.base_logits[:-1],
        conditional_training_logits,
        conditional_mask,
        tolerance=0.1,
    )
    conditional_to_base_kl = jnp.sum(
        conditional_mask
        * jnp.sum(conditional_prob * (conditional_log - base_log), axis=-1)
    ) / conditional_denominator
    residual_mean_square = (
        jnp.sum(
            conditional_mask[..., None] * jnp.square(output.residual_logits[:-1])
        )
        / jnp.maximum(
            conditional_denominator * output.residual_logits.shape[-1], 1.0
        )
    )
    # The residual output layer is exactly zero-initialized by contract.  A
    # raw sqrt here gives an infinite derivative at update one, and the
    # inactive RMS hinge then creates NaN gradients.  Use the stable RMS
    # definition while retaining the same <= 1.0 constraint.
    residual_rms = jnp.sqrt(
        residual_mean_square + jnp.finfo(jnp.float32).tiny
    )
    kl_penalty = jnp.maximum(conditional_to_base_kl - 0.03, 0.0)
    residual_penalty = jnp.maximum(residual_rms - 1.0, 0.0)

    total = (
        actor
        + float(config.ppo.value_weight) * value
        - float(config.ppo.entropy_weight) * entropy
        + float(config.loss.conditional_entropy_weight)
        * entropy_noncollapse
        + float(config.loss.conditional_kl_weight) * kl_penalty
        + float(config.loss.residual_norm_weight)
        * residual_penalty
    )
    metrics = {
        "approx_kl": jnp.maximum(
            base_actor_metrics["approx_kl"], conditional_actor_metrics["approx_kl"]
        ),
        "base_approx_kl": base_actor_metrics["approx_kl"],
        "conditional_approx_kl": conditional_actor_metrics["approx_kl"],
        "clip_fraction": (
            base_fraction * base_actor_metrics["clip_fraction"]
            + conditional_fraction * conditional_actor_metrics["clip_fraction"]
        ),
        "actor_loss": actor,
        "total_loss": total,
        "value_loss": value,
        "entropy": entropy,
        "information_bottleneck": ib,
        "mean_raw_reward": jnp.mean(batch.rewards),
        "mean_shaped_reward": jnp.mean(batch.shaped_rewards),
        "mean_support_score": jnp.mean(output.support_score[:-1]),
        "mean_gate": jnp.mean(output.gate[:-1]),
        "base_entropy": jnp.mean(base_entropy),
        "conditional_entropy": jnp.sum(conditional_entropy * conditional_mask)
        / conditional_denominator,
        "conditional_entropy_noncollapse": entropy_noncollapse,
        "conditional_to_base_kl": conditional_to_base_kl,
        "conditional_residual_rms": residual_rms,
    }
    return LossBundle(total=total, metrics=metrics)


def official_learning_rate_schedule(
    *,
    learning_rate: float,
    warmup_fraction: float,
    update_count: int,
    minibatches_per_epoch: int,
    update_epochs: int,
) -> Any:
    """Exact warmup-plus-cosine schedule from Official ``ippo.py``."""

    import optax

    warmup_updates = int(float(warmup_fraction) * int(update_count))
    steps_per_update = int(minibatches_per_epoch) * int(update_epochs)
    warmup = optax.linear_schedule(
        init_value=0.0,
        end_value=float(learning_rate),
        transition_steps=warmup_updates * steps_per_update,
    )
    cosine_updates = max(int(update_count) - warmup_updates, 1)
    cosine = optax.cosine_decay_schedule(
        init_value=float(learning_rate),
        decay_steps=cosine_updates * steps_per_update,
    )
    return optax.join_schedules(
        schedules=(warmup, cosine),
        boundaries=(warmup_updates * steps_per_update,),
    )


def official_reward_shaping_factor(
    environment_step: int | Any,
    *,
    horizon: int,
) -> Any:
    """Exact clipped linear coefficient used by Official PPO collection."""

    import jax.numpy as jnp

    step = jnp.asarray(environment_step, dtype=jnp.float32)
    return jnp.clip(1.0 - step / float(horizon), 0.0, 1.0)


def make_optimizer(
    params: Any,
    *,
    learning_rate: float,
    gradient_clip_norm: float,
    adam_epsilon: float = 1.0e-5,
    anneal_learning_rate: bool = False,
    warmup_fraction: float = 0.0,
    update_count: int = 1,
    minibatches_per_epoch: int = 1,
    update_epochs: int = 1,
) -> tuple[Any, Any]:
    import optax

    rate = (
        official_learning_rate_schedule(
            learning_rate=learning_rate,
            warmup_fraction=warmup_fraction,
            update_count=update_count,
            minibatches_per_epoch=minibatches_per_epoch,
            update_epochs=update_epochs,
        )
        if anneal_learning_rate
        else float(learning_rate)
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(float(gradient_clip_norm)),
        optax.adam(rate, eps=float(adam_epsilon)),
    )
    return optimizer, optimizer.init(params)


def polyak_update(target: Any, online: Any, coefficient: float) -> Any:
    import jax

    tau = float(coefficient)
    return jax.tree_util.tree_map(
        lambda old, new: (1.0 - tau) * old + tau * new,
        target,
        online,
    )


def apply_training_update(
    *,
    model: Any,
    state: TrainState,
    optimizer: Any,
    batch: RolloutBatch,
    config: Any,
    anchors: CounterfactualAnchorBatch | None = None,
    quotient_pairs: QuotientPairBatch | None = None,
) -> TrainingUpdate:
    core, metrics = apply_training_core_update(
        model=model,
        core=training_core_state(state),
        optimizer=optimizer,
        batch=batch,
        config=config,
        anchors=anchors,
        quotient_pairs=quotient_pairs,
    )
    next_state = state._replace(
        params=core.params,
        target_params=core.target_params,
        ppo_optimizer_state=core.ppo_optimizer_state,
    )
    return TrainingUpdate(state=next_state, metrics=metrics)


def training_core_state(state: TrainState) -> TrainingCoreState:
    """Project serializable run state onto the compiled PPO carry."""

    return TrainingCoreState(
        params=state.params,
        target_params=state.target_params,
        ppo_optimizer_state=state.ppo_optimizer_state,
    )


def merge_training_core(
    state: TrainState,
    core: TrainingCoreState,
) -> TrainState:
    """Merge a completed compiled PPO carry without touching run-level state."""

    return state._replace(
        params=core.params,
        target_params=core.target_params,
        ppo_optimizer_state=core.ppo_optimizer_state,
    )


def apply_training_core_update(
    *,
    model: Any,
    core: TrainingCoreState,
    optimizer: Any,
    batch: RolloutBatch,
    config: Any,
    anchors: CounterfactualAnchorBatch | None = None,
    quotient_pairs: QuotientPairBatch | None = None,
    task_trunk_scale: Any = 1.0,
    base_actor_scale: Any = 1.0,
) -> tuple[TrainingCoreState, Mapping[str, Any]]:
    """Apply exactly one sequential Official PPO minibatch update.

    This is deliberately free of runner, generator, RNG, calibration, and
    accounting state so it can be used unchanged as a ``lax.scan`` body.
    """

    import jax
    import jax.numpy as jnp
    import optax

    def objective(candidate: Any) -> tuple[Any, Mapping[str, Any]]:
        loss = compute_loss(
            model=model,
            params=candidate,
            batch=batch,
            config=config,
            anchors=anchors,
            quotient_pairs=quotient_pairs,
        )
        return loss.total, loss.metrics

    (unused, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(
        core.params
    )
    del unused
    from .gradient_routing import keep_gradients_for_losses

    gradients = keep_gradients_for_losses(
        gradients,
        loss_names=("ppo", "base_ppo", "conditional_ppo", "ppo_value"),
    )
    from .gradient_routing import scale_base_policy_gradients

    gradients = scale_base_policy_gradients(
        gradients,
        task_trunk_scale=task_trunk_scale,
        base_actor_scale=base_actor_scale,
    )
    updates, optimizer_state = optimizer.update(
        gradients, core.ppo_optimizer_state, core.params
    )
    params = optax.apply_updates(core.params, updates)
    candidate = TrainingCoreState(
        params=params,
        # Target actor/belief/raw-Q are immutable inside a 16-update
        # TargetPolicyEpoch and are synchronized only by the lifecycle.
        target_params=core.target_params,
        ppo_optimizer_state=optimizer_state,
    )

    def tree_all_finite(tree: Any) -> Any:
        leaves = jax.tree_util.tree_leaves(tree)
        if not leaves:
            return jnp.asarray(True)
        return jnp.all(
            jnp.stack(
                [jnp.all(jnp.isfinite(jnp.asarray(leaf))) for leaf in leaves]
            )
        )

    finite = (
        jnp.isfinite(jnp.asarray(metrics["total_loss"]))
        & tree_all_finite(gradients)
        & tree_all_finite(candidate.params)
        & tree_all_finite(candidate.ppo_optimizer_state)
    )
    # Never commit a corrupt parameter/optimizer tree.  The scan records and
    # stops on this flag; the host lifecycle then fails with an exact phase
    # error before any auxiliary optimizer or checkpoint can consume it.
    committed = jax.lax.cond(finite, lambda _: candidate, lambda _: core, operand=None)
    return committed, {
        **metrics,
        "optimizer_applied": finite.astype(jnp.float32),
        "nonfinite_update": (~finite).astype(jnp.float32),
    }


def scan_training_updates(
    *,
    model: Any,
    core: TrainingCoreState,
    optimizer: Any,
    batch: RolloutBatch,
    schedule: Any,
    config: Any,
    task_trunk_scale: Any = 1.0,
    base_actor_scale: Any = 1.0,
) -> tuple[TrainingCoreState, Mapping[str, Any]]:
    """Run minibatches sequentially on device without duplicating the rollout."""

    import jax
    import jax.numpy as jnp

    indexes = jnp.asarray(schedule, dtype=jnp.int32)
    if indexes.ndim < 2:
        raise ValueError("The PPO schedule must contain minibatch and lane axes.")
    flat = indexes.reshape((-1, indexes.shape[-1]))

    metric_names = (
        "approx_kl",
        "base_approx_kl",
        "conditional_approx_kl",
        "clip_fraction",
        "actor_loss",
        "total_loss",
        "value_loss",
        "entropy",
        "information_bottleneck",
        "mean_raw_reward",
        "mean_shaped_reward",
        "mean_support_score",
        "mean_gate",
        "base_entropy",
        "conditional_entropy",
        "conditional_entropy_noncollapse",
        "conditional_to_base_kl",
        "conditional_residual_rms",
        "nonfinite_update",
    )

    def skipped(current: TrainingCoreState) -> tuple[TrainingCoreState, Mapping[str, Any]]:
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        return current, {
            **{name: zero for name in metric_names},
            "optimizer_applied": zero,
            "nonfinite_update": zero,
            "ppo_early_stop": jnp.asarray(1.0, dtype=jnp.float32),
        }

    def applied(
        current: TrainingCoreState,
        lane_indexes: Any,
    ) -> tuple[TrainingCoreState, Mapping[str, Any]]:
        updated, metrics = apply_training_core_update(
            model=model,
            core=current,
            optimizer=optimizer,
            batch=slice_rollout_lanes(batch, lane_indexes),
            config=config,
            task_trunk_scale=task_trunk_scale,
            base_actor_scale=base_actor_scale,
        )
        stop = (
            (metrics["nonfinite_update"] > 0.5)
            |
            (metrics["base_approx_kl"] > float(config.ppo.max_approx_kl))
            | (
                metrics["conditional_approx_kl"]
                > float(config.ppo.max_approx_kl)
            )
        )
        return updated, {
            **metrics,
            "ppo_early_stop": stop.astype(jnp.float32),
        }

    def one(
        carry: tuple[TrainingCoreState, Any],
        lane_indexes: Any,
    ) -> tuple[tuple[TrainingCoreState, Any], Mapping[str, Any]]:
        current, active = carry
        updated, metrics = jax.lax.cond(
            active,
            lambda value: applied(value, lane_indexes),
            skipped,
            current,
        )
        still_active = active & (metrics["ppo_early_stop"] < 0.5)
        return (updated, still_active), metrics

    (final_core, unused_active), metrics = jax.lax.scan(
        one, (core, jnp.asarray(True)), flat
    )
    del unused_active
    return final_core, metrics


def slice_rollout_lanes(batch: RolloutBatch, indexes: Any) -> RolloutBatch:
    """Slice environment lanes while preserving the time axis."""

    import jax

    values = {}
    for name, value in batch._asdict().items():
        if name in {"initial_policy_state", "initial_target_policy_state"}:
            values[name] = jax.tree_util.tree_map(lambda leaf: leaf[indexes], value)
        else:
            values[name] = value[:, indexes]
    return RolloutBatch(**values)


def environment_minibatch_schedule(
    key: Any,
    *,
    environment_count: int,
    minibatches_per_epoch: int,
    update_epochs: int,
) -> Any:
    import jax
    import jax.numpy as jnp

    if environment_count % minibatches_per_epoch:
        raise ValueError("Environment lanes must divide exactly into minibatches.")
    lane_count = environment_count // minibatches_per_epoch
    keys = jax.random.split(key, update_epochs)
    return jnp.stack(
        [
            jax.random.permutation(epoch_key, environment_count).reshape(
                (minibatches_per_epoch, lane_count)
            )
            for epoch_key in keys
        ]
    )


__all__ = [
    "apply_training_update",
    "apply_training_core_update",
    "categorical_entropy",
    "conditional_entropy_noncollapse_penalty",
    "stable_root_mean_square",
    "categorical_log_probability",
    "clipped_value_loss",
    "compute_loss",
    "environment_minibatch_schedule",
    "gather_actions",
    "gaussian_mixture_kl_upper_bound",
    "generalized_advantage_estimation",
    "huber",
    "make_optimizer",
    "merge_training_core",
    "official_learning_rate_schedule",
    "official_reward_shaping_factor",
    "polyak_update",
    "ppo_actor_loss",
    "response_prediction_loss",
    "scan_training_updates",
    "slice_rollout_lanes",
    "training_core_state",
]
