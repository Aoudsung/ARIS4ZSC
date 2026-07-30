"""Recurrent PPO and decision-equivalent auxiliary training for DELTA-ZSC."""

from __future__ import annotations

from typing import Any, Mapping

from .decision_geometry import (
    centered_action_values,
    quotient_geometry_loss,
)
from .response_decoder import (
    bernoulli_logit_loss,
    gaussian_negative_log_likelihood,
)
from .types import (
    CounterfactualAnchorBatch,
    LossBundle,
    QuotientPairBatch,
    RolloutBatch,
    TrainState,
    TrainingUpdate,
)


def gather_actions(values: Any, actions: Any, *, action_axis: int = -1) -> Any:
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
    output: Any,
    observations: Any,
    response_next_observations: Any,
    actions: Any,
    rewards: Any,
    dones: Any,
) -> Any:
    import jax.numpy as jnp

    action = jnp.asarray(actions, dtype=jnp.int32)
    # Explicit action axis is immediately before the observation axes.
    action_axis = output.response_observation_delta_mean[:-1].ndim - len(observations.shape[2:]) - 1
    selected_delta_mean = gather_actions(
        output.response_observation_delta_mean[:-1], action, action_axis=action_axis
    )
    selected_delta_log_std = gather_actions(
        output.response_observation_delta_log_std[:-1], action, action_axis=action_axis
    )
    target_delta = jnp.asarray(
        response_next_observations - observations[:-1], dtype=jnp.float32
    )
    observation_nll = gaussian_negative_log_likelihood(
        target_delta, selected_delta_mean, selected_delta_log_std
    )
    selected_reward_mean = gather_actions(output.response_reward_mean[:-1], action, action_axis=-1)
    selected_reward_log_std = gather_actions(
        output.response_reward_log_std[:-1], action, action_axis=-1
    )
    selected_done_logit = gather_actions(output.response_done_logit[:-1], action, action_axis=-1)
    reward_nll = gaussian_negative_log_likelihood(
        rewards, selected_reward_mean, selected_reward_log_std
    )
    done_nll = bernoulli_logit_loss(dones, selected_done_logit)
    obs_axes = tuple(range(observation_nll.ndim - len(observations.shape[2:]), observation_nll.ndim))
    observation_item = jnp.mean(observation_nll, axis=obs_axes) if obs_axes else observation_nll
    return jnp.mean(observation_item + reward_nll + done_nll)


def compute_teacher_latents(
    *,
    model: Any,
    params: Mapping[str, Any],
    batch: RolloutBatch,
    task_features: Any,
) -> Any:
    """Recompute privileged contexts under the candidate parameters.

    Generator and snapshot lanes use the training-only code teacher.  Frozen
    external lanes use the bidirectional full-trajectory teacher.  The returned
    value is not a partner identity label; it is optimized only through real
    return/action-value objectives and is detached when used as a posterior
    target.
    """

    import jax.numpy as jnp

    code_teacher = model.apply(
        {"params": params},
        batch.partner_codes,
        task_features,
        1.0,
        method=model.teacher_from_code,
    ).latent
    full_teacher = model.apply(
        {"params": params},
        batch.observations,
        batch.response_next_observations,
        batch.actions,
        batch.rewards,
        batch.dones,
        method=model.full_trajectory_latents,
    )
    external = jnp.asarray(batch.partner_sources, dtype=jnp.int32) == 2
    return jnp.where(external[..., None], full_teacher, code_teacher)


def _gather_flat_time_lanes(values: Any, indexes: Any) -> Any:
    import jax
    import jax.numpy as jnp

    index = jnp.asarray(indexes, dtype=jnp.int32)

    def one(value: Any) -> Any:
        array = jnp.asarray(value)
        flat = array.reshape((array.shape[0] * array.shape[1],) + array.shape[2:])
        return flat[index]

    return jax.tree_util.tree_map(one, values)


def counterfactual_anchor_loss(
    *,
    model: Any,
    params: Mapping[str, Any],
    teacher_latents: Any,
    anchors: CounterfactualAnchorBatch,
) -> tuple[Any, Any, Mapping[str, Any]]:
    """Real-return anchor loss for both online and privileged contexts.

    Re-running ``model.step`` from the stored legal policy state gives gradients
    to the task encoder, belief encoder, and critic.  The privileged branch gives
    gradients to the code/full-trajectory teacher and the same shared critic.
    """

    import jax
    import jax.numpy as jnp

    unused_state, online = model.apply(
        {"params": params},
        anchors.policy_states,
        anchors.observations,
        jnp.ones(anchors.partner_sources.shape, dtype=jnp.float32),
        method=model.step,
    )
    del unused_state
    flat_teacher = _gather_flat_time_lanes(
        teacher_latents, anchors.rollout_flat_indexes
    )
    # Intervention anchors may use codes that do not occur in the original
    # rollout.  Recompute those contexts from their own code and online task
    # feature; ordinary external anchors retain the full-trajectory teacher.
    code_teacher = model.apply(
        {"params": params},
        anchors.partner_codes,
        online.task_features,
        1.0,
        method=model.teacher_from_code,
    ).latent
    external = jnp.asarray(anchors.partner_sources, dtype=jnp.int32) == 2
    anchor_teacher_latent = jnp.where(
        external[..., None], flat_teacher, code_teacher
    )
    teacher = model.apply(
        {"params": params},
        online.task_features,
        anchor_teacher_latent,
        1.0,
        method=model.from_features_and_latent,
    )
    target = jax.lax.stop_gradient(
        centered_action_values(anchors.fit_returns_by_action)
    )
    mask = jnp.asarray(anchors.action_mask, dtype=jnp.float32)
    online_predicted = centered_action_values(online.action_values)
    teacher_predicted = centered_action_values(teacher.action_values)
    online_items = huber(online_predicted - target) * mask
    teacher_items = huber(teacher_predicted - target) * mask
    denominator = jnp.maximum(jnp.sum(mask), 1.0e-8)
    online_loss = jnp.sum(online_items) / denominator
    teacher_loss = jnp.sum(teacher_items) / denominator
    loss = online_loss + teacher_loss

    selected = jnp.argmax(online_predicted, axis=-1)
    fit_selected = gather_actions(anchors.fit_returns_by_action, selected)
    evaluation_selected = gather_actions(
        anchors.evaluation_returns_by_action, selected
    )
    oracle = jnp.argmax(anchors.fit_returns_by_action, axis=-1)
    evaluation_oracle = gather_actions(
        anchors.evaluation_returns_by_action, oracle
    )
    return loss, anchor_teacher_latent, {
        "counterfactual_loss": loss,
        "counterfactual_online_loss": online_loss,
        "counterfactual_teacher_loss": teacher_loss,
        "counterfactual_selected_eval_return": jnp.mean(evaluation_selected),
        "counterfactual_oracle_eval_return": jnp.mean(evaluation_oracle),
        "counterfactual_selection_regret": jnp.mean(
            evaluation_oracle - evaluation_selected
        ),
        "counterfactual_fit_eval_gap": jnp.mean(
            fit_selected - evaluation_selected
        ),
    }

def teacher_student_losses(
    *,
    model: Any,
    params: Mapping[str, Any],
    task_features: Any,
    student_logits: Any,
    student_action_values: Any,
    teacher_latents: Any,
) -> tuple[Any, Any, Mapping[str, Any]]:
    import jax
    import jax.numpy as jnp

    teacher_latent = jnp.asarray(teacher_latents, dtype=jnp.float32)
    valid = jnp.all(jnp.isfinite(teacher_latent), axis=-1)
    safe = jnp.where(valid[..., None], teacher_latent, 0.0)
    teacher = model.apply(
        {"params": params},
        task_features,
        safe,
        1.0,
        method=model.from_features_and_latent,
    )
    teacher_q = centered_action_values(teacher.action_values)
    student_q = centered_action_values(student_action_values)
    advantage_item = jnp.mean(jnp.square(student_q - jax.lax.stop_gradient(teacher_q)), axis=-1)
    teacher_log = jax.lax.stop_gradient(
        jax.nn.log_softmax(teacher.logits, axis=-1)
    )
    student_log = jax.nn.log_softmax(student_logits, axis=-1)
    teacher_prob = jax.lax.stop_gradient(jnp.exp(teacher_log))
    teacher_best = jnp.max(teacher.action_values, axis=-1)
    sorted_q = jnp.sort(teacher.action_values, axis=-1)
    margin = jax.lax.stop_gradient(jnp.maximum(teacher_best - sorted_q[..., -2], 0.0))
    policy_item = margin * jnp.sum(teacher_prob * (teacher_log - student_log), axis=-1)
    denominator = jnp.sum(valid.astype(jnp.float32)) + 1.0e-8
    advantage = jnp.sum(advantage_item * valid) / denominator
    policy = jnp.sum(policy_item * valid) / denominator
    return advantage, policy, {
        "advantage_distill_loss": advantage,
        "policy_distill_loss": policy,
        "teacher_valid_fraction": jnp.mean(valid.astype(jnp.float32)),
        "teacher_action_margin": jnp.sum(margin * valid) / denominator,
    }


def posterior_consistency_loss(
    *,
    mixture_logits: Any,
    means: Any,
    log_variances: Any,
    teacher_latents: Any,
) -> Any:
    """Negative mixture log-likelihood of a full-information teacher context."""

    import jax
    import jax.numpy as jnp

    target = jax.lax.stop_gradient(
        jnp.asarray(teacher_latents, dtype=jnp.float32)
    )
    valid = jnp.all(jnp.isfinite(target), axis=-1)
    safe = jnp.where(valid[..., None], target, 0.0)
    log_weights = jax.nn.log_softmax(mixture_logits, axis=-1)
    log_var = jnp.asarray(log_variances, dtype=jnp.float32)
    difference = safe[..., None, :] - jnp.asarray(means, dtype=jnp.float32)
    component_log_prob = -0.5 * jnp.sum(
        jnp.square(difference) * jnp.exp(-log_var)
        + log_var
        + jnp.log(2.0 * jnp.pi),
        axis=-1,
    )
    log_prob = jax.scipy.special.logsumexp(log_weights + component_log_prob, axis=-1)
    return -jnp.sum(log_prob * valid) / (jnp.sum(valid) + 1.0e-8)


def compute_loss(
    *,
    model: Any,
    params: Mapping[str, Any],
    batch: RolloutBatch,
    config: Any,
    anchors: CounterfactualAnchorBatch | None = None,
    quotient_pairs: QuotientPairBatch | None = None,
) -> LossBundle:
    import jax.numpy as jnp

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
    actor, actor_metrics = ppo_actor_loss(
        logits=output.execution_logits[:-1],
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
    entropy_items = categorical_entropy(output.execution_logits[:-1])
    entropy = jnp.sum(entropy_items * batch.ppo_mask) / jnp.maximum(
        jnp.sum(batch.ppo_mask), 1.0e-8
    )
    response = response_prediction_loss(
        output=output,
        observations=batch.observations,
        response_next_observations=batch.response_next_observations,
        actions=batch.actions,
        rewards=batch.rewards,
        dones=batch.dones,
    )
    ib = gaussian_mixture_kl_upper_bound(
        output.mixture_logits[:-1],
        output.mixture_means[:-1],
        output.mixture_log_variances[:-1],
        free_bits=config.loss.information_bottleneck_free_bits,
    )
    teacher_latents = compute_teacher_latents(
        model=model,
        params=params,
        batch=batch,
        task_features=output.task_features[:-1],
    )
    advantage_distill, policy_distill, distill_metrics = teacher_student_losses(
        model=model,
        params=params,
        task_features=output.task_features[:-1],
        student_logits=output.execution_logits[:-1],
        student_action_values=output.action_values[:-1],
        teacher_latents=teacher_latents,
    )
    consistency = posterior_consistency_loss(
        mixture_logits=output.mixture_logits[:-1],
        means=output.mixture_means[:-1],
        log_variances=output.mixture_log_variances[:-1],
        teacher_latents=teacher_latents,
    )

    cf = jnp.asarray(0.0, dtype=jnp.float32)
    cf_metrics: Mapping[str, Any] = {
        "counterfactual_loss": jnp.asarray(0.0, dtype=jnp.float32),
        "counterfactual_online_loss": jnp.asarray(0.0, dtype=jnp.float32),
        "counterfactual_teacher_loss": jnp.asarray(0.0, dtype=jnp.float32),
        "counterfactual_selected_eval_return": jnp.asarray(0.0, dtype=jnp.float32),
        "counterfactual_oracle_eval_return": jnp.asarray(0.0, dtype=jnp.float32),
        "counterfactual_selection_regret": jnp.asarray(0.0, dtype=jnp.float32),
        "counterfactual_fit_eval_gap": jnp.asarray(0.0, dtype=jnp.float32),
    }
    if anchors is not None:
        cf, anchor_teacher_latents, cf_metrics = counterfactual_anchor_loss(
            model=model,
            params=params,
            teacher_latents=teacher_latents,
            anchors=anchors,
        )
    else:
        anchor_teacher_latents = None

    quotient = jnp.asarray(0.0, dtype=jnp.float32)
    if quotient_pairs is not None:
        if anchor_teacher_latents is None:
            raise ValueError("Quotient pairs require counterfactual anchors.")
        quotient = quotient_geometry_loss(
            anchor_teacher_latents[quotient_pairs.anchor_index_a],
            anchor_teacher_latents[quotient_pairs.anchor_index_b],
            quotient_pairs.decision_distance,
            equivalence_epsilon=config.loss.quotient_equivalence_epsilon,
            separation_epsilon=config.loss.quotient_separation_epsilon,
            margin=config.loss.quotient_margin,
            weights=quotient_pairs.weights,
        )

    total = (
        actor
        + float(config.ppo.value_weight) * value
        - float(config.ppo.entropy_weight) * entropy
        + float(config.loss.response_weight) * response
        + float(config.loss.counterfactual_weight) * cf
        + float(config.loss.advantage_distill_weight) * advantage_distill
        + float(config.loss.policy_distill_weight) * policy_distill
        + float(config.loss.quotient_weight) * quotient
        + float(config.loss.consistency_weight) * consistency
        + float(config.loss.information_bottleneck_weight) * ib
    )
    metrics = {
        **actor_metrics,
        **distill_metrics,
        **cf_metrics,
        "total_loss": total,
        "value_loss": value,
        "entropy": entropy,
        "response_loss": response,
        "information_bottleneck": ib,
        "posterior_consistency": consistency,
        "quotient_loss": quotient,
        "mean_raw_reward": jnp.mean(batch.rewards),
        "mean_shaped_reward": jnp.mean(batch.shaped_rewards),
        "mean_support_score": jnp.mean(output.support_score[:-1]),
        "mean_gate": jnp.mean(output.gate[:-1]),
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
    import jax
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
        state.params
    )
    del unused
    updates, optimizer_state = optimizer.update(
        gradients, state.optimizer_state, state.params
    )
    params = optax.apply_updates(state.params, updates)
    next_state = state._replace(
        params=params,
        target_params=polyak_update(
            state.target_params, params, config.ppo.polyak_coefficient
        ),
        optimizer_state=optimizer_state,
    )
    return TrainingUpdate(state=next_state, metrics=metrics)


def update_competence_multiplier(
    multiplier: Any,
    *,
    competence: Any,
    threshold: float,
    learning_rate: float,
) -> Any:
    import jax.numpy as jnp

    return jnp.maximum(
        0.0,
        jnp.asarray(multiplier)
        + float(learning_rate) * (float(threshold) - jnp.asarray(competence)),
    )



def generator_score_function_loss(
    *,
    logits: Any,
    values: Any,
    actions: Any,
    source_mask: Any,
    rewards: Any,
    dones: Any,
    diversity_bonus: Any,
    value_weight: float,
    entropy_weight: float,
    competence_multiplier: Any,
    competence_threshold: float,
    cvar_level: float,
) -> LossBundle:
    """Score-function generator objective with a real CVaR gradient path.

    The generator maximizes empirical best-response diversity subject to a
    lower-tail team-return constraint.  The environment is non-differentiable,
    so both terms enter through the complete-trajectory log probability.  A
    shared recurrent value head supplies a Monte-Carlo baseline; no target Q or
    partner-class label is used.
    """

    import jax
    import jax.numpy as jnp

    log_probability = categorical_log_probability(logits, actions)
    entropy = categorical_entropy(logits)
    mask = jnp.asarray(source_mask, dtype=jnp.float32)
    reward = jnp.asarray(rewards, dtype=jnp.float32)
    done = jnp.asarray(dones, dtype=jnp.bool_)
    predicted_value = jnp.asarray(values, dtype=jnp.float32)
    if not (
        log_probability.shape
        == entropy.shape
        == mask.shape
        == reward.shape
        == done.shape
        == predicted_value.shape
    ):
        raise ValueError("Generator trajectory tensors must share [time,lane].")

    active_lane = (jnp.sum(mask, axis=0) > 0).astype(jnp.float32)
    active_count = jnp.maximum(jnp.sum(active_lane).astype(jnp.int32), 1)
    episode_return = jnp.sum(reward * mask, axis=0)

    # Empirical lower-tail CVaR and its score-function weights.  Ties are broken
    # by the fixed lane order, which is independent of the candidate parameters.
    tail_count = jnp.maximum(
        jnp.ceil(float(cvar_level) * active_count).astype(jnp.int32), 1
    )
    order = jnp.argsort(jnp.where(active_lane > 0, episode_return, jnp.inf))
    rank_weight = (
        jnp.arange(episode_return.shape[0], dtype=jnp.int32) < tail_count
    ).astype(jnp.float32)
    tail_indicator = jnp.zeros_like(active_lane).at[order].set(rank_weight)
    tail_indicator = tail_indicator * active_lane
    competence = jnp.sum(episode_return * tail_indicator) / jnp.maximum(
        jnp.sum(tail_indicator), 1.0e-8
    )
    constraint = float(competence_threshold) - competence

    bonus = jax.lax.stop_gradient(
        jnp.asarray(diversity_bonus, dtype=jnp.float32)
    )
    if bonus.shape != episode_return.shape:
        raise ValueError("Generator diversity bonus must have one value per lane.")
    # CVaR is an average over the selected tail.  Rescale lane contributions so
    # the score-function estimate targets that average rather than its sum.
    cvar_weight = tail_indicator * (
        jnp.asarray(active_count, dtype=jnp.float32)
        / jnp.maximum(jnp.sum(tail_indicator), 1.0e-8)
    )
    lane_objective = jax.lax.stop_gradient(
        bonus
        + jnp.asarray(competence_multiplier, dtype=jnp.float32)
        * cvar_weight
        * episode_return
    )

    # Every action in a code-conditioned trajectory affects the terminal
    # diversity/competence objective.  A learned recurrent baseline is trained
    # against the same lane objective and detached in the policy gradient.
    advantage = lane_objective[None, :] - jax.lax.stop_gradient(predicted_value)
    policy_loss = -jnp.sum(log_probability * advantage * mask) / jnp.maximum(
        jnp.sum(mask), 1.0e-8
    )
    value_loss = jnp.sum(
        huber(predicted_value - lane_objective[None, :]) * mask
    ) / jnp.maximum(jnp.sum(mask), 1.0e-8)
    entropy_mean = jnp.sum(entropy * mask) / jnp.maximum(
        jnp.sum(mask), 1.0e-8
    )
    # The explicit Lagrangian constant has no parameter gradient but is retained
    # so the reported scalar is the registered constrained objective.
    total = (
        policy_loss
        + float(value_weight) * value_loss
        - float(entropy_weight) * entropy_mean
        + jax.lax.stop_gradient(
            jnp.asarray(competence_multiplier, dtype=jnp.float32) * constraint
        )
    )
    return LossBundle(
        total=total,
        metrics={
            "generator_total_loss": total,
            "generator_policy_loss": policy_loss,
            "generator_value_loss": value_loss,
            "generator_entropy": entropy_mean,
            "generator_competence_cvar": competence,
            "generator_competence_constraint": constraint,
            "generator_diversity_bonus": jnp.sum(bonus * active_lane)
            / jnp.maximum(jnp.sum(active_lane), 1.0e-8),
            "generator_active_lanes": jnp.sum(active_lane),
            "generator_tail_lanes": jnp.sum(tail_indicator),
        },
    )


def slice_rollout_lanes(batch: RolloutBatch, indexes: Any) -> RolloutBatch:
    """Slice environment lanes while preserving the time axis."""

    import jax

    values = {}
    for name, value in batch._asdict().items():
        if name == "initial_policy_state":
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
    "categorical_entropy",
    "categorical_log_probability",
    "clipped_value_loss",
    "compute_loss",
    "compute_teacher_latents",
    "counterfactual_anchor_loss",
    "environment_minibatch_schedule",
    "gather_actions",
    "gaussian_mixture_kl_upper_bound",
    "generalized_advantage_estimation",
    "generator_score_function_loss",
    "huber",
    "make_optimizer",
    "official_learning_rate_schedule",
    "official_reward_shaping_factor",
    "polyak_update",
    "posterior_consistency_loss",
    "ppo_actor_loss",
    "response_prediction_loss",
    "slice_rollout_lanes",
    "teacher_student_losses",
    "update_competence_multiplier",
]
