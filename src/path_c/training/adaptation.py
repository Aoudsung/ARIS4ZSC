"""Proximal policy optimization and online auxiliary adaptation losses."""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple

from .prefit import auxiliary_losses


class AdaptationLosses(NamedTuple):
    actor: Any
    critic: Any
    entropy: Any
    auxiliary: Any
    total: Any


def generalized_advantage_estimate(
    *, rewards: Any, values: Any, next_values: Any, dones: Any, gamma: float, gae_lambda: float
) -> tuple[Any, Any]:
    """Compute generalized advantage estimates backwards over one segment."""

    import jax
    import jax.numpy as jnp

    rewards = jnp.asarray(rewards, dtype=jnp.float32)
    values = jnp.asarray(values, dtype=jnp.float32)
    next_values = jnp.asarray(next_values, dtype=jnp.float32)
    dones = jnp.asarray(dones, dtype=jnp.float32)
    if rewards.shape != values.shape or rewards.shape != next_values.shape or rewards.shape != dones.shape:
        raise ValueError("Reward, value, next value, and done arrays must share [time, batch] shape.")

    def backward(carry: Any, values_at_step: tuple[Any, ...]) -> tuple[Any, Any]:
        reward, value, next_value, done = values_at_step
        nonterminal = 1.0 - done
        delta = reward + float(gamma) * nonterminal * next_value - value
        advantage = delta + float(gamma) * float(gae_lambda) * nonterminal * carry
        return advantage, advantage

    _, reversed_advantage = jax.lax.scan(
        backward,
        jnp.zeros_like(rewards[-1]),
        (rewards[::-1], values[::-1], next_values[::-1], dones[::-1]),
    )
    advantage = reversed_advantage[::-1]
    return advantage, advantage + values


def clipped_actor_loss(
    *,
    logits: Any,
    actions: Any,
    old_log_probabilities: Any,
    advantages: Any,
    actor_owned_action: Any,
    clip_epsilon: float,
    entropy_coefficient: float,
) -> tuple[Any, Any]:
    """Mask controller-overridden steps out of both actor terms."""

    import jax
    import jax.numpy as jnp

    log_probability = jax.nn.log_softmax(logits, axis=-1)
    selected = jnp.take_along_axis(
        log_probability, jnp.asarray(actions, dtype=jnp.int32)[..., None], axis=-1
    )[..., 0]
    ratio = jnp.exp(selected - old_log_probabilities)
    mask = jnp.asarray(actor_owned_action, dtype=jnp.float32)
    owned_count = jnp.sum(mask)
    denominator = jnp.maximum(owned_count, 1.0)
    masked_mean = jnp.sum(advantages * mask) / denominator
    masked_variance = jnp.sum(jnp.square(advantages - masked_mean) * mask) / denominator
    normalized_advantage = jnp.where(
        owned_count > 0,
        (advantages - masked_mean) / (jnp.sqrt(masked_variance) + 1e-8),
        jnp.zeros_like(advantages),
    )
    unclipped = ratio * normalized_advantage
    clipped = jnp.clip(ratio, 1.0 - float(clip_epsilon), 1.0 + float(clip_epsilon)) * normalized_advantage
    policy_loss = -jnp.sum(jnp.minimum(unclipped, clipped) * mask) / denominator
    probabilities = jax.nn.softmax(logits, axis=-1)
    entropy_per_step = -jnp.sum(probabilities * log_probability, axis=-1)
    entropy = jnp.sum(entropy_per_step * mask) / denominator
    return policy_loss - float(entropy_coefficient) * entropy, entropy


def adaptation_loss(
    outputs: Mapping[str, Any],
    batch: Any,
    *,
    gamma: float,
    clip_epsilon: float,
    entropy_coefficient: float,
    value_coefficient: float,
    auxiliary_loss_weights: Mapping[str, float],
) -> tuple[Any, Mapping[str, Any]]:
    """Combine fixed-rollout actor and critic targets with auxiliary objectives."""

    import jax
    import jax.numpy as jnp

    advantages = jax.lax.stop_gradient(jnp.asarray(batch.advantages, dtype=jnp.float32))
    returns = jax.lax.stop_gradient(jnp.asarray(batch.returns, dtype=jnp.float32))
    if advantages.shape != batch.actions.shape or returns.shape != batch.actions.shape:
        raise ValueError("Fixed advantages and returns must share the action [time, batch] shape.")
    actor, entropy = clipped_actor_loss(
        logits=outputs["actor_logits"],
        actions=batch.actions,
        old_log_probabilities=batch.old_log_probabilities,
        advantages=advantages,
        actor_owned_action=batch.actor_owned_action,
        clip_epsilon=clip_epsilon,
        entropy_coefficient=entropy_coefficient,
    )
    critic = 0.5 * jnp.mean(jnp.square(outputs["shared_value"] - returns))
    auxiliary = auxiliary_losses(
        outputs, batch, gamma=gamma, loss_weights=auxiliary_loss_weights
    )
    total = actor + float(value_coefficient) * critic + auxiliary.total
    losses = AdaptationLosses(actor, critic, entropy, auxiliary.total, total)
    metrics = {**losses._asdict(), **{f"auxiliary_{key}": value for key, value in auxiliary._asdict().items()}}
    return total, metrics


def linear_warmup_decay_schedule(
    *, peak_value: float, total_updates: int, warmup_fraction: float
) -> Any:
    """Warm for the first fraction, then linearly decay to zero."""

    import optax

    if total_updates <= 0 or not 0.0 < float(warmup_fraction) < 1.0:
        raise ValueError("Schedule length must be positive and warmup fraction must lie in (0, 1).")
    warmup = max(1, int(total_updates * float(warmup_fraction)))
    decay = max(1, total_updates - warmup)
    return optax.join_schedules(
        (
            optax.linear_schedule(0.0, float(peak_value), warmup),
            optax.linear_schedule(float(peak_value), 0.0, decay),
        ),
        boundaries=(warmup,),
    )


def adaptation_parameter_labels(params: Mapping[str, Any]) -> Mapping[str, Any]:
    """Assign the backbone and critic, actor, and auxiliary learning rates."""

    import jax

    labels: dict[str, Any] = {}
    for top_name, subtree in params.items():
        name = str(top_name)
        if name == "backbone":
            labels[name] = jax.tree_util.tree_map(lambda unused: "critic", subtree)
        elif name == "shared_heads":
            labels[name] = {
                str(child_name): jax.tree_util.tree_map(
                    lambda unused, label=("actor" if str(child_name).startswith("actor_") else "critic"): label,
                    child,
                )
                for child_name, child in subtree.items()
            }
        else:
            labels[name] = jax.tree_util.tree_map(lambda unused: "auxiliary", subtree)
    return labels


def make_adaptation_optimizer(
    params: Mapping[str, Any],
    *,
    critic_learning_rate: float,
    actor_learning_rate: float,
    auxiliary_learning_rate: float,
    gradient_clip_norm: float,
    total_updates: int,
    warmup_fraction: float,
) -> tuple[Any, Any]:
    import optax

    def transform(rate: float) -> Any:
        schedule = linear_warmup_decay_schedule(
            peak_value=rate,
            total_updates=total_updates,
            warmup_fraction=warmup_fraction,
        )
        return optax.adam(schedule)

    optimizer = optax.chain(
        optax.clip_by_global_norm(float(gradient_clip_norm)),
        optax.multi_transform(
            {
                "critic": transform(critic_learning_rate),
                "actor": transform(actor_learning_rate),
                "auxiliary": transform(auxiliary_learning_rate),
            },
            adaptation_parameter_labels(params),
        ),
    )
    return optimizer, optimizer.init(params)


def adaptation_update(
    *, params: Mapping[str, Any], optimizer_state: Any, optimizer: Any, loss_function: Any
) -> tuple[Mapping[str, Any], Any, Mapping[str, Any]]:
    import jax
    import jax.numpy as jnp
    import optax

    (loss, metrics), gradients = jax.value_and_grad(loss_function, has_aux=True)(params)
    updates, optimizer_state = optimizer.update(gradients, optimizer_state, params)
    next_params = optax.apply_updates(params, updates)
    finite = jnp.all(
        jnp.stack(
            [jnp.all(jnp.isfinite(value)) for value in jax.tree_util.tree_leaves(next_params)]
        )
    )
    return next_params, optimizer_state, {
        "total": loss,
        "gradient_norm": optax.global_norm(gradients),
        "parameters_finite": finite,
        **metrics,
    }


def environment_minibatch(batch: Any, indices: Any) -> Any:
    """Select complete recurrent environment lanes without breaking time order."""

    import jax
    import jax.numpy as jnp

    environment_count = int(batch.actions.shape[1])
    selected = jnp.asarray(indices, dtype=jnp.int32)
    if (
        selected.ndim != 1
        or selected.shape[0] <= 0
        or selected.shape[0] > environment_count
    ):
        raise ValueError("Environment minibatch indices must be a nonempty vector.")

    def slice_leaf(value: Any) -> Any:
        array = jnp.asarray(value)
        if array.ndim >= 2 and array.shape[:2] == batch.actions.shape:
            return array[:, selected, ...]
        return array

    return jax.tree_util.tree_map(slice_leaf, batch)


def environment_minibatches(batch: Any, permutation: Any, num_minibatches: int) -> tuple[Any, ...]:
    """Materialize recurrent minibatches for small tests and diagnostic callers."""

    import jax.numpy as jnp

    environment_count = int(batch.actions.shape[1])
    if num_minibatches <= 0 or environment_count % num_minibatches:
        raise ValueError("Environment lanes must divide evenly into recurrent minibatches.")
    order = jnp.asarray(permutation, dtype=jnp.int32)
    if order.shape != (environment_count,):
        raise ValueError("The environment permutation has the wrong shape.")
    groups = order.reshape((num_minibatches, environment_count // num_minibatches))

    return tuple(
        environment_minibatch(batch, groups[index]) for index in range(num_minibatches)
    )
