"""Clipped PPO losses for complete episodic replay."""

from __future__ import annotations

from typing import Any

from .types import LossResult


def categorical_log_probability(logits: Any, actions: Any) -> Any:
    """Return the log probability of each selected categorical action."""

    import jax.nn as jnn
    import jax.numpy as jnp

    logp = jnn.log_softmax(logits, axis=-1)
    index = jnp.asarray(actions, dtype=jnp.int32)[..., None]
    return jnp.take_along_axis(logp, index, axis=-1)[..., 0]


def categorical_entropy(logits: Any) -> Any:
    """Return categorical entropy for the final logits axis."""

    import jax.nn as jnn
    import jax.numpy as jnp

    logp = jnn.log_softmax(logits, axis=-1)
    return -jnp.sum(jnp.exp(logp) * logp, axis=-1)


def return_to_go(rewards: Any) -> Any:
    """Compute undiscounted return-to-go for complete, single-episode lanes.

    Each lane is exactly one complete episode, so no cross-boundary bootstrap
    is present in the reverse cumulative sum.
    """

    import jax.numpy as jnp

    values = jnp.asarray(rewards, dtype=jnp.float32)
    return jnp.flip(jnp.cumsum(jnp.flip(values, axis=0), axis=0), axis=0)


def joint_normalize_advantages(
    advantages: Any,
    mask: Any,
    epsilon: float = 1.0e-8,
) -> Any:
    """Normalize all selected advantages with one masked mean and deviation."""

    import jax.numpy as jnp

    values = jnp.asarray(advantages, dtype=jnp.float32)
    weights = jnp.asarray(mask, dtype=jnp.float32)
    count = jnp.maximum(jnp.sum(weights), jnp.asarray(1.0, dtype=values.dtype))
    mean = jnp.sum(weights * values) / count
    variance = jnp.sum(weights * jnp.square(values - mean)) / count
    return (values - mean) / (jnp.sqrt(variance) + float(epsilon))


def cetr_ppo_loss(
    model: Any,
    params: Any,
    batch: Any,
    dual_lambda: Any,
) -> LossResult:
    """Evaluate the shared-normalization PPO objective.

    The external actor term is an ordinary average over every time/lane replay
    entry from external lanes.  Lane weights are broadcast over time but are
    not renormalized by their sum, so the term estimates
    ``sum_g q_g * (within-group mean surrogate)``.
    """

    import jax
    import jax.numpy as jnp

    lane_count = int(batch.observations.shape[1])
    sp_count = lane_count // 2
    logits, values = model.sequence(
        params,
        model.initial_carry(lane_count),
        batch.observations,
        batch.episode_starts,
    )
    sp_logits, sp_values = model.sequence(
        params,
        model.initial_carry(sp_count),
        batch.sp_other_observations,
        batch.episode_starts[:, :sp_count],
    )

    targets = jax.lax.stop_gradient(return_to_go(batch.rewards))
    old_values = jnp.asarray(batch.old_values, dtype=jnp.float32)
    sp_old_values = jnp.asarray(batch.sp_other_old_values, dtype=jnp.float32)
    advantages = jax.lax.stop_gradient(targets - old_values)
    sp_advantages = jax.lax.stop_gradient(
        targets[:, :sp_count] - sp_old_values
    )
    combined_advantages = jnp.concatenate((advantages, sp_advantages), axis=1)
    combined_mask = jnp.ones_like(combined_advantages, dtype=jnp.float32)
    normalized = joint_normalize_advantages(combined_advantages, combined_mask)
    normalized_advantages = normalized[:, :lane_count]
    normalized_sp_advantages = normalized[:, lane_count:]

    old_logp = jnp.asarray(batch.old_log_probabilities, dtype=jnp.float32)
    sp_old_logp = jnp.asarray(
        batch.sp_other_old_log_probabilities,
        dtype=jnp.float32,
    )
    new_logp = categorical_log_probability(logits, batch.actions)
    new_sp_logp = categorical_log_probability(sp_logits, batch.sp_other_actions)
    ratio = jnp.exp(new_logp - old_logp)
    sp_ratio = jnp.exp(new_sp_logp - sp_old_logp)
    clip_epsilon = float(model.config.ppo.clip_epsilon)
    clipped_ratio = jnp.clip(
        ratio,
        1.0 - clip_epsilon,
        1.0 + clip_epsilon,
    )
    clipped_sp_ratio = jnp.clip(
        sp_ratio,
        1.0 - clip_epsilon,
        1.0 + clip_epsilon,
    )
    surrogate = jnp.minimum(
        ratio * normalized_advantages,
        clipped_ratio * normalized_advantages,
    )
    sp_surrogate = jnp.minimum(
        sp_ratio * normalized_sp_advantages,
        clipped_sp_ratio * normalized_sp_advantages,
    )

    lane_stream = jnp.asarray(batch.lane_stream, dtype=jnp.int32)
    external_mask = (lane_stream == 1).astype(jnp.float32)
    external_mask_time = jnp.broadcast_to(
        external_mask[None, :], surrogate.shape
    )
    lane_weights = jnp.asarray(batch.lane_weight, dtype=jnp.float32)
    lane_weights_time = jnp.broadcast_to(lane_weights[None, :], surrogate.shape)
    external_sample_mask = external_mask_time
    external_count = jnp.sum(external_sample_mask)
    actor_external = -jnp.sum(
        external_sample_mask * lane_weights_time * surrogate
    ) / jnp.maximum(external_count, 1.0)
    actor_sp = -jnp.mean(
        jnp.concatenate((surrogate[:, :sp_count], sp_surrogate), axis=1)
    )

    value_clip_epsilon = float(model.config.ppo.value_clip_epsilon)
    current_values = jnp.asarray(values, dtype=jnp.float32)
    current_sp_values = jnp.asarray(sp_values, dtype=jnp.float32)
    clipped_values = old_values + jnp.clip(
        current_values - old_values,
        -value_clip_epsilon,
        value_clip_epsilon,
    )
    clipped_sp_values = sp_old_values + jnp.clip(
        current_sp_values - sp_old_values,
        -value_clip_epsilon,
        value_clip_epsilon,
    )
    value_error = jnp.maximum(
        jnp.square(current_values - targets),
        jnp.square(clipped_values - targets),
    )
    sp_value_error = jnp.maximum(
        jnp.square(current_sp_values - targets[:, :sp_count]),
        jnp.square(clipped_sp_values - targets[:, :sp_count]),
    )
    value_loss = 0.5 * jnp.mean(
        jnp.concatenate((value_error, sp_value_error), axis=1)
    )

    entropy = jnp.mean(
        jnp.concatenate(
            (categorical_entropy(logits), categorical_entropy(sp_logits)),
            axis=1,
        )
    )
    all_ratio = jnp.concatenate((ratio, sp_ratio), axis=1)
    all_log_gap = jnp.concatenate(
        (new_logp - old_logp, new_sp_logp - sp_old_logp),
        axis=1,
    )
    ratio_mean = jnp.mean(all_ratio)
    sampled_action_kl = 0.5 * jnp.mean(jnp.square(all_log_gap))

    episode_return = jax.lax.stop_gradient(
        jnp.asarray(batch.episode_return, dtype=jnp.float32)
    )
    sp_return_mean = jax.lax.stop_gradient(jnp.mean(episode_return[:sp_count]))
    external_returns = episode_return * external_mask
    external_return_mean = jax.lax.stop_gradient(
        jnp.sum(external_returns)
        / jnp.maximum(jnp.sum(external_mask), 1.0)
    )
    external_weight_mean = jax.lax.stop_gradient(
        jnp.sum(external_mask * lane_weights)
        / jnp.maximum(jnp.sum(external_mask), 1.0)
    )
    external_weight_max = jax.lax.stop_gradient(
        jnp.max(jnp.where(external_mask > 0.0, lane_weights, 0.0))
    )

    value_weight = float(model.config.ppo.value_weight)
    entropy_weight = float(model.config.ppo.entropy_weight)
    multiplier = jnp.asarray(dual_lambda, dtype=jnp.float32)
    total = (
        actor_external
        + multiplier * actor_sp
        + value_weight * value_loss
        - entropy_weight * entropy
    )
    return LossResult(
        total=total,
        metrics={
            "ppo_total": total,
            "actor_external": actor_external,
            "actor_sp": actor_sp,
            "value_loss": value_loss,
            "entropy": entropy,
            "ratio_mean": ratio_mean,
            "sampled_action_kl": sampled_action_kl,
            "dual_lambda": multiplier,
            "sp_return_mean": sp_return_mean,
            "external_return_mean": external_return_mean,
            "external_weight_mean": external_weight_mean,
            "external_weight_max": external_weight_max,
        },
    )


__all__ = [
    "categorical_entropy",
    "categorical_log_probability",
    "cetr_ppo_loss",
    "joint_normalize_advantages",
    "return_to_go",
]
