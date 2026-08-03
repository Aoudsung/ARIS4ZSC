"""Initialization and continuous PPO updates for the one V6 partner generator."""

from __future__ import annotations

import math
from typing import Any, Mapping

from .decision_geometry import brdiv_marginal_contributions
from .partner_episode import (
    complete_episode_raw_returns,
    generator_ppo_objective,
    generator_weight_schedule,
    lower_tail_cvar_jax,
    source_logit_distillation_loss,
)
from .partner_generator import initial_generator_carry


def source_code_anchors(count: int, dimension: int) -> Any:
    import jax.numpy as jnp
    import numpy as np

    if int(count) != 4 or int(dimension) <= 0:
        raise ValueError("V6 initialization requires one SP/OP/SA/FCP source.")
    rows = np.arange(int(dimension), dtype=np.int32)
    anchors = np.stack(
        (
            1.0 - 2.0 * ((rows >> 0) & 1),
            1.0 - 2.0 * ((rows >> 1) & 1),
            -1.0 + 2.0 * ((rows >> 0) & 1),
            -1.0 + 2.0 * ((rows >> 1) & 1),
        ),
        axis=0,
    ).astype(np.float32)
    return jnp.asarray(anchors)


def distill_generator_sources(
    *,
    generator: Any,
    params: Any,
    batch: Any,
    code_anchors: Any,
    optimizer: Any,
    optimizer_state: Any,
    schedule: Any,
    hidden_dim: int,
) -> tuple[Any, Any, Mapping[str, Any]]:
    import jax
    import jax.numpy as jnp
    import optax

    indexes = jnp.asarray(schedule, dtype=jnp.int32).reshape(
        (-1, int(schedule.shape[-1]))
    )

    def one(carry: Any, lane_indexes: Any):
        current, current_optimizer_state = carry
        members = batch.source_members[:, lane_indexes]
        codes = code_anchors[members]
        observations = batch.observations[:-1, lane_indexes]
        starts = batch.episode_starts[:-1, lane_indexes]
        keys = jnp.zeros(observations.shape[:2] + (2,), dtype=jnp.uint32)
        initial = initial_generator_carry(int(lane_indexes.shape[0]), hidden_dim)

        def objective(candidate: Any):
            _, output = generator.apply(
                {"params": candidate},
                initial,
                observations,
                codes,
                starts,
                keys,
                method=generator.sequence,
            )
            return source_logit_distillation_loss(
                output.logits,
                batch.owner_logits[:, lane_indexes],
                batch.valid_mask[:, lane_indexes],
            )

        loss, gradients = jax.value_and_grad(objective)(current)
        updates, next_optimizer_state = optimizer.update(
            gradients, current_optimizer_state, current
        )
        return (
            optax.apply_updates(current, updates),
            next_optimizer_state,
        ), loss

    (next_params, next_state), losses = jax.lax.scan(
        one, (params, optimizer_state), indexes
    )
    return next_params, next_state, {
        "generator_source_distillation_kl": losses[-1],
        "generator_source_distillation_updates": jnp.asarray(indexes.shape[0]),
    }


def episode_decision_signatures(batch: Any) -> Any:
    import jax.numpy as jnp

    mask = jnp.asarray(batch.valid_mask, dtype=jnp.float32)
    signatures = jnp.asarray(batch.ego_action_signatures, dtype=jnp.float32)
    return jnp.sum(mask[..., None] * signatures, axis=0) / jnp.maximum(
        jnp.sum(mask, axis=0)[:, None], 1.0
    )


def generator_episode_bonus(
    *,
    batch: Any,
    progress: Any,
    competence_multiplier: Any,
    reference_cvar: Any,
    cvar_level: float,
    maximum_probability: float,
    ramp_fraction: float,
    kernel_bandwidth: float,
    kernel_jitter: float,
) -> tuple[Any, Mapping[str, Any]]:
    import jax
    import jax.numpy as jnp

    _, unused_imitation, diversity_weight = generator_weight_schedule(
        progress,
        maximum_probability=maximum_probability,
        ramp_fraction=ramp_fraction,
    )
    del unused_imitation
    signatures = jax.lax.stop_gradient(episode_decision_signatures(batch))
    diversity = brdiv_marginal_contributions(
        signatures, bandwidth=kernel_bandwidth, jitter=kernel_jitter
    )
    raw_returns = jax.lax.stop_gradient(
        jnp.sum(
            jnp.asarray(batch.raw_rewards, dtype=jnp.float32)
            * jnp.asarray(batch.valid_mask, dtype=jnp.float32),
            axis=0,
        )
    )
    sorted_returns = jnp.sort(raw_returns)
    tail_count = max(int(math.ceil(float(cvar_level) * raw_returns.shape[0])), 1)
    threshold = sorted_returns[tail_count - 1]
    tail = (raw_returns <= threshold).astype(jnp.float32)
    competence = (
        jnp.asarray(competence_multiplier, dtype=jnp.float32)
        * tail
        * (raw_returns - jnp.asarray(reference_cvar, dtype=jnp.float32))
        / jnp.maximum(jnp.mean(tail), 1.0e-8)
    )
    bonus = (
        diversity_weight * diversity
        + competence
    )
    return jax.lax.stop_gradient(bonus), {
        "generator_brdiv_weight": diversity_weight,
        "generator_brdiv_bonus_mean": jnp.mean(diversity),
        "generator_competence_bonus_mean": jnp.mean(competence),
        "generator_episode_raw_cvar": lower_tail_cvar_jax(
            raw_returns, level=cvar_level
        ),
    }


def _smoothness_loss(
    *, generator: Any, params: Any, batch: Any, lane_indexes: Any, hidden_dim: int
) -> Any:
    import jax
    import jax.numpy as jnp

    observations = batch.observations[:, lane_indexes]
    codes = batch.codes[:, lane_indexes]
    starts = jnp.zeros(batch.dones[:, lane_indexes].shape, dtype=jnp.bool_).at[0].set(True)
    keys = jnp.zeros(observations.shape[:2] + (2,), dtype=jnp.uint32)
    initial = jax.tree_util.tree_map(lambda value: value[lane_indexes], batch.initial_carries)
    direction = jnp.linspace(-1.0, 1.0, codes.shape[-1], dtype=jnp.float32)
    neighbor = jnp.clip(codes + 0.05 * direction, -1.0, 1.0)
    _, original = generator.apply(
        {"params": params}, initial, observations, codes, starts, keys,
        method=generator.sequence,
    )
    _, shifted = generator.apply(
        {"params": params}, initial, observations, neighbor, starts, keys,
        method=generator.sequence,
    )
    original_log = jax.nn.log_softmax(original.logits, axis=-1)
    shifted_log = jax.nn.log_softmax(shifted.logits, axis=-1)
    probability = jnp.exp(original_log)
    pointwise = jnp.sum(probability * (original_log - shifted_log), axis=-1)
    mask = jnp.asarray(batch.valid_mask[:, lane_indexes], dtype=jnp.float32)
    return jnp.sum(mask * pointwise) / jnp.maximum(jnp.sum(mask), 1.0)


def update_generator(
    *,
    generator: Any,
    params: Any,
    optimizer_state: Any,
    optimizer: Any,
    batch: Any,
    source_batch: Any,
    code_anchors: Any,
    schedule: Any,
    config: Any,
    progress: Any,
    competence_multiplier: Any,
    reference_cvar: Any,
) -> tuple[Any, Any, Mapping[str, Any]]:
    """Run the fixed four-epoch/eight-minibatch generator update."""

    import jax
    import jax.numpy as jnp
    import optax

    indexes = jnp.asarray(schedule, dtype=jnp.int32).reshape(
        (-1, int(schedule.shape[-1]))
    )
    _, imitation_weight, _ = generator_weight_schedule(
        progress,
        maximum_probability=config.partner_generator.maximum_generator_probability,
        ramp_fraction=config.partner_generator.mixture_ramp_fraction,
    )
    episode_bonus, bonus_metrics = generator_episode_bonus(
        batch=batch,
        progress=progress,
        competence_multiplier=competence_multiplier,
        reference_cvar=reference_cvar,
        cvar_level=config.partner_generator.cvar_level,
        maximum_probability=config.partner_generator.maximum_generator_probability,
        ramp_fraction=config.partner_generator.mixture_ramp_fraction,
        kernel_bandwidth=config.partner_generator.kernel_bandwidth,
        kernel_jitter=config.partner_generator.kernel_jitter,
    )

    def one(carry: Any, lane_indexes: Any):
        current, current_optimizer_state = carry
        observations = batch.observations[:, lane_indexes]
        codes = batch.codes[:, lane_indexes]
        starts = jnp.zeros(batch.dones[:, lane_indexes].shape, dtype=jnp.bool_).at[0].set(True)
        keys = jnp.zeros(observations.shape[:2] + (2,), dtype=jnp.uint32)
        initial = jax.tree_util.tree_map(
            lambda value: value[lane_indexes], batch.initial_carries
        )
        source_members = source_batch.source_members[:, lane_indexes]
        source_codes = code_anchors[source_members]
        source_observations = source_batch.observations[:-1, lane_indexes]
        source_starts = source_batch.episode_starts[:-1, lane_indexes]
        source_keys = jnp.zeros(source_observations.shape[:2] + (2,), dtype=jnp.uint32)
        source_initial = initial_generator_carry(
            int(lane_indexes.shape[0]), config.partner_generator.hidden_dim
        )

        def objective(candidate: Any):
            _, output = generator.apply(
                {"params": candidate}, initial, observations, codes, starts, keys,
                method=generator.sequence,
            )
            _, source_output = generator.apply(
                {"params": candidate}, source_initial, source_observations,
                source_codes, source_starts, source_keys, method=generator.sequence,
            )
            imitation = source_logit_distillation_loss(
                source_output.logits,
                source_batch.owner_logits[:, lane_indexes],
                source_batch.valid_mask[:, lane_indexes],
            )
            smoothness = _smoothness_loss(
                generator=generator,
                params=candidate,
                batch=batch,
                lane_indexes=lane_indexes,
                hidden_dim=config.partner_generator.hidden_dim,
            )
            return generator_ppo_objective(
                new_logits=output.logits,
                new_values=output.value,
                actions=batch.actions[:, lane_indexes],
                behavior_log_probabilities=batch.behavior_log_probabilities[:, lane_indexes],
                behavior_values=batch.behavior_values[:, lane_indexes],
                official_shaped_rewards=batch.official_shaped_rewards[:, lane_indexes],
                dones=batch.dones[:, lane_indexes],
                valid_mask=batch.valid_mask[:, lane_indexes],
                gamma=config.ppo.gamma,
                gae_lambda=config.ppo.gae_lambda,
                clip_epsilon=config.ppo.clip_epsilon,
                value_clip_epsilon=config.ppo.value_clip_epsilon,
                value_weight=config.ppo.value_weight,
                entropy_weight=config.ppo.entropy_weight,
                normalize_advantages=config.ppo.normalize_advantages,
                diversity_episode_bonus=episode_bonus[lane_indexes],
                imitation_loss=imitation,
                imitation_weight=imitation_weight,
                smoothness_loss=smoothness,
                smoothness_weight=config.partner_generator.smoothness_weight,
            )

        (loss, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(current)
        updates, next_optimizer_state = optimizer.update(
            gradients, current_optimizer_state, current
        )
        next_params = optax.apply_updates(current, updates)
        return (next_params, next_optimizer_state), {**metrics, "generator_loss": loss}

    (next_params, next_state), scan_metrics = jax.lax.scan(
        one, (params, optimizer_state), indexes
    )
    final_metrics = jax.tree_util.tree_map(lambda value: value[-1], scan_metrics)
    return next_params, next_state, {
        **final_metrics,
        **bonus_metrics,
        "generator_imitation_schedule_weight": imitation_weight,
    }


__all__ = [
    "distill_generator_sources",
    "episode_decision_signatures",
    "generator_episode_bonus",
    "source_code_anchors",
    "update_generator",
]
