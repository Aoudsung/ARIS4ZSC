"""Tail weighting and PPO update transactions for CETR."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping

from .losses import cetr_ppo_loss
from .optimizer import adam_update, init_adam
from .risk import lower_half_cvar_weights, renormalize_weights
from .types import EpisodeBatch, TrainState


def ppo_learning_rate(
    *, config: Any, optimizer_step: Any, total_optimizer_steps: int
) -> Any:
    """Registered warm-up followed by optional cosine annealing."""

    import jax.numpy as jnp

    total = max(int(total_optimizer_steps), 1)
    warmup = int(float(config.ppo.lr_warmup_fraction) * total)
    step = jnp.asarray(optimizer_step, dtype=jnp.float32)
    base = jnp.asarray(config.ppo.learning_rate, dtype=jnp.float32)
    if warmup > 0:
        warmup_rate = base * jnp.minimum((step + 1.0) / float(warmup), 1.0)
    else:
        warmup_rate = base
    if not bool(config.ppo.anneal_learning_rate):
        return warmup_rate
    decay_steps = max(total - warmup, 1)
    progress = jnp.clip((step - float(warmup)) / float(decay_steps), 0.0, 1.0)
    cosine_rate = 0.5 * base * (1.0 + jnp.cos(jnp.pi * progress))
    return jnp.where(step < float(warmup), warmup_rate, cosine_rate)


def environment_minibatch_schedule(
    key: Any,
    *,
    environment_count: int,
    minibatches_per_epoch: int,
    update_epochs: int,
) -> Any:
    """Pair one permutation of self-play lanes with one of external lanes."""

    import jax
    import jax.numpy as jnp

    count = int(environment_count)
    minibatches = int(minibatches_per_epoch)
    epochs = int(update_epochs)
    if count % 4:
        raise ValueError("Mixed training requires a lane count divisible by four.")
    stream_count = count // 2
    if stream_count % minibatches:
        raise ValueError("Each stream must divide into paired minibatches.")
    lanes_per_minibatch = stream_count // minibatches
    keys = jax.random.split(key, epochs)

    def one(epoch_key: Any) -> Any:
        self_key, external_key = jax.random.split(epoch_key)
        self_indexes = jax.random.permutation(self_key, stream_count).reshape(
            (minibatches, lanes_per_minibatch)
        )
        external_indexes = (
            jax.random.permutation(external_key, stream_count).reshape(
                (minibatches, lanes_per_minibatch)
            )
            + stream_count
        )
        return jnp.concatenate((self_indexes, external_indexes), axis=-1)

    return jnp.stack([one(epoch_key) for epoch_key in keys])


def compute_tail_weights(
    batch: EpisodeBatch,
    parent_nominal_weights: Any,
    parent_count: int,
) -> tuple[Any, Mapping[str, Any]]:
    """Compute cross-fitted parent weights from completed episode returns."""

    import jax.numpy as jnp

    count = int(parent_count)
    if count <= 0:
        raise ValueError("Parent count must be positive.")
    nominal = jnp.asarray(parent_nominal_weights, dtype=jnp.float32)
    if nominal.shape != (count,):
        raise ValueError("Parent nominal weights do not match parent count.")

    lane_stream = jnp.asarray(batch.lane_stream, dtype=jnp.int32)
    lane_parent = jnp.asarray(batch.lane_parent, dtype=jnp.int32)
    lane_fold = jnp.asarray(batch.lane_fold, dtype=jnp.int32)
    returns = jnp.asarray(batch.episode_return, dtype=jnp.float32)
    lanes = int(returns.shape[0])
    safe_parent = jnp.clip(lane_parent, 0, count - 1)
    lane_weight = jnp.ones((lanes,), dtype=jnp.float32)
    fold_means = []
    fold_q = []
    fold_nominal = []
    fold_observed = []
    fold_max_weight = []

    for fold in (0, 1):
        selected = (lane_stream == 1) & (lane_fold == fold)
        sums = jnp.zeros((count,), dtype=jnp.float32).at[safe_parent].add(
            jnp.where(selected, returns, 0.0)
        )
        counts = jnp.zeros((count,), dtype=jnp.float32).at[safe_parent].add(
            selected.astype(jnp.float32)
        )
        observed = counts > 0.0
        means = jnp.where(observed, sums / jnp.maximum(counts, 1.0), 0.0)
        q = lower_half_cvar_weights(means, nominal, observed)
        observed_nominal = renormalize_weights(nominal, observed)
        safe_ratio = jnp.where(
            (observed_nominal > 0.0) & (q > 0.0),
            q / jnp.maximum(observed_nominal, 1.0e-30),
            0.0,
        )
        target = (lane_stream == 1) & (lane_fold == (1 - fold))
        lane_weight = jnp.where(target, safe_ratio[safe_parent], lane_weight)
        target_weights = jnp.where(target, safe_ratio[safe_parent], 0.0)
        fold_means.append(means)
        fold_q.append(q)
        fold_nominal.append(observed_nominal)
        fold_observed.append(observed)
        fold_max_weight.append(jnp.max(target_weights))

    tail_objective = 0.5 * sum(
        jnp.sum(q * means) for q, means in zip(fold_q, fold_means)
    )
    metrics = {
        "tail_objective": tail_objective,
        "fold_0_observed_parent_count": jnp.sum(fold_observed[0].astype(jnp.float32)),
        "fold_1_observed_parent_count": jnp.sum(fold_observed[1].astype(jnp.float32)),
        "fold_0_weight_max": fold_max_weight[0],
        "fold_1_weight_max": fold_max_weight[1],
        "tail_weight_max": jnp.max(lane_weight),
    }
    return lane_weight, metrics


def slice_episode_lanes(batch: EpisodeBatch, indexes: Any) -> EpisodeBatch:
    """Slice a minibatch while keeping the self-play companion stream aligned."""

    import jax
    import jax.numpy as jnp

    lane_indexes = jnp.asarray(indexes, dtype=jnp.int32)
    lane_count = int(batch.actions.shape[1])
    self_count = lane_count // 2
    self_indexes = lane_indexes[lane_indexes < self_count]

    return batch._replace(
        observations=batch.observations[:, lane_indexes],
        actions=batch.actions[:, lane_indexes],
        old_log_probabilities=batch.old_log_probabilities[:, lane_indexes],
        old_values=batch.old_values[:, lane_indexes],
        rewards=batch.rewards[:, lane_indexes],
        dones=batch.dones[:, lane_indexes],
        episode_starts=batch.episode_starts[:, lane_indexes],
        sp_other_observations=batch.sp_other_observations[:, self_indexes],
        sp_other_actions=batch.sp_other_actions[:, self_indexes],
        sp_other_old_log_probabilities=batch.sp_other_old_log_probabilities[:, self_indexes],
        sp_other_old_values=batch.sp_other_old_values[:, self_indexes],
        lane_stream=batch.lane_stream[lane_indexes],
        lane_parent=batch.lane_parent[lane_indexes],
        lane_fold=batch.lane_fold[lane_indexes],
        lane_weight=batch.lane_weight[lane_indexes],
        episode_return=batch.episode_return[lane_indexes],
        ego_roles=batch.ego_roles[lane_indexes],
        member_index=batch.member_index[lane_indexes],
    )


def _tree_is_finite(tree: Any) -> Any:
    import jax
    import jax.numpy as jnp

    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(True)
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in leaves]))


def _safe_metrics(metrics: Mapping[str, Any], finite: Any) -> dict[str, Any]:
    import jax.numpy as jnp

    return {
        name: jnp.where(
            finite,
            jnp.asarray(value),
            jnp.zeros_like(jnp.asarray(value)),
        )
        for name, value in metrics.items()
    }


def _update_actor_critic_impl(
    *,
    model: Any,
    params: Any,
    optimizer_state: Any,
    batch: EpisodeBatch,
    dual_lambda: Any,
    total_optimizer_steps: int,
) -> tuple[Any, Any, Mapping[str, Any]]:
    import jax
    import jax.numpy as jnp

    def objective(candidate: Any) -> tuple[Any, Mapping[str, Any]]:
        result = cetr_ppo_loss(model, candidate, batch, dual_lambda)
        return result.total, result.metrics

    (total, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(params)
    learning_rate = ppo_learning_rate(
        config=model.config,
        optimizer_step=optimizer_state.count,
        total_optimizer_steps=total_optimizer_steps,
    )
    updated, next_state, gradient_norm = adam_update(
        params,
        gradients,
        optimizer_state,
        learning_rate=learning_rate,
        maximum_gradient_norm=float(model.config.ppo.gradient_clip_norm),
        epsilon=float(model.config.ppo.adam_epsilon),
    )
    finite = (
        jnp.isfinite(total)
        & jnp.isfinite(gradient_norm)
        & _tree_is_finite((updated, next_state))
    )
    committed_params = jax.lax.cond(
        finite, lambda unused: updated, lambda unused: params, None
    )
    committed_state = jax.lax.cond(
        finite, lambda unused: next_state, lambda unused: optimizer_state, None
    )
    safe = _safe_metrics(metrics, finite)
    safe.update(
        {
            "ppo_total": jnp.where(finite, total, 0.0),
            "ppo_gradient_norm": jnp.where(finite, gradient_norm, 0.0),
            "ppo_learning_rate": jnp.where(finite, learning_rate, 0.0),
            "ppo_update_applied": finite.astype(jnp.float32),
            "ppo_nonfinite_update": (~finite).astype(jnp.float32),
        }
    )
    return committed_params, committed_state, safe


@lru_cache(maxsize=1)
def _compiled_actor_critic_update() -> Any:
    import jax

    return jax.jit(
        _update_actor_critic_impl,
        static_argnames=("model", "total_optimizer_steps"),
    )


def update_actor_critic(
    *,
    model: Any,
    params: Any,
    optimizer_state: Any,
    batch: EpisodeBatch,
    dual_lambda: Any,
    total_optimizer_steps: int,
) -> tuple[Any, Any, Mapping[str, Any]]:
    """Run one finite-guarded, compiled PPO optimizer transaction."""

    return _compiled_actor_critic_update()(
        model=model,
        params=params,
        optimizer_state=optimizer_state,
        batch=batch,
        dual_lambda=dual_lambda,
        total_optimizer_steps=total_optimizer_steps,
    )


def training_update(
    *,
    model: Any,
    params: Any,
    optimizer_state: Any,
    batch: EpisodeBatch,
    schedule: Any,
    dual_lambda: Any,
    total_optimizer_steps: int,
) -> tuple[Any, Any, Mapping[str, Any]]:
    """Apply every scheduled minibatch in one scan.

    The registered schedule supplies exactly one self-play lane and one external
    lane per minibatch.  When a minibatch contains no self-play lane, the
    companion arrays are sliced to length zero; the paired schedule prevents
    that case in supported training.
    """

    import jax
    import jax.numpy as jnp

    flat_schedule = jnp.asarray(schedule, dtype=jnp.int32).reshape(
        (-1, schedule.shape[-1])
    )

    def one(carry: tuple[Any, Any], indexes: Any):
        current_params, current_state = carry
        next_params, next_state, metrics = update_actor_critic(
            model=model,
            params=current_params,
            optimizer_state=current_state,
            batch=slice_episode_lanes(batch, indexes),
            dual_lambda=dual_lambda,
            total_optimizer_steps=total_optimizer_steps,
        )
        return (next_params, next_state), metrics

    (updated_params, updated_state), metrics = jax.lax.scan(
        one,
        (params, optimizer_state),
        flat_schedule,
    )
    mean_metrics = jax.tree_util.tree_map(
        lambda value: jnp.mean(jnp.asarray(value), axis=0), metrics
    )
    return updated_params, updated_state, mean_metrics


def make_training_update_kernel(*, model: Any, total_optimizer_steps: int) -> Any:
    import jax

    @jax.jit
    def kernel(
        params: Any,
        optimizer_state: Any,
        batch: EpisodeBatch,
        schedule: Any,
        dual_lambda: Any,
    ) -> tuple[Any, Any, Mapping[str, Any]]:
        return training_update(
            model=model,
            params=params,
            optimizer_state=optimizer_state,
            batch=batch,
            schedule=schedule,
            dual_lambda=dual_lambda,
            total_optimizer_steps=total_optimizer_steps,
        )

    return kernel


def init_training_state(*, model: Any, params: Any, random_key: Any) -> TrainState:
    import jax.numpy as jnp

    del model
    return TrainState(
        params=params,
        optimizer_state=init_adam(params),
        dual_lambda=jnp.asarray(0.0, dtype=jnp.float32),
        random_key=random_key,
        update_count=jnp.asarray(0, dtype=jnp.int32),
        effective_environment_steps=jnp.asarray(0, dtype=jnp.int32),
        resource_ledger={},
    )


__all__ = [
    "compute_tail_weights",
    "environment_minibatch_schedule",
    "init_training_state",
    "make_training_update_kernel",
    "ppo_learning_rate",
    "slice_episode_lanes",
    "training_update",
    "update_actor_critic",
]
