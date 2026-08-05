"""Separated base-policy and latent-model optimization transactions."""

from __future__ import annotations

from functools import partial
from typing import Any, Mapping

import jax

from .losses import latent_composite_loss, ppo_loss
from .optimizer import adam_update
from .types import AnchorBatch, RolloutBatch


def slice_rollout_lanes(batch: RolloutBatch, indexes: Any) -> RolloutBatch:
    import jax

    values = {}
    for name, value in batch._asdict().items():
        if name == "initial_policy_state":
            values[name] = jax.tree_util.tree_map(lambda leaf: leaf[indexes], value)
        else:
            values[name] = value[:, indexes]
    return RolloutBatch(**values)



def ppo_learning_rate(
    *, config: Any, optimizer_step: Any, total_optimizer_steps: int
) -> Any:
    """Registered warm-up followed by optional cosine annealing.

    The schedule is an optimizer property, not a scientific method parameter.
    It is evaluated from the checkpointed Adam step so save/resume is exactly
    equivalent to an uninterrupted run.
    """

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
    import jax
    import jax.numpy as jnp

    if int(environment_count) % int(minibatches_per_epoch):
        raise ValueError("Environment lanes must divide into minibatches.")
    lanes = int(environment_count) // int(minibatches_per_epoch)
    keys = jax.random.split(key, int(update_epochs))
    return jnp.stack(
        [
            jax.random.permutation(item, int(environment_count)).reshape(
                (int(minibatches_per_epoch), lanes)
            )
            for item in keys
        ]
    )


@partial(jax.jit, static_argnames=("model", "total_optimizer_steps"))
def update_base_policy(
    *,
    model: Any,
    base_params: Any,
    latent_params: Any,
    optimizer_state: Any,
    batch: RolloutBatch,
    total_optimizer_steps: int,
) -> tuple[Any, Any, Mapping[str, Any]]:
    def objective(candidate: Any):
        loss = ppo_loss(model, candidate, latent_params, batch)
        return loss.total, loss.metrics

    (_, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(base_params)
    learning_rate = ppo_learning_rate(
        config=model.config,
        optimizer_step=optimizer_state.count,
        total_optimizer_steps=total_optimizer_steps,
    )
    updated, next_state, gradient_norm = adam_update(
        base_params,
        gradients,
        optimizer_state,
        learning_rate=learning_rate,
        maximum_gradient_norm=float(model.config.ppo.gradient_clip_norm),
        epsilon=float(model.config.ppo.adam_epsilon),
    )
    import jax.numpy as jnp

    leaves = jax.tree_util.tree_leaves((updated, next_state))
    finite = (
        jnp.isfinite(metrics["ppo_total"])
        & jnp.isfinite(gradient_norm)
        & jnp.all(jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in leaves]))
    )
    committed_params = jax.lax.cond(finite, lambda _: updated, lambda _: base_params, None)
    committed_state = jax.lax.cond(finite, lambda _: next_state, lambda _: optimizer_state, None)
    return committed_params, committed_state, {
        **metrics,
        "base_gradient_norm": gradient_norm,
        "base_learning_rate": learning_rate,
        "base_update_applied": finite.astype(jnp.float32),
        "base_nonfinite_update": (~finite).astype(jnp.float32),
    }


@partial(jax.jit, static_argnames=("model",))
def update_latent_model(
    *,
    model: Any,
    latent_params: Any,
    base_params: Any,
    optimizer_state: Any,
    batch: RolloutBatch,
    anchors: AnchorBatch | None,
) -> tuple[Any, Any, Mapping[str, Any]]:
    if model.config.method_variant in {"base", "history_rnn"}:
        return latent_params, optimizer_state, {
            "latent_composite_nll": 0.0,
            "latent_response_nll": 0.0,
            "latent_decision_nll": 0.0,
            "latent_response_observations": 0.0,
            "latent_decision_observations": 0.0,
            "latent_mean_posterior_entropy": 0.0,
            "latent_decision_top_action_agreement": 0.0,
            "latent_decision_empirical_regret": 0.0,
            "latent_gradient_norm": 0.0,
            "latent_update_applied": 0.0,
            "latent_nonfinite_update": 0.0,
        }

    def objective(candidate: Any):
        loss = latent_composite_loss(model, candidate, base_params, batch, anchors)
        return loss.total, loss.metrics

    (_, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(latent_params)
    updated, next_state, gradient_norm = adam_update(
        latent_params,
        gradients,
        optimizer_state,
        learning_rate=float(model.config.latent_optimizer.learning_rate),
        maximum_gradient_norm=float(model.config.latent_optimizer.gradient_clip_norm),
        epsilon=float(model.config.latent_optimizer.adam_epsilon),
    )
    import jax.numpy as jnp

    leaves = jax.tree_util.tree_leaves((updated, next_state))
    finite = (
        jnp.isfinite(metrics["latent_composite_nll"])
        & jnp.isfinite(gradient_norm)
        & jnp.all(jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in leaves]))
    )
    committed_params = jax.lax.cond(finite, lambda _: updated, lambda _: latent_params, None)
    committed_state = jax.lax.cond(finite, lambda _: next_state, lambda _: optimizer_state, None)
    return committed_params, committed_state, {
        **metrics,
        "latent_gradient_norm": gradient_norm,
        "latent_update_applied": finite.astype(jnp.float32),
        "latent_nonfinite_update": (~finite).astype(jnp.float32),
    }


def training_update(
    *,
    model: Any,
    base_params: Any,
    latent_params: Any,
    base_optimizer_state: Any,
    latent_optimizer_state: Any,
    batch: RolloutBatch,
    anchors: AnchorBatch | None,
    schedule: Any,
    total_optimizer_steps: int,
) -> tuple[Any, Any, Any, Any, Mapping[str, Any]]:
    """Update the collection-policy latent model, then commit PPO minibatches.

    Counterfactual anchors are continuations of ``base_params`` at collection
    time.  The latent transaction therefore consumes that exact parameter tree
    before PPO changes it.  PPO is independent of ``latent_params`` because its
    replay path is statically base-only.  This ordering avoids training the
    decision emission against returns generated by a different continuation
    policy.
    """

    import jax.numpy as jnp

    current_latent, current_latent_state, latent_metrics = update_latent_model(
        model=model,
        latent_params=latent_params,
        base_params=base_params,
        optimizer_state=latent_optimizer_state,
        batch=batch,
        anchors=anchors,
    )

    flat = jnp.asarray(schedule).reshape((-1, schedule.shape[-1]))

    def ppo_minibatch(carry: tuple[Any, Any], indexes: Any):
        current_base, current_base_state = carry
        next_base, next_base_state, metrics = update_base_policy(
            model=model,
            base_params=current_base,
            latent_params=current_latent,
            optimizer_state=current_base_state,
            batch=slice_rollout_lanes(batch, indexes),
            total_optimizer_steps=total_optimizer_steps,
        )
        return (next_base, next_base_state), metrics

    # Keep the registered epoch/minibatch order exactly unchanged while
    # lowering the whole sequence as one device program.  The previous Python
    # loop dispatched one tiny four-lane GRU update per minibatch.
    (current_base, current_base_state), ppo_metrics = jax.lax.scan(
        ppo_minibatch,
        (base_params, base_optimizer_state),
        flat,
    )
    mean_ppo = jax.tree_util.tree_map(
        lambda value: jnp.mean(jnp.asarray(value), axis=0), ppo_metrics
    )
    return (
        current_base,
        current_latent,
        current_base_state,
        current_latent_state,
        {"ppo": mean_ppo, "latent": latent_metrics},
    )


def make_training_update_kernel(
    *, model: Any, total_optimizer_steps: int, with_anchors: bool
) -> Any:
    """Create one stable compiled outer update for an anchor signature.

    The PPO lane gathers and the complete optimizer-step sequence stay inside
    this executable.  Separate no-anchor and with-anchor kernels avoid changing
    PyTree signatures at runtime.
    """

    import jax

    if bool(with_anchors):

        @jax.jit
        def kernel(
            base_params: Any,
            latent_params: Any,
            base_optimizer_state: Any,
            latent_optimizer_state: Any,
            batch: RolloutBatch,
            anchors: AnchorBatch,
            schedule: Any,
        ):
            return training_update(
                model=model,
                base_params=base_params,
                latent_params=latent_params,
                base_optimizer_state=base_optimizer_state,
                latent_optimizer_state=latent_optimizer_state,
                batch=batch,
                anchors=anchors,
                schedule=schedule,
                total_optimizer_steps=total_optimizer_steps,
            )

    else:

        @jax.jit
        def kernel(
            base_params: Any,
            latent_params: Any,
            base_optimizer_state: Any,
            latent_optimizer_state: Any,
            batch: RolloutBatch,
            schedule: Any,
        ):
            return training_update(
                model=model,
                base_params=base_params,
                latent_params=latent_params,
                base_optimizer_state=base_optimizer_state,
                latent_optimizer_state=latent_optimizer_state,
                batch=batch,
                anchors=None,
                schedule=schedule,
                total_optimizer_steps=total_optimizer_steps,
            )

    return kernel


__all__ = [
    "environment_minibatch_schedule",
    "make_training_update_kernel",
    "ppo_learning_rate",
    "slice_rollout_lanes",
    "training_update",
    "update_base_policy",
    "update_latent_model",
]
