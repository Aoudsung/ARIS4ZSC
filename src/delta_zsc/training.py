"""Separated base-policy and latent-model optimization transactions."""

from __future__ import annotations

from functools import partial
from typing import Any, Mapping

import jax

from .losses import (
    DECISION_METRIC_NAMES,
    latent_composite_loss,
    latent_contrast_loss,
    ppo_loss,
)
from .optimizer import adam_update
from .types import AnchorBatch, RolloutBatch


LATENT_CHANNEL_SUBTREES = {
    "response": ("component_embeddings", "response", "probe_response"),
    "successor": ("successor_feature",),
    "value": ("belief_value",),
    "contrast": ("belief_value",),
}
"""Which latent subtree each sequenced update writes.

``value`` and ``contrast`` write the same parameters on purpose: they are two
objectives for one critic, and they are given separate Adam states because they
fire at wildly different rates.  An anchor batch reaches roughly one update in
130; sharing a state would let 129 zero gradients decay the second-moment
estimate before the one gradient that carries the ordering information arrives,
and that gradient would then be applied with an enormous effective step.
"""

TARGET_SMOOTHING = 0.01
"""Polyak rate for the critic's bootstrap copy: ``tbar <- tau*t + (1-tau)*tbar``.

A rate of 0.01 gives a time constant of about a hundred outer updates, which is
roughly one anchor interval at the registered cadence.  The target therefore
moves slower than the measurement that corrects it, which is the property that
makes it a target at all.  It is a registered constant rather than a config key
for the same reason the mirror's uncertainty penalty is.
"""


def init_latent_optimizer(latent_params: Any) -> dict[str, Any]:
    """One Adam state per sequenced latent channel."""

    from .optimizer import init_adam

    return {
        name: init_adam(
            {key: latent_params[key] for key in subtrees if key in latent_params}
        )
        for name, subtrees in LATENT_CHANNEL_SUBTREES.items()
    }


def polyak_update(target: Any, online: Any, rate: float = TARGET_SMOOTHING) -> Any:
    import jax

    return jax.tree_util.tree_map(
        lambda slow, fast: (1.0 - float(rate)) * slow + float(rate) * fast,
        target,
        online,
    )


def _apply_channel(
    params: Any,
    gradients: Any,
    optimizer_state: Any,
    channel: str,
    *,
    learning_rate: float,
    maximum_gradient_norm: float,
    epsilon: float,
) -> tuple[Any, Any, Any]:
    """Commit one channel's gradient to its own subtree and Adam state."""

    subtrees = [
        key for key in LATENT_CHANNEL_SUBTREES[channel] if key in params
    ]
    sub_params = {key: params[key] for key in subtrees}
    sub_gradients = {key: gradients[key] for key in subtrees}
    updated, next_state, norm = adam_update(
        sub_params,
        sub_gradients,
        optimizer_state,
        learning_rate=learning_rate,
        maximum_gradient_norm=maximum_gradient_norm,
        epsilon=epsilon,
    )
    return {**params, **updated}, next_state, norm


def _tree_norm(tree: Any) -> Any:
    import jax
    import jax.numpy as jnp

    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    return jnp.sqrt(jnp.sum(jnp.stack([jnp.sum(jnp.square(x)) for x in leaves])))


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
    target_latent_params: Any,
    optimizer_state: Any,
    batch: RolloutBatch,
    anchors: AnchorBatch | None,
) -> tuple[Any, Any, Any, Mapping[str, Any]]:
    """Sequenced latent transaction: response, successor, raw-Q TD, contrast.

    The first three channels are parameter disjoint, so one reverse pass over
    the composite yields each of their gradients exactly; they differ only in
    which Adam state commits them.  The contrast is taken afterwards, in its own
    pass, because it writes the critic the TD step just moved -- and because a
    channel that fires on one update in a hundred and thirty needs its own
    moment estimates, not ones that have decayed to zero in between.
    """

    if model.config.method_variant in {"base", "history_rnn"}:
        return latent_params, target_latent_params, optimizer_state, {
            "latent_composite_nll": 0.0,
            "latent_response_nll": 0.0,
            "latent_shared_response_nll": 0.0,
            "latent_semantic_response_nll": 0.0,
            "latent_probe_shared_nll": 0.0,
            "latent_probe_semantic_nll": 0.0,
            "latent_shared_total_nll": 0.0,
            "latent_semantic_total_nll": 0.0,
            "latent_shared_response_observations": 0.0,
            "latent_semantic_response_observations": 0.0,
            "latent_probe_observations": 0.0,
            "latent_probe_semantic_observations": 0.0,
            "latent_shared_total_observations": 0.0,
            "latent_semantic_total_observations": 0.0,
            "latent_mean_posterior_entropy": 0.0,
            "latent_mean_filter_kl": 0.0,
            "latent_component_event_js": 0.0,
            "latent_probe_component_event_js": 0.0,
            **{name: 0.0 for name in DECISION_METRIC_NAMES},
            "latent_interface_coverage_rate": 0.0,
            "latent_interface_change_rate": 0.0,
            "latent_interface_other_multi_rate": 0.0,
            "latent_recipe_coverage_rate": 0.0,
            "latent_visibility_positive_rate": 0.0,
            "latent_inventory_change_positive_rate": 0.0,
            "latent_recipe_change_positive_rate": 0.0,
            "latent_gradient_norm": 0.0,
            "latent_shared_gradient_norm": 0.0,
            "latent_semantic_gradient_norm": 0.0,
            "latent_response_parameter_gradient_norm": 0.0,
            "latent_component_embedding_gradient_norm": 0.0,
            "latent_contrast_gradient_norm": 0.0,
            "latent_update_applied": 0.0,
            "latent_nonfinite_update": 0.0,
        }

    import jax.numpy as jnp

    learning_rate = float(model.config.latent_optimizer.learning_rate)
    clip = float(model.config.latent_optimizer.gradient_clip_norm)
    epsilon = float(model.config.latent_optimizer.adam_epsilon)

    def objective(candidate: Any):
        loss = latent_composite_loss(
            model,
            candidate,
            base_params,
            batch,
            anchors,
            target_latent_params=target_latent_params,
            include_contrast=False,
        )
        return loss.total, loss.metrics

    (total, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(
        latent_params
    )
    metrics = {**metrics, "latent_composite_nll": total}
    response_gradient_norm = _tree_norm(
        (gradients["response"], gradients["probe_response"])
    )
    embedding_gradient_norm = _tree_norm(gradients["component_embeddings"])

    updated = latent_params
    next_state = dict(optimizer_state)
    norms = []
    for channel in ("response", "successor", "value"):
        updated, next_state[channel], norm = _apply_channel(
            updated,
            gradients,
            optimizer_state[channel],
            channel,
            learning_rate=learning_rate,
            maximum_gradient_norm=clip,
            epsilon=epsilon,
        )
        norms.append(norm)
    gradient_norm = jnp.sqrt(sum(jnp.square(value) for value in norms))

    # The anchor contrast is a separate transaction; see
    # ``update_contrast_channel`` for why it must not share this module.
    contrast_norm = jnp.asarray(0.0, dtype=jnp.float32)

    leaves = jax.tree_util.tree_leaves((updated, next_state))
    finite = (
        jnp.isfinite(metrics["latent_composite_nll"])
        & jnp.isfinite(gradient_norm)
        & jnp.isfinite(contrast_norm)
        & jnp.all(jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in leaves]))
    )
    committed_params = jax.lax.cond(
        finite, lambda _: updated, lambda _: latent_params, None
    )
    committed_state = jax.lax.cond(
        finite, lambda _: next_state, lambda _: optimizer_state, None
    )
    committed_target = jax.lax.cond(
        finite,
        lambda _: polyak_update(target_latent_params, committed_params),
        lambda _: target_latent_params,
        None,
    )
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    return committed_params, committed_target, committed_state, {
        **metrics,
        "latent_gradient_norm": gradient_norm,
        # Per-channel full-tree gradient diagnostics are deliberately not part
        # of the training transaction; zero is an explicit schema value, not an
        # estimated gradient.
        "latent_shared_gradient_norm": zero,
        "latent_semantic_gradient_norm": zero,
        "latent_response_parameter_gradient_norm": response_gradient_norm,
        "latent_component_embedding_gradient_norm": embedding_gradient_norm,
        "latent_contrast_gradient_norm": contrast_norm,
        "latent_update_applied": finite.astype(jnp.float32),
        "latent_nonfinite_update": (~finite).astype(jnp.float32),
    }



@partial(jax.jit, static_argnames=("model",))
def update_contrast_channel(
    *,
    model: Any,
    latent_params: Any,
    base_params: Any,
    optimizer_state: Any,
    batch: RolloutBatch,
    anchors: AnchorBatch,
) -> tuple[Any, Any, Mapping[str, Any]]:
    """Calibrate the critic against the anchor contrasts, in its own module.

    This is deliberately a separate executable rather than a branch inside
    ``update_latent_model``.  Fusing it in put two 256-step recurrent scans in
    one HLO module, and XLA's backend did not finish compiling that module in
    forty minutes -- against 173 seconds for the identical kernel without the
    anchor branch.  Split, the anchor path reuses the ordinary latent module
    (a compilation-cache hit) and adds one small one.

    It runs after ``update_latent_model`` has committed, so the contrast
    corrects a critic that has already absorbed this rollout's returns.
    """

    import jax.numpy as jnp

    def objective(candidate: Any):
        result = latent_contrast_loss(model, candidate, base_params, batch, anchors)
        return result.total, result.metrics

    (_, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(
        latent_params
    )
    updated, next_state, norm = _apply_channel(
        latent_params,
        gradients,
        optimizer_state,
        "contrast",
        learning_rate=float(model.config.latent_optimizer.learning_rate),
        maximum_gradient_norm=float(
            model.config.latent_optimizer.gradient_clip_norm
        ),
        epsilon=float(model.config.latent_optimizer.adam_epsilon),
    )
    leaves = jax.tree_util.tree_leaves((updated, next_state))
    finite = jnp.isfinite(norm) & jnp.all(
        jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in leaves])
    )
    committed = jax.lax.cond(finite, lambda _: updated, lambda _: latent_params, None)
    committed_state = jax.lax.cond(
        finite, lambda _: next_state, lambda _: optimizer_state, None
    )
    return committed, committed_state, {
        **metrics,
        "latent_contrast_gradient_norm": norm,
        "latent_contrast_update_applied": finite.astype(jnp.float32),
    }


def training_update(
    *,
    model: Any,
    base_params: Any,
    latent_params: Any,
    target_latent_params: Any,
    base_optimizer_state: Any,
    latent_optimizer_state: Any,
    batch: RolloutBatch,
    anchors: AnchorBatch | None,
    schedule: Any,
    total_optimizer_steps: int,
) -> tuple[Any, Any, Any, Any, Any, Mapping[str, Any]]:
    """Update the collection-policy latent model, then commit PPO minibatches.

    Counterfactual anchors are continuations of ``base_params`` at collection
    time.  The latent transaction therefore consumes that exact parameter tree
    before PPO changes it.  PPO is independent of ``latent_params`` because its
    replay path is statically base-only.  This ordering avoids training the
    decision emission against returns generated by a different continuation
    policy.
    """

    import jax.numpy as jnp

    (
        current_latent,
        current_target,
        current_latent_state,
        latent_metrics,
    ) = update_latent_model(
        model=model,
        latent_params=latent_params,
        base_params=base_params,
        target_latent_params=target_latent_params,
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
        current_target,
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
            target_latent_params: Any,
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
                target_latent_params=target_latent_params,
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
            target_latent_params: Any,
            base_optimizer_state: Any,
            latent_optimizer_state: Any,
            batch: RolloutBatch,
            schedule: Any,
        ):
            return training_update(
                model=model,
                base_params=base_params,
                latent_params=latent_params,
                target_latent_params=target_latent_params,
                base_optimizer_state=base_optimizer_state,
                latent_optimizer_state=latent_optimizer_state,
                batch=batch,
                anchors=None,
                schedule=schedule,
                total_optimizer_steps=total_optimizer_steps,
            )

    return kernel


__all__ = [
    "LATENT_CHANNEL_SUBTREES",
    "TARGET_SMOOTHING",
    "init_latent_optimizer",
    "update_contrast_channel",
    "polyak_update",
    "environment_minibatch_schedule",
    "make_training_update_kernel",
    "ppo_learning_rate",
    "slice_rollout_lanes",
    "training_update",
    "update_base_policy",
    "update_latent_model",
]
