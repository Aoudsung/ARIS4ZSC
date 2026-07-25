"""Auxiliary prefit losses and a frozen-backbone optimizer."""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple


class AuxiliaryLosses(NamedTuple):
    response: Any
    prototype_value: Any
    transition_response: Any
    reward: Any
    next_feature: Any
    total: Any


def _routed_predictions(outputs: Mapping[str, Any], batch: Any) -> tuple[Any, ...]:
    import jax
    import jax.numpy as jnp

    prototype_count = outputs["prototype_values"].shape[-1]
    action_count = outputs["actor_logits"].shape[-1]
    response_count = outputs["response_logits"].shape[-1]
    prototype = jax.nn.one_hot(batch.partner_indices, prototype_count)
    action = jax.nn.one_hot(batch.actions, action_count)
    response = jax.nn.one_hot(batch.response_tokens, response_count)
    response_logits = jnp.einsum(
        "tbkay,tbk,tba->tby", outputs["response_logits"], prototype, action
    )
    prototype_value = jnp.einsum(
        "tbk,tbk->tb", outputs["prototype_values"], prototype
    )
    transition_logits = jnp.einsum(
        "tbkay,tbk,tba->tby",
        outputs["transition_response_logits"],
        prototype,
        action,
    )
    reward = jnp.einsum(
        "tbka,tbk,tba->tb", outputs["reward_estimates"], prototype, action
    )
    next_feature = jnp.einsum(
        "tbkayd,tbk,tba,tby->tbd",
        outputs["next_feature_summaries"],
        prototype,
        action,
        response,
    )
    next_prototype_value = jnp.einsum(
        "tbk,tbk->tb", batch.next_prototype_values, prototype
    )
    return response_logits, prototype_value, transition_logits, reward, next_feature, next_prototype_value


def auxiliary_losses(
    outputs: Mapping[str, Any],
    batch: Any,
    *,
    gamma: float,
    loss_weights: Mapping[str, float],
) -> AuxiliaryLosses:
    """Compute the five losses routed only by the known training prototype."""

    import jax
    import jax.numpy as jnp

    required = {"response", "prototype_value", "transition_response", "reward", "next_feature"}
    if set(loss_weights) != required:
        raise ValueError("Auxiliary loss weights must name exactly the five registered losses.")
    (
        response_logits,
        prototype_value,
        transition_logits,
        reward_prediction,
        next_feature_prediction,
        next_prototype_value,
    ) = _routed_predictions(outputs, batch)
    tokens = jnp.asarray(batch.response_tokens, dtype=jnp.int32)
    response_loss = jnp.mean(
        -jnp.take_along_axis(
            jax.nn.log_softmax(response_logits, axis=-1), tokens[..., None], axis=-1
        )[..., 0]
    )
    transition_response_loss = jnp.mean(
        -jnp.take_along_axis(
            jax.nn.log_softmax(transition_logits, axis=-1), tokens[..., None], axis=-1
        )[..., 0]
    )
    nonterminal = 1.0 - jnp.asarray(batch.dones, dtype=jnp.float32)
    value_target = jax.lax.stop_gradient(
        batch.rewards + float(gamma) * nonterminal * next_prototype_value
    )
    prototype_value_loss = 0.5 * jnp.mean(jnp.square(prototype_value - value_target))
    reward_loss = 0.5 * jnp.mean(jnp.square(reward_prediction - batch.rewards))
    feature_error = jnp.mean(
        jnp.square(next_feature_prediction - batch.next_feature_targets), axis=-1
    )
    next_feature_loss = jnp.sum(feature_error * nonterminal) / jnp.maximum(
        jnp.sum(nonterminal), 1.0
    )
    parts = {
        "response": response_loss,
        "prototype_value": prototype_value_loss,
        "transition_response": transition_response_loss,
        "reward": reward_loss,
        "next_feature": next_feature_loss,
    }
    total = sum(float(loss_weights[name]) * parts[name] for name in sorted(required))
    return AuxiliaryLosses(
        response_loss,
        prototype_value_loss,
        transition_response_loss,
        reward_loss,
        next_feature_loss,
        total,
    )


def prefit_loss(
    outputs: Mapping[str, Any], batch: Any, *, gamma: float, loss_weights: Mapping[str, float]
) -> tuple[Any, Mapping[str, Any]]:
    losses = auxiliary_losses(outputs, batch, gamma=gamma, loss_weights=loss_weights)
    return losses.total, losses._asdict()


def prefit_parameter_labels(params: Mapping[str, Any]) -> Mapping[str, Any]:
    """Freeze the official backbone and both shared official heads exactly."""

    import jax

    def label_top(name: str, value: Any) -> Any:
        label = "frozen" if name in {"backbone", "shared_heads"} else "auxiliary"
        return jax.tree_util.tree_map(lambda unused: label, value)

    return {str(name): label_top(str(name), value) for name, value in params.items()}


def make_prefit_optimizer(params: Mapping[str, Any], *, learning_rate: float) -> tuple[Any, Any]:
    """Create an optimizer that cannot change backbone or actor parameters."""

    import optax

    optimizer = optax.multi_transform(
        {
            "frozen": optax.set_to_zero(),
            "auxiliary": optax.adam(float(learning_rate)),
        },
        prefit_parameter_labels(params),
    )
    return optimizer, optimizer.init(params)


def prefit_update(
    *,
    params: Mapping[str, Any],
    optimizer_state: Any,
    optimizer: Any,
    loss_function: Any,
) -> tuple[Mapping[str, Any], Any, Mapping[str, Any]]:
    """Apply one already-bound differentiable prefit loss function."""

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
