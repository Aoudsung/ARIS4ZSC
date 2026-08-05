"""Action-conditioned response emission for the shared latent coordination mode."""

from __future__ import annotations

from typing import Any

from .nn import init_linear, init_mlp, linear, mlp
from .observation import (
    ResponseTarget,
    PARTNER_DIRECTION_CLASSES,
    PARTNER_INVENTORY_FACTOR_CLASSES,
    PARTNER_POSITION_CLASSES,
)
from .types import ResponsePrediction


def init_response_params(
    key: Any,
    *,
    observation_shape: tuple[int, ...],
    behavior_dim: int,
    component_embedding_dim: int,
    action_count: int,
    action_embedding_dim: int,
    hidden_dim: int,
    inventory_factor_count: int,
) -> dict[str, Any]:
    import jax

    keys = jax.random.split(key, 8)
    frame_dim = int(observation_shape[0] * observation_shape[1] * observation_shape[2])
    trunk_input = hidden_dim + behavior_dim + component_embedding_dim + action_embedding_dim
    return {
        "frame_encoder": init_mlp(keys[0], (frame_dim, hidden_dim, hidden_dim)),
        "action_embedding": jax.random.normal(
            keys[1], (int(action_count), int(action_embedding_dim))
        ) / max(action_embedding_dim, 1) ** 0.5,
        "trunk": init_mlp(keys[2], (trunk_input, hidden_dim, hidden_dim)),
        "visibility": init_linear(keys[3], hidden_dim, 1, scale=0.01),
        "position": init_linear(
            keys[4], hidden_dim, PARTNER_POSITION_CLASSES, scale=0.01
        ),
        "direction": init_linear(
            keys[5], hidden_dim, PARTNER_DIRECTION_CLASSES, scale=0.01
        ),
        "inventory": init_linear(
            keys[6],
            hidden_dim,
            int(inventory_factor_count) * PARTNER_INVENTORY_FACTOR_CLASSES,
            scale=0.01,
        ),
        "event": init_linear(keys[7], hidden_dim, 1, scale=0.01),
    }


def response_predict(
    params: dict[str, Any],
    component_embeddings: Any,
    frame: Any,
    behavior: Any,
    actions: Any,
) -> ResponsePrediction:
    """Return per-component response distributions while preserving position.

    The 5x5 frame is flattened, not reduced to channel means/maxima.  Therefore
    distinct partner positions remain distinguishable in principle.
    """

    import jax.numpy as jnp

    observation = jnp.asarray(frame, dtype=jnp.float32)
    action = jnp.asarray(actions, dtype=jnp.int32)
    behavior_features = jnp.asarray(behavior, dtype=jnp.float32)
    components = jnp.asarray(component_embeddings, dtype=jnp.float32)
    lead = action.shape
    if observation.shape[:-3] != lead or behavior_features.shape[:-1] != lead:
        raise ValueError("Response batch axes differ.")
    frame_flat = observation.reshape(lead + (-1,))
    frame_encoded = mlp(params["frame_encoder"], frame_flat, final_activation=True)
    action_emb = params["action_embedding"][action]
    count = components.shape[0]

    def add_component(value: Any) -> Any:
        array = jnp.asarray(value)
        return jnp.broadcast_to(
            array[..., None, :], array.shape[:-1] + (count, array.shape[-1])
        )

    comp = jnp.broadcast_to(
        components.reshape((1,) * len(lead) + components.shape),
        lead + components.shape,
    )
    trunk_input = jnp.concatenate(
        (
            add_component(frame_encoded),
            add_component(behavior_features),
            comp,
            add_component(action_emb),
        ),
        axis=-1,
    )
    hidden = mlp(params["trunk"], trunk_input, final_activation=True)
    inventory = linear(params["inventory"], hidden).reshape(
        lead
        + (
            count,
            int(params["inventory"]["bias"].shape[0] // PARTNER_INVENTORY_FACTOR_CLASSES),
            PARTNER_INVENTORY_FACTOR_CLASSES,
        )
    )
    return ResponsePrediction(
        visibility_logit=linear(params["visibility"], hidden)[..., 0],
        relative_position_logits=linear(params["position"], hidden),
        direction_logits=linear(params["direction"], hidden),
        inventory_logits=inventory,
        inventory_change_logit=linear(params["event"], hidden)[..., 0],
    )


def _bernoulli_log_probability(logits: Any, labels: Any) -> Any:
    import jax.nn
    import jax.numpy as jnp

    target = jnp.asarray(labels, dtype=jnp.float32)
    return target * jax.nn.log_sigmoid(logits) + (1.0 - target) * jax.nn.log_sigmoid(-logits)


def _categorical_log_probability(logits: Any, labels: Any) -> Any:
    import jax.nn
    import jax.numpy as jnp

    logp = jax.nn.log_softmax(logits, axis=-1)
    label = jnp.asarray(labels, dtype=jnp.int32)
    index = jnp.broadcast_to(label[..., None, None], logp.shape[:-1] + (1,))
    return jnp.take_along_axis(logp, index, axis=-1)[..., 0]


def _factor_categorical_log_probability(logits: Any, labels: Any) -> Any:
    import jax.nn
    import jax.numpy as jnp

    logp = jax.nn.log_softmax(logits, axis=-1)
    label = jnp.asarray(labels, dtype=jnp.int32)
    index = jnp.broadcast_to(label[..., None, :, None], logp.shape[:-1] + (1,))
    return jnp.take_along_axis(logp, index, axis=-1)[..., 0]


def response_joint_log_probability(
    prediction: ResponsePrediction, target: ResponseTarget
) -> Any:
    """Compute log p(y | z=k,H,a) with one shared component per response."""

    import jax.numpy as jnp

    visible = jnp.asarray(target.visible_mask, dtype=jnp.float32)[..., None]
    event_mask = jnp.asarray(target.event_mask, dtype=jnp.float32)[..., None]
    logp = _bernoulli_log_probability(
        prediction.visibility_logit, jnp.asarray(target.visibility)[..., None]
    )
    logp = logp + visible * _categorical_log_probability(
        prediction.relative_position_logits, target.relative_position
    )
    logp = logp + visible * _categorical_log_probability(
        prediction.direction_logits, target.direction
    )
    logp = logp + visible * jnp.sum(
        _factor_categorical_log_probability(prediction.inventory_logits, target.inventory),
        axis=-1,
    )
    logp = logp + event_mask * _bernoulli_log_probability(
        prediction.inventory_change_logit,
        jnp.asarray(target.inventory_change)[..., None],
    )
    return logp



__all__ = [
    "init_response_params",
    "response_joint_log_probability",
    "response_predict",
]
