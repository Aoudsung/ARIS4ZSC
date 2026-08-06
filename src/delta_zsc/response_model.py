"""Action-conditioned response emission for the shared latent coordination mode."""

from __future__ import annotations

from typing import Any

from .nn import init_linear, init_mlp, linear, mlp
from .observation import (
    INTERFACE_EVENT_CLASSES,
    PARTNER_DIRECTION_CLASSES,
    PARTNER_INVENTORY_FACTOR_CLASSES,
    PARTNER_POSITION_CLASSES,
)
from .types import DirectResponsePrediction, ResponsePrediction, ResponseTarget


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

    keys = jax.random.split(key, 16)
    frame_dim = int(observation_shape[0] * observation_shape[1] * observation_shape[2])
    trunk_input = hidden_dim + behavior_dim + component_embedding_dim + action_embedding_dim
    return {
        "frame_encoder": init_mlp(keys[0], (frame_dim, hidden_dim, hidden_dim)),
        "action_embedding": jax.random.normal(
            keys[1], (int(action_count), int(action_embedding_dim))
        ) / max(action_embedding_dim, 1) ** 0.5,
        "trunk": init_mlp(keys[2], (trunk_input, hidden_dim, hidden_dim)),
        "availability_trunk": init_mlp(
            keys[8],
            (hidden_dim + behavior_dim + action_embedding_dim, hidden_dim, hidden_dim),
        ),
        "availability": init_linear(keys[9], hidden_dim, 1, scale=0.01),
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
        "interface_change": init_linear(keys[10], hidden_dim, 1, scale=0.01),
        "interface_facility": init_linear(keys[11], hidden_dim, 3, scale=0.01),
        "interface_direction": init_linear(keys[12], hidden_dim, 2, scale=0.01),
        "interface_object": init_linear(keys[13], hidden_dim, 5, scale=0.01),
        "interface_other": init_linear(keys[14], hidden_dim, 1, scale=0.01),
        "recipe_change": init_linear(keys[15], hidden_dim, 1, scale=0.01),
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
    facility = linear(params["interface_facility"], hidden)
    change_direction = linear(params["interface_direction"], hidden)
    object_kind = linear(params["interface_object"], hidden)
    structured = (
        facility[..., :, None, None]
        + change_direction[..., None, :, None]
        + object_kind[..., None, None, :]
    )
    structured = structured.reshape(lead + (count, INTERFACE_EVENT_CLASSES - 1))
    interface_event = jnp.concatenate(
        (structured, linear(params["interface_other"], hidden)), axis=-1
    )
    shared_hidden = mlp(
        params["availability_trunk"],
        jnp.concatenate((frame_encoded, behavior_features, action_emb), axis=-1),
        final_activation=True,
    )
    return ResponsePrediction(
        direct=DirectResponsePrediction(
            visibility_logit=linear(params["visibility"], hidden)[..., 0],
            relative_position_logits=linear(params["position"], hidden),
            direction_logits=linear(params["direction"], hidden),
            inventory_logits=inventory,
            inventory_change_logit=linear(params["event"], hidden)[..., 0],
        ),
        interface_availability_logit=linear(params["availability"], shared_hidden)[..., 0],
        interface_change_logit=linear(params["interface_change"], hidden)[..., 0],
        interface_event_logits=interface_event,
        recipe_change_logit=linear(params["recipe_change"], hidden)[..., 0],
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

    factors = response_factor_log_probabilities(prediction, target)
    return sum(factors.values())


def response_factor_log_probabilities(
    prediction: ResponsePrediction, target: ResponseTarget
) -> dict[str, Any]:
    """Return additive complete-response factors for diagnostic decomposition."""

    import jax.numpy as jnp

    direct = target.direct
    predicted_direct = prediction.direct
    visible = jnp.asarray(direct.visible_mask, dtype=jnp.float32)[..., None]
    event_mask = jnp.asarray(direct.event_mask, dtype=jnp.float32)[..., None]
    visibility = _bernoulli_log_probability(
        predicted_direct.visibility_logit, jnp.asarray(direct.visibility)[..., None]
    )
    position = visible * _categorical_log_probability(
        predicted_direct.relative_position_logits, direct.relative_position
    )
    direction = visible * _categorical_log_probability(
        predicted_direct.direction_logits, direct.direction
    )
    inventory = visible * jnp.sum(
        _factor_categorical_log_probability(predicted_direct.inventory_logits, direct.inventory),
        axis=-1,
    )
    inventory_change = event_mask * _bernoulli_log_probability(
        predicted_direct.inventory_change_logit,
        jnp.asarray(direct.inventory_change)[..., None],
    )
    available = jnp.asarray(target.interface_available, dtype=jnp.float32)[..., None]
    changed = jnp.asarray(target.interface_changed, dtype=jnp.float32)[..., None]
    # Availability is deliberately shared (no K axis), hence cannot alter the
    # normalized component posterior even though it remains part of the score.
    availability = _bernoulli_log_probability(
        prediction.interface_availability_logit,
        target.interface_available,
    )[..., None]
    interface_change = available * _bernoulli_log_probability(
        prediction.interface_change_logit, changed
    )
    interface_event = available * changed * _categorical_log_probability(
        prediction.interface_event_logits, target.interface_event
    )
    recipe_mask = jnp.asarray(target.recipe_mask, dtype=jnp.float32)[..., None]
    recipe = recipe_mask * _bernoulli_log_probability(
        prediction.recipe_change_logit,
        jnp.asarray(target.recipe_changed)[..., None],
    )
    return {
        "visibility": visibility,
        "position": position,
        "direction": direction,
        "inventory": inventory,
        "inventory_change": inventory_change,
        "interface_availability": availability,
        "interface_change": interface_change,
        "interface_event": interface_event,
        "recipe_change": recipe,
    }



__all__ = [
    "init_response_params",
    "response_joint_log_probability",
    "response_factor_log_probabilities",
    "response_predict",
]
