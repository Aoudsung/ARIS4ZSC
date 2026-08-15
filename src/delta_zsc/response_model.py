"""DELTA v6 immediate and delayed response emissions.

Occurrence factors are shared across latent components.  Conditional semantic
factors use a pooled baseline plus a centered component residual, with both a
context path and a direct component-embedding skip.  This prevents high-rate,
partner-independent no-change events from creating a global posterior winner
while retaining a normalized predictive model of the complete legal response.
"""

from __future__ import annotations

from typing import Any, Mapping

from .nn import init_linear, init_mlp, linear, mlp
from .observation import (
    INTERFACE_EVENT_CLASSES,
    PARTNER_DIRECTION_CLASSES,
    PARTNER_INVENTORY_FACTOR_CLASSES,
    PARTNER_POSITION_CLASSES,
)
from .semantic_initializer import random_orthogonal_simplex_bias
from .types import (
    DirectResponsePrediction,
    ProbeResponsePrediction,
    ProbeResponseTarget,
    ResponsePrediction,
    ResponseTarget,
)


def _event_shared_logits(params: Mapping[str, Any], hidden: Any) -> Any:
    """Return one normalized 31-class pooled event-logit vector."""

    import jax.numpy as jnp

    facility = linear(params["event_shared_facility"], hidden)
    direction = linear(params["event_shared_direction"], hidden)
    object_kind = linear(params["event_shared_object"], hidden)
    structured = (
        facility[..., :, None, None]
        + direction[..., None, :, None]
        + object_kind[..., None, None, :]
    ).reshape(hidden.shape[:-1] + (INTERFACE_EVENT_CLASSES - 1,))
    return jnp.concatenate(
        (structured, linear(params["event_shared_other"], hidden)), axis=-1
    )


def _broadcast_components(value: Any, count: int) -> Any:
    import jax.numpy as jnp

    array = jnp.asarray(value)
    return jnp.broadcast_to(
        array[..., None, :], array.shape[:-1] + (int(count), array.shape[-1])
    )


def _component_embeddings(components: Any, lead: tuple[int, ...]) -> Any:
    import jax.numpy as jnp

    value = jnp.asarray(components, dtype=jnp.float32)
    return jnp.broadcast_to(
        value.reshape((1,) * len(lead) + value.shape), lead + value.shape
    )


def _center_components(value: Any) -> Any:
    import jax.numpy as jnp

    array = jnp.asarray(value, dtype=jnp.float32)
    return array - jnp.mean(array, axis=-2, keepdims=True)


def _semantic_logits(
    *,
    shared_logits: Any,
    context_residual: Any,
    embedding_residual: Any,
    component_bias: Any | None = None,
) -> Any:
    """Combine a pooled baseline with a zero-mean component residual."""

    import jax.numpy as jnp

    raw = jnp.asarray(context_residual, dtype=jnp.float32) + jnp.asarray(
        embedding_residual, dtype=jnp.float32
    )
    if component_bias is not None:
        bias = jnp.asarray(component_bias, dtype=jnp.float32)
        raw = raw + jnp.broadcast_to(
            bias.reshape((1,) * (raw.ndim - bias.ndim) + bias.shape), raw.shape
        )
    residual = _center_components(raw)
    return jnp.asarray(shared_logits, dtype=jnp.float32)[..., None, :] + residual


def _init_semantic_head(
    keys: tuple[Any, ...],
    *,
    hidden_dim: int,
    component_embedding_dim: int,
    output_dim: int,
    name: str,
) -> dict[str, Any]:
    return {
        f"{name}_shared": init_linear(keys[0], hidden_dim, output_dim, scale=0.01),
        f"{name}_residual_context": init_linear(
            keys[1], hidden_dim, output_dim, scale=1.0
        ),
        f"{name}_residual_embedding": init_linear(
            keys[2], component_embedding_dim, output_dim, scale=1.0
        ),
    }


def _init_event_head(
    keys: tuple[Any, ...],
    *,
    hidden_dim: int,
    component_embedding_dim: int,
    component_count: int,
    semantic_event_bias: Any | None,
) -> dict[str, Any]:
    import jax.numpy as jnp

    if semantic_event_bias is None:
        bias = random_orthogonal_simplex_bias(
            keys[7],
            component_count=component_count,
            event_classes=INTERFACE_EVENT_CLASSES,
        )
    else:
        bias = jnp.asarray(semantic_event_bias, dtype=jnp.float32)
        if bias.shape != (int(component_count), INTERFACE_EVENT_CLASSES):
            raise ValueError("Semantic event initializer has the wrong shape.")
        bias = bias - jnp.mean(bias, axis=0, keepdims=True)
    return {
        "event_shared_facility": init_linear(keys[0], hidden_dim, 3, scale=0.01),
        "event_shared_direction": init_linear(keys[1], hidden_dim, 2, scale=0.01),
        "event_shared_object": init_linear(keys[2], hidden_dim, 5, scale=0.01),
        "event_shared_other": init_linear(keys[3], hidden_dim, 1, scale=0.01),
        "event_residual_context": init_linear(
            keys[4], hidden_dim, INTERFACE_EVENT_CLASSES, scale=1.0
        ),
        "event_residual_embedding": init_linear(
            keys[5], component_embedding_dim, INTERFACE_EVENT_CLASSES, scale=1.0
        ),
        "event_component_bias": bias,
    }


def init_response_params(
    key: Any,
    *,
    observation_shape: tuple[int, ...],
    behavior_dim: int,
    component_count: int,
    component_embedding_dim: int,
    action_count: int,
    action_embedding_dim: int,
    hidden_dim: int,
    inventory_factor_count: int,
    semantic_event_bias: Any | None = None,
) -> dict[str, Any]:
    """Initialize the immediate passive response model."""

    import jax

    keys = tuple(jax.random.split(key, 35))
    frame_dim = int(observation_shape[0] * observation_shape[1] * observation_shape[2])
    shared_input = hidden_dim + behavior_dim + action_embedding_dim
    component_input = hidden_dim + component_embedding_dim
    params: dict[str, Any] = {
        "frame_encoder": init_mlp(keys[0], (frame_dim, hidden_dim, hidden_dim)),
        "action_embedding": jax.random.normal(
            keys[1], (int(action_count), int(action_embedding_dim))
        ) / max(action_embedding_dim, 1) ** 0.5,
        "shared_trunk": init_mlp(keys[2], (shared_input, hidden_dim, hidden_dim)),
        "component_trunk": init_mlp(
            keys[3], (component_input, hidden_dim, hidden_dim)
        ),
        # Shared occurrence heads: these are scored but never enter Bayes ratios.
        "visibility": init_linear(keys[4], hidden_dim, 1, scale=0.01),
        "inventory_change": init_linear(keys[5], hidden_dim, 1, scale=0.01),
        "availability": init_linear(keys[6], hidden_dim, 1, scale=0.01),
        "interface_change": init_linear(keys[7], hidden_dim, 1, scale=0.01),
        "recipe_change": init_linear(keys[8], hidden_dim, 1, scale=0.01),
    }
    params.update(
        _init_semantic_head(
            keys[9:12],
            hidden_dim=hidden_dim,
            component_embedding_dim=component_embedding_dim,
            output_dim=PARTNER_POSITION_CLASSES,
            name="position",
        )
    )
    params.update(
        _init_semantic_head(
            keys[12:15],
            hidden_dim=hidden_dim,
            component_embedding_dim=component_embedding_dim,
            output_dim=PARTNER_DIRECTION_CLASSES,
            name="direction",
        )
    )
    inventory_dim = int(inventory_factor_count) * PARTNER_INVENTORY_FACTOR_CLASSES
    params.update(
        _init_semantic_head(
            keys[15:18],
            hidden_dim=hidden_dim,
            component_embedding_dim=component_embedding_dim,
            output_dim=inventory_dim,
            name="inventory",
        )
    )
    params.update(
        _init_event_head(
            keys[18:26],
            hidden_dim=hidden_dim,
            component_embedding_dim=component_embedding_dim,
            component_count=component_count,
            semantic_event_bias=semantic_event_bias,
        )
    )
    return params


def init_probe_response_params(
    key: Any,
    *,
    observation_shape: tuple[int, ...],
    behavior_dim: int,
    component_count: int,
    component_embedding_dim: int,
    action_count: int,
    action_embedding_dim: int,
    hidden_dim: int,
    semantic_event_bias: Any | None = None,
) -> dict[str, Any]:
    """Initialize the two-step delayed probe-response model."""

    import jax

    keys = tuple(jax.random.split(key, 20))
    frame_dim = int(observation_shape[0] * observation_shape[1] * observation_shape[2])
    return {
        "frame_encoder": init_mlp(keys[0], (frame_dim, hidden_dim, hidden_dim)),
        "action_embedding": jax.random.normal(
            keys[1], (int(action_count), int(action_embedding_dim))
        ) / max(action_embedding_dim, 1) ** 0.5,
        "shared_trunk": init_mlp(
            keys[2],
            (hidden_dim + behavior_dim + action_embedding_dim, hidden_dim, hidden_dim),
        ),
        "component_trunk": init_mlp(
            keys[3], (hidden_dim + component_embedding_dim, hidden_dim, hidden_dim)
        ),
        "visibility": init_linear(keys[4], hidden_dim, 1, scale=0.01),
        "availability": init_linear(keys[5], hidden_dim, 1, scale=0.01),
        "interface_change": init_linear(keys[6], hidden_dim, 1, scale=0.01),
        **_init_event_head(
            keys[7:15],
            hidden_dim=hidden_dim,
            component_embedding_dim=component_embedding_dim,
            component_count=component_count,
            semantic_event_bias=semantic_event_bias,
        ),
    }


def _encoded_context(
    params: Mapping[str, Any], frame: Any, behavior: Any, actions: Any
) -> tuple[Any, Any, tuple[int, ...]]:
    import jax.numpy as jnp

    observation = jnp.asarray(frame, dtype=jnp.float32)
    action = jnp.asarray(actions, dtype=jnp.int32)
    behavior_features = jnp.asarray(behavior, dtype=jnp.float32)
    lead = action.shape
    if observation.shape[:-3] != lead or behavior_features.shape[:-1] != lead:
        raise ValueError("Response batch axes differ.")
    frame_encoded = mlp(
        params["frame_encoder"], observation.reshape(lead + (-1,)), final_activation=True
    )
    action_embedding = params["action_embedding"][action]
    shared = mlp(
        params["shared_trunk"],
        jnp.concatenate((frame_encoded, behavior_features, action_embedding), axis=-1),
        final_activation=True,
    )
    return shared, action_embedding, lead


def _component_hidden(
    params: Mapping[str, Any], shared: Any, component_embeddings: Any, lead: tuple[int, ...]
) -> tuple[Any, Any]:
    import jax.numpy as jnp

    components = jnp.asarray(component_embeddings, dtype=jnp.float32)
    count = int(components.shape[0])
    comp = _component_embeddings(components, lead)
    hidden = mlp(
        params["component_trunk"],
        jnp.concatenate((_broadcast_components(shared, count), comp), axis=-1),
        final_activation=True,
    )
    return hidden, comp


def _predict_event(
    params: Mapping[str, Any], shared: Any, hidden: Any, comp: Any
) -> Any:
    shared_logits = _event_shared_logits(params, shared)
    context = linear(params["event_residual_context"], hidden)
    embedding = linear(params["event_residual_embedding"], comp)
    return _semantic_logits(
        shared_logits=shared_logits,
        context_residual=context,
        embedding_residual=embedding,
        component_bias=params["event_component_bias"],
    )


def response_predict(
    params: Mapping[str, Any],
    component_embeddings: Any,
    frame: Any,
    behavior: Any,
    actions: Any,
) -> ResponsePrediction:
    """Return the immediate shared-occurrence/component-semantic response."""

    import jax.numpy as jnp

    shared, _, lead = _encoded_context(params, frame, behavior, actions)
    hidden, comp = _component_hidden(params, shared, component_embeddings, lead)
    count = int(jnp.asarray(component_embeddings).shape[0])

    def semantic(name: str) -> Any:
        return _semantic_logits(
            shared_logits=linear(params[f"{name}_shared"], shared),
            context_residual=linear(params[f"{name}_residual_context"], hidden),
            embedding_residual=linear(params[f"{name}_residual_embedding"], comp),
        )

    inventory_flat = semantic("inventory")
    inventory = inventory_flat.reshape(
        lead
        + (
            count,
            int(inventory_flat.shape[-1] // PARTNER_INVENTORY_FACTOR_CLASSES),
            PARTNER_INVENTORY_FACTOR_CLASSES,
        )
    )
    return ResponsePrediction(
        direct=DirectResponsePrediction(
            visibility_logit=linear(params["visibility"], shared)[..., 0],
            relative_position_logits=semantic("position"),
            direction_logits=semantic("direction"),
            inventory_logits=inventory,
            inventory_change_logit=linear(params["inventory_change"], shared)[..., 0],
        ),
        interface_availability_logit=linear(params["availability"], shared)[..., 0],
        interface_change_logit=linear(params["interface_change"], shared)[..., 0],
        interface_event_logits=_predict_event(params, shared, hidden, comp),
        recipe_change_logit=linear(params["recipe_change"], shared)[..., 0],
    )


def probe_response_predict(
    params: Mapping[str, Any],
    component_embeddings: Any,
    frame: Any,
    behavior: Any,
    probe_actions: Any,
) -> ProbeResponsePrediction:
    """Predict the response observable one partner reaction after a probe."""

    shared, _, lead = _encoded_context(params, frame, behavior, probe_actions)
    hidden, comp = _component_hidden(params, shared, component_embeddings, lead)
    return ProbeResponsePrediction(
        visibility_logit=linear(params["visibility"], shared)[..., 0],
        interface_availability_logit=linear(params["availability"], shared)[..., 0],
        interface_change_logit=linear(params["interface_change"], shared)[..., 0],
        interface_event_logits=_predict_event(params, shared, hidden, comp),
    )


def _bernoulli_log_probability(logits: Any, labels: Any) -> Any:
    import jax.nn as jnn
    import jax.numpy as jnp

    target = jnp.asarray(labels, dtype=jnp.float32)
    return target * jnn.log_sigmoid(logits) + (1.0 - target) * jnn.log_sigmoid(-logits)


def _component_categorical_log_probability(logits: Any, labels: Any) -> Any:
    import jax.nn as jnn
    import jax.numpy as jnp

    logp = jnn.log_softmax(logits, axis=-1)
    label = jnp.asarray(labels, dtype=jnp.int32)
    index = jnp.broadcast_to(label[..., None, None], logp.shape[:-1] + (1,))
    return jnp.take_along_axis(logp, index, axis=-1)[..., 0]


def _component_factor_categorical_log_probability(logits: Any, labels: Any) -> Any:
    import jax.nn as jnn
    import jax.numpy as jnp

    logp = jnn.log_softmax(logits, axis=-1)
    label = jnp.asarray(labels, dtype=jnp.int32)
    index = jnp.broadcast_to(label[..., None, :, None], logp.shape[:-1] + (1,))
    return jnp.take_along_axis(logp, index, axis=-1)[..., 0]


def response_shared_factor_log_probabilities(
    prediction: ResponsePrediction, target: ResponseTarget
) -> dict[str, Any]:
    """Return partner-independent occurrence scores with no component axis."""

    import jax.numpy as jnp

    direct = target.direct
    available = jnp.asarray(target.interface_available, dtype=jnp.float32)
    recipe_mask = jnp.asarray(target.recipe_mask, dtype=jnp.float32)
    return {
        "visibility": _bernoulli_log_probability(
            prediction.direct.visibility_logit, direct.visibility
        ),
        "inventory_change": jnp.asarray(direct.event_mask, dtype=jnp.float32)
        * _bernoulli_log_probability(
            prediction.direct.inventory_change_logit, direct.inventory_change
        ),
        "interface_availability": _bernoulli_log_probability(
            prediction.interface_availability_logit, target.interface_available
        ),
        "interface_change": available
        * _bernoulli_log_probability(
            prediction.interface_change_logit, target.interface_changed
        ),
        "recipe_change": recipe_mask
        * _bernoulli_log_probability(
            prediction.recipe_change_logit, target.recipe_changed
        ),
    }


def response_semantic_factor_log_probabilities(
    prediction: ResponsePrediction, target: ResponseTarget
) -> dict[str, Any]:
    """Return the only immediate factors permitted to change the posterior."""

    import jax.numpy as jnp

    direct = target.direct
    visible = jnp.asarray(direct.visible_mask, dtype=jnp.float32)[..., None]
    event_mask = (
        jnp.asarray(target.interface_available, dtype=jnp.float32)
        * jnp.asarray(target.interface_changed, dtype=jnp.float32)
    )[..., None]
    return {
        "position": visible
        * _component_categorical_log_probability(
            prediction.direct.relative_position_logits, direct.relative_position
        ),
        "direction": visible
        * _component_categorical_log_probability(
            prediction.direct.direction_logits, direct.direction
        ),
        "inventory": visible
        * jnp.sum(
            _component_factor_categorical_log_probability(
                prediction.direct.inventory_logits, direct.inventory
            ),
            axis=-1,
        ),
        "interface_event": event_mask
        * _component_categorical_log_probability(
            prediction.interface_event_logits, target.interface_event
        ),
    }


def response_shared_log_probability(
    prediction: ResponsePrediction, target: ResponseTarget
) -> Any:
    return sum(response_shared_factor_log_probabilities(prediction, target).values())


def response_semantic_component_log_probability(
    prediction: ResponsePrediction, target: ResponseTarget
) -> Any:
    return sum(response_semantic_factor_log_probabilities(prediction, target).values())


def response_joint_log_probability(
    prediction: ResponsePrediction, target: ResponseTarget
) -> Any:
    """Return full predictive log probability under every component."""

    return response_shared_log_probability(prediction, target)[..., None] + (
        response_semantic_component_log_probability(prediction, target)
    )


def response_factor_log_probabilities(
    prediction: ResponsePrediction, target: ResponseTarget
) -> dict[str, Any]:
    """Diagnostic factor map; shared factors are broadcast but never filtered."""

    import jax.numpy as jnp

    semantic = response_semantic_factor_log_probabilities(prediction, target)
    count = int(prediction.interface_event_logits.shape[-2])
    shared = {
        name: jnp.broadcast_to(value[..., None], value.shape + (count,))
        for name, value in response_shared_factor_log_probabilities(
            prediction, target
        ).items()
    }
    return {**shared, **semantic}


def probe_response_shared_factor_log_probabilities(
    prediction: ProbeResponsePrediction, target: ProbeResponseTarget
) -> dict[str, Any]:
    import jax.numpy as jnp

    valid = jnp.asarray(target.valid_mask, dtype=jnp.float32)
    available = jnp.asarray(target.interface_available, dtype=jnp.float32)
    return {
        "visibility": valid
        * _bernoulli_log_probability(prediction.visibility_logit, target.visibility),
        "interface_availability": valid
        * _bernoulli_log_probability(
            prediction.interface_availability_logit, target.interface_available
        ),
        "interface_change": valid
        * available
        * _bernoulli_log_probability(
            prediction.interface_change_logit, target.interface_changed
        ),
    }


def probe_response_semantic_component_log_probability(
    prediction: ProbeResponsePrediction, target: ProbeResponseTarget
) -> Any:
    import jax.numpy as jnp

    mask = (
        jnp.asarray(target.valid_mask, dtype=jnp.float32)
        * jnp.asarray(target.interface_available, dtype=jnp.float32)
        * jnp.asarray(target.interface_changed, dtype=jnp.float32)
    )[..., None]
    return mask * _component_categorical_log_probability(
        prediction.interface_event_logits, target.interface_event
    )


def probe_response_shared_log_probability(
    prediction: ProbeResponsePrediction, target: ProbeResponseTarget
) -> Any:
    return sum(
        probe_response_shared_factor_log_probabilities(prediction, target).values()
    )


def probe_response_joint_log_probability(
    prediction: ProbeResponsePrediction, target: ProbeResponseTarget
) -> Any:
    return probe_response_shared_log_probability(prediction, target)[..., None] + (
        probe_response_semantic_component_log_probability(prediction, target)
    )


__all__ = [
    "init_probe_response_params",
    "init_response_params",
    "probe_response_joint_log_probability",
    "probe_response_predict",
    "probe_response_semantic_component_log_probability",
    "probe_response_shared_factor_log_probabilities",
    "probe_response_shared_log_probability",
    "response_factor_log_probabilities",
    "response_joint_log_probability",
    "response_predict",
    "response_semantic_component_log_probability",
    "response_semantic_factor_log_probabilities",
    "response_shared_factor_log_probabilities",
    "response_shared_log_probability",
]
