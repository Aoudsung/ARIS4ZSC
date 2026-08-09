"""Episode-static shared-latent coordination model for DELTA v4.

Immediate semantic responses update the legal posterior. Shared occurrence
factors are predicted but cannot change component responsibilities. Delayed
probe responses and sparse current/successor decision observations train the
same component semantics without entering the online filter state.
"""

from __future__ import annotations

from typing import Any

from .behavior_statistics import behavior_features, update_behavior_statistics
from .belief_filter import episode_static_prior, filter_update
from .belief_value import init_belief_value_params
from .successor_feature import init_successor_feature_params
from .nn import tree_stop_gradient
from .observation import extract_response_target
from .response_model import (
    init_probe_response_params,
    init_response_params,
    probe_response_predict,
    response_predict,
    response_semantic_component_log_probability,
    response_shared_log_probability,
)


def init_latent_params(
    key: Any,
    *,
    observation_shape: tuple[int, ...],
    component_count: int,
    component_embedding_dim: int,
    task_dim: int,
    instant_dim: int,
    behavior_dim: int,
    action_count: int,
    action_embedding_dim: int,
    hidden_dim: int,
    value_ensemble_size: int = 4,
    semantic_event_bias: Any | None = None,
) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    keys = jax.random.split(key, 6)
    ingredient_count = (int(observation_shape[-1]) - 27) // 4
    inventory_factor_count = ingredient_count + 2
    return {
        "component_embeddings": jax.random.normal(
            keys[0], (component_count, component_embedding_dim), dtype=jnp.float32
        ) / max(component_embedding_dim, 1) ** 0.5,
        "response": init_response_params(
            keys[1],
            observation_shape=observation_shape,
            behavior_dim=behavior_dim,
            component_count=component_count,
            component_embedding_dim=component_embedding_dim,
            action_count=action_count,
            action_embedding_dim=action_embedding_dim,
            hidden_dim=hidden_dim,
            inventory_factor_count=inventory_factor_count,
            semantic_event_bias=semantic_event_bias,
        ),
        "probe_response": init_probe_response_params(
            keys[2],
            observation_shape=observation_shape,
            behavior_dim=behavior_dim,
            component_count=component_count,
            component_embedding_dim=component_embedding_dim,
            action_count=action_count,
            action_embedding_dim=action_embedding_dim,
            hidden_dim=hidden_dim,
            semantic_event_bias=semantic_event_bias,
        ),
        "belief_value": init_belief_value_params(
            keys[4],
            task_dim=task_dim,
            instant_dim=instant_dim,
            behavior_dim=behavior_dim,
            component_count=component_count,
            hidden_dim=hidden_dim,
            action_count=action_count,
            ensemble_size=value_ensemble_size,
        ),
        "successor_feature": init_successor_feature_params(
            keys[5],
            task_dim=task_dim,
            instant_dim=instant_dim,
            behavior_dim=behavior_dim,
            component_count=component_count,
            action_count=action_count,
            action_embedding_dim=action_embedding_dim,
            hidden_dim=hidden_dim,
            ensemble_size=value_ensemble_size,
        ),
    }


def observe_response(
    params: dict[str, Any],
    belief: Any,
    statistics: Any,
    previous_observation: Any,
    observation: Any,
    previous_action: Any,
    episode_start: Any,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Apply one immediate semantic Bayes update and update legal statistics."""

    import jax.numpy as jnp
    import jax.scipy as jsp

    start = jnp.asarray(episode_start, dtype=jnp.bool_)
    component_count = int(params["component_embeddings"].shape[0])
    prior = episode_static_prior(belief, start, component_count)
    target = extract_response_target(
        previous_observation, observation, previous_action, start
    )
    before_features = behavior_features(statistics)
    prediction = response_predict(
        params["response"],
        params["component_embeddings"],
        previous_observation,
        before_features,
        previous_action,
    )
    shared_logp = response_shared_log_probability(prediction, target)
    semantic_logp = response_semantic_component_log_probability(prediction, target)
    marginal_semantic_logp = jsp.special.logsumexp(
        jnp.log(jnp.maximum(prior, 1.0e-30)) + semantic_logp, axis=-1
    )
    filtered = filter_update(prior, semantic_logp)
    next_belief = jnp.where(start[..., None], prior, filtered)
    next_statistics = update_behavior_statistics(
        statistics, target, episode_start=start
    )
    negative_log_likelihood = jnp.where(
        start, 0.0, -(shared_logp + marginal_semantic_logp)
    )
    return (
        next_belief,
        next_statistics,
        negative_log_likelihood,
        prediction,
        target,
        prior,
    )


__all__ = [
    "init_latent_params",
    "observe_response",
    "predict_probe_response",
]
