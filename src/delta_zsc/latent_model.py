"""Unified latent coordination model.

One exchangeable latent state jointly emits the observable teammate response
and the counterfactual action-value signature. Training minimizes one
shared-latent composite predictive likelihood under the response-only belief; there is no comparator, separation loss, pseudo-label
critic, capability encoder, or auxiliary actor target.
"""

from __future__ import annotations

from typing import Any

from .behavior_statistics import behavior_features, update_behavior_statistics
from .belief_filter import filter_update, uniform_belief
from .decision_model import decision_predict, init_decision_params
from .nn import tree_stop_gradient
from .observation import extract_response_target
from .response_model import init_response_params, response_joint_log_probability, response_predict
from .transition import init_transition_logits, predict_belief


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
) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    keys = jax.random.split(key, 4)
    ingredient_count = (int(observation_shape[-1]) - 27) // 4
    inventory_factor_count = ingredient_count + 2
    return {
        "component_embeddings": jax.random.normal(
            keys[0], (component_count, component_embedding_dim), dtype=jnp.float32
        ) / max(component_embedding_dim, 1) ** 0.5,
        "transition_logits": init_transition_logits(keys[1], component_count),
        "response": init_response_params(
            keys[2],
            observation_shape=observation_shape,
            behavior_dim=behavior_dim,
            component_embedding_dim=component_embedding_dim,
            action_count=action_count,
            action_embedding_dim=action_embedding_dim,
            hidden_dim=hidden_dim,
            inventory_factor_count=inventory_factor_count,
        ),
        "decision": init_decision_params(
            keys[3],
            task_dim=task_dim,
            instant_dim=instant_dim,
            behavior_dim=behavior_dim,
            component_embedding_dim=component_embedding_dim,
            hidden_dim=hidden_dim,
            action_count=action_count,
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
) -> tuple[Any, Any, Any, Any, Any]:
    """Perform one response likelihood update and sufficient-statistic update."""

    import jax.scipy as jsp
    import jax.numpy as jnp

    start = jnp.asarray(episode_start, dtype=jnp.bool_)
    component_count = int(params["component_embeddings"].shape[0])
    prior = uniform_belief(start.shape, component_count)
    previous_belief = jnp.where(start[..., None], prior, belief)
    response_target = extract_response_target(previous_observation, observation)
    before_features = behavior_features(statistics)
    prediction = response_predict(
        params["response"],
        params["component_embeddings"],
        previous_observation,
        before_features,
        previous_action,
    )
    component_logp = response_joint_log_probability(prediction, response_target)
    predictive = predict_belief(previous_belief, params["transition_logits"])
    marginal_logp = jsp.special.logsumexp(
        jnp.log(jnp.maximum(predictive, 1.0e-30)) + component_logp,
        axis=-1,
    )
    filtered = filter_update(
        previous_belief, params["transition_logits"], component_logp
    )
    next_belief = jnp.where(start[..., None], prior, filtered)
    next_statistics = update_behavior_statistics(
        statistics, response_target, episode_start=start
    )
    negative_log_likelihood = jnp.where(start, 0.0, -marginal_logp)
    return (
        next_belief,
        next_statistics,
        negative_log_likelihood,
        prediction,
        response_target,
    )


def predict_decision(
    params: dict[str, Any],
    task_features: Any,
    instant_partner: Any,
    statistics: Any,
) -> Any:
    return decision_predict(
        params["decision"],
        params["component_embeddings"],
        tree_stop_gradient(task_features),
        tree_stop_gradient(instant_partner),
        behavior_features(statistics),
    )


__all__ = ["init_latent_params", "observe_response", "predict_decision"]
