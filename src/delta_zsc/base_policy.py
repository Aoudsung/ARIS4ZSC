"""Task-competence policy trained only by on-policy PPO."""

from __future__ import annotations

from typing import Any

from .nn import gru_step, init_gru, init_linear, init_mlp, layer_normalize, linear, mlp
from .observation import instantaneous_partner_observation, task_only_observation


def init_base_params(
    key: Any,
    *,
    observation_shape: tuple[int, ...],
    task_hidden_dim: int,
    task_embedding_dim: int,
    instant_partner_dim: int,
    action_count: int,
    component_count: int,
) -> dict[str, Any]:
    import jax

    keys = jax.random.split(key, 6)
    frame_dim = int(observation_shape[0] * observation_shape[1] * observation_shape[2])
    # Derive the instantaneous branch size by applying the pinned channel rule.
    from .observation import partner_channel_indexes

    partner_dim = int(
        observation_shape[0]
        * observation_shape[1]
        * len(partner_channel_indexes(observation_shape[-1]))
    )
    # The actor reads the response-only posterior alongside the task features.
    # Without it PPO optimises a single policy against the *marginal* over
    # partners, and protocol-dependent actions cancel in the averaged advantage.
    combined = int(task_hidden_dim) + int(instant_partner_dim) + int(component_count)
    return {
        "task_encoder": init_mlp(
            keys[0], (frame_dim, int(task_embedding_dim), int(task_embedding_dim))
        ),
        "task_gru": init_gru(keys[1], int(task_embedding_dim), int(task_hidden_dim)),
        "instant_encoder": init_mlp(
            keys[2], (partner_dim, int(instant_partner_dim), int(instant_partner_dim))
        ),
        "actor_trunk": init_mlp(
            keys[3], (combined, int(task_hidden_dim), int(task_hidden_dim))
        ),
        "actor": init_linear(keys[4], int(task_hidden_dim), int(action_count), scale=0.01),
        "value": init_linear(keys[5], int(task_hidden_dim), 1, scale=1.0),
    }


def base_policy_step(
    params: dict[str, Any],
    task_carry: Any,
    observation: Any,
    episode_start: Any,
    belief: Any,
    *,
    mask_partner_history: bool,
) -> tuple[Any, Any, Any, Any, Any]:
    """Advance the actor one step conditioned on the current partner posterior.

    ``belief`` is the response-only posterior ``b_t = P(z | h_t)``.  It arrives
    already detached: PPO trains only ``base_params`` through it, so the
    two-estimator boundary is unchanged.
    """

    import jax.numpy as jnp

    frame = (
        task_only_observation(observation)
        if bool(mask_partner_history)
        else jnp.asarray(observation, dtype=jnp.float32)
    )
    start = jnp.asarray(episode_start, dtype=jnp.bool_)
    carry = jnp.where(start[..., None], jnp.zeros_like(task_carry), task_carry)
    task_flat = frame.reshape(frame.shape[:-3] + (-1,))
    task_embedding = layer_normalize(
        mlp(params["task_encoder"], task_flat, final_activation=True)
    )
    next_carry = gru_step(params["task_gru"], carry, task_embedding)
    task_features = layer_normalize(next_carry)

    partner = instantaneous_partner_observation(observation)
    partner_flat = partner.reshape(partner.shape[:-3] + (-1,))
    instant = layer_normalize(
        mlp(params["instant_encoder"], partner_flat, final_activation=True)
    )
    posterior = jnp.asarray(belief, dtype=jnp.float32)
    hidden = mlp(
        params["actor_trunk"],
        jnp.concatenate((task_features, instant, posterior), axis=-1),
        final_activation=True,
    )
    logits = linear(params["actor"], hidden)
    value = linear(params["value"], hidden)[..., 0]
    return next_carry, task_features, instant, logits, value


def base_policy_sequence(
    params: dict[str, Any],
    initial_task_carry: Any,
    observations: Any,
    episode_starts: Any,
    beliefs: Any,
    *,
    mask_partner_history: bool,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Evaluate a time-major base policy with only the GRU left in a scan.

    The observation encoders and policy/value heads are stateless.  Evaluating
    them over the complete ``time x lane`` array is mathematically identical
    to calling :func:`base_policy_step` at every time, but turns hundreds of
    tiny matrix multiplications into a handful of large ones.  Only the task
    GRU has a genuine temporal dependency and therefore remains sequential.
    """

    import jax
    import jax.numpy as jnp

    observation = jnp.asarray(observations, dtype=jnp.float32)
    frame = (
        task_only_observation(observation)
        if bool(mask_partner_history)
        else observation
    )
    task_flat = frame.reshape(frame.shape[:-3] + (-1,))
    task_embeddings = layer_normalize(
        mlp(params["task_encoder"], task_flat, final_activation=True)
    )

    partner = instantaneous_partner_observation(observation)
    partner_flat = partner.reshape(partner.shape[:-3] + (-1,))
    instant = layer_normalize(
        mlp(params["instant_encoder"], partner_flat, final_activation=True)
    )

    def recurrent_step(carry: Any, values: tuple[Any, Any]):
        embedding, episode_start = values
        start = jnp.asarray(episode_start, dtype=jnp.bool_)
        reset_carry = jnp.where(start[..., None], jnp.zeros_like(carry), carry)
        next_carry = gru_step(params["task_gru"], reset_carry, embedding)
        return next_carry, next_carry

    final_carry, task_carries = jax.lax.scan(
        recurrent_step,
        initial_task_carry,
        (task_embeddings, episode_starts),
    )
    task_features = layer_normalize(task_carries)
    posterior = jnp.asarray(beliefs, dtype=jnp.float32)
    hidden = mlp(
        params["actor_trunk"],
        jnp.concatenate((task_features, instant, posterior), axis=-1),
        final_activation=True,
    )
    logits = linear(params["actor"], hidden)
    value = linear(params["value"], hidden)[..., 0]
    return final_carry, task_carries, task_features, instant, logits, value


__all__ = ["base_policy_sequence", "base_policy_step", "init_base_params"]
