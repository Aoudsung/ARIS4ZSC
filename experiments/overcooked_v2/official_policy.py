"""Official evaluator adapters exposing only the legal PPOPolicy surface."""

from __future__ import annotations

from typing import Any

from src.cetr_zsc.model import actor_step

from .deployment import Deployment, reset_deployment_state


def _tree_select(mask: Any, selected: Any, alternative: Any) -> Any:
    import jax
    import jax.numpy as jnp

    def one(left: Any, right: Any):
        value = jnp.asarray(mask, dtype=jnp.bool_)
        expanded = value.reshape(value.shape + (1,) * (jnp.ndim(left) - value.ndim))
        return jnp.where(expanded, left, right)

    return jax.tree_util.tree_map(one, selected, alternative)


def _compute_deployment_action(
    deployment: Deployment,
    obs: Any,
    done: Any,
    hstate: Any,
    key: Any,
) -> tuple[Any, Any]:
    import jax
    import jax.numpy as jnp

    observation = jnp.asarray(obs, dtype=jnp.float32)
    batched = observation.ndim == len(deployment.model.observation_shape) + 1
    if not batched:
        observation = observation[None]
    done_value = jnp.asarray(done, dtype=jnp.bool_)
    if done_value.ndim == 0:
        done_value = done_value[None]
    keys = jnp.asarray(key)
    fresh = reset_deployment_state(deployment, int(observation.shape[0]))
    current = _tree_select(done_value, fresh, hstate)
    next_carry, logits = actor_step(
        deployment.params,
        current.carry,
        observation,
        current.episode_start,
    )
    if keys.ndim == 1:
        keys = jax.random.split(keys, int(observation.shape[0]))
    action = jax.vmap(
        lambda current_key, current_logits: jax.random.categorical(
            current_key, current_logits
        )
    )(keys, logits)
    next_state = current._replace(
        carry=next_carry,
        episode_start=jnp.zeros_like(done_value),
    )
    return (action, next_state) if batched else (action[0], next_state)


class OfficialCetrPolicy:
    """Duck-typed Official ``PPOPolicy`` adapter for a CETR deployment."""

    def __init__(self, deployment: Deployment):
        self.deployment = deployment

    def init_hstate(self, batch_size: int, key: Any | None = None) -> Any:
        del key
        return reset_deployment_state(self.deployment, int(batch_size))

    def compute_action(
        self, obs: Any, done: Any, hstate: Any, key: Any
    ) -> tuple[Any, Any]:
        return _compute_deployment_action(self.deployment, obs, done, hstate, key)


def assert_official_policy_surface(policy: Any) -> None:
    for name in ("compute_action", "init_hstate"):
        if not callable(getattr(policy, name, None)):
            raise TypeError(f"Policy lacks Official interface method {name}.")
    forbidden = (
        "observe_transition",
        "set_partner_id",
        "set_partner_parameters",
        "set_environment_state",
    )
    present = [name for name in forbidden if hasattr(policy, name)]
    if present:
        raise TypeError(f"Official policy exposes forbidden test hooks: {present}")


__all__ = ["OfficialCetrPolicy", "assert_official_policy_surface"]
