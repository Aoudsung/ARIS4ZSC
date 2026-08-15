"""Official evaluator adapter exposing only legal local history."""

from __future__ import annotations

from typing import Any

from .deployment import Deployment, deployment_action, reset_deployment_state


def _tree_select(mask: Any, selected: Any, alternative: Any) -> Any:
    import jax
    import jax.numpy as jnp

    def one(left: Any, right: Any):
        value = jnp.asarray(mask, dtype=jnp.bool_)
        expanded = value.reshape(value.shape + (1,) * (jnp.ndim(left) - value.ndim))
        return jnp.where(expanded, left, right)

    return jax.tree_util.tree_map(one, selected, alternative)


class OfficialDELTAPolicy:
    """Duck-typed OvercookedV2 ``AbstractPolicy`` implementation."""

    def __init__(self, deployment: Deployment, execution_mode: str | None = None):
        self.deployment = deployment
        self.execution_mode = execution_mode

    def init_hstate(self, batch_size: int, key: Any | None = None) -> Any:
        del key
        return reset_deployment_state(self.deployment, batch_size=int(batch_size))

    def compute_action(
        self, obs: Any, done: Any, hstate: Any, key: Any
    ) -> tuple[Any, Any]:
        import jax.numpy as jnp

        observation = jnp.asarray(obs, dtype=jnp.float32)
        batched = observation.ndim == len(self.deployment.model.observation_shape) + 1
        if not batched:
            observation = observation[None]
        done_value = jnp.asarray(done, dtype=jnp.bool_)
        if done_value.ndim == 0:
            done_value = done_value[None]
        keys = jnp.asarray(key)
        if keys.ndim == 1:
            keys = keys[None]
        fresh = self.init_hstate(int(observation.shape[0]))
        current = _tree_select(done_value, fresh, hstate)
        stepped, action, unused_output, unused_logp = deployment_action(
            deployment=self.deployment,
            state=current,
            observation=observation,
            keys=keys,
            execution_mode=self.execution_mode,
        )
        del unused_output, unused_logp
        next_state = stepped._replace(
            previous_action=jnp.asarray(action, dtype=jnp.int32),
            episode_start=jnp.zeros_like(done_value, dtype=jnp.bool_),
        )
        return (action, next_state) if batched else (action[0], next_state)


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
        raise TypeError(f"Official policy exposes forbidden test hooks: {present}.")


__all__ = ["OfficialDELTAPolicy", "assert_official_policy_surface"]
