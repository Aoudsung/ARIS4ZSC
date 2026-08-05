"""Official evaluator adapter for the single deployable DEPI actor."""

from __future__ import annotations

from typing import Any

from experiments.overcooked_v2.deployment import (
    Deployment,
    deployment_action,
    reset_deployment_state,
)
from src.path_c.counterfactual_anchor import tree_select


class OfficialDEPIPolicy:
    """Duck-typed locked ``AbstractPolicy`` implementation."""

    def __init__(self, deployment: Deployment):
        self.deployment = deployment

    def init_hstate(self, batch_size: int, key: Any | None = None) -> Any:
        del key
        return reset_deployment_state(
            self.deployment,
            batch_size=int(batch_size),
            observation_shape=tuple(self.deployment.model.observation_shape),
        )

    def compute_action(
        self, obs: Any, done: Any, hstate: Any, key: Any
    ) -> tuple[Any, Any]:
        import jax.numpy as jnp

        observation = jnp.asarray(obs)
        done_value = jnp.asarray(done, dtype=jnp.bool_)
        batched = observation.ndim == len(self.deployment.model.observation_shape) + 1
        if not batched:
            observation = observation[None, ...]
            done_value = done_value.reshape((1,))
            key = jnp.asarray(key)[None, ...]
        fresh = self.init_hstate(int(observation.shape[0]))
        state = tree_select(done_value, fresh, hstate)
        stepped, action, unused_output, unused_log_probability = deployment_action(
            deployment=self.deployment,
            state=state,
            observation=observation,
            keys=key,
        )
        del unused_output, unused_log_probability
        next_state = stepped._replace(
            previous_action=jnp.asarray(action, dtype=jnp.int32),
            episode_start=jnp.zeros_like(done_value, dtype=jnp.bool_),
        )
        if batched:
            return action, next_state
        return action[0], next_state


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


__all__ = ["OfficialDEPIPolicy", "assert_official_policy_surface"]
