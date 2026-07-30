"""Official-evaluator policy adapters.

The DELTA adapter intentionally exposes only the two methods required by the
locked ``AbstractPolicy`` contract.  In particular, it has no transition hook:
the policy receives the current official observation, the previous done flag,
its own recurrent state, and the official action key.  Calibration statistics
are frozen inside the deployment artifact; calibration examples are not
available at evaluation time.
"""

from __future__ import annotations

from typing import Any

from experiments.overcooked_v2.deployment import (
    Deployment,
    deployment_action,
    reset_deployment_state,
)
from src.path_c.counterfactual_anchor import tree_select


class OfficialDeltaPolicy:
    """Duck-typed implementation of Official ``AbstractPolicy``.

    The public evaluator relies on the interface rather than an ``isinstance``
    check.  Avoiding a module-level import of the Official package keeps config
    and repository checks usable before the pinned runtime is installed.
    """

    def __init__(self, deployment: Deployment, *, force_base: bool = False):
        self.deployment = deployment
        self.force_base = bool(force_base)

    def init_hstate(self, batch_size: int, key: Any | None = None) -> Any:
        del key
        return reset_deployment_state(
            self.deployment,
            batch_size=int(batch_size),
            observation_shape=tuple(self.deployment.model.observation_shape),
        )

    def compute_action(
        self,
        obs: Any,
        done: Any,
        hstate: Any,
        key: Any,
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
            force_base=self.force_base,
        )
        del unused_output, unused_log_probability

        # The locked Official policy interface does not expose transition
        # reward.  Store only information the policy itself produced.  Current
        # official observation at the next call supplies all visible delivery
        # and partner-response evidence.
        next_state = stepped._replace(
            previous_action=jnp.asarray(action, dtype=jnp.int32),
            previous_reward=jnp.zeros_like(
                jnp.asarray(action, dtype=jnp.float32)
            ),
            episode_start=jnp.zeros_like(done_value, dtype=jnp.bool_),
        )
        if batched:
            return action, next_state
        return action[0], next_state


def assert_official_policy_surface(policy: Any) -> None:
    """Fail if an evaluation policy exposes a non-Official transition hook."""

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


__all__ = ["OfficialDeltaPolicy", "assert_official_policy_surface"]
