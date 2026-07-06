from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jaxmarl.environments.overcooked_v2.overcooked import (
        ObservationType,
        OvercookedV2,
    )
except ImportError as exc:  # pragma: no cover - depends on optional JaxMARL install
    raise ImportError(
        "OCV2Adapter requires JAX and a JaxMARL install that provides "
        "jaxmarl.environments.overcooked_v2."
    ) from exc


@dataclass
class OCV2Step:
    obs: dict[str, np.ndarray]
    state: Any
    rewards: dict[str, float]
    dones: dict[str, bool]
    info: dict[str, Any]


class OCV2Adapter:
    def __init__(
        self,
        layout: str,
        max_steps: int = 400,
        observation_type: str = "featurized",
        agent_view_size: int | None = None,
        negative_rewards: bool = True,
        sample_recipe_on_delivery: bool = True,
        random_reset: bool = False,
        random_agent_positions: bool = False,
        force_path_planning: bool = True,
        featurizer: Callable[[Any], dict[str, np.ndarray]] | None = None,
    ):
        self.env = OvercookedV2(
            layout=layout,
            max_steps=max_steps,
            observation_type=_resolve_observation_type(observation_type),
            agent_view_size=agent_view_size,
            negative_rewards=negative_rewards,
            sample_recipe_on_delivery=sample_recipe_on_delivery,
            random_reset=random_reset,
            random_agent_positions=random_agent_positions,
            force_path_planning=force_path_planning,
        )
        self.layout_name = layout
        self.max_steps = max_steps
        self.featurizer = featurizer
        self._jit_reset = jax.jit(self.env.reset)
        self._jit_step = jax.jit(self.env.step_env)
        self.key = None
        self.state = None
        self.obs: dict[str, np.ndarray] | None = None

    @property
    def layout(self) -> Any:
        return self.env.layout

    def set_featurizer(
        self,
        featurizer: Callable[[Any], dict[str, np.ndarray]] | None,
    ) -> None:
        self.featurizer = featurizer

    def reset(self, seed: int) -> tuple[dict[str, np.ndarray], Any]:
        self.key = jax.random.PRNGKey(seed)
        self.key, subkey = jax.random.split(self.key)
        obs, state = self._jit_reset(subkey)
        # PERF (2026-07-06, profile-driven; semantics unchanged): materialize the
        # state as a host/numpy pytree ONCE per boundary. Every downstream reader
        # (state_utils getters, option preconditions, event extractor, featurizer,
        # scripted partners) indexes state fields scalar-by-scalar; on a JAX array
        # each such index dispatches a full slice primitive (~250k dispatches per
        # episode ≈ 2/3 of eval wall time). numpy leaves make those reads ~free.
        # The jitted step/reset accept numpy leaves as inputs unchanged (same
        # shapes/dtypes -> no retrace; CPU backend device_put is near zero-copy).
        state = jax.device_get(state)
        self.obs = self._apply_featurizer(obs, state)
        self.state = state
        return self.obs, self.state

    def step(self, ego_action: int, partner_action: int) -> OCV2Step:
        if self.key is None or self.state is None:
            raise RuntimeError("OCV2Adapter.step() called before reset().")

        self.key, subkey = jax.random.split(self.key)
        actions = {
            "agent_0": jnp.asarray(ego_action, dtype=jnp.int32),
            "agent_1": jnp.asarray(partner_action, dtype=jnp.int32),
        }
        obs, state, rewards, dones, info = self._jit_step(
            subkey,
            self.state,
            actions,
        )
        # PERF: one host materialization for everything downstream (see reset()).
        # device_get preserves pytree structure; leaves become numpy with the same
        # dtypes, so _to_float/_to_bool/np.asarray consumers see identical values.
        state, rewards, dones, info = jax.device_get((state, rewards, dones, info))
        self.state = state
        self.obs = self._apply_featurizer(obs, state)
        return OCV2Step(
            obs=self.obs,
            state=state,
            rewards={key: _to_float(value) for key, value in rewards.items()},
            dones={key: _to_bool(value) for key, value in dones.items()},
            info=self._to_numpy_info(info),
        )

    @staticmethod
    def _to_numpy_obs(obs: Mapping[str, Any]) -> dict[str, np.ndarray]:
        return {key: np.asarray(value) for key, value in obs.items()}

    def _apply_featurizer(
        self,
        raw_obs: Mapping[str, Any],
        state: Any,
    ) -> dict[str, np.ndarray]:
        # PERF: when a featurizer is set, the raw observation tensors are unused —
        # do not pay a per-step host conversion for arrays we immediately discard.
        if self.featurizer is None:
            return self._to_numpy_obs(raw_obs)
        return {
            key: np.asarray(value, dtype=np.float32)
            for key, value in self.featurizer(state).items()
        }

    @staticmethod
    def _to_numpy_info(info: Mapping[str, Any]) -> dict[str, Any]:
        return {key: _to_numpy_value(value) for key, value in info.items()}


def _resolve_observation_type(observation_type: str | ObservationType) -> ObservationType:
    if isinstance(observation_type, ObservationType):
        return observation_type
    if observation_type == ObservationType.FEATURIZED.value:
        return ObservationType.FEATURIZED
    if observation_type == ObservationType.DEFAULT.value:
        return ObservationType.DEFAULT
    raise ValueError(
        f"Unsupported OvercookedV2 observation_type={observation_type!r}. "
        "Expected 'featurized' or 'default'."
    )


def _to_float(value: Any) -> float:
    return float(np.asarray(value).item())


def _to_bool(value: Any) -> bool:
    return bool(np.asarray(value).item())


def _to_numpy_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _to_numpy_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_numpy_value(item) for item in value)
    if isinstance(value, list):
        return [_to_numpy_value(item) for item in value]
    return np.asarray(value)
