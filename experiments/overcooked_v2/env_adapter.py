from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from experiments.overcooked_v2.path_c_seed import (
    canonical_uint64_seed,
    derive_ocv2_execution_seed,
)

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


@dataclass(frozen=True)
class OCV2AdapterSnapshot:
    """Detached OCV2 environment state for deterministic replay or forking."""

    layout_name: str
    max_steps: int
    key: Any
    state: Any
    raw_obs: dict[str, np.ndarray]

    def __post_init__(self) -> None:
        if not isinstance(self.layout_name, str) or not self.layout_name.strip():
            raise ValueError("OCV2 snapshot layout_name must be non-empty.")
        if isinstance(self.max_steps, bool) or int(self.max_steps) <= 0:
            raise ValueError("OCV2 snapshot max_steps must be positive.")
        if self.key is None or self.state is None:
            raise ValueError("OCV2 snapshot requires key and state.")
        if not isinstance(self.raw_obs, Mapping) or not self.raw_obs:
            raise ValueError("OCV2 snapshot requires raw observations.")


@dataclass(frozen=True)
class OCV2PureStep:
    """One primitive transition without mutating the source adapter snapshot."""

    snapshot: OCV2AdapterSnapshot
    step: OCV2Step


class OCV2Adapter:
    def __init__(
        self,
        layout: str,
        max_steps: int = 400,
        observation_type: str = "featurized",
        agent_view_size: int | None = None,
        negative_rewards: bool = True,
        sample_recipe_on_delivery: bool = True,
        indicate_successful_delivery: bool = False,
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
            indicate_successful_delivery=indicate_successful_delivery,
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
        self.canonical_seed: int | None = None
        self.execution_seed: int | None = None
        self.state = None
        self.raw_obs: dict[str, np.ndarray] | None = None
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
        self.canonical_seed = canonical_uint64_seed(seed, name="OCV2 reset seed")
        self.execution_seed = derive_ocv2_execution_seed(self.canonical_seed)
        self.key = jax.random.PRNGKey(self.execution_seed)
        self.key, subkey = jax.random.split(self.key)
        raw_obs, state = self._jit_reset(subkey)
        # PERF (2026-07-06, profile-driven; semantics unchanged): materialize the
        # state as a host/numpy pytree ONCE per boundary. Every downstream reader
        # (state_utils getters, option preconditions, event extractor, featurizer,
        # scripted partners) indexes state fields scalar-by-scalar; on a JAX array
        # each such index dispatches a full slice primitive (~250k dispatches per
        # episode ≈ 2/3 of eval wall time). numpy leaves make those reads ~free.
        # The jitted step/reset accept numpy leaves as inputs unchanged (same
        # shapes/dtypes -> no retrace; CPU backend device_put is near zero-copy).
        state = jax.device_get(state)
        self.raw_obs = self._to_numpy_obs(jax.device_get(raw_obs))
        self.obs = self._apply_featurizer(self.raw_obs, state)
        self.state = state
        return self.obs, self.state

    def step(self, ego_action: int, partner_action: int) -> OCV2Step:
        """Legacy agent-0/agent-1 wrapper retained for the option-based path."""

        return self.step_joint(agent_0_action=ego_action, agent_1_action=partner_action)

    def step_joint(self, agent_0_action: int, agent_1_action: int) -> OCV2Step:
        """Advance one slot-neutral primitive joint action."""

        source = self.capture_state()
        result = self.step_joint_from_state(
            source,
            agent_0_action=agent_0_action,
            agent_1_action=agent_1_action,
        )
        self.restore_state(result.snapshot)
        return result.step

    def capture_state(self) -> OCV2AdapterSnapshot:
        """Capture a detached state; later adapter mutations cannot alter it."""

        if self.key is None or self.state is None or self.raw_obs is None:
            raise RuntimeError("OCV2Adapter.step() called before reset().")
        return OCV2AdapterSnapshot(
            layout_name=str(self.layout_name),
            max_steps=int(self.max_steps),
            key=_clone_pytree(self.key),
            state=_clone_pytree(self.state),
            raw_obs=_clone_observation(self.raw_obs),
        )

    def restore_state(self, snapshot: OCV2AdapterSnapshot) -> tuple[dict[str, np.ndarray], Any]:
        """Restore an exact detached snapshot into this adapter instance."""

        self._validate_snapshot(snapshot)
        self.key = _clone_pytree(snapshot.key)
        self.state = _clone_pytree(snapshot.state)
        self.raw_obs = _clone_observation(snapshot.raw_obs)
        self.obs = self._apply_featurizer(self.raw_obs, self.state)
        return _clone_observation(self.obs), _clone_pytree(self.state)

    def step_from_state(
        self,
        snapshot: OCV2AdapterSnapshot,
        *,
        ego_action: int,
        partner_action: int,
    ) -> OCV2PureStep:
        """Legacy agent-0/agent-1 wrapper for the option-based path."""

        return self.step_joint_from_state(
            snapshot,
            agent_0_action=ego_action,
            agent_1_action=partner_action,
        )

    def step_joint_from_state(
        self,
        snapshot: OCV2AdapterSnapshot,
        *,
        agent_0_action: int,
        agent_1_action: int,
    ) -> OCV2PureStep:
        """Advance a supplied snapshot using slot-neutral primitive actions."""

        self._validate_snapshot(snapshot)
        source_key = _clone_pytree(snapshot.key)
        source_state = _clone_pytree(snapshot.state)

        next_key, subkey = jax.random.split(source_key)
        actions = {
            "agent_0": jnp.asarray(agent_0_action, dtype=jnp.int32),
            "agent_1": jnp.asarray(agent_1_action, dtype=jnp.int32),
        }
        raw_obs, state, rewards, dones, info = self._jit_step(
            subkey,
            source_state,
            actions,
        )
        # PERF: one host materialization for everything downstream (see reset()).
        # device_get preserves pytree structure; leaves become numpy with the same
        # dtypes, so _to_float/_to_bool/np.asarray consumers see identical values.
        state, rewards, dones, info = jax.device_get((state, rewards, dones, info))
        raw_obs_host = self._to_numpy_obs(jax.device_get(raw_obs))
        public_obs = self._apply_featurizer(raw_obs_host, state)
        next_snapshot = OCV2AdapterSnapshot(
            layout_name=str(self.layout_name),
            max_steps=int(self.max_steps),
            key=_clone_pytree(next_key),
            state=_clone_pytree(state),
            raw_obs=_clone_observation(raw_obs_host),
        )
        step = OCV2Step(
            obs=_clone_observation(public_obs),
            state=state,
            rewards={key: _to_float(value) for key, value in rewards.items()},
            dones={key: _to_bool(value) for key, value in dones.items()},
            info=self._to_numpy_info(info),
        )
        return OCV2PureStep(snapshot=next_snapshot, step=step)

    def _validate_snapshot(self, snapshot: OCV2AdapterSnapshot) -> None:
        if not isinstance(snapshot, OCV2AdapterSnapshot):
            raise TypeError("snapshot must be OCV2AdapterSnapshot.")
        if snapshot.layout_name != self.layout_name:
            raise ValueError("OCV2 snapshot layout differs from the adapter.")
        if int(snapshot.max_steps) != int(self.max_steps):
            raise ValueError("OCV2 snapshot max_steps differs from the adapter.")

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


def _clone_pytree(value: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda leaf: np.asarray(leaf).copy(),
        value,
    )


def _clone_observation(value: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {
        str(key): np.asarray(item).copy()
        for key, item in value.items()
    }
