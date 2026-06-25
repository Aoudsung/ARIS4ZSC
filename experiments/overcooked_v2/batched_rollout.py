from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np


class BatchedRolloutUnsupported(RuntimeError):
    pass


@dataclass(frozen=True)
class BatchedResetResult:
    obs: Any
    state: Any
    batch_size: int


def batched_reset(adapter: Any, seeds: np.ndarray | list[int]) -> BatchedResetResult:
    env = getattr(adapter, "env", None)
    if env is None or not hasattr(env, "reset"):
        raise BatchedRolloutUnsupported(
            "batched_reset requires an adapter exposing a JAX env with reset()."
        )

    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise BatchedRolloutUnsupported("batched_reset requires JAX.") from exc

    seed_array = np.asarray(seeds, dtype=np.int64).reshape(-1)
    if seed_array.size == 0:
        raise ValueError("batched_reset requires at least one seed.")

    keys = jax.vmap(lambda seed: jax.random.PRNGKey(seed))(jnp.asarray(seed_array))
    try:
        obs, state = jax.vmap(env.reset)(keys)
    except Exception as exc:  # pragma: no cover - backend specific
        raise BatchedRolloutUnsupported(
            "Underlying OvercookedV2 env.reset could not be vectorized with jax.vmap."
        ) from exc
    return BatchedResetResult(obs=obs, state=state, batch_size=int(seed_array.size))


def assert_batched_shape(value: Any, batch_size: int) -> None:
    if isinstance(value, dict):
        for item in value.values():
            assert_batched_shape(item, batch_size)
        return
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) == 0 or int(shape[0]) != int(batch_size):
        raise ValueError(
            f"Expected leading batch dimension {batch_size}; got shape={shape}."
        )


class NumpyStateSnapshot:
    """Bulk device→host transfer of batched JAX state, then zero-copy numpy views per env."""

    def __init__(self, batched_state: Any) -> None:
        self._pos_x = np.asarray(batched_state.agents.pos.x)
        self._pos_y = np.asarray(batched_state.agents.pos.y)
        self._inventory = np.asarray(batched_state.agents.inventory)
        self._dir = np.asarray(batched_state.agents.dir)
        self._grid = np.asarray(batched_state.grid)
        self._recipe = np.asarray(batched_state.recipe)

    def __getitem__(self, i: int) -> SimpleNamespace:
        return SimpleNamespace(
            agents=SimpleNamespace(
                pos=SimpleNamespace(x=self._pos_x[i], y=self._pos_y[i]),
                dir=self._dir[i],
                inventory=self._inventory[i],
            ),
            grid=self._grid[i],
            recipe=self._recipe[i],
        )


class BatchedEnvPool:
    """N parallel OvercookedV2 envs via jax.vmap for batched stepping."""

    def __init__(self, env: Any, batch_size: int) -> None:
        import jax
        import jax.numpy as jnp

        self.env = env
        self.batch_size = batch_size
        self._jax = jax
        self._jnp = jnp
        self._vmap_reset = jax.jit(jax.vmap(env.reset))
        self._vmap_step = jax.jit(jax.vmap(env.step_env))
        self._vmap_prng = jax.jit(jax.vmap(jax.random.PRNGKey))
        self._vmap_split = jax.jit(jax.vmap(jax.random.split))
        self.keys: Any = None
        self.state: Any = None
        self.obs: Any = None

    def reset(self, seeds: np.ndarray) -> None:
        jnp = self._jnp
        seed_arr = jnp.asarray(np.asarray(seeds, dtype=np.int64).reshape(-1))
        self.keys = self._vmap_prng(seed_arr)
        sub, self.keys = self._split_keys()
        self.obs, self.state = self._vmap_reset(sub)

    def step(
        self,
        ego_actions: np.ndarray,
        partner_actions: np.ndarray,
    ) -> tuple[Any, Any, Any, Any, Any]:
        jnp = self._jnp
        sub, self.keys = self._split_keys()
        actions = {
            "agent_0": jnp.asarray(ego_actions, dtype=jnp.int32),
            "agent_1": jnp.asarray(partner_actions, dtype=jnp.int32),
        }
        obs, state, rewards, dones, info = self._vmap_step(sub, self.state, actions)
        self.state = state
        self.obs = obs
        return obs, state, rewards, dones, info

    def get_state_i(self, i: int) -> Any:
        return self._jax.tree.map(lambda x: x[i], self.state)

    def snapshot(self) -> NumpyStateSnapshot:
        return NumpyStateSnapshot(self.state)

    def snapshot_obs(self) -> dict[str, np.ndarray]:
        return {k: np.asarray(v) for k, v in self.obs.items()}

    def snapshot_info(self, info: Any) -> Any:
        return self._jax.tree.map(
            lambda x: np.asarray(x) if hasattr(x, "shape") else x,
            info,
        )

    def get_obs_i(self, i: int) -> dict[str, np.ndarray]:
        return {k: np.asarray(v[i]) for k, v in self.obs.items()}

    def get_info_i(self, i: int, info: Any) -> dict[str, Any]:
        return self._jax.tree.map(
            lambda x: np.asarray(x[i]) if hasattr(x, "shape") else x,
            info,
        )

    def reset_indices(self, indices: np.ndarray, seeds: np.ndarray) -> None:
        jax = self._jax
        jnp = self._jnp
        idx = np.asarray(indices, dtype=int)
        if idx.size == 0:
            return
        sub_keys = self._vmap_prng(jnp.asarray(np.asarray(seeds, dtype=np.int64)))
        sub_split = jax.vmap(jax.random.split)(sub_keys)
        new_keys = sub_split[:, 0]
        reset_keys = sub_split[:, 1]
        new_obs, new_state = jax.vmap(self.env.reset)(reset_keys)

        def _scatter(old: Any, new_vals: Any) -> Any:
            return old.at[jnp.asarray(idx)].set(new_vals)

        self.state = jax.tree.map(_scatter, self.state, new_state)
        self.obs = jax.tree.map(_scatter, self.obs, new_obs)
        self.keys = self.keys.at[jnp.asarray(idx)].set(new_keys)

    def _split_keys(self) -> tuple[Any, Any]:
        pairs = self._vmap_split(self.keys)
        return pairs[:, 0], pairs[:, 1]
