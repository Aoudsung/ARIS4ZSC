from __future__ import annotations

from dataclasses import dataclass
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
