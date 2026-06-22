from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable, Sequence

import numpy as np

from .state_codec import EnvConfigSnapshot, concrete_state_to_jsonable


class SourceBackendError(RuntimeError):
    pass


@dataclass(frozen=True)
class StepOutcome:
    next_state: Any
    rewards: dict[str, float]
    dones: dict[str, bool]
    info: dict[str, Any]


class JaxmarlOvercookedV2Backend:
    """Strict wrapper around FLAIROx/JaxMARL OvercookedV2.

    Certification runs must use this backend. There is no fixture fallback: if
    JAXMARL cannot be imported, if the requested layout is unavailable, or if a
    source step fails, the run emits a failure report before writing any
    certificate-like artifact.

    The backend provides both single-step and vectorized batch-step execution.
    Vectorization is explicit; a failed vectorized call is not silently replaced
    by sequential stepping. For debugging, callers may request
    ``step_mode='sequential'`` in the config, but that is recorded in the run
    report and is not used as a hidden fallback.
    """

    def __init__(
        self,
        *,
        layout_id: str,
        max_steps: int,
        env_kwargs: dict[str, Any] | None = None,
        jit_batch_step: bool = True,
    ) -> None:
        env_kwargs = dict(env_kwargs or {})
        try:
            import jax  # type: ignore
            import jax.numpy as jnp  # type: ignore
            from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2  # type: ignore
            from jaxmarl.environments.overcooked_v2.common import Actions, StaticObject, DynamicObject, Direction  # type: ignore
        except Exception as exc:  # noqa: BLE001 - explicit top-level failure
            raise SourceBackendError(
                "JAXMARL OvercookedV2 import failed. Install optional runtime dependencies from "
                "https://github.com/FLAIROx/JaxMARL or pip install jaxmarl, then rerun. "
                "The certifier will not replace JAXMARL with a synthetic backend."
            ) from exc
        self.jax = jax
        self.jnp = jnp
        self.Actions = Actions
        self.StaticObject = StaticObject
        self.DynamicObject = DynamicObject
        self.Direction = Direction
        self.layout_id = layout_id
        self.env = self._make_env(OvercookedV2, layout_id=layout_id, max_steps=max_steps, env_kwargs=env_kwargs)
        self.num_agents = int(self.env.num_agents)
        self.num_actions = int(len(self.env.action_set))
        self.action_names = {int(a): getattr(a, "name", str(int(a))) for a in list(Actions)}
        self.config_snapshot = EnvConfigSnapshot.from_env(self.env, layout_id)
        self.jit_batch_step = bool(jit_batch_step)
        self._batch_step_fn = None

    @staticmethod
    def _make_env(OvercookedV2: Any, *, layout_id: str, max_steps: int, env_kwargs: dict[str, Any]) -> Any:
        errors: list[str] = []
        # The public JAXMARL API has used both `layout` and `layout_name` in
        # different downstream examples. Try both explicitly; failures are
        # reported rather than replaced by a default layout.
        for layout_key in ("layout", "layout_name"):
            kwargs = {layout_key: layout_id, "max_steps": max_steps, **env_kwargs}
            try:
                return OvercookedV2(**kwargs)
            except TypeError as exc:
                errors.append(f"{layout_key}: {exc}")
        raise SourceBackendError(
            f"Could not construct OvercookedV2 for layout_id={layout_id!r}. Constructor errors: {errors}"
        )

    def joint_actions(self) -> list[tuple[int, ...]]:
        return [tuple(int(x) for x in a) for a in product(range(self.num_actions), repeat=self.num_agents)]

    def reset_state(self, seed: int):
        key = self.jax.random.PRNGKey(int(seed))
        _, state = self.env.reset(key)
        return state

    def step(self, state: Any, joint_action: tuple[int, ...], seed: int) -> StepOutcome:
        if len(joint_action) != self.num_agents:
            raise ValueError(f"joint_action length {len(joint_action)} != num_agents {self.num_agents}")
        key = self.jax.random.PRNGKey(int(seed))
        actions = {f"agent_{i}": self.jnp.array(int(joint_action[i]), dtype=self.jnp.int32) for i in range(self.num_agents)}
        try:
            _, next_state, rewards, dones, info = self.env.step_env(key, state, actions)
            self.jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, next_state)
        except Exception as exc:  # noqa: BLE001 - turns source failures into explicit run failures
            raise SourceBackendError(f"JAXMARL step_env failed for joint_action={joint_action}") from exc
        return StepOutcome(
            next_state=next_state,
            rewards={k: float(np.asarray(v).item()) for k, v in rewards.items()},
            dones={k: bool(np.asarray(v).item()) for k, v in dones.items()},
            info=_jsonable_info(info),
        )

    def batch_step(self, states: Sequence[Any], joint_actions: Sequence[tuple[int, ...]], base_seed: int) -> list[StepOutcome]:
        """Vectorized source stepping over a frontier-action batch.

        This uses ``jax.vmap`` over a stacked state pytree, a batch of PRNG keys,
        and per-agent action arrays. It is intentionally all-or-fail: exceptions
        are surfaced as ``SourceBackendError``. The caller may choose sequential
        mode explicitly, but vectorized mode never catches-and-continues.
        """

        n = len(states)
        if n != len(joint_actions):
            raise ValueError("states and joint_actions must have identical length")
        if n == 0:
            return []
        for ja in joint_actions:
            if len(ja) != self.num_agents:
                raise ValueError(f"joint_action length {len(ja)} != num_agents {self.num_agents}: {ja}")
        try:
            states_batched = self.jax.tree_util.tree_map(lambda *xs: self.jnp.stack(xs, axis=0), *states)
            keys = self.jax.random.split(self.jax.random.PRNGKey(int(base_seed)), n)
            action_arrays = [self.jnp.asarray([int(ja[i]) for ja in joint_actions], dtype=self.jnp.int32) for i in range(self.num_agents)]

            def one_step(key, state, *per_agent_actions):
                action_dict = {f"agent_{i}": per_agent_actions[i] for i in range(self.num_agents)}
                return self.env.step_env(key, state, action_dict)

            if self._batch_step_fn is None:
                vmapped = self.jax.vmap(one_step, in_axes=(0, 0, *([0] * self.num_agents)))
                self._batch_step_fn = self.jax.jit(vmapped) if self.jit_batch_step else vmapped
            _, next_states_batched, rewards_batched, dones_batched, infos_batched = self._batch_step_fn(keys, states_batched, *action_arrays)
            self.jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, next_states_batched)
            next_states = _unstack_pytree(self.jax, next_states_batched, n)
            return [
                StepOutcome(
                    next_state=next_states[i],
                    rewards=_unbatch_mapping(rewards_batched, i),
                    dones={k: bool(v) for k, v in _unbatch_mapping(dones_batched, i).items()},
                    info=_unbatch_info(infos_batched, i),
                )
                for i in range(n)
            ]
        except Exception as exc:  # noqa: BLE001 - no fallback; explicit source/vectorization failure
            raise SourceBackendError(f"JAXMARL vectorized batch_step failed for batch_size={n}") from exc

    def step_many_sequential_explicit(self, states: Sequence[Any], joint_actions: Sequence[tuple[int, ...]], base_seed: int) -> list[StepOutcome]:
        """Explicit debug mode only; never used as silent vectorized fallback."""

        return [self.step(s, ja, base_seed + i) for i, (s, ja) in enumerate(zip(states, joint_actions))]

    def concrete_hash(self, state: Any) -> str:
        from .canonical import canonical_hash

        return canonical_hash(concrete_state_to_jsonable(state))


def _unstack_pytree(jax_mod: Any, pytree: Any, n: int) -> list[Any]:
    leaves = []
    for i in range(n):
        leaves.append(jax_mod.tree_util.tree_map(lambda x, i=i: x[i], pytree))
    return leaves


def _unbatch_mapping(mapping: dict[str, Any], i: int) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in mapping.items():
        arr = np.asarray(v)
        scalar = arr[i] if arr.shape else arr
        out[k] = float(np.asarray(scalar).item())
    return out


def _unbatch_info(info: dict[str, Any], i: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in info.items():
        if isinstance(v, dict):
            out[k] = {kk: _to_scalar_at(vv, i) for kk, vv in v.items()}
        else:
            out[k] = _to_scalar_at(v, i)
    return out


def _jsonable_info(info: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in info.items():
        if isinstance(v, dict):
            out[k] = {kk: _to_scalar(vv) for kk, vv in v.items()}
        else:
            out[k] = _to_scalar(v)
    return out


def _to_scalar_at(value: Any, i: int) -> Any:
    try:
        arr = np.asarray(value)
        if arr.shape == ():
            return arr.item()
        return _to_scalar(arr[i])
    except Exception:  # noqa: BLE001
        return str(value)


def _to_scalar(value: Any) -> Any:
    try:
        arr = np.asarray(value)
        if arr.shape == ():
            return arr.item()
        return arr.tolist()
    except Exception:  # noqa: BLE001
        return str(value)
