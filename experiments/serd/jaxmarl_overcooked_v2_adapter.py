"""JaxMARL OvercookedV2 adapter for SERD M2 branch probes.

The adapter loads only the official OvercookedV2 modules from a JaxMARL source
checkout. This avoids importing unrelated JaxMARL environments that require
Brax, MuJoCo, Hanabi, or visualization dependencies. The generated records are
an adapter gate, not FCP/PECAN evidence.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .fixture_env import CONTROL_FAMILIES
from .serd_core import BranchRecord


class JaxmarlDependencyError(RuntimeError):
    """Raised when JaxMARL OvercookedV2 cannot be imported."""


V2_ACTION_ALIASES = {
    "right": 0,
    "down": 1,
    "left": 2,
    "up": 3,
    "stay": 4,
    "interact": 5,
}


@dataclass(frozen=True)
class OvercookedV2M2Config:
    layout: str = "test_time_simple"
    policy: str = "handcoded_overcooked_v2_policy"
    disruptions: tuple[str, ...] = ("missed_handoff", "route_block", "hesitation")
    probes_per_disruption: int = 50
    rollout_horizon: int = 20
    warmup_horizon: int = 20
    shock_horizon: int = 1
    shaped_reward_weight: float = 1.0
    agent_view_size: int | None = 2
    negative_rewards: bool = True
    sample_recipe_on_delivery: bool = True
    random_agent_positions: bool = False
    max_steps: int = 400
    seed: int = 0
    jaxmarl_src: str | None = None


@dataclass(frozen=True)
class OvercookedV2HandcodedPolicy:
    """Small deterministic policy used only to exercise M2 branching."""

    action_cycle: tuple[tuple[str, str], ...] = (
        ("right", "left"),
        ("down", "up"),
        ("interact", "interact"),
        ("up", "down"),
        ("left", "right"),
        ("stay", "stay"),
    )

    def action_names(self, state, step_index: int, probe_index: int) -> tuple[str, str]:
        time_value = int(_scalar(getattr(state, "time", 0)))
        offset = time_value + step_index + probe_index
        return self.action_cycle[offset % len(self.action_cycle)]


def _module_from_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise JaxmarlDependencyError(f"cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_jaxmarl_package_root(jaxmarl_src: str | None) -> Path:
    if jaxmarl_src is None:
        spec = importlib.util.find_spec("jaxmarl")
        if spec is None or spec.origin is None:
            raise JaxmarlDependencyError(
                "Cannot find JaxMARL. Pass `--jaxmarl-src /path/to/JaxMARL`."
            )
        return Path(spec.origin).resolve().parent

    src_path = Path(jaxmarl_src).expanduser().resolve()
    if (src_path / "jaxmarl").is_dir():
        return src_path / "jaxmarl"
    if src_path.name == "jaxmarl" and src_path.is_dir():
        return src_path
    raise JaxmarlDependencyError(
        f"JaxMARL source must be the repo root or package dir: {src_path}"
    )


def ensure_jaxmarl_overcooked_v2(jaxmarl_src: str | None = None):
    """Return `(OvercookedV2, Actions, jax)` using a narrow import path."""

    package_root = _resolve_jaxmarl_package_root(jaxmarl_src)
    env_root = package_root / "environments"
    v2_root = env_root / "overcooked_v2"

    for required in (
        env_root / "spaces.py",
        env_root / "multi_agent_env.py",
        v2_root / "common.py",
        v2_root / "settings.py",
        v2_root / "layouts.py",
        v2_root / "utils.py",
        v2_root / "overcooked.py",
    ):
        if not required.exists():
            raise JaxmarlDependencyError(f"missing JaxMARL source file: {required}")

    try:
        import jax
        import flax  # noqa: F401
    except ModuleNotFoundError as exc:
        raise JaxmarlDependencyError(
            "JaxMARL OvercookedV2 requires `jax`, `jaxlib`, `chex`, and `flax`."
        ) from exc

    pkg = types.ModuleType("jaxmarl")
    pkg.__path__ = [str(package_root)]
    sys.modules["jaxmarl"] = pkg

    env_pkg = types.ModuleType("jaxmarl.environments")
    env_pkg.__path__ = [str(env_root)]
    sys.modules["jaxmarl.environments"] = env_pkg

    spaces = _module_from_file("jaxmarl.environments.spaces", env_root / "spaces.py")
    env_pkg.spaces = spaces
    multi_agent_env = _module_from_file(
        "jaxmarl.environments.multi_agent_env", env_root / "multi_agent_env.py"
    )
    env_pkg.MultiAgentEnv = multi_agent_env.MultiAgentEnv
    env_pkg.State = multi_agent_env.State

    v2_pkg = types.ModuleType("jaxmarl.environments.overcooked_v2")
    v2_pkg.__path__ = [str(v2_root)]
    sys.modules["jaxmarl.environments.overcooked_v2"] = v2_pkg

    for module_name in ("common", "settings", "layouts", "utils"):
        _module_from_file(
            f"jaxmarl.environments.overcooked_v2.{module_name}",
            v2_root / f"{module_name}.py",
        )
    overcooked = _module_from_file(
        "jaxmarl.environments.overcooked_v2.overcooked",
        v2_root / "overcooked.py",
    )
    return overcooked.OvercookedV2, overcooked.Actions, jax


def action_index(name: str) -> int:
    try:
        return V2_ACTION_ALIASES[name]
    except KeyError as exc:
        raise ValueError(f"unknown OvercookedV2 action alias: {name}") from exc


def action_divergence(
    left: Sequence[tuple[str, str]],
    right: Sequence[tuple[str, str]],
    shock_horizon: int,
) -> float:
    horizon = max(1, shock_horizon)
    comparisons = 0
    mismatches = 0
    for left_joint, right_joint in zip(left[:horizon], right[:horizon]):
        for left_action, right_action in zip(left_joint, right_joint):
            comparisons += 1
            if left_action != right_action:
                mismatches += 1
    return float(mismatches) / float(max(comparisons, 1))


def _replace_one_action(
    base_joint: tuple[str, str],
    actor_index: int,
    preferred: str,
) -> tuple[str, str]:
    alternatives = ("stay", "interact", "up", "down", "left", "right")
    chosen = preferred if preferred != base_joint[actor_index] else None
    if chosen is None:
        for candidate in alternatives:
            if candidate != base_joint[actor_index]:
                chosen = candidate
                break
    assert chosen is not None
    branch = list(base_joint)
    branch[actor_index] = chosen
    return (branch[0], branch[1])


def semantic_joint(
    base_joint: tuple[str, str],
    disruption: str,
    probe_index: int,
) -> tuple[str, str]:
    if disruption == "missed_handoff":
        return _replace_one_action(base_joint, 0, "stay")
    if disruption == "route_block":
        return _replace_one_action(base_joint, 1, "stay")
    if disruption == "hesitation":
        return _replace_one_action(base_joint, probe_index % 2, "stay")
    raise ValueError(f"unknown disruption: {disruption}")


def control_joint(
    base_joint: tuple[str, str],
    disruption: str,
    family: str,
    probe_index: int,
) -> tuple[str, str]:
    actor_index = 0 if disruption == "missed_handoff" else 1
    if disruption == "hesitation":
        actor_index = probe_index % 2
    preferred_by_family = {
        "random_lag": "interact",
        "state_block": "up",
        "reward_shaping": "down",
        "naive_replanning": "left",
    }
    try:
        preferred = preferred_by_family[family]
    except KeyError as exc:
        raise ValueError(f"unknown control family: {family}") from exc
    return _replace_one_action(base_joint, actor_index, preferred)


def _scalar(value) -> float:
    try:
        import jax

        value = jax.device_get(value)
    except ModuleNotFoundError:
        pass
    try:
        return float(value.item())
    except AttributeError:
        return float(value)


def _array(value):
    import jax
    import numpy as np

    return np.asarray(jax.device_get(value))


def state_phi_pre_h(state) -> dict[str, float]:
    """Extract pre-shock covariates without post-branch outcomes."""

    phi: dict[str, float] = {
        "time": float(_scalar(state.time)),
        "terminal": 1.0 if bool(_scalar(state.terminal)) else 0.0,
        "recipe": float(_scalar(state.recipe)),
    }

    pos_x = _array(state.agents.pos.x).reshape(-1)
    pos_y = _array(state.agents.pos.y).reshape(-1)
    directions = _array(state.agents.dir).reshape(-1)
    inventory = _array(state.agents.inventory).reshape(-1)
    for idx in range(len(pos_x)):
        phi[f"a{idx}_x"] = float(pos_x[idx])
        phi[f"a{idx}_y"] = float(pos_y[idx])
        phi[f"a{idx}_dir"] = float(directions[idx])
        phi[f"a{idx}_inventory"] = float(inventory[idx])

    grid = _array(state.grid)
    phi["grid_dynamic_nonempty"] = float((grid[:, :, 1] != 0).sum())
    phi["grid_dynamic_sum"] = float(grid[:, :, 1].sum())
    phi["grid_extra_sum"] = float(grid[:, :, 2].sum())
    return phi


def _make_env(OvercookedV2, config: OvercookedV2M2Config):
    return OvercookedV2(
        layout=config.layout,
        max_steps=config.max_steps,
        agent_view_size=config.agent_view_size,
        negative_rewards=config.negative_rewards,
        sample_recipe_on_delivery=config.sample_recipe_on_delivery,
        random_agent_positions=config.random_agent_positions,
    )


def _joint_action_dict(env, joint_names: tuple[str, str]) -> dict[str, int]:
    return {
        agent: action_index(joint_names[idx])
        for idx, agent in enumerate(env.agents)
    }


def policy_rollout_return(
    env,
    jax,
    start_state,
    policy: OvercookedV2HandcodedPolicy,
    probe_index: int,
    horizon: int,
    shaped_reward_weight: float,
    seed: int,
    first_joint_override: tuple[str, str] | None = None,
) -> float:
    state = start_state
    key = jax.random.PRNGKey(seed)
    total = 0.0
    for step_index in range(horizon):
        joint_names = policy.action_names(state, step_index, probe_index)
        if step_index == 0 and first_joint_override is not None:
            joint_names = first_joint_override
        key, subkey = jax.random.split(key)
        _obs, state, rewards, _dones, infos = env.step(
            subkey,
            state,
            _joint_action_dict(env, joint_names),
        )
        sparse = sum(float(_scalar(reward)) for reward in rewards.values())
        shaped = 0.0
        for shaped_reward in infos.get("shaped_reward", {}).values():
            shaped += float(_scalar(shaped_reward))
        total += sparse + shaped_reward_weight * shaped
    return total


def _advance_policy_prefix_state(
    env,
    jax,
    start_state,
    policy: OvercookedV2HandcodedPolicy,
    probe_index: int,
    warmup_horizon: int,
    seed: int,
):
    state = start_state
    key = jax.random.PRNGKey(seed)
    for step_index in range(probe_index % max(1, warmup_horizon)):
        joint_names = policy.action_names(state, step_index, probe_index)
        key, subkey = jax.random.split(key)
        _obs, state, _rewards, _dones, _infos = env.step(
            subkey,
            state,
            _joint_action_dict(env, joint_names),
        )
    return state


def determinism_checks(config: OvercookedV2M2Config) -> dict[str, object]:
    OvercookedV2, _Actions, jax = ensure_jaxmarl_overcooked_v2(config.jaxmarl_src)
    env = _make_env(OvercookedV2, config)
    policy = OvercookedV2HandcodedPolicy()

    base_key = jax.random.PRNGKey(config.seed)
    _obs_a, state_a = env.reset(base_key)
    _obs_b, state_b = env.reset(base_key)
    phi_a = state_phi_pre_h(state_a)
    phi_b = state_phi_pre_h(state_b)

    distinct_reset_phis = set()
    for seed_offset in range(1, 11):
        _obs, reset_state = env.reset(jax.random.PRNGKey(config.seed + seed_offset))
        distinct_reset_phis.add(tuple(sorted(state_phi_pre_h(reset_state).items())))

    rollout_seed = config.seed + 100_000
    first = policy_rollout_return(
        env,
        jax,
        state_a,
        policy,
        0,
        min(5, config.rollout_horizon),
        config.shaped_reward_weight,
        rollout_seed,
    )
    second = policy_rollout_return(
        env,
        jax,
        state_a,
        policy,
        0,
        min(5, config.rollout_horizon),
        config.shaped_reward_weight,
        rollout_seed,
    )
    changed_seed = policy_rollout_return(
        env,
        jax,
        state_a,
        policy,
        0,
        min(5, config.rollout_horizon),
        config.shaped_reward_weight,
        rollout_seed + 1,
    )

    return {
        "same_seed_reset_equal": phi_a == phi_b,
        "distinct_reset_phi_count": len(distinct_reset_phis),
        "same_seed_rollout_equal": first == second,
        "recovery_seed_changes_return": first != changed_seed,
        "note": (
            "OvercookedV2 step dynamics may be deterministic for short "
            "no-delivery rollouts; reset seeds still change sampled recipe/state."
        ),
    }


def make_overcooked_v2_m2_policy_records(
    config: OvercookedV2M2Config,
) -> tuple[list[BranchRecord], list[BranchRecord]]:
    if config.shock_horizon != 1:
        raise ValueError("M2 adapter currently supports shock_horizon=1")

    OvercookedV2, _Actions, jax = ensure_jaxmarl_overcooked_v2(config.jaxmarl_src)
    env = _make_env(OvercookedV2, config)
    _obs, start_state = env.reset(jax.random.PRNGKey(config.seed))
    policy = OvercookedV2HandcodedPolicy()

    semantic_records: list[BranchRecord] = []
    control_records: list[BranchRecord] = []

    for disruption in config.disruptions:
        for probe_index in range(config.probes_per_disruption):
            prefix_seed = config.seed + 10_000 + probe_index
            recovery_seed = config.seed + 20_000 + probe_index
            probe_state = _advance_policy_prefix_state(
                env,
                jax,
                start_state,
                policy,
                probe_index,
                config.warmup_horizon,
                prefix_seed,
            )
            phi = state_phi_pre_h(probe_state)
            base_joint = policy.action_names(probe_state, 0, probe_index)
            semantic_first_joint = semantic_joint(base_joint, disruption, probe_index)
            semantic_shock = action_divergence([base_joint], [semantic_first_joint], 1)
            no_shock_return = policy_rollout_return(
                env,
                jax,
                probe_state,
                policy,
                probe_index,
                config.rollout_horizon,
                config.shaped_reward_weight,
                recovery_seed,
            )
            semantic_return = policy_rollout_return(
                env,
                jax,
                probe_state,
                policy,
                probe_index,
                config.rollout_horizon,
                config.shaped_reward_weight,
                recovery_seed,
                first_joint_override=semantic_first_joint,
            )
            probe_id = f"{config.policy}_{config.layout}_{disruption}_{probe_index:03d}"
            semantic_records.append(
                BranchRecord(
                    probe_id=probe_id,
                    policy=config.policy,
                    domain=f"jaxmarl_overcooked_v2:{config.layout}",
                    disruption=disruption,
                    family="semantic",
                    no_shock_return=no_shock_return,
                    branch_return=semantic_return,
                    shock_magnitude=semantic_shock,
                    phi_pre_h=dict(phi),
                )
            )
            for family in CONTROL_FAMILIES:
                control_first_joint = control_joint(
                    base_joint,
                    disruption,
                    family,
                    probe_index,
                )
                control_return = policy_rollout_return(
                    env,
                    jax,
                    probe_state,
                    policy,
                    probe_index,
                    config.rollout_horizon,
                    config.shaped_reward_weight,
                    recovery_seed,
                    first_joint_override=control_first_joint,
                )
                control_records.append(
                    BranchRecord(
                        probe_id=probe_id,
                        policy=config.policy,
                        domain=f"jaxmarl_overcooked_v2:{config.layout}",
                        disruption=disruption,
                        family=family,
                        no_shock_return=no_shock_return,
                        branch_return=control_return,
                        shock_magnitude=action_divergence(
                            [base_joint],
                            [control_first_joint],
                            1,
                        ),
                        phi_pre_h=dict(phi),
                    )
                )

    return semantic_records, control_records


def all_records(
    semantic_records: Iterable[BranchRecord],
    control_records: Iterable[BranchRecord],
) -> list[BranchRecord]:
    return list(semantic_records) + list(control_records)
