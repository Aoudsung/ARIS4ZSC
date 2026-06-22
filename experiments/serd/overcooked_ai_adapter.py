"""Overcooked-AI adapter for SERD branch-record smoke tests.

The adapter intentionally stays MDP-only: it uses OvercookedGridworld state
transitions to verify same-state branching and BranchRecord emission before
FCP/PECAN checkpoints are available. Scripted policies here are not scientific
evidence for the paper claim.
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


ACTION_ALIASES = {
    "north": (0, -1),
    "south": (0, 1),
    "east": (1, 0),
    "west": (-1, 0),
    "stay": (0, 0),
    "interact": "interact",
}


class OvercookedDependencyError(RuntimeError):
    """Raised when Overcooked-AI cannot be imported for adapter execution."""


@dataclass(frozen=True)
class OvercookedSmokeConfig:
    layout: str = "counter_circuit"
    policy: str = "scripted_smoke"
    disruption: str = "missed_handoff"
    probes: int = 8
    horizon: int = 8
    shock_horizon: int = 1
    shaped_reward_weight: float = 1.0
    overcooked_src: str | None = None


@dataclass(frozen=True)
class OvercookedM1Config:
    layout: str = "counter_circuit"
    policy: str = "handcoded_counter_circuit_policy"
    disruptions: tuple[str, ...] = ("missed_handoff", "route_block", "hesitation")
    probes_per_disruption: int = 50
    rollout_horizon: int = 20
    warmup_horizon: int = 20
    shock_horizon: int = 1
    shaped_reward_weight: float = 1.0
    overcooked_src: str | None = None


@dataclass(frozen=True)
class CounterCircuitHandcodedPolicy:
    """Small deterministic policy used to exercise policy-driven branching.

    This is an adapter validation policy, not FCP, PECAN, or a claim-bearing
    baseline. Its purpose is to make no-shock and post-intervention actions come
    from a policy object instead of fixed branch scripts.
    """

    action_cycle: tuple[tuple[str, str], ...] = (
        ("east", "west"),
        ("north", "south"),
        ("interact", "interact"),
        ("south", "north"),
        ("west", "east"),
        ("stay", "stay"),
    )

    def action_names(self, state, step_index: int, probe_index: int) -> tuple[str, str]:
        offset = int(getattr(state, "timestep", 0)) + step_index + probe_index
        return self.action_cycle[offset % len(self.action_cycle)]


def _install_mdp_only_gymnasium_shim() -> None:
    """Install the minimal module shim needed by overcooked_ai_py.__init__.

    The Overcooked-AI MDP code used here does not need Gymnasium, but the
    package-level initializer imports `gymnasium.envs.registration.register`.
    This shim is deliberately narrow and only used when Gymnasium is absent.
    """

    if "gymnasium" in sys.modules:
        return
    if importlib.util.find_spec("gymnasium") is not None:
        return

    registration = types.ModuleType("gymnasium.envs.registration")
    registration.register = lambda *args, **kwargs: None
    envs = types.ModuleType("gymnasium.envs")
    envs.registration = registration
    gymnasium = types.ModuleType("gymnasium")
    gymnasium.envs = envs
    sys.modules["gymnasium"] = gymnasium
    sys.modules["gymnasium.envs"] = envs
    sys.modules["gymnasium.envs.registration"] = registration


def ensure_overcooked_ai(overcooked_src: str | None = None):
    """Return `(OvercookedGridworld, Action)` or raise a visible dependency error."""

    if overcooked_src:
        src_path = Path(overcooked_src).expanduser().resolve()
        if not src_path.exists():
            raise OvercookedDependencyError(
                f"Overcooked-AI source path does not exist: {src_path}"
            )
        sys.path.insert(0, str(src_path))

    _install_mdp_only_gymnasium_shim()

    try:
        from overcooked_ai_py.mdp.actions import Action
        from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
    except ModuleNotFoundError as exc:
        raise OvercookedDependencyError(
            "Cannot import Overcooked-AI. Install `overcooked-ai` or pass "
            "`--overcooked-src /path/to/overcooked_ai/src`."
        ) from exc

    return OvercookedGridworld, Action


def action_from_name(name: str):
    try:
        return ACTION_ALIASES[name]
    except KeyError as exc:
        raise ValueError(f"unknown Overcooked action alias: {name}") from exc


def _joint_action(names: tuple[str, str]):
    return (action_from_name(names[0]), action_from_name(names[1]))


def _extend_script(script: Sequence[tuple[str, str]], horizon: int) -> list[tuple[str, str]]:
    if not script:
        raise ValueError("action script must contain at least one joint action")
    if len(script) >= horizon:
        return list(script[:horizon])
    return list(script) + [script[-1]] * (horizon - len(script))


def _scripted_branches(horizon: int) -> tuple[list[tuple[str, str]], dict[str, list[tuple[str, str]]]]:
    """Return no-shock plus semantic/control scripts with matched first-step size."""

    no_shock = _extend_script([("east", "west"), ("stay", "stay")], horizon)
    branches = {
        "semantic": _extend_script([("stay", "west"), ("stay", "stay")], horizon),
        "random_lag": _extend_script([("east", "stay"), ("stay", "stay")], horizon),
        "state_block": _extend_script([("north", "west"), ("stay", "stay")], horizon),
        "reward_shaping": _extend_script([("interact", "west"), ("stay", "stay")], horizon),
        "naive_replanning": _extend_script([("east", "interact"), ("stay", "stay")], horizon),
    }
    return no_shock, branches


def _action_divergence(
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
    alternatives = ("stay", "interact", "north", "south", "east", "west")
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


def _semantic_joint(
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


def _control_joint(
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
        "state_block": "north",
        "reward_shaping": "south",
        "naive_replanning": "west",
    }
    try:
        preferred = preferred_by_family[family]
    except KeyError as exc:
        raise ValueError(f"unknown control family: {family}") from exc
    return _replace_one_action(base_joint, actor_index, preferred)


def _object_name(obj) -> str:
    return str(getattr(obj, "name", "unknown"))


def state_phi_pre_h(state) -> dict[str, float]:
    """Extract pre-shock covariates without reading post-branch outcomes."""

    phi: dict[str, float] = {"timestep": float(getattr(state, "timestep", 0))}
    for idx, player in enumerate(state.players):
        x, y = player.position
        ox, oy = player.orientation
        phi[f"p{idx}_x"] = float(x)
        phi[f"p{idx}_y"] = float(y)
        phi[f"p{idx}_or_x"] = float(ox)
        phi[f"p{idx}_or_y"] = float(oy)
        phi[f"p{idx}_holding"] = 1.0 if player.has_object() else 0.0

    object_counts = {"onion": 0, "tomato": 0, "dish": 0, "soup": 0}
    for obj in state.objects.values():
        object_counts[_object_name(obj)] = object_counts.get(_object_name(obj), 0) + 1
    for name, count in sorted(object_counts.items()):
        phi[f"objects_{name}"] = float(count)
    return phi


def rollout_return(
    mdp,
    start_state,
    action_script: Sequence[tuple[str, str]],
    horizon: int,
    shaped_reward_weight: float,
) -> float:
    state = start_state.deepcopy()
    total = 0.0
    for joint_names in _extend_script(action_script, horizon):
        next_state, infos = mdp.get_state_transition(state, _joint_action(joint_names))
        sparse = float(sum(infos.get("sparse_reward_by_agent", [0.0, 0.0])))
        shaped = float(sum(infos.get("shaped_reward_by_agent", [0.0, 0.0])))
        total += sparse + shaped_reward_weight * shaped
        state = next_state
    return total


def policy_rollout_return(
    mdp,
    start_state,
    policy: CounterCircuitHandcodedPolicy,
    probe_index: int,
    horizon: int,
    shaped_reward_weight: float,
    first_joint_override: tuple[str, str] | None = None,
) -> float:
    state = start_state.deepcopy()
    total = 0.0
    for step_index in range(horizon):
        joint_names = policy.action_names(state, step_index, probe_index)
        if step_index == 0 and first_joint_override is not None:
            joint_names = first_joint_override
        next_state, infos = mdp.get_state_transition(state, _joint_action(joint_names))
        sparse = float(sum(infos.get("sparse_reward_by_agent", [0.0, 0.0])))
        shaped = float(sum(infos.get("shaped_reward_by_agent", [0.0, 0.0])))
        total += sparse + shaped_reward_weight * shaped
        state = next_state
    return total


def _advance_prefix_state(mdp, start_state, probe_index: int):
    prefix_actions = [("east", "west"), ("stay", "stay"), ("north", "south")]
    state = start_state.deepcopy()
    for idx in range(probe_index % len(prefix_actions)):
        state, _infos = mdp.get_state_transition(state, _joint_action(prefix_actions[idx]))
    return state


def _advance_policy_prefix_state(
    mdp,
    start_state,
    policy: CounterCircuitHandcodedPolicy,
    probe_index: int,
    warmup_horizon: int,
):
    state = start_state.deepcopy()
    for step_index in range(probe_index % max(1, warmup_horizon)):
        joint_names = policy.action_names(state, step_index, probe_index)
        state, _infos = mdp.get_state_transition(state, _joint_action(joint_names))
    return state


def make_overcooked_smoke_records(
    config: OvercookedSmokeConfig,
) -> tuple[list[BranchRecord], list[BranchRecord]]:
    OvercookedGridworld, _Action = ensure_overcooked_ai(config.overcooked_src)
    mdp = OvercookedGridworld.from_layout_name(config.layout)
    start_state = mdp.get_standard_start_state()
    no_shock_script, branch_scripts = _scripted_branches(config.horizon)
    semantic_script = branch_scripts["semantic"]
    semantic_shock = _action_divergence(
        no_shock_script, semantic_script, config.shock_horizon
    )

    semantic_records: list[BranchRecord] = []
    control_records: list[BranchRecord] = []

    for probe_index in range(config.probes):
        probe_state = _advance_prefix_state(mdp, start_state, probe_index)
        phi = state_phi_pre_h(probe_state)
        no_shock_return = rollout_return(
            mdp,
            probe_state,
            no_shock_script,
            config.horizon,
            config.shaped_reward_weight,
        )
        probe_id = f"{config.policy}_{config.layout}_{probe_index:03d}"
        semantic_return = rollout_return(
            mdp,
            probe_state,
            semantic_script,
            config.horizon,
            config.shaped_reward_weight,
        )
        semantic_records.append(
            BranchRecord(
                probe_id=probe_id,
                policy=config.policy,
                domain=f"overcooked_ai:{config.layout}",
                disruption=config.disruption,
                family="semantic",
                no_shock_return=no_shock_return,
                branch_return=semantic_return,
                shock_magnitude=semantic_shock,
                phi_pre_h=dict(phi),
            )
        )
        for family in CONTROL_FAMILIES:
            control_script = branch_scripts[family]
            control_return = rollout_return(
                mdp,
                probe_state,
                control_script,
                config.horizon,
                config.shaped_reward_weight,
            )
            control_records.append(
                BranchRecord(
                    probe_id=probe_id,
                    policy=config.policy,
                    domain=f"overcooked_ai:{config.layout}",
                    disruption=config.disruption,
                    family=family,
                    no_shock_return=no_shock_return,
                    branch_return=control_return,
                    shock_magnitude=_action_divergence(
                        no_shock_script, control_script, config.shock_horizon
                    ),
                    phi_pre_h=dict(phi),
                )
            )

    return semantic_records, control_records


def make_overcooked_m1_policy_records(
    config: OvercookedM1Config,
) -> tuple[list[BranchRecord], list[BranchRecord]]:
    if config.shock_horizon != 1:
        raise ValueError("policy-driven M1 adapter currently supports shock_horizon=1")

    OvercookedGridworld, _Action = ensure_overcooked_ai(config.overcooked_src)
    mdp = OvercookedGridworld.from_layout_name(config.layout)
    start_state = mdp.get_standard_start_state()
    policy = CounterCircuitHandcodedPolicy()

    semantic_records: list[BranchRecord] = []
    control_records: list[BranchRecord] = []

    for disruption in config.disruptions:
        for probe_index in range(config.probes_per_disruption):
            probe_state = _advance_policy_prefix_state(
                mdp,
                start_state,
                policy,
                probe_index,
                config.warmup_horizon,
            )
            phi = state_phi_pre_h(probe_state)
            base_joint = policy.action_names(probe_state, 0, probe_index)
            semantic_joint = _semantic_joint(base_joint, disruption, probe_index)
            semantic_shock = _action_divergence([base_joint], [semantic_joint], 1)
            no_shock_return = policy_rollout_return(
                mdp,
                probe_state,
                policy,
                probe_index,
                config.rollout_horizon,
                config.shaped_reward_weight,
            )
            semantic_return = policy_rollout_return(
                mdp,
                probe_state,
                policy,
                probe_index,
                config.rollout_horizon,
                config.shaped_reward_weight,
                first_joint_override=semantic_joint,
            )
            probe_id = f"{config.policy}_{config.layout}_{disruption}_{probe_index:03d}"
            semantic_records.append(
                BranchRecord(
                    probe_id=probe_id,
                    policy=config.policy,
                    domain=f"overcooked_ai:{config.layout}",
                    disruption=disruption,
                    family="semantic",
                    no_shock_return=no_shock_return,
                    branch_return=semantic_return,
                    shock_magnitude=semantic_shock,
                    phi_pre_h=dict(phi),
                )
            )
            for family in CONTROL_FAMILIES:
                control_joint = _control_joint(base_joint, disruption, family, probe_index)
                control_return = policy_rollout_return(
                    mdp,
                    probe_state,
                    policy,
                    probe_index,
                    config.rollout_horizon,
                    config.shaped_reward_weight,
                    first_joint_override=control_joint,
                )
                control_records.append(
                    BranchRecord(
                        probe_id=probe_id,
                        policy=config.policy,
                        domain=f"overcooked_ai:{config.layout}",
                        disruption=disruption,
                        family=family,
                        no_shock_return=no_shock_return,
                        branch_return=control_return,
                        shock_magnitude=_action_divergence([base_joint], [control_joint], 1),
                        phi_pre_h=dict(phi),
                    )
                )

    return semantic_records, control_records


def all_records(
    semantic_records: Iterable[BranchRecord],
    control_records: Iterable[BranchRecord],
) -> list[BranchRecord]:
    return list(semantic_records) + list(control_records)
