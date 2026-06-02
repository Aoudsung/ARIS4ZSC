"""Classic Overcooked-AI adapter for resettable SRVF-MAPPO interventions."""

from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from raob.benchmarks.base import BenchmarkAdapter, BenchmarkStep, InterventionSnapshot
from raob.types import AffordanceBatch


CLASSIC_OVERCOOKED_LAYOUTS = (
    "cramped_room",
    "asymmetric_advantages",
    "coordination_ring",
    "forced_coordination",
    "counter_circuit",
    "counter_circuit_o_1order",
    "counter_circuit_6x5_2pots_3orders",
    "distant_tomato",
)

CLASSIC_OVERCOOKED_LAYOUT_ALIASES = {
    "multi_strategy_counter": "counter_circuit_6x5_2pots_3orders",
    "diverse_counter_circuit_6x5": "counter_circuit_6x5_2pots_3orders",
}

GOAT_STAGE1_REWARD_SHAPING_PARAMS = {
    "PLACEMENT_IN_POT_REW": 3,
    "DISH_PICKUP_REWARD": 3,
    "SOUP_PICKUP_REWARD": 5,
    "PICKUP_TOMATO_REWARD": 0,
    "DISH_DISP_DISTANCE_REW": 0,
    "POT_DISTANCE_REW": 0,
    "SOUP_DISTANCE_REW": 0,
    "USEFUL_TOMATO_PICKUP": 0,
    "FOLLOW_TOMATO": 0,
    "PLACE_FIRST_TOMATO": 0,
}


def _layout_alias_key(layout: str) -> str:
    return "_".join(str(layout).strip().lower().replace("-", " ").replace("_", " ").split())


def canonical_classic_layout_name(layout: str) -> str:
    """Return the installed classic Overcooked-AI layout name for a user layout label."""

    layout_name = str(layout).strip()
    alias_key = _layout_alias_key(layout_name)
    return CLASSIC_OVERCOOKED_LAYOUT_ALIASES.get(alias_key, layout_name)


@dataclass(frozen=True)
class ClassicOvercookedModules:
    action: Any
    gridworld: Any
    env: Any


def import_classic_overcooked() -> ClassicOvercookedModules:
    """Import classic Overcooked-AI modules installed from overcooked_berkeley."""

    try:
        if not hasattr(np, "Inf"):
            setattr(np, "Inf", np.inf)
        from overcooked_ai_py.mdp.actions import Action
        from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
        from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
    except Exception as exc:  # pragma: no cover - depends on server-only checkout
        raise RuntimeError(
            "Classic Overcooked-AI is not installed. Install goat_overcooked's "
            "mapbt/envs/overcooked/overcooked_berkeley package on the server."
        ) from exc
    return ClassicOvercookedModules(
        action=Action,
        gridworld=OvercookedGridworld,
        env=OvercookedEnv,
    )


class ClassicOvercookedBenchmarkAdapter(BenchmarkAdapter):
    """Resettable adapter for classic Overcooked-AI, not OvercookedV2."""

    benchmark_name = "overcooked_classic"

    def __init__(
        self,
        *,
        layout: str = "cramped_room",
        horizon: int = 400,
        old_dynamics: bool = False,
        reward_shaping_params: Mapping[str, float] | None = None,
        layout_params: Mapping[str, Any] | None = None,
        info_level: int = 0,
    ) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        self.requested_layout = str(layout).strip()
        self.layout = canonical_classic_layout_name(self.requested_layout)
        self.horizon = int(horizon)
        self.old_dynamics = bool(old_dynamics)
        self.reward_shaping_params = (
            dict(GOAT_STAGE1_REWARD_SHAPING_PARAMS)
            if reward_shaping_params is None
            else dict(reward_shaping_params)
        )
        self.layout_params = dict(layout_params or {})
        self.info_level = int(info_level)
        self.modules = import_classic_overcooked()
        self.num_actions = int(self.modules.action.NUM_ACTIONS)
        mdp_kwargs = {
            "old_dynamics": self.old_dynamics,
            "rew_shaping_params": self.reward_shaping_params,
            **self.layout_params,
        }
        self._mdp = self.modules.gridworld.from_layout_name(
            self.layout,
            **mdp_kwargs,
        )
        self._env = self.modules.env.from_mdp(
            self._mdp,
            horizon=self.horizon,
            info_level=self.info_level,
        )
        self.adapter_receives_beta = False
        self.affordance_uses_current_public_state = True

    @property
    def env(self) -> Any:
        return self._env

    @property
    def mdp(self) -> Any:
        return self._mdp

    def reset(self, seed: int) -> Mapping[str, Any]:
        _ = int(seed)
        self._env.reset(regen_mdp=False)
        return self.public_observation()

    def public_observation(self) -> Mapping[str, Any]:
        observation: dict[str, Any] = {
            "benchmark": self.benchmark_name,
            "layout": self.layout,
            "horizon": self.horizon,
            "old_dynamics": self.old_dynamics,
            "mdp": self._mdp,
            "state": self._env.state,
            "timestep": int(self._env.state.timestep),
        }
        if self.requested_layout != self.layout:
            observation["requested_layout"] = self.requested_layout
        return observation

    def _state_g(self, state: Any | None = None) -> torch.Tensor:
        current_state = self._env.state if state is None else state
        encoded = self._mdp.lossless_state_encoding(current_state, self.horizon)
        agent_zero_view = np.asarray(encoded[0], dtype=np.float32)
        return torch.from_numpy(agent_zero_view.reshape(-1).copy()).to(dtype=torch.float32)

    def _agent_encoding(self, agent_index: int, state: Any | None = None) -> torch.Tensor:
        agent_index = int(agent_index)
        if agent_index not in (0, 1):
            raise ValueError("agent_index must be 0 or 1")
        current_state = self._env.state if state is None else state
        encoded = self._mdp.lossless_state_encoding(current_state, self.horizon)
        view = np.asarray(encoded[agent_index], dtype=np.float32)
        return torch.from_numpy(view.reshape(-1).copy()).to(dtype=torch.float32)

    def ego_observation_tensor(self, *, agent_index: int = 0) -> torch.Tensor:
        """Flattened ego observation for SRVF-MAPPO actor input."""

        return self._agent_encoding(agent_index)

    def global_state_tensor(self) -> torch.Tensor:
        """Flattened centralized state for SRVF-MAPPO critic input."""

        return torch.cat([self._agent_encoding(0), self._agent_encoding(1)], dim=0)

    def public_chart_tensor(self) -> torch.Tensor:
        """Canonical public chart `g`; default classic contract uses agent-0 view."""

        return self._state_g()

    def legal_action_mask(self) -> torch.Tensor:
        return torch.ones(self.num_actions, dtype=torch.bool)

    def extract_affordance(self) -> AffordanceBatch:
        state_g = self._state_g()
        return AffordanceBatch(
            g=state_g,
            metadata={
                "benchmark": self.benchmark_name,
                "layout": self.layout,
                **(
                    {"requested_layout": self.requested_layout}
                    if self.requested_layout != self.layout
                    else {}
                ),
                "encoding": "overcooked_ai_py.lossless_state_encoding",
                "encoding_view": "agent_0",
                "goat_reference": "mapbt.envs.overcooked.Overcooked_Env",
                "encoding_shape": tuple(
                    int(dim)
                    for dim in np.asarray(
                        self._mdp.lossless_state_encoding(
                            self._env.state,
                            self.horizon,
                        )[0]
                    ).shape
                ),
            },
        )

    def snapshot(self, *, phase_id: str = "unknown") -> InterventionSnapshot:
        state = self._env.state.deepcopy()
        return InterventionSnapshot(
            state_g=self._state_g(state).detach().cpu(),
            opaque_state={
                "state": state,
                "game_stats": copy.deepcopy(self._env.game_stats),
            },
            observation=self.public_observation(),
            phase_id=phase_id,
            metadata={
                "benchmark": self.benchmark_name,
                "layout": self.layout,
                **(
                    {"requested_layout": self.requested_layout}
                    if self.requested_layout != self.layout
                    else {}
                ),
                "horizon": self.horizon,
                "old_dynamics": self.old_dynamics,
                "encoding": "overcooked_ai_py.lossless_state_encoding",
                "encoding_view": "agent_0",
                "goat_reference": "mapbt.envs.overcooked.Overcooked_Env",
                "reward": "raw sparse reward from OvercookedEnv.step",
            },
        )

    def restore(self, snapshot: InterventionSnapshot) -> Mapping[str, Any]:
        opaque = snapshot.opaque_state
        if not isinstance(opaque, Mapping):
            raise TypeError("classic Overcooked snapshot opaque_state must be a mapping")
        self._env.state = opaque["state"].deepcopy()
        self._env.game_stats = copy.deepcopy(opaque["game_stats"])
        return self.public_observation()

    def action_from_index(self, action_idx: int) -> Any:
        action_idx = int(action_idx)
        if action_idx < 0 or action_idx >= self.num_actions:
            raise ValueError(f"action index out of range: {action_idx}")
        return self.modules.action.INDEX_TO_ACTION[action_idx]

    def step(self, ego_action: int, partner_action: int) -> BenchmarkStep:
        joint_action = (
            self.action_from_index(ego_action),
            self.action_from_index(partner_action),
        )
        _next_state, reward, done, info = self._env.step(joint_action)
        return BenchmarkStep(
            observation=self.public_observation(),
            reward=float(reward),
            done=bool(done),
            info=info,
            affordance=self.extract_affordance(),
            state_g=self._state_g(),
        )

    def close(self) -> None:
        return None
