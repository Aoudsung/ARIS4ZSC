from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from src.aris_bellman.specs import OptionSpec, PartnerAction

from .state_utils import get_agent_pos, get_inventory, is_empty_inventory

GridPos = tuple[int, int]


class PartnerPolicy(Protocol):
    name: str

    def reset(self, seed: int) -> None: ...

    def act(self, obs_partner: Any, state: Any, rng: np.random.Generator) -> PartnerAction: ...


@dataclass(frozen=True)
class ProtocolSpec:
    role: str | None = None
    bottleneck_policy: str | None = None
    pot_preference: str | None = None
    delivery_preference: str | None = None
    serving_style: str | None = None
    button_policy: str | None = None
    counter_preference: str | None = None


TRAINING_PROTOCOLS: tuple[tuple[str, ProtocolSpec], ...] = (
    (
        "ingredient-near",
        ProtocolSpec(role="ingredient_person", pot_preference="near"),
    ),
    (
        "ingredient-far",
        ProtocolSpec(role="ingredient_person", pot_preference="far"),
    ),
    (
        "dish-server",
        ProtocolSpec(role="dish_person", delivery_preference="nearest"),
    ),
    (
        "server-left",
        ProtocolSpec(role="server", delivery_preference="left"),
    ),
    (
        "bottleneck-yield",
        ProtocolSpec(role="flexible", bottleneck_policy="yield"),
    ),
    (
        "flexible-balanced",
        ProtocolSpec(role="flexible", bottleneck_policy="alternate"),
    ),
)


@dataclass
class ScriptedProtocolPartner:
    name: str
    option_library: Any
    protocol: ProtocolSpec
    current_option: int | None = None
    last_state: Any | None = None
    elapsed: int = 0

    def reset(self, seed: int) -> None:
        self.current_option = None
        self.last_state = None
        self.elapsed = 0

    def act(self, obs_partner: Any, state: Any, rng: np.random.Generator) -> PartnerAction:
        del obs_partner

        if self.current_option is None or self._option_done(state, agent_id=1):
            valid = self.option_library.valid_options(state, agent_id=1)
            self.current_option = self._choose_option(state, valid, rng)
            self.elapsed = 0

        primitive = self.option_library.primitive_action(state, 1, self.current_option)
        self.last_state = state
        self.elapsed += 1
        return PartnerAction(
            primitive_action=int(primitive),
            option_id=int(self.current_option),
            option_confidence=1.0,
            option_dist=_one_hot(self.current_option, self.option_library.num_options),
            source="scripted",
        )

    def _choose_option(
        self,
        state: Any,
        valid: np.ndarray,
        rng: np.random.Generator,
    ) -> int:
        valid_ids = np.flatnonzero(valid)
        if valid_ids.size == 0:
            return 0

        scores = np.asarray(
            [self._protocol_score(self.option_library.options[idx], state) for idx in valid_ids],
            dtype=float,
        )
        best = np.flatnonzero(scores == np.max(scores))
        return int(rng.choice(valid_ids[best]))

    def _protocol_score(self, opt: OptionSpec, state: Any) -> float:
        score = _task_progress_score(opt, state)

        if self.protocol.role == "dish_person":
            score += _role_bonus(opt.kind in {"pick_plate", "plate_soup", "serve_soup"})
        elif self.protocol.role == "ingredient_person":
            score += _role_bonus(
                opt.kind in {"fetch_ingredient", "deliver_ingredient_to_pot"}
            )
        elif self.protocol.role == "server":
            score += _role_bonus(opt.kind == "serve_soup")

        if opt.kind == "deliver_ingredient_to_pot":
            score += _positional_preference_bonus(
                opt.target_pos,
                state,
                self.protocol.pot_preference,
                agent_id=1,
            )
        if opt.kind in {"cross_bottleneck", "wait_at_bottleneck"}:
            score += _bottleneck_bonus(opt, self.protocol.bottleneck_policy, self.elapsed)
        if opt.kind == "serve_soup":
            score += _positional_preference_bonus(
                opt.target_pos,
                state,
                self.protocol.delivery_preference,
                agent_id=1,
            )
        if opt.kind == "press_recipe_button":
            score += _role_bonus(self.protocol.button_policy == "check_first")

        score -= _path_length_penalty(self.option_library, state, opt, agent_id=1)
        score -= _congestion_penalty(opt, state)
        return score

    def _option_done(self, state: Any, agent_id: int) -> bool:
        opt = self.option_library.options[self.current_option]
        if self.elapsed >= opt.max_steps:
            return True
        if not self.option_library.is_valid_for_state(state, agent_id, opt.id):
            return True
        if opt.kind == "noop":
            return True
        if opt.kind == "wait_at_bottleneck":
            wait_duration = (opt.metadata or {}).get("wait_duration", 2)
            return self.elapsed >= wait_duration
        if opt.kind == "cross_bottleneck":
            return get_agent_pos(state, agent_id) in _region_cells(opt)
        return False


def make_training_partners(option_library: Any) -> list[ScriptedProtocolPartner]:
    return [
        ScriptedProtocolPartner(name=name, option_library=option_library, protocol=protocol)
        for name, protocol in TRAINING_PROTOCOLS
    ]


def _one_hot(option_id: int, num_options: int) -> np.ndarray:
    dist = np.zeros((num_options,), dtype=np.float32)
    dist[int(option_id)] = 1.0
    return dist


def _task_progress_score(opt: OptionSpec, state: Any) -> float:
    inventory = get_inventory(state, 1)
    if is_empty_inventory(inventory) and opt.kind in {"fetch_ingredient", "pick_plate"}:
        return 2.0
    if opt.kind in {"deliver_ingredient_to_pot", "plate_soup", "serve_soup"}:
        return 2.0
    if opt.kind in {"cross_bottleneck", "wait_at_bottleneck"}:
        return 0.5
    if opt.kind == "noop":
        return -2.0
    return 0.0


def _role_bonus(condition: bool) -> float:
    return 4.0 if condition else -1.0


def _positional_preference_bonus(
    target_pos: GridPos | None,
    state: Any,
    preference: str | None,
    agent_id: int,
) -> float:
    if target_pos is None or preference is None:
        return 0.0

    agent_pos = get_agent_pos(state, agent_id)
    distance = _manhattan(agent_pos, target_pos)
    if preference in {"near", "nearest"}:
        return -0.25 * distance
    if preference == "far":
        return 0.25 * distance
    if preference == "left":
        return -0.25 * target_pos[0]
    if preference == "right":
        return 0.25 * target_pos[0]
    return 0.0


def _bottleneck_bonus(
    opt: OptionSpec,
    policy: str | None,
    elapsed: int,
) -> float:
    if policy == "yield":
        return 3.0 if opt.kind == "wait_at_bottleneck" else -1.0
    if policy == "push":
        return 3.0 if opt.kind == "cross_bottleneck" else -1.0
    if policy == "alternate":
        prefer_cross = elapsed % 2 == 0
        if prefer_cross and opt.kind == "cross_bottleneck":
            return 1.5
        if not prefer_cross and opt.kind == "wait_at_bottleneck":
            return 1.5
    return 0.0


def _path_length_penalty(
    option_library: Any,
    state: Any,
    opt: OptionSpec,
    agent_id: int,
) -> float:
    cost = option_library.expected_cost(state, agent_id, opt.id)
    if not np.isfinite(cost):
        return 10.0
    return 0.1 * cost


def _congestion_penalty(opt: OptionSpec, state: Any) -> float:
    if opt.target_pos is None:
        return 0.0
    ego_pos = get_agent_pos(state, 0)
    partner_pos = get_agent_pos(state, 1)
    return 1.0 if _manhattan(ego_pos, opt.target_pos) <= 1 and partner_pos != ego_pos else 0.0


def _region_cells(opt: OptionSpec) -> tuple[GridPos, ...]:
    return tuple((opt.metadata or {}).get("region_cells", ()))


def _manhattan(a: GridPos, b: GridPos) -> int:
    return int(abs(a[0] - b[0]) + abs(a[1] - b[1]))
