from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np
from jaxmarl.environments.overcooked_v2.common import Actions as _OCActions
from src.aris_bellman.specs import PartnerAction

from .option_termination import OptionRuntime
from .state_utils import (
    get_agent_pos,
    get_inventory,
    has_plate,
    is_cooked,
    is_pot_cooking,
    is_pot_ready_for_plate,
)

POLICY_TO_ID = {"yield": 0, "claim": 1}
ID_TO_POLICY = {value: key for key, value in POLICY_TO_ID.items()}

FAMILY_TO_ID = {
    "static_yield": 0,
    "static_claim": 1,
    "patience": 2,
    "block_switch": 3,
    "tit_for_tat": 4,
    "escalate_after_defer": 5,
}
ID_TO_FAMILY = {value: key for key, value in FAMILY_TO_ID.items()}

TRIGGER_TO_ID = {
    "reset": 0,
    "static": 1,
    "opportunity": 2,
    "patience_cutoff": 3,
    "block_switch": 4,
    "ego_claim": 5,
    "ego_defer": 6,
    "epsilon": 7,
    "yield_abort": 8,
}
ID_TO_TRIGGER = {value: key for key, value in TRIGGER_TO_ID.items()}

# Round-2 expression channel (cert r1: state-only BA 0.849 — trajectories telegraphed
# the current policy). Option SELECTION is always claim-shaped; yield expresses only
# as a last-moment abort of the terminal INTERACT, replaced by a retreat option that
# vacates the stand cell. Disposition becomes an event stream, not a trajectory.
TERMINAL_KINDS = ("serve_soup", "plate_soup", "pick_plate")
RETREAT_KIND_ORDER = (
    "clear_interaction_cell",
    "wait_at_bottleneck",
    "cross_bottleneck",
    "wait_duration_after_arrival",
    "wait_duration",
    "drop_item_to_counter",
)  # noop deliberately LAST-resort (codex r2): it camps the stand cell

VALID_MODE_FAMILIES = frozenset(FAMILY_TO_ID)


@dataclass(frozen=True)
class LatentModeSpec:
    family: str
    param: int | None = None
    epsilon: float = 0.1
    initial_policy: str | None = None

    def __post_init__(self) -> None:
        if self.family not in VALID_MODE_FAMILIES:
            choices = ", ".join(sorted(VALID_MODE_FAMILIES))
            raise ValueError(f"unknown latent mode family {self.family!r}; choices={choices}")
        if not (0.0 <= float(self.epsilon) <= 1.0):
            raise ValueError(f"epsilon must be in [0, 1], got {self.epsilon!r}")
        if self.initial_policy is not None and self.initial_policy not in POLICY_TO_ID:
            raise ValueError(f"initial_policy must be claim/yield, got {self.initial_policy!r}")


@dataclass(frozen=True)
class LatentPartnerSpec:
    geometry_profile: str
    base_protocol: Any
    mode: LatentModeSpec
    curriculum_group: str | None = "latent_v3"


@dataclass
class LatentModeRuntime:
    spec: LatentModeSpec
    policy: str = "yield"
    mode_age: int = 0
    option_decisions: int = 0
    opportunity_count: int = 0
    consecutive_ego_defers: int = 0
    yield_remaining: int = 0
    last_trigger: str = "reset"
    in_opportunity: bool = False
    ego_initiated_current: bool = False
    partner_initiated_current: bool = False

    def reset(self, seed: int | None = None) -> None:
        del seed
        self.policy = self._initial_policy()
        self.mode_age = 0
        self.option_decisions = 0
        self.opportunity_count = 0
        self.consecutive_ego_defers = 0
        self.yield_remaining = 0
        self.last_trigger = "reset"
        self.in_opportunity = False
        self.ego_initiated_current = False
        self.partner_initiated_current = False

    def observe_public_transition(
        self,
        *,
        gate_main: bool,
        ego_initiated: bool,
        partner_initiated: bool,
    ) -> None:
        if gate_main and not self.in_opportunity:
            self.in_opportunity = True
            self.ego_initiated_current = False
            self.partner_initiated_current = False
            self.opportunity_count += 1
            self._on_opportunity_start()

        if ego_initiated:
            self.ego_initiated_current = True
            self.consecutive_ego_defers = 0
            self._on_ego_claim()
        if partner_initiated:
            self.partner_initiated_current = True

        if self.in_opportunity and not gate_main:
            if not self.ego_initiated_current:
                self.consecutive_ego_defers += 1
                self._on_ego_defer()
            self.in_opportunity = False
            self.ego_initiated_current = False
            self.partner_initiated_current = False

    def observe_option_boundary(self) -> None:
        self.option_decisions += 1
        self.mode_age += 1
        if self.spec.family != "block_switch":
            return
        dwell = max(1, int(self.spec.param or 1))
        if self.option_decisions > 1 and (self.option_decisions - 1) % dwell == 0:
            self._set_policy("yield" if self.policy == "claim" else "claim", "block_switch")

    def diagnostic_state(self) -> dict[str, int | str]:
        return {
            "mode_policy": self.policy,
            "mode_policy_id": int(POLICY_TO_ID[self.policy]),
            "mode_family": self.spec.family,
            "mode_family_id": int(FAMILY_TO_ID[self.spec.family]),
            "mode_param": -1 if self.spec.param is None else int(self.spec.param),
            "mode_age": int(self.mode_age),
            "mode_opportunity_count": int(self.opportunity_count),
            "mode_last_trigger": self.last_trigger,
            "mode_last_trigger_id": int(TRIGGER_TO_ID.get(self.last_trigger, -1)),
        }

    def _initial_policy(self) -> str:
        if self.spec.initial_policy is not None:
            return str(self.spec.initial_policy)
        if self.spec.family == "static_claim":
            return "claim"
        if self.spec.family == "static_yield":
            return "yield"
        if self.spec.family == "patience":
            return "yield"
        if self.spec.family == "block_switch":
            return "claim"
        if self.spec.family == "tit_for_tat":
            return "claim"
        if self.spec.family == "escalate_after_defer":
            return "yield"
        raise AssertionError(f"unhandled family {self.spec.family!r}")

    def _on_opportunity_start(self) -> None:
        family = self.spec.family
        if family == "patience":
            limit = max(0, int(self.spec.param or 0))
            self._set_policy("yield" if self.opportunity_count <= limit else "claim",
                             "opportunity" if self.opportunity_count <= limit else "patience_cutoff")
        elif family == "tit_for_tat":
            if self.yield_remaining > 0:
                self._set_policy("yield", "opportunity")
                self.yield_remaining -= 1
            elif self.policy != "claim":
                self._set_policy("claim", "opportunity")
        elif family == "static_claim":
            self._set_policy("claim", "static")
        elif family == "static_yield":
            self._set_policy("yield", "static")

    def _on_ego_claim(self) -> None:
        if self.spec.family != "tit_for_tat":
            return
        self.yield_remaining = max(1, int(self.spec.param or 1))
        self._set_policy("yield", "ego_claim")

    def _on_ego_defer(self) -> None:
        if self.spec.family == "tit_for_tat" and self.consecutive_ego_defers >= 2:
            self.yield_remaining = 0
            self._set_policy("claim", "ego_defer")
        elif self.spec.family == "escalate_after_defer":
            limit = max(1, int(self.spec.param or 1))
            if self.consecutive_ego_defers >= limit:
                self._set_policy("claim", "ego_defer")

    def _set_policy(self, policy: str, trigger: str) -> None:
        if policy not in POLICY_TO_ID:
            raise ValueError(f"unknown policy {policy!r}")
        if policy != self.policy:
            self.mode_age = 0
        self.policy = policy
        self.last_trigger = trigger


class LatentModeController:
    """Partner wrapper whose terminal disposition is latent but public-state driven.

    The wrapped scripted partner still emits only a primitive action. True mode
    diagnostics are exposed through ``diagnostic_mode_state`` and are never routed
    to the method evidence path.
    """

    def __init__(
        self,
        *,
        name: str,
        option_library: Any,
        spec: LatentPartnerSpec,
        partner_cls: Callable[..., Any],
        partner_id: int = 0,
    ) -> None:
        self.name = str(name)
        self.option_library = option_library
        self.spec = spec
        self.protocol = spec
        self.partner_id = int(partner_id)
        self.runtime = LatentModeRuntime(spec.mode)
        self._inner = partner_cls(
            name=str(name),
            option_library=option_library,
            protocol=spec.base_protocol,
            partner_id=partner_id,
        )
        self._last_state: Any | None = None
        self._last_gate_main = False
        self._pot_positions = tuple(
            entity.pos
            for entity in option_library.layout_graph.entities.values()
            if entity.kind == "pot"
        )

    def reset(self, seed: int) -> None:
        self.runtime.reset(seed)
        self._inner.reset(seed)
        self._last_state = None
        self._last_gate_main = False
        if hasattr(self, "_behavior_option_inferencer"):
            setattr(self, "_behavior_option_inferencer", None)

    def act(self, obs_partner: Any, state: Any, rng: np.random.Generator) -> PartnerAction:
        self.sync_public_state(state)
        if self._will_choose_new_option(state):
            self.runtime.observe_option_boundary()
            # Round-2 de-telegraphing: selection is ALWAYS claim-shaped so approach
            # trajectories are mode-invariant; the latent policy expresses only via
            # _maybe_yield_abort below (cert r1 C-1/C-4 anatomy).
            self._inner.protocol = replace(
                self.spec.base_protocol,
                terminal_policy="claim",
            )
            if self._epsilon_force_option(state, rng):
                self.runtime.last_trigger = "epsilon"
        action = self._inner.act(obs_partner, state, rng)
        action = self._maybe_yield_abort(action, state)
        self._last_state = state
        self._last_gate_main = _gate_main(state, self._pot_positions)
        return action

    def _maybe_yield_abort(self, action: PartnerAction, state: Any) -> PartnerAction:
        """Last-moment yield expression: replace the terminal INTERACT with a
        retreat option so the stand cell is vacated for the ego. Runs every step
        while policy stays yield, so the partner orbits (approach, veer off,
        re-approach) instead of converting."""
        if self.runtime.policy != "yield":
            return action
        current = getattr(self._inner, "current_option", None)
        if current is None:
            return action
        if str(self.option_library.options[int(current)].kind) not in TERMINAL_KINDS:
            return action
        if int(action.primitive_action) != int(_OCActions.interact):
            return action
        self.runtime.last_trigger = "yield_abort"
        retreat = self._pick_retreat_option(state)
        if retreat is None:
            return replace(action, primitive_action=int(_OCActions.stay))
        self._inner.current_option = int(retreat)
        self._inner.option_runtime = OptionRuntime(
            option_id=int(retreat),
            start_pos=get_agent_pos(state, 1),
        )
        self._inner.elapsed = 0
        new_primitive = int(self.option_library.primitive_action(state, 1, int(retreat)))
        if hasattr(self._inner, "last_primitive_action"):
            self._inner.last_primitive_action = new_primitive
        return replace(action, primitive_action=new_primitive)

    def _pick_retreat_option(self, state: Any) -> int | None:
        valid = self.option_library.valid_options(state, agent_id=1)
        valid_ids = np.flatnonzero(valid)
        if valid_ids.size == 0:
            return None
        kinds = {int(i): str(self.option_library.options[int(i)].kind) for i in valid_ids}
        for want in RETREAT_KIND_ORDER:
            for vid in valid_ids:
                if kinds[int(vid)] == want:
                    return int(vid)
        for vid in valid_ids:  # any mobile non-terminal before noop (vacate the cell)
            if kinds[int(vid)] not in TERMINAL_KINDS and kinds[int(vid)] != "noop":
                return int(vid)
        for vid in valid_ids:  # noop only as the true last resort
            if kinds[int(vid)] == "noop":
                return int(vid)
        return None

    def sync_public_state(self, state: Any) -> None:
        self._observe_public_state(state)
        self._last_state = state
        self._last_gate_main = _gate_main(state, self._pot_positions)

    def diagnostic_mode_state(self) -> dict[str, int | str]:
        return self.runtime.diagnostic_state()

    @property
    def current_option(self) -> int | None:
        return getattr(self._inner, "current_option", None)

    @property
    def last_primitive_action(self) -> int | None:
        return getattr(self._inner, "last_primitive_action", None)

    def _observe_public_state(self, state: Any) -> None:
        gate = _gate_main(state, self._pot_positions)
        ego_initiated = False
        partner_initiated = False
        if self._last_state is not None:
            ego_initiated = _terminal_initiated(self._last_state, state, agent_id=0)
            partner_initiated = _terminal_initiated(self._last_state, state, agent_id=1)
        self.runtime.observe_public_transition(
            gate_main=gate,
            ego_initiated=ego_initiated,
            partner_initiated=partner_initiated,
        )

    def _will_choose_new_option(self, state: Any) -> bool:
        current = getattr(self._inner, "current_option", None)
        if current is None:
            return True
        return bool(self._inner._option_done(state, agent_id=1))

    def _epsilon_force_option(self, state: Any, rng: np.random.Generator) -> bool:
        epsilon = float(self.spec.mode.epsilon)
        if epsilon <= 0.0 or float(rng.random()) >= epsilon:
            return False
        valid = self.option_library.valid_options(state, agent_id=1)
        valid_ids = np.flatnonzero(valid)
        if valid_ids.size == 0:
            return False
        choice = int(rng.choice(valid_ids))
        self._inner.current_option = choice
        self._inner.option_runtime = OptionRuntime(
            option_id=choice,
            start_pos=get_agent_pos(state, 1),
        )
        self._inner.elapsed = 0
        return True


def _gate_main(state: Any, pot_positions: tuple[tuple[int, int], ...]) -> bool:
    n_ready = any(
        is_pot_ready_for_plate(state, pos, require_correct_recipe=False)
        for pos in pot_positions
    )
    n_cooking = any(is_pot_cooking(state, pos) for pos in pot_positions)
    clean_hands = not _carries_soup(get_inventory(state, 0)) and not _carries_soup(
        get_inventory(state, 1)
    )
    return bool((n_ready or n_cooking) and clean_hands)


def _terminal_initiated(prev_state: Any, next_state: Any, *, agent_id: int) -> bool:
    before = get_inventory(prev_state, agent_id)
    after = get_inventory(next_state, agent_id)
    if _carries_soup(after) and not _carries_soup(before):
        return True
    if _carries_soup(before) and not _carries_soup(after):
        return True
    return False


def _carries_soup(inventory: int) -> bool:
    return bool(is_cooked(inventory) and has_plate(inventory))
