from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from jaxmarl.environments.overcooked_v2.common import Actions
from src.aris_bellman.specs import OptionSpec, PartnerAction

from .option_termination import OptionRuntime, cross_bottleneck_terminated
from .state_utils import (
    get_agent_pos,
    get_inventory,
    is_empty_inventory,
    is_ingredient,
    pot_accepts_inventory_ingredient,
)

GridPos = tuple[int, int]

TERMINAL_KINDS = frozenset({"pick_plate", "plate_soup", "serve_soup"})
PREP_KINDS = frozenset({"fetch_ingredient", "deliver_ingredient_to_pot"})
SUPPORT_KINDS = frozenset({
    "drop_item_to_counter",
    "clear_interaction_cell",
    "wait_at_bottleneck",
    "cross_bottleneck",
})


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
    terminal_policy: str | None = None
    button_policy: str | None = None
    counter_preference: str | None = None
    curriculum_group: str | None = None


STANDARD7_PROTOCOLS: tuple[tuple[str, ProtocolSpec], ...] = (
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
    (
        "terminal-yield",
        ProtocolSpec(
            role="prep_partner",
            pot_preference="near",
            bottleneck_policy="yield",
            terminal_policy="yield",
        ),
    ),
)

TRAINING_PROTOCOLS = STANDARD7_PROTOCOLS  # backward-compat alias

ROLE_CONDITIONED_V1_PROTOCOLS: tuple[tuple[str, ProtocolSpec], ...] = (
    (
        "ingredient-near-yield",
        ProtocolSpec(
            role="ingredient_person",
            pot_preference="near",
            terminal_policy="yield",
            curriculum_group="terminal_yield",
        ),
    ),
    (
        "ingredient-far-yield",
        ProtocolSpec(
            role="ingredient_person",
            pot_preference="far",
            terminal_policy="yield",
            curriculum_group="terminal_yield",
        ),
    ),
    (
        "server-left-claim",
        ProtocolSpec(
            role="server",
            delivery_preference="left",
            terminal_policy="claim",
            curriculum_group="terminal_claim",
        ),
    ),
    (
        "server-right-claim",
        ProtocolSpec(
            role="server",
            delivery_preference="right",
            terminal_policy="claim",
            curriculum_group="terminal_claim",
        ),
    ),
    (
        "bottleneck-yield-terminal-yield",
        ProtocolSpec(
            role="flexible",
            bottleneck_policy="yield",
            terminal_policy="yield",
            curriculum_group="terminal_yield",
        ),
    ),
    (
        "bottleneck-push-terminal-claim",
        ProtocolSpec(
            role="flexible",
            bottleneck_policy="push",
            terminal_policy="claim",
            curriculum_group="terminal_claim",
        ),
    ),
    (
        "heldout-yield-terminal-claim",
        ProtocolSpec(
            role="flexible",
            bottleneck_policy="yield",
            terminal_policy="claim",
            curriculum_group="terminal_claim",
        ),
    ),
    (
        "heldout-push-terminal-yield",
        ProtocolSpec(
            role="flexible",
            bottleneck_policy="push",
            terminal_policy="yield",
            curriculum_group="terminal_yield",
        ),
    ),
)

ROLE_CONDITIONED_V2_PROTOCOLS: tuple[tuple[str, ProtocolSpec], ...] = (
    *ROLE_CONDITIONED_V1_PROTOCOLS[:6],
    (
        "heldout-handoff-alternate-yield",
        ProtocolSpec(
            role="flexible",
            pot_preference="far",
            bottleneck_policy="alternate",
            terminal_policy="yield",
            counter_preference="handoff",
            curriculum_group="heldout_terminal_yield",
        ),
    ),
    (
        "heldout-resource-server-claim",
        ProtocolSpec(
            role="ingredient_person",
            pot_preference="far",
            delivery_preference="right",
            bottleneck_policy="push",
            terminal_policy="claim",
            counter_preference="clear",
            curriculum_group="heldout_terminal_claim",
        ),
    ),
)

BLIND_V1_PROTOCOLS: tuple[tuple[str, ProtocolSpec], ...] = (
    # FORMAL BLIND ROUND partner set (METHOD_LOCK sec18.14, signed 2026-07-06).
    # Constructed from parameter-space principles ONLY — no model behavior was
    # consulted. SINGLE-LOOK RULE: no trained checkpoint may be evaluated against
    # any "blind-*" partner before the formal preregistered run; certification
    # probes are checkpoint-free (scripted-ego only). This registry must NEVER be
    # merged into a training partner_set.
    (
        "blind-dish-yield",
        ProtocolSpec(
            role="dish_person",
            terminal_policy="yield",
            curriculum_group="blind_yield",
        ),
    ),
    (
        "blind-prep-alternate-yield",
        ProtocolSpec(
            role="prep_partner",
            pot_preference="far",
            bottleneck_policy="alternate",
            terminal_policy="yield",
            curriculum_group="blind_yield",
        ),
    ),
    (
        "blind-dish-claim",
        ProtocolSpec(
            role="dish_person",
            terminal_policy="claim",
            delivery_preference="nearest",
            curriculum_group="blind_claim",
        ),
    ),
    (
        "blind-server-nearest-claim-push",
        ProtocolSpec(
            role="server",
            delivery_preference="nearest",
            bottleneck_policy="push",
            terminal_policy="claim",
            counter_preference="clear",
            curriculum_group="blind_claim",
        ),
    ),
    (
        "blind-bottleneck-alternate-neutral",
        ProtocolSpec(
            role="flexible",
            bottleneck_policy="alternate",
            counter_preference="clear",
            curriculum_group="blind_offaxis",
        ),
    ),
    (
        "blind-ingredient-near-neutral",
        ProtocolSpec(
            role="ingredient_person",
            pot_preference="near",
            counter_preference="handoff",
            curriculum_group="blind_offaxis",
        ),
    ),
)

PARTNER_REGISTRIES: dict[str, tuple[tuple[str, ProtocolSpec], ...]] = {
    "standard7": STANDARD7_PROTOCOLS,
    "role_conditioned_v1": ROLE_CONDITIONED_V1_PROTOCOLS,
    "role_conditioned_v2": ROLE_CONDITIONED_V2_PROTOCOLS,
    # Eval-only blind set (sec18.14). Never a training partner_set.
    "blind_v1": BLIND_V1_PROTOCOLS,
}


@dataclass
class ScriptedProtocolPartner:
    name: str
    option_library: Any
    protocol: ProtocolSpec
    partner_id: int = 0
    current_option: int | None = None
    last_state: Any | None = None
    last_primitive_action: int | None = None
    option_runtime: OptionRuntime | None = None
    elapsed: int = 0
    bottleneck_alternate_phase: int = 0

    def reset(self, seed: int) -> None:
        self.current_option = None
        self.last_state = None
        self.last_primitive_action = None
        self.option_runtime = None
        self.elapsed = 0
        self.bottleneck_alternate_phase = 0
        if hasattr(self, "_behavior_option_inferencer"):
            setattr(self, "_behavior_option_inferencer", None)

    def act(self, obs_partner: Any, state: Any, rng: np.random.Generator) -> PartnerAction:
        del obs_partner

        self._update_option_runtime(state, agent_id=1)
        if self.current_option is None or self._option_done(state, agent_id=1):
            valid = self.option_library.valid_options(state, agent_id=1)
            self.current_option = self._choose_option(state, valid, rng)
            self.option_runtime = OptionRuntime(
                option_id=int(self.current_option),
                start_pos=get_agent_pos(state, 1),
            )
            self.elapsed = 0

        primitive = self.option_library.primitive_action(state, 1, self.current_option)
        self.last_state = state
        self.last_primitive_action = int(primitive)
        self.elapsed += 1
        # P1: scripted true option labels are diagnostic truth and must not enter
        # the main train/eval/CE evidence path. The shared executor infers partner
        # options from primitive action and state deltas instead.
        return PartnerAction(
            primitive_action=int(primitive),
            option_id=None,
            option_confidence=0.0,
            option_dist=None,
            source="scripted_primitive_only",
        )

    def _choose_option(
        self,
        state: Any,
        valid: np.ndarray,
        rng: np.random.Generator,
    ) -> int:
        valid_ids = np.flatnonzero(valid)
        if valid_ids.size == 0:
            return _noop_option_id(self.option_library)

        scores = np.asarray(
            [self._protocol_score(self.option_library.options[idx], state) for idx in valid_ids],
            dtype=float,
        )
        best = np.flatnonzero(scores == np.max(scores))
        choice = int(rng.choice(valid_ids[best]))
        chosen_kind = str(self.option_library.options[choice].kind)
        if self.protocol.bottleneck_policy == "alternate" and chosen_kind in {
            "cross_bottleneck",
            "wait_at_bottleneck",
        }:
            self.bottleneck_alternate_phase = 1 - int(self.bottleneck_alternate_phase)
        return choice

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
        elif self.protocol.role == "prep_partner":
            # Train-only curriculum partner: create ego-owned terminal-stage data
            # distribution without adding a fallback controller or supervised loss.
            score += _role_bonus(
                opt.kind in {
                    "fetch_ingredient",
                    "deliver_ingredient_to_pot",
                    "drop_item_to_counter",
                    "clear_interaction_cell",
                    "wait_at_bottleneck",
                }
            )

        score += _terminal_policy_bonus(opt, self.protocol.terminal_policy)

        if opt.kind == "deliver_ingredient_to_pot":
            score += _positional_preference_bonus(
                opt.target_pos,
                state,
                self.protocol.pot_preference,
                agent_id=1,
            )
        if opt.kind == "fetch_ingredient" and not _fetch_has_current_pot_sink(
            self.option_library,
            opt,
            state,
        ):
            score -= 10.0
        if opt.kind == "drop_item_to_counter" and _carrying_unusable_item(
            self.option_library,
            state,
            agent_id=1,
        ):
            score += 100.0
        if opt.kind == "clear_interaction_cell" and _blocking_critical_cell(
            self.option_library,
            state,
            agent_id=1,
        ):
            score += 100.0
        if opt.kind in {"cross_bottleneck", "wait_at_bottleneck"}:
            score += _bottleneck_bonus(
                opt,
                self.protocol.bottleneck_policy,
                self.elapsed,
                self.bottleneck_alternate_phase,
            )
        score += _counter_preference_bonus(opt, self.protocol.counter_preference)
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
            if self.option_runtime is None:
                return False
            return self.option_runtime.wait_elapsed_after_arrival >= int(wait_duration)
        if opt.kind == "cross_bottleneck":
            if self.option_runtime is None or self.last_state is None:
                return False
            return cross_bottleneck_terminated(
                self.option_runtime,
                self.last_state,
                state,
                opt,
                agent_id,
            )
        return False

    def _update_option_runtime(self, state: Any, agent_id: int) -> None:
        if self.current_option is None or self.option_runtime is None:
            return
        opt = self.option_library.options[self.current_option]
        if opt.kind == "wait_at_bottleneck" and get_agent_pos(state, agent_id) in _region_cells(opt):
            self.option_runtime.reached_region = True
            if self.last_primitive_action == int(Actions.stay):
                self.option_runtime.wait_elapsed_after_arrival += 1
        elif opt.kind == "cross_bottleneck" and self.last_state is not None:
            cross_bottleneck_terminated(
                self.option_runtime,
                self.last_state,
                state,
                opt,
                agent_id,
            )


def make_training_partners(
    option_library: Any,
    partner_set: str = "standard7",
) -> list[ScriptedProtocolPartner]:
    try:
        protocols = PARTNER_REGISTRIES[str(partner_set)]
    except KeyError as exc:
        choices = ", ".join(sorted(PARTNER_REGISTRIES))
        raise ValueError(
            f"unknown partner_set {partner_set!r}; expected one of {choices}"
        ) from exc
    return [
        ScriptedProtocolPartner(
            name=name,
            option_library=option_library,
            protocol=protocol,
            partner_id=partner_id,
        )
        for partner_id, (name, protocol) in enumerate(protocols)
    ]



def _noop_option_id(option_library: Any) -> int:
    for opt in option_library.options:
        if str(opt.kind) == "noop":
            return int(opt.id)
    return 0



def _fetch_has_current_pot_sink(option_library: Any, opt: OptionSpec, state: Any) -> bool:
    ingredient_obj = option_library._ingredient_object_for_pile(opt)
    if ingredient_obj is None:
        return False
    for entity_id in option_library._entity_ids_by_kind("pot"):
        pot_pos = option_library.layout_graph.entities[entity_id].pos
        if pot_accepts_inventory_ingredient(
            state,
            pot_pos,
            ingredient_obj,
            require_recipe_useful=True,
        ):
            return True
    return False


def _carrying_unusable_item(option_library: Any, state: Any, agent_id: int) -> bool:
    inv = get_inventory(state, agent_id)
    if is_empty_inventory(inv):
        return False
    if not is_ingredient(inv):
        return False
    for entity_id in option_library._entity_ids_by_kind("pot"):
        pot_pos = option_library.layout_graph.entities[entity_id].pos
        if pot_accepts_inventory_ingredient(
            state,
            pot_pos,
            inv,
            require_recipe_useful=True,
        ):
            return False
    return True


def _blocking_critical_cell(option_library: Any, state: Any, agent_id: int) -> bool:
    return bool(option_library._is_blocking_critical_interaction_cell(state, agent_id))


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
    """Lexicographic role-identity tier (magnitude 10^3)."""
    return 4000.0 if condition else -1000.0


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
    alternate_phase: int = 0,
) -> float:
    if policy == "yield":
        # W3: bottleneck policy must be behaviorally visible within the support/
        # terminal tier, but still below the lexicographic role/terminal tier.
        return 1500.0 if opt.kind == "wait_at_bottleneck" else -500.0
    if policy == "push":
        return 1500.0 if opt.kind == "cross_bottleneck" else -500.0
    if policy == "alternate":
        # W4: alternate must alternate across option choices, not within-option
        # elapsed ticks that reset whenever a new option is selected.
        del elapsed
        prefer_cross = int(alternate_phase) % 2 == 0
        if prefer_cross and opt.kind == "cross_bottleneck":
            return 1500.0
        if not prefer_cross and opt.kind == "wait_at_bottleneck":
            return 1500.0
        if opt.kind in {"cross_bottleneck", "wait_at_bottleneck"}:
            return -500.0
    return 0.0


def _counter_preference_bonus(opt: OptionSpec, preference: str | None) -> float:
    if preference == "handoff" and opt.kind in {"handoff_counter", "drop_item_to_counter"}:
        return 3.0
    if preference == "clear" and opt.kind == "clear_interaction_cell":
        return 3.0
    if preference == "drop" and opt.kind == "drop_item_to_counter":
        return 3.0
    return 0.0


def _terminal_policy_bonus(opt: OptionSpec, policy: str | None) -> float:
    """Lexicographic role-identity tier (magnitude 10^3 - 10^4)."""
    if policy == "yield":
        if opt.kind in TERMINAL_KINDS:
            return -30000.0
        if opt.kind in PREP_KINDS or opt.kind in SUPPORT_KINDS:
            return 5000.0
        return 0.0
    if policy == "claim":
        if opt.kind in TERMINAL_KINDS:
            return 8000.0
        if opt.kind in PREP_KINDS:
            return 1000.0
        if opt.kind == "wait_at_bottleneck":
            return -2000.0
        return 0.0
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
    target = opt.target_pos
    # W5: penalize true target/corridor contention, not the nearly-always-true
    # condition that the partner is simply not on the ego's cell.
    if partner_pos == target:
        return 1.0
    if _manhattan(ego_pos, target) <= 1 and _manhattan(partner_pos, target) <= 1:
        return 0.5
    return 0.0


def _region_cells(opt: OptionSpec) -> tuple[GridPos, ...]:
    return tuple((opt.metadata or {}).get("region_cells", ()))


def _manhattan(a: GridPos, b: GridPos) -> int:
    return int(abs(a[0] - b[0]) + abs(a[1] - b[1]))
