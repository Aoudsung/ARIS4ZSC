from __future__ import annotations

from typing import Any

import numpy as np

from jaxmarl.environments.overcooked_v2.common import Actions
from src.aris_bellman.specs import OptionSpec

from .layout_parser import GridPos, LayoutGraph, adjacent_passable_cells
from .option_termination import OptionRuntime, option_terminated
from .state_utils import (
    agent_facing_pos,
    get_agent_pos,
    get_pot_contents,
    get_inventory,
    has_plate,
    is_empty_inventory,
    is_ingredient,
    is_pot_ready,
    is_pot_usable_for_ingredient,
    is_plated_cooked_soup,
)

_OBJECT_INTERACTION_KINDS = {
    "fetch_ingredient",
    "deliver_ingredient_to_pot",
    "pick_plate",
    "plate_soup",
    "serve_soup",
    "press_recipe_button",
    "handoff_counter",
}

_ACTION_BY_DELTA = {
    (1, 0): int(Actions.right),
    (0, 1): int(Actions.down),
    (-1, 0): int(Actions.left),
    (0, -1): int(Actions.up),
}

class OCV2OptionLibrary:
    def __init__(self, layout_graph: LayoutGraph, max_option_steps: int = 20):
        self.layout_graph = layout_graph
        self.max_option_steps = max_option_steps
        self.options = self._build_options()

    @property
    def num_options(self) -> int:
        return len(self.options)

    def valid_options(self, state: Any, agent_id: int) -> np.ndarray:
        inventory = get_inventory(state, agent_id)
        valid = np.zeros((self.num_options,), dtype=bool)

        for opt in self.options:
            candidate = False
            if opt.kind in {"noop", "cross_bottleneck", "wait_at_bottleneck"}:
                candidate = True
            elif is_empty_inventory(inventory):
                candidate = opt.kind in {
                    "fetch_ingredient",
                    "pick_plate",
                    "press_recipe_button",
                }
            elif is_plated_cooked_soup(inventory):
                candidate = opt.kind == "serve_soup"
            elif is_ingredient(inventory):
                candidate = opt.kind == "deliver_ingredient_to_pot"
            elif has_plate(inventory):
                candidate = opt.kind == "plate_soup"

            valid[opt.id] = bool(
                candidate
                and np.isfinite(self.expected_cost(state, agent_id, opt.id))
                and self._task_precondition(state, agent_id, opt)
            )

        return valid

    def is_valid_for_state(self, state: Any, agent_id: int, option_id: int) -> bool:
        return bool(self.valid_options(state, agent_id)[option_id])

    def primitive_action(self, state: Any, agent_id: int, option_id: int) -> int:
        opt = self.options[option_id]
        if opt.kind == "noop":
            return int(Actions.stay)

        interaction_action = self._interaction_action(state, agent_id, opt)
        if interaction_action is not None:
            return interaction_action

        current = get_agent_pos(state, agent_id)
        targets = self._target_cells(opt)
        if opt.kind == "wait_at_bottleneck" and current in targets:
            return int(Actions.stay)

        target = self._closest_target_cell(state, agent_id, targets)
        if target is None or current == target:
            return int(Actions.stay)

        next_cell = self._next_cell_toward(current, target)
        if next_cell is None:
            return int(Actions.stay)

        return _action_from_to(current, next_cell)

    def expected_cost(self, state: Any, agent_id: int, option_id: int) -> float:
        opt = self.options[option_id]
        if opt.kind == "noop":
            return 0.0
        if opt.kind == "wait_at_bottleneck":
            return float((opt.metadata or {}).get("wait_duration", 2))

        current = get_agent_pos(state, agent_id)
        targets = self._target_cells(opt)
        if not targets:
            return float("inf")

        min_dist = min(
            (
                self.layout_graph.shortest_path_dist.get((current, target), float("inf"))
                for target in targets
            ),
            default=float("inf"),
        )
        interaction_steps = 1 if opt.kind in _OBJECT_INTERACTION_KINDS else 0
        return float(min_dist + interaction_steps)

    def option_terminated(
        self,
        option: OptionSpec,
        prev_state: Any,
        next_state: Any,
        event: Any,
        agent_id: int,
        elapsed: int,
        runtime: OptionRuntime | None = None,
    ) -> tuple[bool, str]:
        return option_terminated(
            option,
            prev_state,
            next_state,
            event,
            agent_id,
            elapsed,
            runtime,
        )

    def _build_options(self) -> list[OptionSpec]:
        options: list[OptionSpec] = []

        for entity_id in self._entity_ids_with_prefix("ingredient_pile"):
            self._add_entity_option(options, "fetch_ingredient", entity_id)

        for entity_id in self._entity_ids_by_kind("pot"):
            self._add_entity_option(options, "deliver_ingredient_to_pot", entity_id)
            self._add_entity_option(options, "plate_soup", entity_id)

        for entity_id in self._entity_ids_by_kind("plate_pile"):
            self._add_entity_option(options, "pick_plate", entity_id)

        for entity_id in self._entity_ids_by_kind("delivery"):
            self._add_entity_option(options, "serve_soup", entity_id)

        for entity_id in self._entity_ids_by_kind("button_recipe_indicator"):
            self._add_entity_option(options, "press_recipe_button", entity_id)

        for idx, bottleneck in enumerate(self.layout_graph.bottlenecks):
            region_id = f"bottleneck:{idx}"
            self._add_bottleneck_option(options, "cross_bottleneck", region_id, bottleneck)
            self._add_bottleneck_option(options, "wait_at_bottleneck", region_id, bottleneck)

        options.append(
            OptionSpec(
                id=len(options),
                name="noop",
                kind="noop",
                target_id=None,
                target_pos=None,
                entity_ids=(),
                region_ids=(),
                max_steps=1,
                terminal_event="noop",
                metadata={},
            )
        )
        return options

    def _add_entity_option(
        self,
        options: list[OptionSpec],
        kind: str,
        entity_id: str,
    ) -> None:
        entity = self.layout_graph.entities[entity_id]
        options.append(
            OptionSpec(
                id=len(options),
                name=f"{kind}:{entity_id}",
                kind=kind,
                target_id=entity_id,
                target_pos=entity.pos,
                entity_ids=(entity_id,),
                region_ids=(),
                max_steps=self.max_option_steps,
                terminal_event=_terminal_event_for_kind(kind),
                metadata={
                    "target_kind": entity.kind,
                    "interaction_cells": tuple(self.layout_graph.interaction_cells[entity_id]),
                },
            )
        )

    def _add_bottleneck_option(
        self,
        options: list[OptionSpec],
        kind: str,
        region_id: str,
        pos: GridPos,
    ) -> None:
        options.append(
            OptionSpec(
                id=len(options),
                name=f"{kind}:{region_id}",
                kind=kind,
                target_id=region_id,
                target_pos=pos,
                entity_ids=(),
                region_ids=(region_id,),
                max_steps=self.max_option_steps,
                terminal_event=_terminal_event_for_kind(kind),
                metadata={
                    "region_cells": (pos,),
                    "interaction_cells": (pos,),
                    "wait_duration": 2,
                },
            )
        )

    def _entity_ids_by_kind(self, kind: str) -> list[str]:
        return list(self.layout_graph.entities_by_kind.get(kind, ()))

    def _entity_ids_with_prefix(self, prefix: str) -> list[str]:
        return [
            entity_id
            for entity_id, entity in self.layout_graph.entities.items()
            if entity.kind.startswith(prefix)
        ]

    def _target_cells(self, opt: OptionSpec) -> tuple[GridPos, ...]:
        metadata = opt.metadata or {}
        if "interaction_cells" in metadata:
            return tuple(metadata["interaction_cells"])
        if "region_cells" in metadata:
            return tuple(metadata["region_cells"])
        if opt.target_pos is not None:
            return (opt.target_pos,)
        return ()

    def _interaction_action(
        self,
        state: Any,
        agent_id: int,
        opt: OptionSpec,
    ) -> int | None:
        if opt.kind not in _OBJECT_INTERACTION_KINDS or opt.target_pos is None:
            return None

        current = get_agent_pos(state, agent_id)
        dx = opt.target_pos[0] - current[0]
        dy = opt.target_pos[1] - current[1]
        if abs(dx) + abs(dy) != 1:
            return None

        if agent_facing_pos(state, agent_id) == opt.target_pos:
            return int(Actions.interact)
        return _ACTION_BY_DELTA[(dx, dy)]

    def _closest_target_cell(
        self,
        state: Any,
        agent_id: int,
        targets: tuple[GridPos, ...],
    ) -> GridPos | None:
        if not targets:
            return None

        current = get_agent_pos(state, agent_id)
        best_target: GridPos | None = None
        best_dist = float("inf")
        for target in targets:
            dist = self.layout_graph.shortest_path_dist.get(
                (current, target),
                float("inf"),
            )
            if dist < best_dist:
                best_target = target
                best_dist = dist
        if not np.isfinite(best_dist):
            return None
        return best_target

    def _next_cell_toward(self, current: GridPos, target: GridPos) -> GridPos | None:
        current_dist = self.layout_graph.shortest_path_dist.get(
            (current, target),
            float("inf"),
        )
        candidates = adjacent_passable_cells(current, self.layout_graph.passable)
        candidates.sort(
            key=lambda cell: self.layout_graph.shortest_path_dist.get(
                (cell, target),
                float("inf"),
            )
        )
        for candidate in candidates:
            candidate_dist = self.layout_graph.shortest_path_dist.get(
                (candidate, target),
                float("inf"),
            )
            if candidate_dist < current_dist:
                return candidate
        return None

    def _task_precondition(self, state: Any, agent_id: int, opt: OptionSpec) -> bool:
        if opt.kind in {"fetch_ingredient", "pick_plate"}:
            return opt.target_pos is not None and bool(self._target_cells(opt))
        if opt.kind in {"noop", "cross_bottleneck"}:
            return True
        if opt.kind == "wait_at_bottleneck":
            return bool(_region_cells(opt))
        if opt.kind == "deliver_ingredient_to_pot":
            return (
                opt.target_pos is not None
                and bool(self._target_cells(opt))
                and is_pot_usable_for_ingredient(get_pot_contents(state, opt.target_pos))
            )
        if opt.kind == "plate_soup":
            return (
                opt.target_pos is not None
                and bool(self._target_cells(opt))
                and is_pot_ready(get_pot_contents(state, opt.target_pos))
            )
        if opt.kind == "serve_soup":
            return opt.target_pos is not None and bool(self._target_cells(opt))
        if opt.kind == "press_recipe_button":
            target_kind = str((opt.metadata or {}).get("target_kind", ""))
            return (
                opt.target_pos is not None
                and bool(self._target_cells(opt))
                and target_kind == "button_recipe_indicator"
            )
        if opt.kind == "handoff_counter":
            return bool(_counter_cells(opt))
        return True


def _action_from_to(current: GridPos, target: GridPos) -> int:
    delta = (target[0] - current[0], target[1] - current[1])
    return _ACTION_BY_DELTA.get(delta, int(Actions.stay))


def _terminal_event_for_kind(kind: str) -> str | None:
    return {
        "fetch_ingredient": "picked_ingredient",
        "deliver_ingredient_to_pot": "ingredient_delivered_to_pot",
        "pick_plate": "picked_plate",
        "plate_soup": "plated_soup",
        "serve_soup": "served_soup",
        "press_recipe_button": "recipe_button_effect",
        "cross_bottleneck": "crossed_bottleneck",
        "wait_at_bottleneck": "wait_duration_after_arrival",
        "noop": "noop",
    }.get(kind)


def _region_cells(opt: OptionSpec) -> tuple[GridPos, ...]:
    return tuple((opt.metadata or {}).get("region_cells", ()))


def _counter_cells(opt: OptionSpec) -> tuple[GridPos, ...]:
    metadata = opt.metadata or {}
    if "counter_cells" in metadata:
        return tuple(metadata["counter_cells"])
    if opt.target_pos is not None:
        return (opt.target_pos,)
    return tuple(metadata.get("interaction_cells", ()))
