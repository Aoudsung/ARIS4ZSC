from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from jaxmarl.environments.overcooked_v2.common import Actions
from src.aris_bellman.specs import OptionSpec

from .state_utils import (
    get_agent_pos,
    get_dynamic_objects_grid,
    get_inventory,
    has_plate,
    is_empty_inventory,
    is_ingredient,
    is_plated_cooked_soup,
)

GridPos = tuple[int, int]


OPTION_SUCCESS_REASONS = {
    "fetch_ingredient": {"picked_ingredient"},
    "deliver_ingredient_to_pot": {"ingredient_delivered_to_pot"},
    "pick_plate": {"picked_plate"},
    "plate_soup": {"plated_soup"},
    "serve_soup": {"served_soup"},
    "press_recipe_button": {"recipe_button_effect", "button_interacted"},
    "drop_item_to_counter": {"dropped_item_to_counter"},
    "clear_interaction_cell": {"cleared_interaction_cell"},
    "handoff_counter": {"handoff_or_counter_event"},
    "wait_at_bottleneck": {"partner_response_observed", "wait_duration_after_arrival"},
    "cross_bottleneck": {"crossed_bottleneck"},
    "noop": {"noop"},
}


def option_success(opt_kind: str, termination_reason: str) -> bool:
    return str(termination_reason) in OPTION_SUCCESS_REASONS.get(str(opt_kind), set())


@dataclass
class OptionRuntime:
    option_id: int
    start_pos: GridPos
    elapsed: int = 0
    reached_region: bool = False
    wait_elapsed_after_arrival: int = 0
    entry_side: GridPos | None = None


def option_terminated(
    opt: OptionSpec,
    prev_state: Any,
    next_state: Any,
    event: Any,
    agent_id: int,
    elapsed: int,
    runtime: OptionRuntime | None = None,
) -> tuple[bool, str]:
    if runtime is not None:
        runtime.elapsed = int(elapsed)

    if opt.kind == "noop":
        return True, "noop"

    inv_before = get_inventory(prev_state, agent_id)
    inv_after = get_inventory(next_state, agent_id)

    if opt.kind == "fetch_ingredient":
        if is_empty_inventory(inv_before) and is_ingredient(inv_after):
            return True, "picked_ingredient"

    if opt.kind == "deliver_ingredient_to_pot":
        if (
            is_ingredient(inv_before)
            and is_empty_inventory(inv_after)
            and pot_changed_near_target(prev_state, next_state, opt)
        ):
            return True, "ingredient_delivered_to_pot"

    if opt.kind == "pick_plate":
        if is_empty_inventory(inv_before) and has_plate(inv_after):
            return True, "picked_plate"

    if opt.kind == "plate_soup":
        if has_plate(inv_before) and is_plated_cooked_soup(inv_after):
            return True, "plated_soup"

    if opt.kind == "serve_soup":
        if (
            is_plated_cooked_soup(inv_before)
            and is_empty_inventory(inv_after)
            and event.delivery_event
        ):
            return True, "served_soup"

    if opt.kind == "press_recipe_button":
        if event.recipe_indicator_event:
            return True, "recipe_button_effect"
        if reached_interaction_target(next_state, agent_id, opt) and event.ego_interacted:
            return True, "button_interacted"

    if opt.kind == "cross_bottleneck":
        if cross_bottleneck_terminated(runtime, prev_state, next_state, opt, agent_id):
            return True, "crossed_bottleneck"

    if opt.kind == "wait_at_bottleneck":
        wait_duration = int((opt.metadata or {}).get("wait_duration", 2))
        reached = wait_bottleneck_update_runtime(
            runtime,
            prev_state,
            next_state,
            event,
            opt,
            agent_id,
        )
        if (
            reached
            and partner_response_observed_near_region(
                event,
                opt.region_ids,
                region_cells=_region_cells(opt),
            )
        ):
            return True, "partner_response_observed"
        if runtime is not None:
            if runtime.wait_elapsed_after_arrival >= wait_duration:
                return True, "wait_duration_after_arrival"
        elif _agent_in_region(next_state, agent_id, opt) and elapsed >= wait_duration:
            return True, "wait_duration_after_arrival"

    if opt.kind == "handoff_counter":
        if object_transfer_or_counter_event(prev_state, next_state, event, opt):
            return True, "handoff_or_counter_event"

    if opt.kind == "drop_item_to_counter":
        if (
            not is_empty_inventory(inv_before)
            and is_empty_inventory(inv_after)
            and counter_changed_near_target(prev_state, next_state, opt)
        ):
            return True, "dropped_item_to_counter"

    if opt.kind == "clear_interaction_cell":
        critical_cells = _critical_interaction_cells(opt)
        if (
            get_agent_pos(prev_state, agent_id) in critical_cells
            and get_agent_pos(next_state, agent_id) not in critical_cells
        ):
            return True, "cleared_interaction_cell"

    if elapsed >= opt.max_steps:
        return True, "max_steps"

    return False, "running"


def cross_bottleneck_terminated(
    runtime: OptionRuntime | None,
    prev_state: Any,
    next_state: Any,
    opt: OptionSpec,
    agent_id: int,
) -> bool:
    region = _region_cells(opt)
    if not region:
        return False

    prev_pos = get_agent_pos(prev_state, agent_id)
    next_pos = get_agent_pos(next_state, agent_id)
    if runtime is None:
        return crossed_region(
            prev_state,
            next_state,
            agent_id,
            opt.region_ids,
            region_cells=region,
        )

    if not runtime.reached_region:
        if prev_pos not in region and next_pos in region:
            runtime.reached_region = True
            runtime.entry_side = prev_pos
        elif prev_pos in region:
            runtime.reached_region = True
            if runtime.start_pos not in region:
                runtime.entry_side = runtime.start_pos

    if not runtime.reached_region or next_pos in region:
        return False
    if runtime.entry_side is None:
        return False
    return _crossed_to_other_side(runtime.entry_side, next_pos, region)


def wait_bottleneck_update_runtime(
    runtime: OptionRuntime | None,
    prev_state: Any,
    next_state: Any,
    event: Any,
    opt: OptionSpec,
    agent_id: int,
) -> bool:
    if runtime is None:
        return _agent_in_region(next_state, agent_id, opt)

    region = _region_cells(opt)
    prev_pos = get_agent_pos(prev_state, agent_id)
    next_pos = get_agent_pos(next_state, agent_id)
    was_in_region = prev_pos in region
    is_in_region = next_pos in region
    if is_in_region:
        runtime.reached_region = True
    if (
        runtime.reached_region
        and was_in_region
        and is_in_region
        and bool(getattr(event, "ego_waited", False))
    ):
        runtime.wait_elapsed_after_arrival += 1
    return runtime.reached_region


def pot_changed_near_target(
    prev_state: Any,
    next_state: Any,
    opt: OptionSpec,
) -> bool:
    if opt.target_pos is None:
        return False

    x, y = opt.target_pos
    prev_dynamic = get_dynamic_objects_grid(prev_state)
    next_dynamic = get_dynamic_objects_grid(next_state)
    return bool(prev_dynamic[y, x] != next_dynamic[y, x])


def counter_changed_near_target(
    prev_state: Any,
    next_state: Any,
    opt: OptionSpec,
) -> bool:
    if opt.target_pos is None:
        return False

    x, y = opt.target_pos
    prev_dynamic = get_dynamic_objects_grid(prev_state)
    next_dynamic = get_dynamic_objects_grid(next_state)
    return bool(prev_dynamic[y, x] != next_dynamic[y, x])


def reached_interaction_target(
    state: Any,
    agent_id: int,
    opt: OptionSpec,
) -> bool:
    agent_pos = get_agent_pos(state, agent_id)
    interaction_cells = _interaction_cells(opt)
    if interaction_cells:
        return agent_pos in interaction_cells
    if opt.target_pos is None:
        return False
    return _manhattan(agent_pos, opt.target_pos) == 1


def crossed_region(
    prev_state: Any,
    next_state: Any,
    agent_id: int,
    region_ids: Iterable[str],
    region_cells: Iterable[GridPos] | None = None,
) -> bool:
    if not tuple(region_ids):
        return False

    prev_pos = get_agent_pos(prev_state, agent_id)
    next_pos = get_agent_pos(next_state, agent_id)

    for cell in tuple(region_cells or ()):
        if _opposite_adjacent_sides(prev_pos, next_pos, cell):
            return True

    return False


def partner_response_observed_near_region(
    event: Any,
    region_ids: Iterable[str],
    region_cells: Iterable[GridPos] | None = None,
) -> bool:
    if not tuple(region_ids):
        return False
    if int(event.partner_action) == int(Actions.stay):
        return False

    partner_pos = tuple(event.partner_pos_after)
    return any(_manhattan(partner_pos, cell) <= 1 for cell in tuple(region_cells or ()))


def object_transfer_or_counter_event(
    prev_state: Any,
    next_state: Any,
    event: Any,
    opt: OptionSpec,
) -> bool:
    if event.object_pickup_or_drop:
        return True

    prev_dynamic = get_dynamic_objects_grid(prev_state)
    next_dynamic = get_dynamic_objects_grid(next_state)
    for cell in _counter_event_cells(opt):
        x, y = cell
        if prev_dynamic[y, x] != next_dynamic[y, x]:
            return True
    return False


def _agent_in_region(state: Any, agent_id: int, opt: OptionSpec) -> bool:
    return get_agent_pos(state, agent_id) in _region_cells(opt)


def _interaction_cells(opt: OptionSpec) -> tuple[GridPos, ...]:
    return tuple((opt.metadata or {}).get("interaction_cells", ()))


def _region_cells(opt: OptionSpec) -> tuple[GridPos, ...]:
    return tuple((opt.metadata or {}).get("region_cells", ()))


def _counter_event_cells(opt: OptionSpec) -> tuple[GridPos, ...]:
    metadata = opt.metadata or {}
    if "counter_cells" in metadata:
        return tuple(metadata["counter_cells"])
    if opt.target_pos is not None:
        return (opt.target_pos,)
    return tuple(metadata.get("interaction_cells", ()))


def _critical_interaction_cells(opt: OptionSpec) -> set[GridPos]:
    return {tuple(cell) for cell in (opt.metadata or {}).get("critical_interaction_cells", ())}


def _crossed_to_other_side(
    entry_side: GridPos,
    exit_pos: GridPos,
    region_cells: Iterable[GridPos],
) -> bool:
    return any(
        _opposite_adjacent_sides(entry_side, exit_pos, cell)
        for cell in tuple(region_cells)
    )


def _opposite_adjacent_sides(
    prev_pos: GridPos,
    next_pos: GridPos,
    region_cell: GridPos,
) -> bool:
    prev_delta = (prev_pos[0] - region_cell[0], prev_pos[1] - region_cell[1])
    next_delta = (next_pos[0] - region_cell[0], next_pos[1] - region_cell[1])
    return (
        abs(prev_delta[0]) + abs(prev_delta[1]) == 1
        and abs(next_delta[0]) + abs(next_delta[1]) == 1
        and prev_delta[0] == -next_delta[0]
        and prev_delta[1] == -next_delta[1]
    )


def _manhattan(a: GridPos, b: GridPos) -> int:
    return int(abs(a[0] - b[0]) + abs(a[1] - b[1]))
