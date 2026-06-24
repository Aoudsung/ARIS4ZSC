from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from jaxmarl.environments.overcooked_v2.common import Actions

from .state_utils import (
    get_agent_pos,
    get_dynamic_objects_grid,
    get_inventory,
    is_empty_inventory,
    is_plated_cooked_soup,
)

GridPos = tuple[int, int]

_MOVE_ACTIONS = {
    int(Actions.right),
    int(Actions.down),
    int(Actions.left),
    int(Actions.up),
}


@dataclass
class OCV2Event:
    ego_pos_before: GridPos
    ego_pos_after: GridPos
    partner_pos_before: GridPos
    partner_pos_after: GridPos
    ego_inventory_before: int
    ego_inventory_after: int
    partner_inventory_before: int
    partner_inventory_after: int
    ego_action: int
    partner_action: int
    partner_option: int | None
    partner_option_dist: np.ndarray | None
    partner_option_confidence: float
    ego_waited: bool
    partner_waited: bool
    ego_interacted: bool
    partner_interacted: bool
    collision_or_block: bool
    delivery_event: bool
    wrong_delivery_event: bool
    pot_changed: bool
    object_pickup_or_drop: bool
    recipe_indicator_event: bool
    button_pressed: bool
    changed_cells: tuple[GridPos, ...]
    changed_object_bits: tuple[int, ...]


def extract_event(
    prev_state: Any,
    ego_action: int,
    partner_action: int,
    next_state: Any,
    info: dict[str, Any],
    partner_option: int | None,
    partner_option_dist: np.ndarray | None,
) -> OCV2Event:
    ego_action = int(ego_action)
    partner_action = int(partner_action)

    ego_pos_before = get_agent_pos(prev_state, 0)
    ego_pos_after = get_agent_pos(next_state, 0)
    partner_pos_before = get_agent_pos(prev_state, 1)
    partner_pos_after = get_agent_pos(next_state, 1)

    ego_inventory_before = get_inventory(prev_state, 0)
    ego_inventory_after = get_inventory(next_state, 0)
    partner_inventory_before = get_inventory(prev_state, 1)
    partner_inventory_after = get_inventory(next_state, 1)

    prev_dynamic = get_dynamic_objects_grid(prev_state)
    next_dynamic = get_dynamic_objects_grid(next_state)
    changed_cells, changed_bits = _changed_dynamic_cells(prev_dynamic, next_dynamic)

    ego_interacted = ego_action == int(Actions.interact)
    partner_interacted = partner_action == int(Actions.interact)
    ego_waited = ego_action == int(Actions.stay)
    partner_waited = partner_action == int(Actions.stay)

    collision_or_block = (
        _blocked_move(ego_action, ego_pos_before, ego_pos_after)
        or _blocked_move(partner_action, partner_pos_before, partner_pos_after)
    )
    delivery_event = _delivered_soup(
        ego_inventory_before,
        ego_inventory_after,
        ego_interacted,
    ) or _delivered_soup(
        partner_inventory_before,
        partner_inventory_after,
        partner_interacted,
    )
    object_pickup_or_drop = (
        ego_inventory_before != ego_inventory_after
        or partner_inventory_before != partner_inventory_after
        or bool(changed_cells)
    )
    pot_changed = bool(changed_cells)
    recipe_indicator_event = _positive_shaped_reward(info) and not object_pickup_or_drop
    button_pressed = (ego_interacted or partner_interacted) and recipe_indicator_event

    return OCV2Event(
        ego_pos_before=ego_pos_before,
        ego_pos_after=ego_pos_after,
        partner_pos_before=partner_pos_before,
        partner_pos_after=partner_pos_after,
        ego_inventory_before=ego_inventory_before,
        ego_inventory_after=ego_inventory_after,
        partner_inventory_before=partner_inventory_before,
        partner_inventory_after=partner_inventory_after,
        ego_action=ego_action,
        partner_action=partner_action,
        partner_option=partner_option,
        partner_option_dist=(
            None if partner_option_dist is None else np.asarray(partner_option_dist)
        ),
        partner_option_confidence=_partner_option_confidence(
            partner_option,
            partner_option_dist,
        ),
        ego_waited=ego_waited,
        partner_waited=partner_waited,
        ego_interacted=ego_interacted,
        partner_interacted=partner_interacted,
        collision_or_block=collision_or_block,
        delivery_event=delivery_event,
        wrong_delivery_event=False,
        pot_changed=pot_changed,
        object_pickup_or_drop=object_pickup_or_drop,
        recipe_indicator_event=recipe_indicator_event,
        button_pressed=button_pressed,
        changed_cells=changed_cells,
        changed_object_bits=changed_bits,
    )


def _blocked_move(action: int, before: GridPos, after: GridPos) -> bool:
    return action in _MOVE_ACTIONS and before == after


def _delivered_soup(inv_before: int, inv_after: int, interacted: bool) -> bool:
    return interacted and is_plated_cooked_soup(inv_before) and is_empty_inventory(inv_after)


def _changed_dynamic_cells(
    prev_dynamic: np.ndarray,
    next_dynamic: np.ndarray,
) -> tuple[tuple[GridPos, ...], tuple[int, ...]]:
    changed = np.asarray(prev_dynamic) != np.asarray(next_dynamic)
    ys, xs = np.nonzero(changed)
    cells: list[GridPos] = []
    bits: list[int] = []
    for y, x in zip(ys, xs, strict=True):
        cells.append((int(x), int(y)))
        before = int(np.asarray(prev_dynamic[y, x]).item())
        after = int(np.asarray(next_dynamic[y, x]).item())
        bits.append(before ^ after)
    return tuple(cells), tuple(bits)


def _partner_option_confidence(
    partner_option: int | None,
    partner_option_dist: np.ndarray | None,
) -> float:
    if partner_option_dist is not None:
        dist = np.asarray(partner_option_dist, dtype=float)
        if dist.size:
            return float(np.max(dist))
    if partner_option is not None:
        return 1.0
    return 0.0


def _positive_shaped_reward(info: dict[str, Any]) -> bool:
    shaped = info.get("shaped_reward", {})
    if not isinstance(shaped, dict):
        return _as_float(shaped) > 0.0
    return any(_as_float(value) > 0.0 for value in shaped.values())


def _as_float(value: Any) -> float:
    return float(np.asarray(value).item())
