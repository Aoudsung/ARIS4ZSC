from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from jaxmarl.environments.overcooked_v2.common import Actions, StaticObject

from .state_utils import (
    agent_facing_pos,
    cell_is_delivery,
    get_agent_pos,
    get_dynamic_objects_grid,
    get_inventory,
    is_empty_inventory,
    is_plain_plate,
    is_pot_full,
    is_pot_ready,
    is_plated_cooked_soup,
)

# Actor-specific sparse-credit helpers live in a dependency-light module so they
# can be unit-tested without the JaxMARL env stack; re-exported here for the
# existing `from .event_extractor import actor_sparse_reward` call sites.
from .sparse_credit import (  # noqa: F401
    DEFAULT_SPARSE_CREDIT_MODE,
    SPARSE_CREDIT_MODES,
    actor_sparse_reward,
    sparse_credit_params,
)

GridPos = tuple[int, int]
EVENT_SEMANTICS_VERSION = 2

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
    ego_delivery_event: bool
    partner_delivery_event: bool
    ego_correct_delivery: bool
    partner_correct_delivery: bool
    ego_wrong_delivery_event: bool
    partner_wrong_delivery_event: bool
    # Actor-LOCAL correct delivery: the ego delivered correctly AND was the sole
    # deliverer this step. correct_delivery below is the env's SHARED bool, so on a
    # coincident ego-wrong + partner-correct step `ego_correct_delivery` is spuriously
    # True; this field is the leak-proof signal the free-rider guard/metrics use.
    ego_sole_correct_delivery: bool
    pot_changed: bool
    object_pickup_or_drop: bool
    recipe_indicator_event: bool
    button_pressed: bool
    changed_cells: tuple[GridPos, ...]
    changed_object_bits: tuple[int, ...]
    changed_object_before: tuple[int, ...]
    changed_object_after: tuple[int, ...]
    pot_changed_cells: tuple[GridPos, ...]
    pot_became_full: bool
    pot_became_cooked: bool
    pot_became_ready: bool
    plate_picked: bool
    soup_picked: bool
    correct_delivery: bool


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
    (
        changed_cells,
        changed_bits,
        changed_before,
        changed_after,
    ) = _changed_dynamic_cells(prev_dynamic, next_dynamic)
    pot_changed_cells = _pot_changed_cells(prev_state, changed_cells)

    ego_interacted = ego_action == int(Actions.interact)
    partner_interacted = partner_action == int(Actions.interact)
    ego_waited = ego_action == int(Actions.stay)
    partner_waited = partner_action == int(Actions.stay)

    collision_or_block = (
        _blocked_move(ego_action, ego_pos_before, ego_pos_after)
        or _blocked_move(partner_action, partner_pos_before, partner_pos_after)
    )
    ego_delivery_event = _canonical_agent_delivery(
        prev_state,
        agent_id=0,
        inventory_before=ego_inventory_before,
        inventory_after=ego_inventory_after,
        interacted=ego_interacted,
    )
    partner_delivery_event = _canonical_agent_delivery(
        prev_state,
        agent_id=1,
        inventory_before=partner_inventory_before,
        inventory_after=partner_inventory_after,
        interacted=partner_interacted,
    )
    correct_delivery = bool(np.asarray(next_state.new_correct_delivery).item())
    delivery_event = bool(ego_delivery_event or partner_delivery_event or correct_delivery)
    wrong_delivery_event = bool(delivery_event and not correct_delivery)
    ego_correct_delivery = bool(ego_delivery_event and correct_delivery)
    partner_correct_delivery = bool(partner_delivery_event and correct_delivery)
    ego_wrong_delivery_event = bool(ego_delivery_event and not correct_delivery)
    partner_wrong_delivery_event = bool(partner_delivery_event and not correct_delivery)
    # Leak-proof actor-local correct delivery: ego delivered correctly AND no partner
    # delivery this step (so the shared correct_delivery bool is unambiguously the ego's).
    ego_sole_correct_delivery = bool(
        ego_delivery_event and not partner_delivery_event and correct_delivery
    )
    object_pickup_or_drop = (
        ego_inventory_before != ego_inventory_after
        or partner_inventory_before != partner_inventory_after
        or bool(changed_cells)
    )
    pot_changed = bool(pot_changed_cells)
    pot_became_full = _pot_became_full(
        pot_changed_cells,
        changed_cells,
        changed_before,
        changed_after,
    )
    pot_became_cooked = _pot_became_cooked(
        pot_changed_cells,
        changed_cells,
        changed_before,
        changed_after,
    )
    pot_became_ready = _pot_became_ready(
        pot_changed_cells,
        changed_cells,
        changed_before,
        changed_after,
    )
    plate_picked = _plate_picked(
        ego_inventory_before,
        ego_inventory_after,
        partner_inventory_before,
        partner_inventory_after,
    )
    soup_picked = _soup_picked(
        ego_inventory_before,
        ego_inventory_after,
        partner_inventory_before,
        partner_inventory_after,
    )
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
        wrong_delivery_event=wrong_delivery_event,
        ego_delivery_event=ego_delivery_event,
        partner_delivery_event=partner_delivery_event,
        ego_correct_delivery=ego_correct_delivery,
        partner_correct_delivery=partner_correct_delivery,
        ego_wrong_delivery_event=ego_wrong_delivery_event,
        partner_wrong_delivery_event=partner_wrong_delivery_event,
        ego_sole_correct_delivery=ego_sole_correct_delivery,
        pot_changed=pot_changed,
        object_pickup_or_drop=object_pickup_or_drop,
        recipe_indicator_event=recipe_indicator_event,
        button_pressed=button_pressed,
        changed_cells=changed_cells,
        changed_object_bits=changed_bits,
        changed_object_before=changed_before,
        changed_object_after=changed_after,
        pot_changed_cells=pot_changed_cells,
        pot_became_full=pot_became_full,
        pot_became_cooked=pot_became_cooked,
        pot_became_ready=pot_became_ready,
        plate_picked=plate_picked,
        soup_picked=soup_picked,
        correct_delivery=correct_delivery,
    )


def _blocked_move(action: int, before: GridPos, after: GridPos) -> bool:
    return action in _MOVE_ACTIONS and before == after


def _delivered_soup(inv_before: int, inv_after: int, interacted: bool) -> bool:
    return interacted and is_plated_cooked_soup(inv_before) and is_empty_inventory(inv_after)


def _canonical_agent_delivery(
    state: Any,
    *,
    agent_id: int,
    inventory_before: int,
    inventory_after: int,
    interacted: bool,
) -> bool:
    return bool(
        _delivered_soup(inventory_before, inventory_after, interacted)
        and _delivery_target(state, agent_id)
    )


def _delivery_target(state: Any, agent_id: int) -> bool:
    return cell_is_delivery(state, agent_facing_pos(state, agent_id))


def _changed_dynamic_cells(
    prev_dynamic: np.ndarray,
    next_dynamic: np.ndarray,
) -> tuple[tuple[GridPos, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    changed = np.asarray(prev_dynamic) != np.asarray(next_dynamic)
    ys, xs = np.nonzero(changed)
    cells: list[GridPos] = []
    bits: list[int] = []
    before_values: list[int] = []
    after_values: list[int] = []
    for y, x in zip(ys, xs, strict=True):
        cells.append((int(x), int(y)))
        before = int(np.asarray(prev_dynamic[y, x]).item())
        after = int(np.asarray(next_dynamic[y, x]).item())
        bits.append(before ^ after)
        before_values.append(before)
        after_values.append(after)
    return tuple(cells), tuple(bits), tuple(before_values), tuple(after_values)


def _pot_changed_cells(state: Any, changed_cells: tuple[GridPos, ...]) -> tuple[GridPos, ...]:
    static = np.asarray(getattr(state, "grid", np.zeros((0, 0, 2), dtype=np.int32))[..., 0])
    cells: list[GridPos] = []
    for x, y in changed_cells:
        if 0 <= y < static.shape[0] and 0 <= x < static.shape[1]:
            if int(static[y, x]) == int(StaticObject.POT):
                cells.append((x, y))
    return tuple(cells)


def _changed_pot_values(
    pot_changed_cells: tuple[GridPos, ...],
    changed_cells: tuple[GridPos, ...],
    before_values: tuple[int, ...],
    after_values: tuple[int, ...],
) -> list[tuple[int, int]]:
    by_cell = {
        cell: (int(before_values[idx]), int(after_values[idx]))
        for idx, cell in enumerate(changed_cells)
    }
    return [by_cell[cell] for cell in pot_changed_cells if cell in by_cell]


def _pot_became_full(
    pot_changed_cells: tuple[GridPos, ...],
    changed_cells: tuple[GridPos, ...],
    before_values: tuple[int, ...],
    after_values: tuple[int, ...],
) -> bool:
    for before, after in _changed_pot_values(
        pot_changed_cells,
        changed_cells,
        before_values,
        after_values,
    ):
        if not is_pot_full(before) and is_pot_full(after):
            return True
    return False


def _pot_became_cooked(
    pot_changed_cells: tuple[GridPos, ...],
    changed_cells: tuple[GridPos, ...],
    before_values: tuple[int, ...],
    after_values: tuple[int, ...],
) -> bool:
    for before, after in _changed_pot_values(
        pot_changed_cells,
        changed_cells,
        before_values,
        after_values,
    ):
        if not is_pot_ready(before) and is_pot_ready(after):
            return True
    return False


def _pot_became_ready(
    pot_changed_cells: tuple[GridPos, ...],
    changed_cells: tuple[GridPos, ...],
    before_values: tuple[int, ...],
    after_values: tuple[int, ...],
) -> bool:
    return _pot_became_cooked(
        pot_changed_cells,
        changed_cells,
        before_values,
        after_values,
    )


def _plate_picked(*inventories: int) -> bool:
    before_after_pairs = ((inventories[0], inventories[1]), (inventories[2], inventories[3]))
    return any(
        not is_plain_plate(before) and is_plain_plate(after)
        for before, after in before_after_pairs
    )


def _soup_picked(*inventories: int) -> bool:
    before_after_pairs = ((inventories[0], inventories[1]), (inventories[2], inventories[3]))
    return any(
        not is_plated_cooked_soup(before) and is_plated_cooked_soup(after)
        for before, after in before_after_pairs
    )


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
