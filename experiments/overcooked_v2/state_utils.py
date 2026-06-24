from __future__ import annotations

from typing import Any

import numpy as np

from jaxmarl.environments.overcooked_v2.common import (
    DIR_TO_VEC,
    DynamicObject,
    StaticObject,
)

GridPos = tuple[int, int]


def _to_int(value: Any) -> int:
    return int(np.asarray(value).item())


def get_agent_pos(state: Any, agent_id: int) -> GridPos:
    x = _to_int(state.agents.pos.x[agent_id])
    y = _to_int(state.agents.pos.y[agent_id])
    return (x, y)


def get_inventory(state: Any, agent_id: int) -> int:
    return _to_int(state.agents.inventory[agent_id])


def is_empty_inventory(inv: int) -> bool:
    return int(inv) == int(DynamicObject.EMPTY)


def has_plate(inv: int) -> bool:
    return (int(inv) & int(DynamicObject.PLATE)) == int(DynamicObject.PLATE)


def is_cooked(inv: int) -> bool:
    return (int(inv) & int(DynamicObject.COOKED)) == int(DynamicObject.COOKED)


def is_ingredient(inv: int) -> bool:
    obj = int(inv)
    return (obj >> 2) != 0 and not has_plate(obj)


def is_plated_cooked_soup(inv: int) -> bool:
    obj = int(inv)
    return has_plate(obj) and is_cooked(obj) and (obj >> 2) != 0


def get_pot_contents(state: Any, pos: GridPos) -> int:
    x, y = pos
    return _to_int(state.grid[y, x, 1])


def get_dynamic_objects_grid(state: Any) -> np.ndarray:
    return np.asarray(state.grid[..., 1])


def is_ingredient_pile(static_obj: int) -> bool:
    return int(static_obj) >= int(StaticObject.INGREDIENT_PILE_BASE)


def agent_facing_pos(state: Any, agent_id: int) -> GridPos:
    x, y = get_agent_pos(state, agent_id)
    direction = _to_int(state.agents.dir[agent_id])
    dx, dy = np.asarray(DIR_TO_VEC[direction], dtype=int)
    return (x + int(dx), y + int(dy))
