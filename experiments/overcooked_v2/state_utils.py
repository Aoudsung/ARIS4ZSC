from __future__ import annotations

from typing import Any

import numpy as np

from jaxmarl.environments.overcooked_v2.common import (
    DIR_TO_VEC,
    DynamicObject,
    MAX_INGREDIENTS,
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


def has_ingredient_bits(inv: int) -> bool:
    return (int(inv) >> 2) != 0


def is_ingredient(inv: int) -> bool:
    obj = int(inv)
    return has_ingredient_bits(obj) and not has_plate(obj)


def is_plated_cooked_soup(inv: int) -> bool:
    obj = int(inv)
    return has_plate(obj) and is_cooked(obj) and has_ingredient_bits(obj)


def is_pot_ready(contents: int) -> bool:
    return is_cooked(contents) and has_ingredient_bits(contents) and not has_plate(contents)


def is_pot_usable_for_ingredient(contents: int) -> bool:
    return not is_cooked(contents) and not has_plate(contents) and not is_pot_full(contents)


def ingredient_count_py(obj: int) -> int:
    """Python mirror of DynamicObject.ingredient_count for scalar predicates."""
    value = int(obj) >> 2
    count = 0
    while value > 0:
        count += value & 0x3
        value >>= 2
    return int(count)


def ingredient_count_of_idx(obj: int, ingredient_idx: int) -> int:
    return int((int(obj) >> (2 + 2 * int(ingredient_idx))) & 0x3)


def ingredient_idx_from_inventory(inv: int) -> int | None:
    if not is_ingredient(inv):
        return None
    value = int(inv) >> 2
    idx = 0
    while value > 0:
        if value & 0x3:
            return idx
        value >>= 2
        idx += 1
    return None


def is_pot_full(contents: int) -> bool:
    return ingredient_count_py(contents) >= int(MAX_INGREDIENTS)


def get_pot_contents(state: Any, pos: GridPos) -> int:
    x, y = pos
    return _to_int(state.grid[y, x, 1])


def get_cell_extra(state: Any, pos: GridPos) -> int:
    grid = np.asarray(state.grid)
    if grid.ndim < 3 or grid.shape[2] <= 2:
        return 0
    x, y = pos
    return _to_int(grid[y, x, 2])


def is_pot_cooking(state: Any, pos: GridPos) -> bool:
    return get_cell_extra(state, pos) > 0


def recipe_needs_inventory_ingredient(state: Any, pot_contents: int, inv: int) -> bool:
    if not is_ingredient(inv):
        return False
    recipe = _to_int(state.recipe)
    ingredient_selector = int(inv) | (int(inv) << 1)
    return (int(pot_contents) & ingredient_selector) < (recipe & ingredient_selector)


def pot_accepts_inventory_ingredient(
    state: Any,
    pot_pos: GridPos,
    inv: int,
    require_recipe_useful: bool = True,
) -> bool:
    contents = get_pot_contents(state, pot_pos)
    if not is_ingredient(inv):
        return False
    if is_pot_cooking(state, pot_pos):
        return False
    if is_cooked(contents):
        return False
    if has_plate(contents):
        return False
    if is_pot_full(contents):
        return False
    if require_recipe_useful and not recipe_needs_inventory_ingredient(state, contents, inv):
        return False
    return True


def any_pot_accepts_inventory_ingredient(
    state: Any,
    pot_positions: tuple[GridPos, ...],
    inv: int,
    require_recipe_useful: bool = True,
) -> bool:
    return any(
        pot_accepts_inventory_ingredient(
            state,
            pos,
            inv,
            require_recipe_useful=require_recipe_useful,
        )
        for pos in pot_positions
    )


def is_pot_ready_for_plate(
    state: Any,
    pot_pos: GridPos,
    require_correct_recipe: bool = True,
) -> bool:
    contents = get_pot_contents(state, pot_pos)
    if not is_pot_ready(contents):
        return False
    if not require_correct_recipe:
        return True
    expected_contents = _to_int(state.recipe) | int(DynamicObject.COOKED)
    return int(contents) == int(expected_contents)


def get_dynamic_objects_grid(state: Any) -> np.ndarray:
    return np.asarray(state.grid[..., 1])


def is_ingredient_pile(static_obj: int) -> bool:
    return int(static_obj) >= int(StaticObject.INGREDIENT_PILE_BASE)


def agent_facing_pos(state: Any, agent_id: int) -> GridPos:
    x, y = get_agent_pos(state, agent_id)
    direction = _to_int(state.agents.dir[agent_id])
    dx, dy = np.asarray(DIR_TO_VEC[direction], dtype=int)
    return (x + int(dx), y + int(dy))
