from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from jaxmarl.environments.overcooked_v2.common import (
    Actions,
    DIR_TO_VEC,
    DynamicObject,
    MAX_INGREDIENTS,
    StaticObject,
)

from experiments.overcooked_v2.event_extractor import (
    _plate_picked,
    _pot_became_full,
    _soup_picked,
    extract_event,
)


def _ingredients(count: int) -> int:
    return sum(int(DynamicObject.ingredient(idx)) for idx in range(int(count)))


def _plated_cooked_soup() -> int:
    return _ingredients(int(MAX_INGREDIENTS)) | int(DynamicObject.COOKED) | int(
        DynamicObject.PLATE
    )


def _direction_for(delta: tuple[int, int]) -> int:
    items = DIR_TO_VEC.items() if hasattr(DIR_TO_VEC, "items") else enumerate(DIR_TO_VEC)
    for idx, vec in items:
        dx, dy = np.asarray(vec, dtype=int)
        if (int(dx), int(dy)) == delta:
            return int(idx)
    raise AssertionError(f"No OvercookedV2 direction for {delta}")


def _grid(target_pos: tuple[int, int], static_obj: int) -> np.ndarray:
    grid = np.zeros((4, 4, 3), dtype=np.int32)
    x, y = target_pos
    grid[y, x, 0] = int(static_obj)
    return grid


def _state(
    *,
    agent0_pos: tuple[int, int] = (1, 1),
    agent1_pos: tuple[int, int] = (3, 3),
    inventory0: int = 0,
    inventory1: int = 0,
    grid: np.ndarray | None = None,
    dir0: int | None = None,
    dir1: int | None = None,
    new_correct_delivery: bool = False,
) -> SimpleNamespace:
    direction = _direction_for((1, 0))
    agents = SimpleNamespace(
        pos=SimpleNamespace(
            x=np.asarray([agent0_pos[0], agent1_pos[0]]),
            y=np.asarray([agent0_pos[1], agent1_pos[1]]),
        ),
        dir=np.asarray([
            direction if dir0 is None else int(dir0),
            direction if dir1 is None else int(dir1),
        ]),
        inventory=np.asarray([int(inventory0), int(inventory1)]),
    )
    return SimpleNamespace(
        agents=agents,
        grid=np.zeros((4, 4, 3), dtype=np.int32) if grid is None else grid,
        recipe=np.asarray(_ingredients(int(MAX_INGREDIENTS))),
        new_correct_delivery=np.asarray(new_correct_delivery),
    )


def test_pot_became_full_only_when_reaching_max_ingredients() -> None:
    pot_cell = (1, 1)

    assert not _pot_became_full(
        (pot_cell,),
        (pot_cell,),
        (0,),
        (_ingredients(1),),
    )
    assert not _pot_became_full(
        (pot_cell,),
        (pot_cell,),
        (_ingredients(1),),
        (_ingredients(2),),
    )
    assert _pot_became_full(
        (pot_cell,),
        (pot_cell,),
        (_ingredients(int(MAX_INGREDIENTS) - 1),),
        (_ingredients(int(MAX_INGREDIENTS)),),
    )


def test_delivery_requires_facing_goal_cell() -> None:
    agent_pos = (1, 1)
    target_pos = (2, 1)
    soup = _plated_cooked_soup()
    direction = _direction_for((1, 0))

    counter_prev = _state(
        agent0_pos=agent_pos,
        inventory0=soup,
        grid=_grid(target_pos, StaticObject.WALL),
        dir0=direction,
    )
    counter_next = _state(
        agent0_pos=agent_pos,
        inventory0=0,
        grid=_grid(target_pos, StaticObject.WALL),
        dir0=direction,
    )

    counter_event = extract_event(
        counter_prev,
        int(Actions.interact),
        int(Actions.stay),
        counter_next,
        {},
        partner_option=None,
        partner_option_dist=None,
    )

    goal_prev = _state(
        agent0_pos=agent_pos,
        inventory0=soup,
        grid=_grid(target_pos, StaticObject.GOAL),
        dir0=direction,
    )
    goal_next = _state(
        agent0_pos=agent_pos,
        inventory0=0,
        grid=_grid(target_pos, StaticObject.GOAL),
        dir0=direction,
        new_correct_delivery=True,
    )

    goal_event = extract_event(
        goal_prev,
        int(Actions.interact),
        int(Actions.stay),
        goal_next,
        {},
        partner_option=None,
        partner_option_dist=None,
    )

    assert counter_event.delivery_event is False
    assert counter_event.correct_delivery is False
    assert counter_event.wrong_delivery_event is False
    assert goal_event.delivery_event is True
    assert goal_event.correct_delivery is True
    assert goal_event.wrong_delivery_event is False


def test_wrong_recipe_goal_drop_is_wrong_delivery() -> None:
    agent_pos = (1, 1)
    target_pos = (2, 1)
    soup = _plated_cooked_soup()
    direction = _direction_for((1, 0))

    prev_state = _state(
        agent0_pos=agent_pos,
        inventory0=soup,
        grid=_grid(target_pos, StaticObject.GOAL),
        dir0=direction,
    )
    next_state = _state(
        agent0_pos=agent_pos,
        inventory0=0,
        grid=_grid(target_pos, StaticObject.GOAL),
        dir0=direction,
        new_correct_delivery=False,
    )

    event = extract_event(
        prev_state,
        int(Actions.interact),
        int(Actions.stay),
        next_state,
        {},
        partner_option=None,
        partner_option_dist=None,
    )

    assert event.delivery_event is True
    assert event.correct_delivery is False
    assert event.wrong_delivery_event is True


def test_counter_drop_is_not_delivery() -> None:
    agent_pos = (1, 1)
    target_pos = (2, 1)
    soup = _plated_cooked_soup()
    direction = _direction_for((1, 0))

    prev_state = _state(
        agent0_pos=agent_pos,
        inventory0=soup,
        grid=_grid(target_pos, StaticObject.WALL),
        dir0=direction,
    )
    next_state = _state(
        agent0_pos=agent_pos,
        inventory0=0,
        grid=_grid(target_pos, StaticObject.WALL),
        dir0=direction,
        new_correct_delivery=False,
    )

    event = extract_event(
        prev_state,
        int(Actions.interact),
        int(Actions.stay),
        next_state,
        {},
        partner_option=None,
        partner_option_dist=None,
    )

    assert event.delivery_event is False
    assert event.correct_delivery is False
    assert event.wrong_delivery_event is False


def test_plate_pickup_excludes_plated_soup_pickup() -> None:
    plate = int(DynamicObject.PLATE)
    soup = _plated_cooked_soup()

    assert _plate_picked(0, plate, 0, 0)
    assert not _plate_picked(plate, soup, 0, 0)
    assert not _plate_picked(0, soup, 0, 0)
    assert _soup_picked(plate, soup, 0, 0)
