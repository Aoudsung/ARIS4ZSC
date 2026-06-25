from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from jaxmarl.environments.overcooked_v2.common import Actions, DynamicObject, StaticObject

from src.aris_bellman.specs import OptionSpec

from experiments.overcooked_v2.layout_parser import (
    LayoutGraph,
    all_pairs_shortest_paths,
    parse_layout,
)
from experiments.overcooked_v2.option_termination import (
    OptionRuntime,
    option_success,
    option_terminated,
)
from experiments.overcooked_v2.options import OCV2OptionLibrary
from experiments.overcooked_v2.state_utils import (
    ingredient_count_py,
    is_pot_full,
    is_pot_usable_for_ingredient,
)


def _recipe(*ingredients: int) -> int:
    return sum(int(DynamicObject.ingredient(idx)) for idx in ingredients)


def _grid(cells: list[tuple[tuple[int, int], int, int, int]] | None = None) -> np.ndarray:
    grid = np.zeros((4, 4, 3), dtype=np.int32)
    for pos, static_obj, dynamic_obj, extra in cells or []:
        x, y = pos
        grid[y, x, 0] = int(static_obj)
        grid[y, x, 1] = int(dynamic_obj)
        grid[y, x, 2] = int(extra)
    return grid


def _state(
    agent0_pos: tuple[int, int],
    *,
    inventory0: int = 0,
    agent1_pos: tuple[int, int] = (3, 3),
    inventory1: int = 0,
    grid: np.ndarray | None = None,
    recipe: int | None = None,
) -> SimpleNamespace:
    agents = SimpleNamespace(
        pos=SimpleNamespace(
            x=np.asarray([agent0_pos[0], agent1_pos[0]]),
            y=np.asarray([agent0_pos[1], agent1_pos[1]]),
        ),
        dir=np.asarray([int(Actions.right), int(Actions.left)]),
        inventory=np.asarray([inventory0, inventory1]),
    )
    return SimpleNamespace(
        agents=agents,
        grid=_grid() if grid is None else grid,
        recipe=np.asarray(_recipe(0, 0, 0) if recipe is None else recipe),
    )


def _layout_graph() -> LayoutGraph:
    passable = np.ones((4, 4), dtype=bool)
    entities = {
        "counter:0:1": SimpleNamespace(id="counter:0:1", kind="counter", pos=(0, 1)),
        "pot:2:1": SimpleNamespace(id="pot:2:1", kind="pot", pos=(2, 1)),
    }
    passable[1, 0] = False
    passable[1, 2] = False
    entities_by_kind = {"counter": ["counter:0:1"], "pot": ["pot:2:1"]}
    interaction_cells = {
        "counter:0:1": [(0, 0), (0, 2), (1, 1)],
        "pot:2:1": [(1, 1), (3, 1), (2, 0), (2, 2)],
    }
    return LayoutGraph(
        layout_name="unit",
        width=4,
        height=4,
        passable=passable,
        entities=entities,
        entities_by_kind=entities_by_kind,
        interaction_cells=interaction_cells,
        bottlenecks=[],
        region_cells={},
        cell_to_entity={(0, 1): "counter:0:1", (2, 1): "pot:2:1"},
        shortest_path_dist=all_pairs_shortest_paths(passable),
    )


def _valid_kinds(option_lib: OCV2OptionLibrary, state: SimpleNamespace, agent_id: int = 0) -> set[str]:
    valid = option_lib.valid_options(state, agent_id)
    return {option_lib.options[idx].kind for idx in np.flatnonzero(valid)}


def _option_id(option_lib: OCV2OptionLibrary, kind: str) -> int:
    for opt in option_lib.options:
        if opt.kind == kind:
            return int(opt.id)
    raise AssertionError(f"missing option kind {kind}")


def test_full_pot_rejects_extra_ingredient():
    assert ingredient_count_py(0xC) == 3
    assert is_pot_full(0xC)
    assert not is_pot_usable_for_ingredient(0xC)


def test_deliver_ingredient_invalid_when_pot_full_cooking_cooked_or_recipe_satisfied():
    option_lib = OCV2OptionLibrary(_layout_graph(), max_option_steps=6)
    ingredient0 = int(DynamicObject.ingredient(0))
    full_recipe = _recipe(0, 0, 0)

    cases = [
        (0xC, 0, full_recipe),
        (ingredient0, 10, full_recipe),
        (0xC | int(DynamicObject.COOKED), 0, full_recipe),
        (ingredient0, 0, ingredient0),
    ]

    for contents, extra, recipe in cases:
        state = _state(
            (1, 1),
            inventory0=ingredient0,
            grid=_grid([((2, 1), StaticObject.POT, contents, extra)]),
            recipe=recipe,
        )
        assert "deliver_ingredient_to_pot" not in _valid_kinds(option_lib, state)


def test_extra_ingredient_has_recovery_option_when_pot_is_unusable():
    option_lib = OCV2OptionLibrary(_layout_graph(), max_option_steps=6)
    ingredient0 = int(DynamicObject.ingredient(0))
    cooked_soup = _recipe(0, 0, 0) | int(DynamicObject.COOKED)
    state = _state(
        (1, 1),
        inventory0=ingredient0,
        grid=_grid(
            [
                ((0, 1), StaticObject.WALL, 0, 0),
                ((2, 1), StaticObject.POT, cooked_soup, 0),
            ]
        ),
        recipe=_recipe(0, 0, 0),
    )

    valid_kinds = _valid_kinds(option_lib, state)

    assert "deliver_ingredient_to_pot" not in valid_kinds
    assert {"drop_item_to_counter", "clear_interaction_cell"} & valid_kinds
    assert "noop" in valid_kinds
    assert len(valid_kinds) > 1


def test_clear_interaction_cell_unblocks_partner_plate_soup():
    option_lib = OCV2OptionLibrary(_layout_graph(), max_option_steps=6)
    soup_ready = _recipe(0, 0, 0) | int(DynamicObject.COOKED)
    prev_state = _state(
        (1, 1),
        agent1_pos=(3, 3),
        grid=_grid([((2, 1), StaticObject.POT, soup_ready, 0)]),
        recipe=_recipe(0, 0, 0),
    )
    next_state = _state(
        (1, 0),
        agent1_pos=(1, 1),
        inventory1=int(DynamicObject.PLATE),
        grid=_grid([((2, 1), StaticObject.POT, soup_ready, 0)]),
        recipe=_recipe(0, 0, 0),
    )
    clear_id = _option_id(option_lib, "clear_interaction_cell")
    opt = option_lib.options[clear_id]

    terminated, reason = option_terminated(
        opt,
        prev_state,
        next_state,
        SimpleNamespace(),
        agent_id=0,
        elapsed=1,
        runtime=OptionRuntime(option_id=clear_id, start_pos=(1, 1)),
    )

    assert (terminated, reason) == (True, "cleared_interaction_cell")
    assert "plate_soup" in _valid_kinds(option_lib, next_state, agent_id=1)


def test_plate_soup_timeout_is_not_success():
    assert not option_success("plate_soup", "max_steps")
    assert not option_success("plate_soup", "env_max_steps")
    assert option_success("plate_soup", "plated_soup")


def test_final_step_plate_soup_success_is_not_labeled_timeout():
    recipe = _recipe(0, 0, 0)
    opt = OptionSpec(
        id=0,
        name="plate",
        kind="plate_soup",
        target_id=None,
        target_pos=None,
        entity_ids=(),
        region_ids=(),
        max_steps=1,
        metadata={},
    )
    prev_state = _state((0, 0), inventory0=int(DynamicObject.PLATE), recipe=recipe)
    next_state = _state(
        (0, 0),
        inventory0=recipe | int(DynamicObject.COOKED) | int(DynamicObject.PLATE),
        recipe=recipe,
    )

    terminated, reason = option_terminated(
        opt,
        prev_state,
        next_state,
        SimpleNamespace(),
        agent_id=0,
        elapsed=1,
    )

    assert (terminated, reason) == (True, "plated_soup")


def test_layout_parser_adds_only_reachable_wall_counters():
    static = np.full((3, 3), int(StaticObject.WALL), dtype=np.int32)
    static[1, 1] = int(StaticObject.EMPTY)
    env = SimpleNamespace(layout=SimpleNamespace(static_objects=static))

    graph = parse_layout(env, "unit")
    counter_positions = {
        graph.entities[entity_id].pos
        for entity_id in graph.entities_by_kind.get("counter", ())
    }

    assert counter_positions == {(1, 0), (0, 1), (2, 1), (1, 2)}
