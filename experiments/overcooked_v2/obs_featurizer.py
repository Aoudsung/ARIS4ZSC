from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from jaxmarl.environments.overcooked_v2.common import DynamicObject, StaticObject

from .layout_parser import GridPos, LayoutGraph, adjacent_passable_cells
from .state_utils import get_agent_pos


PLAYER_FEATURE_DIM = 46
AGENT_OBS_DIM = 96


class NumpyFeaturizer:
    """Pure numpy OvercookedV2 state featurizer with the fixed 2-agent contract."""

    def __init__(
        self,
        layout_graph: LayoutGraph,
        *,
        num_agents: int = 2,
        num_pots: int = 2,
    ):
        if int(num_agents) != 2:
            raise ValueError("NumpyFeaturizer currently supports exactly 2 agents.")
        if int(num_pots) != 2:
            raise ValueError("NumpyFeaturizer keeps a fixed 2-pot, 96-dim contract.")

        self.layout_graph = layout_graph
        self.num_agents = int(num_agents)
        self.num_pots = int(num_pots)
        self.spd = layout_graph.shortest_path_dist
        self.passable = np.asarray(layout_graph.passable, dtype=bool)
        if self.passable.ndim != 2:
            raise ValueError("layout_graph.passable must be a rank-2 boolean grid.")
        self.height, self.width = self.passable.shape
        self._pot_positions = tuple(self._entity_positions("pot"))

    def __call__(self, state: Any) -> dict[str, np.ndarray]:
        self._validate_agent_count(state)
        features = [self._player_features(state, agent_id) for agent_id in range(2)]
        positions = [get_agent_pos(state, agent_id) for agent_id in range(2)]

        obs: dict[str, np.ndarray] = {}
        for agent_id in range(2):
            other_id = 1 - agent_id
            dist_to_other = _delta(positions[other_id], positions[agent_id])
            agent_obs = np.concatenate(
                [
                    features[agent_id],
                    features[other_id],
                    np.asarray(dist_to_other, dtype=np.float32),
                    np.asarray(positions[agent_id], dtype=np.float32),
                ]
            ).astype(np.float32, copy=False)
            if agent_obs.shape != (AGENT_OBS_DIM,):
                raise ValueError(
                    f"NumpyFeaturizer produced shape {agent_obs.shape}; "
                    f"expected ({AGENT_OBS_DIM},)."
                )
            obs[f"agent_{agent_id}"] = agent_obs
        return obs

    def _player_features(self, state: Any, agent_id: int) -> np.ndarray:
        grid = self._state_grid(state)
        pos = get_agent_pos(state, agent_id)
        self._validate_pos(pos, f"agent_{agent_id}")

        direction = _to_int(state.agents.dir[agent_id])
        if direction < 0 or direction >= 4:
            raise ValueError(f"Invalid agent direction {direction}; expected 0..3.")
        inventory = _to_int(state.agents.inventory[agent_id])

        orientation = np.zeros(4, dtype=np.float32)
        orientation[direction] = 1.0

        soup = self._soup_value(state)
        inventory_items = np.asarray(
            [
                _ingredient(0),
                soup,
                int(DynamicObject.PLATE),
                _ingredient(1),
            ],
            dtype=np.int64,
        )
        inventory_features = (inventory == inventory_items).astype(np.float32)

        onion_features = self._delta_to_closest(
            pos,
            grid,
            inventory,
            static_locator=_ingredient_pile(0),
            dynamic_locator=_ingredient(0),
        )
        tomato_features = self._delta_to_closest(
            pos,
            grid,
            inventory,
            static_locator=_ingredient_pile(1),
            dynamic_locator=_ingredient(1),
        )
        dish_features = self._delta_to_closest(
            pos,
            grid,
            inventory,
            static_locator=int(StaticObject.PLATE_PILE),
            dynamic_locator=int(DynamicObject.PLATE),
        )
        soup_features = self._delta_to_closest(
            pos,
            grid,
            inventory,
            dynamic_locator=soup,
        )
        soup_ingredient_features = self._soup_ingredient_features(state, grid, inventory)
        serving_features = self._delta_to_closest(
            pos,
            grid,
            inventory,
            static_locator=int(StaticObject.GOAL),
        )
        empty_counter_features = self._delta_to_closest(
            pos,
            grid,
            inventory,
            static_locator=int(StaticObject.WALL),
            no_ingredients=True,
        )
        pot_features = self._pot_features(pos, grid)
        wall_features = self._wall_features(pos)

        player_features = np.concatenate(
            [
                orientation,
                inventory_features,
                onion_features,
                tomato_features,
                dish_features,
                soup_features,
                soup_ingredient_features,
                serving_features,
                empty_counter_features,
                pot_features,
                wall_features,
            ]
        ).astype(np.float32, copy=False)
        if player_features.shape != (PLAYER_FEATURE_DIM,):
            raise ValueError(
                f"Player feature shape {player_features.shape}; "
                f"expected ({PLAYER_FEATURE_DIM},)."
            )
        return player_features

    def _delta_to_closest(
        self,
        pos: GridPos,
        grid: np.ndarray,
        inventory: int,
        *,
        static_locator: int | None = None,
        dynamic_locator: int | None = None,
        no_ingredients: bool = False,
        not_in_pot: bool = True,
    ) -> np.ndarray:
        candidates: set[GridPos] = set()
        static = grid[:, :, 0]
        dynamic = grid[:, :, 1]

        if static_locator is not None:
            static_mask = static == int(static_locator)
            if no_ingredients:
                static_mask &= dynamic == int(DynamicObject.EMPTY)
            candidates.update(_positions_from_mask(static_mask))

        if dynamic_locator is not None:
            dynamic_mask = dynamic == int(dynamic_locator)
            if not_in_pot:
                dynamic_mask &= static != int(StaticObject.POT)
            candidates.update(_positions_from_mask(dynamic_mask))
            if int(inventory) == int(dynamic_locator):
                candidates.add(pos)

        closest = self._closest_cell(pos, candidates)
        if closest is None:
            return np.zeros(2, dtype=np.float32)
        return np.asarray(_delta(closest, pos), dtype=np.float32)

    def _closest_cell(
        self,
        agent_pos: GridPos,
        candidates: Iterable[GridPos],
    ) -> GridPos | None:
        best: GridPos | None = None
        best_distance = float("inf")
        for candidate in sorted(set(candidates), key=_row_major_key):
            distance = self._target_distance(agent_pos, candidate)
            if distance < best_distance:
                best = candidate
                best_distance = distance
        if not np.isfinite(best_distance):
            return None
        return best

    def _target_distance(self, agent_pos: GridPos, target: GridPos) -> float:
        if agent_pos == target:
            return 0.0
        if not self._in_bounds(target):
            return float("inf")
        x, y = target
        if bool(self.passable[y, x]):
            return float(self.spd.get((agent_pos, target), float("inf")))

        interaction_cells = self._interaction_cells(target)
        distances = [
            self.spd.get((agent_pos, cell), float("inf"))
            for cell in interaction_cells
        ]
        return float(min(distances, default=float("inf")))

    def _interaction_cells(self, target: GridPos) -> list[GridPos]:
        entity_id = self.layout_graph.cell_to_entity.get(target)
        if entity_id is not None:
            return [
                tuple(cell)
                for cell in self.layout_graph.interaction_cells.get(entity_id, ())
            ]
        return adjacent_passable_cells(target, self.passable)

    def _soup_ingredient_features(
        self,
        state: Any,
        grid: np.ndarray,
        inventory: int,
    ) -> np.ndarray:
        soup = self._soup_value(state)
        soup_visible = bool(np.any(grid[:, :, 1] == soup)) or int(inventory) == soup
        if not soup_visible:
            return np.zeros(2, dtype=np.float32)
        recipe = self._recipe_value(state)
        return np.asarray(
            [_ingredient_slot_count(recipe, 0), _ingredient_slot_count(recipe, 1)],
            dtype=np.float32,
        )

    def _pot_features(self, pos: GridPos, grid: np.ndarray) -> np.ndarray:
        candidates = set(self._pot_positions)
        candidates.update(
            _positions_from_mask(grid[:, :, 0] == int(StaticObject.POT))
        )

        features: list[np.ndarray] = []
        remaining = set(candidates)
        for _ in range(self.num_pots):
            pot_pos = self._closest_cell(pos, remaining)
            if pot_pos is None:
                features.append(np.zeros(10, dtype=np.float32))
                continue
            remaining.discard(pot_pos)
            x, y = pot_pos
            pot_contents = _to_int(grid[y, x, 1])
            pot_timer = _to_int(grid[y, x, 2]) if grid.shape[2] > 2 else 0
            pot_feature = np.asarray(
                [
                    1,
                    pot_contents == int(DynamicObject.EMPTY),
                    _ingredient_count(pot_contents) == 3,
                    pot_timer > 0,
                    (pot_contents & int(DynamicObject.COOKED))
                    == int(DynamicObject.COOKED),
                    _ingredient_slot_count(pot_contents, 0),
                    _ingredient_slot_count(pot_contents, 1),
                    pot_timer,
                    *_delta(pot_pos, pos),
                ],
                dtype=np.float32,
            )
            features.append(pot_feature)

        return np.concatenate(features).astype(np.float32, copy=False)

    def _wall_features(self, pos: GridPos) -> np.ndarray:
        x, y = pos
        neighbors = ((x, y - 1), (x, y + 1), (x + 1, y), (x - 1, y))
        return np.asarray(
            [
                (not self._in_bounds(cell))
                or (not bool(self.passable[cell[1], cell[0]]))
                for cell in neighbors
            ],
            dtype=np.float32,
        )

    def _state_grid(self, state: Any) -> np.ndarray:
        if not hasattr(state, "grid"):
            raise ValueError("NumpyFeaturizer requires state.grid.")
        grid = np.asarray(state.grid)
        if grid.ndim != 3 or grid.shape[2] < 2:
            raise ValueError(
                "state.grid must have shape (height, width, channels>=2)."
            )
        if grid.shape[:2] != self.passable.shape:
            raise ValueError(
                f"state.grid spatial shape {grid.shape[:2]} does not match "
                f"layout shape {self.passable.shape}."
            )
        return grid

    def _recipe_value(self, state: Any) -> int:
        if not hasattr(state, "recipe"):
            raise ValueError("NumpyFeaturizer requires state.recipe.")
        return _to_int(state.recipe)

    def _soup_value(self, state: Any) -> int:
        return (
            self._recipe_value(state)
            | int(DynamicObject.COOKED)
            | int(DynamicObject.PLATE)
        )

    def _validate_agent_count(self, state: Any) -> None:
        if not hasattr(state, "agents") or not hasattr(state.agents, "inventory"):
            raise ValueError("NumpyFeaturizer requires state.agents.inventory.")
        inventory = np.asarray(state.agents.inventory)
        if inventory.shape[0] != self.num_agents:
            raise ValueError(
                f"NumpyFeaturizer expected {self.num_agents} agents; "
                f"got inventory shape {inventory.shape}."
            )

    def _validate_pos(self, pos: GridPos, label: str) -> None:
        if not self._in_bounds(pos):
            raise ValueError(f"{label} position {pos} is outside the layout bounds.")

    def _in_bounds(self, pos: GridPos) -> bool:
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def _entity_positions(self, kind: str) -> list[GridPos]:
        positions: list[GridPos] = []
        for entity_id in self.layout_graph.entities_by_kind.get(kind, ()):
            entity = self.layout_graph.entities[entity_id]
            positions.append(tuple(entity.pos))
        return positions


def _to_int(value: Any) -> int:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Expected scalar integer value, got shape {array.shape}.")
    return int(array.reshape(-1)[0].item())


def _ingredient(idx: int) -> int:
    return int(DynamicObject.ingredient(int(idx)))


def _ingredient_pile(idx: int) -> int:
    return int(StaticObject.INGREDIENT_PILE_BASE) + int(idx)


def _ingredient_count(value: int) -> int:
    value = int(value) >> 2
    count = 0
    while value > 0:
        count += value & 0x3
        value >>= 2
    return int(count)


def _ingredient_slot_count(value: int, idx: int) -> int:
    return int((int(value) >> (2 + 2 * int(idx))) & 0x3)


def _positions_from_mask(mask: np.ndarray) -> list[GridPos]:
    return [(int(x), int(y)) for y, x in np.argwhere(np.asarray(mask, dtype=bool))]


def _delta(target: GridPos, source: GridPos) -> tuple[int, int]:
    return (int(target[0]) - int(source[0]), int(target[1]) - int(source[1]))


def _row_major_key(pos: GridPos) -> tuple[int, int]:
    return (int(pos[1]), int(pos[0]))
