from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from jaxmarl.environments.overcooked_v2.common import StaticObject

from .state_utils import is_ingredient_pile

GridPos = tuple[int, int]

_DIRECTION_DELTAS: tuple[GridPos, ...] = (
    (0, -1),
    (0, 1),
    (1, 0),
    (-1, 0),
)


@dataclass(frozen=True)
class Entity:
    id: str
    kind: str
    pos: GridPos


@dataclass
class LayoutGraph:
    layout_name: str
    width: int
    height: int
    passable: np.ndarray
    entities: dict[str, Entity]
    entities_by_kind: dict[str, list[str]]
    interaction_cells: dict[str, list[GridPos]]
    bottlenecks: list[GridPos]
    region_cells: dict[str, list[GridPos]]
    cell_to_entity: dict[GridPos, str]
    shortest_path_dist: dict[tuple[GridPos, GridPos], int]


def parse_layout(env, layout_name: str) -> LayoutGraph:
    """Parse any OvercookedV2-like object exposing .layout.static_objects."""
    static = np.asarray(env.layout.static_objects)
    height, width = static.shape
    passable = static == int(StaticObject.EMPTY)

    entities: dict[str, Entity] = {}
    entities_by_kind: dict[str, list[str]] = {}
    cell_to_entity: dict[GridPos, str] = {}

    def add(kind: str, x: int, y: int) -> None:
        entity_id = f"{kind}:{x}:{y}"
        entities[entity_id] = Entity(id=entity_id, kind=kind, pos=(x, y))
        entities_by_kind.setdefault(kind, []).append(entity_id)
        cell_to_entity[(x, y)] = entity_id

    for y in range(height):
        for x in range(width):
            obj = int(static[y, x])
            if obj == int(StaticObject.GOAL):
                add("delivery", x, y)
            elif obj == int(StaticObject.POT):
                add("pot", x, y)
            elif obj == int(StaticObject.PLATE_PILE):
                add("plate_pile", x, y)
            elif obj == int(StaticObject.RECIPE_INDICATOR):
                add("recipe_indicator", x, y)
            elif obj == int(StaticObject.BUTTON_RECIPE_INDICATOR):
                add("button_recipe_indicator", x, y)
            elif is_ingredient_pile(obj):
                ingredient_idx = obj - int(StaticObject.INGREDIENT_PILE_BASE)
                add(f"ingredient_pile:{ingredient_idx}", x, y)
            elif obj == int(StaticObject.WALL):
                if adjacent_passable_cells((x, y), passable):
                    add("counter", x, y)

    interaction_cells = {
        entity_id: adjacent_passable_cells(entity.pos, passable)
        for entity_id, entity in entities.items()
    }
    bottlenecks = find_bottlenecks_articulation(passable, min_region_size=2)
    region_cells = {
        f"bottleneck:{idx}": [pos] for idx, pos in enumerate(bottlenecks)
    }
    shortest_path_dist = all_pairs_shortest_paths(passable)

    return LayoutGraph(
        layout_name=layout_name,
        width=width,
        height=height,
        passable=passable,
        entities=entities,
        entities_by_kind=entities_by_kind,
        interaction_cells=interaction_cells,
        bottlenecks=bottlenecks,
        region_cells=region_cells,
        cell_to_entity=cell_to_entity,
        shortest_path_dist=shortest_path_dist,
    )


def find_bottlenecks_articulation(
    passable: np.ndarray,
    min_region_size: int = 2,
) -> list[GridPos]:
    graph = _build_grid_graph(passable)
    articulation = _tarjan_articulation_points(graph)
    bottlenecks: list[GridPos] = []

    for cell in sorted(articulation, key=_row_major_key):
        components = _connected_components_after_removal(graph, cell)
        sizes = sorted((len(component) for component in components), reverse=True)
        if len(sizes) < 2:
            continue
        if sizes[1] < min_region_size:
            continue
        bottlenecks.append(cell)

    return bottlenecks


def all_pairs_shortest_paths(
    passable: np.ndarray,
) -> dict[tuple[GridPos, GridPos], int]:
    distances: dict[tuple[GridPos, GridPos], int] = {}

    for source in _passable_cells(passable):
        source_distances = _bfs_distances(source, passable)
        for target, distance in source_distances.items():
            distances[(source, target)] = distance

    return distances


def adjacent_passable_cells(pos: GridPos, passable: np.ndarray) -> list[GridPos]:
    x, y = pos
    height, width = passable.shape
    neighbors: list[GridPos] = []

    for dx, dy in _DIRECTION_DELTAS:
        nx = x + dx
        ny = y + dy
        if 0 <= nx < width and 0 <= ny < height and bool(passable[ny, nx]):
            neighbors.append((nx, ny))

    return neighbors


def _passable_cells(passable: np.ndarray) -> list[GridPos]:
    height, width = passable.shape
    return [
        (x, y)
        for y in range(height)
        for x in range(width)
        if bool(passable[y, x])
    ]


def _bfs_distances(source: GridPos, passable: np.ndarray) -> dict[GridPos, int]:
    distances = {source: 0}
    queue: deque[GridPos] = deque([source])

    while queue:
        cell = queue.popleft()
        next_distance = distances[cell] + 1
        for neighbor in adjacent_passable_cells(cell, passable):
            if neighbor in distances:
                continue
            distances[neighbor] = next_distance
            queue.append(neighbor)

    return distances


def _build_grid_graph(passable: np.ndarray) -> dict[GridPos, list[GridPos]]:
    return {
        cell: adjacent_passable_cells(cell, passable)
        for cell in _passable_cells(passable)
    }


def _tarjan_articulation_points(
    graph: dict[GridPos, list[GridPos]],
) -> set[GridPos]:
    discovery: dict[GridPos, int] = {}
    low: dict[GridPos, int] = {}
    parent: dict[GridPos, GridPos | None] = {}
    articulation: set[GridPos] = set()
    time = 0

    def visit(cell: GridPos) -> None:
        nonlocal time

        discovery[cell] = time
        low[cell] = time
        time += 1
        child_count = 0

        for neighbor in graph[cell]:
            if neighbor not in discovery:
                parent[neighbor] = cell
                child_count += 1
                visit(neighbor)
                low[cell] = min(low[cell], low[neighbor])

                if parent[cell] is None and child_count > 1:
                    articulation.add(cell)
                if parent[cell] is not None and low[neighbor] >= discovery[cell]:
                    articulation.add(cell)
            elif neighbor != parent[cell]:
                low[cell] = min(low[cell], discovery[neighbor])

    for cell in sorted(graph, key=_row_major_key):
        if cell in discovery:
            continue
        parent[cell] = None
        visit(cell)

    return articulation


def _connected_components_after_removal(
    graph: dict[GridPos, list[GridPos]],
    removed: GridPos,
) -> list[set[GridPos]]:
    remaining = set(graph)
    remaining.discard(removed)
    components: list[set[GridPos]] = []

    while remaining:
        start = min(remaining, key=_row_major_key)
        remaining.remove(start)
        component = {start}
        stack = [start]

        while stack:
            cell = stack.pop()
            for neighbor in graph[cell]:
                if neighbor == removed or neighbor not in remaining:
                    continue
                remaining.remove(neighbor)
                component.add(neighbor)
                stack.append(neighbor)

        components.append(component)

    return components


def _row_major_key(pos: GridPos) -> tuple[int, int]:
    x, y = pos
    return (y, x)
