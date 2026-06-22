from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .canonical import canonical_hash, canonical_json
from .models import PublicState


def _arr(value: Any) -> Any:
    if value is None:
        return None
    try:
        return np.asarray(value).tolist()
    except Exception:  # noqa: BLE001 - fail with original value below
        return value


def _as_int(value: Any) -> int:
    return int(np.asarray(value).item())


def _as_bool(value: Any) -> bool:
    return bool(np.asarray(value).item())


@dataclass(frozen=True)
class EnvConfigSnapshot:
    layout_id: str
    max_steps: int
    random_reset: bool
    random_agent_positions: bool
    start_cooking_interaction: bool
    negative_rewards: bool
    sample_recipe_on_delivery: bool
    indicate_successful_delivery: bool
    initial_state_buffer_present: bool
    observation_type: str = "default"

    @classmethod
    def from_env(cls, env: Any, layout_id: str) -> "EnvConfigSnapshot":
        return cls(
            layout_id=layout_id,
            max_steps=int(getattr(env, "max_steps")),
            random_reset=bool(getattr(env, "random_reset", False)),
            random_agent_positions=bool(getattr(env, "random_agent_positions", False)),
            start_cooking_interaction=bool(np.asarray(getattr(env, "start_cooking_interaction", False)).item()),
            negative_rewards=bool(getattr(env, "negative_rewards", False)),
            sample_recipe_on_delivery=bool(np.asarray(getattr(env, "sample_recipe_on_delivery", False)).item()),
            indicate_successful_delivery=bool(getattr(env, "indicate_successful_delivery", False)),
            initial_state_buffer_present=getattr(env, "initial_state_buffer", None) is not None,
            observation_type=str(getattr(env, "observation_type", "default")),
        )


def concrete_state_to_jsonable(state: Any) -> dict[str, Any]:
    """Serialize a JAXMARL OvercookedV2 state while preserving structural fields.

    The public OvercookedV2 state object is a JAX pytree/dataclass. This
    extractor uses the field names present in the Gate 1/2 source audit and
    raises immediately if structural fields are absent. It does not fill missing
    fields with defaults because that would hide a codec/schema mismatch.
    """

    required = ["agents", "grid", "time", "terminal", "recipe"]
    missing = [name for name in required if not hasattr(state, name)]
    if missing:
        raise AttributeError(f"OvercookedV2 state missing required fields for alpha_tau: {missing}")
    agents = state.agents
    for name in ["pos", "dir", "inventory"]:
        if not hasattr(agents, name):
            raise AttributeError(f"OvercookedV2 agents object missing field {name!r}")
    pos = agents.pos
    if not hasattr(pos, "x") or not hasattr(pos, "y"):
        raise AttributeError("OvercookedV2 agents.pos must expose x and y arrays")
    return {
        "agents": {
            "pos": {"x": _arr(pos.x), "y": _arr(pos.y)},
            "dir": _arr(agents.dir),
            "inventory": _arr(agents.inventory),
        },
        "grid": _arr(state.grid),
        "time": _as_int(state.time),
        "terminal": _as_bool(state.terminal),
        "recipe": _as_int(state.recipe),
        "new_correct_delivery": _as_bool(getattr(state, "new_correct_delivery", False)),
        "ingredient_permutations": _arr(getattr(state, "ingredient_permutations", [])),
    }


def split_layout_components(grid_static: list[list[int]]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Deterministic geometric split used for runtime sweeps.

    The interface state keeps the full grid, so this split is not a state-hiding
    compression. A production proof run can swap in the exact Gate 1 component
    table for a layout-specific audit.
    """

    h = len(grid_static)
    w = len(grid_static[0]) if h else 0
    left: list[tuple[int, int]] = []
    right: list[tuple[int, int]] = []
    for y in range(h):
        for x in range(w):
            if grid_static[y][x] == 0:
                (left if x < w / 2 else right).append((x, y))
    return left, right


def static_objects_by_type(static_grid: list[list[int]]) -> dict[str, list[tuple[int, int]]]:
    names = {
        0: "empty",
        1: "wall_or_counter",
        4: "delivery",
        5: "pot",
        6: "recipe_indicator",
        7: "button_recipe_indicator",
        9: "plate_pile",
    }
    out: dict[str, list[tuple[int, int]]] = {}
    for y, row in enumerate(static_grid):
        for x, value in enumerate(row):
            label = names.get(int(value), "ingredient_pile" if int(value) >= 10 else f"static_{value}")
            out.setdefault(label, []).append((x, y))
    return {k: v for k, v in sorted(out.items())}


def adjacent_cell(x: int, y: int, direction: int) -> tuple[int, int]:
    dx, dy = {0: (0, -1), 1: (0, 1), 2: (1, 0), 3: (-1, 0)}.get(int(direction), (0, 0))
    return x + dx, y + dy


class PublicStateCodec:
    """alpha_tau and canonicalization for strict Gate 2 public states."""

    def __init__(self, layout_id: str, config_snapshot: EnvConfigSnapshot, reset_domain_tag: str = "RESET_DEFAULT_OR_SEE_STATE") -> None:
        self.layout_id = layout_id
        self.config_snapshot = config_snapshot
        self.reset_domain_tag = reset_domain_tag

    def alpha_tau(self, state: Any) -> PublicState:
        c = concrete_state_to_jsonable(state)
        grid = c["grid"]
        static_grid = [[int(cell[0]) for cell in row] for row in grid]
        dynamic_grid = [[int(cell[1]) for cell in row] for row in grid]
        extra_grid = [[int(cell[2]) for cell in row] for row in grid]
        objects = static_objects_by_type(static_grid)
        left_cells, right_cells = split_layout_components(static_grid)
        agents = c["agents"]
        agent_rows = []
        for i, (x, y, d, inv) in enumerate(zip(agents["pos"]["x"], agents["pos"]["y"], agents["dir"], agents["inventory"])):
            fx, fy = adjacent_cell(int(x), int(y), int(d))
            agent_rows.append({
                "agent_i": int(i),
                "x": int(x),
                "y": int(y),
                "dir": int(d),
                "inventory": int(inv),
                "facing_x": int(fx),
                "facing_y": int(fy),
                "facing_static": _static_at(static_grid, fx, fy),
                "facing_dynamic": _grid_at(dynamic_grid, fx, fy),
                "facing_extra": _grid_at(extra_grid, fx, fy),
            })
        left_set = set(left_cells)
        right_set = set(right_cells)
        interface_payload = {
            "agents": agent_rows,
            "dynamic_grid": dynamic_grid,
            "extra_grid": extra_grid,
            "new_correct_delivery": c["new_correct_delivery"],
        }
        L = {
            "component_cells": left_cells,
            "agents_in_component": [a for a in agent_rows if (a["x"], a["y"]) in left_set],
        }
        R = {
            "component_cells": right_cells,
            "agents_in_component": [a for a in agent_rows if (a["x"], a["y"]) in right_set],
        }
        C = {
            "static_grid_hash": canonical_hash(static_grid),
            "height": len(grid),
            "width": len(grid[0]) if grid else 0,
            "static_objects_by_type": objects,
            "static_grid_compact": static_grid,
        }
        W_dyn = {
            "walkable_empty_cells": [(x, y) for y, row in enumerate(static_grid) for x, s in enumerate(row) if s == 0],
            "occupied_positions": [(a["x"], a["y"]) for a in agent_rows],
        }
        return PublicState(
            layout_id=self.layout_id,
            L=L,
            R=R,
            I={**interface_payload, "interface_hash": canonical_hash(interface_payload)},
            C=C,
            W_dyn=W_dyn,
            C_cfg=asdict(self.config_snapshot),
            T={"time": c["time"], "terminal": c["terminal"], "max_steps": self.config_snapshot.max_steps},
            R_recipe={"recipe_encoding": c["recipe"]},
            ResetDomainTag=self.reset_domain_tag,
        )

    @staticmethod
    def public_state_hash(v: PublicState) -> str:
        return canonical_hash(v)

    @staticmethod
    def public_state_json(v: PublicState) -> str:
        return canonical_json(v)

    @staticmethod
    def agent_position(v: PublicState, agent_i: int) -> tuple[int, int] | None:
        for row in v.I["agents"]:
            if row["agent_i"] == agent_i:
                return int(row["x"]), int(row["y"])
        return None

    @staticmethod
    def agent_row(v: PublicState, agent_i: int) -> dict[str, Any] | None:
        for row in v.I["agents"]:
            if int(row["agent_i"]) == int(agent_i):
                return row
        return None


def _grid_at(grid: list[list[int]], x: int, y: int) -> int | None:
    if y < 0 or y >= len(grid) or x < 0 or (grid and x >= len(grid[0])):
        return None
    return int(grid[y][x])


def _static_at(static_grid: list[list[int]], x: int, y: int) -> int | None:
    return _grid_at(static_grid, x, y)
