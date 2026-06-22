from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable

from .canonical import canonical_hash
from .models import PublicState, TaskSpec
from .state_codec import PublicStateCodec


DEFAULT_STATIC_TYPES = (
    "pot",
    "delivery",
    "plate_pile",
    "ingredient_pile",
    "recipe_indicator",
    "button_recipe_indicator",
)


class TauFamilyGenerator:
    """Generate a nontrivial tau family from the reset public state.

    The generator deliberately avoids semantic labels or rollout outcomes. Each
    tau uses only represented public fields: positions, facing static object,
    inventory, interface hash, recipe id, and time. This expands the Gate 3
    sweep beyond the initial ``agent_at_any_of`` tau while preserving the Gate 2
    public-state contract.
    """

    def __init__(self, audited_agent_i: int = 0) -> None:
        self.audited_agent_i = int(audited_agent_i)

    def from_initial_state(self, initial: PublicState, requested: Iterable[str] | None = None) -> list[TaskSpec]:
        requested_set = set(requested or [
            "first_move_choice",
            "agent_inventory_nonempty",
            "agent_facing_static_type",
            "interface_changed_from_initial",
            "recipe_equals_initial",
            "time_at_least_2",
        ])
        tasks: list[TaskSpec] = []
        agent_i = self.audited_agent_i
        row = PublicStateCodec.agent_row(initial, agent_i)
        if row is None:
            return tasks
        x, y = int(row["x"]), int(row["y"])
        walkable = [tuple(p) for p in initial.W_dyn.get("walkable_empty_cells", [])]
        neighbor_targets = sorted({(x + dx, y + dy) for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)] if (x + dx, y + dy) in walkable})
        if "first_move_choice" in requested_set and neighbor_targets:
            tasks.append(TaskSpec(
                tau_id=_tau_id(initial.layout_id, "first_move_choice", agent_i, neighbor_targets),
                task_type="agent_at_any_of",
                params={"agent_i": agent_i, "positions": [list(p) for p in neighbor_targets]},
            ))
        if "agent_inventory_nonempty" in requested_set:
            tasks.append(TaskSpec(
                tau_id=_tau_id(initial.layout_id, "inventory_nonempty", agent_i, []),
                task_type="agent_inventory_nonempty",
                params={"agent_i": agent_i},
            ))
        if "agent_facing_static_type" in requested_set:
            object_types = set(initial.C.get("static_objects_by_type", {}).keys())
            for static_type in DEFAULT_STATIC_TYPES:
                if static_type in object_types:
                    tasks.append(TaskSpec(
                        tau_id=_tau_id(initial.layout_id, f"facing_{static_type}", agent_i, []),
                        task_type="agent_facing_static_type",
                        params={"agent_i": agent_i, "static_type": static_type},
                    ))
        if "interface_changed_from_initial" in requested_set:
            tasks.append(TaskSpec(
                tau_id=_tau_id(initial.layout_id, "interface_changed", agent_i, []),
                task_type="interface_changed_from_initial",
                params={"initial_interface_hash": str(initial.I.get("interface_hash"))},
            ))
        if "recipe_equals_initial" in requested_set:
            tasks.append(TaskSpec(
                tau_id=_tau_id(initial.layout_id, "recipe_equals_initial", agent_i, []),
                task_type="recipe_equals",
                params={"recipe_encoding": int(initial.R_recipe.get("recipe_encoding"))},
            ))
        if "time_at_least_2" in requested_set:
            tasks.append(TaskSpec(
                tau_id=_tau_id(initial.layout_id, "time_at_least_2", agent_i, []),
                task_type="time_at_least",
                params={"time_min": 2},
            ))
        # Stable order and duplicate removal.
        by_id = {t.tau_id: t for t in tasks}
        return [by_id[k] for k in sorted(by_id)]


def _tau_id(layout_id: str, name: str, agent_i: int, payload: Any) -> str:
    return f"tau_{layout_id}_{name}_a{agent_i}_{canonical_hash(payload)[:8]}"
