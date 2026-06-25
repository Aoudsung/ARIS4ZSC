from __future__ import annotations

from typing import Any

import numpy as np

from src.aris_bellman.specs import FactorSpec, GraphSpec

GridPos = tuple[int, int]

D_EVID = 64

EVIDENCE_INDEX = {
    "ego_near_entity": 0,
    "partner_near_entity": 1,
    "ego_distance_delta": 2,
    "partner_distance_delta": 3,
    "ego_interacted_near_target": 4,
    "partner_interacted_near_target": 5,
    "collision_or_block_near_region": 6,
    "ego_inventory_changed": 7,
    "partner_inventory_changed": 8,
    "option_progress_fraction": 9,
    "ego_option_is_i": 10,
    "ego_option_is_j": 11,
    "partner_option_is_i": 12,
    "partner_option_is_j": 13,
    "partner_option_dist_i": 14,
    "partner_option_dist_j": 15,
    "partner_option_confidence": 16,
    "partner_option_switched": 17,
    "ego_partner_same_entity_target": 18,
    "ego_partner_same_region_target": 19,
    "ego_waited": 20,
    "partner_waited": 21,
    "partner_waited_after_ego_approach": 22,
    "partner_moved_after_ego_wait": 23,
    "ego_option_elapsed_fraction": 24,
    "changed_cell_touches_entity": 25,
    "changed_cell_touches_region": 26,
    "changed_object_bits_any": 27,
    "pot_changed": 28,
    "object_pickup_or_drop": 29,
    "recipe_indicator_event": 30,
    "button_pressed": 31,
    "delivery_event": 32,
    "wrong_delivery_event": 33,
    "ego_waited_near_region": 34,
    "partner_waited_near_region": 35,
    "ego_moved_toward_target": 36,
    "partner_moved_toward_target": 37,
    "partner_option_known": 38,
    "partner_option_pair_mass": 39,
    "pot_became_full": 40,
    "pot_became_cooked": 41,
    "pot_became_ready": 42,
    "plate_picked": 43,
    "soup_picked": 44,
    "correct_delivery": 45,
    "pot_changed_near_entity": 46,
    "pot_changed_near_region": 47,
}


class OCV2EvidenceRouter:
    def __init__(
        self,
        graph: GraphSpec,
        cell_to_entity: dict[GridPos, str],
        region_cells: dict[str, list[GridPos]],
    ):
        self.graph = graph
        self.cell_to_entity = dict(cell_to_entity)
        self.region_cells = {
            region_id: [tuple(cell) for cell in cells]
            for region_id, cells in region_cells.items()
        }
        self.entity_to_cell = {
            entity_id: tuple(cell)
            for cell, entity_id in self.cell_to_entity.items()
        }
        self.option_by_id = {int(opt.id): opt for opt in graph.options}
        self._factor_active = np.asarray(graph.factor_mask, dtype=bool)
        self._previous_partner_option: int | None = None
        self._validate_factor_refs()

    def reset(self) -> None:
        self._previous_partner_option = None

    def route(
        self,
        event: Any,
        ego_option_id: int | None = None,
        ego_option_elapsed: int | None = None,
        ego_option_max_steps: int | None = None,
    ) -> np.ndarray:
        routed = np.zeros((self.graph.num_factors, D_EVID), dtype=np.float32)
        current_partner_option = _as_optional_int(getattr(event, "partner_option", None))
        partner_switched = (
            current_partner_option is not None
            and self._previous_partner_option is not None
            and current_partner_option != self._previous_partner_option
        )
        for factor_idx, factor in enumerate(self.graph.factors):
            if factor_idx < self._factor_active.size and not bool(self._factor_active[factor_idx]):
                continue
            entity_ids, region_ids = self._route_entities_regions(factor_idx, factor)
            entity_cells = self._entity_cells(entity_ids)
            region = self._region_cells(region_ids)
            targets = entity_cells + region

            routed[factor_idx, EVIDENCE_INDEX["ego_near_entity"]] = _near_any(
                event.ego_pos_after,
                entity_cells,
            )
            routed[factor_idx, EVIDENCE_INDEX["partner_near_entity"]] = _near_any(
                event.partner_pos_after,
                entity_cells,
            )
            routed[factor_idx, EVIDENCE_INDEX["ego_distance_delta"]] = _distance_delta(
                event.ego_pos_before,
                event.ego_pos_after,
                targets,
            )
            routed[factor_idx, EVIDENCE_INDEX["partner_distance_delta"]] = _distance_delta(
                event.partner_pos_before,
                event.partner_pos_after,
                targets,
            )
            routed[factor_idx, EVIDENCE_INDEX["ego_interacted_near_target"]] = float(
                bool(event.ego_interacted) and _near_any(event.ego_pos_after, targets)
            )
            routed[factor_idx, EVIDENCE_INDEX["partner_interacted_near_target"]] = float(
                bool(event.partner_interacted)
                and _near_any(event.partner_pos_after, targets)
            )
            routed[factor_idx, EVIDENCE_INDEX["collision_or_block_near_region"]] = float(
                bool(event.collision_or_block)
                and (
                    _near_any(event.ego_pos_after, region)
                    or _near_any(event.partner_pos_after, region)
                )
            )
            routed[factor_idx, EVIDENCE_INDEX["ego_inventory_changed"]] = float(
                int(event.ego_inventory_before) != int(event.ego_inventory_after)
            )
            routed[factor_idx, EVIDENCE_INDEX["partner_inventory_changed"]] = float(
                int(event.partner_inventory_before)
                != int(event.partner_inventory_after)
            )
            routed[factor_idx, EVIDENCE_INDEX["option_progress_fraction"]] = _option_progress_fraction(
                factor.option_i,
                factor.option_j,
                ego_option_id,
                routed[factor_idx, EVIDENCE_INDEX["ego_distance_delta"]],
            )
            self._route_partner_option_features(
                routed[factor_idx],
                event,
                factor,
                ego_option_id,
                current_partner_option,
                partner_switched,
            )
            self._route_timing_and_object_features(
                routed[factor_idx],
                event,
                entity_cells,
                region,
                targets,
                ego_option_elapsed,
                ego_option_max_steps,
            )
        if current_partner_option is not None:
            self._previous_partner_option = current_partner_option
        return routed

    def _validate_factor_refs(self) -> None:
        for factor_idx, factor in enumerate(self.graph.factors):
            if factor_idx < self._factor_active.size and not bool(self._factor_active[factor_idx]):
                continue
            missing_entities = [
                entity_id
                for entity_id in factor.entity_ids
                if entity_id not in self.entity_to_cell
            ]
            if missing_entities:
                raise KeyError(
                    f"Factor {factor.id} references missing entities {missing_entities}."
                )
            missing_regions = [
                region_id
                for region_id in factor.region_ids
                if region_id not in self.region_cells
            ]
            if missing_regions:
                raise KeyError(
                    f"Factor {factor.id} references missing regions {missing_regions}."
                )
        for factor_idx, source_options in self.graph.route_map.items():
            missing_options = [
                int(option_id)
                for option_id in source_options
                if int(option_id) not in self.option_by_id
            ]
            if missing_options:
                raise KeyError(
                    f"Route map for factor {factor_idx} references missing options "
                    f"{missing_options}."
                )

    def _route_entities_regions(
        self,
        factor_idx: int,
        factor: FactorSpec,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        source_options = self.graph.route_map.get(int(factor_idx))
        if source_options is None:
            return factor.entity_ids, factor.region_ids

        entity_ids: set[str] = set()
        region_ids: set[str] = set()
        for option_id in source_options:
            option = self.option_by_id[int(option_id)]
            entity_ids.update(option.entity_ids)
            region_ids.update(option.region_ids)
        return tuple(sorted(entity_ids)), tuple(sorted(region_ids))

    def _route_partner_option_features(
        self,
        row: np.ndarray,
        event: Any,
        factor: FactorSpec,
        ego_option_id: int | None,
        current_partner_option: int | None,
        partner_switched: bool,
    ) -> None:
        ego_option_id = _as_optional_int(ego_option_id)
        pair = {int(factor.option_i), int(factor.option_j)}
        row[EVIDENCE_INDEX["ego_option_is_i"]] = float(ego_option_id == int(factor.option_i))
        row[EVIDENCE_INDEX["ego_option_is_j"]] = float(ego_option_id == int(factor.option_j))
        row[EVIDENCE_INDEX["partner_option_is_i"]] = float(
            current_partner_option == int(factor.option_i)
        )
        row[EVIDENCE_INDEX["partner_option_is_j"]] = float(
            current_partner_option == int(factor.option_j)
        )
        row[EVIDENCE_INDEX["partner_option_dist_i"]] = _option_dist_value(
            getattr(event, "partner_option_dist", None),
            int(factor.option_i),
        )
        row[EVIDENCE_INDEX["partner_option_dist_j"]] = _option_dist_value(
            getattr(event, "partner_option_dist", None),
            int(factor.option_j),
        )
        row[EVIDENCE_INDEX["partner_option_confidence"]] = float(
            getattr(event, "partner_option_confidence", 0.0)
        )
        row[EVIDENCE_INDEX["partner_option_switched"]] = float(partner_switched)
        row[EVIDENCE_INDEX["partner_option_known"]] = float(current_partner_option is not None)
        row[EVIDENCE_INDEX["partner_option_pair_mass"]] = (
            row[EVIDENCE_INDEX["partner_option_dist_i"]]
            + row[EVIDENCE_INDEX["partner_option_dist_j"]]
        )
        if ego_option_id is not None and current_partner_option is not None:
            ego_option = self.option_by_id.get(ego_option_id)
            partner_option = self.option_by_id.get(current_partner_option)
            if ego_option is not None and partner_option is not None:
                row[EVIDENCE_INDEX["ego_partner_same_entity_target"]] = float(
                    bool(set(ego_option.entity_ids) & set(partner_option.entity_ids))
                )
                row[EVIDENCE_INDEX["ego_partner_same_region_target"]] = float(
                    bool(set(ego_option.region_ids) & set(partner_option.region_ids))
                )
        if current_partner_option in pair:
            row[EVIDENCE_INDEX["partner_option_known"]] = 1.0

    def _route_timing_and_object_features(
        self,
        row: np.ndarray,
        event: Any,
        entity_cells: list[GridPos],
        region: list[GridPos],
        targets: list[GridPos],
        ego_option_elapsed: int | None,
        ego_option_max_steps: int | None,
    ) -> None:
        row[EVIDENCE_INDEX["ego_waited"]] = float(bool(getattr(event, "ego_waited", False)))
        row[EVIDENCE_INDEX["partner_waited"]] = float(
            bool(getattr(event, "partner_waited", False))
        )
        row[EVIDENCE_INDEX["partner_waited_after_ego_approach"]] = float(
            bool(getattr(event, "partner_waited", False))
            and _near_any(getattr(event, "ego_pos_after", (0, 0)), region)
        )
        row[EVIDENCE_INDEX["partner_moved_after_ego_wait"]] = float(
            bool(getattr(event, "ego_waited", False))
            and not bool(getattr(event, "partner_waited", False))
            and _distance_delta(
                getattr(event, "partner_pos_before", (0, 0)),
                getattr(event, "partner_pos_after", (0, 0)),
                targets,
            )
            > 0.0
        )
        row[EVIDENCE_INDEX["ego_option_elapsed_fraction"]] = _elapsed_fraction(
            ego_option_elapsed,
            ego_option_max_steps,
        )
        changed_cells = [tuple(cell) for cell in getattr(event, "changed_cells", ())]
        row[EVIDENCE_INDEX["changed_cell_touches_entity"]] = float(
            bool(set(changed_cells) & set(entity_cells))
        )
        row[EVIDENCE_INDEX["changed_cell_touches_region"]] = float(
            _changed_cell_touches_any(changed_cells, region)
        )
        row[EVIDENCE_INDEX["changed_object_bits_any"]] = float(
            any(int(bit) != 0 for bit in getattr(event, "changed_object_bits", ()))
        )
        row[EVIDENCE_INDEX["pot_changed"]] = float(bool(getattr(event, "pot_changed", False)))
        row[EVIDENCE_INDEX["object_pickup_or_drop"]] = float(
            bool(getattr(event, "object_pickup_or_drop", False))
        )
        row[EVIDENCE_INDEX["recipe_indicator_event"]] = float(
            bool(getattr(event, "recipe_indicator_event", False))
        )
        row[EVIDENCE_INDEX["button_pressed"]] = float(
            bool(getattr(event, "button_pressed", False))
        )
        row[EVIDENCE_INDEX["delivery_event"]] = float(
            bool(getattr(event, "delivery_event", False))
        )
        row[EVIDENCE_INDEX["wrong_delivery_event"]] = float(
            bool(getattr(event, "wrong_delivery_event", False))
        )
        row[EVIDENCE_INDEX["pot_became_full"]] = float(
            bool(getattr(event, "pot_became_full", False))
        )
        row[EVIDENCE_INDEX["pot_became_cooked"]] = float(
            bool(getattr(event, "pot_became_cooked", False))
        )
        row[EVIDENCE_INDEX["pot_became_ready"]] = float(
            bool(getattr(event, "pot_became_ready", False))
        )
        row[EVIDENCE_INDEX["plate_picked"]] = float(
            bool(getattr(event, "plate_picked", False))
        )
        row[EVIDENCE_INDEX["soup_picked"]] = float(
            bool(getattr(event, "soup_picked", False))
        )
        row[EVIDENCE_INDEX["correct_delivery"]] = float(
            bool(getattr(event, "correct_delivery", False))
        )
        pot_changed_cells = [
            tuple(cell) for cell in getattr(event, "pot_changed_cells", ())
        ]
        row[EVIDENCE_INDEX["pot_changed_near_entity"]] = float(
            _changed_cell_touches_any(pot_changed_cells, entity_cells)
        )
        row[EVIDENCE_INDEX["pot_changed_near_region"]] = float(
            _changed_cell_touches_any(pot_changed_cells, region)
        )
        row[EVIDENCE_INDEX["ego_waited_near_region"]] = float(
            bool(getattr(event, "ego_waited", False))
            and _near_any(getattr(event, "ego_pos_after", (0, 0)), region)
        )
        row[EVIDENCE_INDEX["partner_waited_near_region"]] = float(
            bool(getattr(event, "partner_waited", False))
            and _near_any(getattr(event, "partner_pos_after", (0, 0)), region)
        )
        row[EVIDENCE_INDEX["ego_moved_toward_target"]] = float(
            row[EVIDENCE_INDEX["ego_distance_delta"]] > 0.0
        )
        row[EVIDENCE_INDEX["partner_moved_toward_target"]] = float(
            row[EVIDENCE_INDEX["partner_distance_delta"]] > 0.0
        )

    def _entity_cells(self, entity_ids: tuple[str, ...]) -> list[GridPos]:
        return [self.entity_to_cell[entity_id] for entity_id in entity_ids]

    def _region_cells(self, region_ids: tuple[str, ...]) -> list[GridPos]:
        cells: list[GridPos] = []
        for region_id in region_ids:
            cells.extend(self.region_cells[region_id])
        return cells


def _near_any(pos: GridPos, cells: list[GridPos], radius: int = 1) -> float:
    if not cells:
        return 0.0
    return float(any(_manhattan(tuple(pos), cell) <= radius for cell in cells))


def _distance_delta(before: GridPos, after: GridPos, targets: list[GridPos]) -> float:
    if not targets:
        return 0.0
    before_dist = min(_manhattan(tuple(before), target) for target in targets)
    after_dist = min(_manhattan(tuple(after), target) for target in targets)
    return float(np.clip(before_dist - after_dist, -1.0, 1.0))


def _option_progress_fraction(
    option_i: int,
    option_j: int,
    ego_option_id: int | None,
    ego_distance_delta: float,
) -> float:
    if ego_option_id is not None and int(ego_option_id) not in {
        int(option_i),
        int(option_j),
    }:
        return 0.0
    return float(np.clip(max(0.0, ego_distance_delta), 0.0, 1.0))


def _option_dist_value(dist: Any, option_id: int) -> float:
    if dist is None:
        return 0.0
    values = np.asarray(dist, dtype=np.float32).reshape(-1)
    if option_id < 0 or option_id >= values.size:
        return 0.0
    return float(values[option_id])


def _elapsed_fraction(elapsed: int | None, max_steps: int | None) -> float:
    if elapsed is None or max_steps is None or int(max_steps) <= 0:
        return 0.0
    return float(np.clip(float(elapsed) / float(max_steps), 0.0, 1.0))


def _changed_cell_touches_any(changed_cells: list[GridPos], cells: list[GridPos]) -> bool:
    if not changed_cells or not cells:
        return False
    return any(_manhattan(changed_cell, cell) <= 1 for changed_cell in changed_cells for cell in cells)


def _as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _manhattan(a: GridPos, b: GridPos) -> int:
    return int(abs(a[0] - b[0]) + abs(a[1] - b[1]))
