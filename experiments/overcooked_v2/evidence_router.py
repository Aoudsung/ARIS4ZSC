from __future__ import annotations

from typing import Any

import numpy as np

from src.aris_bellman.specs import GraphSpec

GridPos = tuple[int, int]

D_EVID = 64


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
        self._validate_factor_refs()

    def route(self, event: Any, ego_option_id: int | None = None) -> np.ndarray:
        routed = np.zeros((self.graph.num_factors, D_EVID), dtype=np.float32)
        for factor_idx, factor in enumerate(self.graph.factors):
            entity_cells = self._entity_cells(factor.entity_ids)
            region = self._region_cells(factor.region_ids)
            targets = entity_cells + region

            routed[factor_idx, 0] = _near_any(event.ego_pos_after, entity_cells)
            routed[factor_idx, 1] = _near_any(event.partner_pos_after, entity_cells)
            routed[factor_idx, 2] = _distance_delta(
                event.ego_pos_before,
                event.ego_pos_after,
                targets,
            )
            routed[factor_idx, 3] = _distance_delta(
                event.partner_pos_before,
                event.partner_pos_after,
                targets,
            )
            routed[factor_idx, 4] = float(
                bool(event.ego_interacted) and _near_any(event.ego_pos_after, targets)
            )
            routed[factor_idx, 5] = float(
                bool(event.partner_interacted)
                and _near_any(event.partner_pos_after, targets)
            )
            routed[factor_idx, 6] = float(
                bool(event.collision_or_block)
                and (
                    _near_any(event.ego_pos_after, region)
                    or _near_any(event.partner_pos_after, region)
                )
            )
            routed[factor_idx, 7] = float(
                int(event.ego_inventory_before) != int(event.ego_inventory_after)
            )
            routed[factor_idx, 8] = float(
                int(event.partner_inventory_before)
                != int(event.partner_inventory_after)
            )
            routed[factor_idx, 9] = _option_progress_fraction(
                factor.option_i,
                factor.option_j,
                ego_option_id,
                routed[factor_idx, 2],
            )
        return routed

    def _validate_factor_refs(self) -> None:
        for factor in self.graph.factors:
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


def _manhattan(a: GridPos, b: GridPos) -> int:
    return int(abs(a[0] - b[0]) + abs(a[1] - b[1]))
