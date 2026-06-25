from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

GridPos = tuple[int, int]

OptionKind = Literal[
    "fetch_ingredient",
    "deliver_ingredient_to_pot",
    "pick_plate",
    "plate_soup",
    "serve_soup",
    "press_recipe_button",
    "cross_bottleneck",
    "wait_at_bottleneck",
    "handoff_counter",
    "reroute",
    "noop",
]


@dataclass(frozen=True)
class OptionSpec:
    id: int
    name: str
    kind: str
    target_id: str | None
    target_pos: GridPos | None
    entity_ids: tuple[str, ...]
    region_ids: tuple[str, ...]
    max_steps: int
    interruptible: bool = True
    terminal_event: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class FactorSpec:
    id: int
    option_i: int
    option_j: int
    ce_score: float
    num_modes: int
    entity_ids: tuple[str, ...]
    region_ids: tuple[str, ...]
    factor_kind: str = "generic_option_pair"
    metadata: dict[str, Any] | None = None


@dataclass
class GraphSpec:
    layout_name: str
    options: list[OptionSpec]
    factors: list[FactorSpec]
    relevance: np.ndarray
    option_mask: np.ndarray
    factor_mask: np.ndarray
    mode_mask: np.ndarray
    route_map: dict[int, tuple[int, ...]]
    option_features: np.ndarray | None = None
    factor_features: np.ndarray | None = None
    metadata: dict[str, Any] | None = None

    @property
    def num_options(self) -> int:
        return len(self.options)

    @property
    def num_factors(self) -> int:
        return len(self.factors)

    @property
    def max_modes(self) -> int:
        return int(self.mode_mask.shape[1]) if self.mode_mask.size else 0

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "layout_name": self.layout_name,
            "options": [asdict(option) for option in self.options],
            "factors": [asdict(factor) for factor in self.factors],
            "relevance": self.relevance.astype(int).tolist(),
            "option_mask": self.option_mask.astype(int).tolist(),
            "factor_mask": self.factor_mask.astype(int).tolist(),
            "mode_mask": self.mode_mask.astype(int).tolist(),
            "route_map": {str(key): list(value) for key, value in self.route_map.items()},
            "option_features": (
                None if self.option_features is None else self.option_features.tolist()
            ),
            "factor_features": (
                None if self.factor_features is None else self.factor_features.tolist()
            ),
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "GraphSpec":
        options = [
            OptionSpec(
                id=int(item["id"]),
                name=str(item["name"]),
                kind=str(item["kind"]),
                target_id=item.get("target_id"),
                target_pos=_pos_or_none(item.get("target_pos")),
                entity_ids=tuple(item.get("entity_ids", ())),
                region_ids=tuple(item.get("region_ids", ())),
                max_steps=int(item["max_steps"]),
                interruptible=bool(item.get("interruptible", True)),
                terminal_event=item.get("terminal_event"),
                metadata=item.get("metadata"),
            )
            for item in data["options"]
        ]
        factors = [
            FactorSpec(
                id=int(item["id"]),
                option_i=int(item["option_i"]),
                option_j=int(item["option_j"]),
                ce_score=float(item["ce_score"]),
                num_modes=int(item["num_modes"]),
                entity_ids=tuple(item.get("entity_ids", ())),
                region_ids=tuple(item.get("region_ids", ())),
                factor_kind=str(item.get("factor_kind", "generic_option_pair")),
                metadata=item.get("metadata"),
            )
            for item in data["factors"]
        ]
        return cls(
            layout_name=str(data["layout_name"]),
            options=options,
            factors=factors,
            relevance=np.asarray(data["relevance"], dtype=bool),
            option_mask=np.asarray(data["option_mask"], dtype=bool),
            factor_mask=np.asarray(data["factor_mask"], dtype=bool),
            mode_mask=np.asarray(data["mode_mask"], dtype=bool),
            route_map={
                int(key): tuple(int(value) for value in values)
                for key, values in data.get("route_map", {}).items()
            },
            option_features=_array_or_none(data.get("option_features")),
            factor_features=_array_or_none(data.get("factor_features")),
            metadata=data.get("metadata") or {},
        )


@dataclass(frozen=True)
class PartnerAction:
    primitive_action: int
    option_id: int | None
    option_confidence: float
    option_dist: np.ndarray | None
    source: str


@dataclass
class OptionTransition:
    obs_feat_t: np.ndarray
    evidence_t: np.ndarray
    option_id: int
    reward_sum: float
    expected_cost: float
    realized_cost: float
    duration: int
    obs_feat_next: np.ndarray
    evidence_next: np.ndarray
    done: bool
    termination_reason: str
    graph_id: str | None = None
    partner_id: int | None = None
    event_summary: dict[str, Any] | None = None
    event_summary: dict[str, Any] | None = None


def _array_or_none(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=np.float32)


def _pos_or_none(value: Any) -> GridPos | None:
    if value is None:
        return None
    return int(value[0]), int(value[1])
