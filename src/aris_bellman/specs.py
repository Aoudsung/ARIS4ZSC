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
