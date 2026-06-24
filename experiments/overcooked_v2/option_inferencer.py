from __future__ import annotations

from typing import Any

import numpy as np

from src.aris_bellman.specs import OptionSpec, PartnerAction

from .state_utils import get_agent_pos

GridPos = tuple[int, int]


class PartnerOptionInferencer:
    def __init__(self, option_library: Any, temperature: float = 1.0):
        self.option_library = option_library
        self.temperature = temperature
        self.belief: np.ndarray | None = None

    def reset(self, state: Any) -> None:
        valid = self.option_library.valid_options(state, agent_id=1)
        self.belief = _normalize(valid.astype(np.float32))

    def update(
        self,
        prev_state: Any,
        primitive_action: int,
        next_state: Any,
        event: Any,
    ) -> PartnerAction:
        # TODO: Replace this first-pass online likelihood filter with a learned
        # option classifier once scripted rollout labels are available.
        if self.belief is None or self.belief.shape[0] != self.option_library.num_options:
            self.reset(prev_state)

        likelihood = np.zeros_like(self.belief)
        for opt in self.option_library.options:
            if not self.option_library.is_valid_for_state(prev_state, 1, opt.id):
                likelihood[opt.id] = 0.0
                continue

            pred_action = self.option_library.primitive_action(prev_state, 1, opt.id)
            match = float(int(pred_action) == int(primitive_action))
            progress = _option_progress_score(
                opt,
                self.option_library,
                prev_state,
                next_state,
                agent_id=1,
            )
            terminated = self.option_library.option_terminated(
                opt,
                prev_state,
                next_state,
                event,
                agent_id=1,
                elapsed=1,
            )[0]
            likelihood[opt.id] = (
                0.65 * match + 0.25 * progress + 0.10 * float(terminated) + 1e-4
            )

        adjusted = np.power(likelihood, 1.0 / max(self.temperature, 1e-6))
        self.belief = _normalize(self.belief * adjusted)
        option_id = int(np.argmax(self.belief))
        return PartnerAction(
            primitive_action=int(primitive_action),
            option_id=option_id,
            option_confidence=float(np.max(self.belief)),
            option_dist=self.belief.copy(),
            source="inferred",
        )


def _option_progress_score(
    opt: OptionSpec,
    option_library: Any,
    prev_state: Any,
    next_state: Any,
    agent_id: int,
) -> float:
    targets = _target_cells(opt)
    if not targets:
        return 0.0

    prev_pos = get_agent_pos(prev_state, agent_id)
    next_pos = get_agent_pos(next_state, agent_id)
    prev_dist = _min_distance(option_library, prev_pos, targets)
    next_dist = _min_distance(option_library, next_pos, targets)
    if not np.isfinite(prev_dist) or not np.isfinite(next_dist):
        return 0.0
    if prev_dist <= 0:
        return 1.0
    return float(np.clip((prev_dist - next_dist) / max(prev_dist, 1.0), 0.0, 1.0))


def _target_cells(opt: OptionSpec) -> tuple[GridPos, ...]:
    metadata = opt.metadata or {}
    if "interaction_cells" in metadata:
        return tuple(metadata["interaction_cells"])
    if "region_cells" in metadata:
        return tuple(metadata["region_cells"])
    if opt.target_pos is not None:
        return (opt.target_pos,)
    return ()


def _min_distance(
    option_library: Any,
    pos: GridPos,
    targets: tuple[GridPos, ...],
) -> float:
    return float(
        min(
            (
                option_library.layout_graph.shortest_path_dist.get(
                    (pos, target),
                    float("inf"),
                )
                for target in targets
            ),
            default=float("inf"),
        )
    )


def _normalize(values: np.ndarray) -> np.ndarray:
    total = float(np.sum(values))
    if total <= 0.0:
        if values.size == 0:
            return values.astype(np.float32)
        return np.full_like(values, 1.0 / values.size, dtype=np.float32)
    return (values / total).astype(np.float32)
