from __future__ import annotations

from typing import Any

import numpy as np

from src.aris_bellman.specs import OptionSpec, PartnerAction

from .partner_option_classifier import (
    PartnerOptionClassifier,
    classifier_action,
    load_partner_option_classifier,
)
from .state_utils import get_agent_pos

GridPos = tuple[int, int]


class PartnerOptionInferencer:
    def __init__(
        self,
        option_library: Any,
        temperature: float = 1.0,
        classifier_checkpoint: str | None = None,
        allow_heuristic: bool = False,
    ):
        self.option_library = option_library
        self.temperature = temperature
        self.belief: np.ndarray | None = None
        self.classifier: PartnerOptionClassifier | None = None
        self.allow_heuristic = bool(allow_heuristic)
        if classifier_checkpoint is not None:
            self.classifier = load_partner_option_classifier(classifier_checkpoint)
        elif not self.allow_heuristic:
            raise ValueError(
                "PartnerOptionInferencer requires classifier_checkpoint when "
                "allow_heuristic=False."
            )

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
        if self.classifier is not None:
            option_id, confidence, dist = classifier_action(self.classifier, event)
            self.belief = dist.copy()
            return PartnerAction(
                primitive_action=int(primitive_action),
                option_id=option_id,
                option_confidence=confidence,
                option_dist=dist,
                source="classifier",
            )

        num_options = int(
            getattr(self.option_library, "num_options", len(self.option_library.options))
        )
        if self.belief is None or self.belief.shape[0] != num_options:
            self.reset(prev_state)

        likelihood = np.zeros_like(self.belief)
        valid_mask = self.option_library.valid_options(prev_state, agent_id=1)
        for opt in self.option_library.options:
            is_valid_fn = getattr(self.option_library, "is_valid_for_state", None)
            if callable(is_valid_fn):
                is_valid = bool(is_valid_fn(prev_state, 1, opt.id))
            else:
                is_valid = bool(valid_mask[int(opt.id)])
            if not is_valid:
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
                runtime=None,
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
            source="heuristic",
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
    metadata = getattr(opt, "metadata", None) or {}
    if "interaction_cells" in metadata:
        return tuple(metadata["interaction_cells"])
    if "region_cells" in metadata:
        return tuple(metadata["region_cells"])
    if getattr(opt, "target_pos", None) is not None:
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

def make_behavior_option_inferencer(
    option_library: Any,
    config: dict[str, Any] | None = None,
) -> PartnerOptionInferencer:
    """Construct the single train/eval/CE partner-option evidence source.

    P1 boundary: this helper never receives a partner name, id, protocol, role,
    terminal policy, or scripted true option label. The default heuristic consumes
    only primitive partner action, state deltas, validity, and extracted event
    features inside :meth:`PartnerOptionInferencer.update`.
    """
    cfg = ((config or {}).get("evidence", {}) or {}).get("partner_option_inference", {}) or {}
    return PartnerOptionInferencer(
        option_library,
        temperature=float(cfg.get("temperature", 1.0)),
        classifier_checkpoint=cfg.get("classifier_checkpoint"),
        allow_heuristic=bool(cfg.get("allow_heuristic", True)),
    )



def build_behavior_option_inferencer(
    option_library: Any,
    config: dict[str, Any] | None = None,
) -> PartnerOptionInferencer | None:
    """Construct the main-path partner-option evidence source.

    P1: this inferencer may consume primitive partner actions, state deltas, and
    event booleans only. It must not consume partner name/id/protocol/role or
    terminal_policy. The heuristic path is enabled by default so de-oracling does
    not silently make the method untestable.
    """
    cfg = ((config or {}).get("partner_option_inference") or {})
    if not bool(cfg.get("enabled", True)):
        return None
    return PartnerOptionInferencer(
        option_library,
        temperature=float(cfg.get("temperature", 1.0)),
        classifier_checkpoint=cfg.get("classifier_checkpoint"),
        allow_heuristic=bool(cfg.get("allow_heuristic", True)),
    )


def attach_behavior_option_inferencer(
    partner: Any,
    option_library: Any,
    state: Any,
    config: dict[str, Any] | None = None,
) -> PartnerOptionInferencer | None:
    inferencer = build_behavior_option_inferencer(option_library, config)
    if inferencer is not None:
        inferencer.reset(state)
    setattr(partner, "_behavior_option_inferencer", inferencer)
    return inferencer
