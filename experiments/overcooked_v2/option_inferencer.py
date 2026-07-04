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
        mode: str = "inferred",
        support_mix: float = 0.05,
    ):
        self.option_library = option_library
        self.temperature = temperature
        self.belief: np.ndarray | None = None
        self.classifier: PartnerOptionClassifier | None = None
        self.allow_heuristic = bool(allow_heuristic)
        # S27: per-step mixing weight of the uniform-over-currently-valid support
        # floor into the belief prior. 0 disables (reproduces the frozen-support
        # pre-fix behavior — kept reachable for ablation only).
        self.support_mix = float(support_mix)
        if not (0.0 <= self.support_mix < 1.0):
            raise ValueError(f"support_mix must be in [0,1), got {support_mix!r}.")
        # E2 ablation (EXPERIMENT_CHAIN_PLAN §10.4, METHOD_LOCK sec18.7): "zeroed"
        # emits no option-level intent, only the observable primitive action, so the
        # partner_option_* evidence channels go neutral. Default "inferred" is the
        # normal behavior-inference path. This is an EVAL-ONLY ablation knob.
        self.mode = str(mode)
        if self.mode not in {"inferred", "zeroed"}:
            raise ValueError(
                f"PartnerOptionInferencer.mode must be 'inferred' or 'zeroed', got {mode!r}."
            )
        if self.mode == "zeroed":
            # A zeroed inferencer never consults a classifier/heuristic, so skip the
            # (potentially checkpoint-requiring) construction path entirely.
            return
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
        if self.mode == "zeroed":
            # E2: keep the observable primitive action; withhold all option-level
            # intent. option_id=None / option_dist=None make the router's
            # partner_option_* channels resolve to 0.0 (same neutral state as the
            # oracle-stripped path), and `source` tags it for the eval integrity gate.
            return PartnerAction(
                primitive_action=int(primitive_action),
                option_id=None,
                option_confidence=0.0,
                option_dist=None,
                source="zeroed_partner_option",
            )
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

        valid_mask = self.option_library.valid_options(prev_state, agent_id=1)
        # S27 fix (FINDINGS_LEDGER): support injection. The multiplicative Bayes
        # update below can never resurrect an option whose belief mass is exactly 0,
        # and reset() zeroes every option that is invalid at t=0 — so options that
        # only BECOME valid mid-episode (plate/serve once the soup cooks) were
        # permanently uninferable (live evidence: a claim partner delivered 9x in
        # one episode while inferred terminal mass stayed 0.0). Mix a small
        # uniform-over-currently-valid floor into the prior each step (standard
        # forgetting-factor filtering). Uses only valid_options(state) — public,
        # behavior-observable information; the P1 oracle boundary is untouched.
        support_floor = _normalize(np.asarray(valid_mask, dtype=np.float32))
        mix = float(self.support_mix)
        if mix > 0.0:
            self.belief = _normalize((1.0 - mix) * self.belief + mix * support_floor)

        likelihood = np.zeros_like(self.belief)
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
    *,
    require_inferred: bool = False,
) -> PartnerOptionInferencer:
    """Construct the single train/eval/CE partner-option evidence source.

    P1 boundary: this helper never receives a partner name, id, protocol, role,
    terminal policy, or scripted true option label. The default heuristic consumes
    only primitive partner action, state deltas, validity, and extracted event
    features inside :meth:`PartnerOptionInferencer.update`.

    E2 (review BLOCK fix): the ``zeroed`` ablation is EVAL-ONLY. Callers on the
    training/CE path pass ``require_inferred=True`` so that a config which sets
    ``mode: zeroed`` FAILS LOUDLY here rather than silently zeroing the partner-option
    channels during training (a silent experiment-condition override).
    """
    cfg = ((config or {}).get("evidence", {}) or {}).get("partner_option_inference", {}) or {}
    mode = str(cfg.get("mode", "inferred"))
    if require_inferred and mode != "inferred":
        raise ValueError(
            "partner_option_inference.mode must be 'inferred' on the training/CE path; "
            f"the {mode!r} ablation is eval-only (E2, METHOD_LOCK sec18.7)."
        )
    return PartnerOptionInferencer(
        option_library,
        temperature=float(cfg.get("temperature", 1.0)),
        classifier_checkpoint=cfg.get("classifier_checkpoint"),
        allow_heuristic=bool(cfg.get("allow_heuristic", True)),
        mode=mode,
        support_mix=float(cfg.get("support_mix", 0.05)),
    )
