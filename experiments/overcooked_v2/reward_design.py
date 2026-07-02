from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .state_utils import is_empty_inventory, is_plain_plate, is_plated_cooked_soup


DEFAULT_TERMINAL_PROGRESS_SHAPING: dict[str, Any] = {
    "enabled": False,
    "ego_plate_pick_bonus": 0.0,
    "ego_plate_soup_bonus": 0.0,
    "ego_serve_bonus": 0.0,
    "max_bonus_per_step": None,
}


def terminal_progress_params(training_cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Resolve terminal-stage progression shaping config.

    This is a research intervention, not the root-cause reward-credit fix. It is
    deliberately actor-local: only ego inventory transitions and ego sole delivery
    can receive bonus. Partner plate/serve events are never credited to the ego.
    """
    cfg = ((training_cfg or {}).get("terminal_progress_shaping") or {})
    enabled = bool(cfg.get("enabled", DEFAULT_TERMINAL_PROGRESS_SHAPING["enabled"]))
    max_bonus = cfg.get("max_bonus_per_step", DEFAULT_TERMINAL_PROGRESS_SHAPING["max_bonus_per_step"])
    return {
        "enabled": enabled,
        "ego_plate_pick_bonus": float(
            cfg.get("ego_plate_pick_bonus", DEFAULT_TERMINAL_PROGRESS_SHAPING["ego_plate_pick_bonus"])
        ),
        "ego_plate_soup_bonus": float(
            cfg.get("ego_plate_soup_bonus", DEFAULT_TERMINAL_PROGRESS_SHAPING["ego_plate_soup_bonus"])
        ),
        "ego_serve_bonus": float(
            cfg.get("ego_serve_bonus", DEFAULT_TERMINAL_PROGRESS_SHAPING["ego_serve_bonus"])
        ),
        "max_bonus_per_step": None if max_bonus is None else float(max_bonus),
    }


def terminal_progress_bonus(event: Any, *, params: dict[str, Any] | None = None) -> float:
    """Actor-local progression bonus for the ego's terminal-stage chain.

    Bonuses are granted only for unambiguous ego state transitions:
      empty -> plain plate             (pick_plate)
      plain plate -> plated cooked soup (plate_soup / soup pickup)
      plated cooked soup -> empty + ego_sole_correct_delivery (serve)

    This avoids reintroducing the original shared-credit bug: partner delivery,
    partner plating, and coincident partner delivery are not rewarded here.
    """
    cfg = params or DEFAULT_TERMINAL_PROGRESS_SHAPING
    if not bool(cfg.get("enabled", False)):
        return 0.0

    before = int(getattr(event, "ego_inventory_before", 0))
    after = int(getattr(event, "ego_inventory_after", 0))
    bonus = 0.0

    if is_empty_inventory(before) and is_plain_plate(after):
        bonus += float(cfg.get("ego_plate_pick_bonus", 0.0))
    if is_plain_plate(before) and is_plated_cooked_soup(after):
        bonus += float(cfg.get("ego_plate_soup_bonus", 0.0))
    if (
        is_plated_cooked_soup(before)
        and is_empty_inventory(after)
        and bool(getattr(event, "ego_sole_correct_delivery", False))
    ):
        bonus += float(cfg.get("ego_serve_bonus", 0.0))

    cap = cfg.get("max_bonus_per_step")
    if cap is not None:
        bonus = min(float(bonus), float(cap))
    return float(bonus)


CONTRIB_OPTION_KINDS: frozenset[str] = frozenset({
    "deliver_ingredient_to_pot",
    "clear_interaction_cell",
    "drop_item_to_counter",
    "fetch_ingredient",
})


@dataclass
class ContributionLedger:
    """Delivery-delimited ledger of ego contribution to the current dish cycle.

    Set on any support-kind option that produced a visible env effect (pot state
    changed OR object pickup/drop). Reset when a delivery event fires (natural
    dish-cycle boundary — no time hyperparameter).
    """
    ego_contributed: bool = False

    @classmethod
    def from_config(cls, training_cfg: dict[str, Any] | None) -> "ContributionLedger":
        return cls()

    def reset(self) -> None:
        self.ego_contributed = False

    def update(self, event: Any, ego_option_kind: str) -> None:
        if str(ego_option_kind) not in CONTRIB_OPTION_KINDS:
            return
        visible_effect = bool(getattr(event, "pot_changed", False)) or bool(
            getattr(event, "object_pickup_or_drop", False)
        )
        if visible_effect:
            self.ego_contributed = True

    def query_and_reset_on_delivery(self, event: Any) -> bool:
        """Return current contribution flag; reset if this step was a delivery.
        Call this AT the reward-computation site so the value used for reward
        matches the ledger state accumulated up to (and including) the delivery."""
        result = bool(self.ego_contributed)
        if bool(getattr(event, "delivery_event", False)):
            self.ego_contributed = False
        return result
