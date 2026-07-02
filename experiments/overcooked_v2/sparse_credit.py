"""Actor-specific sparse-reward credit (RC root-cause fix) — single source of truth.

This module is intentionally dependency-light (NO JaxMARL / numpy import) so the
credit logic can be unit-tested in the static-maintenance environment and reused
by train_aris / ce_sampler / evaluate_aris without pulling the env stack.

RC root cause: train_aris / ce_sampler / evaluate_aris credited the EGO option's
TD target with step.rewards["agent_0"], which in OvercookedV2 is the SHARED team
delivery reward (identical for both agents). The ego was therefore paid the
+delivery reward when the PARTNER served, making "fetch/deliver/wait while the
partner plates and serves" a high-return ego option. The helpers below are the
ONE place that decides how a per-step team sparse reward is attributed to the ego;
train/CE/eval all route through them so the graph objective, the training target,
and reported returns stay consistent.
"""
from __future__ import annotations

from typing import Any

SPARSE_CREDIT_MODES = (
    "team",
    "ego_delivery",
    "ego_correct_delivery",
    "contrib_team",
    "role_contrib_team",
)
DEFAULT_SPARSE_CREDIT_MODE = "team"


def actor_sparse_reward(
    team_sparse: float,
    event: Any,
    *,
    mode: str = DEFAULT_SPARSE_CREDIT_MODE,
    ego_delivery_reward: float = 20.0,
    ego_wrong_delivery_penalty: float = -20.0,
    ego_contributed: bool = False,
    contrib_scale: float = 1.0,
    partner_terminal_policy: str | None = None,
    ego_terminal_penalty_under_claim: float = 0.3,
) -> float:
    """Attribute one step's *sparse* (delivery) reward to the ego option.

    ``team_sparse`` is the raw per-step ``step.rewards["agent_0"]`` value, which
    in OvercookedV2 is the SHARED team delivery reward. ``event`` is the
    extracted ``OCV2Event`` for that step (only its boolean delivery flags are
    read, via ``getattr``).

    Modes:
      * ``"team"``     — legacy: return the shared team sparse reward unchanged.
                         Reproduces pre-fix runs; this IS the RC root-cause
                         behaviour and must not be used for a train/held-out
                         split where train partners can finish for the ego.
      * ``"ego_delivery"`` — recommended: credit the *real* env sparse reward to
                         the ego ONLY on steps where the EGO is the SOLE deliverer
                         (``event.ego_delivery_event and not event.partner_delivery_event``).
                         A correct ego delivery keeps its +reward, a wrong ego
                         delivery keeps its -penalty (under ``env.negative_rewards``);
                         a partner delivery yields 0.0. No inferred reward constant —
                         the magnitude is the environment's own, so it stays on the
                         same scale as shaping and ``value_bound.vmax``. The
                         ``not partner_delivery_event`` term matters on layouts with
                         multiple goal tiles (e.g. asymm_advantages): the env sparse
                         reward is a single SHARED scalar summed over BOTH agents, so
                         on a coincident ego+partner delivery the raw ``team_sparse``
                         includes the partner's contribution. Crediting it only when
                         the ego delivers ALONE keeps the partner's reward out of the
                         ego target; the rare coincident step is (conservatively)
                         credited 0 rather than re-leaking it.
      * ``"ego_correct_delivery"`` — the minimal-fix-plan literal: explicit
                         ego-owned constants, independent of the env reward table.
                         ``+ego_delivery_reward`` on ``event.ego_correct_delivery``
                         and ``+ego_wrong_delivery_penalty`` on
                         ``event.ego_wrong_delivery_event``, both suppressed on a
                         coincident partner delivery (``correct_delivery`` is a
                         shared bool, so a coincident partner-correct step would
                         otherwise mislabel a wrong ego delivery as correct).
    """
    if mode == "team":
        return float(team_sparse)
    # On layouts with >1 goal tile the env sparse reward is shared and summed over
    # both agents, so credit the ego only when it is the SOLE deliverer this step.
    ego_sole_delivery = bool(getattr(event, "ego_delivery_event", False)) and not bool(
        getattr(event, "partner_delivery_event", False)
    )
    if mode == "ego_delivery":
        return float(team_sparse) if ego_sole_delivery else 0.0
    if mode == "ego_correct_delivery":
        if not ego_sole_delivery:
            return 0.0
        sparse = 0.0
        if bool(getattr(event, "ego_correct_delivery", False)):
            sparse += float(ego_delivery_reward)
        if bool(getattr(event, "ego_wrong_delivery_event", False)):
            sparse += float(ego_wrong_delivery_penalty)
        return sparse
    if mode == "contrib_team":
        ego_sole_delivery = bool(getattr(event, "ego_delivery_event", False)) and not bool(
            getattr(event, "partner_delivery_event", False)
        )
        partner_delivery = bool(getattr(event, "partner_delivery_event", False))
        if ego_sole_delivery:
            return float(team_sparse)
        if partner_delivery and bool(ego_contributed):
            return float(contrib_scale) * float(team_sparse)
        return 0.0
    if mode == "role_contrib_team":
        ego_sole_delivery = bool(getattr(event, "ego_delivery_event", False)) and not bool(
            getattr(event, "partner_delivery_event", False)
        )
        partner_delivery = bool(getattr(event, "partner_delivery_event", False))
        # Yield partner: ego should serve. Reward ego terminal, zero partner terminal.
        if partner_terminal_policy == "yield":
            if ego_sole_delivery:
                return float(team_sparse)
            return 0.0
        # Claim partner: partner should serve. Scale ego terminal down, keep
        # partner terminal credit when the ego contributed to the dish cycle.
        if partner_terminal_policy == "claim":
            if partner_delivery and bool(ego_contributed):
                return float(contrib_scale) * float(team_sparse)
            if ego_sole_delivery:
                return float(ego_terminal_penalty_under_claim) * float(team_sparse)
            return 0.0
        # Unknown policy keeps contrib_team semantics for backward compatibility.
        if ego_sole_delivery:
            return float(team_sparse)
        if partner_delivery and bool(ego_contributed):
            return float(contrib_scale) * float(team_sparse)
        return 0.0
    raise ValueError(
        f"unknown sparse_credit mode {mode!r}; expected one of {SPARSE_CREDIT_MODES}"
    )


def sparse_credit_params(training_cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Resolve :func:`actor_sparse_reward` kwargs from a config ``training`` block.

    A missing ``sparse_credit`` key resolves to legacy ``"team"`` so pre-fix
    configs and CE graphs keep their exact behaviour (no silent change).
    """
    cfg = training_cfg or {}
    mode = str(cfg.get("sparse_credit", DEFAULT_SPARSE_CREDIT_MODE))
    if mode not in SPARSE_CREDIT_MODES:
        raise ValueError(
            f"training.sparse_credit={mode!r} invalid; expected one of {SPARSE_CREDIT_MODES}"
        )
    return {
        "mode": mode,
        "ego_delivery_reward": float(cfg.get("ego_delivery_reward", 20.0)),
        "ego_wrong_delivery_penalty": float(cfg.get("ego_wrong_delivery_penalty", -20.0)),
        "contrib_scale": float(
            (cfg.get("contrib_team") or {}).get("contrib_scale", 1.0)
        ),
        "partner_terminal_policy": None,
        "ego_terminal_penalty_under_claim": float(
            (cfg.get("role_contrib_team") or {}).get(
                "ego_terminal_penalty_under_claim", 0.3
            )
        ),
    }
