"""Shared two-agent option-execution substrate (RC-2b).

Phase 0 — behavior-preserving extraction of the per-primitive-step core that was duplicated across
train (`train_aris._execute_option`), eval (`evaluate_aris._execute_eval_option`), and CE collection
(`ce_sampler._rollout_option`). All three computed the ego primitive action, queried the partner,
stepped the env, and extracted the event identically; this module is the single place that does so.

Later phases add reservation / mechanical-conflict / stand-lock / yield logic INSIDE
`option_primitive_step` (above `env.step`, because collision resolution lives in the JaxMARL env and
cannot be edited). One shared core guarantees train/eval/CE share identical execution semantics (so CE
stays consistent with training) and lets executor behavior be reported and hashed in one place.

Scope boundary (plan B0-B6): this layer is substrate-level and symmetric. It must never read the
method's signals (Q / factor beliefs / Δ_info / MI / partner identity / oracle factor modes), must
never choose high-level options, and must never silently override a black-box partner's action.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from experiments.overcooked_v2.event_extractor import extract_event
from experiments.overcooked_v2.option_inferencer import PartnerOptionInferencer
from src.aris_bellman.specs import PartnerAction


@dataclass
class OptionStep:
    """Outcome of one primitive tick of an option."""

    ego_action: int
    partner_action: Any  # Sanitized PartnerAction: primitive only; no true option labels.
    diagnostic_partner_action: Any  # Raw partner action. Diagnostic-only, never routed to evidence.
    prev_state: Any
    step: Any  # OCV2Step
    event: Any  # OCV2Event


def option_primitive_step(
    env: Any,
    option_lib: Any,
    option_id: int,
    partner: Any,
    obs: Any,
    rng: Any,
    *,
    agent_id: int = 0,
    partner_option_inferencer: PartnerOptionInferencer | None = None,
) -> OptionStep:
    """Advance the ego's current option by one primitive step against the live partner.

    Phase 0: identical calls / order / RNG draws as the three legacy loops (trace-equivalent).
    Conflict-resolution / yield hooks will be inserted here in later phases, strictly above
    `env.step` and within the scope boundary documented in this module.
    """
    ego_action = option_lib.primitive_action(env.state, agent_id, int(option_id))
    partner_obs = obs.get("agent_1") if isinstance(obs, dict) else None
    raw_partner_action = partner.act(partner_obs, env.state, rng)
    partner_action = PartnerAction(
        primitive_action=int(raw_partner_action.primitive_action),
        option_id=None,
        option_confidence=0.0,
        option_dist=None,
        source="behavior_observed:oracle_stripped",
    )
    prev_state = env.state
    step = env.step(ego_action, partner_action.primitive_action)
    # P1: the main event/evidence path must not receive ScriptedProtocolPartner's
    # true option_id / option_dist / confidence. The event is first extracted with
    # no partner-option labels, then optionally annotated by a behavior-only
    # inferencer that consumes primitive actions and state/event deltas.
    event = extract_event(
        prev_state,
        ego_action,
        partner_action.primitive_action,
        step.state,
        step.info,
        partner_option=None,
        partner_option_dist=None,
        partner_option_source="none",
    )
    if partner_option_inferencer is None:
        partner_option_inferencer = getattr(partner, "_behavior_option_inferencer", None)
    if partner_option_inferencer is None:
        partner_option_inferencer = PartnerOptionInferencer(option_lib, allow_heuristic=True)
        partner_option_inferencer.reset(prev_state)
        setattr(partner, "_behavior_option_inferencer", partner_option_inferencer)
    inferred = partner_option_inferencer.update(
        prev_state,
        int(partner_action.primitive_action),
        step.state,
        event,
    )
    event = replace(
        event,
        partner_option=inferred.option_id,
        partner_option_dist=inferred.option_dist,
        partner_option_confidence=float(inferred.option_confidence),
        partner_option_source=str(inferred.source),
    )
    return OptionStep(
        ego_action=ego_action,
        partner_action=partner_action,
        diagnostic_partner_action=raw_partner_action,
        prev_state=prev_state,
        step=step,
        event=event,
    )
