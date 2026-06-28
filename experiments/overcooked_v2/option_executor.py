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

from dataclasses import dataclass
from typing import Any

from experiments.overcooked_v2.event_extractor import extract_event


@dataclass
class OptionStep:
    """Outcome of one primitive tick of an option."""

    ego_action: int
    partner_action: Any  # PartnerAction: .primitive_action, .option_id, .option_dist, .option_confidence
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
) -> OptionStep:
    """Advance the ego's current option by one primitive step against the live partner.

    Phase 0: identical calls / order / RNG draws as the three legacy loops (trace-equivalent).
    Conflict-resolution / yield hooks will be inserted here in later phases, strictly above
    `env.step` and within the scope boundary documented in this module.
    """
    ego_action = option_lib.primitive_action(env.state, agent_id, int(option_id))
    partner_obs = obs.get("agent_1") if isinstance(obs, dict) else None
    partner_action = partner.act(partner_obs, env.state, rng)
    prev_state = env.state
    step = env.step(ego_action, partner_action.primitive_action)
    event = extract_event(
        prev_state,
        ego_action,
        partner_action.primitive_action,
        step.state,
        step.info,
        partner_action.option_id,
        partner_action.option_dist,
    )
    return OptionStep(
        ego_action=ego_action,
        partner_action=partner_action,
        prev_state=prev_state,
        step=step,
        event=event,
    )
