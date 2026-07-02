"""RC root-cause regression: ego must NOT be credited for the partner's delivery.

This pins the exact failure mode found in the asymm_ce replay: a step where the
PARTNER delivers (``partner_correct_delivery=1``, ``ego_correct_delivery=0``) but
the shared team reward ``step.rewards["agent_0"]`` is +20. Under the legacy
``team`` credit that +20 leaked into the ego option's return (free-riding); under
the fixed ego-credit modes it must be 0.

The credit helper lives in the dependency-light ``sparse_credit`` module (no
JaxMARL import), so this test imports it directly and RUNS in the static
environment — it does not need the env stack and does not execute the project.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from experiments.overcooked_v2.sparse_credit import (
    actor_sparse_reward,
    sparse_credit_params,
)


def _event(
    *,
    ego_delivery_event=False,
    ego_correct_delivery=False,
    ego_wrong_delivery_event=False,
    partner_delivery_event=False,
    partner_correct_delivery=False,
):
    """Minimal duck-typed OCV2Event carrying only the fields the helper reads."""
    return SimpleNamespace(
        ego_delivery_event=ego_delivery_event,
        ego_correct_delivery=ego_correct_delivery,
        ego_wrong_delivery_event=ego_wrong_delivery_event,
        partner_delivery_event=partner_delivery_event,
        partner_correct_delivery=partner_correct_delivery,
    )


# --- The decisive failure-mode fixture (asymm_ce replay rows[6], rows[245]) ----
PARTNER_DELIVERS = _event(partner_delivery_event=True, partner_correct_delivery=True)
EGO_DELIVERS = _event(ego_delivery_event=True, ego_correct_delivery=True)
EGO_WRONG_DELIVERS = _event(ego_delivery_event=True, ego_wrong_delivery_event=True)
# Coincident deliveries: asymm_advantages has TWO goal tiles, so both agents can
# deliver in the same env step and step.rewards["agent_0"] is the SUMMED shared
# reward. ego_correct_delivery can be spuriously True here because correct_delivery
# is the env's shared OR-bool — the helper must still not credit the partner share.
EGO_AND_PARTNER_DELIVER = _event(
    ego_delivery_event=True,
    ego_correct_delivery=True,
    partner_delivery_event=True,
    partner_correct_delivery=True,
)
EGO_WRONG_PARTNER_CORRECT = _event(
    ego_delivery_event=True,
    ego_wrong_delivery_event=True,
    ego_correct_delivery=True,  # shared correct_delivery bool spuriously sets this
    partner_delivery_event=True,
    partner_correct_delivery=True,
)
TEAM_DELIVERY_REWARD = 20.0


def test_team_mode_is_legacy_passthrough():
    # Legacy behaviour: ego is paid the shared team reward even on a partner serve
    # (this IS the root-cause bug, preserved only for reproducing pre-fix runs).
    assert actor_sparse_reward(TEAM_DELIVERY_REWARD, PARTNER_DELIVERS, mode="team") == 20.0
    assert actor_sparse_reward(TEAM_DELIVERY_REWARD, EGO_DELIVERS, mode="team") == 20.0
    assert actor_sparse_reward(0.0, PARTNER_DELIVERS, mode="team") == 0.0


def test_ego_delivery_zeroes_partner_serves():
    # The fix: a partner delivery contributes ZERO to the ego option return,
    # even though the shared team reward is +20.
    assert actor_sparse_reward(TEAM_DELIVERY_REWARD, PARTNER_DELIVERS, mode="ego_delivery") == 0.0
    # An ego delivery keeps the real env reward magnitude (no inferred constant).
    assert actor_sparse_reward(TEAM_DELIVERY_REWARD, EGO_DELIVERS, mode="ego_delivery") == 20.0
    # An ego WRONG delivery keeps its negative penalty (env.negative_rewards).
    assert actor_sparse_reward(-20.0, EGO_WRONG_DELIVERS, mode="ego_delivery") == -20.0
    # No delivery at all -> no sparse credit.
    assert actor_sparse_reward(0.0, _event(), mode="ego_delivery") == 0.0


def test_ego_correct_delivery_uses_explicit_constants():
    # Plan-literal mode: explicit ego-owned constants, independent of team reward.
    assert (
        actor_sparse_reward(TEAM_DELIVERY_REWARD, PARTNER_DELIVERS, mode="ego_correct_delivery")
        == 0.0
    )
    assert (
        actor_sparse_reward(0.0, EGO_DELIVERS, mode="ego_correct_delivery", ego_delivery_reward=20.0)
        == 20.0
    )
    assert (
        actor_sparse_reward(
            0.0,
            EGO_WRONG_DELIVERS,
            mode="ego_correct_delivery",
            ego_wrong_delivery_penalty=-20.0,
        )
        == -20.0
    )


def test_sparse_credit_params_defaults_to_team():
    # Absent key -> legacy team (no silent objective change for pre-fix configs).
    assert sparse_credit_params(None)["mode"] == "team"
    assert sparse_credit_params({})["mode"] == "team"
    assert sparse_credit_params({"sparse_credit": "ego_delivery"})["mode"] == "ego_delivery"


def test_sparse_credit_params_rejects_unknown_mode():
    with pytest.raises(ValueError):
        sparse_credit_params({"sparse_credit": "free_ride"})
    with pytest.raises(ValueError):
        actor_sparse_reward(20.0, EGO_DELIVERS, mode="nonsense")


def test_decisive_falsifier_partner_serve_gives_ego_zero():
    # The plan's decisive falsifier would have been a row where partner serves,
    # ego does not, and the ego option return still gets +20. Assert the fix
    # removes exactly that, across BOTH ego-credit modes.
    for mode in ("ego_delivery", "ego_correct_delivery"):
        assert actor_sparse_reward(TEAM_DELIVERY_REWARD, PARTNER_DELIVERS, mode=mode) == 0.0


def test_coincident_delivery_does_not_leak_partner_reward():
    # Both agents deliver in the same env step => step.rewards["agent_0"] is the
    # SUMMED shared reward. The ego must not absorb the partner's share even though
    # the ego also delivered (the sole-deliverer gate). Conservatively credits 0.
    for mode in ("ego_delivery", "ego_correct_delivery"):
        # both correct: raw team reward would be +40 against vmax=20 — must be 0.
        assert actor_sparse_reward(40.0, EGO_AND_PARTNER_DELIVER, mode=mode) == 0.0
        # ego wrong + partner correct (team nets to 0): ego must not be paid the
        # partner's +20, and ego_correct_delivery's shared-bool mislabel is suppressed.
        assert actor_sparse_reward(0.0, EGO_WRONG_PARTNER_CORRECT, mode=mode) == 0.0


def test_event_extractor_reexports_same_helper():
    # Single-source guarantee: event_extractor re-exports the SAME function objects,
    # so the legacy `from .event_extractor import actor_sparse_reward` call sites in
    # train_aris / ce_sampler cannot diverge from the module under test. Needs JaxMARL
    # (event_extractor imports the env stack), so skip cleanly where it is absent.
    ee = pytest.importorskip("experiments.overcooked_v2.event_extractor")
    assert ee.actor_sparse_reward is actor_sparse_reward
    assert ee.sparse_credit_params is sparse_credit_params
