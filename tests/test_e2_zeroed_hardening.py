"""E2 zeroed-channel hardening regressions (LDS-A1 + LDS-B3, FIX window-2).

LDS-A1: the router's zeroed branch must be ALL-OR-NOTHING — a "zeroed" event
still carrying an option id, a distribution, or nonzero confidence is a wiring
bug in the ablation and must fail loud (the old fall-through missed the
bare-option-id and nonzero-confidence cases entirely).

LDS-B3: the zeroed run must be launchable via an explicit eval-only CLI overlay
(_apply_zeroed_override) instead of hand-mutating checkpoint configs.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

try:
    from experiments.overcooked_v2.evidence_router import OCV2EvidenceRouter

    _ROUTER_OK = True
except Exception:  # pragma: no cover - import guard
    _ROUTER_OK = False

try:
    from experiments.overcooked_v2.evaluate_aris import (
        _apply_zeroed_override,
        _evidence_policy_for_config,
    )

    _EVAL_OK = True
except Exception:  # pragma: no cover - import guard
    _EVAL_OK = False


def _minimal_router(policy: str = "behavior_inferred_v1_zeroed_ablation"):
    # Duck-typed graph: _record_partner_option_evidence only needs the counter
    # dict and _current_event; __init__ touches options/factors/factor_mask.
    graph = SimpleNamespace(
        options=[], factors=[], factor_mask=np.zeros(0, dtype=bool), route_map={}
    )
    return OCV2EvidenceRouter(graph, {}, {}, evidence_policy=policy)


def _zeroed_event(**overrides):
    fields = dict(
        partner_option_source="zeroed_partner_option",
        partner_option_confidence=0.0,
        partner_option_dist=None,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.mark.skipif(not _ROUTER_OK, reason="evidence_router stack unavailable")
def test_clean_zeroed_event_counts_as_zeroed():
    router = _minimal_router()
    router._current_event = _zeroed_event()
    router._record_partner_option_evidence(None, None)
    counts = router._partner_option_evidence_counts
    assert counts["zeroed_count"] == 1
    assert counts["missing_count"] == 0
    assert counts["oracle_source_count"] == 0
    assert counts["observed_dist_count"] == 0


@pytest.mark.skipif(not _ROUTER_OK, reason="evidence_router stack unavailable")
def test_zeroed_with_option_id_fails_loud():
    router = _minimal_router()
    router._current_event = _zeroed_event()
    with pytest.raises(RuntimeError, match="all-or-nothing"):
        router._record_partner_option_evidence(None, 7)  # bare option id


@pytest.mark.skipif(not _ROUTER_OK, reason="evidence_router stack unavailable")
def test_zeroed_with_dist_fails_loud():
    router = _minimal_router()
    router._current_event = _zeroed_event()
    with pytest.raises(RuntimeError, match="all-or-nothing"):
        router._record_partner_option_evidence(np.ones(4) / 4.0, None)


@pytest.mark.skipif(not _ROUTER_OK, reason="evidence_router stack unavailable")
def test_zeroed_with_nonzero_confidence_fails_loud():
    """The case the old guard missed entirely (LDS-A1)."""
    router = _minimal_router()
    router._current_event = _zeroed_event(partner_option_confidence=0.8)
    with pytest.raises(RuntimeError, match="confidence=0.8"):
        router._record_partner_option_evidence(None, None)


@pytest.mark.skipif(not _EVAL_OK, reason="evaluate_aris stack unavailable")
def test_zeroed_override_sets_mode_and_policy():
    config = {"evidence": {"partner_option_inference": {"mode": "inferred",
                                                        "support_mix": 0.05}}}
    _apply_zeroed_override(config)
    assert config["evidence"]["partner_option_inference"]["mode"] == "zeroed"
    # other inference settings untouched
    assert config["evidence"]["partner_option_inference"]["support_mix"] == 0.05
    assert _evidence_policy_for_config(config) == "behavior_inferred_v1_zeroed_ablation"


@pytest.mark.skipif(not _EVAL_OK, reason="evaluate_aris stack unavailable")
def test_zeroed_override_on_empty_config():
    config: dict = {}
    _apply_zeroed_override(config)
    assert _evidence_policy_for_config(config) == "behavior_inferred_v1_zeroed_ablation"
