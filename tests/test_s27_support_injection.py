"""S27 regression: the behavior inferencer must be able to infer options that only
BECOME valid mid-episode.

Pins the exact failure found on asymm x v2 (FINDINGS_LEDGER S27): reset() zeroes
belief for options invalid at t=0, and the multiplicative Bayes update can never
resurrect exact-zero mass — so plate/serve (invalid until the soup cooks) were
permanently uninferable even while a claim partner delivered 9x in one episode.

Uses a minimal stub option library (JaxMARL-free); the progress-score helper is
monkeypatched to a constant so states can be plain ints (phase 0 = terminal
option invalid, phase 1 = valid).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

try:
    from experiments.overcooked_v2 import option_inferencer as oi

    _OK = True
except Exception:  # pragma: no cover - import guard
    _OK = False


class _StubLib:
    """3 options; option 2 ('terminal') is invalid in phase 0, valid in phase 1."""

    def __init__(self):
        self.num_options = 3
        self.options = [SimpleNamespace(id=i, kind=f"k{i}") for i in range(3)]

    def valid_options(self, state, agent_id):
        phase = int(state)
        return np.array([1.0, 1.0, 1.0 if phase >= 1 else 0.0], dtype=np.float32)

    def primitive_action(self, state, agent_id, oid):
        return int(oid)  # option oid predicts primitive action oid

    def option_terminated(self, opt, prev, nxt, event, agent_id, elapsed, runtime):
        return (False, "running")


@pytest.fixture(autouse=True)
def _flat_progress(monkeypatch):
    if _OK:
        monkeypatch.setattr(oi, "_option_progress_score", lambda *a, **k: 0.0)


@pytest.mark.skipif(not _OK, reason="option_inferencer unavailable")
def test_mid_episode_valid_option_becomes_inferable():
    inf = oi.PartnerOptionInferencer(_StubLib(), allow_heuristic=True)  # default support_mix
    inf.reset(0)  # phase 0: option 2 invalid -> prior mass 0
    assert inf.belief[2] == 0.0
    # phase 1: option 2 valid; partner repeatedly emits action 2 (matches option 2)
    for _ in range(12):
        act = inf.update(prev_state=1, primitive_action=2, next_state=1, event=None)
    assert inf.belief[2] > 0.5, f"support injection failed: belief={inf.belief}"
    assert act.option_id == 2


@pytest.mark.skipif(not _OK, reason="option_inferencer unavailable")
def test_support_mix_zero_reproduces_freeze():
    """The pre-fix behavior stays reachable (ablation only): mix=0 keeps the freeze."""
    inf = oi.PartnerOptionInferencer(_StubLib(), allow_heuristic=True, support_mix=0.0)
    inf.reset(0)
    for _ in range(12):
        inf.update(prev_state=1, primitive_action=2, next_state=1, event=None)
    assert inf.belief[2] == 0.0  # frozen forever without injection


@pytest.mark.skipif(not _OK, reason="option_inferencer unavailable")
def test_injection_does_not_hijack_consistent_evidence():
    """With option 1 consistently observed, belief must still concentrate on 1."""
    inf = oi.PartnerOptionInferencer(_StubLib(), allow_heuristic=True)
    inf.reset(1)
    for _ in range(12):
        act = inf.update(prev_state=1, primitive_action=1, next_state=1, event=None)
    assert act.option_id == 1
    assert inf.belief[1] > 0.6


@pytest.mark.skipif(not _OK, reason="option_inferencer unavailable")
def test_support_mix_validation():
    with pytest.raises(ValueError):
        oi.PartnerOptionInferencer(_StubLib(), allow_heuristic=True, support_mix=1.0)
