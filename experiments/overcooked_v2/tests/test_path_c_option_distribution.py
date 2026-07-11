from __future__ import annotations

import copy

import numpy as np
import pytest


pytest.importorskip("jaxmarl")

from experiments.overcooked_v2 import partner_pool
from experiments.overcooked_v2.partner_pool import (
    ProtocolSpec,
    ScriptedProtocolPartner,
    option_distribution,
)
from experiments.overcooked_v2.option_termination import OptionRuntime
from src.aris_bellman.specs import OptionSpec


class _FakeOptionLibrary:
    def __init__(self) -> None:
        self.options = [
            OptionSpec(0, "noop", "noop", None, None, (), (), 1),
            OptionSpec(1, "candidate-a", "reroute", None, None, (), (), 1),
            OptionSpec(2, "candidate-b", "reroute", None, None, (), (), 1),
        ]


def _patch_scores(monkeypatch, scores: dict[int, float]) -> None:
    def fake_score(_protocol, option, _state, **_kwargs):
        return float(scores[int(option.id)])

    monkeypatch.setattr(partner_pool, "_protocol_score_for", fake_score)


def test_distribution_respects_valid_support_and_epsilon_mixture(monkeypatch):
    library = _FakeOptionLibrary()
    protocol = ProtocolSpec(role="flexible")
    _patch_scores(monkeypatch, {0: 100.0, 1: 5.0, 2: 1.0})
    runtime = {
        "elapsed": 3,
        "bottleneck_alternate_phase": 1,
        "epsilon": 0.2,
        "valid_options": np.asarray([False, True, True]),
    }
    runtime_before = copy.deepcopy(runtime)
    probabilities = option_distribution(
        protocol,
        runtime,
        public_state={"observable": "state"},
        option_library=library,
    )
    assert probabilities == pytest.approx(np.asarray([0.0, 0.9, 0.1]))
    assert probabilities.sum() == pytest.approx(1.0)
    assert runtime["elapsed"] == runtime_before["elapsed"]
    assert runtime["bottleneck_alternate_phase"] == (
        runtime_before["bottleneck_alternate_phase"]
    )
    assert np.array_equal(
        runtime["valid_options"], runtime_before["valid_options"]
    )


def test_argmax_ties_are_uniform_over_all_best_valid_options(monkeypatch):
    library = _FakeOptionLibrary()
    _patch_scores(monkeypatch, {0: -10.0, 1: 4.0, 2: 4.0})
    probabilities = option_distribution(
        ProtocolSpec(role="flexible"),
        {
            "epsilon": 0.0,
            "valid_options": np.asarray([False, True, True]),
        },
        public_state=None,
        option_library=library,
    )
    assert probabilities == pytest.approx(np.asarray([0.0, 0.5, 0.5]))


def test_empty_valid_support_reroutes_to_registered_noop(monkeypatch):
    library = _FakeOptionLibrary()
    _patch_scores(monkeypatch, {0: 0.0, 1: 0.0, 2: 0.0})
    probabilities = option_distribution(
        ProtocolSpec(),
        {"valid_options": np.asarray([False, False, False])},
        public_state=None,
        option_library=library,
    )
    assert probabilities == pytest.approx(np.asarray([1.0, 0.0, 0.0]))


def test_invalid_or_malformed_support_fails_closed(monkeypatch):
    library = _FakeOptionLibrary()
    _patch_scores(monkeypatch, {0: 0.0, 1: 1.0, 2: 2.0})
    with pytest.raises(ValueError, match="valid_options must have shape"):
        option_distribution(
            ProtocolSpec(),
            {"valid_options": np.asarray([True, False])},
            public_state=None,
            option_library=library,
        )
    with pytest.raises(ValueError, match="epsilon must be in"):
        option_distribution(
            ProtocolSpec(),
            {
                "epsilon": 1.1,
                "valid_options": np.asarray([True, True, True]),
            },
            public_state=None,
            option_library=library,
        )


def test_generation_and_inference_call_the_same_option_distribution(monkeypatch):
    library = _FakeOptionLibrary()
    protocol = ProtocolSpec(role="flexible")
    _patch_scores(monkeypatch, {0: -5.0, 1: 8.0, 2: 1.0})
    shared_implementation = partner_pool.option_distribution
    calls: list[tuple[ProtocolSpec, dict]] = []

    def recording_distribution(theta, runtime_state, public_state, *, option_library):
        calls.append((theta, dict(runtime_state)))
        return shared_implementation(
            theta,
            runtime_state,
            public_state,
            option_library=option_library,
        )

    monkeypatch.setattr(partner_pool, "option_distribution", recording_distribution)
    partner = ScriptedProtocolPartner(
        name="shared-policy-test",
        option_library=library,
        protocol=protocol,
    )
    valid = np.asarray([False, True, True])
    chosen = partner._choose_option(
        state={"public": "state"},
        valid=valid,
        rng=np.random.default_rng(7),
    )

    # The sparse forward likelihood uses the same pure function and indexes the
    # observed emitted option; it does not carry a second policy formula.
    likelihood = partner_pool.option_distribution(
        protocol,
        {
            "elapsed": 0,
            "bottleneck_alternate_phase": 0,
            "epsilon": 0.0,
            "valid_options": valid,
        },
        public_state={"public": "state"},
        option_library=library,
    )
    assert chosen == 1
    assert likelihood[chosen] == pytest.approx(1.0)
    assert len(calls) == 2
    assert calls[0][0] is protocol
    assert calls[1][0] is protocol


def test_scripted_partner_state_round_trip_is_detached():
    partner = ScriptedProtocolPartner(
        name="state-round-trip",
        option_library=_FakeOptionLibrary(),
        protocol=ProtocolSpec(role="flexible"),
    )
    partner.current_option = 1
    partner.last_state = {"counter": np.asarray([3], dtype=np.int32)}
    partner.last_primitive_action = 2
    partner.option_runtime = OptionRuntime(option_id=1, start_pos=(2, 4), elapsed=5)
    partner.elapsed = 5
    partner.bottleneck_alternate_phase = 1
    frozen = partner.get_state()

    partner.last_state["counter"][0] = 99
    partner.reset(0)
    partner.set_state(frozen)
    assert partner.current_option == 1
    assert partner.elapsed == 5
    assert partner.option_runtime is not None
    assert partner.option_runtime.start_pos == (2, 4)
    assert int(partner.last_state["counter"][0]) == 3
