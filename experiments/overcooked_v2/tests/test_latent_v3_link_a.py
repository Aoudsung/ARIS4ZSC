from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.overcooked_v2.partner_modes import (
    LatentModeController,
    LatentModeRuntime,
    LatentModeSpec,
    LatentPartnerSpec,
    _OCActions,
)
from experiments.overcooked_v2.partner_pool import (
    LATENT_V3_DEV_PROTOCOLS,
    PARTNER_REGISTRIES,
    sample_mode_spec,
)
from src.aris_bellman.specs import PartnerAction


def _finish_opportunity(runtime: LatentModeRuntime, *, ego: bool = False) -> None:
    runtime.observe_public_transition(
        gate_main=True,
        ego_initiated=False,
        partner_initiated=False,
    )
    runtime.observe_public_transition(
        gate_main=False,
        ego_initiated=ego,
        partner_initiated=not ego,
    )


def test_patience_yields_then_claims_after_cutoff():
    runtime = LatentModeRuntime(LatentModeSpec("patience", param=2, epsilon=0.0))
    runtime.reset(0)

    runtime.observe_public_transition(gate_main=True, ego_initiated=False, partner_initiated=False)
    assert runtime.policy == "yield"
    runtime.observe_public_transition(gate_main=False, ego_initiated=False, partner_initiated=True)
    runtime.observe_public_transition(gate_main=True, ego_initiated=False, partner_initiated=False)
    assert runtime.policy == "yield"
    runtime.observe_public_transition(gate_main=False, ego_initiated=False, partner_initiated=True)
    runtime.observe_public_transition(gate_main=True, ego_initiated=False, partner_initiated=False)
    assert runtime.policy == "claim"


def test_tit_for_tat_reacts_to_ego_claim_and_defer():
    runtime = LatentModeRuntime(LatentModeSpec("tit_for_tat", param=1, epsilon=0.0))
    runtime.reset(0)
    assert runtime.policy == "claim"

    _finish_opportunity(runtime, ego=True)
    assert runtime.policy == "yield"

    _finish_opportunity(runtime, ego=False)
    _finish_opportunity(runtime, ego=False)
    assert runtime.policy == "claim"


def test_block_switch_uses_option_decision_dwell():
    runtime = LatentModeRuntime(LatentModeSpec("block_switch", param=2, epsilon=0.0))
    runtime.reset(0)
    assert runtime.policy == "claim"

    runtime.observe_option_boundary()
    runtime.observe_option_boundary()
    assert runtime.policy == "claim"
    runtime.observe_option_boundary()
    assert runtime.policy == "yield"


def test_latent_v3_registry_names_are_fixed_and_not_blind_v3():
    names = [name for name, _spec in LATENT_V3_DEV_PROTOCOLS]
    assert len(names) == 18
    assert "latent-ingnear-patience2" in names
    assert "latent-flex-block25" in names
    assert "blind-cert-ingnear-titfortat3" in names
    assert "blind-cert-flex-block16" in names
    assert "latent_v3_dev" in PARTNER_REGISTRIES
    assert "blind_v3" not in PARTNER_REGISTRIES


def test_sample_mode_spec_is_seeded_and_in_declared_ranges():
    rng = np.random.default_rng(7)
    spec = sample_mode_spec(rng, {"block_switch": 1.0})
    assert spec.family == "block_switch"
    assert 12 <= int(spec.param) <= 25
    assert 0.05 <= float(spec.epsilon) <= 0.15


# ---- round-2 expression channel: last-moment yield abort ----

@dataclass
class _Proto:
    terminal_policy: str = "claim"


class _Opt:
    def __init__(self, kind: str):
        self.kind = kind


class _EmptyLayoutGraph:
    entities: dict = {}


class _StubLib:
    def __init__(self):
        self.options = [_Opt("serve_soup"), _Opt("fetch_ingredient")]
        self.layout_graph = _EmptyLayoutGraph()

    def valid_options(self, state, agent_id):  # no retreat available -> stay fallback
        return np.zeros(len(self.options), dtype=bool)

    def primitive_action(self, state, agent_id, option_id):
        return 0


class _StubInner:
    def __init__(self, **kwargs):
        self.current_option = 0
        self.protocol = kwargs.get("protocol")

    def reset(self, seed):
        pass


def _interact_action() -> PartnerAction:
    return PartnerAction(
        primitive_action=int(_OCActions.interact),
        option_id=None,
        option_confidence=0.0,
        option_dist=None,
        source="test",
    )


def test_yield_abort_replaces_terminal_interact_and_claim_passes_through():
    lib = _StubLib()
    spec = LatentPartnerSpec(
        geometry_profile="stub",
        base_protocol=_Proto(),
        mode=LatentModeSpec("static_yield", epsilon=0.0),
    )
    ctrl = LatentModeController(
        name="stub", option_library=lib, spec=spec,
        partner_cls=_StubInner, partner_id=1,
    )
    ctrl.runtime.reset(0)
    assert ctrl.runtime.policy == "yield"

    ctrl._inner.current_option = 0  # serve_soup (terminal)
    out = ctrl._maybe_yield_abort(_interact_action(), state=object())
    assert int(out.primitive_action) == int(_OCActions.stay)
    assert ctrl.runtime.last_trigger == "yield_abort"

    ctrl.runtime.policy = "claim"  # claim mode converts: interact passes through
    out2 = ctrl._maybe_yield_abort(_interact_action(), state=object())
    assert int(out2.primitive_action) == int(_OCActions.interact)

    ctrl.runtime.policy = "yield"  # non-terminal option: no interception
    ctrl._inner.current_option = 1
    out3 = ctrl._maybe_yield_abort(_interact_action(), state=object())
    assert int(out3.primitive_action) == int(_OCActions.interact)

