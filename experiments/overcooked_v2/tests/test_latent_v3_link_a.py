from __future__ import annotations

import numpy as np

from experiments.overcooked_v2.partner_modes import LatentModeRuntime, LatentModeSpec
from experiments.overcooked_v2.partner_pool import (
    LATENT_V3_DEV_PROTOCOLS,
    PARTNER_REGISTRIES,
    sample_mode_spec,
)


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

