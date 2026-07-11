from __future__ import annotations

import copy

import pytest

from experiments.overcooked_v2.path_c_audit_battery import (
    FrozenAuditBatteryV1,
    ProbeScriptV1,
    resolve_probe_step,
)


def _battery() -> FrozenAuditBatteryV1:
    return FrozenAuditBatteryV1(
        battery_id="path-c-test",
        battery_version="v1",
        T_probe=2,
        L_inner=3,
        core_scripts=(
            ProbeScriptV1("core-a", (1, 2), 0.4, scope="core"),
            ProbeScriptV1("core-b", (3, 4), 0.6, scope="core"),
        ),
        design_only_scripts=(
            ProbeScriptV1("design-a", (5, 6), 1.0, scope="design_only"),
        ),
    )


def test_battery_hash_roundtrip_and_role_isolation():
    battery = _battery()
    restored = FrozenAuditBatteryV1.from_manifest(battery.to_manifest())

    assert restored == battery
    assert restored.sha256 == battery.sha256
    assert [item.script_id for item in restored.scripts_for_role("design")] == [
        "core-a",
        "core-b",
        "design-a",
    ]
    assert [item.script_id for item in restored.scripts_for_role("calibration")] == [
        "core-a",
        "core-b",
    ]
    assert [item.script_id for item in restored.scripts_for_role("locked_audit")] == [
        "core-a",
        "core-b",
    ]


def test_battery_hash_rejects_tampering_and_unknown_fields():
    manifest = _battery().to_manifest()
    tampered = copy.deepcopy(manifest)
    tampered["core_scripts"][0]["option_ids"][0] = 99
    with pytest.raises(ValueError, match="SHA-256"):
        FrozenAuditBatteryV1.from_manifest(tampered)

    unknown = copy.deepcopy(manifest)
    unknown["selected_after_audit"] = True
    with pytest.raises(ValueError, match="Unknown audit battery"):
        FrozenAuditBatteryV1.from_manifest(unknown)


def test_T_probe_and_L_inner_are_distinct_and_validated():
    battery = _battery()
    assert battery.T_probe == 2
    assert battery.L_inner == 3

    with pytest.raises(ValueError, match="T_probe"):
        FrozenAuditBatteryV1(
            battery_id="bad-length",
            battery_version="v1",
            T_probe=3,
            L_inner=2,
            core_scripts=(ProbeScriptV1("core", (1, 2), 1.0),),
        )
    with pytest.raises(ValueError, match="sum to one"):
        FrozenAuditBatteryV1(
            battery_id="bad-probability",
            battery_version="v1",
            T_probe=1,
            L_inner=1,
            core_scripts=(
                ProbeScriptV1("left", (1,), 0.4),
                ProbeScriptV1("right", (2,), 0.4),
            ),
        )


def test_invalid_option_fallback_is_part_of_the_frozen_intervention():
    script = ProbeScriptV1("core", (7, 8), 1.0)
    valid = resolve_probe_step(
        script,
        0,
        {7, 9},
        noop_primitive_action=0,
    )
    invalid = resolve_probe_step(
        script,
        1,
        {7, 9},
        noop_primitive_action=0,
    )

    assert valid.executed_option_id == 7
    assert valid.fallback_primitive_action is None
    assert valid.support_violation is False
    assert invalid.executed_option_id is None
    assert invalid.fallback_primitive_action == 0
    assert invalid.support_violation is True
    assert invalid.support_violation_token == "support_violation"
    assert invalid.intervention_sha256 != valid.intervention_sha256


def test_probe_script_rejects_adaptive_invalid_option_policy():
    with pytest.raises(ValueError, match="frozen"):
        ProbeScriptV1(
            "adaptive",
            (1,),
            1.0,
            invalid_option_policy="choose_nearest_valid_option",
        )
