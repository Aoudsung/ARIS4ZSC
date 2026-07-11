from __future__ import annotations

import copy

import pytest

from experiments.overcooked_v2.path_c_split import (
    MIN_GROUPS_PER_MECHANISM,
    PREFERRED_GROUPS_PER_MECHANISM,
    SPLIT_ROLES,
    SplitGroupV1,
    SplitManifestV1,
)


def _groups(
    mechanisms: tuple[str, ...] = ("m0", "m1"),
    *,
    include_spare_layout: bool = True,
) -> tuple[SplitGroupV1, ...]:
    groups = []
    for mechanism in mechanisms:
        for index in range(MIN_GROUPS_PER_MECHANISM):
            prefix = f"{mechanism}-primary-{index}"
            groups.append(
                SplitGroupV1(
                    group_id=prefix,
                    mechanism=mechanism,
                    identity_group=f"identity-{prefix}",
                    style_group=f"style-{prefix}",
                    seed_group=f"seed-{prefix}",
                    layout_group=f"layout-template-{prefix}",
                    layout_stratum="layout_control",
                )
            )
        if include_spare_layout:
            prefix = f"{mechanism}-secondary-layout"
            groups.append(
                SplitGroupV1(
                    group_id=prefix,
                    mechanism=mechanism,
                    identity_group=f"identity-{prefix}",
                    style_group=f"style-{prefix}",
                    seed_group=f"seed-{prefix}",
                    layout_group=f"layout-template-{prefix}",
                    layout_stratum="layout_shift",
                )
            )
    return tuple(groups)


def test_split_manifest_is_deterministic_and_input_order_independent():
    groups = _groups()
    left = SplitManifestV1.build(groups, manifest_seed=17)
    right = SplitManifestV1.build(reversed(groups), manifest_seed=17)
    assert left == right
    assert left.assignments == right.assignments
    assert left.sha256 == right.sha256


def test_split_manifest_covers_four_roles_per_mechanism_and_isolates_groups():
    manifest = SplitManifestV1.build(_groups(), manifest_seed=3)
    assignments = manifest.assignment_by_group
    for mechanism in ("m0", "m1"):
        roles = {
            assignments[group.group_id]
            for group in manifest.groups
            if group.mechanism == mechanism
            and group.layout_stratum == "layout_control"
        }
        assert roles == set(SPLIT_ROLES)

    for field_name in (
        "identity_group",
        "style_group",
        "seed_group",
        "layout_group",
    ):
        roles_by_value: dict[str, set[str]] = {}
        for group in manifest.groups:
            roles_by_value.setdefault(getattr(group, field_name), set()).add(
                assignments[group.group_id]
            )
        assert all(len(roles) == 1 for roles in roles_by_value.values())


def test_split_manifest_fails_closed_below_four_independent_groups():
    with pytest.raises(ValueError, match="at least four"):
        SplitManifestV1.build(_groups(("m0",), include_spare_layout=False)[:3])


def test_split_manifest_counts_linked_groups_as_one_isolation_component():
    groups = list(_groups(("m0",), include_spare_layout=False))
    original = groups[1]
    groups[1] = SplitGroupV1(
        group_id=original.group_id,
        mechanism=original.mechanism,
        identity_group=groups[0].identity_group,
        style_group=original.style_group,
        seed_group=original.seed_group,
        layout_group=original.layout_group,
        layout_stratum=original.layout_stratum,
    )
    with pytest.raises(ValueError, match="at least four"):
        SplitManifestV1.build(groups)


def test_primary_identity_shift_requires_a_shared_layout_control_stratum():
    groups = []
    for index in range(MIN_GROUPS_PER_MECHANISM):
        prefix = f"m0-{index}"
        groups.append(
            SplitGroupV1(
                group_id=prefix,
                mechanism="m0",
                identity_group=f"identity-{prefix}",
                style_group=f"style-{prefix}",
                seed_group=f"seed-{prefix}",
                layout_group=f"layout-{prefix}",
                layout_stratum=f"different-layout-stratum-{index}",
            )
        )
    with pytest.raises(ValueError, match="shared layout_stratum"):
        SplitManifestV1.build(groups)


def test_preferred_spare_is_reported_but_four_groups_remain_feasible():
    four_group_manifest = SplitManifestV1.build(
        _groups(("m0",), include_spare_layout=False)
    )
    five_group_manifest = SplitManifestV1.build(_groups(("m0",)))
    assert four_group_manifest.independent_group_count_by_mechanism["m0"] == 4
    assert four_group_manifest.preferred_spare_satisfied_by_mechanism["m0"] is False
    assert five_group_manifest.independent_group_count_by_mechanism["m0"] == (
        PREFERRED_GROUPS_PER_MECHANISM
    )
    assert five_group_manifest.preferred_spare_satisfied_by_mechanism["m0"] is True


def test_primary_identity_and_secondary_layout_views_are_separate():
    manifest = SplitManifestV1.build(_groups(("m0",)), manifest_seed=5)
    payload = manifest.canonical_payload()
    primary = payload["primary_identity_shift"]
    secondary = payload["secondary_layout_shift"]
    assert primary["priority"] == "primary"
    assert primary["layout_shift_must_not_be_pooled"] is True
    assert primary["condition_on"] == ["mechanism", "layout_stratum"]
    assert secondary["priority"] == "secondary"
    assert secondary["may_replace_primary_identity_shift"] is False
    assert secondary["available_by_mechanism"]["m0"] is True
    secondary_group = next(
        group
        for group in manifest.groups
        if group.layout_stratum == "layout_shift"
    )
    assert manifest.role_for(secondary_group.group_id) == "locked_audit"


def test_role_local_cross_fitting_has_a_bound_description_and_hash():
    manifest = SplitManifestV1.build(
        _groups(("m0",)), manifest_seed=11, cross_fit_folds=5
    )
    payload = manifest.cross_fitting_payload()
    assert payload["scope"] == "within_role_only"
    assert payload["cross_role_pooling_allowed"] is False
    assert payload["unit"] == "episode_uid"
    assert payload["split_core_sha256"] == manifest.split_core_sha256
    assert len(manifest.cross_fitting_sha256) == 64
    fold = manifest.cross_fit_fold("design", "run:seed:episode")
    assert fold == manifest.cross_fit_fold("design", "run:seed:episode")
    assert 0 <= fold < 5


def test_numeric_seed_schedule_is_frozen_unique_and_group_specific():
    manifest = SplitManifestV1.build(
        _groups(("m0",)),
        manifest_seed=29,
        evaluation_seeds_per_group=3,
    )
    all_seeds = [
        seed
        for group in manifest.groups
        for seed in manifest.numeric_seeds_for_group(group.group_id)
    ]
    assert len(all_seeds) == len(set(all_seeds))
    assert len(all_seeds) == 3 * len(manifest.groups)
    assert len(manifest.numeric_seed_schedule_sha256) == 64
    payload = manifest.numeric_seed_schedule_payload()
    execution_seeds = [
        seed
        for entry in payload["entries"]
        for seed in entry["ocv2_execution_seeds"]
    ]
    assert len(execution_seeds) == len(set(execution_seeds))
    assert payload["ocv2_execution_seed_version"] == (
        "path_c_ocv2_execution_seed_v1"
    )
    group = manifest.groups[0]
    manifest.validate_numeric_seed(
        group.group_id,
        manifest.numeric_seeds_for_group(group.group_id)[0],
    )
    with pytest.raises(ValueError, match="not registered"):
        manifest.validate_numeric_seed(group.group_id, -1)


def test_numeric_seed_schedule_rejects_execution_seed_collisions(monkeypatch):
    from experiments.overcooked_v2 import path_c_seed

    monkeypatch.setattr(path_c_seed, "derive_ocv2_execution_seed", lambda _seed: 7)
    manifest = SplitManifestV1.build(
        _groups(("m0",)),
        manifest_seed=31,
        evaluation_seeds_per_group=2,
    )
    with pytest.raises(ValueError, match="execution-seed collision"):
        manifest.numeric_seed_schedule_payload()


def test_split_manifest_round_trip_recomputes_assignments_and_hashes():
    manifest = SplitManifestV1.build(_groups(), manifest_seed=23)
    restored = SplitManifestV1.from_mapping(manifest.to_mapping())
    assert restored == manifest
    assert restored.sha256 == manifest.sha256

    tampered = copy.deepcopy(manifest.to_mapping())
    current_role = tampered["assignments"][0]["role"]
    tampered["assignments"][0]["role"] = next(
        role for role in SPLIT_ROLES if role != current_role
    )
    with pytest.raises(ValueError, match="does not match"):
        SplitManifestV1.from_mapping(tampered)


def test_split_manifest_and_group_reject_unknown_keys():
    group_payload = _groups(("m0",))[0].to_mapping()
    group_payload["unknown"] = 1
    with pytest.raises(ValueError, match="unknown key"):
        SplitGroupV1.from_mapping(group_payload)

    manifest_payload = SplitManifestV1.build(_groups(("m0",))).to_mapping()
    manifest_payload["unknown"] = 1
    with pytest.raises(ValueError, match="unknown key"):
        SplitManifestV1.from_mapping(manifest_payload)
