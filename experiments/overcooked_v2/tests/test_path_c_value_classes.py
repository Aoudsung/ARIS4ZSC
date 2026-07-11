from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from experiments.overcooked_v2.scripts import diag_d1_dataset as dataset_module

from experiments.overcooked_v2.path_c_value_classes import (
    ECOLOGICAL_VALUE_OUTCOME_NAME,
    CrossFittedEcologicalValueClassesV2,
    CrossFittedValueClassesV1,
    cluster_stratified_permutation,
    fit_cross_fitted_ecological_value_estimates,
    fit_cross_fitted_value_classes,
)
from experiments.overcooked_v2.path_c_split import (
    MIN_GROUPS_PER_MECHANISM,
    SplitGroupV1,
    SplitManifestV1,
)
from experiments.overcooked_v2.scripts.diag_d1_dataset import (
    _backfill_held_out_episode_return,
    _frozen_split_assignment,
    _frozen_probe_cost_from_config,
    _validate_path_c_collection_args,
)


def test_frozen_split_assignment_binds_and_returns_numeric_seed(monkeypatch):
    manifest = _manifest()
    group = manifest.groups[0]
    numeric_seed = manifest.numeric_seeds_for_group(group.group_id)[0]
    preregistration = SimpleNamespace(split_manifest=manifest)
    monkeypatch.setattr(
        dataset_module,
        "load_frozen_preregistration",
        lambda _path: preregistration,
    )
    partner = SimpleNamespace(spec=SimpleNamespace(
        value_class_id=group.mechanism,
        style_id=group.style_group,
        surface_identity_key=group.identity_group,
    ))
    assignment = _frozen_split_assignment(
        {"path_c": {"preregistration_path": "frozen.yaml"}},
        partner,
        layout_name=group.layout_group,
        split_group_id=group.group_id,
        numeric_seed=numeric_seed,
    )
    assert assignment["numeric_seed"] == numeric_seed
    assert assignment["seed_group"] == group.seed_group
from experiments.overcooked_v2.scripts.diag_d1_train import (
    _legacy_cross_identity_transfer_summary,
    _path_c_baseline_completeness,
    _path_c_conditional_factorization_audit,
    _synthetic_registry_value_oracle_diagnostic,
    _validated_ecological_artifact_for_rows,
    write_ecological_value_class_artifacts,
)


def _manifest(*, n_folds: int = 3) -> SplitManifestV1:
    return SplitManifestV1.build(
        tuple(
            SplitGroupV1(
                group_id=f"group-{index}",
                mechanism="mechanism-a",
                identity_group=f"identity-{index}",
                style_group=f"style-{index}",
                seed_group=f"seed-{index}",
                layout_group="asymm_advantages",
                layout_stratum="layout-control",
            )
            for index in range(MIN_GROUPS_PER_MECHANISM)
        ),
        manifest_seed=17,
        cross_fit_folds=n_folds,
    )


def _episodes_covering_folds(
    manifest: SplitManifestV1,
    role: str,
    *,
    per_fold: int = 4,
) -> np.ndarray:
    by_fold = {fold_id: [] for fold_id in range(manifest.cross_fit_folds)}
    index = 0
    while (
        index < 10_000
        and any(len(values) < per_fold for values in by_fold.values())
    ):
        episode = f"{role}-episode-{index}"
        fold_id = manifest.cross_fit_fold(role, episode)
        if len(by_fold[fold_id]) < per_fold:
            by_fold[fold_id].append(episode)
        index += 1
    if any(len(values) < per_fold for values in by_fold.values()):
        raise AssertionError("Test fixture could not populate every frozen fold.")
    return np.asarray(sorted(
        episode for values in by_fold.values() for episode in values
    ))


def test_ecological_value_estimate_for_episode_excludes_its_own_return():
    manifest = _manifest()
    role = "design"
    episodes = _episodes_covering_folds(manifest, role, per_fold=4)
    outcomes = np.linspace(0.0, 11.0, episodes.size)
    contexts = np.asarray(["shared-public-context"] * episodes.size)
    first = fit_cross_fitted_ecological_value_estimates(
        outcomes,
        episodes,
        contexts,
        split_manifest=manifest,
        role=role,
        n_classes=3,
    )
    changed = outcomes.copy()
    changed[0] += 1000.0
    second = fit_cross_fitted_ecological_value_estimates(
        changed,
        episodes,
        contexts,
        split_manifest=manifest,
        role=role,
        n_classes=3,
    )
    target = str(episodes[0])
    assert isinstance(first, CrossFittedEcologicalValueClassesV2)
    assert first.assignment_by_episode[target].value_estimate == pytest.approx(
        second.assignment_by_episode[target].value_estimate
    )
    assert first.assignment_by_episode[target].estimation_sample_size >= 2


def test_ecological_value_classes_are_fitted_out_of_episode_fold(tmp_path) -> None:
    manifest = _manifest()
    episodes = _episodes_covering_folds(manifest, "design")
    outcomes = np.linspace(-1.0, 1.0, episodes.size)
    result = fit_cross_fitted_value_classes(
        outcomes,
        episodes,
        split_manifest=manifest,
        role="design",
        outcome_name=ECOLOGICAL_VALUE_OUTCOME_NAME,
        n_classes=3,
    )
    assert result.class_ids.shape == outcomes.shape
    assert result.quantile_method == "linear"
    folds_by_id = {fold.fold_id: fold for fold in result.folds}
    for assignment in result.assignments:
        assert assignment.distance_to_nearest_threshold == pytest.approx(min(
            abs(assignment.held_out_episode_return - threshold)
            for threshold in folds_by_id[assignment.fold_id].bin_upper_bounds
        ))
    assert set(result.class_ids.tolist()).issubset({0, 1, 2})
    for fold in result.folds:
        assert set(fold.fit_episode_uids).isdisjoint(fold.apply_episode_uids)
        assert fold.role == "design"
        assert len(fold.bin_upper_bounds) == 2
        assert all(
            manifest.cross_fit_fold("design", episode) == fold.fold_id
            for episode in fold.apply_episode_uids
        )

    restored = CrossFittedValueClassesV1.from_mapping(
        result.to_mapping(),
        split_manifest=manifest,
    )
    assert restored.sha256 == result.sha256
    reordered = fit_cross_fitted_value_classes(
        outcomes[::-1],
        episodes[::-1],
        split_manifest=manifest,
        role="design",
        outcome_name=ECOLOGICAL_VALUE_OUTCOME_NAME,
        n_classes=3,
    )
    assert reordered.sha256 == result.sha256
    references = write_ecological_value_class_artifacts(
        {"design": result},
        tmp_path,
    )
    assert references[0]["artifact_sha256"] == result.sha256
    assert references[0]["split_manifest_sha256"] == manifest.sha256
    assert references[0]["path"].endswith(
        f"design.{result.sha256}.json"
    )
    with pytest.raises(FileExistsError, match="already exists"):
        write_ecological_value_class_artifacts({"design": result}, tmp_path)

    assignment_by_episode = result.assignment_by_episode
    validated_data = {
        "collection_role": np.asarray(["design"] * episodes.size),
        "episode_uid": episodes,
        "held_out_episode_return": outcomes,
        "ecological_value_class_id": np.asarray([
            assignment_by_episode[item].class_id for item in episodes
        ]),
        "ecological_value_class_fold": np.asarray([
            assignment_by_episode[item].fold_id for item in episodes
        ]),
        "ecological_value_class_threshold_distance": np.asarray([
            assignment_by_episode[item].distance_to_nearest_threshold
            for item in episodes
        ]),
        "ecological_value_class_artifact_sha256": np.asarray(
            [result.sha256] * episodes.size
        ),
        "_validated_ecological_value_class_artifacts": {"design": result},
    }
    role, validated = _validated_ecological_artifact_for_rows(
        validated_data,
        np.arange(episodes.size),
    )
    assert role == "design"
    assert validated.sha256 == result.sha256
    validated_data["ecological_value_class_id"][0] = (
        int(validated_data["ecological_value_class_id"][0]) + 1
    ) % 3
    with pytest.raises(ValueError, match="labels changed"):
        _validated_ecological_artifact_for_rows(
            validated_data,
            np.arange(episodes.size),
        )

    wrong_method = result.to_mapping()
    wrong_method["quantile_method"] = "nearest"
    with pytest.raises(ValueError, match="quantile_method"):
        CrossFittedValueClassesV1.from_mapping(
            wrong_method,
            split_manifest=manifest,
        )
    with pytest.raises(ValueError, match="split manifest"):
        CrossFittedValueClassesV1.from_mapping(
            result.to_mapping(),
            split_manifest=SplitManifestV1.build(
                manifest.groups,
                manifest_seed=manifest.manifest_seed + 1,
                cross_fit_folds=manifest.cross_fit_folds,
            ),
        )


def test_value_class_fit_fails_when_strict_bins_cannot_be_defined() -> None:
    manifest = _manifest(n_folds=2)
    episodes = _episodes_covering_folds(manifest, "calibration")
    with pytest.raises(ValueError, match="strictly ordered classes"):
        fit_cross_fitted_value_classes(
            np.zeros(episodes.size),
            episodes,
            split_manifest=manifest,
            role="calibration",
            outcome_name=ECOLOGICAL_VALUE_OUTCOME_NAME,
            n_classes=3,
        )


def test_value_class_artifact_rejects_tampering_and_row_outcome_drift() -> None:
    manifest = _manifest()
    episodes = _episodes_covering_folds(manifest, "locked_audit")
    outcomes = np.linspace(-2.0, 2.0, episodes.size)
    artifact = fit_cross_fitted_value_classes(
        outcomes,
        episodes,
        split_manifest=manifest,
        role="locked_audit",
        outcome_name=ECOLOGICAL_VALUE_OUTCOME_NAME,
        n_classes=3,
    )
    payload = artifact.to_mapping()
    payload["assignments"][0]["class_id"] = (
        payload["assignments"][0]["class_id"] + 1
    ) % 3
    with pytest.raises(ValueError, match="SHA-256|does not match"):
        CrossFittedValueClassesV1.from_mapping(
            payload,
            split_manifest=manifest,
        )

    repeated_episodes = np.repeat(episodes, 2)
    repeated_outcomes = np.repeat(outcomes, 2)
    labels = artifact.labels_for_rows(
        repeated_episodes,
        held_out_episode_return=repeated_outcomes,
    )
    assert labels.shape == repeated_episodes.shape
    repeated_outcomes[0] += 1.0
    with pytest.raises(ValueError, match="outcome differs"):
        artifact.labels_for_rows(
            repeated_episodes,
            held_out_episode_return=repeated_outcomes,
        )


def test_dataset_backfills_episode_return_from_decision_rewards() -> None:
    rows = {
        "decision_reward": [1.5, -0.25, 0.75],
        "held_out_episode_return": [np.nan, np.nan, np.nan],
    }
    outcome = _backfill_held_out_episode_return(rows, 0, 3)
    assert outcome == 2.0
    assert rows["held_out_episode_return"] == [2.0, 2.0, 2.0]


def test_registry_value_class_is_secondary_oracle_only() -> None:
    data = {
        "synthetic_registry_value_class_id": np.asarray(["a", "b"]),
    }
    report = _synthetic_registry_value_oracle_diagnostic(
        data,
        np.asarray([0, 1]),
    )
    assert report["role"] == "secondary_oracle_diagnostic_only"
    assert report["decision_eligible"] is False
    assert report["may_define_ecological_value_class"] is False

    unavailable = _path_c_conditional_factorization_audit(
        {
            "synthetic_registry_value_class_id": np.asarray(["a", "b"]),
            "identity_key": np.asarray(["i0", "i1"]),
            "episode_uid": np.asarray(["e0", "e1"]),
        },
        np.asarray([True, True]),
        {},
        None,
        {},
        "cpu",
    )
    assert unavailable["available"] is False
    assert "ecological_value_class_id" in unavailable["missing"]
    assert unavailable["synthetic_registry_value_oracle"]["decision_eligible"] is False


def test_dataset_rejects_offline_belief_filter_as_an_acting_variant(tmp_path) -> None:
    import argparse

    args = argparse.Namespace(path_c_collection_variant="belief_filter")
    with pytest.raises(ValueError, match="secondary offline readout"):
        _validate_path_c_collection_args(args, tmp_path / "anchor.pt")


def test_version_three_probe_cost_always_comes_from_frozen_metadata() -> None:
    config = {
        "path_c": {"preregistration": {"probe_cost_per_use": 0.25}}
    }
    assert _frozen_probe_cost_from_config(config) == pytest.approx(0.25)
    with pytest.raises(ValueError, match="requires a frozen per-probe cost"):
        _frozen_probe_cost_from_config({"path_c": {"preregistration": {}}})
    with pytest.raises(ValueError, match="finite and non-negative"):
        _frozen_probe_cost_from_config({
            "path_c": {"preregistration": {"probe_cost_per_use": -0.1}}
        })
    with pytest.raises(ValueError, match="finite and non-negative"):
        _frozen_probe_cost_from_config({
            "path_c": {"preregistration": {"probe_cost_per_use": True}}
        })


def test_legacy_readout_baselines_cannot_create_a_decision_gate() -> None:
    completeness = _path_c_baseline_completeness(
        {
            "probing_ego": np.zeros((2, 1), dtype=np.float32),
            "base_only": np.zeros((2, 1), dtype=np.float32),
        },
        {"required_baseline_variants": ("base_only",)},
    )
    assert completeness["pass"] is True
    assert completeness["decision_eligible"] is False

    summary = _legacy_cross_identity_transfer_summary(
        {"measurement": {"value": 0.2, "threshold": 0.1, "pass": True}},
        {"available": True, "decision_eligible": False},
    )
    assert summary["reported_transfer_threshold_check"] is True
    assert summary["joint_boolean_gate"] is None
    assert summary["decision_eligible"] is False
    assert "pass" not in summary


def test_cluster_permutation_preserves_episode_and_stratum_support() -> None:
    episodes = np.asarray(["a", "a", "b", "b", "c", "c", "d", "d"])
    values = np.asarray([0, 0, 1, 1, 2, 2, 3, 3])
    strata = np.asarray(["left", "left", "left", "left", "right", "right", "right", "right"])
    permuted = cluster_stratified_permutation(
        values, episodes, strata, seed=9
    )
    for episode in np.unique(episodes):
        assert np.unique(permuted[episodes == episode]).size == 1
    assert set(permuted[strata == "left"].tolist()) == {0, 1}
    assert set(permuted[strata == "right"].tolist()) == {2, 3}


def test_cluster_permutation_rejects_row_varying_episode_labels() -> None:
    with pytest.raises(ValueError, match="one cluster label"):
        cluster_stratified_permutation(
            [0, 1],
            ["same-episode", "same-episode"],
            ["registered", "registered"],
            seed=0,
        )
