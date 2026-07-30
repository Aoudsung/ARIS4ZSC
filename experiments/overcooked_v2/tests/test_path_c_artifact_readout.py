from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from experiments.overcooked_v2 import artifact_readout_app
from experiments.overcooked_v2 import path_c as path_c_cli
from experiments.overcooked_v2.artifact_readout_app import (
    ArtifactExpectations,
    ArtifactState,
    DETERMINISTIC_OUTPUT_FILES,
    ReadoutPrediction,
    aggregate_predictions,
    analyze_artifact,
    classify_verdict,
    deterministic_replica_split,
    encode_response_codes,
    evaluate_lopo,
    extract_legal_features,
    fit_ridge,
)
from src.path_c.storage import read_parquet, write_json, write_parquet


def _ego_state(value: float) -> dict[str, object]:
    return {
        "reference_carry": [[value, value + 1.0]],
        "trainable_carry": [[value + 2.0, value + 3.0]],
        "control_carry": [9_999.0 + value],
        "slot_log_belief": [-0.1 - value, -2.0 - value],
        "previous_action": int(value) % 2,
        "previous_team_reward": value / 10.0,
        "episode_start": False,
        "log_temperature": 123.0,
        "generic_log_temperature": 456.0,
    }


def _reconstruction(
    trigger_id: str,
    *,
    partner_index: int | None,
    state_index: int,
    action_count: int,
) -> dict[str, object]:
    trigger_action = state_index % action_count
    use = _ego_state(float(state_index))
    mask = _ego_state(float(state_index))
    mask["control_carry"] = [-8_888.0]
    mask["slot_log_belief"] = [-8.0, -9.0]
    return {
        "trigger_id": trigger_id,
        "source": (
            "self_pairing"
            if partner_index is None
            else f"fixed_partner_{partner_index:02d}"
        ),
        "partner_index": partner_index,
        "episode_index": state_index,
        "episode_seed": 1000 + state_index,
        "trigger_step": 5 + state_index,
        "trigger_action": trigger_action,
        "response_code": state_index % 3,
        "task_phase_before_action": "privileged-do-not-read",
        "pre_environment": {"grid": [[[9_999]]]},
        "post_environment": {"grid": [[[8_888]]]},
        "post_observations": [
            [[float(state_index), 1.0], [float(trigger_action), 2.0]],
            [[-1.0, -2.0], [-3.0, -4.0]],
        ],
        "post_use_ego_state": use,
        "post_mask_ego_state": mask,
        "use_slot_posterior": [0.99, 0.01],
        "dominant_use_slot": 0,
    }


def _artifact_state(
    partner: int | None,
    index: int,
    *,
    response_code: int = 0,
    probe_value: float = 1.0,
) -> ArtifactState:
    values = np.asarray([float(index), 1.0], dtype=np.float64)
    mask = np.asarray([0.0, 0.5], dtype=np.float64)
    use = np.asarray([probe_value, probe_value + 0.5], dtype=np.float64)
    return ArtifactState(
        trigger_id=f"state-{partner}-{index}",
        source="self_pairing" if partner is None else f"fixed_partner_{partner:02d}",
        partner_index=partner,
        episode_index=index,
        episode_seed=100 + index,
        trigger_step=index,
        trigger_action=index % 2,
        response_code=response_code,
        state_features=values,
        history_features=np.concatenate((values, values + 2.0)),
        q_fit_use=use,
        q_fit_mask=mask,
        q_evaluation_use=use,
        q_evaluation_mask=mask,
        q_evaluation_use_discounted=use,
        q_evaluation_mask_discounted=mask,
        q_evaluation_use_correct_deliveries=use,
        q_evaluation_mask_correct_deliveries=mask,
        q_evaluation_use_wrong_deliveries=np.zeros_like(use),
        q_evaluation_mask_wrong_deliveries=np.zeros_like(mask),
        tau_fit=probe_value,
        tau_evaluation=probe_value,
        tau_evaluation_discounted=probe_value,
        tau_evaluation_correct_deliveries=probe_value,
        tau_evaluation_wrong_deliveries=0.0,
    )


def _prediction(state: ArtifactState, lift: float) -> ReadoutPrediction:
    return ReadoutPrediction(
        state=state,
        feature_tier="history",
        evaluation_kind="lopo",
        fold_partner=state.partner_index,
        lambda_mask=1.0,
        lambda_delta=1.0,
        predicted_mask=np.asarray([0.0, 1.0]),
        predicted_use=np.asarray([1.0, 0.0]),
        probe_mask_action=1,
        probe_use_action=0,
        oracle_mask_action=1,
        oracle_use_action=0,
        probe_lift=lift,
        oracle_lift=lift + 1.0,
    )


def test_replica_hash_split_is_exact_paired_and_deterministic() -> None:
    first = deterministic_replica_split("trigger-a", 128)
    second = deterministic_replica_split("trigger-a", 128)
    assert first == second
    fit, evaluation = first
    assert len(fit) == len(evaluation) == 64
    assert set(fit).isdisjoint(evaluation)
    assert set(fit) | set(evaluation) == set(range(128))
    assert deterministic_replica_split("trigger-b", 128) != first


def test_legal_feature_whitelist_ignores_privileged_and_path_c_fields() -> None:
    row = _reconstruction(
        "fixed-partner", partner_index=0, state_index=1, action_count=2
    )
    first = extract_legal_features(row, action_count=2, episode_steps=20)
    changed = json.loads(json.dumps(row))
    changed["partner_index"] = 999
    changed["task_phase_before_action"] = "leaked"
    changed["pre_environment"] = {"grid": [[[123_456]]]}
    changed["post_environment"] = {"grid": [[[654_321]]]}
    changed["use_slot_posterior"] = [0.0, 1.0]
    changed["dominant_use_slot"] = 1
    changed["post_use_ego_state"]["control_carry"] = [123_456.0]
    changed["post_use_ego_state"]["slot_log_belief"] = [-99.0, 0.0]
    changed["post_use_ego_state"]["j_use"] = 999_999.0
    changed["post_use_ego_state"]["lower_score"] = 999_999.0
    second = extract_legal_features(changed, action_count=2, episode_steps=20)
    assert first[0] == second[0]
    assert first[2] == second[2]
    np.testing.assert_array_equal(first[1], second[1])
    np.testing.assert_array_equal(first[3], second[3])


def test_legal_feature_schema_rejects_missing_registered_history() -> None:
    row = _reconstruction(
        "fixed-partner", partner_index=0, state_index=1, action_count=2
    )
    del row["post_use_ego_state"]["reference_carry"]
    with pytest.raises(ValueError, match="official-history field is absent"):
        extract_legal_features(row, action_count=2, episode_steps=20)


def test_unseen_response_code_uses_explicit_unknown_column() -> None:
    training = [_artifact_state(0, 0, response_code=2)]
    held_out = [_artifact_state(1, 1, response_code=9)]
    known = encode_response_codes(training, (2, 3))
    unknown = encode_response_codes(held_out, (2, 3))
    np.testing.assert_array_equal(known, [[1.0, 0.0, 0.0]])
    np.testing.assert_array_equal(unknown, [[0.0, 0.0, 1.0]])


def test_dual_ridge_matches_direct_primal_solution() -> None:
    rng = np.random.default_rng(17)
    features = rng.normal(size=(12, 40))
    targets = rng.normal(size=(12, 3))
    validation = rng.normal(size=(5, 40))
    penalty = 0.1
    model = fit_ridge(features, targets, penalty)
    observed = model.predict(validation)

    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    retained = scale > 1.0e-12
    train = (features[:, retained] - mean[retained]) / scale[retained]
    test = (validation[:, retained] - mean[retained]) / scale[retained]
    target_mean = np.mean(targets, axis=0)
    expected_weights = np.linalg.solve(
        train.T @ train + penalty * np.eye(train.shape[1]),
        train.T @ (targets - target_mean),
    )
    expected = target_mean + test @ expected_weights
    np.testing.assert_allclose(observed, expected, rtol=1.0e-10, atol=1.0e-10)


def test_partner_aggregation_weights_partners_not_states() -> None:
    predictions = [_prediction(_artifact_state(0, 0), 10.0)]
    predictions.extend(
        _prediction(_artifact_state(1, index), 0.0) for index in range(3)
    )
    summary = aggregate_predictions(predictions)
    assert summary["probe_lift"] == 5.0
    assert summary["state_count"] == 4


def test_held_out_labels_and_future_statistics_do_not_change_fold_prediction() -> None:
    states = [
        _artifact_state(partner, 2 * partner + local, response_code=partner)
        for partner in range(4)
        for local in range(2)
    ]
    baseline = evaluate_lopo(states, tier="history")
    changed = []
    for state in states:
        if state.partner_index != 0:
            changed.append(state)
            continue
        changed.append(
            replace(
                state,
                source="changed-but-fold-only-metadata",
                episode_seed=999_999,
                response_code=999,
                q_fit_use=np.asarray([-10_000.0, 10_000.0]),
                q_fit_mask=np.asarray([20_000.0, -20_000.0]),
                q_evaluation_use=np.asarray([30_000.0, -30_000.0]),
                q_evaluation_mask=np.asarray([-40_000.0, 40_000.0]),
            )
        )
    modified = evaluate_lopo(changed, tier="history")
    baseline_fold = sorted(
        (value for value in baseline.predictions if value.fold_partner == 0),
        key=lambda value: value.state.trigger_id,
    )
    modified_fold = sorted(
        (value for value in modified.predictions if value.fold_partner == 0),
        key=lambda value: value.state.trigger_id,
    )
    assert len(baseline_fold) == len(modified_fold) == 2
    for first, second in zip(baseline_fold, modified_fold, strict=True):
        assert first.lambda_mask == second.lambda_mask
        assert first.lambda_delta == second.lambda_delta
        np.testing.assert_array_equal(first.predicted_mask, second.predicted_mask)
        np.testing.assert_array_equal(first.predicted_use, second.predicted_use)
        assert first.probe_mask_action == second.probe_mask_action
        assert first.probe_use_action == second.probe_use_action


def test_fixed_verdict_rules_cover_all_registered_outcomes() -> None:
    positive_partners = {
        "0": {"lcb95": 0.1, "ucb95": 1.0},
        "1": {"lcb95": 0.0, "ucb95": 1.0},
    }
    go = classify_verdict(
        {"lcb95": 0.1, "ucb95": 1.0},
        {"lcb95": 0.2, "ucb95": 2.0},
        positive_partners,
    )
    assert go["verdict"] == "GO"

    no_oracle = classify_verdict(
        {"lcb95": -1.0, "ucb95": 1.0},
        {"lcb95": -2.0, "ucb95": 0.0},
        positive_partners,
    )
    assert no_oracle["verdict"] == "NO-GO"
    assert no_oracle["reason"] == "oracle_ucb_nonpositive"

    unreadable = classify_verdict(
        {"lcb95": -2.0, "ucb95": 0.0},
        {"lcb95": 0.2, "ucb95": 2.0},
        positive_partners,
    )
    assert unreadable["verdict"] == "NO-GO"

    wide = classify_verdict(
        {"lcb95": -0.1, "ucb95": 1.0},
        {"lcb95": -0.1, "ucb95": 2.0},
        positive_partners,
    )
    assert wide["verdict"] == "INCONCLUSIVE"

    conflict = classify_verdict(
        {"lcb95": 0.1, "ucb95": 2.0},
        {"lcb95": 0.2, "ucb95": 3.0},
        {
            "0": {"lcb95": 0.2, "ucb95": 1.0},
            "1": {"lcb95": -2.0, "ucb95": 0.0},
        },
    )
    assert conflict["verdict"] == "INCONCLUSIVE"
    assert conflict["significant_partner_conflict"] is True


def _write_fixture_artifact(root: Path) -> ArtifactExpectations:
    action_count = 2
    replicas = 4
    reconstructions = []
    continuations = []
    state_index = 0
    identities: list[tuple[str, int | None, int]] = []
    for partner in range(4):
        for unused in range(2):
            trigger_id = (
                f"partner-{partner:02d}-episode-{state_index:04d}-"
                f"step-{5 + state_index:03d}"
            )
            identities.append((trigger_id, partner, state_index))
            state_index += 1
    identities.append(("self-episode-0008-step-013", None, state_index))
    for trigger_id, partner, index in identities:
        reconstruction = _reconstruction(
            trigger_id,
            partner_index=partner,
            state_index=index,
            action_count=action_count,
        )
        reconstructions.append(reconstruction)
        identity = {
            name: reconstruction[name]
            for name in (
                "trigger_id",
                "source",
                "partner_index",
                "episode_index",
                "episode_seed",
                "trigger_step",
            )
        }
        preferred = index % action_count
        for replica in range(replicas):
            noise = 0.01 * replica
            for branch in ("use", "mask"):
                continuations.append(
                    {
                        **identity,
                        "estimand": "pre_response",
                        "forced_action": reconstruction["trigger_action"],
                        "replica_index": replica,
                        "branch": branch,
                        "remaining_raw_return": noise + (0.5 if branch == "use" else 0.0),
                        "remaining_discounted_return": noise + (0.25 if branch == "use" else 0.0),
                        "remaining_correct_deliveries": 1 if branch == "use" else 0,
                        "remaining_wrong_deliveries": 0,
                        "privileged_future": 999_999.0,
                    }
                )
            for action in range(action_count):
                mask_value = 2.0 if action == preferred else 0.0
                use_value = mask_value + (1.0 if action == preferred else 0.25)
                for branch, value in (("use", use_value), ("mask", mask_value)):
                    continuations.append(
                        {
                            **identity,
                            "estimand": "post_response",
                            "forced_action": action,
                            "replica_index": replica,
                            "branch": branch,
                            "remaining_raw_return": value + noise,
                            "remaining_discounted_return": 0.5 * value + noise,
                            "remaining_correct_deliveries": int(value > 0.0),
                            "remaining_wrong_deliveries": 0,
                            "privileged_future": 999_999.0,
                        }
                    )
    write_parquet(root / "reconstruction.parquet", reconstructions)
    write_parquet(root / "continuations.parquet", continuations)
    write_parquet(
        root / "trigger_values.parquet",
        [{"trigger_id": value[0], "replicas": replicas} for value in identities],
    )
    write_json(root / "summary.json", {"fixture": True})
    write_json(root / "run_metadata.json", {"fixture": True})
    return ArtifactExpectations(
        trigger_count=9,
        fixed_trigger_count=8,
        self_trigger_count=1,
        partner_count=4,
        replicas=replicas,
        action_count=action_count,
        episode_steps=20,
        continuation_row_count=len(continuations),
        bootstrap_replicates=3,
    )


def test_end_to_end_fixture_writes_reproducible_complete_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "frozen"
    output = tmp_path / "readout"
    expectations = _write_fixture_artifact(artifact)
    before = {
        path.name: path.read_bytes()
        for path in artifact.iterdir()
        if path.is_file()
    }
    monkeypatch.setattr(
        artifact_readout_app, "REGISTERED_EXPECTATIONS", expectations
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "path_c",
            "audit-artifact-readout",
            "--artifact-directory",
            str(artifact),
            "--output",
            str(output),
        ],
    )
    path_c_cli.main()
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["primary_feature_tier"] == "history"
    assert summary["bootstrap"]["pipeline_refit_per_replicate"] is True
    assert len(read_parquet(output / "split_manifest.parquet")) == 9 * 4
    assert len(read_parquet(output / "bootstrap.parquet")) == 2 * 3
    assert (output / "state_readout.parquet").is_file()
    assert (output / "report.md").is_file()
    assert (output / "configuration.json").is_file()
    assert (output / "output_sha256.json").is_file()
    metadata = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["frozen_inputs_unchanged"] is True
    assert metadata["implementation_unchanged"] is True
    assert metadata["implementation_before"]["source_sha256"]
    after = {
        path.name: path.read_bytes()
        for path in artifact.iterdir()
        if path.is_file()
    }
    assert before == after

    second_output = tmp_path / "readout-second"
    second_summary = analyze_artifact(
        artifact, second_output, expectations=expectations
    )
    assert summary == second_summary
    for name in (*DETERMINISTIC_OUTPUT_FILES, "output_sha256.json"):
        assert (output / name).read_bytes() == (second_output / name).read_bytes()
