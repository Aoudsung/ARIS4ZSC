from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from experiments.overcooked_v2.path_c_evaluation import (
    PATH_C_LEAKAGE_CHANNELS,
    assemble_path_c_measurements,
    assign_group_disjoint_folds,
    build_path_c_readout_features,
    canonical_sha256,
    empirical_kernel_distance_audit,
    fingerprint_admission_measurement,
    load_and_validate_path_c_inputs,
    load_frozen_preregistration,
    pass_af_claim_rule,
    phase_b_go_no_go_rule,
    probe_support_measurement,
    require_formal_benchmark_artifact,
    validate_cross_identity_folds,
    validate_runtime_path_c_config,
)
from experiments.overcooked_v2.residual_signature import (
    residual_signature_disagreement,
    select_probe_candidate,
)


TEMPLATE = Path(__file__).resolve().parents[1] / "configs" / "path_c_preregistration.yaml"


def _write_frozen_preregistration(
    tmp_path: Path,
    *,
    compact: bool = False,
    base_checkpoint_sha256: str | None = None,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    payload = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    payload["status"] = "frozen"
    payload["version"] = "path-c-test-v1"
    payload["freeze_timestamp"] = "2026-07-10T00:00:00Z"
    payload["pass_af"]["thresholds"]["kernel_bootstrap_iters"] = 40
    payload["pass_af"]["thresholds"]["kernel_min_context_rows"] = 4
    if base_checkpoint_sha256 is not None:
        payload["probe"]["base_checkpoint_sha256"] = base_checkpoint_sha256
    payload["budget"]["effective_episode_floor"] = 1
    payload["budget"]["effective_transition_floor"] = 1
    if compact:
        payload["budget"]["required_budget_keys"] = ["probe", "random_probe", "no_probe"]
        payload["baselines"]["required_variants"] = ["base_only"]
        payload["baselines"]["strong_history_or_belief_variants"] = ["base_only"]
    path = tmp_path / "path_c_frozen.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path, load_frozen_preregistration(path)


def _leakage(value: float = 0.0, ci_hi: float = 0.001):
    return {
        "representation_scope": "retained_only",
        "channels": {
            channel: {"value": value, "ci_hi": ci_hi}
            for channel in PATH_C_LEAKAGE_CHANNELS
        },
    }


def _passing_variant(required_baselines=("base_only",)):
    comparisons = {
        str(name): {"value": 0.2, "ci_lo": 0.1}
        for name in required_baselines
    }
    return {
        "G1": {"comparisons": copy.deepcopy(comparisons)},
        "G_value": {"value": 0.2, "ci_lo": 0.1},
        "G2": {"comparisons": copy.deepcopy(comparisons)},
        "G3": _leakage(),
        "G4": {
            "power_value": 0.2,
            "power_ci_lo": 0.1,
            "null_ci_lo": -0.01,
            "null_ci_hi": 0.01,
            "null_retained_coordinate_count": 0,
            "fingerprint_visibility_ci_lo": 0.7,
            "fingerprint_chance": 0.5,
            "fingerprint_control_kind": "raw_response_value_null",
            "fingerprint_cross_mechanism_counterbalanced": True,
            "fingerprint_value_null_ci": [-0.01, 0.01],
            "fingerprint_joint_rv_null_ci": [-0.01, 0.01],
            "fingerprint_residual_null_ci": [-0.01, 0.01],
            "same_budget": True,
        },
        "C_fact": {
            "kernel_distance_audit": {
                "delta": 0.01,
                "max_same_W_upper_ci": 0.04,
                "min_diff_W_witness_lower_ci": 0.2,
                "same_mechanism_quantifier": "max_over_identity_context",
                "different_mechanism_quantifier": "min_pair_max_witness",
                "bootstrap_unit": "episode_uid",
                "simultaneous_bootstrap": True,
                "context_strata_source": "preregistered_semantic",
                "all_required_strata_available": True,
            }
        },
        "probe_support": {
            "selected": 200,
            "opportunities": 2000,
            "context_coverage": 0.9,
            "action_coverage": 0.9,
        },
        "C_min": {
            "coordinate_measurements": {
                "coordinate_0": {
                    "metrics": {
                        "heldout_td": {"value": 0.1, "ci_lo": 0.05},
                    }
                }
            },
            "retained_coordinates": ["coordinate_0"],
            "nuisance_gain": 0.0,
            "nuisance_ci_hi": 0.001,
        },
        "C_leak_full": _leakage(),
    }


def _passing_measurements(preregistration, *, demoted: bool = False):
    main = _passing_variant(preregistration.required_baselines)
    budgets = {
        key: {"effective_episodes": 1, "effective_transitions": 1}
        for key in preregistration.payload["budget"]["required_budget_keys"]
    }
    main["artifact_contract"] = {
        "preregistration_sha256": preregistration.sha256,
        "resolved_path_c_sha256": preregistration.runtime_contract_sha256,
        "present_baselines": list(preregistration.required_baselines),
        "hashes_valid": True,
        "split_matched": True,
        "capacity_matched": True,
        "budgets": budgets,
    }
    strong = {}
    for name in preregistration.strong_baselines:
        item = _passing_variant(preregistration.required_baselines)
        if not demoted:
            item["G1"] = {
                "comparisons": {
                    str(baseline): {"value": -0.1, "ci_lo": -0.2}
                    for baseline in preregistration.required_baselines
                }
            }
        strong[name] = item
    main["strong_baselines"] = strong
    return main


def test_phase_b_recomputes_go_demoted_and_missing_no_go(tmp_path):
    _path, preregistration = _write_frozen_preregistration(tmp_path)
    assert phase_b_go_no_go_rule(
        _passing_measurements(preregistration), preregistration
    )["status"] == "GO"
    assert phase_b_go_no_go_rule(
        _passing_measurements(preregistration, demoted=True), preregistration
    )["status"] == "DEMOTED"
    missing = _passing_measurements(preregistration)
    del missing["G2"]
    assert phase_b_go_no_go_rule(missing, preregistration)["status"] == "NO-GO"


def test_runtime_config_must_exactly_match_frozen_contract(tmp_path):
    _path, preregistration = _write_frozen_preregistration(tmp_path)
    runtime = copy.deepcopy(preregistration.runtime_contract)
    runtime["ensemble"]["tie_atol"] = 0.1
    with pytest.raises(ValueError, match="ensemble.tie_atol"):
        validate_runtime_path_c_config(runtime, preregistration)


def test_input_pass_field_cannot_bypass_frozen_threshold(tmp_path):
    _path, preregistration = _write_frozen_preregistration(tmp_path)
    measurements = _passing_measurements(preregistration)
    measurements["G1"] = {"value": -1.0, "ci_lo": -1.0, "pass": True}
    result = phase_b_go_no_go_rule(measurements, preregistration)
    assert result["status"] == "NO-GO"
    assert result["rules"]["G1"]["pass"] is False


def test_pass_af_is_separate_from_phase_b_gate(tmp_path):
    _path, preregistration = _write_frozen_preregistration(tmp_path)
    measurements = _passing_measurements(preregistration)
    assert phase_b_go_no_go_rule(measurements, preregistration)["status"] == "GO"
    measurements["C_min"]["retained_coordinates"] = []
    claim = pass_af_claim_rule(measurements, preregistration)
    assert claim["status"] == "NOT_READY_FOR_TYPE_B_REVIEW"
    assert claim["phase_b"]["status"] == "GO"


def test_terminal_null_requires_interval_containment_and_zero_coordinates(tmp_path):
    _path, preregistration = _write_frozen_preregistration(tmp_path)
    measurements = _passing_measurements(preregistration)
    measurements["G4"]["null_ci_hi"] = 0.06
    assert phase_b_go_no_go_rule(measurements, preregistration)["status"] == "NO-GO"
    measurements = _passing_measurements(preregistration)
    measurements["G4"]["null_retained_coordinate_count"] = 1
    assert phase_b_go_no_go_rule(measurements, preregistration)["status"] == "NO-GO"


def test_readout_features_give_base_only_public_context_not_zero():
    context = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    features = build_path_c_readout_features(
        context,
        {
            "probe": np.ones((2, 1), dtype=np.float32),
            "unpruned_ensemble": np.ones((2, 3), dtype=np.float32),
        },
    )
    assert np.array_equal(features["base_only"], context)
    assert np.array_equal(features["no_admission"], context)
    assert features["probe"].shape == (2, 3)
    assert features["unpruned_ensemble"].shape == (2, 5)


def test_grouped_split_has_zero_surface_seed_layout_intersection():
    surface = np.asarray(["a", "a", "b", "b", "c", "c"])
    seed = np.asarray(["s0", "s0", "s1", "s1", "s2", "s2"])
    layout = np.asarray(["l0", "l0", "l1", "l1", "l2", "l2"])
    mechanism = np.asarray(["m0", "m1", "m0", "m1", "m0", "m1"])
    folds = assign_group_disjoint_folds(surface, seed, layout, n_folds=3)
    report = validate_cross_identity_folds(folds, surface, seed, layout, mechanism)
    assert report["valid"] is True
    assert all(not any(item["overlap"].values()) for item in report["folds"])


def test_kernel_audit_uses_pairwise_existential_witness_and_simultaneous_bootstrap(tmp_path):
    _path, preregistration = _write_frozen_preregistration(tmp_path)
    rv, mechanism, identity, context, episode = [], [], [], [], []
    counter = 0
    for mech in ("m0", "m1"):
        for surface in ("i0", "i1"):
            for stratum in ("separating", "neutral"):
                for _ in range(30):
                    rv.append([1 if mech == "m1" and stratum == "separating" else 0])
                    mechanism.append(mech)
                    identity.append(surface)
                    context.append(stratum)
                    episode.append(f"ep-{counter}")
                    counter += 1
    audit = empirical_kernel_distance_audit(
        rv, mechanism, identity, context, episode, preregistration, seed=3
    )
    assert audit["available"] is True
    assert audit["different_mechanism_quantifier"] == "min_pair_max_witness"
    assert audit["simultaneous_bootstrap"] is True
    assert audit["min_diff_W_witness_lower_ci"] > 0.9


def test_sparse_kernel_strata_are_unavailable(tmp_path):
    _path, preregistration = _write_frozen_preregistration(tmp_path)
    audit = empirical_kernel_distance_audit(
        [[0], [0], [1]],
        ["m0", "m0", "m1"],
        ["i0", "i1", "i0"],
        ["c", "c", "c"],
        ["e0", "e1", "e2"],
        preregistration,
    )
    assert audit["available"] is False
    assert audit["all_required_strata_available"] is False


def test_metadata_only_fingerprint_cannot_be_positive_control(tmp_path):
    _path, preregistration = _write_frozen_preregistration(tmp_path)
    kwargs = dict(
        mechanism=["m0", "m0", "m1", "m1"],
        fingerprint_id=[0, 1, 0, 1],
        visibility_ci_lo=0.7,
        chance_accuracy=0.5,
        value_null_ci=[-0.01, 0.01],
        joint_rv_null_ci=[-0.01, 0.01],
        residual_null_ci=[-0.01, 0.01],
        preregistration=preregistration,
    )
    positive = fingerprint_admission_measurement(
        control_kind="raw_response_value_null", **kwargs
    )
    negative = fingerprint_admission_measurement(control_kind="metadata_only", **kwargs)
    assert positive["available_for_synthetic_power"] is True
    assert negative["available_for_synthetic_power"] is False


def test_probe_return_floor_applies_to_disagreement_candidate():
    decision = select_probe_candidate(
        torch.tensor([0.1, 2.0]),
        torch.tensor([10.0, -1.0]),
        torch.tensor([True, True]),
        disagreement_threshold=0.5,
        return_floor=0.0,
    )
    assert decision["option_id"] == 1
    assert decision["reason"] == "return_floor"
    assert decision["selected"] is False


def test_zero_selected_probes_have_zero_support():
    support = probe_support_measurement(
        [False, False],
        [True, True],
        ["c0", "c1"],
        [0, 1],
    )
    assert support["selected"] == 0
    assert support["context_coverage"] == 0.0
    assert support["action_coverage"] == 0.0


def test_best_option_set_disagreement_respects_ties():
    heads = torch.tensor([[[1.0, 1.0], [1.0, 1.0 + 1.0e-7]]])
    base = torch.zeros(1, 2)
    tied = residual_signature_disagreement(heads, base, tie_atol=1.0e-6)
    untied = residual_signature_disagreement(heads, base, tie_atol=0.0)
    assert torch.all(tied["best_option_set_disagreement"] == 0)
    assert torch.any(untied["best_option_set_disagreement"] > 0)


def test_rv_readout_uses_ordered_joint_likelihood(monkeypatch):
    from experiments.overcooked_v2.scripts import diag_d1_train as d1

    feature_dims = []

    def fake_fit(features, labels, fold_keys, **_kwargs):
        del fold_keys
        feature_dims.append(features.shape[1])
        n_classes = int(np.max(labels)) + 1
        probs = np.full((labels.size, n_classes), 0.1 / max(n_classes - 1, 1))
        probs[np.arange(labels.size), labels] = 0.9
        if n_classes == 1:
            probs[:] = 1.0
        return {"available": True, "probs": probs}

    monkeypatch.setattr(d1, "_fit_predict_classifier", fake_fit)
    data = {
        "rv_first": np.asarray([0, 1, 0, 1]),
        "rv_second": np.asarray([1, 1, 0, 0]),
    }
    score = d1._readout_score_for_columns(
        np.ones((4, 2), dtype=np.float32),
        data,
        np.arange(4),
        ["rv_first", "rv_second"],
        np.asarray(["e0", "e1", "e2", "e3"]),
        device="cpu",
        seed=0,
    )
    assert score["likelihood"] == "autoregressive_joint"
    assert score["factor_order"] == ["rv_first", "rv_second"]
    assert feature_dims == [2, 4]


def test_formal_benchmark_gate_rejects_collection_output():
    with pytest.raises(ValueError, match="Data-collection"):
        require_formal_benchmark_artifact({
            "evaluation_kind": "data_collection_only",
            "collection_role": "data_collection_only",
            "active_probe_collection": True,
            "benchmark_return_eligible": False,
        })
    require_formal_benchmark_artifact({
        "evaluation_kind": "formal_benchmark",
        "collection_role": "formal_benchmark",
        "active_probe_collection": False,
        "benchmark_return_eligible": True,
    })


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _manifest_fixture(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = {}
    for name in ("probe", "base_only", "random_probe", "no_probe"):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(name.encode("ascii"))
        checkpoint_paths[name] = path
    prereg_path, prereg = _write_frozen_preregistration(
        tmp_path,
        compact=True,
        base_checkpoint_sha256=_sha(checkpoint_paths["base_only"]),
    )
    digest = "a" * 64
    schemas = {
        "graph_sha256": digest,
        "option_schema_sha256": "b" * 64,
        "observation_schema_sha256": "c" * 64,
    }
    checkpoints = {}
    input_contracts = {
        "probe": "public_context_plus_retained_residual_representation",
        "base_only": "public_context_only",
        "random_probe": "public_context_plus_retained_residual_representation",
        "no_probe": "public_context_plus_retained_residual_representation",
    }
    for name in ("probe", "base_only", "random_probe", "no_probe"):
        path = checkpoint_paths[name]
        parameter_path = _write_json(tmp_path / f"{name}-parameters.json", {
            "checkpoint_sha256": _sha(path),
            "preregistration_sha256": prereg.sha256,
            "resolved_path_c_sha256": prereg.runtime_contract_sha256,
            "trainable_parameters": {"weight": [2, 5]},
        })
        checkpoints[name] = {
            "path": path.name,
            "sha256": _sha(path),
            "method": name,
            **schemas,
            "resolved_path_c_sha256": prereg.runtime_contract_sha256,
            "preregistration_sha256": prereg.sha256,
            "oracle_baseline": False,
            "input_contract": input_contracts[name],
            "parameter_count": 10,
            "parameter_manifest": {
                "path": parameter_path.name,
                "sha256": _sha(parameter_path),
            },
            "train_episode_uids": [],
            "train_split_ids": [f"train-{name}"],
        }

    calibration = {
        "fold_id": "fold-0",
        "evaluation_split_id": "split-probe",
        "episode_uids": [f"run-probe:10:{index}" for index in range(4)],
        "base_only_checkpoint_sha256": checkpoints["base_only"]["sha256"],
        **schemas,
        "preregistration_sha256": prereg.sha256,
        "resolved_path_c_sha256": prereg.runtime_contract_sha256,
        "prediction_columns": ["q_0"],
        "predictions_by_episode": {
            f"run-probe:10:{index}": [0.0]
            for index in range(4)
        },
    }
    calibration_path = _write_json(tmp_path / "base_calibration.json", calibration)
    checkpoints["base_only"]["calibration_predictions"] = [{
        "path": calibration_path.name,
        "sha256": _sha(calibration_path),
        "fold_id": "fold-0",
        "evaluation_split_id": "split-probe",
        "preregistration_sha256": prereg.sha256,
        "resolved_path_c_sha256": prereg.runtime_contract_sha256,
    }]

    schedule_hash = "d" * 64
    datasets = {}
    for name in ("probe", "random_probe", "no_probe"):
        chunk = tmp_path / f"{name}.npz"
        if name == "probe":
            chunk_values = {
                "episode_uid": np.asarray([f"run-probe:10:{index}" for index in range(4)]),
                "option_transition": np.ones(4, dtype=np.uint8),
                "fold_id": np.asarray([0, 0, 1, 1]),
                "surface_identity_key": np.asarray(["a", "a", "b", "b"]),
                "seed_group": np.asarray(["s0", "s0", "s1", "s1"]),
                "layout_style": np.asarray(["l0", "l0", "l1", "l1"]),
                "mechanism_key": np.asarray(["m0", "m1", "m0", "m1"]),
                "public_context_stratum": np.asarray(["c0", "c0", "c1", "c1"]),
                "probe_selected": np.asarray([1, 0, 1, 0], dtype=np.uint8),
                "probe_action_id": np.asarray([0, -1, 0, -1], dtype=np.int16),
                "probe_skip_reason": np.asarray([6, 4, 6, 4], dtype=np.int16),
                "probe_rule_id": np.asarray([1, 0, 1, 0], dtype=np.int16),
            }
            for rv_column in prereg.payload["rv_summary_spec"]["rv_columns"]:
                chunk_values[str(rv_column)] = np.zeros(4, dtype=np.int16)
        else:
            chunk_values = {
                "episode_uid": np.asarray([f"run-{name}:10:0"]),
                "option_transition": np.asarray([1], dtype=np.uint8),
            }
        np.savez(chunk, **chunk_values)
        required_columns = list(chunk_values)
        schema_payload = {
            "required_columns": required_columns,
            "optional_columns": [],
        }
        episode_uids = sorted(set(chunk_values["episode_uid"].astype(str).tolist()))
        rows = int(chunk_values["episode_uid"].shape[0])
        datasets[name] = {
            "collection_variant": name,
            "collection_role": "readout_evaluation" if name == "probe" else "data_collection_only",
            "collection_mode": "on_policy_independent",
            "split_id": f"split-{name}",
            "run_id": f"run-{name}",
            "base_seed": 10,
            "matching_group_sha256": schedule_hash,
            "schema_sha256": canonical_sha256(schema_payload),
            "required_columns": required_columns,
            "optional_columns": [],
            "episode_uids": episode_uids,
            "effective_episodes": len(episode_uids),
            "effective_transitions": rows,
            "preregistration_sha256": prereg.sha256,
            "resolved_path_c_sha256": prereg.runtime_contract_sha256,
            "rv_summary_spec_sha256": canonical_sha256(prereg.payload["rv_summary_spec"]),
            "cross_identity_split": name == "probe",
            "controller_checkpoint_key": name,
            "controller_checkpoint_sha256": checkpoints[name]["sha256"],
            "chunks": [{"path": chunk.name, "sha256": _sha(chunk), "rows": rows}],
        }

    main = _passing_variant()
    main.update({
        "measurement_kind": "path_c_readout",
        "variant": "probe",
        "source_dataset_keys": ["probe"],
        "readout_capacity": copy.deepcopy(prereg.payload["readout"]),
        "rv_columns": list(prereg.payload["rv_summary_spec"]["joint_likelihood_order"]),
        "cross_identity_audit": {"valid": True, "folds": []},
        "base_only_calibration_sha256_by_split": {
            "split-probe": _sha(calibration_path),
        },
        "preregistration_sha256": prereg.sha256,
        "resolved_path_c_sha256": prereg.runtime_contract_sha256,
        "benchmark_return": {
            "evaluation_kind": "formal_benchmark",
            "collection_role": "formal_benchmark",
            "active_probe_collection": False,
            "benchmark_return_eligible": True,
        },
    })
    baseline = copy.deepcopy(main)
    baseline["variant"] = "base_only"
    baseline["G1"] = {"value": -0.1, "ci_lo": -0.2}
    measurement_refs = {}
    for name, payload in (("probe", main), ("base_only", baseline)):
        path = _write_json(tmp_path / f"measurement-{name}.json", payload)
        measurement_refs[name] = {"path": path.name, "sha256": _sha(path)}

    value_control = {
        "preregistration_sha256": prereg.sha256,
        "resolved_path_c_sha256": prereg.runtime_contract_sha256,
        "source_dataset_keys": ["probe"],
        "schemas": schemas,
        "required_baselines": list(prereg.required_baselines),
        "budget_matched": True,
        "base_only_calibration_sha256_by_split": {
            "split-probe": _sha(calibration_path),
        },
        "provenance_columns": ["heldout_td_error", "action_value_ranking"],
        "held_out": True,
        "main": {
            "comparisons": {"base_only": {"value": 0.2, "ci_lo": 0.1}},
        },
        "baselines": {
            "base_only": {
                "comparisons": {"base_only": {"value": -0.1, "ci_lo": -0.2}},
            }
        },
    }
    value_path = _write_json(tmp_path / "value-control.json", value_control)
    manifest = {
        "schema_version": "path_c_artifacts_v1",
        "preregistration": {"sha256": prereg.sha256, "version": prereg.payload["version"]},
        "resolved_path_c_sha256": prereg.runtime_contract_sha256,
        "schemas": schemas,
        "datasets": datasets,
        "checkpoints": checkpoints,
        "measurement_artifacts": measurement_refs,
        "value_control_artifact": {"path": value_path.name, "sha256": _sha(value_path)},
    }
    manifest_path = _write_json(tmp_path / "manifest.json", manifest)
    return prereg_path, manifest_path, manifest


def test_manifest_readback_produces_go_without_legacy_d1_models(tmp_path):
    prereg_path, manifest_path, _manifest = _manifest_fixture(tmp_path)
    inputs = load_and_validate_path_c_inputs(prereg_path, manifest_path)
    measurements = assemble_path_c_measurements(inputs)
    assert phase_b_go_no_go_rule(measurements, inputs.preregistration)["status"] == "GO"


def test_stage_readout_path_c_never_loads_legacy_d1_models(tmp_path, monkeypatch):
    from experiments.overcooked_v2.scripts import diag_d1_train as d1

    prereg_path, manifest_path, _manifest = _manifest_fixture(tmp_path)

    def fail_legacy_load(*_args, **_kwargs):
        raise AssertionError("Path C readout must not load legacy D1 checkpoints")

    monkeypatch.setattr(d1, "_load_models", fail_legacy_load)
    out_dir = tmp_path / "readout"
    d1.stage_readout(
        None,
        out_dir,
        "cpu",
        str(prereg_path),
        str(manifest_path),
    )
    result = json.loads((out_dir / "readout_frozen.json").read_text(encoding="utf-8"))
    assert result["phase_b_go_no_go"]["status"] == "GO"
    assert result["decision_scope"]["proposal_passed"] is False


def test_manifest_rejects_artifact_tampering(tmp_path):
    prereg_path, manifest_path, manifest = _manifest_fixture(tmp_path)
    measurement_path = tmp_path / manifest["measurement_artifacts"]["probe"]["path"]
    measurement_path.write_text(measurement_path.read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_and_validate_path_c_inputs(prereg_path, manifest_path)


def test_manifest_rejects_authoritative_pass_and_budget_intent(tmp_path):
    prereg_path, manifest_path, manifest = _manifest_fixture(tmp_path)
    measurement_path = tmp_path / manifest["measurement_artifacts"]["probe"]["path"]
    payload = json.loads(measurement_path.read_text(encoding="utf-8"))
    payload["G1"]["pass"] = True
    _write_json(measurement_path, payload)
    manifest["measurement_artifacts"]["probe"]["sha256"] = _sha(measurement_path)
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="authoritative 'pass'"):
        load_and_validate_path_c_inputs(prereg_path, manifest_path)

    prereg_path, manifest_path, manifest = _manifest_fixture(tmp_path / "budget")
    manifest["datasets"]["probe"]["effective_transitions"] = 99
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="not derived from option transitions"):
        load_and_validate_path_c_inputs(prereg_path, manifest_path)


def test_manifest_rejects_duplicate_chunk_and_schema_misalignment(tmp_path):
    prereg_path, manifest_path, manifest = _manifest_fixture(tmp_path)
    manifest["datasets"]["random_probe"]["chunks"] = copy.deepcopy(
        manifest["datasets"]["probe"]["chunks"]
    )
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="reuses a dataset chunk"):
        load_and_validate_path_c_inputs(prereg_path, manifest_path)

    other = tmp_path / "schema"
    prereg_path, manifest_path, manifest = _manifest_fixture(other)
    manifest["datasets"]["probe"]["optional_columns"] = ["missing_column"]
    manifest["datasets"]["probe"]["schema_sha256"] = canonical_sha256({
        "required_columns": manifest["datasets"]["probe"]["required_columns"],
        "optional_columns": ["missing_column"],
    })
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="chunk schema mismatch"):
        load_and_validate_path_c_inputs(prereg_path, manifest_path)


def test_base_only_calibration_cannot_overlap_training_fold(tmp_path):
    prereg_path, manifest_path, manifest = _manifest_fixture(tmp_path)
    manifest["checkpoints"]["base_only"]["train_split_ids"].append("split-probe")
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="training splits overlap evaluation splits"):
        load_and_validate_path_c_inputs(prereg_path, manifest_path)
