from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


PATH_C_ARTIFACT_SCHEMA = "path_c_artifacts_v1"
PATH_C_PRIMARY_VARIANT = "probe"
PATH_C_PROBE_PROVENANCE_IDS = {
    "none": 0,
    "exploration": 1,
    "scripted": 2,
    "disabled": 3,
    "threshold": 4,
    "return_floor": 5,
    "selected": 6,
}
PATH_C_PROBE_RULE_IDS = {
    "none": 0,
    "max_residual_signature_disagreement": 1,
    "random_probe_valid": 2,
}
PATH_C_LEAKAGE_CHANNELS = (
    "fingerprint",
    "identity",
    "seed",
    "layout_style",
    "surface_action_frequency",
    "trajectory_source",
)

_FORBIDDEN_RV_COLUMNS = {
    "identity",
    "seed",
    "style",
    "layout_style",
    "trajectory_source",
    "resp_rv_value_event_target_x",
    "resp_rv_value_event_target_y",
}
_ALLOWED_COLLECTION_ROLES = {
    "data_collection_only",
    "readout_training",
    "readout_evaluation",
    "calibration",
    "synthetic_power",
    "terminal_axis_null",
    "recovery",
}
_EXPECTED_VARIANT_INPUT_CONTRACTS = {
    "probe": "public_context_plus_retained_residual_representation",
    "base_only": "public_context_only",
    "global_gru": "public_context_plus_recurrent_hidden_state",
    "partner_id": "public_context_plus_partner_id_oracle",
    "belief_filter": "public_context_plus_training_split_fitted_posterior",
    "rnn_residualized": "public_context_plus_residual_control_signature",
    "random_probe": "public_context_plus_retained_residual_representation",
    "no_probe": "public_context_plus_retained_residual_representation",
    "no_admission": "public_context_only",
    "unpruned_ensemble": "public_context_plus_all_candidate_coordinates",
}

_REQUIRED_PREREGISTRATION_SECTIONS = (
    "ensemble",
    "probe",
    "rv_summary_spec",
    "fingerprint",
    "cross_identity_split",
    "budget",
    "baselines",
    "readout",
    "pass_af",
    "go_no_go",
)
_REQUIRED_THRESHOLDS = (
    "G1",
    "G_value",
    "G2",
    "G3",
    "G4",
    "epsilon_F",
    "epsilon_R",
    "epsilon_V",
    "null_equivalence_gain",
    "synthetic_power_min_mechanism_advantage",
    "raw_fingerprint_balanced_accuracy_margin",
    "fingerprint_value_null_equivalence",
    "fingerprint_joint_rv_null_equivalence",
    "fingerprint_residual_null_equivalence",
    "value_ablation_min_degradation",
    "nuisance_value_gain_equivalence",
    "full_leakage_max_advantage",
    "kernel_bootstrap_iters",
    "kernel_confidence",
    "kernel_min_context_rows",
)
_RUNTIME_CONTRACT_FIELDS = {
    "ensemble": (
        "n_heads",
        "bootstrap_p",
        "prior_scale",
        "disagreement_stat",
        "tie_atol",
    ),
    "probe": (
        "enable",
        "eval_enable",
        "collection_enable",
        "rule",
        "disagreement_threshold",
        "return_floor",
        "base_checkpoint_sha256",
        "require_public_residual_baseline",
        "allow_raw_q_fallback",
        "min_selected_probes",
        "min_probe_opportunities",
        "min_context_coverage",
        "min_action_coverage",
    ),
    "rv_summary_spec": ("enable",),
    "fingerprint": (
        "enable",
        "kind",
        "vocab",
        "positive_control_kind",
        "negative_control_kind",
    ),
    "cross_identity_split": (
        "enable",
        "group_key",
        "identity_key",
        "split",
        "folds",
    ),
    "pass_af": ("enable",),
}


@dataclass(frozen=True)
class FrozenPathCPreregistration:
    path: Path
    sha256: str
    payload: dict[str, Any]
    thresholds: dict[str, Any]
    required_baselines: tuple[str, ...]
    strong_baselines: tuple[str, ...]
    runtime_contract: dict[str, Any]
    runtime_contract_sha256: str


@dataclass(frozen=True)
class PathCArtifactManifest:
    path: Path
    sha256: str
    payload: dict[str, Any]
    preregistration_sha256: str
    resolved_path_c_sha256: str
    datasets: dict[str, dict[str, Any]]
    checkpoints: dict[str, dict[str, Any]]
    measurement_artifacts: dict[str, dict[str, Any]]
    value_control_artifact: dict[str, Any]


@dataclass(frozen=True)
class ValidatedPathCInputs:
    preregistration: FrozenPathCPreregistration
    manifest: PathCArtifactManifest
    measurements: dict[str, dict[str, Any]]


def assemble_path_c_measurements(inputs: ValidatedPathCInputs) -> dict[str, Any]:
    """Build the only authoritative rule input from validated raw artifacts."""
    main = dict(inputs.measurements[PATH_C_PRIMARY_VARIANT])
    value_control = inputs.measurements["value_control"]
    value_main = _mapping(value_control.get("main"), "value_control_artifact.main")
    value_baselines = _mapping(
        value_control.get("baselines"),
        "value_control_artifact.baselines",
    )
    _reject_authoritative_pass_fields(value_main, "value_control_artifact.main")
    main["G_value"] = _aggregate_value_control(
        value_main,
        inputs.preregistration.required_baselines,
        "value_control_artifact.main",
    )
    strong: dict[str, Any] = {}
    for name in inputs.preregistration.strong_baselines:
        measurement = dict(inputs.measurements[name])
        baseline_value = _mapping(
            value_baselines.get(name),
            f"value_control_artifact.baselines.{name}",
        )
        _reject_authoritative_pass_fields(
            baseline_value,
            f"value_control_artifact.baselines.{name}",
        )
        measurement["G_value"] = _aggregate_value_control(
            baseline_value,
            inputs.preregistration.required_baselines,
            f"value_control_artifact.baselines.{name}",
        )
        strong[name] = measurement
    main["strong_baselines"] = strong
    main["artifact_contract"] = artifact_contract_measurement(inputs)
    return main


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_preregistration(path: str | Path) -> FrozenPathCPreregistration:
    source = Path(path).resolve()
    payload = _load_mapping(source, "Path C preregistration")
    if str(payload.get("status", "")).strip().lower() != "frozen":
        raise ValueError("Path C preregistration status must be exactly 'frozen'.")
    for key in ("version", "freeze_timestamp", *_REQUIRED_PREREGISTRATION_SECTIONS):
        value = payload.get(key)
        if value is None or value == "":
            raise ValueError(f"Path C frozen preregistration is missing required field: {key}")

    ensemble = _mapping(payload["ensemble"], "ensemble")
    _require_mapping_fields(
        ensemble,
        ("n_heads", "bootstrap_p", "prior_scale", "disagreement_stat", "tie_atol"),
        "ensemble",
    )
    if int(ensemble["n_heads"]) <= 1:
        raise ValueError("Path C frozen ensemble.n_heads must be greater than one.")
    if not 0.0 < float(ensemble["bootstrap_p"]) < 1.0:
        raise ValueError("Path C frozen ensemble.bootstrap_p must be in (0, 1).")
    if float(ensemble["prior_scale"]) <= 0.0:
        raise ValueError("Path C frozen ensemble.prior_scale must be positive.")
    if float(ensemble["tie_atol"]) < 0.0:
        raise ValueError("Path C frozen ensemble.tie_atol must be non-negative.")

    pass_af = _mapping(payload["pass_af"], "pass_af")
    if pass_af.get("enable") is not True:
        raise ValueError("Path C frozen pass_af.enable must be true.")
    thresholds = _mapping(pass_af.get("thresholds"), "pass_af.thresholds")
    missing_thresholds = [key for key in _REQUIRED_THRESHOLDS if thresholds.get(key) is None]
    if missing_thresholds:
        raise ValueError(
            "Path C frozen preregistration is missing threshold(s): "
            + ", ".join(missing_thresholds)
        )
    for key in _REQUIRED_THRESHOLDS:
        _finite_number(thresholds[key], f"pass_af.thresholds.{key}")
    confidence = float(thresholds["kernel_confidence"])
    if not 0.0 < confidence < 1.0:
        raise ValueError("pass_af.thresholds.kernel_confidence must be in (0, 1).")
    if int(thresholds["kernel_bootstrap_iters"]) <= 0:
        raise ValueError("pass_af.thresholds.kernel_bootstrap_iters must be positive.")
    if int(thresholds["kernel_min_context_rows"]) <= 1:
        raise ValueError("pass_af.thresholds.kernel_min_context_rows must exceed one.")

    budget = _mapping(payload["budget"], "budget")
    for key in (
        "effective_episode_floor",
        "effective_transition_floor",
        "required_budget_keys",
        "split_unit",
        "matched_budget_required",
    ):
        if budget.get(key) is None:
            raise ValueError(f"Path C frozen preregistration budget is missing: {key}")
    _string_tuple(budget["required_budget_keys"], "budget.required_budget_keys")
    if str(budget["split_unit"]) != "episode_uid":
        raise ValueError("Path C frozen budget.split_unit must be 'episode_uid'.")
    if budget["matched_budget_required"] is not True:
        raise ValueError("Path C frozen budget must require matched budgets.")

    baselines = _mapping(payload["baselines"], "baselines")
    required_baselines = _string_tuple(baselines.get("required_variants"), "baselines.required_variants")
    strong_baselines = _string_tuple(
        baselines.get("strong_history_or_belief_variants"),
        "baselines.strong_history_or_belief_variants",
    )
    if not set(strong_baselines).issubset(set(required_baselines)):
        raise ValueError("Every strong Path C baseline must also be a required baseline.")
    if baselines.get("max_trainable_parameters") is None:
        raise ValueError("Path C frozen preregistration baselines.max_trainable_parameters is required.")
    if baselines.get("matched_capacity_and_split") is not True:
        raise ValueError("Path C frozen baselines must require matched capacity and split.")

    probe = _mapping(payload["probe"], "probe")
    _require_mapping_fields(
        probe,
        (
            "enable", "eval_enable", "collection_enable", "rule",
            "disagreement_threshold", "return_floor",
            "base_checkpoint_sha256",
            "require_public_residual_baseline", "allow_raw_q_fallback",
            "min_selected_probes", "min_probe_opportunities",
            "min_context_coverage", "min_action_coverage",
        ),
        "probe",
    )
    if probe["enable"] is not True:
        raise ValueError("Path C frozen probe.enable must be true.")
    if bool(probe.get("enable", False)):
        for key in (
            "return_floor",
            "base_checkpoint_sha256",
            "min_selected_probes",
            "min_probe_opportunities",
            "min_context_coverage",
            "min_action_coverage",
        ):
            if probe.get(key) is None:
                raise ValueError(f"Path C enabled probe is missing frozen safety/support field: {key}")
        if not _is_sha256(probe["base_checkpoint_sha256"]):
            raise ValueError("Path C probe.base_checkpoint_sha256 must be a SHA-256 digest.")
        if probe["eval_enable"] is not False or probe["collection_enable"] is not True:
            raise ValueError("Path C probes must be collection-only and disabled for formal evaluation.")
        if probe["allow_raw_q_fallback"] is not False:
            raise ValueError("Path C probes cannot fall back to raw-Q disagreement.")
        if probe["rule"] != "max_residual_signature_disagreement":
            raise ValueError("Path C probe rule must use residual-signature disagreement.")

    rv_spec = _mapping(payload["rv_summary_spec"], "rv_summary_spec")
    _require_mapping_fields(
        rv_spec,
        (
            "enable", "source", "window_decisions", "raw_columns", "rv_columns",
            "excludes", "coordinate_policy", "joint_likelihood_order",
            "public_context_strata",
        ),
        "rv_summary_spec",
    )
    if rv_spec["enable"] is not True:
        raise ValueError("Path C frozen rv_summary_spec.enable must be true.")
    rv_columns = _string_tuple(rv_spec["rv_columns"], "rv_summary_spec.rv_columns")
    forbidden_rv = sorted(
        column
        for column in rv_columns
        if column in _FORBIDDEN_RV_COLUMNS
        or column.endswith("_x")
        or column.endswith("_y")
    )
    if forbidden_rv:
        raise ValueError("Path C R^V includes forbidden nuisance columns: " + ", ".join(forbidden_rv))
    excludes = set(_string_tuple(rv_spec["excludes"], "rv_summary_spec.excludes"))
    required_excludes = {
        "identity", "seed", "style", "layout_style", "trajectory_source",
        "absolute_target_coordinates",
    }
    if not required_excludes.issubset(excludes):
        missing = sorted(required_excludes.difference(excludes))
        raise ValueError("Path C R^V excludes are incomplete: " + ", ".join(missing))
    likelihood_order = tuple(
        str(value) for value in _sequence(
            rv_spec["joint_likelihood_order"],
            "rv_summary_spec.joint_likelihood_order",
        )
    )
    if likelihood_order != rv_columns:
        raise ValueError("Path C joint R^V likelihood order must exactly match rv_columns.")
    _string_tuple(rv_spec["public_context_strata"], "rv_summary_spec.public_context_strata")

    fingerprint = _mapping(payload["fingerprint"], "fingerprint")
    _require_mapping_fields(
        fingerprint,
        (
            "enable", "kind", "vocab", "positive_control_kind",
            "negative_control_kind", "admission_required",
            "counterbalanced_across_mechanisms", "require_raw_visibility",
            "require_value_null", "require_joint_rv_null",
            "require_residual_signature_null",
        ),
        "fingerprint",
    )
    if fingerprint["enable"] is not True:
        raise ValueError("Path C frozen fingerprint.enable must be true.")
    _string_tuple(fingerprint["vocab"], "fingerprint.vocab")
    for key in (
        "admission_required", "counterbalanced_across_mechanisms",
        "require_raw_visibility", "require_value_null", "require_joint_rv_null",
        "require_residual_signature_null",
    ):
        if fingerprint[key] is not True:
            raise ValueError(f"Path C fingerprint.{key} must be true.")

    split = _mapping(payload["cross_identity_split"], "cross_identity_split")
    _require_mapping_fields(
        split,
        ("enable", "group_key", "identity_key", "split", "folds"),
        "cross_identity_split",
    )
    if split["enable"] is not True:
        raise ValueError("Path C frozen cross_identity_split.enable must be true.")
    if split["identity_key"] != "surface_identity_key":
        raise ValueError("Path C cross-identity split must use surface_identity_key.")
    if split["split"] != "same_mechanism_disjoint_surface_seed_layout":
        raise ValueError("Path C cross-identity split policy is not the frozen grouped policy.")
    if int(split["folds"]) <= 1:
        raise ValueError("Path C cross-identity split requires at least two folds.")

    readout = _mapping(payload["readout"], "readout")
    for key in ("model", "hidden", "epochs", "batch", "folds", "split_unit", "rv_likelihood"):
        if readout.get(key) is None:
            raise ValueError(f"Path C frozen preregistration readout is missing: {key}")
    if readout["rv_likelihood"] != "autoregressive_joint":
        raise ValueError("Path C R^V readout must use autoregressive_joint likelihood.")
    if str(readout["split_unit"]) != "episode_uid":
        raise ValueError("Path C readout.split_unit must be 'episode_uid'.")

    go_no_go = _mapping(payload["go_no_go"], "go_no_go")
    _require_mapping_fields(
        go_no_go,
        ("logic", "hard_fail_margin", "threshold_source"),
        "go_no_go",
    )
    if go_no_go["threshold_source"] != "artifact_read_back":
        raise ValueError("Path C go/no-go thresholds must use artifact_read_back.")

    runtime_contract = _runtime_contract(payload)
    return FrozenPathCPreregistration(
        path=source,
        sha256=file_sha256(source),
        payload=payload,
        thresholds=dict(thresholds),
        required_baselines=required_baselines,
        strong_baselines=strong_baselines,
        runtime_contract=runtime_contract,
        runtime_contract_sha256=canonical_sha256(runtime_contract),
    )


def validate_runtime_path_c_config(
    runtime_section: Mapping[str, Any],
    preregistration: FrozenPathCPreregistration,
) -> str:
    observed = _runtime_contract(runtime_section)
    if observed != preregistration.runtime_contract:
        mismatches = []
        for section, expected_values in preregistration.runtime_contract.items():
            observed_values = observed.get(section, {})
            for key, expected in expected_values.items():
                actual = observed_values.get(key)
                if actual != expected:
                    mismatches.append(f"{section}.{key}: runtime={actual!r}, frozen={expected!r}")
        raise ValueError(
            "Runtime Path C config does not match the frozen preregistration: "
            + "; ".join(mismatches)
        )
    return canonical_sha256(observed)


def load_and_validate_path_c_inputs(
    preregistration_path: str | Path,
    manifest_path: str | Path,
    *,
    resolved_path_c: Mapping[str, Any] | None = None,
) -> ValidatedPathCInputs:
    preregistration = load_frozen_preregistration(preregistration_path)
    if resolved_path_c is not None:
        resolved_sha = validate_runtime_path_c_config(resolved_path_c, preregistration)
    else:
        resolved_sha = preregistration.runtime_contract_sha256

    source = Path(manifest_path).resolve()
    payload = _load_mapping(source, "Path C artifact manifest")
    if payload.get("schema_version") != PATH_C_ARTIFACT_SCHEMA:
        raise ValueError(
            f"Path C artifact manifest schema_version must be {PATH_C_ARTIFACT_SCHEMA!r}."
        )
    _reject_authoritative_pass_fields(payload, "manifest")
    prereg_ref = _mapping(payload.get("preregistration"), "manifest.preregistration")
    if prereg_ref.get("sha256") != preregistration.sha256:
        raise ValueError("Path C artifact manifest uses a different preregistration SHA-256.")
    if prereg_ref.get("version") != preregistration.payload.get("version"):
        raise ValueError("Path C artifact manifest uses a different preregistration version.")
    if payload.get("resolved_path_c_sha256") != resolved_sha:
        raise ValueError("Path C artifact manifest resolved config SHA-256 does not match runtime.")

    schemas = _mapping(payload.get("schemas"), "manifest.schemas")
    _require_mapping_fields(
        schemas,
        ("graph_sha256", "option_schema_sha256", "observation_schema_sha256"),
        "manifest.schemas",
    )
    for key, value in schemas.items():
        if not _is_sha256(value):
            raise ValueError(f"manifest.schemas.{key} must be a SHA-256 digest.")

    datasets = _mapping(payload.get("datasets"), "manifest.datasets")
    checkpoints = _mapping(payload.get("checkpoints"), "manifest.checkpoints")
    measurement_refs = _mapping(
        payload.get("measurement_artifacts"),
        "manifest.measurement_artifacts",
    )
    _validate_dataset_entries(datasets, preregistration, source.parent)
    _validate_checkpoint_entries(
        checkpoints,
        preregistration,
        datasets,
        schemas,
        source.parent,
    )
    _validate_dataset_checkpoint_bindings(datasets, checkpoints)

    value_control_ref = _mapping(
        payload.get("value_control_artifact"),
        "manifest.value_control_artifact",
    )
    value_control_path = _validated_file_reference(
        value_control_ref,
        source.parent,
        "manifest.value_control_artifact",
    )
    value_control = _load_mapping(value_control_path, "Path C value-control artifact")
    _reject_authoritative_pass_fields(value_control, "value_control_artifact")
    _validate_bound_artifact(
        value_control,
        preregistration,
        resolved_sha,
        "value_control_artifact",
    )
    _require_mapping_fields(
        value_control,
        (
            "source_dataset_keys", "schemas", "required_baselines",
            "budget_matched", "base_only_calibration_sha256_by_split",
            "provenance_columns", "held_out", "main", "baselines",
        ),
        "value_control_artifact",
    )
    value_source_keys = _string_tuple(
        value_control["source_dataset_keys"],
        "value_control_artifact.source_dataset_keys",
    )
    missing_value_sources = sorted(set(value_source_keys).difference(datasets))
    if missing_value_sources:
        raise ValueError(
            "Path C value-control artifact refers to missing datasets: "
            + ", ".join(missing_value_sources)
        )
    if _mapping(value_control["schemas"], "value_control_artifact.schemas") != schemas:
        raise ValueError("Path C value-control artifact uses different graph/option/observation schemas.")
    if tuple(map(str, _sequence(
        value_control["required_baselines"],
        "value_control_artifact.required_baselines",
    ))) != preregistration.required_baselines:
        raise ValueError("Path C value-control artifact changes the required baseline order.")
    if value_control["budget_matched"] is not True:
        raise ValueError("Path C value-control artifact is not matched-budget.")
    value_calibration = _mapping(
        value_control["base_only_calibration_sha256_by_split"],
        "value_control_artifact.base_only_calibration_sha256_by_split",
    )
    base_checkpoint = _mapping(checkpoints.get("base_only"), "checkpoints.base_only")
    expected_calibration = {
        str(_mapping(ref, "base calibration")["evaluation_split_id"]): str(
            _mapping(ref, "base calibration")["sha256"]
        )
        for ref in _sequence(
            base_checkpoint.get("calibration_predictions"),
            "checkpoints.base_only.calibration_predictions",
        )
    }
    value_split_ids = {
        str(_mapping(datasets[key], f"datasets.{key}")["split_id"])
        for key in value_source_keys
        if str(_mapping(datasets[key], f"datasets.{key}").get("collection_role"))
        in {"readout_evaluation", "synthetic_power", "terminal_axis_null", "recovery"}
    }
    if any(
        str(value_calibration.get(split_id)) != expected_calibration.get(split_id)
        for split_id in value_split_ids
    ):
        raise ValueError("Path C value-control artifact is not bound to BaseOnly fold predictions.")
    provenance_columns = tuple(
        str(value).lower()
        for value in _sequence(
            value_control.get("provenance_columns"),
            "value_control_artifact.provenance_columns",
        )
    )
    rejected_value_proxies = sorted(
        column
        for column in provenance_columns
        if column.startswith("rv_")
        or column.startswith("resp_rv")
        or "_rv_" in column
        or "response_summary" in column
    )
    if rejected_value_proxies:
        raise ValueError(
            "Path C value-control artifact uses R^V-derived proxy columns: "
            + ", ".join(rejected_value_proxies)
        )
    if value_control.get("held_out") is not True:
        raise ValueError("Path C value-control artifact must be held out.")
    value_main = _mapping(value_control.get("main"), "value_control_artifact.main")
    _aggregate_value_control(
        value_main,
        preregistration.required_baselines,
        "value_control_artifact.main",
    )
    value_baselines = _mapping(
        value_control.get("baselines"),
        "value_control_artifact.baselines",
    )
    missing_value_baselines = sorted(
        set(preregistration.strong_baselines).difference(value_baselines)
    )
    if missing_value_baselines:
        raise ValueError(
            "Path C value-control artifact lacks strong-baseline measurements: "
            + ", ".join(missing_value_baselines)
        )
    for baseline in preregistration.strong_baselines:
        _aggregate_value_control(
            _mapping(value_baselines[baseline], f"value_control_artifact.baselines.{baseline}"),
            preregistration.required_baselines,
            f"value_control_artifact.baselines.{baseline}",
        )

    required_measurements = {PATH_C_PRIMARY_VARIANT, *preregistration.strong_baselines}
    missing_measurements = sorted(required_measurements.difference(measurement_refs))
    if missing_measurements:
        raise ValueError(
            "Path C artifact manifest is missing measurement artifact(s): "
            + ", ".join(missing_measurements)
        )
    measurements: dict[str, dict[str, Any]] = {}
    for variant, reference in measurement_refs.items():
        ref = _mapping(reference, f"measurement_artifacts.{variant}")
        artifact_path = _validated_file_reference(ref, source.parent, f"measurement_artifacts.{variant}")
        measurement = _load_mapping(artifact_path, f"Path C measurement artifact {variant}")
        _reject_authoritative_pass_fields(measurement, f"measurement_artifacts.{variant}")
        if measurement.get("variant") != variant:
            raise ValueError(f"Path C measurement artifact {variant} has the wrong variant field.")
        if measurement.get("preregistration_sha256") != preregistration.sha256:
            raise ValueError(f"Path C measurement artifact {variant} has the wrong preregistration SHA-256.")
        if measurement.get("resolved_path_c_sha256") != resolved_sha:
            raise ValueError(f"Path C measurement artifact {variant} has the wrong config SHA-256.")
        _validate_measurement_provenance(
            measurement,
            datasets,
            checkpoints,
            variant,
            preregistration,
        )
        measurements[str(variant)] = measurement

    measurements["value_control"] = value_control

    manifest = PathCArtifactManifest(
        path=source,
        sha256=file_sha256(source),
        payload=payload,
        preregistration_sha256=preregistration.sha256,
        resolved_path_c_sha256=resolved_sha,
        datasets={str(k): dict(v) for k, v in datasets.items()},
        checkpoints={str(k): dict(v) for k, v in checkpoints.items()},
        measurement_artifacts={str(k): dict(v) for k, v in measurement_refs.items()},
        value_control_artifact=dict(value_control_ref),
    )
    return ValidatedPathCInputs(preregistration, manifest, measurements)


def artifact_contract_measurement(inputs: ValidatedPathCInputs) -> dict[str, Any]:
    prereg = inputs.preregistration
    datasets = inputs.manifest.datasets
    budgets = {
        name: {
            "effective_episodes": int(item["effective_episodes"]),
            "effective_transitions": int(item["effective_transitions"]),
        }
        for name, item in datasets.items()
    }
    return {
        "preregistration_sha256": prereg.sha256,
        "resolved_path_c_sha256": inputs.manifest.resolved_path_c_sha256,
        "present_baselines": sorted(inputs.manifest.checkpoints),
        "required_baselines": list(prereg.required_baselines),
        "hashes_valid": True,
        "split_matched": True,
        "capacity_matched": True,
        "budgets": budgets,
        "dataset_splits": {
            name: {
                "split_id": str(item["split_id"]),
                "collection_variant": str(item["collection_variant"]),
                "collection_role": str(item["collection_role"]),
                "cross_identity_split": bool(item["cross_identity_split"]),
            }
            for name, item in datasets.items()
        },
        "schemas": dict(inputs.manifest.payload["schemas"]),
        "value_control_artifact_sha256": inputs.manifest.value_control_artifact["sha256"],
    }


def phase_b_go_no_go_rule(
    measurements: Mapping[str, Any],
    preregistration: FrozenPathCPreregistration,
) -> dict[str, Any]:
    main_rules = _phase_rules(measurements, preregistration, require_contract=True)
    main_ready = all(bool(rule.get("pass", False)) for rule in main_rules.values())

    strong_payload = measurements.get("strong_baselines")
    strong_measurements = strong_payload if isinstance(strong_payload, Mapping) else {}
    missing_strong = [name for name in preregistration.strong_baselines if name not in strong_measurements]
    strong_results: dict[str, Any] = {}
    strong_passes: list[str] = []
    for name in preregistration.strong_baselines:
        item = strong_measurements.get(name)
        if not isinstance(item, Mapping):
            continue
        rules = _phase_rules(item, preregistration, require_contract=False)
        passed = all(bool(rule.get("pass", False)) for rule in rules.values())
        strong_results[name] = {"rules": rules, "same_rule_pass": passed}
        if passed:
            strong_passes.append(name)

    strong_audit_complete = not missing_strong
    status = "NO-GO"
    if main_ready and strong_audit_complete:
        status = "DEMOTED" if strong_passes else "GO"
    return {
        "status": status,
        "rules": main_rules,
        "strong_baseline_audit": {
            "complete": strong_audit_complete,
            "missing": missing_strong,
            "results": strong_results,
        },
        "strong_baseline_passes": strong_passes,
        "preregistration_sha256": preregistration.sha256,
        "requires_type_b_review": True,
    }


def pass_af_claim_rule(
    measurements: Mapping[str, Any],
    preregistration: FrozenPathCPreregistration,
) -> dict[str, Any]:
    phase = phase_b_go_no_go_rule(measurements, preregistration)
    c_min = _value_minimality_rule(measurements.get("C_min"), preregistration)
    c_leak = _leakage_rule(
        measurements.get("C_leak_full"),
        preregistration.thresholds["full_leakage_max_advantage"],
        "C_leak_full",
    )
    ready = phase["status"] in {"GO", "DEMOTED"} and c_min["pass"] and c_leak["pass"]
    if not ready:
        status = "NOT_READY_FOR_TYPE_B_REVIEW"
    elif phase["status"] == "DEMOTED":
        status = "DEMOTED_FOR_TYPE_B_REVIEW"
    else:
        status = "READY_FOR_TYPE_B_REVIEW"
    return {
        "status": status,
        "phase_b": phase,
        "rules": {"C_min": c_min, "C_leak_full": c_leak},
        "requires_type_b_review": True,
        "preregistration_sha256": preregistration.sha256,
    }


def select_value_necessary_coordinates(
    coordinate_measurements: Mapping[str, Mapping[str, Any]],
    preregistration: FrozenPathCPreregistration,
) -> dict[str, Any]:
    threshold = float(preregistration.thresholds["value_ablation_min_degradation"])
    retained = []
    rejected = []
    details = {}
    for name, item in sorted(coordinate_measurements.items()):
        raw_metrics = item.get("metrics")
        metrics = raw_metrics if isinstance(raw_metrics, Mapping) else {"aggregate": item}
        metric_details = {}
        metric_passes = []
        for metric_name, metric_item in sorted(metrics.items()):
            if not isinstance(metric_item, Mapping):
                raise ValueError(f"coordinate {name}.{metric_name} must be a mapping.")
            value = _finite_number(
                metric_item.get("value"),
                f"coordinate {name}.{metric_name}.value",
            )
            ci_lo = _finite_number(
                metric_item.get("ci_lo", _nested_ci(metric_item, "lo")),
                f"coordinate {name}.{metric_name}.ci_lo",
            )
            passed = value >= threshold and ci_lo > threshold
            metric_passes.append(passed)
            metric_details[str(metric_name)] = {
                "value": value,
                "ci_lo": ci_lo,
                "threshold": threshold,
                "necessary": passed,
            }
        necessary = any(metric_passes)
        details[name] = {"metrics": metric_details, "necessary": necessary}
        (retained if necessary else rejected).append(name)
    return {
        "retained_coordinates": retained,
        "rejected_coordinates": rejected,
        "retained_coordinate_count": len(retained),
        "details": details,
    }


def fingerprint_admission_measurement(
    *,
    control_kind: str,
    mechanism: Sequence[Any],
    fingerprint_id: Sequence[Any],
    visibility_ci_lo: float,
    chance_accuracy: float,
    value_null_ci: Sequence[float],
    joint_rv_null_ci: Sequence[float],
    residual_null_ci: Sequence[float],
    preregistration: FrozenPathCPreregistration,
) -> dict[str, Any]:
    mechanisms = np.asarray(mechanism).astype(str)
    fingerprints = np.asarray(fingerprint_id).astype(str)
    if mechanisms.shape[0] != fingerprints.shape[0]:
        raise ValueError("Fingerprint mechanism and id arrays must have the same length.")
    expected_kind = str(preregistration.payload["fingerprint"]["positive_control_kind"])
    mechanism_values = sorted(np.unique(mechanisms).tolist())
    fingerprint_values = sorted(np.unique(fingerprints).tolist())
    counterbalanced = bool(
        len(mechanism_values) >= 2
        and len(fingerprint_values) >= 2
        and all(
            set(fingerprints[mechanisms == mechanism_value].tolist())
            == set(fingerprint_values)
            for mechanism_value in mechanism_values
        )
    )
    thresholds = preregistration.thresholds
    visibility = float(visibility_ci_lo)
    chance = float(chance_accuracy)
    value_interval = _interval(value_null_ci)
    rv_interval = _interval(joint_rv_null_ci)
    residual_interval = _interval(residual_null_ci)
    admitted = bool(
        control_kind == expected_kind
        and counterbalanced
        and visibility > chance + float(thresholds["raw_fingerprint_balanced_accuracy_margin"])
        and _interval_inside(
            value_interval,
            -float(thresholds["fingerprint_value_null_equivalence"]),
            float(thresholds["fingerprint_value_null_equivalence"]),
        )
        and _interval_inside(
            rv_interval,
            -float(thresholds["fingerprint_joint_rv_null_equivalence"]),
            float(thresholds["fingerprint_joint_rv_null_equivalence"]),
        )
        and _interval_inside(
            residual_interval,
            -float(thresholds["fingerprint_residual_null_equivalence"]),
            float(thresholds["fingerprint_residual_null_equivalence"]),
        )
    )
    return {
        "fingerprint_control_kind": str(control_kind),
        "expected_positive_control_kind": expected_kind,
        "fingerprint_cross_mechanism_counterbalanced": counterbalanced,
        "fingerprint_visibility_ci_lo": visibility,
        "fingerprint_chance": chance,
        "fingerprint_value_null_ci": value_interval,
        "fingerprint_joint_rv_null_ci": rv_interval,
        "fingerprint_residual_null_ci": residual_interval,
        "admitted": admitted,
        "available_for_synthetic_power": admitted,
    }


def build_path_c_readout_features(
    public_context: np.ndarray,
    representations: Mapping[str, np.ndarray],
    *,
    surface_identity: Sequence[Any] | None = None,
) -> dict[str, np.ndarray]:
    """Construct fair readout inputs for main and hard-baseline variants."""
    context = np.asarray(public_context, dtype=np.float32)
    if context.ndim != 2:
        raise ValueError("Path C public context must be a matrix.")
    features: dict[str, np.ndarray] = {
        "base_only": context.copy(),
        "no_admission": context.copy(),
    }
    for variant, raw in representations.items():
        representation = np.asarray(raw, dtype=np.float32)
        if representation.ndim != 2 or representation.shape[0] != context.shape[0]:
            raise ValueError(f"Path C representation {variant!r} is not row-aligned.")
        if variant in {"base_only", "no_admission"}:
            continue
        if variant == "partner_id":
            if surface_identity is None:
                raise ValueError("Partner-ID baseline requires explicit surface identity.")
            identities = np.asarray(surface_identity).astype(str)
            if identities.shape[0] != context.shape[0]:
                raise ValueError("Partner-ID surface identity is not row-aligned.")
            vocab, labels = np.unique(identities, return_inverse=True)
            identity_onehot = np.zeros((context.shape[0], len(vocab)), dtype=np.float32)
            identity_onehot[np.arange(context.shape[0]), labels] = 1.0
            features[variant] = np.concatenate([context, identity_onehot], axis=1)
        else:
            features[variant] = np.concatenate([context, representation], axis=1)
    return features


def require_formal_benchmark_artifact(artifact: Mapping[str, Any]) -> None:
    if artifact.get("evaluation_kind") != "formal_benchmark":
        raise ValueError("Data-collection output cannot be used as formal benchmark return.")
    if artifact.get("active_probe_collection") is not False:
        raise ValueError("Formal benchmark return must disable active probe collection.")
    if artifact.get("collection_role") != "formal_benchmark":
        raise ValueError("Formal benchmark return has an invalid collection role.")
    if artifact.get("benchmark_return_eligible") is not True:
        raise ValueError("Artifact is not eligible for benchmark-return aggregation.")


def probe_support_measurement(
    selected: Sequence[Any],
    opportunity: Sequence[Any],
    public_context: Sequence[Any],
    action_id: Sequence[Any],
) -> dict[str, Any]:
    selected_arr = np.asarray(selected).astype(bool)
    opportunity_arr = np.asarray(opportunity).astype(bool)
    contexts = np.asarray(public_context).astype(str)
    actions = np.asarray(action_id).astype(str)
    if len({selected_arr.shape[0], opportunity_arr.shape[0], contexts.shape[0], actions.shape[0]}) != 1:
        raise ValueError("Probe support arrays must have the same length.")
    if np.any(selected_arr & ~opportunity_arr):
        raise ValueError("A selected probe must also be a recorded opportunity.")
    opportunity_contexts = set(contexts[opportunity_arr].tolist())
    selected_contexts = set(contexts[selected_arr].tolist())
    opportunity_actions = set(actions[opportunity_arr].tolist())
    selected_actions = set(actions[selected_arr].tolist())
    return {
        "selected": int(np.count_nonzero(selected_arr)),
        "opportunities": int(np.count_nonzero(opportunity_arr)),
        "context_coverage": float(
            len(selected_contexts) / max(1, len(opportunity_contexts))
        ),
        "action_coverage": float(
            len(selected_actions) / max(1, len(opportunity_actions))
        ),
        "public_context_count": len(opportunity_contexts),
        "eligible_action_count": len(opportunity_actions),
    }


def make_run_id(
    *,
    checkpoint_sha256: str,
    resolved_path_c_sha256: str,
    collection_variant: str,
    partner: str,
    ego: str,
    base_seed: int,
) -> str:
    return canonical_sha256({
        "checkpoint_sha256": checkpoint_sha256,
        "resolved_path_c_sha256": resolved_path_c_sha256,
        "collection_variant": collection_variant,
        "partner": partner,
        "ego": ego,
        "base_seed": int(base_seed),
    })[:24]


def make_episode_uid(run_id: str, base_seed: int, episode_id: int) -> str:
    return f"{run_id}:{int(base_seed)}:{int(episode_id)}"


def assign_group_disjoint_folds(
    surface_identity: Sequence[Any],
    seed_group: Sequence[Any],
    layout_style: Sequence[Any],
    *,
    n_folds: int,
) -> np.ndarray:
    """Assign connected surface/seed/layout groups to deterministic folds."""
    surfaces, seeds, layouts = (
        np.asarray(values).astype(str)
        for values in (surface_identity, seed_group, layout_style)
    )
    if len({surfaces.shape[0], seeds.shape[0], layouts.shape[0]}) != 1:
        raise ValueError("Cross-identity grouping arrays must have the same length.")
    if int(n_folds) <= 1:
        raise ValueError("Cross-identity split requires at least two folds.")
    n_rows = int(surfaces.shape[0])
    parent = list(range(n_rows))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for values in (surfaces, seeds, layouts):
        first: dict[str, int] = {}
        for index, value in enumerate(values.tolist()):
            if value in first:
                union(first[value], index)
            else:
                first[value] = index

    components: dict[int, list[int]] = {}
    for index in range(n_rows):
        components.setdefault(find(index), []).append(index)
    if len(components) < int(n_folds):
        raise ValueError(
            "Cross-identity data has fewer disjoint surface/seed/layout components than folds."
        )
    ordered = sorted(
        components.values(),
        key=lambda rows: (
            -len(rows),
            canonical_sha256({
                "surface": sorted(set(surfaces[rows].tolist())),
                "seed": sorted(set(seeds[rows].tolist())),
                "layout": sorted(set(layouts[rows].tolist())),
            }),
        ),
    )
    fold_sizes = [0] * int(n_folds)
    assignments = np.empty(n_rows, dtype=np.int16)
    for rows in ordered:
        fold = min(range(int(n_folds)), key=lambda value: (fold_sizes[value], value))
        assignments[rows] = fold
        fold_sizes[fold] += len(rows)
    return assignments


def validate_cross_identity_folds(
    fold_ids: Sequence[Any],
    surface_identity: Sequence[Any],
    seed_group: Sequence[Any],
    layout_style: Sequence[Any],
    mechanism: Sequence[Any],
) -> dict[str, Any]:
    arrays = [np.asarray(values).astype(str) for values in (
        fold_ids, surface_identity, seed_group, layout_style, mechanism
    )]
    if len({arr.shape[0] for arr in arrays}) != 1:
        raise ValueError("Cross-identity split arrays must have the same length.")
    folds, surfaces, seeds, layouts, mechanisms = arrays
    reports = []
    valid = True
    for fold in sorted(np.unique(folds).tolist()):
        test = folds == fold
        train = ~test
        overlap = {
            "surface_identity": sorted(set(surfaces[train]).intersection(surfaces[test])),
            "seed_group": sorted(set(seeds[train]).intersection(seeds[test])),
            "layout_style": sorted(set(layouts[train]).intersection(layouts[test])),
        }
        mechanism_overlap = sorted(set(mechanisms[train]).intersection(mechanisms[test]))
        fold_valid = not any(overlap.values()) and bool(mechanism_overlap)
        valid = valid and fold_valid
        reports.append({
            "fold": str(fold),
            "overlap": overlap,
            "mechanism_overlap": mechanism_overlap,
            "valid": fold_valid,
        })
    return {"valid": valid, "folds": reports}


def empirical_kernel_distance_audit(
    joint_rv: Sequence[Any],
    mechanism: Sequence[Any],
    surface_identity: Sequence[Any],
    public_context: Sequence[Any],
    episode_uid: Sequence[Any],
    preregistration: FrozenPathCPreregistration,
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """Estimate K^V distances with the proposal's max/min witness quantifiers."""
    rv = np.asarray(joint_rv)
    if rv.ndim == 1:
        rv = rv.reshape(-1, 1)
    mechanisms, identities, contexts, episodes = (
        np.asarray(values).astype(str)
        for values in (mechanism, surface_identity, public_context, episode_uid)
    )
    n_rows = rv.shape[0]
    if any(values.shape[0] != n_rows for values in (mechanisms, identities, contexts, episodes)):
        raise ValueError("Kernel audit arrays must have the same row count.")
    min_rows = int(preregistration.thresholds["kernel_min_context_rows"])
    mechanism_values = sorted(np.unique(mechanisms).tolist())
    if len(mechanism_values) < 2:
        return _unavailable_kernel_audit("fewer_than_two_mechanisms")

    same_specs: list[tuple[str, str, str, str]] = []
    for mech in mechanism_values:
        mech_mask = mechanisms == mech
        identity_values = sorted(np.unique(identities[mech_mask]).tolist())
        for left_index, left in enumerate(identity_values):
            for right in identity_values[left_index + 1:]:
                for context in sorted(np.unique(contexts[mech_mask]).tolist()):
                    left_mask = mech_mask & (identities == left) & (contexts == context)
                    right_mask = mech_mask & (identities == right) & (contexts == context)
                    if int(left_mask.sum()) >= min_rows and int(right_mask.sum()) >= min_rows:
                        same_specs.append((mech, left, right, context))

    different_specs: list[tuple[str, str, str]] = []
    required_mechanism_pairs: list[tuple[str, str]] = []
    for left_index, left in enumerate(mechanism_values):
        for right in mechanism_values[left_index + 1:]:
            required_mechanism_pairs.append((left, right))
            shared_contexts = sorted(
                set(contexts[mechanisms == left]).intersection(contexts[mechanisms == right])
            )
            for context in shared_contexts:
                left_mask = (mechanisms == left) & (contexts == context)
                right_mask = (mechanisms == right) & (contexts == context)
                if int(left_mask.sum()) >= min_rows and int(right_mask.sum()) >= min_rows:
                    different_specs.append((left, right, context))

    covered_pairs = {(left, right) for left, right, _ in different_specs}
    missing_pairs = [pair for pair in required_mechanism_pairs if pair not in covered_pairs]
    if not same_specs:
        return _unavailable_kernel_audit("no_same_mechanism_identity_comparisons")
    if missing_pairs:
        result = _unavailable_kernel_audit("sparse_public_context_strata")
        result["missing_mechanism_pairs"] = [list(pair) for pair in missing_pairs]
        return result

    joint_tokens = np.asarray([
        "|".join(map(str, np.ravel(row).tolist()))
        for row in rv
    ])

    def same_value(spec: tuple[str, str, str, str], indices: np.ndarray) -> float:
        mech, left, right, context = spec
        selected = indices
        left_values = joint_tokens[selected][
            (mechanisms[selected] == mech)
            & (identities[selected] == left)
            & (contexts[selected] == context)
        ]
        right_values = joint_tokens[selected][
            (mechanisms[selected] == mech)
            & (identities[selected] == right)
            & (contexts[selected] == context)
        ]
        return _categorical_total_variation(left_values, right_values)

    def different_value(spec: tuple[str, str, str], indices: np.ndarray) -> float:
        left, right, context = spec
        selected = indices
        left_values = joint_tokens[selected][
            (mechanisms[selected] == left) & (contexts[selected] == context)
        ]
        right_values = joint_tokens[selected][
            (mechanisms[selected] == right) & (contexts[selected] == context)
        ]
        return _categorical_total_variation(left_values, right_values)

    full_indices = np.arange(n_rows, dtype=np.int64)
    observed_same = np.asarray([same_value(spec, full_indices) for spec in same_specs])
    observed_different = np.asarray([
        different_value(spec, full_indices) for spec in different_specs
    ])

    unique_episodes = np.unique(episodes)
    episode_rows = {
        episode: np.flatnonzero(episodes == episode)
        for episode in unique_episodes.tolist()
    }
    iterations = int(preregistration.thresholds["kernel_bootstrap_iters"])
    rng = np.random.default_rng(int(seed))
    max_errors: list[float] = []
    for _ in range(iterations):
        sampled_episodes = rng.choice(unique_episodes, size=len(unique_episodes), replace=True)
        sampled_indices = np.concatenate([
            episode_rows[str(episode)] for episode in sampled_episodes.astype(str).tolist()
        ])
        same_boot = np.asarray([same_value(spec, sampled_indices) for spec in same_specs])
        different_boot = np.asarray([
            different_value(spec, sampled_indices) for spec in different_specs
        ])
        if not np.all(np.isfinite(same_boot)) or not np.all(np.isfinite(different_boot)):
            return _unavailable_kernel_audit("bootstrap_lost_required_stratum")
        max_errors.append(float(max(
            np.max(np.abs(same_boot - observed_same)),
            np.max(np.abs(different_boot - observed_different)),
        )))
    confidence = float(preregistration.thresholds["kernel_confidence"])
    delta = float(np.quantile(np.asarray(max_errors), confidence))

    same_reports = []
    for spec, value in zip(same_specs, observed_same, strict=True):
        same_reports.append({
            "mechanism": spec[0],
            "surface_pair": [spec[1], spec[2]],
            "public_context": spec[3],
            "tv": float(value),
            "upper_ci": float(min(1.0, value + delta)),
        })
    different_reports = []
    for spec, value in zip(different_specs, observed_different, strict=True):
        different_reports.append({
            "mechanism_pair": [spec[0], spec[1]],
            "public_context": spec[2],
            "tv": float(value),
            "lower_ci": float(max(0.0, value - delta)),
        })
    pair_witnesses = []
    for pair in required_mechanism_pairs:
        candidates = [
            report for report in different_reports
            if tuple(report["mechanism_pair"]) == pair
        ]
        witness = max(candidates, key=lambda report: report["lower_ci"])
        pair_witnesses.append(witness)

    return {
        "available": True,
        "delta": delta,
        "max_same_W_upper_ci": max(report["upper_ci"] for report in same_reports),
        "min_diff_W_witness_lower_ci": min(
            report["lower_ci"] for report in pair_witnesses
        ),
        "same_mechanism_quantifier": "max_over_identity_context",
        "different_mechanism_quantifier": "min_pair_max_witness",
        "bootstrap_unit": "episode_uid",
        "simultaneous_bootstrap": True,
        "context_strata_source": "preregistered_semantic",
        "all_required_strata_available": True,
        "same_identity_context_comparisons": same_reports,
        "different_mechanism_context_comparisons": different_reports,
        "mechanism_pair_witnesses": pair_witnesses,
        "bootstrap_iterations": iterations,
        "confidence": confidence,
    }


def _unavailable_kernel_audit(reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "reason": reason,
        "same_mechanism_quantifier": "max_over_identity_context",
        "different_mechanism_quantifier": "min_pair_max_witness",
        "bootstrap_unit": "episode_uid",
        "simultaneous_bootstrap": True,
        "context_strata_source": "preregistered_semantic",
        "all_required_strata_available": False,
    }


def _categorical_total_variation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return float("nan")
    vocab = sorted(set(left.astype(str).tolist()).union(right.astype(str).tolist()))
    left_counts = np.asarray([np.count_nonzero(left == value) for value in vocab], dtype=np.float64)
    right_counts = np.asarray([np.count_nonzero(right == value) for value in vocab], dtype=np.float64)
    left_probs = left_counts / left_counts.sum()
    right_probs = right_counts / right_counts.sum()
    return float(0.5 * np.abs(left_probs - right_probs).sum())


def _phase_rules(
    measurements: Mapping[str, Any],
    preregistration: FrozenPathCPreregistration,
    *,
    require_contract: bool,
) -> dict[str, Any]:
    thresholds = preregistration.thresholds
    rules = {
        "G1": _lower_interval_rule(
            _measurement(measurements, "G1", "C_resp^V"),
            thresholds["G1"],
            "G1",
            required_comparisons=preregistration.required_baselines,
        ),
        "G_value": _lower_interval_rule(_measurement(measurements, "G_value", "C_value"), thresholds["G_value"], "G_value"),
        "G2": _lower_interval_rule(
            _measurement(measurements, "G2", "C_transfer"),
            thresholds["G2"],
            "G2",
            required_comparisons=preregistration.required_baselines,
        ),
        "G3": _leakage_rule(
            _measurement(measurements, "G3", "C_leak"),
            thresholds["G3"],
            "G3",
        ),
        "G4": _power_null_rule(_measurement(measurements, "G4", "C_null"), preregistration),
        "kernel_margin": _kernel_rule(measurements.get("C_fact"), preregistration),
        "probe_support": _probe_support_rule(measurements.get("probe_support"), preregistration),
    }
    if require_contract:
        rules["artifact_contract"] = _artifact_contract_rule(
            measurements.get("artifact_contract"), preregistration
        )
    return rules


def _lower_interval_rule(
    item: Any,
    threshold: Any,
    name: str,
    *,
    required_comparisons: Sequence[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(item, Mapping):
        return {"pass": False, "missing": True, "name": name}
    normalized_comparisons = None
    missing_comparisons: list[str] = []
    if required_comparisons is not None:
        comparisons = item.get("comparisons") if isinstance(item.get("comparisons"), Mapping) else {}
        missing_comparisons = sorted(set(required_comparisons).difference(comparisons))
        normalized_comparisons = {}
        values = []
        lower_bounds = []
        for baseline in required_comparisons:
            measurement = comparisons.get(baseline)
            if not isinstance(measurement, Mapping):
                continue
            comparison_value = _optional_finite_number(measurement.get("value"))
            comparison_ci = _optional_finite_number(
                measurement.get("ci_lo", _nested_ci(measurement, "lo"))
            )
            normalized_comparisons[str(baseline)] = {
                "value": comparison_value,
                "ci_lo": comparison_ci,
            }
            if comparison_value is not None and comparison_ci is not None:
                values.append(comparison_value)
                lower_bounds.append(comparison_ci)
        value = min(values) if len(values) == len(required_comparisons) else None
        ci_lo = min(lower_bounds) if len(lower_bounds) == len(required_comparisons) else None
    else:
        value = _optional_finite_number(item.get("value"))
        ci_lo = _optional_finite_number(item.get("ci_lo", _nested_ci(item, "lo")))
    threshold_value = float(threshold)
    passed = bool(
        not missing_comparisons
        and value is not None
        and ci_lo is not None
        and value >= threshold_value
        and ci_lo > threshold_value
    )
    return {
        "name": name,
        "value": value,
        "ci_lo": ci_lo,
        "threshold": threshold_value,
        "missing_comparisons": missing_comparisons,
        "comparisons": normalized_comparisons,
        "pass": passed,
    }


def _upper_interval_rule(item: Any, threshold: Any, name: str) -> dict[str, Any]:
    if not isinstance(item, Mapping):
        return {"pass": False, "missing": True, "name": name}
    value = _optional_finite_number(item.get("value"))
    ci_hi = _optional_finite_number(item.get("ci_hi", _nested_ci(item, "hi")))
    threshold_value = float(threshold)
    passed = value is not None and ci_hi is not None and value <= threshold_value and ci_hi < threshold_value
    return {"name": name, "value": value, "ci_hi": ci_hi, "threshold": threshold_value, "pass": passed}


def _leakage_rule(item: Any, threshold: Any, name: str) -> dict[str, Any]:
    if not isinstance(item, Mapping):
        return {"pass": False, "missing": True, "name": name}
    threshold_value = float(threshold)
    channels = item.get("channels") if isinstance(item.get("channels"), Mapping) else {}
    missing_channels = [channel for channel in PATH_C_LEAKAGE_CHANNELS if channel not in channels]
    channel_rules = {
        channel: _upper_interval_rule(channels.get(channel), threshold_value, channel)
        for channel in PATH_C_LEAKAGE_CHANNELS
    }
    retained_only = item.get("representation_scope") == "retained_only"
    passed = bool(
        not missing_channels
        and retained_only
        and all(rule["pass"] for rule in channel_rules.values())
    )
    return {
        "name": name,
        "threshold": threshold_value,
        "representation_scope": item.get("representation_scope"),
        "missing_channels": missing_channels,
        "channels": channel_rules,
        "pass": passed,
    }


def _kernel_rule(item: Any, preregistration: FrozenPathCPreregistration) -> dict[str, Any]:
    kernel = item.get("kernel_distance_audit") if isinstance(item, Mapping) else None
    if not isinstance(kernel, Mapping):
        return {"pass": False, "missing": True}
    delta = _optional_finite_number(kernel.get("delta"))
    same_upper = _optional_finite_number(kernel.get("max_same_W_upper_ci"))
    diff_lower = _optional_finite_number(
        kernel.get("min_diff_W_witness_lower_ci", kernel.get("min_diff_W_lower_ci"))
    )
    epsilon_f = float(preregistration.thresholds["epsilon_F"])
    epsilon_r = float(preregistration.thresholds["epsilon_R"])
    margin = None if delta is None else epsilon_f + 2.0 * delta
    provenance_valid = bool(
        kernel.get("same_mechanism_quantifier") == "max_over_identity_context"
        and kernel.get("different_mechanism_quantifier") == "min_pair_max_witness"
        and kernel.get("bootstrap_unit") == "episode_uid"
        and kernel.get("simultaneous_bootstrap") is True
        and kernel.get("context_strata_source") == "preregistered_semantic"
        and kernel.get("all_required_strata_available") is True
    )
    passed = bool(
        delta is not None
        and same_upper is not None
        and diff_lower is not None
        and margin < epsilon_r
        and same_upper <= epsilon_f
        and diff_lower >= epsilon_r
        and provenance_valid
    )
    return {
        "epsilon_F": epsilon_f,
        "epsilon_R": epsilon_r,
        "delta": delta,
        "margin_value": margin,
        "max_same_W_upper_ci": same_upper,
        "min_diff_W_witness_lower_ci": diff_lower,
        "provenance_valid": provenance_valid,
        "pass": passed,
    }


def _power_null_rule(item: Any, preregistration: FrozenPathCPreregistration) -> dict[str, Any]:
    if not isinstance(item, Mapping):
        return {"pass": False, "missing": True}
    t = preregistration.thresholds
    power_value = _optional_finite_number(item.get("power_value"))
    power_ci_lo = _optional_finite_number(item.get("power_ci_lo"))
    null_ci_lo = _optional_finite_number(item.get("null_ci_lo"))
    null_ci_hi = _optional_finite_number(item.get("null_ci_hi"))
    fingerprint_ci_lo = _optional_finite_number(item.get("fingerprint_visibility_ci_lo"))
    fingerprint_chance = _optional_finite_number(item.get("fingerprint_chance"))
    fingerprint_value_null_ci = _interval(item.get("fingerprint_value_null_ci"))
    fingerprint_rv_null_ci = _interval(item.get("fingerprint_joint_rv_null_ci"))
    fingerprint_residual_null_ci = _interval(item.get("fingerprint_residual_null_ci"))
    fingerprint_kind = item.get("fingerprint_control_kind")
    fingerprint_counterbalanced = item.get("fingerprint_cross_mechanism_counterbalanced") is True
    retained_count = _optional_int(item.get("null_retained_coordinate_count"))
    same_budget = item.get("same_budget") is True
    power_threshold = max(
        float(t["G4"]),
        float(t["synthetic_power_min_mechanism_advantage"]),
    )
    null_margin = float(t["null_equivalence_gain"])
    fingerprint_margin = float(t["raw_fingerprint_balanced_accuracy_margin"])
    fingerprint_value_margin = float(t["fingerprint_value_null_equivalence"])
    fingerprint_rv_margin = float(t["fingerprint_joint_rv_null_equivalence"])
    fingerprint_residual_margin = float(t["fingerprint_residual_null_equivalence"])
    passed = bool(
        power_value is not None
        and power_ci_lo is not None
        and power_value >= power_threshold
        and power_ci_lo > power_threshold
        and null_ci_lo is not None
        and null_ci_hi is not None
        and null_ci_lo > -null_margin
        and null_ci_hi < null_margin
        and retained_count is not None
        and retained_count == 0
        and fingerprint_ci_lo is not None
        and fingerprint_chance is not None
        and fingerprint_ci_lo > fingerprint_chance + fingerprint_margin
        and fingerprint_kind == preregistration.payload["fingerprint"]["positive_control_kind"]
        and fingerprint_counterbalanced
        and _interval_inside(fingerprint_value_null_ci, -fingerprint_value_margin, fingerprint_value_margin)
        and _interval_inside(fingerprint_rv_null_ci, -fingerprint_rv_margin, fingerprint_rv_margin)
        and _interval_inside(fingerprint_residual_null_ci, -fingerprint_residual_margin, fingerprint_residual_margin)
        and same_budget
    )
    return {
        "power_value": power_value,
        "power_ci_lo": power_ci_lo,
        "power_threshold": power_threshold,
        "null_ci": [null_ci_lo, null_ci_hi],
        "null_equivalence_interval": [-null_margin, null_margin],
        "null_retained_coordinate_count": retained_count,
        "fingerprint_visibility_ci_lo": fingerprint_ci_lo,
        "fingerprint_chance": fingerprint_chance,
        "fingerprint_margin": fingerprint_margin,
        "fingerprint_control_kind": fingerprint_kind,
        "fingerprint_cross_mechanism_counterbalanced": fingerprint_counterbalanced,
        "fingerprint_value_null_ci": fingerprint_value_null_ci,
        "fingerprint_joint_rv_null_ci": fingerprint_rv_null_ci,
        "fingerprint_residual_null_ci": fingerprint_residual_null_ci,
        "same_budget": same_budget,
        "pass": passed,
    }


def _probe_support_rule(item: Any, preregistration: FrozenPathCPreregistration) -> dict[str, Any]:
    probe = preregistration.payload["probe"]
    if not bool(probe.get("enable", False)):
        return {"pass": True, "not_required": True}
    if not isinstance(item, Mapping):
        return {"pass": False, "missing": True}
    selected = _optional_int(item.get("selected"))
    opportunities = _optional_int(item.get("opportunities"))
    context_coverage = _optional_finite_number(item.get("context_coverage"))
    action_coverage = _optional_finite_number(item.get("action_coverage"))
    passed = bool(
        selected is not None
        and opportunities is not None
        and selected >= int(probe["min_selected_probes"])
        and opportunities >= int(probe["min_probe_opportunities"])
        and context_coverage is not None
        and context_coverage >= float(probe["min_context_coverage"])
        and action_coverage is not None
        and action_coverage >= float(probe["min_action_coverage"])
    )
    return {
        "selected": selected,
        "opportunities": opportunities,
        "context_coverage": context_coverage,
        "action_coverage": action_coverage,
        "pass": passed,
    }


def _artifact_contract_rule(item: Any, preregistration: FrozenPathCPreregistration) -> dict[str, Any]:
    if not isinstance(item, Mapping):
        return {"pass": False, "missing": True}
    present = set(map(str, item.get("present_baselines", [])))
    missing_baselines = sorted(set(preregistration.required_baselines).difference(present))
    budgets = item.get("budgets") if isinstance(item.get("budgets"), Mapping) else {}
    budget_cfg = preregistration.payload["budget"]
    missing_budget_keys = []
    budget_failures = []
    for key in map(str, budget_cfg["required_budget_keys"]):
        value = budgets.get(key)
        if not isinstance(value, Mapping):
            missing_budget_keys.append(key)
            continue
        episodes = _optional_int(value.get("effective_episodes"))
        transitions = _optional_int(value.get("effective_transitions"))
        if episodes is None or episodes < int(budget_cfg["effective_episode_floor"]):
            budget_failures.append(f"{key}.effective_episodes")
        if transitions is None or transitions < int(budget_cfg["effective_transition_floor"]):
            budget_failures.append(f"{key}.effective_transitions")
    passed = bool(
        not missing_baselines
        and not missing_budget_keys
        and not budget_failures
        and item.get("hashes_valid") is True
        and item.get("split_matched") is True
        and item.get("capacity_matched") is True
        and item.get("preregistration_sha256") == preregistration.sha256
        and item.get("resolved_path_c_sha256") == preregistration.runtime_contract_sha256
    )
    return {
        "missing_baselines": missing_baselines,
        "missing_budget_keys": missing_budget_keys,
        "budget_failures": budget_failures,
        "hashes_valid": item.get("hashes_valid") is True,
        "split_matched": item.get("split_matched") is True,
        "capacity_matched": item.get("capacity_matched") is True,
        "pass": passed,
    }


def _value_minimality_rule(item: Any, preregistration: FrozenPathCPreregistration) -> dict[str, Any]:
    if not isinstance(item, Mapping):
        return {"pass": False, "missing": True}
    coordinates = item.get("coordinate_measurements")
    if not isinstance(coordinates, Mapping) or not coordinates:
        return {"pass": False, "missing": True, "reason": "missing_coordinate_measurements"}
    selection = select_value_necessary_coordinates(coordinates, preregistration)
    declared_retained = sorted(map(str, item.get("retained_coordinates", [])))
    computed_retained = sorted(selection["retained_coordinates"])
    nuisance_gain = _optional_finite_number(item.get("nuisance_gain"))
    nuisance_ci_hi = _optional_finite_number(item.get("nuisance_ci_hi"))
    nuisance_limit = float(preregistration.thresholds["nuisance_value_gain_equivalence"])
    passed = bool(
        declared_retained == computed_retained
        and bool(computed_retained)
        and nuisance_gain is not None
        and nuisance_ci_hi is not None
        and nuisance_gain <= nuisance_limit
        and nuisance_ci_hi < nuisance_limit
    )
    return {
        "declared_retained_coordinates": declared_retained,
        "computed_selection": selection,
        "nuisance_gain": nuisance_gain,
        "nuisance_ci_hi": nuisance_ci_hi,
        "nuisance_threshold": nuisance_limit,
        "pass": passed,
    }


def _validate_dataset_entries(
    datasets: Mapping[str, Any],
    preregistration: FrozenPathCPreregistration,
    base_dir: Path,
) -> None:
    seen_chunk_hashes: set[str] = set()
    seen_episode_uids: set[str] = set()
    seen_run_ids: set[str] = set()
    matching_groups: dict[str, str] = {}
    required = set(map(str, preregistration.payload["budget"]["required_budget_keys"]))
    missing = sorted(required.difference(datasets))
    if missing:
        raise ValueError("Path C manifest is missing required dataset budget entries: " + ", ".join(missing))
    for name, raw in datasets.items():
        item = _mapping(raw, f"datasets.{name}")
        _require_mapping_fields(
            item,
            (
                "collection_variant", "collection_role", "split_id", "run_id", "base_seed",
                "matching_group_sha256", "schema_sha256", "required_columns",
                "optional_columns", "episode_uids", "effective_episodes",
                "effective_transitions", "chunks", "preregistration_sha256",
                "resolved_path_c_sha256", "cross_identity_split",
                "collection_mode", "controller_checkpoint_key",
                "controller_checkpoint_sha256", "rv_summary_spec_sha256",
            ),
            f"datasets.{name}",
        )
        if item["preregistration_sha256"] != preregistration.sha256:
            raise ValueError(f"Path C dataset {name} has a different preregistration SHA-256.")
        if item["resolved_path_c_sha256"] != preregistration.runtime_contract_sha256:
            raise ValueError(f"Path C dataset {name} has a different resolved config SHA-256.")
        expected_rv_sha256 = canonical_sha256(preregistration.payload["rv_summary_spec"])
        if item["rv_summary_spec_sha256"] != expected_rv_sha256:
            raise ValueError(f"Path C dataset {name} uses a different frozen R^V schema.")
        role = str(item["collection_role"])
        if role not in _ALLOWED_COLLECTION_ROLES:
            raise ValueError(f"Path C dataset {name} has unsupported collection_role={role!r}.")
        if item["collection_mode"] != "on_policy_independent":
            raise ValueError(
                f"Path C dataset {name} is an offline multi-model diagnostic, not independent collection."
            )
        if not _is_sha256(item["controller_checkpoint_sha256"]):
            raise ValueError(f"Path C dataset {name} controller checkpoint SHA-256 is invalid.")
        run_id = str(item["run_id"])
        if run_id in seen_run_ids:
            raise ValueError(f"Path C datasets reuse run_id={run_id!r}.")
        seen_run_ids.add(run_id)
        matching_group = str(item["matching_group_sha256"])
        if not _is_sha256(matching_group):
            raise ValueError(f"Path C dataset {name} matching_group_sha256 is invalid.")
        variant = str(item["collection_variant"])
        matching_groups[variant] = matching_group

        required_columns = _string_tuple(
            item["required_columns"],
            f"datasets.{name}.required_columns",
        )
        optional_columns = tuple(
            str(value) for value in _sequence(
                item["optional_columns"],
                f"datasets.{name}.optional_columns",
            )
        )
        if len(optional_columns) != len(set(optional_columns)):
            raise ValueError(f"datasets.{name}.optional_columns contains duplicates.")
        if set(required_columns).intersection(optional_columns):
            raise ValueError(f"datasets.{name} declares a column as both required and optional.")
        expected_columns = set(required_columns).union(optional_columns)
        for required_column in ("episode_uid", "option_transition"):
            if required_column not in expected_columns:
                raise ValueError(f"Path C dataset {name} must include {required_column!r}.")
        if item["cross_identity_split"] is True:
            scientific_columns = {
                "fold_id", "surface_identity_key", "seed_group", "layout_style",
                "mechanism_key", "public_context_stratum", "probe_selected",
                "probe_action_id", "probe_skip_reason", "probe_rule_id",
                *map(str, preregistration.payload["rv_summary_spec"]["rv_columns"]),
            }
            missing_scientific_columns = sorted(scientific_columns.difference(expected_columns))
            if missing_scientific_columns:
                raise ValueError(
                    f"Path C dataset {name} lacks scientific readout columns: "
                    + ", ".join(missing_scientific_columns)
                )
        schema_payload = {
            "required_columns": list(required_columns),
            "optional_columns": list(optional_columns),
        }
        if item["schema_sha256"] != canonical_sha256(schema_payload):
            raise ValueError(f"Path C dataset {name} schema_sha256 does not match its columns.")

        chunks = _sequence(item["chunks"], f"datasets.{name}.chunks")
        if not chunks:
            raise ValueError(f"Path C dataset {name} has no chunk artifacts.")
        observed_uids: list[str] = []
        observed_transitions = 0
        fold_columns: dict[str, list[np.ndarray]] = {
            key: []
            for key in (
                "fold_id", "surface_identity_key", "seed_group",
                "layout_style", "mechanism_key",
            )
        }
        for index, raw_ref in enumerate(chunks):
            ref = _mapping(raw_ref, f"datasets.{name}.chunks[{index}]")
            artifact_path = _validated_file_reference(
                ref,
                base_dir,
                f"datasets.{name}.chunks[{index}]",
            )
            digest = str(ref["sha256"])
            if digest in seen_chunk_hashes:
                raise ValueError(f"Path C manifest reuses a dataset chunk SHA-256: {digest}")
            seen_chunk_hashes.add(digest)
            if artifact_path.suffix.lower() != ".npz":
                raise ValueError(f"Path C dataset chunk must be NPZ: {artifact_path}")
            with np.load(artifact_path, allow_pickle=False) as chunk:
                observed_columns = set(map(str, chunk.files))
                if observed_columns != expected_columns:
                    missing_columns = sorted(expected_columns.difference(observed_columns))
                    extra_columns = sorted(observed_columns.difference(expected_columns))
                    raise ValueError(
                        f"Path C dataset {name} chunk schema mismatch; "
                        f"missing={missing_columns}, extra={extra_columns}."
                    )
                lengths = {int(np.asarray(chunk[column]).shape[0]) for column in observed_columns}
                if len(lengths) != 1:
                    raise ValueError(f"Path C dataset {name} chunk has column length mismatch.")
                row_count = lengths.pop()
                if ref.get("rows") is None or int(ref["rows"]) != row_count:
                    raise ValueError(f"Path C dataset {name} chunk rows do not match artifact data.")
                chunk_uids = np.asarray(chunk["episode_uid"]).astype(str)
                if chunk_uids.shape[0] != row_count or np.any(chunk_uids == ""):
                    raise ValueError(f"Path C dataset {name} chunk has invalid episode_uid values.")
                expected_uid_prefix = f"{run_id}:{int(item['base_seed'])}:"
                if any(not uid.startswith(expected_uid_prefix) for uid in chunk_uids.tolist()):
                    raise ValueError(
                        f"Path C dataset {name} episode_uid does not bind run_id and base_seed."
                    )
                observed_uids.extend(chunk_uids.tolist())
                option_transition = np.asarray(chunk["option_transition"])
                if option_transition.shape[0] != row_count:
                    raise ValueError(f"Path C dataset {name} option_transition length mismatch.")
                if not set(np.unique(option_transition).tolist()).issubset({0, 1, False, True}):
                    raise ValueError(f"Path C dataset {name} option_transition must be boolean.")
                observed_transitions += int(np.count_nonzero(option_transition.astype(bool)))
                if item["cross_identity_split"] is True:
                    missing_fold_columns = sorted(
                        set(fold_columns).difference(observed_columns)
                    )
                    if missing_fold_columns:
                        raise ValueError(
                            f"Path C dataset {name} lacks cross-identity columns: "
                            + ", ".join(missing_fold_columns)
                        )
                    for key in fold_columns:
                        fold_columns[key].append(np.asarray(chunk[key]))

        unique_observed_uids = sorted(set(observed_uids))
        if len(observed_uids) != sum(
            int(_mapping(ref, "chunk")["rows"]) for ref in chunks
        ):
            raise ValueError(f"Path C dataset {name} row accounting is inconsistent.")
        declared_uids = sorted(
            str(value)
            for value in _sequence(item["episode_uids"], f"datasets.{name}.episode_uids")
        )
        if declared_uids != unique_observed_uids:
            raise ValueError(f"Path C dataset {name} episode_uids do not match chunk data.")
        if int(item["effective_episodes"]) != len(unique_observed_uids):
            raise ValueError(f"Path C dataset {name} effective_episodes is not derived from unique UIDs.")
        if int(item["effective_transitions"]) != observed_transitions:
            raise ValueError(f"Path C dataset {name} effective_transitions is not derived from option transitions.")
        overlap = seen_episode_uids.intersection(unique_observed_uids)
        if overlap:
            raise ValueError(f"Path C datasets reuse episode UIDs: {sorted(overlap)[:3]}")
        seen_episode_uids.update(unique_observed_uids)
        if item["cross_identity_split"] is True:
            fold_report = validate_cross_identity_folds(
                *(np.concatenate(fold_columns[key]) for key in fold_columns)
            )
            if not fold_report["valid"]:
                raise ValueError(f"Path C dataset {name} cross-identity folds overlap.")

    independent_variants = ("probe", "random_probe", "no_probe")
    missing_independent = [name for name in independent_variants if name not in matching_groups]
    if not missing_independent:
        schedule_hashes = {matching_groups[name] for name in independent_variants}
        if len(schedule_hashes) != 1:
            raise ValueError(
                "Path C probe/random-probe/no-probe collections do not use the same partner/seed/layout schedule."
            )


def _validate_checkpoint_entries(
    checkpoints: Mapping[str, Any],
    preregistration: FrozenPathCPreregistration,
    datasets: Mapping[str, Any],
    schemas: Mapping[str, Any],
    base_dir: Path,
) -> None:
    required = {PATH_C_PRIMARY_VARIANT, *preregistration.required_baselines}
    missing = sorted(required.difference(checkpoints))
    if missing:
        raise ValueError("Path C manifest is missing required checkpoint(s): " + ", ".join(missing))
    max_parameters = int(preregistration.payload["baselines"]["max_trainable_parameters"])
    eval_uids = {
        str(uid)
        for dataset in datasets.values()
        if str(_mapping(dataset, "dataset").get("collection_role"))
        in {"readout_evaluation", "synthetic_power", "terminal_axis_null", "recovery"}
        for uid in _mapping(dataset, "dataset").get("episode_uids", [])
    }
    evaluation_split_ids = {
        str(_mapping(dataset, "dataset")["split_id"])
        for dataset in datasets.values()
        if str(_mapping(dataset, "dataset").get("collection_role"))
        in {"readout_evaluation", "synthetic_power", "terminal_axis_null", "recovery"}
    }
    expected_schema = (
        str(schemas["graph_sha256"]),
        str(schemas["option_schema_sha256"]),
        str(schemas["observation_schema_sha256"]),
    )
    for name, raw in checkpoints.items():
        item = _mapping(raw, f"checkpoints.{name}")
        for key in (
            "path", "sha256", "method", "graph_sha256", "option_schema_sha256",
            "observation_schema_sha256", "resolved_path_c_sha256", "parameter_count",
            "parameter_manifest", "train_episode_uids", "train_split_ids",
            "preregistration_sha256", "oracle_baseline", "input_contract",
        ):
            if item.get(key) is None:
                raise ValueError(f"Path C checkpoint {name} is missing required field: {key}")
        _validated_file_reference(item, base_dir, f"checkpoints.{name}")
        if (
            str(name) == "base_only"
            and item["sha256"] != preregistration.payload["probe"]["base_checkpoint_sha256"]
        ):
            raise ValueError(
                "Path C BaseOnly checkpoint hash differs from the frozen preregistration."
            )
        if (str(name) == "partner_id") != (item["oracle_baseline"] is True):
            raise ValueError(
                f"Path C checkpoint {name} has an incorrect oracle_baseline marker."
            )
        expected_input_contract = _EXPECTED_VARIANT_INPUT_CONTRACTS.get(str(name))
        if expected_input_contract is not None and item["input_contract"] != expected_input_contract:
            raise ValueError(
                f"Path C checkpoint {name} has the wrong input contract."
            )
        if item["preregistration_sha256"] != preregistration.sha256:
            raise ValueError(f"Path C checkpoint {name} has a different preregistration SHA-256.")
        if item["resolved_path_c_sha256"] != preregistration.runtime_contract_sha256:
            raise ValueError(f"Path C checkpoint {name} has a different resolved config SHA-256.")
        if int(item["parameter_count"]) > max_parameters:
            raise ValueError(f"Path C checkpoint {name} exceeds the frozen trainable-parameter budget.")
        parameter_ref = _mapping(
            item["parameter_manifest"],
            f"checkpoints.{name}.parameter_manifest",
        )
        parameter_path = _validated_file_reference(
            parameter_ref,
            base_dir,
            f"checkpoints.{name}.parameter_manifest",
        )
        parameter_artifact = _load_mapping(
            parameter_path,
            f"Path C parameter manifest {name}",
        )
        _reject_authoritative_pass_fields(parameter_artifact, f"parameter_manifest.{name}")
        _require_mapping_fields(
            parameter_artifact,
            (
                "checkpoint_sha256", "preregistration_sha256",
                "resolved_path_c_sha256", "trainable_parameters",
            ),
            f"parameter_manifest.{name}",
        )
        if parameter_artifact["checkpoint_sha256"] != item["sha256"]:
            raise ValueError(f"Path C parameter manifest {name} uses a different checkpoint.")
        if parameter_artifact["preregistration_sha256"] != preregistration.sha256:
            raise ValueError(f"Path C parameter manifest {name} uses a different preregistration.")
        if parameter_artifact["resolved_path_c_sha256"] != preregistration.runtime_contract_sha256:
            raise ValueError(f"Path C parameter manifest {name} uses a different runtime config.")
        shapes = _mapping(
            parameter_artifact["trainable_parameters"],
            f"parameter_manifest.{name}.trainable_parameters",
        )
        derived_parameter_count = 0
        for parameter_name, raw_shape in shapes.items():
            shape = [
                int(value)
                for value in _sequence(
                    raw_shape,
                    f"parameter_manifest.{name}.{parameter_name}",
                )
            ]
            if not shape or any(value <= 0 for value in shape):
                raise ValueError(f"Path C parameter {name}.{parameter_name} has invalid shape.")
            derived_parameter_count += int(np.prod(np.asarray(shape, dtype=np.int64)))
        if derived_parameter_count != int(item["parameter_count"]):
            raise ValueError(f"Path C checkpoint {name} parameter_count is not artifact-derived.")
        train_uids = {str(uid) for uid in _sequence(item["train_episode_uids"], f"checkpoints.{name}.train_episode_uids")}
        if train_uids.intersection(eval_uids):
            raise ValueError(f"Path C checkpoint {name} training episodes overlap evaluation episodes.")
        train_split_ids = {
            str(value)
            for value in _sequence(item["train_split_ids"], f"checkpoints.{name}.train_split_ids")
        }
        if train_split_ids.intersection(evaluation_split_ids):
            raise ValueError(f"Path C checkpoint {name} training splits overlap evaluation splits.")
        schema = (
            str(item["graph_sha256"]),
            str(item["option_schema_sha256"]),
            str(item["observation_schema_sha256"]),
        )
        if schema != expected_schema:
            raise ValueError(f"Path C checkpoint {name} graph/option/observation schema does not match the manifest.")

        if str(name) == "base_only":
            calibration_refs = _sequence(
                item.get("calibration_predictions"),
                "checkpoints.base_only.calibration_predictions",
            )
            calibration_split_ids: set[str] = set()
            for index, raw_ref in enumerate(calibration_refs):
                ref = _mapping(
                    raw_ref,
                    f"checkpoints.base_only.calibration_predictions[{index}]",
                )
                _require_mapping_fields(
                    ref,
                    (
                        "path", "sha256", "fold_id", "evaluation_split_id",
                        "preregistration_sha256", "resolved_path_c_sha256",
                    ),
                    f"checkpoints.base_only.calibration_predictions[{index}]",
                )
                calibration_path = _validated_file_reference(
                    ref,
                    base_dir,
                    f"checkpoints.base_only.calibration_predictions[{index}]",
                )
                if ref["preregistration_sha256"] != preregistration.sha256:
                    raise ValueError("BaseOnly calibration predictions use a different preregistration.")
                if ref["resolved_path_c_sha256"] != preregistration.runtime_contract_sha256:
                    raise ValueError("BaseOnly calibration predictions use a different runtime config.")
                split_id = str(ref["evaluation_split_id"])
                if split_id in train_split_ids:
                    raise ValueError("BaseOnly calibration predictions reuse a training split.")
                calibration_split_ids.add(split_id)
                calibration = _load_mapping(
                    calibration_path,
                    f"BaseOnly calibration predictions for {split_id}",
                )
                _reject_authoritative_pass_fields(
                    calibration,
                    f"base_only_calibration.{split_id}",
                )
                _require_mapping_fields(
                    calibration,
                    (
                        "fold_id", "evaluation_split_id", "episode_uids",
                        "base_only_checkpoint_sha256", "graph_sha256",
                        "option_schema_sha256", "observation_schema_sha256",
                        "preregistration_sha256", "resolved_path_c_sha256",
                        "prediction_columns", "predictions_by_episode",
                    ),
                    f"base_only_calibration.{split_id}",
                )
                if calibration["evaluation_split_id"] != split_id:
                    raise ValueError("BaseOnly calibration split id differs from its manifest reference.")
                if calibration["base_only_checkpoint_sha256"] != item["sha256"]:
                    raise ValueError("BaseOnly calibration predictions use a different checkpoint.")
                if calibration["preregistration_sha256"] != preregistration.sha256:
                    raise ValueError("BaseOnly calibration artifact uses a different preregistration.")
                if calibration["resolved_path_c_sha256"] != preregistration.runtime_contract_sha256:
                    raise ValueError("BaseOnly calibration artifact uses a different runtime config.")
                calibration_schema = (
                    str(calibration["graph_sha256"]),
                    str(calibration["option_schema_sha256"]),
                    str(calibration["observation_schema_sha256"]),
                )
                if calibration_schema != expected_schema:
                    raise ValueError("BaseOnly calibration predictions use different schemas.")
                expected_uids = {
                    str(uid)
                    for dataset in datasets.values()
                    if str(_mapping(dataset, "dataset").get("split_id")) == split_id
                    for uid in _mapping(dataset, "dataset").get("episode_uids", [])
                }
                calibration_uids = {
                    str(uid)
                    for uid in _sequence(
                        calibration["episode_uids"],
                        f"base_only_calibration.{split_id}.episode_uids",
                    )
                }
                if calibration_uids != expected_uids:
                    raise ValueError("BaseOnly calibration episode UIDs do not match the evaluation fold.")
                predictions = _mapping(
                    calibration["predictions_by_episode"],
                    f"base_only_calibration.{split_id}.predictions_by_episode",
                )
                if set(map(str, predictions)) != calibration_uids:
                    raise ValueError("BaseOnly calibration predictions do not cover the evaluation fold.")
                option_counts = set()
                for episode_uid, raw_values in predictions.items():
                    values = [
                        _finite_number(value, f"base_only_calibration.{episode_uid}")
                        for value in _sequence(
                            raw_values,
                            f"base_only_calibration.{episode_uid}",
                        )
                    ]
                    if not values:
                        raise ValueError("BaseOnly calibration prediction vector is empty.")
                    option_counts.add(len(values))
                if len(option_counts) != 1:
                    raise ValueError("BaseOnly calibration prediction vectors have inconsistent option counts.")
            if not evaluation_split_ids.issubset(calibration_split_ids):
                missing_splits = sorted(evaluation_split_ids.difference(calibration_split_ids))
                raise ValueError(
                    "BaseOnly lacks fold-specific calibration predictions for: "
                    + ", ".join(missing_splits)
                )


def _validate_dataset_checkpoint_bindings(
    datasets: Mapping[str, Any],
    checkpoints: Mapping[str, Any],
) -> None:
    for name, raw in datasets.items():
        item = _mapping(raw, f"datasets.{name}")
        checkpoint_key = str(item["controller_checkpoint_key"])
        if checkpoint_key not in checkpoints:
            raise ValueError(
                f"Path C dataset {name} refers to missing checkpoint {checkpoint_key!r}."
            )
        checkpoint = _mapping(checkpoints[checkpoint_key], f"checkpoints.{checkpoint_key}")
        if item["controller_checkpoint_sha256"] != checkpoint.get("sha256"):
            raise ValueError(
                f"Path C dataset {name} controller checkpoint hash does not match the manifest."
            )


def _validated_file_reference(reference: Mapping[str, Any], base_dir: Path, name: str) -> Path:
    if reference.get("path") in {None, ""} or reference.get("sha256") in {None, ""}:
        raise ValueError(f"{name} requires path and sha256.")
    path = Path(str(reference["path"]))
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"{name} artifact does not exist: {path}")
    observed = file_sha256(path)
    if observed != str(reference["sha256"]):
        raise ValueError(f"{name} SHA-256 mismatch: expected {reference['sha256']}, got {observed}")
    return path


def _validate_bound_artifact(
    artifact: Mapping[str, Any],
    preregistration: FrozenPathCPreregistration,
    resolved_path_c_sha256: str,
    name: str,
) -> None:
    if artifact.get("preregistration_sha256") != preregistration.sha256:
        raise ValueError(f"{name} has a different preregistration SHA-256.")
    if artifact.get("resolved_path_c_sha256") != resolved_path_c_sha256:
        raise ValueError(f"{name} has a different resolved config SHA-256.")


def _validate_measurement_provenance(
    measurement: Mapping[str, Any],
    datasets: Mapping[str, Any],
    checkpoints: Mapping[str, Any],
    variant: str,
    preregistration: FrozenPathCPreregistration,
) -> None:
    if measurement.get("measurement_kind") != "path_c_readout":
        raise ValueError(f"Path C measurement artifact {variant} has the wrong measurement_kind.")
    source_keys = _string_tuple(
        measurement.get("source_dataset_keys"),
        f"measurement_artifacts.{variant}.source_dataset_keys",
    )
    missing = sorted(set(source_keys).difference(datasets))
    if missing:
        raise ValueError(
            f"Path C measurement artifact {variant} refers to missing dataset(s): "
            + ", ".join(missing)
        )
    if bool(preregistration.payload["cross_identity_split"]["enable"]):
        if not any(
            _mapping(datasets[key], f"datasets.{key}").get("cross_identity_split") is True
            for key in source_keys
        ):
            raise ValueError(
                f"Path C measurement artifact {variant} is not driven by a cross-identity split."
            )
    readout_capacity = _mapping(
        measurement.get("readout_capacity"),
        f"measurement_artifacts.{variant}.readout_capacity",
    )
    if readout_capacity != preregistration.payload["readout"]:
        raise ValueError(f"Path C measurement artifact {variant} changes readout capacity.")
    rv_columns = tuple(
        str(value)
        for value in _sequence(
            measurement.get("rv_columns"),
            f"measurement_artifacts.{variant}.rv_columns",
        )
    )
    frozen_rv_columns = tuple(
        map(str, preregistration.payload["rv_summary_spec"]["joint_likelihood_order"])
    )
    if rv_columns != frozen_rv_columns:
        raise ValueError(f"Path C measurement artifact {variant} changes joint R^V order.")
    split_audit = _mapping(
        measurement.get("cross_identity_audit"),
        f"measurement_artifacts.{variant}.cross_identity_audit",
    )
    if split_audit.get("valid") is not True:
        raise ValueError(f"Path C measurement artifact {variant} lacks a valid split audit.")
    calibration_binding = _mapping(
        measurement.get("base_only_calibration_sha256_by_split"),
        f"measurement_artifacts.{variant}.base_only_calibration_sha256_by_split",
    )
    base_checkpoint = _mapping(checkpoints.get("base_only"), "checkpoints.base_only")
    expected_calibration = {
        str(_mapping(ref, "base calibration")["evaluation_split_id"]): str(
            _mapping(ref, "base calibration")["sha256"]
        )
        for ref in _sequence(
            base_checkpoint.get("calibration_predictions"),
            "checkpoints.base_only.calibration_predictions",
        )
    }
    source_split_ids = {
        str(_mapping(datasets[key], f"datasets.{key}")["split_id"])
        for key in source_keys
        if str(_mapping(datasets[key], f"datasets.{key}").get("collection_role"))
        in {"readout_evaluation", "synthetic_power", "terminal_axis_null", "recovery"}
    }
    missing_calibration = sorted(source_split_ids.difference(calibration_binding))
    mismatched_calibration = sorted(
        split_id
        for split_id in source_split_ids.intersection(calibration_binding)
        if str(calibration_binding[split_id]) != expected_calibration.get(split_id)
    )
    if missing_calibration or mismatched_calibration:
        raise ValueError(
            f"Path C measurement artifact {variant} is not bound to fold-specific BaseOnly predictions."
        )
    benchmark_return = measurement.get("benchmark_return")
    if benchmark_return is not None:
        item = _mapping(
            benchmark_return,
            f"measurement_artifacts.{variant}.benchmark_return",
        )
        require_formal_benchmark_artifact(item)


def _runtime_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    contract: dict[str, Any] = {}
    for section_name, fields in _RUNTIME_CONTRACT_FIELDS.items():
        section = payload.get(section_name)
        if not isinstance(section, Mapping):
            section = {}
        contract[section_name] = {field: _json_scalar(section.get(field)) for field in fields}
    return contract


def _require_mapping_fields(
    mapping: Mapping[str, Any],
    fields: Sequence[str],
    name: str,
) -> None:
    missing = [field for field in fields if field not in mapping or mapping[field] is None]
    if missing:
        raise ValueError(f"{name} is missing required field(s): " + ", ".join(missing))


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text.lower())


def _reject_authoritative_pass_fields(value: Any, path: str) -> None:
    if isinstance(value, Mapping):
        if "pass" in value:
            raise ValueError(f"{path} contains an authoritative 'pass' field; provide raw measurements instead.")
        for key, item in value.items():
            _reject_authoritative_pass_fields(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_authoritative_pass_fields(item, f"{path}[{index}]")


def _load_mapping(path: Path, name: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a mapping: {path}")
    return payload


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return dict(value)


def _sequence(value: Any, name: str) -> list[Any]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list.")
    return list(value)


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    values = tuple(str(item) for item in _sequence(value, name))
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{name} must be a non-empty list without duplicates.")
    return values


def _measurement(measurements: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in measurements:
            return measurements[name]
    return None


def _aggregate_value_control(
    item: Mapping[str, Any],
    required_baselines: Sequence[str],
    name: str,
) -> dict[str, Any]:
    comparisons = _mapping(item.get("comparisons"), f"{name}.comparisons")
    missing = sorted(set(required_baselines).difference(comparisons))
    if missing:
        raise ValueError(f"{name} lacks required baseline comparison(s): " + ", ".join(missing))
    values = []
    lower_bounds = []
    normalized = {}
    for baseline in required_baselines:
        measurement = _mapping(comparisons[baseline], f"{name}.comparisons.{baseline}")
        value = _finite_number(measurement.get("value"), f"{name}.{baseline}.value")
        ci_lo = _finite_number(
            measurement.get("ci_lo", _nested_ci(measurement, "lo")),
            f"{name}.{baseline}.ci_lo",
        )
        values.append(value)
        lower_bounds.append(ci_lo)
        normalized[str(baseline)] = {"value": value, "ci_lo": ci_lo}
    return {
        "value": min(values),
        "ci_lo": min(lower_bounds),
        "comparisons": normalized,
        "aggregation": "minimum_over_required_baselines",
    }


def _nested_ci(item: Mapping[str, Any], key: str) -> Any:
    ci = item.get("ci")
    return ci.get(key) if isinstance(ci, Mapping) else None


def _optional_finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _interval(value: Any) -> list[float | None]:
    if isinstance(value, Mapping):
        return [
            _optional_finite_number(value.get("lo")),
            _optional_finite_number(value.get("hi")),
        ]
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return [_optional_finite_number(value[0]), _optional_finite_number(value[1])]
    return [None, None]


def _interval_inside(
    interval: Sequence[float | None],
    lower: float,
    upper: float,
) -> bool:
    lo, hi = interval
    return lo is not None and hi is not None and lo > lower and hi < upper


def _finite_number(value: Any, name: str) -> float:
    result = _optional_finite_number(value)
    if result is None:
        raise ValueError(f"{name} must be a finite number.")
    return result


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _json_scalar(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_scalar(item) for item in value]
    if isinstance(value, list):
        return [_json_scalar(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_scalar(item) for key, item in value.items()}
    return value
