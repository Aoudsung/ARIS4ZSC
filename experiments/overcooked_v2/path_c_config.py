from __future__ import annotations

import copy
import hashlib
import json
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from experiments.overcooked_v2.path_c_evaluation import (
    load_frozen_preregistration,
    validate_runtime_path_c_config,
)


PATH_C_RUNTIME_SCHEMA_VERSION = "path_c_runtime_v3"
PATH_C_FEATURE_SECTIONS = (
    "evidence_spec",
    "ensemble",
    "probe",
    "response_summary_spec",
    "fingerprint",
    "split",
    "primary_endpoint",
    "belief_kernel_audit",
    "audit_battery",
    "artifact_contract",
)
_LEGACY_KEYS = {
    "rv_summary_spec": "response_summary_spec",
    "cross_identity_split": "split",
    "pass_af": "decision plus secondary_endpoints in the version-3 preregistration",
}
_DERIVED_TOP_LEVEL_KEYS = {
    "resolved_path_c_sha256",
    "active_sections",
    "preregistration",
}


def default_path_c_config() -> dict[str, Any]:
    """Return the inert version-3 runtime configuration."""

    return {
        "schema_version": PATH_C_RUNTIME_SCHEMA_VERSION,
        "preregistration_path": None,
        "evidence_spec": {
            "enable": False,
            "schema_version": "ego_evidence_spec_v1",
            "max_primitive_actions_per_decision": 64,
            "max_episode_decisions": 128,
        },
        "ensemble": {
            "architecture": "legacy_default_off",
            "n_heads": 1,
            "bootstrap_p": 1.0,
            "bootstrap_scope": "episode",
            "prior_scale": 0.0,
            "prior_network": "independent_frozen_recurrent",
            "recurrent_state": "head_specific",
            "dueling_advantage_center": "valid_actions_only",
            "disagreement_stat": "variance",
            "disagreement_target": "normalized_advantage",
            "tie_atol": 1.0e-6,
        },
        "probe": {
            "enable": False,
            "collection_enable": False,
            "locked_audit_adaptive_probe_enable": False,
            "rule": "max_normalized_advantage_disagreement",
            "disagreement_threshold": None,
            "return_floor": None,
            "record_candidate_mask": True,
            "record_all_scores": True,
            "record_propensity": True,
            "record_budget_and_cost": True,
            "random_probe_support_matched": True,
            "direct_information_baseline": "synthetic_oracle_diagnostic_only",
            "min_selected_probes": None,
            "min_probe_opportunities": None,
            "min_context_coverage": None,
            "min_action_coverage": None,
            "collection_selection_mode": "normalized_advantage",
        },
        "response_summary_spec": {
            "enable": False,
            "schema_version": "path_c_response_summary_v1",
            "path": None,
        },
        "fingerprint": {
            "enable": False,
            "kind": "counterbalanced_value_null_candidate",
            "vocab": [],
            "positive_control_kind": "raw_response_value_null",
            "negative_control_kind": "metadata_only",
        },
        "split": {
            "enable": False,
            "schema_version": "path_c_split_manifest_v2",
            "manifest_path": None,
            "roles": ["train", "design", "calibration", "locked_audit"],
            "minimum_groups_per_mechanism": 4,
            "preferred_groups_per_mechanism": 5,
            "primary_shift": "identity",
            "secondary_shift": "layout",
        },
        "primary_endpoint": {
            "enable": False,
            "schema_version": "path_c_primary_endpoint_v1",
            "baseline_selection_path": None,
        },
        "belief_kernel_audit": {
            "enable": False,
            "schema_version": "path_c_belief_kernel_audit_v1",
            "outer_replicas_M": None,
            "simultaneous_cell_count": None,
            "confidence_delta": None,
            "inner_forks_L_inner": None,
            "probe_horizon_T_probe": None,
            "exact_mode": True,
            "tier1_hypothesis_prune": 0.0,
            "posterior_bias_bound": 0.0,
            "reset_bias_bound": 0.0,
            "rng_key_schedule_version": "path_c_rng_key_schedule_v1",
        },
        "audit_battery": {
            "enable": False,
            "schema_version": "path_c_audit_battery_v1",
            "registry_path": None,
        },
        "artifact_contract": {
            "enable": False,
            "schema_version": "path_c_artifacts_v3",
            "module_registry_path": None,
            "split_manifest_path": None,
        },
    }


def normalize_path_c_config(
    config: dict[str, Any],
    *,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Strictly normalize runtime config and bind active features to a frozen file.

    Unknown keys and version-2 names are rejected. This is deliberate: a prior
    artifact must be migrated explicitly and can never acquire version-3 meaning
    through an alias.
    """

    raw_value = config.get("path_c") or {}
    if not isinstance(raw_value, Mapping):
        raise ValueError("path_c must be a mapping.")
    raw = copy.deepcopy(dict(raw_value))
    previous_derived = {
        key: raw.pop(key)
        for key in tuple(_DERIVED_TOP_LEVEL_KEYS)
        if key in raw
    }
    previous_frozen_evidence = None
    if isinstance(raw.get("evidence_spec"), Mapping):
        raw["evidence_spec"] = dict(raw["evidence_spec"])
        previous_frozen_evidence = raw["evidence_spec"].pop("frozen_spec", None)
    previous_frozen_response = None
    if isinstance(raw.get("response_summary_spec"), Mapping):
        raw["response_summary_spec"] = dict(raw["response_summary_spec"])
        previous_frozen_response = raw["response_summary_spec"].pop(
            "frozen_spec", None
        )
    legacy = sorted(set(raw).intersection(_LEGACY_KEYS))
    if legacy:
        replacements = ", ".join(
            f"{key!r} -> {_LEGACY_KEYS[key]}" for key in legacy
        )
        raise ValueError(
            "Legacy Path C runtime keys require an explicit version-3 migration: "
            + replacements
        )
    defaults = default_path_c_config()
    _reject_unknown_keys(raw, defaults, "path_c")
    section = _deep_merge(defaults, raw)
    if section.get("schema_version") != PATH_C_RUNTIME_SCHEMA_VERSION:
        raise ValueError(
            f"path_c.schema_version must be {PATH_C_RUNTIME_SCHEMA_VERSION!r}."
        )
    _normalize_section(section)
    _resolve_runtime_paths(section, config_path)
    active = active_path_c_sections(section)
    _validate_active_invariants(section, active)
    preregistration = _preregistration_summary(
        section.get("preregistration_path"),
        required=bool(active),
    )
    if active:
        frozen = load_frozen_preregistration(section["preregistration_path"])
        section["resolved_path_c_sha256"] = validate_runtime_path_c_config(
            section,
            frozen,
        )
        section["evidence_spec"]["frozen_spec"] = frozen.evidence_spec.to_dict()
        section["response_summary_spec"]["frozen_spec"] = (
            frozen.response_summary_spec.canonical_payload()
        )
        preregistration["runtime_contract_sha256"] = frozen.runtime_contract_sha256
        preregistration["response_vocabulary_sha256"] = frozen.response_vocabulary_sha256
        preregistration["evidence_spec_sha256"] = frozen.evidence_spec_sha256
        preregistration["probe_cost_per_use"] = float(
            frozen.primary_endpoint["probe_cost_per_use"]
        )
        preregistration["probe_budget_grid"] = list(
            map(int, frozen.primary_endpoint["probe_budget_grid"])
        )
    else:
        section["resolved_path_c_sha256"] = None
    section["active_sections"] = active
    section["preregistration"] = preregistration
    _verify_previous_normalization(
        section,
        previous_derived=previous_derived,
        previous_frozen_evidence=previous_frozen_evidence,
        previous_frozen_response=previous_frozen_response,
    )
    config["path_c"] = section
    return section


def _verify_previous_normalization(
    section: Mapping[str, Any],
    *,
    previous_derived: Mapping[str, Any],
    previous_frozen_evidence: Any,
    previous_frozen_response: Any,
) -> None:
    """Make checkpoint reloading idempotent while rejecting derived-field drift."""

    for key, previous in previous_derived.items():
        if previous != section.get(key):
            raise ValueError(
                f"Stored derived Path C field {key!r} does not match recomputation."
            )
    for name, previous, current in (
        (
            "evidence_spec.frozen_spec",
            previous_frozen_evidence,
            (section.get("evidence_spec") or {}).get("frozen_spec"),
        ),
        (
            "response_summary_spec.frozen_spec",
            previous_frozen_response,
            (section.get("response_summary_spec") or {}).get("frozen_spec"),
        ),
    ):
        if previous is not None and previous != current:
            raise ValueError(
                f"Stored derived Path C field {name!r} does not match recomputation."
            )


def active_path_c_sections(section: Mapping[str, Any]) -> list[str]:
    active: list[str] = []
    if bool((section.get("evidence_spec") or {}).get("enable", False)):
        active.append("evidence_spec")
    ensemble = section.get("ensemble") or {}
    if int(ensemble.get("n_heads", 1)) > 1 or str(
        ensemble.get("architecture", "legacy_default_off")
    ) == "recurrent_sequence_v1":
        active.append("ensemble")
    for key in (
        "probe",
        "response_summary_spec",
        "fingerprint",
        "split",
        "primary_endpoint",
        "belief_kernel_audit",
        "audit_battery",
        "artifact_contract",
    ):
        if bool((section.get(key) or {}).get("enable", False)):
            active.append(key)
    return active


def path_c_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    section = config.get("path_c") or default_path_c_config()
    keys = (
        "schema_version",
        "evidence_spec",
        "ensemble",
        "probe",
        "response_summary_spec",
        "fingerprint",
        "split",
        "primary_endpoint",
        "belief_kernel_audit",
        "audit_battery",
        "artifact_contract",
        "preregistration",
        "resolved_path_c_sha256",
        "active_sections",
    )
    return {
        key: copy.deepcopy(section.get(key))
        for key in keys
        if key in section
    }


def require_preregistration_fields(
    config: Mapping[str, Any],
    required_paths: Iterable[str],
) -> None:
    path_value = (config.get("path_c") or {}).get("preregistration_path")
    if path_value in {None, ""}:
        raise ValueError("Path C preregistration path is required.")
    payload = _load_yaml_mapping(Path(str(path_value)).resolve())
    missing = [
        path
        for path in required_paths
        if _get_path(payload, str(path).split(".")) is None
    ]
    if missing:
        raise ValueError(
            "Path C preregistration is missing required field(s): "
            + ", ".join(sorted(missing))
        )


def _normalize_section(section: dict[str, Any]) -> None:
    evidence = section["evidence_spec"]
    evidence["enable"] = _as_bool(evidence["enable"], "path_c.evidence_spec.enable")
    for key in ("max_primitive_actions_per_decision", "max_episode_decisions"):
        evidence[key] = int(evidence[key])
        if evidence[key] <= 0:
            raise ValueError(f"path_c.evidence_spec.{key} must be positive.")

    ensemble = section["ensemble"]
    ensemble["n_heads"] = int(ensemble["n_heads"])
    ensemble["bootstrap_p"] = float(ensemble["bootstrap_p"])
    ensemble["prior_scale"] = float(ensemble["prior_scale"])
    ensemble["tie_atol"] = float(ensemble["tie_atol"])
    if ensemble["n_heads"] <= 0:
        raise ValueError("path_c.ensemble.n_heads must be positive.")
    if not 0.0 < ensemble["bootstrap_p"] <= 1.0:
        raise ValueError("path_c.ensemble.bootstrap_p must be in (0, 1].")
    if ensemble["prior_scale"] < 0.0 or ensemble["tie_atol"] < 0.0:
        raise ValueError("Path C prior_scale and tie_atol must be non-negative.")
    if ensemble["architecture"] not in {"legacy_default_off", "recurrent_sequence_v1"}:
        raise ValueError("Unknown Path C ensemble architecture.")
    if ensemble["bootstrap_scope"] != "episode":
        raise ValueError("Path C recurrent bootstrap scope must be episode.")
    if ensemble["dueling_advantage_center"] != "valid_actions_only":
        raise ValueError("Path C dueling advantages must center over valid actions only.")
    if ensemble["disagreement_target"] != "normalized_advantage":
        raise ValueError("Path C probe disagreement target must be normalized_advantage.")

    probe = section["probe"]
    for key in (
        "enable",
        "collection_enable",
        "locked_audit_adaptive_probe_enable",
        "record_candidate_mask",
        "record_all_scores",
        "record_propensity",
        "record_budget_and_cost",
        "random_probe_support_matched",
    ):
        probe[key] = _as_bool(probe[key], f"path_c.probe.{key}")
    if probe["rule"] != "max_normalized_advantage_disagreement":
        raise ValueError("Path C probe.rule must use normalized-advantage disagreement.")
    for key in ("disagreement_threshold", "return_floor"):
        probe[key] = None if probe[key] is None else float(probe[key])
    for key in ("min_selected_probes", "min_probe_opportunities"):
        probe[key] = None if probe[key] is None else int(probe[key])
    for key in ("min_context_coverage", "min_action_coverage"):
        probe[key] = None if probe[key] is None else float(probe[key])

    for key in (
        "response_summary_spec",
        "fingerprint",
        "split",
        "primary_endpoint",
        "belief_kernel_audit",
        "audit_battery",
        "artifact_contract",
    ):
        section[key]["enable"] = _as_bool(
            section[key]["enable"], f"path_c.{key}.enable"
        )
    belief = section["belief_kernel_audit"]
    belief["exact_mode"] = _as_bool(
        belief["exact_mode"], "path_c.belief_kernel_audit.exact_mode"
    )
    belief["tier1_hypothesis_prune"] = float(belief["tier1_hypothesis_prune"])
    belief["posterior_bias_bound"] = float(belief["posterior_bias_bound"])
    belief["reset_bias_bound"] = float(belief["reset_bias_bound"])
    if belief["simultaneous_cell_count"] is not None:
        if isinstance(belief["simultaneous_cell_count"], bool) or not isinstance(
            belief["simultaneous_cell_count"], Integral
        ):
            raise ValueError("Path C simultaneous_cell_count must be an integer.")
        belief["simultaneous_cell_count"] = int(belief["simultaneous_cell_count"])
    if belief["confidence_delta"] is not None:
        if isinstance(belief["confidence_delta"], bool) or not isinstance(
            belief["confidence_delta"], Real
        ):
            raise ValueError("Path C confidence_delta must be numeric.")
        belief["confidence_delta"] = float(belief["confidence_delta"])
    if not 0.0 <= belief["tier1_hypothesis_prune"] < 1.0:
        raise ValueError("Path C tier1_hypothesis_prune must lie in [0, 1).")
    if any(
        not 0.0 <= belief[key] <= 1.0
        for key in ("posterior_bias_bound", "reset_bias_bound")
    ):
        raise ValueError("Path C posterior and reset bias bounds must lie in [0, 1].")
    if belief["exact_mode"] and belief["tier1_hypothesis_prune"] != 0.0:
        raise ValueError("Exact Tier 1 forbids positive-mass hypothesis pruning.")
    if belief["exact_mode"] and (
        belief["posterior_bias_bound"] != 0.0
        or belief["reset_bias_bound"] != 0.0
    ):
        raise ValueError("Exact Tier 1 requires zero posterior and reset bias bounds.")
    if (
        not belief["exact_mode"]
        and belief["tier1_hypothesis_prune"] > 0.0
        and belief["posterior_bias_bound"] <= 0.0
    ):
        raise ValueError(
            "Approximate Tier 1 with pruning requires a positive posterior bias bound."
        )
    if (
        belief["simultaneous_cell_count"] is not None
        and belief["simultaneous_cell_count"] < 3
    ):
        raise ValueError("Path C simultaneous_cell_count must be at least three.")
    if belief["confidence_delta"] is not None and not (
        0.0 < belief["confidence_delta"] < 1.0
    ):
        raise ValueError("Path C confidence_delta must lie in (0, 1).")


def _validate_active_invariants(section: Mapping[str, Any], active: Iterable[str]) -> None:
    active_set = set(active)
    ensemble = section["ensemble"]
    if "ensemble" in active_set:
        if ensemble["architecture"] != "recurrent_sequence_v1":
            raise ValueError("Active Path C must use recurrent_sequence_v1.")
        if not bool(section["evidence_spec"]["enable"]):
            raise ValueError("Active recurrent Path C requires evidence_spec.enable=true.")
        if int(ensemble["n_heads"]) > 1 and (
            float(ensemble["bootstrap_p"]) >= 1.0
            or float(ensemble["prior_scale"]) <= 0.0
        ):
            raise ValueError(
                "A multi-head Path C ensemble requires bootstrap_p < 1 and prior_scale > 0."
            )
    if "probe" in active_set:
        if int(ensemble["n_heads"]) <= 1:
            raise ValueError("Active probes require more than one ensemble head.")
        probe = section["probe"]
        if not probe["collection_enable"]:
            raise ValueError("Active probes require collection_enable=true.")
        if probe["locked_audit_adaptive_probe_enable"]:
            raise ValueError("Adaptive training probes cannot run in locked audit.")
        required = (
            "disagreement_threshold",
            "return_floor",
            "min_selected_probes",
            "min_probe_opportunities",
            "min_context_coverage",
            "min_action_coverage",
        )
        missing = [key for key in required if probe.get(key) is None]
        if missing:
            raise ValueError("Active Path C probe fields are missing: " + ", ".join(missing))
    if "belief_kernel_audit" in active_set:
        belief = section["belief_kernel_audit"]
        required = (
            "outer_replicas_M",
            "simultaneous_cell_count",
            "confidence_delta",
            "inner_forks_L_inner",
            "probe_horizon_T_probe",
        )
        missing = [key for key in required if belief.get(key) is None]
        if missing:
            raise ValueError(
                "Active belief-kernel audit has unfrozen field(s): " + ", ".join(missing)
            )
    if "primary_endpoint" in active_set and not bool(section["split"]["enable"]):
        raise ValueError("The primary endpoint requires the frozen four-role split.")


def _resolve_runtime_paths(section: dict[str, Any], config_path: str | Path | None) -> None:
    keys = (
        ("preregistration_path",),
        ("response_summary_spec", "path"),
        ("split", "manifest_path"),
        ("primary_endpoint", "baseline_selection_path"),
        ("audit_battery", "registry_path"),
        ("artifact_contract", "module_registry_path"),
        ("artifact_contract", "split_manifest_path"),
    )
    for path_keys in keys:
        parent: dict[str, Any] = section
        for key in path_keys[:-1]:
            parent = parent[key]
        leaf = path_keys[-1]
        value = parent.get(leaf)
        if value not in {None, ""}:
            parent[leaf] = str(_resolve_path(value, config_path))


def _preregistration_summary(path_value: Any, *, required: bool) -> dict[str, Any]:
    if path_value in {None, ""}:
        if required:
            raise ValueError("path_c.preregistration_path is required for active Path C.")
        return {"loaded": False, "path": None, "sha256": None}
    path = Path(str(path_value)).resolve()
    payload = _load_yaml_mapping(path)
    if required and str(payload.get("status", "")).lower() != "frozen":
        raise ValueError("Active Path C requires preregistration status: frozen.")
    raw = path.read_bytes()
    return {
        "loaded": True,
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "status": payload.get("status"),
        "version": payload.get("version"),
        "schema_version": payload.get("schema_version"),
        "freeze_timestamp": payload.get("freeze_timestamp"),
        "semantic_bindings": copy.deepcopy(payload.get("semantic_bindings") or {}),
    }


def _reject_unknown_keys(observed: Mapping[str, Any], schema: Mapping[str, Any], path: str) -> None:
    unknown = sorted(set(observed).difference(schema))
    if unknown:
        raise ValueError(f"{path} contains unknown key(s): {', '.join(unknown)}")
    for key, value in observed.items():
        expected = schema.get(key)
        if isinstance(value, Mapping) and isinstance(expected, Mapping):
            _reject_unknown_keys(value, expected, f"{path}.{key}")


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _resolve_path(path_value: Any, config_path: str | Path | None) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path.resolve()
    if config_path is not None:
        return (Path(config_path).resolve().parent / path).resolve()
    return path.resolve()


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Path C configuration file does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Path C YAML must contain a mapping: {path}")
    return payload


def _get_path(mapping: Mapping[str, Any], keys: Iterable[str]) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _as_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    raise ValueError(f"{name} must be boolean-like; got {value!r}.")
