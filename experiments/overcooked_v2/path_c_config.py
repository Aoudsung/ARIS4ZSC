from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import yaml

from experiments.overcooked_v2.path_c_evaluation import (
    load_frozen_preregistration,
    validate_runtime_path_c_config,
)


PATH_C_FEATURE_SECTIONS = (
    "ensemble",
    "probe",
    "rv_summary_spec",
    "fingerprint",
    "cross_identity_split",
    "pass_af",
)


def default_path_c_config() -> dict[str, Any]:
    return {
        "preregistration_path": None,
        "ensemble": {
            "n_heads": 1,
            "bootstrap_p": 1.0,
            "prior_scale": 0.0,
            "disagreement_stat": "variance",
            "tie_atol": 1.0e-6,
        },
        "probe": {
            "enable": False,
            "eval_enable": False,
            "collection_enable": False,
            "disagreement_threshold": None,
            "return_floor": None,
            "rule": "max_residual_signature_disagreement",
            "base_checkpoint": None,
            "base_checkpoint_sha256": None,
            "residual_baseline_checkpoint": None,
            "require_public_residual_baseline": True,
            "allow_raw_q_fallback": False,
            "min_selected_probes": None,
            "min_probe_opportunities": None,
            "min_context_coverage": None,
            "min_action_coverage": None,
        },
        "rv_summary_spec": {
            "enable": False,
            "path": None,
        },
        "fingerprint": {
            "enable": False,
            "kind": "counterbalanced_value_null_candidate",
            "vocab": [],
            "positive_control_kind": "raw_response_value_null",
            "negative_control_kind": "metadata_only",
        },
        "cross_identity_split": {
            "enable": False,
            "group_key": ["mode.family", "mode.param"],
            "identity_key": "surface_identity_key",
            "split": "same_mechanism_disjoint_surface_seed_layout",
            "folds": 5,
        },
        "pass_af": {
            "enable": False,
            "thresholds_path": None,
        },
    }


def normalize_path_c_config(
    config: dict[str, Any],
    *,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Normalize Path C config and bind any active feature to frozen preregistration.

    Defaults are inert. Once any Path C section is active, the preregistration file
    must exist, have ``status: frozen``, contain the enabled sections, and is
    recorded by SHA-256 in artifacts. Draft defaults or config intent are not
    admissible as scientific evidence.
    """
    raw_section = config.get("path_c") or {}
    section = _deep_merge(default_path_c_config(), raw_section)
    _normalize_ensemble(section["ensemble"])
    _normalize_probe(section["probe"])
    _normalize_enabled_path(section["rv_summary_spec"], "rv_summary_spec")
    _normalize_enabled_path(section["pass_af"], "pass_af")
    section["fingerprint"]["enable"] = _as_bool(
        section["fingerprint"].get("enable", False),
        "path_c.fingerprint.enable",
    )
    section["cross_identity_split"]["enable"] = _as_bool(
        section["cross_identity_split"].get("enable", False),
        "path_c.cross_identity_split.enable",
    )
    section["cross_identity_split"]["folds"] = int(
        section["cross_identity_split"].get("folds", 5)
    )
    if section["cross_identity_split"]["folds"] <= 1:
        raise ValueError("path_c.cross_identity_split.folds must exceed one.")
    section["fingerprint"]["vocab"] = list(section["fingerprint"].get("vocab", []))
    for key in ("kind", "positive_control_kind", "negative_control_kind"):
        value = section["fingerprint"].get(key)
        if value in {None, ""}:
            raise ValueError(f"path_c.fingerprint.{key} must be explicit.")
        section["fingerprint"][key] = str(value)

    if section.get("preregistration_path") not in {None, ""}:
        section["preregistration_path"] = str(
            _resolve_path(section["preregistration_path"], config_path)
        )
    if (section.get("rv_summary_spec") or {}).get("path") not in {None, ""}:
        section["rv_summary_spec"]["path"] = str(
            _resolve_path(section["rv_summary_spec"]["path"], config_path)
        )
    for probe_path_key in ("base_checkpoint", "residual_baseline_checkpoint"):
        if (section.get("probe") or {}).get(probe_path_key) not in {None, ""}:
            section["probe"][probe_path_key] = str(
                _resolve_path(section["probe"][probe_path_key], config_path)
            )
    if (section.get("pass_af") or {}).get("thresholds_path") not in {None, ""}:
        section["pass_af"]["thresholds_path"] = str(
            _resolve_path(section["pass_af"]["thresholds_path"], config_path)
        )

    active = active_path_c_sections(section)
    if active:
        _validate_explicit_runtime_fields(raw_section)
    preregistration = _load_preregistration_summary(
        section.get("preregistration_path"),
        config_path=config_path,
        required=bool(active),
        required_sections=active,
    )
    _validate_active_path_c_invariants(section, active)
    if active:
        frozen = load_frozen_preregistration(section["preregistration_path"])
        section["resolved_path_c_sha256"] = validate_runtime_path_c_config(
            section,
            frozen,
        )
        preregistration["runtime_contract_sha256"] = frozen.runtime_contract_sha256
    else:
        section["resolved_path_c_sha256"] = None
    section["preregistration"] = preregistration
    section["active_sections"] = active
    config["path_c"] = section
    return section


def active_path_c_sections(section: dict[str, Any]) -> list[str]:
    active: list[str] = []
    ensemble = section.get("ensemble") or {}
    if int(ensemble.get("n_heads", 1)) > 1:
        active.append("ensemble")
    if bool((section.get("probe") or {}).get("enable", False)):
        active.append("probe")
    for key in ("rv_summary_spec", "fingerprint", "cross_identity_split", "pass_af"):
        if bool((section.get(key) or {}).get("enable", False)):
            active.append(key)
    return active


def path_c_metadata(config: dict[str, Any]) -> dict[str, Any]:
    section = config.get("path_c") or default_path_c_config()
    probe = section.get("probe") or {}
    return {
        "active_sections": list(section.get("active_sections", [])),
        "ensemble": {
            "n_heads": int((section.get("ensemble") or {}).get("n_heads", 1)),
            "bootstrap_p": float((section.get("ensemble") or {}).get("bootstrap_p", 1.0)),
            "prior_scale": float((section.get("ensemble") or {}).get("prior_scale", 0.0)),
            "disagreement_stat": str(
                (section.get("ensemble") or {}).get("disagreement_stat", "variance")
            ),
            "tie_atol": float((section.get("ensemble") or {}).get("tie_atol", 1.0e-6)),
        },
        "probe": {
            "enable": bool(probe.get("enable", False)),
            "eval_enable": bool(probe.get("eval_enable", False)),
            "collection_enable": bool(probe.get("collection_enable", False)),
            "rule": str(probe.get("rule", "max_residual_signature_disagreement")),
            "disagreement_threshold": probe.get("disagreement_threshold"),
            "return_floor": probe.get("return_floor"),
            "base_checkpoint": probe.get("base_checkpoint"),
            "base_checkpoint_sha256": probe.get("base_checkpoint_sha256"),
            "residual_baseline_checkpoint": probe.get("residual_baseline_checkpoint"),
            "require_public_residual_baseline": bool(
                probe.get("require_public_residual_baseline", True)
            ),
            "allow_raw_q_fallback": bool(probe.get("allow_raw_q_fallback", False)),
            "min_selected_probes": probe.get("min_selected_probes"),
            "min_probe_opportunities": probe.get("min_probe_opportunities"),
            "min_context_coverage": probe.get("min_context_coverage"),
            "min_action_coverage": probe.get("min_action_coverage"),
        },
        "rv_summary_spec": {
            "enable": bool((section.get("rv_summary_spec") or {}).get("enable", False)),
            "path": (section.get("rv_summary_spec") or {}).get("path"),
        },
        "fingerprint": {
            "enable": bool((section.get("fingerprint") or {}).get("enable", False)),
            "kind": str(
                (section.get("fingerprint") or {}).get(
                    "kind",
                    "counterbalanced_value_null_candidate",
                )
            ),
            "vocab": list((section.get("fingerprint") or {}).get("vocab", [])),
            "positive_control_kind": str(
                (section.get("fingerprint") or {}).get(
                    "positive_control_kind",
                    "raw_response_value_null",
                )
            ),
            "negative_control_kind": str(
                (section.get("fingerprint") or {}).get(
                    "negative_control_kind",
                    "metadata_only",
                )
            ),
        },
        "cross_identity_split": {
            "enable": bool((section.get("cross_identity_split") or {}).get("enable", False)),
            "group_key": list(
                (section.get("cross_identity_split") or {}).get(
                    "group_key",
                    ["mode.family", "mode.param"],
                )
            ),
            "identity_key": str(
                (section.get("cross_identity_split") or {}).get(
                    "identity_key",
                    "surface_identity_key",
                )
            ),
            "split": str(
                (section.get("cross_identity_split") or {}).get(
                    "split",
                    "same_mechanism_disjoint_surface_seed_layout",
                )
            ),
            "folds": int((section.get("cross_identity_split") or {}).get("folds", 5)),
        },
        "pass_af": {
            "enable": bool((section.get("pass_af") or {}).get("enable", False)),
            "thresholds_path": (section.get("pass_af") or {}).get("thresholds_path"),
        },
        "preregistration": dict(section.get("preregistration") or {}),
        "resolved_path_c_sha256": section.get("resolved_path_c_sha256"),
    }


def require_preregistration_fields(
    config: dict[str, Any],
    required_paths: Iterable[str],
) -> None:
    prereg = _load_preregistration_payload(
        (config.get("path_c") or {}).get("preregistration_path"),
        config_path=None,
        required=True,
    )
    missing = [path for path in required_paths if _get_path(prereg, path.split(".")) is None]
    if missing:
        raise ValueError(
            "Path C preregistration is missing required field(s): "
            + ", ".join(sorted(missing))
        )


def _normalize_ensemble(section: dict[str, Any]) -> None:
    n_heads = int(section.get("n_heads", 1))
    if n_heads <= 0:
        raise ValueError("path_c.ensemble.n_heads must be positive.")
    section["n_heads"] = n_heads
    bootstrap_p = float(section.get("bootstrap_p", 1.0))
    if not 0.0 < bootstrap_p <= 1.0:
        raise ValueError("path_c.ensemble.bootstrap_p must be in (0, 1].")
    section["bootstrap_p"] = bootstrap_p
    prior_scale = float(section.get("prior_scale", 0.0))
    if prior_scale < 0.0:
        raise ValueError("path_c.ensemble.prior_scale must be non-negative.")
    section["prior_scale"] = prior_scale
    stat = str(section.get("disagreement_stat", "variance")).lower()
    if stat not in {"variance", "range"}:
        raise ValueError(
            "path_c.ensemble.disagreement_stat must be one of {'variance', 'range'}."
        )
    section["disagreement_stat"] = stat
    tie_atol = float(section.get("tie_atol", 1.0e-6))
    if tie_atol < 0.0:
        raise ValueError("path_c.ensemble.tie_atol must be non-negative.")
    section["tie_atol"] = tie_atol


def _normalize_probe(section: dict[str, Any]) -> None:
    section["enable"] = _as_bool(section.get("enable", False), "path_c.probe.enable")
    section["eval_enable"] = _as_bool(
        section.get("eval_enable", False),
        "path_c.probe.eval_enable",
    )
    section["collection_enable"] = _as_bool(
        section.get("collection_enable", False),
        "path_c.probe.collection_enable",
    )
    threshold = section.get("disagreement_threshold")
    if threshold is not None:
        threshold = float(threshold)
        if threshold < 0.0:
            raise ValueError("path_c.probe.disagreement_threshold must be non-negative.")
    section["disagreement_threshold"] = threshold
    floor = section.get("return_floor")
    section["return_floor"] = None if floor is None else float(floor)
    base_sha = section.get("base_checkpoint_sha256")
    section["base_checkpoint_sha256"] = None if base_sha in {None, ""} else str(base_sha).lower()
    if section["base_checkpoint_sha256"] is not None and (
        len(section["base_checkpoint_sha256"]) != 64
        or any(char not in "0123456789abcdef" for char in section["base_checkpoint_sha256"])
    ):
        raise ValueError("path_c.probe.base_checkpoint_sha256 must be a SHA-256 digest.")
    rule = str(section.get("rule", "max_residual_signature_disagreement")).lower()
    if rule != "max_residual_signature_disagreement":
        raise ValueError(
            "path_c.probe.rule currently supports only 'max_residual_signature_disagreement'."
        )
    section["rule"] = rule
    section["require_public_residual_baseline"] = _as_bool(
        section.get("require_public_residual_baseline", section.get("require_residual_baseline", True)),
        "path_c.probe.require_public_residual_baseline",
    )
    section["allow_raw_q_fallback"] = _as_bool(
        section.get("allow_raw_q_fallback", False),
        "path_c.probe.allow_raw_q_fallback",
    )
    if section["allow_raw_q_fallback"]:
        raise ValueError(
            "Path C cannot enable path_c.probe.allow_raw_q_fallback; probes must target "
            "residual-control-signature disagreement."
        )
    for key in ("min_selected_probes", "min_probe_opportunities"):
        value = section.get(key)
        section[key] = None if value is None else int(value)
        if section[key] is not None and section[key] < 0:
            raise ValueError(f"path_c.probe.{key} must be non-negative.")
    for key in ("min_context_coverage", "min_action_coverage"):
        value = section.get(key)
        section[key] = None if value is None else float(value)
        if section[key] is not None and not 0.0 <= section[key] <= 1.0:
            raise ValueError(f"path_c.probe.{key} must be in [0, 1].")
    for key in ("base_checkpoint", "residual_baseline_checkpoint"):
        value = section.get(key)
        section[key] = None if value in {None, ""} else str(value)


def _normalize_enabled_path(section: dict[str, Any], name: str) -> None:
    section["enable"] = _as_bool(section.get("enable", False), f"path_c.{name}.enable")


def _validate_active_path_c_invariants(section: dict[str, Any], active: Iterable[str]) -> None:
    active_set = set(active)
    ensemble = section.get("ensemble") or {}
    probe = section.get("probe") or {}
    n_heads = int(ensemble.get("n_heads", 1))
    if "ensemble" in active_set and n_heads > 1:
        bootstrap_p = float(ensemble.get("bootstrap_p", 1.0))
        prior_scale = float(ensemble.get("prior_scale", 0.0))
        if bootstrap_p >= 1.0 or prior_scale <= 0.0:
            raise ValueError(
                "Path C ensemble diversity requires both bootstrap_p < 1.0 and "
                "prior_scale > 0.0 when n_heads > 1."
            )
    if "probe" in active_set:
        if n_heads <= 1:
            raise ValueError("path_c.probe.enable=true requires path_c.ensemble.n_heads > 1.")
        if bool(probe.get("require_public_residual_baseline", True)) and not (
            probe.get("base_checkpoint") or probe.get("residual_baseline_checkpoint")
        ):
            raise ValueError(
                "path_c.probe.base_checkpoint or path_c.probe.residual_baseline_checkpoint is "
                "required when probes are enabled so probe selection targets residual-control "
                "signature disagreement, not raw Q."
            )
        if probe.get("return_floor") is None:
            raise ValueError(
                "path_c.probe.return_floor must be frozen and non-null when probes are enabled."
            )
        if probe.get("base_checkpoint_sha256") is None:
            raise ValueError(
                "path_c.probe.base_checkpoint_sha256 must be frozen when probes are enabled."
            )
        missing_support = [
            key for key in (
                "min_selected_probes", "min_probe_opportunities",
                "min_context_coverage", "min_action_coverage",
            )
            if probe.get(key) is None
        ]
        if missing_support:
            raise ValueError(
                "Path C enabled probe lacks frozen support threshold(s): "
                + ", ".join(missing_support)
            )
        if not bool(probe.get("collection_enable", False)):
            raise ValueError(
                "path_c.probe.collection_enable must be true when active probes are enabled."
            )
        if bool(probe.get("eval_enable", False)):
            raise ValueError(
                "path_c.probe.eval_enable must remain false in the frozen runtime config."
            )


def _load_preregistration_summary(
    path_value: Any,
    *,
    config_path: str | Path | None,
    required: bool,
    required_sections: Iterable[str],
) -> dict[str, Any]:
    if path_value in {None, ""}:
        if required:
            raise ValueError(
                "path_c.preregistration_path is required when any Path C feature is enabled."
            )
        return {"loaded": False, "path": None, "sha256": None, "sections": []}
    path = _resolve_path(path_value, config_path)
    payload = _load_preregistration_payload(path, config_path=None, required=True)
    if required and str(payload.get("status", "")).strip().lower() != "frozen":
        raise ValueError(
            "Path C preregistration must have status: frozen before any Path C "
            f"feature is enabled; got {payload.get('status')!r}."
        )
    if required:
        required_top_level = ("version", "freeze_timestamp", "pass_af", "budget", "baselines")
        missing_meta = []
        for key in required_top_level:
            value = payload.get(key)
            if value is None or value == "":
                missing_meta.append(key)
        if missing_meta:
            raise ValueError(
                "Path C frozen preregistration is missing required metadata/decision field(s): "
                + ", ".join(missing_meta)
            )
    missing = [section for section in required_sections if section not in payload]
    if missing:
        raise ValueError(
            "Path C preregistration missing section(s) required by enabled feature(s): "
            + ", ".join(sorted(missing))
        )
    rv_spec = payload.get("rv_summary_spec")
    rv_spec_sha = None
    if isinstance(rv_spec, dict):
        rv_cols = [str(c) for c in rv_spec.get("rv_columns", [])]
        forbidden = {
            "resp_rv_value_event_target_x",
            "resp_rv_value_event_target_y",
            "identity",
            "seed",
            "style",
            "layout_style",
            "trajectory_source",
        }
        bad_cols = sorted(c for c in rv_cols if c in forbidden or c.endswith("_x") or c.endswith("_y"))
        if bad_cols:
            raise ValueError(
                "Path C frozen rv_summary_spec contains forbidden nuisance/coordinate column(s): "
                + ", ".join(bad_cols)
            )
        rv_spec_sha = hashlib.sha256(
            json.dumps(rv_spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    raw = path.read_bytes()
    return {
        "loaded": True,
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "status": payload.get("status"),
        "version": payload.get("version"),
        "freeze_timestamp": payload.get("freeze_timestamp"),
        "rv_summary_spec_sha256": rv_spec_sha,
        "sections": sorted(str(key) for key in payload.keys()),
    }


def _load_preregistration_payload(
    path_value: Any,
    *,
    config_path: str | Path | None,
    required: bool,
) -> dict[str, Any]:
    if path_value in {None, ""}:
        if required:
            raise ValueError("Path C preregistration path is required.")
        return {}
    path = _resolve_path(path_value, config_path)
    if not path.exists():
        raise FileNotFoundError(f"Path C preregistration file does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Path C preregistration file must be a YAML mapping: {path}")
    return payload


def _resolve_path(path_value: Any, config_path: str | Path | None) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    if config_path is not None:
        base = Path(config_path).resolve().parent
        return (base / path).resolve()
    return path.resolve()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _validate_explicit_runtime_fields(section: dict[str, Any]) -> None:
    required = {
        "ensemble": (
            "n_heads", "bootstrap_p", "prior_scale", "disagreement_stat", "tie_atol",
        ),
        "probe": (
            "enable", "eval_enable", "collection_enable", "rule",
            "disagreement_threshold", "return_floor",
            "base_checkpoint_sha256",
            "require_public_residual_baseline", "allow_raw_q_fallback",
            "min_selected_probes", "min_probe_opportunities",
            "min_context_coverage", "min_action_coverage",
        ),
        "rv_summary_spec": ("enable",),
        "fingerprint": (
            "enable", "kind", "vocab", "positive_control_kind",
            "negative_control_kind",
        ),
        "cross_identity_split": (
            "enable", "group_key", "identity_key", "split", "folds",
        ),
        "pass_af": ("enable",),
    }
    missing = []
    for section_name, fields in required.items():
        values = section.get(section_name)
        if not isinstance(values, dict):
            missing.append(section_name)
            continue
        for field in fields:
            if field not in values:
                missing.append(f"{section_name}.{field}")
    if "preregistration_path" not in section:
        missing.append("preregistration_path")
    if missing:
        raise ValueError(
            "Active Path C runtime config must explicitly specify every frozen field; missing: "
            + ", ".join(missing)
        )


def _get_path(mapping: dict[str, Any], keys: list[str]) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _as_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"1", "true", "yes", "on"}:
            return True
        if low in {"0", "false", "no", "off"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    raise ValueError(f"{name} must be boolean-like; got {value!r}.")
