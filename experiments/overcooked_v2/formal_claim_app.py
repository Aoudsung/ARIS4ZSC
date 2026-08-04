"""Fail-closed synthesis of DEPI performance and mechanism-attribution gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from experiments.overcooked_v2.resource_report_app import RESOURCE_METHODS
from src.path_c.calibration import calibration_pass_decision
from src.path_c.experiment import (
    METHOD_VERSION,
    OFFICIAL_PROTOCOL_VERSION,
    OFFICIAL_SOURCE_COMMIT,
    load_config,
)
from src.path_c.official_statistics import registered_superiority_gate
from src.path_c.resources import ResourceLedger
from src.path_c.storage import (
    ensure_run_identity,
    read_parquet,
    runtime_provenance,
    sha256_path,
    validate_formal_repository_state,
    validate_registered_python_runtime,
    write_json,
)


LAYOUTS = ("test_time_simple", "test_time_wide")
MECHANISM_ARTIFACTS = (
    "development_matrix",
    "posterior_calibration",
    "identifiability",
    "recoverable_value",
)


def _json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected a JSON mapping: {path}")
    return payload


def _source(path: Path) -> Mapping[str, str]:
    return {"path": str(path), "sha256": sha256_path(path)}


def _validate_source_ref(value: Any, *, label: str) -> Path:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} source reference fields differ.")
    path = Path(str(value["path"])).resolve()
    if sha256_path(path) != str(value["sha256"]):
        raise ValueError(f"{label} source hash changed: {path}")
    return path


def _validate_source_tree(value: Any, *, label: str) -> None:
    if isinstance(value, Mapping) and set(value) == {"path", "sha256"}:
        _validate_source_ref(value, label=label)
        return
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} source tree is empty or malformed.")
    for name, child in value.items():
        _validate_source_tree(child, label=f"{label}.{name}")


def _validate_official_summary(payload: Mapping[str, Any]) -> bool:
    if (
        payload.get("version") != 2
        or payload.get("artifact_type") != "depi_official_scoreboard_summary"
        or payload.get("method") != METHOD_VERSION
        or payload.get("official_protocol_version") != OFFICIAL_PROTOCOL_VERSION
        or payload.get("official_source_commit") != OFFICIAL_SOURCE_COMMIT
    ):
        raise ValueError("Official-summary identity differs from the registered schema.")
    layouts = payload.get("layouts")
    sources = payload.get("source_results")
    registration = payload.get("statistical_preregistration")
    if not all(isinstance(item, Mapping) for item in (layouts, sources, registration)):
        raise ValueError("Official-summary sections are malformed.")
    if set(layouts) != set(LAYOUTS) or set(registration) != set(LAYOUTS):
        raise ValueError("Official summary must cover both registered layouts.")
    expected_sources = {
        f"{layout}:{method}"
        for layout in LAYOUTS
        for method in ("sp", "state-augmented", "op", "fcp", "depi")
    }
    if set(sources) != expected_sources:
        raise ValueError("Official summary does not bind all 5x2 raw result sources.")
    for label, ref in sources.items():
        directory = _validate_source_ref(ref, label=f"official.{label}")
        layout, method = label.split(":", 1)
        identity = _json(directory / "run_identity.json")
        if (
            identity.get("stage") != "official-10x10-evaluate"
            or identity.get("layout") != layout
            or identity.get("method") != method
        ):
            raise ValueError(f"Official raw source identity differs for {label}.")
        ResourceLedger.from_mapping(
            _json(directory / "budget_ledger.json")
        )
    passed = True
    for layout in LAYOUTS:
        prereg = registration[layout]
        config_ref = {"path": prereg.get("config"), "sha256": prereg.get("config_sha256")}
        config_path = _validate_source_ref(config_ref, label=f"official.config.{layout}")
        config = load_config(config_path, run_kind="formal")
        if (
            config.environment.layout != layout
            or prereg.get("config_fingerprint") != config.fingerprint
            or prereg.get("inference_mode") != "independent_run"
            or float(prereg.get("one_sided_alpha", -1.0)) != 0.05
            or float(prereg.get("superiority_lcb_threshold", -1.0)) != 0.0
            or float(prereg.get("minimum_effect", -1.0)) != 20.0
            or prereg.get("minimum_effect_rule") != "point_estimate"
        ):
            raise ValueError(f"Official statistical registration differs on {layout}.")
        current = layouts[layout]
        comparison = current.get("bootstrap", {}).get("depi_vs_best_baseline")
        if not isinstance(comparison, Mapping):
            raise ValueError(f"Official bootstrap contrast is missing on {layout}.")
        gate = registered_superiority_gate(
            comparison,
            lcb_threshold=0.0,
            minimum_effect=20.0,
            minimum_effect_rule="point_estimate",
        )
        xp = bool(gate["passed"])
        competence_comparison = current.get("bootstrap", {}).get(
            "depi_sp_minus_fcp_sp"
        )
        if not isinstance(competence_comparison, Mapping):
            raise ValueError(f"Official competence contrast is missing on {layout}.")
        competence = float(competence_comparison["one_sided_lcb"]) > -20.0
        if (
            current.get("xp_gate_passed") is not xp
            or current.get("superiority_lcb_passed") is not gate["superiority_lcb_passed"]
            or current.get("minimum_effect_passed") is not gate["minimum_effect_passed"]
            or current.get("competence_gate_passed") is not competence
            or float(current.get("correct_delivery_margin", -1.0)) != 20.0
        ):
            raise ValueError(f"Official gate booleans are not derived from data on {layout}.")
        passed = passed and xp and competence
    if payload.get("primary_benchmark_gate_passed") is not passed:
        raise ValueError("Official primary gate is not the conjunction of layout gates.")
    return passed


def _validate_final_m1_from_official_summary(payload: Mapping[str, Any]) -> bool:
    """Revalidate the ten final-checkpoint M1 nodes evaluated on each layout."""

    from experiments.overcooked_v2.official_evaluation_app import (
        _load_policy_manifest,
    )

    sources = payload.get("source_results")
    if not isinstance(sources, Mapping):
        raise ValueError("Official summary has no raw result sources for M1 validation.")
    all_passed = True
    for layout in LAYOUTS:
        ref = sources.get(f"{layout}:depi")
        directory = _validate_source_ref(ref, label=f"official.{layout}:depi.m1")
        identity = _json(directory / "run_identity.json")
        policy_ref = identity.get("policy_manifest")
        if not isinstance(policy_ref, Mapping) or set(policy_ref) != {
            "path",
            "sha256",
            "content",
        }:
            raise ValueError(f"Official DEPI policy-manifest source is malformed on {layout}.")
        manifest_path = Path(str(policy_ref["path"])).resolve()
        if sha256_path(manifest_path) != str(policy_ref["sha256"]):
            raise ValueError(f"Official DEPI policy manifest changed on {layout}.")
        manifest = _load_policy_manifest(manifest_path, expected_layout=layout)
        if manifest != policy_ref["content"]:
            raise ValueError(
                f"Official DEPI policy-manifest content differs from evaluated identity on {layout}."
            )
        rows = manifest.get("m1_final_evaluations")
        if not isinstance(rows, list) or len(rows) != 10:
            raise ValueError(f"Final-checkpoint M1 evidence is incomplete on {layout}.")
        all_passed = all_passed and all(row["passed"] is True for row in rows)
    return bool(all_passed)


def _validate_capacity_summary(payload: Mapping[str, Any]) -> bool:
    if (
        payload.get("version") != 2
        or payload.get("artifact_type") != "depi_capacity_control_summary"
        or payload.get("method") != METHOD_VERSION
        or payload.get("official_protocol_version") != OFFICIAL_PROTOCOL_VERSION
        or payload.get("official_source_commit") != OFFICIAL_SOURCE_COMMIT
    ):
        raise ValueError("Capacity-summary identity differs from the registered schema.")
    layouts = payload.get("layouts")
    sources = payload.get("source_results")
    expected_sources = {
        f"{layout}:{method}"
        for layout in LAYOUTS
        for method in ("depi", "ippo-large")
    }
    if not isinstance(layouts, Mapping) or set(layouts) != set(LAYOUTS):
        raise ValueError("Capacity control must cover both layouts.")
    if not isinstance(sources, Mapping) or set(sources) != expected_sources:
        raise ValueError("Capacity summary does not bind all raw sources.")
    for label, ref in sources.items():
        directory = _validate_source_ref(ref, label=f"capacity.{label}")
        layout, method = label.split(":", 1)
        identity = _json(directory / "run_identity.json")
        if (
            identity.get("stage") != "official-10x10-evaluate"
            or identity.get("layout") != layout
            or identity.get("method") != method
        ):
            raise ValueError(f"Capacity raw source identity differs for {label}.")
        ResourceLedger.from_mapping(_json(directory / "budget_ledger.json"))
    passed = True
    for layout, current in layouts.items():
        comparison = current.get("bootstrap")
        match = current.get("capacity_match")
        if not isinstance(comparison, Mapping) or not isinstance(match, Mapping):
            raise ValueError(f"Capacity evidence is incomplete on {layout}.")
        current_pass = float(comparison.get("one_sided_lcb", float("-inf"))) > 0.0
        if current.get("capacity_explanation_rejected") is not current_pass:
            raise ValueError(f"Capacity gate is not derived from its contrast on {layout}.")
        if abs(int(match["observed_parameters"]) - int(match["target_parameters"])) != int(
            match["absolute_mismatch"]
        ):
            raise ValueError(f"Capacity parameter matching is inconsistent on {layout}.")
        passed = passed and current_pass
    if payload.get("capacity_control_gate_passed") is not passed:
        raise ValueError("Capacity overall gate is not the conjunction of layout gates.")
    return passed


def _validate_resource_report(payload: Mapping[str, Any]) -> None:
    if set(payload) != {"methods", "sources", "accounting_rule"}:
        raise ValueError("Resource report fields differ from the registered schema.")
    methods = payload["methods"]
    sources = payload["sources"]
    if not isinstance(methods, list) or not isinstance(sources, Mapping):
        raise ValueError("Resource report methods/sources are malformed.")
    index = {}
    for row in methods:
        if not isinstance(row, Mapping) or "method" not in row:
            raise ValueError("Resource report method row is malformed.")
        method = str(row["method"])
        if method in index:
            raise ValueError(f"Duplicate resource-report method: {method}")
        ledger = ResourceLedger.from_mapping(
            {name: value for name, value in row.items() if name != "method"}
        )
        if (
            ledger.total_training_simulator_steps <= 0
            or ledger.deployable_parameters <= 0
            or ledger.inference_latency_ms <= 0.0
        ):
            raise ValueError(f"Formal resource row is incomplete for {method}.")
        index[method] = ledger
    if set(index) != set(RESOURCE_METHODS) or set(sources) != set(RESOURCE_METHODS):
        raise ValueError("Formal resource report does not cover every registered method.")
    for method, artifacts in sources.items():
        if not isinstance(artifacts, list) or not artifacts:
            raise ValueError(f"Resource report has no source ledgers for {method}.")
        for artifact in artifacts:
            if not isinstance(artifact, Mapping) or set(artifact) != {"path", "sha256"}:
                raise ValueError("Resource source fields differ.")
            if sha256_path(artifact["path"]) != artifact["sha256"]:
                raise ValueError(f"Resource source hash changed: {artifact['path']}")


def _validate_development_matrix(payload: Mapping[str, Any]) -> bool:
    if (
        payload.get("version") != 1
        or payload.get("artifact_type") != "depi_development_matrix_summary"
        or payload.get("method") != METHOD_VERSION
        or payload.get("b3_status") != "not_implemented"
    ):
        raise ValueError("Development-matrix summary identity differs.")
    comparisons = payload.get("primary_k4_paired_increments")
    if not isinstance(comparisons, Mapping):
        raise ValueError("Development matrix has no paired increments.")
    required = {"b1_minus_b0", "b2_minus_b1", "b2_minus_b0"}
    if set(comparisons) != required:
        raise ValueError("Development matrix does not cover B0--B2 increments.")
    _validate_source_tree(payload.get("sources"), label="development")
    matrix_path = Path(str(payload["sources"]["matrix"]["path"])).resolve()
    scores_path = Path(str(payload["sources"]["scores"]["path"])).resolve()
    matrix = _json(matrix_path)
    scores = _json(scores_path)
    if (
        matrix.get("artifact_type") != "depi_development_matrix"
        or scores.get("artifact_type") != "depi_development_scores"
        or matrix.get("method") != METHOD_VERSION
    ):
        raise ValueError("Development summary sources have different identities.")
    if matrix.get("budget_capacity_and_key_matching_passed") is not True:
        raise ValueError("Development matrix did not establish budget/capacity/key matching.")
    return bool(
        comparisons["b1_minus_b0"]["bootstrap_99_percent_ci"][0] > 0.0
        and comparisons["b2_minus_b1"]["bootstrap_99_percent_ci"][0] > 0.0
    )


def _validate_common_summary(payload: Mapping[str, Any], *, layout: str) -> bool:
    if (
        payload.get("version") != 2
        or payload.get("artifact_type") != "depi_common_partner_evaluation"
        or payload.get("method") != METHOD_VERSION
        or payload.get("method_variant") != "b2"
        or payload.get("layout") != layout
        or payload.get("official_protocol_version") != OFFICIAL_PROTOCOL_VERSION
        or payload.get("official_source_commit") != OFFICIAL_SOURCE_COMMIT
        or payload.get("paired_crn") is not True
    ):
        raise ValueError(f"Common-Partner identity differs on {layout}.")
    schedule = str(payload.get("episode_key_schedule_sha256", ""))
    if len(schedule) != 64:
        raise ValueError(f"Common-Partner key schedule is not hash-bound on {layout}.")
    sources = payload.get("sources")
    _validate_source_tree(sources, label=f"common.{layout}")
    policy_sources = sources.get("policy_manifests")
    expected_methods = {"sp", "state-augmented", "op", "fcp", "depi"}
    if not isinstance(policy_sources, Mapping) or set(policy_sources) != expected_methods:
        raise ValueError(f"Common-Partner policy manifests are incomplete on {layout}.")
    for method, ref in policy_sources.items():
        manifest = _json(Path(str(ref["path"])).resolve())
        if manifest.get("method") != method or manifest.get("layout") != layout:
            raise ValueError(f"Common-Partner policy lineage differs for {method}/{layout}.")
    methods = payload.get("methods")
    if not isinstance(methods, Mapping) or set(methods) != expected_methods:
        raise ValueError(f"Common-Partner method table is incomplete on {layout}.")
    raw_rows = read_parquet(Path(str(sources["raw_episodes"]["path"])).resolve())
    expected_row_count = 5 * 10 * 18 * 2 * 500
    if (
        len(raw_rows) != expected_row_count
        or {str(row["layout"]) for row in raw_rows} != {layout}
        or {str(row["method"]) for row in raw_rows} != expected_methods
        or {str(row["episode_key_schedule_sha256"]) for row in raw_rows}
        != {schedule}
    ):
        raise ValueError(f"Common-Partner raw episode nodes are incomplete on {layout}.")
    ledger = ResourceLedger.from_mapping(payload.get("resource_ledger"))
    if ledger.evaluation_steps <= 0:
        raise ValueError(f"Common-Partner resource ledger is empty on {layout}.")
    prereg = payload.get("statistical_preregistration")
    if (
        not isinstance(prereg, Mapping)
        or float(prereg.get("superiority_lcb_threshold", -1.0)) != 0.0
        or float(prereg.get("minimum_effect", -1.0)) != 20.0
        or prereg.get("minimum_effect_rule") != "point_estimate"
    ):
        raise ValueError(f"Common-Partner statistical registration differs on {layout}.")
    bootstrap = payload.get("bootstrap")
    if not isinstance(bootstrap, Mapping):
        raise ValueError(f"Common-Partner bootstrap is missing on {layout}.")
    gate = registered_superiority_gate(
        bootstrap,
        lcb_threshold=0.0,
        minimum_effect=20.0,
        minimum_effect_rule="point_estimate",
    )
    if (
        payload.get("common_partner_gate_passed") is not gate["passed"]
        or payload.get("superiority_lcb_passed") is not gate["superiority_lcb_passed"]
        or payload.get("minimum_effect_passed") is not gate["minimum_effect_passed"]
        or payload.get("br_prox_complete") is not True
    ):
        raise ValueError(f"Common-Partner gates are not data-derived on {layout}.")
    return bool(gate["passed"])


def _validate_posterior_calibration(
    payload: Mapping[str, Any], *, layout: str
) -> bool:
    if (
        payload.get("version") != 2
        or payload.get("artifact_type") != "depi_posterior_calibration"
        or payload.get("artifact_name") != "DEPI-Posterior-Calibration"
        or payload.get("method") != METHOD_VERSION
        or payload.get("method_variant") != "b2"
        or payload.get("layout") != layout
        or payload.get("official_protocol_version") != OFFICIAL_PROTOCOL_VERSION
        or payload.get("official_source_commit") != OFFICIAL_SOURCE_COMMIT
    ):
        raise ValueError(f"Posterior-calibration identity differs on {layout}.")
    sources = payload.get("sources")
    _validate_source_tree(sources, label=f"posterior_calibration.{layout}")
    config_path = Path(str(sources["config"]["path"])).resolve()
    config = load_config(config_path, run_kind="formal")
    if (
        config.environment.layout != layout
        or config.method_variant != "b2"
        or payload.get("config_fingerprint") != config.fingerprint
    ):
        raise ValueError(f"Posterior-calibration config lineage differs on {layout}.")
    ledger = ResourceLedger.from_mapping(payload.get("resource_ledger"))
    if ledger.calibration_steps <= 0:
        raise ValueError(f"Posterior-calibration resource ledger is empty on {layout}.")
    registration = payload.get("registered")
    if (
        not isinstance(registration, Mapping)
        or float(registration.get("log_score_margin", -1.0)) != 0.02
        or float(registration.get("coverage_low", -1.0)) != 0.85
        or float(registration.get("coverage_high", -1.0)) != 0.95
        or float(registration.get("brier_ratio", -1.0)) != 0.90
        or registration.get("coverage_targets") != ["relative_position", "direction"]
        or registration.get("primary_aggregation_unit") != "partner_run"
        or registration.get("secondary_aggregation_unit") != "episode"
    ):
        raise ValueError(f"Posterior-calibration registration differs on {layout}.")
    runs = payload.get("runs")
    if not isinstance(runs, list) or len(runs) < 20:
        raise ValueError(f"Posterior calibration has fewer than 20 run blocks on {layout}.")
    score_rows = read_parquet(
        Path(str(sources["calibration_scores"]["path"])).resolve()
    )
    if (
        len(score_rows) != len(runs)
        or {str(row["partner_run_id"]) for row in score_rows}
        != {str(row["partner_run_id"]) for row in runs}
        or any(int(row["episodes"]) != 64 for row in score_rows)
    ):
        raise ValueError(f"Posterior-calibration run-level score nodes differ on {layout}.")
    aggregate = payload.get("aggregate")
    if not isinstance(aggregate, Mapping):
        raise ValueError(f"Posterior-calibration aggregate is missing on {layout}.")
    expected = {
        "log_score_vs_uniform": float(aggregate["log_score"])
        <= float(aggregate["uniform_baseline_log_score"]) - 0.02,
        "log_score_vs_no_history": float(aggregate["log_score"])
        <= float(aggregate["no_history_baseline_log_score"]) - 0.02,
        "position_coverage_in_band": 0.85
        <= float(aggregate["coverage_position"])
        <= 0.95,
        "direction_coverage_in_band": 0.85
        <= float(aggregate["coverage_direction"])
        <= 0.95,
        "event_brier": float(aggregate["event_brier"])
        <= 0.90 * float(aggregate["prior_baseline_brier"]),
    }
    expected_overall = calibration_pass_decision(
        model_log_score=float(aggregate["log_score"]),
        uniform_baseline_log_score=float(aggregate["uniform_baseline_log_score"]),
        no_history_baseline_log_score=float(aggregate["no_history_baseline_log_score"]),
        position_coverage=float(aggregate["coverage_position"]),
        direction_coverage=float(aggregate["coverage_direction"]),
        event_brier=float(aggregate["event_brier"]),
        prior_baseline_brier=float(aggregate["prior_baseline_brier"]),
        log_score_margin=0.02,
        coverage_low=0.85,
        coverage_high=0.95,
        brier_ratio=0.90,
    )
    pass_payload = payload.get("pass")
    if not isinstance(pass_payload, Mapping) or any(
        pass_payload.get(name) is not value for name, value in expected.items()
    ) or pass_payload.get("overall") is not bool(expected_overall):
        raise ValueError(f"Posterior-calibration gates are not score-derived on {layout}.")
    return bool(expected_overall)


def _validate_layout_artifact(
    payload: Mapping[str, Any], *, layout: str, artifact_type: str
) -> bool:
    if payload.get("method") != METHOD_VERSION:
        raise ValueError(f"{artifact_type} belongs to another method version.")
    if payload.get("layout") != layout:
        raise ValueError(f"{artifact_type} layout differs from its label.")
    if payload.get("artifact_type") != artifact_type:
        raise ValueError(f"{artifact_type} artifact identity differs.")
    if payload.get("version") != 2 or payload.get("method_variant") != "b2":
        raise ValueError(f"{artifact_type} schema or B2 identity differs.")
    if payload.get("paired_crn") is not True:
        raise ValueError(f"{artifact_type} is not a paired-CRN evaluation.")
    ledger = payload.get("resource_ledger")
    if (
        not isinstance(ledger, Mapping)
        or int(ledger.get("continuation_steps", 0)) <= 0
        or int(ledger.get("total_simulator_steps", 0))
        < int(ledger.get("continuation_steps", 0))
    ):
        raise ValueError(f"{artifact_type} has no complete simulator ledger.")
    source_path = _validate_source_ref(
        payload.get("source"), label=f"{artifact_type}.{layout}.raw"
    )
    raw = _json(source_path)
    expected_raw_type = artifact_type.replace("_evaluation", "_raw")
    if (
        raw.get("artifact_type") != expected_raw_type
        or raw.get("method") != METHOD_VERSION
        or raw.get("method_variant") != "b2"
        or raw.get("layout") != layout
        or raw.get("policy_manifest_sha256")
        != payload.get("policy_manifest_sha256")
        or raw.get("partner_manifest_sha256")
        != payload.get("partner_manifest_sha256")
    ):
        raise ValueError(f"{artifact_type} raw lineage differs from its summary.")
    if artifact_type == "depi_identifiability_evaluation":
        if payload.get("context_swap", {}).get("measurement") != (
            "real_crn_all_action_continuation"
        ):
            raise ValueError("Context swap is not based on real continuations.")
        task = payload.get("task_leakage", {})
        shuffle = payload.get("history_shuffle", {})
        swap = payload.get("context_swap", {})
        expected_flags = {
            "task_channel_isolation": float(
                task.get("representation_shuffle_drift", float("inf"))
            )
            <= 1.0e-7,
            "task_leakage_probe": float(
                task.get("balanced_accuracy", float("inf"))
            )
            <= float(task.get("chance_accuracy", float("-inf")))
            + float(task.get("maximum_excess_over_chance", float("-inf"))),
            "history_shuffle": float(
                shuffle.get("mean_return_drop", float("-inf"))
            )
            > 0.0
            and float(
                shuffle.get("bootstrap_99_percent_ci", [float("-inf")])[0]
            )
            > 0.0,
            "swap_c_causal_consistency": float(
                swap.get("swap_c_continuation_consistency", float("-inf"))
            )
            > float(swap.get("registered_swap_c_threshold", float("inf"))),
        }
        expected_flags["overall"] = bool(all(expected_flags.values()))
        if payload.get("pass") != expected_flags:
            raise ValueError("Identifiability pass fields are not metric-derived.")
    if artifact_type == "depi_recoverable_value_evaluation":
        fraction = payload.get("recoverable_fraction")
        if (
            not isinstance(fraction, Mapping)
            or fraction.get("estimable") is not True
            or float(fraction.get("signal_threshold", -1.0)) != 20.0
        ):
            raise ValueError("Recoverable-value rho is absent or unregistered.")
        comparisons = payload.get("comparisons", {})
        expected_flags = {
            "legal_history_beats_shuffled": float(
                comparisons.get("g1_minus_g2", {})
                .get("bootstrap_99_percent_ci", [float("-inf")])[0]
            )
            > 0.0,
            "legal_history_beats_state_only": float(
                comparisons.get("g1_minus_g3", {})
                .get("bootstrap_99_percent_ci", [float("-inf")])[0]
            )
            > 0.0,
            "oracle_is_upper_bound": float(
                comparisons.get("g4_minus_g1", {}).get(
                    "point_difference", float("-inf")
                )
            )
            >= 0.0,
            "recoverable_fraction_estimable": bool(fraction["estimable"]),
        }
        expected_flags["overall"] = bool(all(expected_flags.values()))
        if payload.get("pass") != expected_flags:
            raise ValueError("Recoverable-value pass fields are not metric-derived.")
    passed = payload.get("pass")
    if not isinstance(passed, Mapping) or "overall" not in passed:
        raise ValueError(f"{artifact_type} has no registered pass decision.")
    return passed["overall"] is True


def run_formal_claim_report(args: argparse.Namespace) -> None:
    validate_formal_repository_state()
    validate_registered_python_runtime()
    paths = {
        "official": Path(args.official_summary).resolve(),
        "capacity": Path(args.capacity_summary).resolve(),
        "resources": Path(args.resource_report).resolve(),
        "development_matrix": Path(args.development_matrix).resolve(),
    }
    official = _json(paths["official"])
    capacity = _json(paths["capacity"])
    resources = _json(paths["resources"])
    development = _json(paths["development_matrix"])
    _validate_resource_report(resources)

    common_paths = {
        "test_time_simple": Path(args.common_simple).resolve(),
        "test_time_wide": Path(args.common_wide).resolve(),
    }
    calibration_paths = {
        "test_time_simple": Path(args.posterior_calibration_simple).resolve(),
        "test_time_wide": Path(args.posterior_calibration_wide).resolve(),
    }
    identifiability_paths = {
        "test_time_simple": Path(args.identifiability_simple).resolve(),
        "test_time_wide": Path(args.identifiability_wide).resolve(),
    }
    recoverable_paths = {
        "test_time_simple": Path(args.recoverable_value_simple).resolve(),
        "test_time_wide": Path(args.recoverable_value_wide).resolve(),
    }
    common = {layout: _json(path) for layout, path in common_paths.items()}
    calibration = {
        layout: _json(path) for layout, path in calibration_paths.items()
    }
    identifiability = {
        layout: _json(path) for layout, path in identifiability_paths.items()
    }
    recoverable = {
        layout: _json(path) for layout, path in recoverable_paths.items()
    }

    official_passed = _validate_official_summary(official)
    final_m1_passed = _validate_final_m1_from_official_summary(official)
    common_passed = all(
        _validate_common_summary(common[layout], layout=layout)
        for layout in LAYOUTS
    )
    capacity_passed = _validate_capacity_summary(capacity)
    development_passed = _validate_development_matrix(development)
    calibration_passed = all(
        _validate_posterior_calibration(calibration[layout], layout=layout)
        for layout in LAYOUTS
    )
    identifiability_passed = all(
        _validate_layout_artifact(
            identifiability[layout],
            layout=layout,
            artifact_type="depi_identifiability_evaluation",
        )
        for layout in LAYOUTS
    )
    recoverable_passed = all(
        _validate_layout_artifact(
            recoverable[layout],
            layout=layout,
            artifact_type="depi_recoverable_value_evaluation",
        )
        for layout in LAYOUTS
    )
    for layout in LAYOUTS:
        if (
            identifiability[layout].get("policy_manifest_sha256")
            != recoverable[layout].get("policy_manifest_sha256")
            or identifiability[layout].get("partner_manifest_sha256")
            != recoverable[layout].get("partner_manifest_sha256")
        ):
            raise ValueError(
                "Identifiability and recoverable-value controls use different "
                f"checkpoints or partner panels on {layout}."
            )
    mechanism_passed = bool(
        official_passed
        and final_m1_passed
        and common_passed
        and capacity_passed
        and development_passed
        and calibration_passed
        and identifiability_passed
        and recoverable_passed
    )
    sources = {
        **{name: _source(path) for name, path in paths.items()},
        "common": {layout: _source(path) for layout, path in common_paths.items()},
        "posterior_calibration": {
            layout: _source(path) for layout, path in calibration_paths.items()
        },
        "identifiability": {
            layout: _source(path) for layout, path in identifiability_paths.items()
        },
        "recoverable_value": {
            layout: _source(path) for layout, path in recoverable_paths.items()
        },
    }
    result = {
        "method": METHOD_VERSION,
        "official_protocol_gate_passed": official_passed,
        "m1_final_gate_passed": final_m1_passed,
        "common_partner_gate_passed": common_passed,
        "capacity_control_gate_passed": capacity_passed,
        "development_matrix_gate_passed": development_passed,
        "posterior_calibration_gate_passed": calibration_passed,
        "identifiability_gate_passed": identifiability_passed,
        "recoverable_value_gate_passed": recoverable_passed,
        "mechanism_claims_unlocked": mechanism_passed,
        "b3_status": "not_implemented",
        "worst_mechanism_margins_point_only": {
            layout: payload.get("worst_mechanism_margin_point")
            for layout, payload in common.items()
        },
        "sources": sources,
        "claim_boundary": (
            "Benchmark scores are always reported. Protocol-mechanism attribution "
            "requires paired B0--B2 increments, posterior calibration, task leakage, "
            "history shuffle, final-checkpoint M1, real-continuation context swap, "
            "common-partner, capacity, "
            "recoverable-value, and resource evidence on both layouts."
        ),
    }
    output = Path(args.output).resolve()
    ensure_run_identity(
        output,
        {
            "stage": "formal-claim-report",
            "method": METHOD_VERSION,
            "repository_runtime": runtime_provenance(),
            "sources": sources,
        },
    )
    write_json(output / "formal_claim_report.json", result)
    lines = [
        "# DEPI formal claim gates",
        "",
        f"- Official benchmark: `{official_passed}`",
        f"- Final-checkpoint M1: `{final_m1_passed}`",
        f"- Common partner: `{common_passed}`",
        f"- Capacity control: `{capacity_passed}`",
        f"- B0--B2 paired increments: `{development_passed}`",
        f"- Posterior calibration: `{calibration_passed}`",
        f"- Identifiability controls: `{identifiability_passed}`",
        f"- Recoverable value G1--G4: `{recoverable_passed}`",
        f"- Mechanism claims unlocked: `{mechanism_passed}`",
        "",
        "B3 remains explicitly not implemented and is never presented as a completed layer.",
    ]
    (output / "formal_claim_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


__all__ = ["LAYOUTS", "MECHANISM_ARTIFACTS", "run_formal_claim_report"]
