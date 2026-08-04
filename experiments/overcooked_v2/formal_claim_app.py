"""Fail-closed synthesis of DEPI performance and mechanism-attribution gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from experiments.overcooked_v2.resource_report_app import RESOURCE_METHODS
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
        payload.get("version") != 2
        or payload.get("artifact_type") != "depi_development_matrix_summary"
        or payload.get("method") != METHOD_VERSION
        or payload.get("b3_status") != "not_implemented"
    ):
        raise ValueError("Development-matrix summary identity differs.")
    comparisons = payload.get("primary_k4_paired_increments")
    if not isinstance(comparisons, Mapping):
        raise ValueError("Development matrix has no paired increments.")
    required = {
        "b0_minus_r0",
        "b1_minus_b0",
        "b2_minus_b1",
        "b2_minus_b0",
        "b2_minus_r0_total_budget",
        "b2_minus_b0_total_budget",
        "b2_minus_b1_total_budget",
        "b2_minus_deterministic_context",
        "b2_minus_decision_only",
        "b2_minus_q_only",
        "b2_minus_actor_only",
        "b2_minus_no_separation",
        "b2_minus_no_capability",
    }
    if set(comparisons) != required:
        raise ValueError("Development matrix does not cover nested and total-budget contrasts.")
    sources = payload.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != {
        "matrix",
        "evaluations",
        "component_diagnostics",
    }:
        raise ValueError("Development summary sources are malformed.")
    matrix_path = _validate_source_ref(
        sources["matrix"], label="development.matrix"
    )
    matrix = _json(matrix_path)
    if (
        matrix.get("artifact_type") != "depi_development_matrix"
        or matrix.get("method") != METHOD_VERSION
    ):
        raise ValueError("Development summary sources have different identities.")
    if matrix.get("budget_capacity_and_key_matching_passed") is not True:
        raise ValueError("Development matrix did not establish budget/capacity/key matching.")

    from experiments.overcooked_v2.development_matrix_app import (
        DEVELOPMENT_VARIANTS,
        _paired_bootstrap,
        _validate_development_raw_rows,
        validate_development_entry_alignment,
    )

    matrix_entries = matrix.get("entries")
    if not isinstance(matrix_entries, list):
        raise ValueError("Development matrix entries are missing.")
    validate_development_entry_alignment(matrix_entries)
    evaluation_refs = sources["evaluations"]
    if not isinstance(evaluation_refs, list) or not evaluation_refs:
        raise ValueError("Development summary has no evaluator artifacts.")
    score_index: dict[tuple[int, str, int], float] = {}
    observed_blocks: set[tuple[int, str]] = set()
    for index, ref in enumerate(evaluation_refs):
        artifact_path = _validate_source_ref(
            ref, label=f"development.evaluations.{index}"
        )
        evaluation = _json(artifact_path)
        if (
            evaluation.get("version") != 2
            or evaluation.get("artifact_type") != "depi_development_evaluation"
            or evaluation.get("method") != METHOD_VERSION
            or evaluation.get("matrix", {}).get("sha256") != sha256_path(matrix_path)
        ):
            raise ValueError("Development evaluator artifact identity differs.")
        evaluation_identity = _json(artifact_path.parent / "run_identity.json")
        if (
            evaluation_identity.get("stage") != "evaluate-development-matrix"
            or evaluation_identity.get("method") != METHOD_VERSION
            or evaluation_identity.get("matrix", {}).get("sha256")
            != sha256_path(matrix_path)
        ):
            raise ValueError("Development evaluator run identity differs.")
        evaluation_ledger = ResourceLedger.from_mapping(
            _json(artifact_path.parent / "resource_ledger.json")
        )
        variant = str(evaluation.get("variant"))
        component_count = int(evaluation.get("protocol_components", -1))
        if variant not in DEVELOPMENT_VARIANTS or component_count not in {2, 4, 8}:
            raise ValueError("Development evaluator block is outside the matrix.")
        block = (component_count, variant)
        if block in observed_blocks:
            raise ValueError("Duplicate development evaluator block.")
        observed_blocks.add(block)
        raw_ref = evaluation.get("raw_episodes")
        raw_path = _validate_source_ref(
            raw_ref, label=f"development.raw.{component_count}.{variant}"
        )
        raw_rows = read_parquet(raw_path)
        deployments = evaluation.get("deployments")
        if not isinstance(deployments, list) or len(deployments) != 10:
            raise ValueError("Development evaluation must contain ten ego seeds.")
        matrix_index = {
            (
                int(row["protocol_components"]),
                str(row["variant"]),
                int(row["seed_index"]),
            ): row
            for row in matrix_entries
        }
        for deployment in deployments:
            deployment_fields = {
                "seed_index",
                "path",
                "sha256",
                "config_fingerprint",
                "training_run",
                "run_identity",
                "resource_ledger",
                "partner_sampler",
            }
            if not isinstance(deployment, Mapping) or set(deployment) != deployment_fields:
                raise ValueError("Development deployment source is malformed.")
            if sha256_path(deployment["path"]) != deployment["sha256"]:
                raise ValueError("Development deployment source hash changed.")
            seed_index = int(deployment["seed_index"])
            entry = matrix_index.get((component_count, variant, seed_index))
            if entry is None:
                raise ValueError("Development deployment is not a matrix entry.")
            training_run = Path(str(deployment["training_run"])).resolve()
            if (
                training_run != Path(str(entry["run"])).resolve()
                or Path(str(deployment["path"])).resolve()
                != training_run / "final_deployment"
                or deployment["config_fingerprint"] != entry["config_fingerprint"]
                or sha256_path(Path(str(entry["config"]["path"])).resolve())
                != entry["config"]["sha256"]
            ):
                raise ValueError("Development deployment does not match its matrix run.")
            identity_path = _validate_source_ref(
                deployment["run_identity"],
                label=f"development.run-identity.{component_count}.{variant}.{seed_index}",
            )
            resource_path = _validate_source_ref(
                deployment["resource_ledger"],
                label=f"development.resource.{component_count}.{variant}.{seed_index}",
            )
            sampler_path = _validate_source_ref(
                deployment["partner_sampler"],
                label=f"development.sampler.{component_count}.{variant}.{seed_index}",
            )
            identity = _json(identity_path)
            training_ledger = ResourceLedger.from_mapping(_json(resource_path))
            if (
                identity_path != training_run / "run_identity.json"
                or resource_path != training_run / "resource_ledger.json"
                or sampler_path != training_run / "partner_pool.json"
                or sha256_path(identity_path) != entry["run_identity_sha256"]
                or sha256_path(sampler_path) != entry["partner_sampler_sha256"]
                or identity.get("stage") != "train"
                or identity.get("method") != METHOD_VERSION
                or identity.get("method_variant") != variant.removesuffix("_extra")
                or int(identity.get("seed_index", -1)) != seed_index
                or identity.get("config_fingerprint")
                != deployment["config_fingerprint"]
                or training_ledger.ego_policy_steps
                != int(entry["ppo_training_steps"])
                or training_ledger.counterfactual_continuation_steps
                + training_ledger.matched_pair_probe_steps
                != int(entry["auxiliary_training_steps"])
                or training_ledger.total_training_simulator_steps
                != int(entry["total_training_simulator_steps"])
            ):
                raise ValueError("Development training lineage or ledger differs.")
        policy_count = len(deployments)
        expected_rows = policy_count * policy_count * 500
        if (
            evaluation_ledger.evaluation_steps != expected_rows * 400
            or ResourceLedger.from_mapping(evaluation.get("resource_ledger", {}))
            != evaluation_ledger
        ):
            raise ValueError("Development evaluator resource ledger differs.")
        schedule_sha = evaluation.get("episode_key_schedule_sha256")
        if (
            len(raw_rows) != expected_rows
            or int(evaluation.get("episode_count", -1)) != expected_rows
        ):
            raise ValueError("Development raw evaluator rows are incomplete.")
        scores = _validate_development_raw_rows(
            raw_rows,
            variant=variant,
            protocol_components=component_count,
            seed_indexes=[int(row["seed_index"]) for row in deployments],
            schedule_sha256=str(schedule_sha),
        )
        for deployment in deployments:
            seed = int(deployment["seed_index"])
            score_index[(component_count, variant, seed)] = scores[seed]
    expected_blocks = {
        (component_count, variant)
        for component_count in (2, 4, 8)
        for variant in DEVELOPMENT_VARIANTS
    }
    if observed_blocks != expected_blocks:
        raise ValueError("Development evaluator artifacts do not cover all K/variants.")
    seeds = sorted({int(row["seed_index"]) for row in matrix_entries})
    if seeds != list(range(10)):
        raise ValueError("Development inference requires registered seeds 0--9.")
    from src.path_c.component_diagnostics import (
        permutation_aligned_component_stability,
        validate_component_diagnostic_values,
    )

    diagnostic_refs = sources["component_diagnostics"]
    if not isinstance(diagnostic_refs, list) or len(diagnostic_refs) != 30:
        raise ValueError("Development component diagnostics must cover 3 K values x 10 seeds.")
    signatures_by_k: dict[int, dict[int, Any]] = {2: {}, 4: {}, 8: {}}
    for index, ref in enumerate(diagnostic_refs):
        diagnostic_path = _validate_source_ref(
            ref, label=f"development.component-diagnostic.{index}"
        )
        diagnostic = _json(diagnostic_path)
        component_count = int(diagnostic.get("protocol_components", -1))
        seed = int(diagnostic.get("seed_index", -1))
        expected_run = Path(
            str(matrix_index.get((component_count, "b2", seed), {}).get("run", ""))
        ).resolve()
        if (
            diagnostic_path
            != expected_run / "records" / "final_component_diagnostics.json"
            or diagnostic.get("version") != 1
            or diagnostic.get("artifact_type")
            != "depi_exchangeable_response_regime_diagnostics"
            or diagnostic.get("method") != METHOD_VERSION
            or diagnostic.get("method_variant") != "b2"
            or component_count not in signatures_by_k
            or seed not in seeds
            or diagnostic.get("fresh_final_policy_anchors") is not True
            or diagnostic.get("one_hot_component_intervention") is not True
            or seed in signatures_by_k[component_count]
        ):
            raise ValueError("Development component diagnostic identity differs.")
        validate_component_diagnostic_values(
            diagnostic, component_count=component_count
        )
        signatures_by_k[component_count][seed] = diagnostic[
            "component_action_signatures"
        ]
    reported_stability = payload.get("component_permutation_aligned_stability")
    if not isinstance(reported_stability, Mapping):
        raise ValueError("Permutation-aligned component stability is missing.")
    for component_count, signatures in signatures_by_k.items():
        recomputed = permutation_aligned_component_stability(signatures)
        if reported_stability.get(str(component_count)) != recomputed:
            raise ValueError("Permutation-aligned component stability was not recomputed.")
    expected_scores = {
        (
            int(row["protocol_components"]),
            str(row["variant"]),
            int(row["seed_index"]),
        )
        for row in matrix_entries
    }
    if set(score_index) != expected_scores:
        raise ValueError("Development raw scores do not cover the matrix entries.")
    reported_xp = payload.get("xp_by_protocol_components")
    if not isinstance(reported_xp, Mapping):
        raise ValueError("Development per-seed XP is missing.")
    for component_count in (2, 4, 8):
        for variant in DEVELOPMENT_VARIANTS:
            values = np.asarray(
                [score_index[(component_count, variant, seed)] for seed in seeds],
                dtype=np.float64,
            )
            row = reported_xp[str(component_count)][variant]
            if not np.allclose(np.asarray(row["per_seed"]), values) or not np.isclose(
                float(row["mean"]), float(np.mean(values))
            ):
                raise ValueError("Development XP was not recomputed from raw rows.")
    comparison_spec = (
        ("b0_minus_r0", "b0", "r0", 9),
        ("b1_minus_b0", "b1", "b0", 10),
        ("b2_minus_b1", "b2", "b1", 11),
        ("b2_minus_b0", "b2", "b0", 12),
        ("b2_minus_r0_total_budget", "b2", "r0_extra", 13),
        ("b2_minus_b0_total_budget", "b2", "b0_extra", 14),
        ("b2_minus_b1_total_budget", "b2", "b1_extra", 15),
        ("b2_minus_deterministic_context", "b2", "deterministic_context", 16),
        ("b2_minus_decision_only", "b2", "decision_only", 17),
        ("b2_minus_q_only", "b2", "q_only", 18),
        ("b2_minus_actor_only", "b2", "actor_only", 19),
        ("b2_minus_no_separation", "b2", "no_separation", 20),
        ("b2_minus_no_capability", "b2", "no_capability", 21),
    )
    all_comparisons = payload.get("paired_increments_by_protocol_components")
    if not isinstance(all_comparisons, Mapping):
        raise ValueError("Development paired comparisons are missing.")
    for component_count in (2, 4, 8):
        for name, left, right, bootstrap_seed in comparison_spec:
            differences = np.asarray(
                [
                    score_index[(component_count, left, seed)]
                    - score_index[(component_count, right, seed)]
                    for seed in seeds
                ]
            )
            point, interval = _paired_bootstrap(
                differences, seed=bootstrap_seed + 100 * component_count
            )
            reported = all_comparisons[str(component_count)][name]
            if not np.isclose(float(reported["paired_mean_increment"]), point) or not np.allclose(
                np.asarray(reported["bootstrap_99_percent_ci"]), interval
            ):
                raise ValueError("Development comparison was not derived from raw rows.")
    if payload.get("primary_k4_paired_increments") != all_comparisons["4"]:
        raise ValueError("Development K=4 primary comparisons differ from the matrix.")
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
    intervals = payload.get("run_bootstrap_95_ci")
    contrasts = payload.get("run_bootstrap_contrast_95_ci")
    if not isinstance(intervals, Mapping) or not isinstance(contrasts, Mapping):
        raise ValueError(f"Posterior-calibration interval evidence is missing on {layout}.")
    expected = {
        "log_score_vs_uniform": float(contrasts["log_score_minus_uniform"][1])
        <= -0.02,
        "log_score_vs_no_history": float(
            contrasts["log_score_minus_no_history"][1]
        )
        <= -0.02,
        "position_coverage_in_band": float(intervals["coverage_position"][0])
        <= 0.95
        and float(intervals["coverage_position"][1]) >= 0.85,
        "direction_coverage_in_band": float(intervals["coverage_direction"][0])
        <= 0.95
        and float(intervals["coverage_direction"][1]) >= 0.85,
        "event_brier": float(
            contrasts["event_brier_minus_registered_baseline"][1]
        )
        <= 0.0,
    }
    expected_overall = bool(all(expected.values()))
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
    if payload.get("version") != 3 or payload.get("method_variant") != "b2":
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
        transplant = payload.get("protocol_state_transplant", {})
        swap = payload.get("context_swap", {})
        expected_flags = {
            "task_excess_leakage": float(
                task.get("excess_balanced_accuracy", float("inf"))
            )
            <= float(task.get("maximum_excess_over_task_state", float("-inf"))),
            "protocol_state_transplant": float(
                transplant.get("mean_return_drop", float("-inf"))
            )
            > 0.0
            and float(
                transplant.get("bootstrap_99_percent_ci", [float("-inf")])[0]
            )
            > 0.0,
            "source_world_context_value": float(
                swap.get(
                    "source_world_value_alignment_bootstrap_99_percent_ci",
                    [float("-inf")],
                )[0]
            ) > 0.0,
        }
        expected_flags["overall"] = bool(all(expected_flags.values()))
        if payload.get("pass") != expected_flags:
            raise ValueError("Identifiability pass fields are not metric-derived.")
    if artifact_type == "depi_recoverable_value_evaluation":
        fraction = payload.get("recoverable_fraction")
        if not isinstance(fraction, Mapping) or float(
            fraction.get("signal_threshold", -1.0)
        ) != 20.0:
            raise ValueError("Recoverable-value rho registration differs.")
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
            "recoverable_fraction_estimable": bool(fraction["estimable"]),
        }
        proxy = payload.get("cross_fitted_proxy")
        if (
            not isinstance(proxy, Mapping)
            or proxy.get("not_a_mathematical_upper_bound") is not True
            or not 0.0
            <= float(proxy.get("mean_top_action_selection_stability", -1.0))
            <= 1.0
        ):
            raise ValueError("Cross-fitted proxy diagnostics are malformed.")
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
    primary_development = development["primary_k4_paired_increments"]
    filter_architecture_passed = bool(
        primary_development["b1_minus_b0"]["bootstrap_99_percent_ci"][0]
        > 0.0
    )
    decision_increment_passed = bool(
        primary_development["b2_minus_b1"]["bootstrap_99_percent_ci"][0]
        > 0.0
    )
    decision_cost_efficiency_passed = bool(
        primary_development["b2_minus_b1_total_budget"][
            "bootstrap_99_percent_ci"
        ][0]
        > 0.0
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
        # Retained only as a backwards-compatible aggregate diagnostic. It
        # never suppresses or rewrites an independently supported claim below.
        "mechanism_claims_unlocked": mechanism_passed,
        "aggregate_diagnostic_only": True,
        "claims": {
            "performance_claim": official_passed,
            "decision_supervision_claim": bool(
                decision_increment_passed and final_m1_passed
            ),
            "decision_supervision_cost_efficiency_claim": bool(
                decision_cost_efficiency_passed and final_m1_passed
            ),
            "filter_architecture_claim": filter_architecture_passed,
            "predictive_calibration_claim": calibration_passed,
            "history_dependence_claim": bool(
                all(
                    payload.get("pass", {}).get("protocol_state_transplant") is True
                    for payload in identifiability.values()
                )
            ),
            "context_causal_value_claim": bool(
                all(
                    payload.get("pass", {}).get("source_world_context_value") is True
                    for payload in identifiability.values()
                )
            ),
            "recoverable_value_claim": recoverable_passed,
            "capacity_explanation_rejected": capacity_passed,
            "common_partner_generalization_claim": common_passed,
        },
        "b3_status": "not_implemented",
        "worst_mechanism_margins_point_only": {
            layout: payload.get("worst_mechanism_margin_point")
            for layout, payload in common.items()
        },
        "sources": sources,
        "claim_boundary": (
            "Each claim is governed only by its named evidence. The legacy aggregate "
            "conjunction is diagnostic-only and never revokes an independently "
            "supported performance or mechanism statement."
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
        f"- Filter architecture (B1-B0): `{filter_architecture_passed}`",
        f"- Decision supervision increment (B2-B1): `{decision_increment_passed}`",
        f"- Decision supervision cost efficiency (B2-B1-extra): "
        f"`{decision_cost_efficiency_passed}`",
        f"- Posterior calibration: `{calibration_passed}`",
        f"- Identifiability controls: `{identifiability_passed}`",
        f"- Recoverable value G1--G4: `{recoverable_passed}`",
        f"- Aggregate diagnostic only: `{mechanism_passed}`",
        "",
        "B3 remains explicitly not implemented and is never presented as a completed layer.",
    ]
    (output / "formal_claim_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


__all__ = ["LAYOUTS", "MECHANISM_ARTIFACTS", "run_formal_claim_report"]
