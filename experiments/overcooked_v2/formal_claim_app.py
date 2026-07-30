"""Fail-closed synthesis of the preregistered performance and claim gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from experiments.overcooked_v2.resource_report_app import RESOURCE_METHODS
from src.path_c.resources import ResourceLedger
from src.path_c.storage import (
    ensure_run_identity,
    runtime_provenance,
    sha256_path,
    validate_formal_repository_state,
    validate_registered_python_runtime,
    write_json,
)


ABLATION_COMPONENTS = (
    "continuous-belief",
    "real-return-anchors",
    "decision-equivalence",
    "decision-regret",
    "conformal-gate",
    "continuous-generator",
)
LAYOUTS = ("test_time_simple", "test_time_wide")


def _json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected a JSON mapping: {path}")
    return payload


def _validate_ablation_gate(path: Path) -> Mapping[str, Any]:
    payload = _json(path)
    expected = {
        "version",
        "frozen_before_formal_runs",
        "comparisons",
    }
    if set(payload) != expected or int(payload["version"]) != 1:
        raise ValueError("Ablation-gate artifact fields/version differ.")
    if payload["frozen_before_formal_runs"] is not True:
        raise ValueError("Ablation definitions were not frozen before formal runs.")
    comparisons = payload["comparisons"]
    if not isinstance(comparisons, list):
        raise ValueError("Ablation comparisons must be a list.")
    index = {}
    for row in comparisons:
        fields = {
            "layout",
            "component",
            "return_difference_lcb",
            "intermediate_direction_passed",
            "full_total_training_simulator_steps",
            "ablation_total_training_simulator_steps",
            "evidence_artifacts",
        }
        if not isinstance(row, Mapping) or set(row) != fields:
            raise ValueError("Ablation comparison fields differ.")
        key = (str(row["layout"]), str(row["component"]))
        if key in index:
            raise ValueError(f"Duplicate ablation comparison: {key}")
        artifacts = row["evidence_artifacts"]
        if not isinstance(artifacts, list) or not artifacts:
            raise ValueError(f"Ablation comparison lacks raw evidence: {key}")
        for artifact in artifacts:
            if not isinstance(artifact, Mapping) or set(artifact) != {"path", "sha256"}:
                raise ValueError("Ablation evidence fields differ.")
            if sha256_path(artifact["path"]) != artifact["sha256"]:
                raise ValueError(f"Ablation evidence hash changed: {artifact['path']}")
        index[key] = dict(row)
    required = {(layout, component) for layout in LAYOUTS for component in ABLATION_COMPONENTS}
    if set(index) != required:
        raise ValueError(
            "Ablation artifact must cover all registered components on both layouts."
        )
    passed = all(
        float(row["return_difference_lcb"]) > 0.0
        and row["intermediate_direction_passed"] is True
        and int(row["ablation_total_training_simulator_steps"])
        <= int(row["full_total_training_simulator_steps"])
        for row in index.values()
    )
    return {
        "path": str(path),
        "sha256": sha256_path(path),
        "all_component_claim_gates_passed": passed,
        "comparisons": comparisons,
    }


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


def run_formal_claim_report(args: argparse.Namespace) -> None:
    validate_formal_repository_state()
    validate_registered_python_runtime()
    official_path = Path(args.official_summary).resolve()
    capacity_path = Path(args.capacity_summary).resolve()
    resource_path = Path(args.resource_report).resolve()
    official = _json(official_path)
    capacity = _json(capacity_path)
    resources = _json(resource_path)
    _validate_resource_report(resources)
    common_paths = {
        "test_time_simple": Path(args.common_simple).resolve(),
        "test_time_wide": Path(args.common_wide).resolve(),
    }
    common = {layout: _json(path) for layout, path in common_paths.items()}

    official_passed = official.get("primary_benchmark_gate_passed") is True
    common_passed = all(
        payload.get("common_partner_gate_passed") is True
        and payload.get("br_prox_complete") is True
        for payload in common.values()
    )
    capacity_passed = capacity.get("capacity_control_gate_passed") is True
    ablation = (
        None
        if args.ablation_gate is None
        else _validate_ablation_gate(Path(args.ablation_gate).resolve())
    )
    mechanism_passed = (
        official_passed
        and common_passed
        and capacity_passed
        and ablation is not None
        and ablation["all_component_claim_gates_passed"] is True
    )
    result = {
        "official_protocol_gate_passed": official_passed,
        "common_partner_gate_passed": common_passed,
        "capacity_control_gate_passed": capacity_passed,
        "mechanism_claims_unlocked": mechanism_passed,
        "ablation_gate": ablation,
        "worst_mechanism_margins_point_only": {
            layout: payload.get("worst_mechanism_margin_point")
            for layout, payload in common.items()
        },
        "worst_mechanism_margin_note": (
            "The preregistration supplied no numeric 'large negative' threshold; "
            "these values are reported without inventing an inferential gate."
        ),
        "sources": {
            "official": {"path": str(official_path), "sha256": sha256_path(official_path)},
            "capacity": {"path": str(capacity_path), "sha256": sha256_path(capacity_path)},
            "resources": {"path": str(resource_path), "sha256": sha256_path(resource_path)},
            "common": {
                layout: {"path": str(path), "sha256": sha256_path(path)}
                for layout, path in common_paths.items()
            },
        },
        "claim_boundary": (
            "Mechanical tests never unlock scientific claims. Benchmark, common-partner, "
            "capacity, and component-ablation gates are independently required."
        ),
    }
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    ensure_run_identity(
        output,
        {
            "stage": "formal-claim-report",
            "repository_runtime": runtime_provenance(),
            "sources": result["sources"],
            "ablation_gate": (
                None
                if ablation is None
                else {"path": ablation["path"], "sha256": ablation["sha256"]}
            ),
        },
    )
    write_json(output / "formal_claim_report.json", result)
    lines = [
        "# Formal claim gates",
        "",
        f"- Official protocol gate: `{official_passed}`",
        f"- Common-partner gate: `{common_passed}`",
        f"- Parameter-matched capacity gate: `{capacity_passed}`",
        f"- All mechanism claims unlocked: `{mechanism_passed}`",
        "",
        "Mechanical validation never constitutes ZSC evidence. Mechanism claims "
        "remain locked unless every registered full-vs-ablation comparison also "
        "passes on both layouts.",
    ]
    (output / "formal_claim_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


__all__ = ["ABLATION_COMPONENTS", "run_formal_claim_report"]
