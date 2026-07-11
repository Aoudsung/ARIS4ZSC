"""Strict Path C decision entry points over hash-bound artifact read-back."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from experiments.overcooked_v2.path_c_evaluation import (
    FrozenPathCPreregistration,
    assemble_path_c_measurements,
    evaluate_path_c_decision,
    load_and_validate_path_c_inputs,
)


def decision_rule(
    measurements: Mapping[str, Any],
    preregistration: FrozenPathCPreregistration,
) -> dict[str, Any]:
    decision = evaluate_path_c_decision(measurements, preregistration)
    return {**decision.__dict__, "secondary": dict(decision.secondary)}


def evaluate_manifest(
    preregistration_path: str | Path,
    artifact_manifest_path: str | Path,
) -> dict[str, Any]:
    inputs = load_and_validate_path_c_inputs(
        preregistration_path,
        artifact_manifest_path,
    )
    measurements = assemble_path_c_measurements(inputs)
    decision = evaluate_path_c_decision(measurements, inputs.preregistration)
    return {
        "decision": {
            **decision.__dict__,
            "secondary": dict(decision.secondary),
        },
        "preregistration_sha256": inputs.preregistration.sha256,
        "artifact_manifest_sha256": inputs.manifest.sha256,
    }
