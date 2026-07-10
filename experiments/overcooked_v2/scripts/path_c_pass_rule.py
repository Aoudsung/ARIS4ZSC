"""Strict Path C decision entry points over hash-bound artifact read-back."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from experiments.overcooked_v2.path_c_evaluation import (
    FrozenPathCPreregistration,
    assemble_path_c_measurements,
    load_and_validate_path_c_inputs,
    pass_af_claim_rule,
    phase_b_go_no_go_rule,
)


def phase_b_rule(
    measurements: Mapping[str, Any],
    preregistration: FrozenPathCPreregistration,
) -> dict[str, Any]:
    return phase_b_go_no_go_rule(measurements, preregistration)


def pass_af_rule(
    measurements: Mapping[str, Any],
    preregistration: FrozenPathCPreregistration,
) -> dict[str, Any]:
    return pass_af_claim_rule(measurements, preregistration)


def evaluate_manifest(
    preregistration_path: str | Path,
    artifact_manifest_path: str | Path,
) -> dict[str, Any]:
    inputs = load_and_validate_path_c_inputs(
        preregistration_path,
        artifact_manifest_path,
    )
    measurements = assemble_path_c_measurements(inputs)
    return {
        "phase_b": phase_b_go_no_go_rule(measurements, inputs.preregistration),
        "pass_af": pass_af_claim_rule(measurements, inputs.preregistration),
        "preregistration_sha256": inputs.preregistration.sha256,
        "artifact_manifest_sha256": inputs.manifest.sha256,
    }
