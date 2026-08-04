"""Fail-closed public entries for unified evaluation and diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from experiments.overcooked_v2.unified_calibration_app import (
    run_calibration as _run_calibration_core,
)
from experiments.overcooked_v2.unified_causal_app import (
    run_causal_evaluation as _run_causal_core,
)
from experiments.overcooked_v2.unified_evaluation_app import (
    run_evaluation as _run_evaluation_core,
)
from src.delta_zsc.deployment import load_deployment
from src.delta_zsc.manifest import validate_unified_manifest
from src.delta_zsc.model import FULL_VARIANT, JOINT_VARIANT
from src.path_c.experiment import load_partner_manifest


def _deployment_context(path: str | Path) -> tuple[Any, Mapping[str, Any], bool]:
    root = Path(path).resolve()
    deployment = load_deployment(root)
    bundle = json.loads(
        (root / "deployment_bundle.json").read_text(encoding="utf-8")
    )
    training_root = Path(str(bundle["source_training_run"])).resolve()
    identity = json.loads(
        (training_root / "run_identity.json").read_text(encoding="utf-8")
    )
    if identity.get("stage") != "train-unified-delta-zsc":
        raise ValueError("Deployment source is not a unified DELTA training run.")
    return deployment, identity, str(identity.get("run_kind")) == "formal"


def _manifest(args: Any, *, formal: bool, role: str) -> Any:
    if formal and bool(args.skip_manifest_hash_check):
        raise ValueError("Formal unified evaluation cannot skip partner hashes.")
    deployment, _, _ = _deployment_context(args.deployment)
    manifest = load_partner_manifest(
        Path(args.partner_manifest).resolve(),
        expected_layout=deployment.config.environment.layout,
        verify_files=not bool(args.skip_manifest_hash_check),
    )
    validate_unified_manifest(
        manifest,
        formal=formal,
        require_calibration=(role == "calibration"),
        require_confirmatory=(role == "confirmatory"),
    )
    return manifest


def run_evaluation(args: Any) -> None:
    deployment, _, formal = _deployment_context(args.deployment)
    role = str(getattr(args, "partner_role", "confirmatory"))
    if formal:
        if role != "confirmatory":
            raise ValueError("Formal XP evaluation uses the confirmatory panel.")
        episodes = getattr(args, "episodes", None)
        if episodes is not None and int(episodes) != 500:
            raise ValueError("Formal evaluation uses 500 episodes per pairing.")
        if int(getattr(args, "evaluation_seed", 0)) != 0:
            raise ValueError("Formal evaluation root seed is zero.")
    override = getattr(args, "variant_override", None)
    if override == FULL_VARIANT and deployment.variant != JOINT_VARIANT:
        raise ValueError("Full evaluation must reuse a joint deployment.")
    _manifest(args, formal=formal, role=role)
    _run_evaluation_core(args)


def run_calibration(args: Any) -> None:
    _, _, formal = _deployment_context(args.deployment)
    _manifest(args, formal=formal, role="calibration")
    _run_calibration_core(args)


def run_causal_evaluation(args: Any) -> None:
    _, _, formal = _deployment_context(args.deployment)
    role = str(getattr(args, "partner_role", "confirmatory"))
    if formal and role != "confirmatory":
        raise ValueError("Formal causal belief evaluation uses confirmatory partners.")
    _manifest(args, formal=formal, role=role)
    _run_causal_core(args)


__all__ = [
    "run_calibration",
    "run_causal_evaluation",
    "run_evaluation",
]
