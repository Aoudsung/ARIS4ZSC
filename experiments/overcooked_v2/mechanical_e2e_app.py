"""Mechanical end-to-end harness for the single DELTA-ZSC V6 algorithm."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from experiments.overcooked_v2.signal_audit_app import run_signal_audit
from experiments.overcooked_v2.training_app import run_training
from src.path_c.experiment import ENGINEERING_SEED_INDEX, METHOD_VERSION
from src.path_c.storage import read_run_identity, write_json


def run_mechanical_e2e(args: argparse.Namespace) -> None:
    """Run the actual V6 loop and verify its mandatory terminal artifacts.

    This harness never substitutes a fixture policy or a fallback deployment.
    CUDA can be made mandatory with ``--require-cuda``; the same fail-closed
    worker checks used by formal training are then applied.
    """

    output = Path(args.output).resolve()
    training_output = output / "training"
    if bool(getattr(args, "require_cuda", False)):
        os.environ["DELTA_REQUIRE_CUDA"] = "1"
    training_args = argparse.Namespace(
        config=args.config,
        partner_manifest=args.partner_manifest,
        ego_run_id=str(args.ego_run_id),
        seed_index=ENGINEERING_SEED_INDEX,
        run_kind="mechanical",
        output=str(training_output),
        resume=bool(getattr(args, "resume", False)),
        skip_manifest_hash_check=bool(args.skip_manifest_hash_check),
    )
    run_training(training_args)
    audit_output = output / "signal_audit"
    run_signal_audit(
        argparse.Namespace(training_run=str(training_output), output=str(audit_output))
    )
    required = (
        training_output / "run_identity.json",
        training_output / "run_metadata.json",
        training_output / "resource_ledger.json",
        training_output / "final_deployment" / "deployment_bundle.json",
        training_output / "records" / "training_support_latents.json",
        audit_output / "signal_audit.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Mechanical V6 flow omitted artifacts: {missing}")
    identity = read_run_identity(training_output)
    deployment = json.loads(
        (training_output / "final_deployment" / "deployment_bundle.json").read_text(
            encoding="utf-8"
        )
    )
    if identity.get("method") != METHOD_VERSION:
        raise RuntimeError("Mechanical training identity is not V6.")
    if deployment.get("artifact_name") != "DELTA-ZSC-E2E":
        raise RuntimeError("Mechanical flow did not export the single E2E actor.")
    report = {
        "method": METHOD_VERSION,
        "status": "complete",
        "training_run": str(training_output),
        "signal_audit": str(audit_output),
        "cuda_required": bool(getattr(args, "require_cuda", False)),
        "scientific_readout": False,
        "note": "Mechanical completion proves execution, not ZSC effectiveness.",
    }
    write_json(output / "mechanical_e2e_report.json", report)
    print(f"Complete V6 mechanical E2E: {output}")


__all__ = ["run_mechanical_e2e"]
