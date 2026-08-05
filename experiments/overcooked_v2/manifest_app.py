"""Build immutable, hash-bound DEPI partner manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.path_c.experiment import (
    MANIFEST_VERSION,
    PartnerManifest,
    PartnerRun,
    validate_partner_manifest,
)
from src.path_c.storage import sha256_path, write_json

PLAN_VERSION = 3


def _exact_mapping(value: Any, expected: set[str], location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be a mapping.")
    fields = set(value)
    if fields != expected:
        raise ValueError(
            f"{location} fields differ: missing={sorted(expected-fields)}, "
            f"unknown={sorted(fields-expected)}."
        )
    return value


def build_partner_manifest(
    *, plan_path: str | Path, output_path: str | Path
) -> Path:
    """Resolve checkpoint paths and bind their exact SHA-256 identities.

    The plan intentionally omits ``checkpoint_sha256`` so users cannot copy an
    unrelated digest.  The generated manifest is immediately passed through the
    same lineage validator used by training, calibration, and evaluation.
    """

    plan_file = Path(plan_path).resolve()
    payload = json.loads(plan_file.read_text(encoding="utf-8"))
    payload = _exact_mapping(payload, {"version", "layout", "runs"}, "plan")
    if int(payload["version"]) != PLAN_VERSION:
        raise ValueError(f"Manifest plan version must be {PLAN_VERSION}.")
    raw_runs = payload["runs"]
    if not isinstance(raw_runs, Sequence) or isinstance(raw_runs, (str, bytes)):
        raise ValueError("plan.runs must be a sequence.")

    expected_run_fields = {
        "run_id",
        "role",
        "checkpoint",
        "parent_training_run_id",
        "generation_mechanism",
        "checkpoint_stage",
        "hyperparameter_family",
        "seed",
        "seed_index",
        "jax_prng_key",
        "owner_seed_index",
        "co_training_group_id",
        "partner_type_id",
    }
    runs: list[PartnerRun] = []
    for index, raw in enumerate(raw_runs):
        item = _exact_mapping(raw, expected_run_fields, f"plan.runs[{index}]")
        checkpoint = Path(str(item["checkpoint"]))
        if not checkpoint.is_absolute():
            checkpoint = (plan_file.parent / checkpoint).resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        runs.append(
            PartnerRun(
                run_id=str(item["run_id"]),
                role=str(item["role"]),
                checkpoint=checkpoint,
                checkpoint_sha256=sha256_path(checkpoint),
                parent_training_run_id=str(item["parent_training_run_id"]),
                generation_mechanism=str(item["generation_mechanism"]),
                checkpoint_stage=float(item["checkpoint_stage"]),
                hyperparameter_family=str(item["hyperparameter_family"]),
                seed=int(item["seed"]),
                seed_index=(
                    None if item["seed_index"] is None else int(item["seed_index"])
                ),
                jax_prng_key=(
                    None
                    if item["jax_prng_key"] is None
                    else tuple(int(value) for value in item["jax_prng_key"])
                ),
                owner_seed_index=(
                    None
                    if item["owner_seed_index"] is None
                    else int(item["owner_seed_index"])
                ),
                co_training_group_id=(
                    None
                    if item["co_training_group_id"] is None
                    else str(item["co_training_group_id"])
                ),
                partner_type_id=(
                    None
                    if item["partner_type_id"] is None
                    else str(item["partner_type_id"])
                ),
            )
        )
    manifest = PartnerManifest(layout=str(payload["layout"]), runs=tuple(runs))
    validate_partner_manifest(manifest)
    target = write_json(output_path, manifest.to_mapping())
    return target


def add_manifest_command(commands: argparse._SubParsersAction) -> None:
    parser = commands.add_parser("build-partner-manifest")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    parser.set_defaults(
        function=lambda args: print(
            build_partner_manifest(plan_path=args.plan, output_path=args.output)
        ),
        manages_output=False,
    )


__all__ = ["PLAN_VERSION", "add_manifest_command", "build_partner_manifest"]
