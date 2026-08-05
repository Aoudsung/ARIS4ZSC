"""Build and validate lineage-bound partner manifests for unified DELTA-ZSC."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from src.delta_zsc.config import MANIFEST_VERSION
from src.delta_zsc.manifest import load_partner_manifest
from src.delta_zsc.storage import read_json, write_json


_PLAN_FIELDS = {
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


def _resolve_checkpoint(raw: Any, *, plan: Path) -> Path:
    value = Path(str(raw))
    return (plan.parent / value).resolve() if not value.is_absolute() else value.resolve()


def build_partner_manifest(args: Any) -> None:
    """Resolve checkpoint paths from a human-editable manifest plan."""

    plan = Path(args.plan).resolve()
    target = Path(args.output).resolve()
    payload = read_json(plan)
    if not isinstance(payload, Mapping) or set(payload) != {"version", "layout", "runs"}:
        raise ValueError("Partner manifest plan top-level schema differs.")
    if int(payload["version"]) != MANIFEST_VERSION:
        raise ValueError(f"Partner manifest plan version must be {MANIFEST_VERSION}.")
    if not isinstance(payload["runs"], list) or not payload["runs"]:
        raise ValueError("Partner manifest plan requires a non-empty run list.")

    rows = []
    for index, raw in enumerate(payload["runs"]):
        if not isinstance(raw, Mapping) or set(raw) != _PLAN_FIELDS:
            raise ValueError(f"Partner manifest plan row {index} fields differ.")
        checkpoint = _resolve_checkpoint(raw["checkpoint"], plan=plan)
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        # Store paths relative to the manifest.  A complete experiment tree can
        # then be renamed or moved without leaving every checkpoint reference
        # pointing at its former absolute location.
        stored_checkpoint = os.path.relpath(checkpoint, start=target.parent)
        rows.append({**dict(raw), "checkpoint": stored_checkpoint})

    write_json(
        target,
        {
            "version": MANIFEST_VERSION,
            "layout": str(payload["layout"]),
            "runs": rows,
        },
    )
    manifest = load_partner_manifest(
        target,
        expected_layout=(None if getattr(args, "expected_layout", None) is None else str(args.expected_layout)),
        verify_files=True,
    )
    print(
        f"Complete unified DELTA partner manifest: {target} "
        f"({len(manifest.runs)} runs, layout={manifest.layout})"
    )


def validate_partner_manifest_command(args: Any) -> None:
    manifest = load_partner_manifest(
        args.partner_manifest,
        expected_layout=getattr(args, "expected_layout", None),
        verify_files=not bool(getattr(args, "skip_manifest_file_check", False)),
    )
    print(
        f"Valid unified DELTA partner manifest: {len(manifest.runs)} runs, "
        f"layout={manifest.layout}"
    )


__all__ = ["build_partner_manifest", "validate_partner_manifest_command"]
