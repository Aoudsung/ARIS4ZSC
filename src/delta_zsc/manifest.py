"""Lineage-bound partner manifests and their validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from .config import MANIFEST_VERSION


@dataclass(frozen=True, slots=True)
class PartnerRun:
    run_id: str
    role: str
    checkpoint: Path
    parent_training_run_id: str
    generation_mechanism: str
    checkpoint_stage: float
    hyperparameter_family: str
    seed: int
    seed_index: int | None = None
    jax_prng_key: tuple[int, int] | None = None
    owner_seed_index: int | None = None
    co_training_group_id: str | None = None
    partner_type_id: str | None = None


@dataclass(frozen=True, slots=True)
class PartnerManifest:
    layout: str
    runs: tuple[PartnerRun, ...]

    def by_role(self, role: str) -> tuple[PartnerRun, ...]:
        return tuple(row for row in self.runs if row.role == str(role))

    def to_mapping(self) -> Mapping[str, Any]:
        return {
            "version": MANIFEST_VERSION,
            "layout": self.layout,
            "runs": [
                {**asdict(row), "checkpoint": str(row.checkpoint)} for row in self.runs
            ],
        }


def normalized_mechanism(value: str) -> str:
    aliases = {
        "rnn-sp": "sp",
        "sp": "sp",
        "rnn-op": "op",
        "op": "op",
        "state-augmented": "sa",
        "rnn-sa": "sa",
        "sa": "sa",
        "rnn-fcp": "fcp",
        "fcp": "fcp",
    }
    lowered = str(value).strip().lower()
    return aliases.get(lowered, lowered)


def validate_partner_manifest(manifest: PartnerManifest) -> None:
    if not manifest.layout:
        raise ValueError("Partner manifest layout is empty.")
    if not manifest.runs:
        raise ValueError("Partner manifest is empty.")
    run_ids = [row.run_id for row in manifest.runs]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("Partner run IDs must be unique.")
    for row in manifest.runs:
        if not row.run_id or not row.role or not row.parent_training_run_id:
            raise ValueError("Partner manifest contains an empty identity field.")
        if not 0.0 <= float(row.checkpoint_stage) <= 1.0:
            raise ValueError(f"Partner {row.run_id} checkpoint stage is invalid.")
        if row.jax_prng_key is not None and len(row.jax_prng_key) != 2:
            raise ValueError(f"Partner {row.run_id} JAX key must have two words.")
    # Scientific panels must not share parent or co-training lineage.
    active = ("development_support", "calibration", "confirmatory")
    by_role = {role: manifest.by_role(role) for role in active}
    for left_index, left_role in enumerate(active):
        for right_role in active[left_index + 1 :]:
            left = by_role[left_role]
            right = by_role[right_role]
            left_parents = {row.parent_training_run_id for row in left}
            right_parents = {row.parent_training_run_id for row in right}
            if left_parents & right_parents:
                raise ValueError(
                    f"Partner parent lineage overlaps across {left_role}/{right_role}."
                )
            left_groups = {
                row.co_training_group_id
                for row in left
                if row.co_training_group_id is not None
            }
            right_groups = {
                row.co_training_group_id
                for row in right
                if row.co_training_group_id is not None
            }
            if left_groups & right_groups:
                raise ValueError(
                    f"Partner co-training lineage overlaps across {left_role}/{right_role}."
                )


def load_partner_manifest(
    path: str | Path,
    *,
    expected_layout: str | None = None,
    verify_files: bool = True,
) -> PartnerManifest:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or set(payload) != {"version", "layout", "runs"}:
        raise ValueError("Partner manifest top-level schema differs.")
    if int(payload["version"]) != MANIFEST_VERSION:
        raise ValueError(f"Partner manifest version must be {MANIFEST_VERSION}.")
    if not isinstance(payload["runs"], list):
        raise ValueError("Partner manifest runs must be a list.")
    expected_fields = set(PartnerRun.__dataclass_fields__)
    runs = []
    for index, raw in enumerate(payload["runs"]):
        if not isinstance(raw, Mapping) or set(raw) != expected_fields:
            raise ValueError(f"Partner manifest row {index} fields differ.")
        checkpoint = Path(str(raw["checkpoint"]))
        if not checkpoint.is_absolute():
            checkpoint = (source.parent / checkpoint).resolve()
        else:
            checkpoint = checkpoint.resolve()
        row = PartnerRun(
            run_id=str(raw["run_id"]),
            role=str(raw["role"]),
            checkpoint=checkpoint,
            parent_training_run_id=str(raw["parent_training_run_id"]),
            generation_mechanism=str(raw["generation_mechanism"]),
            checkpoint_stage=float(raw["checkpoint_stage"]),
            hyperparameter_family=str(raw["hyperparameter_family"]),
            seed=int(raw["seed"]),
            seed_index=(None if raw["seed_index"] is None else int(raw["seed_index"])),
            jax_prng_key=(
                None
                if raw["jax_prng_key"] is None
                else tuple(int(value) for value in raw["jax_prng_key"])
            ),
            owner_seed_index=(
                None
                if raw["owner_seed_index"] is None
                else int(raw["owner_seed_index"])
            ),
            co_training_group_id=(
                None
                if raw["co_training_group_id"] is None
                else str(raw["co_training_group_id"])
            ),
            partner_type_id=(
                None if raw["partner_type_id"] is None else str(raw["partner_type_id"])
            ),
        )
        if verify_files and not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        runs.append(row)
    manifest = PartnerManifest(layout=str(payload["layout"]), runs=tuple(runs))
    if expected_layout is not None and manifest.layout != str(expected_layout):
        raise ValueError("Partner manifest layout differs from the run config.")
    validate_partner_manifest(manifest)
    return manifest


__all__ = [
    "PartnerManifest",
    "PartnerRun",
    "load_partner_manifest",
    "normalized_mechanism",
    "validate_partner_manifest",
]
