"""Four-mode standard self-play and directed cross-play evaluation contracts."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping, Sequence

from .checkpoint import (
    assert_reference_ownership,
)
from .config import VQBCConfig
from .config import VQBC_DEPLOYMENT_MODES
from .integrity import file_sha256
from .types import CheckpointMetadataV4


VQBC_POPULATION_SCHEMA_VERSION = "path_c_vqbc_population_v2"
VQBC_EVALUATION_ROWS_SCHEMA_VERSION = "path_c_vqbc_evaluation_rows_v2"
VQBC_EVALUATION_SUMMARY_SCHEMA_VERSION = "path_c_vqbc_evaluation_summary_v2"
VQBC_OUTER_UNIT_COUNT = 10
VQBC_PAIRINGS_PER_MODE = 100
VQBC_EPISODES_PER_PAIRING = 500


@dataclass(frozen=True, slots=True)
class VQBCPairing:
    deployment_mode: str
    split: str
    left_outer_unit_id: int
    right_outer_unit_id: int
    pairing_id: str


@dataclass(frozen=True, slots=True)
class VQBCPopulationEntry:
    outer_unit_id: int
    resolved_config_path: Path
    checkpoint_path: Path
    config: VQBCConfig
    checkpoint_metadata: CheckpointMetadataV4


@dataclass(frozen=True, slots=True)
class VQBCPopulation:
    population_id: str
    layout: str
    entries: tuple[VQBCPopulationEntry, ...]
    output_root: Path
    path: Path
    sha256: str

    def entry(self, outer_unit_id: int) -> VQBCPopulationEntry:
        if not 0 <= int(outer_unit_id) < len(self.entries):
            raise ValueError("outer_unit_id must be between zero and nine.")
        entry = self.entries[int(outer_unit_id)]
        if entry.outer_unit_id != int(outer_unit_id):
            raise RuntimeError("Population entries changed order.")
        return entry


def build_population_manifest(
    destination: str | Path,
    *,
    population_id: str,
    layout: str,
    resolved_config_paths: Sequence[str | Path],
    checkpoint_paths: Sequence[str | Path],
    output_root: str | Path,
) -> Path:
    """Write a ten-policy manifest after checking each unit-owned reference."""

    configs = tuple(Path(path).resolve() for path in resolved_config_paths)
    checkpoints = tuple(Path(path).resolve() for path in checkpoint_paths)
    if len(configs) != VQBC_OUTER_UNIT_COUNT or len(checkpoints) != len(configs):
        raise ValueError("A population manifest requires ten configs and checkpoints.")
    policies = []
    reference_runs = set()
    for expected_id, (config_path, checkpoint_path) in enumerate(
        zip(configs, checkpoints, strict=True)
    ):
        if not config_path.is_file():
            raise FileNotFoundError(f"Resolved config is missing: {config_path}")
        config = VQBCConfig.from_mapping(
            json.loads(config_path.read_text(encoding="utf-8")),
            base_dir=config_path.parent,
        )
        if (
            config.run_kind != "formal"
            or config.environment.layout != layout
            or config.outer_unit is None
            or config.outer_unit.outer_unit_id != expected_id
        ):
            raise ValueError("Resolved configs must be ordered formal units zero to nine.")
        checkpoint_manifest = checkpoint_path / "manifest.json"
        if not checkpoint_manifest.is_file():
            raise FileNotFoundError(
                f"Checkpoint manifest is missing: {checkpoint_manifest}"
            )
        checkpoint_payload = json.loads(
            checkpoint_manifest.read_text(encoding="utf-8")
        )
        metadata = CheckpointMetadataV4.from_mapping(
            checkpoint_payload.get("metadata", {})
        )
        assert_reference_ownership(
            metadata,
            config.backbone_init,
            expected_outer_unit_id=expected_id,
        )
        reference_runs.add(config.backbone_init.training_run_id)
        policies.append(
            {
                "outer_unit_id": expected_id,
                "resolved_config_path": str(config_path),
                "resolved_config_sha256": file_sha256(config_path),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_manifest_sha256": file_sha256(
                    checkpoint_manifest
                ),
            }
        )
    if len(reference_runs) != VQBC_OUTER_UNIT_COUNT:
        raise ValueError("Each outer unit must retain an independent reference run.")
    payload = {
        "schema_version": VQBC_POPULATION_SCHEMA_VERSION,
        "population_id": str(population_id),
        "layout": str(layout),
        "scientific_readout_allowed": False,
        "policies": policies,
        "output_root": str(Path(output_root).resolve()),
    }
    target = Path(destination).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    # Re-read through the same strict path used by evaluation.
    load_population(target)
    return target


def load_population(path: str | Path) -> VQBCPopulation:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Fourth-model population manifest must be a mapping.")
    fields = {
        "schema_version",
        "population_id",
        "layout",
        "scientific_readout_allowed",
        "policies",
        "output_root",
    }
    if set(payload) != fields or payload["schema_version"] != VQBC_POPULATION_SCHEMA_VERSION:
        raise ValueError("Fourth-model population manifest changed fields.")
    if payload["scientific_readout_allowed"] is not False:
        raise ValueError("The implementation population is non-claim.")
    layout = str(payload["layout"])
    if layout not in {"test_time_simple", "test_time_wide"}:
        raise ValueError("Population layout is not registered.")
    raw_policies = payload["policies"]
    if (
        isinstance(raw_policies, (str, bytes, bytearray))
        or not isinstance(raw_policies, Sequence)
        or len(raw_policies) != VQBC_OUTER_UNIT_COUNT
    ):
        raise ValueError("A fourth-model population requires ten policies.")
    entries = []
    base_dir = manifest_path.parent
    for expected_id, raw in enumerate(raw_policies):
        if not isinstance(raw, Mapping) or set(raw) != {
            "outer_unit_id",
            "resolved_config_path",
            "resolved_config_sha256",
            "checkpoint_path",
            "checkpoint_manifest_sha256",
        }:
            raise ValueError("A population policy entry changed fields.")
        if raw["outer_unit_id"] != expected_id:
            raise ValueError("Population policies must be ordered zero through nine.")
        config_path = Path(str(raw["resolved_config_path"]))
        if not config_path.is_absolute():
            config_path = base_dir / config_path
        config_path = config_path.resolve()
        if (
            not config_path.is_file()
            or file_sha256(config_path) != raw["resolved_config_sha256"]
        ):
            raise ValueError("A resolved fourth-model config changed content.")
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
        config = VQBCConfig.from_mapping(
            config_payload, base_dir=config_path.parent
        )
        if (
            config.run_kind != "formal"
            or config.environment.layout != layout
            or config.outer_unit is None
            or config.outer_unit.outer_unit_id != expected_id
        ):
            raise ValueError("A population config belongs to another formal unit.")
        checkpoint_path = Path(str(raw["checkpoint_path"]))
        if not checkpoint_path.is_absolute():
            checkpoint_path = base_dir / checkpoint_path
        checkpoint_path = checkpoint_path.resolve()
        checkpoint_manifest = checkpoint_path / "manifest.json"
        if (
            not checkpoint_manifest.is_file()
            or file_sha256(checkpoint_manifest)
            != raw["checkpoint_manifest_sha256"]
        ):
            raise ValueError("A fourth-model checkpoint manifest changed content.")
        checkpoint_payload = json.loads(
            checkpoint_manifest.read_text(encoding="utf-8")
        )
        metadata = CheckpointMetadataV4.from_mapping(
            checkpoint_payload.get("metadata", {})
        )
        assert_reference_ownership(
            metadata,
            config.backbone_init,
            expected_outer_unit_id=expected_id,
        )
        entries.append(
            VQBCPopulationEntry(
                outer_unit_id=expected_id,
                resolved_config_path=config_path,
                checkpoint_path=checkpoint_path,
                config=config,
                checkpoint_metadata=metadata,
            )
        )
    if len(
        {entry.config.backbone_init.training_run_id for entry in entries}
    ) != VQBC_OUTER_UNIT_COUNT:
        raise ValueError("Outer units cannot share one global reference checkpoint.")
    output_root = Path(str(payload["output_root"]))
    if not output_root.is_absolute():
        output_root = base_dir / output_root
    return VQBCPopulation(
        population_id=str(payload["population_id"]),
        layout=layout,
        entries=tuple(entries),
        output_root=output_root.resolve(),
        path=manifest_path,
        sha256=file_sha256(manifest_path),
    )


@dataclass(frozen=True, slots=True)
class VQBCEpisodeRow:
    schema_version: str
    population_id: str
    layout: str
    deployment_mode: str
    split: str
    pairing_id: str
    left_outer_unit_id: int
    right_outer_unit_id: int
    episode_index: int
    episode_seed: int
    environment_steps: int
    raw_return: float
    correct_delivery_count: int
    wrong_delivery_count: int
    cumulative_kl: float
    reference_action_deviation_count: int
    mean_quotient_count: float
    mean_belief_entropy: float
    response_code_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.schema_version != VQBC_EVALUATION_ROWS_SCHEMA_VERSION:
            raise ValueError("Fourth-model evaluation row schema changed.")
        if self.deployment_mode not in VQBC_DEPLOYMENT_MODES:
            raise ValueError("Evaluation row has an unknown deployment mode.")
        expected_split = (
            "sp"
            if self.left_outer_unit_id == self.right_outer_unit_id
            else "xp"
        )
        if self.split != expected_split:
            raise ValueError("Evaluation row split disagrees with its outer units.")
        if (
            isinstance(self.episode_index, bool)
            or not isinstance(self.episode_index, Integral)
            or not 0 <= int(self.episode_index) < VQBC_EPISODES_PER_PAIRING
        ):
            raise ValueError("episode_index is outside the 500-episode block.")
        if self.environment_steps != 400:
            raise ValueError("A standard episode must contain 400 environment steps.")
        for name in (
            "raw_return",
            "cumulative_kl",
            "mean_quotient_count",
            "mean_belief_entropy",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name} must be finite.")
        if self.cumulative_kl < -1.0e-6:
            raise ValueError("Cumulative KL cannot be materially negative.")
        if len(self.response_code_counts) != 16 or any(
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or int(value) < 0
            for value in self.response_code_counts
        ):
            raise ValueError("Response counts must cover all sixteen codes.")

    def to_mapping(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["response_code_counts"] = list(self.response_code_counts)
        return payload


def standard_pairings(
    deployment_modes: Sequence[str] = VQBC_DEPLOYMENT_MODES,
) -> tuple[VQBCPairing, ...]:
    modes = tuple(str(value) for value in deployment_modes)
    if modes != VQBC_DEPLOYMENT_MODES:
        raise ValueError("The standard evaluation requires all four deployment modes.")
    pairings = []
    for mode in modes:
        for left in range(VQBC_OUTER_UNIT_COUNT):
            for right in range(VQBC_OUTER_UNIT_COUNT):
                split = "sp" if left == right else "xp"
                pairings.append(
                    VQBCPairing(
                        deployment_mode=mode,
                        split=split,
                        left_outer_unit_id=left,
                        right_outer_unit_id=right,
                        pairing_id=f"{left:02d}_to_{right:02d}",
                    )
                )
    return tuple(pairings)


def standard_episode_seed(
    *,
    population_id: str,
    layout: str,
    left_outer_unit_id: int,
    right_outer_unit_id: int,
    episode_index: int,
) -> int:
    """Use one mode-independent seed for each matched pairing episode."""

    payload = (
        "path_c_vqbc_standard_episode_v1\0"
        f"{population_id}\0{layout}\0{left_outer_unit_id}\0"
        f"{right_outer_unit_id}\0{episode_index}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def validate_standard_rows(rows: Sequence[VQBCEpisodeRow]) -> None:
    values = tuple(rows)
    expected_count = (
        len(VQBC_DEPLOYMENT_MODES)
        * VQBC_PAIRINGS_PER_MODE
        * VQBC_EPISODES_PER_PAIRING
    )
    if len(values) != expected_count:
        raise ValueError("The four standard matrices have an incomplete row count.")
    keys = {
        (
            row.deployment_mode,
            row.left_outer_unit_id,
            row.right_outer_unit_id,
            row.episode_index,
        )
        for row in values
    }
    if len(keys) != expected_count:
        raise ValueError("The four standard matrices contain duplicate cells.")
    matched_seeds: dict[tuple[int, int, int], set[int]] = defaultdict(set)
    for row in values:
        matched_seeds[
            (
                row.left_outer_unit_id,
                row.right_outer_unit_id,
                row.episode_index,
            )
        ].add(row.episode_seed)
    if any(len(seeds) != 1 for seeds in matched_seeds.values()):
        raise ValueError("Deployment modes did not reuse matched episode seeds.")


def summarize_standard_rows(
    rows: Sequence[VQBCEpisodeRow],
) -> Mapping[str, Any]:
    values = tuple(rows)
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in values:
        grouped[
            (row.deployment_mode, row.split, row.pairing_id)
        ].append(float(row.raw_return))
    summary: dict[str, Any] = {}
    for mode in VQBC_DEPLOYMENT_MODES:
        mode_summary = {}
        for split, expected_pairings in (("sp", 10), ("xp", 90)):
            pairing_means = [
                sum(grouped[(mode, split, pairing_id)])
                / len(grouped[(mode, split, pairing_id)])
                for pairing_id in sorted(
                    {
                        row.pairing_id
                        for row in values
                        if row.deployment_mode == mode and row.split == split
                    }
                )
            ]
            if len(pairing_means) != expected_pairings:
                raise ValueError("A deployment mode has an incomplete pairing matrix.")
            mode_summary[split] = {
                "pairing_count": len(pairing_means),
                "pairing_mean_raw_return": sum(pairing_means)
                / len(pairing_means),
            }
        summary[mode] = mode_summary
    return {
        "schema_version": VQBC_EVALUATION_SUMMARY_SCHEMA_VERSION,
        "scientific_readout_allowed": False,
        "deployment_modes": summary,
    }


def write_rows(path: str | Path, rows: Sequence[VQBCEpisodeRow]) -> Path:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_mapping(), sort_keys=True) + "\n")
    temporary.replace(target)
    return target


__all__ = [
    "VQBCEpisodeRow",
    "VQBCPairing",
    "VQBCPopulation",
    "VQBCPopulationEntry",
    "VQBC_EVALUATION_ROWS_SCHEMA_VERSION",
    "build_population_manifest",
    "load_population",
    "standard_episode_seed",
    "standard_pairings",
    "summarize_standard_rows",
    "validate_standard_rows",
    "write_rows",
]
