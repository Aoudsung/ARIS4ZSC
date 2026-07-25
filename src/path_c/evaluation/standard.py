"""Standard ten-policy OvercookedV2 self-play and cross-play evaluation.

Self-play (SP) is a diagonal pairing of one outer training unit with itself.
Cross-play (XP) is an ordered off-diagonal pairing of two independent outer
training units.  Official summaries use pairing means as their statistical
units and never combine SP and XP.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..contracts.config import CONDITION_CONTROLLERS, PathCModelConfig
from ..contracts.outer_units import (
    FORMAL_OUTER_UNIT_COUNT,
    OuterTrainingUnit,
    OuterUnitsManifest,
    load_outer_units_manifest,
)


POPULATION_MANIFEST_SCHEMA_VERSION = "path_c_population_manifest_v1"
STANDARD_ROWS_SCHEMA_VERSION = "path_c_standard_evaluation_rows_v1"
STANDARD_SUMMARY_SCHEMA_VERSION = "path_c_standard_evaluation_summary_v1"
PROJECT_COMPARISON_SCHEMA_VERSION = "path_c_project_comparison_summary_v1"
EPISODES_PER_PAIRING = 500
EPISODE_STEPS = 400
STANDARD_PAIRING_COUNT = 100
STANDARD_ROW_COUNT = STANDARD_PAIRING_COUNT * EPISODES_PER_PAIRING
STANDARD_ROW_FIELDS = frozenset(
    {
        "schema_version",
        "population_id",
        "population_manifest_sha256",
        "outer_units_manifest_sha256",
        "condition_id",
        "layout",
        "split",
        "pairing_id",
        "outer_unit_0",
        "outer_unit_1",
        "policy_0_id",
        "policy_1_id",
        "episode_index",
        "episode_seed",
        "environment_steps",
        "raw_return",
        "correct_delivery_count",
        "wrong_delivery_count",
        "probe_count",
        "safe_candidate_opportunity_count",
        "maximum_probe_budget",
        "indicator_cost",
    }
)
_HEX = frozenset("0123456789abcdef")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence.")
    return value


def _fields(
    payload: Mapping[str, Any], *, allowed: set[str], required: set[str], name: str
) -> None:
    unknown = sorted(set(payload) - allowed)
    missing = sorted(required - set(payload))
    if unknown or missing:
        raise ValueError(
            f"{name} fields are invalid; unknown={unknown}, missing={missing}."
        )


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or not set(value).issubset(_HEX):
        raise ValueError(f"{name} must be a lowercase SHA-256 string.")
    return value


def _path(value: Any, *, base_dir: Path, name: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError(f"{name} must be a non-empty path.")
    path = Path(value)
    return (path if path.is_absolute() else base_dir / path).resolve()


def _verified_file(
    value: Any, declared_hash: Any, *, base_dir: Path, name: str
) -> Path:
    path = _path(value, base_dir=base_dir, name=name)
    expected = _sha256(declared_hash, f"{name}_sha256")
    if not path.is_file():
        raise FileNotFoundError(f"Population artifact is missing: {path}")
    if file_sha256(path) != expected:
        raise ValueError(f"{name} does not match its declared SHA-256.")
    return path


def _source_matches_unit(config: PathCModelConfig, unit: OuterTrainingUnit) -> bool:
    backbone = (
        config.backbone_init.checkpoint_path,
        config.backbone_init.flax_weights_sha256,
        config.backbone_init.training_run_id,
    )
    expected_backbone = (
        unit.backbone.checkpoint_path,
        unit.backbone.flax_weights_sha256,
        unit.backbone.training_run_id,
    )
    if config.is_family_pool:
        partners = tuple(
            (
                member.checkpoint_path,
                member.flax_weights_sha256,
                member.training_run_id,
                member.prototype_index,
                member.checkpoint_index,
            )
            for member in config.partner_pool
        )
        expected_partners = []
        for prototype_index, sources in (
            (1, (unit.backbone,)),
            (2, unit.partners[:2]),
            (3, unit.partners[2:]),
        ):
            expected_partners.extend(
                (
                    snapshot.checkpoint_path,
                    snapshot.flax_weights_sha256,
                    source.training_run_id,
                    prototype_index,
                    snapshot.checkpoint_index,
                )
                for source in sources
                for snapshot in source.checkpoint_history
            )
        expected_partners = tuple(expected_partners)
    else:
        partners = tuple(
            (
                member.checkpoint_path,
                member.flax_weights_sha256,
                member.training_run_id,
                member.family_id,
            )
            for member in config.partner_pool
        )
        expected_partners = tuple(
            (
                member.checkpoint_path,
                member.flax_weights_sha256,
                member.training_run_id,
                member.family_id,
            )
            for member in unit.partners
        )
    return backbone == expected_backbone and partners == expected_partners


def _official_artifact_manifest_path(checkpoint_path: Path) -> Path:
    candidates = (
        checkpoint_path.parent / "official_artifact_manifest.json",
        checkpoint_path.parent.parent / "official_artifact_manifest.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Official artifact manifest is missing beside checkpoint: {checkpoint_path}"
    )


@dataclass(frozen=True, slots=True)
class StandardPolicyEntry:
    outer_unit_id: int
    policy_id: str
    source_type: str
    checkpoint_path: Path
    checkpoint_manifest_sha256: str
    model_weights_sha256: str | None = None
    calibration_summary_path: Path | None = None
    calibration_summary_sha256: str | None = None
    resolved_config: PathCModelConfig | None = None
    calibration_summary: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class PopulationManifest:
    schema_version: str
    population_id: str
    source_type: str
    condition_id: str | None
    controller: str | None
    layout: str
    run_kind: str
    scientific_readout_allowed: bool
    episodes_per_pairing: int
    outer_units: OuterUnitsManifest
    policies: tuple[StandardPolicyEntry, ...]
    output_root: Path
    path: Path
    sha256: str

    @classmethod
    def load(cls, path: str | Path) -> "PopulationManifest":
        manifest_path = Path(path).resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload = _mapping(payload, POPULATION_MANIFEST_SCHEMA_VERSION)
        fields = {
            "schema_version",
            "population_id",
            "source_type",
            "condition_id",
            "controller",
            "layout",
            "run_kind",
            "scientific_readout_allowed",
            "episodes_per_pairing",
            "outer_units_manifest_path",
            "outer_units_manifest_sha256",
            "policies",
            "output_root",
        }
        _fields(
            payload,
            allowed=fields,
            required=fields,
            name=POPULATION_MANIFEST_SCHEMA_VERSION,
        )
        if payload["schema_version"] != POPULATION_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {POPULATION_MANIFEST_SCHEMA_VERSION}."
            )
        if payload["run_kind"] != "formal":
            raise ValueError("A standard population manifest must be formal.")
        if not isinstance(payload["scientific_readout_allowed"], bool):
            raise ValueError("scientific_readout_allowed must be boolean.")
        if payload["episodes_per_pairing"] != EPISODES_PER_PAIRING:
            raise ValueError("Standard evaluation requires exactly 500 episodes per pairing.")
        base_dir = manifest_path.parent
        outer_manifest_path = _verified_file(
            payload["outer_units_manifest_path"],
            payload["outer_units_manifest_sha256"],
            base_dir=base_dir,
            name="outer_units_manifest_path",
        )
        outer_units = load_outer_units_manifest(outer_manifest_path)
        layout = str(payload["layout"])
        if layout != outer_units.layout:
            raise ValueError("Population and outer-unit manifest layouts differ.")
        source_type = str(payload["source_type"])
        if source_type not in {"adaptation_checkpoint", "official_backbone"}:
            raise ValueError("A population source_type is not registered.")
        population_id = str(payload["population_id"])
        condition = payload["condition_id"]
        controller = payload["controller"]
        if source_type == "adaptation_checkpoint":
            if condition not in CONDITION_CONTROLLERS:
                raise ValueError("An adapted population requires one registered condition.")
            if controller != CONDITION_CONTROLLERS[str(condition)]:
                raise ValueError("Population controller does not match its condition.")
            if population_id != condition:
                raise ValueError("An adapted population identifier must equal its condition.")
        elif (
            population_id != "pre_adaptation_backbone"
            or condition is not None
            or controller is not None
        ):
            raise ValueError("The official-backbone population must not bind a condition.")
        policy_payloads = tuple(_sequence(payload["policies"], "policies"))
        if len(policy_payloads) != FORMAL_OUTER_UNIT_COUNT:
            raise ValueError("A standard population requires exactly ten policies.")
        policies = tuple(
            _load_policy_entry(
                item,
                index=index,
                source_type=source_type,
                condition_id=None if condition is None else str(condition),
                controller=None if controller is None else str(controller),
                outer_units=outer_units,
                base_dir=base_dir,
            )
            for index, item in enumerate(policy_payloads)
        )
        if tuple(item.outer_unit_id for item in policies) != tuple(
            range(FORMAL_OUTER_UNIT_COUNT)
        ):
            raise ValueError("Population policies must be ordered by outer unit 0 through 9.")
        if len({item.policy_id for item in policies}) != FORMAL_OUTER_UNIT_COUNT:
            raise ValueError("Population policy identifiers must be unique.")
        if len({item.checkpoint_path for item in policies}) != FORMAL_OUTER_UNIT_COUNT:
            raise ValueError("Population checkpoint paths must be unique.")
        if (
            any(item.model_weights_sha256 is None for item in policies)
            or len({item.model_weights_sha256 for item in policies})
            != FORMAL_OUTER_UNIT_COUNT
        ):
            raise ValueError("Population policy weights must come from ten distinct runs.")
        if source_type == "adaptation_checkpoint" and (
            len({item.calibration_summary_sha256 for item in policies})
            != FORMAL_OUTER_UNIT_COUNT
        ):
            raise ValueError("Adapted population calibrations must be outer-unit specific.")
        return cls(
            schema_version=POPULATION_MANIFEST_SCHEMA_VERSION,
            population_id=population_id,
            source_type=source_type,
            condition_id=None if condition is None else str(condition),
            controller=None if controller is None else str(controller),
            layout=layout,
            run_kind="formal",
            scientific_readout_allowed=bool(payload["scientific_readout_allowed"]),
            episodes_per_pairing=EPISODES_PER_PAIRING,
            outer_units=outer_units,
            policies=policies,
            output_root=_path(
                payload["output_root"], base_dir=base_dir, name="output_root"
            ),
            path=manifest_path,
            sha256=file_sha256(manifest_path),
        )


def _normalize_checkpoint_resolved_config(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Restore the public probe nesting after dataclass serialization."""

    normalized = dict(payload)
    probe = _mapping(normalized.get("probe"), "checkpoint resolved probe config")
    if "safety" in probe or "threshold" in probe:
        return normalized
    expected = {
        "candidate_window",
        "budget_per_episode",
        "safety_rule",
        "threshold_source",
        "decision_null_quantile",
        "information_quantile",
        "manual_threshold",
        "belief_probability_floor",
    }
    if set(probe) != expected:
        raise ValueError("Checkpoint resolved probe config changed format.")
    threshold = {
        "source": probe["threshold_source"],
        "decision_null_quantile": probe["decision_null_quantile"],
        "information_quantile": probe["information_quantile"],
    }
    if probe["manual_threshold"] is not None:
        threshold["manual_value"] = probe["manual_threshold"]
    normalized["probe"] = {
        "candidate_window": probe["candidate_window"],
        "budget_per_episode": probe["budget_per_episode"],
        "safety": {"rule": probe["safety_rule"]},
        "threshold": threshold,
        "belief_probability_floor": probe["belief_probability_floor"],
    }
    return normalized


def _load_policy_entry(
    payload: Mapping[str, Any],
    *,
    index: int,
    source_type: str,
    condition_id: str | None,
    controller: str | None,
    outer_units: OuterUnitsManifest,
    base_dir: Path,
) -> StandardPolicyEntry:
    name = f"policies[{index}]"
    payload = _mapping(payload, name)
    common = {"outer_unit_id", "policy_id", "checkpoint_path", "checkpoint_manifest_sha256"}
    adaptation = {"calibration_summary_path", "calibration_summary_sha256"}
    allowed = common | (adaptation if source_type == "adaptation_checkpoint" else set())
    _fields(payload, allowed=allowed, required=allowed, name=name)
    unit_id = payload["outer_unit_id"]
    if isinstance(unit_id, bool) or not isinstance(unit_id, Integral) or int(unit_id) != index:
        raise ValueError(f"{name}.outer_unit_id must equal its ordered index.")
    unit = outer_units.unit(int(unit_id))
    policy_id = str(payload["policy_id"])
    if policy_id != f"outer_unit_{index:02d}":
        raise ValueError(f"{name}.policy_id must be condition-independent outer_unit_{index:02d}.")
    checkpoint_path = _path(
        payload["checkpoint_path"], base_dir=base_dir, name=f"{name}.checkpoint_path"
    )
    if source_type == "official_backbone":
        checkpoint_manifest_path = _official_artifact_manifest_path(checkpoint_path)
    else:
        if not checkpoint_path.is_dir():
            raise ValueError("An adaptation checkpoint path must name its checkpoint directory.")
        checkpoint_manifest_path = checkpoint_path / "manifest.json"
    declared_checkpoint_hash = _sha256(
        payload["checkpoint_manifest_sha256"], f"{name}.checkpoint_manifest_sha256"
    )
    if not checkpoint_manifest_path.is_file():
        raise FileNotFoundError(f"Policy checkpoint manifest is missing: {checkpoint_manifest_path}")
    if file_sha256(checkpoint_manifest_path) != declared_checkpoint_hash:
        raise ValueError(f"{name} checkpoint manifest hash does not match.")
    if source_type == "official_backbone":
        if checkpoint_path != unit.backbone.checkpoint_path:
            raise ValueError("An official-backbone population can use only each unit's backbone.")
        official_manifest = _mapping(
            json.loads(checkpoint_manifest_path.read_text(encoding="utf-8")),
            f"{name} official checkpoint manifest",
        )
        official_checkpoint = _mapping(
            official_manifest.get("checkpoint"), f"{name} official checkpoint"
        )
        if (
            official_manifest.get("schema_version")
            != "path_c_official_training_artifact_v2"
            or official_manifest.get("run_status") != "completed"
            or official_manifest.get("layout") != outer_units.layout
            or official_manifest.get("seed") != unit.backbone.training_seed
            or official_manifest.get("training_run_id")
            != unit.backbone.training_run_id
            or official_manifest.get("training_config_sha256")
            != unit.backbone.launch_config_sha256
            or Path(str(official_checkpoint.get("path", ""))).resolve()
            != checkpoint_path
            or official_checkpoint.get("model_weights_sha256")
            != unit.backbone.flax_weights_sha256
        ):
            raise ValueError(
                "An official-backbone population changed its outer-unit provenance."
            )
        return StandardPolicyEntry(
            outer_unit_id=index,
            policy_id=policy_id,
            source_type=source_type,
            checkpoint_path=checkpoint_path,
            checkpoint_manifest_sha256=declared_checkpoint_hash,
            model_weights_sha256=unit.backbone.flax_weights_sha256,
        )
    training_sources = {
        path
        for outer_unit in outer_units.units
        for source in outer_unit.sources
        for path in (
            source.checkpoint_path,
            *(snapshot.checkpoint_path for snapshot in source.checkpoint_history),
        )
    }
    if checkpoint_path in training_sources:
        raise ValueError("A training partner or upstream backbone cannot enter an adapted population.")
    checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
    expected_checkpoint_fields = {
        "schema_version",
        "format",
        "weights_file",
        "weights_file_sha256",
        "model_weights_sha256",
        "metadata",
    }
    if (
        not isinstance(checkpoint_manifest, Mapping)
        or set(checkpoint_manifest) != expected_checkpoint_fields
        or checkpoint_manifest.get("schema_version") != "path_c_flax_checkpoint_v1"
        or checkpoint_manifest.get("format") != "flax_msgpack_file_v1"
        or checkpoint_manifest.get("weights_file") != "weights.msgpack"
    ):
        raise ValueError(f"{name} adaptation checkpoint manifest changed format.")
    model_weights_sha256 = _sha256(
        checkpoint_manifest.get("model_weights_sha256"),
        f"{name}.model_weights_sha256",
    )
    weights_path = checkpoint_path / "weights.msgpack"
    if (
        not weights_path.is_file()
        or file_sha256(weights_path) != checkpoint_manifest.get("weights_file_sha256")
    ):
        raise ValueError(f"{name} adaptation checkpoint weights changed.")
    metadata = _mapping(checkpoint_manifest.get("metadata"), f"{name} checkpoint metadata")
    if (
        metadata.get("stage") != "adaptation"
        or metadata.get("run_kind") != "formal"
        or metadata.get("condition_id") != condition_id
        or metadata.get("controller") != controller
        or metadata.get("shared_across_conditions") is not False
        or metadata.get("environment_steps") != 10_000_000
        or metadata.get("completed_episodes") != 25_000
    ):
        raise ValueError(f"{name} checkpoint metadata belongs to another formal policy.")
    extra = _mapping(metadata.get("extra"), f"{name} checkpoint extra metadata")
    resolved = PathCModelConfig.from_mapping(
        _normalize_checkpoint_resolved_config(
            _mapping(extra.get("resolved_config"), f"{name} resolved config")
        ),
        base_dir=base_dir,
    )
    binding = resolved.formal_outer_unit
    if (
        binding is None
        or binding.outer_unit_id != index
        or binding.outer_units_manifest_sha256 != outer_units.sha256
        or resolved.condition_id != condition_id
        or resolved.controller != controller
        or not _source_matches_unit(resolved, unit)
        or metadata.get("config_sha256") != resolved.config_sha256
        or metadata.get("backbone_origin_sha256")
        != unit.backbone.flax_weights_sha256
        or metadata.get("scientific_readout_allowed")
        is not resolved.scientific_readout_allowed
        or extra.get("model_weights_sha256") != model_weights_sha256
    ):
        raise ValueError(f"{name} resolved config does not bind the declared outer unit.")
    calibration_path = _verified_file(
        payload["calibration_summary_path"],
        payload["calibration_summary_sha256"],
        base_dir=base_dir,
        name=f"{name}.calibration_summary_path",
    )
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration_rows_path = calibration_path.with_name("rows.jsonl")
    required_calibration = {
        "schema_version": "path_c_probe_calibration_v2",
        "run_kind": "formal",
        "scientific_readout_allowed": resolved.scientific_readout_allowed,
        "completed_episodes": 500,
    }
    if any(calibration.get(key) != value for key, value in required_calibration.items()):
        raise ValueError(f"{name} calibration summary is not the bound formal calibration.")
    calibration_bound = (
        (
            calibration.get("parameter_artifact") == "adaptation.manifest"
            and calibration.get("parameter_manifest_sha256")
            == declared_checkpoint_hash
        )
        if resolved.is_family_pool
        else extra.get("calibration_summary_sha256")
        == str(payload["calibration_summary_sha256"])
    )
    if (
        not calibration_bound
        or calibration.get("response_vocabulary_sha256")
        != metadata.get("response_vocabulary_sha256")
        or not calibration_rows_path.is_file()
        or file_sha256(calibration_rows_path) != calibration.get("rows_sha256")
    ):
        raise ValueError(
            f"{name} adaptation checkpoint and calibration artifacts are not bound."
        )
    for key in (
        "decision_threshold",
        "information_threshold",
        "random_trigger_probability",
    ):
        value = calibration.get(key)
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
            raise ValueError(f"{name} calibration summary has an invalid {key}.")
    return StandardPolicyEntry(
        outer_unit_id=index,
        policy_id=policy_id,
        source_type=source_type,
        checkpoint_path=checkpoint_path,
        checkpoint_manifest_sha256=declared_checkpoint_hash,
        model_weights_sha256=model_weights_sha256,
        calibration_summary_path=calibration_path,
        calibration_summary_sha256=str(payload["calibration_summary_sha256"]),
        resolved_config=resolved,
        calibration_summary=calibration,
    )


@dataclass(frozen=True, slots=True)
class StandardPairing:
    outer_unit_0: int
    outer_unit_1: int
    policy_0: StandardPolicyEntry
    policy_1: StandardPolicyEntry

    @property
    def split(self) -> str:
        return "sp" if self.outer_unit_0 == self.outer_unit_1 else "xp"

    @property
    def pairing_id(self) -> str:
        return f"outer_unit_{self.outer_unit_0:02d}__outer_unit_{self.outer_unit_1:02d}"


def standard_pairings(manifest: PopulationManifest) -> tuple[StandardPairing, ...]:
    """Return the complete directed 10 by 10 matrix in stable row-major order."""

    by_unit = {item.outer_unit_id: item for item in manifest.policies}
    pairings = tuple(
        StandardPairing(left, right, by_unit[left], by_unit[right])
        for left in range(FORMAL_OUTER_UNIT_COUNT)
        for right in range(FORMAL_OUTER_UNIT_COUNT)
    )
    if sum(item.split == "sp" for item in pairings) != 10 or sum(
        item.split == "xp" for item in pairings
    ) != 90:
        raise RuntimeError("The standard directed matrix was constructed incorrectly.")
    return pairings


def standard_episode_seed(
    *,
    manifest: PopulationManifest,
    outer_unit_0: int,
    outer_unit_1: int,
    episode_index: int,
) -> int:
    """Derive condition-independent randomness for one ordered matrix cell."""

    if (
        isinstance(episode_index, bool)
        or not isinstance(episode_index, Integral)
        or not 0 <= int(episode_index) < EPISODES_PER_PAIRING
    ):
        raise ValueError("episode_index must be between 0 and 499.")
    left = manifest.outer_units.unit(outer_unit_0)
    right = manifest.outer_units.unit(outer_unit_1)
    payload = (
        f"path_c_standard_evaluation_seed_v2\0{manifest.outer_units.seed_root}\0"
        f"{manifest.layout}\0{left.seeds.evaluation_seed}\0"
        f"{right.seeds.evaluation_seed}\0{int(episode_index)}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _normalize_standard_result(
    *,
    manifest: PopulationManifest,
    pairing: StandardPairing,
    episode_index: int,
    episode_seed: int,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "raw_return",
        "correct_delivery_count",
        "wrong_delivery_count",
        "probe_count",
    }
    allowed = required | {
        "safe_candidate_opportunity_count",
        "maximum_probe_budget",
        "indicator_cost",
    }
    _fields(result, allowed=allowed, required=required, name="standard episode result")
    raw_return = result["raw_return"]
    if isinstance(raw_return, bool) or not isinstance(raw_return, Real) or not math.isfinite(
        float(raw_return)
    ):
        raise ValueError("raw_return must be finite and numeric.")
    counts: dict[str, int] = {}
    for name in (
        "correct_delivery_count",
        "wrong_delivery_count",
        "probe_count",
        "safe_candidate_opportunity_count",
        "maximum_probe_budget",
    ):
        value = result.get(name, 0)
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
            raise ValueError(f"{name} must be a non-negative integer.")
        counts[name] = int(value)
    if counts["probe_count"] > counts["safe_candidate_opportunity_count"]:
        raise ValueError("probe_count cannot exceed safe candidate opportunities.")
    if counts["probe_count"] > counts["maximum_probe_budget"]:
        raise ValueError("probe_count cannot exceed the maximum probe budget.")
    indicator_cost = result.get("indicator_cost", 0.0)
    if isinstance(indicator_cost, bool) or not isinstance(indicator_cost, Real) or not math.isfinite(
        float(indicator_cost)
    ) or float(indicator_cost) < 0.0:
        raise ValueError("indicator_cost must be non-negative, finite, and numeric.")
    return {
        "schema_version": STANDARD_ROWS_SCHEMA_VERSION,
        "population_id": manifest.population_id,
        "population_manifest_sha256": manifest.sha256,
        "outer_units_manifest_sha256": manifest.outer_units.sha256,
        "condition_id": manifest.condition_id,
        "layout": manifest.layout,
        "split": pairing.split,
        "pairing_id": pairing.pairing_id,
        "outer_unit_0": pairing.outer_unit_0,
        "outer_unit_1": pairing.outer_unit_1,
        "policy_0_id": pairing.policy_0.policy_id,
        "policy_1_id": pairing.policy_1.policy_id,
        "episode_index": int(episode_index),
        "episode_seed": int(episode_seed),
        "environment_steps": EPISODE_STEPS,
        "raw_return": float(raw_return),
        "correct_delivery_count": counts["correct_delivery_count"],
        "wrong_delivery_count": counts["wrong_delivery_count"],
        "probe_count": counts["probe_count"],
        "safe_candidate_opportunity_count": counts[
            "safe_candidate_opportunity_count"
        ],
        "maximum_probe_budget": counts["maximum_probe_budget"],
        "indicator_cost": float(indicator_cost),
    }


def execute_standard_pairing(
    manifest: PopulationManifest,
    pairing: StandardPairing,
    *,
    evaluate_pairing: Callable[
        [StandardPairing, Sequence[int]], Sequence[Mapping[str, Any]]
    ],
) -> list[dict[str, Any]]:
    seeds = [
        standard_episode_seed(
            manifest=manifest,
            outer_unit_0=pairing.outer_unit_0,
            outer_unit_1=pairing.outer_unit_1,
            episode_index=index,
        )
        for index in range(EPISODES_PER_PAIRING)
    ]
    if len(set(seeds)) != EPISODES_PER_PAIRING:
        raise RuntimeError("A standard pairing contains duplicate episode seeds.")
    results = tuple(evaluate_pairing(pairing, seeds))
    if len(results) != EPISODES_PER_PAIRING:
        raise RuntimeError("A standard pairing evaluator must return exactly 500 episodes.")
    return [
        _normalize_standard_result(
            manifest=manifest,
            pairing=pairing,
            episode_index=index,
            episode_seed=seeds[index],
            result=result,
        )
        for index, result in enumerate(results)
    ]


def summarize_standard_rows(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Compute official SP and XP statistics from complete raw matrix rows."""

    values = tuple(rows)
    if len(values) != STANDARD_ROW_COUNT:
        raise ValueError("A standard summary requires exactly 50,000 raw episode rows.")
    identities: set[tuple[int, int, int]] = set()
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    episode_seeds: dict[tuple[int, int], set[int]] = defaultdict(set)
    population_ids: set[str] = set()
    population_manifest_hashes: set[str] = set()
    outer_manifest_hashes: set[str] = set()
    condition_ids: set[str | None] = set()
    layouts: set[str] = set()
    for raw_row in values:
        row = _mapping(raw_row, "standard raw row")
        _fields(
            row,
            allowed=set(STANDARD_ROW_FIELDS),
            required=set(STANDARD_ROW_FIELDS),
            name="standard raw row",
        )
        if row.get("schema_version") != STANDARD_ROWS_SCHEMA_VERSION:
            raise ValueError("Standard summaries accept only registered standard raw rows.")
        coordinates = (
            row["outer_unit_0"],
            row["outer_unit_1"],
            row["episode_index"],
        )
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in coordinates):
            raise ValueError("Standard matrix coordinates must be integers.")
        left, right, episode = (int(value) for value in coordinates)
        if (
            not 0 <= left < FORMAL_OUTER_UNIT_COUNT
            or not 0 <= right < FORMAL_OUTER_UNIT_COUNT
            or not 0 <= episode < EPISODES_PER_PAIRING
        ):
            raise ValueError("A standard raw row has an out-of-range matrix coordinate.")
        identity = (left, right, episode)
        if identity in identities:
            raise ValueError("Standard raw rows repeat an outer-unit pairing episode.")
        identities.add(identity)
        expected_split = "sp" if left == right else "xp"
        if row.get("split") != expected_split:
            raise ValueError("A raw row has an incorrect SP or XP label.")
        expected_pairing_id = f"outer_unit_{left:02d}__outer_unit_{right:02d}"
        if row.get("pairing_id") != expected_pairing_id:
            raise ValueError("A raw row has a condition-dependent or incorrect pairing ID.")
        environment_steps = row["environment_steps"]
        if (
            isinstance(environment_steps, bool)
            or not isinstance(environment_steps, Integral)
            or int(environment_steps) != EPISODE_STEPS
        ):
            raise ValueError("Every standard episode must contain 400 environment steps.")
        if (
            row.get("policy_0_id") != f"outer_unit_{left:02d}"
            or row.get("policy_1_id") != f"outer_unit_{right:02d}"
        ):
            raise ValueError("A standard raw row names the wrong outer-unit policy.")
        episode_seed = row.get("episode_seed")
        if (
            isinstance(episode_seed, bool)
            or not isinstance(episode_seed, Integral)
            or not 0 <= int(episode_seed) <= 0xFFFF_FFFF
        ):
            raise ValueError("A standard raw row has an invalid episode seed.")
        if int(episode_seed) in episode_seeds[(left, right)]:
            raise ValueError("A standard matrix cell repeats an episode seed.")
        episode_seeds[(left, right)].add(int(episode_seed))
        raw_return_value = row["raw_return"]
        if (
            isinstance(raw_return_value, bool)
            or not isinstance(raw_return_value, Real)
            or not math.isfinite(float(raw_return_value))
        ):
            raise ValueError("Standard raw returns must be finite.")
        raw_return = float(raw_return_value)
        for count_name in (
            "correct_delivery_count",
            "wrong_delivery_count",
            "probe_count",
            "safe_candidate_opportunity_count",
            "maximum_probe_budget",
        ):
            count = row[count_name]
            if isinstance(count, bool) or not isinstance(count, Integral) or int(count) < 0:
                raise ValueError(f"{count_name} must be a non-negative integer.")
        if int(row["probe_count"]) > int(row["safe_candidate_opportunity_count"]):
            raise ValueError("probe_count cannot exceed safe candidate opportunities.")
        if int(row["probe_count"]) > int(row["maximum_probe_budget"]):
            raise ValueError("probe_count cannot exceed the maximum probe budget.")
        indicator_cost = row["indicator_cost"]
        if (
            isinstance(indicator_cost, bool)
            or not isinstance(indicator_cost, Real)
            or not math.isfinite(float(indicator_cost))
            or float(indicator_cost) < 0.0
        ):
            raise ValueError("indicator_cost must be non-negative and finite.")
        population_id = row["population_id"]
        condition_id = row["condition_id"]
        if not isinstance(population_id, str) or not population_id:
            raise ValueError("A standard raw row requires a population identifier.")
        if condition_id is None:
            if population_id != "pre_adaptation_backbone":
                raise ValueError("Only the backbone population may omit a condition.")
        elif condition_id not in CONDITION_CONTROLLERS or population_id != condition_id:
            raise ValueError("A standard raw row mixes its population and condition.")
        population_manifest_hash = _sha256(
            row["population_manifest_sha256"], "population_manifest_sha256"
        )
        outer_manifest_hash = _sha256(
            row["outer_units_manifest_sha256"], "outer_units_manifest_sha256"
        )
        layout = row["layout"]
        if layout not in {"test_time_simple", "test_time_wide"}:
            raise ValueError("A standard raw row has an unregistered layout.")
        grouped[(left, right)].append(raw_return)
        population_ids.add(population_id)
        population_manifest_hashes.add(population_manifest_hash)
        outer_manifest_hashes.add(outer_manifest_hash)
        condition_ids.add(condition_id)
        layouts.add(str(layout))
    expected_coordinates = {
        (left, right, episode)
        for left in range(FORMAL_OUTER_UNIT_COUNT)
        for right in range(FORMAL_OUTER_UNIT_COUNT)
        for episode in range(EPISODES_PER_PAIRING)
    }
    if identities != expected_coordinates:
        raise ValueError("The standard raw matrix is incomplete.")
    if (
        len(population_ids) != 1
        or len(population_manifest_hashes) != 1
        or len(outer_manifest_hashes) != 1
        or len(condition_ids) != 1
        or len(layouts) != 1
    ):
        raise ValueError("A standard summary cannot mix populations or conditions.")
    if any(len(group) != EPISODES_PER_PAIRING for group in grouped.values()):
        raise ValueError("Each standard pairing must contain exactly 500 episodes.")
    if any(len(seeds) != EPISODES_PER_PAIRING for seeds in episode_seeds.values()):
        raise ValueError("Each standard pairing must contain 500 unique episode seeds.")
    pairing_means = {
        coordinate: float(np.asarray(group, dtype=np.float64).mean())
        for coordinate, group in grouped.items()
    }
    sp_means = np.asarray(
        [pairing_means[(unit, unit)] for unit in range(FORMAL_OUTER_UNIT_COUNT)],
        dtype=np.float64,
    )
    xp_means = np.asarray(
        [
            pairing_means[(left, right)]
            for left in range(FORMAL_OUTER_UNIT_COUNT)
            for right in range(FORMAL_OUTER_UNIT_COUNT)
            if left != right
        ],
        dtype=np.float64,
    )
    sp_mean = float(sp_means.mean())
    xp_mean = float(xp_means.mean())
    return {
        "schema_version": STANDARD_SUMMARY_SCHEMA_VERSION,
        "source": "recomputed_from_standard_raw_episode_rows",
        "population_id": next(iter(population_ids)),
        "population_manifest_sha256": next(iter(population_manifest_hashes)),
        "outer_units_manifest_sha256": next(iter(outer_manifest_hashes)),
        "condition_id": next(iter(condition_ids)),
        "layout": next(iter(layouts)),
        "episodes_per_pairing": EPISODES_PER_PAIRING,
        "sp_pairing_count": int(sp_means.size),
        "sp_raw_row_count": int(sp_means.size * EPISODES_PER_PAIRING),
        "sp_mean_raw_return": sp_mean,
        "sp_pairing_standard_deviation": float(sp_means.std(ddof=0)),
        "xp_pairing_count": int(xp_means.size),
        "xp_raw_row_count": int(xp_means.size * EPISODES_PER_PAIRING),
        "xp_mean_raw_return": xp_mean,
        "xp_pairing_standard_deviation": float(xp_means.std(ddof=0)),
        "sp_minus_xp": sp_mean - xp_mean,
        "effective_environment_steps": len(values) * EPISODE_STEPS,
        "pairings": {
            f"outer_unit_{left:02d}__outer_unit_{right:02d}": {
                "split": "sp" if left == right else "xp",
                "episode_count": EPISODES_PER_PAIRING,
                "mean_raw_return": pairing_means[(left, right)],
            }
            for left in range(FORMAL_OUTER_UNIT_COUNT)
            for right in range(FORMAL_OUTER_UNIT_COUNT)
        },
    }


def validate_standard_summary_payload(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    """Reject mixed-population and episode-standard-error summary fields."""

    payload = _mapping(summary, "standard summary")
    prohibited = {
        "overall_mean_raw_return",
        "overall_raw_return_standard_error",
        "sp_standard_error",
        "xp_standard_error",
        "episode_standard_error",
    }
    present = sorted(prohibited & set(payload))
    if present:
        raise ValueError(
            "Official SP/XP summaries prohibit mixed means and episode-level standard errors: "
            + ", ".join(present)
            + "."
        )
    if payload.get("schema_version") != STANDARD_SUMMARY_SCHEMA_VERSION:
        raise ValueError("Official summary schema version changed.")
    if payload.get("sp_pairing_count") != 10 or payload.get("xp_pairing_count") != 90:
        raise ValueError("Official summary pairing counts changed.")
    if payload.get("sp_raw_row_count") != 5_000 or payload.get("xp_raw_row_count") != 45_000:
        raise ValueError("Official summary raw-row counts changed.")
    return payload


def summarize_project_xp_difference(
    decision_rows: Sequence[Mapping[str, Any]],
    no_probe_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_seed: int,
    bootstrap_samples: int = 9_999,
) -> Mapping[str, Any]:
    """Keep the condition contrast separate from official SP and XP summaries."""

    decision_summary = validate_standard_summary_payload(
        summarize_standard_rows(decision_rows)
    )
    no_probe_summary = validate_standard_summary_payload(
        summarize_standard_rows(no_probe_rows)
    )
    if (
        decision_summary.get("condition_id") != "decision_focused"
        or no_probe_summary.get("condition_id") != "no_probe"
        or decision_summary.get("layout") != no_probe_summary.get("layout")
        or decision_summary.get("outer_units_manifest_sha256")
        != no_probe_summary.get("outer_units_manifest_sha256")
        or decision_summary.get("population_manifest_sha256")
        == no_probe_summary.get("population_manifest_sha256")
    ):
        raise ValueError("Project comparison summaries bind the wrong conditions.")
    decision_seeds = _xp_episode_seeds_by_coordinate(decision_rows)
    no_probe_seeds = _xp_episode_seeds_by_coordinate(no_probe_rows)
    if decision_seeds != no_probe_seeds:
        raise ValueError("Project comparison populations do not share episode seeds.")
    left = _xp_pairing_means_by_coordinate(decision_rows, "decision_focused")
    right = _xp_pairing_means_by_coordinate(no_probe_rows, "no_probe")
    if set(left) != set(right):
        raise ValueError("Project comparison populations have different XP coordinates.")
    differences = np.asarray([left[key] - right[key] for key in sorted(left)], dtype=np.float64)
    if isinstance(bootstrap_samples, bool) or int(bootstrap_samples) <= 0:
        raise ValueError("bootstrap_samples must be positive.")
    rng = np.random.default_rng(int(bootstrap_seed))
    draws = np.empty((int(bootstrap_samples),), dtype=np.float64)
    for index in range(int(bootstrap_samples)):
        sampled_nodes = rng.integers(
            0, FORMAL_OUTER_UNIT_COUNT, size=FORMAL_OUTER_UNIT_COUNT
        )
        while np.unique(sampled_nodes).size < 2:
            sampled_nodes = rng.integers(
                0, FORMAL_OUTER_UNIT_COUNT, size=FORMAL_OUTER_UNIT_COUNT
            )
        sampled = [
            left[(int(first), int(second))] - right[(int(first), int(second))]
            for first in sampled_nodes
            for second in sampled_nodes
            if first != second
        ]
        draws[index] = float(np.asarray(sampled, dtype=np.float64).mean())
    return {
        "schema_version": PROJECT_COMPARISON_SCHEMA_VERSION,
        "source": "paired_xp_pairing_means_with_outer_unit_node_resampling",
        "left_condition": "decision_focused",
        "right_condition": "no_probe",
        "xp_pairing_count": len(differences),
        "mean_xp_difference": float(differences.mean()),
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap_samples": int(bootstrap_samples),
        "outer_unit_node_resampling_interval_95_percent": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "not_an_official_sp_xp_summary": True,
    }


def _xp_pairing_means_by_coordinate(
    rows: Sequence[Mapping[str, Any]], expected_condition: str
) -> dict[tuple[int, int], float]:
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("schema_version") != STANDARD_ROWS_SCHEMA_VERSION:
            raise ValueError("Project comparisons require standard raw rows.")
        if row.get("condition_id") != expected_condition:
            raise ValueError("Project comparison rows belong to another condition.")
        left = int(row["outer_unit_0"])
        right = int(row["outer_unit_1"])
        if left != right:
            grouped[(left, right)].append(float(row["raw_return"]))
    expected = {
        (left, right)
        for left in range(FORMAL_OUTER_UNIT_COUNT)
        for right in range(FORMAL_OUTER_UNIT_COUNT)
        if left != right
    }
    if set(grouped) != expected or any(
        len(values) != EPISODES_PER_PAIRING for values in grouped.values()
    ):
        raise ValueError("Project comparison requires all 90 XP cells with 500 episodes each.")
    return {
        coordinate: float(np.asarray(values, dtype=np.float64).mean())
        for coordinate, values in grouped.items()
    }


def _xp_episode_seeds_by_coordinate(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, int, int], int]:
    return {
        (int(row["outer_unit_0"]), int(row["outer_unit_1"]), int(row["episode_index"])): int(
            row["episode_seed"]
        )
        for row in rows
        if int(row["outer_unit_0"]) != int(row["outer_unit_1"])
    }


__all__ = [
    "EPISODES_PER_PAIRING",
    "POPULATION_MANIFEST_SCHEMA_VERSION",
    "PopulationManifest",
    "STANDARD_ROWS_SCHEMA_VERSION",
    "STANDARD_SUMMARY_SCHEMA_VERSION",
    "StandardPairing",
    "StandardPolicyEntry",
    "canonical_sha256",
    "execute_standard_pairing",
    "file_sha256",
    "standard_episode_seed",
    "standard_pairings",
    "summarize_project_xp_difference",
    "summarize_standard_rows",
    "validate_standard_summary_payload",
]
