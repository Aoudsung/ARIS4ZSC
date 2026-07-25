"""Independent training pipeline with content-addressed shared work."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..contracts.config import FAMILY_POOL_SCHEMA_VERSION, PathCModelConfig


PIPELINE_SCHEMA_VERSION = "path_c_model_pipeline_v1"
STAGE_RECEIPT_SCHEMA_VERSION = "path_c_model_stage_receipt_v1"
LEGACY_STAGES = (
    "pool_check",
    "prefit",
    "calibration",
    "adaptation",
)
FAMILY_POOL_STAGES = (
    "pool_check",
    "prefit",
    "training_calibration",
    "adaptation",
    "deployment_calibration",
)
STAGES = tuple(dict.fromkeys((*LEGACY_STAGES, *FAMILY_POOL_STAGES)))
LEGACY_SHARED_STAGES = frozenset({"pool_check", "prefit", "calibration"})
FAMILY_POOL_SHARED_STAGES = frozenset(
    {"pool_check", "prefit", "training_calibration"}
)


def stages_for_config(config: PathCModelConfig) -> tuple[tuple[str, ...], frozenset[str]]:
    if config.schema_version == FAMILY_POOL_SCHEMA_VERSION:
        return FAMILY_POOL_STAGES, FAMILY_POOL_SHARED_STAGES
    return LEGACY_STAGES, LEGACY_SHARED_STAGES


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


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_records(source_roots: Sequence[str | Path]) -> list[dict[str, str]]:
    """Hash Python trees and explicit source or launch-configuration files."""

    records: list[dict[str, str]] = []
    seen: set[Path] = set()
    for raw_root in source_roots:
        root = Path(raw_root).resolve()
        paths = [root] if root.is_file() else sorted(root.rglob("*.py")) if root.is_dir() else []
        if not paths:
            raise FileNotFoundError(f"Pipeline dependency root has no source file: {root}")
        for path in paths:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            records.append({"path": str(resolved), "sha256": file_sha256(resolved)})
    return sorted(records, key=lambda row: row["path"])


@dataclass(frozen=True, slots=True)
class PipelineContext:
    stage: str
    stage_directory: Path
    config: PathCModelConfig
    stage_input_sha256: str
    preceding_artifacts: Mapping[str, Mapping[str, str]]
    source_files: Sequence[Mapping[str, str]]


def _input_payload(
    *,
    stage: str,
    config: PathCModelConfig,
    sources: Sequence[Mapping[str, str]],
    preceding_artifacts: Mapping[str, Mapping[str, str]],
    shared_stages: frozenset[str],
) -> Mapping[str, Any]:
    config_payload = (
        config.shared_stage_mapping()
        if stage in shared_stages
        else config.to_mapping()
    )
    return {
        "schema_version": "path_c_stage_input_v1",
        "stage": stage,
        "config": config_payload,
        "source_files": list(sources),
        "preceding_artifacts": preceding_artifacts,
    }


def _normalize_artifacts(raw: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    artifacts: dict[str, dict[str, str]] = {}
    for name, value in raw.items():
        if isinstance(value, Mapping):
            if set(value) != {"path", "sha256"}:
                raise ValueError(f"Artifact {name} must contain only path and sha256.")
            path = Path(str(value["path"])).resolve()
            declared = str(value["sha256"])
        else:
            path = Path(value).resolve()
            declared = file_sha256(path) if path.is_file() else ""
        if not path.is_file():
            raise FileNotFoundError(f"Stage artifact is missing: {path}")
        actual = file_sha256(path)
        if declared != actual:
            raise ValueError(f"Stage artifact hash mismatch: {path}")
        artifacts[str(name)] = {"path": str(path), "sha256": actual}
    if not artifacts:
        raise ValueError("A completed stage must expose at least one artifact.")
    return dict(sorted(artifacts.items()))


def _valid_receipt(path: Path, *, stage: str, input_sha256: str) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(receipt, Mapping) or receipt.get("schema_version") != STAGE_RECEIPT_SCHEMA_VERSION:
        return None
    if receipt.get("stage") != stage or receipt.get("stage_input_sha256") != input_sha256:
        return None
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return None
    try:
        normalized = _normalize_artifacts(artifacts)
    except (FileNotFoundError, ValueError, TypeError):
        return None
    if normalized != artifacts:
        return None
    return receipt


def _state_path(config: PathCModelConfig) -> Path:
    return config.output_root / config.condition_id / "pipeline_state.json"


def _load_state(config: PathCModelConfig) -> dict[str, Any]:
    path = _state_path(config)
    if not path.is_file():
        return {
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "condition_id": config.condition_id,
            "stages": {},
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema_version") != PIPELINE_SCHEMA_VERSION:
        raise ValueError("Existing pipeline state uses an unsupported schema.")
    if payload.get("condition_id") != config.condition_id or not isinstance(payload.get("stages"), Mapping):
        raise ValueError("Existing pipeline state belongs to another condition or is malformed.")
    return dict(payload)


def _state_allows_resume(
    state: Mapping[str, Any], *, stage: str, receipt_path: Path, input_sha256: str
) -> bool:
    stages = state.get("stages")
    recorded = stages.get(stage) if isinstance(stages, Mapping) else None
    if not isinstance(recorded, Mapping) or not receipt_path.is_file():
        return False
    return (
        recorded.get("stage_input_sha256") == input_sha256
        and recorded.get("receipt_path") == str(receipt_path)
        and recorded.get("receipt_sha256") == file_sha256(receipt_path)
    )


def run_pipeline(
    config: PathCModelConfig,
    *,
    source_roots: Sequence[str | Path],
    stage_runners: Mapping[str, Callable[[PipelineContext], Mapping[str, Any]]],
    resume: bool = False,
    through: str = "adaptation",
) -> Mapping[str, Any]:
    """Run or safely reuse stages through one requested terminal stage.

    Shared stages are reused across conditions whenever their receipt and every
    artifact still match.  Condition-specific stages are skipped only when
    ``resume`` is requested and the same checks pass.
    """

    stages, shared_stages = stages_for_config(config)
    if through not in stages:
        raise ValueError(f"through must be one of {', '.join(stages)}.")
    required_stages = stages[: stages.index(through) + 1]
    missing = [stage for stage in required_stages if stage not in stage_runners]
    if missing:
        raise ValueError("Pipeline stage runners are missing: " + ", ".join(missing))
    sources = source_records(source_roots)
    state = _load_state(config)
    state["config_sha256"] = config.config_sha256
    state["run_kind"] = config.run_kind
    state["scientific_readout_allowed"] = config.scientific_readout_allowed
    state["source_files"] = sources
    preceding: dict[str, dict[str, str]] = {}

    for stage in required_stages:
        input_payload = _input_payload(
            stage=stage,
            config=config,
            sources=sources,
            preceding_artifacts=preceding,
            shared_stages=shared_stages,
        )
        input_hash = canonical_sha256(input_payload)
        shared = stage in shared_stages
        stage_dir = (
            config.output_root / "shared" / input_hash
            if shared
            else config.output_root / config.condition_id / stage
        )
        receipt_path = stage_dir / "receipt.json"
        reusable = shared or (
            resume
            and _state_allows_resume(
                state,
                stage=stage,
                receipt_path=receipt_path,
                input_sha256=input_hash,
            )
        )
        receipt = _valid_receipt(receipt_path, stage=stage, input_sha256=input_hash) if reusable else None
        if receipt is None:
            context = PipelineContext(
                stage=stage,
                stage_directory=stage_dir,
                config=config,
                stage_input_sha256=input_hash,
                preceding_artifacts=dict(preceding),
                source_files=sources,
            )
            artifacts = _normalize_artifacts(stage_runners[stage](context))
            receipt = {
                "schema_version": STAGE_RECEIPT_SCHEMA_VERSION,
                "stage": stage,
                "shared_across_conditions": shared,
                "condition_id": None if shared else config.condition_id,
                "controller": None if shared else config.controller,
                "run_kind": config.run_kind,
                "scientific_readout_allowed": config.scientific_readout_allowed,
                "stage_input_sha256": input_hash,
                "artifacts": artifacts,
            }
            _atomic_json(receipt_path, receipt)
        stage_artifacts = {
            f"{stage}.{name}": dict(reference)
            for name, reference in receipt["artifacts"].items()
        }
        preceding.update(stage_artifacts)
        state["stages"][stage] = {
            "stage_input_sha256": input_hash,
            "receipt_path": str(receipt_path),
            "receipt_sha256": file_sha256(receipt_path),
            "artifacts": receipt["artifacts"],
            "shared_across_conditions": shared,
        }
        _atomic_json(_state_path(config), state)
    return state
