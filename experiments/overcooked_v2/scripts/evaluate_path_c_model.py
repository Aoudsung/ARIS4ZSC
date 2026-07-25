#!/usr/bin/env python3
"""Evaluate one ten-policy Path C population on the standard directed matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.path_c.evaluation.standard import (
    POPULATION_MANIFEST_SCHEMA_VERSION,
    STANDARD_ROWS_SCHEMA_VERSION,
    PopulationManifest,
    canonical_sha256,
    execute_standard_pairing,
    file_sha256,
    standard_episode_seed,
    standard_pairings,
    summarize_standard_rows,
    validate_standard_summary_payload,
)
from src.path_c.evaluation.response_contrast import (
    RESPONSE_CONTRAST_MANIFEST_SCHEMA_VERSION,
    RESPONSE_CONTRAST_ROW_COUNT,
    RESPONSE_CONTRAST_ROWS_SCHEMA_VERSION,
    ResponseContrastManifest,
    execute_response_contrast_pairing,
    response_contrast_episode_seed,
    response_contrast_pairings,
    summarize_response_contrast_rows,
)
from src.path_c.pipeline.run import source_records

from experiments.overcooked_v2.model_dock.formal_evaluation_runtime import (
    FormalPopulationRuntime,
)
from experiments.overcooked_v2.model_dock.response_contrast_runtime import (
    ResponseContrastRuntime,
)


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _dependency_roots(
    repository_root: Path, manifest: PopulationManifest
) -> tuple[Path, ...]:
    overcooked = repository_root / "experiments" / "overcooked_v2"
    launch_configs = tuple(
        source.launch_config_path
        for unit in manifest.outer_units.units
        for source in unit.sources
    )
    return (
        Path(__file__).resolve(),
        repository_root / "src" / "path_c",
        overcooked / "model_dock",
        overcooked / "official" / "overcooked_v2_experiments_adapter.py",
        overcooked / "official" / "path_c_family_pool_training.py",
        *(
            (
                overcooked
                / "official"
                / "overcooked_v2_experiments_wide_adapter.py",
            )
            if manifest.layout == "test_time_wide"
            else ()
        ),
        overcooked / "path_c_official_artifact.py",
        overcooked / "path_c_flax_policy.py",
        overcooked / "path_c_official_evidence.py",
        overcooked / "path_c_pool_admission.py",
        overcooked / "path_c_response_summary.py",
        overcooked / "path_c_seed.py",
        overcooked / "path_c_standard_training.py",
        manifest.path,
        manifest.outer_units.path,
        *launch_configs,
    )


def _pairing_input(
    manifest: PopulationManifest,
    pairing: Any,
    source_files: Sequence[Mapping[str, str]],
) -> Mapping[str, Any]:
    def policy_payload(policy: Any) -> Mapping[str, Any]:
        return {
            "outer_unit_id": policy.outer_unit_id,
            "policy_id": policy.policy_id,
            "source_type": policy.source_type,
            "checkpoint_path": str(policy.checkpoint_path),
            "checkpoint_manifest_sha256": policy.checkpoint_manifest_sha256,
            "model_weights_sha256": policy.model_weights_sha256,
            "calibration_summary_path": (
                None
                if policy.calibration_summary_path is None
                else str(policy.calibration_summary_path)
            ),
            "calibration_summary_sha256": policy.calibration_summary_sha256,
        }

    return {
        "schema_version": "path_c_standard_pairing_input_v1",
        "population_manifest_sha256": manifest.sha256,
        "outer_units_manifest_sha256": manifest.outer_units.sha256,
        "pairing_id": pairing.pairing_id,
        "split": pairing.split,
        "policy_0": policy_payload(pairing.policy_0),
        "policy_1": policy_payload(pairing.policy_1),
        "episode_seeds_sha256": canonical_sha256(
            [
                standard_episode_seed(
                    manifest=manifest,
                    outer_unit_0=pairing.outer_unit_0,
                    outer_unit_1=pairing.outer_unit_1,
                    episode_index=index,
                )
                for index in range(manifest.episodes_per_pairing)
            ]
        ),
        "source_files": list(source_files),
    }


def _load_reusable_rows(
    *,
    receipt_path: Path,
    rows_path: Path,
    pairing_id: str,
    input_sha256: str,
) -> list[dict[str, Any]] | None:
    if not receipt_path.is_file() or not rows_path.is_file():
        return None
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        rows = [
            json.loads(line)
            for line in rows_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        return None
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("schema_version") != "path_c_standard_pairing_receipt_v1"
        or receipt.get("pairing_id") != pairing_id
        or receipt.get("input_sha256") != input_sha256
        or receipt.get("rows_sha256") != file_sha256(rows_path)
        or len(rows) != 500
    ):
        return None
    for index, row in enumerate(rows):
        if (
            row.get("schema_version") != STANDARD_ROWS_SCHEMA_VERSION
            or row.get("pairing_id") != pairing_id
            or row.get("episode_index") != index
        ):
            return None
    return rows


def run_evaluation(
    manifest: PopulationManifest, *, resume: bool
) -> Mapping[str, Any]:
    repository_root = Path(__file__).resolve().parents[3]
    sources = source_records(_dependency_roots(repository_root, manifest))
    source_sha256 = canonical_sha256(sources)
    output_root = manifest.output_root / manifest.population_id
    # Load and verify all ten parameter artifacts even when every matrix cell is
    # reusable.  A resume request must never hide checkpoint-byte drift.
    runtime = FormalPopulationRuntime(manifest)
    all_rows: list[dict[str, Any]] = []
    pairing_receipts: dict[str, Mapping[str, str]] = {}
    for pairing in standard_pairings(manifest):
        pairing_directory = output_root / "pairings" / pairing.pairing_id
        rows_path = pairing_directory / "rows.jsonl"
        receipt_path = pairing_directory / "receipt.json"
        expected_episode_seeds = tuple(
            standard_episode_seed(
                manifest=manifest,
                outer_unit_0=pairing.outer_unit_0,
                outer_unit_1=pairing.outer_unit_1,
                episode_index=index,
            )
            for index in range(manifest.episodes_per_pairing)
        )
        input_sha256 = canonical_sha256(_pairing_input(manifest, pairing, sources))
        rows = (
            _load_reusable_rows(
                receipt_path=receipt_path,
                rows_path=rows_path,
                pairing_id=pairing.pairing_id,
                input_sha256=input_sha256,
            )
            if resume
            else None
        )
        if rows is not None and any(
            row.get("episode_seed") != expected_episode_seeds[index]
            for index, row in enumerate(rows)
        ):
            rows = None
        if rows is None:
            rows = execute_standard_pairing(
                manifest,
                pairing,
                evaluate_pairing=runtime.evaluate_pairing,
            )
            _atomic_text(
                rows_path,
                "".join(
                    json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                    for row in rows
                ),
            )
            _atomic_json(
                receipt_path,
                {
                    "schema_version": "path_c_standard_pairing_receipt_v1",
                    "pairing_id": pairing.pairing_id,
                    "split": pairing.split,
                    "input_sha256": input_sha256,
                    "row_count": len(rows),
                    "rows_path": str(rows_path),
                    "rows_sha256": file_sha256(rows_path),
                },
            )
        all_rows.extend(rows)
        pairing_receipts[pairing.pairing_id] = {
            "path": str(receipt_path),
            "sha256": file_sha256(receipt_path),
        }
    rows_path = output_root / "rows.jsonl"
    _atomic_text(
        rows_path,
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in all_rows
        ),
    )
    summary = dict(validate_standard_summary_payload(summarize_standard_rows(all_rows)))
    summary.update(
        {
            "run_kind": manifest.run_kind,
            "scientific_readout_allowed": manifest.scientific_readout_allowed,
            "population_manifest_path": str(manifest.path),
            "population_manifest_sha256": manifest.sha256,
            "outer_units_manifest_sha256": manifest.outer_units.sha256,
            "source_sha256": source_sha256,
            "rows_path": str(rows_path),
            "rows_sha256": file_sha256(rows_path),
        }
    )
    summary_path = output_root / "summary.json"
    _atomic_json(summary_path, summary)
    receipt = {
        "schema_version": "path_c_standard_evaluation_receipt_v1",
        "population_id": manifest.population_id,
        "population_manifest_sha256": manifest.sha256,
        "outer_units_manifest_sha256": manifest.outer_units.sha256,
        "source_sha256": source_sha256,
        "rows": {"path": str(rows_path), "sha256": file_sha256(rows_path)},
        "summary": {
            "path": str(summary_path),
            "sha256": file_sha256(summary_path),
        },
        "pairing_receipts": pairing_receipts,
    }
    receipt_path = output_root / "evaluation_receipt.json"
    _atomic_json(receipt_path, receipt)
    return {**receipt, "receipt_path": str(receipt_path)}


def _response_pairing_input(
    manifest: ResponseContrastManifest,
    pairing: Any,
    source_files: Sequence[Mapping[str, str]],
) -> Mapping[str, Any]:
    return {
        "schema_version": "path_c_response_contrast_pairing_input_v1",
        "response_contrast_manifest_sha256": manifest.sha256,
        "population_manifest_sha256": manifest.population.sha256,
        "outer_units_manifest_sha256": manifest.population.outer_units.sha256,
        "pairing_id": pairing.pairing_id,
        "episode_seeds_sha256": canonical_sha256(
            [
                response_contrast_episode_seed(
                    manifest=manifest.population,
                    pairing=pairing,
                    episode_index=index,
                )
                for index in range(manifest.population.episodes_per_pairing)
            ]
        ),
        "source_files": list(source_files),
    }


def _load_reusable_response_rows(
    *,
    receipt_path: Path,
    rows_path: Path,
    pairing_id: str,
    input_sha256: str,
) -> list[dict[str, Any]] | None:
    if not receipt_path.is_file() or not rows_path.is_file():
        return None
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        rows = [
            json.loads(line)
            for line in rows_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        return None
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("schema_version")
        != "path_c_response_contrast_pairing_receipt_v1"
        or receipt.get("pairing_id") != pairing_id
        or receipt.get("input_sha256") != input_sha256
        or receipt.get("rows_sha256") != file_sha256(rows_path)
        or len(rows) != 3 * 500
    ):
        return None
    expected_branches = ("A1", "A2-mask", "A2-use")
    for episode_index in range(500):
        block = rows[3 * episode_index : 3 * episode_index + 3]
        if (
            tuple(row.get("branch") for row in block) != expected_branches
            or any(
                row.get("schema_version") != RESPONSE_CONTRAST_ROWS_SCHEMA_VERSION
                or row.get("pairing_id") != pairing_id
                or row.get("episode_index") != episode_index
                for row in block
            )
        ):
            return None
    return rows


def run_response_contrast(
    manifest: ResponseContrastManifest, *, resume: bool
) -> Mapping[str, Any]:
    repository_root = Path(__file__).resolve().parents[3]
    sources = source_records(
        (
            *_dependency_roots(repository_root, manifest.population),
            manifest.path,
        )
    )
    source_sha256 = canonical_sha256(sources)
    output_root = manifest.output_root / "decision_focused_response_contrast"
    runtime = ResponseContrastRuntime(manifest.population)
    all_rows: list[dict[str, Any]] = []
    pairing_receipts: dict[str, Mapping[str, str]] = {}
    for pairing in response_contrast_pairings(manifest.population):
        pairing_directory = output_root / "pairings" / pairing.pairing_id
        rows_path = pairing_directory / "rows.jsonl"
        receipt_path = pairing_directory / "receipt.json"
        expected_seeds = tuple(
            response_contrast_episode_seed(
                manifest=manifest.population,
                pairing=pairing,
                episode_index=index,
            )
            for index in range(manifest.population.episodes_per_pairing)
        )
        input_sha256 = canonical_sha256(
            _response_pairing_input(manifest, pairing, sources)
        )
        rows = (
            _load_reusable_response_rows(
                receipt_path=receipt_path,
                rows_path=rows_path,
                pairing_id=pairing.pairing_id,
                input_sha256=input_sha256,
            )
            if resume
            else None
        )
        if rows is not None and any(
            rows[3 * index].get("episode_seed") != expected_seeds[index]
            for index in range(500)
        ):
            rows = None
        if rows is None:
            rows = execute_response_contrast_pairing(
                manifest.population,
                pairing,
                evaluate_pairing=runtime.evaluate_pairing,
            )
            _atomic_text(
                rows_path,
                "".join(
                    json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                    for row in rows
                ),
            )
            _atomic_json(
                receipt_path,
                {
                    "schema_version":
                    "path_c_response_contrast_pairing_receipt_v1",
                    "pairing_id": pairing.pairing_id,
                    "input_sha256": input_sha256,
                    "matched_block_count": 500,
                    "row_count": len(rows),
                    "rows_path": str(rows_path),
                    "rows_sha256": file_sha256(rows_path),
                },
            )
        all_rows.extend(rows)
        pairing_receipts[pairing.pairing_id] = {
            "path": str(receipt_path),
            "sha256": file_sha256(receipt_path),
        }
    if len(all_rows) != RESPONSE_CONTRAST_ROW_COUNT:
        raise RuntimeError("The complete response contrast did not produce 135,000 rows.")
    rows_path = output_root / "rows.jsonl"
    _atomic_text(
        rows_path,
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in all_rows
        ),
    )
    summary = dict(summarize_response_contrast_rows(all_rows))
    if (
        summary["pairing_count"] != 90
        or summary["matched_block_count"] != 45_000
        or summary["raw_row_count"] != 135_000
        or summary["executed_environment_steps"] != 54_000_000
        or summary["unique_environment_steps"] > 54_000_000
    ):
        raise RuntimeError("The response contrast failed its formal accounting.")
    summary.update(
        {
            "run_kind": "formal",
            "scientific_readout_allowed": False,
            "response_contrast_manifest_path": str(manifest.path),
            "response_contrast_manifest_sha256": manifest.sha256,
            "population_manifest_sha256": manifest.population.sha256,
            "outer_units_manifest_sha256":
            manifest.population.outer_units.sha256,
            "source_sha256": source_sha256,
            "rows_path": str(rows_path),
            "rows_sha256": file_sha256(rows_path),
        }
    )
    summary_path = output_root / "summary.json"
    _atomic_json(summary_path, summary)
    receipt = {
        "schema_version": "path_c_response_contrast_evaluation_receipt_v1",
        "response_contrast_manifest_sha256": manifest.sha256,
        "population_manifest_sha256": manifest.population.sha256,
        "source_sha256": source_sha256,
        "rows": {"path": str(rows_path), "sha256": file_sha256(rows_path)},
        "summary": {
            "path": str(summary_path),
            "sha256": file_sha256(summary_path),
        },
        "pairing_receipts": pairing_receipts,
    }
    receipt_path = output_root / "evaluation_receipt.json"
    _atomic_json(receipt_path, receipt)
    return {**receipt, "receipt_path": str(receipt_path)}


def _manifest_schema(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("An evaluation manifest must contain a mapping.")
    return str(payload.get("schema_version", ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    schema = _manifest_schema(arguments.manifest)
    if schema == POPULATION_MANIFEST_SCHEMA_VERSION:
        result = run_evaluation(
            PopulationManifest.load(arguments.manifest),
            resume=arguments.resume,
        )
    elif schema == RESPONSE_CONTRAST_MANIFEST_SCHEMA_VERSION:
        result = run_response_contrast(
            ResponseContrastManifest.load(arguments.manifest),
            resume=arguments.resume,
        )
    else:
        raise ValueError("The evaluation manifest schema is not registered.")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
