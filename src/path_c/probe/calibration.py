"""Content-addressed no-probe threshold calibration."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


CALIBRATION_SCHEMA_VERSION = "path_c_probe_calibration_v2"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nearest_rank(values: Sequence[float], quantile: float) -> float:
    """Use the preregistered non-interpolating nearest-rank quantile."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Calibration scores must be one finite non-empty sequence.")
    if not 0.0 < float(quantile) < 1.0:
        raise ValueError("Calibration quantile must lie in (0, 1).")
    ordered = np.sort(array)
    index = max(0, min(int(math.ceil(float(quantile) * ordered.size)) - 1, ordered.size - 1))
    return float(ordered[index])


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    decision_null_quantile: float
    decision_null_quantile_value: float
    decision_zero_anchor: float
    decision_threshold: float
    decision_null_exceedance_count: int
    information_threshold: float
    information_quantile: float
    random_trigger_probability: float
    opportunity_count: int

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "decision_null_quantile": self.decision_null_quantile,
            "decision_null_quantile_value": self.decision_null_quantile_value,
            "decision_zero_anchor": self.decision_zero_anchor,
            "decision_threshold": self.decision_threshold,
            "decision_null_exceedance_count": self.decision_null_exceedance_count,
            "information_threshold": self.information_threshold,
            "information_quantile": self.information_quantile,
            "random_trigger_probability": self.random_trigger_probability,
            "opportunity_count": self.opportunity_count,
            "actual_probe_count": 0,
        }


def calibrate(
    rows: Sequence[Mapping[str, Any]],
    *,
    decision_null_quantile: float,
    information_quantile: float,
    decision_threshold_override: float | None = None,
) -> CalibrationResult:
    if not rows:
        raise ValueError("Calibration requires decision rows.")
    if any(bool(row.get("probed", False)) for row in rows):
        raise ValueError("Calibration rows must come from no-probe episodes.")
    decision = [float(row["max_decision_score"]) for row in rows if bool(row.get("has_safe_candidate"))]
    information = [float(row["max_information_score"]) for row in rows if bool(row.get("has_safe_candidate"))]
    if not decision or len(decision) != len(information):
        raise ValueError("Every safe calibration opportunity needs both registered scores.")
    raw_decision_quantile = nearest_rank(decision, decision_null_quantile)
    if decision_threshold_override is None:
        threshold_source_value = raw_decision_quantile
    else:
        threshold_source_value = float(decision_threshold_override)
        if not math.isfinite(threshold_source_value):
            raise ValueError("A manual decision threshold must be finite.")
    decision_threshold = max(0.0, threshold_source_value)
    information_threshold = nearest_rank(information, information_quantile)
    exceedance_count = int(np.count_nonzero(np.asarray(decision) > decision_threshold))
    frequency = float(exceedance_count / len(decision))
    return CalibrationResult(
        decision_null_quantile=float(decision_null_quantile),
        decision_null_quantile_value=raw_decision_quantile,
        decision_zero_anchor=0.0,
        decision_threshold=decision_threshold,
        decision_null_exceedance_count=exceedance_count,
        information_threshold=information_threshold,
        information_quantile=float(information_quantile),
        random_trigger_probability=frequency,
        opportunity_count=len(decision),
    )


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_calibration_artifacts(
    output_dir: str | Path,
    rows: Sequence[Mapping[str, Any]],
    result: CalibrationResult,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Mapping[str, Mapping[str, str]]:
    root = Path(output_dir).resolve()
    rows_path = root / "rows.jsonl"
    summary_path = root / "summary.json"
    _atomic_text(
        rows_path,
        "".join(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
    )
    summary = result.to_mapping()
    if metadata is not None:
        overlap = set(summary) & set(metadata)
        if overlap:
            raise ValueError("Calibration metadata cannot replace registered summary fields.")
        summary.update(dict(metadata))
    summary["rows_sha256"] = file_sha256(rows_path)
    _atomic_text(summary_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return {
        "rows": {"path": str(rows_path), "sha256": file_sha256(rows_path)},
        "summary": {"path": str(summary_path), "sha256": file_sha256(summary_path)},
    }
