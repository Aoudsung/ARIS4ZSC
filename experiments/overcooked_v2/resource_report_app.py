"""Aggregate marginal, shared, amortized, and fully-loaded DELTA costs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from src.delta_zsc.resources import ResourceLedger
from src.delta_zsc.storage import ensure_run_identity, read_json, write_json


def _read(path: Path) -> ResourceLedger:
    return ResourceLedger.from_mapping(read_json(path))


def _method_row(method: str, paths: list[Path]) -> dict[str, Any]:
    ledgers = [_read(path) for path in paths]
    training = [item for item in ledgers if item.ego_policy_steps > 0]
    if not training:
        raise ValueError(f"Resource method {method} has no ego-training ledger.")
    shared_steps = {item.shared_training_simulator_steps for item in training}
    shared_gpu = {round(item.shared_gpu_hours, 12) for item in training}
    shared_wall = {round(item.shared_wall_clock_hours, 12) for item in training}
    deployable = {item.deployable_parameters for item in training}
    training_only = {item.training_only_parameters for item in training}
    if len(shared_steps) != 1 or len(shared_gpu) != 1 or len(shared_wall) != 1:
        raise ValueError(
            f"Resource method {method} repeats inconsistent shared infrastructure."
        )
    if len(deployable) != 1 or len(training_only) != 1:
        raise ValueError(f"Resource method {method} parameter counts differ across runs.")

    ego_count = len(training)
    marginal_steps = sum(item.marginal_training_simulator_steps for item in training)
    unique_shared_steps = next(iter(shared_steps))
    post_steps = sum(item.post_training_measurement_steps for item in ledgers)
    training_gpu = sum(item.training_gpu_hours for item in training)
    unique_shared_gpu = next(iter(shared_gpu))
    measurement_gpu = sum(item.measurement_gpu_hours for item in ledgers)
    training_wall = sum(item.training_wall_clock_hours for item in training)
    unique_shared_wall = next(iter(shared_wall))
    measurement_wall = sum(item.measurement_wall_clock_hours for item in ledgers)
    latency = [item.inference_latency_ms for item in training if item.inference_latency_ms > 0]
    return {
        "method": method,
        "ego_run_count": ego_count,
        "source_ledger_count": len(ledgers),
        "marginal_training_steps_total": int(marginal_steps),
        "marginal_training_steps_per_ego": float(marginal_steps / ego_count),
        "unique_shared_training_steps": int(unique_shared_steps),
        "amortized_training_steps_per_ego": float(
            marginal_steps / ego_count + unique_shared_steps / ego_count
        ),
        "post_training_measurement_steps": int(post_steps),
        "fully_loaded_study_steps": int(marginal_steps + unique_shared_steps + post_steps),
        "training_gpu_hours_total": float(training_gpu),
        "unique_shared_gpu_hours": float(unique_shared_gpu),
        "measurement_gpu_hours_total": float(measurement_gpu),
        "fully_loaded_gpu_hours": float(
            training_gpu + unique_shared_gpu + measurement_gpu
        ),
        "training_wall_clock_hours_total": float(training_wall),
        "unique_shared_wall_clock_hours": float(unique_shared_wall),
        "measurement_wall_clock_hours_total": float(measurement_wall),
        "fully_loaded_wall_clock_hours": float(
            training_wall + unique_shared_wall + measurement_wall
        ),
        "peak_memory_bytes": int(max(item.peak_memory_bytes for item in ledgers)),
        "deployable_parameters": int(next(iter(deployable))),
        "training_only_parameters": int(next(iter(training_only))),
        "median_inference_latency_ms": (
            None if not latency else float(np.median(np.asarray(latency)))
        ),
    }


def run_resource_report(args: argparse.Namespace) -> None:
    grouped: dict[str, list[Path]] = {}
    sources: dict[str, list[dict[str, str]]] = {}
    for value in args.ledger:
        try:
            method, raw_path = value.split("=", 1)
        except ValueError as error:
            raise ValueError("Resource inputs use METHOD=/path/resource_ledger.json") from error
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        grouped.setdefault(method, []).append(path)
        sources.setdefault(method, []).append({"path": str(path)})
    if not grouped:
        raise ValueError("At least one resource method is required.")
    rows = [_method_row(method, grouped[method]) for method in sorted(grouped)]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    ensure_run_identity(
        output,
        {
            "stage": "delta-resource-report",
            "sources": sources,
        },
    )
    write_json(
        output / "resource_report.json",
        {
            "version": 1,
            "artifact_type": "delta_resource_report",
            "accounting_rule": (
                "shared upstream cost appears in every ego ledger for disclosure, "
                "but is counted once per reproduced study and amortized over ego runs"
            ),
            "methods": rows,
            "sources": sources,
        },
    )
    with (output / "resource_report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# DELTA-ZSC resource report",
        "",
        "| Method | Ego runs | Marginal/ego | Shared | Amortized/ego | Post-training | Fully loaded steps | Fully loaded GPU-h | Peak memory | Deploy params | Inference ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {ego_run_count} | {marginal_training_steps_per_ego:.1f} | "
            "{unique_shared_training_steps} | {amortized_training_steps_per_ego:.1f} | "
            "{post_training_measurement_steps} | {fully_loaded_study_steps} | "
            "{fully_loaded_gpu_hours:.4f} | {peak_memory_bytes} | "
            "{deployable_parameters} | {median_inference_latency_ms} |".format(**row)
        )
    (output / "resource_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = ["run_resource_report"]
