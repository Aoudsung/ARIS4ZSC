"""Create the preregistered full simulator and hardware accounting table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from src.path_c.resources import (
    ResourceLedger,
    aggregate_resource_ledgers,
)
from src.path_c.storage import (
    ensure_run_identity,
    runtime_provenance,
    sha256_path,
    validate_formal_repository_state,
    validate_registered_python_runtime,
    write_json,
)


RESOURCE_METHODS = (
    "sp",
    "state-augmented",
    "op",
    "fcp",
    "depi",
    "ippo-large",
)


def _read_ledger(path: Path) -> ResourceLedger:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ResourceLedger.from_mapping(payload)


def run_resource_report(args: argparse.Namespace) -> None:
    validate_formal_repository_state()
    validate_registered_python_runtime()
    grouped: dict[str, list[Path]] = {}
    seen_paths: set[Path] = set()
    for value in args.ledger:
        try:
            method, raw_path = value.split("=", 1)
        except ValueError as error:
            raise ValueError("Resource inputs use METHOD=/path/resource_ledger.json.") from error
        if method not in RESOURCE_METHODS:
            raise ValueError(f"Unknown resource method: {method}")
        path = Path(raw_path).resolve()
        if path in seen_paths:
            raise ValueError(f"Resource artifact supplied more than once: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        seen_paths.add(path)
        grouped.setdefault(method, []).append(path)
    if set(grouped) != set(RESOURCE_METHODS):
        raise ValueError("Resource table requires all five methods and IPPO-Large.")

    rows: list[dict[str, Any]] = []
    sources: dict[str, list[dict[str, str]]] = {}
    for method in RESOURCE_METHODS:
        paths = grouped[method]
        aggregate = aggregate_resource_ledgers([_read_ledger(path) for path in paths])
        run_count = len(paths)
        marginal_per_run = aggregate.marginal_training_simulator_steps / run_count
        # Every ego ledger deliberately repeats the complete shared-cost
        # disclosure.  Divide once to recover the unique shared source, then
        # amortize that source over the actual number of ego runs.
        unique_shared = aggregate.shared_training_simulator_steps / run_count
        amortized_per_run = marginal_per_run + unique_shared / run_count
        fully_loaded_reproduction = int(
            aggregate.marginal_training_simulator_steps
            + unique_shared
            + aggregate.final_m1_steps
            + aggregate.component_diagnostic_steps
            + aggregate.mechanism_evaluation_steps
            + aggregate.evaluation_steps
        )
        rows.append(
            {
                "method": method,
                **aggregate.to_mapping(),
                "run_count": run_count,
                "marginal_training_steps_per_run": marginal_per_run,
                "amortized_training_steps_per_run": amortized_per_run,
                "fully_loaded_reproduction_steps": fully_loaded_reproduction,
            }
        )
        sources[method] = [
            {"path": str(path), "sha256": sha256_path(path)} for path in paths
        ]

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    ensure_run_identity(
        output,
        {
            "stage": "formal-resource-report",
            "repository_runtime": runtime_provenance(),
            "sources": sources,
        },
    )
    write_json(
        output / "resource_report.json",
        {
            "methods": rows,
            "sources": sources,
            "accounting_rule": {
                "step_counts_gpu_hours_and_wall_clock_hours": "sum",
                "peak_memory_and_latency": "maximum",
                "parameter_counts": "one deployable instance; nonzero inputs must agree",
                "evaluation_excluded_from_total_training_simulator_steps": True,
                "shared_cost_disclosure": (
                    "shared fields repeat in every ego ledger; report divides by "
                    "run_count once for unique fully-loaded cost and twice for "
                    "per-run amortization"
                ),
                "gpu_hours": "wall time multiplied by GPU devices actually used",
                "inference_latency_ms": (
                    "median of 100 synchronized compiled stochastic batch-one "
                    "policy steps after 20 warm-up steps"
                ),
            },
        },
    )
    with (output / "resource_report.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Complete resource ledger",
        "",
        "| Method | Runs | Marginal/run | Amortized/run | Fully loaded reproduction | Wall-h | GPU-h | Comparator GPU-h | Peak memory | Deploy params | Training-only params | Inference ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {run_count} | {marginal_training_steps_per_run:.1f} | "
            "{amortized_training_steps_per_run:.1f} | "
            "{fully_loaded_reproduction_steps} | {wall_clock_hours:.4f} | {gpu_hours:.4f} | "
            "{comparator_gpu_hours:.4f} | {peak_memory_bytes} | {deployable_parameters} | "
            "{training_only_parameters} | {inference_latency_ms:.6f} |".format(
                **row
            )
        )
    lines.extend(
        (
            "",
            "Evaluation steps are disclosed separately and are not included in total training simulator steps.",
        )
    )
    (output / "resource_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = ["RESOURCE_METHODS", "run_resource_report"]
