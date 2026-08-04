"""Executable and paired summary workflow for the DEPI B0--B3 matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from experiments.overcooked_v2.training_app import run_training
from src.path_c.experiment import (
    CONFIG_VERSION,
    METHOD_VERSION,
    official_training_domain_keys,
)
from src.path_c.resources import ResourceLedger
from src.path_c.storage import (
    ensure_run_identity,
    runtime_provenance,
    sha256_path,
    write_json,
)


DEVELOPMENT_VARIANTS = ("b0", "b1", "b2")


def validate_development_entry_alignment(
    entries: Sequence[Mapping[str, Any]],
) -> None:
    """Fail closed unless every per-seed B0--B2 block is scientifically paired."""

    by_seed: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for entry in entries:
        key = (int(entry["protocol_components"]), int(entry["seed_index"]))
        by_seed.setdefault(key, []).append(entry)
    if not by_seed:
        raise ValueError("Development matrix has no run entries.")
    for (component_count, seed_index), rows in by_seed.items():
        if (
            len(rows) != len(DEVELOPMENT_VARIANTS)
            or {str(row["variant"]) for row in rows} != set(DEVELOPMENT_VARIANTS)
        ):
            raise RuntimeError("Every development seed must cover B0--B2 exactly once.")
        checks = {
            "partner sampler": {
                str(row["partner_sampler_sha256"]) for row in rows
            },
            "simulator budget": {
                int(row["total_training_simulator_steps"]) for row in rows
            },
            "deployable parameter capacity": {
                int(row["deployable_parameters"]) for row in rows
            },
            "episode keys": {
                json.dumps(row["episode_key_domains"], sort_keys=True)
                for row in rows
            },
        }
        for label, values in checks.items():
            if len(values) != 1:
                raise RuntimeError(
                    f"{label} differs across K={component_count} variants "
                    f"for seed {seed_index}."
                )


def _variant_config(
    source: Path, target: Path, variant: str, protocol_components: int
) -> None:
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Development base config must be a mapping.")
    payload = dict(payload)
    payload["version"] = CONFIG_VERSION
    payload["method_variant"] = variant
    payload["model"] = dict(payload["model"])
    payload["model"]["protocol_components"] = int(protocol_components)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def run_development_matrix(args: argparse.Namespace) -> None:
    source_config = Path(args.config).resolve()
    manifest = Path(args.partner_manifest).resolve()
    output = Path(args.output).resolve()
    seeds = tuple(int(value) for value in args.seed_index)
    requested_components = getattr(args, "protocol_components", None)
    component_counts = tuple(
        int(value)
        for value in (
            (2, 4, 8) if requested_components is None else requested_components
        )
    )
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Development matrix seed indexes must be unique and non-empty.")
    if set(component_counts) != {2, 4, 8} or len(component_counts) != 3:
        raise ValueError("Development sensitivity must cover K in {2,4,8} exactly once.")
    entries = []
    for component_count in component_counts:
        for variant in DEVELOPMENT_VARIANTS:
            variant_config = (
                output / "configs" / f"k-{component_count}" / f"{variant}.yaml"
            )
            _variant_config(
                source_config, variant_config, variant, component_count
            )
            for seed_index in seeds:
                run_directory = (
                    output
                    / "runs"
                    / f"k-{component_count}"
                    / variant
                    / f"seed-{seed_index}"
                )
                run_training(
                    SimpleNamespace(
                        config=str(variant_config),
                        partner_manifest=str(manifest),
                        ego_run_id=(
                            f"depi-k{component_count}-{variant}-development-"
                            f"seed-{seed_index}"
                        ),
                        seed_index=seed_index,
                        run_kind="development",
                        output=str(run_directory),
                        resume=bool(args.resume),
                        skip_manifest_hash_check=bool(args.skip_manifest_hash_check),
                        _execution_scope="development-matrix",
                    )
                )
                identity = json.loads(
                    (run_directory / "run_identity.json").read_text(encoding="utf-8")
                )
                ledger = ResourceLedger.from_mapping(
                    json.loads(
                        (run_directory / "resource_ledger.json").read_text(
                            encoding="utf-8"
                        )
                    )
                )
                pool_path = run_directory / "partner_pool.json"
                entries.append(
                    {
                        "variant": variant,
                        "protocol_components": component_count,
                        "seed_index": seed_index,
                        "run": str(run_directory),
                        "run_identity_sha256": sha256_path(
                            run_directory / "run_identity.json"
                        ),
                        "config_fingerprint": identity["config_fingerprint"],
                        "partner_sampler_sha256": sha256_path(pool_path),
                        "total_training_simulator_steps": (
                            ledger.total_training_simulator_steps
                        ),
                        "deployable_parameters": ledger.deployable_parameters,
                        "gpu_hours": ledger.gpu_hours,
                        "episode_key_domains": {
                            name: list(value)
                            for name, value in official_training_domain_keys(
                                seed_index
                            ).items()
                        },
                    }
                )
    validate_development_entry_alignment(entries)
    result = {
        "version": 1,
        "artifact_type": "depi_development_matrix",
        "method": METHOD_VERSION,
        "source_config": {
            "path": str(source_config),
            "sha256": sha256_path(source_config),
        },
        "partner_manifest": {
            "path": str(manifest),
            "sha256": sha256_path(manifest),
        },
        "variants": {
            "b0": "full_history_recurrent_ppo",
            "b1": "isolated_protocol_architecture",
            "b2": "decision_supervision",
            "b3": "not_implemented_action_conditioned_voi",
        },
        "b3_status": "not_implemented",
        "protocol_component_sensitivity": list(component_counts),
        "entries": entries,
        "budget_capacity_and_key_matching_passed": True,
    }
    ensure_run_identity(
        output,
        {
            "stage": "run-development-matrix",
            "method": METHOD_VERSION,
            "repository_runtime": runtime_provenance(),
            "source_config": result["source_config"],
            "partner_manifest": result["partner_manifest"],
            "seeds": list(seeds),
            "protocol_component_sensitivity": list(component_counts),
        },
    )
    write_json(output / "development_matrix.json", result)


def _paired_bootstrap(values: np.ndarray, *, seed: int) -> tuple[float, list[float]]:
    if values.ndim != 1 or values.size < 2:
        raise ValueError("Paired development inference needs at least two seeds.")
    rng = np.random.default_rng(int(seed))
    indexes = rng.integers(0, values.size, size=(9_999, values.size))
    draws = values[indexes].mean(axis=1)
    interval = np.quantile(draws, (0.005, 0.995))
    return float(np.mean(values)), [float(interval[0]), float(interval[1])]


def summarize_development_matrix(args: argparse.Namespace) -> None:
    matrix_path = Path(args.matrix).resolve()
    score_path = Path(args.scores).resolve()
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    scores = json.loads(score_path.read_text(encoding="utf-8"))
    if matrix.get("artifact_type") != "depi_development_matrix":
        raise ValueError("Development matrix artifact type differs.")
    if not isinstance(scores, Mapping) or set(scores) != {
        "version",
        "artifact_type",
        "rows",
    }:
        raise ValueError("Development score artifact fields differ.")
    if scores["artifact_type"] != "depi_development_scores" or int(scores["version"]) != 1:
        raise ValueError("Development score artifact type/version differs.")
    required_score_fields = {
        "variant",
        "protocol_components",
        "seed_index",
        "xp_mean",
        "evaluation_key_schedule_sha256",
    }
    score_index = {}
    for row in scores["rows"]:
        if not isinstance(row, Mapping) or set(row) != required_score_fields:
            raise ValueError("Development score row fields differ.")
        key = (
            int(row["protocol_components"]),
            str(row["variant"]),
            int(row["seed_index"]),
        )
        if key in score_index or key[1] not in DEVELOPMENT_VARIANTS:
            raise ValueError("Development score row is duplicate or unknown.")
        score_index[key] = row
    expected = {
        (
            int(row["protocol_components"]),
            str(row["variant"]),
            int(row["seed_index"]),
        )
        for row in matrix["entries"]
    }
    if set(score_index) != expected:
        raise ValueError("Development scores do not cover the complete matrix.")
    seeds = sorted({seed for _, _, seed in expected})
    component_counts = sorted({count for count, _, _ in expected})
    for component_count in component_counts:
        for seed in seeds:
            schedules = {
                score_index[(component_count, variant, seed)][
                    "evaluation_key_schedule_sha256"
                ]
                for variant in DEVELOPMENT_VARIANTS
            }
            if len(schedules) != 1:
                raise ValueError("Paired variants do not share evaluation episode keys.")
    values_by_k = {
        component_count: {
            variant: np.asarray(
                [
                    float(
                        score_index[(component_count, variant, seed)]["xp_mean"]
                    )
                    for seed in seeds
                ],
                dtype=np.float64,
            )
            for variant in DEVELOPMENT_VARIANTS
        }
        for component_count in component_counts
    }
    comparisons_by_k = {}
    for component_count, values in values_by_k.items():
        comparisons = {}
        for name, left, right, seed in (
            ("b1_minus_b0", "b1", "b0", 10),
            ("b2_minus_b1", "b2", "b1", 11),
            ("b2_minus_b0", "b2", "b0", 12),
        ):
            point, interval = _paired_bootstrap(
                values[left] - values[right], seed=seed + 100 * component_count
            )
            comparisons[name] = {
                "paired_mean_increment": point,
                "bootstrap_99_percent_ci": interval,
            }
        comparisons_by_k[str(component_count)] = comparisons
    entries = matrix["entries"]
    compute = {
        variant: {
            "total_training_simulator_steps": int(
                sum(
                    row["total_training_simulator_steps"]
                    for row in entries
                    if row["variant"] == variant
                )
            ),
            "gpu_hours": float(
                sum(row["gpu_hours"] for row in entries if row["variant"] == variant)
            ),
        }
        for variant in DEVELOPMENT_VARIANTS
    }
    result = {
        "version": 1,
        "artifact_type": "depi_development_matrix_summary",
        "method": METHOD_VERSION,
        "paired_seed_count": len(seeds),
        "xp_by_protocol_components": {
            str(component_count): {
                variant: {
                    "mean": float(np.mean(value)),
                    "per_seed": value.tolist(),
                }
                for variant, value in values.items()
            }
            for component_count, values in values_by_k.items()
        },
        "paired_increments_by_protocol_components": comparisons_by_k,
        "primary_k4_paired_increments": comparisons_by_k["4"],
        "total_compute": compute,
        "b3_status": "not_implemented",
        "sources": {
            "matrix": {"path": str(matrix_path), "sha256": sha256_path(matrix_path)},
            "scores": {"path": str(score_path), "sha256": sha256_path(score_path)},
        },
    }
    output = Path(args.output).resolve()
    ensure_run_identity(
        output,
        {
            "stage": "summarize-development-matrix",
            "method": METHOD_VERSION,
            "repository_runtime": runtime_provenance(),
            "sources": result["sources"],
        },
    )
    write_json(output / "development_matrix_summary.json", result)


__all__ = [
    "DEVELOPMENT_VARIANTS",
    "run_development_matrix",
    "summarize_development_matrix",
    "validate_development_entry_alignment",
]
