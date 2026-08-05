"""Train Official OvercookedV2 baselines and export auditable manifests.

This module is deliberately outside the DELTA algorithm package.  It launches
SP, state-augmented, OP, FCP, and the parameter-matched IPPO-Large control
through the pinned Official training entrypoint, then emits the same immutable
policy-manifest and resource-ledger interfaces consumed by DELTA evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Mapping

from .official_adapter import (
    OFFICIAL_BASELINE_EXPERIMENTS,
    OfficialNetwork,
    compose_ippo_large_config,
    compose_official_baseline_config,
    initialize_official_parameters,
    official_policy,
    restore_official_checkpoint,
    validate_official_runtime,
)
from src.delta_zsc.config import (
    OFFICIAL_OP_TOTAL_TIMESTEPS,
    OFFICIAL_SOURCE_COMMIT,
    OFFICIAL_SP_TOTAL_TIMESTEPS,
    OFFICIAL_TRAINING_RUN_COUNT,
)
from src.delta_zsc.resources import (
    ResourceLedger,
    gpu_device_count,
    gpu_hours_for_wall_seconds,
    measure_policy_inference_latency_ms,
    parameter_count,
    peak_device_memory_bytes,
)
from src.delta_zsc.storage import (
    ensure_run_identity,
    read_json,
    sha256_path,
    write_json,
)


BASELINE_METHODS = ("sp", "state-augmented", "op", "fcp", "ippo-large")

def _runtime_identity() -> Mapping[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
    }


def _read_resource_ledger(path: str | Path | None, *, label: str) -> ResourceLedger:
    if path is None:
        raise ValueError(f"{label} requires an explicit resource ledger.")
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return ResourceLedger.from_mapping(read_json(source))


def _read_training_lineage(path: str | Path | None) -> list[Mapping[str, Any]]:
    if path is None:
        return []
    source = Path(path).resolve()
    payload = read_json(source)
    if not isinstance(payload, list):
        raise ValueError("Training-lineage manifest must be a JSON list.")
    expected = {
        "checkpoint",
        "checkpoint_sha256",
        "parent_training_run_id",
        "co_training_group_id",
        "role",
        "jax_prng_key",
    }
    rows: list[Mapping[str, Any]] = []
    for index, raw in enumerate(payload):
        if not isinstance(raw, Mapping) or set(raw) != expected:
            raise ValueError(f"Training-lineage row {index} fields differ.")
        key = raw["jax_prng_key"]
        if not isinstance(key, list) or len(key) != 2:
            raise ValueError(f"Training-lineage row {index} has a malformed JAX key.")
        checkpoint = Path(str(raw["checkpoint"])).resolve()
        if not checkpoint.exists() or sha256_path(checkpoint) != raw["checkpoint_sha256"]:
            raise ValueError(f"Training-lineage checkpoint/hash differs at row {index}.")
        rows.append({**dict(raw), "checkpoint": str(checkpoint)})
    return rows


def _validate_fcp_population(
    *,
    root: Path,
    layout: str,
    lineage: list[Mapping[str, Any]],
    ledger: ResourceLedger,
) -> None:
    population_dirs = tuple(
        sorted(path for path in root.iterdir() if path.is_dir() and "fcp_" in path.name)
    )
    if len(population_dirs) != OFFICIAL_TRAINING_RUN_COUNT:
        raise ValueError("Official FCP requires exactly ten population directories.")
    checkpoints: list[Path] = []
    for population in population_dirs:
        runs = tuple(
            sorted(
                path
                for path in population.iterdir()
                if path.is_dir() and path.name.startswith("run_")
            )
        )
        if {path.name for path in runs} != {f"run_{index}" for index in range(8)}:
            raise ValueError(f"FCP population {population.name} must contain eight SP parents.")
        for run in runs:
            current = tuple(
                sorted(
                    path
                    for path in run.iterdir()
                    if path.is_dir() and path.name.startswith("ckpt_")
                )
            )
            if {path.name for path in current} != {"ckpt_0", "ckpt_1", "ckpt_final"}:
                raise ValueError(f"FCP parent {run} must preserve three checkpoints.")
            checkpoints.extend(current)
    expected_hashes = {sha256_path(path) for path in checkpoints}
    population_rows = [row for row in lineage if row["role"] == "fcp_population_checkpoint"]
    if len(population_rows) != 240 or {
        str(row["checkpoint_sha256"]) for row in population_rows
    } != expected_hashes:
        raise ValueError("FCP lineage must bind all 10x8x3 population checkpoints.")
    parent_ids = {str(row["parent_training_run_id"]) for row in population_rows}
    if len(parent_ids) != 80:
        raise ValueError("FCP population must contain 80 independent SP parent runs.")
    minimum = 80 * OFFICIAL_SP_TOTAL_TIMESTEPS
    if ledger.total_training_simulator_steps < minimum:
        raise ValueError("FCP population ledger omits Official SP parent training cost.")
    checkpoint_config, _ = restore_official_checkpoint(checkpoints[0])
    if str(checkpoint_config["env"]["ENV_KWARGS"]["layout"]) != layout:
        raise ValueError("FCP population layout differs from the requested layout.")


def _official_command(
    *,
    method: str,
    layout: str,
    output: Path,
    fcp_population: Path | None,
    ippo_large_dimension: int | None,
) -> list[str]:
    experiment = (
        "rnn-sp" if method == "ippo-large" else OFFICIAL_BASELINE_EXPERIMENTS[method]
    )
    command = [
        sys.executable,
        "-m",
        "overcooked_v2_experiments.ppo.main",
        f"+experiment={experiment}",
        f"+env={layout}",
        "SEED=42",
        "NUM_CHECKPOINTS=1",
        "VISUALIZE=false",
        "TUNE=false",
        "wandb.WANDB_MODE=disabled",
        f"hydra.run.dir={output / 'official_hydra'}",
        "hydra.job.chdir=true",
    ]
    if method == "fcp":
        if fcp_population is None:
            raise ValueError("FCP requires --fcp-population.")
        command.append(f"+FCP={fcp_population}")
    else:
        command.append(f"NUM_SEEDS={OFFICIAL_TRAINING_RUN_COUNT}")
    if method == "ippo-large":
        if ippo_large_dimension is None:
            raise ValueError("IPPO-Large requires a selected hidden dimension.")
        command.extend(
            (
                f"model.FC_DIM_SIZE={int(ippo_large_dimension)}",
                f"model.GRU_HIDDEN_DIM={int(ippo_large_dimension)}",
            )
        )
    return command


def _deployment_parameter_count(path: str | Path) -> tuple[int, tuple[int, ...]]:
    import pickle

    root = Path(path).resolve()
    bundle = read_json(root / "deployment_bundle.json")
    params = pickle.loads((root / "parameters.pkl").read_bytes())
    return parameter_count(params), tuple(int(value) for value in bundle["observation_shape"])


def _select_ippo_large_dimension(
    *, layout: str, target_parameters: int, observation_shape: tuple[int, ...]
) -> Mapping[str, int]:
    import jax

    target = int(target_parameters)
    if target <= 0:
        raise ValueError("Target parameter count must be positive.")
    cache: dict[int, int] = {}

    def count(dimension: int) -> int:
        dimension = int(dimension)
        if dimension not in cache:
            config = compose_ippo_large_config(layout=layout, hidden_dimension=dimension)
            params = initialize_official_parameters(
                OfficialNetwork(config),
                random_key=jax.random.PRNGKey(0),
                observation_shape=observation_shape,
                batch_size=1,
            )
            cache[dimension] = parameter_count(params)
        return cache[dimension]

    lower, upper = 1, 128
    while count(upper) < target:
        lower, upper = upper, upper * 2
        if upper > 8192:
            raise RuntimeError("IPPO-Large width search exceeded 8192.")
    while upper - lower > 1:
        middle = (lower + upper) // 2
        if count(middle) < target:
            lower = middle
        else:
            upper = middle
    selected = min((lower, upper), key=lambda value: (abs(count(value) - target), value))
    observed = count(selected)
    return {
        "hidden_dimension": int(selected),
        "target_parameters": target,
        "observed_parameters": int(observed),
        "absolute_mismatch": abs(int(observed) - target),
    }


def _baseline_ledger(
    *,
    method: str,
    training_wall_seconds: float,
    inference_measurement_seconds: float,
    deployable_parameters: int,
    inference_latency_ms: float,
    shared_infrastructure: ResourceLedger | None,
) -> ResourceLedger:
    per_run = (
        OFFICIAL_OP_TOTAL_TIMESTEPS if method == "op" else OFFICIAL_SP_TOTAL_TIMESTEPS
    )
    shared_steps = 0
    shared_gpu = 0.0
    shared_wall = 0.0
    training_only_parameters = 0
    if shared_infrastructure is not None:
        shared_steps += shared_infrastructure.total_training_simulator_steps
        shared_gpu += shared_infrastructure.gpu_hours
        shared_wall += shared_infrastructure.wall_clock_hours
        training_only_parameters += (
            shared_infrastructure.deployable_parameters
            + shared_infrastructure.training_only_parameters
        )
    return ResourceLedger(
        ego_policy_steps=OFFICIAL_TRAINING_RUN_COUNT * int(per_run),
        upstream_partner_steps=int(shared_steps),
        training_gpu_hours=gpu_hours_for_wall_seconds(training_wall_seconds),
        shared_gpu_hours=float(shared_gpu),
        measurement_gpu_hours=gpu_hours_for_wall_seconds(inference_measurement_seconds),
        training_wall_clock_hours=float(training_wall_seconds) / 3600.0,
        shared_wall_clock_hours=float(shared_wall),
        measurement_wall_clock_hours=float(inference_measurement_seconds) / 3600.0,
        peak_memory_bytes=peak_device_memory_bytes(),
        deployable_parameters=int(deployable_parameters),
        training_only_parameters=int(training_only_parameters),
        inference_latency_ms=float(inference_latency_ms),
    )


def run_official_baseline(args: argparse.Namespace) -> None:
    method = str(args.method)
    layout = str(args.layout)
    if method not in BASELINE_METHODS:
        raise ValueError(f"Unknown baseline method: {method}")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    runtime = validate_official_runtime()
    fcp_population = (
        None if args.fcp_population is None else Path(args.fcp_population).resolve()
    )
    capacity_match = None
    if method == "ippo-large":
        if args.delta_deployment is None:
            raise ValueError("IPPO-Large requires --delta-deployment.")
        target_count, observation_shape = _deployment_parameter_count(args.delta_deployment)
        capacity_match = _select_ippo_large_dimension(
            layout=layout,
            target_parameters=target_count,
            observation_shape=observation_shape,
        )
        resolved = compose_ippo_large_config(
            layout=layout,
            hidden_dimension=capacity_match["hidden_dimension"],
        )
    else:
        if args.delta_deployment is not None:
            raise ValueError("--delta-deployment is only valid for IPPO-Large.")
        resolved = compose_official_baseline_config(
            layout=layout,
            method=method,
            fcp_population=fcp_population,
        )

    population_ledger = (
        _read_resource_ledger(args.fcp_population_ledger, label="FCP population")
        if method == "fcp"
        else None
    )
    state_augmented_ledger = (
        _read_resource_ledger(
            args.shared_training_ledger, label="State-augmented shared training"
        )
        if method == "state-augmented"
        else None
    )
    if method not in {"state-augmented"} and args.shared_training_ledger is not None:
        raise ValueError("--shared-training-ledger is only valid for state-augmented.")
    lineage = _read_training_lineage(args.training_lineage_manifest)
    if method == "fcp":
        if fcp_population is None or population_ledger is None or not lineage:
            raise ValueError("FCP requires population directory, ledger, and lineage.")
        _validate_fcp_population(
            root=fcp_population,
            layout=layout,
            lineage=lineage,
            ledger=population_ledger,
        )

    identity = {
        "stage": "official-baseline-train",
        "method": method,
        "layout": layout,
        "official_source_commit": OFFICIAL_SOURCE_COMMIT,
        "official_runtime": runtime,
        "runtime": _runtime_identity(),
        "resolved_official_config": resolved,
        "fcp_population": (
            None
            if fcp_population is None
            else {"path": str(fcp_population), "sha256": sha256_path(fcp_population)}
        ),
        "external_training_lineage": lineage,
        "capacity_match": capacity_match,
    }
    ensure_run_identity(output, identity)
    write_json(output / "official_config.json", resolved)
    command = _official_command(
        method=method,
        layout=layout,
        output=output,
        fcp_population=fcp_population,
        ippo_large_dimension=(
            None if capacity_match is None else capacity_match["hidden_dimension"]
        ),
    )
    write_json(output / "official_command.json", {"argv": command})
    started = time.perf_counter()
    subprocess.run(command, cwd=output, check=True)
    training_wall_seconds = time.perf_counter() - started

    checkpoints = tuple(sorted(output.rglob("ckpt_final")))
    if len(checkpoints) != OFFICIAL_TRAINING_RUN_COUNT:
        raise RuntimeError(
            f"Official {method} produced {len(checkpoints)} final checkpoints; expected 10."
        )
    counts = []
    runs = []
    for index, checkpoint in enumerate(checkpoints):
        _, params = restore_official_checkpoint(checkpoint)
        counts.append(parameter_count(params))
        runs.append(
            {
                "run_index": index,
                "run_id": f"{method}-{layout}-{index:02d}",
                "policy": str(checkpoint),
                "policy_sha256": sha256_path(checkpoint),
                "identity": {
                    "parent_training_run_id": f"{method}-{layout}-{index:02d}",
                    "co_training_group_id": None,
                },
            }
        )
    if len(set(counts)) != 1:
        raise RuntimeError("Official final checkpoints have inconsistent parameter counts.")
    if capacity_match is not None and counts[0] != capacity_match["observed_parameters"]:
        raise RuntimeError("Trained IPPO-Large size differs from the frozen capacity match.")

    latency_config, latency_params = restore_official_checkpoint(checkpoints[0])
    import jax
    import jaxmarl
    from jaxmarl.environments.overcooked_v2.overcooked import ObservationType

    latency_environment = jaxmarl.make(
        latency_config["env"]["ENV_NAME"],
        **dict(latency_config["env"]["ENV_KWARGS"]),
        observation_type=ObservationType.DEFAULT,
    )
    latency_observations, _ = latency_environment.reset(jax.random.PRNGKey(0))
    latency_started = time.perf_counter()
    latency_ms = measure_policy_inference_latency_ms(
        official_policy(latency_params, latency_config),
        latency_observations["agent_0"],
    )
    latency_seconds = time.perf_counter() - latency_started
    ledger = _baseline_ledger(
        method=method,
        training_wall_seconds=training_wall_seconds,
        inference_measurement_seconds=latency_seconds,
        deployable_parameters=counts[0],
        inference_latency_ms=latency_ms,
        shared_infrastructure=(
            population_ledger if method == "fcp" else state_augmented_ledger
        ),
    )

    per_run_steps = (
        OFFICIAL_OP_TOTAL_TIMESTEPS if method == "op" else OFFICIAL_SP_TOTAL_TIMESTEPS
    )
    run_ledger_root = output / "resource_ledgers"
    run_ledger_root.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(runs):
        run_ledger = ResourceLedger(
            ego_policy_steps=int(per_run_steps),
            upstream_partner_steps=ledger.upstream_partner_steps,
            training_gpu_hours=ledger.training_gpu_hours / OFFICIAL_TRAINING_RUN_COUNT,
            shared_gpu_hours=ledger.shared_gpu_hours,
            measurement_gpu_hours=(ledger.measurement_gpu_hours if index == 0 else 0.0),
            training_wall_clock_hours=(
                ledger.training_wall_clock_hours / OFFICIAL_TRAINING_RUN_COUNT
            ),
            shared_wall_clock_hours=ledger.shared_wall_clock_hours,
            measurement_wall_clock_hours=(
                ledger.measurement_wall_clock_hours if index == 0 else 0.0
            ),
            peak_memory_bytes=ledger.peak_memory_bytes,
            deployable_parameters=ledger.deployable_parameters,
            training_only_parameters=ledger.training_only_parameters,
            inference_latency_ms=ledger.inference_latency_ms,
        )
        ledger_path = run_ledger_root / f"run-{index:02d}.json"
        write_json(ledger_path, run_ledger.to_mapping())
        row["resource_ledger"] = {
            "path": str(ledger_path),
            "sha256": sha256_path(ledger_path),
        }

    training_lineage = [
        *(
            {
                "checkpoint_sha256": row["checkpoint_sha256"],
                "parent_training_run_id": row["parent_training_run_id"],
                "co_training_group_id": row["co_training_group_id"],
                "role": row["role"],
            }
            for row in lineage
        ),
        *(
            {
                "checkpoint_sha256": row["policy_sha256"],
                "parent_training_run_id": row["identity"]["parent_training_run_id"],
                "co_training_group_id": None,
                "role": "formal_ego",
            }
            for row in runs
        ),
    ]
    write_json(
        output / "policy_manifest.json",
        {
            "version": 1,
            "method": method,
            "layout": layout,
            "policy_kind": "official_checkpoint",
            "runs": runs,
            "training_lineage": training_lineage,
            "official_source_commit": OFFICIAL_SOURCE_COMMIT,
            **({"capacity_match": capacity_match} if capacity_match is not None else {}),
        },
    )
    write_json(output / "resource_ledger.json", ledger.to_mapping())


__all__ = ["BASELINE_METHODS", "run_official_baseline"]
