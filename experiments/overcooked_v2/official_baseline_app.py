"""Launch the four baselines through the fixed Official training entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

from experiments.overcooked_v2.official_adapter import (
    OFFICIAL_BASELINE_EXPERIMENTS,
    OfficialNetwork,
    compose_official_baseline_config,
    compose_ippo_large_config,
    initialize_official_parameters,
    official_policy,
    restore_official_checkpoint,
    validate_official_runtime,
)
from src.path_c.experiment import OFFICIAL_SOURCE_COMMIT
from src.path_c.resources import (
    ResourceLedger,
    gpu_device_count,
    gpu_hours_for_wall_seconds,
    measure_policy_inference_latency_ms,
    parameter_count,
)
from src.path_c.storage import (
    ensure_run_identity,
    runtime_provenance,
    sha256_path,
    validate_formal_repository_state,
    validate_registered_python_runtime,
    write_json,
)


def _read_population_ledger(path: str | Path | None) -> ResourceLedger:
    if path is None:
        raise ValueError(
            "Official FCP requires the complete population resource ledger; "
            "population formation cost cannot be omitted."
        )
    payload = json.loads(Path(path).resolve().read_text(encoding="utf-8"))
    required = {
        "ego_policy_steps",
        "partner_training_steps",
        "counterfactual_steps",
        "state_collection_steps",
        "calibration_steps",
        "evaluation_steps",
        "gpu_hours",
        "peak_memory_bytes",
        "deployable_parameters",
        "training_only_parameters",
        "inference_latency_ms",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"FCP population ledger is incomplete: {sorted(missing)}")
    return ResourceLedger(**{name: payload[name] for name in required})


def _read_training_lineage(path: str | Path | None) -> list[Mapping[str, Any]]:
    if path is None:
        return []
    payload = json.loads(Path(path).resolve().read_text(encoding="utf-8"))
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
    rows = []
    for index, raw in enumerate(payload):
        if not isinstance(raw, Mapping) or set(raw) != expected:
            raise ValueError(f"Training-lineage row {index} fields differ.")
        key = raw["jax_prng_key"]
        if not isinstance(key, list) or len(key) != 2:
            raise ValueError(f"Training-lineage row {index} has a malformed PRNG key.")
        checkpoint = Path(str(raw["checkpoint"])).resolve()
        if not checkpoint.exists() or sha256_path(checkpoint) != raw["checkpoint_sha256"]:
            raise ValueError(f"Training-lineage checkpoint/hash differs at row {index}.")
        rows.append({**dict(raw), "checkpoint": str(checkpoint)})
    return rows


def _validate_fcp_population(
    *, root: Path, layout: str, lineage: list[Mapping[str, Any]], ledger: ResourceLedger
) -> None:
    population_dirs = tuple(
        sorted(path for path in root.iterdir() if path.is_dir() and "fcp_" in path.name)
    )
    if len(population_dirs) != 10:
        raise ValueError("Official FCP requires exactly ten population subdirectories.")
    checkpoints = []
    for population in population_dirs:
        run_dirs = tuple(sorted(path for path in population.iterdir() if path.is_dir() and path.name.startswith("run_")))
        if {path.name for path in run_dirs} != {f"run_{index}" for index in range(8)}:
            raise ValueError(f"FCP population {population.name} must have eight SP parents.")
        for run_dir in run_dirs:
            current = tuple(
                sorted(path for path in run_dir.iterdir() if path.is_dir() and path.name.startswith("ckpt_"))
            )
            if {path.name for path in current} != {"ckpt_0", "ckpt_1", "ckpt_final"}:
                raise ValueError(
                    f"FCP parent {run_dir} must preserve the fixed three checkpoints."
                )
            checkpoints.extend(current)
    expected_hashes = {sha256_path(path) for path in checkpoints}
    population_rows = [row for row in lineage if row["role"] == "fcp_population_checkpoint"]
    if len(population_rows) != 240 or {
        str(row["checkpoint_sha256"]) for row in population_rows
    } != expected_hashes:
        raise ValueError("FCP population lineage must bind all 10x8x3 checkpoints.")
    parents: dict[str, list[Mapping[str, Any]]] = {}
    for row in population_rows:
        parents.setdefault(str(row["parent_training_run_id"]), []).append(row)
    if len(parents) != 80 or {len(rows) for rows in parents.values()} != {3}:
        raise ValueError("FCP lineage must contain 80 independent SP parents with three checkpoints each.")
    keys = set()
    for parent, rows in parents.items():
        parent_keys = {tuple(int(value) for value in row["jax_prng_key"]) for row in rows}
        if len(parent_keys) != 1:
            raise ValueError(f"FCP parent {parent} has inconsistent PRNG provenance.")
        keys.update(parent_keys)
    if len(keys) != 80:
        raise ValueError("FCP SP parent runs must use 80 distinct PRNG keys.")
    minimum_population_steps = 80 * 29_949_952
    if ledger.total_training_simulator_steps < minimum_population_steps:
        raise ValueError(
            "FCP population ledger omits the 80 Official SP parent training costs."
        )
    # Layout/config validation is performed by the fixed loader at training;
    # bind one checkpoint early to reject an obviously different population.
    checkpoint_config, unused = restore_official_checkpoint(checkpoints[0])
    del unused
    observed_layout = checkpoint_config["env"]["ENV_KWARGS"]["layout"]
    if str(observed_layout) != layout:
        raise ValueError("FCP population layout differs from its formal ego layout.")


def _official_command(
    *,
    method: str,
    layout: str,
    output: Path,
    fcp_population: Path | None,
    ippo_large_dimension: int | None = None,
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
        assert fcp_population is not None
        command.append(f"+FCP={fcp_population}")
    else:
        command.append("NUM_SEEDS=10")
    if method == "ippo-large":
        if ippo_large_dimension is None:
            raise ValueError("IPPO-Large requires its mechanically selected width.")
        command.extend(
            (
                f"model.FC_DIM_SIZE={int(ippo_large_dimension)}",
                f"model.GRU_HIDDEN_DIM={int(ippo_large_dimension)}",
            )
        )
    return command


def _deployment_parameter_count(path: str | Path) -> tuple[int, tuple[int, ...]]:
    import orbax.checkpoint as ocp

    root = Path(path).resolve()
    bundle = json.loads((root / "deployment_bundle.json").read_text(encoding="utf-8"))
    params = ocp.PyTreeCheckpointer().restore(str(root / "params"))
    return parameter_count(params), tuple(int(value) for value in bundle["observation_shape"])


def _select_ippo_large_dimension(
    *, layout: str, target_parameters: int, observation_shape: tuple[int, ...]
) -> Mapping[str, int]:
    """Select one shared FC/GRU width by deterministic nearest-count search."""

    import jax

    target = int(target_parameters)
    if target <= 0:
        raise ValueError("Target deployment parameter count must be positive.")

    cache: dict[int, int] = {}

    def count(dimension: int) -> int:
        dimension = int(dimension)
        if dimension not in cache:
            config = compose_ippo_large_config(
                layout=layout, hidden_dimension=dimension
            )
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
            raise RuntimeError("IPPO-Large width search exceeded the registered bound.")
    while upper - lower > 1:
        middle = (lower + upper) // 2
        if count(middle) < target:
            lower = middle
        else:
            upper = middle
    selected = min((lower, upper), key=lambda value: (abs(count(value) - target), value))
    return {
        "hidden_dimension": int(selected),
        "target_parameters": target,
        "observed_parameters": int(count(selected)),
        "absolute_mismatch": abs(int(count(selected)) - target),
    }


def _peak_device_memory() -> int:
    try:
        import jax

        values = []
        for device in jax.devices():
            stats = device.memory_stats() or {}
            for name in ("peak_bytes_in_use", "peak_pool_bytes", "bytes_in_use"):
                if name in stats:
                    values.append(int(stats[name]))
        return max(values, default=0)
    except Exception:
        return 0


def _baseline_ledger(
    *,
    method: str,
    population: ResourceLedger | None,
    wall_seconds: float,
    inference_latency_ms: float,
) -> ResourceLedger:
    per_run = 49_987_584 if method == "op" else 29_949_952
    base = ResourceLedger(
        ego_policy_steps=10 * per_run,
        state_collection_steps=(4_000_000 if method == "state-augmented" else 0),
        gpu_hours=gpu_hours_for_wall_seconds(
            wall_seconds, device_count=gpu_device_count()
        ),
        peak_memory_bytes=_peak_device_memory(),
        inference_latency_ms=float(inference_latency_ms),
    )
    if population is None:
        return base
    return base.plus(
        partner_training_steps=population.total_training_simulator_steps,
        gpu_hours=population.gpu_hours,
        peak_memory_bytes=max(0, population.peak_memory_bytes - base.peak_memory_bytes),
        training_only_parameters=population.deployable_parameters
        + population.training_only_parameters,
    )


def run_official_baseline(args: argparse.Namespace) -> None:
    validate_formal_repository_state()
    validate_registered_python_runtime()
    method = str(args.method)
    layout = str(args.layout)
    output = Path(args.output).resolve()
    fcp_population = (
        None if args.fcp_population is None else Path(args.fcp_population).resolve()
    )
    runtime = validate_official_runtime()
    capacity_match = None
    if method == "ippo-large":
        if args.delta_deployment is None:
            raise ValueError("IPPO-Large requires --delta-deployment.")
        target_count, observation_shape = _deployment_parameter_count(
            args.delta_deployment
        )
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
            layout=layout, method=method, fcp_population=fcp_population
        )
    population_ledger = (
        _read_population_ledger(args.fcp_population_ledger)
        if method == "fcp"
        else None
    )
    external_lineage = _read_training_lineage(args.training_lineage_manifest)
    if method == "fcp" and not external_lineage:
        raise ValueError(
            "Official FCP requires population lineage as well as its resource ledger."
        )
    if method == "fcp":
        assert fcp_population is not None and population_ledger is not None
        _validate_fcp_population(
            root=fcp_population,
            layout=layout,
            lineage=external_lineage,
            ledger=population_ledger,
        )
    identity: Mapping[str, Any] = {
        "stage": "official-baseline-train",
        "method": method,
        "layout": layout,
        "official_source_commit": OFFICIAL_SOURCE_COMMIT,
        "official_runtime": runtime,
        "repository_runtime": runtime_provenance(),
        "resolved_official_config": resolved,
        "fcp_population": (
            None
            if fcp_population is None
            else {"path": str(fcp_population), "sha256": sha256_path(fcp_population)}
        ),
        "fcp_population_ledger": (
            None if population_ledger is None else population_ledger.to_mapping()
        ),
        "external_training_lineage": external_lineage,
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
    wall_seconds = time.perf_counter() - started

    checkpoints = tuple(sorted(output.rglob("ckpt_final")))
    if len(checkpoints) != 10:
        raise RuntimeError(
            f"Official {method} produced {len(checkpoints)} final checkpoints; expected 10."
        )
    runs = []
    deployable_counts = []
    for index, checkpoint in enumerate(checkpoints):
        unused_config, params = restore_official_checkpoint(checkpoint)
        del unused_config
        deployable_counts.append(parameter_count(params))
        runs.append(
            {
                "run_index": index,
                "run_id": f"{method}-{layout}-{index:02d}",
                "parent_training_run_id": f"{method}-{layout}-{index:02d}",
                "co_training_group_id": None,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_path(checkpoint),
            }
        )
    if len(set(deployable_counts)) != 1:
        raise RuntimeError("Official final checkpoints have inconsistent parameter counts.")
    latency_config, latency_params = restore_official_checkpoint(checkpoints[0])
    import jax
    from jaxmarl.environments.overcooked_v2.overcooked import ObservationType
    import jaxmarl

    latency_environment = jaxmarl.make(
        latency_config["env"]["ENV_NAME"],
        **dict(latency_config["env"]["ENV_KWARGS"]),
        observation_type=[ObservationType.DEFAULT, ObservationType.DEFAULT],
    )
    latency_observations, unused_latency_state = latency_environment.reset(
        jax.random.PRNGKey(0)
    )
    del unused_latency_state
    inference_latency_ms = measure_policy_inference_latency_ms(
        official_policy(latency_params, latency_config),
        latency_observations["agent_0"],
    )
    ledger = _baseline_ledger(
        method=method,
        population=population_ledger,
        wall_seconds=wall_seconds,
        inference_latency_ms=inference_latency_ms,
    )
    ledger = ledger.plus(deployable_parameters=deployable_counts[0])
    if capacity_match is not None and deployable_counts[0] != capacity_match["observed_parameters"]:
        raise RuntimeError("Trained IPPO-Large parameter count differs from its frozen match.")
    write_json(
        output / "policy_manifest.json",
        {
            "version": 1,
            "layout": layout,
            "method": method,
            "policy_kind": "official_ppo",
            "official_source_commit": OFFICIAL_SOURCE_COMMIT,
            "runs": runs,
            "training_lineage": [
                *(
                    {
                        "checkpoint_sha256": row["checkpoint_sha256"],
                        "parent_training_run_id": row["parent_training_run_id"],
                        "co_training_group_id": row["co_training_group_id"],
                        "role": row["role"],
                    }
                    for row in external_lineage
                ),
                *(
                    {
                        "checkpoint_sha256": run["checkpoint_sha256"],
                        "parent_training_run_id": run["parent_training_run_id"],
                        "co_training_group_id": run["co_training_group_id"],
                        "role": "formal_ego",
                    }
                    for run in runs
                ),
            ],
            **({"capacity_match": capacity_match} if capacity_match is not None else {}),
        },
    )
    write_json(output / "resource_ledger.json", ledger.to_mapping())


__all__ = ["run_official_baseline"]
