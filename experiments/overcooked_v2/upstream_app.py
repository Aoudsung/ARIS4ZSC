"""Single-GPU Official SP/OP parent jobs for an experiment layout."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any

from src.delta_zsc.config import load_config
from src.delta_zsc.resources import ResourceLedger, parameter_count, peak_device_memory_bytes
from src.delta_zsc.storage import ensure_run_identity, write_json

from .official_adapter import (
    compose_official_config,
    restore_official_checkpoint,
    train_upstream,
    validate_official_runtime,
)


def train_official_parent(args: argparse.Namespace) -> None:
    """Train one exact key from an Official SP or OP population."""

    import jax

    method = str(args.method)
    algorithm = {"sp": "rnn-sp", "op": "rnn-op"}[method]
    config = load_config(args.config, run_kind=args.run_kind)
    devices = [device for device in jax.devices() if device.platform == "gpu"]
    if len(devices) != 1:
        raise RuntimeError("An Official parent job requires exactly one visible CUDA device.")
    runtime = validate_official_runtime()

    root_seed = int(args.root_seed)
    population_size = int(args.population_size)
    seed_index = int(args.seed_index)
    if not 0 <= seed_index < population_size:
        raise ValueError("Official parent seed index lies outside its population.")
    output = Path(args.output).resolve()
    official_config = dict(
        compose_official_config(
            config,
            algorithm=algorithm,
            seed_index=min(seed_index, 9),
            output_directory=output,
        )
    )
    official_config["SEED"] = root_seed
    official_config["NUM_SEEDS"] = population_size
    official_config["RUN_BASE_DIR"] = str(output)
    run_key = jax.random.split(jax.random.PRNGKey(root_seed), population_size)[
        seed_index
    ]
    identity = {
        "stage": "official-parent-train",
        "method": method,
        "layout": config.environment.layout,
        "parent_training_run_id": str(args.parent_training_run_id),
        "root_seed": root_seed,
        "population_size": population_size,
        "seed_index": seed_index,
        "jax_prng_key": [int(word) for word in run_key],
        "resolved_official_config": official_config,
        "official_runtime": runtime,
    }
    ensure_run_identity(output, identity)

    started = time.perf_counter()
    result = train_upstream(
        official_config,
        run_key=run_key,
        seed_index=seed_index,
        checkpoint_progress=config.upstream.checkpoint_progress,
    )
    elapsed = time.perf_counter() - started
    checkpoints = tuple(Path(path).resolve() for path in result["checkpoint_paths"])
    final_config, final_params = restore_official_checkpoint(checkpoints[-1])
    del final_config
    ledger = ResourceLedger(
        ego_policy_steps=int(result["effective_environment_steps"]),
        training_gpu_hours=elapsed / 3600.0,
        training_wall_clock_hours=elapsed / 3600.0,
        peak_memory_bytes=peak_device_memory_bytes(),
        deployable_parameters=parameter_count(final_params),
    )
    write_json(output / "resource_ledger.json", ledger.to_mapping())
    write_json(
        output / "parent_training.json",
        {
            "version": 1,
            "method": method,
            "generation_mechanism": algorithm,
            "layout": config.environment.layout,
            "parent_training_run_id": str(args.parent_training_run_id),
            "root_seed": root_seed,
            "population_size": population_size,
            "seed_index": seed_index,
            "jax_prng_key": [int(word) for word in run_key],
            "checkpoints": [
                {
                    "checkpoint_stage": float(progress),
                    "path": str(path),
                }
                for progress, path in zip(
                    config.upstream.checkpoint_progress, checkpoints, strict=True
                )
            ],
            "resource_ledger": {"path": str(output / "resource_ledger.json")},
        },
    )


__all__ = ["train_official_parent"]
