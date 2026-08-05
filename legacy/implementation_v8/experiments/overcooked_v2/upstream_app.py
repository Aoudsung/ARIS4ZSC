"""Official SP/OP upstream training application for Path C."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from experiments.overcooked_v2.official_adapter import (
    compose_official_config,
    restore_official_checkpoint,
    train_upstream,
    validate_official_runtime,
)
from src.path_c.experiment import load_config, official_training_key
from src.path_c.resources import (
    ResourceLedger,
    configure_bundled_cuda_toolchain,
    gpu_hours_for_wall_seconds,
    parameter_count,
    peak_device_memory_bytes,
    require_single_cuda_worker,
)
from src.path_c.storage import (
    ensure_run_identity,
    upstream_identity,
    validate_formal_repository_state,
    write_array_chunks,
    write_json,
    write_run_metadata,
    validate_registered_python_runtime,
)


def _metric_with_row_axis(values: object) -> object:
    """Normalize scalar Official diagnostics without changing array metrics."""

    import numpy as np

    array = np.asarray(values)
    return array.reshape((1,)) if array.ndim == 0 else values


def run_upstream(args: argparse.Namespace) -> None:
    """Run one locked official upstream training job and preserve all outputs."""

    config = load_config(args.config, run_kind=args.run_kind)
    cuda_runtime = None
    if config.run_kind == "formal":
        cuda_toolchain = configure_bundled_cuda_toolchain()
        validate_formal_repository_state()
        validate_registered_python_runtime()
        cuda_runtime = require_single_cuda_worker()
        cuda_runtime = {**cuda_runtime, "cuda_toolchain": cuda_toolchain}
    runtime = validate_official_runtime()
    seed_index = int(args.seed_index)
    explicit_key = getattr(args, "jax_prng_key", None)
    if explicit_key is not None and config.run_kind != "mechanical":
        raise ValueError(
            "Explicit upstream PRNG keys are restricted to non-scientific "
            "mechanical E2E fixtures."
        )
    if explicit_key is None:
        key_words = official_training_key(seed_index)
    else:
        key_words = tuple(int(value) for value in explicit_key)
        if len(key_words) != 2 or any(
            value < 0 or value > 0xFFFF_FFFF for value in key_words
        ):
            raise ValueError("JAX PRNG keys contain exactly two uint32 words.")
    output = Path(args.output).resolve()
    ensure_run_identity(
        output,
        {
            **upstream_identity(
            config=config,
            seed_index=seed_index,
            jax_prng_key=key_words,
            algorithm=str(args.algorithm),
            ),
            "official_runtime": runtime,
            "formal_cuda_worker": cuda_runtime,
        },
    )
    if cuda_runtime is not None:
        write_json(output / "runtime_gpu.json", cuda_runtime)
    official = compose_official_config(
        config,
        algorithm=args.algorithm,
        seed_index=seed_index,
        output_directory=output,
    )
    started = time.perf_counter()
    result = train_upstream(
        official,
        run_key=key_words,
        seed_index=seed_index,
        checkpoint_progress=config.upstream.checkpoint_progress,
    )
    wall_seconds = time.perf_counter() - started
    write_json(output / "resolved_config.json", config.to_mapping())
    write_json(output / "official_resolved_config.json", official)

    metric_paths: dict[str, list[str]] = {}
    for name, values in result["metrics"].items():
        metric_paths[str(name)] = [
            str(path)
            for path in write_array_chunks(
                output / "metrics",
                name=str(name),
                values=_metric_with_row_axis(values),
                rows_per_chunk=1024,
            )
        ]
    write_json(
        output / "upstream_summary.json",
        {
            "algorithm": str(args.algorithm),
            "seed_index": seed_index,
            "jax_prng_key": list(key_words),
            "explicit_mechanical_key": explicit_key is not None,
            "scientific_readout_allowed": False,
            "effective_environment_steps": int(
                result["effective_environment_steps"]
            ),
            "completed_episodes": int(result["completed_episodes"]),
            "update_count": int(result["update_count"]),
            "checkpoint_paths": [
                str(path) for path in result["checkpoint_paths"]
            ],
            "cuda_only_debug_callbacks_disabled": bool(
                result["cuda_only_debug_callbacks_disabled"]
            ),
            "metric_files": metric_paths,
        },
    )
    write_run_metadata(
        output / "run_metadata.json",
        config=config,
        seed=seed_index,
        effective_environment_steps=int(result["effective_environment_steps"]),
        update_count=int(result["update_count"]),
        completed_episodes=int(result["completed_episodes"]),
    )
    unused_checkpoint_config, final_params = restore_official_checkpoint(
        result["checkpoint_paths"][-1]
    )
    del unused_checkpoint_config
    write_json(
        output / "resource_ledger.json",
        ResourceLedger(
            upstream_partner_steps=int(result["effective_environment_steps"]),
            gpu_hours=gpu_hours_for_wall_seconds(wall_seconds),
            wall_clock_hours=float(wall_seconds) / 3_600.0,
            peak_memory_bytes=peak_device_memory_bytes(),
            deployable_parameters=parameter_count(final_params),
        ).to_mapping(),
    )
    print(f"Complete upstream metrics: {output / 'metrics'}")


__all__ = ["_metric_with_row_axis", "run_upstream"]
