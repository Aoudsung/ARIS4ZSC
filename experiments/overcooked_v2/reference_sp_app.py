"""Measure the seed-matched Official-SP reference target for CETR-ZSC."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.cetr_zsc.config import (
    FORMAL_PEAK_MEMORY_LIMIT_BYTES,
    load_config,
)
from src.cetr_zsc.resources import peak_device_memory_bytes
from src.cetr_zsc.storage import write_json

from .evaluation_app import PAPER_MATRIX_ROOT_SEED, population_sp_keys
from .official_adapter import (
    VectorEnvironment,
    official_pairing_rollouts,
    official_policy,
    restore_official_checkpoint,
    validate_official_runtime,
)


def _reference_returns(
    *,
    policy: Any,
    environment: Any,
    root_key: Any,
    episodes: int,
) -> np.ndarray:
    rollouts, unused_episode_keys = official_pairing_rollouts(
        left_policy=policy,
        right_policy=policy,
        environment=environment,
        root_key=root_key,
        episodes=int(episodes),
    )
    del unused_episode_keys
    return np.asarray(rollouts.total_reward, dtype=np.float64)


def measure_reference_sp(args: argparse.Namespace) -> None:
    import jax

    config = load_config(args.config, run_kind=args.run_kind)
    checkpoint = Path(args.sp_checkpoint).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    seed_index = int(args.seed_index)
    if seed_index not in range(-1, 10):
        raise ValueError("Reference-SP seed index must lie in -1..9.")
    # -1 is engineering-only and binds to the seed-0 diagonal measurement.
    effective_seed = 0 if seed_index == -1 else seed_index
    if config.run_kind == "formal":
        if os.environ.get("JAX_PLATFORMS", "").strip().lower() != "cuda":
            raise RuntimeError("Formal reference-SP measurement requires JAX_PLATFORMS=cuda.")
        devices = [device for device in jax.devices() if device.platform == "gpu"]
        if len(devices) != 1:
            raise RuntimeError(
                "Formal reference-SP measurement requires exactly one visible CUDA GPU."
            )
        validate_official_runtime()

    checkpoint_config, checkpoint_params = restore_official_checkpoint(checkpoint)
    policy = official_policy(checkpoint_params, checkpoint_config)
    environment = VectorEnvironment.create(config).environment
    root_key = population_sp_keys()[effective_seed]
    episodes = int(config.evaluation.episodes_per_pairing)
    returns = _reference_returns(
        policy=policy,
        environment=environment,
        root_key=root_key,
        episodes=episodes,
    )
    if config.run_kind == "formal":
        peak_memory = int(peak_device_memory_bytes())
        if peak_memory >= FORMAL_PEAK_MEMORY_LIMIT_BYTES:
            raise RuntimeError(
                "Formal reference-SP peak device memory must remain below "
                "the registered limit."
            )
    output = Path(args.output).resolve()
    write_json(
        output,
        {
            "artifact_type": "cetr_reference_sp",
            "version": 2,
            "layout": str(config.environment.layout),
            "seed_index": effective_seed,
            "tau_sp": float(np.mean(returns)),
            "episodes_per_pairing": episodes,
            "evaluation_root_seed": int(PAPER_MATRIX_ROOT_SEED),
            "source_checkpoint": str(checkpoint),
        },
    )


__all__ = ["measure_reference_sp"]
