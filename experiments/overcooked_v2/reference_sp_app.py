"""Measure the seed-matched Official-SP reference target for CETR-ZSC."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.cetr_zsc.config import load_config
from src.cetr_zsc.storage import write_json

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
    import jax

    values = []
    for role in (0, 1):
        role_key = jax.random.fold_in(root_key, role)
        rollouts, unused_episode_keys = official_pairing_rollouts(
            left_policy=policy,
            right_policy=policy,
            environment=environment,
            root_key=role_key,
            episodes=int(episodes),
        )
        del unused_episode_keys
        values.append(np.asarray(rollouts.total_reward, dtype=np.float64))
    return np.concatenate(values, axis=0)


def measure_reference_sp(args: argparse.Namespace) -> None:
    import jax

    config = load_config(args.config, run_kind=args.run_kind)
    checkpoint = Path(args.sp_checkpoint).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    seed_index = int(args.seed_index)
    if seed_index not in range(-1, 10):
        raise ValueError("Reference-SP seed index must lie in -1..9.")
    if config.run_kind == "formal":
        devices = [device for device in jax.devices() if device.platform == "gpu"]
        if len(devices) != 1:
            raise RuntimeError("Formal reference-SP measurement requires one CUDA device.")
        validate_official_runtime()

    checkpoint_config, checkpoint_params = restore_official_checkpoint(checkpoint)
    policy = official_policy(checkpoint_params, checkpoint_config)
    environment = VectorEnvironment.create(config).environment
    root_key = jax.random.PRNGKey(int(config.evaluation.evaluation_seed))
    episodes = int(config.evaluation.episodes_per_pairing)
    returns = _reference_returns(
        policy=policy,
        environment=environment,
        root_key=root_key,
        episodes=episodes,
    )
    output = Path(args.output).resolve()
    write_json(
        output,
        {
            "artifact_type": "cetr_reference_sp",
            "version": 1,
            "layout": str(config.environment.layout),
            "seed_index": seed_index,
            "tau_sp": float(np.mean(returns)),
            "episodes_per_pairing": episodes,
            "evaluation_root_seed": int(config.evaluation.evaluation_seed),
            "source_checkpoint": str(checkpoint),
        },
    )


__all__ = ["measure_reference_sp"]
