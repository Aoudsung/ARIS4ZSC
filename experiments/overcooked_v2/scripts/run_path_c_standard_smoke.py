#!/usr/bin/env python3
"""R010 wiring check; its outputs are never eligible for scientific readout."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import jax
import jax.numpy as jnp

from experiments.overcooked_v2.path_c_standard_evaluation import (
    StandardPairingEvaluator,
)
from experiments.overcooked_v2.path_c_standard_training import (
    StandardPathCTrainer,
    load_standard_training_config,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError("R010 requires the CUDA JAX backend.")
    compile_sum = float(jnp.arange(4, dtype=jnp.float32).sum().block_until_ready())
    if compile_sum != 6.0:
        raise RuntimeError("CUDA JAX compilation check returned an unexpected value.")

    base_config = load_standard_training_config(args.config)
    if base_config.get("run_kind") != "smoke":
        raise ValueError("R010 accepts only a smoke configuration.")
    output_dir = Path(args.output_dir).resolve()
    checkpoints = []
    training_manifests = []
    for seed in (1001, 1002):
        config = copy.deepcopy(base_config)
        seed_dir = output_dir / f"seed_{seed}"
        config["seed"] = seed
        config["output_dir"] = str(seed_dir)
        manifest = StandardPathCTrainer(config).run()
        training_manifests.append(manifest)
        checkpoints.append(str(seed_dir / "path_c_final.pt"))

    evaluation_config = {
        "schema_version": "path_c_standard_evaluation_v1",
        "run_kind": "smoke",
        "scientific_readout_allowed": False,
        "evaluation_seed": 9001,
        "device": base_config.get("device", "cuda"),
        "environment": base_config["environment"],
        "checkpoint_paths": checkpoints,
        "output_dir": str(output_dir / "evaluation"),
        # No fallback: a missing probe block must fail the fail-closed validator
        # instead of silently evaluating a degenerate probe configuration.
        "probe": base_config.get("probe"),
        "evaluation": {
            "episodes_per_pairing": 2,
            "evaluation_batch_size": 2,
            "expected_policy_count": 2,
            "standard_deviation_ddof": 0,
            "bootstrap_replicates": 100,
            "bootstrap_confidence": 0.95,
        },
    }
    summary = StandardPairingEvaluator(evaluation_config).run()
    artifact = {
        "schema_version": "path_c_standard_r010_smoke_v1",
        "scientific_readout_allowed": False,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(item) for item in jax.devices()],
        "compile_sum": compile_sum,
        "training_manifests": training_manifests,
        "evaluation_summary": summary,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "r010_smoke_manifest.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
