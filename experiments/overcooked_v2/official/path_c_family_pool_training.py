"""Versioned helpers for the Path C family-level official policy history."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

HISTORY_SCHEMA_VERSION = "path_c_official_checkpoint_history_v1"
HISTORY_SELECTION_RULE = "mechanical_official_update_schedule_v1"
HISTORY_CHECKPOINT_COUNT = 3
ABILITY_ADMISSION_SCHEMA_VERSION = "path_c_official_ability_admission_v1"
ABILITY_ADMISSION_RULE_ID = "official_final_quarter_ability_v1"
ABILITY_ADMISSION_METRIC = "returned_episode_returns"
ABILITY_ADMISSION_MINIMUM_FINAL_MEAN = 100.0
ABILITY_ADMISSION_MINIMUM_FINAL_TO_PEAK_RATIO = 0.9
ABILITY_OBSERVATION_SELECTION_EFFECT = "audit_only_include_all_fixed_seeds_v1"
_EXPECTED_UPDATE_COUNTS = {"rnn-sp": 457, "rnn-op": 1_831}
_OFFICIAL_CHECKPOINT_FORMAT = "orbax_pytree_directory_v1"
_OFFICIAL_PARAMETER_TREE_PATH = ("params",)
_HEX = frozenset("0123456789abcdef")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(_HEX)
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _hash_framed_bytes(digest: "hashlib._Hash", payload: bytes) -> None:
    digest.update(len(payload).to_bytes(8, byteorder="big", signed=False))
    digest.update(payload)


def checkpoint_artifact_sha256(path: str | Path) -> str:
    """Hash a checkpoint without importing JAX, Flax, Orbax, or JaxMARL."""

    target = Path(path)
    if target.is_file():
        return _file_sha256(target)
    if not target.is_dir():
        raise FileNotFoundError(f"Official checkpoint is missing: {target}")
    files = sorted(item for item in target.rglob("*") if item.is_file())
    if not files:
        raise ValueError("An official checkpoint directory cannot be empty.")
    digest = hashlib.sha256(b"path_c_checkpoint_directory_sha256_v1\x00")
    for item in files:
        relative = item.relative_to(target).as_posix()
        _hash_framed_bytes(digest, relative.encode("utf-8"))
        _hash_framed_bytes(digest, bytes.fromhex(_file_sha256(item)))
    return digest.hexdigest()


def validate_checkpoint_history(
    history: Sequence[Mapping[str, Any]],
    *,
    final_weights_sha256: str,
    verify_files: bool = True,
) -> tuple[Mapping[str, Any], ...]:
    """Verify three scheduled checkpoints without consulting return metrics."""

    rows = tuple(dict(item) for item in history)
    if len(rows) != HISTORY_CHECKPOINT_COUNT:
        raise ValueError("An official family member must expose three checkpoints.")
    if tuple(int(row.get("checkpoint_index", -1)) for row in rows) != (0, 1, 2):
        raise ValueError("Official checkpoint indexes must be ordered 0, 1, 2.")
    updates = tuple(int(row.get("update_step", -1)) for row in rows)
    environment_steps = tuple(
        int(row.get("effective_environment_steps", -1)) for row in rows
    )
    if any(left >= right for left, right in zip(updates, updates[1:])):
        raise ValueError("Official checkpoint updates must increase.")
    if any(
        left >= right
        for left, right in zip(environment_steps, environment_steps[1:])
    ):
        raise ValueError("Official checkpoint environment steps must increase.")
    if any(row.get("selection_rule") != HISTORY_SELECTION_RULE for row in rows):
        raise ValueError("Official checkpoint history cannot use return selection.")
    if any(row.get("format") != _OFFICIAL_CHECKPOINT_FORMAT for row in rows):
        raise ValueError("Official checkpoint history uses an unknown format.")
    if any(
        tuple(row.get("parameter_tree_path", ())) != _OFFICIAL_PARAMETER_TREE_PATH
        for row in rows
    ):
        raise ValueError("Official checkpoint history uses the wrong parameter path.")
    if any(
        not _is_sha256(row.get(field))
        for row in rows
        for field in ("checkpoint_sha256", "model_weights_sha256")
    ):
        raise ValueError("Official checkpoint history requires lowercase SHA-256 values.")
    if rows[-1].get("model_weights_sha256") != final_weights_sha256:
        raise ValueError("The admitted final policy differs from the final history member.")
    paths = tuple(Path(str(row.get("path", ""))).resolve() for row in rows)
    if len(set(paths)) != HISTORY_CHECKPOINT_COUNT:
        raise ValueError("Official checkpoint history repeats a path.")
    if verify_files:
        for row, path in zip(rows, paths, strict=True):
            if not path.is_dir():
                raise FileNotFoundError(f"Scheduled checkpoint is missing: {path}")
            if row.get("checkpoint_sha256") != checkpoint_artifact_sha256(path):
                raise ValueError("Scheduled checkpoint content hash changed.")
    return rows


def evaluate_final_policy_ability_admission(
    metrics: Mapping[str, Any],
    *,
    training_run_id: str,
    launch_config_sha256: str,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    model_weights_sha256: str,
) -> Mapping[str, Any]:
    """Decide whether one completed official run may contribute its history.

    The rule reuses the already registered official final-quarter competence
    and no-collapse thresholds.  It adds no evaluation episodes: the evidence
    is the completed run's official raw-return series.  The three history
    checkpoints remain mechanically selected and are never ranked by return.
    """

    if metrics.get("schema_version") != "path_c_official_training_metrics_v1":
        raise ValueError("Official ability admission requires registered training metrics.")
    variant = str(metrics.get("experiment_variant", ""))
    expected_count = _EXPECTED_UPDATE_COUNTS.get(variant)
    if expected_count is None:
        raise ValueError("Official ability admission received an unknown training variant.")
    if int(metrics.get("metric_record_count", -1)) != expected_count:
        raise ValueError("Official ability admission received the wrong update count.")
    metrics_by_update = metrics.get("metrics_by_update")
    if not isinstance(metrics_by_update, Mapping):
        raise ValueError("Official ability admission lacks metrics_by_update.")
    values = np.asarray(
        metrics_by_update.get(ABILITY_ADMISSION_METRIC), dtype=np.float64
    )
    if values.ndim != 1 or values.shape[0] != expected_count:
        raise ValueError(
            "Official ability admission requires one raw-return value per update."
        )
    if not np.isfinite(values).all():
        raise ValueError("Official ability admission received a non-finite return.")
    if not training_run_id:
        raise ValueError("Official ability admission requires a training-run identifier.")
    hashes = {
        "launch_config_sha256": launch_config_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "model_weights_sha256": model_weights_sha256,
    }
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or not set(value).issubset(frozenset("0123456789abcdef"))
        for value in hashes.values()
    ):
        raise ValueError("Official ability admission requires lowercase SHA-256 bindings.")

    boundaries = [index * expected_count // 4 for index in range(5)]
    quarter_means = [
        float(values[boundaries[index] : boundaries[index + 1]].mean())
        for index in range(4)
    ]
    final_mean = quarter_means[-1]
    peak_mean = max(quarter_means)
    final_to_peak = (
        final_mean / peak_mean if peak_mean > 0.0 else float("-inf")
    )
    performance_pass = final_mean >= ABILITY_ADMISSION_MINIMUM_FINAL_MEAN
    no_collapse_pass = (
        final_to_peak >= ABILITY_ADMISSION_MINIMUM_FINAL_TO_PEAK_RATIO
    )
    return {
        "schema_version": ABILITY_ADMISSION_SCHEMA_VERSION,
        "rule_id": ABILITY_ADMISSION_RULE_ID,
        "run_kind": "formal",
        "scientific_readout_allowed": False,
        "experiment_variant": variant,
        "training_seed": int(metrics.get("seed", -1)),
        "effective_environment_steps": int(
            metrics.get("effective_environment_steps", -1)
        ),
        "training_run_id": training_run_id,
        "launch_config_sha256": launch_config_sha256,
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "model_weights_sha256": model_weights_sha256,
        "metric": ABILITY_ADMISSION_METRIC,
        "metric_meaning": "official_wrapper_raw_episode_return",
        "metric_record_count": expected_count,
        "quarter_partition": "contiguous_integer_update_quarters_v1",
        "quarter_boundaries_zero_based": boundaries,
        "quarter_means": quarter_means,
        "final_quarter_mean_raw_return": final_mean,
        "peak_quarter_mean_raw_return": peak_mean,
        "final_to_peak_quarter_ratio": final_to_peak,
        "minimum_final_quarter_mean_raw_return": (
            ABILITY_ADMISSION_MINIMUM_FINAL_MEAN
        ),
        "minimum_final_to_peak_quarter_ratio": (
            ABILITY_ADMISSION_MINIMUM_FINAL_TO_PEAK_RATIO
        ),
        "performance_threshold_pass": performance_pass,
        "no_collapse_threshold_pass": no_collapse_pass,
        "passed": performance_pass and no_collapse_pass,
        "history_selection_consulted_return": False,
        "selection_effect": ABILITY_OBSERVATION_SELECTION_EFFECT,
        "included_in_population": True,
        "seed_replacement_allowed": False,
        "retraining_until_pass_allowed": False,
    }


def run_three_checkpoint_wiring_smoke(
    launch_config_path: str | Path,
    output_directory: str | Path,
) -> Mapping[str, Any]:
    """Run three official updates and round-trip every scheduled checkpoint.

    This is a mechanical wiring check.  It deliberately writes no reward
    statistic and grants no scientific readout.
    """

    import copy

    import jax
    import wandb

    from experiments.overcooked_v2.official import (
        overcooked_v2_experiments_adapter as official_adapter,
    )
    from experiments.overcooked_v2.path_c_official_artifact import (
        checkpoint_artifact_sha256,
        flax_weights_sha256,
    )

    official_adapter._prepare_official_imports()
    from overcooked_v2_experiments.ppo.ippo import make_train
    from overcooked_v2_experiments.utils.utils import mini_batch_pmap

    launch_path = Path(launch_config_path).resolve()
    output_root = Path(output_directory).resolve()
    if output_root.exists():
        raise FileExistsError("The three-checkpoint smoke output already exists.")
    launch = official_adapter.load_official_launch_config(launch_path)
    config = copy.deepcopy(official_adapter.compose_official_training_config(launch))
    config["model"]["TOTAL_TIMESTEPS"] = (
        3 * int(config["model"]["NUM_ENVS"]) * int(config["model"]["NUM_STEPS"])
    )
    config["NUM_CHECKPOINTS"] = 3
    config["RUN_BASE_DIR"] = output_root
    output_root.mkdir(parents=True, exist_ok=False)
    train = make_train(config)
    mapped = mini_batch_pmap(jax.jit(train), 1)
    with wandb.init(
        project="path-c-family-pool-mechanical-smoke",
        mode="disabled",
        reinit=True,
    ):
        output = mapped(
            jax.random.split(jax.random.PRNGKey(int(launch["seed"])), 1)
        )
        jax.block_until_ready(output["runner_state"][1])
    checkpoint_state = output["runner_state"][1]
    leaves = jax.tree_util.tree_leaves(checkpoint_state)
    if not leaves or any(np.asarray(value).shape[:2] != (1, 3) for value in leaves):
        raise RuntimeError("The official smoke did not return three checkpoint states.")
    updates = (0, 1, 3)
    rows = []
    for checkpoint_index, update_step in enumerate(updates):
        params = jax.tree_util.tree_map(
            lambda value, index=checkpoint_index: value[0, index],
            checkpoint_state,
        )
        path = official_adapter.save_official_scheduled_checkpoint(
            output_root,
            config=config,
            params=params,
            update_step=update_step,
        )
        unused_config, restored = official_adapter.restore_official_checkpoint(path)
        del unused_config
        expected_weights = flax_weights_sha256(params)
        if flax_weights_sha256(restored) != expected_weights:
            raise RuntimeError("The official smoke checkpoint failed its round trip.")
        rows.append(
            {
                "checkpoint_index": checkpoint_index,
                "update_step": update_step,
                "effective_environment_steps": (
                    update_step
                    * int(config["model"]["NUM_ENVS"])
                    * int(config["model"]["NUM_STEPS"])
                ),
                "path": str(path),
                "format": official_adapter.OFFICIAL_CHECKPOINT_FORMAT,
                "parameter_tree_path": list(
                    official_adapter.OFFICIAL_PARAMETER_TREE_PATH
                ),
                "selection_rule": HISTORY_SELECTION_RULE,
                "checkpoint_sha256": checkpoint_artifact_sha256(path),
                "model_weights_sha256": expected_weights,
            }
        )
    validated = validate_checkpoint_history(
        rows,
        final_weights_sha256=str(rows[-1]["model_weights_sha256"]),
    )
    report = {
        "schema_version": "path_c_three_checkpoint_wiring_smoke_v1",
        "run_kind": "mechanical_smoke",
        "scientific_readout_allowed": False,
        "launch_config_path": str(launch_path),
        "checkpoint_history_schema_version": HISTORY_SCHEMA_VERSION,
        "checkpoint_history": list(validated),
        "checks": {
            "three_checkpoints_generated": True,
            "all_checkpoints_loaded": True,
            "all_weight_hashes_match": True,
            "return_metrics_read": False,
        },
        "passed": True,
    }
    target = output_root / "mechanical_smoke.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


__all__ = [
    "ABILITY_ADMISSION_METRIC",
    "ABILITY_ADMISSION_MINIMUM_FINAL_MEAN",
    "ABILITY_ADMISSION_MINIMUM_FINAL_TO_PEAK_RATIO",
    "ABILITY_ADMISSION_RULE_ID",
    "ABILITY_ADMISSION_SCHEMA_VERSION",
    "ABILITY_OBSERVATION_SELECTION_EFFECT",
    "HISTORY_CHECKPOINT_COUNT",
    "HISTORY_SCHEMA_VERSION",
    "HISTORY_SELECTION_RULE",
    "checkpoint_artifact_sha256",
    "evaluate_final_policy_ability_admission",
    "run_three_checkpoint_wiring_smoke",
    "validate_checkpoint_history",
]
