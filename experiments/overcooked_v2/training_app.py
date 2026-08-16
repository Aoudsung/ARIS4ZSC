"""CETR-ZSC training and CUDA preflight application flow."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from src.cetr_zsc.config import (
    CHECKPOINT_SCHEMA_VERSION,
    FORMAL_METHOD_LABEL,
    FORMAL_PEAK_MEMORY_LIMIT_BYTES,
    METHOD_VERSION,
    OFFICIAL_ACTION_COUNT,
    load_config,
)
from src.cetr_zsc.manifest import load_partner_manifest
from src.cetr_zsc.model import CetrModel
from src.cetr_zsc.official_init import initialize_from_official, initializer_seed_index
from src.cetr_zsc.partners import build_training_partner_pool
from src.cetr_zsc.resources import (
    ResourceLedger,
    measure_policy_inference_latency_ms,
    parameter_count,
    peak_device_memory_bytes,
)
from src.cetr_zsc.risk import update_sp_dual
from src.cetr_zsc.runner import collect_episodes, make_static_partner_functions
from src.cetr_zsc.storage import (
    ensure_run_identity,
    load_latest_checkpoint,
    read_json,
    save_checkpoint,
    write_json,
)
from src.cetr_zsc.training import (
    compute_tail_weights,
    environment_minibatch_schedule,
    init_training_state,
    make_training_update_kernel,
    ppo_learning_rate,
)

from .deployment import export_deployment_bundle
from .official_adapter import FrozenPartnerPool, VectorEnvironment


def _training_key(seed_index: int) -> Any:
    import jax

    if int(seed_index) == -1:
        return jax.random.PRNGKey(0)
    return jax.random.split(jax.random.PRNGKey(42), 10)[int(seed_index)]


def _host(value: Any) -> Any:
    import jax

    if isinstance(value, Mapping):
        return {str(key): _host(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_host(item) for item in value]
    try:
        array = np.asarray(jax.device_get(value))
    except (TypeError, ValueError):
        return value
    if array.ndim == 0:
        return array.item()
    return array.tolist()


def _write_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _require_cuda() -> None:
    import jax

    if os.environ.get("JAX_PLATFORMS", "").strip().lower() != "cuda":
        raise RuntimeError("CETR CUDA acceptance requires JAX_PLATFORMS=cuda.")
    devices = [device for device in jax.devices() if device.platform == "gpu"]
    if len(devices) != 1:
        raise RuntimeError("CETR CUDA acceptance requires exactly one visible CUDA device.")


def _load_reference_sp(path: str | Path, *, config: Any, seed_index: int) -> Mapping[str, Any]:
    payload = read_json(path)
    required = {
        "artifact_type",
        "version",
        "layout",
        "seed_index",
        "tau_sp",
        "episodes_per_pairing",
        "evaluation_root_seed",
        "source_checkpoint",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ValueError("Reference-SP artifact schema differs.")
    if payload["artifact_type"] != "cetr_reference_sp" or int(payload["version"]) != 1:
        raise ValueError("Reference-SP artifact identity differs.")
    if str(payload["layout"]) != str(config.environment.layout):
        raise ValueError("Reference-SP artifact layout differs from the training config.")
    if int(payload["seed_index"]) != int(seed_index):
        raise ValueError("Reference-SP artifact seed differs from the training seed.")
    if int(payload["episodes_per_pairing"]) != int(config.evaluation.episodes_per_pairing):
        raise ValueError("Reference-SP artifact episode count differs from the config.")
    if int(payload["evaluation_root_seed"]) != int(config.evaluation.evaluation_seed):
        raise ValueError("Reference-SP artifact evaluation seed differs from the config.")
    tau = float(payload["tau_sp"])
    if not np.isfinite(tau):
        raise ValueError("Reference-SP artifact tau is not finite.")
    if not str(payload["source_checkpoint"]):
        raise ValueError("Reference-SP artifact source checkpoint is empty.")
    return dict(payload)


def _upstream_cost(pool: Any) -> tuple[int, float, float, list[Mapping[str, Any]]]:
    total_steps = 0
    gpu_hours = 0.0
    wall_hours = 0.0
    records: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for member in pool.members:
        parent_id = str(member.parent_training_run_id)
        if parent_id in seen:
            continue
        seen.add(parent_id)
        checkpoint = Path(member.checkpoint).resolve()
        source = next(
            (
                candidate / "resource_ledger.json"
                for candidate in (checkpoint, *checkpoint.parents[:6])
                if (candidate / "resource_ledger.json").is_file()
            ),
            None,
        )
        if source is None:
            records.append(
                {
                    "parent_training_run_id": parent_id,
                    "status": "missing",
                    "checkpoint": str(checkpoint),
                }
            )
            continue
        ledger = ResourceLedger.from_mapping(read_json(source))
        total_steps += int(ledger.total_training_simulator_steps)
        gpu_hours += float(ledger.gpu_hours)
        wall_hours += float(ledger.wall_clock_hours)
        records.append(
            {
                "parent_training_run_id": parent_id,
                "status": "counted",
                "checkpoint": str(checkpoint),
                "source": str(source),
                "steps": int(ledger.total_training_simulator_steps),
                "gpu_hours": float(ledger.gpu_hours),
                "wall_clock_hours": float(ledger.wall_clock_hours),
            }
        )
    return total_steps, gpu_hours, wall_hours, records


def _measured_sp(batch: Any) -> float:
    streams = np.asarray(batch.lane_stream)
    returns = np.asarray(batch.episode_return, dtype=np.float64)
    values = returns[streams == 0]
    if values.size == 0:
        raise ValueError("CETR episode batch contains no self-play lanes.")
    return float(np.mean(values))


def _make_ledger(
    *,
    upstream_steps: int,
    upstream_gpu_hours: float,
    upstream_wall_hours: float,
    ego_steps: int = 0,
    training_gpu_hours: float = 0.0,
    training_wall_hours: float = 0.0,
    deployable_parameters: int = 0,
    inference_latency_ms: float = 0.0,
) -> ResourceLedger:
    return ResourceLedger(
        ego_policy_steps=int(ego_steps),
        upstream_partner_steps=int(upstream_steps),
        training_gpu_hours=float(training_gpu_hours),
        shared_gpu_hours=float(upstream_gpu_hours),
        training_wall_clock_hours=float(training_wall_hours),
        shared_wall_clock_hours=float(upstream_wall_hours),
        deployable_parameters=int(deployable_parameters),
        inference_latency_ms=float(inference_latency_ms),
    )


def run_training(args: argparse.Namespace) -> None:
    import jax

    started = time.perf_counter()
    config = load_config(args.config, run_kind=args.run_kind)
    seed_index = int(args.seed_index)
    if seed_index not in range(-1, 10):
        raise ValueError("CETR training seed index must lie in -1..9.")
    requires_cuda = config.run_kind == "formal" or bool(getattr(args, "require_cuda", False))
    if requires_cuda:
        _require_cuda()

    manifest_path = Path(args.partner_manifest).resolve()
    manifest = load_partner_manifest(
        manifest_path,
        expected_layout=config.environment.layout,
        verify_files=not bool(getattr(args, "skip_manifest_file_check", False)),
    )
    pool = build_training_partner_pool(config, manifest)
    upstream_steps, upstream_gpu_hours, upstream_wall_hours, upstream_records = _upstream_cost(pool)
    frozen_pool = FrozenPartnerPool.from_checkpoints(
        [member.checkpoint for member in pool.members],
        parent_training_run_ids=[member.parent_training_run_id for member in pool.members],
    )
    partner_functions = make_static_partner_functions(
        pool_metadata=pool,
        official_pool=frozen_pool,
    )
    environment = VectorEnvironment.create(config)
    model = CetrModel(config, environment.observation_shape, OFFICIAL_ACTION_COUNT)

    reference_path = Path(args.reference_sp_artifact).resolve()
    reference_sp = _load_reference_sp(reference_path, config=config, seed_index=seed_index)
    initializer_path = Path(args.sp_initializer).resolve()
    initializer_seed = int(initializer_seed_index(initializer_path))
    if initializer_seed != seed_index:
        raise ValueError("--sp-initializer seed must equal --seed-index.")

    root_key = _training_key(seed_index)
    init_key, state_key = jax.random.split(root_key)
    params = model.init_parameters(init_key)
    params = initialize_from_official(params, initializer_path)
    output = Path(args.output).resolve()
    parent_manifest_identity = {
        "path": str(manifest_path),
        "run_ids": [str(run.run_id) for run in manifest.runs],
        "parent_training_run_ids": [str(run.parent_training_run_id) for run in manifest.runs],
    }
    identity = {
        "stage": "cetr-train",
        "method": METHOD_VERSION,
        "method_label": FORMAL_METHOD_LABEL,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "ego_run_id": str(args.ego_run_id),
        "seed_index": seed_index,
        "run_kind": config.run_kind,
        "config": config.to_mapping(),
        "reference_sp_artifact": dict(reference_sp),
        "parent_manifest": parent_manifest_identity,
        "sp_initializer": str(initializer_path),
        "observation_shape": list(environment.observation_shape),
        "action_count": OFFICIAL_ACTION_COUNT,
    }
    ensure_run_identity(output, identity)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "partner_manifest.json", read_json(manifest_path))
    write_json(output / "partner_pool.json", {
        "members": [
            {
                "run_id": str(member.run_id),
                "parent_training_run_id": str(member.parent_training_run_id),
                "checkpoint": str(member.checkpoint),
                "mechanism": str(member.mechanism),
                "checkpoint_stage": float(member.checkpoint_stage),
            }
            for member in pool.members
        ],
        "parent_ids": [str(value) for value in pool.parent_ids],
    })
    write_json(output / "upstream_cost_manifest.json", {"parents": upstream_records})

    checkpoint_identity = {
        "method": METHOD_VERSION,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "config": config.to_mapping(),
        "ego_run_id": str(args.ego_run_id),
        "seed_index": seed_index,
        "reference_sp_artifact": dict(reference_sp),
        "parent_manifest": parent_manifest_identity,
    }
    if bool(getattr(args, "resume", False)):
        restored = load_latest_checkpoint(
            output / "checkpoints", expected_identity=checkpoint_identity
        )
        if restored is None:
            raise FileNotFoundError("--resume requested but no checkpoint exists.")
        _, state = restored
    else:
        state = init_training_state(model=model, params=params, random_key=state_key)
        state = state._replace(
            resource_ledger=_make_ledger(
                upstream_steps=upstream_steps,
                upstream_gpu_hours=upstream_gpu_hours,
                upstream_wall_hours=upstream_wall_hours,
                deployable_parameters=parameter_count(params),
            ).to_mapping()
        )

    rollout_steps = int(config.environment.num_envs) * int(config.training.rollout_length)
    total_updates = int(config.training.environment_steps) // rollout_steps
    maximum_updates = getattr(args, "maximum_updates", None)
    if maximum_updates is not None:
        total_updates = min(total_updates, int(maximum_updates))
    total_optimizer_steps = max(
        total_updates * int(config.training.minibatches_per_epoch) * int(config.ppo.update_epochs),
        1,
    )
    update_kernel = make_training_update_kernel(
        model=model,
        total_optimizer_steps=total_optimizer_steps,
    )
    checkpoint_interval = max(
        int(config.training.checkpoint_interval_environment_steps), rollout_steps
    )
    metrics_path = output / "records" / "metrics.jsonl"
    update = int(np.asarray(state.update_count))
    environment_steps = int(np.asarray(state.effective_environment_steps))
    while update < total_updates:
        current_update = update
        next_key, batch, collection_metrics = collect_episodes(
            environment=environment,
            model=model,
            params=state.params,
            partner_functions=partner_functions,
            random_key=state.random_key,
        )
        lane_weight, tail_metrics = compute_tail_weights(
            batch,
            pool.parent_nominal_weights,
            len(pool.parent_members),
        )
        batch = batch._replace(lane_weight=lane_weight)
        schedule = environment_minibatch_schedule(
            jax.random.fold_in(next_key, current_update),
            environment_count=int(config.environment.num_envs),
            minibatches_per_epoch=int(config.training.minibatches_per_epoch),
            update_epochs=int(config.ppo.update_epochs),
        )
        params, optimizer_state, update_metrics = update_kernel(
            state.params,
            state.optimizer_state,
            batch,
            schedule,
            state.dual_lambda,
        )
        measured_sp = _measured_sp(batch)
        eta = ppo_learning_rate(
            config=config,
            optimizer_step=current_update
            * int(config.training.minibatches_per_epoch)
            * int(config.ppo.update_epochs),
            total_optimizer_steps=total_optimizer_steps,
        )
        dual_lambda = update_sp_dual(
            float(np.asarray(state.dual_lambda)),
            float(reference_sp["tau_sp"]),
            measured_sp,
            float(np.asarray(eta)),
        )
        update += 1
        environment_steps += rollout_steps
        state = state._replace(
            params=params,
            optimizer_state=optimizer_state,
            dual_lambda=float(dual_lambda),
            random_key=next_key,
            update_count=update,
            effective_environment_steps=environment_steps,
        )
        metrics = {
            "update": update,
            "environment_steps": environment_steps,
            "collection": _host(collection_metrics),
            "tail": _host(tail_metrics),
            "ppo": _host(update_metrics),
            "self_play_return": measured_sp,
            "reference_sp": float(reference_sp["tau_sp"]),
            "dual_lambda": float(dual_lambda),
            "dual_learning_rate": float(np.asarray(eta)),
        }
        _write_jsonl(metrics_path, metrics)
        if (
            environment_steps % checkpoint_interval == 0
            and environment_steps < int(config.training.environment_steps)
        ):
            save_checkpoint(
                output / "checkpoints",
                step=environment_steps,
                state=state,
                identity=checkpoint_identity,
            )

    elapsed = time.perf_counter() - started
    peak_memory = int(peak_device_memory_bytes())
    ledger = ResourceLedger.from_mapping(state.resource_ledger)
    ledger = ledger.plus(
        training_wall_clock_hours=elapsed / 3600.0,
        training_gpu_hours=(
            elapsed / 3600.0
            if any(device.platform == "gpu" for device in jax.devices())
            else 0.0
        ),
    )
    ledger = ResourceLedger(
        **{
            **{name: getattr(ledger, name) for name in ResourceLedger.__dataclass_fields__},
            "peak_memory_bytes": max(int(ledger.peak_memory_bytes), peak_memory),
        }
    )
    state = state._replace(resource_ledger=ledger.to_mapping())
    if requires_cuda and ledger.peak_memory_bytes >= FORMAL_PEAK_MEMORY_LIMIT_BYTES:
        write_json(output / "resource_ledger.json", ledger.to_mapping())
        raise RuntimeError("CETR CUDA peak device memory must remain below the registered limit.")

    deployment_directory = output / "final_deployment"
    if not deployment_directory.exists():
        export_deployment_bundle(
            deployment_directory,
            ego_run_id=str(args.ego_run_id),
            config=config,
            observation_shape=environment.observation_shape,
            params=state.params,
            source_training_run=output,
            reference_sp_artifact=reference_sp,
        )
    from .deployment import load_deployment
    from .official_policy import OfficialCetrPolicy

    deployment_policy = OfficialCetrPolicy(load_deployment(deployment_directory))
    latency_observation = np.zeros(environment.observation_shape, dtype=np.float32)
    inference_latency_ms = measure_policy_inference_latency_ms(
        deployment_policy,
        latency_observation,
    )
    ledger = ResourceLedger(
        **{
            **{name: getattr(ledger, name) for name in ResourceLedger.__dataclass_fields__},
            "peak_memory_bytes": max(int(ledger.peak_memory_bytes), int(peak_device_memory_bytes())),
            "inference_latency_ms": float(inference_latency_ms),
        }
    )
    state = state._replace(resource_ledger=ledger.to_mapping())
    if requires_cuda and ledger.peak_memory_bytes >= FORMAL_PEAK_MEMORY_LIMIT_BYTES:
        write_json(output / "resource_ledger.json", ledger.to_mapping())
        raise RuntimeError("CETR CUDA peak device memory must remain below the registered limit.")
    save_checkpoint(
        output / "checkpoints",
        step=environment_steps,
        state=state,
        identity=checkpoint_identity,
    )
    write_json(output / "resource_ledger.json", ledger.to_mapping())
    write_json(
        output / "training_summary.json",
        {
            "version": 1,
            "artifact_type": "cetr_training_summary",
            "method": METHOD_VERSION,
            "ego_run_id": str(args.ego_run_id),
            "seed_index": seed_index,
            "environment_steps": environment_steps,
            "update_count": update,
            "reference_sp_artifact": dict(reference_sp),
            "checkpoint": str(output / "checkpoints" / "latest.json"),
            "deployment": str(deployment_directory),
            "resource_ledger": str(output / "resource_ledger.json"),
        },
    )


def run_cuda_preflight(args: argparse.Namespace) -> None:
    if int(args.seed_index) != -1:
        raise ValueError("CUDA preflight uses the engineering seed index -1.")
    if str(args.run_kind) != "mechanical":
        raise ValueError("CUDA preflight uses the mechanical CETR config.")
    args.maximum_updates = 1
    args.require_cuda = True
    args.resume = False
    run_training(args)
    output = Path(args.output).resolve()
    required = (
        output / "final_deployment" / "deployment_bundle.json",
        output / "checkpoints" / "latest.json",
        output / "resource_ledger.json",
    )
    for path in required:
        if not path.is_file():
            raise RuntimeError(f"CUDA preflight artifact is missing: {path}")
    write_json(
        output / "cuda_preflight_report.json",
        {
            "passed": True,
            "method": METHOD_VERSION,
            "run_kind": "mechanical",
            "config": str(Path(args.config).resolve()),
            "partner_manifest": str(Path(args.partner_manifest).resolve()),
            "reference_sp_artifact": str(Path(args.reference_sp_artifact).resolve()),
            "deployment": str(output / "final_deployment"),
            "checkpoint": str(output / "checkpoints" / "latest.json"),
            "resource_ledger": str(output / "resource_ledger.json"),
        },
    )


__all__ = ["run_cuda_preflight", "run_training"]
