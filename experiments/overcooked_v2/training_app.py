"""DELTA-ZSC r3 Signal Contract training lifecycle.

This module owns orchestration only.  Losses, qualification statistics,
target-policy epochs, replay isolation, response routing and fallback selection
live in their dedicated ``src.path_c`` modules.  Formal runs fail closed: an
unqualified signal can never reach a downstream controller.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from typing import Any, Mapping

import numpy as np

from experiments.overcooked_v2.deployment import deployable_parameters
from experiments.overcooked_v2.official_adapter import (
    FrozenPartnerPool,
    VectorEnvironment,
    restore_official_checkpoint,
    validate_official_runtime,
)
from src.path_c.anchor_replay import (
    AnchorReplayState,
    AnchorAuditManifest,
    AnchorAuditRecord,
    append_replay_batch,
    audit_manifest_from_mapping,
    audit_manifest_to_mapping,
    current_epoch_replay_batch,
)
from src.path_c.anchor_sampling import (
    anchor_preflight_world_count,
    collect_anchor_batch,
    collect_external_partner_support_audit,
    collect_matched_code_audit,
    make_anchor_functions,
    preflight_anchor_microbatch_from_records,
)
from src.path_c.base_distillation import (
    collect_owner_behavior,
    gate_zero_consistency,
    owner_behavior_loss,
)
from src.path_c.belief_set_encoder import mixture_moments
from src.path_c.calibration import empty_calibration
from src.path_c.compiled_kernels import (
    CompiledCallable,
    CompiledMemoryLimitError,
    attach_chunked_regret_with_kernels,
    build_anchor_chunk_kernel,
    build_training_kernels,
    configure_persistent_compilation_cache,
)
from src.path_c.curriculum import (
    CurriculumPhase,
    phase_from_qualification,
)
from src.path_c.decision_geometry import (
    brdiv_marginal_contributions,
    centered_action_values,
    smoothness_marginal_penalties,
)
from src.path_c.experiment import (
    METHOD_VERSION,
    ENGINEERING_SEED_INDEX,
    OFFICIAL_CORRECT_DELIVERY_REWARD,
    OFFICIAL_SOURCE_COMMIT,
    load_config,
    load_partner_manifest,
    official_training_domain_keys,
    engineering_training_key,
    official_training_key,
    validate_seed_signal_contract_manifest,
)
from src.path_c.fallback import deployment_tier
from src.path_c.gradient_routing import keep_owned_gradients
from src.path_c.model import build_model, initial_policy_state, initialize_model_parameters
from src.path_c.partner_episode import (
    GeneratorAdmission,
    admitted_source_probabilities,
    collect_complete_generator_episodes,
    complete_episode_raw_returns,
    generator_ppo_objective,
    rollback_unqualified_generator,
    source_logit_distillation_loss,
    validate_complete_episode_batch,
)
from src.path_c.partner_generator import (
    build_partner_generator,
    initial_generator_carry,
    initialize_generator_parameters,
    sample_partner_codes,
)
from src.path_c.partner_sources import MixedPartnerParameters, make_mixed_partner_functions
from src.path_c.policy_epoch import (
    clone_epoch_target_for_live_control,
    combined_policy_fingerprint,
    start_target_policy_epoch,
    validate_epoch_alignment,
)
from src.path_c.qualification import (
    GateDecision,
    SignalQualification,
    bootstrap_interval,
    compound_gate_decision,
    paired_mean_cvar_noninferiority_decision,
    paired_mean_cvar_statistics,
)
from src.path_c.qualification_rollout import (
    paired_conditional_base_returns,
    paired_context_returns,
    paired_delta_params_returns,
    paired_generator_external_returns,
    paired_owner_base_returns,
)
from src.path_c.raw_q import conservative_raw_q, gather_actions
from src.path_c.resources import (
    ResourceLedger,
    configure_bundled_cuda_toolchain,
    gpu_hours_for_wall_seconds,
    parameter_count,
    peak_device_memory_bytes,
    r3_training_anchor_attempted_steps,
    require_single_cuda_worker,
)
from src.path_c.runner import DECISION_REGRET_STATE_CHUNK_SIZE, initialize_runner
from src.path_c.signal_audit import (
    ActiveInformationAudit,
    ConditionalControlAudit,
    GeneratorDecisionAudit,
    InferenceAudit,
    active_information_audit,
    decision_support_audit,
    generator_decision_audit,
    leave_one_partner_out_ordering_accuracy,
    privileged_functional_contexts,
    raw_q_audit,
)
from src.path_c.snapshot_archive import (
    immutable_parameter_snapshot,
    save_generator_snapshot,
    stack_parameter_trees,
)
from src.path_c.storage import (
    ensure_run_identity,
    orbax_manager,
    pytree_fingerprint,
    restore_latest_checkpoint,
    save_checkpoint,
    sha256_path,
    training_identity,
    validate_formal_repository_state,
    validate_registered_python_runtime,
    write_json,
    write_json_atomic,
    write_jsonl,
)
from src.path_c.training import (
    environment_minibatch_schedule,
    make_optimizer,
    merge_training_core,
    official_reward_shaping_factor,
    training_core_state,
)
from src.path_c.types import (
    AuxiliaryCoreState,
    CounterfactualAnchorBatch,
    GaussianMixtureBelief,
    GeneratorCoreState,
    PolicyState,
    TrainState,
)


FORMAL_PEAK_MEMORY_LIMIT_BYTES = 40_000 * 2**20
CUDA_PREFLIGHT_SCOPE = "formal_cuda_single_update_preflight"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _host(value: Any) -> Any:
    import jax
    import numpy as np

    if isinstance(value, Mapping):
        return {str(key): _host(item) for key, item in value.items()}
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return {name: _host(getattr(value, name)) for name in value._fields}
    if isinstance(value, (tuple, list)):
        return [_host(item) for item in value]
    array = np.asarray(jax.device_get(value))
    if array.ndim:
        return array.tolist()
    if np.issubdtype(array.dtype, np.bool_):
        return bool(array)
    if np.issubdtype(array.dtype, np.integer):
        return int(array)
    if np.issubdtype(array.dtype, np.floating):
        return float(array)
    return array.item()


def _jax_runtime_snapshot() -> Mapping[str, Any]:
    import jax

    return {
        "backend": str(jax.default_backend()),
        "jax_platforms": os.environ.get("JAX_PLATFORMS"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "local_device_count": int(jax.local_device_count()),
        "devices": [str(device) for device in jax.devices()],
    }


def _gpu_snapshot() -> Mapping[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,"
        "ecc.errors.uncorrected.volatile.total",
        "--format=csv,noheader,nounits",
    ]
    physical = os.environ.get("DELTA_PHYSICAL_GPU_INDEX", "").strip()
    if physical:
        command.append(f"--id={physical}")
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=10
        )
        values = [item.strip() for item in result.stdout.splitlines()[0].split(",")]
        return {
            "physical_index": values[0],
            "uuid": values[1],
            "name": values[2],
            "total_memory_mib": int(values[3]),
            "used_memory_mib": int(values[4]),
            "utilization_percent": int(values[5]),
            "volatile_uncorrectable_ecc": int(values[6]),
            "jax_peak_memory_bytes": peak_device_memory_bytes(),
        }
    except Exception as error:
        return {
            "physical_index": physical or None,
            "telemetry_error": f"{type(error).__name__}: {error}",
            "jax_peak_memory_bytes": peak_device_memory_bytes(),
        }


def _synchronize(value: Any) -> None:
    import jax

    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _run_phase(
    *, output: Path, phase: str, update: int, operation: Any,
    synchronize: bool, kernels: Any | None = None,
) -> tuple[Any, float]:
    status = output / "runtime_phase.json"
    started = time.perf_counter()
    record = {
        "status": "running",
        "phase": phase,
        "update_count": int(update),
        "started_at": _utc_now(),
        "jax": _jax_runtime_snapshot(),
        "gpu": _gpu_snapshot(),
    }
    write_json_atomic(status, record)
    try:
        value = operation()
        if synchronize:
            _synchronize(value)
        duration = time.perf_counter() - started
        completed = {
            **record,
            "status": "complete",
            "completed_at": _utc_now(),
            "wall_seconds": duration,
            "gpu": _gpu_snapshot(),
            "kernels": {} if kernels is None else kernels(),
        }
        write_json_atomic(status, completed)
        write_json(output / "records" / "phases" / f"update_{update:08d}_{phase}.json", completed)
        return value, duration
    except Exception as error:
        failed = {
            **record,
            "status": "failed",
            "failed_at": _utc_now(),
            "wall_seconds": time.perf_counter() - started,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "gpu": _gpu_snapshot(),
        }
        error_path = output / "records" / "phase_errors" / f"update_{update:08d}_{phase}.json"
        failed["error_log"] = str(error_path)
        write_json(error_path, failed)
        write_json_atomic(status, failed)
        raise


def _model_structure_fingerprint(config: Any, observation_shape: tuple[int, ...]) -> str:
    payload = {
        "model": config.to_mapping()["model"],
        "observation_shape": list(observation_shape),
        "action_count": 6,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _optional_tree_fingerprint(value: Any) -> str:
    return "none" if value is None else pytree_fingerprint(value)


def _state_resume_identity(state: TrainState) -> Mapping[str, Any]:
    """Bind every r3 state component that can change the next update."""

    qualification = _qualification_mapping(state.qualification)
    replay = state.anchor_training_replay
    return {
        "method": METHOD_VERSION,
        "target_policy_epoch_id": int(np.asarray(state.target_policy_epoch.epoch_id)),
        "target_policy_fingerprint": combined_policy_fingerprint(
            state.target_policy_epoch
        ),
        "qualified_base_fingerprint": _optional_tree_fingerprint(
            state.qualified_base_params
        ),
        "last_qualified_generator_fingerprint": _optional_tree_fingerprint(
            state.last_qualified_generator_params
        ),
        "last_qualified_generator_optimizer_step": int(
            np.asarray(state.last_qualified_generator_optimizer_step)
        ),
        "generator_snapshot_archive_fingerprint": _optional_tree_fingerprint(
            state.generator_snapshot_archive
        ),
        "anchor_replay_fingerprint": _optional_tree_fingerprint(replay),
        "anchor_audit_manifest_fingerprint": hashlib.sha256(
            json.dumps(
                audit_manifest_to_mapping(state.anchor_audit_manifest),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "resource_ledger_fingerprint": hashlib.sha256(
            json.dumps(_host(state.resource_ledger), sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "qualification_fingerprint": qualification.fingerprint,
        "curriculum_phase": int(np.asarray(state.curriculum_phase)),
        "optimizer_steps": {
            "ppo": int(np.asarray(state.ppo_optimizer_step)),
            "raw_q": int(np.asarray(state.raw_q_optimizer_step)),
            "response": int(np.asarray(state.response_optimizer_step)),
            "generator": int(np.asarray(state.generator_optimizer_step)),
        },
        "update_count": int(np.asarray(state.update_count)),
        "effective_environment_steps": int(
            np.asarray(state.effective_environment_steps)
        ),
        "random_domains": _host(state.random_domains),
        "random_key": _host(state.random_key),
    }


def _save_r3_checkpoint(
    manager: Any, *, output: Path, step: int, state: TrainState
) -> None:
    checkpoint_state = state._replace(
        anchor_audit_manifest=audit_manifest_to_mapping(
            state.anchor_audit_manifest
        )
    )
    save_checkpoint(manager, step=step, state=checkpoint_state)
    write_json(
        output / "checkpoint_identity" / f"{int(step):012d}.json",
        _state_resume_identity(state),
    )


def _restore_tree_like(template: Any, saved: Any) -> Any:
    """Rebuild registered PyTree container types from an Orbax value tree.

    A generic Orbax restore intentionally returns mappings/lists instead of
    application NamedTuples.  Reapplying the live r3 template recovers Flax,
    Optax, runner and epoch containers while allowing explicitly dynamic
    checkpoint fields (qualification decisions and snapshot archives) to grow.
    """

    if template is None:
        return saved
    if isinstance(template, tuple) and hasattr(template, "_fields"):
        if not template._fields and saved is None:
            # Orbax represents zero-leaf Optax states (for example
            # ``EmptyState``) as ``None`` in an untyped restore.
            return type(template)()
        if not isinstance(saved, Mapping):
            raise TypeError(
                f"Expected mapping for {type(template).__name__}, got {type(saved).__name__}."
            )
        fields = template._fields
        if set(saved) != set(fields):
            raise ValueError(
                f"Checkpoint fields for {type(template).__name__} differ: "
                f"stored={sorted(saved)}, expected={sorted(fields)}."
            )
        return type(template)(
            *(
                _restore_tree_like(getattr(template, name), saved[name])
                for name in fields
            )
        )
    if isinstance(template, tuple):
        if not isinstance(saved, (tuple, list)):
            raise TypeError("Stored tuple checkpoint value is not a sequence.")
        return tuple(
            _restore_tree_like(
                template[index] if index < len(template) else None, value
            )
            for index, value in enumerate(saved)
        )
    if isinstance(template, list):
        if not isinstance(saved, (tuple, list)):
            raise TypeError("Stored list checkpoint value is not a sequence.")
        return [
            _restore_tree_like(
                template[index] if index < len(template) else None, value
            )
            for index, value in enumerate(saved)
        ]
    if isinstance(template, Mapping):
        if not isinstance(saved, Mapping):
            raise TypeError("Stored mapping checkpoint value is not a mapping.")
        values = {
            key: _restore_tree_like(template.get(key), value)
            for key, value in saved.items()
        }
        try:
            return type(template)(values)
        except TypeError:
            return values
    return saved


def _policy_state_from_checkpoint(value: Any) -> PolicyState:
    if isinstance(value, PolicyState):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("Checkpoint policy state must be a mapping.")
    belief = value["belief"]
    if not isinstance(belief, GaussianMixtureBelief):
        if not isinstance(belief, Mapping):
            raise TypeError("Checkpoint belief state must be a mapping.")
        belief = GaussianMixtureBelief(**dict(belief))
    return PolicyState(
        task_carry=value["task_carry"],
        belief=belief,
        previous_observation=value["previous_observation"],
        previous_action=value["previous_action"],
        previous_reward=value["previous_reward"],
        episode_start=value["episode_start"],
    )


def _anchor_replay_from_checkpoint(value: Any) -> AnchorReplayState | None:
    if value is None or isinstance(value, AnchorReplayState):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("Checkpoint anchor replay must be a mapping.")
    raw_batch = value["batch"]
    if not isinstance(raw_batch, Mapping):
        raise TypeError("Checkpoint anchor replay batch must be a mapping.")
    batch = CounterfactualAnchorBatch(
        anchor_ids=raw_batch["anchor_ids"],
        rollout_flat_indexes=raw_batch["rollout_flat_indexes"],
        policy_states=_policy_state_from_checkpoint(raw_batch["policy_states"]),
        observations=raw_batch["observations"],
        partner_codes=raw_batch["partner_codes"],
        partner_sources=raw_batch["partner_sources"],
        fit_returns_by_action=raw_batch["fit_returns_by_action"],
        evaluation_returns_by_action=raw_batch["evaluation_returns_by_action"],
        partner_run_ids=raw_batch["partner_run_ids"],
        action_mask=raw_batch["action_mask"],
    )
    return AnchorReplayState(
        target_policy_epoch_id=value["target_policy_epoch_id"],
        target_policy_fingerprint=str(value["target_policy_fingerprint"]),
        batch=batch,
        item_count=value["item_count"],
        capacity=int(value["capacity"]),
    )


def _rehydrate_r3_checkpoint(
    saved: Any, *, template: TrainState
) -> TrainState:
    """Recover one exact r3 state while rejecting every older state schema."""

    if isinstance(saved, TrainState):
        return saved
    if not isinstance(saved, Mapping):
        raise TypeError("r3 checkpoint root must be a mapping.")
    expected = set(TrainState._fields)
    if set(saved) != expected:
        raise ValueError(
            "Checkpoint is not an exact r3 TrainState: "
            f"missing={sorted(expected.difference(saved))}, "
            f"extra={sorted(set(saved).difference(expected))}."
        )
    restored: dict[str, Any] = {}
    for name in TrainState._fields:
        field_template = getattr(template, name)
        if name == "anchor_training_replay":
            restored[name] = _anchor_replay_from_checkpoint(saved[name])
        elif (
            name == "last_qualified_generator_optimizer_state"
            and field_template is None
            and saved[name] is not None
        ):
            restored[name] = _restore_tree_like(
                template.generator_optimizer_state, saved[name]
            )
        else:
            restored[name] = _restore_tree_like(field_template, saved[name])
    return TrainState(**restored)


def _checkpoint_epoch_validation_update(update_count: Any) -> int:
    """Return the update whose labels are actually stored in a checkpoint.

    The next outer update may begin a new target-policy epoch, but that
    synchronization occurs at the top of the training loop.  Validating the
    restored state against ``update_count + 1`` incorrectly rejects every
    checkpoint saved at an epoch boundary and every completed no-op resume.
    """

    completed = int(np.asarray(update_count))
    if completed < 0:
        raise ValueError("Checkpoint update count cannot be negative.")
    return max(completed, 1)


def _validate_restored_r3_checkpoint(
    *, output: Path, step: int, state: TrainState
) -> None:
    path = output / "checkpoint_identity" / f"{int(step):012d}.json"
    if not path.is_file():
        raise RuntimeError("r3 checkpoint identity sidecar is missing.")
    observed = json.loads(path.read_text(encoding="utf-8"))
    expected = _state_resume_identity(state)
    if observed != expected:
        differing = sorted(
            key
            for key in set(observed) | set(expected)
            if observed.get(key) != expected.get(key)
        )
        raise RuntimeError(
            "Restored r3 checkpoint state identity differs: "
            f"fields={differing}."
        )


def _pool(runs: tuple[Any, ...]) -> FrozenPartnerPool:
    if not runs:
        raise ValueError("A required r3 partner resource group is empty.")
    return FrozenPartnerPool.from_checkpoints(
        [run.checkpoint for run in runs],
        parent_training_run_ids=[run.parent_training_run_id for run in runs],
    )


def _owned_runs(manifest: Any, role: str, seed_index: int) -> tuple[Any, ...]:
    return tuple(
        run for run in manifest.by_role(role)
        if run.owner_seed_index in (None, int(seed_index))
    )


def _source_code_anchors(count: int, dimension: int) -> Any:
    """Deterministic continuous anchors; labels never enter the online model."""

    import jax.numpy as jnp
    import numpy as np

    if count != 4 or dimension <= 0:
        raise ValueError("r3 generator initialization requires four source mechanisms.")
    row = np.arange(dimension, dtype=np.int32)
    anchors = np.stack(
        [1.0 - 2.0 * ((row >> bit) & 1) for bit in range(2)]
        + [-1.0 + 2.0 * ((row >> bit) & 1) for bit in range(2)],
        axis=0,
    ).astype(np.float32)
    return jnp.asarray(anchors)


def _slice_owner_batch(batch: Any, indexes: Any) -> Any:
    return batch._replace(
        observations=batch.observations[:, indexes],
        previous_actions=batch.previous_actions[:, indexes],
        episode_starts=batch.episode_starts[:, indexes],
        owner_logits=batch.owner_logits[:, indexes],
        valid_mask=batch.valid_mask[:, indexes],
        source_members=batch.source_members[:, indexes],
    )


def _distill_owner_base(
    *, model: Any, params: Any, batch: Any, config: Any,
    observation_shape: tuple[int, ...], key: Any,
) -> tuple[Any, Mapping[str, Any]]:
    """One registered Official 4x-minibatch behavioral initialization block."""

    import jax
    import jax.numpy as jnp
    import optax

    optimizer, optimizer_state = make_optimizer(
        params,
        learning_rate=config.ppo.learning_rate,
        gradient_clip_norm=config.ppo.gradient_clip_norm,
        adam_epsilon=config.ppo.adam_epsilon,
    )
    schedule = environment_minibatch_schedule(
        key,
        environment_count=int(batch.owner_logits.shape[1]),
        minibatches_per_epoch=config.training.minibatches_per_epoch,
        update_epochs=config.ppo.update_epochs,
    ).reshape((-1, int(batch.owner_logits.shape[1]) // config.training.minibatches_per_epoch))

    def one(carry: Any, indexes: Any) -> tuple[Any, Any]:
        current, opt_state = carry
        current_batch = _slice_owner_batch(batch, indexes)
        initial = initial_policy_state(
            batch_size=int(indexes.shape[0]), observation_shape=observation_shape,
            action_count=6, task_hidden_dim=config.model.task_hidden_dim,
            belief_hidden_dim=config.model.belief_hidden_dim,
            latent_dim=config.model.latent_dim,
            mixture_components=config.model.mixture_components,
        )

        def objective(candidate: Any) -> tuple[Any, Any]:
            return owner_behavior_loss(
                model=model, params=candidate, initial_state=initial, batch=current_batch
            )

        (loss, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(current)
        gradients = keep_owned_gradients(gradients, loss_name="base_distillation")
        updates, next_opt = optimizer.update(gradients, opt_state, current)
        return (optax.apply_updates(current, updates), next_opt), {**metrics, "loss": loss}

    (distilled, unused_state), metrics = jax.jit(
        lambda initial_params, initial_opt: jax.lax.scan(
            one, (initial_params, initial_opt), schedule
        )
    )(params, optimizer_state)
    del unused_state
    return distilled, jax.tree_util.tree_map(lambda value: value[-1], metrics)


def _base_behavior_contract(
    *, model: Any, params: Any, batch: Any, config: Any,
    observation_shape: tuple[int, ...],
) -> Mapping[str, Any]:
    """Held-out C0 behavior and exact gate-zero execution diagnostics."""

    import jax.numpy as jnp

    initial = initial_policy_state(
        batch_size=int(batch.owner_logits.shape[1]),
        observation_shape=observation_shape,
        action_count=6,
        task_hidden_dim=config.model.task_hidden_dim,
        belief_hidden_dim=config.model.belief_hidden_dim,
        latent_dim=config.model.latent_dim,
        mixture_components=config.model.mixture_components,
    )
    unused_loss, behavior = owner_behavior_loss(
        model=model,
        params=params,
        initial_state=initial,
        batch=batch,
    )
    del unused_loss
    consistency = gate_zero_consistency(
        model=model,
        params=params,
        initial_state=initial,
        batch=batch,
    )
    behavior_kl = jnp.asarray(behavior["base_distillation_kl"])
    residual_rms = jnp.asarray(consistency["conditional_residual_rms"])
    passed = (
        (behavior_kl <= 0.05)
        & jnp.asarray(consistency["passed"], dtype=jnp.bool_)
        & (residual_rms <= 1.0e-7)
    )
    return {
        **behavior,
        **consistency,
        "passed": passed,
        "heldout_behavior_kl_maximum": jnp.asarray(0.05, dtype=jnp.float32),
        "residual_rms_maximum_at_c0": jnp.asarray(1.0e-7, dtype=jnp.float32),
    }


def _distill_generator_sources(
    *, generator: Any, params: Any, batch: Any, config: Any,
    code_anchors: Any, key: Any,
) -> tuple[Any, Mapping[str, Any]]:
    import jax
    import jax.numpy as jnp
    import optax

    member_at_time = batch.source_members
    codes = code_anchors[member_at_time]
    starts = batch.episode_starts[:-1]
    observations = batch.observations[:-1]
    source_logits = batch.owner_logits
    mask = batch.valid_mask
    count = int(source_logits.shape[1])
    if count % config.partner_generator.environment_minibatches:
        raise ValueError("Generator source lanes do not divide into eight minibatches.")
    optimizer, optimizer_state = make_optimizer(
        params, learning_rate=config.ppo.learning_rate,
        gradient_clip_norm=config.ppo.gradient_clip_norm,
        adam_epsilon=config.ppo.adam_epsilon,
    )
    schedule = environment_minibatch_schedule(
        key, environment_count=count,
        minibatches_per_epoch=config.partner_generator.environment_minibatches,
        update_epochs=config.partner_generator.update_epochs,
    ).reshape((-1, count // config.partner_generator.environment_minibatches))

    def one(carry: Any, indexes: Any) -> tuple[Any, Any]:
        current, opt_state = carry
        initial_carry = initial_generator_carry(
            int(indexes.shape[0]), config.partner_generator.hidden_dim
        )
        keys = jnp.zeros(
            observations[:, indexes].shape[:2] + (2,), dtype=jnp.uint32
        )

        def objective(candidate: Any) -> Any:
            unused, output = generator.apply(
                {"params": candidate}, initial_carry,
                observations[:, indexes], codes[:, indexes], starts[:, indexes], keys,
                method=generator.sequence,
            )
            del unused
            return source_logit_distillation_loss(
                output.logits, source_logits[:, indexes], mask[:, indexes]
            )

        loss, gradients = jax.value_and_grad(objective)(current)
        updates, next_opt = optimizer.update(gradients, opt_state, current)
        return (optax.apply_updates(current, updates), next_opt), loss

    (distilled, unused_state), losses = jax.jit(
        lambda initial_params, initial_opt: jax.lax.scan(
            one, (initial_params, initial_opt), schedule
        )
    )(params, optimizer_state)
    del unused_state
    return distilled, {"generator_source_behavior_kl": losses[-1]}


def _generator_behavior_kl(
    *, generator: Any, params: Any, batch: Any, code_anchors: Any,
    hidden_dim: int,
) -> Any:
    """Held-out source-to-generator behavior KL on legal trajectories."""

    import jax.numpy as jnp

    members = batch.source_members
    codes = code_anchors[members]
    observations = batch.observations[:-1]
    starts = batch.episode_starts[:-1]
    keys = jnp.zeros(observations.shape[:2] + (2,), dtype=jnp.uint32)
    initial = initial_generator_carry(int(observations.shape[1]), int(hidden_dim))
    unused, output = generator.apply(
        {"params": params}, initial, observations, codes, starts, keys,
        method=generator.sequence,
    )
    del unused
    return source_logit_distillation_loss(
        output.logits, batch.owner_logits, batch.valid_mask
    )


def _generator_parameter_kl(
    *, generator: Any, reference_params: Any, candidate_params: Any,
    batch: Any, hidden_dim: int,
) -> Any:
    """Forward behavior KL from the last qualified generator to a candidate."""

    import jax
    import jax.numpy as jnp

    observations = batch.observations
    codes = batch.codes
    starts = jnp.zeros(batch.dones.shape, dtype=jnp.bool_).at[0].set(True)
    keys = jnp.zeros(observations.shape[:2] + (2,), dtype=jnp.uint32)
    initial = initial_generator_carry(int(observations.shape[1]), int(hidden_dim))
    unused, reference = generator.apply(
        {"params": reference_params}, initial, observations, codes, starts, keys,
        method=generator.sequence,
    )
    unused, candidate = generator.apply(
        {"params": candidate_params}, initial, observations, codes, starts, keys,
        method=generator.sequence,
    )
    del unused
    reference_log = jax.nn.log_softmax(reference.logits, axis=-1)
    candidate_log = jax.nn.log_softmax(candidate.logits, axis=-1)
    probability = jnp.exp(reference_log)
    pointwise = jnp.sum(
        probability * (reference_log - candidate_log), axis=-1
    )
    mask = jnp.asarray(batch.valid_mask, dtype=jnp.float32)
    return jnp.sum(mask * pointwise) / jnp.maximum(jnp.sum(mask), 1.0)


def _generator_interpolation_uniform_kl(
    *, generator: Any, params: Any, observations: Any, starts: Any,
    key: Any, code_dim: int, hidden_dim: int,
) -> Any:
    """Detect interpolation collapse to a uniform six-action policy."""

    import jax
    import jax.numpy as jnp

    time_count, lane_count = observations.shape[:2]
    codes = sample_partner_codes(
        key, batch_size=lane_count, code_dim=int(code_dim)
    )
    codes = jnp.broadcast_to(codes[None, ...], (time_count,) + codes.shape)
    action_keys = jnp.zeros((time_count, lane_count, 2), dtype=jnp.uint32)
    initial = initial_generator_carry(lane_count, int(hidden_dim))
    unused, output = generator.apply(
        {"params": params}, initial, observations, codes, starts, action_keys,
        method=generator.sequence,
    )
    del unused
    log_probability = jax.nn.log_softmax(output.logits, axis=-1)
    probability = jnp.exp(log_probability)
    uniform_log = -jnp.log(jnp.asarray(output.logits.shape[-1], dtype=jnp.float32))
    return jnp.mean(
        jnp.sum(probability * (log_probability - uniform_log), axis=-1)
    )


def _build_generator_update_kernel(
    *, generator: Any, optimizer: Any, config: Any,
) -> CompiledCallable:
    import jax
    import jax.numpy as jnp
    import optax

    episodes = int(config.partner_generator.episodes_per_update)
    minibatches = int(config.partner_generator.environment_minibatches)
    lane_count = episodes // minibatches
    if episodes % minibatches:
        raise ValueError("Generator episodes must divide into environment minibatches.")

    def execute(core: GeneratorCoreState, batch: Any, schedule: Any, diversity: Any) -> Any:
        def one(carry: GeneratorCoreState, indexes: Any) -> tuple[Any, Any]:
            observations = batch.observations[:, indexes]
            codes = batch.codes[:, indexes]
            starts = jnp.zeros(batch.dones[:, indexes].shape, dtype=jnp.bool_).at[0].set(True)
            keys = jnp.zeros(observations.shape[:2] + (2,), dtype=jnp.uint32)
            initial = jax.tree_util.tree_map(lambda value: value[indexes], batch.initial_carries)

            def objective(candidate: Any) -> tuple[Any, Any]:
                unused, output = generator.apply(
                    {"params": candidate}, initial, observations, codes, starts, keys,
                    method=generator.sequence,
                )
                del unused
                return generator_ppo_objective(
                    new_logits=output.logits, new_values=output.value,
                    actions=batch.actions[:, indexes],
                    behavior_log_probabilities=batch.behavior_log_probabilities[:, indexes],
                    official_shaped_rewards=batch.official_shaped_rewards[:, indexes],
                    dones=batch.dones[:, indexes], valid_mask=batch.valid_mask[:, indexes],
                    gamma=config.ppo.gamma, gae_lambda=config.ppo.gae_lambda,
                    clip_epsilon=config.ppo.clip_epsilon,
                    value_weight=config.ppo.value_weight,
                    entropy_weight=config.ppo.entropy_weight,
                    diversity_episode_bonus=diversity[indexes],
                )

            (loss, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(carry.params)
            updates, opt_state = optimizer.update(gradients, carry.optimizer_state, carry.params)
            params = optax.apply_updates(carry.params, updates)
            return GeneratorCoreState(params, carry.target_params, opt_state), {**metrics, "loss": loss}

        return jax.lax.scan(one, core, schedule.reshape((-1, lane_count)))

    return CompiledCallable("generator_complete_episode_ppo", execute, donate_argnums=(0,))


def _empty_anchor(batch: Any) -> CounterfactualAnchorBatch:
    import jax
    import jax.numpy as jnp

    policy = jax.tree_util.tree_map(lambda value: value[:1], batch.initial_policy_state)
    return CounterfactualAnchorBatch(
        anchor_ids=jnp.zeros((1,), dtype=jnp.int64),
        rollout_flat_indexes=jnp.zeros((1,), dtype=jnp.int32),
        policy_states=policy,
        observations=batch.observations[0, :1],
        partner_codes=batch.partner_codes[0, :1],
        partner_sources=batch.partner_sources[0, :1],
        fit_returns_by_action=jnp.zeros((1, 6), dtype=jnp.float32),
        evaluation_returns_by_action=jnp.full((1, 6), jnp.nan, dtype=jnp.float32),
        partner_run_ids=batch.partner_run_ids[0, :1],
        action_mask=jnp.zeros((1, 6), dtype=jnp.float32),
    )


def _partner_block_means(values: Any, partner_ids: Any) -> tuple[float, ...]:
    import numpy as np

    sample = np.asarray(values, dtype=np.float64)
    ids = np.asarray(partner_ids)
    return tuple(float(np.mean(sample[ids == item])) for item in np.unique(ids))


def _fit_code_signature_readout(codes: Any, signatures: Any, ridge: float = 1.0) -> Any:
    import numpy as np

    x = np.asarray(codes, dtype=np.float64)
    y = np.asarray(signatures, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("Code/signature audit rows are misaligned.")
    design = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
    return np.linalg.solve(
        design.T @ design + float(ridge) * np.eye(design.shape[1]),
        design.T @ y,
    )


def _generator_diversity_bonus(
    *, codes: Any, readout: Any | None, config: Any,
) -> Any:
    import jax.numpy as jnp

    episode_codes = jnp.asarray(codes)[0]
    if readout is None:
        return jnp.zeros((episode_codes.shape[0],), dtype=jnp.float32)
    weights = jnp.asarray(readout, dtype=jnp.float32)
    design = jnp.concatenate(
        (episode_codes, jnp.ones((episode_codes.shape[0], 1), dtype=jnp.float32)),
        axis=-1,
    )
    signatures = design @ weights
    diversity = brdiv_marginal_contributions(
        signatures,
        bandwidth=config.partner_generator.kernel_bandwidth,
        jitter=config.partner_generator.kernel_jitter,
    )
    smoothness = smoothness_marginal_penalties(episode_codes, signatures)
    return (
        config.partner_generator.brdiv_weight * diversity
        - config.partner_generator.smoothness_weight * smoothness
    )


def _anchor_model_outputs(model: Any, params: Any, anchors: Any) -> Any:
    import jax.numpy as jnp

    unused, output = model.apply(
        {"params": params}, anchors.policy_states, anchors.observations,
        jnp.ones(anchors.anchor_ids.shape, dtype=jnp.float32), method=model.step,
    )
    del unused
    return output


def _anchor_predicted_regret(
    *, model: Any, params: Any, anchors: Any, particle_count: int, key: Any,
) -> Any:
    """Registered posterior-particle regret on independent audit anchors."""

    import jax
    import jax.numpy as jnp

    from src.path_c.belief_set_encoder import stratified_mixture_samples
    from src.path_c.regret_potential import decision_regret_from_action_values

    output = _anchor_model_outputs(model, params, anchors)
    keys = jax.random.split(key, int(anchors.anchor_ids.shape[0]))
    samples, weights = jax.vmap(
        lambda lane_key, logits, means, log_variances: stratified_mixture_samples(
            lane_key,
            mixture_logits=logits,
            means=means,
            log_variances=log_variances,
            sample_count=int(particle_count),
        )
    )(
        keys,
        output.mixture_logits,
        output.mixture_means,
        output.mixture_log_variances,
    )
    features = jnp.broadcast_to(
        output.task_features[:, None, :],
        (
            output.task_features.shape[0],
            int(particle_count),
            output.task_features.shape[-1],
        ),
    )
    values = model.apply(
        {"params": params},
        features,
        samples,
        method=model.action_values_from_features_and_latent,
    )
    return decision_regret_from_action_values(values, weights)


def _audit_signal_contracts(
    *, qualification: SignalQualification, anchors: Any, partner_ids: Any,
    model: Any, params: Any, target_epoch: Any, seed: int,
    random_domain: str, artifact_fingerprint: str,
) -> tuple[SignalQualification, Mapping[str, Any]]:
    """Evaluate C1--C2 only from independent fit/evaluation continuations."""

    import numpy as np

    fit = np.asarray(anchors.fit_returns_by_action, dtype=np.float64)
    evaluation = np.asarray(anchors.evaluation_returns_by_action, dtype=np.float64)
    ids = np.asarray(partner_ids)
    output = _anchor_model_outputs(model, target_epoch.target_params, anchors)
    q = np.asarray(conservative_raw_q(output.raw_q1, output.raw_q2))
    base_logits = np.asarray(output.base_logits)
    base_action = np.argmax(base_logits, axis=-1)
    oracle_action = np.argmax(fit, axis=-1)
    q_action = np.argmax(q, axis=-1)
    row = np.arange(fit.shape[0])
    oracle_lift = evaluation[row, oracle_action] - evaluation[row, base_action]
    q_lift = evaluation[row, q_action] - evaluation[row, base_action]
    signatures_fit = centered_action_values(fit)
    signatures_eval = centered_action_values(evaluation)
    posterior = np.asarray(output.belief_embedding)

    metrics: dict[str, Any] = {}
    if qualification.passed("C0"):
        ordering = leave_one_partner_out_ordering_accuracy(
            features=posterior, fit_signatures=signatures_fit,
            evaluation_signatures=signatures_eval, partner_ids=ids,
        )
        stable = [int(np.argmax(np.mean(fit[ids == item], axis=0))) for item in np.unique(ids)]
        partner_count = int(np.unique(ids).size)
        if fit.shape[0] % partner_count:
            raise ValueError("C1 common-world rows do not divide by partner runs.")
        common_world_ids = np.repeat(
            np.arange(fit.shape[0] // partner_count), partner_count
        )
        c1 = decision_support_audit(
            signatures=signatures_fit, partner_ids=ids,
            oracle_lift_blocks=_partner_block_means(oracle_lift, ids),
            stable_best_actions=stable,
            heldout_ordering_blocks=ordering,
            seed=seed,
            common_world_ids=common_world_ids,
        )
        decision = compound_gate_decision(
            passed=c1.passed,
            statistic=c1.between_partner_variance - c1.within_partner_variance,
            lower_bound=min(c1.variance_difference_lcb, c1.oracle_lift_lcb, c1.heldout_ordering_accuracy_lcb - 0.5),
            upper_bound=max(c1.between_partner_variance, c1.oracle_lift_lcb, c1.heldout_ordering_accuracy_lcb),
            sample_count=int(np.unique(ids).size), random_domain=random_domain,
            artifact_fingerprint=artifact_fingerprint,
            reason="C1 variance/oracle/action-region/LOPO subgates",
        )
        qualification = qualification.with_decision("C1", decision)
        metrics["C1"] = {**asdict(c1), "decision": asdict(decision)}

    if qualification.passed("C1"):
        significant = []
        concordance = []
        for partner in np.unique(ids):
            mask = ids == partner
            correct = []
            for predicted, truth in zip(q[mask], evaluation[mask]):
                for left in range(6):
                    for right in range(left + 1, 6):
                        if abs(truth[left] - truth[right]) < 1.0:
                            continue
                        correct.append(float((predicted[left] - predicted[right]) * (truth[left] - truth[right]) > 0))
            concordance.append(0.5 if not correct else float(np.mean(correct)))
            significant.append(float(np.mean(q_lift[mask])))
        errors = q - evaluation
        standard_error = np.nanstd(evaluation - fit, axis=0).max()
        c2 = raw_q_audit(
            selected_action_lift_blocks=significant,
            pairwise_concordance_blocks=concordance,
            prediction_errors=errors.reshape(-1),
            monte_carlo_error_bound=float(standard_error),
            fit_rankings=np.argsort(fit, axis=-1),
            evaluation_rankings=np.argsort(evaluation, axis=-1),
            seed=seed + 10,
        )
        decision = compound_gate_decision(
            passed=c2.passed, statistic=c2.selected_action_lift_lcb,
            lower_bound=min(c2.selected_action_lift_lcb, c2.pairwise_concordance_lcb - 0.5, c2.monte_carlo_error_bound - c2.calibration_error),
            upper_bound=max(c2.selected_action_lift_lcb, c2.pairwise_concordance_lcb, c2.monte_carlo_error_bound),
            sample_count=len(significant), random_domain=random_domain,
            artifact_fingerprint=artifact_fingerprint,
            reason="C2 Q-selection/concordance/calibration/epoch-stability subgates",
        )
        qualification = qualification.with_decision("C2", decision)
        metrics["C2"] = {**asdict(c2), "decision": asdict(decision)}

    return qualification, metrics


def _qualify_c3(
    *, qualification: SignalQualification, paired: Any, seed: int,
    random_domain: str, artifact_fingerprint: str,
) -> tuple[SignalQualification, Mapping[str, Any]]:
    """C3 uses paired complete episodes, never anchor/Q surrogates."""

    import numpy as np

    ids = np.asarray(paired.partner_run_ids)
    oracle = np.asarray(paired.oracle_context) - np.asarray(paired.state_only)
    online = np.asarray(paired.online_context) - np.asarray(paired.state_only)
    _, oracle_lcb, oracle_ucb = bootstrap_interval(
        _partner_block_means(oracle, ids), seed=seed
    )
    _, online_lcb, online_ucb = bootstrap_interval(
        _partner_block_means(online, ids), seed=seed + 1
    )
    audit = InferenceAudit(oracle_lcb, online_lcb)
    decision = compound_gate_decision(
        passed=audit.passed,
        statistic=online_lcb,
        lower_bound=min(oracle_lcb, online_lcb),
        upper_bound=max(oracle_ucb, online_ucb),
        sample_count=int(np.unique(ids).size),
        random_domain=random_domain,
        artifact_fingerprint=artifact_fingerprint,
        reason="C3 paired oracle/online/state-only full-episode subgates",
    )
    return qualification.with_decision("C3", decision), {
        **asdict(audit),
        "recovery_ratio": audit.recovery_ratio,
        "decision": asdict(decision),
    }


def _qualify_c4(
    *, qualification: SignalQualification, paired: Any, diagnostics: Any,
    params: Any, seed: int, random_domain: str, artifact_fingerprint: str,
) -> tuple[SignalQualification, Mapping[str, Any]]:
    import numpy as np

    differences = np.asarray(paired.candidate) - np.asarray(paired.reference)
    ids = np.asarray(paired.partner_run_ids)
    blocks = _partner_block_means(differences, ids)
    _, lift_lcb, _ = bootstrap_interval(blocks, seed=seed)
    negative = float(np.mean(differences < 0.0))
    steps = 400.0
    entropy_sum, kl_sum = (np.asarray(item) for item in diagnostics)
    mean_kl = float(np.mean(kl_sum / steps))
    # Residual RMS is read from parameters, not inferred from return.
    actor = params["universal_actor"]
    residual_leaves = []
    import jax
    for path, value in jax.tree_util.tree_flatten_with_path(actor)[0]:
        name = "/".join(str(getattr(item, "key", item)) for item in path)
        if "context_residual" in name:
            residual_leaves.append(np.asarray(value).reshape(-1))
    residual_rms = 0.0 if not residual_leaves else float(np.sqrt(np.mean(np.square(np.concatenate(residual_leaves)))))
    audit = ConditionalControlAudit(lift_lcb, negative, mean_kl, residual_rms)
    decision = compound_gate_decision(
        passed=audit.passed, statistic=lift_lcb,
        lower_bound=min(lift_lcb, 0.05 - negative, 0.03 - mean_kl, 1.0 - residual_rms),
        upper_bound=max(lift_lcb, 0.05 - negative, 0.03 - mean_kl, 1.0 - residual_rms),
        sample_count=len(blocks), random_domain=random_domain,
        artifact_fingerprint=artifact_fingerprint,
        reason="C4 full-episode lift/negative-transfer/KL/residual subgates",
    )
    return qualification.with_decision("C4", decision), {**asdict(audit), "decision": asdict(decision)}


def _qualification_mapping(value: Any) -> SignalQualification:
    if isinstance(value, SignalQualification):
        return value
    return SignalQualification.from_mapping(value or {})


def _training_anchor_steps(config: Any, *, matched: bool = True) -> int:
    return r3_training_anchor_attempted_steps(
        trigger_count=1,
        ordinary_candidates=config.anchors.pilot_ordinary_candidates,
        matched_code_candidates=(
            config.anchors.pilot_matched_code_candidates if matched else 0
        ),
        selected_ordinary=config.anchors.selected_ordinary,
        selected_matched_code=(
            config.anchors.selected_matched_code if matched else 0
        ),
        action_count=6,
        pilot_replicas=config.anchors.pilot_replicas,
        fit_replicas=config.anchors.fit_replicas,
        continuation_horizon=config.anchors.continuation_horizon,
        probe_steps=config.anchors.probe_steps,
    )


def _write_final_support_latents(
    *, output: Path, environment: Any, model: Any, state: TrainState,
    config: Any, partner_functions: Any, partner_parameters: Any, kernels: Any,
    key: Any,
) -> Mapping[str, Any]:
    import numpy as np
    import jax.numpy as jnp

    path = output / "records" / "training_support_latents.npz"
    metadata_path = output / "records" / "training_support_metadata.json"
    if path.is_file() != metadata_path.is_file():
        raise RuntimeError("Final support cache is incomplete; refusing to recollect over it.")
    if path.is_file():
        cached = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            cached.get("model_fingerprint")
            != pytree_fingerprint(state.params)
            or cached.get("config_fingerprint") != config.fingerprint
            or cached.get("artifact_sha256") != sha256_path(path)
        ):
            raise ValueError("Final support cache identity differs from the restored run.")
        return cached

    support_runner = initialize_runner(
        environment=environment, model_config=config.model,
        partner_functions=partner_functions, random_key=key,
    )
    support_runner, unused_batch, records = kernels.rollout_support(
        support_runner, state.params, state.target_policy_epoch.target_params,
        partner_parameters,
        jnp.ones((environment.num_envs,), dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    del unused_batch
    mean, variance = mixture_moments(
        records["mixture_logits"], records["mixture_means"],
        records["mixture_log_variances"],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        posterior_mean=np.asarray(mean).reshape((-1, mean.shape[-1])),
        posterior_variance=np.asarray(variance).reshape((-1, variance.shape[-1])),
        partner_source=np.asarray(records["partner_source"]).reshape((-1,)),
        partner_run_id=np.asarray(records["partner_run_ids"]).reshape((-1,)),
    )
    metadata = {
        "path": str(path),
        "row_count": int(mean.shape[0] * mean.shape[1]),
        "environment_steps": int(environment.num_envs * config.environment.episode_steps),
        "completed_episodes": int(np.asarray(support_runner.completed_episodes)),
        "model_fingerprint": pytree_fingerprint(state.params),
        "config_fingerprint": config.fingerprint,
        "artifact_sha256": sha256_path(path),
    }
    write_json(metadata_path, metadata)
    return metadata


def run_training(args: argparse.Namespace) -> None:
    import numpy as np

    started = time.perf_counter()
    scope = str(getattr(args, "_execution_scope", "training"))
    preflight = scope == CUDA_PREFLIGHT_SCOPE
    config = load_config(args.config, run_kind=args.run_kind)
    cuda_toolchain = None
    if config.run_kind == "formal" or os.environ.get("DELTA_REQUIRE_CUDA") == "1":
        cuda_toolchain = configure_bundled_cuda_toolchain()
    import jax
    import jax.numpy as jnp

    if config.run_kind == "formal":
        validate_formal_repository_state()
        validate_registered_python_runtime()
    cuda = None
    if config.run_kind == "formal" or os.environ.get("DELTA_REQUIRE_CUDA") == "1":
        cuda = require_single_cuda_worker()
    official_runtime = validate_official_runtime() if config.run_kind == "formal" else None
    manifest = load_partner_manifest(
        args.partner_manifest, expected_layout=config.environment.layout,
        verify_files=not bool(args.skip_manifest_hash_check),
    )
    validate_seed_signal_contract_manifest(
        manifest,
        owner_seed_index=int(args.seed_index),
        formal=(config.run_kind == "formal" and not preflight),
    )
    output = Path(args.output).resolve()
    environment = VectorEnvironment.create(config)
    observation_shape = environment.observation_shape
    outer_key = (
        engineering_training_key()
        if int(args.seed_index) == ENGINEERING_SEED_INDEX
        else official_training_key(int(args.seed_index))
    )
    domains = official_training_domain_keys(int(args.seed_index))
    identity = dict(training_identity(
        config=config, seed_index=int(args.seed_index), jax_prng_key=outer_key,
        partner_manifest=manifest,
    ))
    structure_fingerprint = _model_structure_fingerprint(config, observation_shape)
    compilation_cache = configure_persistent_compilation_cache(
        repository_commit=str(identity["runtime"]["repository_commit"]),
        official_commit=OFFICIAL_SOURCE_COMMIT,
        config_fingerprint=config.fingerprint,
        model_structure_fingerprint=structure_fingerprint,
        cache_root=os.environ.get("DELTA_JAX_COMPILATION_CACHE"),
    )
    identity.update({
        "ego_run_id": str(args.ego_run_id),
        "observation_shape": list(observation_shape),
        "action_count": 6,
        "official_runtime": official_runtime,
        "formal_cuda_worker": cuda,
        "formal_cuda_toolchain": cuda_toolchain,
        "execution_scope": scope,
        "model_structure_fingerprint": structure_fingerprint,
        "compilation_cache": compilation_cache,
        "scientific_readout_allowed": False,
    })
    ensure_run_identity(output, identity)
    write_json(output / "resolved_config.json", config.to_mapping())
    write_json(output / "resolved_partner_manifest.json", manifest.to_mapping())
    write_json(output / "runtime_gpu.json", cuda or _gpu_snapshot())

    owner_runs = _owned_runs(manifest, "owner_source", int(args.seed_index))
    init_runs = _owned_runs(manifest, "generator_init_source", int(args.seed_index))
    development_runs = _owned_runs(manifest, "development_support", int(args.seed_index))
    owner_pool = _pool(owner_runs)
    init_pool = _pool(init_runs)
    development_pool = _pool(development_runs)

    model = build_model(
        observation_shape=observation_shape, action_count=6,
        **{name: getattr(config.model, name) for name in (
            "task_hidden_dim", "belief_hidden_dim", "latent_dim",
            "mixture_components", "belief_embedding_dim", "actor_hidden_dim",
            "critic_hidden_dim", "response_hidden_dim", "modulation_rank",
            "action_embedding_dim", "log_variance_minimum", "log_variance_maximum",
        )},
    )
    generator = build_partner_generator(
        observation_shape=observation_shape, action_count=6,
        code_dim=config.partner_generator.code_dim,
        hidden_dim=config.partner_generator.hidden_dim,
        modulation_rank=config.partner_generator.modulation_rank,
    )
    ego_root = jnp.asarray(domains["ego"], dtype=jnp.uint32)
    generator_root = jnp.asarray(domains["generator"], dtype=jnp.uint32)
    reset_key, model_key, runner_key, state_key, distill_key = jax.random.split(ego_root, 5)
    unused, initial_observations = environment.reset(reset_key)
    del unused
    example_state = initial_policy_state(
        batch_size=environment.num_envs, observation_shape=observation_shape,
        action_count=6, task_hidden_dim=config.model.task_hidden_dim,
        belief_hidden_dim=config.model.belief_hidden_dim,
        latent_dim=config.model.latent_dim,
        mixture_components=config.model.mixture_components,
    )
    params = initialize_model_parameters(
        model, key=model_key, example_state=example_state,
        example_observation=initial_observations[:, 0],
        partner_code_dim=config.partner_generator.code_dim,
    )
    generator_params = initialize_generator_parameters(
        generator, key=jax.random.fold_in(generator_root, 0),
        observation_shape=observation_shape,
        code_dim=config.partner_generator.code_dim,
        batch_size=config.partner_generator.episodes_per_update,
        hidden_dim=config.partner_generator.hidden_dim,
    )

    # r3 never starts either controller from an unqualified random behavior
    # manifold.  The distillation blocks occur before all four live optimizers.
    owner_batch = collect_owner_behavior(
        environment=environment, owner_pool=owner_pool,
        partner_pool=development_pool, length=config.training.rollout_length,
        key=jax.random.fold_in(distill_key, 1),
    )
    params, base_distill_metrics = _distill_owner_base(
        model=model, params=params, batch=owner_batch, config=config,
        observation_shape=observation_shape, key=jax.random.fold_in(distill_key, 2),
    )
    owner_holdout_batch = collect_owner_behavior(
        environment=environment,
        owner_pool=owner_pool,
        partner_pool=development_pool,
        length=config.training.rollout_length,
        key=jax.random.fold_in(distill_key, 3),
    )
    base_behavior_contract = _base_behavior_contract(
        model=model,
        params=params,
        batch=owner_holdout_batch,
        config=config,
        observation_shape=observation_shape,
    )
    generator_environment = VectorEnvironment(
        environment.environment,
        int(config.partner_generator.episodes_per_update),
        int(config.partner_generator.episode_steps),
    )
    init_members = jnp.tile(
        jnp.arange(4, dtype=jnp.int32),
        config.partner_generator.episodes_per_update // 4,
    )
    generator_source_batch = collect_owner_behavior(
        environment=generator_environment, owner_pool=init_pool,
        partner_pool=owner_pool, length=config.partner_generator.episode_steps,
        key=jax.random.fold_in(generator_root, 1), owner_members=init_members,
    )
    code_anchors = _source_code_anchors(4, config.partner_generator.code_dim)
    generator_params, generator_distill_metrics = _distill_generator_sources(
        generator=generator, params=generator_params, batch=generator_source_batch,
        config=config, code_anchors=code_anchors,
        key=jax.random.fold_in(generator_root, 2),
    )
    generator_holdout_batch = collect_owner_behavior(
        environment=generator_environment,
        owner_pool=init_pool,
        partner_pool=owner_pool,
        length=config.partner_generator.episode_steps,
        key=jax.random.fold_in(generator_root, 3),
        owner_members=init_members,
    )
    initial_behavior_kl = float(np.asarray(_generator_behavior_kl(
        generator=generator,
        params=generator_params,
        batch=generator_holdout_batch,
        code_anchors=code_anchors,
        hidden_dim=config.partner_generator.hidden_dim,
    )))
    initial_uniform_kl = float(np.asarray(_generator_interpolation_uniform_kl(
        generator=generator,
        params=generator_params,
        observations=generator_holdout_batch.observations[:-1],
        starts=generator_holdout_batch.episode_starts[:-1],
        key=jax.random.fold_in(generator_root, 4),
        code_dim=config.partner_generator.code_dim,
        hidden_dim=config.partner_generator.hidden_dim,
    )))
    initialization_codes = sample_partner_codes(
        jax.random.fold_in(generator_root, 5),
        batch_size=config.partner_generator.episodes_per_update,
        code_dim=config.partner_generator.code_dim,
    )
    initialization_competence = paired_generator_external_returns(
        environment=generator_environment,
        model=model,
        ego_params=params,
        model_config=config.model,
        generator=generator,
        generator_params=generator_params,
        partner_pool=development_pool,
        codes=initialization_codes,
        key=jax.random.fold_in(generator_root, 6),
    )
    initialization_statistics = paired_mean_cvar_statistics(
        np.asarray(initialization_competence.candidate)
        - np.asarray(initialization_competence.reference),
        partner_run_ids=np.asarray(initialization_competence.partner_run_ids),
        cvar_level=config.partner_generator.cvar_level,
        seed=6_000 + int(args.seed_index),
    )
    generator_initialization_qualified = bool(
        initial_behavior_kl
        <= config.partner_generator.source_distillation_kl_maximum
        and initial_uniform_kl
        >= config.partner_generator.interpolation_uniform_kl_minimum
        and initialization_statistics.mean_lcb
        >= -config.partner_generator.delivery_noninferiority_margin
        and initialization_statistics.cvar_lcb
        >= -config.partner_generator.delivery_noninferiority_margin
    )
    write_json(output / "qualification" / "generator_initialization.json", {
        "passed": generator_initialization_qualified,
        "heldout_behavior_kl": initial_behavior_kl,
        "interpolation_uniform_kl": initial_uniform_kl,
        "interpolation_uniform_kl_minimum": (
            config.partner_generator.interpolation_uniform_kl_minimum
        ),
        "paired_competence": asdict(initialization_statistics),
        "delivery_noninferiority_margin": (
            config.partner_generator.delivery_noninferiority_margin
        ),
        "source_runs": [run.run_id for run in init_runs],
        "scientific_readout_allowed": False,
    })

    total_updates = config.training.environment_steps // (
        config.environment.num_envs * config.training.rollout_length
    )
    ppo_optimizer, ppo_state = make_optimizer(
        params, learning_rate=config.ppo.learning_rate,
        gradient_clip_norm=config.ppo.gradient_clip_norm,
        adam_epsilon=config.ppo.adam_epsilon,
        anneal_learning_rate=config.ppo.anneal_learning_rate,
        warmup_fraction=config.ppo.lr_warmup_fraction,
        update_count=total_updates,
        minibatches_per_epoch=config.training.minibatches_per_epoch,
        update_epochs=config.ppo.update_epochs,
    )
    raw_q_optimizer, raw_q_state = make_optimizer(
        params, learning_rate=config.ppo.learning_rate,
        gradient_clip_norm=config.ppo.gradient_clip_norm,
        adam_epsilon=config.ppo.adam_epsilon,
    )
    response_optimizer, response_state = make_optimizer(
        params, learning_rate=config.ppo.learning_rate,
        gradient_clip_norm=config.ppo.gradient_clip_norm,
        adam_epsilon=config.ppo.adam_epsilon,
    )
    generator_optimizer, generator_state = make_optimizer(
        generator_params, learning_rate=config.ppo.learning_rate,
        gradient_clip_norm=config.ppo.gradient_clip_norm,
        adam_epsilon=config.ppo.adam_epsilon,
    )
    epoch = start_target_policy_epoch(
        params=params, update_number=1,
        updates_per_epoch=config.training.target_policy_epoch_updates,
        fingerprint_function=pytree_fingerprint,
    )

    snapshot_archive: list[Any] = []
    generator_admission_passes = 0
    source_probabilities = admitted_source_probabilities(0)

    def runtime(probabilities: tuple[float, float, float]) -> tuple[Any, Any, Any, Any]:
        functions = make_mixed_partner_functions(
            generator=generator,
            generator_hidden_dim=config.partner_generator.hidden_dim,
            generator_code_dim=config.partner_generator.code_dim,
            snapshot_count=len(snapshot_archive),
            external_pool=development_pool,
            current_probability=probabilities[0],
            snapshot_probability=probabilities[1],
            frozen_external_probability=probabilities[2],
        )
        kernels = build_training_kernels(
            environment=environment, model=model, model_config=config.model,
            partner_functions=functions, config=config, optimizer=ppo_optimizer,
            raw_q_optimizer=raw_q_optimizer, response_optimizer=response_optimizer,
        )
        anchor_functions = make_anchor_functions(
            model=model, model_config=config.model,
            partner_functions=functions, environment=environment,
        )
        training_anchor_kernel = build_anchor_chunk_kernel(
            functions=anchor_functions, config=config,
            memory_limit_bytes=FORMAL_PEAK_MEMORY_LIMIT_BYTES, mode="training",
        )
        audit_anchor_kernel = build_anchor_chunk_kernel(
            functions=anchor_functions, config=config,
            memory_limit_bytes=FORMAL_PEAK_MEMORY_LIMIT_BYTES, mode="audit",
        )
        return functions, kernels, anchor_functions, (training_anchor_kernel, audit_anchor_kernel)

    partner_functions, kernels, anchor_functions, anchor_kernels = runtime(source_probabilities)
    runner = initialize_runner(
        environment=environment, model_config=config.model,
        partner_functions=partner_functions, random_key=runner_key,
    )
    qualification = SignalQualification()
    owner_artifact = {
        "checkpoint": str(owner_runs[0].checkpoint),
        "sha256": owner_runs[0].checkpoint_sha256,
        "parent_training_run_id": owner_runs[0].parent_training_run_id,
    }
    state = TrainState(
        params=params,
        target_params=clone_epoch_target_for_live_control(epoch),
        ppo_optimizer_state=ppo_state,
        raw_q_optimizer_state=raw_q_state,
        response_optimizer_state=response_state,
        generator_optimizer_state=generator_state,
        generator_params=generator_params,
        generator_target_params=jax.tree_util.tree_map(jnp.copy, generator_params),
        target_policy_epoch=epoch,
        qualified_base_params=None,
        owner_source_artifact=owner_artifact,
        last_qualified_generator_params=(
            jax.tree_util.tree_map(jnp.copy, generator_params)
            if generator_initialization_qualified
            else None
        ),
        last_qualified_generator_optimizer_state=(
            generator_state if generator_initialization_qualified else None
        ),
        last_qualified_generator_optimizer_step=jnp.asarray(0, dtype=jnp.int64),
        generator_signature_readout=None,
        generator_snapshot_archive=tuple(),
        generator_admission_passes=jnp.asarray(0, dtype=jnp.int32),
        partner_source_probabilities=jnp.asarray(
            source_probabilities, dtype=jnp.float32
        ),
        qualification=qualification.to_mapping(),
        curriculum_phase=jnp.asarray(CurriculumPhase.BOOTSTRAP_BASE, dtype=jnp.int32),
        anchor_training_replay=None,
        anchor_audit_manifest=AnchorAuditManifest(
            expected_fit_replicas=config.anchors.audit_fit_replicas,
            expected_evaluation_replicas=config.anchors.audit_evaluation_replicas,
            expected_continuation_horizon=config.anchors.audit_continuation_horizon,
        ),
        kl_multiplier=jnp.asarray(0.0, dtype=jnp.float32),
        residual_multiplier=jnp.asarray(0.0, dtype=jnp.float32),
        raw_q_calibration_error=jnp.asarray(0.0, dtype=jnp.float32),
        raw_q_minimum_margin=jnp.asarray(jnp.inf, dtype=jnp.float32),
        ppo_optimizer_step=jnp.asarray(0, dtype=jnp.int64),
        raw_q_optimizer_step=jnp.asarray(0, dtype=jnp.int64),
        response_optimizer_step=jnp.asarray(0, dtype=jnp.int64),
        generator_optimizer_step=jnp.asarray(0, dtype=jnp.int64),
        runner_state=runner,
        random_key=state_key,
        update_count=jnp.asarray(0, dtype=jnp.int64),
        effective_environment_steps=jnp.asarray(0, dtype=jnp.int64),
        random_domains={name: jnp.asarray(0, dtype=jnp.int64) for name in domains},
        resource_ledger=ResourceLedger(
            state_collection_steps=(
                int(owner_batch.owner_logits.shape[0] * owner_batch.owner_logits.shape[1])
                + int(owner_holdout_batch.owner_logits.shape[0] * owner_holdout_batch.owner_logits.shape[1])
                + int(generator_source_batch.owner_logits.shape[0] * generator_source_batch.owner_logits.shape[1])
                + int(generator_holdout_batch.owner_logits.shape[0] * generator_holdout_batch.owner_logits.shape[1])
            ),
            base_distillation_steps=int(
                owner_batch.owner_logits.shape[0] * owner_batch.owner_logits.shape[1]
            ),
        ).to_mapping(),
        calibration=empty_calibration(
            latent_dim=config.model.latent_dim, alpha=config.calibration.alpha,
            model_fingerprint=pytree_fingerprint(deployable_parameters(params)),
        ),
    )

    manager = orbax_manager(output / "checkpoints")
    if manager.latest_step() is not None and not args.resume:
        raise RuntimeError("Output already contains a checkpoint; use --resume.")
    if args.resume:
        try:
            restored = restore_latest_checkpoint(manager)
        except Exception as error:
            raise RuntimeError("r2 or identity-incompatible checkpoints cannot resume r3.") from error
        if restored is not None:
            restored_step, saved_state = restored
            try:
                state = _rehydrate_r3_checkpoint(saved_state, template=state)
            except Exception as error:
                raise RuntimeError(
                    "r2 or identity-incompatible checkpoints cannot resume r3."
                ) from error
            state = state._replace(
                anchor_audit_manifest=audit_manifest_from_mapping(
                    state.anchor_audit_manifest
                )
            )
            _validate_restored_r3_checkpoint(
                output=output, step=int(restored_step), state=state
            )
            # Orbax restores value trees, not the ownership contract required
            # by JAX donation.  Detach both target copies so the moving Polyak
            # tree can be donated without invalidating the frozen epoch labels.
            restored_epoch = state.target_policy_epoch._replace(
                target_params=jax.tree_util.tree_map(
                    lambda value: value.copy(),
                    state.target_policy_epoch.target_params,
                )
            )
            state = state._replace(
                target_policy_epoch=restored_epoch,
                target_params=jax.tree_util.tree_map(
                    lambda value: value.copy(), state.target_params
                ),
            )
            qualification = _qualification_mapping(state.qualification)
            epoch = state.target_policy_epoch
            snapshot_archive = list(state.generator_snapshot_archive)
            generator_admission_passes = int(
                np.asarray(state.generator_admission_passes)
            )
            source_probabilities = tuple(
                float(value)
                for value in np.asarray(state.partner_source_probabilities)
            )
            (
                partner_functions,
                kernels,
                anchor_functions,
                anchor_kernels,
            ) = runtime(source_probabilities)
            validate_epoch_alignment(
                epoch,
                update_number=_checkpoint_epoch_validation_update(
                    state.update_count
                ),
                updates_per_epoch=config.training.target_policy_epoch_updates,
            )
            runner = state.runner_state

    # C0 is a paired full-episode audit, never a loss threshold.  The same
    # episode keys are used by distilled base and exact owner-SP.
    if not qualification.passed("C0") and not preflight:
        paired = paired_owner_base_returns(
            environment=environment, model=model, params=state.params,
            model_config=config.model, owner_pool=owner_pool,
            partner_pool=development_pool,
            episodes_per_partner=config.calibration.episodes_per_run,
            key=jnp.asarray(domains["qualification"], dtype=jnp.uint32),
        )
        return_decision = paired_mean_cvar_noninferiority_decision(
            np.asarray(paired.candidate) - np.asarray(paired.reference),
            partner_run_ids=np.asarray(paired.partner_run_ids),
            margin=OFFICIAL_CORRECT_DELIVERY_REWARD,
            cvar_level=0.20,
            seed=10_000 + int(args.seed_index),
            random_domain="qualification/C0/paired_full_episode",
            artifact_fingerprint=hashlib.sha256(
                np.asarray(paired.candidate).tobytes() + np.asarray(paired.reference).tobytes()
            ).hexdigest(),
        )
        behavior_values = _host(base_behavior_contract)
        decision = compound_gate_decision(
            passed=(return_decision.passed and bool(behavior_values["passed"])),
            statistic=min(
                return_decision.statistic,
                0.05 - float(behavior_values["base_distillation_kl"]),
            ),
            lower_bound=min(
                return_decision.lower_bound,
                0.05 - float(behavior_values["base_distillation_kl"]),
                1.0e-7 - float(behavior_values["conditional_residual_rms"]),
            ),
            upper_bound=max(
                return_decision.upper_bound,
                0.05 - float(behavior_values["base_distillation_kl"]),
            ),
            sample_count=return_decision.sample_count,
            random_domain=return_decision.random_domain,
            artifact_fingerprint=return_decision.artifact_fingerprint,
            reason=(
                "C0 paired mean/CVaR noninferiority plus held-out owner-logit "
                "and exact gate-zero carry/logit contract"
            ),
        )
        qualification = qualification.with_decision("C0", decision)
        write_json(output / "qualification" / "C0.json", {
            "decision": asdict(decision),
            "paired_return_decision": asdict(return_decision),
            "heldout_behavior": behavior_values,
            "paired_episode_count": int(np.asarray(paired.candidate).size),
        })
        if decision.passed:
            state = state._replace(qualified_base_params=jax.tree_util.tree_map(jnp.copy, state.params))
        state = state._replace(qualification=qualification.to_mapping())

    generator_update_kernel = _build_generator_update_kernel(
        generator=generator, optimizer=generator_optimizer, config=config
    )
    manifest_state = audit_manifest_from_mapping(state.anchor_audit_manifest)
    state = state._replace(anchor_audit_manifest=manifest_state)
    completed_audit_milestones = {
        record.milestone for record in manifest_state.records
    }
    def restored_microbatch(name: str) -> int | None:
        path = output / name
        if not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8")).get("branch_count")
        return None if value is None else int(value)

    training_anchor_microbatch = restored_microbatch(
        "training_anchor_microbatch.json"
    )
    audit_anchor_microbatch = restored_microbatch("audit_anchor_microbatch.json")
    restored_ledger = ResourceLedger.from_mapping(state.resource_ledger)
    counterfactual_steps = int(restored_ledger.counterfactual_steps)
    training_anchor_steps = int(restored_ledger.training_anchor_steps)
    audit_anchor_steps = int(restored_ledger.audit_anchor_steps)
    generator_episode_steps = int(restored_ledger.generator_training_steps)
    prior_gpu_hours = float(restored_ledger.gpu_hours)
    prior_peak_memory = int(restored_ledger.peak_memory_bytes)

    def live_resource_ledger(*, ego_steps: int) -> ResourceLedger:
        return ResourceLedger(
            ego_policy_steps=int(ego_steps),
            partner_training_steps=(
                int(restored_ledger.partner_source_training_steps)
                + int(generator_episode_steps)
            ),
            counterfactual_steps=int(counterfactual_steps),
            state_collection_steps=int(restored_ledger.state_collection_steps),
            calibration_steps=int(restored_ledger.calibration_steps),
            evaluation_steps=int(restored_ledger.evaluation_steps),
            base_distillation_steps=int(restored_ledger.base_distillation_steps),
            partner_source_training_steps=int(
                restored_ledger.partner_source_training_steps
            ),
            generator_training_steps=int(generator_episode_steps),
            training_anchor_steps=int(training_anchor_steps),
            audit_anchor_steps=int(audit_anchor_steps),
            gpu_hours=(
                prior_gpu_hours
                + gpu_hours_for_wall_seconds(time.perf_counter() - started)
            ),
            peak_memory_bytes=max(
                prior_peak_memory, int(peak_device_memory_bytes())
            ),
            deployable_parameters=int(restored_ledger.deployable_parameters),
            training_only_parameters=int(restored_ledger.training_only_parameters),
            inference_latency_ms=float(restored_ledger.inference_latency_ms),
        )
    phase_durations: dict[str, float] = {}
    preflight_limit = 1 if preflight else total_updates

    while int(np.asarray(state.update_count)) < min(total_updates, preflight_limit):
        update = int(np.asarray(state.update_count)) + 1
        synchronize = update <= 8
        current_steps = int(np.asarray(state.effective_environment_steps))
        next_steps = current_steps + config.environment.num_envs * config.training.rollout_length
        crosses_15m = current_steps < 15_000_000 <= next_steps
        crosses_22_5m = current_steps < 22_500_000 <= next_steps

        c0_deadline_artifact = output / "qualification" / "C0_15M.json"
        if (
            current_steps >= config.training.base_qualification_deadline_steps
            and not qualification.passed("C0")
            and not c0_deadline_artifact.exists()
            and not preflight
        ):
            paired = paired_owner_base_returns(
                environment=environment, model=model, params=state.params,
                model_config=config.model, owner_pool=owner_pool,
                partner_pool=development_pool,
                episodes_per_partner=config.calibration.episodes_per_run,
                key=jax.random.fold_in(
                    jnp.asarray(domains["qualification"], dtype=jnp.uint32), 15_000_000
                ),
            )
            return_decision = paired_mean_cvar_noninferiority_decision(
                np.asarray(paired.candidate) - np.asarray(paired.reference),
                partner_run_ids=np.asarray(paired.partner_run_ids),
                margin=OFFICIAL_CORRECT_DELIVERY_REWARD,
                cvar_level=0.20, seed=15_000_000 + int(args.seed_index),
                random_domain="qualification/C0/15M",
                artifact_fingerprint=hashlib.sha256(
                    np.asarray(paired.candidate).tobytes()
                    + np.asarray(paired.reference).tobytes()
                ).hexdigest(),
            )
            deadline_behavior = _host(_base_behavior_contract(
                model=model,
                params=state.params,
                batch=owner_holdout_batch,
                config=config,
                observation_shape=observation_shape,
            ))
            decision = compound_gate_decision(
                passed=(
                    return_decision.passed
                    and bool(deadline_behavior["passed"])
                ),
                statistic=min(
                    return_decision.statistic,
                    0.05 - float(deadline_behavior["base_distillation_kl"]),
                ),
                lower_bound=min(
                    return_decision.lower_bound,
                    0.05 - float(deadline_behavior["base_distillation_kl"]),
                    1.0e-7
                    - float(deadline_behavior["conditional_residual_rms"]),
                ),
                upper_bound=max(
                    return_decision.upper_bound,
                    0.05 - float(deadline_behavior["base_distillation_kl"]),
                ),
                sample_count=return_decision.sample_count,
                random_domain=return_decision.random_domain,
                artifact_fingerprint=return_decision.artifact_fingerprint,
                reason=(
                    "C0 15M paired mean/CVaR noninferiority plus held-out "
                    "owner-logit and exact gate-zero contract"
                ),
            )
            qualification = qualification.with_decision("C0", decision)
            write_json(c0_deadline_artifact, {
                "decision": asdict(decision),
                "paired_return_decision": asdict(return_decision),
                "heldout_behavior": deadline_behavior,
                "effective_environment_steps": current_steps,
                "paired_episode_count": int(np.asarray(paired.candidate).size),
            })
            if decision.passed:
                state = state._replace(
                    qualified_base_params=jax.tree_util.tree_map(jnp.copy, state.params)
                )
            else:
                state = state._replace(
                    qualification=qualification.to_mapping(),
                    curriculum_phase=jnp.asarray(
                        CurriculumPhase.SAFE_FALLBACK, dtype=jnp.int32
                    ),
                )
                write_json(output / "qualification" / "base_deadline_fallback.json", {
                    "deployment_tier": "owner_sp_source_fallback",
                    "effective_environment_steps": current_steps,
                    "owner_source_artifact": owner_artifact,
                    "formal_seed_preserved": True,
                    "reason": "C0 failed at the preregistered 15M deadline",
                })
                break
        # Epoch synchronization precedes data collection.  Replay from the old
        # continuation policy is structurally removed, never reweighted.
        if update > 1 and (update - 1) % config.training.target_policy_epoch_updates == 0:
            epoch = start_target_policy_epoch(
                params=state.params, update_number=update,
                updates_per_epoch=config.training.target_policy_epoch_updates,
                fingerprint_function=pytree_fingerprint,
            )
            state = state._replace(
                target_policy_epoch=epoch,
                target_params=clone_epoch_target_for_live_control(epoch),
                anchor_training_replay=None,
            )
            # The target recurrent context is synchronized exactly once with
            # its newly frozen actor/belief/raw-Q snapshot, then evolves in
            # parallel from the same legal observations and executed actions.
            runner = runner._replace(target_ego_policy=runner.ego_policy)
        validate_epoch_alignment(
            state.target_policy_epoch, update_number=update,
            updates_per_epoch=config.training.target_policy_epoch_updates,
        )
        qualification = _qualification_mapping(state.qualification)
        phase = phase_from_qualification(
            qualification, generator_admitted=generator_admission_passes > 0
        )
        conditional_enabled = qualification.passed("C3")
        regret_enabled = qualification.passed("C5")
        anchor_trigger = (update - 1) % config.anchors.interval_updates == 0
        audit_milestone: str | None = None
        if qualification.passed("C0") and "C0" not in completed_audit_milestones:
            audit_milestone = "C0"
        elif crosses_15m and "15M" not in completed_audit_milestones:
            audit_milestone = "15M"
        elif crosses_22_5m and "22.5M" not in completed_audit_milestones:
            audit_milestone = "22.5M"
        elif update == total_updates and "final" not in completed_audit_milestones:
            audit_milestone = "final"
        audit_trigger = audit_milestone is not None
        collection_key, shaping_key, schedule_key, lane_key, next_key = jax.random.split(
            state.random_key, 5
        )
        if conditional_enabled:
            base_lanes = jax.random.bernoulli(
                lane_key, config.training.base_policy_lane_probability,
                (config.environment.num_envs,),
            )
            gates = (~base_lanes).astype(jnp.float32)
        else:
            gates = jnp.zeros((config.environment.num_envs,), dtype=jnp.float32)
        snapshot_params = (
            stack_parameter_trees(snapshot_archive)
            if snapshot_archive else state.generator_params
        )
        partner_parameters = MixedPartnerParameters(
            generator_params=state.generator_params,
            snapshot_params=snapshot_params,
        )
        runner = runner._replace(random_key=collection_key)
        rollout_kernel = (
            kernels.rollout_anchor_full
            if (anchor_trigger or audit_trigger or preflight)
            else kernels.rollout_minimal
        )
        ((runner, batch, records), phase_durations["rollout"]) = _run_phase(
            output=output, phase="rollout", update=update,
            operation=lambda: rollout_kernel(
                runner, state.params, state.target_policy_epoch.target_params,
                partner_parameters, gates,
                official_reward_shaping_factor(
                    state.effective_environment_steps,
                    horizon=config.upstream.reward_shaping_horizon,
                ),
            ),
            synchronize=synchronize, kernels=kernels.metadata,
        )
        ((batch, regret_metrics), phase_durations["regret"]) = _run_phase(
            output=output, phase="regret", update=update,
            operation=lambda: attach_chunked_regret_with_kernels(
                kernels=kernels, batch=batch,
                target_params=state.target_policy_epoch.target_params,
                key=shaping_key,
                weight=(config.loss.decision_regret_weight_maximum if regret_enabled else 0.0),
            ),
            synchronize=synchronize, kernels=kernels.metadata,
        )
        records = dict(records)
        records["decision_regret"] = regret_metrics["decision_regret_values"]

        if preflight:
            if training_anchor_microbatch is None:
                training_anchor_microbatch = preflight_anchor_microbatch_from_records(
                    records=records,
                    target_params=state.target_policy_epoch.target_params,
                    partner_parameters=partner_parameters,
                    config=config,
                    chunk_kernel=anchor_kernels[0],
                    mode="training",
                    key=jax.random.fold_in(shaping_key, 91_001),
                )
                write_json(output / "training_anchor_microbatch.json", {
                    "branch_count": int(training_anchor_microbatch),
                    "maximum_anchor_worlds": anchor_preflight_world_count(
                        config, mode="training"
                    ),
                    "mode": "training",
                    "compile_only": True,
                    "attempted_simulator_transitions": 0,
                    "memory_limit_bytes": FORMAL_PEAK_MEMORY_LIMIT_BYTES,
                    "scientific_identity_effect": "none",
                })
            if audit_anchor_microbatch is None:
                audit_anchor_microbatch = preflight_anchor_microbatch_from_records(
                    records=records,
                    target_params=state.target_policy_epoch.target_params,
                    partner_parameters=partner_parameters,
                    config=config,
                    chunk_kernel=anchor_kernels[1],
                    mode="audit",
                    key=jax.random.fold_in(shaping_key, 91_002),
                )
                write_json(output / "audit_anchor_microbatch.json", {
                    "branch_count": int(audit_anchor_microbatch),
                    "maximum_anchor_worlds": anchor_preflight_world_count(
                        config, mode="audit"
                    ),
                    "mode": "audit",
                    "compile_only": True,
                    "attempted_simulator_transitions": 0,
                    "memory_limit_bytes": FORMAL_PEAK_MEMORY_LIMIT_BYTES,
                    "scientific_identity_effect": "none",
                })
        audit_anchors = None
        matched_audit_anchors = None
        matched_audit_pairs = None
        matched_audit_codes = None
        context_paired = None

        # C1 reference audit uses independent development runs at common
        # physical worlds and full 32/64 continuation replicas.
        if audit_trigger and not preflight:
            audit_key = jax.random.fold_in(
                jnp.asarray(domains["audit_anchor"], dtype=jnp.uint32),
                int.from_bytes(hashlib.sha256(
                    f"{audit_milestone}/{update}".encode("utf-8")
                ).digest()[:4], "big"),
            )
            audit_anchors, audit_partner_ids = collect_external_partner_support_audit(
                anchor_domain=update, key=audit_key, records=records,
                environment=environment, model=model,
                target_params=state.target_policy_epoch.target_params,
                config=config, partner_functions=partner_functions,
                partner_parameters=partner_parameters,
                external_member_count=development_pool.member_count,
                microbatch_size=audit_anchor_microbatch,
                anchor_functions=anchor_functions,
                chunk_kernel=anchor_kernels[1],
            )
            if audit_anchor_microbatch is None:
                audit_anchor_microbatch = anchor_kernels[1].selected_microbatch_size
                if audit_anchor_microbatch is None:
                    raise RuntimeError("Audit anchor microbatch preflight was not recorded.")
                write_json(output / "audit_anchor_microbatch.json", {
                    "branch_count": int(audit_anchor_microbatch),
                    "mode": "audit",
                    "memory_limit_bytes": FORMAL_PEAK_MEMORY_LIMIT_BYTES,
                    "scientific_identity_effect": "none",
                })
            audit_increment = int(audit_anchors.anchor_ids.shape[0]) * 6 * (
                config.anchors.audit_fit_replicas + config.anchors.audit_evaluation_replicas
            ) * config.anchors.audit_continuation_horizon
            counterfactual_steps += audit_increment
            audit_anchor_steps += audit_increment
            fingerprint = hashlib.sha256(
                np.asarray(audit_anchors.fit_returns_by_action).tobytes()
                + np.asarray(audit_anchors.evaluation_returns_by_action).tobytes()
            ).hexdigest()
            qualification, audit_metrics = _audit_signal_contracts(
                qualification=qualification, anchors=audit_anchors,
                partner_ids=audit_partner_ids, model=model, params=state.params,
                target_epoch=state.target_policy_epoch,
                seed=20_000 + update + int(args.seed_index),
                random_domain=f"audit_anchor/{update}",
                artifact_fingerprint=fingerprint,
            )
            if qualification.passed("C2") and "C2" in audit_metrics:
                state = state._replace(
                    raw_q_calibration_error=jnp.asarray(
                        audit_metrics["C2"]["calibration_error"],
                        dtype=jnp.float32,
                    ),
                    raw_q_minimum_margin=jnp.asarray(
                        audit_metrics["C2"]["monte_carlo_error_bound"],
                        dtype=jnp.float32,
                    ),
                )
            matched_audit_anchors, matched_audit_pairs, matched_audit_codes = (
                collect_matched_code_audit(
                    anchor_domain=update + 600_000,
                    key=jax.random.fold_in(audit_key, 600_000),
                    records=records,
                    environment=environment,
                    model=model,
                    target_params=state.target_policy_epoch.target_params,
                    config=config,
                    partner_functions=partner_functions,
                    partner_parameters=partner_parameters,
                    microbatch_size=audit_anchor_microbatch,
                    anchor_functions=anchor_functions,
                    chunk_kernel=anchor_kernels[1],
                )
            )
            audit_increment = int(
                matched_audit_anchors.anchor_ids.shape[0]
            ) * 6 * (
                config.anchors.audit_fit_replicas
                + config.anchors.audit_evaluation_replicas
            ) * config.anchors.audit_continuation_horizon
            audit_increment += int(
                config.anchors.audit_matched_code_states
                * 2
                * config.anchors.probe_steps
            )
            counterfactual_steps += audit_increment
            audit_anchor_steps += audit_increment
            if qualification.passed("C2"):
                functional_contexts = privileged_functional_contexts(
                    signatures=centered_action_values(
                        np.asarray(audit_anchors.fit_returns_by_action)
                    ),
                    partner_ids=np.asarray(audit_partner_ids),
                    latent_dim=config.model.latent_dim,
                )
                context_paired = paired_context_returns(
                    environment=environment,
                    model=model,
                    params=state.params,
                    model_config=config.model,
                    partner_pool=development_pool,
                    episodes_per_partner=config.calibration.episodes_per_run,
                    key=jax.random.fold_in(audit_key, 700_000),
                    privileged_functional_contexts=functional_contexts,
                )
                context_fingerprint = hashlib.sha256(
                    np.asarray(context_paired.oracle_context).tobytes()
                    + np.asarray(context_paired.online_context).tobytes()
                    + np.asarray(context_paired.state_only).tobytes()
                ).hexdigest()
                qualification, c3_metrics = _qualify_c3(
                    qualification=qualification,
                    paired=context_paired,
                    seed=30_000 + update + int(args.seed_index),
                    random_domain=f"qualification/C3/{audit_milestone}/{update}",
                    artifact_fingerprint=context_fingerprint,
                )
                audit_metrics["C3"] = c3_metrics
            write_json(output / "qualification" / f"audit_{update:08d}.json", audit_metrics)
            audit_record = AnchorAuditRecord(
                milestone=str(audit_milestone),
                artifact_path=str(output / "qualification" / f"audit_{update:08d}.json"),
                artifact_fingerprint=fingerprint,
                random_domain=f"audit_anchor/{update}",
                fit_replicas=config.anchors.audit_fit_replicas,
                evaluation_replicas=config.anchors.audit_evaluation_replicas,
                continuation_horizon=config.anchors.audit_continuation_horizon,
            )
            manifest_state = state.anchor_audit_manifest
            if isinstance(manifest_state, Mapping):
                manifest_state = AnchorAuditManifest(
                    records=tuple(
                        AnchorAuditRecord(**dict(item))
                        for item in manifest_state.get("records", ())
                    ),
                    expected_fit_replicas=int(
                        manifest_state.get(
                            "expected_fit_replicas",
                            config.anchors.audit_fit_replicas,
                        )
                    ),
                    expected_evaluation_replicas=int(
                        manifest_state.get(
                            "expected_evaluation_replicas",
                            config.anchors.audit_evaluation_replicas,
                        )
                    ),
                    expected_continuation_horizon=int(
                        manifest_state.get(
                            "expected_continuation_horizon",
                            config.anchors.audit_continuation_horizon,
                        )
                    ),
                )
            state = state._replace(anchor_audit_manifest=manifest_state.add(audit_record))
            completed_audit_milestones.add(str(audit_milestone))

        training_anchors = None
        if anchor_trigger and qualification.passed("C1"):
            training_key = jax.random.fold_in(
                jnp.asarray(domains["training_anchor"], dtype=jnp.uint32), update
            )
            training_anchors, unused_pairs, unused_codes, unused_indexes = collect_anchor_batch(
                mode="training", anchor_domain=update, key=training_key,
                records=records, environment=environment, model=model,
                target_params=state.target_policy_epoch.target_params,
                config=config, partner_functions=partner_functions,
                partner_parameters=partner_parameters,
                microbatch_size=training_anchor_microbatch,
                anchor_functions=anchor_functions,
                chunk_kernel=anchor_kernels[0],
            )
            if training_anchor_microbatch is None:
                training_anchor_microbatch = (
                    anchor_kernels[0].selected_microbatch_size
                )
                if training_anchor_microbatch is None:
                    raise RuntimeError(
                        "Training anchor microbatch preflight was not recorded."
                    )
                write_json(output / "training_anchor_microbatch.json", {
                    "branch_count": int(training_anchor_microbatch),
                    "mode": "training",
                    "memory_limit_bytes": FORMAL_PEAK_MEMORY_LIMIT_BYTES,
                    "scientific_identity_effect": "none",
                })
            del unused_pairs, unused_codes, unused_indexes
            state = state._replace(anchor_training_replay=append_replay_batch(
                state.anchor_training_replay, batch=training_anchors,
                epoch=state.target_policy_epoch,
                capacity=config.anchors.replay_capacity_per_epoch,
            ))
            anchor_increment = _training_anchor_steps(config)
            counterfactual_steps += anchor_increment
            training_anchor_steps += anchor_increment

        schedule = environment_minibatch_schedule(
            schedule_key, environment_count=config.environment.num_envs,
            minibatches_per_epoch=config.training.minibatches_per_epoch,
            update_epochs=config.ppo.update_epochs,
        )
        ((ppo_core, ppo_metrics), phase_durations["ppo"]) = _run_phase(
            output=output, phase="ppo", update=update,
            operation=lambda: kernels.ppo_normal_scan(
                training_core_state(state),
                batch,
                schedule,
                jnp.asarray(
                    0.1 if qualification.passed("C0") else 0.0,
                    dtype=jnp.float32,
                ),
                jnp.asarray(
                    0.1 if qualification.passed("C0") else 1.0,
                    dtype=jnp.float32,
                ),
            ),
            synchronize=synchronize, kernels=kernels.metadata,
        )
        state = merge_training_core(state, ppo_core)
        if bool(np.any(np.asarray(ppo_metrics["nonfinite_update"]) > 0.5)):
            raise FloatingPointError(
                f"Non-finite PPO loss/gradient/update at outer update {update}; "
                "the optimizer update was rejected and no auxiliary phase ran."
            )
        applied_ppo = int(np.asarray(ppo_metrics["optimizer_applied"]).sum())
        state = state._replace(ppo_optimizer_step=state.ppo_optimizer_step + applied_ppo)

        replay_batch = current_epoch_replay_batch(
            state.anchor_training_replay, epoch=state.target_policy_epoch
        )
        anchors_for_loss = replay_batch if replay_batch is not None else _empty_anchor(batch)
        auxiliary = AuxiliaryCoreState(
            params=state.params,
            raw_q_optimizer_state=state.raw_q_optimizer_state,
            response_optimizer_state=state.response_optimizer_state,
            raw_q_optimizer_step=state.raw_q_optimizer_step,
            response_optimizer_step=state.response_optimizer_step,
        )
        ((auxiliary, raw_q_metrics, decision_gradients), phase_durations["raw_q"]) = _run_phase(
            output=output, phase="raw_q", update=update,
            operation=lambda: kernels.raw_q_auxiliary_scan(
                auxiliary, state.target_policy_epoch.target_params, batch,
                anchors_for_loss,
                jnp.asarray(replay_batch is not None and qualification.passed("C1")),
                jnp.asarray(qualification.passed("C3")),
                state.raw_q_calibration_error,
                state.raw_q_minimum_margin,
            ),
            synchronize=synchronize, kernels=kernels.metadata,
        )
        ((auxiliary, response_metrics), phase_durations["response"]) = _run_phase(
            output=output, phase="response", update=update,
            operation=lambda: kernels.response_auxiliary_scan(
                auxiliary, batch, decision_gradients
            ),
            synchronize=synchronize, kernels=kernels.metadata,
        )
        state = state._replace(
            params=auxiliary.params,
            raw_q_optimizer_state=auxiliary.raw_q_optimizer_state,
            response_optimizer_state=auxiliary.response_optimizer_state,
            raw_q_optimizer_step=auxiliary.raw_q_optimizer_step,
            response_optimizer_step=auxiliary.response_optimizer_step,
        )

        # Complete one-code/one-version generator episodes are independent of
        # the 256-step ego rollout.  Failed trust-region candidates are rolled
        # back exactly and can never enter the partner mixture.
        generator_key, generator_schedule_key = jax.random.split(
            jax.random.fold_in(jnp.asarray(domains["generator"], dtype=jnp.uint32), update)
        )
        codes = sample_partner_codes(
            generator_key,
            batch_size=config.partner_generator.episodes_per_update,
            code_dim=config.partner_generator.code_dim,
        )
        generator_batch = collect_complete_generator_episodes(
            environment=generator_environment, model=model,
            ego_params=(
                state.params
                if state.qualified_base_params is None
                else state.qualified_base_params
            ),
            model_config=config.model, generator=generator,
            generator_params=state.generator_params, codes=codes,
            key=generator_key, parameter_version_id=update,
            official_shaping_factor=official_reward_shaping_factor(
                state.effective_environment_steps,
                horizon=config.upstream.reward_shaping_horizon,
            ),
        )
        if config.run_kind != "formal":
            validate_complete_episode_batch(generator_batch)
        generator_schedule = environment_minibatch_schedule(
            generator_schedule_key,
            environment_count=config.partner_generator.episodes_per_update,
            minibatches_per_epoch=config.partner_generator.environment_minibatches,
            update_epochs=config.partner_generator.update_epochs,
        )
        generator_core = GeneratorCoreState(
            state.generator_params, state.generator_target_params,
            state.generator_optimizer_state,
        )
        previous_generator = immutable_parameter_snapshot(state.generator_params)
        previous_generator_optimizer_state = state.generator_optimizer_state
        previous_generator_optimizer_step = state.generator_optimizer_step
        ((generator_result, generator_metrics), phase_durations["generator"]) = _run_phase(
            output=output, phase="generator", update=update,
            operation=lambda: generator_update_kernel(
                generator_core, generator_batch, generator_schedule,
                _generator_diversity_bonus(
                    codes=generator_batch.codes,
                    readout=(
                        state.generator_signature_readout
                        if qualification.passed("C1")
                        and qualification.passed("C2")
                        else None
                    ),
                    config=config,
                ),
            ),
            synchronize=synchronize,
            kernels=lambda: {"generator": list(generator_update_kernel.metadata())},
        )
        candidate_generator, generator_optimizer_candidate = generator_result.params, generator_result.optimizer_state
        reference_generator = (
            state.last_qualified_generator_params
            if state.last_qualified_generator_params is not None
            else previous_generator
        )
        generator_kl = float(np.asarray(_generator_parameter_kl(
            generator=generator,
            reference_params=reference_generator,
            candidate_params=candidate_generator,
            batch=generator_batch,
            hidden_dim=config.partner_generator.hidden_dim,
        )))
        candidate_generator_optimizer_step = (
            state.generator_optimizer_step
            + int(generator_schedule.shape[0] * generator_schedule.shape[1])
        )
        if generator_kl > config.partner_generator.trust_region_kl_maximum:
            candidate_generator = reference_generator
            generator_optimizer_candidate = (
                state.last_qualified_generator_optimizer_state
                if state.last_qualified_generator_optimizer_state is not None
                else previous_generator_optimizer_state
            )
            candidate_generator_optimizer_step = (
                state.last_qualified_generator_optimizer_step
                if state.last_qualified_generator_optimizer_state is not None
                else previous_generator_optimizer_step
            )
        state = state._replace(
            generator_params=candidate_generator,
            generator_target_params=jax.tree_util.tree_map(jnp.copy, candidate_generator),
            generator_optimizer_state=generator_optimizer_candidate,
            generator_optimizer_step=candidate_generator_optimizer_step,
        )
        generator_episode_steps += int(
            config.partner_generator.episodes_per_update * config.partner_generator.episode_steps
        )

        # Admission is read only at independent audit milestones after C1/C2.
        # A failed candidate is never sampled and is rolled back exactly.
        if (
            audit_trigger
            and qualification.passed("C2")
            and matched_audit_anchors is not None
            and matched_audit_codes is not None
            and not preflight
        ):
            admission_key = jax.random.fold_in(
                jnp.asarray(domains["qualification"], dtype=jnp.uint32),
                50_000 + update,
            )
            admission_codes = sample_partner_codes(
                admission_key,
                batch_size=config.partner_generator.episodes_per_update,
                code_dim=config.partner_generator.code_dim,
            )
            paired_generator = paired_generator_external_returns(
                environment=generator_environment,
                model=model,
                ego_params=(
                    state.params
                    if state.qualified_base_params is None
                    else state.qualified_base_params
                ),
                model_config=config.model,
                generator=generator,
                generator_params=state.generator_params,
                partner_pool=development_pool,
                codes=admission_codes,
                key=jax.random.fold_in(admission_key, 1),
            )
            competence = paired_mean_cvar_statistics(
                np.asarray(paired_generator.candidate)
                - np.asarray(paired_generator.reference),
                partner_run_ids=np.asarray(paired_generator.partner_run_ids),
                cvar_level=config.partner_generator.cvar_level,
                seed=50_000 + update + int(args.seed_index),
            )
            matched_fit = np.asarray(
                matched_audit_anchors.fit_returns_by_action, dtype=np.float64
            )
            matched_evaluation = np.asarray(
                matched_audit_anchors.evaluation_returns_by_action,
                dtype=np.float64,
            )
            code_values = np.asarray(matched_audit_codes, dtype=np.float64)
            code_regions = (code_values[:, 0] >= 0.0).astype(np.int32)
            if np.unique(code_regions).size < 2:
                code_regions = np.arange(code_values.shape[0], dtype=np.int32) % 2
            state_only_action = np.argmax(
                np.mean(matched_fit, axis=0), axis=-1
            )
            matched_row = np.arange(matched_fit.shape[0])
            code_oracle_action = np.argmax(matched_fit, axis=-1)
            code_oracle_lift = (
                matched_evaluation[matched_row, code_oracle_action]
                - matched_evaluation[matched_row, state_only_action]
            )
            stable_code_actions = [
                int(np.argmax(np.mean(matched_fit[code_regions == region], axis=0)))
                for region in np.unique(code_regions)
            ]
            generator_structure = generator_decision_audit(
                fit_signatures=centered_action_values(matched_fit),
                evaluation_signatures=centered_action_values(matched_evaluation),
                code_region_ids=code_regions,
                oracle_lift_blocks=tuple(
                    float(np.mean(code_oracle_lift[code_regions == region]))
                    for region in np.unique(code_regions)
                ),
                stable_best_actions=stable_code_actions,
                seed=51_000 + update + int(args.seed_index),
            )
            candidate_signature_readout = (
                _fit_code_signature_readout(
                    code_values, centered_action_values(matched_fit)
                )
                if generator_structure.passed
                else state.generator_signature_readout
            )
            heldout_behavior_kl = float(np.asarray(_generator_behavior_kl(
                generator=generator,
                params=state.generator_params,
                batch=generator_holdout_batch,
                code_anchors=code_anchors,
                hidden_dim=config.partner_generator.hidden_dim,
            )))
            interpolation_uniform_kl = float(np.asarray(
                _generator_interpolation_uniform_kl(
                    generator=generator,
                    params=state.generator_params,
                    observations=generator_holdout_batch.observations[:-1],
                    starts=generator_holdout_batch.episode_starts[:-1],
                    key=jax.random.fold_in(admission_key, 2),
                    code_dim=config.partner_generator.code_dim,
                    hidden_dim=config.partner_generator.hidden_dim,
                )
            ))
            admission = GeneratorAdmission(
                competence_passed=bool(
                    competence.mean_lcb
                    >= -config.partner_generator.delivery_noninferiority_margin
                    and competence.cvar_lcb
                    >= -config.partner_generator.delivery_noninferiority_margin
                ),
                signature_variance_passed=(
                    generator_structure.variance_difference_lcb > 0.0
                ),
                oracle_code_lift_passed=(
                    generator_structure.oracle_code_lift_lcb > 0.0
                ),
                stable_action_regions=generator_structure.stable_action_regions,
                trust_region_kl=generator_kl,
                behavior_kl=heldout_behavior_kl,
                interpolation_uniform_kl=interpolation_uniform_kl,
                interpolation_uniform_kl_minimum=(
                    config.partner_generator.interpolation_uniform_kl_minimum
                ),
                delivery_noninferiority_margin=(
                    config.partner_generator.delivery_noninferiority_margin
                ),
                mean_noninferiority_lcb=competence.mean_lcb,
                cvar_noninferiority_lcb=competence.cvar_lcb,
            )
            admission_record = {
                "admission": asdict(admission),
                "passed": admission.passed,
                "decision_structure": asdict(generator_structure),
                "competence": asdict(competence),
                "target_policy_epoch_id": int(state.target_policy_epoch.epoch_id),
                "scientific_readout_allowed": False,
            }
            write_json(
                output / "qualification" / f"generator_{update:08d}.json",
                admission_record,
            )
            if admission.passed:
                qualified_snapshot = immutable_parameter_snapshot(
                    state.generator_params
                )
                snapshot_path = output / "generator_snapshots" / f"qualified_{update:08d}"
                save_generator_snapshot(snapshot_path, qualified_snapshot)
                snapshot_archive.append(qualified_snapshot)
                generator_admission_passes += 1
                source_probabilities = admitted_source_probabilities(
                    generator_admission_passes
                )
                state = state._replace(
                    last_qualified_generator_params=qualified_snapshot,
                    last_qualified_generator_optimizer_state=(
                        state.generator_optimizer_state
                    ),
                    last_qualified_generator_optimizer_step=(
                        state.generator_optimizer_step
                    ),
                    generator_signature_readout=jnp.asarray(
                        candidate_signature_readout, dtype=jnp.float32
                    ),
                    generator_snapshot_archive=tuple(snapshot_archive),
                    generator_admission_passes=jnp.asarray(
                        generator_admission_passes, dtype=jnp.int32
                    ),
                    partner_source_probabilities=jnp.asarray(
                        source_probabilities, dtype=jnp.float32
                    ),
                )
            else:
                generator_admission_passes = 0
                source_probabilities = admitted_source_probabilities(0)
                rollback_params = (
                    state.last_qualified_generator_params
                    if state.last_qualified_generator_params is not None
                    else previous_generator
                )
                rollback_optimizer = (
                    state.last_qualified_generator_optimizer_state
                    if state.last_qualified_generator_optimizer_state is not None
                    else previous_generator_optimizer_state
                )
                state = state._replace(
                    generator_params=rollback_unqualified_generator(
                        state.generator_params, rollback_params, admission
                    ),
                    generator_target_params=jax.tree_util.tree_map(
                        jnp.copy, rollback_params
                    ),
                    generator_optimizer_state=rollback_optimizer,
                    generator_optimizer_step=(
                        state.last_qualified_generator_optimizer_step
                        if state.last_qualified_generator_optimizer_state is not None
                        else previous_generator_optimizer_step
                    ),
                    generator_admission_passes=jnp.asarray(0, dtype=jnp.int32),
                    partner_source_probabilities=jnp.asarray(
                        source_probabilities, dtype=jnp.float32
                    ),
                )
            (
                partner_functions,
                kernels,
                anchor_functions,
                anchor_kernels,
            ) = runtime(source_probabilities)

        # C4 is a full-episode paired control audit and is read only after C3.
        if qualification.passed("C3") and not qualification.passed("C4") and audit_trigger and not preflight:
            paired_control, control_diagnostics = paired_conditional_base_returns(
                environment=environment, model=model, params=state.params,
                model_config=config.model, partner_pool=development_pool,
                episodes_per_partner=config.calibration.episodes_per_run,
                key=jax.random.fold_in(
                    jnp.asarray(domains["qualification"], dtype=jnp.uint32), 40_000 + update
                ),
            )
            qualification, c4_metrics = _qualify_c4(
                qualification=qualification, paired=paired_control,
                diagnostics=control_diagnostics, params=state.params,
                seed=40_000 + update, random_domain=f"qualification/C4/{update}",
                artifact_fingerprint=hashlib.sha256(
                    np.asarray(paired_control.candidate).tobytes()
                    + np.asarray(paired_control.reference).tobytes()
                ).hexdigest(),
            )
            write_json(output / "qualification" / f"C4_{update:08d}.json", c4_metrics)

        # C5 is a five-part task-value contract.  Its paired control subgate
        # uses two shadow updates from the same core, batch, schedule and keys;
        # neither shadow state is committed to the formal run.
        if (
            qualification.passed("C4")
            and generator_admission_passes > 0
            and not qualification.passed("C5")
            and audit_trigger
            and audit_anchors is not None
            and context_paired is not None
            and not preflight
        ):
            passive_core, unused_passive_metrics = kernels.ppo_normal_scan(
                training_core_state(state),
                batch,
                schedule,
                jnp.asarray(0.1, dtype=jnp.float32),
                jnp.asarray(0.1, dtype=jnp.float32),
            )
            active_batch, unused_active_regret_metrics = kernels.regret_finalize(
                batch,
                regret_metrics["decision_regret_values_with_next"],
                jnp.asarray(
                    config.loss.decision_regret_weight_maximum,
                    dtype=jnp.float32,
                ),
            )
            active_core, unused_active_metrics = kernels.ppo_normal_scan(
                training_core_state(state),
                active_batch,
                schedule,
                jnp.asarray(0.1, dtype=jnp.float32),
                jnp.asarray(0.1, dtype=jnp.float32),
            )
            del (
                unused_passive_metrics,
                unused_active_regret_metrics,
                unused_active_metrics,
            )
            paired_active = paired_delta_params_returns(
                environment=environment,
                model=model,
                candidate_params=active_core.params,
                reference_params=passive_core.params,
                model_config=config.model,
                partner_pool=development_pool,
                episodes_per_partner=config.calibration.episodes_per_run,
                key=jax.random.fold_in(
                    jnp.asarray(domains["qualification"], dtype=jnp.uint32),
                    80_000 + update,
                ),
                gate=1.0,
            )
            fit = np.asarray(
                audit_anchors.fit_returns_by_action, dtype=np.float64
            )
            evaluation = np.asarray(
                audit_anchors.evaluation_returns_by_action, dtype=np.float64
            )
            ids = np.asarray(audit_partner_ids)
            audit_output = _anchor_model_outputs(
                model,
                state.target_policy_epoch.target_params,
                audit_anchors,
            )
            q_values = np.asarray(
                conservative_raw_q(audit_output.raw_q1, audit_output.raw_q2)
            )
            rows = np.arange(fit.shape[0])
            base_actions = np.argmax(np.asarray(audit_output.base_logits), axis=-1)
            oracle_actions = np.argmax(fit, axis=-1)
            q_actions = np.argmax(q_values, axis=-1)
            empirical_lift = (
                evaluation[rows, oracle_actions]
                - evaluation[rows, base_actions]
            )
            q_selection_lift = (
                evaluation[rows, q_actions]
                - evaluation[rows, base_actions]
            )
            predicted_regret = np.asarray(_anchor_predicted_regret(
                model=model,
                params=state.target_policy_epoch.target_params,
                anchors=audit_anchors,
                particle_count=config.model.posterior_particles,
                key=jax.random.fold_in(
                    jnp.asarray(domains["audit_anchor"], dtype=jnp.uint32),
                    81_000 + update,
                ),
            ))
            online_inference = (
                np.asarray(context_paired.online_context)
                - np.asarray(context_paired.state_only)
            )
            active_minus_passive = (
                np.asarray(paired_active.candidate)
                - np.asarray(paired_active.reference)
            )
            c5 = active_information_audit(
                opportunity_blocks=_partner_block_means(empirical_lift, ids),
                raw_q_selection_blocks=_partner_block_means(q_selection_lift, ids),
                online_inference_blocks=_partner_block_means(
                    online_inference,
                    np.asarray(context_paired.partner_run_ids),
                ),
                predicted_regret=predicted_regret,
                empirical_oracle_lift=empirical_lift,
                active_minus_passive_blocks=_partner_block_means(
                    active_minus_passive,
                    np.asarray(paired_active.partner_run_ids),
                ),
                seed=80_000 + update + int(args.seed_index),
            )
            c5_fingerprint = hashlib.sha256(
                empirical_lift.tobytes()
                + q_selection_lift.tobytes()
                + predicted_regret.tobytes()
                + active_minus_passive.tobytes()
            ).hexdigest()
            c5_decision = compound_gate_decision(
                passed=c5.passed,
                statistic=c5.active_minus_passive_lcb,
                lower_bound=min(
                    c5.opportunity_lcb,
                    c5.raw_q_selection_lcb,
                    c5.online_inference_lcb,
                    c5.regret_calibration_lcb,
                    c5.active_minus_passive_lcb,
                ),
                upper_bound=max(
                    c5.opportunity_lcb,
                    c5.raw_q_selection_lcb,
                    c5.online_inference_lcb,
                    c5.regret_calibration_lcb,
                    c5.active_minus_passive_lcb,
                ),
                sample_count=int(np.unique(ids).size),
                random_domain=f"qualification/C5/{audit_milestone}/{update}",
                artifact_fingerprint=c5_fingerprint,
                reason="C5 opportunity/Q/inference/regret-calibration/paired-control subgates",
            )
            qualification = qualification.with_decision("C5", c5_decision)
            write_json(
                output / "qualification" / f"C5_{update:08d}.json",
                {**asdict(c5), "decision": asdict(c5_decision)},
            )

        phase = phase_from_qualification(
            qualification, generator_admitted=generator_admission_passes > 0
        )
        state = state._replace(
            qualification=qualification.to_mapping(),
            curriculum_phase=jnp.asarray(phase, dtype=jnp.int32),
            random_key=next_key,
            update_count=state.update_count + 1,
            effective_environment_steps=runner.effective_environment_steps,
            runner_state=runner,
        )
        state = state._replace(
            resource_ledger=live_resource_ledger(
                ego_steps=int(np.asarray(state.effective_environment_steps))
            ).to_mapping()
        )
        metrics = {
            "update_count": update,
            "effective_environment_steps": int(np.asarray(state.effective_environment_steps)),
            "phase": phase.name,
            "qualification": qualification.to_mapping(),
            "base_distillation": _host(base_distill_metrics),
            "generator_distillation": _host(generator_distill_metrics),
            "ppo": _host(jax.tree_util.tree_map(jnp.mean, ppo_metrics)),
            "raw_q": _host(jax.tree_util.tree_map(jnp.mean, raw_q_metrics)),
            "response": _host(jax.tree_util.tree_map(jnp.mean, response_metrics)),
            "generator": _host(jax.tree_util.tree_map(jnp.mean, generator_metrics)),
            "regret": _host({
                key: value
                for key, value in regret_metrics.items()
                if key not in {
                    "decision_regret_values",
                    "decision_regret_values_with_next",
                }
            }),
            "source_probabilities": list(source_probabilities),
            "phase_wall_seconds": dict(phase_durations),
            "gpu": _gpu_snapshot(),
        }
        write_jsonl(output / "records" / "metrics" / f"update_{update:08d}.jsonl", (metrics,))
        if (
            int(np.asarray(state.effective_environment_steps))
            % config.training.checkpoint_interval_environment_steps == 0
            or preflight
        ):
            _save_r3_checkpoint(
                manager,
                output=output,
                step=int(np.asarray(state.effective_environment_steps)),
                state=state,
            )

    final_step = int(np.asarray(state.effective_environment_steps))
    if manager.latest_step() != final_step:
        _save_r3_checkpoint(
            manager, output=output, step=final_step, state=state
        )
    qualification = _qualification_mapping(state.qualification)
    final_snapshot_params = (
        stack_parameter_trees(snapshot_archive)
        if snapshot_archive else state.generator_params
    )
    final_partner_parameters = MixedPartnerParameters(
        generator_params=state.generator_params,
        snapshot_params=final_snapshot_params,
    )
    support_metadata = _write_final_support_latents(
        output=output, environment=environment, model=model, state=state,
        config=config, partner_functions=partner_functions,
        partner_parameters=final_partner_parameters, kernels=kernels,
        key=jax.random.fold_in(
            jnp.asarray(domains["snapshot"], dtype=jnp.uint32), 900_001
        ),
    )
    tier = deployment_tier(qualification)
    deployable_count = parameter_count(deployable_parameters(state.params))
    training_only_count = (
        parameter_count(state.params) - deployable_count
        + parameter_count(state.target_policy_epoch.target_params)
        + parameter_count(state.generator_params)
        + parameter_count(state.generator_target_params)
    )
    live_ledger = live_resource_ledger(ego_steps=final_step)
    ledger = ResourceLedger(
        **{
            **{
                name: getattr(live_ledger, name)
                for name in ResourceLedger.__dataclass_fields__
            },
            "state_collection_steps": (
                live_ledger.state_collection_steps
                + int(support_metadata["environment_steps"])
            ),
            "deployable_parameters": deployable_count,
            "training_only_parameters": training_only_count,
        }
    )
    write_json(output / "resource_ledger.json", ledger.to_mapping())
    metadata = {
        "method": METHOD_VERSION,
        "seed_index": int(args.seed_index),
        "jax_prng_key": list(outer_key),
        "effective_environment_steps": final_step,
        "update_count": int(np.asarray(state.update_count)),
        "deployment_tier": tier.value,
        "qualification": qualification.to_mapping(),
        "target_policy_epoch": {
            "epoch_id": int(state.target_policy_epoch.epoch_id),
            "policy_fingerprint": combined_policy_fingerprint(state.target_policy_epoch),
            "started_update": int(state.target_policy_epoch.started_update),
        },
        "owner_source_artifact": owner_artifact,
        "generator_admission_passes": generator_admission_passes,
        "support_collection": support_metadata,
        "decision_regret_state_chunk_size": DECISION_REGRET_STATE_CHUNK_SIZE,
        "compiled_kernels": kernels.metadata(),
        "training_anchor_kernel": list(anchor_kernels[0].metadata()),
        "audit_anchor_kernel": list(anchor_kernels[1].metadata()),
        "training_anchor_microbatch": training_anchor_microbatch,
        "audit_anchor_microbatch": audit_anchor_microbatch,
        "generator_kernel": list(generator_update_kernel.metadata()),
        "jax_runtime": _jax_runtime_snapshot(),
        "gpu": _gpu_snapshot(),
        "scientific_readout": False,
        "scientific_readout_allowed": False,
        "resume_allowed": not preflight,
        "note": "Signal-contract qualification is not a benchmark result.",
    }
    write_json(output / "run_metadata.json", metadata)
    if preflight:
        if peak_device_memory_bytes() >= FORMAL_PEAK_MEMORY_LIMIT_BYTES:
            raise RuntimeError("Formal CUDA preflight exceeded the 40,000 MiB limit.")
        write_json(output / "cuda_preflight_report.json", metadata)
    print(f"Complete DELTA-ZSC r3 run: {output}")


def run_cuda_preflight(args: argparse.Namespace) -> None:
    args._execution_scope = CUDA_PREFLIGHT_SCOPE
    args.run_kind = "formal"
    args.resume = False
    run_training(args)


__all__ = ["run_cuda_preflight", "run_training"]
