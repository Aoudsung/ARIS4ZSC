"""DEPI exact-filter decision-supervision training lifecycle.

Every outer update executes the same learning system.  Signal audits never
control losses, partner sampling, checkpoint selection, or deployment.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Callable, Mapping

import numpy as np

from experiments.overcooked_v2.deployment import (
    Deployment,
    deployment_action,
    deployable_parameters,
    export_deployment_bundle,
    load_deployment,
)
from experiments.overcooked_v2.official_adapter import (
    FrozenPartnerPool,
    VectorEnvironment,
    validate_official_runtime,
)
from src.path_c.anchor_sampling import (
    COMPARATOR_RUN_ID_CAPACITY,
    FrozenPairComparator,
    collect_anchor_batch,
    make_anchor_functions,
    preflight_anchor_microbatch_from_records,
    separation_terms_from_matched_pairs,
)
from src.path_c.base_distillation import collect_owner_behavior, distill_owner_actor
from src.path_c.compiled_kernels import (
    build_anchor_chunk_kernel,
    build_training_kernels,
    configure_persistent_compilation_cache,
)
from src.path_c.experiment import (
    CONFIG_VERSION,
    CHECKPOINT_SCHEMA_VERSION,
    ENGINEERING_SEED_INDEX,
    METHOD_VERSION,
    OFFICIAL_SOURCE_COMMIT,
    engineering_training_key,
    load_config,
    load_partner_manifest,
    official_training_domain_keys,
    official_training_key,
    validate_seed_training_manifest,
)
from src.path_c.m1_gate import (
    evaluate_m1_gate_on_anchor_batch,
    initialize_bootstrap_value_ensemble,
    m1_gate_report,
    train_bootstrap_value_ensemble,
)
from src.path_c.model import (
    build_model,
    initial_policy_state,
    initialize_model_parameters,
)
from src.path_c.partner_sources import (
    build_partner_pool,
    make_static_pool_partner_functions,
)
from src.path_c.resources import (
    ResourceLedger,
    configure_bundled_cuda_toolchain,
    gpu_hours_for_wall_seconds,
    parameter_count,
    peak_device_memory_bytes,
    require_single_cuda_worker,
)
from src.path_c.runner import (
    context_dropout_probability,
    initialize_runner,
)
from src.path_c.storage import (
    ensure_run_identity,
    orbax_manager,
    pytree_fingerprint,
    restore_latest_checkpoint,
    save_checkpoint,
    training_identity,
    validate_formal_repository_state,
    validate_registered_python_runtime,
    write_json,
    write_json_atomic,
    write_jsonl,
)
from src.path_c.training import (
    adapt_effective_update_epochs,
    environment_minibatch_schedule,
    make_optimizer,
    official_reward_shaping_factor,
    polyak_update,
)
from src.path_c.types import CounterfactualAnchorBatch, SeparationTerms, TrainState


FORMAL_PEAK_MEMORY_LIMIT_BYTES = 40_000 * 2**20
CUDA_PREFLIGHT_SCOPE = "depi_formal_cuda_single_update_preflight"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _host(value: Any) -> Any:
    """Convert device values to lossless JSON-compatible records."""

    if isinstance(value, Mapping):
        return {str(key): _host(item) for key, item in value.items()}
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return {name: _host(getattr(value, name)) for name in value._fields}
    if isinstance(value, (tuple, list)):
        return [_host(item) for item in value]
    try:
        array = np.asarray(value)
    except Exception:
        return value
    if array.ndim == 0:
        return array.item()
    return array.tolist()


def _gpu_snapshot() -> Mapping[str, Any]:
    try:
        completed = subprocess.run(
            (
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total",
                "--format=csv,noheader,nounits",
            ),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        rows = []
        for line in completed.stdout.splitlines():
            values = [value.strip() for value in line.split(",")]
            if len(values) == 6:
                rows.append(
                    {
                        "physical_index": int(values[0]),
                        "uuid": values[1],
                        "memory_total_mib": int(values[2]),
                        "memory_used_mib": int(values[3]),
                        "utilization_percent": int(values[4]),
                        "volatile_uncorrected_ecc": values[5],
                    }
                )
        return {"gpus": rows, "captured_at": _utc_now()}
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}", "captured_at": _utc_now()}


def _synchronize(value: Any) -> None:
    import jax

    for leaf in jax.tree_util.tree_leaves(value):
        block = getattr(leaf, "block_until_ready", None)
        if callable(block):
            block()


def _phase(
    output: Path,
    *,
    name: str,
    update: int,
    function: Callable[[], Any],
) -> tuple[Any, float]:
    started = time.perf_counter()
    write_json_atomic(
        output / "phase_status.json",
        {
            "status": "running",
            "phase": name,
            "update": int(update),
            "started_at": _utc_now(),
            "gpu": _gpu_snapshot(),
        },
    )
    try:
        value = function()
        _synchronize(value)
    except Exception as error:
        write_json_atomic(
            output / "phase_status.json",
            {
                "status": "failed",
                "phase": name,
                "update": int(update),
                "failed_at": _utc_now(),
                "error": f"{type(error).__name__}: {error}",
                "gpu": _gpu_snapshot(),
            },
        )
        raise
    elapsed = time.perf_counter() - started
    write_json_atomic(
        output / "phase_status.json",
        {
            "status": "complete",
            "phase": name,
            "update": int(update),
            "completed_at": _utc_now(),
            "wall_seconds": elapsed,
            "gpu": _gpu_snapshot(),
        },
    )
    return value, elapsed


def _owned_runs(manifest: Any, role: str, seed_index: int) -> tuple[Any, ...]:
    owner = None if int(seed_index) == ENGINEERING_SEED_INDEX else int(seed_index)
    return tuple(
        run
        for run in manifest.runs
        if run.role == role and run.owner_seed_index == owner
    )


def _pool(runs: tuple[Any, ...]) -> FrozenPartnerPool:
    if not runs:
        raise ValueError("A required DEPI frozen-partner pool is empty.")
    return FrozenPartnerPool.from_checkpoints(
        [run.checkpoint for run in runs],
        parent_training_run_ids=[run.parent_training_run_id for run in runs],
    )


def _model_structure_fingerprint(config: Any, observation_shape: tuple[int, ...]) -> str:
    payload = {
        "method": METHOD_VERSION,
        "model": config.to_mapping()["model"],
        "observation_shape": list(observation_shape),
        "action_count": 6,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _fingerprint_words(tree: Any) -> np.ndarray:
    digest = bytes.fromhex(pytree_fingerprint(tree))[:8]
    return np.frombuffer(digest, dtype=">u4").astype(np.uint32)


def _empty_supervision_payload(
    *, config: Any, observation_shape: tuple[int, ...]
) -> tuple[Any, Any, FrozenPairComparator]:
    """Fixed-shape inactive values used by checkpoint schema 5."""

    import jax.numpy as jnp

    anchor_count = int(config.anchors.ordinary_states) + 2 * int(
        config.anchors.matched_history_pairs
    )
    pair_count = int(config.anchors.matched_history_pairs)
    policy = initial_policy_state(
        batch_size=anchor_count,
        observation_shape=observation_shape,
        action_count=6,
        task_hidden_dim=config.model.task_hidden_dim,
        capability_hidden_dim=config.model.capability_hidden_dim,
        capability_dim=config.model.capability_dim,
        component_embedding_dim=config.model.component_embedding_dim,
        protocol_components=config.model.protocol_components,
    )
    zeros_action = jnp.zeros((anchor_count, 6), dtype=jnp.float32)
    anchors = CounterfactualAnchorBatch(
        anchor_ids=jnp.full((anchor_count,), -1, dtype=jnp.int32),
        rollout_flat_indexes=jnp.zeros((anchor_count,), dtype=jnp.int32),
        policy_states=policy,
        observations=jnp.zeros((anchor_count,) + observation_shape, dtype=jnp.float32),
        partner_sources=jnp.zeros((anchor_count,), dtype=jnp.int32),
        partner_members=jnp.zeros((anchor_count,), dtype=jnp.int32),
        partner_family_ids=jnp.zeros((anchor_count,), dtype=jnp.int32),
        partner_checkpoint_stages=jnp.ones(
            (anchor_count,), dtype=jnp.float32
        ),
        partner_run_ids=jnp.zeros((anchor_count,), dtype=jnp.int32),
        fit_returns_by_action=zeros_action,
        return_sum_by_action=zeros_action,
        return_squared_sum_by_action=zeros_action,
        replica_count=jnp.zeros((anchor_count, 6), dtype=jnp.int32),
        collection_policy_logits=zeros_action,
        collection_update=jnp.zeros((anchor_count,), dtype=jnp.int32),
        collection_target_fingerprint=jnp.zeros((anchor_count, 2), dtype=jnp.uint32),
        matched_pair_ids=jnp.full((anchor_count,), -1, dtype=jnp.int32),
        action_mask=jnp.zeros((anchor_count, 6), dtype=jnp.bool_),
        evaluation_returns_by_action=jnp.zeros_like(zeros_action),
        evaluation_replica_count=jnp.zeros((anchor_count, 6), dtype=jnp.int32),
    )
    pair_policy = jax_tree_take(policy, pair_count)
    separation = SeparationTerms(
        ego_state_a=pair_policy,
        ego_state_b=pair_policy,
        probe_observations=jnp.zeros(
            (2, pair_count) + observation_shape, dtype=jnp.float32
        ),
        equivalent_mask=jnp.zeros((pair_count,), dtype=jnp.bool_),
        weights=jnp.zeros((pair_count,), dtype=jnp.float32),
        margin=jnp.asarray(0.0, dtype=jnp.float32),
        pair_valid=jnp.zeros((pair_count,), dtype=jnp.bool_),
    )
    ingredient_count = (int(observation_shape[-1]) - 27) // 4
    history_dim = int(config.anchors.probe_steps) * (38 + ingredient_count + 2)
    comparator = FrozenPairComparator(
        weights=jnp.zeros((2 * history_dim,), dtype=jnp.float32),
        bias=jnp.asarray(0.0, dtype=jnp.float32),
        train_accuracy=jnp.asarray(jnp.nan, dtype=jnp.float32),
        validation_accuracy=jnp.asarray(jnp.nan, dtype=jnp.float32),
        validation_accuracy_interval=(float("nan"), float("nan")),
        history_feature_dim=history_dim,
        pair_feature_dim=2 * history_dim,
        development_row_count=0,
        validation_row_count=0,
        validation_partner_run_count=0,
        development_partner_run_ids=jnp.full(
            (COMPARATOR_RUN_ID_CAPACITY,), -1, dtype=jnp.int32
        ),
        development_partner_run_count=0,
    )
    return anchors, separation, comparator


def _empty_m1_history(capacity: int) -> Mapping[str, Any]:
    """Fixed-shape M1 ledger stored atomically inside ``TrainState``."""

    import jax.numpy as jnp

    count = max(int(capacity), 1)
    nan = jnp.full
    return {
        "updates": jnp.full((count,), -1, dtype=jnp.int32),
        "passed": jnp.zeros((count,), dtype=jnp.bool_),
        "path_passing_fractions": nan((count, 4), jnp.nan, dtype=jnp.float32),
        "path_mean_spearman": nan((count, 4), jnp.nan, dtype=jnp.float32),
        "path_top_action_agreement": nan((count, 4), jnp.nan, dtype=jnp.float32),
        "path_mean_value_regret": nan((count, 4), jnp.nan, dtype=jnp.float32),
        "bootstrap_training_loss": nan((count, 3), jnp.nan, dtype=jnp.float32),
        "pair_class_fractions": nan((count, 3), jnp.nan, dtype=jnp.float32),
        "irreducible_ambiguity_fraction": nan(
            (count,), jnp.nan, dtype=jnp.float32
        ),
        "separation_margin": nan((count,), jnp.nan, dtype=jnp.float32),
        "valid_pair_fraction": nan((count,), jnp.nan, dtype=jnp.float32),
    }


def _append_m1_history(
    history: Mapping[str, Any],
    *,
    index: int,
    update: int,
    result: Any,
    bootstrap_training_metrics: Mapping[str, Any],
    supervision_readings: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Return a fixed-shape ledger with one deterministic slot replaced."""

    import jax.numpy as jnp

    if not 0 <= int(index) < int(np.asarray(history["updates"]).shape[0]):
        raise RuntimeError("M1 checkpoint history capacity is exhausted.")
    values = dict(history)
    values["updates"] = values["updates"].at[index].set(int(update))
    values["passed"] = values["passed"].at[index].set(result.passed)
    for name in (
        "path_passing_fractions",
        "path_mean_spearman",
        "path_top_action_agreement",
        "path_mean_value_regret",
    ):
        values[name] = values[name].at[index].set(getattr(result, name))
    values["bootstrap_training_loss"] = values["bootstrap_training_loss"].at[
        index
    ].set(bootstrap_training_metrics["bootstrap_training_loss"])
    for name in (
        "pair_class_fractions",
        "irreducible_ambiguity_fraction",
        "separation_margin",
        "valid_pair_fraction",
    ):
        values[name] = values[name].at[index].set(
            jnp.asarray(supervision_readings[name])
        )
    return values


def _m1_history_records(
    history: Mapping[str, Any], *, count: int
) -> list[dict[str, Any]]:
    """Materialize the checkpoint ledger as the human-readable JSON report."""

    host = _host(history)
    records: list[dict[str, Any]] = []
    for index in range(int(count)):
        records.append(
            {
                "update": int(host["updates"][index]),
                "passed": bool(host["passed"][index]),
                "path_passing_fractions": host["path_passing_fractions"][index],
                "path_mean_spearman": host["path_mean_spearman"][index],
                "path_top_action_agreement": host[
                    "path_top_action_agreement"
                ][index],
                "path_mean_value_regret": host["path_mean_value_regret"][index],
                "bootstrap_training_loss": host["bootstrap_training_loss"][index],
                "separation_readings": {
                    "pair_class_fractions": host["pair_class_fractions"][index],
                    "irreducible_ambiguity_fraction": host[
                        "irreducible_ambiguity_fraction"
                    ][index],
                    "separation_margin": host["separation_margin"][index],
                    "valid_pair_fraction": host["valid_pair_fraction"][index],
                },
            }
        )
    return records


def jax_tree_take(tree: Any, count: int) -> Any:
    import jax
    import jax.numpy as jnp

    return jax.tree_util.tree_map(lambda value: jnp.asarray(value)[: int(count)], tree)


def _checkpoint_identity(state: TrainState) -> Mapping[str, Any]:
    """Fingerprint every value whose restoration can change the next update.

    The sidecar is intentionally stricter than Orbax's structural restore.  In
    particular it binds recurrent/random state, supervision state, and every
    optimizer.  A checkpoint from a previous method cannot be adopted merely
    because a subset of parameter leaves happens to match.
    """

    return {
        "method": METHOD_VERSION,
        "config_version": CONFIG_VERSION,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "update_count": int(np.asarray(state.update_count)),
        "effective_environment_steps": int(
            np.asarray(state.effective_environment_steps)
        ),
        "params": pytree_fingerprint(state.params),
        "target_params": pytree_fingerprint(state.target_params),
        "ppo_optimizer_state": pytree_fingerprint(state.ppo_optimizer_state),
        "supervision_anchor_batch": pytree_fingerprint(
            state.supervision_anchor_batch
        ),
        "current_separation_terms": pytree_fingerprint(
            state.current_separation_terms
        ),
        "current_pair_comparator": pytree_fingerprint(
            state.current_pair_comparator
        ),
        "supervision_readings": pytree_fingerprint(state.supervision_readings),
        "anchor_sampling_counter": int(np.asarray(state.anchor_sampling_counter)),
        "anchor_microbatch_size": int(np.asarray(state.anchor_microbatch_size)),
        "effective_update_epochs": int(np.asarray(state.effective_update_epochs)),
        "bootstrap_encoder_params": pytree_fingerprint(
            state.bootstrap_encoder_params
        ),
        "bootstrap_optimizer_states": pytree_fingerprint(
            state.bootstrap_optimizer_states
        ),
        "bootstrap_sampling_counters": pytree_fingerprint(
            state.bootstrap_sampling_counters
        ),
        "m1_summary_state": pytree_fingerprint(state.m1_summary_state),
        "m1_history": pytree_fingerprint(state.m1_history),
        "runner_state": pytree_fingerprint(state.runner_state),
        "optimizer_steps": {
            "ppo": int(np.asarray(state.ppo_optimizer_step)),
        },
        "random_domains": _host(state.random_domains),
        "resource_ledger": dict(state.resource_ledger),
    }


def _save_depi_checkpoint(manager: Any, output: Path, state: TrainState) -> None:
    step = int(np.asarray(state.effective_environment_steps))
    save_checkpoint(manager, step=step, state=state)
    write_json(
        output / "checkpoint_identities" / f"{step:012d}.json",
        _checkpoint_identity(state),
    )


def _restore_depi_checkpoint(
    manager: Any, output: Path, template: TrainState
) -> TrainState | None:
    restored = restore_latest_checkpoint(manager, item=template)
    if restored is None:
        return None
    step, state = restored
    if not isinstance(state, TrainState):
        state = TrainState(*state) if isinstance(state, tuple) else TrainState(**state)
    path = output / "checkpoint_identities" / f"{step:012d}.json"
    if not path.is_file():
        raise RuntimeError("DEPI checkpoint has no atomic resume identity.")
    expected = json.loads(path.read_text(encoding="utf-8"))
    observed = json.loads(json.dumps(_checkpoint_identity(state), sort_keys=True))
    if expected != observed:
        raise RuntimeError(
            "Checkpoint parameters, supervision state, RNG, or optimizer identity differs."
        )
    if int(np.asarray(state.effective_environment_steps)) != int(step):
        raise RuntimeError("Checkpoint step disagrees with DEPI TrainState.")
    return state


def _validate_preflight_roundtrips(
    *,
    manager: Any,
    output: Path,
    state: TrainState,
    deployment_directory: Path,
    config: Any,
) -> Mapping[str, Any]:
    """Fail closed on both §5 preflight persistence round-trips.

    The checkpoint validator fingerprints every continuation-relevant state
    field.  The deployment validator restores only the registered whitelist,
    checks its fingerprint, and executes one real policy forward from the
    checkpointed runner carry and observation.
    """

    import jax
    import jax.numpy as jnp

    restored = _restore_depi_checkpoint(manager, output, state)
    if restored is None:
        raise RuntimeError("CUDA preflight checkpoint round-trip produced no state.")
    expected_checkpoint = _checkpoint_identity(state)
    observed_checkpoint = _checkpoint_identity(restored)
    if expected_checkpoint != observed_checkpoint:
        raise RuntimeError("CUDA preflight checkpoint round-trip is not exact.")

    loaded = load_deployment(deployment_directory, config)
    expected_deployment_fingerprint = pytree_fingerprint(
        deployable_parameters(state.params)
    )
    observed_deployment_fingerprint = pytree_fingerprint(loaded.params)
    if observed_deployment_fingerprint != expected_deployment_fingerprint:
        raise RuntimeError("CUDA preflight deployment round-trip is not exact.")

    runner = restored.runner_state
    lane_count = int(jnp.asarray(runner.ego_roles).shape[0])
    lanes = jnp.arange(lane_count, dtype=jnp.int32)
    observations = runner.observations[lanes, runner.ego_roles]
    keys = jax.random.split(jax.random.PRNGKey(0), lane_count)
    stepped, actions, output_values, log_probability = deployment_action(
        deployment=loaded,
        state=runner.ego_policy,
        observation=observations,
        keys=keys,
    )
    forward_values = {
        "stepped_state": stepped,
        "actions": actions,
        "policy_logits": output_values.policy_logits,
        "state_value": output_values.state_value,
        "protocol_probabilities": output_values.protocol_probabilities,
        "log_probability": log_probability,
    }
    if not _all_finite(forward_values):
        raise RuntimeError("CUDA preflight deployment forward is non-finite.")
    return {
        "checkpoint_roundtrip_passed": True,
        "checkpoint_state_fingerprint": pytree_fingerprint(restored),
        "deployment_roundtrip_passed": True,
        "deployment_params_fingerprint": observed_deployment_fingerprint,
        "deployment_forward_passed": True,
        "deployment_forward_lane_count": lane_count,
    }


def _upstream_partner_cost(runs: tuple[Any, ...], *, formal: bool) -> tuple[int, Any]:
    """Read explicit upstream ledgers; never invent hidden population cost."""

    total = 0
    records = []
    seen: set[str] = set()
    for run in runs:
        parent = str(run.parent_training_run_id)
        if parent in seen:
            continue
        seen.add(parent)
        checkpoint = Path(run.checkpoint).resolve()
        candidates = []
        for directory in (checkpoint, *checkpoint.parents[:5]):
            candidates.extend(
                (directory / "resource_ledger.json", directory / "upstream_summary.json")
            )
        ledger_path = next((path for path in candidates if path.is_file()), None)
        if ledger_path is None:
            records.append({"parent": parent, "status": "missing_explicit_ledger"})
            continue
        payload = json.loads(ledger_path.read_text(encoding="utf-8"))
        value = payload.get(
            "total_training_simulator_steps",
            payload.get("effective_environment_steps", payload.get("actual_timesteps")),
        )
        if value is None:
            records.append({"parent": parent, "status": "ledger_has_no_step_total"})
            continue
        total += int(value)
        records.append(
            {"parent": parent, "status": "counted", "steps": int(value), "path": str(ledger_path)}
        )
    if formal and any(record["status"] != "counted" for record in records):
        raise RuntimeError(
            "Formal DEPI resource accounting requires explicit upstream partner ledgers."
        )
    return total, records


def _all_finite(tree: Any) -> bool:
    import jax

    for leaf in jax.tree_util.tree_leaves(tree):
        values = np.asarray(jax.device_get(leaf))
        if values.size and not bool(np.all(np.isfinite(values))):
            return False
    return True


def _mean_metrics(metrics: Any) -> Any:
    import jax
    import jax.numpy as jnp

    return jax.tree_util.tree_map(lambda value: jnp.mean(value), metrics)


def _account_runtime_resources(
    state: TrainState, *, elapsed_wall_seconds: float
) -> TrainState:
    """Add one non-overlapping process interval to the checkpointed ledger."""

    ledger = ResourceLedger.from_mapping(state.resource_ledger).plus(
        gpu_hours=gpu_hours_for_wall_seconds(float(elapsed_wall_seconds))
    )
    ledger = ResourceLedger(
        **{
            **{
                name: getattr(ledger, name)
                for name in ResourceLedger.__dataclass_fields__
            },
            "peak_memory_bytes": max(
                ledger.peak_memory_bytes, peak_device_memory_bytes()
            ),
        }
    )
    return state._replace(resource_ledger=ledger.to_mapping())


def _anchor_policy_logits(model: Any, params: Any, anchors: Any) -> Any:
    import jax.numpy as jnp

    unused_state, output = model.apply(
        {"params": params},
        anchors.policy_states,
        anchors.observations,
        jnp.zeros(anchors.anchor_ids.shape, dtype=jnp.bool_),
        method=model.step,
    )
    del unused_state
    return output.policy_logits


def _write_training_support_latents(
    output: Path, *, model: Any, params: Any, anchors: Any, config: Any
) -> None:
    """Persist final-model support without adding simulator interaction.

    Posterior calibration must use the primary model's actual training
    support. Re-encoding the fixed anchor set under final parameters avoids mixing
    latent coordinates from earlier checkpoints.
    """

    import jax
    import jax.numpy as jnp

    if anchors is None:
        write_json(
            output / "records" / "training_support_latents.json",
            {
                "method": METHOD_VERSION,
                "config_fingerprint": config.fingerprint,
                "model_fingerprint": pytree_fingerprint(deployable_parameters(params)),
                "row_count": 0,
                "status": "skipped_no_supervision_anchors",
            },
        )
        return
    count = int(np.asarray(anchors.anchor_ids).shape[0])
    batch = anchors
    unused_state, output_values = model.apply(
        {"params": params},
        batch.policy_states,
        batch.observations,
        jnp.zeros((count,), dtype=jnp.bool_),
        method=model.step,
    )
    del unused_state
    write_json(
        output / "records" / "training_support_latents.json",
        {
            "method": METHOD_VERSION,
            "config_fingerprint": config.fingerprint,
            "model_fingerprint": pytree_fingerprint(deployable_parameters(params)),
            "row_count": count,
            "capability": _host(output_values.capability),
            "protocol_probabilities": _host(output_values.protocol_probabilities),
        },
    )


def run_training(args: argparse.Namespace) -> None:
    scope = str(getattr(args, "_execution_scope", "training"))
    preflight = scope == CUDA_PREFLIGHT_SCOPE
    config = load_config(args.config, run_kind=args.run_kind)
    # Every B0--B2 run spends the same registered anchor-continuation
    # simulator budget.  Only B2 is allowed to turn those measurements into
    # decision supervision; B0/B1 collect them as compute/budget controls.
    anchor_enabled = bool(config.anchors.enabled)
    decision_supervision_enabled = config.method_variant == "b2"
    cuda_toolchain = None
    if config.run_kind == "formal" or os.environ.get("DEPI_REQUIRE_CUDA") == "1":
        cuda_toolchain = configure_bundled_cuda_toolchain()

    import jax
    import jax.numpy as jnp

    if config.run_kind == "formal":
        validate_formal_repository_state()
        validate_registered_python_runtime()
    cuda = None
    if config.run_kind == "formal" or os.environ.get("DEPI_REQUIRE_CUDA") == "1":
        cuda = require_single_cuda_worker()
    official_runtime = validate_official_runtime() if config.run_kind == "formal" else None

    manifest = load_partner_manifest(
        args.partner_manifest,
        expected_layout=config.environment.layout,
        verify_files=not bool(args.skip_manifest_hash_check),
    )
    validate_seed_training_manifest(
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
    identity = dict(
        training_identity(
            config=config,
            seed_index=int(args.seed_index),
            jax_prng_key=outer_key,
            partner_manifest=manifest,
        )
    )
    structure_fingerprint = _model_structure_fingerprint(config, observation_shape)
    cache = configure_persistent_compilation_cache(
        repository_commit=str(identity["runtime"]["repository_commit"]),
        official_commit=OFFICIAL_SOURCE_COMMIT,
        config_fingerprint=config.fingerprint,
        model_structure_fingerprint=structure_fingerprint,
        cache_root=os.environ.get("DEPI_JAX_COMPILATION_CACHE"),
    )
    identity.update(
        {
            "ego_run_id": str(args.ego_run_id),
            "observation_shape": list(observation_shape),
            "action_count": 6,
            "official_runtime": official_runtime,
            "cuda_worker": cuda,
            "cuda_toolchain": cuda_toolchain,
            "execution_scope": scope,
            "model_structure_fingerprint": structure_fingerprint,
            "compilation_cache": cache,
            "scientific_readout_allowed": False,
        }
    )
    ensure_run_identity(output, identity)
    write_json(output / "resolved_config.json", config.to_mapping())
    write_json(output / "resolved_partner_manifest.json", manifest.to_mapping())
    write_json(output / "runtime_gpu.json", cuda or _gpu_snapshot())

    owner_runs = _owned_runs(manifest, "owner_source", int(args.seed_index))
    support_runs = tuple(
        run
        for run in manifest.runs
        if run.role == "development_support"
        and run.owner_seed_index in (None, int(args.seed_index))
    )
    owner_pool = _pool(owner_runs)
    support_manifest = type(manifest)(manifest.layout, support_runs)
    support_members = build_partner_pool(config, support_manifest, "train")
    comparator_fit_runs = tuple(manifest.by_role("comparator_fit"))
    comparator_validation_runs = tuple(manifest.by_role("comparator_validation"))
    comparator_fit_members = (
        build_partner_pool(config, manifest, "comparator_fit")
        if comparator_fit_runs
        else ()
    )
    comparator_validation_members = (
        build_partner_pool(config, manifest, "comparator_validation")
        if comparator_validation_runs
        else ()
    )
    if bool(comparator_fit_members) != bool(comparator_validation_members):
        raise RuntimeError("Comparator fit and validation pools must either both exist or both be absent.")
    if config.run_kind == "formal" and comparator_fit_members:
        reserved_lane_count = config.environment.num_envs // 8
        for label, members in (
            ("comparator_fit", comparator_fit_members),
            ("comparator_validation", comparator_validation_members),
        ):
            if min(member.probability for member in members) < 1.0 / reserved_lane_count:
                raise RuntimeError(
                    f"Formal {label} stratification cannot guarantee every run in its reserved lanes."
                )
    static_members = (
        *support_members,
        *comparator_fit_members,
        *comparator_validation_members,
    )
    if any(member.checkpoint is None for member in static_members):
        raise RuntimeError("A training partner-pool member has no frozen checkpoint.")
    support_external_pool = FrozenPartnerPool.from_checkpoints(
        [member.checkpoint for member in support_members],
        parent_training_run_ids=[
            member.parent_training_run_id for member in support_members
        ],
    )
    external_pool = FrozenPartnerPool.from_checkpoints(
        [member.checkpoint for member in static_members],
        parent_training_run_ids=[
            member.parent_training_run_id for member in static_members
        ],
    )
    static_member_probabilities = np.asarray(
        [member.probability for member in static_members], dtype=np.float32
    )
    static_member_sampling_groups = np.asarray(
        [
            0 if index < len(support_members) else (
                1
                if index < len(support_members) + len(comparator_fit_members)
                else 2
            )
            for index in range(len(static_members))
        ],
        dtype=np.int32,
    )
    family_index = {
        family: index
        for index, family in enumerate(
            sorted({member.family_id for member in static_members})
        )
    }
    static_member_family_ids = np.asarray(
        [family_index[member.family_id] for member in static_members], dtype=np.int32
    )
    static_member_stages = np.asarray(
        [member.checkpoint_stage for member in static_members], dtype=np.float32
    )
    write_json(
        output / "partner_pool.json",
        {
            "split": "train",
            "sampling": "family_then_stage_then_seed_uniform",
            "comparator_lane_allocation_before_freeze": (
                {"support": "6/8", "comparator_fit": "1/8", "comparator_validation": "1/8"}
                if comparator_fit_members
                else {"support": "8/8"}
            ),
            "members": [
                {
                    "run_id": member.run_id,
                    "family_id": member.family_id,
                    "hyperparameter_family": getattr(
                        member, "hyperparameter_family", "default"
                    ),
                    "mechanism": member.mechanism,
                    "checkpoint_stage": member.checkpoint_stage,
                    "seed": member.seed,
                    "probability": member.probability,
                    "partition": (
                        "support"
                        if static_member_sampling_groups[index] == 0
                        else (
                            "comparator_fit"
                            if static_member_sampling_groups[index] == 1
                            else "comparator_validation"
                        )
                    ),
                }
                for index, member in enumerate(static_members)
            ],
        },
    )
    upstream_steps, upstream_records = _upstream_partner_cost(
        tuple(
            {
                run.run_id: run
                for run in (
                    *owner_runs,
                    *support_runs,
                    *comparator_fit_runs,
                    *comparator_validation_runs,
                )
            }.values()
        ),
        formal=(config.run_kind == "formal" and not preflight),
    )
    write_json(output / "upstream_cost_manifest.json", {"runs": upstream_records})

    model = build_model(
        observation_shape=observation_shape,
        action_count=6,
        task_hidden_dim=config.model.task_hidden_dim,
        capability_hidden_dim=config.model.capability_hidden_dim,
        capability_dim=config.model.capability_dim,
        protocol_components=config.model.protocol_components,
        component_embedding_dim=config.model.component_embedding_dim,
        actor_hidden_dim=config.model.actor_hidden_dim,
        critic_hidden_dim=config.model.critic_hidden_dim,
        response_hidden_dim=config.model.response_hidden_dim,
        modulation_rank=config.model.modulation_rank,
        action_embedding_dim=config.model.action_embedding_dim,
        method_variant=config.method_variant,
    )
    ego_root = jnp.asarray(domains["ego"], dtype=jnp.uint32)
    reset_key, model_key, runner_key = jax.random.split(ego_root, 3)
    unused_state, initial_observations = environment.reset(reset_key)
    del unused_state
    example_policy = initial_policy_state(
        batch_size=environment.num_envs,
        observation_shape=observation_shape,
        action_count=6,
        task_hidden_dim=config.model.task_hidden_dim,
        capability_hidden_dim=config.model.capability_hidden_dim,
        capability_dim=config.model.capability_dim,
        component_embedding_dim=config.model.component_embedding_dim,
        protocol_components=config.model.protocol_components,
    )
    initial_params = initialize_model_parameters(
        model,
        key=model_key,
        example_state=example_policy,
        example_observation=initial_observations[:, 0],
    )
    bootstrap_models: Any = ()
    bootstrap_params: Any = ()
    bootstrap_optimizer_states: Any = ()
    bootstrap_sampling_counters = jnp.zeros((0,), dtype=jnp.int32)
    if anchor_enabled and decision_supervision_enabled:
        (
            bootstrap_models,
            bootstrap_params,
            bootstrap_optimizer_states,
            bootstrap_sampling_counters,
        ) = initialize_bootstrap_value_ensemble(
            key=jax.random.fold_in(model_key, 60),
            example_policy_state=example_policy,
            example_observation=initial_observations[:, 0],
            action_count=6,
        )
    total_updates = config.training.environment_steps // (
        config.environment.num_envs * config.training.rollout_length
    )

    new_run = not bool(args.resume)
    if new_run:
        if (output / "checkpoints").exists() and any((output / "checkpoints").iterdir()):
            raise RuntimeError("Existing checkpoints require explicit --resume.")
        owner_batch, unused = _phase(
            output,
            name="owner_initialization_collection",
            update=0,
            function=lambda: collect_owner_behavior(
                environment=environment,
                owner_pool=owner_pool,
                partner_pool=support_external_pool,
                length=config.training.rollout_length,
                key=jax.random.fold_in(ego_root, 1),
                partner_members=jax.random.categorical(
                    jax.random.fold_in(ego_root, 101),
                    jnp.log(
                        jnp.maximum(
                            jnp.asarray(
                                [member.probability for member in support_members],
                                dtype=jnp.float32,
                            ),
                            1.0e-12,
                        )
                    ),
                    shape=(environment.num_envs,),
                ).astype(jnp.int32),
            ),
        )
        del unused
        owner_optimizer, owner_optimizer_state = make_optimizer(
            initial_params,
            learning_rate=config.ppo.learning_rate,
            gradient_clip_norm=config.ppo.gradient_clip_norm,
            adam_epsilon=config.ppo.adam_epsilon,
        )
        owner_schedule = environment_minibatch_schedule(
            jax.random.fold_in(ego_root, 2),
            environment_count=environment.num_envs,
            minibatches_per_epoch=config.training.minibatches_per_epoch,
            update_epochs=config.ppo.update_epochs,
        )
        (params, unused_owner_state, owner_metrics), unused = _phase(
            output,
            name="owner_actor_distillation",
            update=0,
            function=lambda: distill_owner_actor(
                model=model,
                params=initial_params,
                batch=owner_batch,
                initial_state_factory=lambda count: initial_policy_state(
                    batch_size=count,
                    observation_shape=observation_shape,
                    action_count=6,
                    task_hidden_dim=config.model.task_hidden_dim,
                    capability_hidden_dim=config.model.capability_hidden_dim,
                    capability_dim=config.model.capability_dim,
                    component_embedding_dim=config.model.component_embedding_dim,
                    protocol_components=config.model.protocol_components,
                ),
                optimizer=owner_optimizer,
                optimizer_state=owner_optimizer_state,
                schedule=owner_schedule,
            ),
        )
        del unused_owner_state, unused
    else:
        params = initial_params
        owner_metrics = {"owner_distillation_kl": jnp.asarray(np.nan)}

    ppo_optimizer, ppo_state = make_optimizer(
        params,
        learning_rate=config.ppo.learning_rate,
        gradient_clip_norm=config.ppo.gradient_clip_norm,
        adam_epsilon=config.ppo.adam_epsilon,
        anneal_learning_rate=config.ppo.anneal_learning_rate,
        warmup_fraction=config.ppo.lr_warmup_fraction,
        update_count=total_updates,
        minibatches_per_epoch=config.training.minibatches_per_epoch,
        update_epochs=config.ppo.update_epochs,
    )
    partner_functions = make_static_pool_partner_functions(
        external_pool=external_pool,
        member_probabilities=static_member_probabilities,
        member_family_ids=static_member_family_ids,
        member_checkpoint_stages=static_member_stages,
        member_sampling_groups=(
            static_member_sampling_groups if comparator_fit_members else None
        ),
    )
    runner = initialize_runner(
        environment=environment,
        model_config=config.model,
        partner_functions=partner_functions,
        random_key=runner_key,
    )
    initial_ledger = ResourceLedger(
        ego_initialization_steps=(
            0 if not new_run else environment.num_envs * config.training.rollout_length
        ),
        upstream_partner_steps=upstream_steps,
    )
    (
        empty_supervision_anchors,
        empty_separation_terms,
        empty_pair_comparator,
    ) = _empty_supervision_payload(
        config=config, observation_shape=observation_shape
    )
    template = TrainState(
        params=params,
        target_params=jax.tree_util.tree_map(jnp.copy, params),
        ppo_optimizer_state=ppo_state,
        supervision_anchor_batch=empty_supervision_anchors,
        current_separation_terms=empty_separation_terms,
        current_pair_comparator=empty_pair_comparator,
        supervision_readings={
            "pair_class_fractions": jnp.zeros((3,), dtype=jnp.float32),
            "irreducible_ambiguity_fraction": jnp.asarray(0.0, dtype=jnp.float32),
            "separation_margin": jnp.asarray(0.0, dtype=jnp.float32),
            "valid_pair_fraction": jnp.asarray(0.0, dtype=jnp.float32),
        },
        anchor_sampling_counter=jnp.asarray(0, dtype=jnp.int32),
        anchor_microbatch_size=jnp.asarray(0, dtype=jnp.int32),
        effective_update_epochs=jnp.asarray(
            config.ppo.update_epochs, dtype=jnp.int32
        ),
        bootstrap_encoder_params=bootstrap_params,
        bootstrap_optimizer_states=bootstrap_optimizer_states,
        bootstrap_sampling_counters=bootstrap_sampling_counters,
        m1_summary_state={
            "evaluations": jnp.asarray(0, dtype=jnp.int32),
            "latest_passed": jnp.asarray(False),
        },
        m1_history=_empty_m1_history(
            (total_updates + int(config.anchors.interval_updates) - 1)
            // int(config.anchors.interval_updates)
            + 1
        ),
        ppo_optimizer_step=jnp.asarray(0, dtype=jnp.int32),
        runner_state=runner,
        random_domains={name: jnp.asarray(value, dtype=jnp.uint32) for name, value in domains.items()},
        update_count=jnp.asarray(0, dtype=jnp.int32),
        effective_environment_steps=jnp.asarray(0, dtype=jnp.int32),
        resource_ledger=initial_ledger.to_mapping(),
    )
    manager = orbax_manager(output / "checkpoints")
    if args.resume:
        restored = _restore_depi_checkpoint(manager, output, template)
        if restored is None:
            raise FileNotFoundError("--resume requested but no DEPI checkpoint exists.")
        state = restored
    else:
        state = template

    kernels = build_training_kernels(
        environment=environment,
        model=model,
        model_config=config.model,
        partner_functions=partner_functions,
        config=config,
        ppo_optimizer=ppo_optimizer,
    )
    anchor_functions = None
    anchor_kernel = None
    if anchor_enabled:
        anchor_functions = make_anchor_functions(
            model=model,
            model_config=config.model,
            partner_functions=partner_functions,
            environment=environment,
        )
        anchor_kernel = build_anchor_chunk_kernel(
            functions=anchor_functions,
            config=config,
            memory_limit_bytes=FORMAL_PEAK_MEMORY_LIMIT_BYTES,
        )

    # §3.3/§5: the most recent anchor batch and its matched-pair separation
    # terms supervise every combined-loss update until the next anchor trigger
    # refreshes them; both stay None while anchors are disabled.
    supervision_active = (
        decision_supervision_enabled
        and int(np.asarray(state.anchor_sampling_counter)) > 0
    )
    supervision_anchors = (
        state.supervision_anchor_batch if supervision_active else None
    )
    supervision_separation = (
        state.current_separation_terms if supervision_active else None
    )
    supervision_readings: dict[str, Any] = dict(state.supervision_readings)
    m1_history_path = output / "m1_gate.json"
    m1_history = _m1_history_records(
        state.m1_history,
        count=int(np.asarray(state.m1_summary_state["evaluations"])),
    )
    stored_microbatch = int(np.asarray(state.anchor_microbatch_size))
    anchor_microbatch = stored_microbatch if stored_microbatch > 0 else None
    start_update = int(np.asarray(state.update_count))
    last_update = min(total_updates, start_update + 1) if preflight else total_updates
    wall_started = time.perf_counter()
    wall_accounted_at = wall_started
    # §3.4: when the combined-policy-KL gate fires inside the scan, shrink the
    # next update's epoch count (floor 2) and recover one epoch per clean update.
    effective_update_epochs = int(np.asarray(state.effective_update_epochs))
    bootstrap_params = state.bootstrap_encoder_params
    bootstrap_optimizer_states = state.bootstrap_optimizer_states
    bootstrap_sampling_counters = state.bootstrap_sampling_counters

    for update in range(start_update, last_update):
        phase_times: dict[str, float] = {}
        snapshot_params = state.params
        snapshot_target = state.target_params
        partner_parameters = jnp.asarray(
            int(state.current_pair_comparator.development_row_count) > 0,
            dtype=jnp.bool_,
        )
        dropout = context_dropout_probability(
            state.effective_environment_steps,
            total_steps=config.training.environment_steps,
            initial=config.training.context_dropout_initial,
            final=config.training.context_dropout_final,
        )
        shaping_factor = official_reward_shaping_factor(
            state.effective_environment_steps,
            horizon=config.upstream.reward_shaping_horizon,
        )
        anchor_trigger = anchor_enabled and update % config.anchors.interval_updates == 0
        rollout_kernel = kernels.rollout_anchor_full if anchor_trigger else kernels.rollout_minimal
        (rollout_result, elapsed) = _phase(
            output,
            name="rollout",
            update=update,
            function=lambda: rollout_kernel(
                state.runner_state,
                snapshot_params,
                snapshot_target,
                partner_parameters,
                dropout,
                jax.random.fold_in(
                    jnp.asarray(domains["context_dropout"], dtype=jnp.uint32),
                    update,
                ),
                shaping_factor,
            ),
        )
        phase_times["rollout"] = elapsed
        runner, batch, records = rollout_result
        if batch is None:
            raise RuntimeError("Training rollout did not return the DEPI PPO batch.")

        next_pair_comparator = state.current_pair_comparator
        next_m1_summary = state.m1_summary_state
        next_m1_history = state.m1_history
        anchor_budget = {
            "counterfactual_continuation_steps": 0,
            "matched_pair_probe_steps": 0,
        }
        if anchor_trigger:
            if anchor_microbatch is None:
                anchor_microbatch = preflight_anchor_microbatch_from_records(
                    records=records,
                    target_params=snapshot_target,
                    partner_parameters=partner_parameters,
                    config=config,
                    chunk_kernel=anchor_kernel,
                    key=jax.random.fold_in(
                        jnp.asarray(domains["training_anchor"], dtype=jnp.uint32),
                        300_000 + update,
                    ),
                )
            (anchor_result, elapsed) = _phase(
                output,
                name="counterfactual_anchor",
                update=update,
                function=lambda: collect_anchor_batch(
                    anchor_domain=update + 1,
                    key=jax.random.fold_in(
                        jnp.asarray(domains["training_anchor"], dtype=jnp.uint32), update
                    ),
                    records=records,
                    environment=environment,
                    model=model,
                    target_params=snapshot_target,
                    config=config,
                    partner_functions=partner_functions,
                    partner_parameters=partner_parameters,
                    collection_update=update,
                    target_fingerprint=jnp.asarray(
                        _fingerprint_words(snapshot_target), dtype=jnp.uint32
                    ),
                    microbatch_size=anchor_microbatch,
                    anchor_functions=anchor_functions,
                    chunk_kernel=anchor_kernel,
                    frozen_comparator=(
                        state.current_pair_comparator
                        if int(state.current_pair_comparator.development_row_count) > 0
                        else None
                    ),
                ),
            )
            phase_times["counterfactual_anchor"] = elapsed
            anchors, quotient_pairs, anchor_budget, next_pair_comparator = anchor_result
            supervision_anchors = anchors if decision_supervision_enabled else None
            supervision_separation = None
            supervision_readings = dict(state.supervision_readings)
            if (
                decision_supervision_enabled
                and quotient_pairs.comparator_distinct_probability is not None
            ):
                # §5.3 -> §3.2: matched-pair classification feeds L_separation.
                # Only the classification (masks/weights/margin) is built here;
                # the loss forward passes run inside the jit-compiled scan on
                # the current params so the gradient is live (§3.2/§3.4).
                (
                    supervision_separation,
                    supervision_readings,
                ) = separation_terms_from_matched_pairs(
                    quotient=quotient_pairs,
                    equivalent_probability_max=float(
                        config.anchors.observable_equivalent_probability_max
                    ),
                    distinct_probability_min=float(
                        config.anchors.decision_distinct_probability_min
                    ),
                    signature_threshold=float(
                        config.anchors.signature_distance_threshold
                    ),
                    margin_scale=float(config.loss_v2.separation_margin_scale),
                )
            if decision_supervision_enabled:
                (
                    bootstrap_params,
                    bootstrap_optimizer_states,
                    bootstrap_sampling_counters,
                    bootstrap_training_metrics,
                ) = train_bootstrap_value_ensemble(
                    models=bootstrap_models,
                    params=bootstrap_params,
                    optimizer_states=bootstrap_optimizer_states,
                    sampling_counters=bootstrap_sampling_counters,
                    anchors=anchors,
                    key=jax.random.fold_in(
                        jnp.asarray(domains["training_anchor"], dtype=jnp.uint32),
                        900_000 + update,
                    ),
                )
                # §6 M1 uses independent evaluation replicas and is report-only.
                (gate_result, elapsed) = _phase(
                    output,
                    name="m1_gate",
                    update=update,
                    function=lambda: evaluate_m1_gate_on_anchor_batch(
                        model=model,
                        params=snapshot_params,
                        bootstrap_models=bootstrap_models,
                        bootstrap_params=bootstrap_params,
                        anchors=anchors,
                        spearman_threshold=float(config.anchors.m1_spearman_threshold),
                        minimum_anchor_fraction=float(
                            config.anchors.m1_minimum_anchor_fraction
                        ),
                    ),
                )
                phase_times["m1_gate"] = elapsed
                gate_payload = {
                    "update": update + 1,
                    **m1_gate_report(gate_result),
                    **_host(bootstrap_training_metrics),
                    "separation_readings": _host(supervision_readings),
                }
                next_m1_summary = {
                    "evaluations": state.m1_summary_state["evaluations"] + 1,
                    "latest_passed": gate_result.passed,
                }
                next_m1_history = _append_m1_history(
                    state.m1_history,
                    index=int(np.asarray(state.m1_summary_state["evaluations"])),
                    update=update + 1,
                    result=gate_result,
                    bootstrap_training_metrics=bootstrap_training_metrics,
                    supervision_readings=supervision_readings,
                )
                m1_history = _m1_history_records(
                    next_m1_history,
                    count=int(np.asarray(next_m1_summary["evaluations"])),
                )
                write_json(
                    m1_history_path,
                    {
                        "specification": "METHOD_SPEC §7 extended M1 gate",
                        "history": m1_history,
                        "latest": gate_payload,
                    },
                )
            # M1 is a report-only model-quality diagnostic. A failed value
            # mechanism is recorded but never changes the training path.
        schedule = environment_minibatch_schedule(
            jax.random.fold_in(
                jnp.asarray(domains["ego"], dtype=jnp.uint32), 400_000 + update
            ),
            environment_count=config.environment.num_envs,
            minibatches_per_epoch=config.training.minibatches_per_epoch,
            update_epochs=effective_update_epochs,
        )
        from src.path_c.types import TrainingCoreState

        (ppo_result, elapsed) = _phase(
            output,
            name="ppo",
            update=update,
            function=lambda: kernels.ppo_scan(
                TrainingCoreState(
                    state.params, state.target_params, state.ppo_optimizer_state
                ),
                batch,
                schedule,
                supervision_anchors,
                supervision_separation,
            ),
        )
        phase_times["ppo"] = elapsed
        ppo_core, ppo_metrics = ppo_result
        if bool(
            np.any(np.asarray(ppo_metrics["training_aborted"]) > 0.5)
        ):
            raise FloatingPointError(
                "NaN or Inf occurred during the DEPI combined-loss minibatch scan."
            )
        # §3.4 combined-policy-KL gate: adapt the next update's epoch count
        # (shrink towards the floor of two, recover one epoch per clean update).
        (
            effective_update_epochs,
            kl_stop_fraction,
            kl_stop_occurred,
        ) = adapt_effective_update_epochs(
            ppo_metrics["kl_early_stop"],
            current_epochs=effective_update_epochs,
            maximum_epochs=int(config.ppo.update_epochs),
        )
        # The single combined objective has one optimizer state.
        next_params = ppo_core.params

        next_target_params = polyak_update(
            state.target_params, next_params, config.ppo.polyak_coefficient
        )
        if next_pair_comparator is None:
            next_pair_comparator = state.current_pair_comparator

        completed_steps = int(np.asarray(runner.effective_environment_steps))
        ledger = ResourceLedger.from_mapping(state.resource_ledger).plus(
            ego_policy_steps=config.environment.num_envs * config.training.rollout_length,
            counterfactual_continuation_steps=int(
                anchor_budget["counterfactual_continuation_steps"]
            ),
            matched_pair_probe_steps=int(anchor_budget["matched_pair_probe_steps"]),
        )
        ppo_applied = int(
            np.asarray(jnp.sum(ppo_metrics["optimizer_applied"], dtype=jnp.int32))
        )
        state = state._replace(
            params=next_params,
            target_params=next_target_params,
            ppo_optimizer_state=ppo_core.ppo_optimizer_state,
            supervision_anchor_batch=(
                supervision_anchors
                if supervision_anchors is not None
                else state.supervision_anchor_batch
            ),
            current_separation_terms=(
                supervision_separation
                if supervision_separation is not None
                else state.current_separation_terms
            ),
            current_pair_comparator=next_pair_comparator,
            supervision_readings=supervision_readings,
            anchor_sampling_counter=(
                state.anchor_sampling_counter + int(anchor_trigger)
            ),
            anchor_microbatch_size=jnp.asarray(
                0 if anchor_microbatch is None else anchor_microbatch,
                dtype=jnp.int32,
            ),
            effective_update_epochs=jnp.asarray(
                effective_update_epochs, dtype=jnp.int32
            ),
            bootstrap_encoder_params=bootstrap_params,
            bootstrap_optimizer_states=bootstrap_optimizer_states,
            bootstrap_sampling_counters=bootstrap_sampling_counters,
            m1_summary_state=next_m1_summary,
            m1_history=next_m1_history,
            ppo_optimizer_step=state.ppo_optimizer_step + ppo_applied,
            runner_state=runner,
            update_count=jnp.asarray(update + 1, dtype=jnp.int32),
            effective_environment_steps=jnp.asarray(completed_steps, dtype=jnp.int32),
            resource_ledger=ledger.to_mapping(),
        )
        if not _all_finite((state.params, state.target_params)):
            raise FloatingPointError("NaN or Inf entered the DEPI training state.")

        metrics = {
            "method": METHOD_VERSION,
            "update_count": update + 1,
            "effective_environment_steps": completed_steps,
            "partner_source": "registered_static_pool",
            "context_dropout_probability": float(np.asarray(dropout)),
            "official_shaping_factor": float(np.asarray(shaping_factor)),
            # Four-loss combined objective metrics (§3): PPO, signature,
            # response, separation plus combined_policy_kl and posterior
            # entropy, all recorded per minibatch inside the scan.
            "ppo": _host(_mean_metrics(ppo_metrics)),
            "combined_policy_kl_stop_fraction": kl_stop_fraction,
            "combined_policy_kl_stop_occurred": kl_stop_occurred,
            "effective_update_epochs": int(effective_update_epochs),
            # §5 anchor supervision readouts (per-anchor-trigger refresh; the
            # carried supervision payload trains every combined-loss update).
            "anchor_supervision": {
                "active": supervision_anchors is not None,
                "anchor_trigger": anchor_trigger,
                "separation_terms": (
                    {
                        "margin": float(np.asarray(supervision_separation.margin)),
                        "pair_count": int(
                            np.asarray(supervision_separation.equivalent_mask).shape[0]
                        ),
                        "equivalent_pair_count": int(
                            np.sum(np.asarray(supervision_separation.equivalent_mask))
                        ),
                    }
                    if supervision_separation is not None
                    else None
                ),
                "readings": _host(supervision_readings),
            },
            "m1_gate": m1_history[-1] if m1_history else {"evaluated": False},
            "anchor_usage": {
                "anchor_trigger": anchor_trigger,
                "stored_anchor_count": (
                    int(np.asarray(state.supervision_anchor_batch.anchor_ids).shape[0])
                    if state.supervision_anchor_batch is not None
                    else 0
                ),
                "invalid_matched_probe_states": int(
                    anchor_budget.get("invalid_matched_probe_states", 0)
                ),
            },
            "phase_wall_seconds": phase_times,
            "gpu": _gpu_snapshot(),
            "scientific_readout": False,
        }
        write_jsonl(
            output / "records" / "metrics" / f"update_{update + 1:08d}.jsonl",
            (metrics,),
        )
        if (
            completed_steps
            % config.training.checkpoint_interval_environment_steps
            == 0
            and completed_steps < config.training.environment_steps
            and not preflight
        ):
            now = time.perf_counter()
            state = _account_runtime_resources(
                state, elapsed_wall_seconds=now - wall_accounted_at
            )
            wall_accounted_at = now
            _save_depi_checkpoint(manager, output, state)

    final_m1_payload: Mapping[str, Any] | None = None
    final_model_fingerprint = pytree_fingerprint(
        deployable_parameters(state.params)
    )
    evaluation_count = int(np.asarray(state.m1_summary_state["evaluations"]))
    if m1_history_path.is_file():
        existing_m1 = json.loads(m1_history_path.read_text(encoding="utf-8"))
        existing_final = existing_m1.get("final")
        if (
            existing_m1.get("version") == 1
            and existing_m1.get("artifact_type") == "depi_final_m1_evaluation"
            and isinstance(existing_final, Mapping)
            and existing_final.get("final_checkpoint_condition") is True
            and existing_final.get("model_fingerprint") == final_model_fingerprint
            and isinstance(existing_m1.get("history"), list)
            and len(existing_m1["history"]) == evaluation_count
        ):
            # A completed run resumed after the final checkpoint must not append
            # a second final evaluation or exhaust the fixed checkpoint ledger.
            final_m1_payload = existing_final
    if (
        decision_supervision_enabled
        and int(np.asarray(state.anchor_sampling_counter)) > 0
        and final_m1_payload is None
    ):
        final_gate = evaluate_m1_gate_on_anchor_batch(
            model=model,
            params=state.params,
            bootstrap_models=bootstrap_models,
            bootstrap_params=state.bootstrap_encoder_params,
            anchors=state.supervision_anchor_batch,
            spearman_threshold=float(config.anchors.m1_spearman_threshold),
            minimum_anchor_fraction=float(config.anchors.m1_minimum_anchor_fraction),
        )
        final_m1_payload = {
            "update": int(np.asarray(state.update_count)),
            "final_checkpoint_condition": True,
            "model_fingerprint": final_model_fingerprint,
            **m1_gate_report(final_gate),
        }
        final_history_index = int(np.asarray(state.m1_summary_state["evaluations"]))
        final_history = _append_m1_history(
            state.m1_history,
            index=final_history_index,
            update=int(np.asarray(state.update_count)),
            result=final_gate,
            bootstrap_training_metrics={
                # This is an evaluation-only row; no bootstrap optimization is
                # performed after the final policy update.  Keep the fixed
                # checkpoint state finite and record that fact explicitly in
                # the enclosing final payload instead of using NaN sentinels.
                "bootstrap_training_loss": jnp.zeros((3,), dtype=jnp.float32)
            },
            supervision_readings=state.supervision_readings,
        )
        state = state._replace(
            m1_summary_state={
                "evaluations": state.m1_summary_state["evaluations"] + 1,
                "latest_passed": final_gate.passed,
            },
            m1_history=final_history,
        )
        m1_history = _m1_history_records(
            final_history,
            count=int(np.asarray(state.m1_summary_state["evaluations"])),
        )
        write_json(
            m1_history_path,
            {
                "version": 1,
                "artifact_type": "depi_final_m1_evaluation",
                "method": METHOD_VERSION,
                "method_variant": config.method_variant,
                "layout": config.environment.layout,
                "seed_index": int(args.seed_index),
                "specification": "METHOD_SPEC §7 final-checkpoint M1 gate",
                "history": m1_history,
                "latest": final_m1_payload,
                "final": final_m1_payload,
            },
        )

    final_step = int(np.asarray(state.effective_environment_steps))
    deployable_count = parameter_count(deployable_parameters(state.params))
    training_only_count = parameter_count(state.target_params) + parameter_count(
        state.bootstrap_encoder_params
    )
    now = time.perf_counter()
    state = _account_runtime_resources(
        state, elapsed_wall_seconds=now - wall_accounted_at
    )
    ledger = ResourceLedger.from_mapping(state.resource_ledger)
    ledger = ResourceLedger(
        **{
            **{
                name: getattr(ledger, name)
                for name in ResourceLedger.__dataclass_fields__
            },
            "deployable_parameters": deployable_count,
            "training_only_parameters": training_only_count,
        }
    )
    state = state._replace(resource_ledger=ledger.to_mapping())
    if manager.latest_step() != final_step:
        _save_depi_checkpoint(manager, output, state)
    write_json(output / "resource_ledger.json", ledger.to_mapping())
    _write_training_support_latents(
        output,
        model=model,
        params=state.params,
        anchors=(
            state.supervision_anchor_batch
            if decision_supervision_enabled
            and int(np.asarray(state.anchor_sampling_counter)) > 0
            else None
        ),
        config=config,
    )
    metadata = {
        "method": METHOD_VERSION,
        "config_version": CONFIG_VERSION,
        "artifact_name": "DEPI",
        "method_variant": config.method_variant,
        "seed_index": int(args.seed_index),
        "jax_prng_key": list(outer_key),
        "effective_environment_steps": final_step,
        "update_count": int(np.asarray(state.update_count)),
        "context_dropout_at_deployment": 0.0,
        "anchor_microbatch": anchor_microbatch,
        "compiled_kernels": kernels.metadata(),
        "anchor_kernel": (
            list(anchor_kernel.metadata()) if anchor_kernel is not None else []
        ),
        "owner_initialization": _host(owner_metrics),
        "partner_training_source": "registered_static_pool_only",
        "gpu": _gpu_snapshot(),
        "scientific_readout": False,
        "resume_allowed": not preflight,
        "note": "Mechanical execution and signal metrics are not ZSC performance evidence.",
    }
    write_json(output / "run_metadata.json", metadata)
    deployment_directory = output / "final_deployment"
    if not deployment_directory.exists() or not any(deployment_directory.iterdir()):
        export_deployment_bundle(
            deployment_directory,
            source_training_run=output,
            deployment=Deployment(
                ego_run_id=str(args.ego_run_id),
                config=config,
                model=model,
                params=state.params,
            ),
        )
    if preflight:
        if peak_device_memory_bytes() >= FORMAL_PEAK_MEMORY_LIMIT_BYTES:
            raise RuntimeError("Formal DEPI CUDA preflight exceeded 40,000 MiB.")
        roundtrip_report = _validate_preflight_roundtrips(
            manager=manager,
            output=output,
            state=state,
            deployment_directory=deployment_directory,
            config=config,
        )
        write_json(
            output / "cuda_preflight_report.json",
            {
                **metadata,
                **roundtrip_report,
                "compiled_rollout_update_count": 1,
                "peak_memory_gate_passed": True,
                "peak_memory_limit_bytes": FORMAL_PEAK_MEMORY_LIMIT_BYTES,
                "observed_peak_memory_bytes": peak_device_memory_bytes(),
            },
        )
    write_json_atomic(
        output / "phase_status.json",
        {
            "status": "complete",
            "phase": "finished",
            "update": int(np.asarray(state.update_count)),
            "completed_at": _utc_now(),
            "gpu": _gpu_snapshot(),
        },
    )
    print(f"Complete DEPI training run: {output}")


def run_cuda_preflight(args: argparse.Namespace) -> None:
    args._execution_scope = CUDA_PREFLIGHT_SCOPE
    args.run_kind = "formal"
    args.resume = False
    run_training(args)


__all__ = ["run_cuda_preflight", "run_training"]
