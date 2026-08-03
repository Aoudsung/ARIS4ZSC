"""DELTA-ZSC V6 end-to-end Bayes Coordination training lifecycle.

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
    deployable_parameters,
    export_deployment_bundle,
)
from experiments.overcooked_v2.official_adapter import (
    FrozenPartnerPool,
    VectorEnvironment,
    validate_official_runtime,
)
from src.path_c.anchor_replay import (
    AnchorReplayState,
    append_replay_batch,
    policy_drift_age_weights,
    replay_fingerprint,
    sample_replay_batch,
)
from src.path_c.anchor_sampling import (
    collect_anchor_batch,
    make_anchor_functions,
    preflight_anchor_microbatch_from_records,
)
from src.path_c.base_distillation import collect_owner_behavior, distill_owner_actor
from src.path_c.compiled_kernels import (
    CompiledCallable,
    attach_chunked_regret_with_kernels,
    build_anchor_chunk_kernel,
    build_training_kernels,
    configure_persistent_compilation_cache,
)
from src.path_c.experiment import (
    CONFIG_VERSION,
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
from src.path_c.generator_training import (
    distill_generator_sources,
    source_code_anchors,
    update_generator,
)
from src.path_c.gradient_routing import (
    BELIEF_OBJECTIVE_ORDER,
    combine_belief_gradients,
)
from src.path_c.model import (
    build_model,
    initial_policy_state,
    initialize_model_parameters,
)
from src.path_c.partner_episode import (
    collect_complete_external_episodes,
    collect_complete_generator_episodes,
    complete_episode_raw_returns,
    lower_tail_cvar,
    update_competence_dual,
    validate_complete_episode_batch,
)
from src.path_c.partner_generator import (
    build_partner_generator,
    initialize_generator_parameters,
    sample_partner_codes,
)
from src.path_c.partner_sources import (
    MixedPartnerParameters,
    make_mixed_partner_functions,
    soft_generator_mixture,
)
from src.path_c.raw_q import centered
from src.path_c.resources import (
    ResourceLedger,
    configure_bundled_cuda_toolchain,
    gpu_hours_for_wall_seconds,
    parameter_count,
    peak_device_memory_bytes,
    require_single_cuda_worker,
)
from src.path_c.runner import (
    DECISION_REGRET_STATE_CHUNK_SIZE,
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
    environment_minibatch_schedule,
    make_optimizer,
    official_reward_shaping_factor,
    polyak_update,
)
from src.path_c.types import CounterfactualAnchorBatch, TrainState


FORMAL_PEAK_MEMORY_LIMIT_BYTES = 40_000 * 2**20
CUDA_PREFLIGHT_SCOPE = "v6_formal_cuda_single_update_preflight"


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
        raise ValueError("A required V6 frozen-partner pool is empty.")
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


def _empty_replay(
    *,
    capacity: int,
    observation_shape: tuple[int, ...],
    code_dim: int,
    model_config: Any,
) -> AnchorReplayState:
    import jax
    import jax.numpy as jnp

    count = int(capacity)
    policy = initial_policy_state(
        batch_size=count,
        observation_shape=observation_shape,
        action_count=6,
        task_hidden_dim=model_config.task_hidden_dim,
        belief_hidden_dim=model_config.belief_hidden_dim,
        latent_dim=model_config.latent_dim,
    )
    zeros_action = jnp.zeros((count, 6), dtype=jnp.float32)
    batch = CounterfactualAnchorBatch(
        anchor_ids=jnp.zeros((count,), dtype=jnp.int64),
        rollout_flat_indexes=jnp.zeros((count,), dtype=jnp.int32),
        policy_states=policy,
        observations=jnp.zeros((count,) + observation_shape, dtype=jnp.float32),
        partner_codes=jnp.zeros((count, int(code_dim)), dtype=jnp.float32),
        partner_sources=jnp.zeros((count,), dtype=jnp.int32),
        partner_run_ids=jnp.zeros((count,), dtype=jnp.int32),
        fit_returns_by_action=zeros_action,
        return_sum_by_action=zeros_action,
        return_squared_sum_by_action=zeros_action,
        replica_count=jnp.zeros((count, 6), dtype=jnp.int32),
        collection_policy_logits=zeros_action,
        collection_update=jnp.zeros((count,), dtype=jnp.int32),
        collection_target_fingerprint=jnp.zeros((count, 2), dtype=jnp.uint32),
        matched_pair_ids=jnp.full((count,), -1, dtype=jnp.int32),
        action_mask=jnp.zeros((count, 6), dtype=jnp.bool_),
    )
    return AnchorReplayState(
        batch=batch,
        item_count=jnp.asarray(0, dtype=jnp.int32),
        capacity=count,
        use_counts=jnp.zeros((count,), dtype=jnp.int32),
    )


def _identity_fingerprint(identity: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(identity), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _source_batch_identity_path(path: Path) -> Path:
    return path.parent / f"{path.name}.identity.json"


def _save_source_batch(
    path: Path, batch: Any, *, run_identity: Mapping[str, Any]
) -> None:
    import orbax.checkpoint as ocp

    path.parent.mkdir(parents=True, exist_ok=True)
    ocp.PyTreeCheckpointer().save(str(path), batch, force=True)
    write_json_atomic(
        _source_batch_identity_path(path),
        {
            "schema": "delta_zsc_v6_generator_source_batch_v1",
            "method": METHOD_VERSION,
            "config_version": CONFIG_VERSION,
            "run_identity_sha256": _identity_fingerprint(run_identity),
            "batch_fingerprint": pytree_fingerprint(batch),
        },
    )


def _load_source_batch(path: Path, *, run_identity: Mapping[str, Any]) -> Any:
    import orbax.checkpoint as ocp

    if not path.is_dir():
        raise FileNotFoundError(f"Generator initialization replay is missing: {path}")
    identity_path = _source_batch_identity_path(path)
    if not identity_path.is_file():
        raise RuntimeError("Generator initialization replay has no V6 identity sidecar.")
    expected = json.loads(identity_path.read_text(encoding="utf-8"))
    if expected.get("schema") != "delta_zsc_v6_generator_source_batch_v1":
        raise RuntimeError("Generator initialization replay uses an incompatible schema.")
    if expected.get("method") != METHOD_VERSION or int(
        expected.get("config_version", -1)
    ) != CONFIG_VERSION:
        raise RuntimeError("Generator initialization replay is not a V6 artifact.")
    if expected.get("run_identity_sha256") != _identity_fingerprint(run_identity):
        raise RuntimeError("Generator initialization replay belongs to another run identity.")
    restored = ocp.PyTreeCheckpointer().restore(str(path))
    if isinstance(restored, Mapping):
        from src.path_c.base_distillation import OwnerBehaviorBatch

        restored = OwnerBehaviorBatch(**restored)
    if expected.get("batch_fingerprint") != pytree_fingerprint(restored):
        raise RuntimeError("Generator initialization replay fingerprint differs from its sidecar.")
    return restored


def _checkpoint_identity(state: TrainState) -> Mapping[str, Any]:
    """Fingerprint every value whose restoration can change the next update.

    The sidecar is intentionally stricter than Orbax's structural restore.  In
    particular it binds recurrent/random state, all moving statistics and the
    continuously trained generator.  A V5 checkpoint therefore cannot be
    adopted merely because a subset of parameter leaves happens to match.
    """

    return {
        "method": METHOD_VERSION,
        "config_version": CONFIG_VERSION,
        "update_count": int(np.asarray(state.update_count)),
        "effective_environment_steps": int(
            np.asarray(state.effective_environment_steps)
        ),
        "params": pytree_fingerprint(state.params),
        "target_params": pytree_fingerprint(state.target_params),
        "generator_params": pytree_fingerprint(state.generator_params),
        "generator_target_params": pytree_fingerprint(state.generator_target_params),
        "ppo_optimizer_state": pytree_fingerprint(state.ppo_optimizer_state),
        "raw_q_optimizer_state": pytree_fingerprint(state.raw_q_optimizer_state),
        "response_optimizer_state": pytree_fingerprint(state.response_optimizer_state),
        "belief_optimizer_state": pytree_fingerprint(state.belief_optimizer_state),
        "generator_optimizer_state": pytree_fingerprint(state.generator_optimizer_state),
        "anchor_replay": replay_fingerprint(state.anchor_replay),
        "generator_competence_multiplier": pytree_fingerprint(
            state.generator_competence_multiplier
        ),
        "generator_cvar_ema": pytree_fingerprint(state.generator_cvar_ema),
        "external_reference_cvar_ema": pytree_fingerprint(
            state.external_reference_cvar_ema
        ),
        "anchor_sampling_counter": int(np.asarray(state.anchor_sampling_counter)),
        "belief_gradient_norm_ema": pytree_fingerprint(
            state.belief_gradient_norm_ema
        ),
        "action_range_ema": pytree_fingerprint(state.action_range_ema),
        "anchor_advantage_scale_ema": pytree_fingerprint(
            state.anchor_advantage_scale_ema
        ),
        "runner_state": pytree_fingerprint(state.runner_state),
        "optimizer_steps": {
            "ppo": int(np.asarray(state.ppo_optimizer_step)),
            "raw_q": int(np.asarray(state.raw_q_optimizer_step)),
            "response": int(np.asarray(state.response_optimizer_step)),
            "belief": int(np.asarray(state.belief_optimizer_step)),
            "generator": int(np.asarray(state.generator_optimizer_step)),
        },
        "random_domains": _host(state.random_domains),
        "resource_ledger": dict(state.resource_ledger),
    }


def _save_v6_checkpoint(manager: Any, output: Path, state: TrainState) -> None:
    step = int(np.asarray(state.effective_environment_steps))
    save_checkpoint(manager, step=step, state=state)
    write_json(
        output / "checkpoint_identities" / f"{step:012d}.json",
        _checkpoint_identity(state),
    )


def _restore_v6_checkpoint(
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
        raise RuntimeError("V6 checkpoint has no atomic resume identity.")
    expected = json.loads(path.read_text(encoding="utf-8"))
    observed = json.loads(json.dumps(_checkpoint_identity(state), sort_keys=True))
    if expected != observed:
        raise RuntimeError("Checkpoint parameters, replay, RNG, or optimizer identity differs.")
    if int(np.asarray(state.effective_environment_steps)) != int(step):
        raise RuntimeError("Checkpoint step disagrees with V6 TrainState.")
    return state


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
            "Formal V6 resource accounting requires explicit upstream partner ledgers."
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


def _tree_average(trees: list[Any]) -> Any:
    import jax

    if not trees:
        raise ValueError("Cannot average an empty gradient sequence.")
    return jax.tree_util.tree_map(
        lambda *values: sum(values) / float(len(values)), *trees
    )


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


def _external_cvar_from_rollout(records: Mapping[str, Any], level: float) -> float | None:
    completed = np.asarray(records["completed_returns"])
    mask = np.asarray(records["completed_mask"], dtype=bool)
    source = np.asarray(records["partner_source"])
    values = completed[mask & (source == 2)]
    if values.size == 0:
        return None
    return lower_tail_cvar(values.tolist(), level=float(level))


def _write_training_support_latents(
    output: Path, *, model: Any, params: Any, replay: AnchorReplayState, config: Any
) -> None:
    """Persist final-model support without adding simulator interaction.

    The optional safety wrapper must be calibrated against the primary model's
    actual training support.  Re-encoding the fixed replay under final
    parameters avoids mixing latent coordinates from earlier checkpoints.
    """

    import jax
    import jax.numpy as jnp

    count = int(np.asarray(replay.item_count))
    if count < 2:
        raise RuntimeError("V6 final support requires at least two replay states.")
    batch = jax.tree_util.tree_map(lambda value: value[:count], replay.batch)
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
            "posterior_mean": _host(output_values.belief_mean),
        },
    )


def run_training(args: argparse.Namespace) -> None:
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
        cache_root=os.environ.get("DELTA_JAX_COMPILATION_CACHE"),
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
    init_runs = _owned_runs(manifest, "generator_init_source", int(args.seed_index))
    support_runs = _owned_runs(manifest, "development_support", int(args.seed_index))
    owner_pool, init_pool, external_pool = (
        _pool(owner_runs),
        _pool(init_runs),
        _pool(support_runs),
    )
    upstream_steps, upstream_records = _upstream_partner_cost(
        tuple({run.run_id: run for run in (*owner_runs, *init_runs, *support_runs)}.values()),
        formal=(config.run_kind == "formal" and not preflight),
    )
    write_json(output / "upstream_cost_manifest.json", {"runs": upstream_records})

    model = build_model(
        observation_shape=observation_shape,
        action_count=6,
        task_hidden_dim=config.model.task_hidden_dim,
        belief_hidden_dim=config.model.belief_hidden_dim,
        latent_dim=config.model.latent_dim,
        actor_hidden_dim=config.model.actor_hidden_dim,
        critic_hidden_dim=config.model.critic_hidden_dim,
        response_hidden_dim=config.model.response_hidden_dim,
        modulation_rank=config.model.modulation_rank,
        action_embedding_dim=config.model.action_embedding_dim,
        log_standard_deviation_minimum=config.model.log_standard_deviation_minimum,
        log_standard_deviation_maximum=config.model.log_standard_deviation_maximum,
    )
    generator = build_partner_generator(
        observation_shape=observation_shape,
        action_count=6,
        code_dim=config.partner_generator.code_dim,
        hidden_dim=config.partner_generator.hidden_dim,
        modulation_rank=config.partner_generator.modulation_rank,
    )
    generator_environment = VectorEnvironment(
        environment.environment,
        int(config.partner_generator.episodes_per_update),
        int(config.partner_generator.episode_steps),
    )

    ego_root = jnp.asarray(domains["ego"], dtype=jnp.uint32)
    generator_root = jnp.asarray(domains["generator"], dtype=jnp.uint32)
    reset_key, model_key, runner_key = jax.random.split(ego_root, 3)
    unused_state, initial_observations = environment.reset(reset_key)
    del unused_state
    example_policy = initial_policy_state(
        batch_size=environment.num_envs,
        observation_shape=observation_shape,
        action_count=6,
        task_hidden_dim=config.model.task_hidden_dim,
        belief_hidden_dim=config.model.belief_hidden_dim,
        latent_dim=config.model.latent_dim,
    )
    initial_params = initialize_model_parameters(
        model,
        key=model_key,
        example_state=example_policy,
        example_observation=initial_observations[:, 0],
        partner_code_dim=config.partner_generator.code_dim,
    )
    initial_generator_params = initialize_generator_parameters(
        generator,
        key=jax.random.fold_in(generator_root, 0),
        observation_shape=observation_shape,
        code_dim=config.partner_generator.code_dim,
        batch_size=config.partner_generator.episodes_per_update,
        hidden_dim=config.partner_generator.hidden_dim,
    )
    total_updates = config.training.environment_steps // (
        config.environment.num_envs * config.training.rollout_length
    )

    source_cache = output / "training_cache" / "generator_source_batch"
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
                partner_pool=external_pool,
                length=config.training.rollout_length,
                key=jax.random.fold_in(ego_root, 1),
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
                    belief_hidden_dim=config.model.belief_hidden_dim,
                    latent_dim=config.model.latent_dim,
                ),
                optimizer=owner_optimizer,
                optimizer_state=owner_optimizer_state,
                schedule=owner_schedule,
            ),
        )
        del unused_owner_state, unused

        episode_count = config.partner_generator.episodes_per_update
        if episode_count % 4:
            raise ValueError("Generator source initialization needs balanced SP/OP/SA/FCP lanes.")
        members = jnp.tile(jnp.arange(4, dtype=jnp.int32), episode_count // 4)
        source_batch, unused = _phase(
            output,
            name="generator_initialization_collection",
            update=0,
            function=lambda: collect_owner_behavior(
                environment=generator_environment,
                owner_pool=init_pool,
                partner_pool=owner_pool,
                length=config.partner_generator.episode_steps,
                key=jax.random.fold_in(generator_root, 1),
                owner_members=members,
            ),
        )
        del unused
        _save_source_batch(source_cache, source_batch, run_identity=identity)
        init_generator_optimizer, init_generator_state = make_optimizer(
            initial_generator_params,
            learning_rate=config.ppo.learning_rate,
            gradient_clip_norm=config.ppo.gradient_clip_norm,
            adam_epsilon=config.ppo.adam_epsilon,
        )
        generator_init_schedule = environment_minibatch_schedule(
            jax.random.fold_in(generator_root, 2),
            environment_count=episode_count,
            minibatches_per_epoch=config.partner_generator.environment_minibatches,
            update_epochs=config.partner_generator.update_epochs,
        )
        code_anchors = source_code_anchors(4, config.partner_generator.code_dim)
        (
            generator_params,
            unused_generator_init_state,
            generator_initialization_metrics,
        ), unused = _phase(
            output,
            name="generator_source_distillation",
            update=0,
            function=lambda: distill_generator_sources(
                generator=generator,
                params=initial_generator_params,
                batch=source_batch,
                code_anchors=code_anchors,
                optimizer=init_generator_optimizer,
                optimizer_state=init_generator_state,
                schedule=generator_init_schedule,
                hidden_dim=config.partner_generator.hidden_dim,
            ),
        )
        del unused_generator_init_state, unused
        external_returns, unused = _phase(
            output,
            name="external_reference_initialization",
            update=0,
            function=lambda: collect_complete_external_episodes(
                environment=generator_environment,
                model=model,
                ego_params=params,
                model_config=config.model,
                external_pool=external_pool,
                key=jax.random.fold_in(generator_root, 3),
            ),
        )
        del unused
        reference_cvar = lower_tail_cvar(
            np.asarray(external_returns.raw_returns).tolist(),
            level=config.partner_generator.cvar_level,
        )
    else:
        params = initial_params
        generator_params = initial_generator_params
        source_batch = _load_source_batch(source_cache, run_identity=identity)
        owner_metrics = {"owner_distillation_kl": jnp.asarray(np.nan)}
        generator_initialization_metrics = {
            "generator_source_distillation_kl": jnp.asarray(np.nan)
        }
        reference_cvar = 0.0
        code_anchors = source_code_anchors(4, config.partner_generator.code_dim)

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
    raw_q_optimizer, raw_q_state = make_optimizer(
        params,
        learning_rate=config.ppo.learning_rate,
        gradient_clip_norm=config.ppo.gradient_clip_norm,
        adam_epsilon=config.ppo.adam_epsilon,
    )
    response_optimizer, response_state = make_optimizer(
        params,
        learning_rate=config.ppo.learning_rate,
        gradient_clip_norm=config.ppo.gradient_clip_norm,
        adam_epsilon=config.ppo.adam_epsilon,
    )
    belief_optimizer, belief_state = make_optimizer(
        params,
        learning_rate=config.ppo.learning_rate,
        gradient_clip_norm=config.ppo.gradient_clip_norm,
        adam_epsilon=config.ppo.adam_epsilon,
    )
    generator_optimizer, generator_state = make_optimizer(
        generator_params,
        learning_rate=config.ppo.learning_rate,
        gradient_clip_norm=config.ppo.gradient_clip_norm,
        adam_epsilon=config.ppo.adam_epsilon,
        anneal_learning_rate=config.ppo.anneal_learning_rate,
        warmup_fraction=config.ppo.lr_warmup_fraction,
        update_count=total_updates,
        minibatches_per_epoch=(
            config.partner_generator.environment_minibatches
        ),
        update_epochs=config.partner_generator.update_epochs,
    )
    partner_functions = make_mixed_partner_functions(
        generator=generator,
        generator_hidden_dim=config.partner_generator.hidden_dim,
        generator_code_dim=config.partner_generator.code_dim,
        external_pool=external_pool,
    )
    runner = initialize_runner(
        environment=environment,
        model_config=config.model,
        partner_functions=partner_functions,
        random_key=runner_key,
    )
    replay = _empty_replay(
        capacity=config.anchors.replay_capacity,
        observation_shape=observation_shape,
        code_dim=config.partner_generator.code_dim,
        model_config=config.model,
    )
    initial_ledger = ResourceLedger(
        ego_initialization_steps=(
            0 if not new_run else environment.num_envs * config.training.rollout_length
        ),
        generator_initialization_steps=(
            0
            if not new_run
            else 2
            * config.partner_generator.episodes_per_update
            * config.partner_generator.episode_steps
        ),
        upstream_partner_steps=upstream_steps,
    )
    norm_ema = {
        name: jnp.asarray(1.0, dtype=jnp.float32)
        for name in BELIEF_OBJECTIVE_ORDER
    }
    template = TrainState(
        params=params,
        target_params=jax.tree_util.tree_map(jnp.copy, params),
        ppo_optimizer_state=ppo_state,
        raw_q_optimizer_state=raw_q_state,
        response_optimizer_state=response_state,
        belief_optimizer_state=belief_state,
        generator_optimizer_state=generator_state,
        generator_params=generator_params,
        generator_target_params=jax.tree_util.tree_map(jnp.copy, generator_params),
        generator_competence_multiplier=jnp.asarray(0.0, dtype=jnp.float32),
        generator_cvar_ema=jnp.asarray(reference_cvar, dtype=jnp.float32),
        external_reference_cvar_ema=jnp.asarray(reference_cvar, dtype=jnp.float32),
        anchor_replay=replay,
        anchor_sampling_counter=jnp.asarray(0, dtype=jnp.int32),
        belief_gradient_norm_ema=norm_ema,
        action_range_ema=jnp.asarray(1.0, dtype=jnp.float32),
        anchor_advantage_scale_ema=jnp.asarray(1.0, dtype=jnp.float32),
        ppo_optimizer_step=jnp.asarray(0, dtype=jnp.int64),
        raw_q_optimizer_step=jnp.asarray(0, dtype=jnp.int64),
        response_optimizer_step=jnp.asarray(0, dtype=jnp.int64),
        belief_optimizer_step=jnp.asarray(0, dtype=jnp.int64),
        generator_optimizer_step=jnp.asarray(0, dtype=jnp.int64),
        runner_state=runner,
        random_domains={name: jnp.asarray(value, dtype=jnp.uint32) for name, value in domains.items()},
        update_count=jnp.asarray(0, dtype=jnp.int32),
        effective_environment_steps=jnp.asarray(0, dtype=jnp.int64),
        resource_ledger=initial_ledger.to_mapping(),
    )
    manager = orbax_manager(output / "checkpoints")
    if args.resume:
        restored = _restore_v6_checkpoint(manager, output, template)
        if restored is None:
            raise FileNotFoundError("--resume requested but no V6 checkpoint exists.")
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
        raw_q_optimizer=raw_q_optimizer,
        response_optimizer=response_optimizer,
        belief_optimizer=belief_optimizer,
    )
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
    anchor_logits_kernel = CompiledCallable(
        "anchor_policy_logits",
        lambda candidate, batch: _anchor_policy_logits(model, candidate, batch),
    )
    generator_collect_kernel = CompiledCallable(
        "generator_episode_collection",
        lambda ego, gen, codes, key, version, factor: collect_complete_generator_episodes(
            environment=generator_environment,
            model=model,
            ego_params=ego,
            model_config=config.model,
            generator=generator,
            generator_params=gen,
            codes=codes,
            key=key,
            parameter_version_id=version,
            official_shaping_factor=factor,
        ),
    )
    generator_update_kernel = CompiledCallable(
        "generator_ppo_scan",
        lambda gen, opt_state, episodes, source, schedule, progress, multiplier, reference: update_generator(
            generator=generator,
            params=gen,
            optimizer_state=opt_state,
            optimizer=generator_optimizer,
            batch=episodes,
            source_batch=source,
            code_anchors=code_anchors,
            schedule=schedule,
            config=config,
            progress=progress,
            competence_multiplier=multiplier,
            reference_cvar=reference,
        ),
    )

    anchor_microbatch = None
    start_update = int(np.asarray(state.update_count))
    last_update = min(total_updates, start_update + 1) if preflight else total_updates
    wall_started = time.perf_counter()
    wall_accounted_at = wall_started
    belief_weights = {
        "ppo": config.loss.policy_belief_gradient_scale,
        "raw_q": config.loss.raw_q_weight,
        "counterfactual": config.loss.counterfactual_weight,
        "response": config.loss.response_weight,
        "decision_equivalence": config.loss.decision_equivalence_weight,
        "information_bottleneck": config.loss.information_bottleneck_weight,
        "q_policy": config.loss.q_policy_weight,
        "robust": config.loss.robust_generalist_weight,
    }

    for update in range(start_update, last_update):
        phase_times: dict[str, float] = {}
        snapshot_params = state.params
        snapshot_target = state.target_params
        progress = jnp.asarray(update / float(max(total_updates, 1)), dtype=jnp.float32)
        probabilities = soft_generator_mixture(
            progress=progress,
            generator_cvar_ema=state.generator_cvar_ema,
            reference_cvar_ema=state.external_reference_cvar_ema,
            maximum_probability=config.partner_generator.maximum_generator_probability,
            ramp_fraction=config.partner_generator.mixture_ramp_fraction,
            competence_temperature=config.partner_generator.competence_temperature,
        )
        partner_parameters = MixedPartnerParameters(
            state.generator_params, state.generator_target_params, probabilities
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
        anchor_trigger = update % config.anchors.interval_updates == 0
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
            raise RuntimeError("Training rollout did not return the V6 PPO batch.")

        (regret_result, elapsed) = _phase(
            output,
            name="decision_regret",
            update=update,
            function=lambda: attach_chunked_regret_with_kernels(
                kernels=kernels,
                batch=batch,
                target_params=snapshot_target,
                key=jax.random.fold_in(
                    jnp.asarray(domains["ego"], dtype=jnp.uint32), 200_000 + update
                ),
                action_range_ema=state.action_range_ema,
                effective_steps=state.effective_environment_steps,
            ),
        )
        phase_times["decision_regret"] = elapsed
        batch, regret_metrics = regret_result
        next_replay = state.anchor_replay
        anchor_budget = {
            "counterfactual_continuation_steps": 0,
            "matched_code_probe_steps": 0,
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
                ),
            )
            phase_times["counterfactual_anchor"] = elapsed
            anchors, unused_pairs, anchor_budget = anchor_result
            del unused_pairs
            next_replay = append_replay_batch(
                next_replay,
                batch=anchors,
                capacity=config.anchors.replay_capacity,
            )

        replay_samples: list[Any] = []
        replay_weights: list[Any] = []
        sampled_replay = next_replay
        for replay_index in range(config.loss.raw_q_updates_per_outer_update):
            sampled_replay, sample = sample_replay_batch(
                sampled_replay,
                key=jax.random.fold_in(
                    jnp.asarray(domains["anchor_replay"], dtype=jnp.uint32),
                    update * config.loss.raw_q_updates_per_outer_update + replay_index,
                ),
                batch_size=config.anchors.replay_minibatch_size,
            )
            current_logits = anchor_logits_kernel(snapshot_params, sample)
            weights = policy_drift_age_weights(
                current_policy_logits=current_logits,
                collection_policy_logits=sample.collection_policy_logits,
                current_update=update,
                collection_update=sample.collection_update,
                kl_decay=config.anchors.policy_kl_decay,
                age_decay_updates=config.anchors.age_decay_updates,
                minimum_weight=config.anchors.minimum_replay_weight,
            )
            replay_samples.append(sample)
            replay_weights.append(weights)

        (rollout_belief, elapsed) = _phase(
            output,
            name="belief_gradient_rollout",
            update=update,
            function=lambda: kernels.belief_rollout_objectives(
                snapshot_params, snapshot_target, batch
            ),
        )
        phase_times["belief_gradient_rollout"] = elapsed
        belief_values, belief_gradients = rollout_belief
        anchor_values: list[Mapping[str, Any]] = []
        anchor_gradients: list[Mapping[str, Any]] = []
        for sample, weights in zip(replay_samples, replay_weights):
            values, gradients = kernels.belief_anchor_objectives(
                snapshot_params,
                sample,
                weights,
                state.anchor_advantage_scale_ema,
            )
            anchor_values.append(values)
            anchor_gradients.append(gradients)
        belief_values["counterfactual"] = sum(
            value["counterfactual"] for value in anchor_values
        ) / float(len(anchor_values))
        belief_values["decision_equivalence"] = sum(
            value["decision_equivalence"] for value in anchor_values
        ) / float(len(anchor_values))
        belief_gradients["counterfactual"] = _tree_average(
            [value["counterfactual"] for value in anchor_gradients]
        )
        belief_gradients["decision_equivalence"] = _tree_average(
            [value["decision_equivalence"] for value in anchor_gradients]
        )
        combined_belief, next_norm_ema, belief_gradient_metrics = (
            combine_belief_gradients(
                belief_gradients,
                state.belief_gradient_norm_ema,
                weights=belief_weights,
                decay=config.loss.belief_gradient_ema_decay,
            )
        )

        schedule = environment_minibatch_schedule(
            jax.random.fold_in(
                jnp.asarray(domains["ego"], dtype=jnp.uint32), 400_000 + update
            ),
            environment_count=config.environment.num_envs,
            minibatches_per_epoch=config.training.minibatches_per_epoch,
            update_epochs=config.ppo.update_epochs,
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
            ),
        )
        phase_times["ppo"] = elapsed
        ppo_core, ppo_metrics = ppo_result
        if bool(
            np.any(np.asarray(ppo_metrics["training_aborted_nonfinite"]) > 0.5)
        ):
            raise FloatingPointError(
                "NaN or Inf occurred during the V6 PPO minibatch scan."
            )
        params_after_heads = ppo_core.params
        (
            params_after_heads,
            next_raw_q_state,
            raw_q_metrics,
        ) = kernels.raw_q_retrace_update(
            params_after_heads, snapshot_target, state.raw_q_optimizer_state, batch
        )
        anchor_head_metrics = []
        for sample, weights in zip(replay_samples, replay_weights):
            params_after_heads, next_raw_q_state, metrics = kernels.anchor_head_update(
                params_after_heads, next_raw_q_state, sample, weights
            )
            anchor_head_metrics.append(metrics)
        next_response_state = state.response_optimizer_state
        response_metrics = []
        for unused_index in range(config.loss.response_updates_per_outer_update):
            del unused_index
            params_after_heads, next_response_state, metrics = kernels.response_head_update(
                params_after_heads, next_response_state, batch
            )
            response_metrics.append(metrics)
        (belief_result, elapsed) = _phase(
            output,
            name="belief_apply",
            update=update,
            function=lambda: kernels.belief_apply(
                params_after_heads, state.belief_optimizer_state, combined_belief
            ),
        )
        phase_times["belief_apply"] = elapsed
        next_params, next_belief_state = belief_result

        generator_codes = sample_partner_codes(
            jax.random.fold_in(generator_root, 500_000 + update),
            batch_size=config.partner_generator.episodes_per_update,
            code_dim=config.partner_generator.code_dim,
        )
        (generator_episodes, elapsed) = _phase(
            output,
            name="generator_collection",
            update=update,
            function=lambda: generator_collect_kernel(
                next_params,
                state.generator_params,
                generator_codes,
                jax.random.fold_in(generator_root, 600_000 + update),
                jnp.asarray(update, dtype=jnp.int32),
                shaping_factor,
            ),
        )
        phase_times["generator_collection"] = elapsed
        validate_complete_episode_batch(generator_episodes)
        generator_schedule = environment_minibatch_schedule(
            jax.random.fold_in(generator_root, 700_000 + update),
            environment_count=config.partner_generator.episodes_per_update,
            minibatches_per_epoch=config.partner_generator.environment_minibatches,
            update_epochs=config.partner_generator.update_epochs,
        )
        (generator_result, elapsed) = _phase(
            output,
            name="generator_ppo",
            update=update,
            function=lambda: generator_update_kernel(
                state.generator_params,
                state.generator_optimizer_state,
                generator_episodes,
                source_batch,
                generator_schedule,
                progress,
                state.generator_competence_multiplier,
                state.external_reference_cvar_ema,
            ),
        )
        phase_times["generator_ppo"] = elapsed
        next_generator_params, next_generator_optimizer_state, generator_metrics = generator_result
        generator_returns = complete_episode_raw_returns(generator_episodes)
        generator_cvar = lower_tail_cvar(
            generator_returns.tolist(), level=config.partner_generator.cvar_level
        )
        generator_cvar_ema = (
            config.partner_generator.cvar_ema_decay * state.generator_cvar_ema
            + (1.0 - config.partner_generator.cvar_ema_decay) * generator_cvar
        )
        external_cvar = _external_cvar_from_rollout(
            records, config.partner_generator.cvar_level
        )
        external_reference_cvar_ema = state.external_reference_cvar_ema
        if external_cvar is not None:
            external_reference_cvar_ema = (
                config.partner_generator.cvar_ema_decay
                * state.external_reference_cvar_ema
                + (1.0 - config.partner_generator.cvar_ema_decay) * external_cvar
            )
        next_dual = update_competence_dual(
            state.generator_competence_multiplier,
            reference_cvar=external_reference_cvar_ema,
            generator_cvar=generator_cvar_ema,
            learning_rate=config.partner_generator.lagrangian_learning_rate,
            maximum=config.partner_generator.competence_multiplier_maximum,
        )
        next_target_params = polyak_update(
            state.target_params, next_params, config.ppo.polyak_coefficient
        )
        next_generator_target = polyak_update(
            state.generator_target_params,
            next_generator_params,
            config.partner_generator.target_polyak_coefficient,
        )

        empirical_scale = np.asarray(
            jnp.mean(jnp.linalg.norm(centered(replay_samples[0].fit_returns_by_action), axis=-1))
        )
        next_advantage_scale = (
            0.99 * state.anchor_advantage_scale_ema + 0.01 * float(empirical_scale)
        )
        completed_steps = int(np.asarray(runner.effective_environment_steps))
        ledger = ResourceLedger.from_mapping(state.resource_ledger).plus(
            ego_policy_steps=config.environment.num_envs * config.training.rollout_length,
            generator_training_steps=(
                config.partner_generator.episodes_per_update
                * config.partner_generator.episode_steps
            ),
            counterfactual_continuation_steps=int(
                anchor_budget["counterfactual_continuation_steps"]
            ),
            matched_code_probe_steps=int(anchor_budget["matched_code_probe_steps"]),
        )
        ppo_applied = int(
            np.asarray(jnp.sum(ppo_metrics["optimizer_applied"], dtype=jnp.int64))
        )
        state = state._replace(
            params=next_params,
            target_params=next_target_params,
            ppo_optimizer_state=ppo_core.ppo_optimizer_state,
            raw_q_optimizer_state=next_raw_q_state,
            response_optimizer_state=next_response_state,
            belief_optimizer_state=next_belief_state,
            generator_optimizer_state=next_generator_optimizer_state,
            generator_params=next_generator_params,
            generator_target_params=next_generator_target,
            generator_competence_multiplier=next_dual,
            generator_cvar_ema=jnp.asarray(generator_cvar_ema),
            external_reference_cvar_ema=jnp.asarray(external_reference_cvar_ema),
            anchor_replay=sampled_replay,
            anchor_sampling_counter=(
                state.anchor_sampling_counter + int(anchor_trigger)
            ),
            belief_gradient_norm_ema=next_norm_ema,
            action_range_ema=regret_metrics["action_range_ema"],
            anchor_advantage_scale_ema=jnp.asarray(next_advantage_scale),
            ppo_optimizer_step=state.ppo_optimizer_step + ppo_applied,
            raw_q_optimizer_step=(
                state.raw_q_optimizer_step
                + 1
                + config.loss.raw_q_updates_per_outer_update
            ),
            response_optimizer_step=(
                state.response_optimizer_step
                + config.loss.response_updates_per_outer_update
            ),
            belief_optimizer_step=state.belief_optimizer_step + 1,
            generator_optimizer_step=(
                state.generator_optimizer_step
                + config.partner_generator.update_epochs
                * config.partner_generator.environment_minibatches
            ),
            runner_state=runner,
            update_count=jnp.asarray(update + 1, dtype=jnp.int32),
            effective_environment_steps=jnp.asarray(completed_steps, dtype=jnp.int64),
            resource_ledger=ledger.to_mapping(),
        )
        if not _all_finite(
            (
                state.params,
                state.target_params,
                state.generator_params,
                state.generator_target_params,
                state.generator_competence_multiplier,
                state.generator_cvar_ema,
            )
        ):
            raise FloatingPointError("NaN or Inf entered the V6 training state.")

        metrics = {
            "method": METHOD_VERSION,
            "update_count": update + 1,
            "effective_environment_steps": completed_steps,
            "partner_source_probabilities": _host(probabilities),
            "context_dropout_probability": float(np.asarray(dropout)),
            "official_shaping_factor": float(np.asarray(shaping_factor)),
            "ppo": _host(_mean_metrics(ppo_metrics)),
            "raw_q": _host(raw_q_metrics),
            "counterfactual": _host(
                _mean_metrics(jax.tree_util.tree_map(lambda *values: jnp.stack(values), *anchor_head_metrics))
            ),
            "response": _host(
                _mean_metrics(jax.tree_util.tree_map(lambda *values: jnp.stack(values), *response_metrics))
            ),
            "belief_objective_losses": _host(belief_values),
            "belief_gradient": _host(belief_gradient_metrics),
            "regret": _host(
                {
                    key: value
                    for key, value in regret_metrics.items()
                    if key != "decision_regret_values"
                }
            ),
            "generator": {
                **_host(generator_metrics),
                "raw_cvar": generator_cvar,
                "raw_cvar_ema": float(np.asarray(generator_cvar_ema)),
                "external_reference_cvar_ema": float(
                    np.asarray(external_reference_cvar_ema)
                ),
                "competence_multiplier": float(np.asarray(next_dual)),
            },
            "replay": {
                "item_count": int(np.asarray(state.anchor_replay.item_count)),
                "fingerprint": replay_fingerprint(state.anchor_replay),
                "anchor_trigger": anchor_trigger,
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
            _save_v6_checkpoint(manager, output, state)

    final_step = int(np.asarray(state.effective_environment_steps))
    elapsed_total = time.perf_counter() - wall_started
    deployable_count = parameter_count(deployable_parameters(state.params))
    training_only_count = (
        parameter_count(state.target_params)
        + parameter_count(state.generator_params)
        + parameter_count(state.generator_target_params)
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
        _save_v6_checkpoint(manager, output, state)
    write_json(output / "resource_ledger.json", ledger.to_mapping())
    _write_training_support_latents(
        output,
        model=model,
        params=state.params,
        replay=state.anchor_replay,
        config=config,
    )
    metadata = {
        "method": METHOD_VERSION,
        "config_version": CONFIG_VERSION,
        "artifact_name": "DELTA-ZSC-E2E",
        "seed_index": int(args.seed_index),
        "jax_prng_key": list(outer_key),
        "effective_environment_steps": final_step,
        "update_count": int(np.asarray(state.update_count)),
        "context_dropout_at_deployment": 0.0,
        "decision_regret_state_chunk_size": DECISION_REGRET_STATE_CHUNK_SIZE,
        "anchor_microbatch": anchor_microbatch,
        "compiled_kernels": kernels.metadata(),
        "anchor_kernel": list(anchor_kernel.metadata()),
        "generator_collection_kernel": list(generator_collect_kernel.metadata()),
        "generator_update_kernel": list(generator_update_kernel.metadata()),
        "owner_initialization": _host(owner_metrics),
        "generator_initialization": _host(generator_initialization_metrics),
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
            raise RuntimeError("Formal V6 CUDA preflight exceeded 40,000 MiB.")
        write_json(output / "cuda_preflight_report.json", metadata)
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
    print(f"Complete DELTA-ZSC V6 E2E run: {output}")


def run_cuda_preflight(args: argparse.Namespace) -> None:
    args._execution_scope = CUDA_PREFLIGHT_SCOPE
    args.run_kind = "formal"
    args.resume = False
    run_training(args)


__all__ = ["run_cuda_preflight", "run_training"]
