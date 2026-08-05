"""End-to-end Official OvercookedV2 training for unified DELTA-ZSC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from src.delta_zsc.anchors import AnchorFunctions, collect_anchor_batch
from src.delta_zsc.config import (
    CHECKPOINT_SCHEMA_VERSION,
    METHOD_VERSION,
    OFFICIAL_ACTION_COUNT,
    load_config,
)
from src.delta_zsc.manifest import load_partner_manifest
from src.delta_zsc.model import DeltaModel, observe_after_transition
from src.delta_zsc.optimizer import init_adam
from src.delta_zsc.partners import build_training_partner_pool, make_static_partner_functions
from src.delta_zsc.resources import ResourceLedger, parameter_count, peak_device_memory_bytes
from src.delta_zsc.runner import collect_rollout, initialize_runner
from src.delta_zsc.storage import (
    ensure_run_identity,
    load_latest_checkpoint,
    pytree_fingerprint,
    save_checkpoint,
    sha256_path,
    write_json,
)
from src.delta_zsc.training import environment_minibatch_schedule, training_update
from src.delta_zsc.types import TrainState

from .deployment import export_deployment_bundle


def _training_key(seed_index: int) -> Any:
    import jax

    if int(seed_index) < 0:
        return jax.random.PRNGKey(0)
    return jax.random.split(jax.random.PRNGKey(42), 10)[int(seed_index)]


def _write_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _host(value: Any) -> Any:
    import jax

    def convert(item: Any) -> Any:
        array = np.asarray(jax.device_get(item))
        if array.ndim == 0:
            return array.item()
        return array.tolist()

    return jax.tree_util.tree_map(convert, value)




def _upstream_partner_cost(
    members: tuple[Any, ...], *, formal: bool
) -> tuple[int, float, float, list[Mapping[str, Any]]]:
    """Resolve each unique parent policy's explicit simulator cost.

    New DELTA upstream runs expose ``ResourceLedger``.  Existing Official
    checkpoints may carry an older ledger/summary, so a positive explicit step
    total is accepted, but formal runs never invent a missing cost.
    """

    total = 0
    gpu_hours = 0.0
    wall_clock_hours = 0.0
    records: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for member in members:
        parent = str(member.parent_training_run_id)
        if parent in seen:
            continue
        seen.add(parent)
        checkpoint = Path(member.checkpoint).resolve()
        candidates: list[Path] = []
        for directory in (checkpoint, *checkpoint.parents[:6]):
            candidates.extend(
                (directory / "resource_ledger.json", directory / "upstream_summary.json")
            )
        source = next((value for value in candidates if value.is_file()), None)
        steps = None
        parent_gpu = 0.0
        parent_wall = 0.0
        if source is not None:
            payload = json.loads(source.read_text(encoding="utf-8"))
            if isinstance(payload, Mapping):
                try:
                    registered = ResourceLedger.from_mapping(payload)
                except (TypeError, ValueError):
                    registered = None
                if registered is not None:
                    steps = int(registered.total_training_simulator_steps)
                    parent_gpu = float(registered.gpu_hours)
                    parent_wall = float(registered.wall_clock_hours)
                else:
                    for name in (
                        "total_training_simulator_steps",
                        "upstream_partner_steps",
                        "effective_environment_steps",
                        "actual_timesteps",
                    ):
                        if payload.get(name) is not None and int(payload[name]) > 0:
                            steps = int(payload[name])
                            break
                    parent_gpu = float(
                        payload.get("gpu_hours", payload.get("training_gpu_hours", 0.0))
                    )
                    parent_wall = float(
                        payload.get(
                            "wall_clock_hours",
                            payload.get("training_wall_clock_hours", 0.0),
                        )
                    )
        if steps is None:
            records.append({"parent_training_run_id": parent, "status": "missing"})
            continue
        total += steps
        gpu_hours += parent_gpu
        wall_clock_hours += parent_wall
        records.append(
            {
                "parent_training_run_id": parent,
                "status": "counted",
                "steps": steps,
                "gpu_hours": parent_gpu,
                "wall_clock_hours": parent_wall,
                "source": str(source),
                "source_sha256": sha256_path(source),
            }
        )
    if formal and any(row["status"] != "counted" for row in records):
        raise RuntimeError(
            "Formal DELTA requires an explicit upstream resource ledger for every "
            "training-support parent."
        )
    return total, gpu_hours, wall_clock_hours, records

def _anchor_functions(
    *, model: DeltaModel, partner_functions: Any, partner_parameters: Any, environment: Any
) -> AnchorFunctions:
    import jax

    def ego_step(base: Any, latent: Any, state: Any, observation: Any, keys: Any):
        next_state, output = model.step(
            base,
            latent,
            state,
            observation,
            compute_latent=True,
            execute_adaptation=False,
        )
        action = jax.vmap(lambda key, logits: jax.random.categorical(key, logits))(
            keys, output.base_policy_logits
        )
        return next_state, action

    def ego_observe(state: Any, action: Any, done: Any):
        return observe_after_transition(state, action=action, done=done)

    def partner_step_fixed(
        state: Any, observation: Any, episode_start: Any, keys: Any
    ):
        # The anchor world stores the actual snapshot episode-start flag.  This
        # is essential for states sampled at an episode boundary: partner
        # recurrent policies must reset exactly as they did in the legal
        # rollout, rather than being silently forced to continue an old carry.
        action, next_state, context, unused_value = partner_functions.step(
            partner_parameters, state, observation, episode_start, keys
        )
        del unused_value
        return action, next_state, context

    def partner_observe(
        state: Any,
        context: Any,
        observation: Any,
        action: Any,
        reward: Any,
        done: Any,
        next_observation: Any,
    ):
        return partner_functions.observe(
            partner_parameters,
            state,
            context,
            observation,
            action,
            reward,
            done,
            next_observation,
        )

    return AnchorFunctions(
        ego_step=ego_step,
        ego_observe=ego_observe,
        partner_step=partner_step_fixed,
        partner_observe=partner_observe,
        environment_step=environment.step_with_keys,
    )


def _measure_inference_latency_ms(
    *, model: DeltaModel, base_params: Any, latent_params: Any
) -> float:
    """Median synchronized batch-one deployment-step latency."""

    import jax
    import jax.numpy as jnp

    state = model.initial_state(1)
    observation = jnp.zeros((1,) + tuple(model.observation_shape), dtype=jnp.float32)
    step = jax.jit(lambda current, frame: model.step(
        base_params, latent_params, current, frame
    ))
    for _ in range(20):
        state, output = step(state, observation)
        jax.block_until_ready(output.policy_logits)
    values = []
    for _ in range(100):
        started = time.perf_counter()
        state, output = step(state, observation)
        jax.block_until_ready(output.policy_logits)
        values.append((time.perf_counter() - started) * 1_000.0)
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _final_decision_audit(
    *,
    model: DeltaModel,
    base_params: Any,
    latent_params: Any,
    batch: Any,
    anchors: Any,
) -> Mapping[str, Any]:
    import jax.numpy as jnp

    _, output = model.sequence(
        base_params,
        latent_params,
        batch.initial_policy_state,
        batch.observations,
        batch.previous_actions,
        batch.episode_starts,
    )
    predicted = output.expected_decision_values[
        anchors.time_indexes, anchors.lane_indexes
    ]
    target = anchors.evaluation_returns_by_action
    predicted_order = jnp.argsort(jnp.argsort(predicted, axis=-1), axis=-1)
    target_order = jnp.argsort(jnp.argsort(target, axis=-1), axis=-1)
    centered_p = predicted_order - jnp.mean(predicted_order, axis=-1, keepdims=True)
    centered_t = target_order - jnp.mean(target_order, axis=-1, keepdims=True)
    spearman = jnp.sum(centered_p * centered_t, axis=-1) / jnp.sqrt(
        jnp.sum(jnp.square(centered_p), axis=-1)
        * jnp.sum(jnp.square(centered_t), axis=-1)
        + 1.0e-8
    )
    chosen = jnp.argmax(predicted, axis=-1)
    oracle = jnp.argmax(target, axis=-1)
    rows = jnp.arange(chosen.shape[0])
    regret = target[rows, oracle] - target[rows, chosen]
    anchor_voi = output.active_voi[anchors.time_indexes, anchors.lane_indexes]
    anchor_voi_raw = output.active_voi_raw[
        anchors.time_indexes, anchors.lane_indexes
    ]
    anchor_information_gain = output.active_information_gain[
        anchors.time_indexes, anchors.lane_indexes
    ]
    anchor_quadrature_error = output.active_voi_quadrature_error[
        anchors.time_indexes, anchors.lane_indexes
    ]
    anchor_adaptation_kl = output.adaptation_kl[
        anchors.time_indexes, anchors.lane_indexes
    ]
    base_greedy = jnp.argmax(
        output.base_policy_logits[anchors.time_indexes, anchors.lane_indexes],
        axis=-1,
    )
    deployment_greedy = jnp.argmax(
        output.policy_logits[anchors.time_indexes, anchors.lane_indexes],
        axis=-1,
    )
    return {
        "mean_spearman": float(jnp.mean(spearman)),
        "top_action_agreement": float(jnp.mean((chosen == oracle).astype(jnp.float32))),
        "mean_empirical_action_regret": float(jnp.mean(regret)),
        "mean_action_voi": float(jnp.mean(anchor_voi)),
        "mean_max_action_voi": float(jnp.mean(jnp.max(anchor_voi, axis=-1))),
        "raw_voi_negative_fraction": float(
            jnp.mean((anchor_voi_raw < 0.0).astype(jnp.float32))
        ),
        "mean_information_gain": float(jnp.mean(anchor_information_gain)),
        "mean_voi_quadrature_error": float(jnp.mean(anchor_quadrature_error)),
        "max_voi_quadrature_error": float(jnp.max(anchor_quadrature_error)),
        "mean_adaptation_kl": float(jnp.mean(anchor_adaptation_kl)),
        "greedy_action_disagreement": float(
            jnp.mean((base_greedy != deployment_greedy).astype(jnp.float32))
        ),
        "anchor_count": int(chosen.shape[0]),
        "report_only": True,
    }


def run_training(args: argparse.Namespace) -> None:
    import jax
    import jax.numpy as jnp

    started = time.perf_counter()
    config = load_config(args.config, run_kind=args.run_kind)
    if config.run_kind == "formal" or bool(getattr(args, "require_cuda", False)):
        devices = [device for device in jax.devices() if device.platform == "gpu"]
        if len(devices) != 1:
            raise RuntimeError("Formal DELTA requires exactly one visible CUDA device.")
    from .official_adapter import FrozenPartnerPool, VectorEnvironment, validate_official_runtime

    official_runtime = (
        validate_official_runtime() if config.run_kind == "formal" else None
    )
    manifest_path = Path(args.partner_manifest).resolve()
    manifest = load_partner_manifest(
        manifest_path,
        expected_layout=config.environment.layout,
        verify_files=not bool(getattr(args, "skip_manifest_hash_check", False)),
    )
    members = build_training_partner_pool(config, manifest)
    (
        upstream_steps,
        upstream_gpu_hours,
        upstream_wall_clock_hours,
        upstream_records,
    ) = _upstream_partner_cost(
        members, formal=(config.run_kind == "formal")
    )
    frozen_pool = FrozenPartnerPool.from_checkpoints(
        [member.checkpoint for member in members],
        parent_training_run_ids=[member.parent_training_run_id for member in members],
    )
    partner_functions = make_static_partner_functions(
        pool=frozen_pool,
        probabilities=np.asarray([member.probability for member in members], dtype=np.float32),
        run_ids=np.arange(len(members), dtype=np.int32),
    )
    partner_parameters = None
    environment = VectorEnvironment.create(config)
    model = DeltaModel(config, environment.observation_shape, OFFICIAL_ACTION_COUNT)
    root = _training_key(int(args.seed_index))
    init_key, runner_key, loop_key = jax.random.split(root, 3)
    base_params, latent_params = model.init_parameters(init_key)
    base_optimizer_state = init_adam(base_params)
    latent_optimizer_state = init_adam(latent_params)
    runner = initialize_runner(
        environment=environment,
        model=model,
        partner_functions=partner_functions,
        random_key=runner_key,
    )
    output = Path(args.output).resolve()
    identity = {
        "stage": "train",
        "method": METHOD_VERSION,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "ego_run_id": str(args.ego_run_id),
        "seed_index": int(args.seed_index),
        "run_kind": config.run_kind,
        "layout": config.environment.layout,
        "method_variant": config.method_variant,
        "config": config.to_mapping(),
        "config_fingerprint": config.fingerprint,
        "partner_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_path(manifest_path),
            "fingerprint": manifest.fingerprint,
        },
        "official_runtime": official_runtime,
        "observation_shape": list(environment.observation_shape),
        "action_count": OFFICIAL_ACTION_COUNT,
    }
    ensure_run_identity(output, identity)
    write_json(
        output / "partner_pool.json",
        {
            "sampling": "mechanism_then_family_then_stage_then_run_uniform",
            "members": [
                {
                    "run_id": member.run_id,
                    "mechanism": member.mechanism,
                    "hyperparameter_family": member.hyperparameter_family,
                    "checkpoint_stage": member.checkpoint_stage,
                    "probability": member.probability,
                }
                for member in members
            ],
        },
    )
    write_json(output / "upstream_cost_manifest.json", {"parents": upstream_records})
    ledger = ResourceLedger(
        upstream_partner_steps=upstream_steps,
        shared_gpu_hours=upstream_gpu_hours,
        shared_wall_clock_hours=upstream_wall_clock_hours,
        deployable_parameters=parameter_count(base_params) + parameter_count(latent_params),
    )
    state = TrainState(
        base_params=base_params,
        latent_params=latent_params,
        base_optimizer_state=base_optimizer_state,
        latent_optimizer_state=latent_optimizer_state,
        runner_state=runner,
        update_count=jnp.asarray(0, dtype=jnp.int32),
        effective_environment_steps=jnp.asarray(0, dtype=jnp.int32),
        resource_ledger=ledger.to_mapping(),
    )
    checkpoint_identity = {
        "method": METHOD_VERSION,
        "config_fingerprint": config.fingerprint,
        "manifest_fingerprint": manifest.fingerprint,
        "seed_index": int(args.seed_index),
    }
    if bool(getattr(args, "resume", False)):
        restored = load_latest_checkpoint(
            output / "checkpoints", expected_identity=checkpoint_identity
        )
        if restored is None:
            raise FileNotFoundError("--resume requested but no checkpoint exists.")
        _, state = restored
    rollout_steps = config.environment.num_envs * config.training.rollout_length
    total_updates = config.training.environment_steps // rollout_steps
    maximum_updates = getattr(args, "maximum_updates", None)
    if maximum_updates is not None:
        total_updates = min(total_updates, int(maximum_updates))
    anchor_functions = _anchor_functions(
        model=model,
        partner_functions=partner_functions,
        partner_parameters=partner_parameters,
        environment=environment,
    )
    for update in range(int(np.asarray(state.update_count)), total_updates):
        shaping_horizon = max(int(config.upstream.reward_shaping_horizon), 1)
        shaping = max(
            1.0
            - int(np.asarray(state.effective_environment_steps)) / float(shaping_horizon),
            0.0,
        )
        next_steps = int(np.asarray(state.effective_environment_steps)) + rollout_steps
        anchor_trigger = bool(
            config.anchors.enabled
            and config.method_variant in {"delta_passive", "delta_active"}
            and (
                next_steps % int(config.anchors.interval_environment_steps) == 0
                or (bool(getattr(args, "force_anchor", False)) and update == 0)
            )
        )
        runner, batch, records = collect_rollout(
            state=state.runner_state,
            length=config.training.rollout_length,
            environment=environment,
            model=model,
            base_params=state.base_params,
            latent_params=state.latent_params,
            partner_functions=partner_functions,
            partner_parameters=partner_parameters,
            official_shaping_factor=shaping,
            record_anchors=anchor_trigger,
        )
        anchors = None
        if anchor_trigger:
            anchors = collect_anchor_batch(
                key=jax.random.fold_in(loop_key, 100_000 + update),
                records=records,
                functions=anchor_functions,
                base_params=state.base_params,
                latent_params=state.latent_params,
                states_per_trigger=config.anchors.states_per_trigger,
                action_count=OFFICIAL_ACTION_COUNT,
                fit_replicas=config.anchors.fit_replicas,
                evaluation_replicas=config.anchors.evaluation_replicas,
                horizon=config.method.continuation_horizon,
                gamma=config.ppo.gamma,
            )
        schedule = environment_minibatch_schedule(
            jax.random.fold_in(loop_key, update),
            environment_count=config.environment.num_envs,
            minibatches_per_epoch=config.training.minibatches_per_epoch,
            update_epochs=config.ppo.update_epochs,
        )
        (
            base_params,
            latent_params,
            base_optimizer_state,
            latent_optimizer_state,
            metrics,
        ) = training_update(
            model=model,
            base_params=state.base_params,
            latent_params=state.latent_params,
            base_optimizer_state=state.base_optimizer_state,
            latent_optimizer_state=state.latent_optimizer_state,
            batch=batch,
            anchors=anchors,
            schedule=schedule,
            total_optimizer_steps=(
                (config.training.environment_steps // rollout_steps)
                * config.training.minibatches_per_epoch
                * config.ppo.update_epochs
            ),
        )
        if (
            float(np.asarray(metrics["ppo"]["base_update_applied"])) < 1.0
            or float(np.asarray(metrics["ppo"]["base_nonfinite_update"])) > 0.0
            or float(np.asarray(metrics["latent"]["latent_nonfinite_update"])) > 0.0
        ):
            raise FloatingPointError(
                "A DELTA optimizer transaction produced NaN/Inf and was rolled back."
            )

        anchor_steps = 0
        if anchors is not None:
            anchor_steps = (
                config.anchors.states_per_trigger
                * OFFICIAL_ACTION_COUNT
                * (config.anchors.fit_replicas + config.anchors.evaluation_replicas)
                * config.method.continuation_horizon
            )
        ledger = ResourceLedger.from_mapping(state.resource_ledger).plus(
            ego_policy_steps=rollout_steps,
            anchor_continuation_steps=anchor_steps,
        )
        state = TrainState(
            base_params=base_params,
            latent_params=latent_params,
            base_optimizer_state=base_optimizer_state,
            latent_optimizer_state=latent_optimizer_state,
            runner_state=runner,
            update_count=jnp.asarray(update + 1, dtype=jnp.int32),
            effective_environment_steps=jnp.asarray(next_steps, dtype=jnp.int32),
            resource_ledger=ledger.to_mapping(),
        )
        _write_jsonl(
            output / "records" / "metrics.jsonl",
            {
                "update": update + 1,
                "environment_steps": next_steps,
                "anchor_trigger": anchor_trigger,
                "official_shaping_factor": shaping,
                "metrics": _host(metrics),
            },
        )
        if (
            next_steps % config.training.checkpoint_interval_environment_steps == 0
            and next_steps < config.training.environment_steps
        ):
            save_checkpoint(
                output / "checkpoints",
                step=next_steps,
                state=state,
                identity=checkpoint_identity,
            )
    elapsed = time.perf_counter() - started
    ledger = ResourceLedger.from_mapping(state.resource_ledger).plus(
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
            "peak_memory_bytes": max(
                int(ledger.peak_memory_bytes), peak_device_memory_bytes()
            ),
        }
    )
    state = state._replace(resource_ledger=ledger.to_mapping())

    deployment_directory = output / "final_deployment"
    if deployment_directory.exists() and any(deployment_directory.iterdir()):
        from .deployment import load_deployment

        existing = load_deployment(deployment_directory)
        if (
            pytree_fingerprint(existing.base_params)
            != pytree_fingerprint(state.base_params)
            or pytree_fingerprint(existing.latent_params)
            != pytree_fingerprint(state.latent_params)
        ):
            raise RuntimeError("Existing final deployment differs from the final checkpoint.")
    else:
        export_deployment_bundle(
            deployment_directory,
            ego_run_id=str(args.ego_run_id),
            config=config,
            observation_shape=environment.observation_shape,
            base_params=state.base_params,
            latent_params=state.latent_params,
            source_training_run=output,
        )

    # Fresh report-only decision audit.  It never alters learned parameters and
    # is idempotent across resume of a completed run.
    audit_path = output / "final_decision_audit.json"
    base_fingerprint = pytree_fingerprint(state.base_params)
    latent_fingerprint = pytree_fingerprint(state.latent_params)
    audit_complete = False
    if audit_path.is_file():
        existing_audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit_complete = (
            existing_audit.get("base_params_fingerprint") == base_fingerprint
            and existing_audit.get("latent_params_fingerprint") == latent_fingerprint
        )
    if (
        config.method_variant in {"delta_passive", "delta_active"}
        and not audit_complete
    ):
        audit_started = time.perf_counter()
        audit_runner, audit_batch, audit_records = collect_rollout(
            state=state.runner_state,
            length=config.training.rollout_length,
            environment=environment,
            model=model,
            base_params=state.base_params,
            latent_params=state.latent_params,
            partner_functions=partner_functions,
            partner_parameters=partner_parameters,
            official_shaping_factor=0.0,
            record_anchors=True,
        )
        del audit_runner
        audit_anchors = collect_anchor_batch(
            key=jax.random.fold_in(loop_key, 999_999),
            records=audit_records,
            functions=anchor_functions,
            base_params=state.base_params,
            latent_params=state.latent_params,
            states_per_trigger=config.anchors.states_per_trigger,
            action_count=OFFICIAL_ACTION_COUNT,
            fit_replicas=config.anchors.fit_replicas,
            evaluation_replicas=config.anchors.evaluation_replicas,
            horizon=config.method.continuation_horizon,
            gamma=config.ppo.gamma,
        )
        audit_steps = (
            config.environment.num_envs * config.training.rollout_length
            + config.anchors.states_per_trigger
            * OFFICIAL_ACTION_COUNT
            * (config.anchors.fit_replicas + config.anchors.evaluation_replicas)
            * config.method.continuation_horizon
        )
        audit_elapsed = time.perf_counter() - audit_started
        ledger = ledger.plus(
            final_decision_audit_steps=audit_steps,
            measurement_wall_clock_hours=audit_elapsed / 3600.0,
            measurement_gpu_hours=(
                audit_elapsed / 3600.0
                if any(device.platform == "gpu" for device in jax.devices())
                else 0.0
            ),
        )
        state = state._replace(resource_ledger=ledger.to_mapping())
        write_json(
            audit_path,
            {
                **_final_decision_audit(
                    model=model,
                    base_params=state.base_params,
                    latent_params=state.latent_params,
                    batch=audit_batch,
                    anchors=audit_anchors,
                ),
                "base_params_fingerprint": base_fingerprint,
                "latent_params_fingerprint": latent_fingerprint,
            },
        )

    measured_latency = _measure_inference_latency_ms(
        model=model,
        base_params=state.base_params,
        latent_params=state.latent_params,
    )
    current_ledger = ResourceLedger.from_mapping(state.resource_ledger)
    if current_ledger.inference_latency_ms == 0.0:
        current_ledger = ResourceLedger(
            **{
                **{
                    name: getattr(current_ledger, name)
                    for name in ResourceLedger.__dataclass_fields__
                },
                "inference_latency_ms": measured_latency,
            }
        )
        state = state._replace(resource_ledger=current_ledger.to_mapping())

    save_checkpoint(
        output / "checkpoints",
        step=int(np.asarray(state.effective_environment_steps)),
        state=state,
        identity=checkpoint_identity,
    )
    write_json(output / "resource_ledger.json", ResourceLedger.from_mapping(state.resource_ledger).to_mapping())


__all__ = ["run_cuda_preflight", "run_training"]


def run_cuda_preflight(args: argparse.Namespace) -> None:
    """Execute one registered real-partner mechanical update and persistence round trip."""
    args.maximum_updates = 1
    args.force_anchor = True
    args.require_cuda = True
    args.resume = False
    run_training(args)
    output = Path(args.output).resolve()
    deployment = output / "final_deployment"
    checkpoint = output / "checkpoints" / "latest.json"
    audit = output / "final_decision_audit.json"
    ledger = output / "resource_ledger.json"
    for path in (deployment / "deployment_bundle.json", checkpoint, audit, ledger):
        if not path.is_file():
            raise RuntimeError(f"CUDA preflight artifact is missing: {path}")
    write_json(
        output / "cuda_preflight_report.json",
        {
            "passed": True,
            "method": METHOD_VERSION,
            "config": str(Path(args.config).resolve()),
            "partner_manifest": str(Path(args.partner_manifest).resolve()),
            "deployment_sha256": sha256_path(deployment),
            "checkpoint_descriptor_sha256": sha256_path(checkpoint),
            "final_decision_audit_sha256": sha256_path(audit),
            "resource_ledger_sha256": sha256_path(ledger),
        },
    )


