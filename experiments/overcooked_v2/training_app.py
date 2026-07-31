"""End-to-end DELTA-ZSC v5 training workflow for OvercookedV2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Mapping

from experiments.overcooked_v2.deployment import deployable_parameters
from experiments.overcooked_v2.official_adapter import (
    FrozenPartnerPool,
    VectorEnvironment,
    restore_official_checkpoint,
    validate_official_runtime,
    validate_official_partner_checkpoint,
)
from src.path_c.anchor_sampling import collect_anchor_batch
from src.path_c.counterfactual_anchor import anchor_microbatch_candidates
from src.path_c.belief_set_encoder import mixture_moments
from src.path_c.calibration import calibration_to_mapping, empty_calibration
from src.path_c.decision_geometry import (
    brdiv_marginal_contributions,
    centered_action_values,
    smoothness_marginal_penalties,
)
from src.path_c.experiment import (
    load_config,
    load_partner_manifest,
    official_training_domain_keys,
    official_training_key,
)
from src.path_c.model import (
    build_model,
    initial_policy_state,
    initialize_model_parameters,
)
from src.path_c.partner_generator import (
    build_partner_generator,
    initialize_generator_parameters,
)
from src.path_c.partner_sources import (
    MixedPartnerParameters,
    make_mixed_partner_functions,
)
from src.path_c.runner import (
    attach_decision_regret_shaping,
    collect_rollout,
    initialize_runner,
)
from src.path_c.resources import (
    ResourceLedger,
    gpu_hours_for_wall_seconds,
    parameter_count,
    peak_device_memory_bytes,
)
from src.path_c.snapshot_archive import (
    load_snapshot_archive,
    save_generator_snapshot,
    stack_parameter_trees,
)
from src.path_c.storage import (
    ensure_run_identity,
    orbax_manager,
    pytree_fingerprint,
    restore_latest_checkpoint,
    save_checkpoint,
    training_identity,
    validate_formal_repository_state,
    write_json,
    validate_registered_python_runtime,
    write_jsonl,
)
from src.path_c.training import (
    apply_training_update,
    environment_minibatch_schedule,
    generator_score_function_loss,
    make_optimizer,
    official_reward_shaping_factor,
    polyak_update,
    slice_rollout_lanes,
    update_competence_multiplier,
)
from src.path_c.types import TrainState


def _host(tree: Any) -> Any:
    import jax
    import numpy as np

    if isinstance(tree, Mapping):
        return {str(key): _host(value) for key, value in tree.items()}
    if isinstance(tree, tuple) and hasattr(tree, "_fields"):
        return {name: _host(getattr(tree, name)) for name in tree._fields}
    array = np.asarray(tree)
    if array.ndim:
        return array.tolist()
    if np.issubdtype(array.dtype, np.integer):
        return int(array)
    if np.issubdtype(array.dtype, np.bool_):
        return bool(array)
    return float(array)


def _collect_with_adaptive_microbatch(
    *,
    candidates: tuple[int, ...],
    collection_kwargs: Mapping[str, Any],
) -> tuple[Any, Any, Any, Any, int]:
    """Choose the largest complete-world microbatch that the device accepts."""

    import jax

    failures = []
    for candidate in candidates:
        try:
            result = collect_anchor_batch(
                **collection_kwargs,
                microbatch_size=int(candidate),
            )
            # Host-side bound validation in the collector already synchronizes
            # returns; this explicit leaf barrier also covers future changes.
            leaves = jax.tree_util.tree_leaves(result[0])
            if leaves:
                jax.block_until_ready(leaves[0])
            return (*result, int(candidate))
        except Exception as error:
            message = f"{type(error).__name__}: {error}".lower()
            if not any(
                token in message
                for token in ("out of memory", "resource exhausted", "resource_exhausted")
            ):
                raise
            failures.append(f"{candidate}: {type(error).__name__}: {error}")
            jax.clear_caches()
    raise RuntimeError(
        "No registered anchor microbatch fits the device; scientific budgets "
        "were not reduced. Attempts=" + " | ".join(failures)
    )


def _generator_update(
    *,
    generator: Any,
    generator_optimizer: Any,
    state: TrainState,
    records: Mapping[str, Any],
    anchors: Any,
    anchor_codes: Any,
    config: Any,
) -> tuple[TrainState, Mapping[str, Any]]:
    import jax
    import jax.numpy as jnp
    import optax

    source_mask = (records["partner_source"] == 0).astype(jnp.float32)
    if float(jnp.sum(source_mask)) <= 0.0:
        return state, {"generator_update_skipped": jnp.asarray(1.0)}
    initial_carry = records["partner_state"].generator_carry[0]
    partner_observations = records["partner_observations"]
    codes = records["partner_code"]
    starts = records["episode_starts"]
    # Actions are fixed by the collected on-policy trajectory.  Keys only satisfy
    # the generator API; sampled candidate actions are ignored by the loss.
    keys = jnp.zeros(source_mask.shape + (2,), dtype=jnp.uint32)

    signatures = centered_action_values(anchors.fit_returns_by_action)
    valid_anchor = jnp.all(jnp.isfinite(anchor_codes), axis=-1)
    valid_count = int(jnp.sum(valid_anchor))
    lane_codes = codes[0]
    if valid_count >= 2:
        valid_codes = anchor_codes[valid_anchor]
        valid_signatures = signatures[valid_anchor]
        contributions = jax.lax.stop_gradient(
            brdiv_marginal_contributions(
                valid_signatures,
                config.partner_generator.kernel_bandwidth,
                config.partner_generator.kernel_jitter,
            )
        )
        squared = jnp.sum(
            jnp.square(lane_codes[:, None, :] - valid_codes[None, :, :]), axis=-1
        )
        nearest = jnp.argmin(squared, axis=-1)
        smoothness_penalty = jax.lax.stop_gradient(
            smoothness_marginal_penalties(valid_codes, valid_signatures)
        )
        diversity_bonus = (
            float(config.partner_generator.brdiv_weight) * contributions[nearest]
            - float(config.partner_generator.smoothness_weight)
            * smoothness_penalty[nearest]
        )
    else:
        diversity_bonus = jnp.zeros((lane_codes.shape[0],), dtype=jnp.float32)

    def objective(candidate: Any) -> tuple[Any, Mapping[str, Any]]:
        unused_carry, output = generator.apply(
            {"params": candidate},
            initial_carry,
            partner_observations,
            codes[0],
            starts,
            keys,
            method=generator.sequence,
        )
        del unused_carry
        loss = generator_score_function_loss(
            logits=output.logits,
            values=output.value,
            actions=records["partner_actions"],
            source_mask=source_mask,
            rewards=records["rewards"],
            dones=records["dones"],
            diversity_bonus=diversity_bonus,
            value_weight=config.ppo.value_weight,
            entropy_weight=config.ppo.entropy_weight,
            competence_multiplier=state.competence_multiplier,
            competence_threshold=config.partner_generator.competence_threshold,
            cvar_level=config.partner_generator.cvar_level,
        )
        return loss.total, loss.metrics

    (unused, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(
        state.generator_params
    )
    del unused
    updates, optimizer_state = generator_optimizer.update(
        gradients, state.generator_optimizer_state, state.generator_params
    )
    generator_params = optax.apply_updates(state.generator_params, updates)
    multiplier = update_competence_multiplier(
        state.competence_multiplier,
        competence=metrics["generator_competence_cvar"],
        threshold=config.partner_generator.competence_threshold,
        learning_rate=config.partner_generator.lagrangian_learning_rate,
    )
    return state._replace(
        generator_params=generator_params,
        generator_target_params=polyak_update(
            state.generator_target_params,
            generator_params,
            config.ppo.polyak_coefficient,
        ),
        generator_optimizer_state=optimizer_state,
        competence_multiplier=multiplier,
    ), metrics




def _write_final_support_latents(
    *,
    output: Path,
    config: Any,
    environment: Any,
    model: Any,
    state: TrainState,
    partner_functions: Any,
    partner_parameters: Any,
    key: Any,
) -> Mapping[str, Any]:
    """Collect final-model legal-history support from the complete train mixture."""

    import jax.numpy as jnp
    import numpy as np

    runner = initialize_runner(
        environment=environment,
        model_config=config.model,
        partner_functions=partner_functions,
        random_key=key,
    )
    runner, unused_batch, records = collect_rollout(
        state=runner,
        length=config.environment.episode_steps,
        environment=environment,
        model=model,
        params=state.params,
        model_config=config.model,
        partner_functions=partner_functions,
        partner_parameters=partner_parameters,
        gate_values=jnp.ones((environment.num_envs,), dtype=jnp.float32),
        teacher_lane_mask=jnp.zeros((environment.num_envs,), dtype=jnp.bool_),
        official_shaping_factor=0.0,
    )
    del unused_batch
    mean, variance = mixture_moments(
        records["mixture_logits"],
        records["mixture_means"],
        records["mixture_log_variances"],
    )
    target = output / "records" / "training_support_latents.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        posterior_mean=np.asarray(mean).reshape((-1, mean.shape[-1])),
        posterior_variance=np.asarray(variance).reshape((-1, variance.shape[-1])),
        partner_source=np.asarray(records["partner_source"]).reshape((-1,)),
        partner_run_id=np.asarray(records["partner_run_ids"]).reshape((-1,)),
    )
    metadata = {
        "path": str(target),
        "row_count": int(mean.shape[0] * mean.shape[1]),
        "environment_steps": int(environment.num_envs * config.environment.episode_steps),
        "completed_episodes": int(np.asarray(runner.completed_episodes)),
        "model_fingerprint": pytree_fingerprint(state.params),
        "config_fingerprint": config.fingerprint,
        "source_counts": {
            str(int(source)): int(
                np.sum(np.asarray(records["partner_source"]).reshape((-1,)) == source)
            )
            for source in np.unique(
                np.asarray(records["partner_source"]).reshape((-1,))
            )
        },
    }
    write_json(output / "records" / "training_support_metadata.json", metadata)
    return metadata


def run_training(args: argparse.Namespace) -> None:
    import jax
    import jax.numpy as jnp
    import numpy as np

    started = time.perf_counter()
    config = load_config(args.config, run_kind=args.run_kind)
    if config.run_kind == "formal":
        validate_formal_repository_state()
        validate_registered_python_runtime()
    official_runtime = (
        validate_official_runtime() if config.run_kind == "formal" else None
    )
    manifest = load_partner_manifest(
        args.partner_manifest,
        expected_layout=config.environment.layout,
        verify_files=not bool(args.skip_manifest_hash_check),
    )
    output = Path(args.output).resolve()
    environment = VectorEnvironment.create(config)
    observation_shape = environment.observation_shape
    outer_key = official_training_key(int(args.seed_index))
    domain_keys = official_training_domain_keys(int(args.seed_index))
    identity = dict(
        training_identity(
            config=config,
            seed_index=int(args.seed_index),
            jax_prng_key=outer_key,
            partner_manifest=manifest,
        )
    )
    identity.update(
        {
            "ego_run_id": str(args.ego_run_id),
            "observation_shape": list(observation_shape),
            "action_count": 6,
            "official_runtime": official_runtime,
        }
    )
    ensure_run_identity(output, identity)
    write_json(output / "resolved_config.json", config.to_mapping())
    write_json(output / "resolved_partner_manifest.json", manifest.to_mapping())

    external_runs = tuple(
        run
        for run in manifest.by_role("frozen_external_train")
        if run.owner_seed_index == int(args.seed_index)
    )
    parent_training_steps_by_id: dict[str, int] = {}
    for run in external_runs:
        if run.parent_training_run_id in parent_training_steps_by_id:
            continue
        checkpoint_config, unused_checkpoint_params = restore_official_checkpoint(
            run.checkpoint
        )
        del unused_checkpoint_params
        checkpoint_model = checkpoint_config["model"]
        steps_per_update = int(checkpoint_model["NUM_ENVS"]) * int(
            checkpoint_model["NUM_STEPS"]
        )
        parent_training_steps_by_id[run.parent_training_run_id] = (
            int(checkpoint_model["TOTAL_TIMESTEPS"]) // steps_per_update
        ) * steps_per_update
    if config.run_kind == "formal":
        mechanisms = {run.generation_mechanism for run in external_runs}
        if not {"rnn-sp", "rnn-op"}.issubset(mechanisms):
            raise ValueError(
                "Formal DELTA run requires its own Official SP and OP parent support."
            )
        unowned = [
            run.run_id
            for run in manifest.by_role("frozen_external_train")
            if run.owner_seed_index is None
        ]
        if unowned:
            raise ValueError(
                "Formal frozen training partners must bind one owner seed: "
                f"{unowned}"
            )
        parents = {}
        for run in external_runs:
            parents.setdefault(run.parent_training_run_id, []).append(run)
        if len(parents) != 2 or {len(values) for values in parents.values()} != {3}:
            raise ValueError(
                "Each formal DELTA run requires exactly one SP parent and one OP "
                "parent, with three frozen checkpoints per parent."
            )
        if {
            values[0].generation_mechanism for values in parents.values()
        } != {"rnn-sp", "rnn-op"}:
            raise ValueError("Formal frozen parent mechanisms must be SP and OP.")
        expected_key = tuple(official_training_key(int(args.seed_index)))
        for parent, values in parents.items():
            if (
                {run.owner_seed_index for run in values} != {int(args.seed_index)}
                or {run.seed_index for run in values} != {int(args.seed_index)}
                or {tuple(run.jax_prng_key or ()) for run in values} != {expected_key}
                or len({run.generation_mechanism for run in values}) != 1
            ):
                raise ValueError(
                    f"Formal frozen parent provenance is inconsistent: {parent}."
                )
            algorithm = values[0].generation_mechanism
            for run in values:
                validate_official_partner_checkpoint(
                    run.checkpoint,
                    config=config,
                    algorithm=algorithm,
                    seed_index=int(args.seed_index),
                )
    external_paths = [run.checkpoint for run in external_runs]
    external_pool = (
        FrozenPartnerPool.from_checkpoints(
            external_paths,
            parent_training_run_ids=[
                run.parent_training_run_id for run in external_runs
            ],
        )
        if external_paths
        else None
    )
    model = build_model(
        observation_shape=observation_shape,
        action_count=6,
        **{
            name: getattr(config.model, name)
            for name in (
                "task_hidden_dim",
                "belief_hidden_dim",
                "latent_dim",
                "mixture_components",
                "belief_embedding_dim",
                "actor_hidden_dim",
                "critic_hidden_dim",
                "response_hidden_dim",
                "modulation_rank",
                "action_embedding_dim",
                "log_variance_minimum",
                "log_variance_maximum",
                "response_log_std_minimum",
                "response_log_std_maximum",
            )
        },
    )
    generator = build_partner_generator(
        observation_shape=observation_shape,
        action_count=6,
        code_dim=config.partner_generator.code_dim,
        hidden_dim=config.partner_generator.hidden_dim,
        modulation_rank=config.partner_generator.modulation_rank,
    )

    ego_root = jnp.asarray(domain_keys["ego"], dtype=jnp.uint32)
    generator_root = jnp.asarray(domain_keys["generator"], dtype=jnp.uint32)
    anchor_root = jnp.asarray(domain_keys["anchor"], dtype=jnp.uint32)
    reset_key, model_key, runner_key, state_key = jax.random.split(ego_root, 4)
    generator_key = jax.random.fold_in(generator_root, 0)
    unused_environment_state, observations = environment.reset(reset_key)
    del unused_environment_state
    example_state = initial_policy_state(
        batch_size=environment.num_envs,
        observation_shape=observation_shape,
        action_count=6,
        task_hidden_dim=config.model.task_hidden_dim,
        belief_hidden_dim=config.model.belief_hidden_dim,
        latent_dim=config.model.latent_dim,
        mixture_components=config.model.mixture_components,
    )
    params = initialize_model_parameters(
        model,
        key=model_key,
        example_state=example_state,
        example_observation=observations[:, 0],
        partner_code_dim=config.partner_generator.code_dim,
    )
    generator_params = initialize_generator_parameters(
        generator,
        key=generator_key,
        observation_shape=observation_shape,
        code_dim=config.partner_generator.code_dim,
        batch_size=environment.num_envs,
        hidden_dim=config.partner_generator.hidden_dim,
    )
    total_updates = (
        config.training.environment_steps
        // config.environment.num_envs
        // config.training.rollout_length
    )
    optimizer, optimizer_state = make_optimizer(
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
    generator_optimizer, generator_optimizer_state = make_optimizer(
        generator_params,
        learning_rate=config.ppo.learning_rate,
        gradient_clip_norm=config.ppo.gradient_clip_norm,
        adam_epsilon=config.ppo.adam_epsilon,
    )

    snapshot_root = output / "generator_snapshots"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    bootstrap_snapshot = snapshot_root / "snapshot_00000000"
    if (
        config.partner_generator.snapshot_probability > 0.0
        and not bootstrap_snapshot.exists()
    ):
        save_generator_snapshot(bootstrap_snapshot, generator_params)
    snapshot_paths = [
        *(
            run.checkpoint
            for run in manifest.by_role("generator_snapshot")
            if run.owner_seed_index == int(args.seed_index)
        ),
        *sorted(snapshot_root.glob("snapshot_*")),
    ]
    snapshot_archive = (
        list(load_snapshot_archive(snapshot_paths)) if snapshot_paths else []
    )

    def teacher_latent_apply(teacher_params: Any, code: Any, task_features: Any) -> Any:
        return model.apply(
            {"params": teacher_params},
            code,
            task_features,
            1.0,
            method=model.teacher_from_code,
        ).latent

    partner_functions = make_mixed_partner_functions(
        generator=generator,
        generator_hidden_dim=config.partner_generator.hidden_dim,
        generator_code_dim=config.partner_generator.code_dim,
        snapshot_count=len(snapshot_archive),
        external_pool=external_pool,
        current_probability=config.partner_generator.current_probability,
        snapshot_probability=config.partner_generator.snapshot_probability,
        frozen_external_probability=config.partner_generator.frozen_external_probability,
        teacher_latent_apply=teacher_latent_apply,
    )
    runner_state = initialize_runner(
        environment=environment,
        model_config=config.model,
        partner_functions=partner_functions,
        random_key=runner_key,
    )
    state = TrainState(
        params=params,
        target_params=params,
        optimizer_state=optimizer_state,
        generator_params=generator_params,
        generator_target_params=generator_params,
        generator_optimizer_state=generator_optimizer_state,
        competence_multiplier=jnp.asarray(0.0, dtype=jnp.float32),
        random_key=state_key,
        update_count=jnp.asarray(0, dtype=jnp.int64),
        effective_environment_steps=jnp.asarray(0, dtype=jnp.int64),
        runner_state=runner_state,
        calibration=empty_calibration(
            latent_dim=config.model.latent_dim,
            alpha=config.calibration.alpha,
        ),
    )
    manager = orbax_manager(output / "checkpoints")
    if manager.latest_step() is not None and not args.resume:
        raise RuntimeError("Output already has a checkpoint; use --resume.")
    if args.resume:
        restored = restore_latest_checkpoint(manager, item=state)
        if restored is not None:
            unused_step, state = restored
            del unused_step
            runner_state = state.runner_state

    metrics_path = output / "records" / "metrics"
    anchor_path = output / "records" / "anchors"
    resource_progress_path = output / "resource_progress.json"
    previous_gpu_hours = 0.0
    if resource_progress_path.is_file():
        previous_gpu_hours = float(
            json.loads(resource_progress_path.read_text(encoding="utf-8")).get(
                "gpu_hours", 0.0
            )
        )
    anchor_microbatch_size: int | None = None
    completed_anchor_triggers = (
        int(np.asarray(state.update_count)) // config.anchors.interval_updates
    )
    matched_pair_count = min(
        config.anchors.states_per_interval,
        config.partner_generator.codes_per_update,
    )
    counterfactual_steps = (
        completed_anchor_triggers
        * (config.anchors.states_per_interval + 2 * matched_pair_count)
        * 6
        * (config.anchors.fit_replicas + config.anchors.evaluation_replicas)
        * config.anchors.continuation_horizon
    )
    anchor_candidates = anchor_microbatch_candidates(
        maximum_anchor_worlds=max(
            config.anchors.states_per_interval,
            2 * min(
                config.anchors.states_per_interval,
                config.partner_generator.codes_per_update,
            ),
        ),
        action_count=6,
        replicas=(
            config.anchors.fit_replicas
            + config.anchors.evaluation_replicas
        ),
    )
    while int(np.asarray(state.effective_environment_steps)) < config.training.environment_steps:
        (
            collection_key,
            shaping_key,
            schedule_key,
            lane_key,
            next_key,
        ) = jax.random.split(state.random_key, 5)
        if snapshot_archive:
            stacked_snapshots = stack_parameter_trees(snapshot_archive)
        else:
            stacked_snapshots = state.generator_params
        partner_parameters = MixedPartnerParameters(
            generator_params=state.generator_params,
            snapshot_params=stacked_snapshots,
            teacher_params=state.params,
        )
        teacher_key, base_key = jax.random.split(lane_key)
        teacher_mask = jax.random.bernoulli(
            teacher_key,
            config.training.teacher_lane_probability,
            (config.environment.num_envs,),
        )
        base_mask = jax.random.bernoulli(
            base_key,
            config.training.base_policy_lane_probability,
            (config.environment.num_envs,),
        )
        gate_values = (~base_mask).astype(jnp.float32)
        runner_state = runner_state._replace(random_key=collection_key)
        runner_state, batch, records = collect_rollout(
            state=runner_state,
            length=config.training.rollout_length,
            environment=environment,
            model=model,
            params=state.params,
            model_config=config.model,
            partner_functions=partner_functions,
            partner_parameters=partner_parameters,
            gate_values=gate_values,
            teacher_lane_mask=teacher_mask,
            official_shaping_factor=float(
                official_reward_shaping_factor(
                    state.effective_environment_steps,
                    horizon=config.upstream.reward_shaping_horizon,
                )
            ),
        )
        batch, shaping_metrics = attach_decision_regret_shaping(
            batch=batch,
            model=model,
            target_params=state.target_params,
            config=config,
            key=shaping_key,
        )
        shaping_metrics = dict(shaping_metrics)
        records = dict(records)
        records["decision_regret"] = shaping_metrics.pop(
            "decision_regret_values"
        )

        update_number = int(np.asarray(state.update_count)) + 1
        anchors = None
        quotient_pairs = None
        anchor_codes = None
        if update_number % config.anchors.interval_updates == 0:
            collection_kwargs = dict(
                anchor_domain=update_number,
                key=jax.random.fold_in(anchor_root, update_number),
                records=records,
                environment=environment,
                model=model,
                target_params=state.target_params,
                config=config,
                partner_functions=partner_functions,
                partner_parameters=partner_parameters,
            )
            if anchor_microbatch_size is None:
                (
                    anchors,
                    quotient_pairs,
                    anchor_codes,
                    unused_anchor_indexes,
                    anchor_microbatch_size,
                ) = _collect_with_adaptive_microbatch(
                    candidates=anchor_candidates,
                    collection_kwargs=collection_kwargs,
                )
                write_json(
                    output / "anchor_microbatch.json",
                    {
                        "candidates": list(anchor_candidates),
                        "selected": anchor_microbatch_size,
                        "changes_scientific_samples": False,
                    },
                )
            else:
                (
                    anchors,
                    quotient_pairs,
                    anchor_codes,
                    unused_anchor_indexes,
                ) = collect_anchor_batch(
                    **collection_kwargs,
                    microbatch_size=anchor_microbatch_size,
                )
            del unused_anchor_indexes
            counterfactual_steps += (
                int(anchors.anchor_ids.shape[0])
                * 6
                * (
                    config.anchors.fit_replicas
                    + config.anchors.evaluation_replicas
                )
                * config.anchors.continuation_horizon
            )
            write_json(
                anchor_path / f"update_{update_number:08d}.json",
                {
                    "anchor_count": int(anchors.anchor_ids.shape[0]),
                    "fit_returns_by_action": _host(anchors.fit_returns_by_action),
                    "evaluation_returns_by_action": _host(
                        anchors.evaluation_returns_by_action
                    ),
                    "partner_run_ids": _host(anchors.partner_run_ids),
                },
            )

        schedule = environment_minibatch_schedule(
            schedule_key,
            environment_count=config.environment.num_envs,
            minibatches_per_epoch=config.training.minibatches_per_epoch,
            update_epochs=config.ppo.update_epochs,
        )
        update_metrics = []
        first_minibatch = True
        for epoch in range(config.ppo.update_epochs):
            for minibatch in range(config.training.minibatches_per_epoch):
                indexes = schedule[epoch, minibatch]
                current_anchors = anchors if first_minibatch else None
                current_quotient = quotient_pairs if first_minibatch else None
                update = apply_training_update(
                    model=model,
                    state=state,
                    optimizer=optimizer,
                    batch=slice_rollout_lanes(batch, indexes),
                    config=config,
                    anchors=current_anchors,
                    quotient_pairs=current_quotient,
                )
                state = update.state
                update_metrics.append(update.metrics)
                first_minibatch = False

        generator_metrics: Mapping[str, Any] = {
            "generator_update_skipped": jnp.asarray(1.0)
        }
        if (
            anchors is not None
            and update_number % config.partner_generator.update_interval == 0
        ):
            state, generator_metrics = _generator_update(
                generator=generator,
                generator_optimizer=generator_optimizer,
                state=state,
                records=records,
                anchors=anchors,
                anchor_codes=anchor_codes,
                config=config,
            )

        if update_number % config.partner_generator.snapshot_interval == 0:
            snapshot = save_generator_snapshot(
                snapshot_root / f"snapshot_{update_number:08d}",
                state.generator_params,
            )
            snapshot_archive.append(state.generator_params)
            # Rebuild static runtime so the next collector can address the new archive.
            partner_functions = make_mixed_partner_functions(
                generator=generator,
                generator_hidden_dim=config.partner_generator.hidden_dim,
                generator_code_dim=config.partner_generator.code_dim,
                snapshot_count=len(snapshot_archive),
                external_pool=external_pool,
                current_probability=config.partner_generator.current_probability,
                snapshot_probability=config.partner_generator.snapshot_probability,
                frozen_external_probability=config.partner_generator.frozen_external_probability,
                teacher_latent_apply=teacher_latent_apply,
            )

        state = state._replace(
            random_key=next_key,
            update_count=state.update_count + 1,
            effective_environment_steps=runner_state.effective_environment_steps,
            runner_state=runner_state,
        )
        mean_metrics = jax.tree_util.tree_map(
            lambda *values: jnp.mean(jnp.stack(values)), *update_metrics
        )
        write_jsonl(
            metrics_path / f"update_{update_number:08d}.jsonl",
            (
                {
                    "update_count": update_number,
                    "effective_environment_steps": int(
                        np.asarray(state.effective_environment_steps)
                    ),
                    "completed_episodes": int(
                        np.asarray(runner_state.completed_episodes)
                    ),
                    "losses": _host(mean_metrics),
                    "shaping": _host(shaping_metrics),
                    "generator": _host(generator_metrics),
                    "ppo_early_stop_for_kl": False,
                    "calibration": calibration_to_mapping(state.calibration),
                },
            ),
        )
        step = int(np.asarray(state.effective_environment_steps))
        if step % config.training.checkpoint_interval_environment_steps == 0:
            save_checkpoint(manager, step=step, state=state)
        write_json(
            resource_progress_path,
            {
                "effective_environment_steps": step,
                "counterfactual_steps": counterfactual_steps,
                "gpu_hours": previous_gpu_hours
                + gpu_hours_for_wall_seconds(time.perf_counter() - started),
            },
        )

    final_step = int(np.asarray(state.effective_environment_steps))
    if manager.latest_step() != final_step:
        save_checkpoint(manager, step=final_step, state=state)
    final_snapshot_params = (
        stack_parameter_trees(snapshot_archive)
        if snapshot_archive
        else state.generator_params
    )
    final_partner_parameters = MixedPartnerParameters(
        generator_params=state.generator_params,
        snapshot_params=final_snapshot_params,
        teacher_params=state.params,
    )
    support_metadata = _write_final_support_latents(
        output=output,
        config=config,
        environment=environment,
        model=model,
        state=state,
        partner_functions=partner_functions,
        partner_parameters=final_partner_parameters,
        key=jax.random.fold_in(
            jnp.asarray(domain_keys["snapshot"], dtype=jnp.uint32), 900_001
        ),
    )
    write_json(
        output / "run_metadata.json",
        {
            "method": identity["method"],
            "seed_index": int(args.seed_index),
            "jax_prng_key": list(outer_key),
            "domain_keys": {
                name: list(value) for name, value in domain_keys.items()
            },
            "ego_run_id": str(args.ego_run_id),
            "effective_environment_steps": final_step,
            "update_count": int(np.asarray(state.update_count)),
            "completed_episodes": int(np.asarray(runner_state.completed_episodes)),
            "support_collection": support_metadata,
            "scientific_readout_allowed": False,
        },
    )
    partner_training_steps = sum(parent_training_steps_by_id.values())
    deployable_count = parameter_count(deployable_parameters(state.params))
    training_only_count = (
        parameter_count(state.params)
        - deployable_count
        + parameter_count(state.target_params)
        + parameter_count(state.generator_params)
        + parameter_count(state.generator_target_params)
    )
    ledger = ResourceLedger(
        ego_policy_steps=final_step,
        partner_training_steps=partner_training_steps,
        counterfactual_steps=counterfactual_steps,
        calibration_steps=int(support_metadata["environment_steps"]),
        gpu_hours=previous_gpu_hours
        + gpu_hours_for_wall_seconds(time.perf_counter() - started),
        peak_memory_bytes=peak_device_memory_bytes(),
        deployable_parameters=deployable_count,
        training_only_parameters=training_only_count,
    )
    write_json(output / "resource_ledger.json", ledger.to_mapping())
    print(f"Complete DELTA-ZSC training run: {output}")


__all__ = ["run_training"]
