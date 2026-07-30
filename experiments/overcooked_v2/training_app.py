"""End-to-end DELTA-ZSC v5 training workflow for OvercookedV2."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from experiments.overcooked_v2.official_adapter import FrozenPartnerPool, VectorEnvironment
from src.path_c.anchor_sampling import collect_anchor_batch
from src.path_c.belief_set_encoder import mixture_moments
from src.path_c.calibration import calibration_to_mapping, empty_calibration
from src.path_c.decision_geometry import (
    brdiv_marginal_contributions,
    centered_action_values,
    smoothness_marginal_penalties,
)
from src.path_c.experiment import load_config, load_partner_manifest
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
    write_json,
    write_jsonl,
)
from src.path_c.training import (
    apply_training_update,
    environment_minibatch_schedule,
    generator_score_function_loss,
    make_optimizer,
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
    partner_observations = records["joint_observations"][:, :, 1]
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

    config = load_config(args.config, run_kind=args.run_kind)
    manifest = load_partner_manifest(
        args.partner_manifest,
        expected_layout=config.environment.layout,
        verify_files=not bool(args.skip_manifest_hash_check),
    )
    output = Path(args.output).resolve()
    environment = VectorEnvironment.create(config)
    observation_shape = environment.observation_shape
    identity = dict(
        training_identity(config=config, seed=args.seed, partner_manifest=manifest)
    )
    identity.update(
        {
            "ego_run_id": str(args.ego_run_id),
            "observation_shape": list(observation_shape),
            "action_count": 6,
        }
    )
    ensure_run_identity(output, identity)
    write_json(output / "resolved_config.json", config.to_mapping())
    write_json(output / "resolved_partner_manifest.json", manifest.to_mapping())

    external_paths = [
        run.checkpoint for run in manifest.by_role("frozen_external_train")
    ]
    external_pool = (
        FrozenPartnerPool.from_checkpoints(external_paths) if external_paths else None
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

    root = jax.random.PRNGKey(int(args.seed))
    root, reset_key, model_key, generator_key, runner_key, state_key = jax.random.split(
        root, 6
    )
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
    optimizer, optimizer_state = make_optimizer(
        params,
        learning_rate=config.ppo.learning_rate,
        gradient_clip_norm=config.ppo.gradient_clip_norm,
    )
    generator_optimizer, generator_optimizer_state = make_optimizer(
        generator_params,
        learning_rate=config.ppo.learning_rate,
        gradient_clip_norm=config.ppo.gradient_clip_norm,
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
        *(run.checkpoint for run in manifest.by_role("generator_snapshot")),
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
    while int(np.asarray(state.effective_environment_steps)) < config.training.environment_steps:
        (
            collection_key,
            shaping_key,
            schedule_key,
            anchor_key,
            lane_key,
            next_key,
        ) = jax.random.split(state.random_key, 6)
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
            anchors, quotient_pairs, anchor_codes, unused_anchor_indexes = collect_anchor_batch(
                anchor_domain=update_number,
                key=anchor_key,
                records=records,
                environment=environment,
                model=model,
                target_params=state.target_params,
                config=config,
                partner_functions=partner_functions,
                partner_parameters=partner_parameters,
            )
            del unused_anchor_indexes
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
        stop_for_kl = False
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
                if float(np.asarray(update.metrics["approx_kl"])) > float(
                    config.ppo.max_approx_kl
                ):
                    stop_for_kl = True
                    break
            if stop_for_kl:
                break

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
                    "ppo_early_stop_for_kl": bool(stop_for_kl),
                    "calibration": calibration_to_mapping(state.calibration),
                },
            ),
        )
        step = int(np.asarray(state.effective_environment_steps))
        if step % config.training.checkpoint_interval_environment_steps == 0:
            save_checkpoint(manager, step=step, state=state)

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
        key=jax.random.fold_in(state.random_key, 900_001),
    )
    write_json(
        output / "run_metadata.json",
        {
            "method": identity["method"],
            "seed": int(args.seed),
            "ego_run_id": str(args.ego_run_id),
            "effective_environment_steps": final_step,
            "update_count": int(np.asarray(state.update_count)),
            "completed_episodes": int(np.asarray(runner_state.completed_episodes)),
            "support_collection": support_metadata,
            "scientific_readout_allowed": False,
        },
    )
    print(f"Complete DELTA-ZSC training run: {output}")


__all__ = ["run_training"]
