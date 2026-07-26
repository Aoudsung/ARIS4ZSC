"""One command-line entry for Path C upstream training, method training, and evaluation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, NamedTuple

from experiments.overcooked_v2.official_adapter import (
    FrozenPartnerPool,
    OfficialNetwork,
    VectorEnvironment,
    compose_official_config,
    restore_official_checkpoint,
    train_upstream,
)
from src.path_c.evaluation import (
    EpisodeRow,
    Pairing,
    PopulationEntry,
    ResponseContrastRow,
    load_population,
    standard_episode_seed,
    standard_pairings,
    summarize_standard_rows,
    summarize_response_contrast,
    validate_standard_rows,
    validate_response_contrast_rows,
)
from src.path_c.method import (
    deployment_belief_after_response,
    empty_codebook,
    policy_effect_trigger_tolerance,
    slot_bayes_update,
    uniform_slot_log_belief,
)
from src.path_c.model import (
    build_model,
    encode_response_codes,
    initialize_heads,
)
from src.path_c.runner import (
    RunnerFunctions,
    compiled_collector,
    initialize_runner,
    partner_callbacks,
)
from src.path_c.storage import (
    CompleteConsoleLog,
    load_config,
    orbax_manager,
    restore_latest_checkpoint,
    save_checkpoint,
    write_array_chunks,
    write_json,
    write_jsonl,
    write_parquet,
    read_parquet,
    write_run_metadata,
)
from src.path_c.training import (
    ModelFunctions,
    OptimizerBundle,
    TrainState,
    apply_rollout_updates,
    environment_minibatch_schedule,
    make_optimizers,
    prepare_frozen_assignments,
    rollout_kl_means,
    sample_bootstrap_mask,
    update_codebook_from_assignments,
    update_policy_temperatures,
)


def _load_unit(
    manifest_path: str | Path, outer_unit: int
) -> tuple[Path, tuple[Path, ...]]:
    path = Path(manifest_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    units = payload["units"]
    item = units[int(outer_unit)]
    if int(item["outer_unit_id"]) != int(outer_unit):
        raise ValueError("The requested training unit is not in manifest order.")

    def resolve(value: Any) -> Path:
        candidate = Path(str(value))
        return (candidate if candidate.is_absolute() else path.parent / candidate).resolve()

    return (
        resolve(item["reference_checkpoint"]),
        tuple(resolve(value) for value in item["partner_checkpoints"]),
    )


def _host(value: Any) -> Any:
    import numpy as np

    if isinstance(value, Mapping):
        return {str(name): _host(item) for name, item in value.items()}
    if isinstance(value, tuple):
        return [_host(item) for item in value]
    array = np.asarray(value)
    if array.ndim:
        return array.tolist()
    if np.issubdtype(array.dtype, np.bool_):
        return bool(array)
    if np.issubdtype(array.dtype, np.integer):
        return int(array)
    return float(array)


def _jsonl_row_count(path: str | Path) -> int:
    count = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            json.loads(line)
            count += 1
    return count


def _decision_rows(
    batch: Any,
    records: Mapping[str, Any],
    *,
    update_count: int,
) -> Iterable[Mapping[str, Any]]:
    import numpy as np

    arrays = {
        **{
            name: np.asarray(value)
            for name, value in batch._asdict().items()
        },
        **{name: np.asarray(value) for name, value in records.items()},
    }
    time_count, environment_count = arrays["actions"].shape
    for time_index in range(time_count):
        for environment_index in range(environment_count):
            yield {
                "update_count": int(update_count),
                "time_index": time_index,
                "environment_index": environment_index,
                "episode_id": int(
                    arrays["episode_ids"][time_index, environment_index]
                ),
                "episode_step": int(
                    arrays["episode_steps"][time_index, environment_index]
                ),
                "partner_member": int(
                    arrays["partner_members"][time_index, environment_index]
                ),
                "ego_seat": int(
                    arrays["ego_seats"][time_index, environment_index]
                ),
                "action": int(
                    arrays["actions"][time_index, environment_index]
                ),
                "reward": float(
                    arrays["rewards"][time_index, environment_index]
                ),
                "done": bool(
                    arrays["dones"][time_index, environment_index]
                ),
                "response_code": int(
                    arrays["response_codes"][time_index, environment_index]
                ),
                "reference_logits": arrays["reference_logits"][
                    time_index, environment_index
                ].tolist(),
                "execution_logits": arrays["execution_logits"][
                    time_index, environment_index
                ].tolist(),
                "mask_execution_logits": arrays["mask_execution_logits"][
                    time_index, environment_index
                ].tolist(),
                "generic_execution_logits": arrays[
                    "generic_execution_logits"
                ][time_index, environment_index].tolist(),
                "slot_log_belief": arrays["slot_log_beliefs"][
                    time_index, environment_index
                ].tolist(),
                "value_class_count": int(
                    arrays["value_class_counts"][
                        time_index, environment_index
                    ]
                ),
                "j_use": arrays["j_use"][
                    time_index, environment_index
                ].tolist(),
                "j_mask": arrays["j_mask"][
                    time_index, environment_index
                ].tolist(),
                "per_action_response_value": arrays[
                    "per_action_response_values"
                ][time_index, environment_index].tolist(),
                "per_action_net_value": arrays["per_action_net_values"][
                    time_index, environment_index
                ].tolist(),
                "predicted_response_effect": float(
                    arrays["predicted_response_effects"][
                        time_index, environment_index
                    ]
                ),
                "predicted_policy_cost": float(
                    arrays["predicted_policy_costs"][
                        time_index, environment_index
                    ]
                ),
                "predicted_net_effect": float(
                    arrays["predicted_net_effects"][
                        time_index, environment_index
                    ]
                ),
                "predicted_regularized_net_effect": float(
                    arrays["predicted_regularized_net_effects"][
                        time_index, environment_index
                    ]
                ),
                "predicted_policy_total_variation": float(
                    arrays["predicted_policy_total_variations"][
                        time_index, environment_index
                    ]
                ),
                "kl_divergence": float(
                    arrays["kl_divergences"][
                        time_index, environment_index
                    ]
                ),
                "reference_greedy_action": int(
                    arrays["reference_greedy_actions"][
                        time_index, environment_index
                    ]
                ),
                "belief_entropy": float(
                    arrays["belief_entropies"][
                        time_index, environment_index
                    ]
                ),
                "executed_action_response_value": float(
                    arrays["executed_action_response_values"][
                        time_index, environment_index
                    ]
                ),
                "executed_action_net_value": float(
                    arrays["executed_action_net_values"][
                        time_index, environment_index
                    ]
                ),
                "maximum_action_net_value": float(
                    arrays["maximum_action_net_values"][
                        time_index, environment_index
                    ]
                ),
            }


def _episode_rows(
    records: Mapping[str, Any], *, update_count: int
) -> Iterable[Mapping[str, Any]]:
    import numpy as np

    arrays = {name: np.asarray(value) for name, value in records.items()}
    mask = np.asarray(arrays["completed_episode_mask"], dtype=np.bool_)
    time_indexes, environment_indexes = np.nonzero(mask)
    for time_index, environment_index in zip(
        time_indexes.tolist(), environment_indexes.tolist(), strict=True
    ):
        yield {
            "update_count": int(update_count),
            "environment_index": int(environment_index),
            "episode_id": int(arrays["episode_ids"][time_index, environment_index]),
            "partner_member": int(
                arrays["partner_members"][time_index, environment_index]
            ),
            "ego_seat": int(
                arrays["ego_seats"][time_index, environment_index]
            ),
            "environment_steps": int(
                arrays["episode_steps"][time_index, environment_index]
            ) + 1,
            "raw_return": float(
                arrays["completed_episode_returns"][
                    time_index, environment_index
                ]
            ),
            "correct_delivery_count": int(
                arrays["completed_correct_deliveries"][
                    time_index, environment_index
                ]
            ),
            "wrong_delivery_count": int(
                arrays["completed_wrong_deliveries"][
                    time_index, environment_index
                ]
            ),
            "indicator_activation_count": int(
                arrays["completed_indicator_activations"][
                    time_index, environment_index
                ]
            ),
        }


def _assert_finite(tree: Any, *, step: int) -> None:
    import jax
    import numpy as np

    for index, value in enumerate(jax.tree_util.tree_leaves(tree)):
        if not np.isfinite(np.asarray(value)).all():
            raise FloatingPointError(
                f"Non-finite value at training step {step}, leaf {index}."
            )


def _assert_nonnegative_deliveries(
    records: Mapping[str, Any], *, step: int
) -> None:
    import numpy as np

    completed = np.asarray(
        records["completed_episode_mask"], dtype=np.bool_
    )
    correct = np.asarray(records["completed_correct_deliveries"])[completed]
    wrong = np.asarray(records["completed_wrong_deliveries"])[completed]
    if np.any(correct < 0) or np.any(wrong < 0):
        raise RuntimeError(
            f"Negative delivery count at training step {step}; "
            "the official event semantics must be checked."
        )


def _training_functions(
    *,
    reference_network: OfficialNetwork,
    reference_params: Mapping[str, Any],
    model: Any,
    partner_pool: FrozenPartnerPool,
    config: Any,
) -> tuple[RunnerFunctions, ModelFunctions, Any]:
    def online_step(
        params: Mapping[str, Any],
        carry: Any,
        observations: Any,
        episode_start: Any,
    ) -> tuple[Any, Any, Any, Any]:
        return reference_network.step(
            params, carry, observations, episode_start
        )

    def reference_step(
        carry: Any, observations: Any, episode_start: Any
    ) -> tuple[Any, Any, Any]:
        next_carry, unused_feature, logits, value = reference_network.step(
            reference_params, carry, observations, episode_start
        )
        del unused_feature
        return next_carry, logits, value

    def heads_apply(
        params: Mapping[str, Any],
        features: Any,
        previous_actions: Any,
        previous_team_rewards: Any,
        belief: Any,
    ) -> Mapping[str, Any]:
        return model.apply(
            {"params": params},
            features,
            previous_actions,
            previous_team_rewards,
            belief,
        )

    def response_apply(
        params: Mapping[str, Any],
        observations: Any,
        actions: Any,
        next_observations: Any,
        dones: Any,
        codebook_embeddings: Any,
        terminal_response: int,
    ) -> tuple[Any, Any, Any]:
        return encode_response_codes(
            model=model,
            params=params,
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            dones=dones,
            codebook_embeddings=codebook_embeddings,
            terminal_response=terminal_response,
        )

    base = RunnerFunctions(
        online_step=online_step,
        reference_step=reference_step,
        heads_apply=heads_apply,
        encode_response=response_apply,
        partner_step=lambda *unused: None,
        partner_observe=lambda *unused: None,
    )
    partner_initial, partner_step, partner_observe = partner_callbacks(
        base_functions=base,
        official_initial_carry=reference_network.initial_carry,
        static_partner_step=partner_pool.step,
        slot_count=config.model.slot_count,
        action_count=config.model.action_count,
        initial_temperature=config.kl.initial_temperature,
        gamma=config.training.gamma,
        terminal_response=config.model.response_count - 1,
    )
    runner_functions = base._replace(
        partner_step=partner_step,
        partner_observe=partner_observe,
    )
    return (
        runner_functions,
        ModelFunctions(
            heads=model,
            official_sequence=reference_network.sequence,
        ),
        partner_initial,
    )


def run_training(args: argparse.Namespace) -> None:
    import jax
    import jax.numpy as jnp
    import numpy as np

    config = load_config(args.config)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    reference_path, partner_paths = _load_unit(
        args.unit_manifest, args.outer_unit
    )
    official_config, reference_params = restore_official_checkpoint(
        reference_path
    )
    reference_network = OfficialNetwork(official_config)
    partner_pool = FrozenPartnerPool.from_checkpoints(partner_paths)
    environment = VectorEnvironment.create(config)
    model = build_model(
        hidden_dim=config.model.hidden_dim,
        slot_count=config.model.slot_count,
        action_count=config.model.action_count,
        response_count=config.model.response_count,
        prior_scale=config.model.prior_scale,
        action_embedding_dim=config.model.action_embedding_dim,
        log_standard_deviation_minimum=(
            config.model.log_standard_deviation_minimum
        ),
        log_standard_deviation_maximum=(
            config.model.log_standard_deviation_maximum
        ),
    )

    root_key = jax.random.PRNGKey(int(args.seed))
    root_key, reset_key, head_key, runner_key, state_key = jax.random.split(
        root_key, 5
    )
    unused_environment_state, observations = environment.reset(reset_key)
    del unused_environment_state
    ego_observations = observations[:, 0]
    starts = jnp.ones((environment.num_envs,), dtype=jnp.bool_)
    unused_carry, example_features, unused_logits, unused_value = (
        reference_network.step(
            reference_params,
            reference_network.initial_carry(environment.num_envs),
            ego_observations,
            starts,
        )
    )
    del unused_carry, unused_logits, unused_value
    head_params = initialize_heads(
        model,
        random_key=head_key,
        example_features=example_features,
        example_previous_actions=jnp.full(
            (environment.num_envs,),
            config.model.action_count,
            dtype=jnp.int32,
        ),
        example_previous_team_rewards=jnp.zeros(
            (environment.num_envs,), dtype=jnp.float32
        ),
        example_slot_log_belief=uniform_slot_log_belief(
            (environment.num_envs,), config.model.slot_count
        ),
        example_observations=ego_observations[None, ...],
    )
    params = {"official": reference_params, "heads": head_params}
    functions, model_functions, partner_initial = _training_functions(
        reference_network=reference_network,
        reference_params=reference_params,
        model=model,
        partner_pool=partner_pool,
        config=config,
    )
    runner_state = initialize_runner(
        environment=environment,
        official_initial_carry=reference_network.initial_carry,
        partner_initial_state=partner_initial,
        random_key=runner_key,
        slot_count=config.model.slot_count,
        action_count=config.model.action_count,
        partner_member_count=partner_pool.member_count,
        initial_temperature=config.kl.initial_temperature,
    )
    optimizers = make_optimizers(
        params,
        bellman_learning_rate=config.training.bellman_learning_rate,
        outcome_learning_rate=config.training.outcome_learning_rate,
        gradient_clip_norm=config.training.gradient_clip_norm,
    )
    state = TrainState(
        online_params=params,
        target_params=params,
        bellman_optimizer_state=optimizers.bellman_state,
        outcome_optimizer_state=optimizers.outcome_state,
        codebook=empty_codebook(
            code_count=config.model.response_count - 1,
            signature_dim=config.model.action_count,
        ),
        runner_state=runner_state,
        random_key=state_key,
        update_count=jnp.asarray(0, dtype=jnp.int64),
    )

    manager = orbax_manager(output / "checkpoints")
    existing_step = manager.latest_step()
    if existing_step is not None and not args.resume:
        raise RuntimeError(
            "The output already contains a checkpoint; use --resume or a new directory."
        )
    if args.resume:
        restored = restore_latest_checkpoint(manager)
        if restored is not None:
            unused_step, state = restored
            del unused_step

    collector = compiled_collector(
        length=config.environment.episode_steps,
        environment=environment,
        functions=functions,
        partner_member_count=partner_pool.member_count,
        deployment_mode="posterior_use",
        gamma=config.training.gamma,
        terminal_response=config.model.response_count - 1,
    )

    def assignment_step(
        target_params: Any,
        batch: Any,
        embeddings: Any,
        key: Any,
        temperature: Any,
        generic_temperature: Any,
        bootstrap_mask: Any,
    ) -> Any:
        return prepare_frozen_assignments(
            functions=model_functions,
            target_params=target_params,
            batch=batch,
            codebook_embeddings=embeddings,
            key=key,
            temperature=temperature,
            generic_temperature=generic_temperature,
            gamma=config.training.gamma,
            responsibility_temperature=(
                config.training.responsibility_temperature
            ),
            bootstrap_probability=config.training.bootstrap_probability,
            terminal_response=config.model.response_count - 1,
            bootstrap_mask=bootstrap_mask,
        )

    compiled_assignments = jax.jit(assignment_step)

    def update_step(
        online_params: Any,
        target_params: Any,
        bellman_state: Any,
        outcome_state: Any,
        batch: Any,
        assignments: Any,
        embeddings: Any,
        schedule: Any,
    ) -> Any:
        return apply_rollout_updates(
            functions=model_functions,
            params=online_params,
            target_params=target_params,
            optimizer_bundle=OptimizerBundle(
                bellman_optimizer=optimizers.bellman_optimizer,
                outcome_optimizer=optimizers.outcome_optimizer,
                bellman_state=bellman_state,
                outcome_state=outcome_state,
            ),
            batch=batch,
            assignments=assignments,
            codebook_embeddings=embeddings,
            schedule=schedule,
            polyak_coefficient=config.training.polyak_coefficient,
        )

    compiled_updates = jax.jit(update_step)
    decisions_path = output / "records" / "decisions"
    episodes_path = output / "records" / "episodes"
    metrics_path = output / "records" / "metrics"
    write_json(output / "resolved_config.json", config.to_mapping())

    rollout_steps = (
        config.environment.num_envs * config.environment.episode_steps
    )
    if config.training.environment_steps % rollout_steps:
        raise ValueError(
            "Training budget must be a whole number of complete vector rollouts."
        )

    while int(
        np.asarray(state.runner_state.effective_environment_steps)
    ) < (
        config.training.environment_steps
    ):
        (
            assignment_key,
            schedule_key,
            codebook_key,
            bootstrap_key,
            next_key,
        ) = jax.random.split(state.random_key, 5)
        runner_state, batch, rollout_records = collector(
            state=state.runner_state,
            params=state.online_params,
            codebook_embeddings=state.codebook.embeddings,
        )
        bootstrap_mask = sample_bootstrap_mask(
            bootstrap_key,
            environment_count=config.environment.num_envs,
            slot_count=config.model.slot_count,
            probability=config.training.bootstrap_probability,
        )
        provisional_assignments = compiled_assignments(
            state.target_params,
            batch,
            state.codebook.embeddings,
            assignment_key,
            jnp.exp(jnp.ravel(state.runner_state.ego_policy.log_temperature)[0]),
            jnp.exp(
                jnp.ravel(
                    state.runner_state.ego_policy.generic_log_temperature
                )[0]
            ),
            bootstrap_mask,
        )
        codebook = update_codebook_from_assignments(
            state.codebook,
            assignments=provisional_assignments,
            dones=batch.dones,
            key=codebook_key,
            decay=config.training.codebook_decay,
            replacement_after_rollouts=(
                config.training.code_replacement_rollouts
            ),
        )
        schedule = environment_minibatch_schedule(
            schedule_key,
            environment_count=config.environment.num_envs,
            minibatches_per_epoch=(
                config.training.minibatches_per_epoch
            ),
            update_epochs=config.training.update_epochs,
        )
        online_params = state.online_params
        target_params = state.target_params
        bellman_state = state.bellman_optimizer_state
        outcome_state = state.outcome_optimizer_state
        for epoch in range(config.training.update_epochs):
            assignments = compiled_assignments(
                target_params,
                batch,
                codebook.embeddings,
                jax.random.fold_in(assignment_key, epoch + 1),
                jnp.exp(
                    jnp.ravel(
                        state.runner_state.ego_policy.log_temperature
                    )[0]
                ),
                jnp.exp(
                    jnp.ravel(
                        state.runner_state.ego_policy.generic_log_temperature
                    )[0]
                ),
                bootstrap_mask,
            )
            update = compiled_updates(
                online_params,
                target_params,
                bellman_state,
                outcome_state,
                batch,
                assignments,
                codebook.embeddings,
                schedule[epoch],
            )
            online_params = update.params
            target_params = update.target_params
            bellman_state = update.bellman_optimizer_state
            outcome_state = update.outcome_optimizer_state
        posterior_kl, generic_kl = rollout_kl_means(batch)
        policy_state = update_policy_temperatures(
            runner_state.ego_policy,
            posterior_mean_kl=posterior_kl,
            generic_mean_kl=generic_kl,
            target_kl=config.kl.target_per_step,
            learning_rate=config.kl.dual_learning_rate,
            minimum_temperature=config.kl.minimum_temperature,
            maximum_temperature=config.kl.maximum_temperature,
        )
        runner_state = runner_state._replace(ego_policy=policy_state)
        state = TrainState(
            online_params=update.params,
            target_params=update.target_params,
            bellman_optimizer_state=update.bellman_optimizer_state,
            outcome_optimizer_state=update.outcome_optimizer_state,
            codebook=codebook,
            runner_state=runner_state,
            random_key=next_key,
            update_count=state.update_count + 1,
        )
        step = int(
            np.asarray(state.runner_state.effective_environment_steps)
        )
        _assert_finite(
            (state, batch, rollout_records), step=step
        )
        _assert_nonnegative_deliveries(rollout_records, step=step)
        update_number = int(np.asarray(state.update_count))
        write_jsonl(
            decisions_path / f"update_{update_number:08d}.jsonl",
            _decision_rows(
                batch, rollout_records, update_count=update_number
            ),
        )
        write_jsonl(
            episodes_path / f"update_{update_number:08d}.jsonl",
            _episode_rows(
                rollout_records, update_count=update_number
            ),
        )
        write_jsonl(
            metrics_path / f"update_{update_number:08d}.jsonl",
            (
                {
                    "effective_environment_steps": step,
                    "completed_episodes": int(
                        np.asarray(state.runner_state.completed_episodes)
                    ),
                    "update_count": int(np.asarray(state.update_count)),
                    "posterior_mean_kl": float(np.asarray(posterior_kl)),
                    "generic_mean_kl": float(np.asarray(generic_kl)),
                    "temperature": float(
                        np.exp(
                            np.asarray(
                                jnp.ravel(policy_state.log_temperature)[0]
                            )
                        )
                    ),
                    "generic_temperature": float(
                        np.exp(
                            np.asarray(
                                jnp.ravel(
                                    policy_state.generic_log_temperature
                                )[0]
                            )
                        )
                    ),
                    "losses": _host(update.metrics),
                },
            ),
        )
        if step % config.training.checkpoint_interval_environment_steps == 0:
            save_checkpoint(manager, step=step, state=state)

    final_step = int(
        np.asarray(state.runner_state.effective_environment_steps)
    )
    if manager.latest_step() != final_step:
        save_checkpoint(manager, step=final_step, state=state)
    write_run_metadata(
        output / "run_metadata.json",
        config=config,
        seed=int(args.seed),
        run_kind=args.run_kind,
        effective_environment_steps=final_step,
        update_count=int(np.asarray(state.update_count)),
        completed_episodes=int(
            np.asarray(state.runner_state.completed_episodes)
        ),
    )
    print(f"Complete decision records: {decisions_path}")
    print(f"Complete episode records: {episodes_path}")
    print(f"Complete metric records: {metrics_path}")


def run_upstream(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    output = Path(args.output).resolve()
    official = compose_official_config(
        config,
        algorithm=args.algorithm,
        seed=int(args.seed),
        output_directory=output,
    )
    result = train_upstream(
        official,
        seed=int(args.seed),
        checkpoint_progress=config.upstream.checkpoint_progress,
    )
    write_json(output / "resolved_config.json", config.to_mapping())
    write_json(output / "official_resolved_config.json", official)
    metric_paths = {}
    for name, values in result["metrics"].items():
        metric_paths[name] = [
            str(path)
            for path in write_array_chunks(
                output / "metrics",
                name=str(name),
                values=values,
                rows_per_chunk=1024,
            )
        ]
    write_json(
        output / "upstream_summary.json",
        {
            "algorithm": args.algorithm,
            "seed": int(args.seed),
            "effective_environment_steps": result[
                "effective_environment_steps"
            ],
            "update_count": result["update_count"],
            "checkpoint_paths": [
                str(path) for path in result["checkpoint_paths"]
            ],
            "metric_files": metric_paths,
        },
    )
    write_run_metadata(
        output / "run_metadata.json",
        config=config,
        seed=int(args.seed),
        run_kind=args.run_kind,
        effective_environment_steps=result["effective_environment_steps"],
        update_count=result["update_count"],
        completed_episodes=result["completed_episodes"],
    )
    print(f"Complete upstream metrics: {output / 'metrics'}")


@dataclass(frozen=True, slots=True)
class _Deployment:
    outer_unit_id: int
    network: OfficialNetwork
    reference_params: Any
    online_params: Any
    heads: Any
    head_params: Any
    codebook: Any
    log_temperature: Any
    generic_log_temperature: Any

    def functions(self) -> RunnerFunctions:
        def online_step(
            params: Mapping[str, Any],
            carry: Any,
            observations: Any,
            episode_start: Any,
        ) -> tuple[Any, Any, Any, Any]:
            return self.network.step(
                params, carry, observations, episode_start
            )

        def reference_step(
            carry: Any, observations: Any, episode_start: Any
        ) -> tuple[Any, Any, Any]:
            next_carry, unused_feature, logits, value = self.network.step(
                self.reference_params,
                carry,
                observations,
                episode_start,
            )
            del unused_feature
            return next_carry, logits, value

        def heads_apply(
            params: Mapping[str, Any],
            features: Any,
            previous_actions: Any,
            previous_team_rewards: Any,
            belief: Any,
        ) -> Mapping[str, Any]:
            return self.heads.apply(
                {"params": params},
                features,
                previous_actions,
                previous_team_rewards,
                belief,
            )

        def response_apply(
            params: Mapping[str, Any],
            observations: Any,
            actions: Any,
            next_observations: Any,
            dones: Any,
            embeddings: Any,
            terminal_response: int,
        ) -> tuple[Any, Any, Any]:
            return encode_response_codes(
                model=self.heads,
                params=params,
                observations=observations,
                actions=actions,
                next_observations=next_observations,
                dones=dones,
                codebook_embeddings=embeddings,
                terminal_response=terminal_response,
            )

        return RunnerFunctions(
            online_step=online_step,
            reference_step=reference_step,
            heads_apply=heads_apply,
            encode_response=response_apply,
            partner_step=lambda *unused: None,
            partner_observe=lambda *unused: None,
        )


def _load_deployment(
    entry: PopulationEntry, config: Any
) -> _Deployment:
    reference_config, reference_params = restore_official_checkpoint(
        entry.reference_checkpoint_path
    )
    manager = orbax_manager(entry.checkpoint_path, create=False)
    restored = restore_latest_checkpoint(manager)
    if restored is None:
        raise FileNotFoundError(
            f"No Orbax step in {entry.checkpoint_path}."
        )
    unused_step, train_state = restored
    del unused_step
    heads = build_model(
        hidden_dim=config.model.hidden_dim,
        slot_count=config.model.slot_count,
        action_count=config.model.action_count,
        response_count=config.model.response_count,
        prior_scale=config.model.prior_scale,
        action_embedding_dim=config.model.action_embedding_dim,
        log_standard_deviation_minimum=(
            config.model.log_standard_deviation_minimum
        ),
        log_standard_deviation_maximum=(
            config.model.log_standard_deviation_maximum
        ),
    )
    return _Deployment(
        outer_unit_id=entry.outer_unit_id,
        network=OfficialNetwork(reference_config),
        reference_params=reference_params,
        online_params=train_state.online_params["official"],
        heads=heads,
        head_params=train_state.online_params["heads"],
        codebook=train_state.codebook,
        log_temperature=train_state.runner_state.ego_policy.log_temperature,
        generic_log_temperature=(
            train_state.runner_state.ego_policy.generic_log_temperature
        ),
    )


def _reset_deployment_state(
    deployment: _Deployment,
    *,
    batch_size: int,
    config: Any,
) -> Any:
    import jax.numpy as jnp

    state = initialize_policy_state(
        official_initial_carry=deployment.network.initial_carry,
        batch_size=batch_size,
        slot_count=config.model.slot_count,
        action_count=config.model.action_count,
        initial_temperature=config.kl.initial_temperature,
    )
    return state._replace(
        log_temperature=jnp.full(
            (batch_size,),
            jnp.ravel(deployment.log_temperature)[0],
            dtype=jnp.float32,
        ),
        generic_log_temperature=jnp.full(
            (batch_size,),
            jnp.ravel(deployment.generic_log_temperature)[0],
            dtype=jnp.float32,
        ),
    )


def _update_deployment_after_transition(
    *,
    deployment: _Deployment,
    functions: RunnerFunctions,
    state: Any,
    output: Any,
    observations: Any,
    actions: Any,
    next_observations: Any,
    rewards: Any,
    dones: Any,
    deployment_mode: str,
    terminal_response: int,
    mask_response: Any = False,
) -> tuple[Any, Any]:
    import jax.numpy as jnp

    response_codes, unused_logits, unused_signatures = (
        functions.encode_response(
            deployment.head_params,
            observations[None, ...],
            actions[None, ...],
            next_observations[None, ...],
            dones[None, ...],
            deployment.codebook.embeddings,
            terminal_response,
        )
    )
    del unused_logits, unused_signatures
    response_codes = response_codes[0]
    posterior = slot_bayes_update(
        slot_log_belief=state.slot_log_belief,
        slot_response_probabilities=output.response_probabilities,
        action=actions,
        response_code=response_codes,
    )
    mask = jnp.asarray(mask_response, dtype=jnp.bool_)
    if mask.ndim == 0:
        mask = jnp.broadcast_to(mask, posterior.shape[:-1])
    posterior = jnp.where(
        mask[..., None], state.slot_log_belief, posterior
    )
    posterior = deployment_belief_after_response(
        mode=deployment_mode,
        current_log_belief=state.slot_log_belief,
        updated_log_belief=posterior,
    )
    uniform = uniform_slot_log_belief(
        posterior.shape[:-1], int(posterior.shape[-1])
    )
    return state._replace(
        slot_log_belief=jnp.where(dones[:, None], uniform, posterior),
        previous_action=jnp.where(
            dones,
            jnp.full(dones.shape, output.q_values.shape[-1], dtype=jnp.int32),
            actions,
        ),
        previous_team_reward=jnp.where(dones, 0.0, rewards),
        episode_start=dones,
    ), response_codes


def _pairing_batch(
    *,
    config: Any,
    left: _Deployment,
    right: _Deployment,
    pairing: Pairing,
    population_name: str,
    evaluation_seed: int,
) -> tuple[tuple[EpisodeRow, ...], Iterable[Mapping[str, Any]]]:
    import jax
    import jax.numpy as jnp
    import numpy as np

    episode_count = config.evaluation.episodes_per_pairing
    template = VectorEnvironment.create(config)
    environment = VectorEnvironment(
        environment=template.environment,
        num_envs=episode_count,
        episode_steps=config.environment.episode_steps,
    )
    seeds = np.asarray(
        [
            standard_episode_seed(
                evaluation_seed=evaluation_seed,
                layout=config.environment.layout,
                left_outer_unit_id=pairing.left_outer_unit_id,
                right_outer_unit_id=pairing.right_outer_unit_id,
                episode_index=index,
            )
            for index in range(episode_count)
        ],
        dtype=np.uint32,
    )
    root_keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray(seeds))
    reset_keys = jax.vmap(lambda key: jax.random.fold_in(key, 0))(root_keys)
    environment_state, observations = environment.reset_with_keys(reset_keys)
    left_state = _reset_deployment_state(
        left, batch_size=episode_count, config=config
    )
    right_state = _reset_deployment_state(
        right, batch_size=episode_count, config=config
    )
    left_functions = left.functions()
    right_functions = right.functions()

    def one_step(carry: tuple[Any, ...], step: Any) -> tuple[Any, tuple[Any, ...]]:
        (
            current_environment,
            current_observations,
            current_left,
            current_right,
        ) = carry
        left_keys = jax.vmap(
            lambda key: jax.random.fold_in(key, 1 + 3 * step)
        )(root_keys)
        right_keys = jax.vmap(
            lambda key: jax.random.fold_in(key, 2 + 3 * step)
        )(root_keys)
        environment_keys = jax.vmap(
            lambda key: jax.random.fold_in(key, 3 + 3 * step)
        )(root_keys)
        stepped_left, left_action, left_output, left_record, unused_left_generic = (
            policy_action(
                functions=left_functions,
                params={"official": left.online_params, "heads": left.head_params},
                policy_state=current_left,
                observations=current_observations[:, 0],
                key=left_keys,
                deployment_mode=pairing.deployment_mode,
                gamma=config.training.gamma,
            )
        )
        stepped_right, right_action, right_output, right_record, unused_right_generic = (
            policy_action(
                functions=right_functions,
                params={"official": right.online_params, "heads": right.head_params},
                policy_state=current_right,
                observations=current_observations[:, 1],
                key=right_keys,
                deployment_mode=pairing.deployment_mode,
                gamma=config.training.gamma,
            )
        )
        del unused_left_generic, unused_right_generic
        actions = jnp.stack((left_action, right_action), axis=-1)
        (
            next_environment,
            next_observations,
            rewards,
            dones,
            info,
        ) = environment.step_with_keys(
            current_environment, actions, environment_keys
        )
        terminal = info["terminal_observations"]
        observation_mask = dones.reshape(
            dones.shape + (1,) * (next_observations.ndim - 2)
        )
        response_next = jnp.where(
            observation_mask[:, None, ...], terminal, next_observations
        )
        next_left, left_response = _update_deployment_after_transition(
            deployment=left,
            functions=left_functions,
            state=stepped_left,
            output=left_output,
            observations=current_observations[:, 0],
            actions=left_action,
            next_observations=response_next[:, 0],
            rewards=rewards,
            dones=dones,
            deployment_mode=pairing.deployment_mode,
            terminal_response=config.model.response_count - 1,
        )
        next_right, right_response = _update_deployment_after_transition(
            deployment=right,
            functions=right_functions,
            state=stepped_right,
            output=right_output,
            observations=current_observations[:, 1],
            actions=right_action,
            next_observations=response_next[:, 1],
            rewards=rewards,
            dones=dones,
            deployment_mode=pairing.deployment_mode,
            terminal_response=config.model.response_count - 1,
        )
        return (
            next_environment,
            next_observations,
            next_left,
            next_right,
        ), (
            rewards,
            info["correct_delivery"],
            info["wrong_delivery"],
            info["indicator_activation"],
            left_record,
            right_record,
            left_response,
            right_response,
        )

    unused_final, recorded = jax.lax.scan(
        one_step,
        (environment_state, observations, left_state, right_state),
        jnp.arange(config.environment.episode_steps),
    )
    del unused_final
    (
        rewards,
        correct,
        wrong,
        indicator,
        left_records,
        right_records,
        left_codes,
        right_codes,
    ) = recorded
    trigger_left = left_records.predicted_net_effect > policy_effect_trigger_tolerance(
        left_records.j_use, left_records.j_mask
    )
    trigger_right = right_records.predicted_net_effect > policy_effect_trigger_tolerance(
        right_records.j_use, right_records.j_mask
    )
    response_count = config.model.response_count

    def count_codes(left_values: Any, right_values: Any) -> Any:
        return jnp.bincount(
            jnp.concatenate((left_values, right_values)),
            length=response_count,
        )

    counts = jax.vmap(count_codes, in_axes=(1, 1))(left_codes, right_codes)
    raw_returns = np.asarray(jnp.sum(rewards, axis=0))
    correct_totals = np.asarray(jnp.sum(correct, axis=0))
    wrong_totals = np.asarray(jnp.sum(wrong, axis=0))
    indicator_totals = np.asarray(jnp.sum(indicator, axis=0))
    cumulative_kl = np.asarray(
        jnp.sum(left_records.kl_divergence + right_records.kl_divergence, axis=0)
    )
    deviations = np.asarray(
        jnp.sum(
            (
                left_records.action
                != left_records.reference_greedy_action
            ).astype(jnp.int32)
            + (
                right_records.action
                != right_records.reference_greedy_action
            ).astype(jnp.int32),
            axis=0,
        )
    )
    class_means = np.asarray(
        jnp.mean(
            0.5
            * (
                left_records.value_class_count
                + right_records.value_class_count
            ),
            axis=0,
        )
    )
    entropy_means = np.asarray(
        jnp.mean(
            0.5
            * (left_records.belief_entropy + right_records.belief_entropy),
            axis=0,
        )
    )
    trigger_counts = np.asarray(
        jnp.sum(
            trigger_left.astype(jnp.int32) + trigger_right.astype(jnp.int32),
            axis=0,
        )
    )
    counts_host = np.asarray(counts)
    rows = tuple(
        EpisodeRow(
            population=population_name,
            layout=config.environment.layout,
            deployment_mode=pairing.deployment_mode,
            split=pairing.split,
            pairing_id=pairing.pairing_id,
            left_outer_unit_id=pairing.left_outer_unit_id,
            right_outer_unit_id=pairing.right_outer_unit_id,
            episode_index=index,
            episode_seed=int(seeds[index]),
            environment_steps=config.environment.episode_steps,
            raw_return=float(raw_returns[index]),
            correct_delivery_count=int(correct_totals[index]),
            wrong_delivery_count=int(wrong_totals[index]),
            indicator_activation_count=int(indicator_totals[index]),
            positive_predicted_response_effect_count=int(
                trigger_counts[index]
            ),
            cumulative_kl=float(cumulative_kl[index]),
            reference_action_deviation_count=int(deviations[index]),
            mean_value_class_count=float(class_means[index]),
            mean_belief_entropy=float(entropy_means[index]),
            response_code_counts=tuple(
                int(value) for value in counts_host[index]
            ),
        )
        for index in range(episode_count)
    )

    host_records = {
        "left": jax.tree_util.tree_map(lambda value: np.asarray(value), left_records),
        "right": jax.tree_util.tree_map(lambda value: np.asarray(value), right_records),
    }
    host_codes = {
        "left": np.asarray(left_codes),
        "right": np.asarray(right_codes),
    }

    def decisions() -> Iterable[Mapping[str, Any]]:
        for step in range(config.environment.episode_steps):
            for episode_index in range(episode_count):
                for side in ("left", "right"):
                    record = host_records[side]
                    yield {
                        "deployment_mode": pairing.deployment_mode,
                        "pairing_id": pairing.pairing_id,
                        "episode_index": episode_index,
                        "episode_seed": int(seeds[episode_index]),
                        "step": step,
                        "side": side,
                        "action": int(record.action[step, episode_index]),
                        "reference_logits": record.reference_logits[
                            step, episode_index
                        ].tolist(),
                        "execution_logits": record.execution_logits[
                            step, episode_index
                        ].tolist(),
                        "mask_execution_logits": record.mask_execution_logits[
                            step, episode_index
                        ].tolist(),
                        "j_use": record.j_use[step, episode_index].tolist(),
                        "j_mask": record.j_mask[step, episode_index].tolist(),
                        "per_action_response_value": (
                            record.per_action_response_value[
                                step, episode_index
                            ].tolist()
                        ),
                        "per_action_net_value": record.per_action_net_value[
                            step, episode_index
                        ].tolist(),
                        "response_code": int(
                            host_codes[side][step, episode_index]
                        ),
                        "belief_entropy": float(
                            record.belief_entropy[step, episode_index]
                        ),
                        "value_class_count": int(
                            record.value_class_count[step, episode_index]
                        ),
                        "kl_divergence": float(
                            record.kl_divergence[step, episode_index]
                        ),
                        "reference_greedy_action": int(
                            record.reference_greedy_action[
                                step, episode_index
                            ]
                        ),
                        "predicted_response_effect": float(
                            record.predicted_response_effect[
                                step, episode_index
                            ]
                        ),
                        "predicted_policy_cost": float(
                            record.predicted_policy_cost[
                                step, episode_index
                            ]
                        ),
                        "predicted_net_effect": float(
                            record.predicted_net_effect[
                                step, episode_index
                            ]
                        ),
                        "predicted_regularized_net_effect": float(
                            record.predicted_regularized_net_effect[
                                step, episode_index
                            ]
                        ),
                        "predicted_policy_total_variation": float(
                            record.predicted_policy_total_variation[
                                step, episode_index
                            ]
                        ),
                        "executed_action_response_value": float(
                            record.executed_action_response_value[
                                step, episode_index
                            ]
                        ),
                        "executed_action_net_value": float(
                            record.executed_action_net_value[
                                step, episode_index
                            ]
                        ),
                        "maximum_action_net_value": float(
                            record.maximum_action_net_value[
                                step, episode_index
                            ]
                        ),
                    }

    return rows, decisions()


def _evaluate_standard(
    *,
    config: Any,
    population: Any,
    output: Path,
    evaluation_seed: int,
    resume: bool,
) -> tuple[int, int]:
    deployments = {
        entry.outer_unit_id: _load_deployment(entry, config)
        for entry in population.entries
    }
    row_paths = []
    for pairing in standard_pairings(config.evaluation.deployment_modes):
        pairing_root = (
            output
            / pairing.deployment_mode
            / pairing.pairing_id
        )
        row_path = pairing_root / "episodes.parquet"
        decision_path = pairing_root / "decisions.jsonl"
        if resume and row_path.is_file() and decision_path.is_file():
            row_count = len(read_parquet(row_path))
            decision_count = _jsonl_row_count(decision_path)
            expected_decisions = (
                2
                * config.environment.episode_steps
                * config.evaluation.episodes_per_pairing
            )
            if (
                row_count != config.evaluation.episodes_per_pairing
                or decision_count != expected_decisions
            ):
                raise RuntimeError(
                    f"Incomplete evaluation output exists in {pairing_root}."
                )
            row_paths.append(row_path)
            continue
        if row_path.exists() or decision_path.exists():
            raise RuntimeError(
                f"Incomplete evaluation output exists in {pairing_root}."
            )
        rows, decisions = _pairing_batch(
            config=config,
            left=deployments[pairing.left_outer_unit_id],
            right=deployments[pairing.right_outer_unit_id],
            pairing=pairing,
            population_name=population.name,
            evaluation_seed=evaluation_seed,
        )
        write_parquet(row_path, [row.to_mapping() for row in rows])
        write_jsonl(decision_path, decisions)
        row_paths.append(row_path)

    all_rows = [
        EpisodeRow(
            **{
                **row,
                "response_code_counts": tuple(row["response_code_counts"]),
            }
        )
        for path in row_paths
        for row in read_parquet(path)
    ]
    validate_standard_rows(
        all_rows,
        deployment_modes=config.evaluation.deployment_modes,
        episodes_per_pairing=config.evaluation.episodes_per_pairing,
    )
    write_json(
        output / "summary.json", summarize_standard_rows(all_rows)
    )
    return len(all_rows), sum(row.environment_steps for row in all_rows)


class _DeploymentStep(NamedTuple):
    state: Any
    action: Any
    output: Any
    record: Any


class _ContrastBranch(NamedTuple):
    environment_state: Any
    observations: Any
    left_state: Any
    right_state: Any
    raw_return: Any
    correct_delivery_count: Any
    wrong_delivery_count: Any
    indicator_activation_count: Any
    last_left_action: Any
    last_left_response: Any
    last_reward: Any


class _ContrastState(NamedTuple):
    a1: _ContrastBranch
    a2_mask: _ContrastBranch
    a2_use: _ContrastBranch
    triggered: Any
    trigger_step: Any
    trigger_tolerance: Any
    predicted_response_effect: Any
    predicted_policy_cost: Any
    predicted_net_effect: Any
    predicted_regularized_net_effect: Any
    predicted_policy_total_variation: Any
    maximum_action_net_value: Any
    executed_action_net_value: Any
    executed_action_response_value: Any
    executed_action: Any
    maximum_net_action: Any
    post_response_belief_l1: Any
    first_left_action_difference_step: Any
    first_observation_difference_step: Any
    first_response_code_difference_step: Any
    first_reward_difference_step: Any
    left_action_difference_count: Any
    observation_difference_count: Any
    response_code_difference_count: Any
    reward_difference_count: Any


def _deployment_step(
    *,
    deployment: _Deployment,
    state: Any,
    observations: Any,
    keys: Any,
    config: Any,
) -> _DeploymentStep:
    next_state, action, output, record, unused_generic = policy_action(
        functions=deployment.functions(),
        params={
            "official": deployment.online_params,
            "heads": deployment.head_params,
        },
        policy_state=state,
        observations=observations,
        key=keys,
        deployment_mode="posterior_use",
        gamma=config.training.gamma,
    )
    del unused_generic
    return _DeploymentStep(next_state, action, output, record)


def _masked_action_step(step: _DeploymentStep, keys: Any) -> _DeploymentStep:
    import jax

    action = jax.vmap(
        lambda key, logits: jax.random.categorical(key, logits)
    )(keys, step.output.mask_execution_logits)
    return step._replace(action=action)


def _advance_contrast_branch(
    *,
    branch: _ContrastBranch,
    left: _Deployment,
    right: _Deployment,
    left_step: _DeploymentStep,
    right_step: _DeploymentStep,
    environment: VectorEnvironment,
    environment_keys: Any,
    config: Any,
    mask_left_response: Any,
) -> _ContrastBranch:
    import jax.numpy as jnp

    actions = jnp.stack((left_step.action, right_step.action), axis=-1)
    next_environment, next_observations, rewards, dones, info = (
        environment.step_with_keys(
            branch.environment_state, actions, environment_keys
        )
    )
    terminal = info["terminal_observations"]
    mask = dones.reshape(
        dones.shape + (1,) * (next_observations[:, 0].ndim - 1)
    )
    left_next_observation = jnp.where(
        mask, terminal[:, 0], next_observations[:, 0]
    )
    right_next_observation = jnp.where(
        mask, terminal[:, 1], next_observations[:, 1]
    )
    left_state, left_response = _update_deployment_after_transition(
        deployment=left,
        functions=left.functions(),
        state=left_step.state,
        output=left_step.output,
        observations=branch.observations[:, 0],
        actions=left_step.action,
        next_observations=left_next_observation,
        rewards=rewards,
        dones=dones,
        deployment_mode="posterior_use",
        terminal_response=config.model.response_count - 1,
        mask_response=mask_left_response,
    )
    right_state, unused_right_response = _update_deployment_after_transition(
        deployment=right,
        functions=right.functions(),
        state=right_step.state,
        output=right_step.output,
        observations=branch.observations[:, 1],
        actions=right_step.action,
        next_observations=right_next_observation,
        rewards=rewards,
        dones=dones,
        deployment_mode="posterior_use",
        terminal_response=config.model.response_count - 1,
    )
    del unused_right_response
    return _ContrastBranch(
        environment_state=next_environment,
        observations=next_observations,
        left_state=left_state,
        right_state=right_state,
        raw_return=branch.raw_return + rewards,
        correct_delivery_count=(
            branch.correct_delivery_count + info["correct_delivery"]
        ),
        wrong_delivery_count=(
            branch.wrong_delivery_count + info["wrong_delivery"]
        ),
        indicator_activation_count=(
            branch.indicator_activation_count + info["indicator_activation"]
        ),
        last_left_action=left_step.action,
        last_left_response=left_response,
        last_reward=rewards,
    )


def _contrast_pairing_batch(
    *,
    config: Any,
    left: _Deployment,
    right: _Deployment,
    pairing: Pairing,
    evaluation_seed: int,
) -> tuple[ResponseContrastRow, ...]:
    import jax
    import jax.numpy as jnp
    import numpy as np

    count = config.evaluation.episodes_per_pairing
    template = VectorEnvironment.create(config)
    environment = VectorEnvironment(
        environment=template.environment,
        num_envs=count,
        episode_steps=config.environment.episode_steps,
    )
    seeds = np.asarray(
        [
            standard_episode_seed(
                evaluation_seed=evaluation_seed,
                layout=config.environment.layout,
                left_outer_unit_id=pairing.left_outer_unit_id,
                right_outer_unit_id=pairing.right_outer_unit_id,
                episode_index=index,
            )
            for index in range(count)
        ],
        dtype=np.uint32,
    )
    root_keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray(seeds))
    reset_keys = jax.vmap(lambda key: jax.random.fold_in(key, 0))(root_keys)
    environment_state, observations = environment.reset_with_keys(reset_keys)
    minus_one = jnp.full((count,), -1, dtype=jnp.int32)
    zeros_int = jnp.zeros((count,), dtype=jnp.int32)
    zeros_float = jnp.zeros((count,), dtype=jnp.float32)
    initial_branch = _ContrastBranch(
        environment_state=environment_state,
        observations=observations,
        left_state=_reset_deployment_state(
            left, batch_size=count, config=config
        ),
        right_state=_reset_deployment_state(
            right, batch_size=count, config=config
        ),
        raw_return=zeros_float,
        correct_delivery_count=zeros_int,
        wrong_delivery_count=zeros_int,
        indicator_activation_count=zeros_int,
        last_left_action=minus_one,
        last_left_response=minus_one,
        last_reward=zeros_float,
    )
    initial = _ContrastState(
        a1=initial_branch,
        a2_mask=initial_branch,
        a2_use=initial_branch,
        triggered=jnp.zeros((count,), dtype=jnp.bool_),
        trigger_step=minus_one,
        trigger_tolerance=zeros_float,
        predicted_response_effect=zeros_float,
        predicted_policy_cost=zeros_float,
        predicted_net_effect=zeros_float,
        predicted_regularized_net_effect=zeros_float,
        predicted_policy_total_variation=zeros_float,
        maximum_action_net_value=zeros_float,
        executed_action_net_value=zeros_float,
        executed_action_response_value=zeros_float,
        executed_action=minus_one,
        maximum_net_action=minus_one,
        post_response_belief_l1=zeros_float,
        first_left_action_difference_step=minus_one,
        first_observation_difference_step=minus_one,
        first_response_code_difference_step=minus_one,
        first_reward_difference_step=minus_one,
        left_action_difference_count=zeros_int,
        observation_difference_count=zeros_int,
        response_code_difference_count=zeros_int,
        reward_difference_count=zeros_int,
    )

    def one_step(current: _ContrastState, step: Any) -> tuple[Any, None]:
        left_keys = jax.vmap(
            lambda key: jax.random.fold_in(key, 1 + 3 * step)
        )(root_keys)
        right_keys = jax.vmap(
            lambda key: jax.random.fold_in(key, 2 + 3 * step)
        )(root_keys)
        environment_keys = jax.vmap(
            lambda key: jax.random.fold_in(key, 3 + 3 * step)
        )(root_keys)

        a1_left = _deployment_step(
            deployment=left,
            state=current.a1.left_state,
            observations=current.a1.observations[:, 0],
            keys=left_keys,
            config=config,
        )
        a1_right = _deployment_step(
            deployment=right,
            state=current.a1.right_state,
            observations=current.a1.observations[:, 1],
            keys=right_keys,
            config=config,
        )
        mask_left = _deployment_step(
            deployment=left,
            state=current.a2_mask.left_state,
            observations=current.a2_mask.observations[:, 0],
            keys=left_keys,
            config=config,
        )
        mask_right = _deployment_step(
            deployment=right,
            state=current.a2_mask.right_state,
            observations=current.a2_mask.observations[:, 1],
            keys=right_keys,
            config=config,
        )
        use_left = _deployment_step(
            deployment=left,
            state=current.a2_use.left_state,
            observations=current.a2_use.observations[:, 0],
            keys=left_keys,
            config=config,
        )
        use_right = _deployment_step(
            deployment=right,
            state=current.a2_use.right_state,
            observations=current.a2_use.observations[:, 1],
            keys=right_keys,
            config=config,
        )
        tolerance = policy_effect_trigger_tolerance(
            use_left.output.j_use, use_left.output.j_mask
        )
        newly_triggered = (~current.triggered) & (
            use_left.output.predicted_net_effect > tolerance
        )
        masked_reference = _masked_action_step(use_left, left_keys)
        a1_left = a1_left._replace(
            action=jnp.where(
                newly_triggered,
                masked_reference.action,
                a1_left.action,
            )
        )
        mask_left = mask_left._replace(
            action=jnp.where(
                newly_triggered, use_left.action, mask_left.action
            )
        )
        next_a1 = _advance_contrast_branch(
            branch=current.a1,
            left=left,
            right=right,
            left_step=a1_left,
            right_step=a1_right,
            environment=environment,
            environment_keys=environment_keys,
            config=config,
            mask_left_response=newly_triggered,
        )
        next_mask = _advance_contrast_branch(
            branch=current.a2_mask,
            left=left,
            right=right,
            left_step=mask_left,
            right_step=mask_right,
            environment=environment,
            environment_keys=environment_keys,
            config=config,
            mask_left_response=newly_triggered,
        )
        next_use = _advance_contrast_branch(
            branch=current.a2_use,
            left=left,
            right=right,
            left_step=use_left,
            right_step=use_right,
            environment=environment,
            environment_keys=environment_keys,
            config=config,
            mask_left_response=False,
        )
        active = current.triggered | newly_triggered
        action_difference = active & (
            next_mask.last_left_action != next_use.last_left_action
        )
        observation_difference = active & jnp.any(
            next_mask.observations != next_use.observations,
            axis=tuple(range(1, next_use.observations.ndim)),
        )
        response_difference = active & (
            next_mask.last_left_response != next_use.last_left_response
        )
        reward_difference = active & (
            next_mask.last_reward != next_use.last_reward
        )

        def capture(old: Any, new: Any) -> Any:
            return jnp.where(newly_triggered, new, old)

        def first(old: Any, condition: Any) -> Any:
            return jnp.where((old < 0) & condition, step, old)

        use_belief = jax.nn.softmax(
            next_use.left_state.slot_log_belief, axis=-1
        )
        mask_belief = jax.nn.softmax(
            next_mask.left_state.slot_log_belief, axis=-1
        )
        return _ContrastState(
            a1=next_a1,
            a2_mask=next_mask,
            a2_use=next_use,
            triggered=active,
            trigger_step=capture(current.trigger_step, step),
            trigger_tolerance=capture(
                current.trigger_tolerance, tolerance
            ),
            predicted_response_effect=capture(
                current.predicted_response_effect,
                use_left.output.predicted_response_effect,
            ),
            predicted_policy_cost=capture(
                current.predicted_policy_cost,
                use_left.output.predicted_policy_cost,
            ),
            predicted_net_effect=capture(
                current.predicted_net_effect,
                use_left.output.predicted_net_effect,
            ),
            predicted_regularized_net_effect=capture(
                current.predicted_regularized_net_effect,
                use_left.output.predicted_regularized_net_effect,
            ),
            predicted_policy_total_variation=capture(
                current.predicted_policy_total_variation,
                use_left.output.predicted_policy_total_variation,
            ),
            maximum_action_net_value=capture(
                current.maximum_action_net_value,
                use_left.record.maximum_action_net_value,
            ),
            executed_action_net_value=capture(
                current.executed_action_net_value,
                use_left.record.executed_action_net_value,
            ),
            executed_action_response_value=capture(
                current.executed_action_response_value,
                use_left.record.executed_action_response_value,
            ),
            executed_action=capture(
                current.executed_action, use_left.action
            ),
            maximum_net_action=capture(
                current.maximum_net_action,
                jnp.argmax(
                    use_left.output.per_action_net_value, axis=-1
                ),
            ),
            post_response_belief_l1=capture(
                current.post_response_belief_l1,
                jnp.sum(jnp.abs(use_belief - mask_belief), axis=-1),
            ),
            first_left_action_difference_step=first(
                current.first_left_action_difference_step,
                action_difference,
            ),
            first_observation_difference_step=first(
                current.first_observation_difference_step,
                observation_difference,
            ),
            first_response_code_difference_step=first(
                current.first_response_code_difference_step,
                response_difference,
            ),
            first_reward_difference_step=first(
                current.first_reward_difference_step,
                reward_difference,
            ),
            left_action_difference_count=(
                current.left_action_difference_count
                + action_difference.astype(jnp.int32)
            ),
            observation_difference_count=(
                current.observation_difference_count
                + observation_difference.astype(jnp.int32)
            ),
            response_code_difference_count=(
                current.response_code_difference_count
                + response_difference.astype(jnp.int32)
            ),
            reward_difference_count=(
                current.reward_difference_count
                + reward_difference.astype(jnp.int32)
            ),
        ), None

    final, unused = jax.lax.scan(
        one_step,
        initial,
        jnp.arange(config.environment.episode_steps),
    )
    del unused

    def optional_step(values: Any, lane: int) -> int | None:
        value = int(np.asarray(values[lane]))
        return None if value < 0 else value

    rows = []
    for lane in range(count):
        triggered = bool(np.asarray(final.triggered[lane]))

        def optional_float(values: Any) -> float | None:
            return float(np.asarray(values[lane])) if triggered else None

        def optional_int(values: Any) -> int | None:
            return int(np.asarray(values[lane])) if triggered else None

        rows.append(
            ResponseContrastRow(
                pairing_id=pairing.pairing_id,
                episode_index=lane,
                episode_seed=int(seeds[lane]),
                triggered=triggered,
                trigger_step=(
                    int(np.asarray(final.trigger_step[lane]))
                    if triggered
                    else None
                ),
                trigger_tolerance=optional_float(final.trigger_tolerance),
                predicted_response_effect=optional_float(
                    final.predicted_response_effect
                ),
                predicted_policy_cost=optional_float(
                    final.predicted_policy_cost
                ),
                predicted_net_effect=optional_float(
                    final.predicted_net_effect
                ),
                predicted_regularized_net_effect=optional_float(
                    final.predicted_regularized_net_effect
                ),
                predicted_policy_total_variation=optional_float(
                    final.predicted_policy_total_variation
                ),
                maximum_action_net_value=optional_float(
                    final.maximum_action_net_value
                ),
                executed_action_net_value=optional_float(
                    final.executed_action_net_value
                ),
                executed_action_response_value=optional_float(
                    final.executed_action_response_value
                ),
                executed_action=optional_int(final.executed_action),
                maximum_net_action=optional_int(final.maximum_net_action),
                post_response_belief_l1=optional_float(
                    final.post_response_belief_l1
                ),
                first_left_action_difference_step=(
                    optional_step(
                        final.first_left_action_difference_step, lane
                    )
                    if triggered
                    else None
                ),
                first_observation_difference_step=(
                    optional_step(
                        final.first_observation_difference_step, lane
                    )
                    if triggered
                    else None
                ),
                first_response_code_difference_step=(
                    optional_step(
                        final.first_response_code_difference_step, lane
                    )
                    if triggered
                    else None
                ),
                first_reward_difference_step=(
                    optional_step(final.first_reward_difference_step, lane)
                    if triggered
                    else None
                ),
                left_action_difference_count=optional_int(
                    final.left_action_difference_count
                ),
                observation_difference_count=optional_int(
                    final.observation_difference_count
                ),
                response_code_difference_count=optional_int(
                    final.response_code_difference_count
                ),
                reward_difference_count=optional_int(
                    final.reward_difference_count
                ),
                environment_steps=3 * config.environment.episode_steps,
                a1_raw_return=float(np.asarray(final.a1.raw_return[lane])),
                a1_correct_delivery_count=int(
                    np.asarray(final.a1.correct_delivery_count[lane])
                ),
                a1_wrong_delivery_count=int(
                    np.asarray(final.a1.wrong_delivery_count[lane])
                ),
                a1_indicator_activation_count=int(
                    np.asarray(final.a1.indicator_activation_count[lane])
                ),
                a2_mask_raw_return=float(
                    np.asarray(final.a2_mask.raw_return[lane])
                ),
                a2_mask_correct_delivery_count=int(
                    np.asarray(final.a2_mask.correct_delivery_count[lane])
                ),
                a2_mask_wrong_delivery_count=int(
                    np.asarray(final.a2_mask.wrong_delivery_count[lane])
                ),
                a2_mask_indicator_activation_count=int(
                    np.asarray(final.a2_mask.indicator_activation_count[lane])
                ),
                a2_use_raw_return=float(
                    np.asarray(final.a2_use.raw_return[lane])
                ),
                a2_use_correct_delivery_count=int(
                    np.asarray(final.a2_use.correct_delivery_count[lane])
                ),
                a2_use_wrong_delivery_count=int(
                    np.asarray(final.a2_use.wrong_delivery_count[lane])
                ),
                a2_use_indicator_activation_count=int(
                    np.asarray(final.a2_use.indicator_activation_count[lane])
                ),
            )
        )
    return tuple(rows)


def _evaluate_response_contrast(
    *,
    config: Any,
    population: Any,
    output: Path,
    evaluation_seed: int,
    resume: bool,
) -> tuple[int, int]:
    deployments = {
        entry.outer_unit_id: _load_deployment(entry, config)
        for entry in population.entries
    }
    paths = []
    for left_id in range(10):
        for right_id in range(10):
            if left_id == right_id:
                continue
            pairing = Pairing(
                deployment_mode="posterior_use",
                split="xp",
                left_outer_unit_id=left_id,
                right_outer_unit_id=right_id,
            )
            path = output / pairing.pairing_id / "branches.parquet"
            if resume and path.is_file():
                if len(read_parquet(path)) != config.evaluation.episodes_per_pairing:
                    raise RuntimeError(
                        f"Incomplete response contrast output exists: {path}"
                    )
                paths.append(path)
                continue
            if path.exists():
                raise RuntimeError(
                    f"Response contrast output already exists: {path}"
                )
            rows = _contrast_pairing_batch(
                config=config,
                left=deployments[left_id],
                right=deployments[right_id],
                pairing=pairing,
                evaluation_seed=evaluation_seed,
            )
            write_parquet(path, [row.to_mapping() for row in rows])
            paths.append(path)
    rows = [
        ResponseContrastRow(**row)
        for path in paths
        for row in read_parquet(path)
    ]
    validate_response_contrast_rows(
        rows,
        evaluation_seed=evaluation_seed,
        layout=config.environment.layout,
        episodes_per_pairing=config.evaluation.episodes_per_pairing,
    )
    write_json(
        output / "summary.json", summarize_response_contrast(rows)
    )
    return len(rows), sum(row.environment_steps for row in rows)


def run_evaluation(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    population = load_population(args.manifest)
    if population.layout != config.environment.layout:
        raise ValueError("Population and configuration layouts differ.")
    evaluation_kind = population.evaluation_kind
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "resolved_config.json", config.to_mapping())
    if evaluation_kind == "standard_matrix":
        completed_episodes, effective_environment_steps = _evaluate_standard(
            config=config,
            population=population,
            output=output,
            evaluation_seed=int(args.seed),
            resume=bool(args.resume),
        )
    elif evaluation_kind == "response_contrast":
        matched_episode_count, effective_environment_steps = (
            _evaluate_response_contrast(
                config=config,
                population=population,
                output=output,
                evaluation_seed=int(args.seed),
                resume=bool(args.resume),
            )
        )
        completed_episodes = 3 * matched_episode_count
    else:
        raise ValueError(f"Unknown evaluation_kind: {evaluation_kind}")
    write_run_metadata(
        output / "run_metadata.json",
        config=config,
        seed=int(args.seed),
        run_kind=args.run_kind,
        effective_environment_steps=effective_environment_steps,
        update_count=0,
        completed_episodes=completed_episodes,
    )
    print(f"Complete evaluation rows: {output}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m experiments.overcooked_v2.path_c")
    commands = parser.add_subparsers(dest="command", required=True)

    upstream = commands.add_parser("upstream")
    upstream.add_argument("--config", required=True)
    upstream.add_argument("--algorithm", choices=("rnn-sp", "rnn-op"), required=True)
    upstream.add_argument("--seed", type=int, required=True)
    upstream.add_argument("--run-kind", choices=("development", "formal"), required=True)
    upstream.add_argument("--output", required=True)
    upstream.set_defaults(function=run_upstream)

    train = commands.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--unit-manifest", required=True)
    train.add_argument("--outer-unit", type=int, choices=range(10), required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--run-kind", choices=("development", "formal"), required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--resume", action="store_true")
    train.set_defaults(function=run_training)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--seed", type=int, required=True)
    evaluate.add_argument("--run-kind", choices=("development", "formal"), required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--resume", action="store_true")
    evaluate.set_defaults(function=run_evaluation)
    return parser


def main() -> None:
    args = _parser().parse_args()
    output = Path(args.output).resolve()
    with CompleteConsoleLog(output / "logs") as console_log:
        print(f"Complete standard output: {console_log.stdout_path}")
        print(f"Complete standard error: {console_log.stderr_path}")
        args.function(args)


if __name__ == "__main__":
    main()
