"""Upstream and Path C V4.4 training workflows for OvercookedV2."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable, Mapping

from experiments.overcooked_v2.official_adapter import (
    FrozenPartnerPool,
    OfficialNetwork,
    VectorEnvironment,
    restore_official_checkpoint,
)
from src.path_c.experiment import load_config, load_training_unit_manifest
from src.path_c.method import empty_codebook, uniform_slot_log_belief
from src.path_c.model import build_model, encode_response_codes, initialize_heads
from src.path_c.runner import (
    RunnerFunctions,
    compiled_collector,
    initialize_runner,
    partner_callbacks,
)
from src.path_c.storage import (
    ensure_run_identity,
    orbax_manager,
    restore_latest_checkpoint,
    save_checkpoint,
    training_identity,
    write_json,
    write_jsonl,
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
    recompute_policy_scores,
    rollout_kl_means,
    sample_bootstrap_mask,
    solve_policy_temperatures,
    update_codebook_from_assignments,
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
                "behavior_logits": arrays["behavior_logits"][
                    time_index, environment_index
                ].tolist(),
                "exploration_logits": arrays["exploration_logits"][
                    time_index, environment_index
                ].tolist(),
                "behavior_slot": int(
                    arrays["behavior_slots"][time_index, environment_index]
                ),
                "behavior_estimator": int(
                    arrays["behavior_estimators"][time_index, environment_index]
                ),
                "behavior_action_probability": float(
                    arrays["behavior_action_probabilities"][
                        time_index, environment_index
                    ]
                ),
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
                "per_action_policy_mediated_gain": arrays[
                    "per_action_policy_mediated_gains"
                ][time_index, environment_index].tolist(),
                "per_action_policy_mediated_gain_lcb": arrays[
                    "per_action_policy_mediated_gain_lcbs"
                ][time_index, environment_index].tolist(),
                "per_action_policy_gain_uncertainty": arrays[
                    "per_action_policy_gain_uncertainties"
                ][time_index, environment_index].tolist(),
                "per_action_expected_next_policy_tv": arrays[
                    "per_action_expected_next_policy_tvs"
                ][time_index, environment_index].tolist(),
                "information_gain": arrays["information_gains"][
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
                "predicted_policy_mediated_effect": float(
                    arrays["predicted_policy_mediated_effects"][
                        time_index, environment_index
                    ]
                ),
                "predicted_policy_mediated_effect_lcb": float(
                    arrays["predicted_policy_mediated_effect_lcbs"][
                        time_index, environment_index
                    ]
                ),
                "predicted_policy_gain_uncertainty": float(
                    arrays["predicted_policy_gain_uncertainties"][
                        time_index, environment_index
                    ]
                ),
                "predicted_next_policy_total_variation": float(
                    arrays["predicted_next_policy_total_variations"][
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
                "executed_action_policy_mediated_gain": float(
                    arrays["executed_action_policy_mediated_gains"][
                        time_index, environment_index
                    ]
                ),
                "executed_action_policy_mediated_gain_lcb": float(
                    arrays["executed_action_policy_mediated_gain_lcbs"][
                        time_index, environment_index
                    ]
                ),
                "executed_action_policy_gain_uncertainty": float(
                    arrays["executed_action_policy_gain_uncertainties"][
                        time_index, environment_index
                    ]
                ),
                "executed_action_expected_next_policy_tv": float(
                    arrays["executed_action_expected_next_policy_tvs"][
                        time_index, environment_index
                    ]
                ),
                "maximum_action_net_value": float(
                    arrays["maximum_action_net_values"][
                        time_index, environment_index
                    ]
                ),
                "maximum_action_policy_mediated_gain": float(
                    arrays["maximum_action_policy_mediated_gains"][
                        time_index, environment_index
                    ]
                ),
                "maximum_action_policy_mediated_gain_lcb": float(
                    arrays["maximum_action_policy_mediated_gain_lcbs"][
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


def _responsibility_rows(
    assignments: Any,
    records: Mapping[str, Any],
    *,
    update_count: int,
    epoch: int,
) -> Iterable[Mapping[str, Any]]:
    """Persist the complete E-step; audit labels never enter the losses."""

    import numpy as np

    responsibilities = np.asarray(assignments.responsibilities)
    td = np.asarray(assignments.td_energies)
    response = np.asarray(assignments.response_energies)
    reward = np.asarray(assignments.reward_energies)
    use = np.asarray(assignments.next_q_use_energies)
    mask = np.asarray(assignments.next_q_mask_energies)
    reference = np.asarray(assignments.next_reference_energy)
    available = np.asarray(assignments.bootstrap_mask, dtype=np.bool_)
    importance_ratio = np.asarray(assignments.importance_ratio_mean)
    trace_coefficient = np.asarray(assignments.trace_coefficient_mean)
    partner = np.asarray(records["partner_members"])[0]
    episode = np.asarray(records["episode_ids"])[0]
    environment_count, slot_count = responsibilities.shape
    for environment_index in range(environment_count):
        for slot in range(slot_count):
            yield {
                "update_count": int(update_count),
                "epoch": int(epoch),
                "environment_index": int(environment_index),
                "episode_id": int(episode[environment_index]),
                "audit_partner_member": int(partner[environment_index]),
                "slot": int(slot),
                "responsibility": float(
                    responsibilities[environment_index, slot]
                ),
                "td_energy": float(td[environment_index, slot]),
                "response_nll_energy": float(
                    response[environment_index, slot]
                ),
                "reward_nll_energy": float(
                    reward[environment_index, slot]
                ),
                "next_q_use_nll_energy": float(
                    use[environment_index, slot]
                ),
                "next_q_mask_nll_energy": float(
                    mask[environment_index, slot]
                ),
                "next_reference_mse_energy": float(
                    reference[environment_index]
                ),
                "bootstrap_available": bool(
                    available[environment_index, slot]
                ),
                "mean_importance_ratio": float(
                    importance_ratio[environment_index]
                ),
                "mean_trace_coefficient": float(
                    trace_coefficient[environment_index]
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
        control_carry: Any,
        features: Any,
        previous_actions: Any,
        previous_team_rewards: Any,
        episode_start: Any,
        belief: Any,
    ) -> tuple[Any, Mapping[str, Any]]:
        return model.apply(
            {"params": params},
            control_carry,
            features,
            previous_actions,
            previous_team_rewards,
            episode_start,
            belief,
            method=model.step,
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
        hidden_dim=config.model.hidden_dim,
        initial_temperature=config.kl.initial_temperature,
        gamma=config.training.gamma,
        uncertainty_penalty=config.model.uncertainty_penalty,
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

    config = load_config(args.config, run_kind=args.run_kind)
    output = Path(args.output).resolve()
    manifest = load_training_unit_manifest(
        args.unit_manifest,
        expected_layout=config.environment.layout,
        run_kind=config.run_kind,
    )
    unit = manifest.unit(args.outer_unit)
    reference_path = unit.reference_checkpoint
    partner_paths = unit.partner_checkpoints
    ensure_run_identity(
        output,
        training_identity(
            config=config,
            seed=args.seed,
            outer_unit_id=args.outer_unit,
            reference_checkpoint=reference_path,
            partner_checkpoints=partner_paths,
        ),
    )
    official_config, reference_params = restore_official_checkpoint(
        reference_path
    )
    reference_network = OfficialNetwork(official_config)
    partner_pool = FrozenPartnerPool.from_checkpoints(partner_paths)
    if (
        reference_network.layout != config.environment.layout
        or partner_pool.network.layout != config.environment.layout
    ):
        raise ValueError("Reference, partners, and training environment must share a layout.")
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
        example_official_features=example_features,
        example_previous_actions=jnp.full(
            (environment.num_envs,),
            config.model.action_count,
            dtype=jnp.int32,
        ),
        example_previous_team_rewards=jnp.zeros(
            (environment.num_envs,), dtype=jnp.float32
        ),
        example_episode_start=starts,
        example_slot_log_belief=uniform_slot_log_belief(
            (environment.num_envs,), config.model.slot_count
        ),
        example_observations=ego_observations[None, ...],
        hidden_dim=config.model.hidden_dim,
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
        hidden_dim=config.model.hidden_dim,
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
            signature_dim=2 * config.model.action_count,
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
        restored = restore_latest_checkpoint(manager, item=state)
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
        uncertainty_penalty=config.model.uncertainty_penalty,
        terminal_response=config.model.response_count - 1,
        behavior_exploration_mix=(
            config.training.behavior_exploration_mix
        ),
        behavior_exploration_temperature=(
            config.training.behavior_exploration_temperature
        ),
        behavior_uniform_floor=config.training.behavior_uniform_floor,
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
            retrace_lambda=config.training.retrace_lambda,
            importance_ratio_clip=config.training.importance_ratio_clip,
            uncertainty_penalty=config.model.uncertainty_penalty,
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
    responsibilities_path = output / "records" / "responsibilities"
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
            write_jsonl(
                responsibilities_path
                / f"update_{int(np.asarray(state.update_count)) + 1:08d}_epoch_{epoch:02d}.jsonl",
                _responsibility_rows(
                    assignments,
                    rollout_records,
                    update_count=int(np.asarray(state.update_count)) + 1,
                    epoch=epoch,
                ),
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
        updated_posterior_scores, updated_generic_scores = recompute_policy_scores(
            functions=model_functions,
            params=update.params,
            batch=batch,
            temperature=jnp.exp(
                jnp.ravel(state.runner_state.ego_policy.log_temperature)[0]
            ),
            generic_temperature=jnp.exp(
                jnp.ravel(
                    state.runner_state.ego_policy.generic_log_temperature
                )[0]
            ),
            gamma=config.training.gamma,
            uncertainty_penalty=config.model.uncertainty_penalty,
        )
        temperature_batch = batch._replace(
            posterior_scores=updated_posterior_scores,
            generic_scores=updated_generic_scores,
        )
        policy_state, solved_temperature_metrics = solve_policy_temperatures(
            runner_state.ego_policy,
            batch=temperature_batch,
            target_kl=config.kl.target_per_step,
            minimum_temperature=config.kl.minimum_temperature,
            maximum_temperature=config.kl.maximum_temperature,
            iterations=config.kl.bisection_iterations,
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
                    "rollout_posterior_mean_kl": float(
                        np.asarray(posterior_kl)
                    ),
                    "rollout_generic_mean_kl": float(
                        np.asarray(generic_kl)
                    ),
                    "solved_posterior_mean_kl": float(
                        np.asarray(
                            solved_temperature_metrics["posterior_solved_kl"]
                        )
                    ),
                    "solved_generic_mean_kl": float(
                        np.asarray(
                            solved_temperature_metrics["generic_solved_kl"]
                        )
                    ),
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
        effective_environment_steps=final_step,
        update_count=int(np.asarray(state.update_count)),
        completed_episodes=int(
            np.asarray(state.runner_state.completed_episodes)
        ),
    )
    print(f"Complete decision records: {decisions_path}")
    print(f"Complete episode records: {episodes_path}")
    print(f"Complete metric records: {metrics_path}")
    print(f"Complete responsibility records: {responsibilities_path}")

__all__ = ["run_training"]
