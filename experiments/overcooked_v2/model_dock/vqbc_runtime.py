"""OvercookedV2 training runtime for the fourth Path C model.

This module is imported only by explicitly authorized remote runs.  Importing
the configuration and contract modules remains possible without JAX, Flax,
Optax, JaxMARL, or the external official experiments package.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, NamedTuple

from src.path_c.vqbc.checkpoint import (
    load_vqbc_checkpoint,
    save_vqbc_checkpoint,
)
from src.path_c.vqbc.codebook import empty_codebook, nearest_codes
from src.path_c.vqbc.config import VQBCConfig
from src.path_c.vqbc.integrity import tree_sha256
from src.path_c.vqbc.model import (
    build_vqbc_model,
    encode_response_codes,
    initialize_vqbc_from_official,
)
from src.path_c.vqbc.policy import uniform_slot_log_belief
from src.path_c.vqbc.quotient import slot_bayes_update
from src.path_c.vqbc.rollout import (
    VQBCRolloutCallbacks,
    initialize_policy_state,
    initialize_rollout,
    make_compiled_collector,
    policy_action,
)
from src.path_c.vqbc.training import (
    OptimizerBundle,
    apply_rollout_updates,
    environment_minibatch_schedule,
    make_optimizers,
    prepare_frozen_assignments,
    rollout_health_diagnostics,
    rollout_kl_means,
    sample_bootstrap_mask,
    update_codebook_from_assignments,
    update_kl_state,
)
from src.path_c.vqbc.types import (
    VQBCKLState,
    VQBCTrainState,
)

from .env_dock import OvercookedV2VectorEnvironment
from .official_dock import (
    OfficialBackboneDock,
    load_frozen_partner_pool,
    make_batched_partner_step,
)


class TrainingPartnerCarry(NamedTuple):
    dynamic_policy: Any
    static_carry: Any


class TrainingPartnerContext(NamedTuple):
    dynamic_lane: Any
    dynamic_output: Any
    dynamic_action: Any


def _tree_select(mask: Any, selected: Any, alternative: Any) -> Any:
    import jax
    import jax.numpy as jnp

    def choose(selected_leaf: Any, alternative_leaf: Any) -> Any:
        expanded = jnp.asarray(mask, dtype=jnp.bool_).reshape(
            jnp.asarray(mask).shape
            + (1,) * (jnp.ndim(selected_leaf) - jnp.asarray(mask).ndim)
        )
        return jnp.where(expanded, selected_leaf, alternative_leaf)

    return jax.tree_util.tree_map(choose, selected, alternative)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_json_line(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _host_number(value: Any) -> float:
    import numpy as np

    return float(np.asarray(value))


def _host_value(value: Any) -> Any:
    import numpy as np

    if isinstance(value, Mapping):
        return {
            str(name): _host_value(member)
            for name, member in value.items()
        }
    array = np.asarray(value)
    if array.ndim:
        return array.tolist()
    if np.issubdtype(array.dtype, np.bool_):
        return bool(array)
    if np.issubdtype(array.dtype, np.integer):
        return int(array)
    return float(array)


@dataclass(frozen=True)
class VQBCWiring:
    config: VQBCConfig
    environment: Any
    official_dock: OfficialBackboneDock
    official_params: Mapping[str, Any]
    model: Any
    initial_params: Mapping[str, Any]
    callbacks: VQBCRolloutCallbacks
    partner_initial_carry: Any
    reference_parity: Mapping[str, Any]


def _reference_callback(dock: OfficialBackboneDock, params: Mapping[str, Any]) -> Any:
    def apply(carry: Any, observations: Any, episode_start: Any) -> tuple[Any, Any]:
        next_carry, logits, unused_value = dock.apply_reference_actor_critic(
            params,
            carry,
            observations[None, ...],
            episode_start[None, ...],
        )
        del unused_value
        return next_carry, logits[0]

    return apply


def _model_callback(model: Any) -> Any:
    def apply(
        params: Mapping[str, Any],
        carry: Any,
        observations: Any,
        previous_actions: Any,
        previous_team_rewards: Any,
        episode_start: Any,
    ) -> tuple[Any, Mapping[str, Any]]:
        return model.apply(
            {"params": params},
            carry,
            observations,
            previous_actions,
            previous_team_rewards,
            episode_start,
        )

    return apply


def _partner_callbacks(
    *,
    model: Any,
    model_apply: Any,
    reference_apply: Any,
    reference_initial_carry: Any,
    static_partners: tuple[Any, ...],
    slot_count: int,
    action_count: int,
    initial_temperature: float,
    gamma: float,
) -> tuple[Any, Any, Any]:
    import jax
    import jax.numpy as jnp

    static_step = make_batched_partner_step(static_partners)
    policy_callbacks = VQBCRolloutCallbacks(
        model_apply=model_apply,
        reference_apply=reference_apply,
        partner_step=lambda *unused: None,
        partner_observe=lambda *unused: None,
    )

    def initial(batch_size: int) -> TrainingPartnerCarry:
        return TrainingPartnerCarry(
            dynamic_policy=initialize_policy_state(
                model_initial_carry=model.initial_carry,
                reference_initial_carry=reference_initial_carry,
                batch_size=batch_size,
                slot_count=slot_count,
                action_count=action_count,
                initial_temperature=initial_temperature,
            ),
            static_carry=static_partners[0].initial_state(batch_size),
        )

    def step(
        frozen_params: Mapping[str, Any],
        member_index: Any,
        observations: Any,
        carry: TrainingPartnerCarry,
        episode_start: Any,
        key: Any,
    ) -> tuple[Any, TrainingPartnerCarry, TrainingPartnerContext]:
        dynamic_key, static_key = jax.random.split(key)
        dynamic_lane = jnp.asarray(member_index) == 0
        static_indexes = jnp.maximum(jnp.asarray(member_index) - 1, 0)
        static_action, static_next = static_step(
            static_indexes,
            observations,
            carry.static_carry,
            episode_start,
            static_key,
        )
        (
            dynamic_next,
            dynamic_action,
            dynamic_output,
            unused_decision,
            unused_generic,
        ) = policy_action(
            callbacks=policy_callbacks,
            params=frozen_params,
            policy_state=carry.dynamic_policy,
            observations=observations,
            key=dynamic_key,
            deployment_mode="posterior_use",
            gamma=gamma,
        )
        del unused_decision, unused_generic
        selected_dynamic = _tree_select(
            dynamic_lane, dynamic_next, carry.dynamic_policy
        )
        selected_static = _tree_select(
            dynamic_lane, carry.static_carry, static_next
        )
        return (
            jnp.where(dynamic_lane, dynamic_action, static_action),
            TrainingPartnerCarry(
                dynamic_policy=selected_dynamic,
                static_carry=selected_static,
            ),
            TrainingPartnerContext(
                dynamic_lane=dynamic_lane,
                dynamic_output=dynamic_output,
                dynamic_action=dynamic_action,
            ),
        )

    def observe(
        frozen_params: Mapping[str, Any],
        member_index: Any,
        carry: TrainingPartnerCarry,
        context: TrainingPartnerContext,
        observations: Any,
        next_observations: Any,
        actions: Any,
        rewards: Any,
        dones: Any,
        codebook_embeddings: Any,
    ) -> TrainingPartnerCarry:
        del member_index, actions
        response_codes, unused_logits, unused_signatures = encode_response_codes(
            model=model,
            params=frozen_params,
            observations=observations[None, ...],
            actions=context.dynamic_action[None, ...],
            next_observations=next_observations[None, ...],
            dones=dones[None, ...],
            codebook_embeddings=codebook_embeddings,
        )
        del unused_logits, unused_signatures
        response_codes = response_codes[0]
        output = context.dynamic_output
        updated_belief = slot_bayes_update(
            slot_log_belief=carry.dynamic_policy.slot_log_belief,
            slot_response_probabilities=output.response_probabilities,
            action=context.dynamic_action,
            response_code=response_codes,
        )
        uniform = uniform_slot_log_belief(
            updated_belief.shape[:-1], slot_count
        )
        candidate = carry.dynamic_policy._replace(
            slot_log_belief=jnp.where(
                dones[:, None], uniform, updated_belief
            ),
            previous_action=jnp.where(
                dones,
                jnp.full(dones.shape, action_count, dtype=jnp.int32),
                context.dynamic_action,
            ),
            previous_team_reward=jnp.where(dones, 0.0, rewards),
            episode_start=dones,
        )
        updated_dynamic = _tree_select(
            context.dynamic_lane, candidate, carry.dynamic_policy
        )
        # Every completed lane is reset even when the previous partner was
        # static, because the next episode may select the dynamic partner.
        reset_dynamic = carry.dynamic_policy._replace(
            slot_log_belief=uniform,
            previous_action=jnp.full(dones.shape, action_count, dtype=jnp.int32),
            previous_team_reward=jnp.zeros(dones.shape, dtype=jnp.float32),
            episode_start=jnp.ones(dones.shape, dtype=jnp.bool_),
        )
        updated_dynamic = _tree_select(dones, reset_dynamic, updated_dynamic)
        return TrainingPartnerCarry(
            dynamic_policy=updated_dynamic,
            static_carry=carry.static_carry,
        )

    return initial, step, observe


def _verify_reference_parity(
    *,
    dock: OfficialBackboneDock,
    official_params: Mapping[str, Any],
    model: Any,
    initialized_params: Mapping[str, Any],
    observations: Any,
) -> Mapping[str, Any]:
    import jax
    import jax.numpy as jnp
    import numpy as np

    batch_size = int(observations.shape[1])
    carry = dock.initial_recurrent_state(batch_size)
    starts = jnp.ones(observations.shape[:2], dtype=jnp.bool_)
    official_carry, official_logits, unused_value = dock.apply_reference_actor_critic(
        official_params, carry, observations, starts
    )
    del unused_value
    model_carry, unused_output = model.apply(
        {"params": initialized_params},
        carry,
        observations,
        jnp.full(observations.shape[:2], 6, dtype=jnp.int32),
        jnp.zeros(observations.shape[:2], dtype=jnp.float32),
        starts,
    )
    del unused_output
    carry_equal = all(
        np.array_equal(np.asarray(left), np.asarray(right))
        for left, right in zip(
            jax.tree_util.tree_leaves(official_carry),
            jax.tree_util.tree_leaves(model_carry),
            strict=True,
        )
    )
    if not carry_equal:
        raise RuntimeError(
            "Zero-initialized history inputs changed the official recurrent carry."
        )
    if int(official_logits.shape[-1]) != 6:
        raise RuntimeError("The unit-owned official reference has a non-six-action actor.")
    return {
        "reference_training_run_id": "",
        "carry_exact": True,
        "reference_logit_shape": [int(value) for value in official_logits.shape],
        "reference_output_sha256": tree_sha256(
            {
                "carry": official_carry,
                "logits": official_logits,
            }
        ),
    }


def build_vqbc_wiring(config: VQBCConfig) -> VQBCWiring:
    import jax
    import jax.numpy as jnp

    if (
        not config.backbone_init.launch_config_path.is_file()
        or config.backbone_init.launch_config_sha256
        != _file_sha256(config.backbone_init.launch_config_path)
    ):
        raise ValueError("The reference launch configuration hash changed.")
    environment = OvercookedV2VectorEnvironment.create(
        num_envs=config.environment.num_envs,
        layout=config.environment.layout,
    )
    dock = OfficialBackboneDock.from_launch_config(
        config.backbone_init.launch_config_path
    )
    if tuple(dock.action_order()) != (
        "right",
        "down",
        "left",
        "up",
        "stay",
        "interact",
    ):
        raise ValueError("Official action order changed.")
    official_params = dock.official_parameter_tree(config.backbone_init)
    model = build_vqbc_model(
        model_config=config.model,
        official_dimensions=dock.network_dimensions(),
    )
    reset_key, init_key = jax.random.split(jax.random.PRNGKey(config.seeds.model_seed))
    unused_environment_state, observations = environment.reset(reset_key)
    del unused_environment_state
    ego_observations = observations[:, 0][None, ...]
    initial_params = initialize_vqbc_from_official(
        model,
        random_key=init_key,
        official_params=official_params,
        example_observations=ego_observations,
        example_previous_actions=jnp.full(
            ego_observations.shape[:2], config.model.action_count, dtype=jnp.int32
        ),
        example_previous_team_rewards=jnp.zeros(
            ego_observations.shape[:2], dtype=jnp.float32
        ),
        example_episode_start=jnp.ones(
            ego_observations.shape[:2], dtype=jnp.bool_
        ),
    )
    reference_apply = _reference_callback(dock, official_params)
    model_apply = _model_callback(model)
    members = tuple(
        member.checkpoint for member in config.partner_sampling.members
    )
    launch_paths = tuple(
        member.checkpoint.launch_config_path
        for member in config.partner_sampling.members
    )
    static_partners = load_frozen_partner_pool(
        members, launch_config_paths=launch_paths
    )
    partner_initial, partner_step, partner_observe = _partner_callbacks(
        model=model,
        model_apply=model_apply,
        reference_apply=reference_apply,
        reference_initial_carry=dock.initial_recurrent_state,
        static_partners=static_partners,
        slot_count=config.model.slot_count,
        action_count=config.model.action_count,
        initial_temperature=config.kl_control.initial_temperature,
        gamma=config.training.gamma,
    )
    parity = _verify_reference_parity(
        dock=dock,
        official_params=official_params,
        model=model,
        initialized_params=initial_params,
        observations=ego_observations,
    )
    parity = {
        **parity,
        "reference_training_run_id": config.backbone_init.training_run_id,
        "reference_weights_sha256": config.backbone_init.flax_weights_sha256,
    }
    return VQBCWiring(
        config=config,
        environment=environment,
        official_dock=dock,
        official_params=official_params,
        model=model,
        initial_params=initial_params,
        callbacks=VQBCRolloutCallbacks(
            model_apply=model_apply,
            reference_apply=reference_apply,
            partner_step=partner_step,
            partner_observe=partner_observe,
        ),
        partner_initial_carry=partner_initial,
        reference_parity=parity,
    )


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _initial_train_state(wiring: VQBCWiring) -> tuple[VQBCTrainState, Any, Any]:
    import jax
    import jax.numpy as jnp

    config = wiring.config
    root_key = jax.random.PRNGKey(config.seeds.environment_seed)
    rollout_key, state_key = jax.random.split(root_key)
    rollout_state = initialize_rollout(
        environment=wiring.environment,
        model_initial_carry=wiring.model.initial_carry,
        reference_initial_carry=wiring.official_dock.initial_recurrent_state,
        partner_initial_carry=wiring.partner_initial_carry,
        random_key=rollout_key,
        slot_count=config.model.slot_count,
        action_count=config.model.action_count,
        partner_member_count=len(config.partner_sampling.members),
        initial_temperature=config.kl_control.initial_temperature,
    )
    optimizers = make_optimizers(
        wiring.initial_params,
        bellman_learning_rate=config.training.bellman_learning_rate,
        outcome_learning_rate=config.training.outcome_learning_rate,
        gradient_clip_norm=config.training.gradient_clip_norm,
    )
    state = VQBCTrainState(
        online_params=wiring.initial_params,
        target_params=wiring.initial_params,
        bellman_optimizer_state=optimizers.bellman_state,
        outcome_optimizer_state=optimizers.outcome_state,
        codebook=empty_codebook(
            code_count=config.model.response_count - 1,
            signature_dim=config.model.action_count,
        ),
        kl_state=VQBCKLState(
            log_temperature=jnp.log(config.kl_control.initial_temperature),
            generic_log_temperature=jnp.log(
                config.kl_control.initial_temperature
            ),
        ),
        rollout_state=rollout_state,
        random_key=state_key,
        effective_environment_steps=jnp.asarray(0, dtype=jnp.int64),
        completed_episodes=jnp.asarray(0, dtype=jnp.int64),
        update_count=jnp.asarray(0, dtype=jnp.int64),
    )
    return state, optimizers.bellman_optimizer, optimizers.outcome_optimizer


def run_vqbc_training(
    config: VQBCConfig, *, resume_checkpoint: str | Path | None = None
) -> Mapping[str, Path]:
    """Train from fresh complete episodes and emit resumable fourth-model state."""

    import jax
    import jax.numpy as jnp
    import numpy as np

    wiring = build_vqbc_wiring(config)
    state, bellman_optimizer, outcome_optimizer = _initial_train_state(wiring)
    if resume_checkpoint is not None:
        state, unused_metadata, unused_manifest = load_vqbc_checkpoint(
            resume_checkpoint,
            target_state=state,
            expected_config=config,
        )
        del unused_metadata, unused_manifest
    metrics_path = config.output_root / "training" / "metrics.jsonl"
    summary_path = config.output_root / "training" / "summary.json"
    resolved_config_path = config.output_root / "training" / "resolved_config.json"
    _atomic_json(resolved_config_path, config.to_mapping())
    checkpoint_root = config.output_root / "training" / "checkpoints"

    def apply_compiled_updates(
        online_params: Any,
        target_params: Any,
        bellman_optimizer_state: Any,
        outcome_optimizer_state: Any,
        batch: Any,
        assignments: Any,
        codebook_embeddings: Any,
        schedule: Any,
    ) -> Any:
        return apply_rollout_updates(
            model=wiring.model,
            params=online_params,
            target_params=target_params,
            optimizer_bundle=OptimizerBundle(
                bellman_optimizer=bellman_optimizer,
                outcome_optimizer=outcome_optimizer,
                bellman_state=bellman_optimizer_state,
                outcome_state=outcome_optimizer_state,
            ),
            batch=batch,
            assignments=assignments,
            codebook_embeddings=codebook_embeddings,
            schedule=schedule,
            polyak_coefficient=config.training.polyak_coefficient,
        )

    compiled_rollout_updates = jax.jit(apply_compiled_updates)
    compiled_rollout_collector = make_compiled_collector(
        length=config.training.unroll_length,
        environment=wiring.environment,
        callbacks=wiring.callbacks,
        model=wiring.model,
        partner_member_count=len(config.partner_sampling.members),
        deployment_mode="posterior_use",
        gamma=config.training.gamma,
    )

    def apply_compiled_assignments(
        online_params: Any,
        target_params: Any,
        batch: Any,
        codebook_embeddings: Any,
        key: Any,
        temperature: Any,
        generic_temperature: Any,
        bootstrap_mask: Any,
    ) -> Any:
        return prepare_frozen_assignments(
            model=wiring.model,
            online_params=online_params,
            target_params=target_params,
            batch=batch,
            codebook_embeddings=codebook_embeddings,
            key=key,
            temperature=temperature,
            generic_temperature=generic_temperature,
            gamma=config.training.gamma,
            responsibility_temperature=(
                config.training.responsibility_temperature
            ),
            bootstrap_probability=config.training.bootstrap_probability,
            bootstrap_mask=bootstrap_mask,
            environment_chunk_size=max(
                1,
                config.environment.num_envs
                // config.training.minibatches_per_epoch,
            ),
        )

    compiled_assignments = jax.jit(apply_compiled_assignments)
    while int(np.asarray(state.effective_environment_steps)) < (
        config.training.environment_steps
    ):
        (
            assignment_key,
            schedule_key,
            codebook_key,
            bootstrap_key,
            next_state_key,
        ) = jax.random.split(state.random_key, 5)
        rollout_state, batch = compiled_rollout_collector(
            state=state.rollout_state,
            params=state.online_params,
            codebook_embeddings=state.codebook.embeddings,
        )
        bootstrap_mask = sample_bootstrap_mask(
            bootstrap_key,
            environment_count=config.environment.num_envs,
            slot_count=config.model.slot_count,
            probability=config.training.bootstrap_probability,
        )
        provisional = compiled_assignments(
            state.online_params,
            state.target_params,
            batch=batch,
            codebook_embeddings=state.codebook.embeddings,
            key=assignment_key,
            temperature=jnp.exp(state.kl_state.log_temperature),
            generic_temperature=jnp.exp(
                state.kl_state.generic_log_temperature
            ),
            bootstrap_mask=bootstrap_mask,
        )
        codebook_update = update_codebook_from_assignments(
            state.codebook,
            assignments=provisional,
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
        update = None
        assignments = provisional
        for epoch in range(config.training.update_epochs):
            assignments = compiled_assignments(
                online_params,
                target_params,
                batch=batch,
                codebook_embeddings=codebook_update.state.embeddings,
                key=jax.random.fold_in(assignment_key, epoch + 1),
                temperature=jnp.exp(state.kl_state.log_temperature),
                generic_temperature=jnp.exp(
                    state.kl_state.generic_log_temperature
                ),
                bootstrap_mask=bootstrap_mask,
            )
            update = compiled_rollout_updates(
                online_params,
                target_params,
                bellman_state,
                outcome_state,
                batch=batch,
                assignments=assignments,
                codebook_embeddings=codebook_update.state.embeddings,
                schedule=schedule[epoch],
            )
            online_params = update.params
            target_params = update.target_params
            bellman_state = update.bellman_optimizer_state
            outcome_state = update.outcome_optimizer_state
        if update is None:
            raise RuntimeError("VQBC completed no M-step epoch.")
        posterior_kl, generic_kl = rollout_kl_means(batch)
        kl_state = update_kl_state(
            state.kl_state,
            posterior_mean_kl=posterior_kl,
            generic_mean_kl=generic_kl,
            target_kl=config.kl_control.target_per_step,
            learning_rate=config.kl_control.dual_learning_rate,
            minimum_temperature=config.kl_control.minimum_temperature,
            maximum_temperature=config.kl_control.maximum_temperature,
        )
        policy_state = rollout_state.policy_state._replace(
            log_temperature=jnp.full_like(
                rollout_state.policy_state.log_temperature,
                kl_state.log_temperature,
            ),
            generic_log_temperature=jnp.full_like(
                rollout_state.policy_state.generic_log_temperature,
                kl_state.generic_log_temperature,
            ),
        )
        rollout_state = rollout_state._replace(policy_state=policy_state)
        state = VQBCTrainState(
            online_params=update.params,
            target_params=update.target_params,
            bellman_optimizer_state=update.bellman_optimizer_state,
            outcome_optimizer_state=update.outcome_optimizer_state,
            codebook=codebook_update.state,
            kl_state=kl_state,
            rollout_state=rollout_state,
            random_key=next_state_key,
            effective_environment_steps=(
                state.effective_environment_steps
                + config.environment.num_envs * config.training.unroll_length
            ),
            completed_episodes=rollout_state.completed_episodes,
            update_count=state.update_count + 1,
        )
        effective_steps = int(np.asarray(state.effective_environment_steps))
        scheduled_metric = (
            effective_steps
            % config.training.metrics_interval_environment_steps
            == 0
        )
        first_rollout_metric = int(np.asarray(state.update_count)) == 1
        if scheduled_metric or first_rollout_metric:
            training_diagnostics = rollout_health_diagnostics(
                batch=batch,
                assignments=assignments,
                response_count=config.model.response_count,
            )
            _append_json_line(
                metrics_path,
                {
                    "schema_version": "path_c_vqbc_training_metric_v3",
                    "run_kind": config.run_kind,
                    "scientific_readout_allowed": False,
                    "diagnostic_window": "latest_complete_rollout",
                    "scheduled_metric": scheduled_metric,
                    "effective_environment_steps": effective_steps,
                    "completed_episodes": int(
                        np.asarray(state.completed_episodes)
                    ),
                    "update_count": int(np.asarray(state.update_count)),
                    "posterior_mean_kl": _host_number(posterior_kl),
                    "generic_mean_kl": _host_number(generic_kl),
                    "temperature": _host_number(
                        jnp.exp(kl_state.log_temperature)
                    ),
                    "generic_temperature": _host_number(
                        jnp.exp(kl_state.generic_log_temperature)
                    ),
                    "codebook_replacement_count": int(
                        np.asarray(state.codebook.replacement_count)
                    ),
                    "codebook_last_replaced_codes": [
                        int(index)
                        for index in np.flatnonzero(
                            np.asarray(state.codebook.last_replaced_codes)
                        ).tolist()
                    ],
                    "losses": {
                        name: _host_number(value)
                        for name, value in update.metrics.items()
                    },
                    "training_diagnostics": _host_value(
                        training_diagnostics
                    ),
                },
            )
        if (
            effective_steps
            % config.training.checkpoint_interval_environment_steps
            == 0
        ):
            save_vqbc_checkpoint(
                checkpoint_root / f"step_{effective_steps:09d}",
                config=config,
                train_state=state,
            )
    final_checkpoint = checkpoint_root / "final"
    save_vqbc_checkpoint(
        final_checkpoint, config=config, train_state=state
    )
    _atomic_json(
        summary_path,
        {
            "schema_version": "path_c_vqbc_training_summary_v2",
            "run_kind": config.run_kind,
            "scientific_readout_allowed": False,
            "effective_environment_steps": int(
                np.asarray(state.effective_environment_steps)
            ),
            "completed_episodes": int(np.asarray(state.completed_episodes)),
            "update_count": int(np.asarray(state.update_count)),
            "final_checkpoint": str(final_checkpoint),
            "reference_training_run_id": config.backbone_init.training_run_id,
            "outer_unit_id": (
                None
                if config.outer_unit is None
                else config.outer_unit.outer_unit_id
            ),
        },
    )
    return {
        "metrics": metrics_path,
        "summary": summary_path,
        "checkpoint_manifest": final_checkpoint / "manifest.json",
        "resolved_config": resolved_config_path,
    }


__all__ = [
    "TrainingPartnerCarry",
    "VQBCWiring",
    "build_vqbc_wiring",
    "run_vqbc_training",
]
