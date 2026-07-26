"""Fresh-rollout training, separate optimizers, target updates, and KL control."""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple

from .codebook import (
    nearest_codes,
    target_response_signatures,
    update_codebook,
)
from .objectives import (
    FrozenAssignments,
    bellman_loss,
    bellman_targets,
    episode_responsibilities,
    outcome_loss,
    response_encoder_loss,
    sample_bootstrap_mask,
)
from .policy import regularized_policy, update_log_temperature
from .model import vqbc_forward
from .types import VQBCKLState, VQBCRolloutBatch


class OptimizerBundle(NamedTuple):
    bellman_optimizer: Any
    outcome_optimizer: Any
    bellman_state: Any
    outcome_state: Any


class TrainingUpdate(NamedTuple):
    params: Any
    bellman_optimizer_state: Any
    outcome_optimizer_state: Any
    metrics: Mapping[str, Any]


def _labels_for_top_level(params: Mapping[str, Any], selected: set[str]) -> Any:
    import jax

    return {
        str(name): jax.tree_util.tree_map(
            lambda unused, label=("train" if str(name) in selected else "frozen"): label,
            subtree,
        )
        for name, subtree in params.items()
    }


def _bellman_labels(params: Mapping[str, Any]) -> Any:
    import jax

    labels = _labels_for_top_level(params, {"backbone"})
    if "q_heads" not in params:
        raise ValueError("Fourth-model parameters are missing q_heads.")
    labels["q_heads"] = {
        str(name): jax.tree_util.tree_map(
            lambda unused, label=(
                "train" if str(name).startswith("LearnedEstimator_") else "frozen"
            ): label,
            subtree,
        )
        for name, subtree in params["q_heads"].items()
    }
    return labels


def make_optimizers(
    params: Mapping[str, Any],
    *,
    bellman_learning_rate: float,
    outcome_learning_rate: float,
    gradient_clip_norm: float,
) -> OptimizerBundle:
    """Clip Bellman and outcome gradients separately before their Adam updates."""

    import optax

    def transform(rate: float, labels: Any) -> Any:
        return optax.multi_transform(
            {
                "train": optax.chain(
                    optax.clip_by_global_norm(float(gradient_clip_norm)),
                    optax.adam(float(rate)),
                ),
                "frozen": optax.set_to_zero(),
            },
            labels,
        )

    bellman = transform(
        bellman_learning_rate, _bellman_labels(params)
    )
    outcome = transform(
        outcome_learning_rate,
        _labels_for_top_level(params, {"outcome", "response_encoder"}),
    )
    return OptimizerBundle(
        bellman_optimizer=bellman,
        outcome_optimizer=outcome,
        bellman_state=bellman.init(params),
        outcome_state=outcome.init(params),
    )


def apply_model_sequence(
    *, model: Any, params: Mapping[str, Any], batch: VQBCRolloutBatch
) -> Mapping[str, Any]:
    unused_carry, output = model.apply(
        {"params": params},
        batch.initial_value_carry,
        batch.observations,
        batch.previous_actions,
        batch.previous_team_rewards,
        batch.episode_start,
    )
    del unused_carry
    return output


def apply_control_sequence(
    *, model: Any, params: Mapping[str, Any], batch: VQBCRolloutBatch
) -> Mapping[str, Any]:
    unused_carry, output = model.apply(
        {"params": params},
        batch.initial_value_carry,
        batch.observations,
        batch.previous_actions,
        batch.previous_team_rewards,
        batch.episode_start,
        method=model.control_only,
    )
    del unused_carry
    return output


def slice_rollout_batch(
    batch: VQBCRolloutBatch, indexes: Any
) -> VQBCRolloutBatch:
    return VQBCRolloutBatch(
        observations=batch.observations[:, indexes],
        response_next_observations=batch.response_next_observations[:, indexes],
        episode_start=batch.episode_start[:, indexes],
        previous_actions=batch.previous_actions[:, indexes],
        previous_team_rewards=batch.previous_team_rewards[:, indexes],
        initial_value_carry=batch.initial_value_carry[indexes],
        reference_logits=batch.reference_logits[:, indexes],
        execution_logits=batch.execution_logits[:, indexes],
        generic_execution_logits=batch.generic_execution_logits[:, indexes],
        slot_log_beliefs=batch.slot_log_beliefs[:, indexes],
        actions=batch.actions[:, indexes],
        rewards=batch.rewards[:, indexes],
        dones=batch.dones[:, indexes],
        response_codes=batch.response_codes[:, indexes],
        episode_ids=batch.episode_ids[:, indexes],
        episode_steps=batch.episode_steps[:, indexes],
        completed_episode_returns=batch.completed_episode_returns[:, indexes],
        quotient_counts=batch.quotient_counts[:, indexes],
        j_use=batch.j_use[:, indexes],
        j_mask=batch.j_mask[:, indexes],
    )


def _next_policy_probabilities(
    *,
    target_output: Mapping[str, Any],
    batch: VQBCRolloutBatch,
    temperature: Any,
    generic_temperature: Any,
    gamma: float,
) -> Any:
    forward = vqbc_forward(
        raw_output={
            name: value[1:]
            for name, value in target_output.items()
        },
        reference_logits=batch.reference_logits[1:],
        slot_log_belief=batch.slot_log_beliefs[1:],
        temperature=temperature,
        generic_temperature=generic_temperature,
        deployment_mode="posterior_use",
        gamma=gamma,
    )
    import jax

    return jax.nn.softmax(forward.execution_logits, axis=-1)


def prepare_frozen_assignments(
    *,
    model: Any,
    online_params: Mapping[str, Any],
    target_params: Mapping[str, Any],
    batch: VQBCRolloutBatch,
    codebook_embeddings: Any,
    key: Any,
    temperature: Any,
    generic_temperature: Any,
    gamma: float,
    responsibility_temperature: float,
    bootstrap_probability: float,
    terminal_response: int = 15,
    environment_chunk_size: int | None = None,
) -> FrozenAssignments:
    """Freeze assignments while bounding the large outcome tensor by lane chunks."""

    import jax
    import jax.numpy as jnp

    environment_count = int(batch.actions.shape[1])
    chunk_size = (
        environment_count
        if environment_chunk_size is None
        else int(environment_chunk_size)
    )
    if chunk_size <= 0:
        raise ValueError("Assignment environment chunks must be positive.")
    if environment_count % chunk_size:
        raise ValueError(
            "Assignment environment count must divide exactly into lane chunks."
        )
    mask = sample_bootstrap_mask(
        key,
        environment_count=environment_count,
        slot_count=8,
        probability=bootstrap_probability,
    )
    chunk_offsets = jnp.arange(chunk_size, dtype=jnp.int32)

    def prepare_chunk(start: Any) -> tuple[Any, Any, Any, Any]:
        indexes = start + chunk_offsets
        current = slice_rollout_batch(batch, indexes)
        online = apply_control_sequence(
            model=model, params=online_params, batch=current
        )
        target = jax.tree_util.tree_map(
            jax.lax.stop_gradient,
            apply_model_sequence(
                model=model, params=target_params, batch=current
            ),
        )
        next_probabilities = _next_policy_probabilities(
            target_output=target,
            batch=current,
            temperature=temperature,
            generic_temperature=generic_temperature,
            gamma=gamma,
        )
        targets = jax.lax.stop_gradient(
            bellman_targets(
                rewards=current.rewards,
                dones=current.dones,
                next_execution_probabilities=next_probabilities,
                target_next_q_values=target["q_values"][1:],
                gamma=gamma,
            )
        )
        responsibilities = episode_responsibilities(
            q_values=online["q_values"][:-1],
            actions=current.actions,
            targets=targets,
            temperature=responsibility_temperature,
        )
        signatures = jax.lax.stop_gradient(
            target_response_signatures(
                target_centered_advantages=target[
                    "centered_advantages"
                ][1:],
                stopped_responsibilities=responsibilities,
            )
        )
        code_targets, unused_error = nearest_codes(
            signatures, codebook_embeddings
        )
        del unused_error
        return (
            responsibilities,
            targets,
            signatures,
            jax.lax.stop_gradient(
                jnp.where(current.dones, terminal_response, code_targets)
            ),
        )

    starts = jnp.arange(0, environment_count, chunk_size, dtype=jnp.int32)
    (
        chunk_responsibilities,
        chunk_targets,
        chunk_signatures,
        chunk_code_targets,
    ) = jax.lax.map(prepare_chunk, starts)
    time_count = int(batch.actions.shape[0])
    return FrozenAssignments(
        responsibilities=chunk_responsibilities.reshape(
            (environment_count, chunk_responsibilities.shape[-1])
        ),
        bootstrap_mask=mask,
        bellman_targets=jnp.swapaxes(chunk_targets, 0, 1).reshape(
            (time_count, environment_count, chunk_targets.shape[-1])
        ),
        response_signature_targets=jnp.swapaxes(
            chunk_signatures, 0, 1
        ).reshape(
            (time_count, environment_count, chunk_signatures.shape[-1])
        ),
        response_code_targets=jnp.swapaxes(
            chunk_code_targets, 0, 1
        ).reshape(
            (time_count, environment_count)
        ),
    )


def slice_environment_lanes(
    batch: VQBCRolloutBatch, assignments: FrozenAssignments, indexes: Any
) -> tuple[VQBCRolloutBatch, FrozenAssignments]:
    import jax

    batch_slice = slice_rollout_batch(batch, indexes)
    assignment_slice = FrozenAssignments(
        responsibilities=assignments.responsibilities[indexes],
        bootstrap_mask=assignments.bootstrap_mask[indexes],
        bellman_targets=assignments.bellman_targets[:, indexes],
        response_signature_targets=assignments.response_signature_targets[
            :, indexes
        ],
        response_code_targets=assignments.response_code_targets[:, indexes],
    )
    return batch_slice, jax.tree_util.tree_map(
        jax.lax.stop_gradient, assignment_slice
    )


def environment_minibatch_schedule(
    key: Any,
    *,
    environment_count: int,
    minibatches_per_epoch: int,
    update_epochs: int,
) -> Any:
    """Use partitions when possible and keyed one-lane cycling for 32-env development."""

    import jax
    import jax.numpy as jnp

    if environment_count >= minibatches_per_epoch:
        if environment_count % minibatches_per_epoch:
            raise ValueError("Formal environment lanes must divide into 50 minibatches.")
        lane_count = environment_count // minibatches_per_epoch
        keys = jax.random.split(key, update_epochs)
        schedules = [
            jax.random.permutation(keys[index], environment_count).reshape(
                (minibatches_per_epoch, lane_count)
            )
            for index in range(update_epochs)
        ]
    else:
        keys = jax.random.split(key, update_epochs)
        repeats = (
            minibatches_per_epoch + environment_count - 1
        ) // environment_count
        schedules = []
        for index in range(update_epochs):
            epoch_keys = jax.random.split(keys[index], repeats)
            order = jnp.concatenate(
                [
                    jax.random.permutation(epoch_key, environment_count)
                    for epoch_key in epoch_keys
                ],
                axis=0,
            )[:minibatches_per_epoch]
            schedules.append(order[:, None])
    return jnp.concatenate(schedules, axis=0)


def _losses(
    *,
    model: Any,
    params: Mapping[str, Any],
    target_params: Mapping[str, Any],
    batch: VQBCRolloutBatch,
    assignments: FrozenAssignments,
    codebook_embeddings: Any,
) -> tuple[Any, Any]:
    import jax

    output = apply_model_sequence(model=model, params=params, batch=batch)
    target_output = jax.tree_util.tree_map(
        jax.lax.stop_gradient,
        apply_model_sequence(model=model, params=target_params, batch=batch),
    )
    bellman = bellman_loss(
        q_values=output["q_values"][:-1],
        actions=batch.actions,
        targets=assignments.bellman_targets,
        stopped_responsibilities=assignments.responsibilities,
        bootstrap_mask=assignments.bootstrap_mask,
    )
    outcome = outcome_loss(
        response_logits=output["response_logits"][:-1],
        reward_mean=output["reward_mean"][:-1],
        reward_log_standard_deviation=output[
            "reward_log_standard_deviation"
        ][:-1],
        next_q_mean=output["next_q_mean"][:-1],
        next_q_log_standard_deviation=output[
            "next_q_log_standard_deviation"
        ][:-1],
        response_codes=batch.response_codes,
        rewards=batch.rewards,
        target_next_q_values=target_output["q_values"][1:],
        dones=batch.dones,
        actions=batch.actions,
        stopped_responsibilities=assignments.responsibilities,
    )
    predicted_signature = model.apply(
        {"params": params},
        batch.observations[:-1],
        batch.actions,
        batch.response_next_observations,
        batch.dones,
        method=model.encode_response,
    )
    encoder = response_encoder_loss(
        predicted_signatures=predicted_signature,
        target_signatures=assignments.response_signature_targets,
        target_codes=assignments.response_code_targets,
        codebook_embeddings=codebook_embeddings,
    )
    return bellman, (
        outcome.total + encoder.total,
        {**outcome.metrics, **encoder.metrics},
    )


def apply_minibatch_update(
    *,
    model: Any,
    params: Mapping[str, Any],
    target_params: Mapping[str, Any],
    bellman_optimizer: Any,
    outcome_optimizer: Any,
    bellman_optimizer_state: Any,
    outcome_optimizer_state: Any,
    batch: VQBCRolloutBatch,
    assignments: FrozenAssignments,
    codebook_embeddings: Any,
) -> TrainingUpdate:
    import jax
    import optax

    def bellman_function(candidate: Mapping[str, Any]) -> tuple[Any, Mapping[str, Any]]:
        bellman, unused_outcome = _losses(
            model=model,
            params=candidate,
            target_params=target_params,
            batch=batch,
            assignments=assignments,
            codebook_embeddings=codebook_embeddings,
        )
        del unused_outcome
        return bellman.total, bellman.metrics

    (unused_bellman_loss, bellman_metrics), bellman_gradients = (
        jax.value_and_grad(bellman_function, has_aux=True)(params)
    )
    del unused_bellman_loss
    bellman_updates, bellman_optimizer_state = bellman_optimizer.update(
        bellman_gradients, bellman_optimizer_state, params
    )
    params = optax.apply_updates(params, bellman_updates)

    def outcome_function(candidate: Mapping[str, Any]) -> tuple[Any, Mapping[str, Any]]:
        unused_bellman, outcome = _losses(
            model=model,
            params=candidate,
            target_params=target_params,
            batch=batch,
            assignments=assignments,
            codebook_embeddings=codebook_embeddings,
        )
        del unused_bellman
        return outcome

    (unused_outcome_loss, outcome_metrics), outcome_gradients = (
        jax.value_and_grad(outcome_function, has_aux=True)(params)
    )
    del unused_outcome_loss
    outcome_updates, outcome_optimizer_state = outcome_optimizer.update(
        outcome_gradients, outcome_optimizer_state, params
    )
    params = optax.apply_updates(params, outcome_updates)
    return TrainingUpdate(
        params=params,
        bellman_optimizer_state=bellman_optimizer_state,
        outcome_optimizer_state=outcome_optimizer_state,
        metrics={**bellman_metrics, **outcome_metrics},
    )


def apply_rollout_updates(
    *,
    model: Any,
    params: Mapping[str, Any],
    target_params: Mapping[str, Any],
    optimizer_bundle: OptimizerBundle,
    batch: VQBCRolloutBatch,
    assignments: FrozenAssignments,
    codebook_embeddings: Any,
    schedule: Any,
) -> TrainingUpdate:
    """Run four epochs over fifty complete-lane minibatches per epoch."""

    import jax
    import jax.numpy as jnp

    initial = (
        params,
        optimizer_bundle.bellman_state,
        optimizer_bundle.outcome_state,
    )

    def update_one(carry: tuple[Any, Any, Any], indexes: Any) -> tuple[Any, Any]:
        current_params, bellman_state, outcome_state = carry
        minibatch, mini_assignments = slice_environment_lanes(
            batch, assignments, indexes
        )
        updated = apply_minibatch_update(
            model=model,
            params=current_params,
            target_params=target_params,
            bellman_optimizer=optimizer_bundle.bellman_optimizer,
            outcome_optimizer=optimizer_bundle.outcome_optimizer,
            bellman_optimizer_state=bellman_state,
            outcome_optimizer_state=outcome_state,
            batch=minibatch,
            assignments=mini_assignments,
            codebook_embeddings=codebook_embeddings,
        )
        return (
            updated.params,
            updated.bellman_optimizer_state,
            updated.outcome_optimizer_state,
        ), updated.metrics

    final, metrics = jax.lax.scan(update_one, initial, jnp.asarray(schedule))
    return TrainingUpdate(
        params=final[0],
        bellman_optimizer_state=final[1],
        outcome_optimizer_state=final[2],
        metrics=jax.tree_util.tree_map(lambda value: jnp.mean(value), metrics),
    )


def polyak_update(target_params: Any, online_params: Any, coefficient: float) -> Any:
    import jax

    tau = float(coefficient)
    return jax.tree_util.tree_map(
        lambda target, online: (1.0 - tau) * target + tau * online,
        target_params,
        online_params,
    )


def rollout_kl_means(batch: VQBCRolloutBatch) -> tuple[Any, Any]:
    import jax
    import jax.numpy as jnp

    reference = jax.nn.log_softmax(batch.reference_logits[:-1], axis=-1)

    def one(logits: Any) -> Any:
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        probabilities = jnp.exp(log_probs)
        return jnp.mean(
            jnp.sum(probabilities * (log_probs - reference), axis=-1)
        )

    return one(batch.execution_logits), one(batch.generic_execution_logits)


def _distribution_summary(values: Any) -> Mapping[str, Any]:
    import jax.numpy as jnp

    flattened = jnp.asarray(values, dtype=jnp.float32).reshape((-1,))
    quantiles = jnp.quantile(
        flattened, jnp.asarray((0.05, 0.50, 0.95), dtype=jnp.float32)
    )
    return {
        "count": jnp.asarray(flattened.size, dtype=jnp.int32),
        "mean": jnp.mean(flattened),
        "standard_deviation": jnp.std(flattened),
        "minimum": jnp.min(flattened),
        "p05": quantiles[0],
        "median": quantiles[1],
        "p95": quantiles[2],
        "maximum": jnp.max(flattened),
    }


def _code_usage_summary(codes: Any, *, response_count: int) -> Mapping[str, Any]:
    import jax.numpy as jnp

    counts = jnp.bincount(
        jnp.asarray(codes, dtype=jnp.int32).reshape((-1,)),
        length=int(response_count),
    )

    def summarize(current_counts: Any) -> Mapping[str, Any]:
        total = jnp.sum(current_counts)
        probabilities = current_counts / jnp.maximum(total, 1)
        entropy = -jnp.sum(
            probabilities * jnp.log(jnp.maximum(probabilities, 1.0e-12))
        )
        return {
            "counts": current_counts,
            "probabilities": probabilities,
            "active_code_count": jnp.sum(
                current_counts > 0, dtype=jnp.int32
            ),
            "entropy": entropy,
            "perplexity": jnp.where(total > 0, jnp.exp(entropy), 0.0),
        }

    return {
        "all_codes": summarize(counts),
        "nonterminal_codes": summarize(counts[:-1]),
    }


def rollout_health_diagnostics(
    *,
    batch: VQBCRolloutBatch,
    assignments: FrozenAssignments,
    response_count: int = 16,
) -> Mapping[str, Any]:
    """Summarize slot, response, posterior, policy, and control health."""

    import jax
    import jax.numpy as jnp

    responsibilities = jnp.asarray(
        assignments.responsibilities, dtype=jnp.float32
    )
    responsibility_entropy = -jnp.sum(
        responsibilities
        * jnp.log(jnp.maximum(responsibilities, 1.0e-12)),
        axis=-1,
    )
    slot_count = int(responsibilities.shape[-1])
    responsibility_mass = jnp.sum(responsibilities, axis=0)
    responsibility_fraction = responsibility_mass / jnp.maximum(
        jnp.sum(responsibility_mass), 1.0
    )
    responsibility_mass_entropy = -jnp.sum(
        responsibility_fraction
        * jnp.log(jnp.maximum(responsibility_fraction, 1.0e-12))
    )

    quotient_counts = jnp.asarray(batch.quotient_counts, dtype=jnp.int32)
    quotient_histogram = jnp.bincount(
        quotient_counts.reshape((-1,)), length=slot_count + 1
    )[1:]

    reference_log_probabilities = jax.nn.log_softmax(
        batch.reference_logits[:-1], axis=-1
    )

    def policy_kl(logits: Any) -> Any:
        log_probabilities = jax.nn.log_softmax(logits, axis=-1)
        probabilities = jnp.exp(log_probabilities)
        return jnp.maximum(
            jnp.sum(
                probabilities
                * (log_probabilities - reference_log_probabilities),
                axis=-1,
            ),
            0.0,
        )

    posterior_kl = policy_kl(batch.execution_logits)
    generic_kl = policy_kl(batch.generic_execution_logits)
    reference_argmax = jnp.argmax(batch.reference_logits[:-1], axis=-1)

    slot_log_probabilities = jax.nn.log_softmax(
        batch.slot_log_beliefs[:-1], axis=-1
    )
    slot_probabilities = jnp.exp(slot_log_probabilities)
    posterior_entropy = -jnp.sum(
        slot_probabilities * slot_log_probabilities, axis=-1
    )

    value_difference = jnp.asarray(batch.j_use) - jnp.asarray(batch.j_mask)
    selected_value_difference = jnp.take_along_axis(
        value_difference,
        jnp.asarray(batch.actions, dtype=jnp.int32)[..., None],
        axis=-1,
    )[..., 0]

    return {
        "responsibility": {
            "entropy": _distribution_summary(responsibility_entropy),
            "normalized_entropy_mean": (
                jnp.mean(responsibility_entropy) / jnp.log(float(slot_count))
            ),
            "effective_mass_by_slot": responsibility_mass,
            "effective_fraction_by_slot": responsibility_fraction,
            "mass_entropy": responsibility_mass_entropy,
            "effective_slot_count": jnp.exp(responsibility_mass_entropy),
        },
        "quotient": {
            "active_count": _distribution_summary(quotient_counts),
            "count_histogram": quotient_histogram,
        },
        "response_code": {
            "observed": _code_usage_summary(
                batch.response_codes, response_count=response_count
            ),
            "value_target": _code_usage_summary(
                assignments.response_code_targets,
                response_count=response_count,
            ),
        },
        "policy": {
            "posterior_kl": _distribution_summary(posterior_kl),
            "generic_kl": _distribution_summary(generic_kl),
            "posterior_argmax_deviation_rate": jnp.mean(
                jnp.argmax(batch.execution_logits, axis=-1)
                != reference_argmax
            ),
            "generic_argmax_deviation_rate": jnp.mean(
                jnp.argmax(batch.generic_execution_logits, axis=-1)
                != reference_argmax
            ),
            "posterior_entropy": _distribution_summary(posterior_entropy),
            "normalized_posterior_entropy_mean": (
                jnp.mean(posterior_entropy) / jnp.log(float(slot_count))
            ),
        },
        "value_information": {
            "all_action_j_use_minus_j_mask": {
                **_distribution_summary(value_difference),
                "positive_fraction": jnp.mean(value_difference > 0.0),
            },
            "executed_action_j_use_minus_j_mask": {
                **_distribution_summary(selected_value_difference),
                "positive_fraction": jnp.mean(
                    selected_value_difference > 0.0
                ),
            },
        },
    }


def update_kl_state(
    state: VQBCKLState,
    *,
    posterior_mean_kl: Any,
    generic_mean_kl: Any,
    target_kl: float,
    learning_rate: float,
    minimum_temperature: float,
    maximum_temperature: float,
) -> VQBCKLState:
    return VQBCKLState(
        log_temperature=update_log_temperature(
            log_temperature=state.log_temperature,
            mean_kl=posterior_mean_kl,
            target_kl=target_kl,
            learning_rate=learning_rate,
            minimum_temperature=minimum_temperature,
            maximum_temperature=maximum_temperature,
        ),
        generic_log_temperature=update_log_temperature(
            log_temperature=state.generic_log_temperature,
            mean_kl=generic_mean_kl,
            target_kl=target_kl,
            learning_rate=learning_rate,
            minimum_temperature=minimum_temperature,
            maximum_temperature=maximum_temperature,
        ),
    )


def update_codebook_from_assignments(
    codebook: Any,
    *,
    assignments: FrozenAssignments,
    dones: Any,
    key: Any,
    decay: float,
    replacement_after_rollouts: int,
) -> Any:
    return update_codebook(
        codebook,
        signatures=assignments.response_signature_targets,
        valid_mask=~dones,
        key=key,
        decay=decay,
        replacement_after_rollouts=replacement_after_rollouts,
    )


__all__ = [
    "OptimizerBundle",
    "TrainingUpdate",
    "apply_minibatch_update",
    "apply_control_sequence",
    "apply_model_sequence",
    "apply_rollout_updates",
    "environment_minibatch_schedule",
    "make_optimizers",
    "polyak_update",
    "prepare_frozen_assignments",
    "rollout_health_diagnostics",
    "rollout_kl_means",
    "slice_environment_lanes",
    "slice_rollout_batch",
    "update_codebook_from_assignments",
    "update_kl_state",
]
