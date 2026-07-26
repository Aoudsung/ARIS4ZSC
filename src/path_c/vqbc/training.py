"""Fresh-rollout VQBC V4.2 training with behavior-consistent targets."""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple

from .codebook import nearest_codes, target_response_signatures, update_codebook
from .model import vqbc_forward
from .objectives import (
    FrozenAssignments,
    LossBundle,
    bellman_loss,
    bellman_targets,
    episode_responsibility_evidence,
    outcome_loss,
    raw_policy_continuation_targets,
    response_encoder_loss,
    sample_bootstrap_mask,
)
from .policy import uniform_slot_log_belief, update_log_temperature
from .quotient import slot_bayes_update
from .types import VQBCKLState, VQBCRolloutBatch


class OptimizerBundle(NamedTuple):
    bellman_optimizer: Any
    outcome_optimizer: Any
    bellman_state: Any
    outcome_state: Any


class TrainingUpdate(NamedTuple):
    params: Any
    target_params: Any
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

    bellman = transform(bellman_learning_rate, _bellman_labels(params))
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
        batch.slot_log_beliefs,
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
        batch.slot_log_beliefs,
        method=model.control_only,
    )
    del unused_carry
    return output


def apply_heads_from_features(
    *,
    model: Any,
    params: Mapping[str, Any],
    features: Any,
    slot_log_beliefs: Any,
) -> Mapping[str, Any]:
    return model.apply(
        {"params": params},
        features,
        slot_log_beliefs,
        method=model.from_features,
    )


def slice_rollout_batch(batch: VQBCRolloutBatch, indexes: Any) -> VQBCRolloutBatch:
    return VQBCRolloutBatch(
        **{
            name: (
                value[:, indexes]
                if name != "initial_value_carry"
                else value[indexes]
            )
            for name, value in batch._asdict().items()
        }
    )


def _forward(
    *,
    raw_output: Mapping[str, Any],
    reference_logits: Any,
    slot_log_belief: Any,
    temperature: Any,
    generic_temperature: Any,
    gamma: float,
) -> Any:
    return vqbc_forward(
        raw_output=raw_output,
        reference_logits=reference_logits,
        slot_log_belief=slot_log_belief,
        temperature=temperature,
        generic_temperature=generic_temperature,
        deployment_mode="posterior_use",
        gamma=gamma,
    )


def _branch_targets(
    *,
    model: Any,
    target_params: Mapping[str, Any],
    target_output: Mapping[str, Any],
    batch: VQBCRolloutBatch,
    temperature: Any,
    generic_temperature: Any,
    gamma: float,
) -> tuple[Any, Any, Any, Any, Any]:
    """Exact runtime policy targets for use and one-response-mask branches."""

    import jax

    use_raw = {name: value[1:] for name, value in target_output.items()}
    use_forward = _forward(
        raw_output=use_raw,
        reference_logits=batch.reference_logits[1:],
        slot_log_belief=batch.slot_log_beliefs[1:],
        temperature=temperature,
        generic_temperature=generic_temperature,
        gamma=gamma,
    )
    # Same next observation/history; only the controller belief is masked.
    mask_raw = jax.tree_util.tree_map(
        jax.lax.stop_gradient,
        apply_heads_from_features(
            model=model,
            params=target_params,
            features=target_output["features"][1:],
            slot_log_beliefs=batch.slot_log_beliefs[:-1],
        ),
    )
    mask_forward = _forward(
        raw_output=mask_raw,
        reference_logits=batch.reference_logits[1:],
        slot_log_belief=batch.slot_log_beliefs[:-1],
        temperature=temperature,
        generic_temperature=generic_temperature,
        gamma=gamma,
    )
    import jax.numpy as jnp

    use_probabilities = jax.nn.softmax(use_forward.execution_logits, axis=-1)
    mask_probabilities = jax.nn.softmax(mask_forward.execution_logits, axis=-1)
    use_targets = raw_policy_continuation_targets(
        execution_probabilities=use_probabilities,
        target_q_values=use_raw["q_values"],
        dones=batch.dones,
    )
    mask_targets = raw_policy_continuation_targets(
        execution_probabilities=mask_probabilities,
        target_q_values=mask_raw["q_values"],
        dones=batch.dones,
    )
    return use_forward, mask_forward, use_probabilities, use_targets, mask_targets


def _stale_belief_targets(
    *,
    model: Any,
    target_params: Mapping[str, Any],
    target_output: Mapping[str, Any],
    batch: VQBCRolloutBatch,
    response_codes: Any,
    temperature: Any,
    generic_temperature: Any,
    gamma: float,
) -> tuple[Any, Any]:
    """Bellman targets for the one-response-stale states used by A2-mask."""

    import jax

    stale_current = batch.slot_log_beliefs[:-2]
    stale_current_raw = apply_heads_from_features(
        model=model,
        params=target_params,
        features=target_output["features"][1:-1],
        slot_log_beliefs=stale_current,
    )
    stale_next = slot_bayes_update(
        slot_log_belief=stale_current,
        slot_response_probabilities=stale_current_raw["response_probabilities"],
        action=batch.actions[1:],
        response_code=response_codes[1:],
    )
    stale_next_raw = jax.tree_util.tree_map(
        jax.lax.stop_gradient,
        apply_heads_from_features(
            model=model,
            params=target_params,
            features=target_output["features"][2:],
            slot_log_beliefs=stale_next,
        ),
    )
    stale_forward = _forward(
        raw_output=stale_next_raw,
        reference_logits=batch.reference_logits[2:],
        slot_log_belief=stale_next,
        temperature=temperature,
        generic_temperature=generic_temperature,
        gamma=gamma,
    )
    probabilities = jax.nn.softmax(stale_forward.execution_logits, axis=-1)
    targets = bellman_targets(
        rewards=batch.rewards[1:],
        dones=batch.dones[1:],
        next_execution_probabilities=probabilities,
        target_next_q_values=stale_next_raw["q_values"],
        gamma=gamma,
    )
    return jax.lax.stop_gradient(stale_current), jax.lax.stop_gradient(targets)


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
    bootstrap_mask: Any | None = None,
    terminal_response: int = 15,
    environment_chunk_size: int | None = None,
) -> FrozenAssignments:
    """Cross-fitted target-network E-step over complete episode lanes."""

    import jax
    import jax.numpy as jnp

    del online_params
    environment_count = int(batch.actions.shape[1])
    chunk_size = environment_count if environment_chunk_size is None else int(
        environment_chunk_size
    )
    if chunk_size <= 0 or environment_count % chunk_size:
        raise ValueError("Assignment chunks must divide environment lanes.")
    slot_count = int(batch.slot_log_beliefs.shape[-1])
    mask = (
        sample_bootstrap_mask(
            key,
            environment_count=environment_count,
            slot_count=slot_count,
            probability=bootstrap_probability,
        )
        if bootstrap_mask is None
        else jnp.asarray(bootstrap_mask, dtype=jnp.bool_)
    )
    if mask.shape != (environment_count, slot_count):
        raise ValueError("Bootstrap mask has the wrong shape.")
    offsets = jnp.arange(chunk_size, dtype=jnp.int32)

    def prepare_chunk(start: Any) -> tuple[Any, ...]:
        indexes = start + offsets
        current = slice_rollout_batch(batch, indexes)
        target = jax.tree_util.tree_map(
            jax.lax.stop_gradient,
            apply_model_sequence(model=model, params=target_params, batch=current),
        )
        (
            unused_use_forward,
            unused_mask_forward,
            use_probabilities,
            use_continuation_targets,
            mask_continuation_targets,
        ) = _branch_targets(
            model=model,
            target_params=target_params,
            target_output=target,
            batch=current,
            temperature=temperature,
            generic_temperature=generic_temperature,
            gamma=gamma,
        )
        del unused_use_forward, unused_mask_forward
        targets = jax.lax.stop_gradient(
            bellman_targets(
                rewards=current.rewards,
                dones=current.dones,
                next_execution_probabilities=use_probabilities,
                target_next_q_values=target["q_values"][1:],
                gamma=gamma,
            )
        )
        # Response codes are properties of the observable transition, not of
        # the controller belief used by a particular counterfactual branch.
        # Re-evaluate the target heads under a canonical uniform belief so an
        # identical (o, a, o') transition cannot receive different code targets
        # solely because use and mask carry different explicit beliefs.
        canonical_belief = uniform_slot_log_belief(
            target["features"][1:].shape[:-1], slot_count
        )
        canonical = jax.tree_util.tree_map(
            jax.lax.stop_gradient,
            apply_heads_from_features(
                model=model,
                params=target_params,
                features=target["features"][1:],
                slot_log_beliefs=canonical_belief,
            ),
        )
        signatures = jax.lax.stop_gradient(
            target_response_signatures(
                target_centered_advantages=canonical["centered_advantages"]
            )
        )
        code_targets, unused_error = nearest_codes(signatures, codebook_embeddings)
        del unused_error
        code_targets = jax.lax.stop_gradient(
            jnp.where(current.dones, terminal_response, code_targets)
        )
        stale_beliefs, stale_targets = _stale_belief_targets(
            model=model,
            target_params=target_params,
            target_output=target,
            batch=current,
            response_codes=current.response_codes,
            temperature=temperature,
            generic_temperature=generic_temperature,
            gamma=gamma,
        )
        responsibilities, energies = episode_responsibility_evidence(
            q_values=target["q_values"][:-1],
            actions=current.actions,
            targets=targets,
            temperature=responsibility_temperature,
            response_logits=target["response_logits"][:-1],
            reward_mean=target["reward_mean"][:-1],
            reward_log_standard_deviation=target[
                "reward_log_standard_deviation"
            ][:-1],
            continuation_use_mean=target["continuation_use_mean"][:-1],
            continuation_use_log_standard_deviation=target[
                "continuation_use_log_standard_deviation"
            ][:-1],
            continuation_mask_mean=target["continuation_mask_mean"][:-1],
            continuation_mask_log_standard_deviation=target[
                "continuation_mask_log_standard_deviation"
            ][:-1],
            response_codes=current.response_codes,
            rewards=current.rewards,
            continuation_use_targets=use_continuation_targets,
            continuation_mask_targets=mask_continuation_targets,
            availability_mask=mask[indexes],
        )
        return (
            responsibilities,
            energies,
            targets,
            stale_targets,
            stale_beliefs,
            signatures,
            code_targets,
            use_continuation_targets,
            mask_continuation_targets,
        )

    starts = jnp.arange(0, environment_count, chunk_size, dtype=jnp.int32)
    chunks = jax.lax.map(prepare_chunk, starts)
    time_count = int(batch.actions.shape[0])

    def restore_time(values: Any, length: int) -> Any:
        return jnp.swapaxes(values, 0, 1).reshape(
            (length, environment_count) + values.shape[3:]
        )

    return FrozenAssignments(
        responsibilities=chunks[0].reshape((environment_count, slot_count)),
        responsibility_energies=chunks[1].reshape(
            (environment_count, slot_count)
        ),
        bootstrap_mask=mask,
        bellman_targets=restore_time(chunks[2], time_count),
        stale_bellman_targets=restore_time(chunks[3], time_count - 1),
        stale_current_beliefs=restore_time(chunks[4], time_count - 1),
        response_signature_targets=restore_time(chunks[5], time_count),
        response_code_targets=restore_time(chunks[6], time_count),
        continuation_use_targets=restore_time(chunks[7], time_count),
        continuation_mask_targets=restore_time(chunks[8], time_count),
    )


def slice_environment_lanes(
    batch: VQBCRolloutBatch, assignments: FrozenAssignments, indexes: Any
) -> tuple[VQBCRolloutBatch, FrozenAssignments]:
    import jax

    assignment_slice = FrozenAssignments(
        responsibilities=assignments.responsibilities[indexes],
        responsibility_energies=assignments.responsibility_energies[indexes],
        bootstrap_mask=assignments.bootstrap_mask[indexes],
        bellman_targets=assignments.bellman_targets[:, indexes],
        stale_bellman_targets=assignments.stale_bellman_targets[:, indexes],
        stale_current_beliefs=assignments.stale_current_beliefs[:, indexes],
        response_signature_targets=assignments.response_signature_targets[:, indexes],
        response_code_targets=assignments.response_code_targets[:, indexes],
        continuation_use_targets=assignments.continuation_use_targets[:, indexes],
        continuation_mask_targets=assignments.continuation_mask_targets[:, indexes],
    )
    return slice_rollout_batch(batch, indexes), jax.tree_util.tree_map(
        jax.lax.stop_gradient, assignment_slice
    )


def environment_minibatch_schedule(
    key: Any,
    *,
    environment_count: int,
    minibatches_per_epoch: int,
    update_epochs: int,
) -> Any:
    import jax
    import jax.numpy as jnp

    if environment_count < minibatches_per_epoch:
        raise ValueError("Minibatches cannot exceed complete environment lanes.")
    if environment_count % minibatches_per_epoch:
        raise ValueError("Environment lanes must divide exactly into minibatches.")
    lane_count = environment_count // minibatches_per_epoch
    keys = jax.random.split(key, update_epochs)
    return jnp.stack(
        [
            jax.random.permutation(epoch_key, environment_count).reshape(
                (minibatches_per_epoch, lane_count)
            )
            for epoch_key in keys
        ],
        axis=0,
    )


def _losses(
    *,
    model: Any,
    params: Mapping[str, Any],
    batch: VQBCRolloutBatch,
    assignments: FrozenAssignments,
    codebook_embeddings: Any,
) -> tuple[LossBundle, tuple[Any, Mapping[str, Any]]]:
    output = apply_model_sequence(model=model, params=params, batch=batch)
    actual = bellman_loss(
        q_values=output["q_values"][:-1],
        actions=batch.actions,
        targets=assignments.bellman_targets,
        stopped_responsibilities=assignments.responsibilities,
        bootstrap_mask=assignments.bootstrap_mask,
        metric_prefix="bellman",
    )
    stale_raw = apply_heads_from_features(
        model=model,
        params=params,
        features=output["features"][1:-1],
        slot_log_beliefs=assignments.stale_current_beliefs,
    )
    stale = bellman_loss(
        q_values=stale_raw["q_values"],
        actions=batch.actions[1:],
        targets=assignments.stale_bellman_targets,
        stopped_responsibilities=assignments.responsibilities,
        bootstrap_mask=assignments.bootstrap_mask,
        metric_prefix="stale_belief_bellman",
    )
    bellman = LossBundle(
        total=actual.total + stale.total,
        metrics={**actual.metrics, **stale.metrics},
    )
    outcome = outcome_loss(
        response_logits=output["response_logits"][:-1],
        reward_mean=output["reward_mean"][:-1],
        reward_log_standard_deviation=output[
            "reward_log_standard_deviation"
        ][:-1],
        continuation_use_mean=output["continuation_use_mean"][:-1],
        continuation_use_log_standard_deviation=output[
            "continuation_use_log_standard_deviation"
        ][:-1],
        continuation_mask_mean=output["continuation_mask_mean"][:-1],
        continuation_mask_log_standard_deviation=output[
            "continuation_mask_log_standard_deviation"
        ][:-1],
        response_codes=batch.response_codes,
        rewards=batch.rewards,
        continuation_use_targets=assignments.continuation_use_targets,
        continuation_mask_targets=assignments.continuation_mask_targets,
        actions=batch.actions,
        stopped_responsibilities=assignments.responsibilities,
        bootstrap_mask=assignments.bootstrap_mask,
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
    polyak_coefficient: float,
) -> TrainingUpdate:
    import jax
    import optax

    def bellman_function(candidate: Mapping[str, Any]) -> tuple[Any, Mapping[str, Any]]:
        bellman, unused = _losses(
            model=model,
            params=candidate,
            batch=batch,
            assignments=assignments,
            codebook_embeddings=codebook_embeddings,
        )
        del unused
        return bellman.total, bellman.metrics

    (unused, bellman_metrics), gradients = jax.value_and_grad(
        bellman_function, has_aux=True
    )(params)
    del unused
    updates, bellman_optimizer_state = bellman_optimizer.update(
        gradients, bellman_optimizer_state, params
    )
    params = optax.apply_updates(params, updates)

    def outcome_function(candidate: Mapping[str, Any]) -> tuple[Any, Mapping[str, Any]]:
        unused_bellman, outcome = _losses(
            model=model,
            params=candidate,
            batch=batch,
            assignments=assignments,
            codebook_embeddings=codebook_embeddings,
        )
        del unused_bellman
        return outcome

    (unused, outcome_metrics), gradients = jax.value_and_grad(
        outcome_function, has_aux=True
    )(params)
    del unused
    updates, outcome_optimizer_state = outcome_optimizer.update(
        gradients, outcome_optimizer_state, params
    )
    params = optax.apply_updates(params, updates)
    target_params = polyak_update(target_params, params, polyak_coefficient)
    return TrainingUpdate(
        params=params,
        target_params=target_params,
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
    polyak_coefficient: float,
) -> TrainingUpdate:
    import jax
    import jax.numpy as jnp

    schedule_array = jnp.asarray(schedule)
    if schedule_array.ndim != 2:
        raise ValueError("One M-step requires [minibatch, lane] schedule axes.")
    initial = (
        params,
        target_params,
        optimizer_bundle.bellman_state,
        optimizer_bundle.outcome_state,
    )

    def update_one(carry: tuple[Any, ...], indexes: Any) -> tuple[Any, Any]:
        current_params, current_target, bellman_state, outcome_state = carry
        minibatch, mini_assignments = slice_environment_lanes(
            batch, assignments, indexes
        )
        updated = apply_minibatch_update(
            model=model,
            params=current_params,
            target_params=current_target,
            bellman_optimizer=optimizer_bundle.bellman_optimizer,
            outcome_optimizer=optimizer_bundle.outcome_optimizer,
            bellman_optimizer_state=bellman_state,
            outcome_optimizer_state=outcome_state,
            batch=minibatch,
            assignments=mini_assignments,
            codebook_embeddings=codebook_embeddings,
            polyak_coefficient=polyak_coefficient,
        )
        return (
            updated.params,
            updated.target_params,
            updated.bellman_optimizer_state,
            updated.outcome_optimizer_state,
        ), updated.metrics

    final, metrics = jax.lax.scan(update_one, initial, schedule_array)
    return TrainingUpdate(
        params=final[0],
        target_params=final[1],
        bellman_optimizer_state=final[2],
        outcome_optimizer_state=final[3],
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
        log_probabilities = jax.nn.log_softmax(logits, axis=-1)
        probabilities = jnp.exp(log_probabilities)
        return jnp.mean(
            jnp.sum(probabilities * (log_probabilities - reference), axis=-1)
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
            "active_code_count": jnp.sum(current_counts > 0, dtype=jnp.int32),
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
    import jax
    import jax.numpy as jnp

    responsibilities = jnp.asarray(assignments.responsibilities)
    responsibility_entropy = -jnp.sum(
        responsibilities * jnp.log(jnp.maximum(responsibilities, 1.0e-12)),
        axis=-1,
    )
    slot_count = int(responsibilities.shape[-1])
    mass = jnp.sum(responsibilities, axis=0)
    fraction = mass / jnp.maximum(jnp.sum(mass), 1.0)
    mass_entropy = -jnp.sum(
        fraction * jnp.log(jnp.maximum(fraction, 1.0e-12))
    )
    energies = jnp.sort(assignments.responsibility_energies, axis=-1)
    beliefs = jax.nn.log_softmax(batch.slot_log_beliefs[:-1], axis=-1)
    probabilities = jnp.exp(beliefs)
    posterior_entropy = -jnp.sum(probabilities * beliefs, axis=-1)
    return {
        "responsibility": {
            "entropy": _distribution_summary(responsibility_entropy),
            "normalized_entropy_mean": jnp.mean(responsibility_entropy)
            / jnp.log(float(slot_count)),
            "effective_mass_by_slot": mass,
            "effective_fraction_by_slot": fraction,
            "mass_entropy": mass_entropy,
            "effective_slot_count": jnp.exp(mass_entropy),
            "energy_margin": _distribution_summary(energies[..., 1] - energies[..., 0]),
            "energy_spread": _distribution_summary(energies[..., -1] - energies[..., 0]),
            "bootstrap_active_fraction": jnp.mean(
                assignments.bootstrap_mask.astype(jnp.float32)
            ),
        },
        "quotient": {
            "posterior_supported_active_count": _distribution_summary(
                batch.quotient_counts
            ),
            "count_histogram": jnp.bincount(
                batch.quotient_counts.reshape((-1,)), length=slot_count + 1
            )[1:],
        },
        "response_code": {
            "observed": _code_usage_summary(
                batch.response_codes, response_count=response_count
            ),
            "value_target": _code_usage_summary(
                assignments.response_code_targets, response_count=response_count
            ),
        },
        "policy": {
            "posterior_entropy": _distribution_summary(posterior_entropy),
            "normalized_posterior_entropy_mean": jnp.mean(posterior_entropy)
            / jnp.log(float(slot_count)),
            "use_mask_total_variation": _distribution_summary(
                batch.predicted_policy_total_variations
            ),
        },
        "value_information": {
            "per_action_response_value": _distribution_summary(
                batch.per_action_response_values
            ),
            "per_action_net_value": _distribution_summary(
                batch.per_action_net_values
            ),
            "executed_action_response_value": _distribution_summary(
                batch.executed_action_response_values
            ),
            "executed_action_net_value": _distribution_summary(
                batch.executed_action_net_values
            ),
            "maximum_action_net_value": _distribution_summary(
                batch.maximum_action_net_values
            ),
            "predicted_response_effect": _distribution_summary(
                batch.predicted_response_effects
            ),
            "predicted_policy_cost": _distribution_summary(
                batch.predicted_policy_costs
            ),
            "predicted_net_effect": _distribution_summary(
                batch.predicted_net_effects
            ),
            "regularized_net_effect": _distribution_summary(
                batch.predicted_regularized_net_effects
            ),
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
    "apply_control_sequence",
    "apply_heads_from_features",
    "apply_minibatch_update",
    "apply_model_sequence",
    "apply_rollout_updates",
    "environment_minibatch_schedule",
    "make_optimizers",
    "polyak_update",
    "prepare_frozen_assignments",
    "rollout_health_diagnostics",
    "rollout_kl_means",
    "sample_bootstrap_mask",
    "slice_environment_lanes",
    "slice_rollout_batch",
    "update_codebook_from_assignments",
    "update_kl_state",
]
